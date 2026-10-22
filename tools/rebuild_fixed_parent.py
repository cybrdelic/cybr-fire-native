"""Checkpointed parent solve with physically consistent spatial remeshing."""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time,hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cybrfire.config import scene_from_dict
from cybrfire.gpu import ComputeContext
from cybrfire.solver import Simulation
from cybrfire.remesh import conservative_scalars, staggered_velocity

p=argparse.ArgumentParser()
p.add_argument('--start', type=Path, required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--frames',type=int,default=81)
p.add_argument('--batch',type=int,default=14)
p.add_argument('--shape',type=int,nargs=3,default=(64,112,64))
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
mfile=a.out/'manifest.json'; cp=a.out/'restart.npz'; start_q=a.out/'initial-q.npy'
with np.load(a.start,allow_pickle=False) as src:
 orig=scene_from_dict(json.loads(str(src['config'])))
 base_time=int(src['tick'])*orig.domain.tick
 old=tuple(orig.domain.shape); new=tuple(a.shape)
 assert all(x>0 and x%8==0 for x in new)
 old_ext=np.array(old)*orig.domain.spacing
 new_h=old_ext[0]/new[0]
 assert np.allclose(np.array(new)*new_h,old_ext), (old_ext,new_h,new)
 assert all(all(0<=e.center[k]<=old_ext[k] for k in range(3)) for e in orig.emitters)
 scene=replace(orig,domain=replace(orig.domain,shape=new,spacing=new_h,tick=1/120,native_fields=True,trace_courant=1.,max_trace_steps=16))
 assert all(all(0<=e.center[k]<=new[k]*new_h for k in range(3)) for e in scene.emitters)
 scene.validate()
 ctx=ComputeContext(); sim=Simulation(ctx,scene)
 try:
  if mfile.exists() and cp.exists():
   manifest=json.loads(mfile.read_text())
   assert np.array_equal(manifest['grid'],new)
   assert abs(manifest['spacing']-new_h)<1e-12
   with np.load(cp,allow_pickle=False) as c:
    sim.q.write(c['q']);sim.vel.write(c['velocity']);sim.tick=int(c['tick'])
   i=len(manifest['frames'])
  else:
   q=np.asarray(src['q']).reshape(old[2],old[1],old[0],4)
   v=np.asarray(src['velocity']).reshape(old[2]+1,old[1]+1,old[0]+1,4)
   qnew=conservative_scalars(q,new).reshape(-1,4).astype(np.float32)
   vnew=staggered_velocity(v,old,new).reshape(-1,4).astype(np.float32)
   sim.q.write(qnew);sim.vel.write(vnew);sim.tick=round(base_time/scene.domain.tick)
   np.save(start_q,qnew)
   manifest={'scene':scene.to_dict(),'grid':new,'spacing':new_h,'extent_m':old_ext.tolist(), 'emitter_inside_domain':True,
      'import_equivalent_restart':False,'start_time':base_time,'frame_dt':1/24,'frames':[],'complete':False,
      'source_checkpoint_sha256':hashlib.sha256(a.start.read_bytes()).hexdigest()}
   mfile.write_text(json.dumps(manifest,indent=2));i=0
  for index in range(i,min(a.frames,i+a.batch)):
   t=time.perf_counter()
   if index:sim.step(5)
   ctx.finish()
   dims=tuple(n+1 for n in new)
   dest=np.lib.format.open_memmap(a.out/f'velocity-{index:03d}.npy',mode='w+',dtype=np.float32,shape=sim.vel.shape)
   for z in range(0,dims[2],8):
    slab=sim.vel.read_slab(z,min(8,dims[2]-z))
    start=z*dims[0]*dims[1]; dest[start:start+slab.size//4]=slab.reshape(-1,4)
   dest.flush();del dest
   diag=sim.diagnostics() if index%8==0 else {'time':sim.time}
   manifest['frames'].append({'index':index,'time':sim.time,'seconds':time.perf_counter()-t,'diagnostics':diag})
   # Write restart checkpoint before extending manifest so no half-commit.
   temp=a.out/'restart.tmp.npz'
   np.savez(temp,q=sim.q.read(),velocity=sim.vel.read(),tick=np.int64(sim.tick))
   temp.replace(cp)
   (a.out/'manifest.tmp.json').write_text(json.dumps(manifest,indent=2))
   (a.out/'manifest.tmp.json').replace(mfile)
   print('PARENT',index+1,'/',a.frames,'at',round(sim.time,3),'wall',round(time.perf_counter()-t,2),flush=True)
  if len(manifest['frames'])==a.frames:
   manifest['complete']=True
   mfile.write_text(json.dumps(manifest,indent=2))
   print('PARENT COMPLETE',flush=True)
 finally:
  ctx.close()
