"""Evolve the existing MAC solver and publish immutable velocity snapshots.
The archive's start is an explicitly imported initial condition, not an equivalent
restart under the finer tick. Output uses world units and conservative fields.
"""
from pathlib import Path
from dataclasses import replace
import sys,json,time,zipfile,hashlib,ctypes,gc
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import scene_from_dict
from cybrfire.gpu import ComputeContext
from cybrfire.solver import Simulation
root=Path(__file__).resolve().parents[1];out=root/'cache/parent';out.mkdir(parents=True,exist_ok=True)
archive=Path('input/CYBR_FIRE_V7_complete.zip')
with zipfile.ZipFile(archive) as z:
 with z.open('CYBR_FIRE_V7/cache/v7-start.npz') as src,(out/'start.npz').open('wb') as dst:
  import shutil;shutil.copyfileobj(src,dst)
with np.load(out/'start.npz',allow_pickle=False) as z:
 old=scene_from_dict(json.loads(str(z['config'])));oldtime=int(z['tick'])*old.domain.tick
 scene=replace(old,domain=replace(old.domain,native_fields=True,tick=1/120,trace_courant=1.,max_trace_steps=16))
 ctx=ComputeContext();sim=Simulation(ctx,scene)
 sim.q.write(z['q']);sim.vel.write(z['velocity']);sim.tick=round(oldtime/scene.domain.tick)
 # Preserve the imported instantaneous field. The first advancement reprojects.
 np.save(out/'initial-q.npy',z['q'])
manifest={'scene':scene.to_dict(),'start_time':oldtime,'frame_dt':1/24,'grid':sim.shape,'adapter':ctx.info,'frames':[],'input_sha256':hashlib.sha256((out/'start.npz').read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
for i in range(33):
 t=time.perf_counter()
 if i:sim.step(5)
 ctx.finish()
 # Sparse host slabs avoid retaining full snapshot copies.
 d=tuple(n+1 for n in sim.shape)
 dest=np.lib.format.open_memmap(out/f'velocity-{i:03}.npy',mode='w+',dtype=np.float32,shape=sim.vel.shape)
 for k in range(0,d[2],8):
  a=sim.vel.read_slab(k,min(8,d[2]-k));start=k*d[0]*d[1];dest[start:start+a.size//4]=a.reshape(-1,4)
 dest.flush();del dest
 diag=sim.diagnostics() if i%8==0 else {'time':sim.time}
 manifest['frames'].append({'index':i,'time':sim.time,'seconds':time.perf_counter()-t,'diagnostics':diag})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(i,sim.time,manifest['frames'][-1]['seconds'],flush=True)
manifest['complete']=True;(out/'manifest.json').write_text(json.dumps(manifest,indent=2));ctx.close()
