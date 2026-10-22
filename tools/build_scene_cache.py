"""Generate independent physical scene states; checkpoint full gas and surface.

This is deliberately NOT a V7 mature-fire checkpoint import. Each scenario
begins with quiescent ambient fields at t=0 to avoid shared legacy plumes.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.gpu import ComputeContext
from cybrfire.solver import Simulation
from cybrfire.config import scene_from_dict
from real_scenarios import make_scene
from quality_profiles import get_profile


def publish(path,value):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--scene',required=True,choices=['jet','pool','fireball'])
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--quality',choices=['native-v8','preview'],default='native-v8')
    a.add_argument('--shape',type=int,nargs=3,default=None)
    a.add_argument('--frames',type=int,default=29)
    a.add_argument('--batch',type=int,default=4)
    args=a.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    prof=get_profile(args.quality)
    selected=tuple(args.shape) if args.shape is not None else prof.parent
    if args.quality=='native-v8' and selected!=prof.parent:
        raise ValueError('Native V8 forbids reduced parent grid; use --quality preview for a preview')
    mpath=args.out/'manifest.json'; statepath=args.out/'restart.npz'
    ctx=ComputeContext()
    try:
        if mpath.exists() and statepath.exists():
            m=json.loads(mpath.read_text());scene=scene_from_dict(m['scene'])
            if tuple(scene.domain.shape)!=selected or m.get('quality_profile')!=args.quality:
                raise RuntimeError('Attempted resume with a different quality grid/profile')
        else:
            scene=make_scene(args.scene,selected);m=dict(
                scene=scene.to_dict(),start_time=0.,frame_dt=1/24,grid=scene.domain.shape,
                frames=[],complete=False,scenario=args.scene,quality_profile=args.quality,independent_zero_initial_conditions=True,
                adapter=ctx.info)
        sim=Simulation(ctx,scene)
        if statepath.exists():
            with np.load(statepath,allow_pickle=False) as z:
                sim.q.write(z['q']);sim.vel.write(z['velocity']);sim.tick=int(z['tick'])
                if sim.surface_fuel:sim.surface_fuel.state.write(z['surface'])
        else:
            # The initial q is all zero by construction. Ambient O2 is encoded as
            # oxygen DEFICIT=0, and ambient T=300 K as energy density=0.
            np.save(args.out/'initial-q.npy',sim.q.read())
            publish(mpath,m)
        first=len(m['frames']);end=min(args.frames,first+args.batch)
        for idx in range(first,end):
            t=time.perf_counter()
            if idx:sim.step(5)
            ctx.finish()
            dims=tuple(n+1 for n in sim.shape)
            f=args.out/f'velocity-{idx:03d}.npy'
            dest=np.lib.format.open_memmap(f,mode='w+',dtype=np.float32,shape=sim.vel.shape)
            for z in range(0,dims[2],8):
                slab=sim.vel.read_slab(z,min(8,dims[2]-z));at=z*dims[0]*dims[1]
                dest[at:at+slab.size//4]=slab.reshape(-1,4)
            dest.flush();del dest
            d=sim.diagnostics() if idx%6==0 else {'time':sim.time}
            if sim.surface_fuel:d['surface_fuel']=sim.surface_fuel.inventory()
            # Atomic state publication: write gas + fuel in same checkpoint.
            temp=args.out/'restart.tmp.npz'
            val=dict(q=sim.q.read(),velocity=sim.vel.read(),tick=np.int64(sim.tick))
            if sim.surface_fuel:val['surface']=sim.surface_fuel.state.read()
            np.savez(temp,**val);temp.replace(statepath)
            m['frames'].append({'index':idx,'time':sim.time,'seconds':time.perf_counter()-t,'diagnostics':d})
            publish(mpath,m)
            print('PARENT',args.scene,idx+1,'/',args.frames,'time',round(sim.time,4),'wall',round(time.perf_counter()-t,3),flush=True)
        if len(m['frames'])>=args.frames:
            m['complete']=True;publish(mpath,m)
            print('PARENT COMPLETE',args.scene,flush=True)
    finally:
        ctx.close()

if __name__=='__main__':main()
