#!/usr/bin/env python3
"""Native parent micro-batches, protected by verified remote checkpoint receipts.

Resume exactly one or more 1/120-second ticks. Publish a velocity sample on
24-Hz boundaries only; preserve partial physical intervals between invocations.
No simulation setting or grid is reduced to fit an execution timeout.
"""
from __future__ import annotations
import argparse, fcntl, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
from checkpoint_guard import digest, atomic_text, verify
from v8_guarded import audit, commit, require_remote

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--exports',type=Path,required=True)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--ticks',type=int,default=1)
    a=p.parse_args()
    if not 1<=a.ticks<=5:raise ValueError('Use 1..5 ticks per guarded micro-batch')
    lock=open(a.store/'BATCH.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=require_remote(a.store);verified=verify(a.store)
    for f in verified['files']:
        if digest(a.run/f['path'])!=f['sha256']:
            raise RuntimeError(f'Live state differs from last verified generation: {f["path"]}')
    before=audit(a.run)
    if not a.engine.exists():
        with zipfile.ZipFile(a.run/'provenance/engine_source.zip') as z:z.extractall(a.engine.parent)
    sys.path.insert(0,str(a.engine))
    from cybrfire.gpu import ComputeContext
    from cybrfire.config import scene_from_dict
    from cybrfire.solver import Simulation
    parent=a.run/'parent';mp=parent/'manifest.json'
    m=json.loads(mp.read_text());scene=scene_from_dict(m['scene'])
    started=time.perf_counter();ctx=ComputeContext()
    try:
        sim=Simulation(ctx,scene)
        with np.load(parent/'restart.npz',allow_pickle=False) as z:
            sim.q.write(z['q']);sim.vel.write(z['velocity'])
            sim.mg.pressure.write(z['pressure']);sim.tick=int(z['tick'])
            if sim.surface_fuel:sim.surface_fuel.state.write(z['surface'])
            if sim.embers:sim.embers.state.write(z['particles'])
        print('RESTORED tick',sim.tick,'native',sim.shape,'adapter',ctx.info['renderer'],flush=True)
        frame_ticks=round(m['frame_dt']/scene.domain.tick)
        for _ in range(a.ticks):
            t=time.perf_counter();sim.step();ctx.finish()
            print('EVOLVED tick',sim.tick,'wall',round(time.perf_counter()-t,3),flush=True)
            if sim.tick%frame_ticks==0:
                idx=len(m['frames']);nx,ny,nz=sim.vel.grid_shape
                arr=np.lib.format.open_memmap(parent/f'.velocity-{idx:03d}.npy',mode='w+',dtype=np.float32,shape=sim.vel.shape)
                for z in range(0,nz,8):
                    n=min(8,nz-z);arr[z*nx*ny:(z+n)*nx*ny]=sim.vel.read_slab(z,n).reshape(-1,4)
                arr.flush();del arr
                os.replace(parent/f'.velocity-{idx:03d}.npy',parent/f'velocity-{idx:03d}.npy')
                m['frames'].append({'index':idx,'time':sim.time,'seconds':time.perf_counter()-t})
        diag=sim.diagnostics()
        if not diag['finite']:raise RuntimeError('Nonfinite state')
        values={'q':sim.q.read(),'velocity':sim.vel.read(),'pressure':sim.mg.pressure.read(),'tick':np.int64(sim.tick)}
        if sim.surface_fuel:values['surface']=sim.surface_fuel.state.read()
        if sim.embers:values['particles']=sim.embers.state.read()
        tmp=parent/'restart.next.npz'
        with tmp.open('wb') as f:
            np.savez(f,**values);f.flush();os.fsync(f.fileno())
        del values;os.replace(tmp,parent/'restart.npz')
        m['pending_parent_tick']=sim.tick if sim.tick%frame_ticks else None
        m['restart_includes_pressure']=True;m['complete']=False
        m['latest_restart_diagnostics']=diag
        atomic_text(mp,json.dumps(m,indent=2))
        print('SAVED tick',sim.tick,'temperature',diag['max_temperature_K'],flush=True)
    finally:ctx.close()
    for path in Path(__file__).parent.glob('*.py'):
        (a.run/'provenance'/path.name).write_bytes(path.read_bytes())
    settings=json.loads((a.run/'RUN_SETTINGS.json').read_text())
    settings.update(status='paused_pending_remote_backup',restart_driver='advance_guarded_parent.py',
        microstep_seconds=scene.domain.tick)
    atomic_text(a.run/'RUN_SETTINGS.json',json.dumps(settings,indent=2))
    result=commit(a.run,a.store,a.exports)
    result.update(resumed_from_remote_file_id=receipt['remote_file_id'],started_from=before,
        total_wall_seconds=time.perf_counter()-started)
    atomic_text(a.exports/'last_batch.json',json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
