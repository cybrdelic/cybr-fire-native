#!/usr/bin/env python3
"""One fine-transport interval, bounded-RAM checkpointing, then Guard commit.

This stage does not render or claim a frame. Warmup progress is explicit.
Uses the unmodified 99,090,432-cell V8 numerical profile and two 1/48s steps.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, zipfile, gc
from dataclasses import asdict,replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.file_lock import lock_exclusive
import numpy as np
from checkpoint_guard import verify,digest,atomic_text
from v8_guarded import audit,require_remote,commit

def write_field(field,path:Path):
    tmp=path.with_name('.'+path.name+'.partial')
    nx,ny,nz=field.grid_shape
    with tmp.open('wb') as f:
        np.lib.format.write_array_header_2_0(f,{'descr':'<f4','fortran_order':False,'shape':field.shape})
        for z in range(0,nz,4):
            slab=field.read_slab(z,min(4,nz-z))
            f.write(memoryview(slab).cast('B'))
            del slab
            # Avoid resident mmap pages competing with the two full GPU fields.
            if z%32==28:
                f.flush();getattr(os,'fdatasync',os.fsync)(f.fileno())
                if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser()
    for n in ('run','store','exports','engine'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();lock=open(a.store/'BATCH.lock','a');lock_exclusive(lock)
    receipt=require_remote(a.store);g=verify(a.store)
    for f in g['files']:
        if digest(a.run/f['path'])!=f['sha256']:raise RuntimeError('Live state changed: '+f['path'])
    before=audit(a.run)
    if not a.engine.exists():
        with zipfile.ZipFile(a.run/'provenance/engine_source.zip') as z:z.extractall(a.engine.parent)
    sys.path.insert(0,str(a.engine));sys.path.insert(0,str(a.engine/'tools'))
    from cybrfire.config import scene_from_dict
    from cybrfire.gpu import ComputeContext
    from cybrfire.detail import DetailConfig,DetailTransport,upload_npy_slabs
    from quality_profiles import get_profile,require_native
    pm=json.loads((a.run/'parent/manifest.json').read_text())
    settings=json.loads((a.run/'RUN_SETTINGS.json').read_text())
    s=scene_from_dict(pm['scene']);profile=get_profile('native-v8')
    s=replace(s,render=replace(s.render,width=3840,height=2160,samples=2,exposure=.009,
       camera_distance=1.25,fov=42.,target_height=.65,bloom=0.,floor=False,key_intensity=0.,
       shadow_resolution_scale=8,reconstruction_mode=1,empty_space_skipping=False,step_scale=.60))
    require_native(s.domain.shape,profile.fine,(3840,2160))
    cfg=DetailConfig(shape=profile.fine,flow_scale=profile.flow_scale,rms_velocity=.4,wavelengths=profile.wavelengths)
    cap=a.run/'capture';cap.mkdir(exist_ok=True);rp=cap/'capture.json'
    identity=json.dumps({k:pm[k] for k in ('scene','scenario','start_time','grid','quality_profile')},sort_keys=True)
    if rp.exists():
        r=json.loads(rp.read_text());cur=int(r['current'])
        if cur>=settings['warmup']:raise RuntimeError('Next interval needs a render-capable guarded driver')
    else:
        cur=0;r={'status':'warmup_checkpointed','parent_identity_hash':hashlib.sha256(identity.encode()).hexdigest(),
          'initial_q_hash':digest(a.run/'parent/initial-q.npy'),'scenario':pm['scenario'],
          'scene':s.to_dict(),'fine':asdict(cfg),'quality_profile':'native-v8','current':0,'tick':0,'frames':[],
          'fps':24,'resolution':[3840,2160],'samples':2,'fine_pressure_solve':False,
          'synthetic_frame_interpolation':False,'initial_condition':'quiescent ambient independent scene at t=0',
          'warmup_target_intervals':settings['warmup']}
    if len(pm['frames'])<=cur+1:raise RuntimeError('Parent velocity pair not available')
    ctx=ComputeContext();started=time.perf_counter()
    try:
        sim=DetailTransport(ctx,s,cfg,a.run/'parent/initial-q.npy',a.run/'parent')
        if cur:
            upload_npy_slabs(sim.q,cap/r['checkpoint_npy']);sim.tick=int(r['tick'])
            if sim.surface_fuel:sim.surface_fuel.state.write(np.load(cap/r['surface_checkpoint_npy']))
        ctx.finish();print('NATIVE_FINE_READY',sim.shape,flush=True)
        sim.set_parent_pair(cur,cur+1)
        for j in range(2):
            sim.step(pm['start_time']+cur/24+j/48,1/48,(j+.5)/2);ctx.finish()
            print('FINE_SUBSTEP',sim.tick,'of',2*(cur+1),flush=True)
        name=f'fine-state-{cur+1:05d}.npy'
        write_field(sim.q,cap/name)
        if sim.surface_fuel:
            sn=f'surface-state-{cur+1:05d}.npy';np.save(cap/sn,sim.surface_fuel.state.read());r['surface_checkpoint_npy']=sn
        r.update(current=cur+1,tick=sim.tick,checkpoint_npy=name,status='warmup_checkpointed',
                 parent_source_states_observed=len(pm['frames']),adapter=ctx.info)
        atomic_text(rp,json.dumps(r,indent=2))
        print('FINE_STATE_SAVED',name,(cap/name).stat().st_size,'bytes',flush=True)
        sim.close()
    finally:ctx.close()
    for pth in Path(__file__).parent.glob('*.py'):(a.run/'provenance'/pth.name).write_bytes(pth.read_bytes())
    settings.update(status='paused_pending_remote_backup',fine_checkpoint_driver='advance_guarded_fine.py')
    atomic_text(a.run/'RUN_SETTINGS.json',json.dumps(settings,indent=2))
    # Keep commit/export separate from evolution under a short execution budget.
    audit(a.run)
    print('READY_FOR_GUARD_COMMIT',round(time.perf_counter()-started,3),'seconds; no video frame claimed',flush=True)
if __name__=='__main__':main()
