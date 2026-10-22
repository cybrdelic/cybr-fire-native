#!/usr/bin/env python3
"""Apply Checkpoint Guard to a complete, dependency-checked native V8 job.

A local generation is not called remotely durable. An external upload receipt
and a byte-identical download are required before the next batch is allowed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path
import numpy as np
from checkpoint_guard import snapshot, verify, restore, digest, atomic_text

PARENT = (160, 280, 160)
FINE = (384, 672, 384)

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def safe_file(root: Path, rel: str) -> Path:
    p=Path(rel)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Unsafe path: {rel}')
    out=root/p
    if not out.is_file() or out.is_symlink() or root.resolve() not in out.resolve().parents:
        raise FileNotFoundError(f'Missing or unsafe dependency: {out}')
    return out

def require_array(path: Path, shape: tuple, finite: bool=False) -> None:
    a=np.load(path, mmap_mode='r', allow_pickle=False)
    if a.shape!=shape or a.dtype!=np.float32:
        raise ValueError(f'Wrong array shape/type: {path}, {a.shape}, {a.dtype}')
    if finite:
        for start in range(0,len(a),262144):
            if not np.isfinite(a[start:start+262144]).all():
                raise ValueError(f'Nonfinite data: {path}')
    del a

def audit(run: Path) -> dict:
    """Check semantic restart consistency, not just whether files exist."""
    run=run.resolve()
    m=load_json(safe_file(run,'parent/manifest.json'))
    if tuple(m['grid'])!=PARENT or tuple(m['scene']['domain']['shape'])!=PARENT:
        raise ValueError('Native V8 parent grid changed')
    if m.get('quality_profile')!='native-v8':
        raise ValueError('Not a native V8 run')
    frames=m.get('frames',[])
    if not frames or [x['index'] for x in frames]!=list(range(len(frames))):
        raise ValueError('Nonconsecutive parent snapshots')
    dt=float(m['frame_dt']); start=float(m.get('start_time',0))
    for i,f in enumerate(frames):
        if abs(float(f['time'])-(start+i*dt))>1e-8:
            raise ValueError(f'Parent timestamp discontinuity at {i}')
    n=int(np.prod(PARENT)); vn=int(np.prod(np.array(PARENT)+1))
    require_array(safe_file(run,'parent/initial-q.npy'),(n,4))
    for i in range(len(frames)):
        require_array(safe_file(run,f'parent/velocity-{i:03d}.npy'),(vn,4))
    with np.load(safe_file(run,'parent/restart.npz'),allow_pickle=False) as z:
        required={'q':(n,4),'velocity':(vn,4),'pressure':(n,)}
        if m['scene'].get('surface_fuel',{}).get('enabled'):
            if 'surface' not in z: raise ValueError('Missing surface fuel state')
        for key,shape in required.items():
            if key not in z: raise ValueError(f'Incomplete parent restart: missing {key}')
            a=z[key]
            if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():
                raise ValueError(f'Invalid parent {key}')
            if key=='q' and float(a.min()) < -1e-4:
                raise ValueError('Negative species')
            del a
        tick=int(z['tick'])
        last_tick=round(frames[-1]['time']/m['scene']['domain']['tick'])
        frame_ticks=round(dt/m['scene']['domain']['tick'])
        if not last_tick<=tick<last_tick+frame_ticks:
            raise ValueError('Restart tick is outside the next parent snapshot interval')
        if tick!=last_tick and m.get('pending_parent_tick')!=tick:
            raise ValueError('Unrecorded partial parent interval')
        if tick==last_tick and m.get('pending_parent_tick') is not None:
            raise ValueError('Stale pending parent interval')
    source=['parent/manifest.json','parent/restart.npz','parent/initial-q.npy']
    source += [f'parent/velocity-{i:03d}.npy' for i in range(len(frames))]
    current=rendered=0
    if (run/'capture/capture.json').exists():
        c=load_json(run/'capture/capture.json'); current=int(c['current'])
        if c['quality_profile']!='native-v8' or tuple(c['fine']['shape'])!=FINE:
            raise ValueError('Fine quality changed')
        if current<0 or current>=len(frames):
            raise ValueError('Fine progress has no corresponding parent state')
        if c['tick']!=current*2:
            raise ValueError('Fine tick/frame mismatch')
        source.append('capture/capture.json')
        if current:
            rel='capture/'+c['checkpoint_npy']; safe_file(run,rel)
            require_array(run/rel,(int(np.prod(FINE)),4));source.append(rel)
        if c.get('surface_checkpoint_npy'):
            rel='capture/'+c['surface_checkpoint_npy'];safe_file(run,rel);source.append(rel)
        fs=c['frames'];rendered=len(fs)
        if [x['index'] for x in fs]!=list(range(rendered)):
            raise ValueError('Nonconsecutive rendered frames')
        if len({x['sha256'] for x in fs})!=rendered:
            raise ValueError('Duplicate source frames')
        for f in fs:
            rel=f'capture/frames/{f["index"]:04d}.png'
            if digest(safe_file(run,rel))!=f['sha256']:
                raise ValueError('Frame content mismatch')
            source.append(rel)
        # Include committed tile metadata and images plus its independent field.
        tile=run/'capture/band_resume.json'
        if tile.exists():
            t=load_json(tile);source.append('capture/band_resume.json')
            if t.get('state_path'):
                rel='capture/'+t['state_path'];safe_file(run,rel);source.append(rel)
            for band in t.get('bands',{}).values():
                rel='capture/'+band['path']
                if digest(safe_file(run,rel))!=band['hash']:
                    raise ValueError('Band content mismatch')
                source.append(rel)
    for p in sorted((run/'provenance').glob('*')):
        if p.is_file():source.append(p.relative_to(run).as_posix())
    source.append('RUN_SETTINGS.json');safe_file(run,'RUN_SETTINGS.json')
    return {'parent_snapshots':len(frames),'parent_tick':tick,
        'parent_sim_time':tick*m['scene']['domain']['tick'],'last_parent_snapshot_time':frames[-1]['time'],'fine_intervals':current,
        'rendered_frames':rendered,'dependencies':sorted(set(source))}

def commit(run: Path, store: Path, exports: Path) -> dict:
    a=audit(run)
    gen=f"jet-p{a['parent_snapshots']:04d}-t{a['parent_tick']:06d}-f{a['fine_intervals']:04d}"
    m=snapshot(run,store,a['dependencies'],generation=gen)
    verify(store,gen)
    # Verify dependencies after the immutable copy, not only before it.
    audit(store/'generations'/gen/'payload')
    return export_generation(store,exports,gen)

def export_generation(store: Path, exports: Path, generation: str | None=None) -> dict:
    # Export can be retried independently after a timeout. It never changes
    # the already committed immutable numeric generation.
    m=verify(store,generation);gen=m['generation']
    a=audit(store/'generations'/gen/'payload')
    exports.mkdir(parents=True,exist_ok=True)
    dst=exports/f'CYBR_FIRE_GUARDED_{gen}.zip'; tmp=dst.with_suffix('.zip.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        z.writestr('store/LATEST',gen+'\n')
        root=store/'generations'/gen
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,'store/generations/'+gen+'/'+p.relative_to(root).as_posix())
        for p in sorted(Path(__file__).parent.glob('*.py')):z.write(p,'guard/'+p.name)
        z.writestr('RECOVERY.txt',
            'Unzip into a new directory. Run: python guard/checkpoint_guard.py verify --store store\n'
            'Then: python guard/checkpoint_guard.py restore --store store --into restored_run\n'
            'Then: python guard/v8_guarded.py audit --run restored_run\n'
            'Engine source is restored_run/provenance/engine_source.zip.\n')
    os.replace(tmp,dst)
    report={'local_generation':gen,'archive':str(dst),'archive_sha256':digest(dst),
            'archive_bytes':dst.stat().st_size,'payload_bytes':sum(f['bytes'] for f in m['files']),
            'semantic_audit':a,'remote_status':'not_yet_uploaded'}
    atomic_text(dst.with_suffix('.json'),json.dumps(report,indent=2))
    return report

def confirm(store: Path, original: Path, downloaded: Path, remote_file_id: str) -> dict:
    if digest(original)!=digest(downloaded):raise ValueError('Downloaded archive differs')
    gen=(store/'LATEST').read_text().strip(); verify(store,gen)
    report={'generation':gen,'status':'uploaded_and_downloaded_hash_match',
            'archive_sha256':digest(original),'remote_file_id':remote_file_id,
            'downloaded_bytes':downloaded.stat().st_size,'checked_unix_time':time.time()}
    atomic_text(store/'REMOTE_CONFIRMED.json',json.dumps(report,indent=2))
    return report

def require_remote(store: Path) -> dict:
    m=verify(store);r=load_json(store/'REMOTE_CONFIRMED.json')
    if r['generation']!=m['generation'] or r['status']!='uploaded_and_downloaded_hash_match':
        raise RuntimeError('Next batch blocked: latest generation has no verified remote backup')
    return r

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('audit');a.add_argument('--run',type=Path,required=True)
    a=s.add_parser('commit');a.add_argument('--run',type=Path,required=True);a.add_argument('--store',type=Path,required=True);a.add_argument('--exports',type=Path,required=True)
    a=s.add_parser('export');a.add_argument('--store',type=Path,required=True);a.add_argument('--exports',type=Path,required=True)
    a=s.add_parser('confirm');a.add_argument('--store',type=Path,required=True);a.add_argument('--original',type=Path,required=True);a.add_argument('--downloaded',type=Path,required=True);a.add_argument('--remote-file-id',required=True)
    a=s.add_parser('check-remote');a.add_argument('--store',type=Path,required=True)
    x=p.parse_args()
    if x.cmd=='audit':r=audit(x.run)
    elif x.cmd=='commit':r=commit(x.run,x.store,x.exports)
    elif x.cmd=='export':r=export_generation(x.store,x.exports)
    elif x.cmd=='confirm':r=confirm(x.store,x.original,x.downloaded,x.remote_file_id)
    else:r=require_remote(x.store)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
