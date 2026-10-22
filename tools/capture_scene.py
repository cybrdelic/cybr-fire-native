"""Resumable scene-specific V8 fine transport, native ray-march, and audits.

Every frame has a new physical integration; checkpoints include surface mass
and heat state. No frame interpolation, loops or held-image synthesis.
"""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,gc,hashlib,json,subprocess,sys,time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.gpu import ComputeContext
from cybrfire.config import scene_from_dict
from cybrfire.detail import DetailTransport,DetailConfig
from cybrfire.render import VolumeRenderer
from cybrfire.detail import upload_npy_slabs
from quality_profiles import get_profile,require_native

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(8<<20),b''):h.update(buf)
    return h.hexdigest()

def atomic_json(path,data):
    t=path.with_suffix('.tmp.json');t.write_text(json.dumps(data,indent=2));t.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=24)
    ap.add_argument('--warmup',type=int,default=4)
    ap.add_argument('--batch',type=int,default=2)
    ap.add_argument('--width',type=int,default=None)
    ap.add_argument('--height',type=int,default=None)
    ap.add_argument('--samples',type=int,default=None)
    ap.add_argument('--quality',choices=['native-v8','preview'],default='native-v8')
    ap.add_argument('--shape',type=int,nargs=3,default=None)
    ap.add_argument('--exposure',type=float,default=None)
    ap.add_argument('--fov',type=float,default=None)
    ap.add_argument('--distance',type=float,default=None)
    ap.add_argument('--target-height',type=float,default=None)
    ap.add_argument('--emit-batch-clip',action='store_true',help='Encode only NEW frames from this independent batch, never repeat')
    ap.add_argument('--allow-growing-parent',action='store_true',help='Allow unfinished parent cache and stop before missing states')
    args=ap.parse_args()
    profile=get_profile(args.quality)
    args.shape=tuple(args.shape) if args.shape is not None else profile.fine
    args.width=args.width if args.width is not None else profile.video[0]
    args.height=args.height if args.height is not None else profile.video[1]
    args.samples=args.samples if args.samples is not None else profile.samples
    args.out.mkdir(parents=True,exist_ok=True);d=args.out/'frames';d.mkdir(exist_ok=True)
    m=json.loads((args.parent/'manifest.json').read_text())
    if args.allow_growing_parent:
        if len(m['frames']) < 2:
            raise RuntimeError('Need at least two parent states')
    elif not m['complete'] or len(m['frames'])<=args.frames+args.warmup:
        raise RuntimeError('Need complete parent velocity snapshots covering the entire capture')
    scene=scene_from_dict(m['scene'])
    if tuple(scene.domain.shape)!=profile.parent and args.quality=='native-v8':
        raise ValueError('Native V8 requires the original 160x280x160 parent')
    if m.get('quality_profile', 'preview')!=args.quality:
        raise ValueError('Cannot capture native V8 from preview cache (or vice versa)')
    if args.quality=='native-v8':
        require_native(scene.domain.shape,args.shape,(args.width,args.height))
    camera_updates=dict(width=args.width,height=args.height,samples=args.samples)
    if args.quality=='native-v8':
        # Preserve the original V8 ray marcher settings; scene-specific framing
        # and explicit user camera overrides remain possible.
        camera_updates.update(step_scale=.60,shadow_resolution_scale=8,
                              reconstruction_mode=1,empty_space_skipping=False,
                              bloom=0.,floor=False,key_intensity=0.)
    for key,value in [('exposure',args.exposure),('fov',args.fov),('camera_distance',args.distance),('target_height',args.target_height)]:
        if value is not None:camera_updates[key]=value
    scene=replace(scene,render=replace(scene.render,**camera_updates))
    cfg=DetailConfig(shape=tuple(args.shape),flow_scale=profile.flow_scale,
                     rms_velocity=.4,wavelengths=profile.wavelengths)
    record_path=args.out/'capture.json';checkpoint=args.out/'fine-state.npz'
    parent_identity=json.dumps({k:m[k] for k in ('scene','scenario','start_time','grid','quality_profile')},sort_keys=True)
    parent_identity_hash=hashlib.sha256(parent_identity.encode()).hexdigest()
    first_source_hash=digest(args.parent/'initial-q.npy')
    if record_path.exists():
        record=json.loads(record_path.read_text());current=record['current'];tick=record['tick']
        if current and not record.get('checkpoint_npy') and not checkpoint.exists():
            raise RuntimeError('Capture record has progress but no persisted fine state')
        if record.get('parent_identity_hash',record.get('parent_hash'))!=parent_identity_hash or record.get('initial_q_hash',first_source_hash)!=first_source_hash:
            raise RuntimeError('Parent scene/initial state changed after capture started')
        if len(m['frames']) <= current:
            raise RuntimeError('Missing parent states needed for resume')
        if record['scene']!=json.loads(json.dumps(scene.to_dict())):
            raise RuntimeError('Scene or camera settings changed in a resumed capture')
        if record.get('fine')!=json.loads(json.dumps(asdict(cfg))) or record.get('quality_profile')!=args.quality:
            raise RuntimeError('Fine-grid settings changed during resumed capture')
        if record['frames']:
            observed_start=int(round(24*record['frames'][0]['sim_time']))-record['frames'][0]['index']-1
            if observed_start!=args.warmup:
                raise RuntimeError(f'Warmup mismatch: resume requires {observed_start}, got {args.warmup}')
    else:
        record={'status':'running','parent_identity_hash':parent_identity_hash,'initial_q_hash':first_source_hash,'scenario':m['scenario'],
            'scene':scene.to_dict(),'fine':asdict(cfg),'quality_profile':args.quality,'current':0,'tick':0,'frames':[],
            'fps':24,'resolution':[args.width,args.height], 'samples':args.samples,
            'fine_pressure_solve':False,'synthetic_frame_interpolation':False,
            'initial_condition':'quiescent ambient independent scene at t=0'}
        current=tick=0
    first_output=len(record['frames'])
    available = len(m['frames'])-1
    until=min(current+args.batch,args.warmup+args.frames,available)
    if until<=current:
        print('WAITING_FOR_PARENT',m['scenario'],'current',current,'parent_states',len(m['frames']),flush=True)
        return
    if args.allow_growing_parent:
        record['parent_source_states_observed']=len(m['frames'])
    ctx=ComputeContext()
    try:
        sim=DetailTransport(ctx,scene,cfg,args.parent/'initial-q.npy',args.parent)
        ckpt_name=record.get('checkpoint_npy')
        if ckpt_name:
            upload_npy_slabs(sim.q,args.out/ckpt_name)
            sim.tick=tick
            if sim.surface_fuel:
                surf=record.get('surface_checkpoint_npy')
                if not surf:raise RuntimeError('Missing surface checkpoint in capture record')
                sim.surface_fuel.state.write(np.load(args.out/surf))
        elif checkpoint.exists():
            # Read compatibility for old preview checkpoints only.
            if args.quality=='native-v8':
                raise RuntimeError('Native V8 requires slab-streamed checkpoints, not a full-array NPZ')
            with np.load(checkpoint,allow_pickle=False) as z:
                sim.q.write(z['q']);sim.tick=tick
                if sim.surface_fuel:sim.surface_fuel.state.write(z['surface'])
        r=VolumeRenderer(ctx,sim,optical_workspace=sim.next,lighting_scale=4)
        for f in range(current,until):
            sim.set_parent_pair(f,f+1)
            t=time.perf_counter()
            for j in range(2):sim.step(m['start_time']+f/24+j/48,1/48,(j+.5)/2)
            ctx.finish();simtime=time.perf_counter()-t
            item=None
            if f>=args.warmup:
                idx=f-args.warmup;r.optical_texture=sim.next.handle
                t=time.perf_counter()
                pixels=r.render_tiled_rgb(width=args.width,height=args.height,samples=args.samples,seed=idx,prepare=True,band_height=64)
                render_time=time.perf_counter()-t
                path=d/f'{idx:04d}.png';Image.fromarray(pixels).save(path)
                bright=int(np.count_nonzero((pixels[:,:,0]>170)&(pixels[:,:,0]>pixels[:,:,2]*1.2)))
                item={'index':idx,'sim_time':(f+1)/24,'sha256':digest(path),'bright_flame_px':bright,
                      'simulation_wall_seconds':simtime,'render_wall_seconds':render_time}
                if sim.surface_fuel:item['surface']=sim.surface_fuel.inventory()
                record['frames'].append(item)
                if idx in (0,args.frames//2,args.frames-1):
                    img=Image.fromarray(pixels);img.thumbnail((540,960));img.save(args.out/f'preview-{idx:03d}.png')
                print('RENDER',m['scenario'],idx,'bright',bright,'sim',round(simtime,2),'render',round(render_time,2),flush=True)
                del pixels
            else:print('WARMUP',m['scenario'],f+1,'sim',round(simtime,2),flush=True)
            # 99M-cell native fields cannot be checkpointed using q.read():
            # that allocates another ~1.59GB host array on a 4GB runtime.
            # Slab streaming bounds transient RAM to 8 z-slices.
            name=f'fine-state-{f+1:05d}.npy'
            tmp=args.out/f'.{name}'
            nx,ny,nz=sim.q.grid_shape
            mapped=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=sim.q.shape)
            for z in range(0,nz,8):
                n=min(8,nz-z);offset=z*nx*ny
                mapped[offset:offset+n*nx*ny]=sim.q.read_slab(z,n).reshape(-1,4)
            mapped.flush();del mapped
            tmp.replace(args.out/name)
            old_surface=record.get('surface_checkpoint_npy')
            if sim.surface_fuel:
                surface_name=f'surface-state-{f+1:05d}.npy'
                st=args.out/f'.{surface_name}'
                np.save(st,sim.surface_fuel.state.read())
                st.replace(args.out/surface_name)
                record['surface_checkpoint_npy']=surface_name
            old_ckpt=record.get('checkpoint_npy')
            record['checkpoint_npy']=name
            record['current']=f+1;record['tick']=sim.tick
            atomic_json(record_path,record)
            if old_ckpt and old_ckpt!=name:
                (args.out/old_ckpt).unlink(missing_ok=True)
            if old_surface and old_surface!=record.get('surface_checkpoint_npy'):
                (args.out/old_surface).unlink(missing_ok=True)
            gc.collect()
        r.close();sim.close()
    finally:
        ctx.close()
    if record['current']==args.warmup+args.frames:
        vid=args.out/f"{m['scenario']}_{args.width}x{args.height}_24fps.mp4"
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','24','-i',str(d/'%04d.png'),'-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(vid)],check=True)
        subprocess.run(['ffmpeg','-v','error','-i',str(vid),'-f','null','-'],check=True)
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(vid)]))['streams'][0]
        if int(info['nb_read_frames'])!=args.frames:raise RuntimeError('Encoded frame count mismatch')
        if len(set(x['sha256'] for x in record['frames']))!=args.frames:raise RuntimeError('Repeated PNG source frames')
        # Sustained sources must remain luminous; transient sources are
        # expected to extinguish and leave a cooling afterglow or smoke.
        latest_emitter_end=max(e.end for e in scene.emitters if e.replacement_rate>0.) if any(e.replacement_rate>0. for e in scene.emitters) else float('inf')
        transient=latest_emitter_end < (args.warmup+args.frames)/24
        if transient:
            if max(x['bright_flame_px'] for x in record['frames'])<1000:
                raise RuntimeError('Transient source never ignited')
            if record['frames'][-1]['bright_flame_px']>=max(x['bright_flame_px'] for x in record['frames']):
                raise RuntimeError('Transient source has no decay after extinction')
        elif not all(x['bright_flame_px']>1000 for x in record['frames']):
            raise RuntimeError('Sustained source lost flame; refusing successful status')
        record['transient_extinction_expected']=transient
        record.update(status='completed_verified',probe=info,video_hash=digest(vid))
        atomic_json(record_path,record)
        print('FINISHED',vid,flush=True)
    elif args.emit_batch_clip and len(record['frames'])>first_output:
        batch_items=record['frames'][first_output:]
        start=batch_items[0]['index']; end=batch_items[-1]['index']
        if [v['index'] for v in batch_items]!=list(range(start,end+1)):
            raise RuntimeError('Nonconsecutive batch frame indices')
        if len({v['sha256'] for v in batch_items})!=len(batch_items):
            raise RuntimeError('Batch contained repeated PNG source frames')
        batches=args.out/'batches';batches.mkdir(exist_ok=True)
        outfile=batches/f'{m["scenario"]}_{start:04d}-{end:04d}_{args.width}x{args.height}.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','24',
            '-start_number',str(start),'-i',str(d/'%04d.png'),'-frames:v',str(len(batch_items)),
            '-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(outfile)],check=True)
        subprocess.run(['ffmpeg','-v','error','-i',str(outfile),'-f','null','-'],check=True)
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration',
            '-of','json',str(outfile)]))['streams'][0]
        if int(info['nb_read_frames'])!=len(batch_items) or (int(info['width']),int(info['height']))!=(args.width,args.height):
            raise RuntimeError('Batch decode/size mismatch')
        metadata={'scene':m['scenario'],'status':'verified_partial_batch','first_frame':start,'last_frame':end,
            'unique_simulated_frames':len(batch_items),'fps':24,'duration_seconds':len(batch_items)/24,
            'native_simulation_grid':list(profile.parent),'native_fine_grid':list(profile.fine),
            'resolution':[args.width,args.height],'frame_sha256':[v['sha256'] for v in batch_items],
            'video_sha256':digest(outfile),'probe':info}
        atomic_json(outfile.with_suffix('.json'),metadata)
        print('BATCH_VERIFIED',outfile,flush=True)

if __name__=='__main__':main()
