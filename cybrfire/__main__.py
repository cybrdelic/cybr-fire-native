"""Reproducible CLI. Every reported time is synchronized wall time, not FPS."""
import argparse
from dataclasses import replace
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from .gpu import ComputeContext
from .config import example,scene_from_dict
from .solver import Simulation
from .render import VolumeRenderer,tone_map

def code_digest():
    root=Path(__file__).resolve().parents[1];h=hashlib.sha256()
    for folder in ['cybrfire','shaders']:
        for p in sorted((root/folder).glob('*')):
            if p.is_file() and p.name not in ("__main__.py","live.py","__init__.py"):h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def capture(args):
    if args.frames<1 or args.frames>24000 or not 1<=args.fps<=120:raise ValueError('Invalid capture length/rate')
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('FFmpeg is required for MP4 capture')
    ctx=ComputeContext()
    scene=scene_from_dict(json.loads(Path(args.config).read_text())) if args.config else example(args.scene,args.quality)
    pressure=args.pressure or (scene.domain.pressure_backend if args.config else ('spectral' if ctx.info['software_adapter'] else 'multigrid'))
    args.width=args.width if args.width is not None else scene.render.width
    args.height=args.height if args.height is not None else scene.render.height
    args.samples=args.samples if args.samples is not None else scene.render.samples
    if args.width%2 or args.height%2 or min(args.width,args.height)<64:raise ValueError('Video dimensions must be even and at least 64')
    scene=replace(scene,domain=replace(scene.domain,pressure_backend=pressure),render=replace(scene.render,width=args.width,height=args.height,samples=args.samples))
    scene.validate();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'scene.json').write_text(json.dumps(scene.to_dict(),indent=2))
    sim=Simulation(ctx,scene);renderer=VolumeRenderer(ctx,sim)
    if args.resume:sim.load(args.resume)
    resume_sha256=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest() if args.resume else None
    print(json.dumps({'adapter':ctx.info,'scene':scene.to_dict(),'source_sha256':code_digest()}),flush=True)
    source_digest=code_digest()
    startwall=time.perf_counter();warmstart=time.perf_counter()
    sim.advance_to(max(sim.time,args.warmup));ctx.finish();warm_seconds=time.perf_counter()-warmstart
    capture_start=sim.time
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{args.width}x{args.height}','-r',str(args.fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'simulation.partial.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    stats=[];still_frames=sorted(set([args.frames//3,2*args.frames//3]));saved=[]
    try:
        base_yaw=scene.render.yaw
        for f in range(args.frames):
            renderer.cfg=replace(renderer.cfg,yaw=base_yaw+args.orbit_degrees*f/max(args.frames-1,1))
            t=time.perf_counter();requested=capture_start+f/args.fps;sim.advance_to(requested);ctx.finish();sim_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();rgb=renderer.render(width=args.width,height=args.height,samples=args.samples,seed=f);ctx.finish();render_ms=(time.perf_counter()-t)*1000
            proc.stdin.write(rgb.tobytes())
            if f == 0: Image.fromarray(rgb).save(out/'first-frame.png')
            if f%8==0 or f==args.frames-1:diag=sim.diagnostics()
            else:diag={'time':sim.time,'tick':sim.tick}
            stats.append({'frame':f,'requested_time':requested,'simulation_ms':sim_ms,'render_ms':render_ms,**diag})
            if f in still_frames:
                suffix='01' if not saved else '02'
                t=time.perf_counter();hdr=renderer.render_hdr(width=args.still_width,height=args.still_height,samples=4,seed=f,prepare=False)
                Image.fromarray(tone_map(hdr,renderer.cfg.exposure,renderer.cfg.bloom)).save(out/f'still-{suffix}.png')
                saved.append({'frame':f,'time':sim.time,'file':f'still-{suffix}.png','width':args.still_width,'height':args.still_height,'render_ms':(time.perf_counter()-t)*1000})
                if args.hdr and suffix=='01':np.savez_compressed(out/'linear-hdr.npz',radiance=hdr,exposure=scene.render.exposure)
            if f%8==0 or f==args.frames-1:
                (out/'progress.json').write_text(json.dumps({'frame':f,'frames':args.frames,'simulation_time':sim.time,'wall_seconds':time.perf_counter()-startwall}))
                print(f'frame {f}/{args.frames} t={sim.time:.3f} sim={sim_ms:.1f}ms render={render_ms:.1f}ms',flush=True)
        proc.stdin.close();err=proc.stderr.read();status=proc.wait()
        if status:raise RuntimeError('FFmpeg failed: '+err.decode(errors='replace'))
        os.replace(out/'simulation.partial.mp4',out/'simulation.mp4')
    except Exception:
        proc.kill();proc.wait();raise
    if args.checkpoint:sim.save(out/'final-state.npz')
    final=sim.diagnostics();ctx.finish()
    report={'engine':'CYBR FIRE 7.0.0','orbit_degrees':args.orbit_degrees,'render_acceleration':renderer.statistics(),'source_sha256':source_digest,'adapter':ctx.info,'pressure_backend':'hybrid_masked_PCG' if scene.solids else scene.domain.pressure_backend,'transport':'bounded subtraced RK2 MacCormack; full-precision texture cache' if scene.domain.texture_transport else 'bounded RK2 MacCormack; SSBO reference','solid_boundary':'voxelized free-slip' if scene.solids else None,'one_way_ember_particles':bool(scene.embers.enabled),'mass_conservative_advection':False,'image_generation_or_external_footage':False,'render_time_density_warp':False,'gpu_hardware_benchmark':False if ctx.info['software_adapter'] else True,'embergen_comparison':False,'physical_calibration':False,'frames':args.frames,'playback_fps':args.fps,'video_dimensions':[args.width,args.height],'simulated_cells':sim.n,'grid':list(sim.shape),'spacing_m':sim.h,'resume_checkpoint_sha256':resume_sha256,'capture_start_s':capture_start,'capture_end_s':sim.time,'warmup_wall_s':warm_seconds,'wall_seconds':time.perf_counter()-startwall,'simulation_median_ms_per_frame':float(np.median([s['simulation_ms'] for s in stats[1:]])) if len(stats)>1 else 0.,'render_median_ms_per_frame':float(np.median([s['render_ms'] for s in stats])),'stills':saved,'final_state':final,'measurements':stats,'timing_scope':'Synchronized wall time on the identified adapter. Includes dispatch; render includes readback and camera processing. Excludes video encoding and stills from per-frame timings. Spectral pressure performs CPU transforms and readback/upload.'}
    (out/'capture.partial.json').write_text(json.dumps(report,indent=2));os.replace(out/'capture.partial.json',out/'capture.json');ctx.close()
    print('COMPLETE',out,flush=True)

def main():
    p=argparse.ArgumentParser(prog='python -m cybrfire');sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('device');sub.add_parser('test')
    live=sub.add_parser('live');live.add_argument('--port',type=int,default=8765);live.add_argument('--quality',default='preview',choices=['test','preview','standard','high','ultra'])
    cp=sub.add_parser('capture');cp.add_argument('--scene',default='burner',choices=['burner','crosswind','vortex','pulse','two-burners','sweeping-jet','shutdown','ember-burner','deflector','grate','ember-shutdown']);cp.add_argument('--quality',default='standard',choices=['test','preview','standard','high','ultra'])
    cp.add_argument('--pressure',choices=['multigrid','spectral']);cp.add_argument('--config');cp.add_argument('--resume');cp.add_argument('--frames',type=int,default=72);cp.add_argument('--fps',type=int,default=24);cp.add_argument('--warmup',type=float,default=2.)
    cp.add_argument('--width',type=int,default=None);cp.add_argument('--height',type=int,default=None);cp.add_argument('--samples',type=int,default=None);cp.add_argument('--still-width',type=int,default=1080);cp.add_argument('--still-height',type=int,default=1440)
    cp.add_argument('--orbit-degrees',type=float,default=0.);cp.add_argument('--checkpoint',action='store_true');cp.add_argument('--hdr',action='store_true');cp.add_argument('--out',default='captures/burner')
    args=p.parse_args()
    if args.command=='device':
        c=ComputeContext();print(json.dumps(c.info,indent=2));c.close()
    elif args.command=='test':
        root=Path(__file__).resolve().parents[1]
        failed=0
        for file in ['run_tests.py','test_extended.py','test_runtime.py','test_v5.py','test_v6.py','test_particle_allocation.py']:
            failed|=subprocess.run([sys.executable,str(root/'tests'/file)]).returncode
        sys.exit(bool(failed))
    elif args.command=='live':
        from .live import serve
        serve(args.port,args.quality)
    else:capture(args)
if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError) as e:print('ERROR:',e,file=sys.stderr);sys.exit(1)
