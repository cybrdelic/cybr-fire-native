#!/usr/bin/env python3
"""Render an explicitly two-grid numerical VFX capture, with no image synthesis.

Requires a COMPLETE parent cache from build_parent_cache.py. Outputs individual
lossless frames before encoding. All grid counts and time/quality settings are
recorded; the fine scalar grid must not be reported as a fine pressure solve.
"""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,hashlib,json,os,subprocess,sys,time,gc
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.detail import DetailConfig,DetailTransport
from cybrfire.config import scene_from_dict
from cybrfire.gpu import ComputeContext
from cybrfire.render import VolumeRenderer


def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def core_digest(root):
 h=hashlib.sha256()
 for p in sorted([*root.glob('cybrfire/*.py'),*root.glob('shaders/*')]):h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
 return h.hexdigest()


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--parent-cache',type=Path,default=Path('cache/parent'))
 ap.add_argument('--out',type=Path,default=Path('captures/v8'))
 ap.add_argument('--shape',nargs=3,type=int,default=(384,672,384))
 ap.add_argument('--warmup',type=int,default=8)
 ap.add_argument('--frames',type=int,default=24)
 ap.add_argument('--substeps',type=int,default=2)
 ap.add_argument('--width',type=int,default=2160);ap.add_argument('--height',type=int,default=3840)
 ap.add_argument('--samples',type=int,default=1);ap.add_argument('--strength',type=float,default=.40)
 a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True);framesdir=out/'frames';framesdir.mkdir(exist_ok=True)
 m=json.loads((a.parent_cache/'manifest.json').read_text())
 if not m.get('complete'):raise RuntimeError('Parent cache is incomplete')
 if a.warmup+a.frames>=len(m['frames']):raise ValueError('Insufficient cached parent states')
 if a.frames<1 or a.substeps<1:raise ValueError('Invalid capture length')
 scene=scene_from_dict(m['scene'])
 scene=replace(scene,render=replace(scene.render,bloom=0.,floor=False,width=a.width,height=a.height,
     fov=40.,camera_distance=3.4,camera_height=1.6,target_height=1.05,yaw=12.,
     empty_space_skipping=False,key_intensity=0.,shadow_resolution_scale=8,
     reconstruction_mode=1,step_scale=.60,samples=a.samples))
 cfg=DetailConfig(shape=tuple(a.shape),rms_velocity=a.strength)
 record={'status':'running','method':'fine_reactive_scalar_transport_with_prescribed_MAC_flow_and_transverse_mode_closure',
    'fine_config':asdict(cfg),'fine_scalar_cells':int(np.prod(cfg.shape)),'parent_grid':m['grid'],
    'parent_velocity_pressure_cells':int(np.prod(m['grid'])),'fine_pressure_solve':False,'fine_scalar_format':'RGBA32F','velocity_sampling_cache_format':'RGBA16F',
    'fine_two_way_thermal_momentum_feedback':False,'new_subgrid_modes_are_modeled':True,
    'synthetic_images_or_external_fire_footage':False,'render_time_warp_or_image_sharpening':False,
    'initialization':'trilinear initial-condition transfer followed by explicit fine-grid source, transport and reaction updates',
    'scene':scene.to_dict(),'warmup_frames':a.warmup,'fps':24,'physical_dt_per_frame':1/24,
    'substeps':a.substeps,'native_video_dimensions':[a.width,a.height], 'parent_manifest_sha256':digest(a.parent_cache/'manifest.json'),
    'source_sha256':core_digest(Path(__file__).resolve().parents[1]),'frames':[]}
 report=out/'capture.json';report.write_text(json.dumps(record,indent=2))
 ctx=ComputeContext();record['adapter']=ctx.info;t=time.perf_counter()
 sim=DetailTransport(ctx,scene,cfg,a.parent_cache/'initial-q.npy',a.parent_cache)
 ctx.finish();record['initialization_seconds']=time.perf_counter()-t
 print('initialized',record['fine_scalar_cells'],'cells, seconds',record['initialization_seconds'],flush=True)
 r=VolumeRenderer(ctx,sim,optical_workspace=sim.next,lighting_scale=4)
 journal=(out/'frame-journal.jsonl').open('w');start=time.perf_counter()
 # Release no simulation state on a failed render; lossless frames and journal
 # are individually durable, but this run is only complete after MP4 validation.
 try:
  for frame in range(a.warmup+a.frames):
   sim.set_parent_pair(frame,frame+1);t=time.perf_counter()
   for sub in range(a.substeps):
    sim.step(m['start_time']+frame/24+sub/(24*a.substeps),1/(24*a.substeps),(sub+.5)/a.substeps)
   ctx.finish();sim_seconds=time.perf_counter()-t
   print('evolved',frame+1,'seconds',sim_seconds,flush=True)
   if frame<a.warmup:continue
   i=frame-a.warmup;r.optical_texture=sim.next.handle;t=time.perf_counter()
   image=r.render_tiled_rgb(width=a.width,height=a.height,samples=a.samples,prepare=True,band_height=64)
   render_seconds=time.perf_counter()-t;path=framesdir/f'{i:04}.png';Image.fromarray(image).save(path)
   item={'index':i,'time':m['start_time']+(frame+1)/24,'fine_steps_completed':sim.tick,
       'simulation_seconds':sim_seconds,'render_seconds':render_seconds,'sha256':digest(path)}
   record['frames'].append(item);journal.write(json.dumps(item)+'\n');journal.flush();os.fsync(journal.fileno())
   report.write_text(json.dumps(record,indent=2));print('rendered',i,'seconds',render_seconds,flush=True)
   if i in (0,a.frames//2,a.frames-1):
    Image.fromarray(image).save(out/f'V8_detail_still_{i:02}.png')
    small=Image.fromarray(image);small.thumbnail((900,1600));small.save(out/f'preview-{i:02}.jpg',quality=94)
   del image;gc.collect()
  record['final_state']=sim.state_statistics();record['total_wall_seconds']=time.perf_counter()-start
  record['final_fine_steps']=sim.tick;record['status']='frames_complete';report.write_text(json.dumps(record,indent=2))
 finally:
  journal.close();r.close();sim.close();ctx.close()
 video=out/'V8_detail_rebuild_4K.mp4';temp=out/'V8_encoding.tmp.mp4'
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','24','-i',str(framesdir/'%04d.png'),'-c:v','libx264','-preset','slow','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',str(temp)],check=True)
 subprocess.run(['ffmpeg','-v','error','-i',str(temp),'-f','null','-'],check=True)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(temp)]))
 if int(probe['streams'][0]['nb_read_frames'])!=a.frames:raise RuntimeError('Decoded frame count mismatch')
 temp.replace(video);record.update(status='completed_and_fully_decoded',video_sha256=digest(video),video_bytes=video.stat().st_size,probe=probe)
 report.write_text(json.dumps(record,indent=2));print('COMPLETE',video,flush=True)

if __name__=='__main__':main()
