"""Resume-compatible V8 fine transport and native volume capture with fuel/heat diagnostics."""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,hashlib,json,subprocess,sys,time,gc
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import scene_from_dict
from cybrfire.detail import DetailConfig, DetailTransport
from cybrfire.gpu import ComputeContext
from cybrfire.render import VolumeRenderer

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--parent',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--frames',type=int,default=24)
p.add_argument('--warmup',type=int,default=8)
p.add_argument('--batch',type=int,default=3)
p.add_argument('--width',type=int,default=720)
p.add_argument('--height',type=int,default=1280)
p.add_argument('--samples',type=int,default=2)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
framesdir=a.out/'frames';framesdir.mkdir(exist_ok=True)
m=json.loads((a.parent/'manifest.json').read_text())
if not m['complete'] or len(m['frames'])<=a.warmup+a.frames:raise RuntimeError('Parent snapshots incomplete')
scene=scene_from_dict(m['scene']);extent=np.array(scene.domain.shape)*scene.domain.spacing
assert np.allclose(extent,[1.92,3.36,1.92]),extent
assert all(all(0<=e.center[k]<=extent[k] for k in range(3)) for e in scene.emitters)
scene=replace(scene,render=replace(scene.render,bloom=0.,floor=False,width=a.width,height=a.height,
    fov=40.,camera_distance=3.4,camera_height=1.6,target_height=1.05,yaw=12.,
    empty_space_skipping=False,key_intensity=0.,shadow_resolution_scale=4,
    reconstruction_mode=1,step_scale=.60,samples=a.samples,stable_sampling=False))
cfg=DetailConfig(shape=(192,336,192),flow_scale=1,rms_velocity=.40,wavelengths=(.16,.08,.05))
recordpath=a.out/'capture.json';statepath=a.out/'state.npy'
if recordpath.exists() and statepath.exists():
 record=json.loads(recordpath.read_text());current=record['current'];tick=record['tick']
 assert record['parent_manifest_hash']==digest(a.parent/'manifest.json')
else:
 record={'status':'in_progress','scene':scene.to_dict(),'detail':asdict(cfg),'parent_manifest_hash':digest(a.parent/'manifest.json'),
         'original_extent_m':extent.tolist(),'emitter_inside_domain':True,'current':0,'tick':0,'frames':[],
         'fps':24,'width':a.width,'height':a.height,'samples':a.samples,'warmup':a.warmup,
         'source':'CYBR FIRE V8 reduced physically-correct parent + fine reaction transport'}
 current=tick=0
ctx=ComputeContext();sim=DetailTransport(ctx,scene,cfg,a.parent/'initial-q.npy',a.parent)
if statepath.exists():
 sim.q.write(np.load(statepath));sim.tick=tick
renderer=VolumeRenderer(ctx,sim,optical_workspace=sim.next,lighting_scale=4)
for f in range(current,min(current+a.batch,a.warmup+a.frames)):
 sim.set_parent_pair(f,f+1)
 t=time.perf_counter()
 for j in range(2):sim.step(m['start_time']+f/24+j/48,1/48,(j+.5)/2)
 ctx.finish();simtime=time.perf_counter()-t
 if f>=a.warmup:
  idx=f-a.warmup; renderer.optical_texture=sim.next.handle
  t=time.perf_counter()
  im=renderer.render_tiled_rgb(width=a.width,height=a.height,samples=a.samples,seed=idx,prepare=True,band_height=64)
  render_time=time.perf_counter()-t
  path=framesdir/f'{idx:04d}.png';Image.fromarray(im).save(path)
  lum=float(np.mean(im.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)))
  bright_pixels=int(np.count_nonzero(np.max(im,axis=2)>165))
  # The actual active source is at z=.96 m, so sample a narrow slab around z=.96.
  slab=sim.q.read_slab(int(.96/sim.h)-4,8)
  source_heat=float(slab[...,2].max()); source_fuel=float(slab[...,0].max())
  record['frames'].append({'index':idx,'time':m['start_time']+(f+1)/24,'simulation_seconds':simtime,'render_seconds':render_time,
      'sha256':digest(path),'mean_luminance_8bit':lum,'bright_pixel_count':bright_pixels,
      'local_max_fuel_density':source_fuel,'local_max_heat_density':source_heat})
  print('FRAME',idx,'brightpx',bright_pixels,'fuel',round(source_fuel,6),'heat',round(source_heat),'render',round(render_time,2),flush=True)
  if idx in (0,12,a.frames-1):Image.fromarray(im).save(a.out/f'still_{idx:03d}.png')
  del im
 else:
  print('WARMUP',f+1,'simulation_seconds',round(simtime,2),flush=True)
 record['current']=f+1;record['tick']=sim.tick
 temp=a.out/'state.tmp.npy';np.save(temp,sim.q.read());temp.replace(statepath)
 recordpath.write_text(json.dumps(record,indent=2));gc.collect()
renderer.close();sim.close();ctx.close()
if record['current']==a.warmup+a.frames:
 video=a.out/'corrected_fire_24fps.mp4'
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','24','-i',str(framesdir/'%04d.png'),'-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(video)]))
 assert int(probe['streams'][0]['nb_read_frames'])==a.frames
 subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
 record['status']='complete_verified';record['video_sha256']=digest(video);record['probe']=probe
 recordpath.write_text(json.dumps(record,indent=2));print('COMPLETE',video)
else:print('PROGRESS',record['current'],'of',a.warmup+a.frames)
