"""Resume original-resolution CYBR FIRE V8 inside a 4K frame, one row-band/process.

advance: integrate exactly one V8 video frame, saving the 99M-cell fine state.
band: render one immutable full-width 4K band, with global pixel coordinates.
assemble: verify all band hashes and compose a lossless native 4K frame.
"""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib,json,sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.gpu import ComputeContext
from cybrfire.config import scene_from_dict
from cybrfire.detail import DetailTransport,DetailConfig,upload_npy_slabs
from cybrfire.render import VolumeRenderer,tone_map
from quality_profiles import get_profile,require_native

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def publish(p,data):
 tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def load(parent,state,scene_override=None):
 m=json.loads((parent/'manifest.json').read_text()); scene=scene_from_dict(m['scene'])
 profile=get_profile('native-v8')
 require_native(scene.domain.shape,profile.fine,(3840,2160))
 render=replace(scene.render,width=3840,height=2160,samples=2,exposure=.009,
  camera_distance=1.25,fov=42.,target_height=.65,bloom=0.,floor=False,key_intensity=0.,
  shadow_resolution_scale=8,reconstruction_mode=1,empty_space_skipping=False,step_scale=.60)
 scene=replace(scene,render=render)
 cfg=DetailConfig(shape=profile.fine,flow_scale=profile.flow_scale,rms_velocity=.4,wavelengths=profile.wavelengths)
 ctx=ComputeContext()
 sim=DetailTransport(ctx,scene,cfg,parent/'initial-q.npy',parent)
 upload_npy_slabs(sim.q,state)
 return m,ctx,sim

def save(field,dest):
 nx,ny,nz=field.grid_shape;tmp=dest.with_name('.'+dest.name)
 mm=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=field.shape)
 for z in range(0,nz,8):
  n=min(8,nz-z);mm[z*nx*ny:(z+n)*nx*ny]=field.read_slab(z,n).reshape(-1,4)
 mm.flush();del mm;tmp.replace(dest)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('mode',choices=['advance','band','assemble'])
 ap.add_argument('--parent',type=Path,required=True)
 ap.add_argument('--capture',type=Path,required=True)
 ap.add_argument('--band-height',type=int,default=24)
 ap.add_argument('--band-index',type=int)
 a=ap.parse_args();p=a.capture;record_path=p/'band_resume.json';cap=json.loads((p/'capture.json').read_text())
 if cap['current']<4:raise RuntimeError('Native warmup not completed')
 if a.mode=='advance':
  if record_path.exists():
   print('EXISTING_ADVANCE',record_path);return
  f=cap['current']
  m,ctx,sim=load(a.parent,p/cap['checkpoint_npy'])
  try:
   if len(m['frames'])<=f+1:raise RuntimeError('More parent states required')
   sim.tick=cap['tick'];sim.set_parent_pair(f,f+1)
   for j in range(2):sim.step(m['start_time']+f/24+j/48,1/48,(j+.5)/2)
   ctx.finish()
   state=p/f'band-state-{f+1:05d}.npy';save(sim.q,state)
   record=dict(status='advanced',fine_step_from=f,fine_step_to=f+1,
    video_frame=f-4,sim_time=(f+1)/24,tick=sim.tick,state_path=state.name,
    state_sha256=None,resolution=[3840,2160],parent_grid=[160,280,160],fine_grid=[384,672,384],
    band_height=a.band_height,bands={})
   publish(record_path,record);print('ADVANCE_COMMITTED',state,flush=True)
  finally:sim.close();ctx.close()
  return
 if not record_path.exists():raise RuntimeError('Run advance first')
 record=json.loads(record_path.read_text());bh=record['band_height'];total=(2160+bh-1)//bh
 if a.mode=='band':
  index=next((i for i in range(total) if str(i) not in record['bands']),None) if a.band_index is None else a.band_index
  if index is None:print('ALL_BANDS_COMPLETE');return
  if not 0<=index<total:raise ValueError('Invalid band')
  if str(index) in record['bands']:print('BAND_ALREADY_COMPLETE',index);return
  _,ctx,sim=load(a.parent,p/record['state_path'])
  try:
   sim.tick=record['tick'];r=VolumeRenderer(ctx,sim,optical_workspace=sim.next,lighting_scale=4)
   y=index*bh;h=min(bh,2160-y)
   r.optical_texture=sim.next.handle
   hdr=r.render_hdr(width=3840,height=2160,samples=2,seed=record['video_frame'],prepare=True,region=(0,y,3840,h))
   pixels=tone_map(hdr,r.cfg.exposure,0.)
   dst=p/'bands'/f'{record["video_frame"]:04d}'/f'{index:04d}.png';dst.parent.mkdir(parents=True,exist_ok=True)
   Image.fromarray(pixels).save(dst)
   record['bands'][str(index)]=dict(path=str(dst.relative_to(p)),sha256=digest(dst),y=y,height=h)
   if len(record['bands'])==total:record['status']='bands_complete'
   publish(record_path,record)
   print('BAND_COMMITTED',index+1,'/',total,'rows',y,'-',y+h,flush=True)
   r.close()
  finally:sim.close();ctx.close()
  return
 if len(record['bands'])!=total:raise RuntimeError(f'Only {len(record["bands"])} / {total} bands complete')
 canvas=np.empty((2160,3840,3),dtype=np.uint8)
 for i in range(total):
  entry=record['bands'][str(i)];src=p/entry['path']
  if digest(src)!=entry['sha256']:raise RuntimeError('Band hash failed: '+str(src))
  arr=np.asarray(Image.open(src).convert('RGB'))
  if arr.shape!=(entry['height'],3840,3):raise RuntimeError('Unexpected band shape')
  canvas[entry['y']:entry['y']+entry['height']]=arr
 dst=p/'frames'/f'{record["video_frame"]:04d}.png';dst.parent.mkdir(exist_ok=True)
 Image.fromarray(canvas).save(dst)
 record['status']='assembled';record['assembled_frame']=str(dst.relative_to(p));record['frame_sha256']=digest(dst)
 # Commit only after every band and final lossless frame are validated. This
 # preserves the exact pre-advance capture checkpoint on interrupted renders.
 if cap['current']==record['fine_step_from']:
  next_record=dict(cap)
  if [x['index'] for x in next_record['frames']]!=list(range(len(next_record['frames']))):
   raise RuntimeError('Nonconsecutive historical render frames')
  next_record['frames'].append(dict(index=record['video_frame'],sim_time=record['sim_time'],
   sha256=record['frame_sha256'],bright_flame_px=int(np.count_nonzero((canvas[:,:,0]>170)&(canvas[:,:,0]>canvas[:,:,2]*1.2))),
   simulation_wall_seconds=None,render_wall_seconds=None,rendering_method='verified_independent_4K_bands'))
  next_record['current']=record['fine_step_to'];next_record['tick']=record['tick']
  next_record['checkpoint_npy']=record['state_path']
  publish(p/'capture.json',next_record)
  record['status']='committed'
 publish(record_path,record);print('FRAME_ASSEMBLED_AND_COMMITTED',dst,flush=True)
if __name__=='__main__':main()
