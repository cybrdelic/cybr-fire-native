"""Optical-only camera diagnostic from an already solved native fine state.

Never interpolates or generates motion, and never changes numerical grid.
A low raster here is only a camera/exposure preview, not the deliverable.
"""
from pathlib import Path
from dataclasses import replace
import argparse, json, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import scene_from_dict
from cybrfire.detail import DetailTransport, DetailConfig, upload_npy_slabs
from cybrfire.gpu import ComputeContext
from cybrfire.render import VolumeRenderer
from quality_profiles import get_profile

p=argparse.ArgumentParser()
p.add_argument('--parent',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--width',type=int,default=1280)
p.add_argument('--height',type=int,default=720)
p.add_argument('--x',type=float,default=.65)
p.add_argument('--target-height',type=float,default=.82)
p.add_argument('--distance',type=float,default=3.25)
p.add_argument('--fov',type=float,default=42.)
p.add_argument('--exposures',type=float,nargs='+',default=[.25,.42,.65])
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
m=json.loads((a.parent/'manifest.json').read_text()); profile=get_profile('native-v8')
scene=scene_from_dict(m['scene'])
scene=replace(scene,render=replace(scene.render, camera_target_x=a.x,target_height=a.target_height,camera_distance=a.distance,
                                   fov=a.fov,yaw=0.,bloom=0.,floor=False,step_scale=.60,reconstruction_mode=1,
                                   empty_space_skipping=False,shadow_resolution_scale=8,samples=2))
cfg=DetailConfig(shape=profile.fine,flow_scale=profile.flow_scale,rms_velocity=.4,wavelengths=profile.wavelengths)
ctx=ComputeContext()
try:
 t=time.perf_counter()
 sim=DetailTransport(ctx,scene,cfg,a.parent/'initial-q.npy',a.parent)
 upload_npy_slabs(sim.q,a.checkpoint)
 print('RESTORED',time.perf_counter()-t,flush=True)
 r=VolumeRenderer(ctx,sim,optical_workspace=sim.next,lighting_scale=4)
 for exp in a.exposures:
  r.cfg=replace(r.cfg,exposure=exp)
  t=time.perf_counter()
  im=r.render_tiled_rgb(width=a.width,height=a.height,samples=2,seed=5,prepare=True,band_height=64)
  filename=a.out/f'jet_native_optical_exp{exp:.4f}.png';Image.fromarray(im).save(filename)
  bright=int(np.count_nonzero(np.max(im,axis=2)>240));orange=int(np.count_nonzero((im[:,:,0]>100)&(im[:,:,1]<im[:,:,0])&(im[:,:,0]>im[:,:,2]*1.5)))
  print('FRAME',filename,'seconds',time.perf_counter()-t,'bright',bright,'orange',orange,flush=True)
 r.close();sim.close()
finally:ctx.close()
