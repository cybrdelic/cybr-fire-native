from pathlib import Path
from dataclasses import replace
import json,sys,time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import scene_from_dict
from cybrfire.detail import DetailConfig, DetailTransport
from cybrfire.gpu import ComputeContext
from cybrfire.render import VolumeRenderer
base=Path('outputs/fire_fix'); parent=base/'parent'; out=base/'fire_proof.png'
m=json.loads((parent/'manifest.json').read_text());scene=scene_from_dict(m['scene'])
scene=replace(scene,render=replace(scene.render,bloom=0.,floor=False,width=720,height=1280,
 fov=40.,camera_distance=3.4,camera_height=1.6,target_height=1.05,yaw=12.,
 empty_space_skipping=False,key_intensity=0.,shadow_resolution_scale=4,
 reconstruction_mode=1,step_scale=.60,samples=2,stable_sampling=False))
cfg=DetailConfig(shape=(192,336,192),flow_scale=1,rms_velocity=.4,wavelengths=(.16,.08,.05))
t=time.perf_counter();ctx=ComputeContext();sim=DetailTransport(ctx,scene,cfg,parent/'initial-q.npy',parent)
print('setup',time.perf_counter()-t,flush=True)
sim.q.write(np.load(base/'capture/state.npy'))
r=VolumeRenderer(ctx,sim,optical_workspace=sim.next,lighting_scale=4)
t=time.perf_counter();img=r.render_tiled_rgb(width=720,height=1280,samples=2,seed=3,prepare=True,band_height=64)
print('render',time.perf_counter()-t,flush=True)
Image.fromarray(img).save(out);print(out, 'bright_pixels',int(np.count_nonzero(np.max(img,axis=2)>165)),flush=True)
r.close();sim.close();ctx.close()
