import sys,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import scene_from_dict
from cybrfire.detail import DetailTransport,DetailConfig
from cybrfire.render import VolumeRenderer
from cybrfire.gpu import ComputeContext
scene_name=sys.argv[1] if len(sys.argv)>1 else 'fireball'
root=Path('outputs/fire_upgrade')/scene_name
m=json.load(open(root/'parent/manifest.json'))
record=json.load(open(root/'capture/capture.json'))
s=scene_from_dict(m['scene']);ctx=ComputeContext();sim=DetailTransport(ctx,s,DetailConfig(shape=(144,252,144),flow_scale=1,rms_velocity=.4,wavelengths=(.20,.12,.08)),root/'parent/initial-q.npy',root/'parent')
with np.load(root/'capture/fine-state.npz') as z:sim.q.write(z['q'])
imgs=[]
for exposure in (1.4,.3,.08,.025):
 s0=replace(s,render=replace(s.render,exposure=exposure,width=480,height=800,samples=2))
 sim.scene=s0
 r=VolumeRenderer(ctx,sim,lighting_scale=4)
 img=Image.fromarray(r.render_tiled_rgb(width=480,height=800,samples=2,seed=4,prepare=True,band_height=64))
 r.close()
 imgs.append((exposure,img))
w=960;h=1600
out=Image.new('RGB',(w,h),(0,0,0));d=ImageDraw.Draw(out)
for i,(ex,img) in enumerate(imgs):
 x=i%2*480;y=i//2*800
 out.paste(img,(x,y));d.text((x+10,y+10),f'exposure {ex}',fill='white')
p=root/'exposure_comparison.png';out.save(p);print(p)
sim.close();ctx.close()
