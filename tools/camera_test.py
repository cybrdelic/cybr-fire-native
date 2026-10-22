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
scene_name=sys.argv[1] if len(sys.argv)>1 else 'pool'
root=Path('outputs/fire_upgrade')/scene_name
m=json.load(open(root/'parent/manifest.json'))
s=scene_from_dict(m['scene']);ctx=ComputeContext();sim=DetailTransport(ctx,s,DetailConfig(shape=(144,252,144),flow_scale=1,rms_velocity=.4,wavelengths=(.20,.12,.08)),root/'parent/initial-q.npy',root/'parent')
with np.load(root/'capture/fine-state.npz') as z:sim.q.write(z['q'])
opts=[(1.4,40.,3.55,1.3),(.13,55.,4.25,1.6),(.04,55.,4.25,1.6),(.015,55.,4.25,1.6)]
if scene_name=='pool':opts=[(1.4,40.,3.55,1.05),(.15,48.,3.7,1.0),(.04,48.,3.7,1.0),(.015,48.,3.7,1.0)]
imgs=[]
for e,fov,dist,target in opts:
    sc=replace(s,render=replace(s.render,exposure=e,fov=fov,camera_distance=dist,target_height=target,width=480,height=800,samples=2))
    sim.scene=sc;r=VolumeRenderer(ctx,sim,lighting_scale=4)
    img=Image.fromarray(r.render_tiled_rgb(width=480,height=800,samples=2,seed=4,prepare=True,band_height=64))
    r.close();imgs.append(img)
out=Image.new('RGB',(960,1600),(0,0,0));d=ImageDraw.Draw(out)
for i,img in enumerate(imgs):
    x=i%2*480;y=i//2*800
    out.paste(img,(x,y));d.text((x+10,y+10),f'{opts[i]}',fill='white')
p=root/'camera_comparison.png';out.save(p);print(p)
sim.close();ctx.close()
