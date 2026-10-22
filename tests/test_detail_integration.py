"""Cache/transport/render ownership checks on real small-grid compute."""
from pathlib import Path
import sys,json,numpy as np
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import scene_from_dict
from cybrfire.detail import DetailTransport,DetailConfig
from cybrfire.gpu import ComputeContext
from cybrfire.render import VolumeRenderer
root=Path(__file__).resolve().parents[1];cache=root/'cache/parent';m=json.loads((cache/'manifest.json').read_text())
scene=scene_from_dict(m['scene']);scene=replace(scene,render=replace(scene.render,reconstruction_mode=1,floor=False,bloom=0.,empty_space_skipping=False,shadow_resolution_scale=8))
cfg=DetailConfig(shape=(128,224,128),flow_scale=1,wavelengths=(.24,.12,.06));checks=[]
def check(name,passed,**kw):checks.append({'name':name,'passed':bool(passed),**kw});print('PASS' if passed else 'FAIL',name,kw,flush=True)
ctx=ComputeContext();s=DetailTransport(ctx,scene,cfg,cache/'initial-q.npy',cache)
s.set_parent_pair(0,1);s.step(m['start_time'],1/48,.25);a=s.q.read()
check('fine_step_finite_nonnegative',np.isfinite(a).all() and a.min()>=0)
check('fine_step_oxygen_bounded',a[:,1].max()<=scene.chemistry.oxygen_ambient+1e-6)
check('FP32_combustion_storage',ctx.texture_formats[s.q.handle]==0x8814 and ctx.texture_formats[s.next.handle]==0x8814)
check('declared_FP16_velocity_cache',ctx.texture_formats[s.parent[0].handle]==0x881A and ctx.texture_formats[s.flow_field.handle]==0x881A)
r=VolumeRenderer(ctx,s,optical_workspace=s.next,lighting_scale=4);im=r.render(width=192,height=256)
check('nonblack_fine_render',im.max()>80)
check('render_leaves_current_state_unchanged',np.array_equal(a,s.q.read()))
next_handle=s.next.handle;r.close();check('borrowed_optical_workspace_survives_renderer_close',next_handle in ctx.textures)
s.step(m['start_time']+1/48,1/48,.75);rendered_path=s.q.read();s.close()
s=DetailTransport(ctx,scene,cfg,cache/'initial-q.npy',cache);s.set_parent_pair(0,1)
s.step(m['start_time'],1/48,.25);s.step(m['start_time']+1/48,1/48,.75)
err=float(np.max(np.abs(s.q.read()-rendered_path)))
check('intervening_render_does_not_change_next_state',err==0.,maximum_absolute_error=err)
s.close();ctx.close()
report={'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'checks':checks}
(root/'evidence-v8/detail-integration-tests.json').write_text(json.dumps(report,indent=2));raise SystemExit(bool(report['failed']))
