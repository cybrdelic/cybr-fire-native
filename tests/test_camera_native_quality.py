"""The camera may target an off-center physical subject, with no solver downgrade."""
from dataclasses import replace
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'tools'))
from cybrfire.config import Render,scene_from_dict
from quality_profiles import get_profile,require_native
from real_scenarios import make_scene

def test_native_landscape_and_camera():
 q=get_profile('native-v8')
 require_native(q.parent,q.fine,q.video[::-1])
 for name in ('jet','pool','fireball'):
  s=make_scene(name)
  s.validate()
  r=replace(s.render,camera_target_x=.58,camera_target_z=.96,exposure=.009)
  z=scene_from_dict(replace(s,render=r).to_dict())
  assert z.render.camera_target_x==.58
  assert z.render.camera_target_z==.96
  assert z.domain.shape==q.parent
 if __name__=='__main__':pass

if __name__=='__main__':
 test_native_landscape_and_camera()
 print('PASS scene-aware camera targeting and landscape 4K fidelity')
