"""Regression tests: no accidental V8 downgrade to preview numerical grids."""
import json, sys, tempfile
from pathlib import Path
from math import prod
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from quality_profiles import get_profile, require_native
from real_scenarios import make_scene
from cybrfire.detail import DetailConfig


def test_default_profile_is_original_v8():
    native=get_profile('native-v8')
    assert native.parent == (160,280,160)
    assert native.fine == (384,672,384)
    assert native.video == (2160,3840)
    assert native.parent_cells==7_168_000
    assert native.fine_cells==99_090_432
    assert native.flow_scale==2
    assert native.wavelengths==(.16,.08,.04)
    for name in ('jet','pool','fireball'):
        scene=make_scene(name)
        assert scene.domain.shape == native.parent
        assert scene.domain.native_fields
        expected_extent=[1.92,3.36,1.92]
        assert all(abs(n*h - e)<1e-10 for n,h,e in zip(scene.domain.shape,[scene.domain.spacing]*3,expected_extent))
        scene.validate()
        cfg=DetailConfig(shape=native.fine,flow_scale=native.flow_scale,wavelengths=native.wavelengths)
        cfg.validate(tuple(expected_extent))
        require_native(scene.domain.shape,cfg.shape,native.video)


def test_preview_rejected_as_native():
    prev=get_profile('preview')
    native=get_profile('native-v8')
    try:
        require_native(prev.parent, prev.fine, prev.video)
    except ValueError:
        pass
    else:
        raise AssertionError('Preview was incorrectly labeled as native')
    assert prev.parent_cells < native.parent_cells
    assert prev.fine_cells < native.fine_cells

if __name__=='__main__':
    test_default_profile_is_original_v8()
    test_preview_rejected_as_native()
    print('PASS production numeric fidelity requirements')
    print('PASS preview/native mismatch rejection')
