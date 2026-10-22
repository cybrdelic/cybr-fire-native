"""Temporal, media and finite surface-fuel mass-ledger acceptance tests."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.real_scenarios import make_scene
from cybrfire.gpu import ComputeContext
from cybrfire.image_fields import ImageField
from cybrfire.surface_fuel import SurfaceCombustion

def mass_balance_test():
    scene=make_scene('pool',(32,56,32)); ctx=ComputeContext()
    try:
        q=ImageField(ctx,scene.domain.shape)
        surf=SurfaceCombustion(ctx,scene,scene.domain.shape,scene.domain.spacing)
        initial=surf.initial_inventory
        for _ in range(10):surf.step(scene.domain.tick,q,scene.chemistry.heat_capacity)
        gas=np.sum(q.read()[:,0],dtype=np.float64)*scene.domain.spacing**3
        inventory=surf.inventory()
        # Pyrolysis gas mass is in 2 grid layers, from the 2D fuel budget.
        error=abs((initial-inventory['fuel_kg'])-gas)
        max_tol=1e-5*max(initial,1.)
        assert error<max_tol,(error,max_tol)
        assert inventory['char_kg']>0
        assert abs(inventory['fuel_kg']+inventory['char_kg']-initial)<max_tol
        return dict(initial_kg=initial,remaining_kg=inventory['fuel_kg'],pyrolyzed_kg=inventory['char_kg'],gas_injected_kg=gas,absolute_error_kg=error,pass_=True)
    finally:ctx.close()

def temporal_test(root,scene,capture):
    p=root/scene/'parent';d=root/scene/capture
    m=json.loads((p/'manifest.json').read_text())
    r=json.loads((d/'capture.json').read_text())
    assert m['complete'] and len(m['frames'])==29
    assert r['status']=='completed_verified',r['status']
    assert len(r['frames'])==24
    hashes=[f['sha256'] for f in r['frames']]
    assert len(set(hashes))==24
    pngs=sorted((d/'frames').glob('*.png'))
    assert len(pngs)==24
    # Resize and blur to reject mere shader sampling/antialias noise as motion.
    pixels=[]
    for i,f in enumerate(pngs):
        assert f.name==f'{i:04d}.png'
        raw=f.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==hashes[i]
        im=Image.open(f).convert('RGB').resize((90,160),Image.Resampling.BICUBIC)
        pixels.append(np.asarray(im,np.int16))
    deltas=[float(np.mean(np.abs(pixels[i+1]-pixels[i]))) for i in range(23)]
    assert min(deltas)>.05,('Frozen consecutive frames',min(deltas))
    frames=r['frames'];times=[f['sim_time'] for f in frames]
    assert all(abs((times[i+1]-times[i])-1/24)<1e-8 for i in range(23))
    video=d/f'{scene}_720x1280_24fps.mp4'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,duration,nb_read_frames','-of','json',str(video)]))['streams'][0]
    assert int(probe['nb_read_frames'])==24
    assert int(probe['width'])==720 and int(probe['height'])==1280
    assert probe['r_frame_rate']=='24/1'
    assert abs(float(probe['duration'])-1.)<1e-5
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    return dict(scene=scene,video=str(video),frames=24,fps=24,duration_seconds=1.,resolution=[720,1280],
                unique_frame_sha256=24,consecutive_mean_abs_diff_8bit_min=min(deltas),
                consecutive_mean_abs_diff_8bit_median=float(np.median(deltas)),
                first_frame_time=times[0],last_frame_time=times[-1],
                bright_flame_pixels_first=frames[0]['bright_flame_px'],bright_flame_pixels_last=frames[-1]['bright_flame_px'],
                probe=probe,temporal_motion_verified=True)


def main():
    root=Path('outputs/fire_upgrade')
    results=dict(surface_mass_balance=mass_balance_test(),scenes={},engine_limitations=[
      'Parent grid 64x112x64, fine reactive scalar grid 144x252x144; not original V8 4K grid',
      'Ray-integrated 720x1280, not native 2160x3840',
      'Fluid model uses low-Mach pressure projection; fireball radial impulse is not a compressible shock solver',
      'Surface phase-change+entrainment are calibrated VFX closures, not laboratory-validated combustion models',
      'No vegetation geometry, ember-driven spread, solid heat transfer or structural-fire coupling in this build',
      'Each scene 1.0s; original requested 5-10 seconds not reached'])
    for name,capture in [('jet','capture'),('pool','capture_framed'),('fireball','capture_framed')]:
        results['scenes'][name]=temporal_test(root,name,capture)
    m=json.load(open(root/'pool/parent/manifest.json'))
    s0=m['frames'][0]['diagnostics']['surface_fuel']; sf=m['frames'][-1]['diagnostics']['surface_fuel']
    assert sf['fuel_kg']<s0['fuel_kg']
    assert sf['char_kg']>s0['char_kg']
    assert sf['max_surface_temperature']>m['scene']['surface_fuel']['ignition_temperature']
    results['pool_mass_trajectory']={'initial':s0,'final':sf,'fuel_consumed_kg':s0['fuel_kg']-sf['fuel_kg'],'mass_balance_error_kg':abs((s0['fuel_kg']-sf['fuel_kg'])-(sf['char_kg']-s0['char_kg']))}
    p=root/'V8_UPGRADE_VALIDATION.json';p.write_text(json.dumps(results,indent=2))
    print(json.dumps({'passed':True,'mass_balance':results['surface_mass_balance'],'clips':{k:v['consecutive_mean_abs_diff_8bit_median'] for k,v in results['scenes'].items()},'pool':results['pool_mass_trajectory']},indent=2))
if __name__=='__main__':main()
