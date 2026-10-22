"""Native image fields, tiled rays and slab I/O must preserve their contracts."""
from pathlib import Path
from dataclasses import replace
import sys,json,tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import example
from cybrfire.gpu import ComputeContext
from cybrfire.solver import Simulation
from cybrfire.render import VolumeRenderer
rows=[]
def check(name,ok,**data):
 rows.append(dict(name=name,passed=bool(ok),**data));print(('PASS ' if ok else 'FAIL ')+name+' '+json.dumps(data),flush=True)
 if not ok:raise AssertionError(name)
c=ComputeContext();base=example('ember-burner','test');base=replace(base,domain=replace(base.domain,pressure_backend='spectral'),render=replace(base.render,bloom=0.,empty_space_skipping=False))
a=Simulation(c,base);b=Simulation(c,replace(base,domain=replace(base.domain,native_fields=True)));a.step(8);b.step(8)
check('reaction_ledger_alias',np.array_equal(a.budget.read(),b.budget.read()))
for key,field in [('q',b.q),('v',b.vel)]:
 slabs=np.concatenate([field.read_slab(z,min(7,field.grid_shape[2]-z)) for z in range(0,field.grid_shape[2],7)],axis=0).reshape(-1,4)
 check('slab_read_'+key,np.array_equal(slabs,field.read()))
A=a.diagnostics();B=b.diagnostics()
for key in ['integrals','divergence_rms','divergence_max','max_component_speed','min_species','max_species','floor_normal_max']:
 check('diagnostic_'+key,np.allclose(A[key],B[key],rtol=1e-11,atol=1e-11))
r=VolumeRenderer(c,b);hdr=r.render_hdr(width=101,height=167,samples=2);reference=r.render(width=101,height=167,samples=2)
for band in [17,31,64]:
 image=r.render_tiled_rgb(width=101,height=167,samples=2,band_height=band)
 check('tiled_pixels_'+str(band),np.array_equal(image,reference),maximum_8bit_error=int(abs(image.astype(int)-reference).max()))
 check('tiled_ray_accounting_'+str(band),r.statistics()['camera_rays']==101*167*2)
sub=r.render_hdr(width=101,height=167,samples=2,region=(3,7,50,61),prepare=False)
check('arbitrary_region_is_full_frame_crop',np.array_equal(sub,hdr[7:68,3:53]),max_hdr_error=float(abs(sub-hdr[7:68,3:53]).max()))
q=b.q.read();v=b.vel.read();p=b.mg.pressure.read()
with tempfile.TemporaryDirectory() as folder:
 path=Path(folder)/'state.npz';b.save(path)
 with np.load(path,allow_pickle=False) as z:
  check('streamed_checkpoint_version',int(z['version'])==7)
  for name,expected in [('q',q),('velocity',v),('pressure',p)]:check('streamed_checkpoint_'+name,np.array_equal(z[name],expected))
 b.step();b.load(path);check('streamed_restore_state',np.array_equal(b.q.read(),q) and np.array_equal(b.vel.read(),v))
 b.step();a.step();check('streamed_continuation',np.array_equal(a.q.read(),b.q.read()) and np.array_equal(a.vel.read(),b.vel.read()))
for name,action in [('invalid_slab',lambda:b.q.read_slab(-1,3)),('invalid_region',lambda:r.render_hdr(width=101,height=167,region=(99,1,4,4)))]:
 try:action();rejected=False
 except ValueError:rejected=True
 check(name+'_rejected',rejected)
r.cfg=replace(r.cfg,bloom=.1)
try:r.render_tiled_rgb(width=101,height=167);rejected=False
except ValueError:rejected=True
check('nonlocal_bloom_rejected_in_banded_rgb',rejected)
r.close();c.close()
report={'passed':sum(x['passed'] for x in rows),'failed':sum(not x['passed'] for x in rows),'checks':rows}
(Path(__file__).resolve().parents[1]/'evidence/streaming-tests.json').write_text(json.dumps(report,indent=2))
