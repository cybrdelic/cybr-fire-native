"""Executed V7 storage, complete-operator and render parity regressions."""
from dataclasses import replace
from pathlib import Path
import sys,json,time,tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.gpu import ComputeContext
from cybrfire.config import example,Domain
from cybrfire.solver import Simulation
from cybrfire.render import VolumeRenderer
from cybrfire.image_fields import ImageField
rows=[]
def check(name,cond,**metrics):
 rows.append(dict(name=name,passed=bool(cond),**metrics));print(('PASS' if cond else 'FAIL'),name,metrics,flush=True)
 if not cond:raise AssertionError(name)
ctx=ComputeContext()
a=example('ember-burner','test');a=replace(a,domain=replace(a.domain,pressure_backend='spectral'))
b=replace(a,domain=replace(a.domain,native_fields=True))
s=Simulation(ctx,a);t=Simulation(ctx,b)
for count in [1,3,8]:
 s.step(count);t.step(count)
 for name,x,y in [('gas',s.q,t.q),('velocity',s.vel,t.vel),('particles',s.embers.state,t.embers.state)]:
  A=x.read();B=y.read();err=float(np.max(np.abs(A-B)));check(f'{name}_parity_tick_{s.tick}',err==0,maximum_absolute_error=err)
for kind in (0,1):
 for periodic in (False,True):
  f=(s.vel,s.vel_pred,s.vel_out) if kind else (s.q,s.q_pred,s.q_out)
  g=(t.vel,t.vel_pred,t.vel_out) if kind else (t.q,t.q_pred,t.q_out)
  s.advect(*f,s.vel,kind,.04,periodic=periodic);t.advect(*g,t.vel,kind,.04,periodic=periodic)
  err=float(np.max(np.abs(f[2].read()-g[2].read())))
  check(f'transport_kind_{kind}_periodic_{periodic}',err==0,maximum_absolute_error=err)
r=VolumeRenderer(ctx,s);rr=VolumeRenderer(ctx,t)
A=r.render_hdr(width=128,height=192);B=rr.render_hdr(width=128,height=192);err=float(np.max(abs(A-B)))
check('render_hdr_parity',err==0,maximum_absolute_error=err)
check('scratch_aliases_are_predictors',t.omega is t.q_pred and t.budget is t.q_pred and t.assembled_force is t.vel_pred)
check('published_state_disjoint_from_scratch',t.q not in (t.q_pred,t.vel_pred) and t.vel not in (t.q_pred,t.vel_pred))
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'state.npz';t.save(p);q=t.q.read();t.step(1);t.load(p)
 check('native_checkpoint_restore',np.array_equal(t.q.read(),q))
 for f in (t.q,t.vel):
  f.write(np.ones(f.shape,np.float32));f.zero();check('native_zero_'+str(f.shape[0]),not np.any(f.read()))
check('new_grid_budget',Domain(shape=(224,392,224),spacing=1.92/224,native_fields=True).validate() is None)
try:Domain(shape=(224,392,224),spacing=1.92/224).validate();accepted=True
except ValueError:accepted=False
check('oversized_legacy_backend_rejected',not accepted)
r.close();rr.close();ctx.close()
# Each production field exceeds the old adapter SSBO cap. Exercise an actual
# allocation above that cap, not merely the new dimension validator.
ctx=ComputeContext();f=ImageField(ctx,(224,176,224));host=np.zeros(f.shape,np.float32)
indices=[0,8_388_607,8_388_608,f.shape[0]-1]
for i,n in enumerate(indices):host[n]=[i+.25,i+.5,i+.75,i+1.]
f.write(host);got=f.read();check('beyond_128MiB_roundtrip',np.array_equal(got,host),bytes=f.nbytes,cells=f.shape[0],sentinel_indices=indices)
f.close();f.close();check('field_close_idempotent',f.handle==0);ctx.close()
report={'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows),'checks':rows}
(Path(__file__).resolve().parents[1]/'evidence/native-field-tests.json').write_text(json.dumps(report,indent=2))
