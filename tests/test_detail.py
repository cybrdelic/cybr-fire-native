"""Executed contracts for V8's one-way fine transport, not a CFD certificate."""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.detail import DetailConfig,transverse_modes
from cybrfire.gpu import ComputeContext,bind_texture,bind_image
from cybrfire.image_fields import ImageField
from cybrfire.solver import SHADERS
out=Path(__file__).resolve().parents[1]/'evidence-v8/detail-tests.json';checks=[]
def check(name,passed,**values):
 checks.append({'name':name,'passed':bool(passed),**values});print('PASS' if passed else 'FAIL',name,values,flush=True)

a=transverse_modes(DetailConfig());dots=np.sum(a[::2,:3].astype(float)*a[1::2,:3],axis=1)
check('transverse_mode_analytic_divergence',max(abs(dots))<2e-5,maximum_k_dot_p=float(max(abs(dots))))
check('mode_seed_repeatability',np.array_equal(a,transverse_modes(DetailConfig())))
check('declared_spectrum_norm',abs(float(np.sum(a[1::2,:3]**2)*.5)-1)<1e-6)
try:DetailConfig(wavelengths=(.001,)).validate((1.92,3.36,1.92));ok=False
except ValueError:ok=True
check('underresolved_spectral_modes_rejected',ok)
ctx=ComputeContext();shape=(48,64,48);spacing=.02;extent=tuple(np.array(shape)*spacing)
f=ImageField(ctx,shape);g=ImageField(ctx,shape);v=ImageField(ctx,(8,8,8));k=ctx.program(SHADERS/'detail_advect.comp')
z,y,x=np.indices(shape[::-1],dtype=np.float32);r2=((x-17.25)**2+(y-32.5)**2+(z-24.5)**2)
q=np.zeros((*shape[::-1],4),np.float32);q[...,0]=np.exp(-r2/25);q[...,1]=q[...,0]*.2;q[...,2]=q[...,0]*1e6;q[...,3]=q[...,0]*.001
vel=np.zeros((512,4),np.float32);vel[:,0]=.17;v.write(vel)
results=[]
for cubic in (0,1):
 f.write(q.reshape(-1,4));mass0=float(q[...,0].sum(dtype=np.float64));t=time.perf_counter()
 for step in range(20):
  bind_texture(ctx,f.handle,0);bind_texture(ctx,v.handle,1);bind_image(ctx,g.handle,0)
  k.run(shape,[],dims=shape,spacing=spacing,dt=.01,extent=extent,cubic=cubic,traceBudget=1.5,traceLimit=64);f,g=g,f
 result=f.read().reshape(*shape[::-1],4)
 exact=np.exp(-((x-(17.25+.17*.2/.02))**2+(y-32.5)**2+(z-24.5)**2)/25)
 error=float(np.linalg.norm(result[...,0]-exact)/np.linalg.norm(exact));drift=float(result[...,0].sum(dtype=np.float64)/mass0-1)
 results.append({'cubic':cubic,'error':error,'mass_drift':drift,'seconds':time.perf_counter()-t})
 check(f'nonnegative_transport_{cubic}',result.min()>=0,minimum=float(result.min()))
 check(f'bounded_transport_{cubic}',float(result[...,0].max())<=float(q[...,0].max())+1e-6)
check('cubic_translation_error_lower',results[1]['error']<results[0]['error'],measurements=results)
# Zero velocity: actual compute must exactly preserve the field at grid centers.
v.zero();f.write(q.reshape(-1,4));bind_texture(ctx,f.handle,0);bind_texture(ctx,v.handle,1);bind_image(ctx,g.handle,0)
k.run(shape,[],dims=shape,spacing=spacing,dt=.01,extent=extent,cubic=1,traceBudget=1.5,traceLimit=64)
err=float(np.max(np.abs(g.read()-q.reshape(-1,4))))
check('zero_velocity_stationarity',err<2.,maximum_energy_error=err)
# Constant state is preserved away from open boundaries; no hidden detail source.
constant=np.tile(np.array([.02,.15,1e6,.001],np.float32),(np.prod(shape),1));f.write(constant)
bind_texture(ctx,f.handle,0);bind_image(ctx,g.handle,0)
k.run(shape,[],dims=shape,spacing=spacing,dt=.01,extent=extent,cubic=1,traceBudget=1.5,traceLimit=64)
check('constant_state_stationarity',np.allclose(g.read(),constant,rtol=2e-6,atol=1e-8))
ctx.close();report={'passed':sum(x['passed'] for x in checks),'failed':sum(not x['passed'] for x in checks),'checks':checks};out.write_text(json.dumps(report,indent=2));raise SystemExit(bool(report['failed']))
