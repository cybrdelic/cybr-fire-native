"""Actual GPU dispatch/ground-light integration and error-boundary regressions."""
import sys,json,time,ctypes as C
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.gpu import ComputeContext,bind_texture
from cybrfire.config import example,Domain
from cybrfire.solver import Simulation
from cybrfire.render import VolumeRenderer,blackbody_lut
R=Path(__file__).resolve().parents[1];results=[]
def check(n,p,**kw):results.append(dict(test=n,passed=bool(p),**kw));print(('PASS ' if p else 'FAIL ')+n+' '+json.dumps(kw),flush=True)
def hash32(x):
    x^=x>>16;x=(x*0x7feb352d)&0xffffffff;x^=x>>15;x=(x*0x846ca68b)&0xffffffff;return x^(x>>16)
def main():
 c=ComputeContext();start=time.perf_counter();scene=example('burner','test');scene=replace(scene,domain=Domain((16,24,16),.1,pressure_backend='spectral'),render=replace(scene.render,width=16,height=24,ground_resolution=16,ground_directions=64,scatter_albedo=0.,ambient=0.,bloom=0.))
 a=Simulation(c,scene);b=Simulation(c,scene);a.step(4);c.dispatch_width=3;b.step(4);c.dispatch_width=65535
 check('two_dimensional_dispatch_equivalence',np.array_equal(a.q.read(),b.q.read()) and np.array_equal(a.vel.read(),b.vel.read()),forced_X_workgroups=3)
 old=a.snapshot();original=a.mg.solve
 def fail(*args,**kwargs):raise RuntimeError('Test-injected solver failure')
 a.mg.solve=fail;raised=False
 try:a.step()
 except RuntimeError:raised=True
 blocked=False
 try:a.snapshot()
 except RuntimeError:blocked=True
 check('partial_step_failure_blocks_publication',raised and blocked and a.fault is not None and old['tick']==4)
 a.mg.solve=original
 # Ground irradiance is compared with independently evaluated ray/box
 # intersections and the closed-form homogeneous transfer equation.
 s=Simulation(c,scene);r=VolumeRenderer(c,s);state=np.zeros((s.n,4),np.float32);sigma=.45;temperature=1800.
 state[:,2]=(temperature-300.)*scene.chemistry.heat_capacity;state[:,3]=sigma/scene.render.soot_extinction;s.q.write(state);r.prepare()
 c._fn('glGetTexImage',None,C.c_uint,C.c_int,C.c_uint,C.c_uint,C.c_void_p)
 def ground_data():
    out=np.empty((16,1,16,4),np.float32);bind_texture(c,r.ground_texture,5);c.glGetTexImage(0x806F,0,0x1908,0x1406,out.ctypes.data);return out[:,0,:,:3]
 measured=ground_data();table=blackbody_lut();pos=(temperature-300)/5700*511;lo=int(pos);B=table[lo,:3]*(1-(pos-lo))+table[lo+1,:3]*(pos-lo)
 size=np.array(s.shape)*s.h
 for gx,gz in [(8,8),(3,3),(12,6)]:
    n=gx+gz*16;rotation=2*np.pi*((hash32(n*173+7919)>>8)/16777216.)
    ro=np.array(r.patch_origin)+np.array([(gx+.5)/16*r.patch_extent[0],.001,(gz+.5)/16*r.patch_extent[2]])
    value=np.zeros(3)
    for k in range(64):
        u=(k+.5)/64;phi=2.39996322973*k+rotation;rad=np.sqrt(u);d=np.array([rad*np.cos(phi),np.sqrt(1-u),rad*np.sin(phi)])
        t0=-ro/d;t1=(size-ro)/d;entry=max(float(np.minimum(t0,t1).max()),0.);exit=float(np.maximum(t0,t1).min());length=max(exit-entry,0.)
        value+=B*(-np.expm1(-sigma*length))/64
    error=float(np.max(abs(measured[gz,gx]-value))/max(float(value.max()),1e-9))
    check('ground_hemisphere_integral_'+str((gx,gz)),error<3e-4,relative_error=error,measured=measured[gz,gx].tolist(),expected=value.tolist())
 s.q.zero();r.prepare();check('empty_volume_has_no_fire_ground_light',float(np.max(abs(ground_data())))==0)
 # With scattering disabled the volume emission is exactly kappa*B. With
 # scattering enabled it uses absorption, kappa*(1-albedo), not extinction.
 s.q.write(state);r.cfg=replace(r.cfg,scatter_albedo=.4);r.prepare();optical=np.empty((s.shape[2],s.shape[1],s.shape[0],4),np.float32);bind_texture(c,r.optical_texture,0);c.glGetTexImage(0x806F,0,0x1908,0x1406,optical.ctypes.data)
 value=optical[4,4,4];expected=B*sigma*.6;err=float(np.max(abs(value[:3]-expected))/float(expected.max()))
 check('thermal_emission_uses_absorption',err<2e-6,relative_error=err)
 c.check();report={'adapter':c.info,'passed':sum(x['passed'] for x in results),'failed':sum(not x['passed'] for x in results),'wall_seconds':time.perf_counter()-start,'tests':results};(R/'evidence/runtime-tests.json').write_text(json.dumps(report,indent=2));c.close();return report['failed']
if __name__=='__main__':sys.exit(bool(main()))
