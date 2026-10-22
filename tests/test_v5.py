"""V5-specific numerical, optical, acceleration, persistence, and resource tests."""
import sys,json,time,tempfile,ctypes as C
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import example,Domain,scene_from_dict
from cybrfire.gpu import ComputeContext,bind_texture
from cybrfire.solver import Simulation
from cybrfire.render import VolumeRenderer
from cybrfire.lighting import quadrature
ROOT=Path(__file__).resolve().parents[1];results=[]
def check(name,passed,**metrics):
    results.append(dict(test=name,passed=bool(passed),**metrics));print(('PASS ' if passed else 'FAIL ')+name+' '+json.dumps(metrics),flush=True)
def rejects(fn):
    try:fn()
    except (ValueError,RuntimeError):return True
    return False

def texture_read(c,handle,shape):
    if not hasattr(c,'glGetTexImage'):c._fn('glGetTexImage',None,C.c_uint,C.c_int,C.c_uint,C.c_uint,C.c_void_p)
    out=np.empty((shape[2],shape[1],shape[0],4),np.float32);bind_texture(c,handle,1)
    c.glGetTexImage(0x806F,0,0x1908,0x1406,out.ctypes.data);return out

def main():
    start=time.perf_counter();ctx=ComputeContext();rng=np.random.default_rng(814)
    scene=example('burner','test');scene=replace(scene,domain=Domain((24,32,24),.08,pressure_backend='spectral'),
        render=replace(scene.render,width=72,height=96,ground_resolution=16,ground_directions=16,traversal_policy="forced"))
    for count in (6,14):
        nodes=quadrature(count);w=np.array([x[1] for x in nodes]);v=np.array([x[0] for x in nodes],float);v/=np.linalg.norm(v,axis=1)[:,None]
        check(f'quadrature_{count}_normalization',abs(w.sum()-1)<1e-14 and w.min()>0,weight_sum=float(w.sum()))
        first=np.einsum('n,ni->i',w,v);second=np.einsum('n,ni,nj->ij',w,v,v)
        check(f'quadrature_{count}_isotropy',np.max(abs(first))<1e-14 and np.max(abs(second-np.eye(3)/3))<1e-14)
    fourth=np.einsum('n,ni->i',w,v**4)
    check('quadrature_14_fourth_moment',np.max(abs(fourth-.2))<1e-14,max_error=float(np.max(abs(fourth-.2))))
    invalid=[replace(scene,domain=replace(scene.domain,max_trace_steps=0)),replace(scene,render=replace(scene.render,brick_size=3)),
        replace(scene,render=replace(scene.render,illumination_directions=12)),replace(scene,render=replace(scene.render,scattering_iterations=0)),
        replace(scene,emitters=(replace(scene.emitters[0],attack=2.,release=2.,end=3.),))]
    check('v5_invalid_configs_rejected',all(rejects(s.validate) for s in invalid),cases=len(invalid))
    from cybrfire.traversal import TraversalSelector
    selector=TraversalSelector()
    selected=selector.choose(.8,lambda enabled:1. if enabled else 4.)
    check('selector_accepts_measured_speedup',selected)
    reused=selector.choose(.75,lambda enabled:(_ for _ in ()).throw(RuntimeError('unnecessary recalibration')))
    check('selector_reuses_compatible_occupancy',reused and len(selector.measurements)==1)
    rejected=selector.choose(.4,lambda enabled:4. if enabled else 1.)
    check('selector_rejects_slowdown_after_occupancy_change',not rejected and len(selector.measurements)==2)
    check('v5_config_roundtrip',scene_from_dict(json.loads(json.dumps(scene.to_dict())))==scene)
    # Affine rotation is reproduced exactly by MAC trilinear velocity sampling;
    # this isolates departure-point ODE integration from velocity reconstruction.
    d=scene.domain.shape;h=scene.domain.spacing;nx,ny,nz=d
    z,y,x=np.indices((nz,ny,nx));cx=nx*h/2;cz=nz*h/2;cy=ny*h/2;omega=5.;dt=.24
    q=np.zeros((nx*ny*nz,4),np.float32)
    f=lambda xc,zc:np.exp(-(((x+.5)*h-xc)**2+((z+.5)*h-zc)**2+((y+.5)*h-cy)**2)/(.18**2))
    q[:,0]=f(cx+.35,cz).ravel()
    vz,vy,vx=np.indices((nz+1,ny+1,nx+1));velocity=np.zeros((nz+1,ny+1,nx+1,4),np.float32)
    velocity[...,0]=-omega*((vz+.5)*h-cz);velocity[...,2]=omega*((vx+.5)*h-cx)
    expected=f(cx+.35*np.cos(omega*dt),cz+.35*np.sin(omega*dt))
    errors=[]
    for steps in (1,4):
        s=Simulation(ctx,replace(scene,domain=replace(scene.domain,max_trace_steps=steps,trace_courant=scene.domain.trace_courant)))
        s.q.write(q);s.vel.write(velocity.reshape(s.vn,4));s.advect(s.q,s.q_pred,s.q_out,s.vel,0,dt,corrected=False)
        got=s.q_out.read()[:,0].reshape(nz,ny,nx);errors.append(float(np.linalg.norm(got-expected)/np.linalg.norm(expected)))
        check(f'rotation_trace_{steps}_bounded',got.min()>=0 and got.max()<=q[:,0].max()+1e-6)
    check('subtracing_reduces_curved_characteristic_error',errors[1]<errors[0]*.55,one_segment_relative_L2=errors[0],subtraced_relative_L2=errors[1])
    # Full attack/hold/release exchange, including a start and end within a tick.
    e=replace(scene.emitters[0],center=(cx,cy,cz),radii=(.5,.5,.5),modulation=0.,turbulent_speed=0.,start=.073,end=.817,attack=.19,release=.27,replacement_rate=4.)
    a=Simulation(ctx,replace(scene,emitters=(e,)));b=Simulation(ctx,replace(scene,emitters=(e,)))
    def emit(sim,t,dt):sim.source.run(sim.shape,[sim.q,sim.vel,sim.emitters],dims=sim.shape,spacing=sim.h,dt=dt,time=t,sourceCount=1,operation=0)
    emit(a,0.,1.)
    for t in np.arange(0.,1.,.01):emit(b,float(t),.01)
    qa=a.q.read();qb=b.q.read();scale=np.array([1,1,1e6,1])
    error=float(np.max(abs(qa-qb)/scale));check('ramped_source_time_partition',error<3e-6,max_scaled_difference=error)
    pos=np.stack(((x+.5)*h,(y+.5)*h,(z+.5)*h),axis=-1);support=np.maximum(0.,1.-np.sum(((pos-np.array(e.center))/np.array(e.radii))**2,axis=-1))**2
    area=e.end-e.start-.5*(e.attack+e.release)
    oracle=e.fuel*(-np.expm1(-e.replacement_rate*support*area))
    error=float(np.max(abs(qa[:,0].reshape(nz,ny,nx)-oracle)))
    check('ramped_source_integrated_reservoir_oracle',error<1e-7,max_fuel_error=error,integrated_envelope_s=area)
    moving=replace(e,start=0.,end=4.,attack=0.,release=0.,motion_velocity=(.2,-.1,.15))
    time0=.4;dt0=.03;center=np.array(moving.center)+np.array(moving.motion_velocity)*(time0+.5*dt0)
    static=replace(moving,center=tuple(center),motion_velocity=(0.,0.,0.))
    ma=Simulation(ctx,replace(scene,emitters=(moving,)));mb=Simulation(ctx,replace(scene,emitters=(static,)))
    emit(ma,time0,dt0);emit(mb,time0,dt0)
    error=float(np.max(abs(ma.q.read()-mb.q.read())/scale));check('moving_support_world_space_equivalence',error<1e-6,max_scaled_difference=error)
    # Discrete diagonal rays visit every cell exactly once. In a zero-extinction
    # volume, the accumulated environment must be uniform everywhere.
    sim=Simulation(ctx,scene);renderer=VolumeRenderer(ctx,sim);renderer.prepare()
    light=texture_read(ctx,renderer.light_texture,renderer.light_shape)[...,:3]
    env=np.array([.65,.78,1.])*scene.render.ambient*(4.25/6)
    error=float(np.max(abs(light-env)));check('diagonal_sweeps_cover_volume_once',error<2e-6,max_environment_error=error)
    # A localized scalar field with tiny tails, positioned across brick faces.
    q=np.zeros((sim.n,4),np.float32);field=q.reshape(nz,ny,nx,4)
    field[5:14,4:20,7:17,2]=1.8e6
    field[5:14,4:20,7:17,3]=rng.random((9,16,10))*.0015
    # The far corner is in a different brick from the ordinary-density
    # patch, so another emitter cannot accidentally satisfy the tail check.
    field[nz-1,ny-1,nx-1,2]=1.8e6;field[nz-1,ny-1,nx-1,3]=1e-20
    sim.q.write(q);renderer.prepare();bounds=renderer.bricks.bounds.read().reshape(renderer.bricks.dimensions[::-1])
    check('occupancy_preserves_nonzero_tails',0<bounds[-1,-1,-1]<1e-8,isolated_corner_brick_bound=float(bounds[-1,-1,-1]))
    check('occupancy_marks_interpolation_halo',bounds[0,0,0]>0 and bounds[0,0,1]>0)
    snapshots=sim.snapshot()
    for yaw in (-70.,0.,37.,120.):
        renderer.cfg=replace(renderer.cfg,yaw=yaw,empty_space_skipping=False)
        dense=renderer.render_hdr(seed=2)
        renderer.cfg=replace(renderer.cfg,empty_space_skipping=True)
        accelerated=renderer.render_hdr(seed=2)
        error=float(np.max(abs(dense-accelerated)))
        check(f'exact_empty_skip_yaw_{yaw}',error<2e-6,max_HDR_error=error,**renderer.statistics())
    renderer.cfg=replace(renderer.cfg,yaw=15.)
    im1=renderer.render_hdr(seed=8);im2=renderer.render_hdr(seed=87)
    check('stable_sampling_no_frame_index_flicker',np.array_equal(im1,im2),max_HDR_difference=float(np.max(abs(im1-im2))))
    check('v5_render_preserves_simulation',np.array_equal(snapshots['q'],sim.q.read()) and np.array_equal(snapshots['velocity'],sim.vel.read()))
    renderer.cfg=replace(renderer.cfg,stable_sampling=False)
    im1=renderer.render_hdr(seed=8);im2=renderer.render_hdr(seed=87,prepare=False)
    check('frame_varying_ablation_is_exercised',float(np.max(abs(im1-im2)))>1e-5)
    renderer.cfg=replace(renderer.cfg,scattering_iterations=1);renderer.prepare();one=texture_read(ctx,renderer.light_texture,renderer.light_shape)
    renderer.cfg=replace(renderer.cfg,scattering_iterations=2);renderer.prepare();two=texture_read(ctx,renderer.light_texture,renderer.light_shape)
    check('positive_scattering_iteration',np.isfinite(two).all() and np.min(two[...,:3]-one[...,:3])>=-1e-6,maximum_added_radiance=float(np.max(two[...,:3]-one[...,:3])))
    renderer.prepare();again=texture_read(ctx,renderer.light_texture,renderer.light_shape)
    check('lighting_has_no_stale_previous_frame',np.array_equal(two,again))
    # Renderer closes only its resources, not simulation data or later users.
    extra=ctx.buffer(8);renderer.close();renderer.close()
    check('renderer_close_idempotent_and_scoped',extra.handle!=0 and np.array_equal(sim.q.read(),snapshots['q']) and rejects(renderer.render_hdr))
    sim.q.zero();r=VolumeRenderer(ctx,sim);r.render_hdr(width=48,height=64);st=r.statistics()
    check('zero_volume_fully_skipped',st['primary_evaluated_segments']==0 and st['primary_skipped_segments']==st['primary_candidate_segments'],**st)
    print('EMPTY_METRICS',json.dumps(st),flush=True)
    ctx.check();report={'version':'5.0.0','adapter':ctx.info,'passed':sum(t['passed'] for t in results),'failed':sum(not t['passed'] for t in results),'wall_seconds':time.perf_counter()-start,'tests':results}
    (ROOT/'evidence/v5-tests.json').write_text(json.dumps(report,indent=2));ctx.close();return report['failed']
if __name__=='__main__':sys.exit(bool(main()))
