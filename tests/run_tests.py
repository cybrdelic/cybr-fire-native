#!/usr/bin/env python3
"""Executable assertions and measured errors, not a static checklist."""
import sys,json,time,tempfile,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.gpu import ComputeContext
from cybrfire.config import example,Domain,Chemistry,Emitter,Scene
from cybrfire.solver import Simulation
from cybrfire.render import VolumeRenderer,blackbody_lut

ROOT=Path(__file__).resolve().parents[1]
results=[]
def check(name,condition,**measurements):
    results.append({'test':name,'passed':bool(condition),**measurements});print(('PASS ' if condition else 'FAIL ')+name+' '+json.dumps(measurements),flush=True)

def manufactured_A(p):
    out=np.zeros_like(p)
    for axis in (0,1,2):
        # NumPy z,y,x layout; floor is y=0, all remaining boundaries open.
        for sign in (-1,1):
            neighbor=np.roll(p,-sign,axis=axis)
            face=[slice(None)]*3;face[axis]=0 if sign==-1 else -1
            neighbor[tuple(face)]=p[tuple(face)] if axis==1 and sign==-1 else -p[tuple(face)]
            out+=p-neighbor
    return out

def cpu_cell_reaction(a,dt,c):
    q=a.astype(np.float64).copy();O=np.maximum(c.oxygen_ambient-q[:,1],0);T=300+q[:,2]/c.heat_capacity
    u=np.clip((T-620)/330,0,1);activation=u*u*(3-2*u)
    stoich=3.43-c.soot_yield*2.667
    b=np.minimum(q[:,0],O/stoich)*(-np.expm1(-c.reaction_rate*activation*dt));formed=b*c.soot_yield
    q[:,0]-=b;q[:,1]+=b*stoich;q[:,3]+=formed;q[:,2]+=b*c.heat_release-formed*32.8e6
    return q

def main():
    start=time.perf_counter();ctx=ComputeContext();base=example('burner','test')
    base=replace(base,domain=Domain((24,32,24),.1,1/60,3),render=replace(base.render,width=64,height=80))
    sim=Simulation(ctx,base);rng=np.random.default_rng(107)
    # Manufactured nonseparable pressure solution tests the actual boundary rows.
    d=base.domain.shape;shape=(d[2],d[1],d[0]);z,y,x=np.indices(shape)
    exact=(np.sin(.21*x)+.4*np.cos(.19*y)+.2*np.sin(.23*z)*np.cos(.11*x)).astype(np.float32)
    rhs=manufactured_A(exact);sim.mg.rhs.write(rhs.ravel());sim.mg.solve(cycles=10,reset=True)
    measured=sim.mg.pressure.read().reshape(shape);err=float(np.linalg.norm(measured-exact)/np.linalg.norm(exact))
    residual=float(np.linalg.norm(manufactured_A(measured)-rhs)/np.linalg.norm(rhs))
    check('multigrid_manufactured_solution',err<.002,relative_solution_error=err,relative_residual=residual)
    # Projection uses the matching discrete divergence and gradient.
    vel=rng.normal(0,.2,(sim.vn,4)).astype(np.float32);vel[:,3]=0
    v3=vel.reshape(d[2]+1,d[1]+1,d[0]+1,4);v3[:,0,:,1]=0
    sim.vel.write(vel);before=sim.diagnostics()['divergence_rms'];sim.project_velocity(cycles=8,reset=True);after=sim.diagnostics()
    check('projection_divergence_reduction',after['divergence_rms']<before*1e-3,before=before,after=after['divergence_rms'])
    check('floor_normal_flux_zero',after['floor_normal_max']==0,maximum=after['floor_normal_max'])
    # Reaction parity is independent NumPy math, with losses and soot oxidation off.
    chemistry=replace(base.chemistry,cooling_rate=0.,radiation_coefficient=0.,soot_oxidation=0.)
    chem=Simulation(ctx,replace(base,chemistry=chemistry))
    a=np.zeros((chem.n,4),np.float32);a[:,0]=rng.random(chem.n)*.1;a[:,1]=rng.random(chem.n)*.27;a[:,2]=rng.random(chem.n)*2e6;a[:,3]=rng.random(chem.n)*.002
    chem.q.write(a);dt=.017;expected=cpu_cell_reaction(a,dt,chemistry);chem.react(dt);got=chem.q.read()
    scale=np.array([.1,.27,4e6,.004]);parity=float(np.max(np.abs(got-expected)/scale))
    check('reaction_numpy_compute_parity',parity<3e-6,max_scaled_absolute_error=parity)
    beforeE=a[:,2].astype(float)+a[:,0].astype(float)*chemistry.heat_release+a[:,3].astype(float)*32.8e6
    afterE=got[:,2].astype(float)+got[:,0].astype(float)*chemistry.heat_release+got[:,3].astype(float)*32.8e6
    error=float(np.max(abs(afterE-beforeE)/np.maximum(beforeE,1.)))
    check('chemical_plus_sensible_energy',error<5e-7,max_relative_error=error)
    check('oxygen_budget',got[:,1].max()<=chemistry.oxygen_ambient+1e-7,maximum=float(got[:,1].max()))
    check('reaction_nonnegative',bool((got>=0).all()),minimum=float(got.min()))
    a.fill(0);a[:,0]=.1;a[:,2]=2e6;a[:,1]=.27;chem.q.write(a);chem.react(.1);starved=chem.q.read()
    check('oxygen_starvation',np.array_equal(starved,a))
    a.fill(0);a[:,2]=9.6e6;chem.q.write(a);chem.react(.1);noclip=chem.q.read()
    check('no_temperature_ceiling',np.array_equal(noclip,a),temperature_K=float(300+noclip[:,2].max()/chemistry.heat_capacity))
    a.fill(0);a[:,0]=.1;chem.q.write(a);chem.react(.1)
    check('cold_fuel_does_not_ignite',np.array_equal(chem.q.read(),a))
    # Bounded periodic scalar transport: constants, extrema, Gaussian accuracy.
    q=np.tile(np.array([.032,.12,1e6,.001],np.float32),(sim.n,1));sim.q.write(q)
    vel=np.zeros((sim.vn,4),np.float32);vel[:,0]=.65;sim.vel.write(vel)
    sim.advect(sim.q,sim.q_pred,sim.q_out,sim.vel,0,.037,periodic=True)
    terr=float(np.max(abs(sim.q_out.read()-q)/np.array([1,1,1e6,1])))
    check('constant_transport',terr<2e-6,max_scaled_error=terr)
    # Translate 7.4 cells through a periodic domain. CPU first-order oracle.
    g=np.exp(-(((x-10)/3.)**2+((y-16)/4.)**2+((z-12)/3.)**2)).astype(np.float32)
    q.fill(0);q[:,0]=g.ravel();sim.q.write(q);vel[:,0]=.37;sim.vel.write(vel)
    donor=g.copy();steps=20;dt=.1;C=.37
    for _ in range(steps):
        sim.advect(sim.q,sim.q_pred,sim.q_out,sim.vel,0,dt,periodic=True);sim.q,sim.q_out=sim.q_out,sim.q
        donor=(1-C)*donor+C*np.roll(donor,1,axis=2)
    moved=sim.q.read()[:,0].reshape(shape)
    # Periodic Gaussian oracle accounts for wrap-around explicitly.
    target=np.zeros(shape)
    for wrap in [-1,0,1]:target+=np.exp(-(((x-10-C*steps+wrap*d[0])/3.)**2+((y-16)/4.)**2+((z-12)/3.)**2))
    ecor=float(np.linalg.norm(moved-target)/np.linalg.norm(target));edon=float(np.linalg.norm(donor-target)/np.linalg.norm(target));drift=float(moved.astype(float).sum()/g.astype(float).sum()-1.)
    check('corrected_transport_less_diffusive',ecor<edon,corrected_relative_L2=ecor,donor_relative_L2=edon,mass_drift_fraction=drift)
    check('transport_no_new_extrema',moved.min()>=-1e-7 and moved.max()<=g.max()+1e-6,minimum=float(moved.min()),maximum=float(moved.max()))
    # Coupled source mixing has a closed-form composition and order invariance.
    e1=replace(base.emitters[0],center=(1.05,.8,1.0),radii=(.5,.4,.5),modulation=0.,replacement_rate=3.)
    e2=replace(e1,center=(1.3,.9,1.0),fuel=.02,temperature=700.,replacement_rate=5.)
    sa=Simulation(ctx,replace(base,emitters=(e1,e2)));sb=Simulation(ctx,replace(base,emitters=(e2,e1)))
    def emit(s,dt):s.source.run(s.shape,[s.q,s.vel,s.emitters],dims=s.shape,spacing=s.h,dt=dt,time=0.,sourceCount=2,operation=0)
    emit(sa,.1);emit(sb,.1);ea=sa.q.read();eb=sb.q.read()
    order=float(np.max(abs(ea-eb)/np.array([1,1,1e6,1])))
    check('overlapping_sources_order_independent',order<2e-6,max_scaled_error=order)
    sb.q.zero()
    for _ in range(10):emit(sb,.01)
    partition=float(np.max(abs(ea-sb.q.read())/np.array([1,1,1e6,1])))
    check('source_time_partition',partition<2e-6,max_scaled_error=partition)
    # Restart and presentation independence; no source parameters are changed.
    r1=Simulation(ctx,base);r1.step(6)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'restart.npz';r1.save(path);frozen=r1.snapshot();r1.step(5);end=r1.snapshot()
        r2=Simulation(ctx,base);r2.load(path);r2.step(5);resumed=r2.snapshot()
        diff=float(np.max(abs(end['q']-resumed['q'])));vdiff=float(np.max(abs(end['velocity']-resumed['velocity'])))
        check('restart_continuation',diff==0 and vdiff==0,scalar_max_difference=diff,velocity_max_difference=vdiff)
        check('snapshot_owned_not_mutated',not np.shares_memory(frozen['q'],end['q']) and frozen['tick']==6)
        bad=Path(tmp)/'bad.npz';bad.write_bytes(path.read_bytes()[:100]);pre=r2.snapshot()
        rejected=False
        try:r2.load(bad)
        except Exception:rejected=True
        check('corrupt_restore_transactional',rejected and np.array_equal(pre['q'],r2.q.read()))
    r3=Simulation(ctx,base)
    for t in [1/120,1/40,1/24,6/60,11/60]:r3.advance_to(t)
    diff=float(np.max(abs(r3.q.read()-end['q'])))
    check('presentation_cadence_independent',diff==0,scalar_max_difference=diff)
    wrong=False
    try:r3.advance_to(-1)
    except ValueError:wrong=True
    check('backward_time_rejected',wrong)
    # Renderer reads only; clean-air background differs from combustion state.
    r1.step(12);renderer=VolumeRenderer(ctx,r1);pre=r1.snapshot();image=renderer.render_hdr(width=64,height=80)
    check('renderer_read_only',np.array_equal(pre['q'],r1.q.read()) and np.array_equal(pre['velocity'],r1.vel.read()))
    check('render_finite_nonzero',np.isfinite(image).all() and image[...,:3].max()>.1,maximum=float(image[...,:3].max()))
    r1.q.zero();empty=renderer.render_hdr(width=64,height=80)
    difference=float(np.mean(abs(image-empty)))
    check('render_is_not_background_only',difference>1e-4,mean_HDR_difference=difference)
    lut=blackbody_lut();check('blackbody_lut_nonnegative_monotone_luminance',(lut[:,:3]>=0).all() and np.all(np.diff(lut[:,:3]@np.array([.2126,.7152,.0722]))>=0))
    # Uniform-slab transmittance/emission has an analytic solution.
    sigma=2.3;distance=1.7;B=np.array([2.,.8,.2]);analytic=B*(-np.expm1(-sigma*distance))
    integrated=np.zeros(3);T=1.
    for ds in [distance/71]*71:a=-np.expm1(-sigma*ds);integrated+=T*a*B;T*=1-a
    check('homogeneous_volume_integral',np.max(abs(integrated-analytic))<1e-12,max_error=float(np.max(abs(integrated-analytic))))
    # Actual compute ray marcher, not merely a Python segment formula.
    slab_scene=replace(base,render=replace(base.render,width=1,height=1,yaw=0.,fov=.001,
        camera_height=1.6,target_height=1.6,camera_distance=4.,floor=False,ambient=0.,
        scatter_albedo=0.,background=(0.,0.,0.),step_scale=.7,bloom=0.))
    slab=Simulation(ctx,slab_scene);sr=VolumeRenderer(ctx,slab)
    state=np.zeros((slab.n,4),np.float32);state[:,2]=(1800.-300.)*base.chemistry.heat_capacity
    pos=(1800.-300.)/5700.*511;lo=int(pos);B=lut[lo,:3]*(1-(pos-lo))+lut[lo+1,:3]*(pos-lo)
    for sigma in [0.,1.e-7,.23,2.3]:
        state[:,3]=sigma/slab_scene.render.soot_extinction;slab.q.write(state)
        computed=sr.render_hdr(width=1,height=1,samples=1)[0,0,:3]
        expected=B*(-np.expm1(-sigma*slab.shape[2]*slab.h))
        err=float(np.max(abs(computed-expected))/max(float(expected.max()),1.e-12))
        check('GPU_homogeneous_slab_'+str(sigma),err<2.e-4,relative_error=err,computed=computed.tolist(),expected=expected.tolist())
    ctx.check();report={'version':'5.0.0','adapter':ctx.info,'passed':sum(r['passed'] for r in results),'failed':sum(not r['passed'] for r in results),'wall_seconds':time.perf_counter()-start,'tests':results}
    (ROOT/'evidence/tests.json').write_text(json.dumps(report,indent=2));ctx.close()
    return report['failed']
if __name__=='__main__':sys.exit(bool(main()))
