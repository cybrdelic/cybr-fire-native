#!/usr/bin/env python3
"""Actual V6 compute, transport/visibility, geometry and restart experiments."""
import sys,json,time,tempfile,math
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import example,Solid,Embers,scene_from_dict
from cybrfire.gpu import ComputeContext
from cybrfire.solver import Simulation
from cybrfire.solids import signed_distance
from cybrfire.embers import envelope_integral
from cybrfire.render import VolumeRenderer
from cybrfire.remesh import conservative_scalars,staggered_velocity
ROOT=Path(__file__).resolve().parents[1];results=[]
def check(name,condition,**data):
    results.append({'test':name,'passed':bool(condition),**data});print(('PASS ' if condition else 'FAIL ')+name+' '+json.dumps(data),flush=True)
def reject(fn):
    try:fn();return False
    except (ValueError,RuntimeError,KeyError):return True

def main():
    started=time.perf_counter();ctx=ComputeContext();rng=np.random.default_rng(972)
    scene=example('deflector','test');sim=Simulation(ctx,scene);A=sim.mg.A
    x=rng.normal(size=sim.n).astype(np.float32);y=rng.normal(size=sim.n).astype(np.float32)
    symmetry=float(abs(np.dot(x.astype(float),A@y)-np.dot(y.astype(float),A@x))/max(abs(np.dot(x.astype(float),A@y)),1))
    check('masked_pressure_symmetric',symmetry<2.e-5,error=symmetry)
    check('masked_pressure_positive_energy',float(x@(A@x))>0,quadratic_form=float(x@(A@x)))
    exact=np.sin(np.arange(sim.n)*.013).astype(np.float32);exact[~sim.solids.fluid.ravel()]=0
    sim.mg.rhs.write(A@exact);sim.mg.solve(reset=True);measured=sim.mg.pressure.read()
    res=float(np.linalg.norm(A@measured-A@exact)/np.linalg.norm(A@exact))
    check('masked_pressure_manufactured_solution',res<4.e-5,residual=res,iterations=sim.mg.last['iterations'])
    v=rng.normal(0,.2,size=(sim.vn,4)).astype(np.float32);v[:,3]=0;sim.vel.write(v);sim.solids.enforce(sim.q,sim.vel)
    before=sim.diagnostics()['divergence_rms'];sim.project_velocity(reset=True);diag=sim.diagnostics()
    check('masked_projection_reduces_divergence',diag['divergence_rms']<before*5e-5,before=before,after=diag['divergence_rms'])
    vv=sim.vel.read().reshape(*(np.array(sim.shape[::-1])+1),4);solid=~sim.solids.fluid;blocked=0.
    for a,axis in enumerate((2,1,0)):
        low=[slice(None)]*3;high=low.copy();low[axis]=slice(None,-1);high[axis]=slice(1,None)
        mask=solid[tuple(low)]|solid[tuple(high)];f=[slice(0,n) for n in solid.shape];f[axis]=slice(1,solid.shape[axis])
        blocked=max(blocked,float(np.max(np.abs(vv[tuple(f)+(a,)][mask]))))
    check('solid_normal_faces_zero',blocked==0.,maximum=blocked)
    check('solid_projection_floor_closed',diag['floor_normal_max']==0.)
    sim.q.write(np.ones((sim.n,4),np.float32));sim.solids.enforce(sim.q,sim.vel)
    check('solid_cells_have_no_material',float(np.max(np.abs(sim.q.read()[solid.ravel()])))==0.)
    sphere=Solid(center=(.7,1.1,.9),radius=.2);p=np.array([[.7,1.1,.9],[.9,1.1,.9],[1.2,1.1,.9]])
    check('sphere_distance',np.allclose(signed_distance(p,sphere),[-.2,0,.3]))
    check('box_distance',np.allclose(signed_distance(p,replace(sphere,kind='box',half_extents=(.2,.3,.4))),[-.2,0,.3]))
    cap=replace(sphere,kind='capsule',end=(.7,1.5,.9));check('capsule_distance',np.allclose(signed_distance(p,cap),[-.2,0,.3]))
    delta=np.array([2.,-1.,.6]);moved=replace(cap,center=tuple(np.array(cap.center)+delta),end=tuple(np.array(cap.end)+delta))
    check('collider_translation_invariance',np.allclose(signed_distance(p,cap),signed_distance(p+delta,moved)))
    check('unknown_geometry_rejected',reject(lambda:replace(scene,solids=(replace(sphere,kind='missing-mesh'),)).validate()))
    check('insufficient_particle_capacity_rejected',reject(lambda:replace(scene,embers=replace(scene.embers,capacity=1)).validate()))
    check('zero_light_direction_rejected',reject(lambda:replace(scene,render=replace(scene.render,key_direction=(0.,0.,0.))).validate()))
    check('scene_json_roundtrip',scene_from_dict(scene.to_dict())==scene)
    failed=Simulation(ctx,scene);failed.mg.maxiter=1;failed.vel.write(v);fault=False
    try:failed.step()
    except RuntimeError:fault=True
    check('failed_pressure_faults_owner',fault and failed.fault is not None)
    check('failed_pressure_cannot_publish',reject(lambda:failed.snapshot()))
    base=example('burner','test');base=replace(base,domain=replace(base.domain,pressure_backend='spectral'))
    without=Simulation(ctx,base);withp=Simulation(ctx,replace(base,embers=Embers(enabled=True)));without.step(30);withp.step(30)
    check('one_way_particles_do_not_change_gas',np.array_equal(without.q.read(),withp.q.read()) and np.array_equal(without.vel.read(),withp.vel.read()))
    e=replace(base.emitters[0],start=.25,end=3.5,attack=.4,release=.8)
    check('integrated_birth_envelope',abs(envelope_integral(e,10)-2.65)<1.e-12,integral=envelope_integral(e,10))
    check('declared_particle_birth_rate',withp.embers.spawned==55,actual=withp.embers.spawned,expected=55)
    direct=Simulation(ctx,replace(base,embers=Embers(enabled=True)));direct.advance_to(.5)
    cadence=Simulation(ctx,replace(base,embers=Embers(enabled=True)))
    for t in np.arange(1,31)/60:cadence.advance_to(float(t))
    check('particle_capture_cadence_invariant',np.array_equal(direct.embers.state.read(),cadence.embers.state.read()))
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'state.npz';direct.save(path);owned=direct.snapshot();direct.step(10)
        restarted=Simulation(ctx,replace(base,embers=Embers(enabled=True)));restarted.load(path);restarted.step(10)
        check('particle_restart_equivalent',np.array_equal(direct.embers.state.read(),restarted.embers.state.read()))
        check('restart_gas_equivalent',np.array_equal(direct.q.read(),restarted.q.read()))
        check('particle_snapshot_owned',not np.array_equal(owned['particles'],direct.embers.state.read()))
        with np.load(path,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
        payload['particles'][0,1,3]=np.nan;bad=Path(temp)/'bad.npz';np.savez(bad,**payload)
        previous=restarted.snapshot();ok=reject(lambda:restarted.load(bad))
        check('bad_particle_restore_transactional',ok and np.array_equal(restarted.q.read(),previous['q']) and np.array_equal(restarted.embers.state.read(),previous['particles']))
    emb=Embers(enabled=True,rate_per_emitter=0,emissivity=0.,burn_rate=0.,heat_transfer=45.,drag_coefficient=0.)
    thermal=Simulation(ctx,replace(base,embers=emb));q=np.zeros((thermal.n,4),np.float32);q[:,2]=800*base.chemistry.heat_capacity;thermal.q.write(q)
    a=np.zeros(thermal.embers.state.shape,np.float32);a[0,0]=(.96,1.,.96,.002);a[0,1,3]=1900.;a[0,2,:3]=a[0,0,:3];thermal.embers.state.write(a)
    dt=base.domain.tick;thermal.embers.step(0,dt);ctx.check();out=thermal.embers.state.read()
    k=3*emb.heat_transfer/(emb.density*emb.heat_capacity*.002);expected=(1900+dt*k*1100)/(1+dt*k)
    check('particle_implicit_thermal_exchange',abs(float(out[0,1,3])-expected)<.001,actual_K=float(out[0,1,3]),expected_K=expected)
    check('integer_time_uploaded_as_float_uniform',True)
    emb2=replace(emb,burn_rate=.00022,heat_transfer=0.,retained_heat_fraction=.35)
    burning=Simulation(ctx,replace(base,embers=emb2));burning.embers.state.write(a);burning.embers.step(0,dt);out=burning.embers.state.read();radius=.002-dt*.00022
    expected=1900+3*emb2.oxidation_heat*emb2.retained_heat_fraction/emb2.heat_capacity*math.log(.002/radius)
    check('shrinking_char_thermal_integral',abs(float(out[0,1,3])-expected)<.02,actual_K=float(out[0,1,3]),expected_K=expected)
    check('char_radius_update',abs(float(out[0,0,3])-radius)<1.e-8)
    collision=Simulation(ctx,replace(scene,embers=emb));a=np.zeros(collision.embers.state.shape,np.float32)
    a[0,0]=(.5,.92,.96,.002);a[0,1]=(40.,0.,0.,1600.);a[0,2,:3]=a[0,0,:3];collision.embers.state.write(a);collision.embers.step(0,dt)
    out=collision.embers.state.read();distance=float(signed_distance(out[0,0,:3],scene.solids[0]))
    check('swept_particle_no_tunnelling',distance>=float(out[0,0,3])-.0001 and out[0,1,0]<0,clearance_m=distance,vx=float(out[0,1,0]))
    rc=replace(base.render,width=80,height=100,samples=1,floor=False,ambient=0.,key_intensity=0.,background=(0.,0.,0.),camera_height=1.,target_height=1.,yaw=0.,camera_distance=4.,scatter_albedo=0.,bloom=0.,transmittance_cutoff=0.,empty_space_skipping=False)
    visible=replace(emb,emissivity=.86)
    occ=Simulation(ctx,replace(base,render=rc,embers=visible,solids=(Solid(center=(.96,1.,.96),radius=.25),)))
    renderer=VolumeRenderer(ctx,occ);blank=renderer.render_hdr();a=np.zeros(occ.embers.state.shape,np.float32)
    a[0,0]=(.96,1.,.55,.008);a[0,1,3]=2400.;occ.embers.state.write(a);behind=renderer.render_hdr(prepare=False)
    check('solid_occludes_particle',np.array_equal(blank,behind));a[0,0,2]=1.35;occ.embers.state.write(a);front=renderer.render_hdr(prepare=False)
    check('unoccluded_particle_visible',float(np.max(front[...,:3]-blank[...,:3]))>.05,peak_added_radiance=float(np.max(front[...,:3]-blank[...,:3])))
    pre=occ.snapshot();renderer.render_hdr(prepare=False)
    check('render_does_not_mutate_particles',np.array_equal(occ.embers.state.read(),pre['particles']) and occ.tick==pre['tick']);renderer.close()
    fog=Simulation(ctx,replace(base,render=rc,embers=visible));a=np.zeros(fog.embers.state.shape,np.float32);a[0,0]=(.96,1.,.96,.008);a[0,1,3]=2400.;fog.embers.state.write(a)
    r=VolumeRenderer(ctx,fog);clean=r.render_hdr();q=np.zeros((fog.n,4),np.float32);q[:,3]=.001;fog.q.write(q);attenuated=r.render_hdr()
    index=np.unravel_index(np.argmax(clean[...,:3]),clean[...,:3].shape);ratio=float(attenuated[index]/clean[index]);expected=math.exp(-rc.soot_extinction*.001*.96)
    check('volume_attenuates_particle_at_depth',abs(ratio-expected)<.003,measured_transmittance=ratio,analytic_transmittance=expected);r.close()
    mu,w=np.polynomial.legendre.leggauss(128);error=0.
    for g in (-.8,-.3,0.,.3,.8):error=max(error,abs(.5*np.sum(w*(1-g*g)/(1+g*g-2*g*mu)**1.5)-1))
    check('phase_normalization',error<1.e-10,error=error)
    check('baseline_has_no_particle_or_solid_allocations',without.embers is None and without.solids.mask is None)
    old=rng.random((12,16,12,4),dtype=np.float32);new=conservative_scalars(old,(20,24,20))
    err=float(np.max(np.abs(new.astype(float).mean((0,1,2))-old.astype(float).mean((0,1,2)))))
    check('remesh_channel_integrals',err<1.e-7,maximum_mean_error=err)
    check('remesh_preserves_bounds',new.min()>=old.min()-1.e-6 and new.max()<=old.max()+1.e-6)
    constant=np.ones_like(old)*.4;cn=conservative_scalars(constant,(20,24,20))
    check('remesh_preserves_constants',np.allclose(cn,.4,rtol=0,atol=1.e-7))
    v=np.zeros((13,17,13,4),np.float32);v[...,0]=.7;v[...,2]=-.3;vn=staggered_velocity(v,(12,16,12),(20,24,20))
    check('staggered_remesh_constant_flow',np.allclose(vn[:20,:24,:21,0],.7) and np.allclose(vn[:21,:24,:20,2],-.3) and np.max(np.abs(vn[:,0,:,1]))==0.)
    ctx.close();report={'version':6,'passed':sum(r['passed'] for r in results),'failed':sum(not r['passed'] for r in results),'seconds':time.perf_counter()-started,'checks':results}
    (ROOT/'evidence/v6-tests.json').write_text(json.dumps(report,indent=2));return int(report['failed']>0)
if __name__=='__main__':sys.exit(main())
