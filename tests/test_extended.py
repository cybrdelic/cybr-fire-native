"""Cross-backend, rendering, ownership, and long-run regression tests.

Run explicitly. This is not a claim of physical validation or vendor benchmarking.
"""
import sys,json,tempfile,time
from pathlib import Path
from dataclasses import replace,FrozenInstanceError
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.config import example,Domain,Scene,scene_from_dict
from cybrfire.gpu import ComputeContext
from cybrfire.solver import Simulation
from cybrfire.render import VolumeRenderer
from run_tests import manufactured_A
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,passed,**values):
    checks.append(dict(test=name,passed=bool(passed),**values));print(('PASS ' if passed else 'FAIL ')+name+' '+json.dumps(values),flush=True)
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError,FrozenInstanceError):return True
    return False

def main():
    c=ComputeContext();start=time.perf_counter();base=example('burner','test')
    base=replace(base,domain=Domain((24,32,24),.08),render=replace(base.render,width=40,height=48))
    rng=np.random.default_rng(2435)
    # A round-trip must retain immutable vector data and the same identity.
    rebuilt=scene_from_dict(json.loads(json.dumps(base.to_dict())))
    check('configuration_roundtrip',rebuilt==base)
    check('configuration_vectors_immutable',all(isinstance(x,tuple) for x in [rebuilt.domain.shape,rebuilt.emitters,rebuilt.emitters[0].center,rebuilt.flow.wind,rebuilt.render.background]))
    invalid=[replace(base,domain=replace(base.domain,shape=(24.,32,24))),replace(base,chemistry=replace(base.chemistry,oxygen_ambient=0.)),replace(base,chemistry=replace(base.chemistry,cooling_rate=-1.)),replace(base,render=replace(base.render,scatter_albedo=1.1)),replace(base,emitters=(replace(base.emitters[0],turbulent_length=0.),)),replace(base,render=replace(base.render,exposure=float('nan')))]
    check('invalid_configurations_rejected',all(reject(s.validate) for s in invalid),cases=len(invalid))
    fast=Simulation(c,base);slow=Simulation(c,replace(base,domain=replace(base.domain,texture_transport=False)))
    # Nonconstant, nonperiodic flow exercises true hardware-filtered path.
    vel=rng.normal(0,.1,(fast.vn,4)).astype(np.float32);vel[:,3]=0
    q=rng.random((fast.n,4),dtype=np.float32)*np.array([.08,.26,2e6,.002],np.float32)
    for s in [fast,slow]:s.vel.write(vel);s.q.write(q)
    for kind in [0,1]:
        outs=[]
        for s in [fast,slow]:
            fields=(s.q,s.q_pred,s.q_out) if kind==0 else (s.vel,s.vel_pred,s.vel_out)
            s.advect(*fields,s.vel,kind,.013);outs.append(fields[2].read())
        denom=np.array([.08,.26,2e6,.002]) if kind==0 else np.ones(4)
        error=float(np.max(abs(outs[0]-outs[1])/denom))
        check('texture_buffer_parity_'+str(kind),error<1e-5,max_scaled_absolute_difference=error)
    # Cache updates after the same authoritative buffer changes.
    fast.q.zero();fast.advect(fast.q,fast.q_pred,fast.q_out,fast.vel,0,.013)
    check('sampling_cache_invalidation',np.max(abs(fast.q_out.read()))==0)
    # Courant > 1 is permitted by the semi-Lagrangian method, not an accuracy promise.
    fast.q.write(q);uniform=np.zeros_like(vel);uniform[:,0]=8.;uniform[:,1]=3.;fast.vel.write(uniform)
    fast.advect(fast.q,fast.q_pred,fast.q_out,fast.vel,0,.04,periodic=True);result=fast.q_out.read()
    check('large_Courant_bounded',np.isfinite(result).all() and (result>=q.min(axis=0)-1e-6).all() and (result<=q.max(axis=0)+1e-5).all(),component_Courant=4.)
    # Independently evaluated mixed-boundary Poisson problem, both backends.
    nx,ny,nz=base.domain.shape;z,y,x=np.indices((nz,ny,nx));exact=(np.sin(.31*x)*np.cos(.12*y)+.7*np.sin(.2*z)).astype(np.float32)
    rhs=manufactured_A(exact).ravel();spec=Simulation(c,replace(base,domain=replace(base.domain,pressure_backend='spectral')))
    for s in [fast,spec]:s.mg.rhs.write(rhs);s.mg.solve(cycles=12,reset=True)
    mp=fast.mg.pressure.read();sp=spec.mg.pressure.read();diff=float(np.linalg.norm(mp-sp)/np.linalg.norm(sp));se=float(np.linalg.norm(sp-exact.ravel())/np.linalg.norm(exact))
    check('multigrid_spectral_pressure_agreement',diff<1e-3,relative_pressure_difference=diff,spectral_exact_solution_error=se)
    check('spectral_boundary_operator',se<1e-5)
    # Test soot oxidation shares oxygen and transfers chemical energy to sensible heat.
    chemistry=replace(base.chemistry,reaction_rate=0.,cooling_rate=0.,radiation_coefficient=0.,soot_oxidation=4.)
    cs=Simulation(c,replace(base,chemistry=chemistry));state=np.zeros((cs.n,4),np.float32);state[:,3]=.015;state[:,1]=.23;state[:,2]=1.8e6
    cs.q.write(state);cs.react(.05);after=cs.q.read();soot=state[:,3]-after[:,3]
    error=float(np.max(abs((after[:,1]-state[:,1])-soot*2.667)))
    check('soot_oxidation_consumes_oxygen',soot.min()>0 and error<1e-7,max_oxygen_stoichiometry_error=error)
    energy_error=float(np.max(abs((after[:,2].astype(float)-state[:,2])-(soot.astype(float)*32.8e6)))/1.8e6)
    check('soot_oxidation_energy_accounted',energy_error<1e-6,relative_energy_error=energy_error)
    # The reaction-loss buffer is a per-step diagnostic, not a global advection ledger.
    loss=Simulation(c,replace(base,chemistry=replace(base.chemistry,reaction_rate=0.,soot_oxidation=0.)))
    state[:,3]=.001;state[:,1]=0.;loss.q.write(state);loss.react(.03);a=loss.q.read();budget=loss.budget.read()
    led=float(np.max(abs((state[:,2]-a[:,2])-budget[:,2])))
    check('cooling_nonnegative_and_accounted',a[:,2].min()>=0 and a[:,2].max()<state[:,2].min() and led<.1,max_ledger_error_J_per_m3=led)
    # Rendering and scene labels are not part of a restart's physical identity.
    fast=Simulation(c,base);fast.step(3)
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'valid.npz';fast.save(path)
        camera_scene=replace(base,name='renamed',render=replace(base.render,exposure=2.,yaw=25.))
        camera=Simulation(c,camera_scene);camera.load(path)
        check('restart_camera_independent',np.array_equal(fast.q.read(),camera.q.read()) and fast.tick==camera.tick)
        renderer=VolumeRenderer(c,camera);renderer.render_hdr(width=16,height=24);initial_buffers=len(c.buffers)
        for wh in [(24,32),(16,24),(32,40),(16,24)]:renderer.render_hdr(width=wh[0],height=wh[1],prepare=False)
        check('render_resize_releases_previous_buffer',len(c.buffers)==initial_buffers,buffer_count_before=initial_buffers,buffer_count_after=len(c.buffers))
        broken=camera.q.read();broken[0,0]=np.nan;camera.q.write(broken)
        detected=reject(camera.diagnostics)
        check('fault_blocks_state_publication',detected and reject(camera.snapshot) and reject(camera.step) and reject(lambda:renderer.render_hdr(width=8,height=8)))
        camera.load(path);check('validated_restart_recovers_fault',camera.fault is None and np.array_equal(camera.q.read(),fast.q.read()))
    check('invalid_step_count_rejected',reject(lambda:fast.step(-1)) and reject(lambda:fast.step(.5)))
    c.check();report={'adapter':c.info,'passed':sum(t['passed'] for t in checks),'failed':sum(not t['passed'] for t in checks),'wall_seconds':time.perf_counter()-start,'tests':checks}
    (ROOT/'evidence/extended-tests.json').write_text(json.dumps(report,indent=2));c.close()
    return report['failed']
if __name__=='__main__':sys.exit(bool(main()))
