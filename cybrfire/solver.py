"""GPU-resident staggered-grid reacting flow and geometric multigrid.

The operator graph owns advancement; presentation cannot change integration ticks.
Snapshots are copies; no renderer owns a mutable simulation field.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .config import Scene

SHADERS=Path(__file__).resolve().parent.parent/'shaders'

class Multigrid:
    def __init__(self,ctx,shape):
        self.ctx=ctx;self.levels=[]
        self.kernel=ctx.program(SHADERS/'pressure.comp')
        d=tuple(shape)
        while True:
            n=int(np.prod(d));level={'dims':d,'p':ctx.buffer(n),'tmp':ctx.buffer(n),'rhs':ctx.buffer(n),'res':ctx.buffer(n)}
            self.levels.append(level)
            if min(d)<8 or any(n%2 for n in d):break
            d=tuple(n//2 for n in d)
    @property
    def pressure(self):return self.levels[0]['p']
    @property
    def rhs(self):return self.levels[0]['rhs']
    def smooth(self,i,count):
        l=self.levels[i]
        for _ in range(count):
            self.kernel.run(l['dims'],[l['p'],l['rhs'],l['tmp']],dims=l['dims'],operation=0)
            l['p'],l['tmp']=l['tmp'],l['p']
    def cycle(self,i=0):
        l=self.levels[i]
        if i==len(self.levels)-1:
            self.smooth(i,30);return
        self.smooth(i,3)
        self.kernel.run(l['dims'],[l['p'],l['rhs'],l['res']],dims=l['dims'],operation=1)
        co=self.levels[i+1]
        self.kernel.run(co['dims'],[co['p'],l['res'],co['rhs']],dims=co['dims'],otherDims=l['dims'],operation=2)
        # Coarse correction starts at zero, never a stale previous-frame solution.
        co['p'].zero();co['tmp'].zero()
        self.cycle(i+1)
        self.kernel.run(l['dims'],[l['p'],co['p'],l['tmp']],dims=l['dims'],otherDims=co['dims'],operation=3)
        l['p'],l['tmp']=l['tmp'],l['p']
        self.smooth(i,3)
    def solve(self,cycles=3,reset=False):
        if reset:self.levels[0]['p'].zero()
        for _ in range(cycles):self.cycle()
    def residual(self):
        l=self.levels[0]
        self.kernel.run(l['dims'],[l['p'],l['rhs'],l['res']],dims=l['dims'],operation=1)
        return l['res'].read()

class Simulation:
    def __init__(self,ctx,scene:Scene):
        scene.validate();self.ctx=ctx;self.scene=scene;self.shape=tuple(scene.domain.shape);self.h=scene.domain.spacing
        self.n=int(np.prod(self.shape));self.vn=int(np.prod(np.array(self.shape)+1));self.tick=0;self.fault=None
        self.native_fields=scene.domain.native_fields
        self.kernel_defines={"NATIVE_FIELDS":1} if self.native_fields else None
        if self.native_fields:
            from .image_fields import ImageField
            make_q=lambda:ImageField(ctx,self.shape)
            make_v=lambda:ImageField(ctx,tuple(n+1 for n in self.shape))
        else:
            make_q=lambda:ctx.buffer((self.n,4))
            make_v=lambda:ctx.buffer((self.vn,4))
        self.q=make_q();self.q_pred=make_q();self.q_out=make_q()
        self.vel=make_v();self.vel_pred=make_v();self.vel_out=make_v()
        # Nonoverlapping lifetime reuse: force stages precede prediction, and
        # the reaction ledger is produced after scalar prediction is consumed.
        # q/velocity publication and pressure never alias scratch.
        self.omega=self.q_pred if self.native_fields else make_q()
        self.assembled_force=self.vel_pred if self.native_fields else make_q()
        self.budget=self.q_pred if self.native_fields else make_q()
        from .solids import SolidSet,MaskedPressure
        self.solids=SolidSet(ctx,self.shape,self.h,scene.solids,native_fields=self.native_fields)
        from .spectral import SpectralPressure
        self.mg=MaskedPressure(ctx,self.shape,self.solids) if scene.solids else (Multigrid(ctx,self.shape) if scene.domain.pressure_backend=="multigrid" else SpectralPressure(ctx,self.shape))
        from .transport import TextureTransport
        if self.native_fields:
            from .image_fields import ImageTransport
            self.texture_transport=ImageTransport(ctx,self.shape,scene.domain.trace_courant,scene.domain.max_trace_steps)
        else:self.texture_transport=TextureTransport(ctx,self.shape,scene.domain.trace_courant,scene.domain.max_trace_steps) if scene.domain.texture_transport else None
        self.transport=ctx.program(SHADERS/'transport.comp',self.kernel_defines);self.project=ctx.program(SHADERS/'project.comp',self.kernel_defines)
        self.source=ctx.program(SHADERS/'source.comp',self.kernel_defines);self.chemistry=ctx.program(SHADERS/'chemistry.comp',self.kernel_defines)
        self.forces=ctx.program(SHADERS/'forces.comp',self.kernel_defines)
        self.air_mixing=ctx.program(SHADERS/'air_mixing.comp',self.kernel_defines) if scene.flow.entrainment_rate>0 else None
        entries=[]
        for e in scene.emitters:
            entries.extend([(*e.center,e.start),(*e.radii,e.end),(e.fuel,e.oxygen_deficit,(e.temperature-300)*scene.chemistry.heat_capacity,e.soot),(*e.velocity,e.replacement_rate),(e.modulation,e.frequency,float(e.seed),e.swirl),(e.turbulent_speed,e.turbulent_length,e.turbulent_time,e.expansion_speed),(e.attack,e.release,e.radial_speed,0.),(*e.motion_velocity,0.)])
        self.emitters=ctx.buffer((len(entries),4),data=np.array(entries,np.float32))
        from .surface_fuel import SurfaceCombustion
        self.surface_fuel=SurfaceCombustion(ctx,scene,self.shape,self.h) if scene.surface_fuel.enabled else None
        from .embers import EmberSystem
        self.embers=EmberSystem(ctx,self) if scene.embers.enabled else None
        self.last_ms={}
        # Presentation is not part of the dynamical state: camera/exposure/name
        # changes must not invalidate a physically identical restart.
        physics={k:v for k,v in scene.to_dict().items() if k not in ('name','render')}
        self.config_hash=hashlib.sha256(json.dumps(physics,sort_keys=True).encode()).hexdigest()
    @property
    def time(self):return self.tick*self.scene.domain.tick
    def advect(self,original,pred,out,velocity,kind,dt,corrected=True,periodic=False):
        self.solids.bind()
        if self.texture_transport:
            self.texture_transport.solid_count=len(self.scene.solids)
            return self.texture_transport.advect(original,pred,out,velocity,kind,dt,self.h,corrected,periodic)
        d=tuple(np.array(self.shape)+(1 if kind else 0))
        kw={'dims':self.shape,'spacing':self.h,'dt':dt,'velocityField':int(kind),'periodic':int(periodic),'traceCourant':self.scene.domain.trace_courant,'maxTraceSteps':self.scene.domain.max_trace_steps,'solidCount':len(self.scene.solids)}
        self.transport.run(d,[original,pred,velocity,pred],correction=0,**kw)
        if corrected:self.transport.run(d,[original,pred,velocity,out],correction=1,**kw)
        else:
            # Explicitly copy the prediction through buffer read/write for test-only
            # first-order ablation. Production uses the bounded correction.
            out.write(pred.read())
    def project_velocity(self,cycles=None,reset=False):
        self.solids.enforce(self.q,self.vel,operation=1)
        self.project.run(self.shape,[self.vel,self.mg.pressure,self.mg.rhs],dims=self.shape,spacing=self.h,operation=0)
        self.mg.solve(cycles or self.scene.domain.pressure_cycles,reset)
        self.project.run(tuple(np.array(self.shape)+1),[self.vel,self.mg.pressure,self.mg.rhs],dims=self.shape,spacing=self.h,operation=1)
        self.solids.enforce(self.q,self.vel,operation=1)
    def react(self,dt):
        c=self.scene.chemistry
        self.chemistry.run(self.shape,[self.q,self.budget],dims=self.shape,dt=dt,oxygenAmbient=c.oxygen_ambient,capacity=c.heat_capacity,heatRelease=c.heat_release,sootYield=c.soot_yield,reactionRate=c.reaction_rate,coolingRate=c.cooling_rate,radiationCoefficient=c.radiation_coefficient,sootOxidation=c.soot_oxidation)
    def step(self,count=1):
        if type(count) is not int or count<0:raise ValueError('Step count must be a nonnegative integer')
        if self.fault:raise RuntimeError('Simulation is faulted: '+self.fault)
        dt=self.scene.domain.tick;fl=self.scene.flow;c=self.scene.chemistry
        try:
            for _ in range(count):
                kw=dict(dims=self.shape,spacing=self.h,dt=dt,time=self.time,sourceCount=len(self.scene.emitters))
                self.source.run(self.shape,[self.q,self.vel,self.emitters],operation=0,**kw)
                self.source.run(tuple(np.array(self.shape)+1),[self.q,self.vel,self.emitters],operation=1,**kw)
                if self.surface_fuel:self.surface_fuel.step(dt,self.q,c.heat_capacity)
                self.solids.enforce(self.q,self.vel)
                self.forces.run(self.shape,[self.vel,self.q,self.omega,self.assembled_force],dims=self.shape,spacing=self.h,operation=0)
                self.forces.run(self.shape,[self.vel,self.q,self.omega,self.assembled_force],dims=self.shape,spacing=self.h,dt=dt,operation=1,capacity=c.heat_capacity,confinement=fl.confinement,buoyancy=fl.buoyancy,wind=tuple(float(v) for v in fl.wind),drag=fl.drag,maxTurbulentAcceleration=fl.max_turbulent_acceleration)
                self.forces.run(tuple(np.array(self.shape)+1),[self.vel,self.q,self.omega,self.assembled_force],dims=self.shape,spacing=self.h,dt=dt,operation=2,wind=tuple(float(v) for v in fl.wind),drag=fl.drag)
                self.advect(self.vel,self.vel_pred,self.vel_out,self.vel,1,dt)
                self.vel,self.vel_out=self.vel_out,self.vel
                self.project_velocity()
                self.advect(self.q,self.q_pred,self.q_out,self.vel,0,dt)
                self.q,self.q_out=self.q_out,self.q
                self.solids.enforce(self.q,self.vel)
                if self.air_mixing:
                    self.air_mixing.run(self.shape,[self.q],dims=self.shape,spacing=self.h,dt=dt,
                        entrainmentRate=fl.entrainment_rate,entrainmentHeight=fl.entrainment_height)
                self.react(dt)
                if self.embers:self.embers.step(self.time,dt)
                self.tick+=1
            self.ctx.check()
        except Exception as exc:
            self.fault='Step failed: '+str(exc)
            raise
    def advance_to(self,t):
        if not np.isfinite(t) or t<self.time-1e-10:raise ValueError('Invalid or backward simulation time')
        target=int(np.floor(t/self.scene.domain.tick+1e-8));self.step(target-self.tick)
        return self.time
    def diagnostics(self):
        if self.native_fields:
            from .streaming import diagnostics
            return diagnostics(self)
        q=self.q.read();v=self.vel.read();d=self.shape;v3=v.reshape((d[2]+1,d[1]+1,d[0]+1,4))
        div=(v3[:-1,:-1,1:,0]-v3[:-1,:-1,:-1,0]+v3[:-1,1:,:-1,1]-v3[:-1,:-1,:-1,1]+v3[1:,:-1,:-1,2]-v3[:-1,:-1,:-1,2])/self.h
        finite=bool(np.isfinite(q).all() and np.isfinite(v).all())
        r={'time':self.time,'tick':self.tick,'finite':finite,'min_species':q.min(axis=0).astype(float).tolist(),'max_species':q.max(axis=0).astype(float).tolist(),'integrals':(q.sum(axis=0,dtype=np.float64)*self.h**3).tolist(),'max_temperature_K':float(300+q[:,2].max()/self.scene.chemistry.heat_capacity),'max_component_speed':float(np.abs(v[:,:3]).max()),'divergence_rms':float(np.sqrt(np.mean(div.astype(np.float64)**2))),'divergence_max':float(np.abs(div).max()),'cfl_component':float(np.abs(v[:,:3]).max()*self.scene.domain.tick/self.h),'floor_normal_max':float(np.abs(v3[:,0,:,1]).max())}
        if self.embers:r['embers']=self.embers.diagnostics()
        if self.surface_fuel:r['surface_fuel']=self.surface_fuel.inventory()
        if self.scene.solids:
            r['solid_scalar_max']=float(np.max(np.abs(q[~self.solids.fluid.ravel()])))
            r['pressure_solver']=self.mg.last
        if not finite or q.min()<-1e-4 or q[:,1].max()>self.scene.chemistry.oxygen_ambient+1e-5:
            self.fault='Nonfinite or inadmissible species detected';raise RuntimeError(self.fault)
        return r
    def snapshot(self):
        if self.fault:raise RuntimeError('Cannot publish a faulted simulation')
        return {'q':self.q.read(),'velocity':self.vel.read(),'pressure':self.mg.pressure.read(),'tick':self.tick,'config_hash':self.config_hash,**({'particles':self.embers.state.read()} if self.embers else {})}
    def save(self,path):
        if self.native_fields:
            from .streaming import save_checkpoint
            return save_checkpoint(self,path)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);s=self.snapshot()
        tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,**s,version=7,config=json.dumps(self.scene.to_dict(),sort_keys=True))
        tmp.replace(path)
    def load(self,path):
        # Verify every field before mutating any live GPU buffer.
        with np.load(path,allow_pickle=False) as z:
            if int(z['version'])!=7 or str(z['config_hash'])!=self.config_hash:raise ValueError('Checkpoint version/config mismatch')
            vals={k:z[k].copy() for k in ['q','velocity','pressure']+(['particles'] if self.embers else [])};tick=int(z['tick'])
        for key,b in [('q',self.q),('velocity',self.vel),('pressure',self.mg.pressure)]+([('particles',self.embers.state)] if self.embers else []):
            a=vals[key]
            if a.shape!=b.shape or not np.isfinite(a).all():raise ValueError('Invalid checkpoint '+key)
        if tick<0 or vals['q'].min()<-1e-4 or vals['q'][:,1].max()>self.scene.chemistry.oxygen_ambient+1e-5:raise ValueError('Invalid checkpoint state')
        self.q.write(vals['q']);self.vel.write(vals['velocity']);self.mg.pressure.write(vals['pressure']);self.tick=tick;self.fault=None
        if self.embers:self.embers.state.write(vals['particles'])
