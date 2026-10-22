"""Deterministic one-way hot fragments. Rendering never advances this state."""
from pathlib import Path
import math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def _ramp_integral(u):
    u=min(max(u,0.),1.);return u*u*u*(1.-.5*u)
def envelope_integral(e,t):
    t=min(max(t,e.start),e.end);a=e.attack;r=e.release;value=0.
    if a:value+=a*_ramp_integral(min((t-e.start)/a,1.))
    hold_begin=e.start+a;hold_end=e.end-r;value+=max(0.,min(t,hold_end)-hold_begin)
    if r and t>hold_end:
        u=(t-hold_end)/r;value+=r*(u-_ramp_integral(u))
    return value
class EmberSystem:
    def __init__(self,context,simulation):
        self.ctx,self.sim,self.cfg=context,simulation,simulation.scene.embers
        self.state=context.buffer((self.cfg.capacity,4,4));self.commands=context.buffer((self.cfg.capacity,4))
        self.kernel=context.program(ROOT/'shaders/embers.comp',simulation.kernel_defines);self.spawned=0;self.last_births=0
    def births(self,t,dt):
        cfg=self.cfg;emitters=self.sim.scene.emitters
        before=[int(math.floor(cfg.rate_per_emitter*envelope_integral(e,t)+1.e-9)) for e in emitters]
        after=[int(math.floor(cfg.rate_per_emitter*envelope_integral(e,t+dt)+1.e-9)) for e in emitters]
        global_id=sum(before);commands=[]
        if global_id>16_000_000:raise RuntimeError('Particle identity exceeds supported exact command range')
        for source,(a,b,e) in enumerate(zip(before,after,emitters)):
            for ordinal in range(a,b):
                lo=t;hi=t+dt;target=(ordinal+1)/cfg.rate_per_emitter
                for _ in range(28):
                    mid=.5*(lo+hi)
                    if envelope_integral(e,mid)<target:lo=mid
                    else:hi=mid
                commands.append((global_id%cfg.capacity,source,.5*(lo+hi),global_id));global_id+=1
        if len(commands)>cfg.capacity:raise RuntimeError('Particle birth burst exceeds capacity')
        return np.asarray(commands,np.float32).reshape(-1,4)
    def step(self,t,dt):
        c=self.cfg;s=self.sim;commands=self.births(t,dt);self.last_births=len(commands);self.spawned+=len(commands)
        kw=dict(dims=s.shape,spacing=s.h,dt=dt,time=t,capacityParticles=c.capacity,density=c.density,
            particleCapacity=c.heat_capacity,dragCoefficient=c.drag_coefficient,heatTransfer=c.heat_transfer,
            emissivity=c.emissivity,burnRate=c.burn_rate,lifetime=c.lifetime,restitution=c.restitution,friction=c.friction,
            gasCapacity=s.scene.chemistry.heat_capacity,oxygenAmbient=s.scene.chemistry.oxygen_ambient,
            oxidationHeat=c.oxidation_heat,retainedFraction=c.retained_heat_fraction,launchVelocity=c.launch_velocity,
            radiusRange=(c.radius_min,c.radius_max,0.,0.),birthTemperature=c.temperature,seed=c.seed,solidCount=len(s.scene.solids))
        s.solids.bind();bindings=[self.state,s.q,s.vel,s.emitters,s.solids.records,self.commands]
        self.kernel.run(c.capacity,bindings,operation=0,**kw)
        if len(commands):
            self.ctx.glBindBuffer(0x90D2,self.commands.handle);self.ctx.glBufferSubData(0x90D2,0,commands.nbytes,commands.ctypes.data)
            self.kernel.run(len(commands),bindings,operation=1,birthCount=len(commands),**kw)
    def diagnostics(self):
        a=self.state.read();live=a[:,0,3]>0
        return {'alive':int(live.sum()),'spawned_this_run':self.spawned,'minimum_y':float(a[live,0,1].min()) if np.any(live) else None,
            'maximum_temperature_K':float(a[live,1,3].max()) if np.any(live) else None,'finite':bool(np.isfinite(a).all()),'one_way_coupled':True}
    def close(self):self.state.close();self.commands.close()
