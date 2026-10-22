"""Finite combustible fuel bed coupled to the 3D gas solve.

Conserves fuel mass across consumption/injection for cell-area to gas-volume
conversion. This is a coarse pyrolysis model, not structural char chemistry.
"""
import numpy as np
from .solver import SHADERS

class SurfaceCombustion:
    def __init__(self, ctx, scene, shape, spacing):
        self.ctx,self.scene,self.shape,self.h=ctx,scene,tuple(shape),float(spacing)
        cfg=scene.surface_fuel
        nx,_,nz=self.shape
        x=(np.arange(nx)+.5)*self.h
        z=(np.arange(nz)+.5)*self.h
        X,Z=np.meshgrid(x,z)
        e=((X-cfg.center[0])/cfg.radii[0])**2+((Z-cfg.center[1])/cfg.radii[1])**2
        # Spatial fuel coverage is the initial state, not a frame animation.
        coverage=np.clip((1.05-e)/.12,0,1)
        fuel=cfg.inventory*coverage
        radius=((X-cfg.center[0])**2+(Z-cfg.center[1])**2)**.5
        temp=np.full_like(fuel,cfg.initial_temperature)
        temp[radius<cfg.initial_ignition_radius]=max(cfg.ignition_temperature+250.,cfg.initial_temperature)
        st=np.stack([fuel,temp,np.zeros_like(fuel),np.zeros_like(fuel)],axis=-1).astype(np.float32).reshape(-1,4)
        self.state=ctx.buffer(st.shape,data=st)
        self.next=ctx.buffer(st.shape)
        self.step_shader=ctx.program(SHADERS/'surface_fuel.comp',{'NATIVE_FIELDS':1} if scene.domain.native_fields else None)
        self.inject_shader=ctx.program(SHADERS/'surface_inject.comp',{'NATIVE_FIELDS':1} if scene.domain.native_fields else None)
        self.initial_inventory=float(np.sum(st[:,0],dtype=np.float64)*self.h**2)
    def step(self,dt,gas,capacity):
        c=self.scene.surface_fuel
        uniforms=dict(dims=self.shape,spacing=self.h,dt=dt,ambientTemperature=300.,
          initialInventory=c.inventory,ignitionTemperature=c.ignition_temperature,maximumBurnRate=c.burn_rate,
          surfaceHeatCapacity=c.surface_heat_capacity,diffusivity=c.diffusivity,heatTransfer=c.heat_transfer,
          radiationSpread=c.spread_radiation,coolingRate=c.cooling_rate,capacity=capacity)
        self.step_shader.run((self.shape[0],self.shape[2]),[self.state,self.next,gas],**uniforms)
        self.state,self.next=self.next,self.state
        self.inject_shader.run(self.shape,[gas,self.state],dims=self.shape,spacing=self.h,dt=dt,
                                pyrolysisHeat=c.pyrolysis_heat,capacity=capacity)
    def inventory(self):
        a=self.state.read()
        return {'fuel_kg':float(np.sum(a[:,0],dtype=np.float64)*self.h**2),
                'char_kg':float(np.sum(a[:,2],dtype=np.float64)*self.h**2),
                'active_cells':int(np.sum(a[:,3]>1e-8)),
                'max_surface_temperature':float(a[:,1].max())}
    def close(self):
        self.state.close();self.next.close()
