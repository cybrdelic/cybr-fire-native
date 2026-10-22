"""Fine scalar evolution driven by a separately resolved, cached MAC flow.

This is an explicitly one-way VFX detail model. Fine fuel, oxygen deficit,
sensible energy and soot are advected, sourced, and reacted on the fine grid.
The parent pressure/velocity solution is NOT advertised as a fine-grid solve.
Synthetic transverse modes are a declared subgrid closure, not resolved CFD.
No field deformation, sharpening or noise is performed in camera space.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
from types import SimpleNamespace
import ctypes,gc,json,math,mmap
import numpy as np
from .gpu import bind_texture,bind_image
from .image_fields import ImageField
from .solids import SolidSet
from .solver import SHADERS

@dataclass(frozen=True)
class DetailConfig:
    shape:tuple=(384,672,384)
    flow_scale:int=2
    rms_velocity:float=.40
    wavelengths:tuple=(.16,.08,.04)
    modes_per_band:int=4
    decorrelation_speed:float=.30
    seed:int=94713
    cubic:bool=True
    trace_budget:float=1.5
    trace_limit:int=64
    def validate(self,extent):
        if len(self.shape)!=3 or any(type(x) is not int or x<8 or x>1024 for x in self.shape):raise ValueError('Invalid detail dimensions')
        if math.prod(self.shape)>105_000_000:raise ValueError('Detail grid exceeds the explicit memory budget')
        if self.flow_scale not in (1,2,4):raise ValueError('Invalid flow cache scale')
        if not np.allclose(np.asarray(extent)/self.shape,extent[0]/self.shape[0]):raise ValueError('Detail cells must be cubic')
        if self.rms_velocity<0 or not np.isfinite(self.rms_velocity):raise ValueError('Invalid mode amplitude')
        if not self.wavelengths or min(self.wavelengths)<3.9*self.flow_scale*extent[0]/self.shape[0]:raise ValueError('Modes underresolved by the flow cache')
        if self.modes_per_band<1 or self.trace_limit<1 or self.trace_budget<=0:raise ValueError('Invalid integration setting')


def transverse_modes(config):
    rng=np.random.default_rng(config.seed);values=[];weights=[]
    for wavelength in config.wavelengths:
        for _ in range(config.modes_per_band):
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
            pol=rng.normal(size=3);pol-=direction*np.dot(direction,pol);pol/=np.linalg.norm(pol)
            weight=(wavelength/max(config.wavelengths))**(1/3)
            wave=direction*(2*np.pi/wavelength)
            omega=(2*np.pi/wavelength)*config.decorrelation_speed*rng.uniform(.6,1.4)
            values.extend([(*wave,rng.uniform(0,2*np.pi)),(*pol,omega)]);weights.append(weight)
    a=np.array(values,np.float32);weights=np.asarray(weights)
    a[1::2,:3]*=(weights*np.sqrt(2./np.sum(weights**2)))[:,None]
    return a


def upload_npy_slabs(field,path,depth=8):
    """Bounded host-memory upload; do not retain a mapped cache in the RSS."""
    a=np.load(path,mmap_mode='r',allow_pickle=False)
    if a.shape!=field.shape or a.dtype!=np.float32:raise ValueError('Cache array shape/dtype mismatch')
    nx,ny,nz=field.grid_shape;ctx=field.ctx;bind_texture(ctx,field.handle,8)
    for z in range(0,nz,depth):
        n=min(depth,nz-z);slab=np.ascontiguousarray(a[z*nx*ny:(z+n)*nx*ny])
        ctx.glTexSubImage3D(0x806F,0,0,0,z,nx,ny,n,0x1908,0x1406,slab.ctypes.data)
    ctx.glMemoryBarrier(0xFFFFFFFF);ctx.check();del a
    gc.collect()
    try:ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError,AttributeError):pass


class DetailTransport:
    def __init__(self,ctx,parent_scene,config,initial_q_path,parent_directory):
        if parent_scene.solids:raise ValueError('Fine scalar detail currently supports free-space burners, not solid masks')
        self.ctx=ctx;self.scene=parent_scene;self.cfg=config
        self.extent=tuple(np.array(parent_scene.domain.shape)*parent_scene.domain.spacing)
        config.validate(self.extent);self.shape=tuple(config.shape);self.h=self.extent[0]/self.shape[0]
        self.parent_shape=tuple(parent_scene.domain.shape);self.parent_directory=Path(parent_directory)
        self.native_fields=True;self.kernel_defines={'NATIVE_FIELDS':1};self.embers=None;self.fault=None;self.tick=0
        self.solids=SolidSet(ctx,self.shape,self.h,(),native_fields=True)
        self.q=ImageField(ctx,self.shape)
        init=ctx.program(SHADERS/'detail_initialize.comp')
        parent_q=ImageField(ctx,self.parent_shape);upload_npy_slabs(parent_q,initial_q_path)
        bind_texture(ctx,parent_q.handle,0);bind_image(ctx,self.q.handle,0)
        init.run(self.shape,[],dims=self.shape);ctx.finish();parent_q.close()
        self.next=ImageField(ctx,self.shape)
        self.flow_shape=tuple(n//config.flow_scale for n in self.shape)
        self.flow_field=ImageField(ctx,self.flow_shape,full_precision=False)
        self.parent=[ImageField(ctx,tuple(n+1 for n in self.parent_shape),full_precision=False)]
        self.parent_indices=[None,None]
        self.mode_array=transverse_modes(config);self.modes=ctx.buffer(self.mode_array.shape,data=self.mode_array)
        self.flow_program=ctx.program(SHADERS/'detail_flow.comp',{'FLOW_FP16':1})
        self.advect_program=ctx.program(SHADERS/'detail_advect.comp')
        self.reaction_program=ctx.program(SHADERS/'chemistry.comp',{'NATIVE_FIELDS':1,'DETAIL_NO_BUDGET':1})
        self.source_program=ctx.program(SHADERS/'source.comp',{'NATIVE_FIELDS':1})
        self.air_mixing=ctx.program(SHADERS/'air_mixing.comp',{'NATIVE_FIELDS':1}) if parent_scene.flow.entrainment_rate>0 else None
        entries=[]
        for e in parent_scene.emitters:
            entries.extend([(*e.center,e.start),(*e.radii,e.end),(e.fuel,e.oxygen_deficit,(e.temperature-300)*parent_scene.chemistry.heat_capacity,e.soot),(*e.velocity,e.replacement_rate),(e.modulation,e.frequency,float(e.seed),e.swirl),(e.turbulent_speed,e.turbulent_length,e.turbulent_time,e.expansion_speed),(e.attack,e.release,e.radial_speed,0.),(*e.motion_velocity,0.)])
        self.emitters=ctx.buffer((len(entries),4),data=np.asarray(entries,np.float32))
        from .surface_fuel import SurfaceCombustion
        self.surface_fuel=SurfaceCombustion(ctx,parent_scene,self.shape,self.h) if parent_scene.surface_fuel.enabled else None
        self.last={}
    def set_parent_pair(self,a,b):
        self.parent_indices=[a,b]
        for index in (a,b):
            if not (self.parent_directory/f'velocity-{index:03}.npy').exists():raise FileNotFoundError('Missing parent state')
    def upload_mixed_parent(self,blend):
        field=self.parent[0];nx,ny,nz=field.grid_shape;ctx=self.ctx
        a,b=[np.load(self.parent_directory/f'velocity-{i:03}.npy',mmap_mode='r',allow_pickle=False) for i in self.parent_indices]
        if a.shape!=field.shape or b.shape!=field.shape:raise ValueError('Parent shape mismatch')
        bind_texture(ctx,field.handle,8)
        for z in range(0,nz,4):
            depth=min(4,nz-z);start=z*nx*ny;end=(z+depth)*nx*ny
            mixed=(1.-np.float32(blend))*a[start:end]+np.float32(blend)*b[start:end]
            ctx.glTexSubImage3D(0x806F,0,0,0,z,nx,ny,depth,0x1908,0x1406,mixed.ctypes.data)
            del mixed
            # Prevent mapped cache pages from competing with the fine textures.
            a._mmap.madvise(mmap.MADV_DONTNEED);b._mmap.madvise(mmap.MADV_DONTNEED)
        del a,b
        ctx.glMemoryBarrier(0xFFFFFFFF);ctx.check()
    def flow(self,time,blend):
        self.upload_mixed_parent(blend)
        for unit in (4,5):bind_texture(self.ctx,self.parent[0].handle,unit)
        bind_image(self.ctx,self.flow_field.handle,0)
        self.flow_program.run(self.flow_shape,[self.modes],dims=self.flow_shape,parentDims=self.parent_shape,
            extent=self.extent,blend=0.,time=float(time),modeCount=len(self.mode_array)//2,detailStrength=self.cfg.rms_velocity,floorLayer=float(max(self.cfg.wavelengths)))
    def react(self,dt):
        c=self.scene.chemistry
        self.reaction_program.run(self.shape,[self.q,self.next],dims=self.shape,dt=float(dt),oxygenAmbient=c.oxygen_ambient,capacity=c.heat_capacity,heatRelease=c.heat_release,sootYield=c.soot_yield,reactionRate=c.reaction_rate,coolingRate=c.cooling_rate,radiationCoefficient=c.radiation_coefficient,sootOxidation=c.soot_oxidation)
    def step(self,time,dt,blend):
        if self.fault:raise RuntimeError(self.fault)
        if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid detail timestep')
        try:
            self.source_program.run(self.shape,[self.q,self.next,self.emitters],dims=self.shape,spacing=self.h,dt=float(dt),time=float(time),sourceCount=len(self.scene.emitters),operation=0)
            if self.surface_fuel:self.surface_fuel.step(dt,self.q,self.scene.chemistry.heat_capacity)
            self.react(.5*dt)
            self.flow(time+.5*dt,blend)
            bind_texture(self.ctx,self.q.handle,0);bind_texture(self.ctx,self.flow_field.handle,1)
            bind_image(self.ctx,self.next.handle,0)
            self.advect_program.run(self.shape,[],dims=self.shape,spacing=self.h,dt=float(dt),extent=self.extent,
                cubic=int(self.cfg.cubic),traceBudget=self.cfg.trace_budget,traceLimit=self.cfg.trace_limit)
            self.q,self.next=self.next,self.q
            if self.air_mixing:
                fl=self.scene.flow
                self.air_mixing.run(self.shape,[self.q],dims=self.shape,spacing=self.h,dt=dt,
                    entrainmentRate=fl.entrainment_rate,entrainmentHeight=fl.entrainment_height)
            self.react(.5*dt);self.ctx.check();self.tick+=1
        except Exception as exc:
            self.fault=f'Detail step failed: {exc}';raise
    def state_statistics(self):
        lo=np.full(4,np.inf);hi=np.full(4,-np.inf);total=np.zeros(4);active=0;finite=True
        for z in range(0,self.shape[2],4):
            a=self.q.read_slab(z,min(4,self.shape[2]-z));lo=np.minimum(lo,a.min(axis=(0,1,2)));hi=np.maximum(hi,a.max(axis=(0,1,2)));total+=a.sum(axis=(0,1,2),dtype=np.float64)
            active+=int(np.count_nonzero(a[...,2]>1200*300));finite=finite and bool(np.isfinite(a).all())
        return dict(grid=self.shape,cells=math.prod(self.shape),spacing=self.h,minimum=lo.tolist(),maximum=hi.tolist(),integrals=(total*self.h**3).tolist(),hot_cells=active,finite=finite)
    def close(self):
        for field in (self.q,self.next,self.flow_field,*self.parent):field.close()
        self.modes.close();self.emitters.close();self.solids.close()
        if self.surface_fuel:self.surface_fuel.close()
