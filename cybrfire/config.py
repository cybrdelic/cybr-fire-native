"""Immutable, serializable configuration. Scenes provide data, not solver branches."""
from dataclasses import dataclass, asdict, field, replace
import math

@dataclass(frozen=True)
class Domain:
    shape: tuple[int,int,int]=(96,160,96)
    spacing: float=.035
    tick: float=1/60
    pressure_cycles: int=3
    pressure_backend: str="multigrid"
    texture_transport: bool=True
    native_fields: bool=False
    trace_courant: float=2.0
    max_trace_steps: int=4
    def __post_init__(self):object.__setattr__(self,"shape",tuple(self.shape))
    def validate(self):
        if len(self.shape)!=3 or any(type(n) is not int for n in self.shape):raise ValueError("Grid must contain three integers")
        if any(n<8 or n>512 or n%8 for n in self.shape):raise ValueError('Grid dimensions must be multiples of 8 in [8,512]')
        if type(self.native_fields) is not bool:raise ValueError('native_fields must be boolean')
        if math.prod(self.shape)>(32_000_000 if self.native_fields else 8_000_000):raise ValueError('Grid exceeds the selected storage backend cell budget')
        if self.native_fields and not self.texture_transport:raise ValueError('Native image fields require texture transport')
        if not 0<self.spacing<10 or not 0<self.tick<=.05:raise ValueError('Invalid spacing or tick')
        if self.pressure_backend not in ("multigrid","spectral"):raise ValueError("Unknown pressure backend")
        if type(self.max_trace_steps) is not int or not 1<=self.max_trace_steps<=16 or not .25<=self.trace_courant<=16:raise ValueError('Invalid characteristic integration budget')
        if not 1<=self.pressure_cycles<=20:raise ValueError('Invalid multigrid cycle count')

@dataclass(frozen=True)
class Chemistry:
    oxygen_ambient: float=.27
    heat_capacity: float=1200.
    heat_release: float=46e6
    soot_yield: float=.035
    reaction_rate: float=70.
    cooling_rate: float=.5
    radiation_coefficient: float=4*5.670374419e-8*150.
    soot_oxidation: float=.5

@dataclass(frozen=True)
class Flow:
    buoyancy: float=13.
    confinement: float=1.8
    max_turbulent_acceleration: float=8.
    drag: float=.4
    wind: tuple[float,float,float]=(0.,0.,0.)
    entrainment_rate: float=0.
    entrainment_height: float=.4
    def __post_init__(self):object.__setattr__(self,"wind",tuple(self.wind))

@dataclass(frozen=True)
class SurfaceFuel:
    """Finite ground fuel reservoir coupled to gas temperature and reaction.

    Surface state (kg/m^2, K, kg/m^2 of char) is evolved on its own 2D grid.
    These are model parameters, not per-frame flame geometry or image effects.
    """
    enabled: bool = False
    center: tuple[float,float] = (.96,.96)
    radii: tuple[float,float] = (.75,.75)
    inventory: float = .55
    initial_ignition_radius: float = .22
    initial_temperature: float = 300.
    burn_rate: float = .045
    ignition_temperature: float = 650.
    surface_heat_capacity: float = 1700.
    diffusivity: float = .008
    heat_transfer: float = 25.
    cooling_rate: float = .18
    spread_radiation: float = 14.
    pyrolysis_heat: float = 3.0e6
    def __post_init__(self):
        for k in ('center','radii'): object.__setattr__(self,k,tuple(getattr(self,k)))

@dataclass(frozen=True)
class Emitter:
    center: tuple[float,float,float]
    radii: tuple[float,float,float]
    velocity: tuple[float,float,float]=(0.,.65,0.)
    fuel: float=.045
    oxygen_deficit: float=.20
    temperature: float=1100.
    soot: float=0.
    replacement_rate: float=70.
    start: float=0.
    end: float=1e6
    modulation: float=.25
    frequency: float=8.
    seed: int=1847
    swirl: float=0.
    turbulent_speed: float=.8
    turbulent_length: float=.32
    turbulent_time: float=.24
    attack: float=0.0
    release: float=0.0
    motion_velocity: tuple[float,float,float]=(0.,0.,0.)
    expansion_speed: float=0.
    radial_speed: float=0.
    def __post_init__(self):
        for key in ["center","radii","velocity","motion_velocity"]:object.__setattr__(self,key,tuple(getattr(self,key)))

@dataclass(frozen=True)
class Solid:
    """Static world-space collider; voxel flow boundary and analytic visibility."""
    kind: str="sphere"
    center: tuple[float,float,float]=(.96,1.,.96)
    radius: float=.25
    half_extents: tuple[float,float,float]=(.2,.2,.2)
    end: tuple[float,float,float]=(1.3,1.,.96)
    albedo: tuple[float,float,float]=(.18,.20,.23)
    roughness: float=.32
    metallic: float=.8
    def __post_init__(self):
        for k in ("center","half_extents","end","albedo"):object.__setattr__(self,k,tuple(getattr(self,k)))

@dataclass(frozen=True)
class Embers:
    """One-way hot-char fragments. No gas feedback or pyrolysis is modeled."""
    enabled: bool=False
    capacity: int=4096
    rate_per_emitter: float=110.
    radius_min: float=.0009
    radius_max: float=.0024
    temperature: float=1900.
    density: float=120.
    heat_capacity: float=1100.
    drag_coefficient: float=.65
    heat_transfer: float=45.
    emissivity: float=.86
    burn_rate: float=.00022
    oxidation_heat: float=32.8e6
    retained_heat_fraction: float=.35
    launch_velocity: tuple[float,float,float]=(0.,1.8,0.)
    lifetime: float=5.
    restitution: float=.22
    friction: float=.3
    seed: int=619
    def __post_init__(self):object.__setattr__(self,"launch_velocity",tuple(self.launch_velocity))

@dataclass(frozen=True)
class Render:
    width: int=600
    height: int=800
    exposure: float=1.4
    fov: float=39.
    yaw: float=15.
    camera_distance: float=4.8
    camera_height: float=1.8
    target_height: float=1.45
    camera_target_x: float | None=None
    camera_target_z: float | None=None
    step_scale: float=.65
    samples: int=1
    soot_extinction: float=1900.
    scatter_albedo: float=.35
    ambient: float=.65
    background: tuple[float,float,float]=(.008,.012,.021)
    bloom: float=.08
    ground_resolution: int=96
    ground_directions: int=64
    floor: bool=True
    illumination_directions: int=14
    scattering_iterations: int=2
    empty_space_skipping: bool=True
    traversal_policy: str="auto"
    brick_size: int=8
    stable_sampling: bool=True
    transmittance_cutoff: float=.002
    key_direction: tuple[float,float,float]=(-.55,.70,-.45)
    key_color: tuple[float,float,float]=(.8,.9,1.)
    key_intensity: float=0.
    phase_g: float=.25
    shadow_step_cells: float=1.5
    shadow_resolution_scale: int=2
    ember_tile_capacity: int=128
    reconstruction_mode: int=0
    def __post_init__(self):
        for k in ("background","key_direction","key_color"):object.__setattr__(self,k,tuple(getattr(self,k)))

@dataclass(frozen=True)
class Scene:
    name: str
    domain: Domain=field(default_factory=Domain)
    chemistry: Chemistry=field(default_factory=Chemistry)
    flow: Flow=field(default_factory=Flow)
    render: Render=field(default_factory=Render)
    emitters: tuple[Emitter,...]=()
    solids: tuple[Solid,...]=()
    embers: Embers=field(default_factory=Embers)
    surface_fuel: SurfaceFuel=field(default_factory=SurfaceFuel)
    def __post_init__(self):
        object.__setattr__(self,"emitters",tuple(self.emitters))
        object.__setattr__(self,"solids",tuple(self.solids))
    def validate(self):
        self.domain.validate()
        if not 1<=len(self.emitters)<=64:raise ValueError('Scene requires 1-64 emitters')
        sf=self.surface_fuel
        extent=tuple(a*self.domain.spacing for a in self.domain.shape)
        if sf.enabled:
            if min(sf.radii)<=0 or min(sf.center[0],sf.center[1])<0 or sf.center[0]>extent[0] or sf.center[1]>extent[2]:
                raise ValueError('Surface fuel patch outside domain')
            if min(sf.inventory,sf.initial_ignition_radius,sf.burn_rate,sf.diffusivity,sf.heat_transfer,sf.cooling_rate,sf.spread_radiation,sf.pyrolysis_heat)<0 or sf.surface_heat_capacity<=0 or sf.ignition_temperature<=300:
                raise ValueError('Invalid surface combustion coefficients')
        if not 0<=self.chemistry.soot_yield<.2:raise ValueError('Invalid soot yield')
        if self.chemistry.heat_capacity<=0 or self.chemistry.oxygen_ambient<=0 or self.chemistry.heat_release<=0:raise ValueError('Invalid thermodynamics')
        if min(self.chemistry.reaction_rate,self.chemistry.cooling_rate,self.chemistry.radiation_coefficient,self.chemistry.soot_oxidation)<0:raise ValueError('Negative reaction/loss rate')
        if len(self.flow.wind)!=3 or min(self.flow.confinement,self.flow.max_turbulent_acceleration,self.flow.drag,self.flow.entrainment_rate)<0 or self.flow.entrainment_height<=0:raise ValueError('Invalid flow configuration')
        if len(self.solids)>16:raise ValueError("At most 16 analytic solids")
        for o in self.solids:
            if o.kind not in ("sphere","box","capsule"):raise ValueError("Unknown solid type")
            if any(len(getattr(o,k))!=3 for k in ("center","half_extents","end","albedo")):raise ValueError("Solid vectors must be 3D")
            if o.radius<=0 or min(o.half_extents)<=0 or not .04<=o.roughness<=1 or not 0<=o.metallic<=1 or min(o.albedo)<0 or max(o.albedo)>1:raise ValueError("Invalid solid material/geometry")
            if o.kind=="capsule" and sum((a-b)**2 for a,b in zip(o.center,o.end))<1.e-12:raise ValueError("Capsule endpoints coincide")
        b=self.embers
        if len(b.launch_velocity)!=3 or b.oxidation_heat<0 or not 0<=b.retained_heat_fraction<=1:raise ValueError("Invalid particle launch/oxidation model")
        if type(b.capacity) is not int or not 1<=b.capacity<=32768 or b.rate_per_emitter<0 or not 0<b.radius_min<=b.radius_max<.05:raise ValueError("Invalid particle allocation/radii/rate")
        if b.enabled and b.rate_per_emitter*len(self.emitters)*b.lifetime>b.capacity:raise ValueError("Particle capacity must cover rate times lifetime")
        if not 300<=b.temperature<=5900 or min(b.density,b.heat_capacity,b.lifetime)<=0 or min(b.drag_coefficient,b.heat_transfer,b.burn_rate)<0:raise ValueError("Invalid ember thermodynamics")
        if not 0<=b.emissivity<=1 or not 0<=b.restitution<=1 or not 0<=b.friction<=1 or type(b.seed) is not int:raise ValueError("Invalid ember material")
        r=self.render
        if any(len(getattr(r,k))!=3 for k in ("key_direction","key_color")) or sum(x*x for x in r.key_direction)<1.e-10 or min(r.key_color)<0 or r.key_intensity<0:raise ValueError("Invalid directional light")
        if not -.8<=r.phase_g<=.8 or not .25<=r.shadow_step_cells<=8 or r.shadow_resolution_scale not in (1,2,4) or not 16<=r.ember_tile_capacity<=512:raise ValueError("Invalid lighting/particle budget")
        if r.illumination_directions not in (6,14) or type(r.scattering_iterations) is not int or not 1<=r.scattering_iterations<=4:raise ValueError("Invalid illumination quadrature")
        if r.traversal_policy not in ("auto","forced"):raise ValueError("Invalid traversal policy")
        if r.brick_size not in (4,8,16) or not 0<=r.transmittance_cutoff<.1:raise ValueError("Invalid volume traversal policy")
        if type(r.width) is not int or type(r.height) is not int or min(r.width,r.height)<1 or max(r.width,r.height)>8192:raise ValueError('Invalid image dimensions')
        if not 0<r.fov<160 or r.camera_distance<=0 or r.exposure<=0 or not .1<=r.step_scale<=4 or not 1<=r.samples<=64:raise ValueError('Invalid camera/sampling configuration')
        if type(r.ground_resolution) is not int or not 8<=r.ground_resolution<=512 or type(r.ground_directions) is not int or not 4<=r.ground_directions<=256:raise ValueError('Invalid ground irradiance sampling')
        if min(r.soot_extinction,r.ambient,r.bloom)<0 or not 0<=r.scatter_albedo<=1 or len(r.background)!=3 or min(r.background)<0:raise ValueError('Invalid optical configuration')
        for point in (r.camera_target_x,r.camera_target_z):
            if point is not None and (not math.isfinite(point) or point<0): raise ValueError("Invalid camera look-at target")
        for e in self.emitters:
            if any(len(getattr(e,k))!=3 for k in ["center","radii","velocity","motion_velocity"]):raise ValueError("Emitter vectors require three components")
            if e.attack<0 or e.release<0 or e.attack+e.release>e.end-e.start:raise ValueError("Source ramps must be nonnegative and nonoverlapping")
            if e.soot<0 or e.turbulent_speed<0 or e.turbulent_length<=0 or e.turbulent_time<=0 or type(e.seed) is not int:raise ValueError("Invalid emitter turbulence/soot")
            if min(e.radii)<=0 or not 0<=e.modulation<1 or e.end<=e.start:raise ValueError('Invalid emitter geometry/time/modulation')
            if e.fuel<0 or not 0<=e.oxygen_deficit<=self.chemistry.oxygen_ambient:raise ValueError('Invalid source composition')
            if e.temperature<300 or e.replacement_rate<0:raise ValueError('Invalid source thermodynamics')
            if e.expansion_speed<0 or e.radial_speed<0:raise ValueError('Invalid source expansion')
        # Reject NaN/Inf throughout nested data, rather than allowing failure later.
        def check(o):
            if isinstance(o,dict):
                for v in o.values():check(v)
            elif isinstance(o,(tuple,list)):
                for v in o:check(v)
            elif isinstance(o,(float,int)) and not math.isfinite(o):raise ValueError('Nonfinite configuration')
        check(asdict(self))
    def to_dict(self):return asdict(self)

def scene_from_dict(d):
    return Scene(name=d['name'],domain=Domain(**d['domain']),chemistry=Chemistry(**d['chemistry']),flow=Flow(**d['flow']),render=Render(**d['render']),emitters=tuple(Emitter(**e) for e in d['emitters']),solids=tuple(Solid(**o) for o in d.get('solids',())),embers=Embers(**d.get('embers',{})),surface_fuel=SurfaceFuel(**d.get('surface_fuel',{})))

def example(name='burner',quality='standard'):
    # Physical domain is invariant when changing quality.
    grids={'test':(32,56,32),'preview':(64,112,64),'standard':(96,168,96),'high':(128,224,128),'ultra':(160,280,160)}
    grid=grids[quality];h=1.92/grid[0];dom=Domain(shape=grid,spacing=h)
    mid=.96
    basic=Emitter(center=(mid,.12,mid),radii=(.30,.16,.30))
    scene=Scene(name,domain=dom,emitters=(basic,))
    if name=='ember-burner':
        return replace(scene,embers=Embers(enabled=True),render=replace(scene.render,key_intensity=.24,ambient=.5))
    if name=='deflector':
        return replace(scene,solids=(Solid(center=(mid,.92,mid),radius=.27),),embers=Embers(enabled=True,rate_per_emitter=90.),render=replace(scene.render,key_intensity=.32,ambient=.5,yaw=25.))
    if name=='grate':
        bars=tuple(Solid(kind="capsule",center=(.45,.66,z),end=(1.47,.66,z),radius=.065,roughness=.34) for z in (.67,.96,1.25))
        return replace(scene,solids=bars,embers=Embers(enabled=True,rate_per_emitter=130.),render=replace(scene.render,key_intensity=.32,yaw=30.,ambient=.5))
    if name=='ember-shutdown':
        return replace(scene,emitters=(replace(basic,end=3.8,release=1.2),),embers=Embers(enabled=True,rate_per_emitter=160.),render=replace(scene.render,key_intensity=.32,ambient=.45))
    if name=='burner':return scene
    if name=='crosswind':return replace(scene,flow=replace(scene.flow,wind=(1.1,0.,0.),drag=.65),render=replace(scene.render,yaw=-12.))
    if name=='vortex':return replace(scene,emitters=(replace(basic,swirl=7.,radii=(.35,.18,.35)),))
    if name=='pulse':
        e=replace(basic,center=(mid,.65,mid),radii=(.43,.43,.43),velocity=(0.,2.,0.),fuel=.14,oxygen_deficit=.16,temperature=1300.,replacement_rate=22.,end=.28,modulation=.6)
        return replace(scene,emitters=(e,),flow=replace(scene.flow,confinement=2.2))
    if name=='two-burners':
        e1=replace(basic,center=(mid-.32,.12,mid),radii=(.22,.15,.22),seed=97)
        e2=replace(e1,center=(mid+.32,.12,mid),seed=701,velocity=(-.3,1.8,0.))
        return replace(scene,emitters=(e1,e2))
    if name=='sweeping-jet':
        e=replace(basic,center=(.4,.25,mid),radii=(.2,.14,.2),velocity=(.5,1.6,0.),motion_velocity=(.22,0.,0.),attack=.35,release=.6,end=4.5)
        return replace(scene,emitters=(e,))
    if name=='shutdown':
        return replace(scene,emitters=(replace(basic,end=3.8,release=1.2),))
    raise ValueError('Unknown scene '+name)
