"""Direct field rendering: thermal radiance, soot extinction, directional lighting.

No render-time noise, domain warping, flame masks, or preset selection.
"""
from pathlib import Path
import math
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from .gpu import texture3d,bind_texture
from .solver import SHADERS

def blackbody_lut(count=512,max_kelvin=6000.):
    # Planck spectra integrated against Wyman/Sloan/Shirley analytic CIE fits.
    nm=np.arange(360.,831.,2.)
    def g(mu,left,right):
        s=np.where(nm<mu,left,right);return np.exp(-.5*((nm-mu)*s)**2)
    xyz=np.array([1.056*g(599.8,.0264,.0323)+.362*g(442.,.0624,.0374)-.065*g(501.1,.049,.0382),.821*g(568.8,.0213,.0247)+.286*g(530.9,.0613,.0322),1.217*g(437.,.0845,.0278)+.681*g(459.,.0385,.0725)]).T
    wavelengths=nm*1e-9;temperatures=np.linspace(300,max_kelvin,count)
    spectra=1.191042972e-16/(wavelengths[None,:]**5*np.expm1(np.minimum(1.438776877e-2/(wavelengths[None,:]*temperatures[:,None]),700)))
    tristimulus=spectra@xyz*2e-9
    mat=np.array([[3.2406,-1.5372,-.4986],[-.9689,1.8758,.0415],[.0557,-.204,1.057]])
    rgb=np.maximum(tristimulus@mat.T,0)
    # Radiometric working scale: integrated spectral XYZ Y at 1800 K = 1.
    at1800=1.191042972e-16/(wavelengths**5*np.expm1(1.438776877e-2/(wavelengths*1800.)))
    rgb/=float(at1800@xyz[:,1]*2e-9)
    out=np.ones((count,4),np.float32);out[:,:3]=rgb
    return out

def tone_map(hdr,exposure=1.,bloom=.0):
    x=np.maximum(hdr[...,:3],0).astype(np.float32)*exposure
    if bloom:
        bright=np.maximum(x-1.,0)
        # Optical glare derived from HDR, never from source or preset identity.
        for sigma,weight in [(2.,.45),(7.,.35),(20.,.2)]:x+=bloom*weight*gaussian_filter(bright,(sigma,sigma,0))
    # ACES-style display shoulder. Applied only in camera space after physical
    # radiance integration; raw linear HDR remains available in captures.
    x=(x*(2.51*x+.03))/(x*(2.43*x+.59)+.14)
    x=np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)
    return np.clip(x*255+.5,0,255).astype(np.uint8)

class VolumeRenderer:
    """Read-only optical reconstruction, lighting, acceleration, and camera passes."""
    def __init__(self,ctx,simulation,config=None,optical_workspace=None,lighting_scale=2):
        from .acceleration import OpticalBricks
        from .lighting import IncidentLighting
        self.ctx=ctx;self.sim=simulation;self.cfg=config or simulation.scene.render
        self.shape=simulation.shape;self.closed=False
        self.borrowed_optical=optical_workspace is not None
        self.optical_texture=optical_workspace.handle if self.borrowed_optical else texture3d(ctx,self.shape,0,full_precision=True,edge=True)
        self.incident=IncidentLighting(ctx,self.shape,scale=lighting_scale)
        self.light_shape=self.incident.shape;self.light=self.incident.accumulation
        self.light_texture=self.incident.texture
        self.bricks=OpticalBricks(ctx,self.shape,self.cfg.brick_size)
        self.lut=ctx.buffer((512,4),data=blackbody_lut())
        from .image_fields import ImageField
        self.radiance_field=ImageField(ctx,(512,1,1));self.radiance_field.write(blackbody_lut())
        self.physical_copy=None
        if not simulation.native_fields:
            self.physical_copy=ImageField(ctx,self.shape)
        self.optics=ctx.program(SHADERS/'optical.comp',simulation.kernel_defines);self.march=ctx.program(SHADERS/'render.comp',{'PARTICLE_EVENTS':1} if simulation.embers else None)
        self.particle_event_capacity=32
        self.particle_event_variants={32:self.march} if simulation.embers else {}
        self.particle_event_growths=0
        from .direct_light import DirectLight
        self.direct=DirectLight(ctx,self.shape,self.cfg.shadow_resolution_scale)
        from .particle_render import ParticleRenderer
        self.particles=ParticleRenderer(ctx,simulation,self.lut) if simulation.embers else None
        self.ground_resolution=self.cfg.ground_resolution
        self.ground_texture=texture3d(ctx,(self.ground_resolution,1,self.ground_resolution),5,full_precision=True)
        self.ground=ctx.program(SHADERS/'ground.comp')
        size=np.array(self.shape)*simulation.h
        self.patch_origin=(-float(size[0]),0.,-float(size[2]))
        self.patch_extent=(float(3*size[0]),1.,float(3*size[2]))
        self.hdr=None;self.ray_metrics=None;self.last_size=None
        self.prepare_count=0
        from .traversal import TraversalSelector
        self.selector=TraversalSelector()

    def prepare(self):
        from .acceleration import OpticalBricks
        if self.closed:raise RuntimeError('Renderer is closed')
        if self.sim.fault:raise RuntimeError('Cannot render a faulted simulation')
        c=self.cfg;ctx=self.ctx;sim=self.sim
        if self.bricks.brick_size!=c.brick_size:
            self.bricks.close();self.bricks=OpticalBricks(ctx,self.shape,c.brick_size)
        bind_texture(ctx,self.optical_texture,0,True)
        self.optics.run(self.shape,[sim.q,self.lut],dims=self.shape,
            capacity=sim.scene.chemistry.heat_capacity,sootExtinction=c.soot_extinction,
            scatterAlbedo=c.scatter_albedo,lutMaxK=6000.)
        self.bricks.update(self.optical_texture)
        sim.solids.bind()
        self.light_texture=self.incident.prepare(self.optical_texture,sim.h,c,sim.solids)
        self.direct.prepare(self.optical_texture,sim,c)
        if c.floor:
            if self.ground_resolution!=c.ground_resolution:
                from .gpu import delete_texture
                delete_texture(ctx,self.ground_texture)
                self.ground_resolution=c.ground_resolution
                self.ground_texture=texture3d(ctx,(self.ground_resolution,1,self.ground_resolution),5,full_precision=True)
            bind_texture(ctx,self.ground_texture,5,True)
            self.ground.run(self.ground_resolution**2,[None,self.bricks.bounds],dims=self.shape,
                spacing=sim.h,resolution=self.ground_resolution,directions=c.ground_directions,
                scatterAlbedo=c.scatter_albedo,patchOrigin=self.patch_origin,patchExtent=self.patch_extent,solidCount=len(sim.scene.solids),
                **self.bricks.uniforms(self._skip_enabled(),c.transmittance_cutoff))
        if self.physical_copy is not None and self.cfg.reconstruction_mode:
            self.physical_copy.write(sim.q.read())
        self.prepare_count+=1

    def _skip_enabled(self):
        if not self.cfg.empty_space_skipping:return False
        return self.selector.selected if self.cfg.traversal_policy=='auto' else True

    def render_hdr(self,width=None,height=None,samples=None,seed=0,prepare=True,region=None):
        if self.closed:raise RuntimeError('Renderer is closed')
        if self.sim.fault:raise RuntimeError('Cannot render a faulted simulation')
        c=self.cfg;w=width or c.width;h=height or c.height
        if type(w) is not int or type(h) is not int or min(w,h)<1 or max(w,h)>8192:raise ValueError('Invalid render dimensions')
        self._tiled_metrics=None
        x0,y0,ow,oh=(0,0,w,h) if region is None else region
        if any(type(v) is not int for v in (x0,y0,ow,oh)) or min(x0,y0)<0 or min(ow,oh)<1 or x0+ow>w or y0+oh>h:raise ValueError('Invalid render region')
        if ow*oh>8_388_608:raise ValueError('Single render region exceeds SSBO limit; use render_tiled_rgb')
        if self.last_size!=(ow,oh):
            if self.hdr is not None:self.hdr.close();self.ray_metrics.close()
            self.hdr=self.ctx.buffer((ow*oh,4));self.ray_metrics=self.ctx.buffer((ow*oh,4));self.last_size=(ow,oh)
        if prepare:self.prepare()
        elif not self.prepare_count:raise RuntimeError('prepare=False requires a prepared optical field')
        bind_texture(self.ctx,self.optical_texture,0);bind_texture(self.ctx,self.light_texture,1);bind_texture(self.ctx,self.ground_texture,5)
        bind_texture(self.ctx, self.sim.q.handle if self.sim.native_fields else self.physical_copy.handle,2)
        bind_texture(self.ctx,self.radiance_field.handle,3)
        size=np.array(self.shape)*self.sim.h;yaw=math.radians(c.yaw)
        look_x=size[0]/2 if c.camera_target_x is None else c.camera_target_x
        look_z=size[2]/2 if c.camera_target_z is None else c.camera_target_z
        target=np.array([look_x,c.target_height,look_z]);eye=np.array([look_x+c.camera_distance*math.sin(yaw),c.camera_height,look_z+c.camera_distance*math.cos(yaw)])
        forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0.,1.,0.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
        uniforms=dict(dims=self.shape,spacing=self.sim.h,
            reconstructionMode=c.reconstruction_mode,physicalCapacity=self.sim.scene.chemistry.heat_capacity,physicalExtinction=c.soot_extinction,
            resolution=(w,h),outputOrigin=(x0,y0),outputResolution=(ow,oh),patchOrigin=self.patch_origin,patchExtent=self.patch_extent,cameraPosition=tuple(eye),
            cameraForward=tuple(forward),cameraRight=tuple(right),cameraUp=tuple(up),
            tanHalfFov=math.tan(math.radians(c.fov)/2),stepScale=c.step_scale,scatterAlbedo=c.scatter_albedo,
            sampleSeed=int(seed),stableSampling=int(c.stable_sampling),samples=int(samples or c.samples),
            background=c.background,floorEnabled=int(c.floor),solidCount=len(self.sim.scene.solids),
            keyDirection=c.key_direction,keyColor=c.key_color,keyIntensity=c.key_intensity,phaseG=c.phase_g,
            **self.bricks.uniforms(self._skip_enabled(),c.transmittance_cutoff))
        self.sim.solids.bind()
        bind_texture(self.ctx,self.direct.texture,7)
        bindings=[self.hdr,self.bricks.bounds,self.ray_metrics,None,self.sim.solids.records]
        if self.particles:
            camera={k:uniforms[k] for k in ("cameraPosition","cameraForward","cameraRight","cameraUp","tanHalfFov")}
            self.particles.prepare(w,h,camera,c)
            bindings=[self.hdr,self.bricks.bounds,self.ray_metrics,self.particles.counts,self.sim.solids.records,self.particles.indices,self.particles.projected,self.particles.overflow]
            uniforms.update(self.particles.uniforms())
        if c.empty_space_skipping and c.traversal_policy=='auto' and not self.particles:
            occupancy=self.bricks.statistics()['exact_empty_fraction']
            def measure(enabled):
                cw=min(ow,96);ch=min(oh,128)
                trial={**uniforms,'resolution':(cw,ch),'outputOrigin':(0,0),'outputResolution':(cw,ch),'emptySpaceSkipping':int(enabled),'samples':1}
                self.ctx.finish();start=time.perf_counter()
                self.march.run(cw*ch,bindings,**trial);self.ctx.finish()
                return max((time.perf_counter()-start)*1000,1.e-9)
            uniforms['emptySpaceSkipping']=int(self.selector.choose(occupancy,measure))
        # An overflowed trial is never published. Grow only from measured
        # candidate demand, cache the specialization, then integrate again from
        # the same immutable gas/particle state. No clipping or particle loss.
        while True:
            if self.particles:self.particles.overflow.zero()
            self.march.run(ow*oh,bindings,**uniforms)
            self.ctx.check()
            needed=self.particles.event_demand() if self.particles else 0
            if not needed:break
            from .particle_render import grown_capacity
            if needed<=self.particle_event_capacity:
                raise RuntimeError('Inconsistent per-sample particle event demand')
            capacity=grown_capacity(needed,self.sim.scene.embers.capacity)
            if capacity not in self.particle_event_variants:
                self.particle_event_variants[capacity]=self.ctx.program(SHADERS/'render.comp',
                    {'PARTICLE_EVENTS':1,'PARTICLE_EVENT_CAPACITY':capacity})
            self.march=self.particle_event_variants[capacity]
            self.particle_event_capacity=capacity
            self.particle_event_growths+=1
        if self.particles:self.particles.check()
        return self.hdr.read().reshape(oh,ow,4)

    def render_tiled_rgb(self,width=None,height=None,samples=None,seed=0,prepare=True,band_height=128):
        """Native full-resolution rays in bounded row bands, never upscaling.

        Camera, random sequence, particle tiles, and volume samples use global
        pixel coordinates. A frozen field is identical to untiled integration.
        Bloom must be disabled: per-band convolution would create seams.
        """
        w=width or self.cfg.width;h=height or self.cfg.height
        if self.cfg.bloom!=0:raise ValueError('Tiled RGB requires bloom=0; bloom is nonlocal')
        if type(band_height) is not int or band_height<1:raise ValueError('Invalid band height')
        result=np.empty((h,w,3),np.uint8);totals={k:0 for k in ('primary_evaluated_segments','primary_skipped_segments','primary_candidate_segments','camera_rays')}
        if prepare:self.prepare()
        for y in range(0,h,band_height):
            bh=min(band_height,h-y)
            hdr=self.render_hdr(width=w,height=h,samples=samples,seed=seed,prepare=False,region=(0,y,w,bh))
            result[y:y+bh]=tone_map(hdr,self.cfg.exposure,0.)
            stats=self.statistics()
            for key in totals:totals[key]+=stats[key]
        self._tiled_metrics=totals
        return result

    def statistics(self):
        if self.ray_metrics is None:raise RuntimeError('Render at least one image before reading statistics')
        a=self.ray_metrics.read().astype(np.float64).sum(axis=0)
        return {'particle_tile_capacity':self.particles.shape[2] if self.particles else 0,
            'particle_max_tile_occupancy':self.particles.max_tile_occupancy if self.particles else 0,
            'particle_tile_growths':self.particles.tile_growths if self.particles else 0,
            'particle_event_capacity':self.particle_event_capacity if self.particles else 0,
            'particle_event_growths':self.particle_event_growths,
            **self.bricks.statistics(),'primary_evaluated_segments':int(a[0]),
            'primary_skipped_segments':int(a[1]),'primary_candidate_segments':int(a[2]),
            'camera_rays':int(a[3]),'sampling':'fixed spatial sequence' if self.cfg.stable_sampling else 'frame-varying',
            'selected_traversal':'bricks' if self._skip_enabled() else 'dense','traversal_policy':self.cfg.traversal_policy,'calibrations':self.selector.measurements,'illumination_directions':self.cfg.illumination_directions,'scattering_iterations':self.cfg.scattering_iterations,**(getattr(self,'_tiled_metrics',None) or {})}

    def render(self,**kwargs):
        return tone_map(self.render_hdr(**kwargs),self.cfg.exposure,self.cfg.bloom)

    def close(self):
        from .gpu import delete_texture
        if self.closed:return
        self.bricks.close();self.incident.close();self.direct.close();self.lut.close()
        self.radiance_field.close()
        if self.physical_copy:self.physical_copy.close()
        if self.particles:self.particles.close()
        if self.hdr is not None:self.hdr.close();self.ray_metrics.close()
        if not self.borrowed_optical:delete_texture(self.ctx,self.optical_texture)
        delete_texture(self.ctx,self.ground_texture)
        self.closed=True
