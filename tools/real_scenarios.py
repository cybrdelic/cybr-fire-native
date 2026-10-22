"""Physically distinguishable independent scene initial conditions.

All cameras are presentation only. The expansion model is a compressible-
impulse VFX approximation and is not a compressible shock solver.
"""
from dataclasses import replace
from cybrfire.config import Scene, Domain, Chemistry, Flow, Render, Emitter, SurfaceFuel


def make_scene(name: str, shape=(160,280,160)) -> Scene:
    shape=tuple(shape)
    domain=Domain(shape=shape,spacing=1.92/shape[0],tick=1/120,
                  native_fields=True,trace_courant=1.,max_trace_steps=16)
    render=replace(Render(),floor=False,width=540,height=960,fov=40.,
                   camera_distance=3.55,camera_height=1.4,target_height=1.05,
                   yaw=12.,empty_space_skipping=False,key_intensity=0.,
                   shadow_resolution_scale=4,reconstruction_mode=1,step_scale=.65,
                   samples=2,stable_sampling=False,bloom=0.)
    if name=='jet':
        # Long lateral momentum jet; sustained fuel supply with transverse modes.
        src=Emitter(center=(.18,.34,.96),radii=(.11,.12,.13),
             velocity=(3.5,.5,0.),fuel=.12,oxygen_deficit=.13,
             temperature=1640.,replacement_rate=170.,
             turbulent_speed=1.5,turbulent_length=.18,frequency=7.,modulation=.12)
        scene=Scene(name='Momentum-driven flame jet',domain=domain,emitters=(src,),
              flow=Flow(wind=(.45,0.,0.),drag=.25,confinement=1.4),
              render=replace(render,yaw=0.,target_height=.65,camera_distance=1.25,camera_target_x=.58,fov=42.,exposure=.009))
    elif name=='pool':
        # No gas burner: ONLY a finite-area fuel sheet supplies pyrolysis gas.
        disabled=Emitter(center=(.96,.12,.96),radii=(.05,.05,.05),replacement_rate=0.,fuel=0.)
        surface=SurfaceFuel(enabled=True,center=(.96,.96),radii=(.74,.64),
             inventory=.78,initial_ignition_radius=.38,burn_rate=.16,
             ignition_temperature=565.,surface_heat_capacity=1500.,
             diffusivity=.009,heat_transfer=32.,spread_radiation=18.,
             pyrolysis_heat=6.0e6)
        scene=Scene(name='Finite-mass spreading pool fire',domain=domain,
                    emitters=(disabled,),surface_fuel=surface,
                    flow=Flow(wind=(.2,0.,0.),confinement=1.5,entrainment_rate=12.,entrainment_height=.38),
                    render=replace(render,yaw=22.,target_height=.96,camera_distance=3.25))
    elif name=='fireball':
        # Transient expanding fuel volume with radial momentum impulse; no
        # perpetual burner. The impulse is a VFX low-Mach approximation.
        src=Emitter(center=(.96,1.15,.96),radii=(.17,.17,.17),
             velocity=(0.,.4,0.),fuel=.18,oxygen_deficit=.12,
             temperature=1750.,replacement_rate=160.,
             start=0.,end=.27,attack=.02,release=.035,
             expansion_speed=1.55,radial_speed=3.8,
             turbulent_speed=1.0,turbulent_length=.24,modulation=.10)
        scene=Scene(name='Finite-duration expanding fireball',domain=domain,
                   emitters=(src,),flow=Flow(drag=.3,confinement=2.1),
                   render=replace(render,yaw=10.,target_height=1.3,camera_distance=3.55))
    else:
        raise ValueError('Unrecognized fire scene '+name)
    scene.validate()
    return scene
