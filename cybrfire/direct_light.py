"""Directional transmittance cache, independent of thermal self-illumination."""
from pathlib import Path
from .gpu import texture3d,bind_texture,delete_texture
class DirectLight:
    def __init__(self,context,shape,scale):
        self.ctx=context;self.shape=tuple(max(1,n//scale) for n in shape)
        self.texture=texture3d(context,self.shape,7,full_precision=True,edge=True)
        self.kernel=context.program(Path(__file__).resolve().parents[1]/'shaders/direct_light.comp')
    def prepare(self,optical,sim,cfg):
        bind_texture(self.ctx,optical,0);bind_texture(self.ctx,self.texture,7,True);sim.solids.bind()
        self.kernel.run(self.shape,[],dims=self.shape,volumeDims=sim.shape,spacing=sim.h,
            keyDirection=cfg.key_direction,keyIntensity=cfg.key_intensity,shadowStep=cfg.shadow_step_cells,solidCount=len(sim.scene.solids))
        bind_texture(self.ctx,self.texture,7)
    def close(self):delete_texture(self.ctx,self.texture)
