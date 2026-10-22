"""Transport-local full-precision texture cache, invalidated at every mutation.

Buffers remain authoritative. These textures are numerical sampling caches,
not render detail, alternate scalar fields, or an upsampled flow simulation.
"""
from pathlib import Path
import numpy as np
from .gpu import texture3d,bind_texture

class TextureTransport:
    def __init__(self,ctx,shape,trace_courant=2.,max_trace_steps=4,specialize=True):
        self.solid_count=0;self.trace_courant=trace_courant;self.max_trace_steps=max_trace_steps
        self.ctx=ctx;self.shape=tuple(shape);self.shapes=[self.shape,tuple(n+1 for n in shape)]
        self.original=[texture3d(ctx,d,2,True) for d in self.shapes]
        self.predicted=[texture3d(ctx,d,3,True) for d in self.shapes]
        shaders=Path(__file__).resolve().parents[1]/'shaders'
        self.copy=ctx.program(shaders/'copy_texture.comp')
        self.specialized=specialize
        if specialize:
            self.kernels={(kind,stage):ctx.program(shaders/'transport.comp',
                {'TEXTURE_TRANSPORT':1,'TRANSPORT_KIND':kind,'TRANSPORT_CORRECTION':stage})
                for kind in (0,1) for stage in (0,1)}
        else:
            reference=ctx.program(shaders/'transport.comp',{'TEXTURE_TRANSPORT':1})
            self.kernels={(kind,stage):reference for kind in (0,1) for stage in (0,1)}
    def advect(self,original,pred,out,velocity,kind,dt,spacing,corrected=True,periodic=False):
        ctx=self.ctx;d=self.shapes[kind]
        # Always refresh after source/force/projection writes. A frame number is
        # deliberately NOT used as a cache key: several stages mutate per tick.
        if kind==0:
            bind_texture(ctx,self.original[1],2,True)
            self.copy.run(self.shapes[1],[velocity],dims=self.shapes[1])
        bind_texture(ctx,self.original[kind],2,True)
        self.copy.run(d,[original],dims=d)
        bind_texture(ctx,self.original[kind],2)
        bind_texture(ctx,self.predicted[kind],3,True)
        bind_texture(ctx,self.original[1],4)
        kw=dict(dims=self.shape,spacing=spacing,dt=dt,velocityField=int(kind),periodic=int(periodic),traceCourant=self.trace_courant,maxTraceSteps=self.max_trace_steps,solidCount=self.solid_count)
        self.kernels[kind,0].run(d,[original,pred,velocity,pred],correction=0,**kw)
        if corrected:self.kernels[kind,1].run(d,[original,pred,velocity,out],correction=1,**kw)
        else:out.write(pred.read())
