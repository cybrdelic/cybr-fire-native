"""Authoritative RGBA32F volume storage with zero-copy transport sampling.

Unlike an SSBO containing vec4[cell_count], a 3D image is not constrained by
GL_MAX_SHADER_STORAGE_BLOCK_SIZE. The source, force, projection, chemistry,
particle and optical kernels read the same 32-bit values. No half precision,
resampled render field, or synthesized detail is introduced by this backend.

Image units 8..11 are reserved for numerical fields; units 0..7 retain the
renderer/legacy-cache contract. Field ownership is independent of stage binding.
"""
from pathlib import Path
import ctypes as C
import numpy as np
from .gpu import texture3d, bind_texture, delete_texture

class ImageField:
    def __init__(self, ctx, dimensions, full_precision=True):
        self.ctx=ctx;self.grid_shape=tuple(int(n) for n in dimensions)
        if len(self.grid_shape)!=3 or min(self.grid_shape)<1:raise ValueError('Invalid 3D field dimensions')
        limit=C.c_int();ctx.glGetIntegerv(0x8073,C.byref(limit))
        if max(self.grid_shape)>limit.value:raise ValueError('Field exceeds GL_MAX_3D_TEXTURE_SIZE')
        ctx.glGetIntegerv(0x8F38,C.byref(limit)) # GL_MAX_IMAGE_UNITS
        if limit.value<12:raise RuntimeError('Native field path requires 12 image binding units')
        self.shape=(int(np.prod(self.grid_shape)),4);self.dtype=np.dtype(np.float32)
        self.full_precision=full_precision
        self.handle=texture3d(ctx,self.grid_shape,8,full_precision=full_precision)
        if not hasattr(ctx,'glTexSubImage3D'):
            U=C.c_uint;I=C.c_int;P=C.c_void_p
            ctx._fn('glTexSubImage3D',None,U,I,I,I,I,I,I,I,U,U,P)
            ctx._fn('glGetTexImage',None,U,I,U,U,P)
        if not hasattr(ctx,'fields'):ctx.fields=[]
        ctx.fields.append(self);self.zero();ctx.check()
    @property
    def nbytes(self):return int(np.prod(self.shape))*4
    def bind_field(self,logical_binding):
        if not self.handle:raise RuntimeError('Field is closed')
        # Read/write permits stages to use their more restrictive GLSL access
        # qualifier; visibility is established by Program.run's memory barrier.
        self.ctx.glBindImageTexture(8+logical_binding,self.handle,0,1,0,0x88BA,self.ctx.texture_formats[self.handle])
    def read(self):
        if not self.handle:raise RuntimeError('Field is closed')
        out=np.empty(self.shape,np.float32)
        bind_texture(self.ctx,self.handle,8)
        self.ctx.glGetTexImage(0x806F,0,0x1908,0x1406,out.ctypes.data)
        self.ctx.check();return out
    def read_slab(self,z,depth):
        """Bounded-memory readback. Requires the OpenGL 4.5 subimage API."""
        if not self.handle:raise RuntimeError('Field is closed')
        nx,ny,nz=self.grid_shape
        if type(z) is not int or type(depth) is not int or z<0 or depth<1 or z+depth>nz:raise ValueError('Invalid slab')
        ctx=self.ctx
        if not hasattr(ctx,'glGetTextureSubImage'):
            U=C.c_uint;I=C.c_int;P=C.c_void_p
            ctx._fn('glGetTextureSubImage',None,U,I,I,I,I,I,I,I,U,U,I,P)
        out=np.empty((depth,ny,nx,4),np.float32)
        ctx.glGetTextureSubImage(self.handle,0,0,0,z,nx,ny,depth,0x1908,0x1406,out.nbytes,out.ctypes.data)
        ctx.check();return out
    def write(self,values):
        if not self.handle:raise RuntimeError('Field is closed')
        a=np.asarray(values)
        if a.shape!=self.shape:raise ValueError(f'Shape mismatch {a.shape} != {self.shape}')
        a=np.ascontiguousarray(a,dtype=np.float32)
        bind_texture(self.ctx,self.handle,8)
        self.ctx.glTexSubImage3D(0x806F,0,0,0,0,*self.grid_shape,0x1908,0x1406,a.ctypes.data)
        self.ctx.glMemoryBarrier(0xFFFFFFFF);self.ctx.check()
    def zero(self):
        nx,ny,nz=self.grid_shape
        # Bound transient host memory rather than allocating another full field.
        depth=min(nz,8);slab=np.zeros((depth,ny,nx,4),np.float32)
        bind_texture(self.ctx,self.handle,8)
        for z in range(0,nz,depth):
            self.ctx.glTexSubImage3D(0x806F,0,0,0,z,nx,ny,min(depth,nz-z),0x1908,0x1406,slab.ctypes.data)
        self.ctx.glMemoryBarrier(0xFFFFFFFF)
    def close(self):
        if self.handle:delete_texture(self.ctx,self.handle);self.handle=0

class ImageTransport:
    """Same corrected characteristic integrator, no field-to-cache copies."""
    def __init__(self,ctx,shape,trace_courant=2.,max_trace_steps=4):
        self.ctx=ctx;self.shape=tuple(shape);self.shapes=[self.shape,tuple(n+1 for n in shape)]
        self.trace_courant=trace_courant;self.max_trace_steps=max_trace_steps;self.solid_count=0
        shader=Path(__file__).resolve().parents[1]/'shaders/transport.comp'
        self.kernels={(kind,stage):ctx.program(shader,{'TEXTURE_TRANSPORT':1,'NATIVE_FIELDS':1,
            'TRANSPORT_KIND':kind,'TRANSPORT_CORRECTION':stage}) for kind in (0,1) for stage in (0,1)}
    def advect(self,original,pred,out,velocity,kind,dt,spacing,corrected=True,periodic=False):
        for unit,f in ((2,original),(3,pred),(4,velocity)):bind_texture(self.ctx,f.handle,unit)
        kw=dict(dims=self.shape,spacing=spacing,dt=dt,velocityField=int(kind),periodic=int(periodic),
                traceCourant=self.trace_courant,maxTraceSteps=self.max_trace_steps,solidCount=self.solid_count)
        self.kernels[kind,0].run(self.shapes[kind],[original,pred,velocity,pred],correction=0,**kw)
        if corrected:self.kernels[kind,1].run(self.shapes[kind],[original,pred,velocity,out],correction=1,**kw)
        else:out.write(pred.read())
