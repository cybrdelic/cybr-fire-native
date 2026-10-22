"""Minimal EGL/OpenGL compute runtime. No Python graphics package required.

This runtime explicitly requires a compute-capable OpenGL 4.3+ implementation.
Software adapters are identified, never relabeled as hardware acceleration.
"""
import ctypes as C
import ctypes.util
import os
from pathlib import Path
import numpy as np

class ComputeContext:
    def __init__(self):
        name = ctypes.util.find_library('EGL')
        if not name:
            raise RuntimeError('EGL not installed. Use Linux/WSL with Mesa or GPU vendor EGL.')
        self.egl = C.CDLL(name)
        e = self.egl
        e.eglGetProcAddress.argtypes=[C.c_char_p];e.eglGetProcAddress.restype=C.c_void_p
        platform_ptr=e.eglGetProcAddress(b'eglGetPlatformDisplayEXT')
        if not platform_ptr:raise RuntimeError('EGL_EXT_platform_base is unavailable')
        platform=C.CFUNCTYPE(C.c_void_p,C.c_uint,C.c_void_p,C.POINTER(C.c_int))(platform_ptr)
        e.eglInitialize.argtypes=[C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_int)];e.eglInitialize.restype=C.c_uint
        # EGL_PLATFORM_DEVICE_EXT supports headless vendor GPUs; Mesa's
        # surfaceless platform is a fallback, not an assumed NVIDIA capability.
        candidates=[];requested=os.environ.get('CYBR_EGL_DEVICE')
        query_ptr=e.eglGetProcAddress(b'eglQueryDevicesEXT')
        if query_ptr:
            query=C.CFUNCTYPE(C.c_uint,C.c_int,C.POINTER(C.c_void_p),C.POINTER(C.c_int))(query_ptr)
            devices=(C.c_void_p*32)();count=C.c_int()
            if query(32,devices,C.byref(count)):
                indices=range(count.value) if requested is None else [int(requested)]
                for index in indices:
                    if not 0<=index<count.value:raise RuntimeError('CYBR_EGL_DEVICE is outside the enumerated device range')
                    candidates.append((platform(0x313F,devices[index],None),'device',index))
        if requested is not None and not candidates:raise RuntimeError('Requested EGL device is not available')
        if requested is None:candidates.append((platform(0x31DD,None,None),'surfaceless',None))
        major,minor=C.c_int(),C.c_int();self.display=None
        for display,kind,index in candidates:
            if display and e.eglInitialize(display,C.byref(major),C.byref(minor)):
                self.display=display;self.egl_platform=kind;self.egl_device_index=index;break
        if not self.display:raise RuntimeError('Cannot initialize a device or surfaceless EGL display')
        e.eglBindAPI.argtypes=[C.c_uint];e.eglBindAPI.restype=C.c_uint
        if not e.eglBindAPI(0x30A2):raise RuntimeError('Cannot bind OpenGL')
        attrs=(C.c_int*7)(0x3033,1,0x3040,8,0x3024,8,0x3038)
        config=C.c_void_p();count=C.c_int()
        e.eglChooseConfig.argtypes=[C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_void_p),C.c_int,C.POINTER(C.c_int)];e.eglChooseConfig.restype=C.c_uint
        if not e.eglChooseConfig(self.display,attrs,C.byref(config),1,C.byref(count)) or not count.value:raise RuntimeError('No EGL OpenGL configuration')
        ca=(C.c_int*7)(0x3098,4,0x30FB,3,0x30FD,1,0x3038)
        e.eglCreateContext.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_int)];e.eglCreateContext.restype=C.c_void_p
        self.context=e.eglCreateContext(self.display,config,None,ca)
        if not self.context:raise RuntimeError('OpenGL 4.3 compute context unavailable')
        e.eglMakeCurrent.argtypes=[C.c_void_p]*4;e.eglMakeCurrent.restype=C.c_uint
        if not e.eglMakeCurrent(self.display,None,None,self.context):raise RuntimeError('Cannot make EGL context current')
        self._load()
        self.info={k:self.glGetString(v).decode() for k,v in [('vendor',0x1F00),('renderer',0x1F01),('version',0x1F02)]}
        self.info['egl_platform']=self.egl_platform;self.info['egl_device_index']=self.egl_device_index
        self.dispatch_width=65535
        self.info['software_adapter']=any(x in self.info['renderer'].lower() for x in ['llvmpipe','softpipe','swiftshader','software'])
        self.buffers=[];self.programs=[];self.closed=False
    def _fn(self,name,restype,*args):
        ptr=self.egl.eglGetProcAddress(name.encode())
        if not ptr:raise RuntimeError('Missing OpenGL function '+name)
        fn=C.CFUNCTYPE(restype,*args)(ptr);setattr(self,name,fn)
    def _load(self):
        U=C.c_uint;I=C.c_int;P=C.c_void_p;F=C.c_float;S=C.c_char_p
        for name,ret,args in [
            ('glGetString',S,[U]),('glGetError',U,[]),('glGetIntegerv',None,[U,C.POINTER(I)]),
            ('glCreateShader',U,[U]),('glShaderSource',None,[U,I,C.POINTER(S),C.POINTER(I)]),('glCompileShader',None,[U]),
            ('glGetShaderiv',None,[U,U,C.POINTER(I)]),('glGetShaderInfoLog',None,[U,I,C.POINTER(I),P]),
            ('glCreateProgram',U,[]),('glAttachShader',None,[U,U]),('glLinkProgram',None,[U]),('glGetProgramiv',None,[U,U,C.POINTER(I)]),
            ('glGetProgramInfoLog',None,[U,I,C.POINTER(I),P]),('glGetActiveUniform',None,[U,U,I,C.POINTER(I),C.POINTER(I),C.POINTER(U),P]),('glDeleteShader',None,[U]),('glDeleteProgram',None,[U]),
            ('glUseProgram',None,[U]),('glGetUniformLocation',I,[U,S]),('glUniform1i',None,[I,I]),('glUniform1f',None,[I,F]),
            ('glUniform2i',None,[I,I,I]),('glUniform2f',None,[I,F,F]),('glUniform4i',None,[I,I,I,I,I]),('glUniform3i',None,[I,I,I,I]),('glUniform3f',None,[I,F,F,F]),('glUniform4f',None,[I,F,F,F,F]),
            ('glGenBuffers',None,[I,C.POINTER(U)]),('glBindBuffer',None,[U,U]),('glBufferData',None,[U,C.c_ssize_t,P,U]),
            ('glBufferSubData',None,[U,C.c_ssize_t,C.c_ssize_t,P]),('glGetBufferSubData',None,[U,C.c_ssize_t,C.c_ssize_t,P]),
            ('glBindBufferBase',None,[U,U,U]),('glDeleteBuffers',None,[I,C.POINTER(U)]),
            ('glDispatchCompute',None,[U,U,U]),('glMemoryBarrier',None,[U]),('glFinish',None,[]),
        ]:self._fn(name,ret,*args)
    def buffer(self,shape,dtype=np.float32,data=None):
        arr=np.zeros(shape,dtype=dtype) if data is None else np.ascontiguousarray(data,dtype=dtype)
        handle=C.c_uint();self.glGenBuffers(1,C.byref(handle));self.glBindBuffer(0x90D2,handle.value)
        self.glBufferData(0x90D2,arr.nbytes,arr.ctypes.data,0x88E8)
        b=Buffer(self,handle.value,arr.shape,arr.dtype);self.buffers.append(b);return b
    def program(self,path,defines=None):
        path=Path(path)
        def expand(p):
            lines=[]
            for line in p.read_text().splitlines():
                if line.startswith('#include "'):lines.append(expand(p.parent/line.split('"')[1]))
                else:lines.append(line)
            return '\n'.join(lines)
        src=expand(path)
        if defines:
            src=src.replace('#version 430','#version 430\n'+'\n'.join('#define '+k+' '+str(v) for k,v in defines.items()))
        sh=self.glCreateShader(0x91B9);raw=src.encode();pp=C.c_char_p(raw)
        self.glShaderSource(sh,1,C.byref(pp),None);self.glCompileShader(sh)
        ok=C.c_int();self.glGetShaderiv(sh,0x8B81,C.byref(ok))
        if not ok.value:
            log=C.create_string_buffer(32768);self.glGetShaderInfoLog(sh,len(log),None,log)
            raise RuntimeError(f'{path.name}: '+log.value.decode())
        pr=self.glCreateProgram();self.glAttachShader(pr,sh);self.glLinkProgram(pr);self.glDeleteShader(sh)
        self.glGetProgramiv(pr,0x8B82,C.byref(ok))
        if not ok.value:
            log=C.create_string_buffer(32768);self.glGetProgramInfoLog(pr,len(log),None,log)
            raise RuntimeError('Link '+path.name+': '+log.value.decode())
        self.programs.append(pr);return Program(self,pr,path.name)
    def finish(self):self.glFinish()
    def check(self):
        err=self.glGetError()
        if err:raise RuntimeError('OpenGL error '+hex(err))
    def close(self):
        if self.closed:return
        self.finish()
        for b in list(self.buffers):b.close()
        for f in getattr(self,"fields",[]):f.close()
        for p in self.programs:self.glDeleteProgram(p)
        if hasattr(self,'textures'):
            for handle in self.textures:
                v=C.c_uint(handle);self.glDeleteTextures(1,C.byref(v))
            self.textures=[]
        self.buffers=[];self.programs=[];self.closed=True
        self.egl.eglMakeCurrent(self.display,None,None,None)
        self.egl.eglDestroyContext.argtypes=[C.c_void_p,C.c_void_p]
        self.egl.eglDestroyContext(self.display,self.context)
        self.egl.eglTerminate.argtypes=[C.c_void_p];self.egl.eglTerminate(self.display)

class Buffer:
    def __init__(self,ctx,handle,shape,dtype):self.ctx,self.handle,self.shape,self.dtype=ctx,handle,shape,dtype
    def close(self):
        if self.handle:
            v=C.c_uint(self.handle);self.ctx.glDeleteBuffers(1,C.byref(v));self.handle=0
            if self in self.ctx.buffers:self.ctx.buffers.remove(self)
    def read(self):
        if not self.handle:raise RuntimeError("Buffer is closed")
        out=np.empty(self.shape,self.dtype);self.ctx.glBindBuffer(0x90D2,self.handle)
        self.ctx.glGetBufferSubData(0x90D2,0,out.nbytes,out.ctypes.data);return out
    def write(self,data):
        if not self.handle:raise RuntimeError("Buffer is closed")
        a=np.ascontiguousarray(data,dtype=self.dtype)
        if a.shape!=self.shape:raise ValueError(f'Shape mismatch {a.shape} != {self.shape}')
        self.ctx.glBindBuffer(0x90D2,self.handle);self.ctx.glBufferSubData(0x90D2,0,a.nbytes,a.ctypes.data)
    def zero(self):self.write(np.zeros(self.shape,self.dtype))

class Program:
    def __init__(self,ctx,handle,name):
        self.ctx,self.handle,self.name,self.locations=ctx,handle,name,{}
        # Shader declarations determine upload types, not JSON integer spelling.
        count=C.c_int();ctx.glGetProgramiv(handle,0x8B86,C.byref(count));self.types={}
        for index in range(count.value):
            length=C.c_int();size=C.c_int();kind=C.c_uint();label=C.create_string_buffer(512)
            ctx.glGetActiveUniform(handle,index,512,C.byref(length),C.byref(size),C.byref(kind),label)
            self.types[label.value.decode().removesuffix('[0]')]=kind.value
    def run(self,shape,bindings=(),**uniforms):
        c=self.ctx
        if c.closed:raise RuntimeError("Compute context is closed")
        c.glUseProgram(self.handle)
        for i,b in enumerate(bindings):
            if b is not None:
                if hasattr(b,"bind_field"):b.bind_field(i)
                else:c.glBindBufferBase(0x90D2,i,b.handle)
        for name,val in uniforms.items():
            if name not in self.locations:self.locations[name]=c.glGetUniformLocation(self.handle,name.encode())
            l=self.locations[name]
            if l<0:continue
            kind=self.types.get(name)
            if kind==0x1406:c.glUniform1f(l,float(val))
            elif kind in (0x1404,0x8B56):c.glUniform1i(l,int(val))
            elif kind==0x8B50:c.glUniform2f(l,*map(float,val))
            elif kind==0x8B51:c.glUniform3f(l,*map(float,val))
            elif kind==0x8B52:c.glUniform4f(l,*map(float,val))
            elif kind in (0x8B53,0x8B57):c.glUniform2i(l,*map(int,val))
            elif kind in (0x8B54,0x8B58):c.glUniform3i(l,*map(int,val))
            elif kind in (0x8B55,0x8B59):c.glUniform4i(l,*map(int,val))
            else:raise TypeError(name+' has unsupported GLSL uniform type '+str(kind))
        # All kernels use 128 one-dimensional invocations per workgroup.
        n=int(np.prod(shape));groups=(n+127)//128
        if n<1:return
        gx=min(groups,c.dispatch_width);gy=(groups+gx-1)//gx
        c.glDispatchCompute(gx,gy,1);c.glMemoryBarrier(0xFFFFFFFF)

# Texture functions are loaded lazily to keep the compute-only tests independent.
def texture3d(ctx, dims, unit, full_precision=False, edge=False):
    if not hasattr(ctx,'glGenTextures'):
        U=C.c_uint;I=C.c_int;P=C.c_void_p
        for name,ret,args in [
          ('glGenTextures',None,[I,C.POINTER(U)]),('glBindTexture',None,[U,U]),
          ('glTexStorage3D',None,[U,I,U,I,I,I]),('glTexParameteri',None,[U,U,I]),
          ('glActiveTexture',None,[U]),('glBindImageTexture',None,[U,U,I,C.c_ubyte,I,U,U]),
          ('glDeleteTextures',None,[I,C.POINTER(U)]),
        ]:ctx._fn(name,ret,*args)
        ctx.textures=[];ctx.texture_formats={}
    handle=C.c_uint();ctx.glGenTextures(1,C.byref(handle));ctx.glActiveTexture(0x84C0+unit)
    fmt=0x8814 if full_precision else 0x881A
    ctx.glBindTexture(0x806F,handle.value);ctx.glTexStorage3D(0x806F,1,fmt,*dims)
    for p in [0x2801,0x2800]:ctx.glTexParameteri(0x806F,p,0x2601)
    for p in [0x2802,0x2803,0x8072]:ctx.glTexParameteri(0x806F,p,0x812F if edge else 0x812D)
    ctx.glBindImageTexture(unit,handle.value,0,1,0,0x88B9,fmt)
    ctx.textures.append(handle.value);ctx.texture_formats[handle.value]=fmt;return handle.value

def bind_texture(ctx, handle, unit, image=False):
    ctx.glActiveTexture(0x84C0+unit);ctx.glBindTexture(0x806F,handle)
    if image:ctx.glBindImageTexture(unit,handle,0,1,0,0x88B9,ctx.texture_formats[handle])


def delete_texture(ctx, handle):
    """Release a texture and remove it from context ownership exactly once."""
    if handle in getattr(ctx, 'textures', []):
        value = C.c_uint(handle)
        ctx.glDeleteTextures(1, C.byref(value))
        ctx.textures.remove(handle)
        ctx.texture_formats.pop(handle, None)


def bind_image(ctx, handle, unit, access=0x88B9):
    """Bind compute image storage without changing any sampled-texture binding.

    Texture unit N and image unit N are distinct namespaces. Coupling these
    states can make a pass sample its output while writing it, which is a race.
    """
    if handle not in ctx.texture_formats:raise ValueError("Unknown image handle")
    ctx.glBindImageTexture(unit,handle,0,1,0,access,ctx.texture_formats[handle])
