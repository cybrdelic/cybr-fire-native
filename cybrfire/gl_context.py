"""Context creation only: EGL headless or a hidden GLFW desktop context.

Numerical kernels and renderer are shared by both backends. This is not a CPU
solver fallback. Apple's OpenGL 4.1 does not provide the required compute;
native Mesa EGL can supply the same API through its software implementation.
"""
import ctypes as C
import os
import sys
import threading
from .egl_library import load_egl_library

_glfw_users = 0

class EGLContext:
    def __init__(self):
        self.display = self.context = None
        try:
            self._open()
        except Exception:
            self.close()
            raise
        self.info = {**self.library_info, "backend": "egl", "egl_platform": self.egl_platform,
                     "egl_device_index": self.egl_device_index}

    def _open(self):
        requested=os.environ.get('CYBR_EGL_DEVICE')
        selected=os.environ.get('CYBR_EGL_PLATFORM','auto').lower()
        if selected not in ('auto','device','surfaceless'):
            raise ValueError('CYBR_EGL_PLATFORM must be auto, device, or surfaceless')
        if requested is not None and selected=='surfaceless':
            raise ValueError('CYBR_EGL_DEVICE cannot be combined with CYBR_EGL_PLATFORM=surfaceless')
        self.egl, self.library_info = load_egl_library()
        e = self.egl
        e.eglGetProcAddress.argtypes=[C.c_char_p];e.eglGetProcAddress.restype=C.c_void_p
        platform_ptr=e.eglGetProcAddress(b'eglGetPlatformDisplayEXT')
        if not platform_ptr:raise RuntimeError('EGL_EXT_platform_base is unavailable')
        platform=C.CFUNCTYPE(C.c_void_p,C.c_uint,C.c_void_p,C.POINTER(C.c_int))(platform_ptr)
        e.eglInitialize.argtypes=[C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_int)];e.eglInitialize.restype=C.c_uint
        # EGL_PLATFORM_DEVICE_EXT supports headless vendor GPUs; Mesa's
        # surfaceless platform is preferred on macOS and explicitly selectable.
        candidates=[]
        surfaceless_first = selected=='surfaceless' or (selected=='auto' and sys.platform=='darwin' and requested is None)
        if surfaceless_first:
            candidates.append((platform(0x31DD,None,None),'surfaceless',None))
        query_ptr=e.eglGetProcAddress(b'eglQueryDevicesEXT')
        if query_ptr and selected!='surfaceless':
            query=C.CFUNCTYPE(C.c_uint,C.c_int,C.POINTER(C.c_void_p),C.POINTER(C.c_int))(query_ptr)
            devices=(C.c_void_p*32)();count=C.c_int()
            if query(32,devices,C.byref(count)):
                indices=range(count.value) if requested is None else [int(requested)]
                for index in indices:
                    if not 0<=index<count.value:raise RuntimeError('CYBR_EGL_DEVICE is outside the enumerated device range')
                    candidates.append((platform(0x313F,devices[index],None),'device',index))
        if (requested is not None or selected=='device') and not candidates:raise RuntimeError('Requested EGL device is not available')
        if requested is None and selected=='auto' and not surfaceless_first:
            candidates.append((platform(0x31DD,None,None),'surfaceless',None))
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
        ca=(C.c_int*7)(0x3098,4,0x30FB,5,0x30FD,1,0x3038)
        e.eglCreateContext.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_int)];e.eglCreateContext.restype=C.c_void_p
        self.context=e.eglCreateContext(self.display,config,None,ca)
        if not self.context:raise RuntimeError('OpenGL 4.5 compute context unavailable')
        e.eglMakeCurrent.argtypes=[C.c_void_p]*4;e.eglMakeCurrent.restype=C.c_uint
        if not e.eglMakeCurrent(self.display,None,None,self.context):raise RuntimeError('Cannot make EGL context current')

    def address(self, name):
        return self.egl.eglGetProcAddress(name.encode())

    def close(self):
        if not getattr(self, "display", None) or not hasattr(self, "egl"):
            return
        if self.context:
            self.egl.eglMakeCurrent(self.display, None, None, None)
            self.egl.eglDestroyContext.argtypes = [C.c_void_p, C.c_void_p]
            self.egl.eglDestroyContext(self.display, self.context)
            self.context = None
        self.egl.eglTerminate.argtypes = [C.c_void_p]
        self.egl.eglTerminate(self.display)
        self.display = None

class GLFWContext:
    def __init__(self):
        global _glfw_users
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("GLFW context creation must run on the main thread.")
        try:
            import glfw
        except ImportError as exc:
            raise RuntimeError("GLFW binding missing; install requirements.txt.") from exc
        self.glfw = glfw
        self.window = None
        self.active = False
        if not _glfw_users and not glfw.init():
            code, message = glfw.get_error()
            raise RuntimeError(f"GLFW initialization failed ({code}): {message!r}")
        _glfw_users += 1
        self.active = True
        try:
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            self.window = glfw.create_window(16, 16, "CYBR FIRE compute", None, None)
            if not self.window:
                code, message = glfw.get_error()
                raise RuntimeError(f"OpenGL 4.5 context unavailable ({code}): {message!r}")
            glfw.make_context_current(self.window)
            self.info = {"backend": "glfw", "egl_platform": None, "egl_device_index": None}
        except Exception:
            self.close()
            raise

    def address(self, name):
        pointer = self.glfw.get_proc_address(name)
        return pointer if isinstance(pointer, int) else C.cast(pointer, C.c_void_p).value

    def close(self):
        global _glfw_users
        if not self.active:
            return
        if self.window:
            self.glfw.destroy_window(self.window)
            self.window = None
        _glfw_users -= 1
        if not _glfw_users:
            self.glfw.terminate()
        self.active = False

def open_context():
    requested = os.environ.get("CYBR_GL_BACKEND", "auto").lower()
    choices = {"egl": EGLContext, "glfw": GLFWContext}
    if requested not in ("auto", *choices):
        raise ValueError("CYBR_GL_BACKEND must be auto, egl, or glfw")
    order = (["egl", "glfw"] if sys.platform.startswith("linux") or sys.platform=="darwin" else ["glfw", "egl"])
    if requested != "auto":
        order = [requested]
    elif any(os.environ.get(key) is not None for key in ("CYBR_EGL_DEVICE", "CYBR_EGL_LIBRARY", "CYBR_EGL_PLATFORM")):
        order = ["egl"]
    errors = []
    for name in order:
        try:
            return choices[name]()
        except (RuntimeError, OSError, ImportError) as exc:
            errors.append(f"{name}: {exc}")
    hint = (" Apple's OpenGL stops at 4.1; use an existing native Mesa EGL/llvmpipe "
            "installation with CYBR_GL_BACKEND=egl and CYBR_EGL_PLATFORM=surfaceless, "
            "or the documented Linux container. Native macOS compute still requires actual driver verification." if sys.platform == "darwin" else
            " Install a current OpenGL 4.5 GPU driver, or Mesa for Linux software compute.")
    raise RuntimeError("No supported OpenGL compute context. " + " | ".join(errors) + hint)
