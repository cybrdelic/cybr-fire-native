"""CPU-only loader/selection/lifetime contracts; these do not prove macOS GL."""
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cybrfire import egl_library, gl_context


class FakeEGL:
    """Exercise ctypes callback boundaries without claiming driver execution."""
    def __init__(self, fail_current=False):
        self.initialized = []
        self.destroyed = []
        self.terminated = []
        self.platform_callback = C.CFUNCTYPE(C.c_void_p, C.c_uint, C.c_void_p, C.POINTER(C.c_int))(
            lambda kind, device, attrs: 100 if kind == 0x31DD else 200 + device)
        def query(size, devices, count):
            devices[0], devices[1], count[0] = 1, 2, 2
            return 1
        self.query_callback = C.CFUNCTYPE(C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))(query)
        pointers = {b"eglGetPlatformDisplayEXT": C.cast(self.platform_callback, C.c_void_p).value,
                    b"eglQueryDevicesEXT": C.cast(self.query_callback, C.c_void_p).value}
        self.eglGetProcAddress = Mock(side_effect=lambda name: pointers.get(name))
        def initialize(display, major, minor):
            self.initialized.append(display)
            major._obj.value, minor._obj.value = 1, 5
            return 1
        self.eglInitialize = Mock(side_effect=initialize)
        self.eglBindAPI = Mock(return_value=1)
        def choose(display, attrs, config, capacity, count):
            config._obj.value, count._obj.value = 7, 1
            return 1
        self.eglChooseConfig = Mock(side_effect=choose)
        self.eglCreateContext = Mock(return_value=42)
        self.eglMakeCurrent = Mock(side_effect=lambda display, draw, read, context: int(not context or not fail_current))
        self.eglDestroyContext = Mock(side_effect=lambda display, context: self.destroyed.append(context))
        self.eglTerminate = Mock(side_effect=lambda display: self.terminated.append(display))


class EGLContracts(unittest.TestCase):
    def test_explicit_missing_library_does_not_fall_back(self):
        with patch.dict(os.environ, {"CYBR_EGL_LIBRARY": "relative.dylib"}, clear=True), \
             patch.object(egl_library.ctypes.util, "find_library") as find:
            with self.assertRaisesRegex(RuntimeError, "existing absolute"):
                egl_library.load_egl_library()
            find.assert_not_called()

    def test_explicit_load_failure_is_reported_without_other_library(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libEGL.dylib"
            library.touch()
            with patch.dict(os.environ, {"CYBR_EGL_LIBRARY": str(library)}, clear=True), \
                 patch.object(egl_library.C, "CDLL", side_effect=OSError("wrong architecture")) as load:
                with self.assertRaisesRegex(RuntimeError, "No fallback"):
                    egl_library.load_egl_library()
                load.assert_called_once_with(str(library))

    def test_package_lookup_is_read_only_bounded_and_custom_prefix_works(self):
        with tempfile.TemporaryDirectory() as directory:
            libdir = Path(directory) / "custom-mesa" / "lib"
            libdir.mkdir(parents=True)
            library = libdir / "libEGL.1.dylib"
            library.touch()
            stub = types.SimpleNamespace(eglGetProcAddress=lambda: None, eglInitialize=lambda: None,
                                         eglTerminate=lambda: None)
            with patch.dict(os.environ, {}, clear=True), patch.object(egl_library.sys, "platform", "darwin"), \
                 patch.object(egl_library.ctypes.util, "find_library", return_value=None), \
                 patch.object(egl_library.Path, "is_file", lambda path: path == library), \
                 patch.object(egl_library.shutil, "which", side_effect=lambda name: "/pkg-config" if name == "pkg-config" else None), \
                 patch.object(egl_library.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout=str(libdir))) as run, \
                 patch.object(egl_library.C, "CDLL", return_value=stub):
                _, info = egl_library.load_egl_library()
                self.assertEqual(info, {"egl_library": library.name, "egl_library_source": "pkg-config"})
                run.assert_called_once_with(["/pkg-config", "--variable=libdir", "egl"],
                                            capture_output=True, text=True, timeout=5, check=False)

    def test_package_query_timeout_is_not_a_hang(self):
        with patch.object(egl_library.subprocess, "run", side_effect=subprocess.TimeoutExpired("query", 5)):
            self.assertIsNone(egl_library._command_output(["query", "--prefix", "mesa"]))

    def test_linux_soname_without_external_discovery(self):
        library = FakeEGL()
        with patch.dict(os.environ, {}, clear=True), patch.object(egl_library.sys, "platform", "linux"), \
             patch.object(egl_library.ctypes.util, "find_library", return_value=None), \
             patch.object(egl_library.C, "CDLL", return_value=library) as load:
            loaded, info = egl_library.load_egl_library()
            self.assertIs(loaded, library)
            self.assertEqual(info["egl_library_source"], "linux-soname")
            load.assert_called_once_with("libEGL.so.1")

    def test_darwin_auto_prefers_egl_and_explicit_override_is_not_ignored(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(gl_context.sys, "platform", "darwin"), \
             patch.object(gl_context, "EGLContext", return_value="egl-context") as egl, \
             patch.object(gl_context, "GLFWContext") as glfw:
            self.assertEqual(gl_context.open_context(), "egl-context")
            egl.assert_called_once()
            glfw.assert_not_called()
        with patch.dict(os.environ, {"CYBR_EGL_LIBRARY": "explicit"}, clear=True), \
             patch.object(gl_context, "EGLContext", side_effect=RuntimeError("unavailable")), \
             patch.object(gl_context, "GLFWContext") as glfw:
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                gl_context.open_context()
            glfw.assert_not_called()

    def test_platform_conflict_rejected_before_loading_library(self):
        with patch.dict(os.environ, {"CYBR_EGL_PLATFORM": "surfaceless", "CYBR_EGL_DEVICE": "0"}, clear=True), \
             patch.object(gl_context, "load_egl_library") as load:
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                gl_context.EGLContext()
            load.assert_not_called()

    def test_surfaceless_and_explicit_device_selection_and_close(self):
        cases = [({}, "darwin", "surfaceless", None, 100),
                 ({}, "linux", "device", 0, 201),
                 ({"CYBR_EGL_PLATFORM": "surfaceless"}, "darwin", "surfaceless", None, 100),
                 ({"CYBR_EGL_DEVICE": "1"}, "darwin", "device", 1, 202)]
        for environment, host, expected, device, display in cases:
            with self.subTest(environment=environment):
                fake = FakeEGL()
                with patch.dict(os.environ, environment, clear=True), patch.object(gl_context.sys, "platform", host), \
                     patch.object(gl_context, "load_egl_library", return_value=(fake, {})):
                    context = gl_context.EGLContext()
                    self.assertEqual(context.info["egl_platform"], expected)
                    self.assertEqual(context.info["egl_device_index"], device)
                    self.assertEqual(fake.initialized, [display])
                    context.close()
                    context.close()
                    self.assertEqual(fake.destroyed, [42])
                    self.assertEqual(fake.terminated, [display])

    def test_failed_make_current_releases_context_and_display(self):
        fake = FakeEGL(fail_current=True)
        with patch.dict(os.environ, {"CYBR_EGL_PLATFORM": "surfaceless"}, clear=True), \
             patch.object(gl_context, "load_egl_library", return_value=(fake, {})):
            with self.assertRaisesRegex(RuntimeError, "make EGL context current"):
                gl_context.EGLContext()
        self.assertEqual(fake.destroyed, [42])
        self.assertEqual(fake.terminated, [100])


if __name__ == "__main__":
    unittest.main(verbosity=2)
