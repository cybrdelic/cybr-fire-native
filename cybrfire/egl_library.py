"""Discover an existing EGL implementation without installing or configuring it."""
import ctypes as C
import ctypes.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _command_output(arguments):
    """Only bounded, read-only package-location queries are permitted here."""
    try:
        result = subprocess.run(arguments, capture_output=True, text=True,
                                timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value and "\n" not in value else None


def library_candidates():
    override = os.environ.get("CYBR_EGL_LIBRARY")
    if override is not None:
        path = Path(override).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError("CYBR_EGL_LIBRARY must name an existing absolute library file; no fallback was attempted.")
        yield str(path), "explicit"
        return

    system = ctypes.util.find_library("EGL")
    if system:
        yield system, "system-loader"
    if sys.platform.startswith("linux"):
        # Runtime ld.so can resolve the ABI soname even when find_library's
        # external discovery helpers are absent in a minimal installation.
        yield "libEGL.so.1", "linux-soname"
    if sys.platform != "darwin":
        return

    # Homebrew libraries may be keg-only or absent from dyld's search path.
    # Inspect known prefixes first; commands never install or update packages.
    directories = [Path("/opt/homebrew/opt/mesa/lib"), Path("/usr/local/opt/mesa/lib")]
    for directory in directories:
        for filename in ("libEGL.1.dylib", "libEGL.dylib"):
            path = directory / filename
            if path.is_file():
                yield str(path), "homebrew-prefix"
    for command, argument, origin in (("pkg-config", "--variable=libdir", "pkg-config"),
                                      ("brew", "--prefix", "homebrew")):
        executable = shutil.which(command)
        if executable:
            value = _command_output([executable, argument, "egl" if command == "pkg-config" else "mesa"])
            if value and Path(value).is_absolute():
                directory = Path(value) if command == "pkg-config" else Path(value) / "lib"
                for filename in ("libEGL.1.dylib", "libEGL.dylib"):
                    path = directory / filename
                    if path.is_file():
                        yield str(path), origin


def load_egl_library():
    seen = set()
    failures = []
    for name, origin in library_candidates():
        if name in seen:
            continue
        seen.add(name)
        try:
            library = C.CDLL(name)
            for symbol in ("eglGetProcAddress", "eglInitialize", "eglTerminate"):
                getattr(library, symbol)
        except (OSError, AttributeError) as exc:
            failures.append(f"{origin}: {type(exc).__name__}")
            if origin == "explicit":
                raise RuntimeError("Cannot load CYBR_EGL_LIBRARY or its dependencies; check architecture and install linkage. No fallback was attempted.") from exc
            continue
        # Do not leak a private installation prefix into shareable diagnostics.
        return library, {"egl_library": Path(name).name, "egl_library_source": origin}
    detail = " (" + ", ".join(failures) + ")" if failures else ""
    raise RuntimeError("No loadable EGL library found" + detail +
                       ". Supply CYBR_EGL_LIBRARY for an existing Mesa/vendor library; on macOS use a native Mesa build with EGL and llvmpipe.")
