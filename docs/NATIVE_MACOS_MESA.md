# Native macOS software execution

This path loads native Mesa EGL and executes FIRE's existing GLSL through llvmpipe on the CPU. It does not translate the solver to a new numerical model, play historical footage, or use Metal. Implementation and a real execution verifier are present; **execution on macOS has not yet been verified**. Linux container success and macOS CPU contract tests are separate evidence.

## Requirements and setup

Use an existing Mesa installation with EGL, surfaceless support, llvmpipe, OpenGL 4.5 and at least 12 image units. The Python process and dylib architecture must match. Current [Mesa macOS documentation](https://docs.mesa3d.org/macos.html) supports software Gallium drivers. The [Homebrew recipe](https://github.com/Homebrew/homebrew-core/blob/master/Formula/m/mesa.rb) builds llvmpipe and zink; its current macOS bottles are Apple Silicon. Intel Macs may need an independently built compatible Mesa installation. No package is bundled, installed or configured by FIRE's loader.

Create an isolated Python environment and install `requirements.txt`, then select the installed native library in the current shell:

```sh
export CYBR_EGL_LIBRARY="$(brew --prefix mesa)/lib/libEGL.dylib"
export CYBR_GL_BACKEND=egl
export CYBR_EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export LP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
python -m cybrfire.doctor --require-compute
python tests/test_portable_compute.py
python tests/test_runtime.py
python tests/test_detail.py
python tools/verify_native_mesa.py --require-platform darwin --pressure multigrid
```

For a custom installation, replace `CYBR_EGL_LIBRARY` with its absolute existing dylib path. A conflicting `MESA_LOADER_DRIVER_OVERRIDE=zink` should be removed from this shell's launch configuration deliberately; FIRE does not silently rewrite it. Explicit EGL library/platform selection does not fall back to Apple's OpenGL. Missing dependencies, incompatible architecture, insufficient GL capabilities or absent EGL are failures, not successful simulation checks.

Library discovery also checks the system loader, common Homebrew prefixes, `pkg-config --variable=libdir egl`, and `brew --prefix mesa`. Package queries are read-only with five-second timeouts. macOS prefers surfaceless EGL, while explicit EGL device targeting remains supported elsewhere. OSMesa is not used because current Mesa removed that API.

## What the verifier establishes when it passes

The native verifier requires an actual llvmpipe EGL surfaceless context. It executes four integration ticks for both SSBO and RGBA32F native-image storage on a 16 x 24 x 16 grid, with identical explicitly selected multigrid pressure, emitter, chemistry, transport and rendering parameters. It checks finite evolved fields, exact storage-path parity, actual GL 4.5 slab readback, checkpoint restore and a nonconstant 64 x 96 HDR render. `--pressure spectral` is a separate explicit option; it is never selected implicitly because the adapter is software.

The generated JSON records the actual host, adapter, scene settings and shader hashes; the PNG comes from that execution. These tests do not establish experimental calibration, convergence, performance at production size, a finished 4K film, GPU acceleration, or universal Mac compatibility. The hosted macOS workflow is prepared to run them, but an unexecuted workflow is not proof.

## Implementation evidence

The narrow portability layer adds dylib discovery and surfaceless selection; numerical kernels, physics configuration classes, storage equations and renderer equations are unchanged. Mesa's [EGL build](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.5/src/egl/meson.build) includes its [surfaceless software implementation](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.5/src/egl/drivers/dri2/platform_surfaceless.c). These primary sources support feasibility, not a claim that this repository has already run on a Mac.

Local Windows regression checks passed after this change: nine CPU EGL discovery/selection/lifetime contracts, five existing portability contracts, two real GLSL dispatch/readback checks with context recreation, and the existing four-tick 16 x 24 x 16 solver plus finite 64 x 96 render. The latter explicitly uses spectral pressure, as before; it is not the new multigrid/native-image verifier. The Intel test adapter has eight image units, so it cannot validate the twelve-unit native-image route. The new verifier also correctly rejected a Windows run requesting native Darwin before context creation. None of these results establishes macOS execution.
