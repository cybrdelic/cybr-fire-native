# FIRE runtime portability

The numerical solver, chemistry and GLSL kernels are shared across backends. Context creation and filesystem operations are the portability layer. No historical video substitutes for a simulation.

## Choose an execution path

| Host | Path | Requirement and scope |
| --- | --- | --- |
| Windows | Native hidden GLFW OpenGL context | OpenGL 4.5 driver and a desktop session; native numerical fields additionally require at least 12 image units; headless services may not expose the driver. |
| Linux | Native EGL, then hidden GLFW fallback | OpenGL 4.5 vendor driver or Mesa; GLFW additionally needs a working display. |
| macOS | Native Mesa EGL / llvmpipe | Existing compatible Mesa EGL installation; implemented discovery and verifier, pending actual Mac execution. CPU software execution of the same GLSL, not Metal. See [the native setup](docs/NATIVE_MACOS_MESA.md). |
| macOS, Windows, Linux | Linux Mesa container | Existing Docker with Linux containers; a separate software-compute path. Linux container execution does not prove native macOS support. |
| Any of these hosts | CPU contract/checkpoint tools | Python dependencies; these checks do not establish simulation correctness. |

`CYBR_GL_BACKEND=auto` selects EGL first on Linux/macOS and GLFW first on Windows. Set it explicitly to `egl` or `glfw` to require a particular backend. `CYBR_EGL_LIBRARY` selects an existing absolute library file; macOS also discovers Homebrew prefixes and uses bounded, read-only package-location queries. `CYBR_EGL_PLATFORM=surfaceless` selects headless Mesa; `device` selects enumerated EGL devices. `CYBR_EGL_DEVICE` retains explicit device selection and conflicts with explicit surfaceless selection. Explicit EGL settings are never silently replaced by GLFW. No loader installs packages or changes global environment/configuration. Backend errors are reported rather than hidden. GLFW context creation requires the main thread. `doctor` separately reports `max_image_units` and `native_field_bindings_supported`. The Intel test adapter exposes 8 image units: small SSBO solver and sampled-volume paths can run, while native numerical fields reserve slots 8 through 11 and must reject that hardware. Allocating a texture alone no longer unnecessarily binds slot 8.

## Fresh native setup

Use Python 3.11 or newer. Create an isolated environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a POSIX shell, then run:

```sh
python -m pip install --no-compile -r requirements.txt
python -m cybrfire.doctor --require-compute
python checkpoint_guard/test_checkpoint_guard.py
python tests/test_portable.py
python tests/test_egl_backend.py
python tests/test_portable_compute.py
python tests/test_runtime.py
python tests/test_detail.py
```

`doctor` reports actual runtime and driver availability; creating a context alone is not a physics test. `test_portable_compute` compiles/dispatches/reads real GLSL and reopens a context. `test_runtime` steps a 16 x 24 x 16 coupled solver and checks its ground-light integral; `test_detail` advects a 48 x 64 x 48 scalar field. These bounded tests do not qualify native-size production or hardware fire safety.

FFmpeg/ffprobe are separate prerequisites for movie encoding and validation. Bash is needed only for historical `.sh` wrappers; Python batch entry points work without Bash. No compiler is needed for the ctypes runtime. Dependency wheels and graphics drivers remain platform prerequisites.

## Portable container execution

Install and start Docker separately if it is not already available. The launcher does not install software, start a daemon, alter global configuration or grant GPU access. Docker must be using Linux containers. Initial image building downloads the base image, Debian Mesa/FFmpeg packages and Python dependencies; later execution disables network access.

These commands have the same syntax in PowerShell, Linux and macOS shells:

```sh
python tools/run_portable.py --build
python tools/run_portable.py -- tests/test_portable_compute.py
python tools/run_portable.py -- tests/test_runtime.py
python tools/run_portable.py -- tests/test_detail.py
```

The image copies this checkout at build time. Rebuild after source edits. It selects `CYBR_GL_BACKEND=egl` and `LIBGL_ALWAYS_SOFTWARE=1`, running the same FP32 kernels and explicit scene settings under Mesa. Existing solver behavior still chooses spectral pressure for the generic software-adapter capture CLI unless `--pressure` is supplied; no new precision or numerical shortcut is introduced by the container. Native independent batch defaults are unchanged.

Files inside a `--rm` container are temporary. To persist a generated result, mount only its destination and use `/output` in the Python command:

```sh
python tools/run_portable.py --output ./portable-output -- tools/portable_smoke.py --out /output
```

Full fine fields require gigabytes of memory and software rendering can be very slow. Allocate resources deliberately; a passing smoke test does not imply the unfinished 4K films are complete. Docker Desktop on macOS/Windows is a prerequisite, not tested automatically by Linux CI.

## Checkpoint durability

Checkpoint copies now flush writable handles, avoiding Windows' read-only `fsync` error. POSIX retains file and directory `fsync`; Windows supports file `fsync` plus atomic replacement but Python does not provide equivalent directory `fsync`. The manifest records `file-fsync-atomic-replace-no-directory-fsync` on Windows, so it does not claim the same power-loss metadata durability. Hash validation, immutable generations and the remote-receipt gate remain intact. Advisory batch locks use POSIX `flock` or Windows byte-range locking and release when the owned file closes.

## Verification boundaries

Windows 11 / Python 3.12.10 / Intel UHD OpenGL 4.5 driver 31.0.101.4146: checkpoint regressions (2), portable CPU contracts (including native image-unit rejection), native-quality guards, camera contracts, and real compute dispatch/readback passed locally. Actual solver/transport/render results are recorded separately in the audit evidence; a full native production film has not been produced by these checks.

The workflow runs CPU contracts on Windows, Ubuntu and macOS with Python 3.11/3.12, and actual small solver/transport tests inside Linux Mesa. A separate native macOS job installs Mesa only on its hosted runner, verifies real compute/transport, and compares native-image and SSBO simulation/render output with identical explicit multigrid pressure settings. Workflow configuration is not execution evidence: native macOS remains unverified until that job actually passes. Neither a Linux container run nor macOS CPU-only CI establishes native macOS execution or an actual macOS Docker Desktop run.

The original mature burner still requires the absent historical `v7-start.npz`. New independent scenes can start from their initial conditions; no replacement starting state has been fabricated.
