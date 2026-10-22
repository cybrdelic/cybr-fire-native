# CYBR FIRE V7 — native high-resolution capture

A continuation of the supplied V6 implementation, not replacement footage. This
release adds authoritative full-precision image fields and memory-bounded camera
and checkpoint paths. The capture configuration is **224 × 392 × 224 = 19,668,992
simulated cells**, with staggered velocity on a separate face lattice. Each
published video frame is integrated directly at **2160 × 3840**; the two detail
stills are separately integrated at **3240 × 5760** with two camera samples per
pixel. No image generator, stock footage, render-time density warp, sharpening,
image upscaler, or frame interpolator is used.

The video is a **5× slow-motion detail study**: one 1/120-second physical tick per
frame, played at 24 frames per second. It is not a real-time performance demo.
A finer grid and more camera samples do not establish physical accuracy or
convergence. The gas remains a reduced constant-density combustion model;
transport is bounded but not exactly mass conservative, illumination is
approximate, and hot-char particles have one-way coupling.

## Environment

Python 3.10+, NumPy, SciPy, Pillow, FFmpeg, and Linux with EGL/OpenGL **4.5+**.
The native path uses `glGetTextureSubImage` for slab readback. At least 12 image
binding units and sufficient 3D texture dimensions are required. The driver used
for the delivered capture is recorded in `media/capture.json`: Mesa llvmpipe,
a software OpenGL adapter. No NVIDIA/AMD hardware throughput was measured.
Native Windows/WGL and macOS execution are not implemented. WSL depends on its
actual EGL driver; it is not a tested platform claim.

The tested capture fits the available 4 GiB process-group memory limit by
streaming readback and separating encoding from simulation. This is not a
universal minimum-VRAM guarantee. Storage is dense, not sparse.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m cybrfire device
python -m cybrfire test
python tests/test_native_fields.py
python tests/test_streaming.py
```

## Reproduce the delivered video

The complete package includes `cache/v7-capture-start.npz`. The source-only ZIP
does not include this approximately 456 MiB cache. From the project root:

```bash
LP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
python tools/replay_v7.py \
  --checkpoint cache/v7-capture-start.npz \
  --out captures/v7-4k \
  --width 2160 --height 3840 --frames 48 --fps 24 \
  --ticks-per-frame 1 --samples 1 \
  --still-width 3240 --still-height 5760 --still-samples 2
```

This first computes lossless full-resolution PNG frames and independent stills.
It then closes simulation/render resources and encodes MP4. A codec failure does
not erase the frames. Physical and presentation timestamps, hashes, camera ray
counts, and timings are recorded separately. `progress.json` is progress only;
`capture.json` is written after the complete capture and encoder succeed.

The starting checkpoint was explicitly transferred from the supplied V5 state
at 4.45 s using conservative scalar cell averages and interpolated MAC velocity,
reprojected, then evolved for 24 V7 fine-grid ticks. The published first frame is
at 4.65 s. It is not an equivalent V5 restart or a simulation run from zero at
19.67 million cells. The numerical fields—not the old images—are the input.
The transfer provenance is included in `media/initialization.json`.

To create a *different*, cold-start initial-value history with the same scene:

```bash
python tools/initialize_v7.py --cold-start \
  --scene examples/19-million-4k.json \
  --warmup-ticks 600 --output cache/my-cold-start.npz
```

To explicitly transfer a compatible checkpoint from V5/V6/V7, substitute
`--input /path/to/checkpoint.npz` for `--cold-start`. This utility rejects a
mismatched physical domain or a time that the new clock cannot represent.
It does not silently reinterpret the normal restart loader.

## What changes, and what stays the same

* **Storage:** full-precision native 3D images replace oversized vec4 SSBOs in
  numerical fields. Source, forcing, reaction, transport, projection, particles,
  and optics consume those same authoritative values. This avoids the measured
  128 MiB per-SSBO limit without reducing precision or splitting the simulation
  into independently evolving tiles.
* **Working set:** six persistent images hold scalar/velocity ping-pong state.
  Nonoverlapping stage scratch reuses prediction storage. Camera output is
  integrated in bands with global pixel coordinates. Checkpoints and diagnostics
  read bounded-depth slabs. These are storage/work-scheduling changes, not
  lower-resolution fallbacks.
* **Numerical configuration:** this capture uses a 1/120 s master tick and up to
  16 RK2 characteristic subdivisions with a nominal two-cell local trace budget.
  The previous capture's 1/60 s and four-subdivision configuration is not retained.
  No convergence or arbitrary-Courant accuracy guarantee is implied.
* **Particle ordering:** equal-depth events are ordered by stable particle ID,
  not nondeterministic atomic tile insertion order. The original failing test
  and the repaired passing tests are preserved in the evidence.
* **Presentation:** tighter framing, a portrait 9:16 camera, no rendered floor,
  and no external directional key light are explicit scene settings. The
  thermal material response is not sharpened or replaced. This is not a
  controlled camera-matched V6-to-V7 aesthetic benchmark.

`docs/ARCHITECTURE.md` describes ownership and lifetime constraints.
`docs/VALIDATION.md` separates tests, capture checks, and missing physical/hardware
validation. The measured release evidence—not a planned filename—is authoritative
for completed media and test counts.
