# CYBR FIRE V8 — Native-quality execution and honest limitations

This branch keeps the original 160×280×160 fluid solve (7,168,000 cells),
384×672×384 reactive fine scalar grid (99,090,432 cells), and true native
4K raymarch (2160×3840 or 3840×2160). Fine transport is *not* an independent
99-million-cell pressure solver. No image generation, image warping,
frame duplication, or resolution upscaling is used.

## What was executed here

- Unit tests for the native numerical profile, subgrid detail transport, and
  off-center camera / full-raster landscape quality constraints.
- A full-resolution **jet parent solve**, producing 9 distinct 160×280×160
  parent velocity states through t=0.333 seconds, from independent cold initial
  conditions. Maximum recorded source-region combustion temperature during
  the solve reached ~3015 K in an intermediate diagnostics pass.
- A full-resolution fine transport stage, including 4 warmup steps and **two
  independently evolved native 2160×3840 rendered frames**. Their original
  portrait camera and 1.4-scale optics grossly overexposed and cropped the jet.
- An optical-only camera/exposure calibration using the **same native 99M-cell
  fine state**, without changing or resimulating it. The successful calibrated
  camera was then raymarched at an actual 3840×2160 raster, with 2 samples per
  pixel and a measured 85-second render time on software OpenGL.

## Why this is not yet the requested showcase

- The fire is still young: only around 0.25 seconds after cold initialization.
  It has not developed the original V8's mature turbulent plume, filamentary
  flame sheets, stratified soot and long-lived recirculating structure.
- The calibration repairs white-clipped imagery and horizontal-jet composition
  but does **not** fix the short simulation age or guarantee photorealism.
- The checked output is **one true native 4K landscape still**, not a 5-second
  video and not a comparison proving equivalence to the original mature V8.
- The available environment uses Mesa llvmpipe on CPU under a 4 GiB memory
  cgroup. This is barely enough for native fields and much too slow for three
  5–10-second full-quality scenarios within a practical interactive session.

## Production execution

`tools/run_native_production.sh jet /path/to/output/jet`

Analogously `pool` and `fireball`. Requires Linux OpenGL 4.5 GPU/EGL compute,
adequate device memory, tens of GiB of temporary disk **per scene**, and
long-running persistent compute. The script executes the native profile only,
with physical pre-roll for sustained sources, 120 distinct temporal frames,
lossless frame dumps, restartable parent/fine passes, and frame-count/hash
verification. It never fabricates longer scenes by repeating video frames.

Importantly, these commands are *prepared and unit-tested*, not claimed to have
finished all three production jobs on the current software adapter. The next
quality gate is evaluating a physically aged simulation (>1 second for sustained
jet/pool), tuning source combustion/soot/radiation without scene-specific
image-space cheats, then capturing and grading full-quality motion.
