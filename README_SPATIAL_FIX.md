# CYBR FIRE V8 — reduced-grid spatial consistency repair

## Failure mechanism

The prior reduced parent path changed grid shape from `(160, 280, 160)` to `(64, 112, 64)` without increasing cell spacing from `0.012 m` to `0.030 m`. That accidentally reduced the physical domain from `1.92 × 3.36 × 1.92 m` to `0.768 × 1.344 × 0.768 m`. The unchanged burner center `(0.96, 0.12, 0.96) m` consequently lay outside the domain, so fuel injection stopped and a residual plume cooled into smoke.

The corrected import holds physical extent constant, conservatively transfers the four scalar fields to the new cell grid, interpolates staggered MAC velocities to the new faces and advances the same V7/V8 pressure-projected reacting-flow parent solver. The fine V8 scalar stage is then initialized on a `192 × 336 × 192` grid, with source positions still in physical world coordinates.

## Reproduction

Obtain the original `v7-start.npz` from `CYBR_FIRE_V8_project.zip` at `CYBR_FIRE_V8/input/v7-start.npz`.

```bash
python tools/rebuild_fixed_parent.py \
  --start /path/to/v7-start.npz \
  --out work/parent --frames 41 --batch 8
# Repeat until work/parent/manifest.json reports complete:true.

python tools/capture_fixed_fire.py \
  --parent work/parent --out work/capture \
  --frames 24 --warmup 8 --batch 1 \
  --width 540 --height 960 --samples 2
# Repeat until capture.json reports complete_verified.
```

## Quality and numerical boundaries

- The produced video is a **1-second, 24-FPS, 540×960 native ray-integrated** reduced-detail capture with 24 new simulated states. It is **not** full original V8 2160×3840 / 99-million-scalar-cell quality.
- Parent pressure/velocity grid: 64×112×64 = 458,752 cells; fine scalar grid: 192×336×192 = 12,386,304 cells.
- Pressure projection is performed on the parent grid, not the fine scalar grid. Subgrid transverse velocity modes drive one-way fine detail.
- The import at different spatial and temporal discretization is **not a mathematically equivalent restart** of the 7M-cell full-resolution fluid solver.
- The corrected render uses anti-aliasing through two samples per pixel and frame-varying sampling; no image generation, held video frames or looping.
- Validation includes emitter-inside-domain checks, nonzero fuel/heat at the emitter, 24 distinct rendered PNG hashes, nonzero bright-flame pixel coverage across every frame, full H.264 decoding and exact 24/24 frame-count verification.
- Persistent fire over durations longer than one second remains to be validated independently. More physics snapshots and independently rendered frames are required for longer clips.
