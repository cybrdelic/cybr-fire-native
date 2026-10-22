# CYBR FIRE — V8 native-quality restoration branch

This is a *source-level restoration of native numerical and output quality defaults* in the recent scenario upgrade, not a completed new production-quality movie. **No new native 4K scene film is included.** Existing released preview videos are below the quality of the original V8 reference.

## Objective regression

| Characteristic | Original CYBR FIRE V8 | Earlier upgrade scenes | Corrected `native-v8` profile |
|---|---:|---:|---:|
| Parent pressure/velocity cells | 160×280×160 (7,168,000) | 64×112×64 (458,752) | 160×280×160 |
| Fine reactive scalar cells | 384×672×384 (99,090,432) | 144×252×144 (5,225,472) | 384×672×384 |
| Native raster | 2160×3840 | 720×1280 | 2160×3840 |
| Detail mode wavelengths (m) | .16, .08, .04 | .20, .12, .08 | .16, .08, .04 |
| Velocity sampling cache scale | 2 | 1 | 2 |
| V8 volume renderer | native ray march | same underlying renderer | same underlying renderer + original V8 optical defaults |

The original has much finer luminous sheets, smoke tendrils, soot variation and transparency than the current upgrade captures. **Reinstating the original grid is necessary but not sufficient**: the original rendered an evolved parent V7 flame state whereas the new scenes begin at cold ambient conditions, and the upgrade includes different source geometry/combustion terms and camera placement. Physically and artistically matching the quality still requires further numerical and optical calibration followed by actual full-grid simulations.

## Implemented

- Native profile is the default for `build_scene_cache.py` and `capture_scene.py`, and native resolution cannot be silently overridden to a preview grid.
- Preview profile is opt-in with `--quality preview`.
- `capture_scene.py` restores original detail wavelengths/flow cache scale and original V8 raymarch sampling defaults for `native-v8`.
- Native fine scalar snapshots are streamed in 8-slab increments (instead of allocating ~1.6GB in `q.read()`), with frame-specific atomic checkpoint publication to avoid mixing gas and surface states from different frames.
- CLI rejects use of preview parent cache for native capture and rejects changes in fine-grid config during resume.
- Added `tests/test_native_quality_guard.py`.

## Verified in this environment

- Native 160×280×160 **jet parent snapshot 0** completed.
- Preview branch compiled and rendered 2 consecutive frames end-to-end with streaming snapshots.
- Numerical `tests/test_detail.py` passed.
- Profile validation test passed.
- Attempting a native capture from a preview cache was rejected before GPU allocation.

**Not yet verified:** a 99,090,432-cell native fine transport step/render after the upgrades, or full-length native 4K scenario output. A complete 24-FPS, 5-second clip requires at least 120 independently evolved output frames plus warmup; CPU-only llvmpipe would require considerable compute and memory. The current environment has a ~4GB memory cgroup, while the original V8 itself needed around 4GB and recommended at least 6GB available. For extended native clips plan for a GPU with adequate memory and persistent job execution.

## Example usage for a native scene

Run parent and fine stages **sequentially**, not concurrently. Repeating commands continues from saved checkpoints.

```bash
# Native profile default; 120 rendered frames at 24 FPS + 24 warmup + 1 velocity sample.
python tools/build_scene_cache.py --quality native-v8 --scene jet \
  --out cache/jet-native --frames 146 --batch 4

# Repeat parent command until manifest.json has complete=true.
python tools/capture_scene.py --quality native-v8 --parent cache/jet-native \
  --out captures/jet-native --warmup 24 --frames 120 --batch 1
# Repeat capture command until completed_verified; if GPU/runtime constraints allow.
```

For local tests on a lower-memory CPU use `--quality preview` **in both stages**; do not label that result as full-quality V8. Pool and fireball share the numerical core but have different ignition behavior; the fireball fades once its short finite source has stopped, so sustained 5s flames require a physically distinct fuel-continuation model, not looped frames.

This branch does not yet implement full solid-fluid thermal coupling, vegetation spread, a compressible explosion shock solver, or high-quality environmental geometry.
