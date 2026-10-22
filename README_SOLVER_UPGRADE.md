# CYBR FIRE — solver-level scenario branch

This branch is derived directly from the recovered CYBR FIRE V8 source, rather
than from stills, retimed footage, or synthesized images. It contains actual
changes to gas/surface evolution and explicit independent scene initial states.

## Changes

- `SurfaceFuel` is a first-class, serialized scene component. `surface_fuel.comp`
  integrates finite mass, surface temperature, pyrolyzed/char mass, neighbor
  thermal diffusion, and local gas heat coupling on a separate 2-D grid.
- `surface_inject.comp` converts consumed surface fuel (kg/m²) into gas fuel
  density (kg/m³) in two adjacent cells. The finite-volume mass ledger was
  validated with absolute error < 1e-8 kg in the isolated GPU test.
- `air_mixing.comp` is a parameterized unresolved turbulent entrainment closure
  that mixes ambient air into the near-ground volume. It limits otherwise
  unbounded pool-fuel buildup and supplies oxidizer. This is an open-system VFX
  approximation, **not** a detailed atmospheric combustion model.
- `Emitter.expansion_speed` grows a transient source radius at a specified
  physical rate (m/s), while `Emitter.radial_speed` adds radial impulse to the
  inlet velocity. The fluid solver is low-Mach; this is **not a shock-wave model**.
- Parent and fine checkpoint state now includes the surface fuel field; restart
  does not reset the solid-fuel reservoir.
- Parent scenes start independently at t=0 with ambient gas and no inherited
  mature V7 plume. The V8 GPU projection, transport, chemistry and radiance
  operators remain the primary pipeline.

## Current verified capture sizes

- Parent velocity/pressure grid: **64 × 112 × 64** (458,752 cells).
- Fine scalar transport: **144 × 252 × 144** (5,225,472 cells).
- Render: **720 × 1280**, 2 samples/pixel, 24 FPS.
- Three independent clips, each **1 second, 24 distinct physical frames**.
- `jet` has sustained transverse momentum and a continuous fuel inlet.
- `pool` has **zero gas-emitter input** and is fed by finite surface pyrolysis.
- `fireball` uses finite-duration expanding injection with radial momentum, then
  extinguishes. Its fading is intended behavior, not a persistent burner.

The current results are numerical/VFX proof clips, **not the user's requested
native 4K, 5–10-second, world-class completed showcase**. There is no terrain,
vegetation, geometry-coupled heat transfer, ember-driven spread, compressible
explosion shock or photoreal environmental integration in this branch.

## Running

```sh
python tools/build_scene_cache.py --scene pool --out cache/pool --frames 29 --batch 4
# Repeat the same command until `cache/pool/manifest.json` has complete:true.
python tools/capture_scene.py --parent cache/pool --out captures/pool \
  --frames 24 --warmup 4 --batch 2 --shape 144 252 144 \
  --width 720 --height 1280 --samples 2
# Repeat until capture status is `completed_verified`.
```

Use `jet`, `pool`, or `fireball` for the scene argument. See
`tools/real_scenarios.py` for independent scene parameters. For improved
fireball composition, pass `--exposure .13 --fov 55 --distance 4.25
--target-height 1.6` to every capture invocation. For pool, pass
`--exposure .08 --fov 48 --distance 3.7 --target-height 1.0`.

## Validation

`python tools/validate_upgrade.py`

Includes GPU-level mass conservation, source-frame SHA-256 uniqueness, 24/24
frame decode and simulation-time verification, and downsampled frame-change
checks to distinguish real motion from sampling-only differences. The validation
report is delivered separately.

Existing V8 detail and native field tests pass. The original
`tests/test_detail_integration.py` requires its own `cache/parent` and cannot
run without the prebuilt integration fixture; this upgrade was independently
rendered and audited with three scene-specific parent caches.
