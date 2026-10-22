# CYBR FIRE V8 — resumable independent batches

This is a **production job system, not a finished 5-second movie**. It advances the original V8 numerical resolution in separate resumable batches, allowing a machine with enough time and memory to build up the film incrementally. Neither image generation nor video stretching is used.

## Fixed fidelity invariants

- Native parent fluid/pressure grid: **160 × 280 × 160** (7,168,000 cells).
- Native fine reacting scalar grid: **384 × 672 × 384** (99,090,432 cells). Fine pressure **is not** solved on this grid; it is driven by the parent MAC flow and fine transverse modes.
- Native image size: **2160 × 3840** for fireball; **3840 × 2160** for jet and pool; two pixel samples per frame.
- Temporal integration: **120 Hz** parent steps and two fine steps per 24-FPS output frame.
- Native output target: 120 new frames per scene = 5 seconds @ 24 FPS.

## One independent batch at a time

In the extracted directory, call commands below in a shell (Python 3.11+, FFmpeg, GL 4.5 EGL backend, NumPy, SciPy, PIL); the batch script uses an exclusive filesystem lock:

```bash
# Repeat: one more full-grid parent velocity snapshot in fireball.
python3 tools/native_batch.py --scene fireball --out production/fireball --stage parent --frames 120 --batch 1

# Repeat: advance at most one fine-grid simulation frame and render it if past warmup.
# This can run while the parent manifest is incomplete, provided it has the
# two velocity states needed for the next fine integration step.
python3 tools/native_batch.py --scene fireball --out production/fireball --stage capture --frames 120 --batch 1

# Once >=2 fresh rendered frames exist, encode their ACTUAL temporal span;
# the filename explicitly says INCOMPLETE until 120 unique frames are verified.
python3 tools/native_batch.py --scene fireball --out production/fireball --stage encode-partial --frames 120

# Inspect progress without performing a simulation step.
python3 tools/native_batch.py --scene fireball --out production/fireball --stage status --frames 120
```

For `jet` and `pool`, substitute the `--scene` and output directory. Jet and pool require 24 physical warmup steps; fireball requires 1. The 120-frame target needs **145 parent velocity snapshots** for jet/pool or **122** for fireball. One-way parent/fine coupling is unchanged from V8. The two stages must **not run concurrently on the same 4 GiB machine**.

## Resume guarantees and limitations

- Parent `restart.npz` stores gas, velocity, simulation clock and (if present) surface-fuel state. Individual velocity snapshots (`velocity-NNN.npy`) are durable.
- Fine `fine-state-NNNNN.npy` is written using bounded-slab GPU readback and atomic checkpoint publication. A new process uploads the last checkpoint; previously rendered frames are not recreated unless the previous invocation failed before the state commit.
- Fine `capture.json` is tied to the parent's immutable physics configuration, not the changing snapshot-count manifest. The requested frame count, warmup, camera, full fine-grid profile and resolution cannot change mid-capture.
- `batch_receipt.json` reports actual parent snapshots, fine steps, genuinely rendered PNGs and native duration. It verifies increasing time and no identical frame hashes.
- Intermediate diagnostic videos are named `INCOMPLETE`. Their duration is exactly the number of independently generated PNGs divided by 24, with no holds, interpolation, or loops.
- A completed scene video is encoded only when **all 120 independently simulated and rendered frames** exist. Completion also verifies full MP4 decoding, frame count, frame hashes and sustained-flame tests where applicable.

**Visual fidelity remains a separate acceptance test.** A technically genuine native-V8 image can still contain a clipped, smooth or immature fireball. Numerical size and uniqueness do not by themselves establish matching the original V8's mature turbulent appearance. The actual production frames must be reviewed before calling a scene showcase-quality. The software OpenGL runtime here is extraordinarily slow; running ~120 frames per scene can take many hours. For practical use run these batches on a computer with a strong GPU and adequate VRAM. Do not describe a partial diagnostic as a finished 5-second film.
