# Applied V8 checkpoint guard

These are the supplied later checkpoint tools, not a new solver or an automatic cloud backup service. `checkpoint_guard.py` uses immutable generations, file hashes, atomic pointer updates and clean-directory restore. `v8_guarded.py` adds native-grid dependency and semantic checks, plus a requirement for an externally verified upload/download receipt before another physical batch.

The public source omits private storage identifiers, the per-user latest-checkpoint registry, raw logs and numerical state archives. Obtain your own complete checkpoint and independently retained hash before restoring. Do not invent a receipt or disable the guard to make an incomplete job advance.

## Restore your own verified archive

```bash
python checkpoint_guard/restore_saved_job.py \
  --archive /path/to/downloaded-checkpoint.zip \
  --expected-sha256 YOUR_INDEPENDENTLY_RECORDED_SHA256 \
  --remote-file-id YOUR_ACTUAL_REMOTE_OBJECT_IDENTIFIER \
  --into /path/to/empty/recovered
```

The tool checks the archive hash, member paths, CRC, all committed files and native state semantics, and refuses a nonempty restore destination. The restored directories are `recovered/run` and `recovered/store`. Checkpoint provenance may contain the exact matching engine archive. No network requests occur in these Python scripts.

## Audit and guarded advancement

```bash
python checkpoint_guard/v8_guarded.py audit --run /path/to/recovered/run
python checkpoint_guard/v8_guarded.py check-remote --store /path/to/recovered/store

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 LP_NUM_THREADS=4 \
python checkpoint_guard/advance_guarded_parent.py \
  --run /path/to/recovered/run --store /path/to/recovered/store \
  --exports /path/to/exports --engine /path/to/matching/engine --ticks 1
```

The driver requires the last verified generation, exact matching live bytes and a real remote receipt. It preserves pressure and partial parent intervals. After committing, the next batch stays blocked until that exact archive is externally uploaded and downloaded, and the two files match. Use `v8_guarded.py confirm --store ... --original ... --downloaded ... --remote-file-id ...` only after that real external roundtrip. An interrupted export may be retried using `v8_guarded.py export --store ... --exports ...` without rerunning physics.

`advance_guarded_fine.py` advances a full native fine warmup interval. Rendering remains a separate native-band stage. The latest archived experiment reached one of four warmup intervals and zero video frames. The stricter audit rejects old parent restarts that omit pressure; do not synthesize pressure for a nonzero state or substitute the older independent-batch restart format.

The tools improve persistence and dependency validation, not throughput, fire realism, or hardware qualification. Historical report claims have not been rerun during source publication. Run `python checkpoint_guard/test_checkpoint_guard.py` separately for the supplied small guard tests, which are not a native-size recovery proof.
