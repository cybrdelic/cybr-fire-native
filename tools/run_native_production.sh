#!/usr/bin/env bash
# Verified original-V8 numerical grids, independent simulation frames, resumable.
# Requires a GL4.5 compute GPU with adequate memory and free disk (>20 GiB/scene).
set -euo pipefail
cd "$(dirname "$0")/.."
scene="${1:?Usage: $0 jet|pool|fireball [output_directory]}"
case "$scene" in
  jet) warmup=24; width=3840; height=2160; exposure=.009; distance=1.25; fov=42; target=.65 ;;
  pool) warmup=24; width=3840; height=2160; exposure=.03; distance=3.25; fov=40; target=.96 ;;
  fireball) warmup=1; width=2160; height=3840; exposure=.03; distance=3.55; fov=40; target=1.30 ;;
  *) echo "Unsupported scenario $scene" >&2; exit 2;;
esac
out="${2:-$PWD/production/$scene}"
frames="${FRAMES:-120}"
parent_count=$((warmup+frames+1))
mkdir -p "$out/parent" "$out/capture"
python3 tests/test_native_quality_guard.py
python3 tests/test_detail.py
# Advance checkpoints in small bounded segments. Parent must finish first.
for iteration in $(seq 1 "$((parent_count+5))"); do
  python3 tools/build_scene_cache.py --scene "$scene" --quality native-v8 --out "$out/parent" --frames "$parent_count" --batch 2
  if python3 - "$out/parent/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));sys.exit(0 if m['complete'] and m['quality_profile']=='native-v8' else 1)
PY
  then break; fi
done
# Capture independent 24-FPS states. Save every frame in lossless PNG, never stretch/loop.
for iteration in $(seq 1 "$((warmup+frames+5))"); do
  python3 tools/capture_scene.py --quality native-v8 --parent "$out/parent" --out "$out/capture" \
    --warmup "$warmup" --frames "$frames" --batch 1 \
    --width "$width" --height "$height" --exposure "$exposure" \
    --distance "$distance" --fov "$fov" --target-height "$target"
  if python3 - "$out/capture/capture.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));sys.exit(0 if m.get('status')=='completed_verified' else 1)
PY
  then break; fi
done
python3 - "$out/capture/capture.json" "$frames" "$width" "$height" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));n,w,h=map(int,sys.argv[2:])
assert m['status']=='completed_verified'
assert m['resolution']==[w,h] and m['quality_profile']=='native-v8'
assert len(m['frames'])==n and len(set(x['sha256'] for x in m['frames']))==n
assert int(m['probe']['nb_read_frames'])==n and m['probe']['width']==w and m['probe']['height']==h
assert [x['index'] for x in m['frames']]==list(range(n))
print('PASS independent native-V8 frames',n, 'native dimensions',w,h)
PY
