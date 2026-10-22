#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 LP_NUM_THREADS=4
input="${1:?Usage: bash tools/reproduce_v8.sh /path/to/CYBR_FIRE_V7_complete.zip}"
python tests/test_detail.py
python tools/build_parent_cache.py "$input" --out cache/parent
python tests/test_detail_integration.py
python tools/capture_detail.py --parent-cache cache/parent --out captures/v8
