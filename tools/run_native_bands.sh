#!/usr/bin/env bash
# Resume original V8 from an immutable fine-state checkpoint, one 4K band per invocation.
set -euo pipefail
cd "$(dirname "$0")/.."
parent="${1:?parent directory}";capture="${2:?capture directory}";count="${3:-1}"
if [[ ! "$count" =~ ^[0-9]+$ ]] || ((count<1));then echo 'Count must be positive' >&2;exit 2;fi
if [[ ! -e "$capture/band_resume.json" ]];then
  python3 tools/native_band_resume.py advance --parent "$parent" --capture "$capture" --band-height 24
fi
for ((i=0;i<count;i++));do
 python3 tools/native_band_resume.py band --parent "$parent" --capture "$capture"
done
python3 - "$capture" <<'PY'
from pathlib import Path
import sys,json
p=Path(sys.argv[1]);r=json.loads((p/'band_resume.json').read_text())
print('BAND_STATUS',len(r['bands']),'of',(2160+r['band_height']-1)//r['band_height'],'frame',r['video_frame'])
PY
