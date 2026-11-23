#!/usr/bin/env bash
# Selected independently audited physical demonstration, NOT policy inference.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ruiz/miniconda3/envs/env_robocup_sim2sim/bin/python"
MANIFEST="$ROOT/data/motions/robotcupmini/style_reference_manifest.json"
"$PYTHON" "$ROOT/tools/motion/verified_style_data.py" --manifest "$MANIFEST"
MOTION="$("$PYTHON" - "$ROOT" "$MANIFEST" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);m=json.loads(Path(sys.argv[2]).read_text())
if 'style_quality_report' not in m:raise ValueError('No strict straight-walking quality report')
print((root/m['motion']['path']).resolve())
PY
)"
printf '[PLAY] Selected straight-gait demonstration; 1x recorded time, fixed lateral camera.\n'
printf '[NOTE] No AMP training, no policy inference, no certified last-to-first transition.\n'
exec "$PYTHON" "$ROOT/tools/motion/play_retarget_motion.py" --motion "$MOTION" --speed 1.0 --fixed-lateral-camera "$@"
