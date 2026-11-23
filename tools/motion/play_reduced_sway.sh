#!/usr/bin/env bash
# Actual recorded physical states, not policy inference or a new physics test.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ruiz/miniconda3/envs/env_robocup_sim2sim/bin/python"
MANIFEST="$ROOT/data/motions/robotcupmini/validated/sway_correction_20261119/selected_reduced_sway_motion.json"
"$PYTHON" "$ROOT/tools/motion/verified_style_data.py" --manifest "$MANIFEST"
"$PYTHON" - "$MANIFEST" "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]).resolve()
entry=manifest['lateral_improvement_report'];path=(root/entry['path']).resolve()
if not path.is_relative_to(root) or 'lilin' in str(path).lower(): raise ValueError('Outside user project')
if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']: raise ValueError('Lateral report changed')
r=json.loads(path.read_text())
if not all(r['checks'].values()) or r['candidate']['sha256']!=manifest['motion']['sha256']:
    raise ValueError('Lateral improvement not verified for this motion')
print('[CHECK] Physics + lateral improvement reports verified. Playback is recorded states, NOT policy inference.')
print('[PLAY] Default 1x speed; source clip is already slow walking. No automatic training or looping.')
PY
exec "$PYTHON" "$ROOT/tools/motion/play_retarget_motion.py" \
 --motion "$ROOT/data/motions/robotcupmini/retargeted/walk1_v14_reduced_sway_accepted/candidate_motion.npz" \
 --speed 1.0 "$@"
