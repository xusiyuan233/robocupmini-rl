#!/usr/bin/env bash
# User's own project only. Recorded accepted physics playback, not policy inference.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ruiz/miniconda3/envs/env_robocup_sim2sim/bin/python"
MANIFEST="$ROOT/data/motions/robotcupmini/validated/dynamics_acceptance_20261119/selected_style_motion.json"
"$PYTHON" "$ROOT/tools/motion/verified_style_data.py" --manifest "$MANIFEST"
exec "$PYTHON" "$ROOT/tools/motion/play_retarget_motion.py" \
 --motion "$ROOT/data/motions/robotcupmini/retargeted/walk1_v13_physics_accepted/candidate_motion.npz" "$@"
