#!/usr/bin/env bash
# v5 measured-human retarget playback ONLY. No training or real-robot control.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ruiz/miniconda3/envs/env_robocup_sim2sim/bin/python"
MOTION="$ROOT/data/motions/robotcupmini/retargeted/walk1_candidate_v5_human/candidate_motion.npz"
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$MOTION" ]] || { echo "Missing reference: $MOTION" >&2; exit 1; }
# No --loop: the BVH clip is not periodic and must not hide a boundary jump.
export MUJOCO_GL=glfw
exec "$PYTHON" "$ROOT/tools/motion/play_retarget_motion.py" --motion "$MOTION" --speed 1.0 --view oblique "$@"
