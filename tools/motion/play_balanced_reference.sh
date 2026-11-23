#!/usr/bin/env bash
# 仅回放 v4 参考姿态；不会加载策略、训练网络或驱动真机。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROBOCUP_MUJOCO_PYTHON:-$HOME/miniconda3/envs/env_robocup_sim2sim/bin/python}"
exec env MUJOCO_GL=glfw "$PYTHON" -u "$ROOT/tools/motion/play_retarget_motion.py" \
  --motion "$ROOT/data/motions/robotcupmini/retargeted/walk1_candidate_v4_balanced/candidate_motion.npz" \
  --view oblique --speed 1.0 --loop "$@"
