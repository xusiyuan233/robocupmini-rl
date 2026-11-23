#!/usr/bin/env bash
# Playback only. play.py also exports policy and encoder next to the checkpoint.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CHECKPOINT="${CHECKPOINT:-$ROOT/artifacts/checkpoints/amp_align_v2/model_150.pt}"
[[ -f "$CHECKPOINT" ]] || { echo 'Restore the checkpoint Release asset first.' >&2; exit 1; }
exec "${PYTHON:-python}" scripts/rsl_rl/play.py \
    --task=Robocup-Style-Prep-Play-v0 --device=cuda:0 --num_envs=1 \
    --checkpoint_path="$CHECKPOINT" "$@"
