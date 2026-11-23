#!/usr/bin/env bash
# Execute only when the user intentionally wants to train.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CHECKPOINT="${CHECKPOINT:-$ROOT/artifacts/checkpoints/amp_align_v2/model_150.pt}"
[[ -f "$CHECKPOINT" ]] || { echo 'Restore the checkpoint Release asset first.' >&2; exit 1; }
exec "${PYTHON:-python}" scripts/rsl_rl/train.py \
    --task=Robocup-Style-Prep-v0 --amp --amp_weight=1.0 --resume=True \
    --checkpoint_path="$CHECKPOINT" --device=cuda:0 \
    --num_envs="${NUM_ENVS:-4096}" --max_iterations="${ITERATIONS:-2000}" \
    --save_interval=500 --logger=tensorboard \
    --run_name=amp_align_v2_from150_2000 --headless "$@"
