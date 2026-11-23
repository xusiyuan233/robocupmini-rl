#!/usr/bin/env bash
# Install into an explicitly activated Python 3.11 environment. Never starts training.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/robocup/IsaacLab}"
case "${ROOT,,}|${ISAACLAB_DIR,,}" in *lilin*) echo 'Refusing another user project.' >&2; exit 1;; esac
"$PYTHON" -c 'import sys,os; assert sys.version_info[:2] == (3,11), "Activate a Python 3.11 environment first"; assert sys.prefix != sys.base_prefix or os.environ.get("CONDA_PREFIX"), "Use an isolated environment"'
COMMIT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["isaaclab"]["commit"])' "$ROOT/reproducibility/snapshot.json")"
PATCH="$ROOT/reproducibility/patches/isaaclab-local.patch"
if [[ -e "$ISAACLAB_DIR" && ! -d "$ISAACLAB_DIR/.git" ]]; then
    echo "Refusing to modify an existing non-Git directory: $ISAACLAB_DIR" >&2; exit 1
fi
if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
    mkdir -p "$(dirname "$ISAACLAB_DIR")"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
    git -C "$ISAACLAB_DIR" checkout --detach "$COMMIT"
fi
if [[ "$(git -C "$ISAACLAB_DIR" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo 'Existing IsaacLab is at a different commit. Choose an empty ISAACLAB_DIR; no checkout performed.' >&2; exit 1
fi
if git -C "$ISAACLAB_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo 'Recorded IsaacLab patch is already present.'
elif git -C "$ISAACLAB_DIR" apply --check "$PATCH"; then
    git -C "$ISAACLAB_DIR" apply "$PATCH"
else
    echo 'IsaacLab patch cannot be safely applied; stopping without reset/overwrite.' >&2; exit 1
fi
# Do not upgrade packaging or torch to arbitrary latest versions.
"$PYTHON" -m pip install --timeout 120 --retries 8 \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "$ROOT/reproducibility/requirements-runtime.lock"
for package in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    "$PYTHON" -m pip install --no-deps --no-build-isolation -e "$ISAACLAB_DIR/source/$package"
done
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$ROOT/rsl_rl"
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$ROOT/exts/bipedal_locomotion"
"$PYTHON" -m pip check
cd "$ROOT"
"$PYTHON" -m tools.repro.verify_snapshot
"$PYTHON" -m tools.amp.motion_loader
printf '\nInstallation checks complete. No policy training was started.\n'
printf 'First simulator launch may ask you to review NVIDIA license terms and populate shader caches.\n'
