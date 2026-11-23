#!/usr/bin/env bash
# Current selected demonstration. V16-specific artifacts remain unchanged.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/play_style_reference.sh" "$@"
