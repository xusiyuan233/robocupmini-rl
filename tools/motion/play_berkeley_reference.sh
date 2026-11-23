#!/usr/bin/env bash
# Review recorded Berkeley adaptation; this does NOT run training or a policy.
# Prefer FFmpeg playback: desktop Totem can start successfully without H.264.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
INDEX="$ROOT/data/motions/robotcupmini/validated/berkeley_20261120/adaptation_result.json"
CHECK_ONLY=0
case "${1:-}" in
  --check) CHECK_ONLY=1 ;;
  -h|--help)
    printf '用法：bash tools/motion/play_berkeley_reference.sh [--check]\n'
    printf '默认：用本机播放器打开 Berkeley 对照视频。空格暂停，Esc/q 退出。\n'
    printf -- '--check：仅检查文件、解码和播放器，不打开窗口。\n'
    exit 0 ;;
  '') ;;
  *) printf '未知参数：%s；使用 --help 查看用法。\n' "$1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf '参数过多；使用 --help 查看用法。\n' >&2
  exit 2
fi
if [[ -x /usr/bin/python3 ]]; then
  PYTHON=/usr/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  printf '[错误] 找不到 Python 3。视频索引读取仅需标准库，无需激活 Conda 环境。\n' >&2
  exit 1
fi
if [[ ! -f "$INDEX" ]]; then
  printf '[错误] 找不到 Berkeley 回放索引：%s\n' "$INDEX" >&2
  exit 1
fi
VIDEO="$("$PYTHON" - "$ROOT" "$INDEX" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
result = json.loads(Path(sys.argv[2]).read_text())
print('[状态] ' + result['status'], file=sys.stderr)
print('[说明] 这是记录运动的对照回放，不是训练或策略推理。', file=sys.stderr)
if not result['ready_for_style_dataset_selection']:
    print('[提醒] 候选示范尚未通过全部风格检查，未选为正式训练示范。', file=sys.stderr)
path = (root / result['comparison_video']).resolve()
if not path.is_relative_to(root) or 'lilin' in str(path).lower() or not path.is_file():
    raise ValueError('视频不存在或超出本用户项目范围')
if path.stat().st_size == 0:
    raise ValueError('视频文件为空')
print(path)
PY
)"
# Do not rely on xdg-open's zero exit status as proof of successful playback.
# It only proves that the desktop accepted an open request, not that it decoded.
PLAYER=''
for candidate in ffplay mpv vlc; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PLAYER="$candidate"
    break
  fi
done
if [[ -z "$PLAYER" ]]; then
  printf '[错误] 未发现 ffplay、mpv 或 VLC；不调用已知缺少解码器的默认播放器。\n' >&2
  printf '视频位置：%s\n' "$VIDEO" >&2
  exit 1
fi
printf '[视频] %s\n[播放器] %s\n' "$VIDEO" "$(command -v "$PLAYER")"
if (( CHECK_ONLY )); then
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -v error -threads 2 -i "$VIDEO" -map 0:v:0 -an -f null -
    printf '[通过] 整段视频解码成功；文件与播放器检查通过。未打开窗口。\n'
  else
    printf '[通过] 文件与播放器存在；未安装 ffmpeg，未执行整段解码检查。\n'
  fi
  exit 0
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  printf '[错误] 当前终端没有图形显示会话；请在本机桌面终端运行。\n' >&2
  exit 1
fi
printf '[操作] 空格暂停/继续；Esc 或 q 退出。视频只播放一次，不自动循环。\n'
case "$PLAYER" in
  ffplay)
    # Use software decoding; avoid consuming the shared CUDA training device.
    exec ffplay -hide_banner -loglevel warning -threads 2 -an -autoexit \
      -window_title 'Berkeley -> ROBOCUPMINI | recorded review (not training)' \
      -x 1200 -y 742 "$VIDEO" ;;
  mpv) exec mpv --hwdec=no --loop-file=no "$VIDEO" ;;
  vlc) exec vlc --no-loop --no-repeat "$VIDEO" ;;
esac
