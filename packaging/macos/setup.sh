#!/bin/bash
set -euo pipefail
PAYLOAD="$1"
DATA_ROOT="$2"
UPDATER="$3"
fail() { echo "[SoyaComfy] $*" >&2; exit 1; }

command -v git >/dev/null || fail "Install Git first: xcode-select --install"
command -v ffmpeg >/dev/null || fail "Install FFmpeg first: brew install ffmpeg"
command -v ffprobe >/dev/null || fail "Install FFmpeg first: brew install ffmpeg"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"
UV_VERSION="0.11.8"
UV_TOOL_DIR="$PWD/.tools/uv-$UV_VERSION"
UV_EXE="$UV_TOOL_DIR/uv"
if [ ! -x "$UV_EXE" ]; then
    (export UV_INSTALL_DIR="$UV_TOOL_DIR" UV_NO_MODIFY_PATH=1
     curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh) \
        || fail "Could not install uv $UV_VERSION"
fi
# No virtual environment is created before the payload is installed.
"$UV_EXE" run --no-project --no-config --python 3.12 python "$UPDATER" "$PAYLOAD" "$DATA_ROOT" \
    || fail "App update failed; inspect the error above and launch again to retry."
chmod +x "$DATA_ROOT/run_en.sh"
exec /bin/bash "$DATA_ROOT/run_en.sh"
