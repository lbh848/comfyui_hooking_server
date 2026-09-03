#!/usr/bin/env bash
# macOS Launcher — run_en.bat의 macOS 대응 버전입니다.
# Windows 전용 단계(PowerShell을 통한 프로젝트 로컬 uv 설치, winget Git, Real-ESRGAN/FFmpeg
# Windows binary 다운로드)는 macOS에 상응하는 절차로 대체됩니다.
set -uo pipefail
cd "$(dirname "$0")"

UV_VERSION="0.11.8"
UV_TOOL_DIR="$PWD/.tools/uv-$UV_VERSION"
UV_EXE="$UV_TOOL_DIR/uv"

fail() { echo "[ERROR] $*" >&2; exit 1; }

echo "[1/7] Checking project-local uv $UV_VERSION..."
if [ ! -x "$UV_EXE" ]; then
    echo "      Project-local uv not found. Installing pinned version..."
    # 주의: 파이프라인 앞의 접두어 대입은 curl에만 적용되므로 반드시 export로 전달해야 합니다.
    ( export UV_INSTALL_DIR="$UV_TOOL_DIR" UV_NO_MODIFY_PATH=1
      curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh ) \
        || fail "Failed to install project-local uv $UV_VERSION. Check the network connection."
    [ -x "$UV_EXE" ] || fail "uv installer completed but the binary is missing: $UV_EXE"
    echo "      Project-local uv installed."
else
    echo "      Project-local uv OK."
fi
export PATH="$UV_TOOL_DIR:$PATH"

echo "[2/7] Checking system Git..."
command -v git >/dev/null 2>&1 \
    || fail "Git not found. Install the Xcode Command Line Tools: xcode-select --install"
echo "      $(git --version)"

# Python 3.12 + packages
echo "[3/7] Setting up Python 3.12 environment..."
"$UV_EXE" sync || fail "Failed to set up environment. Check network connection."

# macOS GPU 가속: onnxruntime-directml은 Windows 전용이므로 pyproject에서
# sys_platform 마커를 통해 분기 처리됩니다. macOS에서는 순정 onnxruntime(CPU/CoreML)을 사용합니다.

echo "[4/7] Checking project-local video tools..."
# ensure_video_tools.py는 Real-ESRGAN(macOS universal build)을 프로젝트 로컬로 다운로드하며,
# FFmpeg은 시스템 PATH에 존재하는 버전을 검증하여 사용합니다 (gyan.dev 빌드는 Windows 전용입니다).
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "[WARN] ffmpeg/ffprobe가 PATH에 존재하지 않습니다. 설치 명령어: brew install ffmpeg"
fi
"$UV_EXE" run --no-sync python ensure_video_tools.py \
    || fail "Failed to prepare Real-ESRGAN or FFmpeg. Check the messages above and network connection."

echo "[5/7] Checking model files..."
"$UV_EXE" run --no-sync python ensure_models.py \
    || fail "Failed to prepare model files. Check the messages above and network connection."

echo "[6/7] Packages, models, and video tools ready."

echo "[7/7] Creating folders..."
for d in workflow current_work workflow_backup frontend logs mode_workflow \
         asset_data auto_complete pose_data chain_presets key; do
    mkdir -p "$d"
done

echo
echo "============================================"
echo "  ComfyUI Proxy Server Start (port 8189)"
echo "  Frontend: http://127.0.0.1:8189/"
echo "============================================"
echo
# -u: 로그를 파일이나 파이프(pipe)로 redirect하더라도 block buffering 없이 즉시 출력되도록 설정합니다.
exec "$UV_EXE" run --no-sync python -u server.py
