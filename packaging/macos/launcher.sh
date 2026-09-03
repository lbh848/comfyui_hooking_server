#!/bin/bash
# SoyaComfy.app 실행기 — 번들 안의 사본을 쓰기 가능한 곳으로 옮긴 뒤 run_en.sh 를 띄운다.
#
# 왜 복사하나: /Applications 안의 번들은 읽기 전용이고, 이 앱은 ComfyUI(12 GiB) ·
# 모델 · config.json · logs 를 **자기 폴더 안에** 만든다. 번들 안에서 돌리면 첫
# 설치부터 실패한다. 그래서 코드만 번들에 담고, 작업 트리는 사용자 폴더에 둔다.
#
# 왜 Terminal 을 여나: 첫 실행은 uv 설치 · 파이썬 의존성 동기화 · 영상 도구
# 내려받기로 수 분이 걸리고, 그동안 아무 창도 없으면 멈춘 것처럼 보인다.
# 서버 자체도 콘솔 로그가 본체다.

set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAYLOAD="$BUNDLE/Contents/Resources/app"
DATA_ROOT="${SOYA_DATA_ROOT:-$HOME/Library/Application Support/SoyaComfy}"
VERSION_FILE="$DATA_ROOT/.soya-app-version"
BUNDLE_VERSION="$(defaults read "$BUNDLE/Contents/Info" CFBundleShortVersionString 2>/dev/null || echo "0")"

# 사용자 데이터는 절대 건드리지 않는다. 갱신 대상은 여기 적힌 것뿐이다.
# (config.json · key/ · logs/ · comfy/ · models/ · workflow*/ · asset*/ 등은 제외)
CODE_PATHS=(
    server.py queue_manager.py comfy_allocation.py comfy_runtime.py
    frontend_auth.py frontend_ws_manager.py video_engine_runtime.py
    ensure_models.py ensure_video_tools.py run_en.sh run_en.bat
    pyproject.toml uv.lock
    modes modal_backend comfy_installer frontend customprompt fonts docker scripts
)

die() {
    echo "[SoyaComfy] $*" >&2
    /usr/bin/osascript -e "display alert \"SoyaComfy\" message \"$*\" as critical" >/dev/null 2>&1 || true
    exit 1
}

[ -d "$PAYLOAD" ] || die "번들이 손상되었습니다: $PAYLOAD 가 없습니다."

# 필수 도구를 먼저 확인합니다. 없으면 run_en.sh 가 Terminal 안에서 실패하는데,
# 그 창은 사용자가 읽지 않고 닫기 쉽습니다. 무엇을 설치해야 하는지 알림으로 알립니다.
#
# **반드시 로그인 셸로 확인해야 합니다.** Finder 로 띄운 앱의 PATH 에는
# /opt/homebrew/bin 이 없어서, Homebrew 로 깐 ffmpeg 이 있어도 "없음"으로 보입니다.
# 아래 Terminal 은 로그인 셸로 열리므로 그쪽 PATH 를 기준으로 판정해야 맞습니다.
LOGIN_SHELL="${SHELL:-/bin/zsh}"
has_tool() { "$LOGIN_SHELL" -lc "command -v $1 >/dev/null 2>&1"; }

missing=""
has_tool git     || missing="$missing"$'\n'"· Git — 터미널에서: xcode-select --install"
has_tool ffmpeg  || missing="$missing"$'\n'"· FFmpeg — 터미널에서: brew install ffmpeg"
has_tool ffprobe || missing="$missing"$'\n'"· ffprobe (FFmpeg 에 포함) — brew install ffmpeg"
if [ -n "$missing" ]; then
    die "먼저 설치해야 하는 것이 있습니다:$missing"
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "[SoyaComfy] 첫 실행: 작업 폴더를 만듭니다 → $DATA_ROOT"
    mkdir -p "$DATA_ROOT"
    # ditto 는 권한·심볼릭 링크를 보존한다. cp -R 은 macOS 에서 메타데이터를 흘린다.
    /usr/bin/ditto "$PAYLOAD" "$DATA_ROOT" || die "작업 폴더 생성에 실패했습니다."
    printf '%s\n' "$BUNDLE_VERSION" > "$VERSION_FILE"
elif [ "$(cat "$VERSION_FILE" 2>/dev/null || echo 0)" != "$BUNDLE_VERSION" ]; then
    echo "[SoyaComfy] 새 버전($BUNDLE_VERSION)을 반영합니다. 사용자 데이터는 그대로 둡니다."
    for path in "${CODE_PATHS[@]}"; do
        [ -e "$PAYLOAD/$path" ] || continue
        if [ -d "$PAYLOAD/$path" ]; then
            /usr/bin/rsync -a --delete "$PAYLOAD/$path/" "$DATA_ROOT/$path/"
        else
            /usr/bin/rsync -a "$PAYLOAD/$path" "$DATA_ROOT/$path"
        fi
    done
    printf '%s\n' "$BUNDLE_VERSION" > "$VERSION_FILE"
fi

chmod +x "$DATA_ROOT/run_en.sh" 2>/dev/null || true

# osascript 문자열 안으로 들어가므로 역슬래시와 큰따옴표를 먼저 이스케이프한다.
escaped_root="${DATA_ROOT//\\/\\\\}"
escaped_root="${escaped_root//\"/\\\"}"

/usr/bin/osascript \
    -e "tell application \"Terminal\" to do script \"cd \\\"$escaped_root\\\" && ./run_en.sh\"" \
    -e 'tell application "Terminal" to activate' \
    >/dev/null || die "Terminal 을 열지 못했습니다. 직접 실행하세요: cd \"$DATA_ROOT\" && ./run_en.sh"
