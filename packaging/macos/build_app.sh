#!/bin/bash
# SoyaComfy.app 를 만든다. macOS 에서만 돈다.
#
# 사용법: packaging/macos/build_app.sh [출력폴더] [버전]
#
# 담는 것은 **git 이 추적하는 파일**뿐이다(`git archive`). 작업 산출물이나
# 사용자 데이터가 섞여 들어갈 길을 원천적으로 막는다 — 제외 목록을 관리하다
# 빠뜨리는 것보다 이쪽이 안전하다.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-$REPO_ROOT/dist}"
VERSION="${2:-0.0.0}"
APP_NAME="SoyaComfy"
APP="$OUT_DIR/$APP_NAME.app"

[ "$(uname -s)" = "Darwin" ] || { echo "[BUILD] macOS 에서만 만들 수 있습니다: $(uname -s)" >&2; exit 1; }

echo "[BUILD] 출력: $APP (version=$VERSION)"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/app"

# ── 페이로드: 추적 파일만 ───────────────────────────────────────────────
echo "[BUILD] 페이로드 복사 (git archive)"
git -C "$REPO_ROOT" archive --format=tar HEAD \
    | tar -x -C "$APP/Contents/Resources/app"

# 앱 안에서 쓸모없는 것은 뺀다. 없어도 동작에는 지장이 없다.
for drop in tests upstream_patches tools .github packaging updater_dev GitUpdater.exe; do
    rm -rf "$APP/Contents/Resources/app/$drop"
done

# ── 실행기 ──────────────────────────────────────────────────────────────
cp "$REPO_ROOT/packaging/macos/launcher.sh" "$APP/Contents/MacOS/$APP_NAME"
chmod +x "$APP/Contents/MacOS/$APP_NAME"
chmod +x "$APP/Contents/Resources/app/run_en.sh"

# ── Info.plist ─────────────────────────────────────────────────────────
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$APP_NAME</string>
    <key>CFBundleDisplayName</key><string>$APP_NAME</string>
    <key>CFBundleIdentifier</key><string>com.github.lbh848.comfyui-hooking-server</string>
    <key>CFBundleExecutable</key><string>$APP_NAME</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# ── 임시 서명 ───────────────────────────────────────────────────────────
# 공증(notarization)이 아니다. Apple 계정 없이 무료로 되는 것만 한다.
# Apple Silicon 은 서명 없는 실행 파일을 아예 거부하므로 이건 선택이 아니다.
echo "[BUILD] 임시(ad-hoc) 서명"
codesign --force --deep --sign - "$APP"
codesign --verify --verbose=2 "$APP"

echo "[BUILD] 완료: $APP"
du -sh "$APP"
