#!/bin/bash
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")/../.." && pwd)"
PAYLOAD="$BUNDLE/Contents/Resources/app"
DATA_ROOT="${SOYA_DATA_ROOT:-$HOME/Library/Application Support/SoyaComfy}"
UPDATER="$BUNDLE/Contents/Resources/update_payload.py"

# Pass paths as data; AppleScript quotes them for the Terminal shell.
/usr/bin/osascript - "$PAYLOAD" "$DATA_ROOT" "$UPDATER" <<'APPLESCRIPT'
on run argv
    set payload to item 1 of argv
    set dataRoot to item 2 of argv
    set updater to item 3 of argv
    set setupScript to payload & "/packaging/macos/setup.sh"
    set commandText to "/bin/bash " & quoted form of setupScript & " " & quoted form of payload & " " & quoted form of dataRoot & " " & quoted form of updater
    tell application "Terminal"
        do script commandText
        activate
    end tell
end run
APPLESCRIPT
