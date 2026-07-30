from __future__ import annotations

from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _function_source(name: str, next_name: str) -> str:
    return FRONTEND.split(f"function {name}", 1)[1].split(
        f"function {next_name}", 1
    )[0]


def test_memo_launcher_is_independent_and_above_the_queue() -> None:
    launcher = FRONTEND.index('id="memo-launcher"')
    queue = FRONTEND.index('id="asset-queue-container"')

    assert launcher < queue
    assert 'aria-controls="memo-overlay"' in FRONTEND
    assert "z-index: 2147483647;" in FRONTEND
    assert "z-index: 2147483646;" in FRONTEND
    assert "syncMemoLauncherPosition()" in FRONTEND
    assert "queueContainer.classList.contains('collapsed')" in FRONTEND
    assert "width: 30px;" in FRONTEND
    assert "min-height: 84px;" in FRONTEND
    assert "window.innerHeight - rect.top + 10" in FRONTEND
    assert "memoQueueTransitionTimer" in FRONTEND
    assert "}, 260);" in FRONTEND


def test_memo_dialog_is_centered_accessible_and_closable() -> None:
    assert 'id="memo-overlay" hidden' in FRONTEND
    assert 'role="dialog" aria-modal="true"' in FRONTEND
    assert 'id="memo-textarea"' in FRONTEND
    assert 'maxlength="100000"' in FRONTEND
    assert 'onclick="closeMemoPad()"' in FRONTEND  # ✕ 버튼으로 닫힘
    assert "handleMemoOverlayClick" not in FRONTEND  # 바깥 클릭으로 닫히지 않음
    assert "event.key !== 'Escape'" not in FRONTEND  # ESC로 닫히지 않음
    assert "align-items: center;" in FRONTEND
    assert "justify-content: center;" in FRONTEND


def test_memo_uses_server_memory_instead_of_browser_storage() -> None:
    load_source = _function_source("loadMemoFromServer()", "saveMemoToServer()")
    save_source = _function_source("saveMemoToServer()", "queueMemoSave()")
    beacon_source = _function_source(
        "flushMemoWithBeacon()", "initializeMemoPadUi()"
    )

    assert "fetchJSON('/api/memo')" in load_source
    assert "fetchJSON('/api/memo'," in save_source
    assert "navigator.sendBeacon('/api/memo'" in beacon_source
    assert "memoServerInstanceId" in load_source + save_source + beacon_source
    assert "HTTP 409" in save_source
    assert "localStorage" not in load_source + save_source
    assert "sessionStorage" not in load_source + save_source


def test_common_settings_explains_and_persists_memo_toggle() -> None:
    assert 'id="setting-memo-enabled"' in FRONTEND
    assert 'id="memo-setting-status"' in FRONTEND
    assert "새로고침해도 유지되지만 서버를 다시 시작하면 초기화됩니다." in FRONTEND
    assert (
        "memo_enabled: document.getElementById('setting-memo-enabled').checked"
        in FRONTEND
    )
    assert "applyMemoFeatureEnabled(currentConfig.memo_enabled !== false)" in FRONTEND
