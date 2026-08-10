from __future__ import annotations

from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _function_source(name: str, next_name: str) -> str:
    return FRONTEND.split(f"function {name}", 1)[1].split(
        f"function {next_name}", 1
    )[0]


def test_modal_worker_panel_is_attached_above_memo_and_collapsible() -> None:
    widget = FRONTEND.index('id="modal-worker-widget"')
    memo = FRONTEND.index('id="memo-launcher"')
    queue = FRONTEND.index('id="asset-queue-container"')

    assert 'id="left-utility-stack" hidden' in FRONTEND
    assert widget < memo < queue
    assert 'id="modal-worker-panel"' in FRONTEND
    assert 'id="modal-worker-wall-tab"' in FRONTEND
    assert "let modalWorkerExpanded = false;" in FRONTEND
    assert 'class="modal-worker-wall-label">M</span>' in FRONTEND
    assert 'modal-worker-wall-count' not in FRONTEND
    assert 'modal-worker-wall-arrow' not in FRONTEND
    assert "width: 30px;" in FRONTEND
    assert "min-height: 52px;" in FRONTEND
    assert 'onclick="collapseModalWorkerPanel()"' in FRONTEND
    assert 'onclick="expandModalWorkerPanel()"' in FRONTEND
    assert "panel.inert = !modalWorkerExpanded;" in FRONTEND
    assert "wallTab.tabIndex = modalWorkerExpanded ? -1 : 0;" in FRONTEND


def test_modal_worker_panel_uses_lightweight_adaptive_polling() -> None:
    refresh = _function_source(
        "refreshModalWorkerStatus(manual = false)",
        "applyModalWorkerFeatureConfig(config)",
    )
    schedule = _function_source(
        "scheduleModalWorkerPolling()",
        "modalWorkerCount(data, key)",
    )
    queue_sync = _function_source(
        "notifyModalWorkerQueueState(queueStatus)",
        "handleModalWorkerVisibilityChange()",
    )

    assert "const MODAL_WORKER_IDLE_REFRESH_MS = 60000;" in FRONTEND
    assert "modalWorkerActiveRefreshSeconds * 1000" in FRONTEND
    assert "fetchJSON('/api/modal/worker-status')" in refresh
    assert "document.hidden" in schedule
    assert "area === 'modal'" in queue_sync
    assert "area === 'comfy_parallel'" in queue_sync
    assert "void refreshModalWorkerStatus(false);" in queue_sync
    assert "visibilitychange" in FRONTEND


def test_modal_worker_panel_explains_deployment_and_network_failures() -> None:
    error_copy = _function_source(
        "modalWorkerErrorCopy(reason)",
        "renderModalWorkerError(message, reason = 'runtime_unavailable')",
    )
    render_error = _function_source(
        "renderModalWorkerError(message, reason = 'runtime_unavailable')",
        "renderModalWorkerStatus(data)",
    )
    render_status = _function_source(
        "renderModalWorkerStatus(data)",
        "refreshModalWorkerStatus(manual = false)",
    )
    render_deploying = _function_source(
        "renderModalWorkerDeploying(data)",
        "renderModalWorkerStatus(data)",
    )

    assert "case 'app_not_deployed':" in error_copy
    assert "state: '작업 App 미배포'" in error_copy
    assert "동기화된 자산은 유지됩니다 · MODAL 탭에서 기본 앱 재배포를 실행하세요" in error_copy
    assert "원격 ComfyUI 미설치" not in error_copy
    assert "워크플로우를 동기화하세요" not in error_copy
    assert "case 'network_unavailable':" in error_copy
    assert "state: 'Modal 네트워크 장애'" in error_copy
    assert "Modal 서버에 연결하지 못했습니다" in error_copy
    assert "elements.state.textContent = copy.state;" in render_error
    assert "elements.checked.textContent = `${copy.detail}" in render_error
    assert "elements.state.textContent = 'Modal 배포 중';" in render_deploying
    assert "elements.widget.dataset.state = 'checking';" in render_deploying
    assert "worker.reason === 'deployment_in_progress'" in render_status
    assert "renderModalWorkerDeploying(data);" in render_status
    assert "worker.reason," in render_status


def test_modal_worker_panel_follows_config_and_shared_queue_offset() -> None:
    apply_config = _function_source(
        "applyModalWorkerFeatureConfig(config)",
        "notifyModalWorkerQueueState(queueStatus)",
    )
    position = _function_source(
        "syncMemoLauncherPosition()",
        "scheduleMemoLauncherPositionSync()",
    )

    assert "modalWorkerFeatureEnabled = config.modal_enabled === true;" in apply_config
    assert "widget.hidden = !modalWorkerFeatureEnabled;" in apply_config
    assert "stopModalWorkerPolling();" in apply_config
    assert "syncLeftUtilityStackVisibility();" in apply_config
    assert "document.getElementById('left-utility-stack')" in position
    assert "window.innerHeight - rect.top + 10" in position
    assert "queueContainer.classList.contains('collapsed')" in position


def test_modal_runtime_status_and_logs_poll_independently() -> None:
    refresh = _function_source(
        "modalRuntimeRefresh(showResult = false)",
        "modalRuntimeSaveAndApply()",
    )
    start_polling = _function_source(
        "modalRuntimeStartPolling()",
        "modalRuntimeRefresh(showResult = false)",
    )

    assert "fetchJSON('/api/modal/status?runtime=1')" in refresh
    assert "modalRuntimeRefreshLogs" not in refresh
    assert "void modalRuntimeRefresh(false);" in start_polling
    assert "void modalRuntimeRefreshLogs(false);" in start_polling
    assert "const MODAL_RUNTIME_LOG_REFRESH_MS = 30000;" in FRONTEND
