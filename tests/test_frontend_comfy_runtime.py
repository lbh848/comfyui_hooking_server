from __future__ import annotations

from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_comfy_runtime_tab_is_next_to_common_settings() -> None:
    common = FRONTEND.index("switchSettingsTab('common')")
    runtime = FRONTEND.index("switchSettingsTab('comfy_runtime')")
    api = FRONTEND.index("switchSettingsTab('api')")

    assert common < runtime < api
    assert ">Comfy 실행 관리</button>" in FRONTEND
    assert 'id="settings-tab-comfy_runtime"' in FRONTEND


def test_common_settings_keep_three_independent_comfy_ports() -> None:
    assert 'id="setting-comfyui-port"' in FRONTEND
    assert 'id="setting-comfyui-port-2"' in FRONTEND
    assert 'id="setting-comfyui-port-3"' in FRONTEND
    assert 'placeholder="8188"' in FRONTEND
    assert 'placeholder="8187"' in FRONTEND
    assert 'placeholder="8186"' in FRONTEND
    assert 'id="setting-illust-port-enabled"' not in FRONTEND
    assert 'id="setting-comfyui-port-illustration"' not in FRONTEND
    assert "comfyui_port_2:" in FRONTEND
    assert "comfyui_port_3:" in FRONTEND


def test_runtime_has_three_instance_tabs_controls_and_raw_terminal() -> None:
    for value in (
        'id="comfy-runtime-tab-1"',
        'id="comfy-runtime-tab-2"',
        'id="comfy-runtime-tab-3"',
        'id="comfy-runtime-enable-cors" checked',
        'id="comfy-runtime-listen-all" checked',
        'id="comfy-runtime-fast"',
        'id="comfy-runtime-disable-dynamic-vram"',
        'id="comfy-runtime-vram-mode"',
        'id="comfy-runtime-cuda-device"',
        'id="comfy-runtime-auto-start"',
        'id="comfy-runtime-start"',
        'id="comfy-runtime-free-memory"',
        'id="comfy-runtime-stop"',
        'id="comfy-runtime-terminal"',
    ):
        assert value in FRONTEND

    assert "ComfyUI CMD 출력" in FRONTEND
    assert "terminal.textContent = comfyRuntimeLogText" in FRONTEND
    assert "terminal.innerHTML = comfyRuntimeLogText" not in FRONTEND


def test_runtime_frontend_uses_dedicated_process_apis_and_persists_profiles() -> None:
    for endpoint in (
        "/api/comfy-runtime/status",
        "/api/comfy-runtime/start",
        "/api/comfy-runtime/stop",
        "/api/comfy-runtime/free-memory",
    ):
        assert endpoint in FRONTEND

    assert "comfy_launch_profiles: comfyRuntimeProfilesForSave()" in FRONTEND
    assert "auto_start: false" in FRONTEND
    assert "auto_start: source.auto_start === true" in FRONTEND
    assert "매니저 시작 시 자동 실행" in FRONTEND
    assert "enable_cors: true" in FRONTEND
    assert "listen_all: true" in FRONTEND
    assert "fast: false" in FRONTEND
    assert "disable_dynamic_vram: false" in FRONTEND
    assert "disable_dynamic_vram: source.disable_dynamic_vram === true" in FRONTEND
    assert "'3': comfyRuntimeNormalizeProfile(comfyRuntimeProfiles[3])" in FRONTEND
    assert "vram_mode: 'auto'" in FRONTEND
    assert "async function comfyRuntimeFreeMemory()" in FRONTEND
    assert "freeMemory.disabled = comfyRuntimeActionBusy || !status?.running" in FRONTEND


def test_vram_cleanup_control_is_only_in_local_comfy_instance_panel() -> None:
    local_start = FRONTEND.index('id="comfy-runtime-instance-panel"')
    local_end = FRONTEND.index('id="comfy-runtime-allocation-panel"', local_start)
    local_panel = FRONTEND[local_start:local_end]
    modal_start = FRONTEND.index('id="comfy-modal-runtime-panel"')
    modal_end = FRONTEND.index('id="settings-tab-comfy_install"', modal_start)
    modal_panel = FRONTEND[modal_start:modal_end]

    assert local_panel.count('id="comfy-runtime-free-memory"') == 1
    assert "VRAM/RAM 정리" in local_panel
    assert "VRAM/RAM 정리" not in modal_panel


def test_runtime_has_task_allocation_tab_next_to_second_instance() -> None:
    instance_1 = FRONTEND.index('id="comfy-runtime-tab-1"')
    instance_2 = FRONTEND.index('id="comfy-runtime-tab-2"')
    modal = FRONTEND.index('id="comfy-runtime-tab-modal"')
    allocation = FRONTEND.index('id="comfy-runtime-tab-allocation"')

    assert instance_1 < instance_2 < modal < allocation
    assert "Comfy 배분 관리" in FRONTEND
    assert 'id="comfy-runtime-allocation-panel"' in FRONTEND
    assert 'id="comfy-allocation-list"' in FRONTEND


def test_video_engine_tab_is_next_to_vast_and_exposes_managed_runtime() -> None:
    vast = FRONTEND.index('id="comfy-runtime-tab-vast"')
    engine = FRONTEND.index('id="comfy-runtime-tab-video-engine"')
    allocation = FRONTEND.index('id="comfy-runtime-tab-allocation"')

    assert vast < engine < allocation
    for value in (
        'id="comfy-video-engine-panel"',
        'id="setting-video-engine-port"',
        'id="setting-video-engine-project-path"',
        'id="setting-video-engine-auto-start"',
        'id="video-engine-start"',
        'id="video-engine-stop"',
        'id="video-engine-terminal"',
        'id="video-engine-raise"',
        'id="video-engine-lower"',
        "/api/video-engine/status",
        "/api/video-engine/start",
        "/api/video-engine/stop",
        "/api/video-engine/resources",
        "video_engine_port: videoEnginePort",
        "video_engine_project_path: videoEngineProjectPathValue(false)",
        "video_engine_auto_start:",
        "외부 실행 중",
        "4060의 4B VL은 유지됩니다.",
    ):
        assert value in FRONTEND
    assert "engineOption.textContent = '영상 전용 엔진'" in FRONTEND
    assert "videoEngineSupported: true" in FRONTEND


def test_runtime_has_managed_modal_tab_lifecycle_sync_and_log_controls() -> None:
    allocation = FRONTEND.index('id="comfy-runtime-tab-allocation"')
    modal = FRONTEND.index('id="comfy-runtime-tab-modal"')

    assert modal < allocation
    for value in (
        'id="comfy-modal-runtime-panel"',
        'id="modal-runtime-scaledown-window"',
        'id="modal-runtime-max-concurrency"',
        'id="modal-runtime-container-start-max-retries"',
        'id="modal-runtime-status-refresh"',
        'id="modal-runtime-worker-gpu"',
        'id="modal-runtime-web-gpu"',
        'id="modal-runtime-vram-mode"',
        'id="modal-runtime-worker-gpu-cost"',
        'id="modal-runtime-web-gpu-cost"',
        'id="modal-runtime-combined-cost"',
        'id="modal-runtime-web-fast"',
        'id="modal-runtime-workflow-query-btn"',
        'id="modal-runtime-workflow-query-status"',
        'id="modal-runtime-redeploy-btn"',
        'id="modal-redeploy-choice-modal"',
        'id="modal-operation-lock-modal"',
        'id="modal-runtime-deployment-status"',
        'id="modal-deploy-progress"',
        'id="modal-deploy-progress-title"',
        'id="modal-deploy-progress-elapsed"',
        'id="modal-deploy-progress-log"',
        'id="modal-runtime-web-start-btn"',
        'id="modal-runtime-web-connect-btn"',
        'id="modal-runtime-web-stop-btn"',
        'id="modal-runtime-log"',
        'data-modal-log-filter="all"',
        'data-modal-log-filter="web"',
        'data-modal-log-filter="jobs"',
        'data-modal-log-filter="sync"',
        'data-modal-log-filter="diagnostic"',
        'id="modal-runtime-enabled-status"',
        'id="modal-runtime-metered-cost"',
        'id="modal-runtime-adjustments"',
        'id="modal-runtime-billed-cost"',
        'id="modal-runtime-credit-remaining"',
        "/api/modal/status?runtime=1",
        "/api/modal/billing?refresh=1",
        "/api/modal/workflows/remote",
        "/api/modal/custom-nodes",
        "/api/modal/redeploy",
        "/api/modal/custom-nodes/sync",
        "/api/modal/probe",
        "/api/modal/runtime/logs?entries=500",
        "/api/modal/web/start",
        "/api/modal/web/stop",
    ):
        assert value in FRONTEND

    assert 'id="modal-runtime-enabled"' not in FRONTEND
    assert "modalRuntimeSetEnabled" not in FRONTEND
    assert "외부 API 설정에서만 변경할 수 있습니다." in FRONTEND
    assert "외부 접속 허용" in FRONTEND
    assert "--listen 0.0.0.0 · 웹 UI 필수" in FRONTEND
    assert "modal_web_fast: document.getElementById('modal-runtime-web-fast')?.checked === true" in FRONTEND
    assert "modalWebFastEl.checked = currentConfig.modal_web_fast === true" in FRONTEND
    assert "modal_container_start_max_retries: modalStartRetries" in FRONTEND
    assert "modal_worker_gpu: modalWorkerGpu" in FRONTEND
    assert "modal_web_gpu: modalWebGpu" in FRONTEND
    assert "modal_vram_mode: modalVramMode" in FRONTEND
    assert "currentConfig.modal_vram_mode || 'highvram'" in FRONTEND
    assert "작업 워커 GPU는 설정 저장 후 다음 호출부터 재배포 없이 동적으로 적용됩니다." in FRONTEND
    assert "L4, A10, L40S, A100 40GB, RTX PRO 6000" in FRONTEND
    assert "CUDA 아키텍처 8.0, 8.6, 8.9, 12.0" in FRONTEND
    assert "SageAttention 실제 커널 실행을 확인합니다." in FRONTEND
    assert "currentConfig.modal_container_start_max_retries ?? 2" in FRONTEND
    assert "최초 실행 제외 · 초과 시 강제 취소" in FRONTEND
    assert "60초 캐시" in FRONTEND
    assert "modalRuntimeQueryWorkflows" in FRONTEND
    assert "modalRuntimeRedeploy" in FRONTEND
    assert "modalStartRedeploy" in FRONTEND
    assert 'id="modal-runtime-custom-nodes-btn"' not in FRONTEND
    assert "modalRuntimeSyncCustomNodes" not in FRONTEND
    assert "modalRenderDeploymentProgress" in FRONTEND
    assert "modalRemoteWorkflowStateText" in FRONTEND
    assert "modalRuntimeRunWorkflow" not in FRONTEND
    assert "modal-runtime-result-image" not in FRONTEND
    assert "Modal에서 실행" not in FRONTEND


def test_modal_panel_has_web_url_access_button_and_drops_local_debug_shortcut() -> None:
    for value in (
        'id="modal-runtime-web-start-btn"',
        'id="modal-runtime-web-connect-btn"',
        'id="modal-runtime-web-stop-btn"',
        "modalRuntimeStartWeb",
        "modalRuntimeStopWeb",
        "modalRuntimeOpenWeb",
        "/api/modal/web-url",
        "↗ ComfyUI 접속",
        "GPU 연결 테스트 도움말",
        "유료 진단 기능",
    ):
        assert value in FRONTEND

    assert "로컬 실행 탭" not in FRONTEND
    assert "modalOpenDebugWorkflow" not in FRONTEND


def test_modal_gpu_selectors_are_independent_and_show_hourly_costs() -> None:
    runtime_start = FRONTEND.index('id="comfy-modal-runtime-panel"')
    runtime_end = FRONTEND.index('id="modal-runtime-log"', runtime_start)
    runtime_html = FRONTEND[runtime_start:runtime_end]

    assert runtime_html.count('value="L4" data-vram-gib="24"') == 2
    assert runtime_html.count('value="A10" data-vram-gib="24"') == 2
    assert runtime_html.count('value="L40S" data-vram-gib="48"') == 2
    assert runtime_html.count('value="A100-40GB" data-vram-gib="40"') == 2
    assert runtime_html.count('value="RTX-PRO-6000" data-vram-gib="96"') == 2
    for unsupported_gpu in ("T4", "A100-80GB", "H100"):
        assert f'value="{unsupported_gpu}"' not in runtime_html
    assert "GPU $0.80/시간" in runtime_html
    assert "GPU $1.10/시간" in runtime_html
    assert "GPU $2.10/시간" in runtime_html
    assert "GPU $3.03/시간" in runtime_html
    assert "CUDA 아키텍처 8.0, 8.6, 8.9, 12.0" in runtime_html
    assert "MODAL_CPU_MEMORY_PER_HOUR = 0.3165" in FRONTEND
    assert "modalGpuContainerCostText(worker)" in FRONTEND
    assert "worker.gpuHour + web.gpuHour" in FRONTEND
    assert "Modal L4 실행 관리" not in FRONTEND
    assert "최대 병렬 L4" not in FRONTEND


def test_remote_vram_mode_selectors_are_provider_specific_and_default_high() -> None:
    modal_start = FRONTEND.index('id="modal-runtime-vram-mode"')
    modal_end = FRONTEND.index("</select>", modal_start)
    modal_select = FRONTEND[modal_start:modal_end]
    vast_start = FRONTEND.index('id="setting-vast-vram-mode"')
    vast_end = FRONTEND.index("</select>", vast_start)
    vast_select = FRONTEND[vast_start:vast_end]

    for select in (modal_select, vast_select):
        assert '<option value="highvram" selected>' in select
        for mode in ("normalvram", "lowvram", "novram", "auto"):
            assert f'<option value="{mode}">' in select
    assert "vast_vram_mode: vastVramMode" in FRONTEND
    assert "currentConfig.vast_vram_mode || 'highvram'" in FRONTEND


def test_modal_web_start_stays_locked_until_server_state_acknowledges_request() -> None:
    assert "if (modalWebStartRequestPending || modalOperationLockActive()) return;" in FRONTEND
    assert "modalWebStartRequestPending = true;" in FRONTEND
    assert "if (modalWebStartRequestPending && webState !== 'stopped')" in FRONTEND
    assert (
        "webStartButton.disabled = modalWebStartRequestPending || deploymentBusy || "
        "webBusy || webRunning || !data.connected"
    ) in FRONTEND


def test_modal_redeploy_has_dedicated_live_progress_and_terminal_notifications() -> None:
    for value in (
        'data-phase="inventory"',
        'data-phase="worker"',
        'data-phase="web"',
        'data-phase="shutdown"',
        'id="modal-deploy-progress-track"',
        'id="modal-deploy-progress-current"',
        'id="modal-deploy-progress-count"',
        "const phaseOrder = ['inventory', 'worker', 'web', 'shutdown', 'complete'];",
        "deployment.logs.slice(-10)",
        "modalDeploymentRequestStartedAt = Date.now();",
        "deploymentStartedAt >= modalDeploymentRequestStartedAt - 1500",
        "const staleWhilePending = pending",
        "const endpoint = customNodes ? '/api/modal/custom-nodes/sync' : '/api/modal/redeploy';",
        "button.textContent = `${label} 요청 중…`;",
        "deployment.state === 'completed'",
        "deployment.state === 'failed'",
        "Modal 배포가 완료되었습니다",
        "Modal 배포에 실패했습니다",
    ):
        assert value in FRONTEND

    assert "modal-deployment-busy" in FRONTEND
    assert "@keyframes modal-deployment-spin" in FRONTEND


def test_modal_redeploy_uses_one_button_choice_dialog_and_locked_progress() -> None:
    for value in (
        'id="modal-redeploy-choice-modal" hidden',
        'name="modal-redeploy-kind" value="redeploy" checked',
        'name="modal-redeploy-kind" value="custom_nodes"',
        'id="modal-redeploy-confirm-btn"',
        'id="modal-redeploy-custom-preview"',
        '>기본 앱 재배포</strong>',
        '>Custom Node 포함 재배포</strong>',
        'id="modal-operation-lock-modal" hidden',
        'id="modal-operation-lock-title"',
        'id="modal-operation-lock-log"',
        "function modalOpenRedeployChoice()",
        "function modalConfirmRedeployChoice()",
        "async function modalStartRedeploy(kind)",
        "window.addEventListener('beforeunload'",
        "if (modalOperationLockActive() || modalRedeployChoiceActive()) return;",
    ):
        assert value in FRONTEND

    assert FRONTEND.count('id="modal-runtime-redeploy-btn"') == 1
    assert 'id="modal-runtime-custom-nodes-btn"' not in FRONTEND
    assert "modalRuntimeSyncCustomNodes" not in FRONTEND
    assert 'id="modal-operation-lock-modal" onclick=' not in FRONTEND
    assert 'id="modal-redeploy-choice-modal" onclick=' not in FRONTEND


def test_modal_web_start_uses_same_locked_progress_and_keeps_cancel_action() -> None:
    start = FRONTEND.split("async function modalRuntimeStartWeb()", 1)[1].split(
        "async function modalRuntimeStopWeb()", 1
    )[0]

    assert "modalShowOperationLock(" in start
    assert "'web_start'" in start
    assert "modalSyncOperationLock({web});" in start
    assert "modalHideOperationLock();" in start
    assert 'id="modal-operation-cancel-web-btn"' in FRONTEND
    assert 'onclick="modalRuntimeStopWeb()" hidden' in FRONTEND
    assert "modalOperationLockKind === 'web_start' && webState === 'stopping'" in FRONTEND


def test_modal_web_start_changes_to_cancellable_stop_action() -> None:
    assert 'id="modal-runtime-web-stop-btn" onclick="modalRuntimeStopWeb()" disabled hidden' in FRONTEND
    assert (
        "const showWebStopAction = webState === 'starting' || webRunning || "
        "webState === 'stopping';"
    ) in FRONTEND
    assert "webStartButton.hidden = showWebStopAction;" in FRONTEND
    assert "webStopButton.hidden = !showWebStopAction;" in FRONTEND
    assert "webStopButton.disabled = deploymentBusy || webState === 'stopping';" in FRONTEND
    assert "? 'ComfyUI 시작 취소'" in FRONTEND
    assert "ComfyUI 완전 종료" in FRONTEND


def test_modal_group_is_last_in_external_api_page() -> None:
    api_start = FRONTEND.index('id="settings-tab-api"')
    api_end = FRONTEND.index('id="settings-tab-api_route"')
    api_html = FRONTEND[api_start:api_end]
    assert api_html.rindex('Modal 원격 ComfyUI') > api_html.rindex('챈섭')


def test_modal_installer_is_inside_runtime_modal_panel_not_installer_page() -> None:
    runtime_start = FRONTEND.index('id="settings-tab-comfy_runtime"')
    runtime_end = FRONTEND.index('id="settings-tab-api"')
    runtime_html = FRONTEND[runtime_start:runtime_end]

    installer_start = FRONTEND.index('id="settings-tab-comfy_install"')
    installer_end = FRONTEND.index('id="settings-tab-queue"')
    installer_html = FRONTEND[installer_start:installer_end]

    assert 'id="modal-installer-card"' in runtime_html
    assert 'id="modal-installer-card"' not in installer_html
    assert "modalOpenInstaller" not in FRONTEND

    assert "name.textContent = item.source_name || item.id;" in FRONTEND
    assert 'id="modal-runtime-workflow-query-btn"' in runtime_html
    assert runtime_html.index('id="modal-runtime-workflow-query-btn"') > runtime_html.index(
        'id="modal-installer-card"'
    )

    child_layout_start = FRONTEND.index('.comfy-modal-runtime-panel > *')
    child_layout_end = FRONTEND.index('}', child_layout_start)
    child_layout_rule = FRONTEND[child_layout_start:child_layout_end]
    assert 'flex: 0 0 auto;' in child_layout_rule


def test_modal_workflow_query_renders_filterable_state_badges() -> None:
    for value in (
        'id="modal-workflow-sync-summary"',
        'data-modal-workflow-sync-filter="all"',
        'data-modal-workflow-sync-filter="synced"',
        'data-modal-workflow-sync-filter="different"',
        'data-modal-workflow-sync-filter="missing"',
        'data-modal-workflow-sync-filter="invalid"',
        'id="modal-workflow-sync-count-all"',
        'id="modal-workflow-sync-count-synced"',
        'id="modal-workflow-sync-count-different"',
        'id="modal-workflow-sync-count-missing"',
        'id="modal-workflow-sync-count-invalid"',
        'className = `modal-workflow-sync-badge state-${syncState}`',
        "function modalSetRemoteWorkflowFilter(state)",
        "function modalRenderWorkflowSyncSummary()",
        "동기화 완료",
        "내용 다름",
        "Modal에 없음",
        "원격 파일 오류",
    ):
        assert value in FRONTEND


def test_modal_installer_exposes_live_phase_progress_and_logs() -> None:
    for value in (
        'id="modal-install-progress"',
        'id="modal-install-progress-title"',
        'id="modal-install-progress-elapsed"',
        'data-phase="assets">1. 자산 분석',
        'data-phase="upload">2. Volume 업로드',
        'data-phase="complete">3. 완료',
        'id="modal-install-progress-track"',
        'id="modal-install-progress-current"',
        'id="modal-install-progress-count"',
        'id="modal-install-progress-log"',
        "function modalRenderInstallProgress(install)",
        "progress.mode === 'determinate'",
        "comfyInstallerFormatBytes(Math.min(completedBytes, totalBytes))",
        "modalRenderInstallProgress(data.install",
        "runtime?.reason === 'deployment_in_progress'",
    ):
        assert value in FRONTEND

    assert "modal-install-progress-slide" in FRONTEND
    assert "button.disabled = state === 'running'" in FRONTEND
    assert "if (button && !started) button.disabled = false;" in FRONTEND
    assert "const phaseOrder = ['assets', 'upload', 'complete'];" in FRONTEND


def test_modal_sync_and_app_deployment_have_separate_responsibilities() -> None:
    assert ">선택 워크플로우·모델 동기화</button>" in FRONTEND
    assert ">앱·Custom Node 재배포</button>" in FRONTEND
    assert "앱 코드와 Custom Node 이미지는 재배포하지 않습니다." in FRONTEND
    assert "동기화된 워크플로우·모델은 건드리지 않습니다." in FRONTEND


def test_runtime_persists_detailed_task_allocations_and_shows_fallback_state() -> None:
    for task_key in (
        "illustration",
        "restore_regenerate",
        "asset_generation",
        "qwen_edit",
        "tag_analysis",
        "outfit",
        "asset_lora_training",
        "bot_lora_training",
        "instance_lora",
        "face_extract",
        "utility_debug",
    ):
        assert f"key: '{task_key}'" in FRONTEND

    assert "comfy_task_allocations: comfyAllocationsForSave()" in FRONTEND
    assert "comfy_task_modal_parallel: comfyModalParallelForSave()" in FRONTEND
    assert "comfy_task_vast_parallel: comfyVastParallelForSave()" in FRONTEND
    assert "modalOption.textContent = 'MODAL'" in FRONTEND
    assert "comfyAllocationCreateRemoteParallel(item, 'modal')" in FRONTEND
    assert "comfyAllocationCreateRemoteParallel(item, 'vast')" in FRONTEND
    assert "comfy-allocation-remote-parallel provider-${provider}" in FRONTEND
    assert "comfy-allocation-remote-parallel ${provider}" not in FRONTEND
    assert "const providerPrimary = select.value === provider" in FRONTEND
    assert "checkbox.disabled = providerPrimary" in FRONTEND
    assert "Comfy #1만 실행 중 · 로컬 대상 작업은 #1로 폴백" in FRONTEND
    assert "Comfy #2만 실행 중 · 로컬 대상 작업은 #2로 폴백" in FRONTEND


def test_runtime_layout_gives_remaining_space_to_terminal() -> None:
    assert ".settings-modal.comfy-runtime-layout .modal-content" in FRONTEND
    assert ".comfy-runtime-page" in FRONTEND
    assert ".comfy-runtime-terminal-card" in FRONTEND
    assert "flex: 1;" in FRONTEND
