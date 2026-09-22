"""High-level contracts for the monolithic frontend bundle.

These checks intentionally cover one representative marker per UI responsibility.
Detailed behavior belongs in the Node and server tests; repeating every HTML string as
an independent pytest case made harmless copy/layout changes look like regressions.
"""

from __future__ import annotations

from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _assert_markers(*, required: tuple[str, ...], forbidden: tuple[str, ...] = ()) -> None:
    missing = [marker for marker in required if marker not in FRONTEND]
    unexpected = [marker for marker in forbidden if marker in FRONTEND]
    assert not missing, f"frontend contract markers are missing: {missing}"
    assert not unexpected, f"removed frontend markers returned: {unexpected}"


def test_frontend_shell_and_navigation_contract() -> None:
    _assert_markers(
        required=(
            'id="header-controls-handle"',
            "toggleHeaderControlsDock(event)",
            'aria-expanded="false"',
            'id="left-utility-stack" hidden',
            'id="memo-launcher"',
            'id="memo-overlay" hidden',
            "/api/memo",
            'id="asset-queue-container"',
            'id="queue-panel-collapse-btn"',
            'id="asset-queue-wall-tab"',
            "setAssetQueuePanelExpanded(expanded)",
            '<div class="sysmon-host-rows">',
            '<div id="sysmon-gpu-rows" class="sysmon-gpu-rows"></div>',
            "while (container.children.length < want * 2)",
            "function showNotiReminderBubble(unreadCount)",
            "const safeLeft = Math.min(Math.max(centeredLeft, viewportMargin), maxLeft);",
            "function syncIndependentScrollHeight()",
            "_assetScrollCache[1] = main.scrollTop;",
        ),
        forbidden=(
            "localStorage.getItem('memo'",
            "localStorage.setItem('memo'",
            "header-controls:hover",
        ),
    )
    assert FRONTEND.index('id="modal-worker-widget"') < FRONTEND.index(
        'id="memo-launcher"'
    ) < FRONTEND.index('id="asset-queue-container"')


def test_frontend_asset_and_illustration_contract() -> None:
    _assert_markers(
        required=(
            'id="bot-lora-character-search"',
            "filterBotLoraCharacters(_botLoraCharacterSearchQuery)",
            "function buildAssetGalleryControls(scope)",
            '<option value="representative_missing"',
            "applyAssetGalleryControls",
            'id="at-fill-top-n" value="3" min="1" max="12"',
            "embedding: atCreateFillRunState('embedding')",
            "call1_backtranslate_slow_retry_enabled",
            "call1_parallel_max_concurrency",
            "call2_parallel_max_concurrency",
            "key: 'multi_char_mask_enabled'",
            "{key: 'minimal_background_description'",
            "{key: 'profile_resolve_enabled'",
            "LLM 실행 연결 정보",
        ),
        forbidden=(
            "expression-profile-breadcrumb",
            "setting-illust-port-enabled",
            "setting-comfyui-port-illustration",
        ),
    )


def test_frontend_character_maker_contract() -> None:
    _assert_markers(
        required=(
            "switchTab('character-maker')",
            'id="tab-character-maker-content"',
            'class="cm-workspace"',
            'class="cm-context-rail"',
            'class="cm-stage"',
            'class="cm-editor-wall"',
            'id="cm-chat-log"',
            'id="cm-settings-wall" class="cm-settings-wall collapsed"',
            "async function cmResetSession()",
            "async function cmAccept()",
            "async function cmConfirmCharacter()",
            "function cmLoadAppearancePreset()",
            "async function cmBuildPromptData(source = 'user')",
            "/api/character_maker/session",
            "/api/character_maker/rag/runtime",
            "/api/character_maker/rag/install",
        ),
        forbidden=(
            "character-maker-model-patcher-trace",
            "character-maker-ipadapter-strength",
        ),
    )
    automatch = FRONTEND.index('id="tab-btn-smart-asset"')
    maker = FRONTEND.index('id="tab-btn-character-maker"', automatch)
    assert automatch < maker


def test_frontend_video_contract() -> None:
    _assert_markers(
        required=(
            'id="video-modal"',
            "openVideoWorkspace(",
            'id="video-generation-source"',
            'id="video-generation-last-preview"',
            'id="video-frame-references"',
            'id="video-reference-picker-modal"',
            "loadMoreVideoReferenceOptions",
            "loadVideoAssetPickerLevel",
            'id="video-reprocess-modal"',
            "/api/video/reprocess/enqueue",
            "/api/video/enqueue",
            "video_workflow_source_paths",
            "calculateVideoFastModeResolution",
            "calculateVideoRefResolution",
            "videoInstructionOriginalInput",
            "videoInstructionLlmTrace",
            "openVideoDirectionEditModal",
            "acceptVideoDirectionEdit",
            "secondary_motion: secondaryMotion",
            "prompt_generation_mode: promptGenerationMode",
            'class="asset-video-postprocess-btn"',
            "video_postprocess",
        ),
        forbidden=(
            'id="tab-video-content"',
            "switchTab('video')",
            "video-generation-overlay",
            'value="t2v"',
        ),
    )
    workflow_choices = (
        "i2v:standard",
        "i2v:fast",
        "first_last:standard",
        "first_last:fast",
        "ref2v:standard",
        "ref2v:fast",
    )
    positions = [FRONTEND.index(f'value="{choice}"') for choice in workflow_choices]
    assert positions == sorted(positions)
    assert FRONTEND.count('type="radio" name="video-mode-choice"') == len(
        workflow_choices
    )


def test_frontend_comfy_runtime_and_installer_contract() -> None:
    _assert_markers(
        required=(
            'id="settings-tab-comfy_runtime"',
            'id="comfy-runtime-tab-1"',
            'id="comfy-runtime-tab-2"',
            'id="comfy-runtime-tab-3"',
            'id="comfy-runtime-tab-vast"',
            'id="comfy-runtime-tab-video-engine"',
            'id="comfy-runtime-tab-modal"',
            'id="comfy-runtime-tab-allocation"',
            'id="comfy-video-engine-panel"',
            "/api/video-engine/start",
            "/api/video-engine/stop",
            'id="comfy-modal-runtime-panel"',
            'id="modal-runtime-worker-gpu"',
            'id="modal-runtime-web-gpu"',
            'id="modal-runtime-vram-mode"',
            'id="modal-operation-lock-modal" hidden',
            "modalRuntimeStartWeb",
            "modalRuntimeStopWeb",
            "modalRenderDeploymentProgress",
            "modalRenderInstallProgress",
            "/api/modal/status?runtime=1",
            "/api/modal/workflows/remote",
            "/api/modal/redeploy",
            'id="settings-tab-comfy_install"',
            "/api/comfy-installer/status",
            "/api/comfy-installer/start",
            "/api/comfy-installer/image-diagnostic",
            "comfy_task_allocations: comfyAllocationsForSave()",
        ),
        forbidden=(
            'id="modal-runtime-enabled"',
            "modalRuntimeRunWorkflow",
            "modal-runtime-result-image",
            'value="normalvram"',
        ),
    )
    common = FRONTEND.index("switchSettingsTab('common')")
    runtime = FRONTEND.index("switchSettingsTab('comfy_runtime')")
    api = FRONTEND.index("switchSettingsTab('api')")
    assert common < runtime < api


def test_frontend_modal_worker_and_installer_failure_contract() -> None:
    _assert_markers(
        required=(
            'id="modal-worker-widget"',
            'id="modal-worker-panel"',
            "refreshModalWorkerStatus",
            "scheduleModalWorkerPolling",
            "case 'app_not_deployed':",
            "case 'network_unavailable':",
            "worker.reason === 'deployment_in_progress'",
            "MODAL_WORKER_IDLE_REFRESH_MS = 60000",
            "MODAL_RUNTIME_LOG_REFRESH_MS = 30000",
            "settingsFormPopulated",
            "await populateSettingsForm()",
            "/api/comfy-installer/workflow-integrity",
            "civitai",
            "SageAttention",
        ),
        forbidden=(
            "원격 ComfyUI 미설치",
            "modalOpenDebugWorkflow",
        ),
    )
