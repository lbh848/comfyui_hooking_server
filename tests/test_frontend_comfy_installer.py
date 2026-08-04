from __future__ import annotations

from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_settings_has_comfy_installer_tab_and_key_inputs() -> None:
    assert "switchSettingsTab('comfy_install')" in FRONTEND
    assert ">업데이트 및 설치</button>" in FRONTEND
    assert 'id="settings-tab-comfy_install"' in FRONTEND
    assert (
        'type="text" id="comfy-installer-civitai-key"' in FRONTEND
    )
    assert (
        'type="password" id="comfy-installer-workflow-key"' in FRONTEND
    )
    assert 'id="comfy-installer-pack"' in FRONTEND
    assert 'id="comfy-installer-unpack-btn"' in FRONTEND
    assert 'id="comfy-installer-update-btn"' in FRONTEND
    assert 'id="comfy-installer-compat-start-btn"' in FRONTEND
    assert 'id="comfy-installer-migrate-btn"' in FRONTEND
    assert 'id="comfy-installer-retarget-config-btn"' in FRONTEND
    assert ">문제 해결</h3>" in FRONTEND
    assert 'id="comfy-installer-repair-civitai-key"' in FRONTEND
    assert 'id="comfy-installer-repair-civitai-btn"' in FRONTEND
    assert "이사하기(V4 사용자용)" in FRONTEND
    assert "config를 내장 Comfy로 수정" in FRONTEND
    assert "comfy/.installer-state/backups/config" in FRONTEND
    assert 'id="comfy-installer-restore-after-success"' not in FRONTEND
    assert "절대 자동 덮어쓰기 안 함" in FRONTEND


def test_comfy_installer_uses_polished_step_layout() -> None:
    assert ".settings-modal.comfy-installer-layout .modal-content" in FRONTEND
    assert "'comfy-installer-layout'" in FRONTEND
    assert 'class="comfy-installer-step"' in FRONTEND
    assert 'class="comfy-installer-card comfy-installer-run-card"' in FRONTEND
    assert 'class="comfy-installer-card comfy-installer-monitor-card"' in FRONTEND


def test_frontend_uses_dedicated_installer_apis_and_does_not_persist_keys() -> None:
    for endpoint in (
        "/api/comfy-installer/preflight",
        "/api/comfy-installer/workflow-pack",
        "/api/comfy-installer/start",
        "/api/comfy-installer/update",
        "/api/comfy-installer/shutdown-after-update",
        "/api/comfy-installer/status",
        "/api/comfy-installer/cancel",
        "/api/comfy-installer/unpack-workflow-pack",
        "/api/comfy-installer/workflow-library",
        "/api/comfy-installer/civitai-key",
        "/api/comfy-installer/troubleshooting/civitai-key",
        "/api/comfy-installer/migrate",
        "/api/comfy-installer/retarget-config",
    ):
        assert endpoint in FRONTEND
    assert "localStorage.setItem('comfy-installer" not in FRONTEND
    assert "api_key: input?.value || ''" in FRONTEND
    assert "release_version: releaseVersion" in FRONTEND
    assert "selected_item_ids: selectedItemIds" in FRONTEND
    assert "install_mode: installMode" in FRONTEND
    assert "restore_config_after_success" not in FRONTEND
    assert "workflowKeyInput.value = ''" in FRONTEND


def test_troubleshooting_civitai_key_replacement_requires_comfy_restart() -> None:
    assert "async function comfyInstallerReplaceLoraManagerCivitaiKey()" in FRONTEND
    assert "설치기와 LoRA Manager 설정에 함께 저장합니다" in FRONTEND
    assert "ComfyUI를 재시작한 뒤 다시 다운로드하세요" in FRONTEND


def test_nvidia_compatibility_install_warns_about_sageattention_workflows() -> None:
    assert "comfyInstallerStart('nvidia_compatibility')" in FRONTEND
    assert "RTX 2070/2080 같은 Turing(sm75)" in FRONTEND
    assert "SageAttention과 전용 Triton을 제외합니다" in FRONTEND
    assert "SageAttention 노드를 제거하거나" in FRONTEND
    assert "disabled로 설정" in FRONTEND
    assert "compatibility_warning" in FRONTEND


def test_successful_update_shows_restart_message_then_requests_shutdown() -> None:
    assert "comfyInstallerLastState === 'running'" in FRONTEND
    assert "data.operation === 'update'" in FRONTEND
    assert "업데이트가 완료되었습니다. 매니저를 재시작해주세요." in FRONTEND
    assert "comfyInstallerShutdownAfterUpdate()" in FRONTEND


def test_v4_migration_reports_config_path_retargeting() -> None:
    assert "result.config?.updated_paths" in FRONTEND
    assert "result.config?.missing_targets" in FRONTEND
    assert "Comfy 관련 설정 경로를 하나씩 검사" in FRONTEND
    assert "data.operation === 'migrate'" in FRONTEND
    assert "comfyInstallerFormatDuration(progress.eta_seconds)" in FRONTEND
    assert "robocopy 병렬" in FRONTEND
    assert "started.state === 'succeeded'" in FRONTEND
    assert "result.config?.already_retargeted" in FRONTEND
    assert "설정 이미 전환됨" in FRONTEND
    assert "comfyInstallerReloadSettingsAfterConfigChange()" in FRONTEND


def test_comfy_installer_tab_is_right_of_debug() -> None:
    debug = FRONTEND.index("switchSettingsTab('debug')")
    comfy = FRONTEND.index("switchSettingsTab('comfy_install')")
    assert debug < comfy


def test_normal_settings_payload_does_not_include_installer_secrets() -> None:
    save_start = FRONTEND.index("async function saveSettings()")
    save_end = FRONTEND.index("\n        //", save_start + 40)
    save_source = FRONTEND[save_start:save_end]
    assert "comfy-installer-civitai-key" not in save_source
    assert "comfy-installer-workflow-key" not in save_source
