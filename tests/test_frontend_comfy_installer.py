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
    assert 'id="comfy-installer-migrate-btn"' in FRONTEND
    assert "이사하기(V4 사용자용)" in FRONTEND
    assert "config.json</code>을 <code>요구사항/</code>에 먼저 백업" in FRONTEND
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
        "/api/comfy-installer/migrate",
    ):
        assert endpoint in FRONTEND
    assert "localStorage.setItem('comfy-installer" not in FRONTEND
    assert "api_key: input?.value || ''" in FRONTEND
    assert "release_version: releaseVersion" in FRONTEND
    assert "selected_item_ids: selectedItemIds" in FRONTEND
    assert "restore_config_after_success" not in FRONTEND
    assert "workflowKeyInput.value = ''" in FRONTEND


def test_successful_update_shows_restart_message_then_requests_shutdown() -> None:
    assert "comfyInstallerLastState === 'running'" in FRONTEND
    assert "data.operation === 'update'" in FRONTEND
    assert "업데이트가 완료되었습니다. 매니저를 재시작해주세요." in FRONTEND
    assert "comfyInstallerShutdownAfterUpdate()" in FRONTEND


def test_v4_migration_reports_config_path_retargeting() -> None:
    assert "result.config?.updated_paths" in FRONTEND
    assert "result.config?.missing_targets" in FRONTEND
    assert "설정 경로를 모두 전환합니다" in FRONTEND


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
