from __future__ import annotations

from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_settings_has_comfy_installer_tab_and_secret_inputs() -> None:
    assert "switchSettingsTab('comfy_install')" in FRONTEND
    assert 'id="settings-tab-comfy_install"' in FRONTEND
    assert (
        'type="password" id="comfy-installer-civitai-key"' in FRONTEND
    )
    assert (
        'type="password" id="comfy-installer-workflow-key"' in FRONTEND
    )
    assert 'id="comfy-installer-pack"' in FRONTEND
    assert 'id="comfy-installer-restore-after-success" checked' in FRONTEND
    assert "17개 워크플로우를 실제 큐에서 실행" in FRONTEND


def test_frontend_uses_dedicated_installer_apis_and_does_not_persist_keys() -> None:
    for endpoint in (
        "/api/comfy-installer/preflight",
        "/api/comfy-installer/workflow-pack",
        "/api/comfy-installer/start",
        "/api/comfy-installer/status",
        "/api/comfy-installer/cancel",
        "/api/comfy-installer/restore-config",
    ):
        assert endpoint in FRONTEND
    assert "localStorage.setItem('comfy-installer" not in FRONTEND
    assert 'civitai_key: civitaiKeyInput.value' in FRONTEND
    assert (
        "restore_config_after_success: "
        "Boolean(restoreAfterSuccessInput?.checked)" in FRONTEND
    )
    assert "civitaiKeyInput.value = ''" in FRONTEND
    assert "workflowKeyInput.value = ''" in FRONTEND


def test_normal_settings_payload_does_not_include_installer_secrets() -> None:
    save_start = FRONTEND.index("async function saveSettings()")
    save_end = FRONTEND.index("\n        //", save_start + 40)
    save_source = FRONTEND[save_start:save_end]
    assert "comfy-installer-civitai-key" not in save_source
    assert "comfy-installer-workflow-key" not in save_source
