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


def test_common_settings_keep_two_independent_comfy_ports() -> None:
    assert 'id="setting-comfyui-port"' in FRONTEND
    assert 'id="setting-comfyui-port-2"' in FRONTEND
    assert 'placeholder="8188"' in FRONTEND
    assert 'placeholder="8187"' in FRONTEND
    assert 'id="setting-illust-port-enabled"' in FRONTEND
    assert 'id="setting-comfyui-port-illustration"' not in FRONTEND
    assert "comfyui_port_2:" in FRONTEND


def test_runtime_has_two_instance_tabs_controls_and_raw_terminal() -> None:
    for value in (
        'id="comfy-runtime-tab-1"',
        'id="comfy-runtime-tab-2"',
        'id="comfy-runtime-enable-cors" checked',
        'id="comfy-runtime-listen-all" checked',
        'id="comfy-runtime-fast"',
        'id="comfy-runtime-vram-mode"',
        'id="comfy-runtime-cuda-device"',
        'id="comfy-runtime-start"',
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
    ):
        assert endpoint in FRONTEND

    assert "comfy_launch_profiles: comfyRuntimeProfilesForSave()" in FRONTEND
    assert "enable_cors: true" in FRONTEND
    assert "listen_all: true" in FRONTEND
    assert "fast: false" in FRONTEND
    assert "vram_mode: 'auto'" in FRONTEND


def test_runtime_layout_gives_remaining_space_to_terminal() -> None:
    assert ".settings-modal.comfy-runtime-layout .modal-content" in FRONTEND
    assert ".comfy-runtime-page" in FRONTEND
    assert ".comfy-runtime-terminal-card" in FRONTEND
    assert "flex: 1;" in FRONTEND
