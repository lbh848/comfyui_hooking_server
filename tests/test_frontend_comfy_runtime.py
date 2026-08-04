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
    assert 'id="setting-illust-port-enabled"' not in FRONTEND
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
        'id="comfy-runtime-auto-start"',
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
    assert "auto_start: false" in FRONTEND
    assert "auto_start: source.auto_start === true" in FRONTEND
    assert "매니저 시작 시 자동 실행" in FRONTEND
    assert "enable_cors: true" in FRONTEND
    assert "listen_all: true" in FRONTEND
    assert "fast: false" in FRONTEND
    assert "vram_mode: 'auto'" in FRONTEND


def test_runtime_has_task_allocation_tab_next_to_second_instance() -> None:
    instance_1 = FRONTEND.index('id="comfy-runtime-tab-1"')
    instance_2 = FRONTEND.index('id="comfy-runtime-tab-2"')
    allocation = FRONTEND.index('id="comfy-runtime-tab-allocation"')

    assert instance_1 < instance_2 < allocation
    assert "Comfy 배분 관리" in FRONTEND
    assert 'id="comfy-runtime-allocation-panel"' in FRONTEND
    assert 'id="comfy-allocation-list"' in FRONTEND


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
    assert "Comfy #1만 실행 중 · 모든 내부 Comfy 작업은 자동으로 #1에 들어갑니다." in FRONTEND
    assert "Comfy #2만 실행 중 · 모든 내부 Comfy 작업은 자동으로 #2에 들어갑니다." in FRONTEND


def test_runtime_layout_gives_remaining_space_to_terminal() -> None:
    assert ".settings-modal.comfy-runtime-layout .modal-content" in FRONTEND
    assert ".comfy-runtime-page" in FRONTEND
    assert ".comfy-runtime-terminal-card" in FRONTEND
    assert "flex: 1;" in FRONTEND
