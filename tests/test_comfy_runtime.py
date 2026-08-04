from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfy_runtime import (
    ComfyRuntimeManager,
    ComfyRuntimeValidationError,
    autostart_comfy_instances,
    normalize_comfy_launch_profile,
    register_comfy_runtime_routes,
)


def test_launch_profile_defaults_enable_network_options() -> None:
    profile = normalize_comfy_launch_profile({})

    assert profile == {
        "auto_start": False,
        "enable_cors": True,
        "listen_all": True,
        "fast": False,
        "vram_mode": "auto",
        "cuda_device": None,
    }


def test_build_command_uses_supported_comfy_arguments(tmp_path: Path) -> None:
    manager = ComfyRuntimeManager(tmp_path)

    command, port, profile = manager.build_command(
        port=8187,
        profile={
            "enable_cors": True,
            "listen_all": True,
            "fast": True,
            "vram_mode": "lowvram",
            "cuda_device": 1,
        },
    )

    assert port == 8187
    assert profile["cuda_device"] == 1
    assert command[:3] == [
        str(manager.python_path),
        "-u",
        str(manager.main_path),
    ]
    assert command[command.index("--port") + 1] == "8187"
    assert command[command.index("--listen") + 1] == "0.0.0.0"
    assert command[command.index("--enable-cors-header") + 1] == "*"
    assert command[command.index("--cuda-device") + 1] == "1"
    assert "--lowvram" in command
    assert command[-1] == "--fast"


@pytest.mark.parametrize(
    "profile",
    (
        {"auto_start": "yes"},
        {"enable_cors": "yes"},
        {"vram_mode": "unknown"},
        {"cuda_device": -1},
    ),
)
def test_launch_profile_rejects_invalid_values(profile: dict) -> None:
    with pytest.raises(ComfyRuntimeValidationError):
        normalize_comfy_launch_profile(profile)


class _FakeProcess:
    def __init__(self, output: bytes) -> None:
        self.pid = 43210
        self.stdout = io.BytesIO(output)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 1

    def kill(self) -> None:
        self.returncode = 1


def test_runtime_forwards_process_stdout_without_summarizing(tmp_path: Path) -> None:
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _FakeProcess(b"Comfy raw stdout\ncustom node output\n")

    manager = ComfyRuntimeManager(tmp_path, popen_factory=fake_popen)
    manager.python_path.parent.mkdir(parents=True)
    manager.python_path.write_text("", encoding="utf-8")
    manager.main_path.parent.mkdir(parents=True, exist_ok=True)
    manager.main_path.write_text("", encoding="utf-8")
    manager._port_is_in_use = lambda _port: False  # type: ignore[method-assign]
    manager._create_windows_job = lambda _process: None  # type: ignore[method-assign]

    manager.start(instance_id=1, port=8188, profile={})
    deadline = time.monotonic() + 1
    payload = manager.status(instance_id=1, after=0)
    while not payload["logs"] and time.monotonic() < deadline:
        time.sleep(0.01)
        payload = manager.status(instance_id=1, after=0)

    assert "".join(item["text"] for item in payload["logs"]) == (
        "Comfy raw stdout\ncustom node output\n"
    )
    assert captured["kwargs"]["stderr"] is not None


def test_runtime_reports_managed_process_running_state(tmp_path: Path) -> None:
    manager = ComfyRuntimeManager(tmp_path)
    process = _FakeProcess(b"")
    manager._states[1].process = process

    assert manager.is_running(instance_id=1) is True
    process.returncode = 0
    assert manager.is_running(instance_id=1) is False


def test_autostart_starts_only_enabled_instances() -> None:
    calls: list[dict] = []

    class _Manager:
        def start(self, **kwargs):
            calls.append(kwargs)
            return {"instance_id": kwargs["instance_id"], "running": True}

    started = autostart_comfy_instances(
        _Manager(),  # type: ignore[arg-type]
        profiles={
            "1": {"auto_start": True},
            "2": {"auto_start": False},
        },
        ports={1: 8188, 2: 8187},
    )

    assert [call["instance_id"] for call in calls] == [1]
    assert calls[0]["port"] == 8188
    assert started == {1: {"instance_id": 1, "running": True}}


def test_autostart_keeps_instance_failures_isolated() -> None:
    calls: list[dict] = []

    class _Manager:
        def start(self, **kwargs):
            calls.append(kwargs)
            if kwargs["instance_id"] == 1:
                raise RuntimeError("first instance failed")
            return {"instance_id": kwargs["instance_id"], "running": True}

    started = autostart_comfy_instances(
        _Manager(),  # type: ignore[arg-type]
        profiles={
            "1": {"auto_start": True},
            "2": {"auto_start": True, "fast": True},
        },
        ports={1: 8188, 2: 8187},
    )

    assert [call["instance_id"] for call in calls] == [1, 2]
    assert calls[0]["port"] == 8188
    assert calls[1]["port"] == 8187
    assert calls[1]["profile"]["fast"] is True
    assert started == {2: {"instance_id": 2, "running": True}}


@pytest.mark.asyncio
async def test_runtime_http_status_and_validation(tmp_path: Path) -> None:
    app = web.Application()
    register_comfy_runtime_routes(app, project_root=tmp_path)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/api/comfy-runtime/status?instance=1&after=0")
        assert response.status == 200
        payload = await response.json()
        assert payload["ok"] is True
        assert payload["state"] == "stopped"

        response = await client.post(
            "/api/comfy-runtime/start",
            json={
                "instance_id": 1,
                "port": 8188,
                "profile": {"vram_mode": "invalid"},
            },
        )
        assert response.status == 400
        assert (await response.json())["ok"] is False

        response = await client.post(
            "/api/comfy-runtime/start",
            json={"instance_id": 1, "port": 8188, "profile": {}},
        )
        assert response.status == 409
        assert "Python" in (await response.json())["error"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_runtime_http_routes_can_require_dashboard_session(tmp_path: Path) -> None:
    app = web.Application()
    register_comfy_runtime_routes(
        app,
        project_root=tmp_path,
        authorize=lambda _request: False,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/api/comfy-runtime/status?instance=1")
        assert response.status == 401
        payload = await response.json()
        assert payload["ok"] is False
        assert "로그인" in payload["error"]
    finally:
        await client.close()
