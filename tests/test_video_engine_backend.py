from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from video_engine_backend import (
    VIDEO_ENGINE_DEFAULT_PORT,
    VIDEO_ENGINE_DEFAULT_PROFILE,
    VIDEO_ENGINE_DEFAULT_STEPS,
    VideoEngineError,
    VideoEngineService,
    VideoEngineUnavailableError,
    normalize_video_engine_port,
    normalize_video_engine_profile,
    normalize_video_engine_steps,
    register_video_engine_routes,
)


def _service() -> VideoEngineService:
    return VideoEngineService(
        get_config=lambda: {
            "video_engine_port": VIDEO_ENGINE_DEFAULT_PORT,
            "video_engine_profile": "stock",
        },
        get_comfy_ports=lambda: [(1, 8188), (2, 8187), (3, 8186)],
    )


@pytest.mark.parametrize("value", (8093, "8093", 1, 65535))
def test_video_engine_port_normalization(value) -> None:
    assert normalize_video_engine_port(value) == int(value)


@pytest.mark.parametrize("value", (True, 0, 65536, 8093.5, "08093", "port"))
def test_video_engine_port_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        normalize_video_engine_port(value)


def test_video_engine_profile_normalization() -> None:
    assert normalize_video_engine_profile(None) == VIDEO_ENGINE_DEFAULT_PROFILE
    assert normalize_video_engine_profile("  dasiwa_8turbo_v1_int4  ") == (
        "dasiwa_8turbo_v1_int4"
    )
    with pytest.raises(ValueError, match="문자열"):
        normalize_video_engine_profile(8)


@pytest.mark.parametrize("value", (None, 4, 8, 13, 20, 25, "25", "20", " 8 "))
def test_video_engine_steps_normalization(value) -> None:
    assert normalize_video_engine_steps(value) == (
        VIDEO_ENGINE_DEFAULT_STEPS if value is None else int(value)
    )


@pytest.mark.parametrize("value", (True, False, 3, 26, 8.5, "fast", "", float("inf")))
def test_video_engine_steps_respect_daemon_request_range(value) -> None:
    with pytest.raises(ValueError, match="4~25"):
        normalize_video_engine_steps(value)


@pytest.mark.asyncio
async def test_unconfigured_profile_uses_daemon_default() -> None:
    service = VideoEngineService(
        get_config=lambda: {"video_engine_port": VIDEO_ENGINE_DEFAULT_PORT},
        get_comfy_ports=lambda: [],
    )
    service.models = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "default_profile": "stock",
            "profiles": {"stock": {"label": "MiniMax H3 Stock"}},
        }
    )

    assert await service.selected_profile() == "stock"


@pytest.mark.asyncio
async def test_prepare_video_frees_comfy_before_warming_engine() -> None:
    service = _service()
    events: list[object] = []

    async def free_comfy():
        events.append("free_comfy")
        return []

    async def set_warm(enabled: bool, *, mode: str, profile: str | None = None):
        events.append(("warmup", enabled, mode, profile))
        return {
            "status": "ready",
            "residency": {"model_mode": mode, "model_profile": profile},
        }

    service.free_comfy_memory = free_comfy  # type: ignore[method-assign]
    service._set_warmup = set_warm  # type: ignore[method-assign]

    result = await service.prepare_video(mode="ref2v")

    assert events == ["free_comfy", ("warmup", True, "ref2v", "stock")]
    assert result["residency"]["model_mode"] == "ref2v"
    assert result["residency"]["model_profile"] == "stock"


@pytest.mark.asyncio
async def test_warmup_requires_mode_and_profile_to_match() -> None:
    service = _service()
    service.status = AsyncMock(  # type: ignore[method-assign]
        side_effect=(
            {
                "status": "ready",
                "residency": {
                    "model_mode": "i2v",
                    "model_profile": "dasiwa_8turbo_v1_int4",
                },
            },
            {
                "status": "ready",
                "residency": {
                    "model_mode": "i2v",
                    "model_profile": "stock",
                },
            },
        )
    )
    service._request_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"status": "warming"}
    )

    result = await service._set_warmup(True, mode="i2v", profile="stock")

    assert result["residency"]["model_profile"] == "stock"
    service._request_json.assert_awaited_once_with(
        "PUT",
        "/api/warmup",
        body={"enabled": True, "mode": "i2v", "profile": "stock"},
        timeout_seconds=15.0,
    )


@pytest.mark.asyncio
async def test_generation_forces_configured_profile_into_daemon_request() -> None:
    service = _service()
    service.models = AsyncMock(return_value={"profiles": {"stock": {}}})  # type: ignore[method-assign]
    service.prepare_video = AsyncMock(return_value={"status": "ready"})  # type: ignore[method-assign]
    service._request_json = AsyncMock(  # type: ignore[method-assign]
        side_effect=(
            {"job_id": "job-1"},
            {"status": "completed", "progress": 1.0, "result": {}},
        )
    )
    service._request_bytes = AsyncMock(return_value=b"mp4")  # type: ignore[method-assign]

    video, descriptor = await service.generate_video(
        {"mode": "i2v", "prompt": "prompt"}
    )

    assert video == b"mp4"
    service.prepare_video.assert_awaited_once_with(mode="i2v", profile="stock")
    assert service._request_json.await_args_list[0].kwargs["body"] == {
        "mode": "i2v",
        "prompt": "prompt",
        "profile": "stock",
        "steps": VIDEO_ENGINE_DEFAULT_STEPS,
    }
    assert descriptor["profile"] == "stock"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("i2v", "first_last", "ref2v"))
@pytest.mark.parametrize(
    ("profile", "fixed_steps", "configured_steps", "expected_steps"),
    (
        ("stock", None, 20, 20),
        ("stock", None, 25, 25),
        ("hybrid", None, 4, 4),
        ("hybrid", None, 8, 8),
        ("hybrid", None, 13, 13),
        ("dasiwa_8turbo_v1_int4", 8, 4, 8),
        ("dasiwa_8turbo_v1_int4", 8, 20, 8),
        ("dasiwa_8turbo_v1_int4", 8, 25, 8),
        ("another_fixed_model", 12, 4, 12),
        ("turbo_named_but_unlocked", None, 20, 20),
        ("", 8, 20, 8),
    ),
)
async def test_generation_uses_setting_or_profile_fixed_steps(
    mode, profile, fixed_steps, configured_steps, expected_steps,
) -> None:
    selected = profile or "daemon_default"
    service = VideoEngineService(
        get_config=lambda: {
            "video_engine_profile": profile,
            "video_engine_steps": configured_steps,
        },
        get_comfy_ports=lambda: [],
    )
    service.models = AsyncMock(return_value={  # type: ignore[method-assign]
        "default_profile": "daemon_default",
        "profiles": {selected: {"fixed_steps": fixed_steps}},
    })
    service.prepare_video = AsyncMock(return_value={"status": "ready"})  # type: ignore[method-assign]
    service._request_json = AsyncMock(side_effect=(  # type: ignore[method-assign]
        {"job_id": "job-steps"},
        {"status": "completed", "progress": 1, "result": {}},
    ))
    service._request_bytes = AsyncMock(return_value=b"mp4")  # type: ignore[method-assign]
    incoming = {"mode": mode, "prompt": "A moving scene", "profile": "stale", "steps": 6}

    _, descriptor = await service.generate_video(incoming)

    sent = service._request_json.await_args_list[0].kwargs["body"]
    assert sent["steps"] == expected_steps
    assert sent["profile"] == selected
    assert incoming["steps"] == 6
    assert incoming["profile"] == "stale"
    assert descriptor["steps"] == expected_steps
    service.prepare_video.assert_awaited_once_with(mode=mode, profile=selected)
    service.models.assert_awaited_once()


@pytest.mark.asyncio
async def test_comfy_transition_treats_offline_video_engine_as_already_released() -> None:
    service = _service()
    service._set_warmup = AsyncMock(  # type: ignore[method-assign]
        side_effect=VideoEngineUnavailableError("offline")
    )

    result = await service.ensure_cold_for_comfy()

    assert result == {
        "reachable": False,
        "port": VIDEO_ENGINE_DEFAULT_PORT,
        "status": "offline",
        "released": True,
    }


@pytest.mark.asyncio
async def test_warmup_error_state_fails_immediately_instead_of_polling_to_timeout() -> None:
    service = _service()
    service.status = AsyncMock(  # type: ignore[method-assign]
        return_value={"status": "error", "error": "insufficient 4080 headroom"}
    )
    service._request_json = AsyncMock(return_value={"status": "warming"})  # type: ignore[method-assign]

    with pytest.raises(VideoEngineError, match="insufficient 4080 headroom"):
        await service._set_warmup(True, mode="i2v")

    assert service.status.await_count == 2
    service._request_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_cooldown_can_recover_engine_from_failed_generation_state() -> None:
    service = _service()
    service.status = AsyncMock(  # type: ignore[method-assign]
        side_effect=(
            {"status": "error", "error": "generation failed"},
            {"status": "cold", "residency": {"encoder_4b": {"loaded_gib": 4.0}}},
        )
    )
    service._request_json = AsyncMock(return_value={"status": "cooling"})  # type: ignore[method-assign]

    result = await service._set_warmup(False, mode="i2v")

    assert result["status"] == "cold"
    service._request_json.assert_awaited_once()


def test_comfy_queue_busy_requires_running_and_pending_lists() -> None:
    assert VideoEngineService._comfy_queue_busy(
        {"queue_running": [], "queue_pending": []}
    ) is False
    assert VideoEngineService._comfy_queue_busy(
        {"queue_running": [[1]], "queue_pending": []}
    ) is True
    with pytest.raises(VideoEngineError):
        VideoEngineService._comfy_queue_busy({"queue_running": []})


class _FakeRuntimeManager:
    def __init__(self) -> None:
        self.running = False
        self.started: list[tuple[str, int]] = []
        self.stop_calls = 0

    def status(self, *, after=0):
        return {
            "state": "running" if self.running else "stopped",
            "running": self.running,
            "managed": self.running,
            "pid": 44556 if self.running else None,
            "port": 8093 if self.running else None,
            "project_path": "E:\\minmax_low_vram" if self.running else "",
            "exit_code": None,
            "logs": [{"seq": 1, "text": "daemon log\n"}] if int(after) < 1 else [],
            "log_seq": 1,
            "log_reset": False,
        }

    def start(self, *, project_path, port):
        self.started.append((project_path, port))
        self.running = True
        return self.status(after=0)

    def stop(self):
        self.stop_calls += 1
        self.running = False
        return self.status(after=0)

    def stop_if_running(self):
        self.running = False


@pytest.mark.asyncio
async def test_status_marks_reachable_unowned_daemon_as_external(monkeypatch) -> None:
    async def fake_status(_self):
        return {"reachable": True, "port": 8093, "status": "cold"}

    monkeypatch.setattr(VideoEngineService, "status", fake_status)
    runtime = _FakeRuntimeManager()
    app = web.Application()
    register_video_engine_routes(
        app,
        get_config=lambda: {
            "video_engine_port": 8093,
            "video_engine_project_path": r"E:\minmax_low_vram",
        },
        get_comfy_ports=lambda: [],
        runtime_manager=runtime,  # type: ignore[arg-type]
    )

    async with TestClient(TestServer(app)) as client:
        response = await client.get("/api/video-engine/status?after=0")
        payload = await response.json()

    assert response.status == 200
    assert payload["runtime"]["external"] is True
    assert payload["runtime"]["managed"] is False
    assert payload["runtime"]["logs"][0]["text"] == "daemon log\n"


@pytest.mark.asyncio
async def test_models_route_proxies_daemon_profiles_and_configured_value(monkeypatch) -> None:
    async def fake_models(_self):
        return {
            "default_profile": "stock",
            "profiles": {
                "stock": {"label": "MiniMax H3 Stock"},
                "dasiwa": {"label": "DaSiWa"},
            },
        }

    monkeypatch.setattr(VideoEngineService, "models", fake_models)
    app = web.Application()
    register_video_engine_routes(
        app,
        get_config=lambda: {
            "video_engine_port": 8093,
            "video_engine_profile": "dasiwa",
        },
        get_comfy_ports=lambda: [],
    )

    async with TestClient(TestServer(app)) as client:
        response = await client.get("/api/video-engine/models")
        payload = await response.json()

    assert response.status == 200
    assert payload["port"] == 8093
    assert payload["default_profile"] == "stock"
    assert payload["configured_profile"] == "dasiwa"
    assert set(payload["profiles"]) == {"stock", "dasiwa"}


@pytest.mark.asyncio
async def test_runtime_start_and_stop_routes_use_configured_identity(monkeypatch) -> None:
    async def fake_status(_self):
        return {"reachable": True, "port": 8093, "status": "cold", "queue_size": 0}

    monkeypatch.setattr(VideoEngineService, "status", fake_status)
    runtime = _FakeRuntimeManager()
    app = web.Application()
    register_video_engine_routes(
        app,
        get_config=lambda: {
            "video_engine_port": 8093,
            "video_engine_project_path": r"E:\minmax_low_vram",
        },
        get_comfy_ports=lambda: [],
        runtime_manager=runtime,  # type: ignore[arg-type]
    )

    async with TestClient(TestServer(app)) as client:
        started = await client.post("/api/video-engine/start", json={})
        stopped = await client.post("/api/video-engine/stop", json={})

    assert started.status == 200
    assert stopped.status == 200
    assert runtime.started == [(r"E:\minmax_low_vram", 8093)]
    assert runtime.stop_calls == 1


@pytest.mark.asyncio
async def test_runtime_stop_is_rejected_while_generation_is_busy(monkeypatch) -> None:
    async def fake_status(_self):
        return {"reachable": True, "port": 8093, "status": "busy", "queue_size": 1}

    monkeypatch.setattr(VideoEngineService, "status", fake_status)
    runtime = _FakeRuntimeManager()
    runtime.running = True
    app = web.Application()
    register_video_engine_routes(
        app,
        get_config=lambda: {"video_engine_port": 8093},
        get_comfy_ports=lambda: [],
        runtime_manager=runtime,  # type: ignore[arg-type]
    )

    async with TestClient(TestServer(app)) as client:
        response = await client.post("/api/video-engine/stop", json={})

    assert response.status == 409
    assert runtime.stop_calls == 0
