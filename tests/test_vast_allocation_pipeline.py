from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from comfy_allocation import (
    CURRENT_COMFY_EXECUTION_TARGET,
    VAST_COMFY_TARGET,
)
from queue_manager import QueueItem, QueueManager
from vast_backend.service import VastService


@pytest.mark.asyncio
async def test_vast_service_runs_comfy_workflow_and_downloads_image(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sockets: list[web.WebSocketResponse] = []

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sockets.append(ws)
        async for _message in ws:
            pass
        return ws

    async def prompt(_request: web.Request) -> web.Response:
        for ws in sockets:
            await ws.send_json(
                {
                    "type": "executing",
                    "data": {"prompt_id": "prompt-1", "node": None},
                }
            )
        return web.json_response({"prompt_id": "prompt-1"})

    async def history(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "prompt-1": {
                    "status": {"completed": True},
                    "outputs": {
                        "2": {
                            "images": [
                                {
                                    "filename": "result.png",
                                    "subfolder": "",
                                    "type": "output",
                                }
                            ],
                            "text": ["tag_one, tag_two"],
                        }
                    },
                }
            }
        )

    async def view(_request: web.Request) -> web.Response:
        return web.Response(body=b"png-bytes", content_type="image/png")

    app = web.Application()
    app.router.add_get("/ws", websocket)
    app.router.add_post("/prompt", prompt)
    app.router.add_get("/history/{prompt_id}", history)
    app.router.add_get("/view", view)
    server = TestServer(app)
    await server.start_server()
    service = VastService(tmp_path, lambda: {"vast_enabled": True})
    service.launch["state"] = "ready"
    service.launch["comfy_base_url"] = str(server.make_url("")).rstrip("/")
    service._comfy_tunnel = object()  # type: ignore[assignment]
    service._active_ssh_endpoint = ("host", 22, "key")
    monkeypatch.setattr(service, "_upload_workflow_inputs_sync", lambda *_args: [])
    monkeypatch.setattr(service, "_snapshot_lora_artifacts_sync", lambda *_args: {})
    monkeypatch.setattr(service, "_collect_lora_artifacts_sync", lambda *_args: [])

    try:
        result = await service.run_workflow(
            {
                "2": {
                    "class_type": "SaveImage",
                    "inputs": {},
                    "_meta": {"title": "WD_TAG_TEXT"},
                }
            }
        )
    finally:
        await server.close()

    assert result["prompt_id"] == "prompt-1"
    assert result["images"][0]["bytes"] == b"png-bytes"
    assert result["images"][0]["content_type"] == "image/png"
    assert result["text_outputs"] == [
        {
            "node_id": "2",
            "node_title": "WD_TAG_TEXT",
            "text": "tag_one, tag_two",
        }
    ]


@pytest.mark.asyncio
async def test_vast_training_runs_and_downloads_lora_before_completion() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {"vast_enabled": True}
    frontend_events: list[tuple[str, dict]] = []
    completed: list[str] = []

    async def notify(event_type: str, data: dict) -> None:
        frontend_events.append((event_type, dict(data)))

    async def run_vast_workflow(_workflow: dict, **kwargs):
        assert kwargs["input_paths"] == ["input/alice"]
        assert kwargs["artifact_prefixes"] == ["SOYA_INSTANCE_LORA/alice"]
        assert kwargs["require_images"] is False
        await kwargs["progress_callback"](
            {"phase": "training", "step": 5, "total": 10}
        )
        return {
            "prompt_id": "vast-prompt-1",
            "deferred_artifacts": [
                {
                    "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
                    "remote_path": (
                        "/root/ComfyUI/models/loras/SOYA_CHAR_LORA/"
                        "SOYA_INSTANCE_LORA/alice.safetensors"
                    ),
                    "size": 6,
                }
            ],
        }

    async def download_vast_artifacts(artifacts: list[dict], *, progress_callback):
        assert len(artifacts) == 1
        await progress_callback(
            {
                "phase": "vast_downloading",
                "percentage": 100,
                "index": 1,
                "total_files": 1,
            }
        )
        return {
            "artifacts": [{"local_path": "alice.safetensors"}],
            "remote_delete_queued": [],
        }

    manager.notify_frontend = notify
    manager.run_vast_workflow = run_vast_workflow
    manager.download_vast_artifacts = download_vast_artifacts
    item = QueueItem(
        id="vast-training-1",
        type="instance_lora_training",
        label="Alice Vast 학습",
        params={},
    )
    token = CURRENT_COMFY_EXECUTION_TARGET.set(VAST_COMFY_TARGET)
    try:
        prompt_id, result = await manager._monitor_training_ws(
            item,
            {"1": {"class_type": "Test", "inputs": {}}},
            event_type="instance_lora_training_progress",
            on_complete=lambda: completed.append("done"),
            modal_input_paths=["input/alice"],
            modal_artifact_prefixes=["SOYA_INSTANCE_LORA/alice"],
        )
    finally:
        CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert prompt_id == "vast-prompt-1"
    assert result["download"]["artifacts"][0]["local_path"] == "alice.safetensors"
    assert completed == ["done"]
    assert any(
        data.get("phase") == "all_complete"
        for _, data in frontend_events
    )


@pytest.mark.asyncio
async def test_vast_availability_callback_wakes_waiting_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = QueueManager()
    calls: list[str] = []

    async def ensure_workers() -> None:
        calls.append("workers")

    async def notify_updated() -> None:
        calls.append("notify")

    async def process_loop() -> None:
        calls.append("process")

    monkeypatch.setattr(manager, "_ensure_vast_workers", ensure_workers)
    monkeypatch.setattr(manager, "_notify_queue_updated", notify_updated)
    monkeypatch.setattr(manager, "_process_loop", process_loop)

    await manager.notify_vast_availability_changed()
    await asyncio.sleep(0)

    assert set(calls) == {"workers", "notify", "process"}
    assert manager._vast_wakeup.is_set()
