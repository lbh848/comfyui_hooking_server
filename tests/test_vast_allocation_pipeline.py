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
async def test_vast_training_returns_before_background_download_finishes() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {"vast_enabled": True}
    frontend_events: list[tuple[str, dict]] = []
    completed: list[str] = []
    download_started = asyncio.Event()
    release_download = asyncio.Event()

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
        download_started.set()
        await progress_callback(
            {
                "phase": "vast_downloading",
                "percentage": 50,
                "index": 1,
                "total_files": 1,
            }
        )
        await release_download.wait()
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
    manager.items.append(item)
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

    await asyncio.wait_for(download_started.wait(), timeout=1)
    download_item = next(
        queued for queued in manager.items if queued.type == "vast_lora_download"
    )
    assert prompt_id == "vast-prompt-1"
    assert result["download_item_id"] == download_item.id
    assert download_item.status == "processing"
    assert manager._item_execution_area(download_item) == (
        "vast_download",
        "vast-sftp",
    )
    assert manager.get_status()["vast_download_active"] == 1
    assert completed == []
    assert any(
        data.get("phase") == "training_complete"
        for _, data in frontend_events
    )

    release_download.set()
    await asyncio.wait_for(download_item.completion_future, timeout=1)
    assert download_item.status == "completed"
    assert completed == ["done"]


@pytest.mark.asyncio
async def test_next_vast_gpu_job_starts_while_previous_lora_downloads() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {
        "vast_enabled": True,
        "comfy_task_allocations": {"instance_lora": "vast"},
    }
    manager.is_vast_ready = lambda: True
    manager.notify_frontend = lambda *_args, **_kwargs: asyncio.sleep(0)
    manager.get_vast_cleanup_status = lambda: {
        "pending_count": 0,
        "pending_bytes": 0,
    }
    manager.check_vast_storage_headroom = lambda **_kwargs: asyncio.sleep(
        0,
        result={
            "safe": True,
            "free_bytes": 2_000_000_000,
            "required_bytes": 600_000_000,
        },
    )
    download_started = asyncio.Event()
    release_download = asyncio.Event()
    second_started = asyncio.Event()

    async def run_vast_workflow(_workflow: dict, **_kwargs):
        return {
            "prompt_id": "vast-first",
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

    async def download_vast_artifacts(_artifacts: list[dict], *, progress_callback):
        download_started.set()
        await progress_callback(
            {
                "phase": "vast_downloading",
                "percentage": 50,
                "remaining_bytes": 3,
            }
        )
        await release_download.wait()
        return {"artifacts": [{"local_path": "alice.safetensors"}], "remote_delete_queued": []}

    async def execute(item: QueueItem) -> dict:
        if item.label == "first":
            _prompt_id, result = await manager._monitor_training_ws(
                item,
                {"1": {"class_type": "Test", "inputs": {}}},
                modal_input_paths=["input/alice"],
                modal_artifact_prefixes=["SOYA_INSTANCE_LORA/alice"],
            )
            return result
        second_started.set()
        return {"success": True}

    manager.run_vast_workflow = run_vast_workflow
    manager.download_vast_artifacts = download_vast_artifacts
    manager._execute_item = execute
    manager._deferred_prune = lambda _item: asyncio.sleep(0)

    first = await manager.add_item("instance_lora_training", "first", {})
    second = await manager.add_item("instance_lora_training", "second", {})

    try:
        await asyncio.wait_for(download_started.wait(), timeout=1)
        await asyncio.wait_for(second_started.wait(), timeout=1)
        assert release_download.is_set() is False
        assert second.status in ("processing", "completed")

        release_download.set()
        await asyncio.wait_for(
            asyncio.gather(first.completion_future, second.completion_future),
            timeout=1,
        )
        download_item = next(
            item for item in manager.items if item.type == "vast_lora_download"
        )
        await asyncio.wait_for(download_item.completion_future, timeout=1)
    finally:
        release_download.set()
        tasks = [
            task
            for task in (
                *manager._vast_worker_tasks.values(),
                *manager._vast_download_tasks.values(),
            )
            if not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


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


@pytest.mark.asyncio
async def test_local_modal_and_vast_claim_parallel_items_without_duplicates() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "modal_max_concurrency": 1,
        "vast_enabled": True,
        "comfy_task_allocations": {"asset_generation": 1},
        "comfy_task_modal_parallel": {"asset_generation": True},
        "comfy_task_vast_parallel": {"asset_generation": True},
    }
    manager.is_vast_ready = lambda: True
    all_started = asyncio.Event()
    release = asyncio.Event()
    starts: list[tuple[str, str | None]] = []

    async def execute(item: QueueItem) -> dict:
        starts.append((item.id, item.comfy_execution_target))
        if len(starts) == 3:
            all_started.set()
        await release.wait()
        return {"ok": True}

    async def no_prune(_item: QueueItem) -> None:
        return None

    manager._execute_item = execute
    manager._deferred_prune = no_prune
    items = [
        await manager.add_item("asset_generation", f"asset-{index}", {})
        for index in range(3)
    ]

    try:
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert {item_id for item_id, _target in starts} == {
            item.id for item in items
        }
        assert {target for _item_id, target in starts} == {
            "local",
            "modal",
            "vast",
        }

        release.set()
        await asyncio.wait_for(
            asyncio.gather(*(item.completion_future for item in items)),
            timeout=1,
        )
        assert all(item.status == "completed" for item in items)
    finally:
        release.set()
        tasks = [
            task
            for task in (
                *manager._modal_worker_tasks.values(),
                *manager._vast_worker_tasks.values(),
            )
            if not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_modal_primary_and_vast_parallel_do_not_use_local_lane() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "modal_max_concurrency": 1,
        "vast_enabled": True,
        "comfy_task_allocations": {"asset_generation": "modal"},
        "comfy_task_vast_parallel": {"asset_generation": True},
    }
    manager.is_vast_ready = lambda: True
    both_started = asyncio.Event()
    release = asyncio.Event()
    starts: list[tuple[str, str | None]] = []

    async def execute(item: QueueItem) -> dict:
        starts.append((item.id, item.comfy_execution_target))
        if len(starts) == 2:
            both_started.set()
        await release.wait()
        return {"ok": True}

    async def no_prune(_item: QueueItem) -> None:
        return None

    manager._execute_item = execute
    manager._deferred_prune = no_prune
    items = [
        await manager.add_item("asset_generation", f"asset-{index}", {})
        for index in range(2)
    ]

    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert {item_id for item_id, _target in starts} == {
            item.id for item in items
        }
        assert {target for _item_id, target in starts} == {"modal", "vast"}

        release.set()
        await asyncio.wait_for(
            asyncio.gather(*(item.completion_future for item in items)),
            timeout=1,
        )
    finally:
        release.set()
        tasks = [
            task
            for task in (
                *manager._modal_worker_tasks.values(),
                *manager._vast_worker_tasks.values(),
            )
            if not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
