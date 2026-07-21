import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline


@pytest.mark.asyncio
async def test_process_prompt_raises_when_provider_returns_no_image(monkeypatch):
    prompt_id = "provider-failure-test"
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": None,
        "image_bytes": None,
    }

    async def fail_generation(*args, **kwargs):
        return None, "remote server unavailable"

    monkeypatch.setitem(server.app_config, "bot_selected", "")
    monkeypatch.setattr(server, "generate_image_with_prompt", fail_generation)

    try:
        with pytest.raises(RuntimeError, match="remote server unavailable"):
            await server.process_prompt(prompt_id, {}, {})
        assert server.prompts[prompt_id]["status"] == "completed"
        assert server.prompts[prompt_id]["outputs"] == {"images": []}
        assert server.prompts[prompt_id]["image_bytes"] is None
    finally:
        server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_context_queue_retries_at_tail_and_returns_partial_success(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("d" * 64)
    original_prompt_id = "partial-retry-original"
    pipeline.create_session(session_id, "")
    server.prompts[original_prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }

    attempts = {0: 0, 1: 0, 2: 0}
    enqueue_order = []
    child_prompt_ids = []

    async def fake_add_item(item_type, label, params, priority=10, **kwargs):
        assert item_type == "illustration"
        assert priority == 0
        slot_index = int(params["raw_body"]["illustration_context_index"]) - 1
        enqueue_order.append(slot_index)
        attempts[slot_index] += 1
        child_prompt_id = params["prompt_id"]
        child_prompt_ids.append(child_prompt_id)
        future = asyncio.get_running_loop().create_future()
        item = SimpleNamespace(status="completed", completion_future=future)

        succeeds = slot_index == 0 or (slot_index == 1 and attempts[slot_index] == 2)
        if succeeds:
            server.prompts[child_prompt_id]["image_bytes"] = (
                f"image-{slot_index}-attempt-{attempts[slot_index]}".encode()
            )
            future.set_result({"success": True})
        else:
            item.status = "failed"
            future.set_exception(RuntimeError(f"slot {slot_index} failed"))
        return item

    async def fake_complete_prompt(prompt_id, save_node_id, filename):
        server.prompts[prompt_id]["status"] = "completed"

    async def ignore_progress(*args, **kwargs):
        return None

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "set_prompt_by_title", lambda *args, **kwargs: True)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete_prompt)

    descriptors = [
        {
            "kind": "scene",
            "slot": slot,
            "raw_positive": f"positive {slot}",
            "raw_negative": f"negative {slot}",
        }
        for slot in range(3)
    ]
    parent_item = SimpleNamespace(
        params={
            "prompt_id": original_prompt_id,
            "payload": {
                "protocol": "prompt_batch_v1",
                "session_id": session_id,
                "items": descriptors,
            },
            "prompt_data": {},
            "raw_body": {},
        }
    )

    try:
        result = await server.process_illustration_context_queue_item(parent_item)

        # Initial slots are all enqueued first. Both failures are then appended in
        # their original order before either retry is awaited.
        assert enqueue_order == [0, 1, 2, 1, 2]
        assert result == {
            "success": True,
            "session_id": session_id,
            "count": 2,
            "requested_count": 3,
            "failed_count": 1,
        }

        session = pipeline.get_session(session_id)
        assert [item["slot"] for item in session["items"]] == [0, 1]
        assert session["images"] == [b"image-0-attempt-1", b"image-1-attempt-2"]
        assert session["progress"]["phase"] == "ready_partial"
        assert session["progress"]["done"] == 2
        assert session["progress"]["total"] == 3
        assert session["failure_count"] == 1
        assert session["failures"][0]["slot"] == 2
        assert server.prompts[original_prompt_id]["image_bytes"] == b"image-0-attempt-1"
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop("d" * 24, None)
        server.prompts.pop(original_prompt_id, None)
        for child_prompt_id in child_prompt_ids:
            server.prompts.pop(child_prompt_id, None)
