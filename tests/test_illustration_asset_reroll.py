import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline


def _context_payload(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "action": "asset_reroll",
        "target_slotted": (
            "First\n\n[Slot 0]\n\nSecond\n\n[Slot 1]\n\n"
            "Third\n\n[Slot 2]\n\nFourth\n\n[Slot 3]\n\nFifth"
        ),
        "chats": [
            {"role": "user", "data": "Show Aoi after class."},
            {"role": "char", "data": "First\n\nSecond\n\nThird\n\nFourth\n\nFifth"},
        ],
    }


def _regular_descriptor(slot: int = 0) -> dict:
    return {
        "kind": "scene",
        "slot": slot,
        "raw_positive": "regular scene",
        "raw_negative": "",
        "characters": [],
    }


def _asset_descriptor(slot: int, command: str) -> dict:
    return {
        "kind": "original_asset",
        "slot": slot,
        "anchor_before": "Before",
        "anchor_after": "After",
        "anchor_version": 1,
        "original_asset": {
            "bot_name": "sample",
            "character": "Aoi",
            "filename": f"{command}.webp",
            "command": command,
        },
    }


def test_asset_reroll_context_action_uses_the_existing_context_transport() -> None:
    session_id = "risu_" + ("a" * 64)
    parsed = pipeline.parse_context_request(
        pipeline.CONTEXT_PREFIX
        + "\n"
        + json.dumps(_context_payload(session_id), ensure_ascii=False)
    )

    assert parsed is not None
    assert parsed["session_id"] == session_id
    assert parsed["action"] == "asset_reroll"
    assert parsed["slot"] is None
    assert "[Slot 3]" in parsed["target_slotted"]


@pytest.mark.asyncio
async def test_prompt_route_dispatches_asset_reroll_only_to_llm_build_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("d" * 64)
    lookup_key = "d" * 24
    asset_bytes = b"RIFF\x00\x00\x00\x00WEBProute-asset"
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(
        session_id,
        [_asset_descriptor(2, "Aoi_Route.webp")],
        [asset_bytes],
    )
    marker = (
        pipeline.CONTEXT_PREFIX
        + "\n"
        + json.dumps(_context_payload(session_id), ensure_ascii=False)
    )
    queued = []

    async def fake_add_item(item_type, label, params, **kwargs):
        queued.append((item_type, label, params, kwargs))
        return SimpleNamespace()

    class Request:
        async def json(self):
            return {"prompt": {}, "client_id": "client", "extra_data": {}}

    monkeypatch.setattr(server, "log_to_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "cleanup_logs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "extract_prompts_by_title", lambda *_args: marker)
    monkeypatch.setattr(server, "find_save_image_node", lambda *_args: "9")
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    created_prompt_id = None
    try:
        response = await server.handle_prompt(Request())
        body = json.loads(response.text)
        created_prompt_id = body["prompt_id"]
        await asyncio.sleep(0)

        assert response.status == 200
        assert len(queued) == 1
        assert queued[0][0] == "illustration_llm_build"
        assert queued[0][2]["payload"]["action"] == "asset_reroll"
        assert pipeline.get_session(session_id)["items"][0]["slot"] == 2
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        if created_prompt_id:
            server.prompts.pop(created_prompt_id, None)


@pytest.mark.asyncio
async def test_asset_reroll_reuses_selector_moves_slots_and_never_enqueues_comfy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("b" * 64)
    lookup_key = "b" * 24
    prompt_id = "asset-reroll-prompt"
    regular_bytes = server.create_placeholder_png()
    old_asset_bytes = b"RIFF\x00\x00\x00\x00WEBPold-asset"
    new_asset_bytes = b"RIFF\x00\x00\x00\x00WEBPnew-asset"
    old_regular = _regular_descriptor(0)
    old_asset = _asset_descriptor(2, "Aoi_School_old.webp")
    new_asset = _asset_descriptor(3, "Aoi_School_new.webp")

    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(
        session_id,
        [old_regular, old_asset],
        [regular_bytes, old_asset_bytes],
    )
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }

    monkeypatch.setattr(
        server,
        "_capture_illustration_runtime_snapshot",
        lambda: {
            "bot_name": "sample",
            "provider": "comfy",
            "illustration_workflow_type": "v3",
            "illustration_context_toggles": {
                "illustration_output_mode": "both",
                "original_asset_count": 9,
                "original_asset_instruction": "Aoi: choose an uploaded source image",
            },
        },
    )

    async def fake_select(**kwargs):
        assert pipeline.get_session(session_id)["status"] == "building"
        assert kwargs["payload"]["action"] == "asset_reroll"
        assert kwargs["toggles"]["original_asset_count"] == 1
        assert kwargs["used_slots"] == {0}
        assert kwargs["reserve_slot_count"] == 0
        return {
            "items": [new_asset],
            "images": [new_asset_bytes],
            "failures": [],
            "requested_count": 1,
            "target_slotted": kwargs["payload"]["target_slotted"],
        }

    async def should_not_enqueue(*_args, **_kwargs):
        raise AssertionError("에셋 리롤이 Comfy/일반 삽화 큐에 진입하면 안 됩니다")

    async def ignore_progress(*_args, **_kwargs):
        return None

    async def complete_prompt(completed_prompt_id, _save_node_id, filename):
        entry = server.prompts[completed_prompt_id]
        entry["status"] = "completed"
        entry["filename"] = filename

    monkeypatch.setattr(server, "_select_original_asset_outputs", fake_select)
    monkeypatch.setattr(server.queue_manager, "add_item", should_not_enqueue)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", complete_prompt)

    parent_item = SimpleNamespace(
        id="asset-reroll-queue-item",
        params={
            "prompt_id": prompt_id,
            "payload": _context_payload(session_id),
            "prompt_data": {},
            "raw_body": {},
        },
    )

    try:
        result = await server.process_illustration_context_queue_item(parent_item)

        assert result == {
            "success": True,
            "session_id": session_id,
            "asset_count": 1,
            "asset_slots": [3],
        }
        session = pipeline.get_session(session_id)
        assert [item["slot"] for item in session["items"]] == [0, 3]
        assert [item["kind"] for item in session["items"]] == [
            "scene",
            "original_asset",
        ]
        assert session["images"] == [regular_bytes, new_asset_bytes]
        assert server.prompts[prompt_id]["image_bytes"] == new_asset_bytes
        assert server.prompts[prompt_id]["_serve_image_bytes_original"] is True
        assert server.prompts[prompt_id]["status"] == "completed"
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_asset_reroll_cancellation_restores_ready_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("e" * 64)
    lookup_key = "e" * 24
    prompt_id = "asset-reroll-cancelled"
    asset_bytes = b"RIFF\x00\x00\x00\x00WEBPcancel-preserved"
    old_items = [_asset_descriptor(2, "Aoi_Cancel.webp")]

    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(session_id, old_items, [asset_bytes])
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }

    monkeypatch.setattr(
        server,
        "_capture_illustration_runtime_snapshot",
        lambda: {
            "bot_name": "sample",
            "illustration_context_toggles": {},
        },
    )

    async def cancel_select(**_kwargs):
        assert pipeline.get_session(session_id)["status"] == "building"
        raise asyncio.CancelledError()

    async def ignore_progress(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "_select_original_asset_outputs", cancel_select)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    parent_item = SimpleNamespace(
        id="asset-reroll-cancelled-item",
        params={
            "prompt_id": prompt_id,
            "payload": _context_payload(session_id),
            "prompt_data": {},
            "raw_body": {},
        },
    )

    try:
        with pytest.raises(asyncio.CancelledError):
            await server.process_illustration_context_queue_item(parent_item)

        session = pipeline.get_session(session_id)
        assert session["status"] == "ready"
        assert session["items"] == old_items
        assert session["images"] == [asset_bytes]
        assert session["progress"]["phase"] == "ready"
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_asset_reroll_failure_keeps_previous_session_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("c" * 64)
    lookup_key = "c" * 24
    prompt_id = "asset-reroll-failure"
    regular_bytes = server.create_placeholder_png()
    asset_bytes = b"RIFF\x00\x00\x00\x00WEBPpreserved"
    old_items = [_regular_descriptor(0), _asset_descriptor(2, "Aoi_Keep.webp")]
    old_images = [regular_bytes, asset_bytes]

    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(session_id, old_items, old_images)
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }

    monkeypatch.setattr(
        server,
        "_capture_illustration_runtime_snapshot",
        lambda: {
            "bot_name": "sample",
            "provider": "comfy",
            "illustration_workflow_type": "v3",
            "illustration_context_toggles": {
                "original_asset_instruction": "rules",
            },
        },
    )

    async def fail_select(**_kwargs):
        raise RuntimeError("selector failed")

    async def should_not_enqueue(*_args, **_kwargs):
        raise AssertionError("실패 경로도 Comfy/일반 삽화 큐에 진입하면 안 됩니다")

    async def ignore_progress(*_args, **_kwargs):
        return None

    async def ignore_frontend(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "_select_original_asset_outputs", fail_select)
    monkeypatch.setattr(server.queue_manager, "add_item", should_not_enqueue)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "notify_frontend", ignore_frontend)

    parent_item = SimpleNamespace(
        id="asset-reroll-failure-item",
        params={
            "prompt_id": prompt_id,
            "payload": _context_payload(session_id),
            "prompt_data": {},
            "raw_body": {},
        },
    )

    try:
        with pytest.raises(RuntimeError, match="selector failed"):
            await server.process_illustration_context_queue_item(parent_item)

        session = pipeline.get_session(session_id)
        assert session["status"] == "ready"
        assert session["items"] == old_items
        assert session["images"] == old_images
        assert session["progress"]["phase"] == "error"
        assert server.prompts[prompt_id]["status"] == "completed"
        assert server.prompts[prompt_id]["outputs"] == {"images": []}
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        server.prompts.pop(prompt_id, None)
