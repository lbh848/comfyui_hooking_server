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
async def test_process_prompt_defers_publication_and_backup_for_call3(monkeypatch):
    prompt_id = "deferred-postprocess-test"
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }

    async def generate_raw(*args, **kwargs):
        return b"raw-image", {}

    async def fail_if_backup_runs(*args, **kwargs):
        raise AssertionError("CALL3 합류 전에는 백업/후처리를 실행하면 안 됩니다")

    monkeypatch.setitem(server.app_config, "bot_selected", "")
    monkeypatch.setattr(server, "generate_image_with_prompt", generate_raw)
    monkeypatch.setattr(server, "save_backup", fail_if_backup_runs)

    try:
        await server.process_prompt(
            prompt_id,
            {},
            {
                "illustration_defer_postprocess": True,
                "illustration_provider": "comfy",
            },
        )

        entry = server.prompts[prompt_id]
        assert entry["status"] == "running"
        assert entry["image_bytes"] is None
        assert entry["_deferred_image_bytes"] == b"raw-image"
        assert entry["_deferred_finalize"]["provider"] == "comfy"
        assert entry["_deferred_finalize"]["positive"] == ""
    finally:
        server.prompts.pop(prompt_id, None)


def test_deferred_speak_uses_same_raw_name_replacements(monkeypatch):
    prompt_id = "deferred-speak-word-rule-test"
    server.prompts[prompt_id] = {
        "_deferred_finalize": {"bot_name": "word-rule-bot"},
    }
    descriptor = {
        "speak": 'Alias: "hello"',
        "raw_positive": '[SPEAK]\nAlias: "hello"\n[NAME]\nAlias',
    }

    monkeypatch.setattr(
        server,
        "apply_raw_prompt_word_replacements",
        lambda raw, bot_name, rules=None: raw.replace("Alias", "Canonical"),
    )

    try:
        assert server._resolve_deferred_speak_text(prompt_id, descriptor) == (
            'Canonical: "hello"'
        )
    finally:
        server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_process_prompt_uses_queued_runtime_snapshot(monkeypatch):
    prompt_id = "runtime-snapshot-test"
    captured = {}
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
        "_illustration_runtime_snapshot": {
            "bot_name": "",
            "provider": "comfy",
            "chansub_workflow_type": "anima",
            "clamp_enabled": True,
            "clamp_value": 1.2,
            "illustration_context_toggles": {
                "call1_backtranslate_max_concurrency": 3,
            },
            "word_rules": [],
        },
    }

    def fake_extract(_prompt, title):
        return "(lighting:2.0)" if title == "긍정프롬프트" else ""

    async def fake_generate(positive, negative, **kwargs):
        captured["positive"] = positive
        captured["provider"] = kwargs.get("provider")
        return b"raw-image", {}

    monkeypatch.setitem(server.app_config, "clamp_enabled", False)
    monkeypatch.setattr(server, "extract_prompts_by_title", fake_extract)
    monkeypatch.setattr(server, "generate_image_with_prompt", fake_generate)

    try:
        await server.process_prompt(
            prompt_id,
            {},
            {"illustration_defer_postprocess": True},
        )

        assert captured == {
            "positive": "(lighting:1.2)",
            "provider": "comfy",
        }
        assert "_illustration_runtime_snapshot" not in server.prompts[prompt_id]
        assert server.prompts[prompt_id]["_deferred_finalize"]["word_rules"] == []
    finally:
        server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_finalize_deferred_prompt_publishes_only_postprocessed_image(monkeypatch):
    prompt_id = "deferred-finalize-test"
    workflow_snapshot = {"nodes": [{"id": 1}]}
    server.prompts[prompt_id] = {
        "status": "running",
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
        "_deferred_image_bytes": b"raw-image",
        "_deferred_finalize": {
            "positive": "positive",
            "negative": "negative",
            "generation_time": 3.5,
            "bot_name": "test-bot",
            "gen_method": "",
            "provider": "comfy",
            "generation_params": None,
            "original_workflow": workflow_snapshot,
            "api_workflow": {"1": {"class_type": "Test"}},
            "conversion_info": {"source": "snapshot"},
        },
    }
    captured = {}

    async def fake_save_backup(image_bytes, *args, **kwargs):
        captured["input"] = image_bytes
        captured.update(kwargs)
        return "backup-name", b"postprocessed-image"

    monkeypatch.setattr(server, "save_backup", fake_save_backup)
    monkeypatch.setattr(
        server,
        "_get_illustration_postprocess_settings",
        lambda bot_name: {"placement": "bottom"},
    )

    try:
        result = await server._finalize_deferred_illustration_prompt(
            prompt_id,
            'Hero: "done"',
        )

        entry = server.prompts[prompt_id]
        assert result == b"postprocessed-image"
        assert entry["status"] == "completed"
        assert entry["image_bytes"] == b"postprocessed-image"
        assert "_deferred_image_bytes" not in entry
        assert "_deferred_finalize" not in entry
        assert captured["input"] == b"raw-image"
        assert captured["speak_text"] == 'Hero: "done"'
        assert captured["original_workflow_snapshot"] == workflow_snapshot
        assert captured["conversion_info_snapshot"] == {"source": "snapshot"}
    finally:
        server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_context_queue_keeps_generating_until_call3_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("e" * 64)
    original_prompt_id = "parallel-call3-original"
    pipeline.create_session(session_id, "")
    server.prompts[original_prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }

    lifecycle = []
    child_prompt_ids = []
    generated_count = 0
    both_generated = asyncio.Event()

    preliminary = [
        {
            "kind": "scene",
            "slot": slot,
            "raw_positive": f"raw positive {slot}",
            "raw_negative": f"raw negative {slot}",
        }
        for slot in (0, 1)
    ]
    final_items = [
        {
            **descriptor,
            "speak": f'Hero: "line {descriptor["slot"]}"',
            "raw_positive": (
                f'[SPEAK]\nHero: "line {descriptor["slot"]}"\n'
                f'{descriptor["raw_positive"]}'
            ),
        }
        for descriptor in preliminary
    ]

    async def fake_build(*args, on_call2_ready=None, **kwargs):
        assert on_call2_ready is not None
        assert args[1]["call1_backtranslate_max_concurrency"] == 3
        await on_call2_ready({
            "session_id": session_id,
            "context": "private context",
            "prompt_format": "v3",
            "items": preliminary,
        })
        lifecycle.append("call3_running")
        await asyncio.wait_for(both_generated.wait(), timeout=1)
        in_flight_session = pipeline.get_session(session_id)
        assert in_flight_session["status"] == "building"
        assert in_flight_session["images"] == []
        assert pipeline.session_manifest(session_id).startswith("STATUS|building\nCOUNT|0")
        assert all(
            server.prompts[child_id]["image_bytes"] is None
            for child_id in child_prompt_ids
        )
        lifecycle.append("call3_returned")
        return {
            "session_id": session_id,
            "context": "private context",
            "prompt_format": "v3",
            "items": final_items,
        }

    async def fake_add_item(item_type, label, params, priority=10, **kwargs):
        nonlocal generated_count
        assert item_type == "illustration"
        assert params["raw_body"]["illustration_defer_postprocess"] is True
        assert params["provider"] == "comfy"
        child_prompt_id = params["prompt_id"]
        assert server.prompts[child_prompt_id]["_illustration_runtime_snapshot"][
            "illustration_context_toggles"
        ]["call1_backtranslate_max_concurrency"] == 3
        child_prompt_ids.append(child_prompt_id)
        future = asyncio.get_running_loop().create_future()
        queue_item = SimpleNamespace(
            id=f"queue-{len(child_prompt_ids)}",
            status="processing",
            completion_future=future,
        )
        lifecycle.append(f"enqueue-{len(child_prompt_ids)}")

        async def finish_generation():
            nonlocal generated_count
            await asyncio.sleep(0)
            generated_count += 1
            server.prompts[child_prompt_id]["_deferred_image_bytes"] = (
                f"raw-{generated_count}".encode()
            )
            server.prompts[child_prompt_id]["_deferred_finalize"] = {}
            queue_item.status = "completed"
            lifecycle.append(f"generated-{generated_count}")
            future.set_result({"success": True})
            if generated_count == 2:
                both_generated.set()

        asyncio.create_task(finish_generation())
        return queue_item

    async def fake_finalize(child_prompt_id, speak_text):
        entry = server.prompts[child_prompt_id]
        raw_image = entry.pop("_deferred_image_bytes")
        entry.pop("_deferred_finalize")
        final_image = raw_image + b"|" + speak_text.encode("utf-8")
        entry["status"] = "completed"
        entry["image_bytes"] = final_image
        lifecycle.append(f"finalized-{child_prompt_id}")
        return final_image

    async def fake_complete_prompt(prompt_id, save_node_id, filename):
        server.prompts[prompt_id]["status"] = "completed"

    async def ignore_progress(*args, **kwargs):
        # 첫 progress await 중 전역 설정이 바뀌어도 이미 캡처한 CALL1 병렬값을 쓴다.
        server.app_config["illustration_context_toggles"][
            "call1_backtranslate_max_concurrency"
        ] = 1
        return None

    monkeypatch.setitem(server.app_config, "bot_selected", "")
    monkeypatch.setitem(server.app_config, "illustration_provider", "comfy")
    monkeypatch.setitem(
        server.app_config,
        "illustration_context_toggles",
        {"call1_backtranslate_max_concurrency": 3},
    )
    monkeypatch.setattr(pipeline, "build_from_context", fake_build)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "set_prompt_by_title", lambda *args, **kwargs: True)
    monkeypatch.setattr(server, "_finalize_deferred_illustration_prompt", fake_finalize)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete_prompt)
    monkeypatch.setattr(server, "build_active_lb_extra", lambda *args: "")
    monkeypatch.setattr(server, "build_lb_extra_costume", lambda *args: "")
    monkeypatch.setattr(server, "build_lb_extra_names", lambda *args: "")
    monkeypatch.setattr(server, "build_bot_character_names", lambda *args: "")

    parent_item = SimpleNamespace(
        params={
            "prompt_id": original_prompt_id,
            "payload": {"session_id": session_id, "chats": []},
            "prompt_data": {},
            "raw_body": {},
        }
    )

    try:
        result = await server.process_illustration_context_queue_item(parent_item)

        assert lifecycle.index("generated-2") < lifecycle.index("call3_returned")
        assert lifecycle.index("call3_returned") < lifecycle.index(
            f"finalized-{child_prompt_ids[0]}"
        )
        assert result["count"] == 2
        session = pipeline.get_session(session_id)
        assert len(session["images"]) == 2
        assert b'Hero: "line 0"' in session["images"][0]
        assert b'Hero: "line 1"' in session["images"][1]
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop("e" * 24, None)
        server.prompts.pop(original_prompt_id, None)
        for child_prompt_id in child_prompt_ids:
            server.prompts.pop(child_prompt_id, None)


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
