import asyncio
import json
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
async def test_process_prompt_allows_unregistered_multi_char_as_prompt_only(monkeypatch):
    import importlib

    from modes import multi_char_mask

    bot_mode_module = importlib.import_module("modes.bot_mode")

    prompt_id = "multi-char-prompt-only-character-test"
    captured = {}
    registered_name = "Registered Hero"
    prompt_only_name = "Passing Stranger"
    bot = {
        "name": "mixed-character-bot",
        "characters": [{
            "name": registered_name,
            "gender_tag": "1girl",
            "loras_group": [{
                "source": "asset",
                "lora_path": "hero.safetensors",
                "trigger": "hero trigger",
                "BASE": "anima",
            }],
        }],
        "illust_settings_group": {},
        "illust_settings_solo": {},
    }
    bot_data = {
        "bots": [bot],
        "positive_whitelist": [],
        "positive_blacklist": [],
    }
    raw_positive = (
        "[Positive]\n"
        "[SETUP]\nwide shot, classroom\n"
        "[CHAR]\ngirl, silver hair | boy, black hair\n"
        "[SUPPLEMENT]\ntwo people standing apart\n"
        f"[NAME]\n{registered_name}, {prompt_only_name}"
    )
    layout = {
        "background_prompt": "wide shot, classroom",
        "composition_prompt": "two people standing apart",
        "regions": [{
            "name": registered_name,
            "character_prompt": "model rewrite, purple hair",
            "x": 0.0,
            "y": 0.0,
            "width": 0.5,
            "height": 1.0,
        }, {
            "name": prompt_only_name,
            "character_prompt": "model rewrite, white coat",
            "x": 0.5,
            "y": 0.0,
            "width": 0.5,
            "height": 1.0,
        }],
    }
    raw_body = {
        "illustration_provider": "comfy",
        "illustration_defer_postprocess": True,
        "illustration_multi_char": {
            "enable": True,
            "characters": [
                {
                    "name": registered_name,
                    "positive": "girl, silver hair, school uniform",
                },
                {
                    "name": prompt_only_name,
                    "positive": "boy, black hair, black jacket",
                },
            ],
            "character_order": [registered_name, prompt_only_name],
            "layout": layout,
            "mask_location": multi_char_mask.DEFAULT_MASK_LOCATION,
        },
    }
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
        "_illustration_runtime_snapshot": {
            "bot_name": bot["name"],
            "provider": "comfy",
            "illustration_workflow_type": "v3",
            "clamp_enabled": False,
            "word_rules": [],
        },
    }

    def fake_extract(_prompt, title):
        return raw_positive if title == "긍정프롬프트" else "lowres"

    async def fake_generate(positive, negative, **kwargs):
        captured["positive"] = positive
        captured["negative"] = negative
        return b"raw-image", {}

    monkeypatch.setattr(server, "extract_prompts_by_title", fake_extract)
    monkeypatch.setattr(server, "generate_image_with_prompt", fake_generate)
    monkeypatch.setattr(server, "log_to_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_mode_module, "_load_bot_data", lambda: bot_data)
    monkeypatch.setattr(bot_mode_module, "_load_lb_extra", lambda _bot_name: [])
    monkeypatch.setattr(bot_mode_module, "_load_patch_settings", lambda _bot_name: {})
    monkeypatch.setattr(server.asset_mode, "_tags", {})

    try:
        await server.process_prompt(prompt_id, {}, raw_body)

        blocks = server.llm_prompt_edit.parse_blocks(captured["positive"])
        multi_payload = json.loads(blocks["MULTI_CHAR"])
        cache_payload = json.loads(blocks["CACHE_PATH"])
        face_id_payload = json.loads(blocks["FACE_ID_DIR"])
        assert blocks["CHAR_LIST"] == f"{registered_name},{prompt_only_name}"
        assert multi_payload["enable"] is True
        assert multi_payload["char_name_list"] == [registered_name, prompt_only_name]
        assert multi_payload["char_inform"] == [
            f"{registered_name}, girl, silver hair, school uniform",
            "boy, black hair, black jacket",
        ]
        assert all(
            "model rewrite" not in value
            for value in multi_payload["char_inform"]
        )
        assert [entry["CHAR"] for entry in cache_payload["list"]] == [registered_name]
        assert [entry["CHAR"] for entry in face_id_payload["list"]] == [registered_name]
        assert raw_body["illustration_multi_char"]["character_order"] == [
            registered_name,
            prompt_only_name,
        ]
        assert server.prompts[prompt_id]["_deferred_finalize"]["word_rule_character_count"] == 2
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
async def test_context_queue_defers_multi_character_scene_until_layout_is_ready(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("c" * 64)
    original_prompt_id = "mixed-multi-original"
    pipeline.create_session(session_id, "")
    server.prompts[original_prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }

    single = {
        "kind": "scene",
        "slot": 0,
        "characters": [{"name": "Solo", "positive": "solo tags"}],
        "raw_positive": "single positive",
        "raw_negative": "single negative",
    }
    multi_preliminary = {
        "kind": "scene",
        "slot": 1,
        "characters": [
            {"name": "Left", "positive": "left tags"},
            {"name": "Right", "positive": "right tags"},
        ],
        "raw_positive": "multi preliminary",
        "raw_negative": "multi negative",
    }
    multi_final = {
        **multi_preliminary,
        "raw_positive": "multi final",
        "multi_char_layout": {
            "background_prompt": "shared clean background",
            "composition_prompt": "two distinct people, one on the left and one on the right",
            "character_order": ["Left", "Right"],
            "regions": [
                {
                    "name": "Left", "character_prompt": "left clean tags",
                    "x": 0.0, "y": 0.0, "width": 0.55, "height": 1.0,
                },
                {
                    "name": "Right", "character_prompt": "right clean tags",
                    "x": 0.45, "y": 0.0, "width": 0.55, "height": 1.0,
                },
            ],
        },
    }
    enqueue_log = []
    child_prompt_ids = []

    async def fake_build(*args, on_call2_ready=None, **kwargs):
        assert kwargs["enable_multi_char_layout"] is True
        await on_call2_ready({
            "context": "context",
            "prompt_format": "v3",
            "items": [single, multi_preliminary],
        })
        # CALL2 직후에는 단일 캐릭터만 큐에 들어간다.
        assert [entry[0] for entry in enqueue_log] == [0]
        return {
            "context": "context",
            "prompt_format": "v3",
            "items": [single, multi_final],
        }

    async def fake_add_item(item_type, label, params, priority=10, **kwargs):
        child_id = params["prompt_id"]
        child_prompt_ids.append(child_id)
        raw = params["raw_body"]
        enqueue_log.append((
            priority,
            raw.get("illustration_multi_char"),
            raw["illustration_defer_postprocess"],
        ))
        future = asyncio.get_running_loop().create_future()
        queue_item = SimpleNamespace(status="completed", completion_future=future)
        server.prompts[child_id]["image_bytes"] = f"image-priority-{priority}".encode()
        future.set_result({"success": True})
        return queue_item

    async def fake_complete_prompt(prompt_id, save_node_id, filename):
        server.prompts[prompt_id]["status"] = "completed"

    async def ignore_progress(*args, **kwargs):
        return None

    monkeypatch.setitem(server.app_config, "bot_selected", "")
    monkeypatch.setitem(server.app_config, "illustration_provider", "comfy")
    monkeypatch.setitem(server.app_config, "illustration_context_toggles", {"prompt_format": "v3"})
    monkeypatch.setattr(pipeline, "build_from_context", fake_build)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "set_prompt_by_title", lambda *args, **kwargs: True)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete_prompt)
    monkeypatch.setattr(server, "build_active_lb_extra", lambda *args: "")
    monkeypatch.setattr(server, "build_lb_extra_costume", lambda *args: "")
    monkeypatch.setattr(server, "build_lb_extra_names", lambda *args: "")
    monkeypatch.setattr(server, "build_bot_character_names", lambda *args: "")

    parent_item = SimpleNamespace(params={
        "prompt_id": original_prompt_id,
        "payload": {"session_id": session_id, "chats": []},
        "prompt_data": {},
        "raw_body": {},
    })

    try:
        result = await server.process_illustration_context_queue_item(parent_item)

        assert [entry[0] for entry in enqueue_log] == [0, 1]
        assert enqueue_log[0][1] is None
        assert enqueue_log[0][2] is True
        assert enqueue_log[1][1]["enable"] is True
        assert enqueue_log[1][1]["character_order"] == ["Left", "Right"]
        assert enqueue_log[1][1]["background_prompt"] == "shared clean background"
        assert enqueue_log[1][1]["composition_prompt"] == (
            "two distinct people, one on the left and one on the right"
        )
        assert enqueue_log[1][2] is False
        assert result["count"] == 2
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop("c" * 24, None)
        server.prompts.pop(original_prompt_id, None)
        for child_prompt_id in child_prompt_ids:
            server.prompts.pop(child_prompt_id, None)


@pytest.mark.asyncio
async def test_context_queue_enqueues_multi_character_scene_without_mask_when_disabled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("b" * 64)
    original_prompt_id = "multi-mask-disabled-original"
    pipeline.create_session(session_id, "")
    server.prompts[original_prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }
    descriptor = {
        "kind": "scene",
        "slot": 0,
        "characters": [
            {"name": "Left", "positive": "left tags"},
            {"name": "Right", "positive": "right tags"},
        ],
        "raw_positive": "ordinary multi-character prompt",
        "raw_negative": "negative",
    }
    enqueued = []
    child_prompt_ids = []

    async def fake_build(*args, on_call2_ready=None, **kwargs):
        assert kwargs["enable_multi_char_layout"] is False
        await on_call2_ready({
            "context": "context",
            "prompt_format": "v3",
            "items": [descriptor],
        })
        return {
            "context": "context",
            "prompt_format": "v3",
            "items": [descriptor],
        }

    async def fake_add_item(item_type, label, params, priority=10, **kwargs):
        child_id = params["prompt_id"]
        child_prompt_ids.append(child_id)
        raw = params["raw_body"]
        enqueued.append((priority, raw.copy()))
        future = asyncio.get_running_loop().create_future()
        queue_item = SimpleNamespace(status="completed", completion_future=future)
        server.prompts[child_id]["image_bytes"] = b"ordinary-multi-image"
        future.set_result({"success": True})
        return queue_item

    async def fake_complete_prompt(prompt_id, save_node_id, filename):
        server.prompts[prompt_id]["status"] = "completed"

    async def ignore_progress(*args, **kwargs):
        return None

    monkeypatch.setitem(server.app_config, "bot_selected", "")
    monkeypatch.setitem(server.app_config, "illustration_provider", "comfy")
    monkeypatch.setitem(
        server.app_config,
        "illustration_context_toggles",
        {"prompt_format": "v3", "multi_char_mask_enabled": False},
    )
    monkeypatch.setattr(pipeline, "build_from_context", fake_build)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "set_prompt_by_title", lambda *args, **kwargs: True)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", fake_complete_prompt)
    monkeypatch.setattr(server, "build_active_lb_extra", lambda *args: "")
    monkeypatch.setattr(server, "build_lb_extra_costume", lambda *args: "")
    monkeypatch.setattr(server, "build_lb_extra_names", lambda *args: "")
    monkeypatch.setattr(server, "build_bot_character_names", lambda *args: "")

    parent_item = SimpleNamespace(params={
        "prompt_id": original_prompt_id,
        "payload": {"session_id": session_id, "chats": []},
        "prompt_data": {},
        "raw_body": {},
    })

    try:
        result = await server.process_illustration_context_queue_item(parent_item)

        assert len(enqueued) == 1
        assert enqueued[0][0] == 0
        assert "illustration_multi_char" not in enqueued[0][1]
        assert enqueued[0][1]["illustration_defer_postprocess"] is True
        assert result["count"] == 1
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop("b" * 24, None)
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
