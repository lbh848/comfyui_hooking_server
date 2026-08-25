import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline
from modes import illustration_original_assets as original_assets


def _write_image(path: Path, data: bytes = b"RIFFtestWEBP") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_original_asset_index_uses_direct_character_files_and_logical_webp_id(
    tmp_path: Path,
) -> None:
    bot_dir = tmp_path / "bot"
    _write_image(bot_dir / "sample" / "Yuu" / "Yuu_Casual_happy.webp.webp")
    _write_image(bot_dir / "sample" / "Yuu" / "_face_image.webp")
    _write_image(
        bot_dir / "sample" / "Yuu" / "_visual_profiles" / "card_1" / "derived.webp"
    )
    _write_image(bot_dir / "sample" / "Lora" / "duplicate.webp")

    index = original_assets.build_original_asset_index(
        str(bot_dir),
        "sample",
        ["Yuu"],
    )

    key = original_assets.canonical_asset_command("Yuu_Casual_happy.webp")
    assert list(index) == [key]
    assert index[key][0].filename == "Yuu_Casual_happy.webp.webp"
    assert index[key][0].command == "Yuu_Casual_happy.webp"


@pytest.mark.asyncio
async def test_one_step_selector_does_not_send_uploaded_filename_index_to_llm(
    tmp_path: Path,
) -> None:
    bot_dir = tmp_path / "bot"
    _write_image(bot_dir / "sample" / "Aoi" / "Aoi_School_happy.webp.webp")
    _write_image(bot_dir / "sample" / "Aoi" / "SHOULD_NOT_ENTER_PROMPT.webp")
    index = original_assets.build_original_asset_index(
        str(bot_dir),
        "sample",
        ["Aoi"],
    )
    calls = []

    async def fake_llm(messages, validator):
        calls.append(messages)
        raw = json.dumps({
            "selections": [{"src": "Aoi_School_happy.webp", "slot": 2}]
        })
        assert validator(raw) == (True, "")
        return raw

    selected = await original_assets.select_original_assets(
        call_llm=fake_llm,
        instruction=(
            "Aoi: Aoi_School\n"
            "Normal emotions: happy\n"
            'Use <img src="<Character>_<State>.webp">.'
        ),
        conversation_context="Aoi smiles after class.",
        target_slotted="First paragraph.\n\n[Slot 2]\n\nSecond paragraph.",
        allowed_slots=[2],
        requested_count=1,
        asset_index=index,
    )

    assert len(calls) == 1
    prompt_text = "\n".join(message["content"] for message in calls[0])
    assert "SHOULD_NOT_ENTER_PROMPT.webp" not in prompt_text
    assert selected[0]["src"] == "Aoi_School_happy.webp"
    assert selected[0]["candidate"].filename == "Aoi_School_happy.webp.webp"


def test_original_asset_session_reloads_source_bytes_after_metadata_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "sessions"
    bot_dir = tmp_path / "bot"
    image_bytes = b"RIFForiginalWEBP"
    _write_image(bot_dir / "sample" / "Aoi" / "Aoi_School_happy.webp.webp", image_bytes)
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(session_dir))
    monkeypatch.setattr(pipeline, "BASE_DIR", str(tmp_path))
    session_id = "risu_" + ("7" * 64)
    lookup_key = "7" * 24
    descriptor = {
        "kind": "original_asset",
        "slot": 0,
        "original_asset": {
            "bot_name": "sample",
            "character": "Aoi",
            "filename": "Aoi_School_happy.webp.webp",
            "command": "Aoi_School_happy.webp",
        },
    }
    try:
        pipeline.create_session(session_id, "")
        pipeline.set_session_result(session_id, [descriptor], [image_bytes])
        pipeline._SESSIONS.pop(session_id)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)

        assert pipeline.session_image_by_slot(session_id, 0) == image_bytes
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)


def test_original_asset_settings_and_routing_are_registered() -> None:
    toggles = pipeline.merged_toggles({
        "illustration_enabled": False,
        "original_asset_enabled": True,
        "original_asset_count": 99,
        "original_asset_instruction": "rules",
    })

    assert toggles["illustration_enabled"] is False
    assert toggles["original_asset_enabled"] is True
    assert toggles["original_asset_count"] == 30
    assert toggles["original_asset_instruction"] == "rules"
    assert pipeline._CALL_TASK_KEYS["ORIGINAL-ASSET"] == "illustration_original_asset"
    assert "illustration_original_asset" in server.DEFAULT_CONFIG["llm_routing"]
    assert server.DEFAULT_CONFIG["llm_routing"]["illustration_original_asset"][
        "json_mode"
    ] is True


def test_frontend_places_original_asset_tab_after_output_count() -> None:
    frontend = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    groups_start = frontend.index("const ILLUSTRATION_CONTEXT_TOGGLE_GROUPS = [")
    groups_end = frontend.index("const ILLUSTRATION_CONTEXT_TOGGLE_FIELDS", groups_start)
    groups = frontend[groups_start:groups_end]

    assert groups.index("key: 'output_count'") < groups.index("key: 'original_asset'")
    assert "key: 'illustration_enabled'" in groups
    assert "key: 'original_asset_count'" in groups
    assert "key: 'original_asset_instruction'" in groups
    assert "key: 'illustration_original_asset'" in frontend


@pytest.mark.asyncio
async def test_asset_only_queue_skips_regular_pipeline_and_comfy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("8" * 64)
    lookup_key = "8" * 24
    prompt_id = "original-asset-only-prompt"
    pipeline.create_session(session_id, "")
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }
    image_bytes = b"RIFF\x00\x00\x00\x00WEBPasset-only"
    descriptor = {
        "kind": "original_asset",
        "slot": 0,
        "anchor_before": "Before",
        "anchor_after": "After",
        "anchor_version": 1,
        "original_asset": {
            "bot_name": "sample",
            "character": "Aoi",
            "filename": "Aoi_School_happy.webp.webp",
            "command": "Aoi_School_happy.webp",
        },
    }

    monkeypatch.setattr(
        server,
        "_capture_illustration_runtime_snapshot",
        lambda: {
            "bot_name": "sample",
            "provider": "comfy",
            "illustration_workflow_type": "v3",
            "illustration_context_toggles": {
                "illustration_enabled": False,
                "original_asset_enabled": True,
                "original_asset_count": 1,
                "original_asset_instruction": "Aoi: Aoi_School; happy",
            },
        },
    )

    async def fake_select(**_kwargs):
        return {
            "items": [descriptor],
            "images": [image_bytes],
            "failures": [],
            "requested_count": 1,
        }

    async def should_not_build(*_args, **_kwargs):
        raise AssertionError("일반 삽화 파이프라인이 호출되면 안 됩니다")

    async def should_not_enqueue(*_args, **_kwargs):
        raise AssertionError("Comfy 이미지 큐가 호출되면 안 됩니다")

    async def ignore_progress(*_args, **_kwargs):
        return None

    async def complete_prompt(completed_prompt_id, _save_node_id, _filename):
        server.prompts[completed_prompt_id]["status"] = "completed"

    monkeypatch.setattr(server, "_select_original_asset_outputs", fake_select)
    monkeypatch.setattr(pipeline, "build_from_context", should_not_build)
    monkeypatch.setattr(server.queue_manager, "add_item", should_not_enqueue)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", complete_prompt)
    monkeypatch.setattr(server.illustration_chat_history, "prepare_history", lambda *_args: None)
    monkeypatch.setattr(server, "build_active_lb_instruction", lambda *_args: "")
    monkeypatch.setattr(server, "build_lb_extra_costume", lambda *_args: "")
    monkeypatch.setattr(server, "build_lb_extra_names", lambda *_args: "")
    monkeypatch.setattr(server, "build_bot_character_names", lambda *_args: "")
    monkeypatch.setattr(server, "build_visual_profile_catalog", lambda *_args: "")
    monkeypatch.setattr(server, "build_effective_visual_profiles", lambda *_args: {})

    parent_item = SimpleNamespace(params={
        "prompt_id": prompt_id,
        "payload": {
            "session_id": session_id,
            "target_slotted": "Before\n\n[Slot 0]\n\nAfter",
            "chats": [
                {"role": "user", "data": "Hello"},
                {"role": "char", "data": "Before\n\nAfter"},
            ],
        },
        "prompt_data": {},
        "raw_body": {},
    })

    try:
        result = await server.process_illustration_context_queue_item(parent_item)

        assert result == {
            "success": True,
            "session_id": session_id,
            "count": 1,
            "requested_count": 1,
            "failed_count": 0,
        }
        session = pipeline.get_session(session_id)
        assert session["items"][0]["kind"] == "original_asset"
        assert session["images"] == [image_bytes]
        assert server.prompts[prompt_id]["status"] == "completed"
        view_response = await server.handle_view(SimpleNamespace(query={
            "filename": server.prompts[prompt_id]["filename"],
        }))
        assert view_response.body == image_bytes
        assert view_response.content_type == "image/webp"
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        server.prompts.pop(prompt_id, None)


@pytest.mark.asyncio
async def test_mixed_output_reserves_original_asset_slot_before_regular_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("9" * 64)
    lookup_key = "9" * 24
    prompt_id = "mixed-original-asset-prompt"
    pipeline.create_session(session_id, "")
    server.prompts[prompt_id] = {
        "status": "running",
        "prompt": {},
        "outputs": {},
        "filename": None,
        "save_node_id": "9",
        "image_bytes": None,
    }
    original_bytes = b"RIFF\x00\x00\x00\x00WEBPoriginal"
    generated_bytes = server.create_placeholder_png()
    original_descriptor = {
        "kind": "original_asset",
        "slot": 0,
        "anchor_before": "First",
        "anchor_after": "Second",
        "anchor_version": 1,
        "original_asset": {
            "bot_name": "sample",
            "character": "Aoi",
            "filename": "Aoi_School_happy.webp.webp",
            "command": "Aoi_School_happy.webp",
        },
    }
    generated_descriptor = {
        "kind": "scene",
        "slot": 1,
        "characters": [],
        "raw_positive": "scene",
        "raw_negative": "",
    }
    child_ids = []

    monkeypatch.setattr(
        server,
        "_capture_illustration_runtime_snapshot",
        lambda: {
            "bot_name": "sample",
            "provider": "comfy",
            "illustration_workflow_type": "v3",
            "illustration_context_toggles": {
                "illustration_enabled": True,
                "original_asset_enabled": True,
                "original_asset_count": 1,
                "original_asset_instruction": "Aoi: Aoi_School; happy",
                "scene_mode": "manual",
                "output_count_min": 1,
                "output_count_max": 1,
                "multi_char_mask_enabled": False,
            },
        },
    )

    async def fake_select(**kwargs):
        assert kwargs["reserve_slot_count"] == 1
        return {
            "items": [original_descriptor],
            "images": [original_bytes],
            "failures": [],
            "requested_count": 1,
            "target_slotted": "First\n\n[Slot 0]\n\nSecond\n\n[Slot 1]\n\nThird",
        }

    async def fake_build(build_payload, *_args, **kwargs):
        assert "[Slot 0]" not in build_payload["target_slotted"]
        assert "[Slot 1]" in build_payload["target_slotted"]
        await kwargs["on_call2_ready"]({
            "context": "context",
            "prompt_format": "v3",
            "items": [generated_descriptor],
        })
        return {
            "context": "context",
            "prompt_format": "v3",
            "items": [generated_descriptor],
            "llm_trace": [],
        }

    async def fake_add_item(_item_type, _label, params, **_kwargs):
        child_id = params["prompt_id"]
        child_ids.append(child_id)
        future = asyncio.get_running_loop().create_future()
        server.prompts[child_id]["image_bytes"] = generated_bytes
        future.set_result({"success": True})
        return SimpleNamespace(status="completed", completion_future=future)

    async def ignore_progress(*_args, **_kwargs):
        return None

    async def complete_prompt(completed_prompt_id, _save_node_id, _filename):
        server.prompts[completed_prompt_id]["status"] = "completed"

    monkeypatch.setattr(server, "_select_original_asset_outputs", fake_select)
    monkeypatch.setattr(pipeline, "build_from_context", fake_build)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server.queue_manager, "_notify_progress", ignore_progress)
    monkeypatch.setattr(server, "complete_prompt_from_reschedule", complete_prompt)
    monkeypatch.setattr(server, "set_prompt_by_title", lambda *_args: True)
    monkeypatch.setattr(server, "build_active_lb_instruction", lambda *_args: "")
    monkeypatch.setattr(server, "build_lb_extra_costume", lambda *_args: "")
    monkeypatch.setattr(server, "build_lb_extra_names", lambda *_args: "")
    monkeypatch.setattr(server, "build_bot_character_names", lambda *_args: "")
    monkeypatch.setattr(server, "build_visual_profile_catalog", lambda *_args: "")
    monkeypatch.setattr(server, "build_effective_visual_profiles", lambda *_args: {})

    parent_item = SimpleNamespace(params={
        "prompt_id": prompt_id,
        "payload": {
            "session_id": session_id,
            "target_slotted": "First\n\n[Slot 0]\n\nSecond\n\n[Slot 1]\n\nThird",
            "chats": [],
        },
        "prompt_data": {},
        "raw_body": {},
    })

    try:
        result = await server.process_illustration_context_queue_item(parent_item)

        assert result["count"] == 2
        assert result["requested_count"] == 2
        session = pipeline.get_session(session_id)
        assert [item["slot"] for item in session["items"]] == [0, 1]
        assert [item["kind"] for item in session["items"]] == [
            "original_asset",
            "scene",
        ]
        assert session["images"] == [original_bytes, generated_bytes]
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        server.prompts.pop(prompt_id, None)
        for child_id in child_ids:
            server.prompts.pop(child_id, None)
