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


def test_similar_asset_commands_caps_at_thirty_and_keeps_diverse_matches() -> None:
    commands = [
        f"Hero_School_sad_{index:02}.webp"
        for index in range(40)
    ] + [
        "Hero_Overcome_School_sad.webp",
        "Hero_Corruption_School_sad.webp",
        "Hero_Casual_sad.webp",
        "Hero_School_crying.webp",
    ]
    asset_index = {}
    for command in commands:
        candidate = original_assets.OriginalAssetCandidate(
            command=command,
            bot_name="sample",
            character="Hero",
            filename=f"{command}.webp",
            path=str(Path("Hero") / f"{command}.webp"),
        )
        asset_index[original_assets.canonical_asset_command(command)] = [candidate]

    candidates = original_assets.similar_asset_commands(
        "Hero_School_sad.webp",
        asset_index,
    )

    assert len(candidates) == 30
    assert len(candidates) < len(commands)
    assert "Hero_Overcome_School_sad.webp" in candidates
    assert "Hero_Corruption_School_sad.webp" in candidates


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


def test_selector_prompt_requires_present_subject_and_post_evidence_slot() -> None:
    messages = original_assets.build_selection_messages(
        instruction="Aoi: Aoi_School; happy, smirk, crazy smile",
        conversation_context="Shiho appeared in the previous scene.",
        target_slotted=(
            "Aoi enters the classroom.\n\n"
            "[Slot 0]\n\n"
            "Aoi smiles.\n\n"
            "[Slot 1]\n\n"
            "She briefly remembers Shiho."
        ),
        allowed_slots=[0, 1],
        requested_count=1,
        profile_authority=(
            "### START · Aoi · Aoi_School\n"
            "- authoritative appearance: black hair\n"
            "- profile default outfit: school uniform"
        ),
    )

    system = messages[0]["content"]
    assert "only selection target is [CURRENT RESPONSE WITH INSERTION SLOTS]" in system
    assert "physically present in the active narrative scene" in system
    assert "visualized in a brief recollection" in system
    assert "Never substitute a default, habitual" in system
    assert "Do not exaggerate a smile or smirk into a crazy smile" in system
    assert "after the paragraph above it and before the paragraph below it" in system
    assert "Do not spend multiple selections on near-duplicate states" in system
    assert "Never relax presence, outfit/form, state, or non-duplication rules" in system
    assert "Return no more than the requested maximum" in system
    assert "was resolved before this stage" in system
    assert "Aoi_School" in messages[1]["content"]
    assert "profile default outfit: school uniform" in messages[1]["content"]
    assert "[REQUESTED MAXIMUM OUTPUT COUNT]" in messages[1]["content"]


def test_original_asset_context_excludes_current_target_from_recent_context() -> None:
    payload = {
        "chats": [
            {"role": "user", "data": "opening request"},
            {"role": "char", "data": "previous response with Shiho"},
            {"role": "user", "data": "continue"},
            {"role": "char", "data": "current classroom response with Aoi"},
        ],
        "target_slotted": "current classroom\n\n[Slot 0]\n\nresponse with Aoi",
    }

    context, target = server._original_asset_context(payload, context_turns=3)

    assert "opening request" in context
    assert "previous response with Shiho" in context
    assert "continue" in context
    assert "current classroom response with Aoi" not in context
    assert target == payload["target_slotted"]


@pytest.mark.asyncio
async def test_original_asset_output_keeps_underfilled_valid_selections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot_dir = tmp_path / "bot"
    valid_bytes = b"RIFFpartialWEBP"
    _write_image(
        bot_dir / "sample" / "Aoi" / "Aoi_School_happy.webp.webp",
        valid_bytes,
    )
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        server,
        "_original_asset_bot_character_names",
        lambda _bot_name: ["Aoi"],
    )

    raw = json.dumps({
        "selections": [{"src": "Aoi_School_happy.webp", "slot": 1}],
    })
    calls = []

    async def fake_pipeline_llm(call_name, _messages, **kwargs):
        calls.append(call_name)
        assert call_name == "ORIGINAL-ASSET"
        assert kwargs["result_validator"](raw) == (True, "")
        return raw

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_llm)

    result = await server._select_original_asset_outputs(
        payload={
            "chats": [{"role": "char", "data": "Aoi smiles after class."}],
            "target_slotted": "First\n\n[Slot 1]\n\nSecond\n\n[Slot 2]\n\nThird",
        },
        toggles={
            "original_asset_count": 2,
            "original_asset_instruction": "Aoi: Aoi_School; happy",
            "call2_context_turns": 5,
        },
        active_bot="sample",
        used_slots=set(),
        reserve_slot_count=0,
        stream_notify=None,
        llm_trace=[],
    )

    assert calls == ["ORIGINAL-ASSET"]
    assert len(result["items"]) == 1
    assert result["items"][0]["slot"] == 1
    assert result["images"] == [valid_bytes]
    assert result["failures"] == [{
        "slot": None,
        "error": "원본 에셋 선택 수 부족: requested=2, returned=1",
    }]


@pytest.mark.asyncio
async def test_original_asset_output_keeps_valid_items_when_one_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot_dir = tmp_path / "bot"
    valid_bytes = b"RIFFvalidWEBP"
    _write_image(
        bot_dir / "sample" / "Aoi" / "Aoi_School_happy.webp.webp",
        valid_bytes,
    )
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        server,
        "_original_asset_bot_character_names",
        lambda _bot_name: ["Aoi"],
    )

    raw = json.dumps({
        "selections": [
            {"src": "Aoi_School_happy.webp", "slot": 1},
            {"src": "Aoi_School_missing.webp", "slot": 2},
        ]
    })

    async def fake_pipeline_llm(call_name, _messages, **kwargs):
        if call_name == "ORIGINAL-ASSET":
            assert kwargs["result_validator"](raw) == (True, "")
            return raw
        assert call_name == "ORIGINAL-ASSET-RECOVERY"
        raise RuntimeError("recovery unavailable")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_llm)

    result = await server._select_original_asset_outputs(
        payload={
            "chats": [{"role": "char", "data": "First\n\nSecond\n\nThird"}],
            "target_slotted": (
                "First\n\n[Slot 1]\n\nSecond\n\n[Slot 2]\n\nThird"
            ),
        },
        toggles={
            "original_asset_count": 2,
            "original_asset_instruction": "Aoi: Aoi_School; happy, missing",
            "call2_context_turns": 5,
        },
        active_bot="sample",
        used_slots=set(),
        reserve_slot_count=0,
        stream_notify=None,
        llm_trace=[],
    )

    assert len(result["items"]) == 1
    assert result["items"][0]["slot"] == 1
    assert result["images"] == [valid_bytes]
    assert result["failures"] == [{
        "slot": 2,
        "error": "selection 2 실제 업로드 파일 없음: 'Aoi_School_missing.webp'",
    }]


@pytest.mark.asyncio
async def test_original_asset_output_recovers_missing_id_from_real_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot_dir = tmp_path / "bot"
    valid_bytes = b"RIFFvalidWEBP"
    recovered_bytes = b"RIFFrecoveredWEBP"
    _write_image(
        bot_dir / "sample" / "Aoi" / "Aoi_Casual_happy.webp.webp",
        valid_bytes,
    )
    _write_image(
        bot_dir / "sample" / "Aoi" / "Aoi_Overcome_School_sad.webp.webp",
        recovered_bytes,
    )
    _write_image(
        bot_dir / "sample" / "Aoi" / "Aoi_Corruption_School_sad.webp.webp"
    )
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        server,
        "_original_asset_bot_character_names",
        lambda _bot_name: ["Aoi"],
    )

    initial_raw = json.dumps({
        "selections": [
            {"src": "Aoi_Casual_happy.webp", "slot": 1},
            {"src": "Aoi_School_sad.webp", "slot": 2},
        ]
    })
    recovery_raw = json.dumps({
        "selections": [
            {"src": "Aoi_Overcome_School_sad.webp", "slot": 2},
        ]
    })
    calls = []

    async def fake_pipeline_llm(call_name, messages, **kwargs):
        calls.append(call_name)
        if call_name == "ORIGINAL-ASSET":
            assert kwargs["result_validator"](initial_raw) == (True, "")
            return initial_raw
        assert call_name == "ORIGINAL-ASSET-RECOVERY"
        prompt_text = "\n".join(message["content"] for message in messages)
        assert "Rejected src: Aoi_School_sad.webp" in prompt_text
        assert "Aoi_Overcome_School_sad.webp" in prompt_text
        assert "Aoi_Corruption_School_sad.webp" in prompt_text
        assert kwargs["result_validator"](recovery_raw) == (True, "")
        return recovery_raw

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_llm)

    result = await server._select_original_asset_outputs(
        payload={
            "chats": [{
                "role": "char",
                "data": "Aoi changes back into her school uniform and looks sad.",
            }],
            "target_slotted": (
                "First\n\n[Slot 1]\n\nSecond\n\n[Slot 2]\n\nThird"
            ),
        },
        toggles={
            "original_asset_count": 2,
            "original_asset_instruction": (
                "Aoi: Aoi_Casual, Aoi_Overcome_School, "
                "Aoi_Corruption_School; happy, sad"
            ),
            "call2_context_turns": 5,
        },
        active_bot="sample",
        used_slots=set(),
        reserve_slot_count=0,
        stream_notify=None,
        llm_trace=[],
    )

    assert calls == ["ORIGINAL-ASSET", "ORIGINAL-ASSET-RECOVERY"]
    assert [item["slot"] for item in result["items"]] == [1, 2]
    assert result["images"] == [valid_bytes, recovered_bytes]
    assert result["failures"] == []


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
        "illustration_output_mode": "original_asset",
        "original_asset_count": 99,
        "original_asset_instruction": "rules",
    })

    assert toggles["illustration_output_mode"] == "original_asset"
    assert toggles["illustration_enabled"] is False
    assert toggles["original_asset_enabled"] is True
    assert toggles["original_asset_count"] == 30
    assert toggles["original_asset_instruction"] == "rules"
    assert pipeline._CALL_TASK_KEYS["ORIGINAL-ASSET"] == "illustration_original_asset"
    assert (
        pipeline._CALL_TASK_KEYS["ORIGINAL-ASSET-RECOVERY"]
        == "illustration_original_asset_recovery"
    )
    assert pipeline._CALL_QUEUE_SUBTASK_GROUPS["ORIGINAL-ASSET-RECOVERY"] == (
        "original_asset_recovery",
        "원본 에셋 실패 항목 복구",
    )
    assert "illustration_original_asset" in server.DEFAULT_CONFIG["llm_routing"]
    assert server.DEFAULT_CONFIG["llm_routing"]["illustration_original_asset"][
        "json_mode"
    ] is True
    assert "illustration_original_asset_recovery" in server.DEFAULT_CONFIG["llm_routing"]
    assert server.DEFAULT_CONFIG["llm_routing"][
        "illustration_original_asset_recovery"
    ]["json_mode"] is True


def test_illustration_output_mode_derives_booleans_and_legacy_fallback() -> None:
    # 단일 모드가 두 불린으로 전개된다.
    both = pipeline.merged_toggles({"illustration_output_mode": "both"})
    assert both["illustration_enabled"] is True
    assert both["original_asset_enabled"] is True
    assert both["illustration_output_mode"] == "both"

    illustration = pipeline.merged_toggles({"illustration_output_mode": "illustration"})
    assert illustration["illustration_enabled"] is True
    assert illustration["original_asset_enabled"] is False

    # 구버전 저장값(두 불린, 모드 없음)은 조합에서 모드를 추론한다.
    legacy_both = pipeline.merged_toggles({
        "illustration_enabled": True,
        "original_asset_enabled": True,
    })
    assert legacy_both["illustration_output_mode"] == "both"

    legacy_asset_only = pipeline.merged_toggles({
        "illustration_enabled": False,
        "original_asset_enabled": True,
    })
    assert legacy_asset_only["illustration_output_mode"] == "original_asset"

    # 빈 입력은 기본 일반 삽화.
    defaults = pipeline.merged_toggles({})
    assert defaults["illustration_output_mode"] == "illustration"
    assert defaults["illustration_enabled"] is True
    assert defaults["original_asset_enabled"] is False


def test_frontend_places_original_asset_tab_after_output_count() -> None:
    frontend = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    groups_start = frontend.index("const ILLUSTRATION_CONTEXT_TOGGLE_GROUPS = [")
    groups_end = frontend.index("const ILLUSTRATION_CONTEXT_TOGGLE_FIELDS", groups_start)
    groups = frontend[groups_start:groups_end]

    assert groups.index("key: 'output_count'") < groups.index("key: 'original_asset'")
    assert "key: 'illustration_output_mode'" in groups
    assert "key: 'original_asset_count'" in groups
    assert "key: 'original_asset_instruction'" in groups
    assert "key: 'illustration_original_asset'" in frontend
    assert "key: 'illustration_original_asset_recovery'" in frontend


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
                "illustration_output_mode": "original_asset",
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
    stage_order = []
    resolved_profile_output = '{"profile_events":[]}'
    resolved_profile_result = {
        "profile_events": [],
        "initial_visual_bases": [],
        "visual_base_events": [],
        "validation_warnings": [],
        "validation_errors": [],
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
                "original_asset_count": 1,
                "original_asset_instruction": "Aoi: Aoi_School; happy",
                "scene_mode": "manual",
                "output_count_min": 1,
                "output_count_max": 1,
                "multi_char_mask_enabled": False,
            },
        },
    )

    async def fake_profile_resolve(**kwargs):
        stage_order.append("profile")
        assert kwargs["payload"]["target_slotted"].startswith("First")
        return resolved_profile_output, resolved_profile_result

    async def fake_select(**kwargs):
        stage_order.append("asset")
        assert stage_order == ["profile", "asset"]
        assert kwargs["reserve_slot_count"] == 1
        assert kwargs["profile_authority"] == "selected profile authority"
        return {
            "items": [original_descriptor],
            "images": [original_bytes],
            "failures": [],
            "requested_count": 1,
            "target_slotted": "First\n\n[Slot 0]\n\nSecond\n\n[Slot 1]\n\nThird",
        }

    async def fake_build(build_payload, *_args, **kwargs):
        stage_order.append("build")
        assert stage_order == ["profile", "asset", "build"]
        assert "[Slot 0]" not in build_payload["target_slotted"]
        assert "[Slot 1]" in build_payload["target_slotted"]
        assert kwargs["pre_resolved_profile_output"] == resolved_profile_output
        assert kwargs["pre_resolved_profile_result"] == resolved_profile_result
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

    monkeypatch.setattr(pipeline, "resolve_profiles_before_generation", fake_profile_resolve)
    monkeypatch.setattr(
        pipeline,
        "profile_authority_text",
        lambda *_args: "selected profile authority",
    )
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
        assert stage_order == ["profile", "asset", "build"]
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        server.prompts.pop(prompt_id, None)
        for child_id in child_ids:
            server.prompts.pop(child_id, None)
