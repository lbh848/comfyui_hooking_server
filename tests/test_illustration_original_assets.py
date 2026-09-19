import asyncio
import json
import sys
from copy import deepcopy
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


class _ToggleRequest:
    def __init__(self, method: str, body: dict | None = None):
        self.method = method
        self._body = body or {}

    async def json(self):
        return deepcopy(self._body)


def test_character_preprocess_reuses_one_lb_extra_collection_per_request(monkeypatch):
    calls = []
    collected = {
        "system_prompt": "shared illustration instruction",
        "system_prompt_preset": server.FIRST_PASS_SINGLE_V5_PRESET,
        "characters": [{
            "name": "Aria",
            "appearance": "silver hair",
            "outfit": "blue coat",
        }],
        "bot_character_names": ["Aria"],
        "visual_profiles": {"Aria": {"default_profile_id": "default"}},
        "visual_profile_catalog": "Aria has one default visual profile.",
    }

    def fake_collect(bot_name):
        calls.append(bot_name)
        return collected

    monkeypatch.setattr(server, "_collect_lb_extra_impl", fake_collect)
    request_cache = {}

    assert server.build_active_lb_instruction("bot-a", "trace-a", request_cache) == (
        "shared illustration instruction"
    )
    assert "### Aria" in server.build_lb_extra_costume(
        "bot-a", "trace-a", request_cache
    )
    assert server.build_lb_extra_names("bot-a", "trace-a", request_cache) == "Aria"
    assert server.build_bot_character_names("bot-a", "trace-a", request_cache) == "Aria"
    assert server.build_visual_profile_catalog(
        "bot-a", "trace-a", request_cache
    ) == "Aria has one default visual profile."
    assert server.build_effective_visual_profiles(
        "bot-a", "trace-a", request_cache
    ) == collected["visual_profiles"]
    assert server.active_bot_uses_first_pass_single_v5(
        "bot-a", "trace-a", request_cache
    ) is True
    assert calls == ["bot-a"]


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
    assert "Never substitute a habitual, more familiar" in system
    assert "Do not exaggerate a smile or smirk into a crazy smile" in system
    assert "after the paragraph above it and before the paragraph below it" in system
    assert "Do not spend multiple selections on near-duplicate states" in system
    assert "Never relax presence, outfit/form, state, or non-duplication rules" in system
    assert "Return no more than the requested maximum" in system
    assert "was resolved before this stage" in system
    assert "Aoi_School" in messages[1]["content"]
    assert "profile default outfit: school uniform" in messages[1]["content"]
    assert "[REQUESTED MAXIMUM OUTPUT COUNT]" in messages[1]["content"]


def test_selector_prompt_keeps_profile_identifier_separate_from_scene_outfit() -> None:
    messages = original_assets.build_selection_messages(
        instruction=(
            "Hoshino Yui: Hoshino_Normal_Casual, Hoshino_Normal_School\n"
            "Normal emotions: smile"
        ),
        conversation_context="Hoshino remains in the same after-school scene.",
        target_slotted=(
            "Hoshino walks out of the school gate in her school uniform.\n\n"
            "[Slot 12]\n\n"
            "She looks back and smiles."
        ),
        allowed_slots=[12],
        requested_count=1,
        profile_authority=(
            "### START · Hoshino · Hoshino_Normal_Casual\n"
            "- profile state: ordinary untransformed state\n"
            "- authoritative appearance: black hair, yellow eyes\n"
            "- profile default outfit: school uniform"
        ),
    )

    system = messages[0]["content"]
    prompt_text = "\n".join(message["content"] for message in messages)
    assert "profile name is opaque internal metadata" in system
    assert "not wardrobe evidence" in system
    assert "must never be copied or lexically matched to a command ID" in system
    assert "semantic form/state and the subject's current wardrobe as separate decisions" in system
    assert "Selecting that wardrobe variant does not constitute changing" in system
    assert "profile-default outfit, only when all narrative sources above" in system
    assert "Hoshino_Normal_Casual" in prompt_text
    assert "Hoshino_Normal_School" in prompt_text
    assert "school uniform" in prompt_text


def test_recovery_prompt_keeps_profile_identifier_separate_from_scene_outfit() -> None:
    messages = original_assets.build_recovery_messages(
        instruction="Hoshino: Hoshino_Normal_Casual, Hoshino_Normal_School",
        conversation_context="The after-school scene continues without a wardrobe change.",
        target_slotted=(
            "Hoshino is still wearing her school uniform.\n\n"
            "[Slot 12]\n\n"
            "She smiles."
        ),
        recovery_items=[{
            "slot": 12,
            "rejected_src": "Hoshino_Normal_Casual_smile.webp",
            "error": "uploaded file missing",
            "candidates": [
                "Hoshino_Normal_Casual_smile.webp",
                "Hoshino_Normal_School_smile.webp",
            ],
        }],
        profile_authority=(
            "### START · Hoshino · Hoshino_Normal_Casual\n"
            "- profile state: ordinary untransformed state\n"
            "- profile default outfit: school uniform"
        ),
    )

    system = messages[0]["content"]
    assert "profile name is opaque internal metadata" in system
    assert "Resolve semantic form/state and current wardrobe separately" in system
    assert "choosing the variant matching the narrative wardrobe does not change" in system
    assert "never keyword or string similarity" in system


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


def test_original_asset_context_fallback_keeps_module_filtered_xml() -> None:
    current = (
        '<output target=input priority=extra-high source=ai sender=ai recipient=user>\n'
        "Aoi looks toward the classroom door.\n\n"
        "Aoi raises one hand.\n"
        "</output>"
    )

    context, target = server._original_asset_context(
        {"chats": [{"role": "char", "data": current}]},
        context_turns=3,
    )

    assert context == ""
    assert "<output target=input" in target
    assert "Aoi looks toward the classroom door." in target
    assert "</output>" in target
    assert "[Slot 0]" in target


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


@pytest.mark.asyncio
async def test_original_asset_recovery_is_consumed_once_for_duplicate_failed_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot_dir = tmp_path / "bot"
    recovered_bytes = b"RIFFrecovered-onceWEBP"
    _write_image(
        bot_dir / "sample" / "Aoi" / "Aoi_Overcome_School_sad.webp.webp",
        recovered_bytes,
    )
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        server,
        "_original_asset_bot_character_names",
        lambda _bot_name: ["Aoi"],
    )

    initial_raw = json.dumps({
        "selections": [
            {"src": "Aoi_School_missing_one.webp", "slot": 1},
            {"src": "Aoi_School_missing_two.webp", "slot": 1},
        ]
    })
    recovery_raw = json.dumps({
        "selections": [
            {"src": "Aoi_Overcome_School_sad.webp", "slot": 1},
        ]
    })
    calls = []

    async def fake_pipeline_llm(call_name, _messages, **kwargs):
        calls.append(call_name)
        raw = initial_raw if call_name == "ORIGINAL-ASSET" else recovery_raw
        assert kwargs["result_validator"](raw) == (True, "")
        return raw

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_llm)

    result = await server._select_original_asset_outputs(
        payload={
            "chats": [{"role": "char", "data": "Aoi looks sad after class."}],
            "target_slotted": (
                "First\n\n[Slot 1]\n\nSecond\n\n[Slot 2]\n\nThird"
            ),
        },
        toggles={
            "original_asset_count": 2,
            "original_asset_instruction": "Aoi: Aoi_School; sad",
            "call2_context_turns": 5,
        },
        active_bot="sample",
        used_slots=set(),
        reserve_slot_count=0,
        stream_notify=None,
        llm_trace=[],
    )

    assert calls == ["ORIGINAL-ASSET", "ORIGINAL-ASSET-RECOVERY"]
    assert [item["slot"] for item in result["items"]] == [1]
    assert result["images"] == [recovered_bytes]
    assert result["failures"] == [{
        "slot": 1,
        "error": (
            "selection 2 실제 업로드 파일 없음: "
            "'Aoi_School_missing_two.webp'"
        ),
    }]


def test_final_result_slot_collision_prefers_original_asset_and_keeps_session_alive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pairs = [
        ({"kind": "scene", "slot": 35}, b"generated"),
        ({"kind": "original_asset", "slot": 35}, b"original"),
        ({"kind": "scene", "slot": 36}, b"next"),
    ]

    unique_pairs, failures = server._deduplicate_illustration_result_pairs(
        pairs,
        session_id="test-session",
    )

    assert [pair[0]["slot"] for pair in unique_pairs] == [35, 36]
    assert [pair[0]["kind"] for pair in unique_pairs] == ["original_asset", "scene"]
    assert [pair[1] for pair in unique_pairs] == [b"original", b"next"]
    assert failures == [{
        "slot": 35,
        "error": (
            "중복 삽입 슬롯 결과 제외: "
            "slot=35, kept=original_asset, dropped=scene"
        ),
    }]
    assert "최종 결과 중복 slot 한 건 제외" in capsys.readouterr().out


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
    assert (
        "original_asset_instruction"
        not in server.DEFAULT_CONFIG["illustration_context_toggles"]
    )
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


def test_runtime_snapshot_uses_bot_instruction_and_ignores_legacy_global_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(server.DEFAULT_CONFIG)
    config["bot_selected"] = "sample"
    config["illustration_context_toggles"] = {
        "illustration_output_mode": "original_asset",
        "original_asset_instruction": "legacy global rules",
    }
    monkeypatch.setattr(server, "_load_word_rules_snapshot", lambda _bot_name: [])
    monkeypatch.setattr(
        server.bot_mode,
        "get_asset_output_instruction",
        lambda bot_name: "saved bot-only rules" if bot_name == "sample" else "",
    )

    snapshot = server._capture_illustration_runtime_snapshot(config)

    assert snapshot["illustration_context_toggles"][
        "original_asset_instruction"
    ] == "saved bot-only rules"


@pytest.mark.asyncio
async def test_global_toggle_api_drops_legacy_asset_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(server.DEFAULT_CONFIG)
    config["illustration_context_toggles"] = {
        "illustration_output_mode": "both",
        "original_asset_instruction": "legacy global rules",
    }
    saved_configs = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(
        server,
        "save_config",
        lambda value: saved_configs.append(deepcopy(value)),
    )

    get_response = await server.handle_api_illustration_context_toggles(
        _ToggleRequest("GET")
    )
    get_payload = json.loads(get_response.text)
    assert get_response.status == 200
    assert "original_asset_instruction" not in get_payload["toggles"]

    post_response = await server.handle_api_illustration_context_toggles(
        _ToggleRequest("POST", {
            "toggles": {
                "illustration_output_mode": "both",
                "original_asset_instruction": "attempted legacy overwrite",
            }
        })
    )
    post_payload = json.loads(post_response.text)
    assert post_response.status == 200
    assert "original_asset_instruction" not in post_payload["toggles"]
    assert (
        "original_asset_instruction"
        not in saved_configs[-1]["illustration_context_toggles"]
    )


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
    assert "key: 'original_asset_instruction'" not in groups
    assert "에셋 출력 지침 세팅" in groups
    assert "기존 전역 에셋 선택 지침은 사용하지 않습니다" in groups
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

    async def fake_select(**kwargs):
        assert kwargs["used_slots"] == set()
        assert kwargs["reserve_slot_count"] == 0
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
@pytest.mark.parametrize("multi_character", [False, True])
async def test_mixed_output_starts_original_asset_after_detail_gpu_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multi_character: bool,
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
        "characters": (
            [{"name": "Aoi"}, {"name": "Ren"}]
            if multi_character
            else []
        ),
        "raw_positive": "scene",
        "raw_negative": "",
    }
    if multi_character:
        generated_descriptor["multi_char_layout"] = {
            "character_order": ["Aoi", "Ren"],
            "background_prompt": "classroom",
            "composition_prompt": "two characters",
            "regions": [],
        }
    child_ids = []
    stage_order = []
    asset_started = asyncio.Event()
    gpu_futures = []
    gpu_completion_tasks = []
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
                "multi_char_mask_enabled": multi_character,
            },
        },
    )

    async def fake_profile_resolve(**kwargs):
        stage_order.append("profile")
        assert kwargs["payload"]["target_slotted"].startswith("First")
        return resolved_profile_output, resolved_profile_result

    async def fake_select(**kwargs):
        stage_order.append("asset_start")
        assert kwargs["used_slots"] == {1}
        assert kwargs["reserve_slot_count"] == 0
        assert kwargs["profile_authority"] == "selected profile authority"
        assert "detail_3_done" in stage_order
        assert "gpu_enqueued" in stage_order
        assert gpu_futures and all(not future.done() for future in gpu_futures)
        asset_started.set()
        await asyncio.sleep(0)
        stage_order.append("asset_done")
        return {
            "items": [original_descriptor],
            "images": [original_bytes],
            "failures": [],
            "requested_count": 1,
            "target_slotted": "First\n\n[Slot 0]\n\nSecond\n\n[Slot 1]\n\nThird",
        }

    async def fake_build(build_payload, *_args, **kwargs):
        stage_order.append("call1_start")
        assert build_payload["target_slotted"].count("[Slot") == 2
        assert kwargs["pre_resolved_profile_output"] == resolved_profile_output
        assert kwargs["pre_resolved_profile_result"] == resolved_profile_result
        assert "before_call2" not in kwargs
        assert not asset_started.is_set(), "ORIGINAL-ASSET must not start before PLAN"
        stage_order.append("call1_done")

        await kwargs["on_call2_plan_ready"]({
            "session_id": session_id,
            "mode": "plan",
            "scene_slots": [1],
            "target_slotted": build_payload["target_slotted"],
        })
        stage_order.append("call2_after_plan")

        # ORIGINAL-ASSET은 PLAN 직후 LLM 슬롯을 선점하지 않고 마지막 DETAIL까지 양보한다.
        for detail_index in range(1, 4):
            await asyncio.sleep(0)
            assert not asset_started.is_set()
            stage_order.append(f"detail_{detail_index}_done")

        await kwargs["on_call2_ready"]({
            "context": "context",
            "prompt_format": "v3",
            "items": [generated_descriptor],
        })
        if multi_character:
            # 다중 장면은 CALL3/레이아웃 뒤 실제 GPU 큐 등록까지 한 번 더 보류한다.
            await asyncio.sleep(0)
            assert not asset_started.is_set()
        else:
            await asyncio.wait_for(asset_started.wait(), timeout=1.0)
        return {
            "context": "context",
            "prompt_format": "v3",
            "items": [generated_descriptor],
            "llm_trace": [],
        }

    async def fake_add_item(_item_type, _label, params, **_kwargs):
        child_id = params["prompt_id"]
        child_ids.append(child_id)
        stage_order.append("gpu_enqueued")
        future = asyncio.get_running_loop().create_future()
        gpu_futures.append(future)

        async def complete_gpu_after_asset_starts():
            await asset_started.wait()
            stage_order.append("gpu_done")
            server.prompts[child_id]["image_bytes"] = generated_bytes
            future.set_result({"success": True})

        gpu_completion_tasks.append(asyncio.create_task(complete_gpu_after_asset_starts()))
        return SimpleNamespace(status="processing", completion_future=future)

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
        result = await asyncio.wait_for(
            server.process_illustration_context_queue_item(parent_item),
            timeout=2.0,
        )

        assert result["count"] == 2
        assert result["requested_count"] == 2
        session = pipeline.get_session(session_id)
        assert [item["slot"] for item in session["items"]] == [0, 1]
        assert [item["kind"] for item in session["items"]] == [
            "original_asset",
            "scene",
        ]
        assert session["images"] == [original_bytes, generated_bytes]
        assert stage_order.index("call2_after_plan") < stage_order.index("detail_1_done")
        assert stage_order.index("detail_3_done") < stage_order.index("gpu_enqueued")
        assert stage_order.index("gpu_enqueued") < stage_order.index("asset_start")
        assert stage_order.index("asset_start") < stage_order.index("gpu_done")
    finally:
        for task in gpu_completion_tasks:
            if not task.done():
                task.cancel()
        if gpu_completion_tasks:
            await asyncio.gather(*gpu_completion_tasks, return_exceptions=True)
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        server.prompts.pop(prompt_id, None)
        for child_id in child_ids:
            server.prompts.pop(child_id, None)


def test_original_asset_plan_reserves_slots_but_gpu_dispatch_starts_selection() -> None:
    server_source = Path(server.__file__).read_text(encoding="utf-8")
    pipeline_source = Path(pipeline.__file__).read_text(encoding="utf-8")

    assert "before_call2" not in server_source
    assert "before_call2" not in pipeline_source
    assert "on_call2_plan_ready=None" in pipeline_source
    assert "CALL2-PLAN 직후 선택 시작" not in server_source
    assert "CALL2-PLAN slot 확정 · DETAIL/GPU 등록까지 선택 보류" in server_source

    plan_join = pipeline_source.index("call2_plan_output = await illustration_flow.join(plan_task)")
    plan_parse = pipeline_source.index("parsed_plan, plan_reason = parse_call2_plan(", plan_join)
    plan_callback = pipeline_source.index("await on_call2_plan_ready({", plan_parse)
    detail_stage = pipeline_source.index('parallel_stage = "CALL2-DETAIL"', plan_callback)
    assert plan_join < plan_parse < plan_callback < detail_stage

    server_plan_callback = server_source.index("async def _on_call2_plan_ready")
    slot_reservation = server_source.index(
        "original_asset_plan_slots = {",
        server_plan_callback,
    )
    call2_ready = server_source.index("async def _on_call2_ready", slot_reservation)
    gpu_enqueue = server_source.index("child_pairs.append(await _enqueue_child(", call2_ready)
    asset_start = server_source.index(
        "_start_original_asset_selection(",
        gpu_enqueue,
    )
    assert server_plan_callback < slot_reservation < call2_ready < gpu_enqueue < asset_start


def test_asset_only_path_is_free_and_mixed_path_waits_for_plan_slots() -> None:
    source = Path(server.__file__).read_text(encoding="utf-8")
    asset_only = source.index("if not illustration_enabled:")
    free_select = source.index("_run_original_asset_selection(set())", asset_only)
    plan_wait = source.index("일반 삽화 PLAN slot 확정 대기", free_select)
    plan_callback = source.index("async def _on_call2_plan_ready", plan_wait)
    plan_reservation = source.index("original_asset_plan_slots = {", plan_callback)
    call2_ready = source.index("async def _on_call2_ready", plan_reservation)
    gpu_enqueue = source.index("child_pairs.append(await _enqueue_child(", call2_ready)
    mixed_select = source.index("_start_original_asset_selection(", gpu_enqueue)
    assert (
        asset_only
        < free_select
        < plan_wait
        < plan_callback
        < plan_reservation
        < call2_ready
        < gpu_enqueue
        < mixed_select
    )
