import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import illustration_context_pipeline as pipeline


def test_context_and_result_transport_markers():
    session_id = "session_12345678"
    context_payload = {
        "session_id": session_id,
        "target_slotted": "첫 문장.\n\n[Slot 0]\n\n둘째 문장.",
        "chats": [
            {"role": "user", "data": "질문"},
            {"role": "assistant", "content": "응답"},
        ],
    }
    parsed = pipeline.parse_context_request(
        pipeline.CONTEXT_PREFIX + "\n" + json.dumps(context_payload, ensure_ascii=False)
    )

    assert parsed["session_id"] == session_id
    assert parsed["chats"] == [
        {"role": "user", "data": "질문"},
        {"role": "char", "data": "응답"},
    ]
    assert parsed["action"] == "regenerate"
    assert parsed["slot"] is None
    assert "[Slot 0]" in parsed["target_slotted"]
    assert pipeline.parse_result_request(
        pipeline.RESULT_PREFIX + "\n" + json.dumps({"session_id": session_id, "slot": -1})
    ) == {"session_id": session_id, "index": None, "slot": -1}
    assert pipeline.parse_regenerate_request(
        pipeline.REGENERATE_PREFIX + "\n" + json.dumps({"session_id": session_id, "slot": 0})
    ) == {"session_id": session_id, "slot": 0}


def test_descriptor_slots_trust_call2_with_light_sanitization():
    descriptors = [
        {"kind": "keyvis", "slot": -1},
        *({"kind": "scene", "slot": slot} for slot in (1, 1, 999, 4, 2)),
    ]
    slotted = "\n\n".join(f"문단 {index}\n\n[Slot {index}]" for index in range(10))

    normalized = pipeline.sanitize_descriptor_slots(descriptors, slotted)

    # keyvis=-1; CALL2 picks: 1(keep), 1(dup->nearest unused 0), 999(out->nearest 9),
    # 4(keep), 2(keep).
    assert [item["slot"] for item in normalized] == [-1, 1, 0, 9, 4, 2]


def test_descriptor_slot_sanitization_drops_excess_when_candidates_exhausted():
    descriptors = [
        {"kind": "keyvis", "slot": -1},
        *({"kind": "scene", "slot": index} for index in range(5)),
    ]

    normalized = pipeline.sanitize_descriptor_slots(
        descriptors,
        "첫째\n\n[Slot 2]\n\n둘째\n\n[Slot 8]\n\n셋째",
    )

    # 후보 [2,8]: scene 0->2, 1->8, 나머지(2,3,4)는 빈 후보 없음 -> 드롭.
    assert [item["slot"] for item in normalized] == [-1, 2, 8]


def test_call2_context_anchors_survive_slot_sanitization():
    descriptors = [
        {"kind": "keyvis", "slot": -1},
        {"kind": "scene", "slot": 1},
        {"kind": "scene", "slot": 3},
    ]
    slotted = (
        "첫 문단의 마지막 문구.\n\n[Slot 0]\n\n"
        "둘째 문단의 핵심 문구.\n\n[Slot 1]\n\n"
        "셋째 문단의 장면 문구.\n\n[Slot 2]\n\n"
        "넷째 문단의 시작 문구.\n\n[Slot 3]\n\n"
        "다섯째 문단의 후속 문구."
    )

    pipeline.attach_descriptor_anchors(descriptors, slotted)
    normalized = pipeline.sanitize_descriptor_slots(descriptors, slotted)

    scenes = [item for item in normalized if item["kind"] == "scene"]
    # CALL2 원 슬롯(1, 3)을 신뢰하여 그대로 유지.
    assert [item["slot"] for item in scenes] == [1, 3]
    assert scenes[0]["anchor_before"] == "둘째 문단의 핵심 문구."
    assert scenes[0]["anchor_after"] == "셋째 문단의 장면 문구."
    assert scenes[1]["anchor_before"] == "넷째 문단의 시작 문구."
    assert scenes[1]["anchor_after"] == "다섯째 문단의 후속 문구."


def test_call2_context_anchor_text_is_bounded_and_survives_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    long_before = "앞" * 240
    long_after = "뒤" * 240
    items = [{"kind": "scene", "slot": 0}]
    pipeline.attach_descriptor_anchors(
        items,
        f"{long_before}\n\n[Slot 0]\n\n{long_after}",
    )

    assert len(items[0]["anchor_before"]) == 180
    assert len(items[0]["anchor_after"]) == 180
    pipeline.create_session("anchor_cache_test_1234", "context")
    pipeline.set_session_result("anchor_cache_test_1234", items, [b"scene"])
    pipeline._SESSIONS.pop("anchor_cache_test_1234")

    restored = pipeline.session_item_by_slot("anchor_cache_test_1234", 0)
    assert restored["anchor_before"] == items[0]["anchor_before"]
    assert restored["anchor_after"] == items[0]["anchor_after"]
    assert "anchor_slot" not in restored


def test_context_transport_actions_keep_chat_data_and_validate_slot():
    session_id = "session_actions_1234"
    base = {
        "session_id": session_id,
        "target_slotted": "첫 문장.\n\n[Slot 0]\n\n둘째 문장.",
        "chats": [
            {"role": "user", "data": "질문"},
            {"role": "char", "data": "응답"},
        ],
    }

    for action in ("generate", "result"):
        payload = {**base, "action": action, "slot": -1}
        parsed = pipeline.parse_context_request(
            pipeline.CONTEXT_PREFIX + "\n" + json.dumps(payload, ensure_ascii=False)
        )
        assert parsed["action"] == action
        assert parsed["slot"] == -1
        assert parsed["chats"] == base["chats"]

    invalid_action = {**base, "action": "fallback"}
    assert pipeline.parse_context_request(
        pipeline.CONTEXT_PREFIX + "\n" + json.dumps(invalid_action, ensure_ascii=False)
    ) is None

    missing_slot = {**base, "action": "generate"}
    assert pipeline.parse_context_request(
        pipeline.CONTEXT_PREFIX + "\n" + json.dumps(missing_slot, ensure_ascii=False)
    ) is None


def test_toon_parse_and_slot_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    toon = """<lb-xnai>
keyvis:
  camera: close-up
  characters[1]:
    - name: hana
      positive: 1girl, black hair
      negative: bad hands
  scene: classroom
scenes[1]:
  - camera: medium shot
    characters[1]:
      - name: hana
        positive: 1girl, black hair, smile
    scene: classroom, sunset
    slot: 0
</lb-xnai>"""
    items = pipeline.parse_toon_plan(toon, pipeline.merged_toggles({"scene_max": 11}))

    assert [item["slot"] for item in items] == [-1, 0]
    pipeline.create_session("cache_test_1234", "context")
    pipeline.set_session_result("cache_test_1234", items, [b"keyvis", b"scene"])
    assert pipeline.session_image_by_slot("cache_test_1234", -1) == b"keyvis"
    assert pipeline.session_image_by_slot("cache_test_1234", 0) == b"scene"
    assert pipeline.session_item_by_slot("cache_test_1234", 0)["scene"] == "classroom, sunset"


def test_short_lookup_key_returns_only_ready_slots_and_survives_metadata_reload(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("a" * 64)
    lookup_key = "a" * 24
    pipeline._SESSIONS.pop(session_id, None)
    pipeline._LOOKUP_KEYS.pop(lookup_key, None)

    try:
        session = pipeline.create_session(session_id, "private context")
        assert session["lookup_key"] == lookup_key
        pipeline.set_session_result(
            session_id,
            [
                {"kind": "keyvis", "slot": -1},
                {"kind": "scene", "slot": 3},
                {"kind": "scene", "slot": 10},
            ],
            [b"keyvis", b"scene-3", b"scene-10"],
        )
        assert pipeline.session_slots_by_lookup_key(lookup_key) == [-1, 3, 10]

        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        assert pipeline.session_slots_by_lookup_key(lookup_key) == [-1, 3, 10]
    finally:
        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)


def test_short_lookup_key_collision_is_rejected_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    lookup_key = "b" * 24
    first_id = "risu_" + lookup_key + ("1" * 40)
    second_id = "risu_" + lookup_key + ("2" * 40)
    pipeline._LOOKUP_KEYS.pop(lookup_key, None)

    try:
        first = pipeline.create_session(first_id, "first")
        assert first["lookup_key"] == lookup_key
        with pytest.raises(ValueError, match="lookup key collision"):
            pipeline.create_session(second_id, "second")
        assert pipeline._LOOKUP_KEYS[lookup_key] == first_id
        assert second_id not in pipeline._SESSIONS
    finally:
        pipeline._SESSIONS.pop(first_id, None)
        pipeline._SESSIONS.pop(second_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
    assert pipeline.update_session_image_by_slot("cache_test_1234", 0, b"rerolled")
    assert pipeline.session_image_by_slot("cache_test_1234", 0) == b"rerolled"
    pipeline._SESSIONS.pop("cache_test_1234")
    assert pipeline.session_item_by_slot("cache_test_1234", 0)["scene"] == "classroom, sunset"


def test_toon_parse_preserves_yaml_punctuation_in_unquoted_text_fields():
    toon = """[TOON]
scenes[1]:
  - camera: close-up: from side
    characters[1]:
      - positive: 1girl, (solo focus:1.2), sign: blue # not a YAML comment
        negative: lowres, bad hands
        name: Bbyakbbyak
    scene: guildhouse: warm interior
    slot: 7
    supplement: A split-screen composition: the left side shows Kai's dark room #1.
[/TOON]"""

    items = pipeline.parse_toon_plan(
        toon,
        pipeline.merged_toggles({"scene_max": 11, "key_visual": False}),
    )

    assert len(items) == 1
    assert items[0]["camera"] == "close-up: from side"
    assert items[0]["scene"] == "guildhouse: warm interior"
    assert items[0]["supplement"] == (
        "A split-screen composition: the left side shows Kai's dark room #1."
    )
    assert items[0]["characters"][0]["positive"] == (
        "1girl, (solo focus:1.2), sign: blue # not a YAML comment"
    )


def test_toon_parse_error_uses_actual_call_source(capsys):
    assert pipeline.parse_toon_plan(
        "[TOON]\nscenes: invalid\n[/TOON]",
        pipeline.merged_toggles({}),
        "CALL3",
    ) == []

    captured = capsys.readouterr()
    assert "[ILLUST_CONTEXT:CALL3] scenes가 list가 아님" in captured.out
    assert "[ILLUST_CONTEXT:CALL2]" not in captured.out


def test_session_progress_tracks_call_and_image_counts_without_payload_data():
    session_id = "progress_test_1234"
    pipeline.create_session(session_id, "private chat context")
    try:
        pipeline.set_session_progress(
            session_id,
            "CALL3",
            "CALL3 대사 빌드",
            value=52,
            done=0,
            total=0,
        )
        progress = pipeline.get_session(session_id)["progress"]
        assert progress == {
            "phase": "call3",
            "label": "CALL3 대사 빌드",
            "value": 52.0,
            "done": 0,
            "total": 0,
        }
        assert "private chat context" not in progress.values()

        pipeline.set_session_progress(
            session_id,
            "generating",
            "이미지 4/3 완료",
            value=130,
            done=4,
            total=3,
        )
        progress = pipeline.get_session(session_id)["progress"]
        assert progress["phase"] == "generating"
        assert progress["value"] == 100.0
        assert progress["done"] == 3
        assert progress["total"] == 3

        pipeline.set_session_error(session_id, "remote failed")
        progress = pipeline.get_session(session_id)["progress"]
        assert progress["phase"] == "error"
        assert progress["done"] == 3
        assert progress["total"] == 3
    finally:
        pipeline._SESSIONS.pop(session_id, None)


def test_call2_macro_render_has_no_risu_macros():
    prompts = pipeline.load_prompt_files()
    rendered = pipeline.render_call2_prompt(
        prompts["call2_system"],
        pipeline.merged_toggles({
            "nsfw": False,
            "supplement": True,
            "key_visual": True,
            "character_limit": 2,
            "context_history": True,
            "focus": "hana",
            "direction": "Prefer cinematic lighting.",
        }),
        "### hana\n-Appearance: 1girl, black hair",
    )

    assert "{{" not in rendered
    assert "Limit characters to max 2" in rendered
    assert "### hana" in rendered
    assert 'focus on the character(s): "hana"' in rendered
    assert "Prefer cinematic lighting." in rendered


def test_prompt_format_migrates_legacy_preset_and_rejects_unknown_value(capsys):
    assert pipeline.merged_toggles({"preset": "tutorial"})["prompt_format"] == "v3"
    assert pipeline.merged_toggles({"preset": "v1"})["prompt_format"] == "v1"
    assert pipeline.merged_toggles({"prompt_format": "V1"})["prompt_format"] == "v1"
    assert pipeline.merged_toggles({"prompt_format": "future"})["prompt_format"] == "v3"
    assert "지원하지 않는 프롬프트 입력 형식" in capsys.readouterr().out


def test_call3_prompt_mode_defaults_and_rejects_unknown_value(capsys):
    assert pipeline.merged_toggles({})["call3_prompt_mode"] == "speak"
    assert pipeline.merged_toggles({"call3_prompt_mode": "MANGA"})["call3_prompt_mode"] == "manga"
    assert pipeline.merged_toggles({"call3_prompt_mode": "future"})["call3_prompt_mode"] == "speak"
    assert "지원하지 않는 CALL3 대사 프롬프트" in capsys.readouterr().out


def test_call3_dialogue_prompt_selects_speak_or_manga_and_scopes_emotions():
    prompts = {
        "call3_speak": "SPEAK PROMPT",
        "call3_manga": "MANGA PROMPT",
    }

    speak_mode, speak_prompt = pipeline.build_call3_dialogue_system_prompt(
        prompts,
        pipeline.merged_toggles({
            "call3_prompt_mode": "speak",
            "speak_emotion_enabled": True,
            "speak_emotions": "joy; anger",
        }),
        "CHARACTER REFERENCE",
    )
    manga_mode, manga_prompt = pipeline.build_call3_dialogue_system_prompt(
        prompts,
        pipeline.merged_toggles({
            "call3_prompt_mode": "manga",
            "speak_emotion_enabled": True,
            "speak_emotions": "joy; anger",
        }),
        "CHARACTER REFERENCE",
    )

    assert speak_mode == "speak"
    assert speak_prompt.startswith("SPEAK PROMPT")
    assert "Add one #emotion tag" in speak_prompt
    assert "Allowed labels: joy; anger" in speak_prompt
    assert manga_mode == "manga"
    assert manga_prompt.startswith("MANGA PROMPT")
    assert "CHARACTER REFERENCE" in manga_prompt
    assert "#emotion" not in manga_prompt


def test_parse_speak_output_enforces_two_entry_limit_per_scene(capsys):
    parsed = pipeline.parse_speak_output(
        '''[Scene slot=3]
Alice: "First"
Bob: "Second"
Alice: "Third"
[Scene slot=4] Bob: "Only"''',
        max_entries_per_scene=2,
    )

    assert parsed == {
        3: 'Alice: "First"\nBob: "Second"',
        4: 'Bob: "Only"',
    }
    assert "slot=3, limit=2, dropped=1" in capsys.readouterr().out


def test_manga_prompt_declares_all_balloon_labels_and_short_dialogue_rules():
    manga = pipeline.load_prompt_files()["call3_manga"]

    for label in (
        "#normal",
        "#angular",
        "#monologue_box",
        "#thought_cloud",
        "#trembling",
        "#burst",
        "#whisper",
        "#charming",
    ):
        assert label in manga
    assert "Do not write paragraphs, long monologues, or multi-sentence speeches." in manga


@pytest.mark.asyncio
async def test_build_from_context_uses_selected_manga_prompt(monkeypatch):
    calls = []
    responses = [
        """<lb-xnai>
scenes[1]:
  - camera: close-up
    characters[1]:
      - name: hana
        positive: 1girl, hana, black hair
    scene: classroom
    slot: 0
</lb-xnai>""",
        """[Scene slot=0]
Hana: "No way!" #burst""",
    ]

    async def fake_call(task_key, messages, **kwargs):
        calls.append(messages)
        return responses[len(calls) - 1]

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "manga_prompt_test_1234",
            "target_slotted": "하나가 놀란다.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "무슨 일이야?"},
                {"role": "char", "data": "하나가 놀란다."},
            ],
        },
        {
            "call1_enabled": False,
            "call3_enabled": True,
            "speak_enabled": True,
            "call3_prompt_mode": "manga",
            "key_visual": False,
        },
        "### hana\n-Appearance: 1girl, black hair",
    )

    assert len(calls) == 2
    assert "manga dialogue and balloon-style editor" in calls[1][0]["content"]
    assert "#normal" in calls[1][0]["content"]
    assert result["items"][0]["speak"] == 'Hana: "No way!" #burst'
    assert '[SPEAK]\nHana: "No way!" #burst' in result["items"][0]["raw_positive"]


def test_build_raw_prompt_uses_v1_or_v3_input_shape():
    descriptor = {
        "slot": 2,
        "camera": "medium shot",
        "scene": "classroom",
        "supplement": "sunset lighting",
        "speak": "Hana: hello",
        "characters": [{
            "name": "hana",
            "positive": "1girl, hana, black hair",
            "negative": "bad hands",
        }],
    }
    prompts = pipeline.load_prompt_files()

    v1_positive, v1_negative = pipeline.build_raw_prompt(
        descriptor,
        "창가에 선다.",
        prompts,
        pipeline.merged_toggles({"prompt_format": "v1"}),
    )
    # V1도 이제 V3 마커(SETUP/CHAR/SUPPLEMENT)를 내보낸다.
    # 포맷별 최종 조립(ILXL 등)은 후속 처리기(process_prompt)가 수행.
    assert "[SETUP]\nmedium shot, classroom" in v1_positive
    assert "[CHAR]\n1girl, hana, black hair" in v1_positive
    assert "[ILXL]" not in v1_positive
    assert "[CHAT]\n창가에 선다." in v1_positive
    assert "bad hands" in v1_negative

    v3_positive, v3_negative = pipeline.build_raw_prompt(
        descriptor,
        "창가에 선다.",
        prompts,
        pipeline.merged_toggles({"prompt_format": "v3"}),
    )
    assert "[SETUP]\nmedium shot, classroom" in v3_positive
    assert "[CHAR]\n1girl, hana, black hair" in v3_positive
    assert "[ILXL]" not in v3_positive
    assert "bad hands" in v3_negative


@pytest.mark.asyncio
async def test_pipeline_llm_records_success_in_lighbd_history(monkeypatch):
    records = []
    events = []
    messages = [{"role": "user", "content": "scene"}]

    async def fake_call(task_key, actual_messages):
        assert task_key == "illustration_call1"
        assert actual_messages == messages
        return "completed output"

    async def fake_notify(event):
        events.append(event)

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)

    result = await pipeline._call_pipeline_llm("CALL1", messages, fake_notify)

    assert result == "completed output"
    assert [event["type"] for event in events] == ["start", "done"]
    assert len(records) == 1
    assert records[0]["call_name"] == "CALL1"
    assert records[0]["task_key"] == "illustration_call1"
    assert records[0]["input"] == messages
    assert records[0]["output"] == "completed output"
    assert records[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_pipeline_llm_records_failure_in_lighbd_history(monkeypatch):
    records = []
    events = []
    messages = [{"role": "user", "content": "broken scene"}]

    async def fake_call(task_key, actual_messages):
        return "[LLM 실패] upstream unavailable"

    async def fake_notify(event):
        events.append(event)

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await pipeline._call_pipeline_llm("CALL2", messages, fake_notify)

    assert [event["type"] for event in events] == ["start", "error"]
    assert len(records) == 1
    assert records[0]["call_name"] == "CALL2"
    assert records[0]["input"] == messages
    assert records[0]["output"] == ""
    assert records[0]["status"] == "error"
    assert "upstream unavailable" in records[0]["error"]


@pytest.mark.asyncio
async def test_call1_enrichment_is_passed_to_call2_without_changing_slots(monkeypatch):
    calls = []
    responses = [
        """[Position]
첫 문장.
[/Position]
[Visual Content #01]
A character turns toward the window.
[DynamicPrompt scene="01"]
looking away, window
[/DynamicPrompt]
[CharacterBaseTags]
hana : 1girl, black hair
[/CharacterBaseTags]""",
        """<lb-xnai>
scenes[1]:
  - camera: medium shot
    characters[1]:
      - name: hana
        positive: 1girl, black hair
    scene: classroom
    slot: 0
</lb-xnai>""",
    ]

    async def fake_call(task_key, messages, **kwargs):
        calls.append(messages)
        return responses[len(calls) - 1]

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "pipeline_test_1234",
            "target_slotted": "첫 문장.\n\n[Slot 0]\n\n둘째 문장.",
            "chats": [
                {"role": "user", "data": "창밖을 봐."},
                {"role": "char", "data": "첫 문장.\n\n둘째 문장."},
            ],
        },
        {"call3_enabled": False, "speak_enabled": False, "key_visual": False},
        "### hana\n-Appearance: 1girl, black hair",
    )

    call2_text = "\n".join(message["content"] for message in calls[1])
    assert "[Visual Content #01]" in call2_text
    assert "[Slot 0]" in call2_text
    assert result["items"][0]["slot"] == 0
    assert result["items"][0]["raw_positive"]
