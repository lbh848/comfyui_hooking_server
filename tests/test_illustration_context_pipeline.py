import ast
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


def test_partial_result_returns_only_successful_slots_and_tracks_failures(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    lookup_key = "c" * 24
    session_id = "risu_" + lookup_key + ("3" * 40)
    pipeline._SESSIONS.pop(session_id, None)
    pipeline._LOOKUP_KEYS.pop(lookup_key, None)

    try:
        pipeline.create_session(session_id, "private context")
        pipeline.set_session_result(
            session_id,
            [
                {"kind": "keyvis", "slot": -1},
                {"kind": "scene", "slot": 10},
            ],
            [b"keyvis", b"scene-10"],
            requested_count=3,
            failures=[{"slot": 3, "error": "remote\nfailed"}],
        )

        session = pipeline.get_session(session_id)
        assert session["status"] == "ready"
        assert session["progress"] == {
            "phase": "ready_partial",
            "label": "성공 2/3장 반환 준비 완료 · 최종 실패 1장 제외",
            "value": 100,
            "done": 2,
            "total": 3,
        }
        assert session["requested_count"] == 3
        assert session["success_count"] == 2
        assert session["failure_count"] == 1
        assert session["failures"] == [{"slot": 3, "error": "remote failed"}]
        assert pipeline.session_slots_by_lookup_key(lookup_key) == [-1, 10]
        assert pipeline.session_image_by_slot(session_id, 10) == b"scene-10"
        assert pipeline.session_image_by_slot(session_id, 3) is None

        summary = next(
            item
            for item in pipeline.recent_session_summaries()
            if item["session_id"] == session_id
        )
        assert summary["requested_count"] == 3
        assert summary["success_count"] == 2
        assert summary["failure_count"] == 1

        pipeline._SESSIONS.pop(session_id, None)
        pipeline._LOOKUP_KEYS.pop(lookup_key, None)
        assert pipeline.session_slots_by_lookup_key(lookup_key) == [-1, 10]
        restored = pipeline.get_session(session_id)
        assert restored["progress"]["phase"] == "ready_partial"
        assert restored["requested_count"] == 3
        assert restored["failure_count"] == 1
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


def test_raw_full_generation_progress_is_active_until_slot_image_updates(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "raw_full_progress_1234"
    pipeline.create_session(session_id, "private chat context")
    pipeline.set_session_result(
        session_id,
        [
            {"kind": "keyvis", "slot": -1},
            {"kind": "scene", "slot": 3},
            {"kind": "scene", "slot": 8},
        ],
        [b"old keyvis", b"old scene 3", b"old scene 8"],
    )

    try:
        pipeline.set_session_regenerate_started(
            session_id,
            -1,
            "전체 생성",
            whole_session=True,
        )

        active_session = pipeline.get_session(session_id)
        assert active_session["status"] == "ready"
        assert active_session["progress"] == {
            "phase": "regenerating",
            "label": "전체 3장 중 1장째 · 키비주얼 서버 전체 생성 중",
            "value": 0.0,
            "done": 0,
            "total": 3,
        }
        summary = next(
            item
            for item in pipeline.recent_session_summaries()
            if item["session_id"] == session_id
        )
        assert summary["status"] == "ready"
        assert summary["progress"] == active_session["progress"]

        assert pipeline.update_session_image_by_slot(session_id, -1, b"new keyvis")
        completed_session = pipeline.get_session(session_id)
        assert completed_session["progress"]["phase"] == "ready"
        assert completed_session["progress"] == {
            "phase": "ready",
            "label": "전체 3장 중 1장 완료 · 키비주얼 서버 전체 생성 완료",
            "value": 33.3,
            "done": 1,
            "total": 3,
        }
        assert pipeline.session_image_by_slot(session_id, -1) == b"new keyvis"

        pipeline.set_session_regenerate_started(
            session_id,
            3,
            "전체 생성",
            whole_session=True,
        )
        assert pipeline.get_session(session_id)["progress"] == {
            "phase": "regenerating",
            "label": "전체 3장 중 2장째 · 슬롯 3 서버 전체 생성 중",
            "value": 33.3,
            "done": 1,
            "total": 3,
        }
    finally:
        pipeline._SESSIONS.pop(session_id, None)


def test_raw_full_generation_queue_marks_session_active():
    server_path = Path(__file__).resolve().parents[1] / "server.py"
    tree = ast.parse(server_path.read_text(encoding="utf-8-sig"))
    enqueue_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_enqueue_illustration_session_slot"
    )
    progress_calls = [
        node
        for node in ast.walk(enqueue_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_session_regenerate_started"
    ]

    assert len(progress_calls) == 1
    argument_names = [
        argument.id
        for argument in progress_calls[0].args
        if isinstance(argument, ast.Name)
    ]
    assert argument_names == [
        "session_id",
        "slot",
        "operation_label",
    ]
    assert [keyword.arg for keyword in progress_calls[0].keywords] == ["whole_session"]
    assert isinstance(progress_calls[0].keywords[0].value, ast.Name)
    assert progress_calls[0].keywords[0].value.id == "whole_session"

    handle_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_prompt"
    )
    enqueue_calls = [
        node
        for node in ast.walk(handle_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_enqueue_illustration_session_slot"
    ]
    assert len(enqueue_calls) == 1
    enqueue_keywords = {keyword.arg: keyword.value for keyword in enqueue_calls[0].keywords}
    assert isinstance(enqueue_keywords["whole_session"], ast.Constant)
    assert enqueue_keywords["whole_session"].value is True


def test_individual_regeneration_keeps_single_slot_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "single_regenerate_progress_1234"
    pipeline.create_session(session_id, "private chat context")
    pipeline.set_session_result(
        session_id,
        [
            {"kind": "keyvis", "slot": -1},
            {"kind": "scene", "slot": 3},
        ],
        [b"old keyvis", b"old scene"],
    )

    try:
        pipeline.set_session_regenerate_started(session_id, 3)
        assert pipeline.get_session(session_id)["progress"] == {
            "phase": "regenerating",
            "label": "슬롯 3 서버 재생성 중",
            "value": 0.0,
            "done": 0,
            "total": 1,
        }

        assert pipeline.update_session_image_by_slot(session_id, 3, b"new scene")
        assert pipeline.get_session(session_id)["progress"] == {
            "phase": "ready",
            "label": "슬롯 3 서버 재생성 완료",
            "value": 100,
            "done": 1,
            "total": 1,
        }
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


def test_backtranslation_defaults_and_concurrency_clamp():
    defaults = pipeline.merged_toggles({})
    assert defaults["call1_backtranslate_enabled"] is False
    assert defaults["call1_backtranslate_max_concurrency"] == 4
    assert defaults["call1_backtranslate_failure_strategy"] == "fallback"
    assert pipeline.merged_toggles({
        "call1_backtranslate_max_concurrency": 0,
    })["call1_backtranslate_max_concurrency"] == 1
    assert pipeline.merged_toggles({
        "call1_backtranslate_max_concurrency": 99,
    })["call1_backtranslate_max_concurrency"] == 16
    assert pipeline.merged_toggles({
        "call1_backtranslate_failure_strategy": "retry_abort",
    })["call1_backtranslate_failure_strategy"] == "retry_abort"
    assert pipeline.merged_toggles({
        "call1_backtranslate_failure_strategy": "unknown",
    })["call1_backtranslate_failure_strategy"] == "fallback"


def test_backtranslation_chunks_balance_contiguous_slot_groups():
    source = "\n\n".join(
        f"문단 {index}\n\n[Slot {index}]"
        for index in range(5)
    ) + "\n\n마지막 문단"

    chunks = pipeline.split_backtranslation_chunks(source, 2)

    assert len(chunks) == 2
    assert pipeline._SLOT_MARKER_RE.findall(chunks[0]) == ["0", "1", "2"]
    assert pipeline._SLOT_MARKER_RE.findall(chunks[1]) == ["3", "4"]
    assert "".join(chunks) == source


@pytest.mark.asyncio
async def test_backtranslation_stream_events_include_queue_subtask_metadata(monkeypatch):
    source = "첫 문단.\n\n[Slot 0]\n\n둘째 문단.\n\n[Slot 1]\n\n셋째 문단.\n\n[Slot 2]"
    chunks = pipeline.split_backtranslation_chunks(source, 3)
    events = []

    async def fake_pipeline_call(
        call_name, messages, stream_notify=None, result_validator=None
    ):
        index = int(call_name.rsplit(" ", 1)[1].split("/", 1)[0])
        translated = chunks[index - 1]
        await stream_notify({"type": "start", "call_name": call_name})
        await stream_notify({
            "type": "done",
            "call_name": call_name,
            "text": translated,
        })
        return translated

    async def capture(event):
        events.append(event)

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        3,
        stream_notify=capture,
    )

    assert translated == "\n\n".join(chunk.strip() for chunk in chunks)
    assert [status["status"] for status in statuses] == [
        "translated",
        "translated",
        "translated",
    ]
    start_metadata = [
        event["queue_subtask"]
        for event in events
        if event["type"] == "start"
    ]
    assert sorted(metadata["index"] for metadata in start_metadata) == [1, 2, 3]
    assert all(metadata["group_id"] == "backtranslation" for metadata in start_metadata)
    assert all(metadata["group_label"] == "역번역" for metadata in start_metadata)
    assert all(metadata["total"] == 3 for metadata in start_metadata)


@pytest.mark.asyncio
async def test_backtranslation_empty_response_falls_back_only_failed_chunk(monkeypatch, capsys):
    source = "첫 문단.\n\n[Slot 0]\n\n둘째 문단.\n\n[Slot 1]\n\n끝 문단."

    async def fake_pipeline_call(
        call_name, messages, stream_notify=None, result_validator=None
    ):
        if call_name.endswith("1/2"):
            return "First paragraph.\n\n[Slot 0]"
        return "   "

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Protected names: {character_names}",
        "Bbyakbbyak",
        2,
    )

    assert translated == (
        "First paragraph.\n\n[Slot 0]\n\n"
        "둘째 문단.\n\n[Slot 1]\n\n끝 문단."
    )
    assert [status["status"] for status in statuses] == ["translated", "fallback"]
    assert statuses[1]["reason"] == "응답 길이가 0임"
    assert "output_len=3" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_backtranslation_slot_mismatch_falls_back_to_original_chunk(monkeypatch):
    source = "장면.\n\n[Slot 7]"

    async def fake_pipeline_call(
        call_name, messages, stream_notify=None, result_validator=None
    ):
        return "Scene.\n\n[Slot 8]"

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        4,
    )

    assert translated == source
    assert statuses[0]["status"] == "fallback"
    assert "슬롯 마커 불일치" in statuses[0]["reason"]


@pytest.mark.asyncio
async def test_backtranslation_strict_strategy_uses_central_route_retry(monkeypatch):
    calls = []
    source = "장면.\n\n[Slot 3]"

    async def fake_pipeline_call(
        call_name, messages, stream_notify=None, result_validator=None
    ):
        calls.append(call_name)
        assert result_validator is not None
        assert result_validator("Scene.\n\n[Slot 8]")[0] is False
        successful = "Scene.\n\n[Slot 3]"
        assert result_validator(successful)[0] is True
        return successful

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        4,
        failure_strategy="retry_abort",
    )

    assert translated == "Scene.\n\n[Slot 3]"
    assert len(calls) == 1
    assert statuses == [{
        "index": 1,
        "status": "translated",
        "reason": "",
        "attempts": 2,
    }]


@pytest.mark.asyncio
async def test_backtranslation_strict_strategy_aborts_after_route_retries(monkeypatch):
    calls = []

    async def fake_pipeline_call(
        call_name, messages, stream_notify=None, result_validator=None
    ):
        calls.append(call_name)
        assert result_validator is not None
        invalid = "Scene.\n\n[Slot 8]"
        assert result_validator(invalid)[0] is False
        assert result_validator(invalid)[0] is False
        return invalid

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    with pytest.raises(RuntimeError, match="엄격 전략 실패"):
        await pipeline.backtranslate_current_context(
            "장면.\n\n[Slot 9]",
            "Translate. {character_names}",
            "Hana",
            4,
            failure_strategy="retry_abort",
        )

    assert len(calls) == 1


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


@pytest.mark.asyncio
async def test_call2_ready_callback_runs_before_call3_and_receives_generation_raw(monkeypatch):
    events = []
    early_payloads = []

    async def fake_call(task_key, messages, **kwargs):
        events.append(task_key)
        if task_key == "illustration_call2":
            return """<lb-xnai>
scenes[1]:
  - camera: close-up
    characters[1]:
      - name: hana
        positive: 1girl, hana, black hair
    scene: classroom
    slot: 0
</lb-xnai>"""
        assert task_key == "illustration_call3"
        assert events == ["illustration_call2", "dispatch", "illustration_call3"]
        return '[Scene slot=0]\nHana: "Ready." #normal'

    async def on_call2_ready(payload):
        events.append("dispatch")
        early_payloads.append(payload)
        assert payload["items"][0].get("speak") in (None, "")
        assert 'Hana: "Ready."' not in payload["items"][0]["raw_positive"]

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_early_dispatch_test",
            "target_slotted": "하나가 교실을 둘러본다.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "어디야?"},
                {"role": "char", "data": "하나가 교실을 둘러본다."},
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
        on_call2_ready=on_call2_ready,
    )

    assert events == ["illustration_call2", "dispatch", "illustration_call3"]
    assert len(early_payloads) == 1
    assert result["items"][0]["speak"] == 'Hana: "Ready." #normal'
    assert '[SPEAK]\nHana: "Ready." #normal' in result["items"][0]["raw_positive"]


@pytest.mark.asyncio
async def test_build_from_context_uses_backtranslated_current_response_only(monkeypatch):
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        if task_key == "illustration_call1_backtranslate":
            body = messages[-1]["content"]
            assert "Bbyakbbyak" in messages[0]["content"]
            if "[Slot 0]" in body:
                return "Bbyakbbyak opens the door.\n\n[Slot 0]"
            return "She looks inside.\n\n[Slot 1]\n\nThe room is quiet."
        assert task_key == "illustration_call2"
        return """<lb-xnai>
scenes[1]:
  - camera: medium shot
    characters[1]:
      - name: Bbyakbbyak
        positive: 1girl, Bbyakbbyak, white hair
    scene: quiet room
    slot: 0
</lb-xnai>"""

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "backtranslate_pipeline_1234",
            "target_slotted": (
                "뺙뺙이 문을 연다.\n\n[Slot 0]\n\n"
                "그녀가 안을 본다.\n\n[Slot 1]\n\n방은 조용하다."
            ),
            "chats": [
                {"role": "user", "data": "과거 질문은 한국어다."},
                {"role": "char", "data": "과거 답변도 한국어다."},
                {"role": "user", "data": "문을 열어 봐."},
                {"role": "char", "data": "뺙뺙이 문을 연다.\n\n그녀가 안을 본다.\n\n방은 조용하다."},
            ],
        },
        {
            "call1_backtranslate_enabled": True,
            "call1_backtranslate_max_concurrency": 2,
            "call1_enabled": False,
            "call3_enabled": False,
            "speak_enabled": False,
            "key_visual": False,
        },
        "### Bbyakbbyak\n-Appearance: 1girl, white hair",
        backtranslate_names="Bbyakbbyak, Bbyakbbyak_reallife",
    )

    assert [task_key for task_key, _messages in calls].count(
        "illustration_call1_backtranslate"
    ) == 2
    call2_text = "\n".join(
        message["content"]
        for task_key, messages in calls
        if task_key == "illustration_call2"
        for message in messages
    )
    assert "과거 답변도 한국어다." in call2_text
    assert "Bbyakbbyak opens the door." in call2_text
    assert "뺙뺙이 문을 연다." not in call2_text
    assert "[CHAT]\nBbyakbbyak opens the door." in result["items"][0]["raw_positive"]
    assert "[CHAR]\n과거 답변도 한국어다." in result["context"]
    assert "[CHAR]\nBbyakbbyak opens the door." in result["context"]
    assert result["backtranslated_slotted"].startswith("Bbyakbbyak opens the door.")
    assert [entry["status"] for entry in result["backtranslation_chunks"]] == [
        "translated",
        "translated",
    ]


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
