import ast
import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import illustration_context_pipeline as pipeline


def _toon_for_slots(slots):
    rows = []
    for slot in slots:
        rows.append(
            "  - camera: medium shot\n"
            "    characters[1]:\n"
            "      - name: Hana\n"
            "        positive: 1girl, black hair\n"
            "        negative: lowres\n"
            "        outfit_state:\n"
            "          body_state: clothed\n"
            "          worn: [school uniform]\n"
            "          removed: []\n"
            "    scene: classroom\n"
            f"    plan_id: S{slot + 1:03d}\n"
            f"    slot: {slot}\n"
            "    supplement: daylight"
        )
    return "<lb-xnai>\nscenes[%d]:\n%s\n</lb-xnai>" % (len(slots), "\n".join(rows))


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


def test_call2_plan_selects_global_slots_and_builds_key_visual():
    slotted = "\n\n".join(
        ["첫 문단."]
        + [f"[Slot {index}]\n\n문단 {index + 2}." for index in range(5)]
    )
    raw = json.dumps({
        "scene_plan": [
            {
                "plan_id": f"raw-{index}",
                "slot": index,
                "source_segments": [f"C{index + 1:03d}"],
                "characters": ["Hana"],
                "scene_brief": f"장면 {index}",
            }
            for index in range(5)
        ],
        "keyvis": {
            "camera": "full body",
            "characters": [{
                "name": "Hana",
                "positive": "1girl, black hair",
                "negative": "lowres",
                "outfit_state": {
                    "body_state": "clothed",
                    "worn": ["school uniform"],
                    "removed": [],
                },
            }],
            "scene": "classroom",
            "supplement": "daylight",
        },
    }, ensure_ascii=False)

    plan, reason = pipeline.parse_call2_plan(
        raw,
        pipeline.merged_toggles({"scene_min": 5, "scene_max": 5}),
        slotted,
    )

    assert reason == ""
    assert [item["plan_id"] for item in plan["scene_plan"]] == [
        "S001", "S002", "S003", "S004", "S005"
    ]
    assert [item["slot"] for item in plan["scene_plan"]] == [0, 1, 2, 3, 4]
    assert plan["keyvis_descriptor"]["kind"] == "keyvis"
    assert plan["keyvis_descriptor"]["slot"] == -1


def test_segment_slot_map_binds_segments_to_following_server_slot():
    slotted = (
        "첫 문단.\n\n[Slot 0]\n\n"
        "둘째 문단.\n\n세부 문단.\n\n[Slot 1]\n\n"
        "마지막 문단."
    )
    _rendered, segments = pipeline._segment_current_context(
        pipeline.remove_slot_markers(slotted)
    )

    mapping, annotated, reason = pipeline.build_segment_slot_map(slotted, segments)

    assert reason == ""
    assert mapping == {"C001": 0, "C002": 1, "C003": 1, "C004": 1}
    assert "[C001 slot=0]" in annotated
    assert "[C004 slot=1]" in annotated


def test_call2_plan_uses_anchor_mapping_and_ignores_model_slot_number():
    slotted = "첫 문단.\n\n[Slot 0]\n\n둘째 문단.\n\n[Slot 1]"
    raw = json.dumps({
        "scene_plan": [{
            "plan_id": "raw-plan",
            "slot": 67,
            "anchor_segment": "C002",
            "source_segments": ["C002"],
            "characters": [{
                "name": "Hana",
                "outfit_state": {
                    "body_state": "clothed",
                    "worn": ["blue dress"],
                    "removed": [],
                },
            }],
            "scene_brief": "둘째 문단의 장면",
        }],
        "keyvis": None,
    }, ensure_ascii=False)

    plan, reason = pipeline.parse_call2_plan(
        raw,
        pipeline.merged_toggles({
            "scene_min": 1,
            "scene_max": 1,
            "key_visual": False,
        }),
        slotted,
        segment_slot_map={"C001": 0, "C002": 1},
    )

    assert reason == ""
    assert plan["scene_plan"][0]["anchor_segment"] == "C002"
    assert plan["scene_plan"][0]["slot"] == 1
    assert plan["scene_plan"][0]["planned_outfits"]["Hana"] == {
        "body_state": "clothed",
        "worn": ["blue dress"],
        "removed": [],
    }


def test_scene_plan_wardrobe_snapshot_uses_plan_over_tracked_timeline():
    plans = [{
        "plan_id": "S001",
        "slot": 0,
        "anchor_segment": "C001",
        "source_segments": ["C001"],
        "characters": ["Hana"],
        "planned_outfits": {
            "Hana": {"body_state": "clothed", "worn": ["red coat"], "removed": []},
        },
    }, {
        "plan_id": "S002",
        "slot": 1,
        "anchor_segment": "C003",
        "source_segments": ["C003"],
        "characters": ["Hana"],
        "planned_outfits": {
            "Hana": {"body_state": "topless", "worn": [], "removed": ["red coat"]},
        },
    }]
    state_before = {
        "hana": {
            "canonical_name": "Hana",
            "current_wardrobe": {
                "body_state": "clothed",
                "worn": ["blue dress"],
                "removed": [],
            },
        },
    }
    events = [{
        "segment_id": "C003",
        "character": "Hana",
        "operation": "remove",
        "items": ["blue dress"],
        "state_after": "topless",
    }]

    bound = pipeline.bind_scene_plan_wardrobes(
        plans,
        ["C001", "C002", "C003"],
        state_before,
        [{"name": "Hana", "confidence": 1.0}],
        events,
        "message-1",
    )

    assert bound[0]["wardrobe_snapshot"]["Hana"] == {
        "body_state": "clothed",
        "worn": ["red coat"],
        "removed": [],
    }
    assert bound[1]["wardrobe_snapshot"]["Hana"] == {
        "body_state": "topless",
        "worn": [],
        "removed": ["red coat"],
    }
    assert bound[0]["wardrobe_sources"]["Hana"] == "call2_plan"
    assert bound[1]["wardrobe_sources"]["Hana"] == "call2_plan"


def test_scene_plan_uses_each_outfit_decided_by_global_plan():
    plans = [{
        "plan_id": "S001",
        "slot": 0,
        "anchor_segment": "C001",
        "characters": ["Hana"],
        "planned_outfits": {
            "Hana": {"body_state": "clothed", "worn": ["red coat"], "removed": []},
        },
    }, {
        "plan_id": "S002",
        "slot": 1,
        "anchor_segment": "C002",
        "characters": ["Hana"],
        "planned_outfits": {
            "Hana": {"body_state": "clothed", "worn": ["blue dress"], "removed": []},
        },
    }]

    bound = pipeline.bind_scene_plan_wardrobes(
        plans,
        ["C001", "C002"],
        {},
        [{"name": "Hana", "confidence": 1.0}],
        [],
        "message-1",
    )

    assert bound[0]["wardrobe_snapshot"]["Hana"]["worn"] == ["red coat"]
    assert bound[1]["wardrobe_snapshot"]["Hana"]["worn"] == ["blue dress"]
    assert bound[0]["wardrobe_sources"]["Hana"] == "call2_plan"
    assert bound[1]["wardrobe_sources"]["Hana"] == "call2_plan"


def test_call2_detail_assigns_plan_ids_from_validated_slots():
    output_without_plan_ids = re.sub(
        r"\n\s+plan_id:\s*[^\r\n]+",
        "",
        _toon_for_slots([4, 9]),
    )
    descriptors, reason = pipeline._parse_call2_detail_output(
        output_without_plan_ids,
        pipeline.merged_toggles({"key_visual": False}),
        [4, 9],
        ["S021", "S022"],
        "TEST-CALL2-DETAIL-SERVER-PLAN-ID",
    )

    assert reason == ""
    assert [item["slot"] for item in descriptors] == [4, 9]
    assert [item["plan_id"] for item in descriptors] == ["S021", "S022"]


def test_call2_detail_rejects_wardrobe_different_from_plan_snapshot():
    descriptors, reason = pipeline._parse_call2_detail_output(
        _toon_for_slots([4]),
        pipeline.merged_toggles({"key_visual": False}),
        [4],
        ["S021"],
        "TEST-CALL2-DETAIL-WARDROBE",
        {
            4: {
                "Hana": {
                    "body_state": "clothed",
                    "worn": ["blue dress"],
                    "removed": [],
                },
            },
        },
    )

    assert descriptors == []
    assert "권위 복장 충돌" in reason


def test_call2_detail_accepts_superset_and_normalizes_to_plan_snapshot():
    detail_output = _toon_for_slots([4]).replace(
        "worn: [school uniform]",
        "worn: [school uniform, red scarf]",
    )
    descriptors, reason = pipeline._parse_call2_detail_output(
        detail_output,
        pipeline.merged_toggles({"key_visual": False}),
        [4],
        ["S021"],
        "TEST-CALL2-DETAIL-WARDROBE-SUPERSET",
        {
            4: {
                "Hana": {
                    "body_state": "clothed",
                    "worn": ["school uniform"],
                    "removed": [],
                },
            },
        },
    )

    assert reason == ""
    assert descriptors[0]["characters"][0]["outfit_state"] == {
        "body_state": "clothed",
        "worn": ["school uniform"],
        "removed": [],
    }


@pytest.mark.asyncio
async def test_call2_detail_wardrobe_conflict_retries_only_that_shard(monkeypatch, capsys):
    call_names = []
    initial = _toon_for_slots([4])
    corrected = initial.replace("school uniform", "blue dress")

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        if "WARDROBE-CORRECTION" in call_name:
            return corrected
        return initial

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    descriptors, raw_outputs = await pipeline._run_parallel_call2_details(
        scene_plan=[{
            "plan_id": "S001",
            "slot": 4,
            "anchor_segment": "C001",
            "source_segments": ["C001"],
            "characters": ["Hana"],
            "scene_brief": "Hana in a blue dress",
            "wardrobe_snapshot": {
                "Hana": {
                    "body_state": "clothed",
                    "worn": ["blue dress"],
                    "removed": [],
                },
            },
        }],
        keyvis_descriptor=None,
        call2_context_messages=[{"role": "system", "content": "Build detail."}],
        call2_format="Return TOON.",
        toggles=pipeline.merged_toggles({
            "key_visual": False,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
        }),
        stream_notify=None,
    )

    assert call_names == [
        "CALL2-DETAIL 1/1",
        "CALL2-DETAIL 1/1 [WARDROBE-CORRECTION]",
    ]
    assert len(raw_outputs) == 1
    assert descriptors[0]["characters"][0]["outfit_state"]["worn"] == ["blue dress"]
    assert "권위 복장 충돌 작업만 교정 재호출" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_call2_parallel_failure_is_named_fallback_and_logs_reason(
    monkeypatch,
    capsys,
):
    call_names = []

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        if call_name == "CALL2-PLAN":
            return "not valid plan json"
        if call_name == "CALL2-FALLBACK":
            return _toon_for_slots([0])
        raise AssertionError(f"unexpected call: {call_name}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_fallback_reason_test",
            "target_slotted": "Hana waits.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Continue."},
                {"role": "char", "data": "Hana waits."},
            ],
        },
        {
            "call1_enabled": False,
            "call2_parallel_enabled": True,
            "scene_min": 1,
            "scene_max": 1,
            "key_visual": False,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        "### Hana\n-default_outfit\nschool uniform",
        extra_costume="### Hana\n-default_outfit\nschool uniform",
        extra_names="Hana",
        backtranslate_names="Hana",
    )

    output = capsys.readouterr().out
    assert call_names == ["CALL2-PLAN", "CALL2-FALLBACK"]
    assert "[ILLUST_CONTEXT:CALL2-FALLBACK] 폴백 시작" in output
    assert "failed_stage=CALL2-PLAN" in output
    assert "CALL2-PLAN JSON object를 찾지 못함" in output
    assert result["call2_fallback_stage"] == "CALL2-PLAN"
    assert "CALL2-PLAN JSON object를 찾지 못함" in result["call2_fallback_reason"]


@pytest.mark.asyncio
async def test_parallel_job_tail_hedge_uses_shared_concurrency_and_duplicate_wins(monkeypatch):
    active = 0
    max_active = 0
    attempts = []
    history_updates = []

    async def invoke(job, index, total, attempt_kind, observer, history_id, notify):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        attempts.append((index, attempt_kind))
        try:
            if index < 3 or attempt_kind == "duplicate":
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.3)
            return {"raw": f"job-{index}-{attempt_kind}", "winner": attempt_kind}
        finally:
            active -= 1

    monkeypatch.setattr(
        pipeline.lighbd_service,
        "_update_lighbd_history_records",
        history_updates.append,
    )
    results = await pipeline._run_parallel_pipeline_jobs(
        [{"weight": 1}, {"weight": 1}, {"weight": 1}],
        group_id="TEST_HEDGE",
        group_label="테스트 경주",
        max_concurrency=3,
        slow_retry_enabled=True,
        slow_retry_remaining=1,
        slow_retry_progress_enabled=True,
        slow_retry_progress_threshold=50,
        slow_retry_tps_enabled=False,
        slow_retry_tps_threshold=5.0,
        slow_retry_condition_operator="and",
        stream_notify=None,
        invoke=invoke,
    )

    assert max_active <= 3
    assert (3, "duplicate") in attempts
    assert results[2]["winner"] == "duplicate"
    assert history_updates
    assert {item["status"] for item in history_updates[0].values()} == {
        "race_won", "race_lost"
    }


@pytest.mark.asyncio
async def test_call1_segments_and_call2_details_run_in_parallel_batches(monkeypatch):
    paragraphs = [f"Hana paragraph {index}." for index in range(10)]
    narrative = "\n\n".join(paragraphs)
    target_slotted = pipeline.insert_slots(narrative)
    call1_active = 0
    call1_max_active = 0
    detail_active = 0
    detail_max_active = 0
    call_names = []

    async def fake_call(task_key, messages, **kwargs):
        nonlocal call1_active, call1_max_active, detail_active, detail_max_active
        text = "\n".join(str(message.get("content") or "") for message in messages)
        metadata = pipeline.llm_service._stream_metadata_ctx.get({})
        call_names.append(str(metadata.get("call_name") or task_key))
        if task_key == "illustration_call1":
            call1_active += 1
            call1_max_active = max(call1_max_active, call1_active)
            try:
                await asyncio.sleep(0.02)
                assigned_match = re.search(
                    r"# ASSIGNED SEGMENT IDS\s*(\[[^\n]+\])",
                    text,
                )
                assert assigned_match
                assigned = json.loads(assigned_match.group(1))
                return json.dumps({
                    "reference_assignments": [],
                    "history_characters": [],
                    "current_characters": [{"name": "Hana", "confidence": 0.99}],
                    "wardrobe_events": [],
                    "unresolved_references": [],
                    "assigned": assigned,
                })
            finally:
                call1_active -= 1

        assert task_key == "illustration_call2"
        if "# GLOBAL CALL2 PLAN" in text:
            return json.dumps({
                "scene_plan": [
                    {
                        "plan_id": f"S{slot + 1:03d}",
                        "anchor_segment": f"C{slot + 1:03d}",
                        "source_segments": [f"C{slot + 1:03d}"],
                        "characters": [{
                            "name": "Hana",
                            "outfit_state": {
                                "body_state": "clothed",
                                "worn": ["school uniform"],
                                "removed": [],
                            },
                        }],
                        "scene_brief": f"Hana scene {slot}",
                    }
                    for slot in range(9)
                ],
                "keyvis": {
                    "camera": "full body",
                    "characters": [{
                        "name": "Hana",
                        "positive": "1girl, black hair, school uniform",
                        "negative": "lowres",
                        "outfit_state": {
                            "body_state": "clothed",
                            "worn": ["school uniform"],
                            "removed": [],
                        },
                    }],
                    "scene": "classroom",
                    "supplement": "daylight",
                },
            })
        assert "# ASSIGNED GLOBAL SCENE PLAN" in text
        plan_match = re.search(
            r"# ASSIGNED GLOBAL SCENE PLAN\s*(\[[\s\S]*?\])\s*\n\nExpand each plan",
            text,
        )
        assert plan_match
        plans = json.loads(plan_match.group(1))
        detail_active += 1
        detail_max_active = max(detail_max_active, detail_active)
        try:
            await asyncio.sleep(0.02)
            return _toon_for_slots([int(item["slot"]) for item in plans])
        finally:
            detail_active -= 1

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "parallel_call1_call2_test",
            "target_slotted": target_slotted,
            "chats": [
                {"role": "user", "data": "Continue."},
                {"role": "char", "data": narrative},
            ],
        },
        {
            "call1_parallel_enabled": True,
            "call1_parallel_max_concurrency": 3,
            "call2_parallel_enabled": True,
            "call2_parallel_max_concurrency": 3,
            "scene_min": 9,
            "scene_max": 9,
            "key_visual": True,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        "### Hana\n-default_outfit\nschool uniform",
        extra_costume="### Hana\n-default_outfit\nschool uniform",
        extra_names="Hana",
        backtranslate_names="Hana",
        history_plan={
            "state_before": {},
            "call1_history": [],
            "call2_fallback_history": [],
            "record_before": {"last_pipeline": {}},
        },
    )

    assert call1_max_active == 3
    assert detail_max_active == 3
    assert len(result["call2_detail_outputs"]) == 3
    assert len(result["items"]) == 10
    assert result["items"][0]["kind"] == "keyvis"
    assert [item["slot"] for item in result["items"][1:]] == list(range(9))
    reparsed = pipeline.parse_toon_plan(
        result["call2_output"],
        pipeline.merged_toggles({"scene_min": 9, "scene_max": 9}),
        "TEST-MERGED-CALL2",
    )
    assert len(reparsed) == 10
    assert [item["plan_id"] for item in reparsed[1:]] == [
        f"S{index:03d}" for index in range(1, 10)
    ]
    assert "CALL2-PLAN" in call_names
    assert sum(name.startswith("CALL2-DETAIL") for name in call_names) >= 3


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
    assert "Count, group, and focus tags" in rendered
    assert "belong only in `scene`" in rendered
    assert "keep exactly one visible head and one visible face" in rendered
    assert "Choose exactly one head orientation and one eye direction" in rendered
    assert "positive: 1girl" not in rendered


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
    assert defaults["call1_backtranslate_slow_retry_enabled"] is False
    assert defaults["call1_backtranslate_slow_retry_remaining"] == 1
    assert defaults["call1_backtranslate_slow_retry_progress_enabled"] is True
    assert defaults["call1_backtranslate_slow_retry_progress_threshold"] == 50
    assert defaults["call1_backtranslate_slow_retry_tps_enabled"] is False
    assert defaults["call1_backtranslate_slow_retry_tps_threshold"] == 5.0
    assert defaults["call1_backtranslate_slow_retry_condition_operator"] == "and"
    assert defaults["call1_backtranslate_failure_strategy"] == "fallback"
    assert pipeline.merged_toggles({
        "call1_backtranslate_max_concurrency": 0,
    })["call1_backtranslate_max_concurrency"] == 1
    assert pipeline.merged_toggles({
        "call1_backtranslate_max_concurrency": 99,
    })["call1_backtranslate_max_concurrency"] == 16
    assert pipeline.merged_toggles({
        "call1_backtranslate_slow_retry_remaining": 0,
        "call1_backtranslate_slow_retry_progress_threshold": 100,
    })["call1_backtranslate_slow_retry_remaining"] == 1
    assert pipeline.merged_toggles({
        "call1_backtranslate_slow_retry_remaining": 99,
        "call1_backtranslate_slow_retry_progress_threshold": 0,
    })["call1_backtranslate_slow_retry_remaining"] == 16
    assert pipeline.merged_toggles({
        "call1_backtranslate_slow_retry_progress_threshold": 100,
    })["call1_backtranslate_slow_retry_progress_threshold"] == 99
    assert pipeline.merged_toggles({
        "call1_backtranslate_slow_retry_progress_threshold": 0,
    })["call1_backtranslate_slow_retry_progress_threshold"] == 1
    assert pipeline.merged_toggles({
        "call1_backtranslate_slow_retry_tps_threshold": 0,
    })["call1_backtranslate_slow_retry_tps_threshold"] == 0.1
    assert pipeline.merged_toggles({
        "call1_backtranslate_slow_retry_tps_threshold": 5000,
    })["call1_backtranslate_slow_retry_tps_threshold"] == 1000.0
    assert pipeline.merged_toggles({
        "call1_backtranslate_slow_retry_condition_operator": "OR",
    })["call1_backtranslate_slow_retry_condition_operator"] == "or"
    assert pipeline.merged_toggles({
        "call1_backtranslate_slow_retry_condition_operator": "xor",
    })["call1_backtranslate_slow_retry_condition_operator"] == "and"


def test_call1_call2_parallel_defaults_and_clamps():
    defaults = pipeline.merged_toggles({})
    for prefix in ("call1_parallel", "call2_parallel"):
        assert defaults[f"{prefix}_enabled"] is True
        assert defaults[f"{prefix}_max_concurrency"] == 3
        assert defaults[f"{prefix}_slow_retry_enabled"] is False
        assert defaults[f"{prefix}_slow_retry_remaining"] == 1
        assert defaults[f"{prefix}_slow_retry_progress_enabled"] is True
        assert defaults[f"{prefix}_slow_retry_progress_threshold"] == 50
        assert defaults[f"{prefix}_slow_retry_tps_enabled"] is False
        assert defaults[f"{prefix}_slow_retry_tps_threshold"] == 5.0
        assert defaults[f"{prefix}_slow_retry_condition_operator"] == "and"
    assert "call1_parallel_chunk_size" not in defaults
    assert "call2_parallel_batch_size" not in defaults

    clamped = pipeline.merged_toggles({
        "call1_parallel_chunk_size": 0,
        "call1_parallel_max_concurrency": 99,
        "call1_parallel_slow_retry_remaining": 0,
        "call2_parallel_batch_size": 99,
        "call2_parallel_max_concurrency": 0,
        "call2_parallel_slow_retry_progress_threshold": 100,
        "call2_parallel_slow_retry_tps_threshold": 0,
        "call2_parallel_slow_retry_condition_operator": "xor",
    })
    assert "call1_parallel_chunk_size" not in clamped
    assert clamped["call1_parallel_max_concurrency"] == 16
    assert clamped["call1_parallel_slow_retry_remaining"] == 1
    assert "call2_parallel_batch_size" not in clamped
    assert clamped["call2_parallel_max_concurrency"] == 1
    assert clamped["call2_parallel_slow_retry_progress_threshold"] == 99
    assert clamped["call2_parallel_slow_retry_tps_threshold"] == 0.1
    assert clamped["call2_parallel_slow_retry_condition_operator"] == "and"
    assert pipeline.merged_toggles({
        "call1_backtranslate_failure_strategy": "retry_abort",
    })["call1_backtranslate_failure_strategy"] == "retry_abort"
    assert pipeline.merged_toggles({
        "call1_backtranslate_failure_strategy": "unknown",
    })["call1_backtranslate_failure_strategy"] == "fallback"


def test_call2_plan_batches_balance_across_available_detail_workers():
    scene_plan = [{"slot": index} for index in range(1, 12)]
    batches = pipeline._balanced_call2_scene_plan_batches(scene_plan, 3)
    assert [len(batch) for batch in batches] == [4, 4, 3]
    assert [item["slot"] for batch in batches for item in batch] == list(range(1, 12))

    eight_scene_batches = pipeline._balanced_call2_scene_plan_batches(scene_plan[:8], 3)
    assert [len(batch) for batch in eight_scene_batches] == [3, 3, 2]

    two_scene_batches = pipeline._balanced_call2_scene_plan_batches(scene_plan[:2], 3)
    assert [len(batch) for batch in two_scene_batches] == [1, 1]


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


def test_backtranslation_slot_protection_round_trips_exact_markers():
    source = "첫 문단.\n\n[Slot 7]\n\n둘째 문단.\n\n[Slot   11]"

    protected, markers = pipeline._protect_slot_markers(source)

    assert "[Slot" not in protected
    assert pipeline._PROTECTED_SLOT_TOKEN_RE.findall(protected) == [
        "__SLOT_0__",
        "__SLOT_1__",
    ]
    assert [token for token, _marker in markers] == ["__SLOT_0__", "__SLOT_1__"]
    restored, valid, reason = pipeline._restore_slot_markers(protected, markers)
    assert valid is True
    assert reason == ""
    assert restored == source


def test_backtranslation_slot_protection_rejects_reordered_tokens():
    source = "첫 문단.\n\n[Slot 0]\n\n둘째 문단.\n\n[Slot 1]"
    protected, markers = pipeline._protect_slot_markers(source)
    first, second = [token for token, _marker in markers]
    reordered = protected.replace(first, "__TEMP_SLOT__").replace(
        second, first
    ).replace("__TEMP_SLOT__", second)

    restored, valid, reason = pipeline._restore_slot_markers(reordered, markers)

    assert restored == reordered
    assert valid is False
    assert "보호 슬롯 토큰 불일치" in reason


def test_call1_position_merges_into_slotted_body_without_placeholder():
    slotted = "첫 문장.\n\n[Slot 0]\n\n둘째 문장."
    call1_output = """[Position]둘째 문장.[/Position]
[VisualSupplement]
창가의 빛이 얼굴을 비춘다.
[/VisualSupplement]"""

    merged = pipeline._merge_call1_output_into_slotted(slotted, call1_output)

    assert "__SLOT" not in merged
    assert "[Slot 0]" in merged
    assert "[Position]둘째 문장.[/Position]" in merged
    assert "[VisualSupplement]" in merged


def test_call1_position_mapping_can_cross_a_hidden_slot_boundary():
    slotted = "첫 문장.\n\n[Slot 0]\n\n둘째 문장."
    call1_output = """[Position]
첫 문장.

둘째 문장.
[/Position]
[VisualSupplement]
두 문장은 같은 장면이다.
[/VisualSupplement]"""

    merged = pipeline._merge_call1_output_into_slotted(slotted, call1_output)

    assert "[Position]첫 문장.\n\n[Slot 0]\n\n둘째 문장.[/Position]" in merged
    assert merged.count("[Slot 0]") == 1


def test_call1_merge_removes_unexpected_slots_from_llm_output(capsys):
    slotted = "첫 문장.\n\n[Slot 0]\n\n둘째 문장."
    call1_output = """[Position]둘째 문장.[/Position]
[VisualSupplement]
[Slot 99]
창가의 빛이 얼굴을 비춘다.
[/VisualSupplement]"""

    merged = pipeline._merge_call1_output_into_slotted(slotted, call1_output)

    assert pipeline._SLOT_MARKER_RE.findall(merged) == ["0"]
    assert "예상하지 못한 슬롯 마커" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_backtranslation_stream_events_include_queue_subtask_metadata(monkeypatch):
    source = "첫 문단.\n\n[Slot 0]\n\n둘째 문단.\n\n[Slot 1]\n\n셋째 문단.\n\n[Slot 2]"
    chunks = pipeline.split_backtranslation_chunks(source, 3)
    events = []

    async def fake_pipeline_call(
        call_name, messages, stream_notify=None, result_validator=None
    ):
        index = int(call_name.rsplit(" ", 1)[1].split("/", 1)[0])
        translated, _markers = pipeline._protect_slot_markers(chunks[index - 1])
        assert translated in messages[-1]["content"]
        assert "[Slot " not in messages[-1]["content"]
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
            token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
                messages[-1]["content"]
            )[0]
            return f"First paragraph.\n\n{token}"
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
async def test_backtranslation_protected_slot_mismatch_falls_back_to_original_chunk(monkeypatch):
    source = "장면.\n\n[Slot 7]"

    async def fake_pipeline_call(
        call_name, messages, stream_notify=None, result_validator=None
    ):
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        return "Scene.\n\n" + token.replace("_0__", "_1__")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        4,
    )

    assert translated == source
    assert statuses[0]["status"] == "fallback"
    assert "보호 슬롯 토큰 불일치" in statuses[0]["reason"]


@pytest.mark.asyncio
async def test_backtranslation_strict_strategy_uses_central_route_retry(monkeypatch):
    calls = []
    source = "장면.\n\n[Slot 3]"

    async def fake_pipeline_call(
        call_name, messages, stream_notify=None, result_validator=None
    ):
        calls.append(call_name)
        assert result_validator is not None
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        assert result_validator("Scene without protected token.")[0] is False
        successful = f"Scene.\n\n{token}"
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
        invalid = "Scene without protected token."
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


@pytest.mark.asyncio
async def test_backtranslation_slow_retry_duplicates_non_streaming_tail_and_uses_first_valid(monkeypatch):
    source = "빠른 문장.\n\n[Slot 0]\n\n느린 문장.\n\n[Slot 1]"
    calls = []
    history_updates = {}
    primary_cancelled = asyncio.Event()

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        stream_observer=None,
        history_id="",
    ):
        calls.append(call_name)
        index = int(re.search(r"BACKTRANSLATE (\d+)/", call_name).group(1))
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        if index == 1:
            return f"Fast sentence.\n\n{token}"
        if "느리다고? 다시해!" in call_name:
            return f"Duplicate wins.\n\n{token}"
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            primary_cancelled.set()
            raise

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    monkeypatch.setattr(
        pipeline.lighbd_service,
        "_update_lighbd_history_records",
        lambda updates: history_updates.update(updates) or len(updates),
    )
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        2,
        slow_retry_enabled=True,
        slow_retry_remaining=1,
        slow_retry_progress_threshold=50,
    )

    assert translated == (
        "Fast sentence.\n\n[Slot 0]\n\n"
        "Duplicate wins.\n\n[Slot 1]"
    )
    assert len(calls) == 3
    assert primary_cancelled.is_set()
    assert statuses[1]["hedged"] is True
    assert statuses[1]["winner"] == "duplicate"
    assert statuses[1]["requests"] == 2
    assert any(
        update["call_name"]
        == "CALL1-BACKTRANSLATE 2/2 [느리다고? 다시해! · 승리]"
        and update["status"] == "race_won"
        for update in history_updates.values()
    )
    assert any(
        update["call_name"]
        == "CALL1-BACKTRANSLATE 2/2 [원본 · 패배 · 진행률 0% (비스트리밍)]"
        and update["status"] == "race_lost"
        for update in history_updates.values()
    )


@pytest.mark.asyncio
async def test_backtranslation_slow_retry_marks_primary_as_winner(monkeypatch):
    source = "빠른 문장.\n\n[Slot 0]\n\n느린 문장.\n\n[Slot 1]"
    history_updates = {}
    duplicate_started = asyncio.Event()

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        stream_observer=None,
        history_id="",
    ):
        index = int(re.search(r"BACKTRANSLATE (\d+)/", call_name).group(1))
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        if index == 1:
            return f"Fast sentence.\n\n{token}"
        if "느리다고? 다시해!" in call_name:
            duplicate_started.set()
            await asyncio.Future()
        await duplicate_started.wait()
        return f"Primary wins.\n\n{token}"

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    monkeypatch.setattr(
        pipeline.lighbd_service,
        "_update_lighbd_history_records",
        lambda updates: history_updates.update(updates) or len(updates),
    )

    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        2,
        slow_retry_enabled=True,
        slow_retry_remaining=1,
        slow_retry_progress_threshold=50,
    )

    assert "Primary wins." in translated
    assert statuses[1]["winner"] == "primary"
    assert any(
        update["call_name"] == "CALL1-BACKTRANSLATE 2/2 [원본 · 승리]"
        and update["status"] == "race_won"
        for update in history_updates.values()
    )
    assert any(
        update["call_name"]
        == "CALL1-BACKTRANSLATE 2/2 [느리다고? 다시해! · 패배 · 진행률 0% (비스트리밍)]"
        and update["status"] == "race_lost"
        for update in history_updates.values()
    )


@pytest.mark.asyncio
async def test_backtranslation_slow_retry_uses_completed_ratio_for_stream_progress(monkeypatch):
    source = "빠른 문장.\n\n[Slot 0]\n\n진행 중인 문장.\n\n[Slot 1]"
    calls = []

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        stream_observer=None,
        history_id="",
    ):
        calls.append(call_name)
        index = int(re.search(r"BACKTRANSLATE (\d+)/", call_name).group(1))
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        if index == 1:
            return f"Fast sentence.\n\n{token}"
        assert stream_observer is not None
        stream_observer({
            "type": "stream_open",
            "stream_id": "streaming-primary",
            "partial_length": 0,
        })
        stream_observer({
            "type": "delta",
            "stream_id": "streaming-primary",
            "partial_length": 200,
        })
        await asyncio.sleep(0.01)
        return f"Primary completes.\n\n{token}"

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        2,
        slow_retry_enabled=True,
        slow_retry_remaining=1,
        slow_retry_progress_threshold=50,
    )

    assert "Primary completes." in translated
    assert len(calls) == 2
    assert "hedged" not in statuses[1]


@pytest.mark.asyncio
async def test_backtranslation_slow_retry_duplicates_stream_below_threshold(monkeypatch):
    source = "빠른 문장.\n\n[Slot 0]\n\n느린 문장.\n\n[Slot 1]"
    calls = []
    history_updates = {}

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        stream_observer=None,
        history_id="",
    ):
        calls.append(call_name)
        index = int(re.search(r"BACKTRANSLATE (\d+)/", call_name).group(1))
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        if index == 1:
            return f"Fast sentence.\n\n{token}"
        if "느리다고? 다시해!" in call_name:
            return f"Hedged stream wins.\n\n{token}"
        assert stream_observer is not None
        stream_observer({
            "type": "stream_open",
            "stream_id": "slow-stream",
            "partial_length": 0,
        })
        stream_observer({
            "type": "delta",
            "stream_id": "slow-stream",
            "partial_length": 1,
        })
        await asyncio.Future()

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    monkeypatch.setattr(
        pipeline.lighbd_service,
        "_update_lighbd_history_records",
        lambda updates: history_updates.update(updates) or len(updates),
    )
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        2,
        slow_retry_enabled=True,
        slow_retry_remaining=1,
        slow_retry_progress_threshold=50,
    )

    assert "Hedged stream wins." in translated
    assert len(calls) == 3
    assert statuses[1]["winner"] == "duplicate"
    assert any(
        "[원본 · 패배 · 진행률 " in update["call_name"]
        and "(비스트리밍)" not in update["call_name"]
        and update["status"] == "race_lost"
        for update in history_updates.values()
    )


@pytest.mark.asyncio
async def test_backtranslation_slow_retry_and_requires_progress_and_tps(monkeypatch):
    source = "빠른 문장.\n\n[Slot 0]\n\n느린 문장.\n\n[Slot 1]"
    calls = []

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        stream_observer=None,
        history_id="",
    ):
        calls.append(call_name)
        index = int(re.search(r"BACKTRANSLATE (\d+)/", call_name).group(1))
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        if index == 1:
            await asyncio.sleep(0.02)
            return f"Fast sentence.\n\n{token}"
        if "느리다고? 다시해!" in call_name:
            return f"Unexpected duplicate.\n\n{token}"
        assert stream_observer is not None
        stream_observer({
            "type": "stream_open",
            "stream_id": "and-primary",
            "partial_length": 0,
        })
        stream_observer({
            "type": "delta",
            "stream_id": "and-primary",
            "partial_length": 1,
        })
        await asyncio.sleep(0.04)
        return f"Primary completes.\n\n{token}"

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        2,
        slow_retry_enabled=True,
        slow_retry_remaining=1,
        slow_retry_progress_enabled=True,
        slow_retry_progress_threshold=50,
        slow_retry_tps_enabled=True,
        slow_retry_tps_threshold=1,
        slow_retry_condition_operator="and",
    )

    assert "Primary completes." in translated
    assert len(calls) == 2
    assert "hedged" not in statuses[1]


@pytest.mark.asyncio
async def test_backtranslation_slow_retry_or_accepts_progress_or_tps(monkeypatch):
    source = "빠른 문장.\n\n[Slot 0]\n\n느린 문장.\n\n[Slot 1]"
    calls = []

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        stream_observer=None,
        history_id="",
    ):
        calls.append(call_name)
        index = int(re.search(r"BACKTRANSLATE (\d+)/", call_name).group(1))
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        if index == 1:
            await asyncio.sleep(0.02)
            return f"Fast sentence.\n\n{token}"
        if "느리다고? 다시해!" in call_name:
            return f"OR duplicate wins.\n\n{token}"
        assert stream_observer is not None
        stream_observer({
            "type": "stream_open",
            "stream_id": "or-primary",
            "partial_length": 0,
        })
        stream_observer({
            "type": "delta",
            "stream_id": "or-primary",
            "partial_length": 1,
        })
        await asyncio.Future()

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        2,
        slow_retry_enabled=True,
        slow_retry_remaining=1,
        slow_retry_progress_enabled=True,
        slow_retry_progress_threshold=50,
        slow_retry_tps_enabled=True,
        slow_retry_tps_threshold=1,
        slow_retry_condition_operator="or",
    )

    assert "OR duplicate wins." in translated
    assert len(calls) == 3
    assert statuses[1]["hedged"] is True


@pytest.mark.asyncio
async def test_backtranslation_slow_retry_does_nothing_when_all_conditions_are_off(monkeypatch):
    source = "빠른 문장.\n\n[Slot 0]\n\n느린 문장.\n\n[Slot 1]"
    calls = []

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        stream_observer=None,
        history_id="",
    ):
        calls.append(call_name)
        index = int(re.search(r"BACKTRANSLATE (\d+)/", call_name).group(1))
        token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(
            messages[-1]["content"]
        )[0]
        if index == 1:
            return f"Fast sentence.\n\n{token}"
        await asyncio.sleep(0.01)
        return f"Primary completes.\n\n{token}"

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    translated, statuses = await pipeline.backtranslate_current_context(
        source,
        "Translate. {character_names}",
        "Hana",
        2,
        slow_retry_enabled=True,
        slow_retry_remaining=1,
        slow_retry_progress_enabled=False,
        slow_retry_tps_enabled=False,
        slow_retry_condition_operator="or",
    )

    assert "Primary completes." in translated
    assert len(calls) == 2
    assert "hedged" not in statuses[1]


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
            "speak_language": "한국어",
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
    assert speak_prompt.startswith("# OUTPUT LANGUAGE — HARD REQUIREMENT")
    assert "required output language: 한국어" in speak_prompt
    assert "SPEAK PROMPT" in speak_prompt
    assert "Add one #emotion tag" in speak_prompt
    assert "Allowed labels: joy; anger" in speak_prompt
    assert manga_mode == "manga"
    assert manga_prompt.startswith("# OUTPUT LANGUAGE — HARD REQUIREMENT")
    assert "MANGA PROMPT" in manga_prompt
    assert "CHARACTER REFERENCE" in manga_prompt
    assert "#emotion" not in manga_prompt


def test_call3_prompts_separate_internal_speaker_ids_from_in_story_address():
    prompts = pipeline.load_prompt_files()

    for prompt_key in ("call3_speak", "call3_manga"):
        prompt = prompts[prompt_key]
        assert "internal identifiers for machine-readable speaker attribution only" in prompt
        assert "use the exact matching identifier from the character roster" in prompt
        assert "never copy an internal roster identifier merely because it appears" in prompt
        assert "Infer any in-story name, nickname, title, kinship term" in prompt
        assert "Do not force a name, nickname, or direct address" in prompt
        assert "If the proper form of address is uncertain, omit it" in prompt


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


def test_call3_scene_selection_excludes_key_visual_and_keeps_only_scene_context():
    selected_slots, payload = pipeline.build_call3_scene_selection([
        {
            "kind": "keyvis",
            "slot": -1,
            "scene": "poster key visual",
            "characters": [{"name": "hana", "position": "center"}],
        },
        {
            "kind": "scene",
            "slot": 2,
            "scene": "quiet classroom",
            "camera": "close-up",
            "supplement": "sunset",
            "characters": [{"name": "hana", "position": "left"}],
        },
        {
            "kind": "scene",
            "slot": 7,
            "scene": "school hallway",
            "camera": "medium shot",
            "characters": [{"name": "minsu", "position": "right"}],
        },
    ])

    decoded = json.loads(payload)
    assert selected_slots == [2, 7]
    assert [scene["slot"] for scene in decoded["selected_scenes"]] == [2, 7]
    assert decoded["selected_scenes"][0]["characters"] == [
        {"name": "hana", "position": "left"}
    ]
    assert "poster key visual" not in payload
    assert '"slot": -1' not in payload


def test_call3_slot_coverage_requires_every_selected_slot_and_rejects_others(capsys):
    valid, reason = pipeline.validate_call3_slot_coverage(
        '[Scene slot=2]\nHana: "Ready."\n[Scene slot=7]\nMinsu: (Wait.)',
        [2, 7],
    )
    assert valid is True
    assert reason == ""

    valid, reason = pipeline.validate_call3_slot_coverage(
        '[Scene slot=2]\nHana: "Ready."\n[Scene slot=-1]\nHana: "Poster."',
        [2, 7],
    )
    assert valid is False
    assert "missing=[7]" in reason
    assert "unexpected=[-1]" in reason
    assert "CALL3 선택 slot 불일치" in capsys.readouterr().out

    valid, reason = pipeline.validate_call3_slot_coverage(
        '[Scene slot=7]\nMinsu: "Later."\n[Scene slot=2]\nHana: "First."',
        [2, 7],
    )
    assert valid is False
    assert "headers=[7, 2]" in reason


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
    assert "Never omit or skip a supplied selected scene." in manga
    assert "Never output slot -1." in manga


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
    assert "manga dialogue writer and balloon-style editor" in calls[1][0]["content"]
    assert "#normal" in calls[1][0]["content"]
    assert result["items"][0]["speak"] == 'Hana: "No way!" #burst'
    assert '[SPEAK]\nHana: "No way!" #burst' in result["items"][0]["raw_positive"]


@pytest.mark.asyncio
async def test_call3_uses_original_narrative_and_only_call2_selected_scene_slots(monkeypatch):
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages, kwargs))
        if task_key == "illustration_call1_backtranslate":
            body = messages[-1]["content"].split(
                "slot markers. Copy every token exactly once and in the same order.\n\n",
                1,
            )[1]
            return body.replace("원문의 첫 문장.", "Translated first sentence.").replace(
                "원문의 둘째 문장.",
                "Translated second sentence.",
            )
        if task_key == "illustration_call2":
            return """<lb-xnai>
keyvis:
  camera: portrait
  characters[1]:
    - name: hana
      positive: 1girl, hana, black hair
  scene: poster key visual
scenes[2]:
  - camera: close-up
    characters[1]:
      - name: hana
        positive: 1girl, hana, black hair
        position: left
    scene: first selected moment
    slot: 2
  - camera: medium shot
    characters[1]:
      - name: hana
        positive: 1girl, hana, black hair
        position: center
    scene: second selected moment
    slot: 5
</lb-xnai>"""

        assert task_key == "illustration_call3"
        request = messages[-1]["content"]
        assert "[Original narrative]" in request
        assert "원문의 첫 문장." in request
        assert "원문의 둘째 문장." in request
        assert "Translated first sentence." not in request
        assert "Translated second sentence." not in request
        assert "poster key visual" not in request
        assert '"slot": -1' not in request
        selected = json.loads(request.split("[Selected scenes from CALL2]\n", 1)[1].split(
            "\n\nLanguage:",
            1,
        )[0])
        assert [scene["slot"] for scene in selected["selected_scenes"]] == [2, 5]
        return """[Scene slot=2]
Hana: "첫 장면이야." #normal
[Scene slot=5]
Hana: (다음은 어떤 장면일까?) #thought_cloud"""

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call3_original_selected_slots_test",
            "target_slotted": (
                "원문의 첫 문장.\n\n[Slot 2]\n\n"
                "원문의 둘째 문장.\n\n[Slot 5]"
            ),
            "chats": [
                {"role": "user", "data": "장면을 보여줘."},
                {"role": "char", "data": "원문의 첫 문장.\n\n원문의 둘째 문장."},
            ],
        },
        {
            "call1_backtranslate_enabled": True,
            "call1_backtranslate_max_concurrency": 1,
            "call1_enabled": False,
            "call3_enabled": True,
            "speak_enabled": True,
            "call3_prompt_mode": "manga",
            "key_visual": True,
        },
        "### hana\n-Appearance: 1girl, black hair",
        extra_names="Hana",
        backtranslate_names="Hana",
    )

    assert [task_key for task_key, _messages, _kwargs in calls] == [
        "illustration_call1_backtranslate",
        "illustration_call2",
        "illustration_call3",
    ]
    assert [item["kind"] for item in result["items"]] == ["keyvis", "scene", "scene"]
    assert result["items"][0]["slot"] == -1
    assert result["items"][0]["speak"] == ""
    assert result["items"][1]["speak"] == 'Hana: "첫 장면이야." #normal'
    assert result["items"][2]["speak"] == "Hana: (다음은 어떤 장면일까?) #thought_cloud"


@pytest.mark.asyncio
async def test_call3_retries_once_when_a_selected_slot_is_missing(monkeypatch, capsys):
    call3_attempts = 0

    async def fake_call(task_key, messages, **kwargs):
        nonlocal call3_attempts
        if task_key == "illustration_call2":
            return """<lb-xnai>
scenes[2]:
  - camera: close-up
    characters[1]:
      - name: hana
        positive: 1girl, hana, black hair
    scene: first moment
    slot: 0
  - camera: medium shot
    characters[1]:
      - name: hana
        positive: 1girl, hana, black hair
    scene: second moment
    slot: 1
</lb-xnai>"""

        assert task_key == "illustration_call3"
        call3_attempts += 1
        if call3_attempts == 1:
            assert "result_validator" not in kwargs
            assert "required output language: 한국어" in messages[0]["content"]
            assert "Language: 한국어" in messages[-1]["content"]
            return '[Scene slot=0]\nHana: "첫 장면." #normal'

        assert "Required slots, in order: [0, 1]" in messages[-1]["content"]
        assert "Write every dialogue and thought in 한국어" in messages[-1]["content"]
        validator = kwargs.get("result_validator")
        assert validator is not None
        corrected = """[Scene slot=0]
Hana: "첫 장면." #normal
[Scene slot=1]
Hana: "둘째 장면." #normal"""
        assert validator(corrected) == (True, "")
        return corrected

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call3_missing_slot_retry_test",
            "target_slotted": "첫 장면.\n\n[Slot 0]\n\n둘째 장면.\n\n[Slot 1]",
            "chats": [
                {"role": "user", "data": "계속해."},
                {"role": "char", "data": "첫 장면.\n\n둘째 장면."},
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
        extra_names="Hana",
    )

    assert call3_attempts == 2
    assert result["call3_correction_used"] is True
    assert result["call3_initial_output"] == '[Scene slot=0]\nHana: "첫 장면." #normal'
    assert [item["speak"] for item in result["items"]] == [
        'Hana: "첫 장면." #normal',
        'Hana: "둘째 장면." #normal',
    ]
    captured = capsys.readouterr().out
    assert "missing=[1]" in captured
    assert "CALL3-CORRECTION" in captured
    assert "교정 호출 1회 실행" in captured


@pytest.mark.asyncio
async def test_call3_skips_dialogue_when_call2_selected_only_key_visual(monkeypatch, capsys):
    task_keys = []

    async def fake_call(task_key, messages, **kwargs):
        task_keys.append(task_key)
        assert task_key == "illustration_call2"
        return """<lb-xnai>
keyvis:
  camera: portrait
  characters[1]:
    - name: hana
      positive: 1girl, hana, black hair
  scene: poster key visual
</lb-xnai>"""

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call3_key_visual_only_test",
            "target_slotted": "포스터 장면.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "포스터를 보여줘."},
                {"role": "char", "data": "포스터 장면."},
            ],
        },
        {
            "call1_enabled": False,
            "call3_enabled": True,
            "speak_enabled": True,
            "call3_prompt_mode": "manga",
            "key_visual": True,
        },
        "### hana\n-Appearance: 1girl, black hair",
        extra_names="Hana",
    )

    assert task_keys == ["illustration_call2"]
    assert len(result["items"]) == 1
    assert result["items"][0]["kind"] == "keyvis"
    assert result["items"][0]["slot"] == -1
    assert result["items"][0]["speak"] == ""
    assert "일반 장면 slot이 없어 대사 생성 건너뜀: key_visuals=1" in capsys.readouterr().out


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
            token = pipeline._PROTECTED_SLOT_TOKEN_RE.findall(body)[0]
            assert "[Slot " not in body
            backtranslation_call_count = sum(
                1 for called_task, _messages in calls
                if called_task == "illustration_call1_backtranslate"
            )
            if backtranslation_call_count == 1:
                return f"Bbyakbbyak opens the door.\n\n{token}"
            return f"She looks inside.\n\n{token}\n\nThe room is quiet."
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
async def test_multi_char_layout_reorders_call2_characters_left_to_right(monkeypatch):
    descriptor = {
        "kind": "scene",
        "slot": 4,
        "camera": "wide shot",
        "scene": "two characters talking",
        "supplement": "soft light",
        "speak": 'Right: "hello"',
        "characters": [
            {"name": "Right", "positive": "green hair", "position": "on the right"},
            {"name": "Left", "positive": "red hair", "position": "on the left"},
        ],
    }
    calls = []

    async def fake_call(call_name, messages, stream_notify=None, result_validator=None, json_mode=False):
        calls.append((call_name, messages, json_mode))
        result = """{
          "background_prompt": "wide shot, classroom, soft light",
          "composition_prompt": "two distinct people, one listener on the left and one speaker on the right",
          "regions": [
            {"name":"Right","character_prompt":"green hair, waving","x":0.55,"y":0.1,"width":0.4,"height":0.8},
            {"name":"Left","character_prompt":"red hair, listening","x":0.05,"y":0.1,"width":0.4,"height":0.8}
          ]
        }"""
        assert result_validator(result)[0] is True
        return result

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)

    await pipeline.calculate_multi_char_layouts(
        [descriptor],
        "Return layout JSON",
        positive_note="cinematic color grading",
    )

    assert calls[0][0].startswith("MULTI-CHAR-MASK slot=4")
    assert calls[0][2] is True
    assert json.loads(calls[0][1][1]["content"])["positive_note"] == (
        "cinematic color grading"
    )
    assert [character["name"] for character in descriptor["characters"]] == ["Left", "Right"]
    assert descriptor["multi_char_layout"]["character_order"] == ["Left", "Right"]
    assert descriptor["multi_char_layout"]["background_prompt"] == (
        "wide shot, classroom, soft light"
    )
    assert descriptor["multi_char_layout"]["composition_prompt"].startswith(
        "two distinct people"
    )
    assert descriptor["multi_char_layout_request"]["slot"] == 4
    assert "composition_prompt" in descriptor["multi_char_layout_raw_response"]
    assert [
        region["character_prompt"]
        for region in descriptor["multi_char_layout"]["regions"]
    ] == ["red hair, listening", "green hair, waving"]


@pytest.mark.asyncio
async def test_multi_char_layout_rejects_unseparated_prompt(monkeypatch):
    descriptor = {
        "kind": "scene",
        "slot": 5,
        "camera": "wide shot",
        "scene": "rooftop",
        "supplement": "blue hour",
        "characters": [
            {"name": "Left", "positive": "grey hair, coat"},
            {"name": "Right", "positive": "black hair, crop top"},
        ],
    }

    async def fake_call(call_name, messages, stream_notify=None, result_validator=None, json_mode=False):
        result = """{
          "regions": [
            {"name":"Left","x":0.0,"y":0.0,"width":0.5,"height":1.0},
            {"name":"Right","x":0.5,"y":0.0,"width":0.5,"height":1.0}
          ]
        }"""
        valid, reason = result_validator(result)
        assert valid is False
        assert "background_prompt" in reason
        return result

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)

    await pipeline.calculate_multi_char_layouts([descriptor], "Return separated JSON")

    assert "multi_char_layout" not in descriptor
    assert "background_prompt" in descriptor["multi_char_layout_error"]


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
async def test_pipeline_llm_records_cancelled_hedge_in_lighbd_history(monkeypatch):
    records = []
    events = []
    call_started = asyncio.Event()
    messages = [{"role": "user", "content": "slow scene"}]

    async def fake_call(task_key, actual_messages, **kwargs):
        call_started.set()
        await asyncio.Future()

    async def fake_notify(event):
        events.append(event)

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)

    task = asyncio.create_task(
        pipeline._call_pipeline_llm(
            "CALL1-BACKTRANSLATE 2/2",
            messages,
            fake_notify,
            history_id="hedge-loser-id",
        )
    )
    await call_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(records) == 1
    assert records[0]["history_id"] == "hedge-loser-id"
    assert records[0]["status"] == "cancelled"
    assert "패배" in records[0]["error"]
    assert [event["type"] for event in events] == ["start", "cancelled"]


def test_lighbd_history_records_are_updated_by_history_id_with_backup(
    monkeypatch,
    tmp_path,
):
    history_path = tmp_path / "logs" / "lighbd_history.jsonl"
    history_path.parent.mkdir(parents=True)
    original_records = [
        {"history_id": "winner-id", "call_name": "original", "status": "ok"},
        {"history_id": "loser-id", "call_name": "retry", "status": "cancelled"},
    ]
    original_text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in original_records
    )
    history_path.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(
        pipeline.lighbd_service,
        "LIGHBD_HISTORY_PATH",
        str(history_path),
    )
    monkeypatch.setattr(pipeline.lighbd_service, "BASE_DIR", str(tmp_path))

    updated = pipeline.lighbd_service._update_lighbd_history_records({
        "winner-id": {"call_name": "원본 · 승리", "status": "race_won"},
        "loser-id": {"call_name": "재요청 · 패배", "status": "race_lost"},
    })

    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    backup_path = tmp_path / "요구사항" / "lighbd_history.jsonl.bak"
    assert updated == 2
    assert records[0]["status"] == "race_won"
    assert records[1]["status"] == "race_lost"
    assert backup_path.read_text(encoding="utf-8") == original_text


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
        {"call1_parallel_enabled": False, "call3_enabled": False, "speak_enabled": False, "key_visual": False},
        "### hana\n-Appearance: 1girl, black hair",
    )

    call1_text = "\n".join(message["content"] for message in calls[0])
    call2_text = "\n".join(message["content"] for message in calls[1])
    assert "[Slot 0]" not in call1_text
    assert "__SLOT_" not in call1_text
    assert "[Visual Content #01]" in call2_text
    assert "[Slot 0]" in call2_text
    assert "[Position]첫 문장.[/Position]" in call2_text
    assert "[CharacterBaseTags]" in call2_text
    assert "[Position]" not in result["enhanced_narrative"]
    assert result["items"][0]["slot"] == 0
    assert result["items"][0]["raw_positive"]


def test_call1_structured_assignments_preserve_slot_markers():
    current = "She enters the room.\n\nHer blue dress rustles."
    _rendered, segments = pipeline._segment_current_context(current)
    analysis = pipeline.parse_call1_analysis(
        json.dumps({
            "reference_assignments": [
                {
                    "segment_id": "C001",
                    "surface": "She",
                    "occurrence": 1,
                    "canonical_name": "Hana",
                    "replacement": "Hana",
                    "confidence": 0.99,
                },
                {
                    "segment_id": "C002",
                    "surface": "Her",
                    "occurrence": 1,
                    "canonical_name": "Hana",
                    "replacement": "Hana's",
                    "confidence": 0.99,
                },
            ],
            "history_characters": ["Hana"],
            "current_characters": [{"name": "Hana", "confidence": 0.99}],
            "wardrobe_events": [],
            "unresolved_references": [],
        }),
        current,
        segments,
        "Hana, Bob",
    )
    assert analysis is not None
    resolved, errors, variables = pipeline.apply_reference_assignments(
        current,
        segments,
        analysis["reference_assignments"],
    )
    assert not errors
    assert resolved == "Hana enters the room.\n\nHana's blue dress rustles."
    assert variables == {"__REF_001__": "Hana", "__REF_002__": "Hana's"}

    slotted, slotted_errors = pipeline.apply_reference_assignments_to_slotted(
        "She enters the room.\n\n[Slot 0]\n\nHer blue dress rustles.\n\n[Slot 1]",
        segments,
        analysis["reference_assignments"],
    )
    assert not slotted_errors
    assert "Hana enters" in slotted
    assert "Hana's blue dress" in slotted
    assert re.findall(r"\[Slot\s+\d+\]", slotted) == ["[Slot 0]", "[Slot 1]"]


def test_call1_recoverable_item_errors_do_not_require_balanced_fallback():
    current = "Sena enters the room.\n\nKai waits by the door."
    _rendered, segments = pipeline._segment_current_context(current)
    analysis = pipeline.parse_call1_analysis(
        json.dumps({
            "reference_assignments": [{
                "segment_id": "C001",
                "surface": "Sen\u0430",
                "canonical_name": "Sena",
                "replacement": "Sena",
                "confidence": 0.99,
            }],
            "history_characters": [],
            "current_characters": [{"name": "Kai", "confidence": 0.99}],
            "wardrobe_events": [{
                "segment_id": "C002",
                "character": "Kai",
                "operation": "remove",
                "items": ["coat"],
                "evidence": "Kai removes his coat.",
                "confidence": 0.99,
            }],
            "unresolved_references": [],
        }),
        current,
        segments,
        "Sena, Kai",
    )

    assert analysis is not None
    assert analysis["fallback_required"] is False
    assert analysis["fallback_errors"] == []
    assert analysis["reference_assignments"] == []
    assert analysis["wardrobe_events"] == []
    assert {item["name"] for item in analysis["current_characters"]} == {"Sena", "Kai"}
    assert any("지칭 원문 불일치로 폐기" in item for item in analysis["validation_warnings"])
    assert any("복장 변경 근거 불일치로 폐기" in item for item in analysis["validation_warnings"])
    assert any("서버가 보완: Sena" in item for item in analysis["validation_warnings"])


def test_call1_unresolved_reference_still_requires_balanced_fallback():
    current = "Kai waits by the door."
    _rendered, segments = pipeline._segment_current_context(current)
    analysis = pipeline.parse_call1_analysis(
        json.dumps({
            "reference_assignments": [],
            "history_characters": [],
            "current_characters": [{"name": "Kai", "confidence": 0.99}],
            "wardrobe_events": [],
            "unresolved_references": [{
                "segment_id": "C001",
                "surface": "someone",
            }],
        }),
        current,
        segments,
        "Kai",
    )

    assert analysis is not None
    assert analysis["fallback_required"] is True
    assert analysis["fallback_errors"] == ["미해결 지칭 1건"]


def test_call1_shard_scope_violations_are_warnings_but_conflicts_are_fatal():
    outside_value = {
        "assigned_segment_ids": ["C001"],
        "value": {
            "reference_assignments": [{
                "segment_id": "C002",
                "surface": "she",
                "canonical_name": "Hana",
            }],
            "history_characters": [],
            "current_characters": [{"name": "Hana", "confidence": 0.99}],
            "wardrobe_events": [],
            "unresolved_references": [],
        },
    }
    _merged, warnings, fallback_errors = pipeline._merge_call1_shard_values(
        [outside_value],
        ["C001", "C002"],
    )
    assert any("담당 밖 지칭 할당 폐기" in item for item in warnings)
    assert fallback_errors == []

    conflict_value = {
        "assigned_segment_ids": ["C001"],
        "value": {
            **outside_value["value"],
            "reference_assignments": [
                {
                    "segment_id": "C001",
                    "surface": "she",
                    "occurrence": 1,
                    "canonical_name": "Hana",
                },
                {
                    "segment_id": "C001",
                    "surface": "she",
                    "occurrence": 1,
                    "canonical_name": "Mina",
                },
            ],
        },
    }
    _merged, _warnings, fallback_errors = pipeline._merge_call1_shard_values(
        [conflict_value],
        ["C001"],
    )
    assert any("지칭 충돌" in item for item in fallback_errors)


def test_call3_scene_selection_contains_bounded_upper_and_lower_windows():
    slots, payload = pipeline.build_call3_scene_selection(
        [
            {"kind": "scene", "slot": 0, "scene": "first", "characters": []},
            {"kind": "scene", "slot": 1, "scene": "second", "characters": []},
        ],
        "위쪽 첫 대사\n\n[Slot 0]\n\n두 삽화 사이 대사\n\n[Slot 1]\n\n아래쪽 마지막 대사",
    )
    decoded = json.loads(payload)
    assert slots == [0, 1]
    first, second = decoded["selected_scenes"]
    assert first["upper_window"] == "위쪽 첫 대사"
    assert first["lower_window"] == "두 삽화 사이 대사"
    assert second["upper_window"] == "두 삽화 사이 대사"
    assert second["lower_window"] == "아래쪽 마지막 대사"
    assert "아래쪽 마지막 대사" not in first["lower_window"]


def test_call3_scene_selection_never_crosses_an_unselected_illustration_slot():
    slots, payload = pipeline.build_call3_scene_selection(
        [
            {"kind": "scene", "slot": 0, "scene": "first", "characters": []},
            {"kind": "scene", "slot": 2, "scene": "third", "characters": []},
        ],
        (
            "첫 장면 위\n\n[Slot 0]\n\n첫 장면 아래\n\n"
            "[Slot 1]\n\n선택되지 않은 삽화 아래\n\n"
            "[Slot 2]\n\n셋째 장면 아래"
        ),
    )
    decoded = json.loads(payload)
    assert slots == [0, 2]
    first, third = decoded["selected_scenes"]
    assert first["lower_window"] == "첫 장면 아래"
    assert "선택되지 않은 삽화 아래" not in first["lower_window"]
    assert third["upper_window"] == "선택되지 않은 삽화 아래"


@pytest.mark.asyncio
async def test_persistent_history_path_uses_compact_call2_and_updates_wardrobe(monkeypatch):
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        if task_key == "illustration_call1":
            return json.dumps({
                "reference_assignments": [{
                    "segment_id": "C001",
                    "surface": "She",
                    "occurrence": 1,
                    "canonical_name": "Hana",
                    "replacement": "Hana",
                    "confidence": 0.99,
                }],
                "history_characters": ["Hana"],
                "current_characters": [{"name": "Hana", "confidence": 0.99}],
                "wardrobe_events": [{
                    "segment_id": "C002",
                    "character": "Hana",
                    "operation": "remove",
                    "items": ["blue dress"],
                    "state_after": "nude",
                    "evidence": "Hana removes the blue dress.",
                    "confidence": 0.99,
                }],
                "unresolved_references": [],
            })
        assert task_key == "illustration_call2"
        request_text = "\n".join(message["content"] for message in messages)
        assert "very old fallback history" not in request_text
        assert "### Hana" in request_text
        assert "### Bob" not in request_text
        assert "Hana enters the room." in request_text
        assert "She enters the room." not in request_text
        assert "[Slot 0]" in request_text
        return """<lb-xnai>
scenes[1]:
  - camera: full body
    characters[1]:
      - name: Hana
        positive: 1girl, nude
        outfit_state:
          body_state: nude
          worn: []
          removed: [blue dress]
    scene: bedroom
    slot: 1
</lb-xnai>"""

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    state_before = {
        "hana": {
            "canonical_name": "Hana",
            "current_wardrobe": {
                "body_state": "clothed",
                "worn": ["blue dress"],
                "removed": [],
            },
        }
    }
    result = await pipeline.build_from_context(
        {
            "session_id": "persistent_history_pipeline_test",
            "target_slotted": (
                "She enters the room.\n\n[Slot 0]\n\n"
                "Hana removes the blue dress.\n\n[Slot 1]"
            ),
            "chats": [
                {"role": "user", "data": "Continue."},
                {
                    "role": "char",
                    "data": "She enters the room.\n\nHana removes the blue dress.",
                },
            ],
        },
        {
            "call1_enabled": True,
            "call1_parallel_enabled": False,
            "call3_enabled": False,
            "speak_enabled": False,
            "key_visual": False,
        },
        "World prompt\n\n### Hana\n-default_outfit\nblue dress\n\n### Bob\n-default_outfit\nblack suit",
        extra_costume="### Hana\n-default_outfit\nblue dress",
        extra_names="Hana, Bob",
        backtranslate_names="Hana, Bob",
        history_plan={
            "history_id": "hist_test",
            "operation": "append",
            "current_message_id": "msg_current",
            "current_context_hash": "current",
            "base_context_hash": "base",
            "state_before": state_before,
            "call1_history": [{"role": "char", "data": "Hana previously wore a blue dress."}],
            "call2_fallback_history": [{"role": "char", "data": "very old fallback history"}],
            "call3_fallback_history": [],
            "record_before": {"last_pipeline": {}},
        },
    )

    assert [task_key for task_key, _messages in calls] == [
        "illustration_call1",
        "illustration_call2",
    ]
    assert result["balanced_fallback_used"] is False
    assert result["enhanced_narrative"].startswith("Hana enters")
    wardrobe = result["character_states_after"]["hana"]["current_wardrobe"]
    assert wardrobe["body_state"] == "nude"
    assert wardrobe["worn"] == []
    assert "blue dress" in wardrobe["removed"]
    assert result["last_visual_by_character"]["Hana"]["outfit_state"]["body_state"] == "nude"


@pytest.mark.asyncio
async def test_persistent_call2_only_uses_bounded_history_and_visual_candidate(monkeypatch):
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        assert task_key == "illustration_call2"
        request_text = "\n".join(message["content"] for message in messages)
        assert "bounded past marker" in request_text
        assert "### Hana" in request_text
        assert "She waits by the door." in request_text
        return """<lb-xnai>
scenes[1]:
  - camera: full body
    characters[1]:
      - name: Hana
        positive: 1girl, white shirt, black skirt
        outfit_state:
          body_state: clothed
          worn: [white shirt, black skirt]
          removed: []
    scene: hallway
    slot: 0
</lb-xnai>"""

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "persistent_call2_only_test",
            "target_slotted": "She waits by the door.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Wait there."},
                {"role": "char", "data": "She waits by the door."},
            ],
        },
        {
            "call1_backtranslate_enabled": False,
            "call1_enabled": False,
            "call3_enabled": False,
            "speak_enabled": False,
            "key_visual": False,
        },
        "### Hana\n-default_outfit\nwhite shirt, black skirt",
        extra_names="Hana",
        backtranslate_names="Hana",
        history_plan={
            "history_id": "hist_call2_only",
            "operation": "append",
            "current_message_id": "msg_current",
            "current_context_hash": "current",
            "base_context_hash": "base",
            "state_before": {},
            "call1_history": [],
            "call2_fallback_history": [{"role": "char", "data": "bounded past marker"}],
            "call3_fallback_history": [],
            "record_before": {"last_pipeline": {}},
        },
    )

    assert [task_key for task_key, _messages in calls] == ["illustration_call2"]
    assert result["balanced_fallback_used"] is True
    assert result["enhanced_narrative"] == "She waits by the door."
    state = result["character_states_after"]["hana"]
    assert state["current_wardrobe"]["source"] == "call2_visual_candidate"
    assert state["current_wardrobe"]["worn"] == ["white shirt", "black skirt"]


@pytest.mark.asyncio
async def test_persistent_history_recovers_missing_prior_wardrobe_with_balanced_fallback(monkeypatch):
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        if task_key == "illustration_call1":
            return json.dumps({
                "reference_assignments": [],
                "history_characters": ["Hana"],
                "current_characters": [{"name": "Hana", "confidence": 0.99}],
                "wardrobe_events": [],
                "unresolved_references": [],
            })
        assert task_key == "illustration_call2"
        request_text = "\n".join(message["content"] for message in messages)
        assert "past wardrobe recovery marker" in request_text
        assert "### Hana" in request_text
        assert "### Bob" in request_text
        return """<lb-xnai>
scenes[1]:
  - camera: full body
    characters[1]:
      - name: Hana
        positive: 1girl, red cardigan, pleated skirt
        outfit_state:
          body_state: clothed
          worn: [red cardigan, pleated skirt]
          removed: []
    scene: classroom
    slot: 0
</lb-xnai>"""

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "persistent_missing_state_recovery",
            "target_slotted": "Hana looks outside.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Continue."},
                {"role": "char", "data": "Hana looks outside."},
            ],
        },
        {
            "call1_backtranslate_enabled": False,
            "call1_enabled": True,
            "call3_enabled": False,
            "speak_enabled": False,
            "key_visual": False,
        },
        (
            "### Hana\n-default_outfit\nred cardigan, pleated skirt\n\n"
            "### Bob\n-default_outfit\nblack suit"
        ),
        extra_costume="### Hana\n-default_outfit\nred cardigan, pleated skirt",
        extra_names="Hana, Bob",
        backtranslate_names="Hana, Bob",
        history_plan={
            "history_id": "hist_missing_state",
            "operation": "new",
            "current_message_id": "msg_current",
            "current_context_hash": "current",
            "base_context_hash": "base",
            "state_before": {},
            "call1_history": [{"role": "char", "data": "past wardrobe recovery marker: Hana arrived."}],
            "call2_fallback_history": [{"role": "char", "data": "past wardrobe recovery marker"}],
            "call3_fallback_history": [],
            "record_before": {"last_pipeline": {}},
        },
    )

    assert [task_key for task_key, _messages in calls] == [
        "illustration_call1",
        "illustration_call2",
    ]
    assert result["balanced_fallback_used"] is True
    state = result["character_states_after"]["hana"]
    assert state["current_wardrobe"]["source"] == "call2_visual_candidate"
    assert state["current_wardrobe"]["worn"] == ["red cardigan", "pleated skirt"]


@pytest.mark.asyncio
async def test_persistent_backtranslation_off_keeps_call1_call2_call3_combination(monkeypatch):
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        if task_key == "illustration_call1":
            return json.dumps({
                "reference_assignments": [{
                    "segment_id": "C001",
                    "surface": "She",
                    "occurrence": 1,
                    "canonical_name": "Hana",
                    "replacement": "Hana",
                    "confidence": 0.99,
                }],
                "history_characters": ["Hana"],
                "current_characters": [{"name": "Hana", "confidence": 0.99}],
                "wardrobe_events": [],
                "unresolved_references": [],
            })
        if task_key == "illustration_call2":
            return """<lb-xnai>
scenes[1]:
  - camera: medium shot
    characters[1]:
      - name: Hana
        positive: 1girl, blue dress
        outfit_state:
          body_state: clothed
          worn: [blue dress]
          removed: []
    scene: hallway
    slot: 0
</lb-xnai>"""
        assert task_key == "illustration_call3"
        request_text = "\n".join(message["content"] for message in messages)
        assert "[Original narrative]\nHana waits by the door." in request_text
        assert "She waits by the door." not in request_text
        return '[Scene slot=0]\nHana: "I will wait." #normal'

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "persistent_no_backtranslation_all_calls",
            "target_slotted": "She waits by the door.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Wait there."},
                {"role": "char", "data": "She waits by the door."},
            ],
        },
        {
            "call1_backtranslate_enabled": False,
            "call1_enabled": True,
            "call3_enabled": True,
            "speak_enabled": True,
            "key_visual": False,
        },
        "### Hana\n-default_outfit\nblue dress",
        extra_names="Hana",
        backtranslate_names="Hana",
        history_plan={
            "history_id": "hist_no_backtranslation",
            "operation": "append",
            "current_message_id": "msg_current",
            "current_context_hash": "current",
            "base_context_hash": "base",
            "state_before": {
                "hana": {
                    "canonical_name": "Hana",
                    "current_wardrobe": {
                        "body_state": "clothed",
                        "worn": ["blue dress"],
                        "removed": [],
                    },
                },
            },
            "call1_history": [{"role": "char", "data": "Hana wore a blue dress."}],
            "call2_fallback_history": [{"role": "char", "data": "unused call2 fallback"}],
            "call3_fallback_history": [{"role": "char", "data": "unused call3 fallback"}],
            "record_before": {"last_pipeline": {}},
        },
    )

    assert [task_key for task_key, _messages in calls] == [
        "illustration_call1",
        "illustration_call2",
        "illustration_call3",
    ]
    assert result["balanced_fallback_used"] is False
    assert result["items"][0]["speak"] == 'Hana: "I will wait." #normal'


@pytest.mark.asyncio
async def test_persistent_call1_off_keeps_call2_call3_with_separate_bounded_histories(monkeypatch):
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        request_text = "\n".join(message["content"] for message in messages)
        if task_key == "illustration_call2":
            assert "call2 bounded marker" in request_text
            assert "call3 bounded marker" not in request_text
            return """<lb-xnai>
scenes[1]:
  - camera: medium shot
    characters[1]:
      - name: Hana
        positive: 1girl, blue dress
        outfit_state:
          body_state: clothed
          worn: [blue dress]
          removed: []
    scene: hallway
    slot: 0
</lb-xnai>"""
        assert task_key == "illustration_call3"
        assert "call3 bounded marker" in request_text
        assert "[Original narrative]\nShe waits by the door." in request_text
        return '[Scene slot=0]\nHana: "Still here." #normal'

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "persistent_call1_off_call3_on",
            "target_slotted": "She waits by the door.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Wait there."},
                {"role": "char", "data": "She waits by the door."},
            ],
        },
        {
            "call1_backtranslate_enabled": False,
            "call1_enabled": False,
            "call3_enabled": True,
            "speak_enabled": True,
            "key_visual": False,
        },
        "### Hana\n-default_outfit\nblue dress",
        extra_names="Hana",
        history_plan={
            "history_id": "hist_call1_off_call3_on",
            "operation": "append",
            "current_message_id": "msg_current",
            "current_context_hash": "current",
            "base_context_hash": "base",
            "state_before": {},
            "call1_history": [],
            "call2_fallback_history": [{"role": "char", "data": "call2 bounded marker"}],
            "call3_fallback_history": [{"role": "char", "data": "call3 bounded marker"}],
            "record_before": {"last_pipeline": {}},
        },
    )

    assert [task_key for task_key, _messages in calls] == [
        "illustration_call2",
        "illustration_call3",
    ]
    assert result["balanced_fallback_used"] is True
    assert result["enhanced_narrative"] == "She waits by the door."
    assert result["items"][0]["speak"] == 'Hana: "Still here." #normal'
