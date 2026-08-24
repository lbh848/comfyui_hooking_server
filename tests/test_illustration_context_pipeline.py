import ast
import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import illustration_context_pipeline as pipeline


@pytest.fixture(autouse=True)
def _isolate_lighbd_history_writes(monkeypatch):
    """파이프라인 단위 테스트가 운영 LLM 이력 파일을 수정하지 않게 한다."""
    monkeypatch.setattr(
        pipeline.lighbd_service,
        "_log_lighbd_history",
        lambda _record: None,
    )


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


def _toon_without_named_characters(slot=0):
    return (
        "<lb-xnai>\n"
        "scenes[1]:\n"
        "  - camera: wide shot, straight-on\n"
        "    characters: []\n"
        "    scene: interior, classroom, students, teacher, warm afternoon light\n"
        f"    slot: {slot}\n"
        "    supplement: Students rise while the teacher exits through a wooden door.\n"
        "</lb-xnai>"
    )


def _call_name(task_key):
    metadata = pipeline.llm_service._stream_metadata_ctx.get({})
    return str(metadata.get("call_name") or task_key)


def _authority_audit_response(
    messages,
    *,
    authority_exceptions=None,
    forbidden_additions=None,
    conflicts=None,
    required_additions=None,
    scene_additions=None,
    camera_replacement="",
):
    request = str(messages[-1].get("content") or "")
    entries = json.loads(request.split("# AUDIT ENTRIES\n", 1)[1])
    return json.dumps({
        "entries": [{
            "id": int(entry["id"]),
            "authority_exceptions": list(authority_exceptions or []),
            "forbidden_additions": list(forbidden_additions or []),
            "conflicts": list(conflicts or []),
            "required_additions": list(required_additions or []),
            "scene_additions": list(scene_additions or []),
            "camera_replacement": str(camera_replacement or ""),
        } for entry in entries],
    })


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
    assert pipeline.parse_easy_edit_request(
        pipeline.EASY_EDIT_PREFIX + "\n" + json.dumps({
            "session_id": session_id,
            "slot": 0,
            "direction": "배경을 밤으로 바꿔줘",
        }, ensure_ascii=False)
    ) == {
        "session_id": session_id,
        "slot": 0,
        "direction": "배경을 밤으로 바꿔줘",
    }
    assert pipeline.parse_easy_edit_request(
        pipeline.EASY_EDIT_PREFIX + "\n" + json.dumps({
            "session_id": session_id,
            "slot": 0,
            "direction": " ",
        })
    ) is None


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
        pipeline.merged_toggles({"output_count_min": 5, "output_count_max": 5}),
        slotted,
    )

    assert reason == ""
    assert [item["plan_id"] for item in plan["scene_plan"]] == [
        "S001", "S002", "S003", "S004", "S005"
    ]
    assert [item["slot"] for item in plan["scene_plan"]] == [0, 1, 2, 3, 4]
    assert plan["keyvis_descriptor"]["kind"] == "keyvis"
    assert plan["keyvis_descriptor"]["slot"] == -1


def test_call2_plan_trims_overcount_via_diversity_maximin():
    """PLAN이 maximum 초과 반환하면 유사도 maximin으로 maximum개만 남긴다.

    비슷한 brief/캐릭터의 연속 장면(slot 1, 4는 slot 0과 거의 동일한 묘사)은
    유사도가 높아 잘리고, 서로 다른 장면이 생존한다. 정상(max 이하)일 때는
    PLAN 결과를 그대로 쓴다.
    """
    slotted = "\n\n".join(
        ["첫 문단."] + [f"[Slot {index}]\n\n문단 {index + 2}." for index in range(5)]
    )
    raw = json.dumps({
        "scene_plan": [
            {
                "slot": 0,
                "source_segments": ["C001"],
                "characters": ["Hana"],
                "scene_brief": "Hana sits at the desk reading a thick book",
            },
            {
                "slot": 1,
                "source_segments": ["C002"],
                "characters": ["Hana"],
                "scene_brief": "Hana sits at the desk reading a thick book closely",
            },
            {
                "slot": 2,
                "source_segments": ["C003"],
                "characters": ["Hana"],
                "scene_brief": "Hana walks alone through a sunny garden",
            },
            {
                "slot": 3,
                "source_segments": ["C004"],
                "characters": ["Hana", "Mina"],
                "scene_brief": "Hana talks with her friend Mina by the window",
            },
            {
                "slot": 4,
                "source_segments": ["C005"],
                "characters": ["Hana"],
                "scene_brief": "Hana sits at the desk reading a thick book again",
            },
        ],
        "keyvis": None,
        "keyvis_plan": None,
    }, ensure_ascii=False)

    plan, reason = pipeline.parse_call2_plan(
        raw,
        pipeline.merged_toggles({
            "output_count_min": 1,
            "output_count_max": 3,
            "key_visual": False,
        }),
        slotted,
    )

    assert reason == ""
    kept_slots = [item["slot"] for item in plan["scene_plan"]]
    assert len(kept_slots) == 3
    assert len(set(kept_slots)) == 3
    # 시드(slot 0)는 생존, 유사 연속 beat(slot 1, 4)는 잘린다.
    assert 0 in kept_slots
    assert 1 not in kept_slots
    assert 4 not in kept_slots


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


def test_segment_slot_map_excludes_only_unmatched_segment_and_resynchronizes():
    slotted = (
        "첫 문단.\n\n[Slot 0]\n\n"
        "둘째 원문.\n\n세부 원문.\n\n[Slot 1]\n\n"
        "마지막 문단."
    )
    segments = {
        "C001": {"text": "첫 문단."},
        "C002": {"text": "둘째 가공문과 세부 가공문."},
        "C003": {"text": "마지막 문단."},
    }

    mapping, annotated, reason = pipeline.build_segment_slot_map(slotted, segments)

    assert mapping == {"C001": 0, "C003": 1}
    assert "[C001 slot=0]" in annotated
    assert "[C002" not in annotated
    assert "[C003 slot=1]" in annotated
    assert "mapped=2/3" in reason
    assert "excluded=['C002']" in reason


def test_segment_slot_map_returns_empty_only_when_no_segment_can_be_mapped():
    slotted = "원문 하나.\n\n[Slot 0]\n\n원문 둘."
    segments = {
        "C001": {"text": "가공문 하나."},
        "C002": {"text": "가공문 둘."},
        "C003": {"text": "가공문 셋."},
    }

    mapping, annotated, reason = pipeline.build_segment_slot_map(slotted, segments)

    assert mapping == {}
    assert annotated == ""
    assert "mapped=0/3" in reason
    assert "excluded=['C001', 'C002', 'C003']" in reason


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
            "output_count_min": 1,
            "output_count_max": 1,
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


def test_call2_plan_accepts_compact_server_derived_fields_and_keyvis_plan():
    raw = json.dumps({
        "scene_plan": [{
            "anchor_segment": "C002",
            "characters": ["Hana"],
            "scene_brief": "Hana pauses beside the classroom door",
        }],
        "keyvis_plan": {
            "characters": ["Hana"],
            "scene_brief": "Hana framed as the emotional center of the school day",
        },
    }, ensure_ascii=False)

    plan, reason = pipeline.parse_call2_plan(
        raw,
        pipeline.merged_toggles({"output_count_min": 1, "output_count_max": 1}),
        "첫 문단.\n\n[Slot 0]\n\n둘째 문단.\n\n[Slot 1]",
        segment_slot_map={"C001": 0, "C002": 1},
    )

    assert reason == ""
    assert plan["scene_plan"][0]["plan_id"] == "S001"
    assert plan["scene_plan"][0]["slot"] == 1
    assert plan["scene_plan"][0]["source_segments"] == ["C002"]
    assert plan["scene_plan"][0]["planned_outfits"] == {}
    assert plan["keyvis_descriptor"] is None
    assert plan["keyvis_plan"] == {
        "characters": ["Hana"],
        "scene_brief": "Hana framed as the emotional center of the school day",
    }


def test_call2_plan_accepts_scene_without_named_tracked_characters(capsys):
    raw = json.dumps({
        "scene_plan": [{
            "anchor_segment": "C001",
            "characters": [],
            "scene_brief": (
                "Wide classroom establishing shot with students rising and the teacher exiting"
            ),
        }],
        "keyvis_plan": None,
    })

    plan, reason = pipeline.parse_call2_plan(
        raw,
        pipeline.merged_toggles({
            "output_count_min": 1,
            "output_count_max": 1,
            "key_visual": False,
        }),
        "Lunch bell rings.\n\n[Slot 0]",
        segment_slot_map={"C001": 0},
    )

    assert reason == ""
    assert plan["scene_plan"][0]["characters"] == []
    assert plan["scene_plan"][0]["planned_outfits"] == {}
    assert "이름 있는 추적 캐릭터가 없는 장면 수용" in capsys.readouterr().out


def test_call2_plan_repairs_duplicate_server_slot_without_reroll(capsys):
    raw = json.dumps({
        "scene_plan": [
            {
                "anchor_segment": "C041",
                "characters": ["Rito"],
                "scene_brief": "Rito hides behind a book while watching carefully",
            },
            {
                "anchor_segment": "C042",
                "characters": ["Rito"],
                "scene_brief": "Rito points toward the examination room",
            },
        ],
    })

    plan, reason = pipeline.parse_call2_plan(
        raw,
        pipeline.merged_toggles({
            "output_count_min": 2,
            "output_count_max": 2,
            "key_visual": False,
        }),
        "Before.\n\n[Slot 40]\n\nNear.\n\n[Slot 41]\n\nAfter.\n\n[Slot 42]",
        segment_slot_map={"C041": 41, "C042": 41},
    )

    assert reason == ""
    assert [item["anchor_segment"] for item in plan["scene_plan"]] == [
        "C041",
        "C042",
    ]
    assert [item["slot"] for item in plan["scene_plan"]] == [40, 41]
    output = capsys.readouterr().out
    assert "중복 slot 로컬 보정" in output
    assert "중복 slot 권위 위치 유지" in output


def test_call2_plan_drops_only_unplaceable_earlier_duplicate(capsys):
    raw = json.dumps({
        "scene_plan": [
            {
                "anchor_segment": "C041",
                "characters": ["Rito"],
                "scene_brief": "Rito hides behind a book",
            },
            {
                "anchor_segment": "C042",
                "characters": ["Rito"],
                "scene_brief": "Rito points toward the examination room",
            },
        ],
    })

    plan, reason = pipeline.parse_call2_plan(
        raw,
        pipeline.merged_toggles({
            "output_count_min": 2,
            "output_count_max": 2,
            "key_visual": False,
        }),
        "Only insertion point.\n\n[Slot 41]",
        segment_slot_map={"C041": 41, "C042": 41},
    )

    assert reason == ""
    assert [item["anchor_segment"] for item in plan["scene_plan"]] == ["C042"]
    assert [item["slot"] for item in plan["scene_plan"]] == [41]
    assert "중복 slot 장면 제외" in capsys.readouterr().out


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


def test_scene_plan_carries_literal_wardrobe_change_as_natural_continuity():
    plans = [{
        "plan_id": "S001",
        "slot": 4,
        "anchor_segment": "C002",
        "characters": ["Sato"],
        "scene_brief": "Sato leans forward during the ongoing intimate moment.",
    }]
    events = [{
        "segment_id": "C002",
        "character": "Sato",
        # Deliberately coarse/wrong hints reproduce the original failure mode.
        "operation": "open",
        "wardrobe_change": (
            "He pulled down his pants and underwear, leaving his penis exposed."
        ),
        "state_after": "partial",
        "evidence": (
            "He pulled down his pants and underwear.\n"
            "His penis remained fully visible."
        ),
    }]

    bound = pipeline.bind_scene_plan_wardrobes(
        plans,
        ["C001", "C002"],
        {},
        [{"name": "Sato", "confidence": 1.0}],
        events,
        "message-1",
        default_outfits={"Sato": ["white shirt", "blue pants", "underwear"]},
    )

    note = bound[0]["continuity_note"]
    assert note.startswith("By this point in the story")
    assert "pulled down his pants and underwear" in note
    assert "penis remained fully visible" in note
    assert bound[0]["_continuity_characters"] == ["Sato"]


def test_downstream_call2_plan_handoff_omits_internal_wardrobe_structure():
    public = pipeline._public_call2_scene_plan({
        "plan_id": "S001",
        "slot": 4,
        "anchor_segment": "C002",
        "characters": ["Sato"],
        "scene_brief": "Sato leans forward while his lowered clothing remains visible.",
        "continuity_note": "Sato pulled down his pants and underwear.",
        "wardrobe_snapshot": {
            "Sato": {
                "body_state": "partial",
                "worn": ["white shirt", "blue pants"],
                "removed": [],
            },
        },
        "wardrobe_sources": {"Sato": "default_base_plus_sparse_history"},
        "_continuity_characters": ["Sato"],
    })

    assert public == {
        "slot": 4,
        "characters": ["Sato"],
        "scene_brief": "Sato leans forward while his lowered clothing remains visible.",
        "continuity_note": "Sato pulled down his pants and underwear.",
    }


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


def test_call2_detail_preserves_contextual_outfit_resolution_until_audit():
    output = """<lb-xnai>
scenes[1]:
  - camera: full body, straight-on
    characters[1]:
      - name: Sato
        positive: boy, white shirt, bottomless, penis, pants down
        outfit_state:
          body_state: bottomless
          worn: [white shirt]
          removed: [blue pants, underwear]
    scene: 1boy, interior, bedroom, nsfw
    slot: 4
</lb-xnai>"""

    descriptors, reason = pipeline._parse_call2_detail_output(
        output,
        pipeline.merged_toggles({"key_visual": False}),
        [4],
        ["S001"],
        "TEST-CALL2-DETAIL-NATURAL-CONTINUITY",
        assigned_wardrobes_by_slot={
            4: {
                "Sato": {
                    "body_state": "partial",
                    "worn": ["white shirt", "blue pants", "underwear"],
                    "removed": [],
                },
            },
        },
        assigned_characters_by_slot={4: ["Sato"]},
        assigned_scene_context_by_slot={
            4: {
                "scene_brief": "Sato's explicit lower-body exposure is visible.",
                "continuity_note": (
                    "Sato pulled down his pants and underwear, leaving his penis exposed."
                ),
                "continuity_characters": ["Sato"],
            },
        },
    )

    assert reason == ""
    character = descriptors[0]["characters"][0]
    assert character["outfit_state"] == {
        "body_state": "bottomless",
        "worn": ["white shirt"],
        "removed": ["blue pants", "underwear"],
    }
    assert "pants and underwear" in descriptors[0]["continuity_note"]


def test_call2_detail_accepts_empty_characters_for_characterless_plan():
    descriptors, reason = pipeline._parse_call2_detail_output(
        _toon_without_named_characters(4),
        pipeline.merged_toggles({"key_visual": False}),
        [4],
        ["S021"],
        "TEST-CALL2-DETAIL-NO-NAMED-CHARACTERS",
        assigned_characters_by_slot={4: []},
    )

    assert reason == ""
    assert len(descriptors) == 1
    assert descriptors[0]["slot"] == 4
    assert descriptors[0]["characters"] == []


def test_call2_detail_still_requires_characters_for_named_plan():
    descriptors, reason = pipeline._parse_call2_detail_output(
        _toon_without_named_characters(4),
        pipeline.merged_toggles({"key_visual": False}),
        [4],
        ["S021"],
        "TEST-CALL2-DETAIL-NAMED-CHARACTER-REQUIRED",
        assigned_characters_by_slot={4: ["Hana"]},
    )

    assert descriptors == []
    assert "이름 있는 PLAN 캐릭터가 누락됨" in reason


def test_call2_detail_repairs_sparse_wardrobe_from_plan_snapshot():
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

    assert reason == ""
    assert descriptors[0]["characters"][0]["outfit_state"] == {
        "body_state": "clothed",
        "worn": ["blue dress"],
        "removed": [],
    }


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


def test_call2_detail_recovers_one_unnamed_character_from_plan(capsys):
    detail_output = _toon_for_slots([5]).replace("      - name: Hana\n", "      - positive: 1girl, black hair\n").replace(
        "        positive: 1girl, black hair\n",
        "",
        1,
    )

    descriptors, reason = pipeline._parse_call2_detail_output(
        detail_output,
        pipeline.merged_toggles({"key_visual": False}),
        [5],
        ["S001"],
        "TEST-CALL2-DETAIL-MISSING-NAME",
        {
            5: {
                "Masachika": {
                    "body_state": "clothed",
                    "worn": ["school uniform"],
                    "removed": [],
                },
            },
        },
    )

    assert reason == ""
    assert descriptors[0]["characters"][0]["name"] == "Masachika"
    assert "PLAN으로 단일 누락 캐릭터 이름 복구" in capsys.readouterr().out


def test_call2_detail_discards_only_slot_with_character_mismatch(capsys):
    detail_output = _toon_for_slots([4, 9])
    slot_nine_marker = (
        "    scene: classroom\n"
        "    plan_id: S010\n"
        "    slot: 9"
    )
    detail_output = detail_output.replace(
        slot_nine_marker,
        "      - name: Alisa\n"
        "        positive: 1girl, silver hair\n"
        "        negative: lowres\n"
        "        outfit_state:\n"
        "          body_state: clothed\n"
        "          worn: [school uniform]\n"
        "          removed: []\n"
        + slot_nine_marker,
    )
    school_uniform = {
        "body_state": "clothed",
        "worn": ["school uniform"],
        "removed": [],
    }

    descriptors, reason = pipeline._parse_call2_detail_output(
        detail_output,
        pipeline.merged_toggles({"key_visual": False}),
        [4, 9],
        ["S001", "S002"],
        "TEST-CALL2-DETAIL-CHARACTER-DISCARD",
        {
            4: {"Hana": school_uniform},
            9: {
                "Hana": school_uniform,
                "Alisa": school_uniform,
                "Maria": school_uniform,
            },
        },
    )

    assert reason == ""
    assert [item["slot"] for item in descriptors] == [4]
    output = capsys.readouterr().out
    assert "PLAN 캐릭터 불일치로 해당 슬롯 폐기" in output
    assert "slot=9" in output
    assert "expected=['Hana', 'Alisa', 'Maria']" in output
    assert "actual=['Hana', 'Alisa']" in output


def test_call2_detail_keeps_keyvis_character_mismatch_as_error():
    detail_output = _toon_for_slots([4]).replace(
        "</lb-xnai>",
        "keyvis:\n"
        "  camera: full body\n"
        "  characters[1]:\n"
        "    - name: Hana\n"
        "      positive: 1girl, black hair\n"
        "      negative: lowres\n"
        "  scene: classroom, daylight\n"
        "  supplement: Hana stands at the center.\n"
        "</lb-xnai>",
    )

    descriptors, reason = pipeline._parse_call2_detail_output(
        detail_output,
        pipeline.merged_toggles({"key_visual": True}),
        [4],
        ["S001"],
        "TEST-CALL2-DETAIL-KEYVIS-CHARACTER-MISMATCH",
        assigned_keyvis_plan={
            "characters": ["Maria"],
            "scene_brief": "Maria stands at the center.",
        },
    )

    assert descriptors == []
    assert "PLAN 캐릭터 불일치: keyvis" in reason


@pytest.mark.asyncio
async def test_call2_detail_sparse_wardrobe_is_repaired_without_retry(monkeypatch, capsys):
    call_names = []
    initial = _toon_for_slots([4])
    corrected = initial.replace("school uniform", "blue dress")

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        combined = "\n".join(str(message.get("content") or "") for message in messages)
        assert '"scene_brief": "Hana in a blue dress"' in combined
        assert '"wardrobe_snapshot"' not in combined
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
        call2_context_messages=[{"role": "system", "content": "Build detail."}],
        call2_format="Return TOON.",
        toggles=pipeline.merged_toggles({
            "key_visual": False,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
        }),
        stream_notify=None,
    )

    assert len(call_names) == 1
    assert call_names[0].startswith("CALL2-DETAIL 1/1")
    assert len(raw_outputs) == 1
    assert descriptors[0]["characters"][0]["outfit_state"]["worn"] == ["blue dress"]
    assert "희소 DETAIL 복장" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_call2_detail_preserves_successful_shards_and_retries_only_failed(monkeypatch, capsys):
    call_names = []

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        shard = int(re.search(r"CALL2-DETAIL (\d+)/3", call_name).group(1))
        if shard == 2 and "FAILED-SHARD-RETRY" not in call_name:
            return "not toon"
        return _toon_for_slots([shard])

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    scene_plan = []
    for slot in (1, 2, 3):
        scene_plan.append({
            "plan_id": f"S{slot:03d}",
            "slot": slot,
            "anchor_segment": f"C{slot:03d}",
            "source_segments": [f"C{slot:03d}"],
            "characters": ["Hana"],
            "scene_brief": f"Hana scene {slot}",
            "wardrobe_snapshot": {
                "Hana": {
                    "body_state": "clothed",
                    "worn": ["school uniform"],
                    "removed": [],
                },
            },
        })

    descriptors, raw_outputs = await pipeline._run_parallel_call2_details(
        scene_plan=scene_plan,
        call2_context_messages=[{"role": "system", "content": "Build detail."}],
        call2_format="Return TOON.",
        toggles=pipeline.merged_toggles({
            "key_visual": False,
            "call2_parallel_max_concurrency": 3,
            "call2_parallel_slow_retry_enabled": False,
        }),
        stream_notify=None,
    )

    assert [item["slot"] for item in descriptors] == [1, 2, 3]
    assert len(raw_outputs) == 3
    assert sum("CALL2-DETAIL 1/3" in name for name in call_names) == 1
    assert sum("CALL2-DETAIL 3/3" in name for name in call_names) == 1
    assert sum("CALL2-DETAIL 2/3" in name for name in call_names) == 3
    assert any("FAILED-SHARD-RETRY" in name for name in call_names)
    assert "성공 shard 보존 후 실패 shard만 재시도" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_call2_detail_accepts_empty_shard_after_character_discard(monkeypatch, capsys):
    call_names = []

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        return _toon_for_slots([60])

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    descriptors, raw_outputs = await pipeline._run_parallel_call2_details(
        scene_plan=[{
            "plan_id": "S001",
            "slot": 60,
            "anchor_segment": "C001",
            "source_segments": ["C001"],
            "characters": ["Maria"],
            "scene_brief": "Maria stands alone",
            "wardrobe_snapshot": {
                "Maria": {
                    "body_state": "clothed",
                    "worn": ["school uniform"],
                    "removed": [],
                },
            },
        }],
        call2_context_messages=[{"role": "system", "content": "Build detail."}],
        call2_format="Return TOON.",
        toggles=pipeline.merged_toggles({
            "key_visual": False,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
        }),
        stream_notify=None,
    )

    assert descriptors == []
    assert len(raw_outputs) == 1
    assert len(call_names) == 1
    assert call_names[0].startswith("CALL2-DETAIL 1/1")
    assert "PLAN 캐릭터 불일치" in capsys.readouterr().out


def test_call2_detail_partial_keeps_good_slots_and_reports_missing():
    """slot 하나가 빠지거나 할당 밖 번호여도 나머지 좋은 슬롯은 보존한다."""
    toggles = pipeline.merged_toggles({"key_visual": False})
    # assigned=[1,2,3] 이지만 출력은 1,2 + 할당 밖 stray 99 (3이 빠짐).
    text = _toon_for_slots([1, 2, 99])
    kept, missing, discarded, hard = pipeline._parse_call2_detail_partial(
        text, toggles, [1, 2, 3], ["S001", "S002", "S003"], "PARTIAL-TEST",
        None, None,
    )
    assert sorted(kept.keys()) == [1, 2]
    assert missing == [3]
    assert discarded == []
    assert hard == ""
    # 보존된 슬롯에는 서버 plan_id가 주입된다.
    assert kept[1]["plan_id"] == "S001"
    assert kept[2]["plan_id"] == "S002"


def test_call2_detail_partial_marks_character_discard_as_not_missing(capsys):
    """캐릭터 불일치로 폐기된 슬롯은 missing이 아니라 discarded로 분류(재시도 무의미)."""
    toggles = pipeline.merged_toggles({"key_visual": False})
    text = _toon_for_slots([5])  # 항상 "Hana"
    kept, missing, discarded, hard = pipeline._parse_call2_detail_partial(
        text, toggles, [5], ["S005"], "PARTIAL-DISCARD-TEST",
        {5: {"Maria": {"body_state": "clothed", "worn": ["school uniform"], "removed": []}}},
        {5: ["Maria"]},  # PLAN은 Maria, 출력은 Hana → 불일치
    )
    assert kept == {}
    assert missing == []
    assert discarded == [5]
    assert hard == ""


@pytest.mark.asyncio
async def test_call2_detail_partial_loop_fills_only_missing_slot(monkeypatch):
    """①전부예측이 일부 슬롯만 맞춰도 좋은 슬롯은 보존하고 ②실패분만 채운다."""
    calls = []
    partial_requests = []

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        calls.append(call_name)
        if "[PARTIAL" in call_name:
            partial_requests.append(
                "\n".join(str(m.get("content") or "") for m in messages)
            )
            return _toon_for_slots([3])
        # ① 전부 예측: slot 1,2는 맞지만 3 대신 할당 밖 stray 99를 내보낸다.
        return _toon_for_slots([1, 2, 99])

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    scene_plan = []
    for slot in (1, 2, 3):
        scene_plan.append({
            "plan_id": f"S{slot:03d}",
            "slot": slot,
            "anchor_segment": f"C{slot:03d}",
            "source_segments": [f"C{slot:03d}"],
            "characters": ["Hana"],
            "scene_brief": f"Hana scene {slot}",
            "wardrobe_snapshot": {
                "Hana": {"body_state": "clothed", "worn": ["school uniform"], "removed": []},
            },
        })

    descriptors, _raw_outputs = await pipeline._run_parallel_call2_details(
        scene_plan=scene_plan,
        call2_context_messages=[{"role": "system", "content": "Build detail."}],
        call2_format="Return TOON.",
        toggles=pipeline.merged_toggles({
            "key_visual": False,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
        }),
        stream_notify=None,
    )

    assert [item["slot"] for item in descriptors] == [1, 2, 3]
    assert sum("[FULL" in name for name in calls) == 1
    assert sum("[PARTIAL" in name for name in calls) == 1
    # ② 실패분만 요청에는 빠진 slot 3만 들어있고, 이미 확보한 1/2는 없다.
    assert partial_requests
    assert '"slot": 3' in partial_requests[0]
    assert '"slot": 1' not in partial_requests[0]
    assert '"slot": 2' not in partial_requests[0]


@pytest.mark.asyncio
async def test_call2_pipeline_returns_empty_without_fallback_when_all_slots_discarded(
    monkeypatch,
    capsys,
):
    call_names = []

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        if call_name == "CALL2-PLAN":
            return json.dumps({
                "scene_plan": [{
                    "anchor_segment": "C001",
                    "characters": ["Maria"],
                    "scene_brief": "Maria stands alone",
                }],
                "keyvis_plan": None,
            })
        if call_name.startswith("CALL2-DETAIL 1/1"):
            return _toon_for_slots([0])
        raise AssertionError(f"unexpected call: {call_name}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_all_character_mismatch_discard_test",
            "target_slotted": "Hana waits.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Continue."},
                {"role": "char", "data": "Hana waits."},
            ],
        },
        {
            "call1_enabled": False,
            "call2_parallel_enabled": True,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
            "output_count_min": 1,
            "output_count_max": 1,
            "key_visual": False,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        "### Hana\n-default_outfit\nschool uniform",
        extra_costume="### Hana\n-default_outfit\nschool uniform",
        extra_names="Hana",
        backtranslate_names="Hana",
    )

    assert call_names[0] == "CALL2-PLAN"
    assert sum(name.startswith("CALL2-DETAIL") for name in call_names) == 1
    assert "CALL2-FALLBACK" not in call_names
    assert result["items"] == []
    assert result["call2_fix_output"] == ""
    assert result["call2_fallback_stage"] == ""
    assert "scenes: []" in result["call2_output"]
    assert "모든 scene 슬롯이 폐기되어 빈 결과로 완료" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_call2_pipeline_generates_characterless_scene_without_fallback(monkeypatch):
    call_names = []
    detail_messages = []

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        if call_name == "CALL2-PLAN":
            return json.dumps({
                "scene_plan": [{
                    "anchor_segment": "C001",
                    "characters": [],
                    "scene_brief": (
                        "Wide classroom establishing shot with students rising and the teacher exiting"
                    ),
                }],
                "keyvis_plan": None,
            })
        if call_name.startswith("CALL2-DETAIL 1/1"):
            detail_messages.extend(messages)
            return _toon_without_named_characters(0)
        raise AssertionError(f"unexpected call: {call_name}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_characterless_scene_test",
            "target_slotted": (
                "The lunch bell rings. Students rise while the teacher exits.\n\n[Slot 0]"
            ),
            "chats": [
                {"role": "user", "data": "Continue."},
                {
                    "role": "char",
                    "data": "The lunch bell rings. Students rise while the teacher exits.",
                },
            ],
        },
        {
            "call1_enabled": False,
            "call2_parallel_enabled": True,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
            "output_count_min": 1,
            "output_count_max": 1,
            "key_visual": False,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        "### Hana\n-default_outfit\nschool uniform",
        extra_costume="### Hana\n-default_outfit\nschool uniform",
        extra_names="Hana",
        backtranslate_names="Hana",
    )

    assert call_names[0] == "CALL2-PLAN"
    assert sum(name.startswith("CALL2-DETAIL 1/1") for name in call_names) == 1
    assert "CALL2-FALLBACK" not in call_names
    assert len(result["items"]) == 1
    assert result["items"][0]["characters"] == []
    assert result["call2_fallback_stage"] == ""
    assert "students" in result["items"][0]["scene"]
    detail_prompt = "\n".join(str(message.get("content") or "") for message in detail_messages)
    assert "preserve characters: []" in detail_prompt


@pytest.mark.asyncio
async def test_call2_pipeline_continues_with_partial_segment_slot_map(monkeypatch, capsys):
    call_names = []
    plan_requests = []

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        if call_name == "CALL2-PLAN":
            plan_requests.append(
                "\n".join(str(message.get("content") or "") for message in messages)
            )
            return json.dumps({
                "scene_plan": [
                    {
                        "anchor_segment": "C001",
                        "characters": ["Hana"],
                        "scene_brief": "Hana at the first moment",
                    },
                    {
                        "anchor_segment": "C003",
                        "characters": ["Hana"],
                        "scene_brief": "Hana at the final moment",
                    },
                ],
                "keyvis_plan": None,
            })
        if call_name.startswith("CALL2-DETAIL 1/1"):
            return _toon_for_slots([0, 1])
        raise AssertionError(f"unexpected call: {call_name}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_partial_segment_slot_map_test",
            "target_slotted": (
                "첫 문단.\n\n[Slot 0]\n\n"
                "둘째 원문.\n\n세부 원문.\n\n[Slot 1]\n\n"
                "마지막 문단."
            ),
            "chats": [
                {"role": "user", "data": "Continue."},
                {
                    "role": "char",
                    "data": (
                        "첫 문단.\n\n둘째 가공문과 세부 가공문.\n\n마지막 문단."
                    ),
                },
            ],
        },
        {
            "call1_enabled": False,
            "call2_parallel_enabled": True,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
            "output_count_min": 3,
            "output_count_max": 3,
            "key_visual": False,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        "### Hana\n-default_outfit\nschool uniform",
        extra_costume="### Hana\n-default_outfit\nschool uniform",
        extra_names="Hana",
        backtranslate_names="Hana",
    )

    assert call_names[0] == "CALL2-PLAN"
    assert sum(name.startswith("CALL2-DETAIL 1/1") for name in call_names) == 1
    assert "CALL2-FALLBACK" not in call_names
    assert result["call2_fallback_stage"] == ""
    assert [item["slot"] for item in result["items"]] == [0, 1]
    assert plan_requests
    assert "[C001 slot=0]" in plan_requests[0]
    assert "[C002" not in plan_requests[0]
    assert "[C003 slot=1]" in plan_requests[0]
    assert "minimum of 2 and a maximum of 2" in plan_requests[0]
    output = capsys.readouterr().out
    assert "부분 segment-slot 매핑으로 계속 진행" in output
    assert "effective=2..2" in output


@pytest.mark.asyncio
async def test_independent_call2_keyvis_returns_one_object_and_rejects_scenes(monkeypatch):
    seen_messages = []
    keyvis_output = (
        "<lb-xnai>\n"
        "keyvis:\n"
        "  camera: full body\n"
        "  characters[1]:\n"
        "    - name: Hana\n"
        "      positive: 1girl, black hair, school uniform\n"
        "      negative: lowres\n"
        "  scene: classroom, daylight\n"
        "  supplement: Hana stands at the center of the classroom.\n"
        "scenes: []\n"
        "</lb-xnai>"
    )

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        assert call_name == "CALL2-KEYVIS"
        seen_messages.extend(messages)
        return keyvis_output

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    descriptor, raw_output = await pipeline._run_call2_keyvis(
        call2_context_messages=[{"role": "system", "content": "Build detail."}],
        allowed_character_names=["Hana"],
        toggles=pipeline.merged_toggles({
            "key_visual": True,
        }),
        stream_notify=None,
    )

    assert descriptor["kind"] == "keyvis"
    assert descriptor["slot"] == -1
    assert raw_output == keyvis_output
    combined = "\n".join(str(message.get("content") or "") for message in seen_messages)
    assert "# INDEPENDENT KEY VISUAL" in combined
    assert "# ASSIGNED GLOBAL SCENE PLAN" not in combined
    assert "Output exactly one keyvis object and no scene objects" in combined

    leaked_scene_output = keyvis_output.replace(
        "scenes: []",
        "scenes[1]:\n"
        "  - camera: close-up\n"
        "    characters[1]:\n"
        "      - name: Hana\n"
        "        positive: 1girl, black hair\n"
        "    scene: classroom\n"
        "    slot: 4",
    )
    rejected, reason = pipeline._parse_call2_keyvis_output(
        leaked_scene_output,
        pipeline.merged_toggles({"key_visual": True}),
        ["Hana"],
        "TEST-CALL2-KEYVIS-SCENE-LEAK",
    )
    assert rejected is None
    assert "KEYVIS 전용 응답에 scene이 포함됨" in reason


@pytest.mark.asyncio
async def test_call2_role_inputs_are_isolated_without_mutating_stored_state(monkeypatch):
    request_by_call = {}
    state_before = {
        "hana": {
            "canonical_name": "Hana",
            "current_wardrobe": {
                "body_state": "clothed",
                "worn": ["blue dress"],
                "removed": [],
            },
            "last_visual_reference": {
                "positive_tags": "nested generated visual marker",
                "outfit_state": {
                    "body_state": "clothed",
                    "worn": ["blue dress"],
                    "removed": [],
                },
            },
        },
    }
    original_state = json.loads(json.dumps(state_before))

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        request_by_call[call_name] = "\n".join(
            str(message.get("content") or "") for message in messages
        )
        if call_name == "CALL1":
            return json.dumps({
                "reference_assignments": [],
                "history_characters": ["Hana"],
                "current_characters": [{"name": "Hana", "confidence": 0.99}],
                "wardrobe_events": [{
                    "segment_id": "C001",
                    "character": "Hana",
                    "operation": "keep",
                    "items": ["timeline event marker"],
                    "evidence": "Hana waits in the blue dress.",
                    "confidence": 0.99,
                }],
                "unresolved_references": [],
            })
        if call_name == "CALL2-PLAN":
            return json.dumps({
                "scene_plan": [{
                    "anchor_segment": "C001",
                    "characters": ["Hana"],
                    "scene_brief": "Hana waits by the window",
                }],
            })
        if call_name == "CALL2-KEYVIS":
            return """<lb-xnai>
keyvis:
  camera: full body
  characters[1]:
    - name: Hana
      positive: 1girl, black hair, blue dress
      outfit_state:
        body_state: clothed
        worn: [blue dress]
        removed: []
  scene: bedroom, window, daylight
  supplement: Hana waits beside the window.
scenes: []
</lb-xnai>"""
        if call_name.startswith("CALL2-DETAIL 1/1"):
            return _toon_for_slots([0]).replace("school uniform", "blue dress")
        raise AssertionError(f"unexpected call: {call_name}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_role_input_isolation_test",
            "target_slotted": "Hana waits in the blue dress.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Continue."},
                {"role": "char", "data": "Hana waits in the blue dress."},
            ],
        },
        {
            "call1_enabled": True,
            "call1_parallel_enabled": False,
            "call2_parallel_enabled": True,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
            "output_count_min": 1,
            "output_count_max": 1,
            "key_visual": True,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        (
            "### Hana\n-Appearance\n1girl, black hair\n-default_outfit\nblue dress\n\n"
            "### Bob\n-Appearance\n1boy, brown hair\n-default_outfit\nblack suit"
        ),
        extra_instruction=(
            "## Single identifiable character rule\n"
            "ACTIVE BOT INSTRUCTION MARKER\n\n"
            "### Nested instruction heading\n"
            "Keep this nested instruction even when character cards are filtered."
        ),
        extra_costume="### Hana\n-default_outfit\nblue dress",
        extra_names="Hana",
        backtranslate_names="Hana",
        history_plan={
            "history_id": "hist_call2_role_input_isolation",
            "operation": "append",
            "current_message_id": "msg_current",
            "base_context_hash": "base-current",
            "state_before": state_before,
            "call1_history": [],
            "call2_fallback_history": [],
            "call3_fallback_history": [],
            "record_before": {
                "source": {"branch_id": "main"},
                "active_turn": {"base_context_hash": "base-previous"},
                "last_pipeline": {
                    "last_visual_by_character": {
                        "Hana": {
                            "positive_tags": "dedicated last visual marker",
                            "outfit_state": {
                                "body_state": "clothed",
                                "worn": ["blue dress"],
                                "removed": [],
                            },
                        },
                    },
                },
            },
        },
    )

    plan_request = request_by_call["CALL2-PLAN"]
    assert "# ACTIVE BOT IMAGE INSTRUCTIONS" in plan_request
    assert "ACTIVE BOT INSTRUCTION MARKER" in plan_request
    assert "### Nested instruction heading" in plan_request
    assert "# CHARACTER DICTIONARY" in plan_request
    assert "### Hana" in plan_request
    assert "### Bob" not in plan_request
    assert "# AUTHORITATIVE FIXED APPEARANCE" not in plan_request
    assert "# AUTHORITATIVE WARDROBE CONTINUITY STATE" not in plan_request
    assert "# CURRENT WARDROBE EVENT TIMELINE" not in plan_request
    assert "# CLASSIFIED LAST VISUAL REFERENCE" not in plan_request
    assert "nested generated visual marker" not in plan_request
    assert "dedicated last visual marker" not in plan_request
    assert "timeline event marker" not in plan_request

    keyvis_request = request_by_call["CALL2-KEYVIS"]
    assert "### Key Visual" in keyvis_request
    assert "### Scene" not in keyvis_request
    assert "# Example" not in keyvis_request
    assert "# Server limits" not in keyvis_request
    assert "# ACTIVE BOT IMAGE INSTRUCTIONS" in keyvis_request
    assert "ACTIVE BOT INSTRUCTION MARKER" in keyvis_request
    assert "### Nested instruction heading" in keyvis_request
    assert "# CHARACTER DICTIONARY" in keyvis_request
    assert "### Hana" in keyvis_request
    assert "### Bob" not in keyvis_request
    assert "# AUTHORITATIVE FIXED APPEARANCE" in keyvis_request
    assert "# DEFAULT-BASED WARDROBE CONTINUITY BASE" in keyvis_request
    assert "# SPARSE CURRENT WARDROBE CHANGE HISTORY" in keyvis_request
    assert "blue dress" in keyvis_request
    assert "timeline event marker" in keyvis_request
    assert "# CLASSIFIED LAST VISUAL REFERENCE" not in keyvis_request
    assert "nested generated visual marker" not in keyvis_request
    assert "dedicated last visual marker" not in keyvis_request
    assert "negative: ..." not in keyvis_request

    detail_request = next(
        content
        for name, content in request_by_call.items()
        if name.startswith("CALL2-DETAIL 1/1")
    )
    assert "# DETAIL CHECKLIST" in detail_request
    assert "Reason silently and return only the requested final <lb-xnai> block." in detail_request
    assert "\nkeyvis:\n" not in detail_request
    assert "negative: ..." not in detail_request
    assert "# ACTIVE BOT IMAGE INSTRUCTIONS" in detail_request
    assert "ACTIVE BOT INSTRUCTION MARKER" in detail_request
    assert "### Nested instruction heading" in detail_request
    assert "# CHARACTER DICTIONARY" in detail_request
    assert "### Hana" in detail_request
    assert "### Bob" not in detail_request
    assert "# AUTHORITATIVE FIXED APPEARANCE" in detail_request
    assert "# DEFAULT-BASED WARDROBE CONTINUITY BASE" in detail_request
    assert "# SPARSE CURRENT WARDROBE CHANGE HISTORY" in detail_request
    assert "# CLASSIFIED LAST VISUAL REFERENCE" in detail_request
    assert "nested generated visual marker" not in detail_request
    assert "dedicated last visual marker" in detail_request
    assert "timeline event marker" in detail_request
    assert result["last_visual_reference_classification"]["reference_type"] == "CONTINUITY"
    assert state_before == original_state
    assert [item["kind"] for item in result["items"]] == ["keyvis", "scene"]


def test_complete_call2_validation_rejects_one_shard_as_global_fallback():
    toggles = pipeline.merged_toggles({
        "output_count_min": 3,
        "output_count_max": 3,
        "key_visual": False,
    })
    slotted = "\n\n".join(
        f"문단 {slot}.\n\n[Slot {slot}]" for slot in (1, 2, 3)
    )

    descriptors, reason = pipeline.validate_complete_call2_output(
        _toon_for_slots([1]),
        toggles,
        slotted,
        "TEST-CALL2-FALLBACK",
        [1, 2, 3],
    )

    assert descriptors == []
    assert "PLAN scene slot 불일치" in reason


def test_complete_call2_validation_accepts_scene_without_named_characters():
    descriptors, reason = pipeline.validate_complete_call2_output(
        _toon_without_named_characters(0),
        pipeline.merged_toggles({
            "output_count_min": 1,
            "output_count_max": 1,
            "key_visual": False,
        }),
        "Lunch bell rings.\n\n[Slot 0]",
        "TEST-CALL2-NO-NAMED-CHARACTERS",
        [0],
    )

    assert reason == ""
    assert len(descriptors) == 1
    assert descriptors[0]["characters"] == []


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
        if call_name == "CALL2-AUTHORITY-AUDIT":
            return json.dumps({
                "entries": [{
                    "id": 1,
                    "authority_exceptions": [],
                    "conflicts": [],
                }],
            })
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
            "output_count_min": 1,
            "output_count_max": 1,
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
    assert call_names == [
        "CALL2-PLAN",
        "CALL2-FALLBACK",
        "CALL2-AUTHORITY-AUDIT",
    ]
    assert "[ILLUST_CONTEXT:CALL2-FALLBACK] 폴백 시작" in output
    assert "failed_stage=CALL2-PLAN" in output
    assert "CALL2-PLAN JSON object를 찾지 못함" in output
    assert result["call2_fallback_stage"] == "CALL2-PLAN"
    assert "CALL2-PLAN JSON object를 찾지 못함" in result["call2_fallback_reason"]


@pytest.mark.asyncio
async def test_call2_plan_failure_cancels_hanging_independent_keyvis(monkeypatch):
    keyvis_started = asyncio.Event()
    keyvis_cancelled = asyncio.Event()

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        if call_name == "CALL2-PLAN":
            await asyncio.wait_for(keyvis_started.wait(), timeout=1)
            return "not valid plan json"
        if call_name == "CALL2-KEYVIS":
            keyvis_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                keyvis_cancelled.set()
                raise
        if call_name == "CALL2-FALLBACK":
            return _toon_for_slots([0]).replace(
                "<lb-xnai>\n",
                "<lb-xnai>\n"
                "keyvis:\n"
                "  camera: full body\n"
                "  characters[1]:\n"
                "    - name: Hana\n"
                "      positive: 1girl, black hair, school uniform\n"
                "  scene: classroom\n"
                "  supplement: daylight\n",
            )
        raise AssertionError(f"unexpected call: {call_name}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_plan_failure_cancels_keyvis_test",
            "target_slotted": "Hana waits.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Continue."},
                {"role": "char", "data": "Hana waits."},
            ],
        },
        {
            "call1_enabled": False,
            "call2_parallel_enabled": True,
            "output_count_min": 1,
            "output_count_max": 1,
            "key_visual": True,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        "### Hana\n-Appearance\n1girl, black hair\n-default_outfit\nschool uniform",
        extra_costume="### Hana\n-default_outfit\nschool uniform",
        extra_names="Hana",
        backtranslate_names="Hana",
    )

    await asyncio.wait_for(keyvis_cancelled.wait(), timeout=1)
    assert result["call2_fallback_stage"] == "CALL2-PLAN"
    assert [item["kind"] for item in result["items"]] == ["keyvis", "scene"]


@pytest.mark.asyncio
async def test_call2_detail_failure_reuses_preserved_plan_in_global_fallback(monkeypatch):
    call_names = []

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        if call_name == "CALL2-PLAN":
            return json.dumps({
                "scene_plan": [{
                    "anchor_segment": "C001",
                    "characters": ["Hana"],
                    "scene_brief": "Hana waits",
                }],
                "keyvis_plan": None,
            })
        if call_name.startswith("CALL2-DETAIL"):
            return "not toon"
        if call_name == "CALL2-FALLBACK":
            combined = "\n".join(str(message.get("content") or "") for message in messages)
            assert "# PRESERVED GLOBAL PLAN AFTER DETAIL FAILURE" in combined
            assert '"slot": 0' in combined
            assert '"scene_brief": "Hana waits"' in combined
            assert '"wardrobe_snapshot"' not in combined
            return _toon_for_slots([0])
        if call_name == "CALL2-AUTHORITY-AUDIT":
            return json.dumps({
                "entries": [{
                    "id": 1,
                    "authority_exceptions": [],
                    "conflicts": [],
                }],
            })
        raise AssertionError(f"unexpected call: {call_name}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_preserved_plan_fallback_test",
            "target_slotted": "Hana waits.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Continue."},
                {"role": "char", "data": "Hana waits."},
            ],
        },
        {
            "call1_enabled": False,
            "call2_parallel_enabled": True,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
            "output_count_min": 1,
            "output_count_max": 1,
            "key_visual": False,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        "### Hana\n-default_outfit\nschool uniform",
        extra_costume="### Hana\n-default_outfit\nschool uniform",
        extra_names="Hana",
        backtranslate_names="Hana",
    )

    assert call_names[0] == "CALL2-PLAN"
    assert call_names[-2:] == ["CALL2-FALLBACK", "CALL2-AUTHORITY-AUDIT"]
    assert sum(name.startswith("CALL2-DETAIL 1/1") for name in call_names) == 4
    assert any("FAILED-SHARD-RETRY" in name for name in call_names)
    assert any("[FULL" in name for name in call_names)
    assert any("[PARTIAL" in name for name in call_names)
    assert [item["slot"] for item in result["items"]] == [0]
    assert result["call2_plan_output"]
    assert result["call2_fallback_stage"] == "CALL2-DETAIL"


@pytest.mark.asyncio
async def test_call2_detail_failure_preserves_independent_keyvis_in_scene_only_fallback(
    monkeypatch,
):
    call_names = []
    keyvis_output = """<lb-xnai>
keyvis:
  camera: full body
  characters[1]:
    - name: Hana
      positive: 1girl, black hair, school uniform
      outfit_state:
        body_state: clothed
        worn: [school uniform]
        removed: []
  scene: classroom, daylight
  supplement: Hana stands alone in the classroom.
scenes: []
</lb-xnai>"""

    async def fake_pipeline_call(call_name, messages, *args, **kwargs):
        call_names.append(call_name)
        if call_name == "CALL2-PLAN":
            return json.dumps({
                "scene_plan": [{
                    "anchor_segment": "C001",
                    "characters": ["Hana"],
                    "scene_brief": "Hana waits",
                }],
            })
        if call_name == "CALL2-KEYVIS":
            return keyvis_output
        if call_name.startswith("CALL2-DETAIL"):
            return "not toon"
        if call_name == "CALL2-FALLBACK":
            combined = "\n".join(
                str(message.get("content") or "") for message in messages
            )
            assert "A separately validated Key Visual is already preserved" in combined
            assert "Omit keyvis completely" in combined
            return _toon_for_slots([0])
        if call_name == "CALL2-AUTHORITY-AUDIT":
            return json.dumps({
                "entries": [{
                    "id": 1,
                    "authority_exceptions": [],
                    "conflicts": [],
                }],
            })
        raise AssertionError(f"unexpected call: {call_name}")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call2_preserved_keyvis_scene_fallback_test",
            "target_slotted": "Hana waits.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "Continue."},
                {"role": "char", "data": "Hana waits."},
            ],
        },
        {
            "call1_enabled": False,
            "call2_parallel_enabled": True,
            "call2_parallel_max_concurrency": 1,
            "call2_parallel_slow_retry_enabled": False,
            "output_count_min": 1,
            "output_count_max": 1,
            "key_visual": True,
            "call3_enabled": False,
            "speak_enabled": False,
        },
        "### Hana\n-Appearance\n1girl, black hair\n-default_outfit\nschool uniform",
        extra_costume="### Hana\n-default_outfit\nschool uniform",
        extra_names="Hana",
        backtranslate_names="Hana",
    )

    assert call_names.count("CALL2-KEYVIS") == 1
    assert any(name.startswith("CALL2-DETAIL 1/1") for name in call_names)
    assert any("FAILED-SHARD-RETRY" in name for name in call_names)
    assert call_names[-2:] == ["CALL2-FALLBACK", "CALL2-AUTHORITY-AUDIT"]
    assert result["call2_keyvis_output"] == keyvis_output
    assert result["call2_fallback_stage"] == "CALL2-DETAIL"
    assert [item["kind"] for item in result["items"]] == ["keyvis", "scene"]
    assert "Hana stands alone in the classroom." in result["call2_output"]


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
async def test_call1_segments_and_call2_plan_keyvis_and_details_run_in_parallel(monkeypatch):
    paragraphs = [f"Hana paragraph {index}." for index in range(10)]
    narrative = "\n\n".join(paragraphs)
    target_slotted = pipeline.insert_slots(narrative)
    call1_active = 0
    call1_max_active = 0
    detail_active = 0
    detail_max_active = 0
    plan_keyvis_active = 0
    plan_keyvis_max_active = 0
    plan_started = asyncio.Event()
    keyvis_started = asyncio.Event()
    call_names = []

    async def fake_call(task_key, messages, **kwargs):
        nonlocal call1_active, call1_max_active, detail_active, detail_max_active
        nonlocal plan_keyvis_active, plan_keyvis_max_active
        text = "\n".join(str(message.get("content") or "") for message in messages)
        metadata = pipeline.llm_service._stream_metadata_ctx.get({})
        call_name = str(metadata.get("call_name") or task_key)
        call_names.append(call_name)
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
        if call_name == "CALL2-PLAN":
            assert "Follow these steps and output each and all step" not in text
            assert '"keyvis_plan"' not in text
            assert '"plan_id": "S001"' not in text
            assert '"outfit_state"' not in text.split("# GLOBAL CALL2 PLAN", 1)[1]
            plan_keyvis_active += 1
            plan_keyvis_max_active = max(plan_keyvis_max_active, plan_keyvis_active)
            plan_started.set()
            try:
                await asyncio.wait_for(keyvis_started.wait(), timeout=1)
                await asyncio.sleep(0.01)
                return json.dumps({
                    "scene_plan": [
                        {
                            "anchor_segment": f"C{slot + 1:03d}",
                            "characters": ["Hana"],
                            "scene_brief": f"Hana scene {slot}",
                        }
                        for slot in range(9)
                    ],
                })
            finally:
                plan_keyvis_active -= 1

        if call_name == "CALL2-KEYVIS":
            assert "# INDEPENDENT KEY VISUAL" in text
            assert "# GLOBAL CALL2 PLAN" not in text
            assert "# ASSIGNED GLOBAL SCENE PLAN" not in text
            plan_keyvis_active += 1
            plan_keyvis_max_active = max(plan_keyvis_max_active, plan_keyvis_active)
            keyvis_started.set()
            try:
                await asyncio.wait_for(plan_started.wait(), timeout=1)
                await asyncio.sleep(0.01)
                return """<lb-xnai>
keyvis:
  camera: full body
  characters[1]:
    - name: Hana
      positive: 1girl, black hair, school uniform
      negative: lowres
      outfit_state:
        body_state: clothed
        worn: [school uniform]
        removed: []
  scene: classroom
  supplement: daylight
scenes: []
</lb-xnai>"""
            finally:
                plan_keyvis_active -= 1

        assert call_name.startswith("CALL2-DETAIL")
        assert "# ASSIGNED GLOBAL SCENE PLAN" in text
        assert "Omit keyvis" in text
        assert "# INDEPENDENT KEY VISUAL" not in text
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
            "output_count_min": 9,
            "output_count_max": 9,
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
    assert plan_keyvis_max_active == 2
    assert detail_max_active == 3
    assert len(result["call2_detail_outputs"]) == 3
    assert len(result["items"]) == 10
    assert result["items"][0]["kind"] == "keyvis"
    assert result["call2_keyvis_output"].startswith("<lb-xnai>")
    assert [item["slot"] for item in result["items"][1:]] == list(range(9))
    reparsed = pipeline.parse_toon_plan(
        result["call2_output"],
        pipeline.merged_toggles({"output_count_min": 9, "output_count_max": 9}),
        "TEST-MERGED-CALL2",
    )
    assert len(reparsed) == 10
    assert [item["plan_id"] for item in reparsed[1:]] == [
        f"S{index:03d}" for index in range(1, 10)
    ]
    assert "CALL2-PLAN" in call_names
    assert "CALL2-KEYVIS" in call_names
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
    items = pipeline.parse_toon_plan(toon, pipeline.merged_toggles({"output_count_max": 11}))

    assert [item["slot"] for item in items] == [-1, 0]
    pipeline.create_session("cache_test_1234", "context")
    pipeline.set_session_result("cache_test_1234", items, [b"keyvis", b"scene"])
    assert pipeline.session_image_by_slot("cache_test_1234", -1) == b"keyvis"
    assert pipeline.session_image_by_slot("cache_test_1234", 0) == b"scene"
    assert pipeline.session_item_by_slot("cache_test_1234", 0)["scene"] == "classroom, sunset"


def test_prompt_batch_accepts_65_items_and_rejects_66():
    session_id = "risu_" + ("d" * 64)
    items = [
        {
            "slot": slot,
            "positive": f"scene {slot}",
            "negative": "low quality",
        }
        for slot in [-1, *range(64)]
    ]

    accepted = pipeline.parse_prompt_batch_request(
        pipeline.PROMPT_BATCH_PREFIX
        + "\n"
        + json.dumps({"session_id": session_id, "items": items})
    )

    assert pipeline.MAX_ILLUSTRATION_SLOT_COUNT == 65
    assert accepted is not None
    assert len(accepted["items"]) == 65
    assert [item["slot"] for item in accepted["items"]] == [-1, *range(64)]

    rejected = pipeline.parse_prompt_batch_request(
        pipeline.PROMPT_BATCH_PREFIX
        + "\n"
        + json.dumps({
            "session_id": session_id,
            "items": [*items, {
                "slot": 64,
                "positive": "scene 64",
                "negative": "low quality",
            }],
        })
    )
    assert rejected is None


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


def test_short_lookup_key_accepts_65_slots_and_rejects_66(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    accepted_key = "e" * 24
    accepted_id = "risu_" + accepted_key + ("1" * 40)
    rejected_key = "f" * 24
    rejected_id = "risu_" + rejected_key + ("2" * 40)
    accepted_items = [
        {"kind": "keyvis" if slot == -1 else "scene", "slot": slot}
        for slot in [-1, *range(64)]
    ]
    rejected_items = [
        *accepted_items,
        {"kind": "scene", "slot": 64},
    ]

    try:
        pipeline.create_session(accepted_id, "context")
        pipeline.set_session_result(
            accepted_id,
            accepted_items,
            [b"image"] * len(accepted_items),
        )
        assert pipeline.session_slots_by_lookup_key(accepted_key) == [-1, *range(64)]

        pipeline.create_session(rejected_id, "context")
        pipeline.set_session_result(
            rejected_id,
            rejected_items,
            [b"image"] * len(rejected_items),
        )
        with pytest.raises(ValueError, match=r"count=66, max=65"):
            pipeline.session_slots_by_lookup_key(rejected_key)
    finally:
        pipeline._SESSIONS.pop(accepted_id, None)
        pipeline._SESSIONS.pop(rejected_id, None)
        pipeline._LOOKUP_KEYS.pop(accepted_key, None)
        pipeline._LOOKUP_KEYS.pop(rejected_key, None)


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
        pipeline.merged_toggles({"output_count_max": 11, "key_visual": False}),
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
        assert "exact roster identifier belongs only on the left side of the colon" in prompt
        assert "never leave that Latin-script roster identifier inside" in prompt


def test_call3_output_contract_rejects_roster_ids_only_inside_localized_dialogue(capsys):
    leaked = """[Scene slot=44]
Maria: "Masachika 군. 수업이 많이 힘들었나 보네?" #charming
[Scene slot=52]
Alisa: (Alisa라면 어떻게 했을까?) #thought_cloud"""
    valid, reason = pipeline.validate_call3_output_contract(
        leaked,
        [44, 52],
        "Masachika, Alisa, Maria",
        "한국어",
    )

    assert valid is False
    assert "대사 본문에 내부 발화자 ID 유출" in reason
    assert "'names': ['Masachika']" in reason
    assert "'names': ['Alisa']" in reason
    assert "내부 발화자 ID 유출" in capsys.readouterr().out

    corrected = """[Scene slot=44]
Maria: "마사치카 군. 수업이 많이 힘들었나 보네?" #charming
[Scene slot=52]
Alisa: (나라면 어떻게 했을까?) #thought_cloud"""
    assert pipeline.validate_call3_output_contract(
        corrected,
        [44, 52],
        "Masachika, Alisa, Maria",
        "한국어",
    ) == (True, "")


def test_call3_output_contract_allows_roster_ids_in_speaker_prefix_and_english_body(capsys):
    korean = '[Scene slot=7]\nMasachika: (차갑다...) #monologue_box'
    assert pipeline.validate_call3_output_contract(
        korean,
        [7],
        "Masachika, Alisa",
        "한국어",
    ) == (True, "")

    english = '[Scene slot=7]\nAlisa: "Masachika, are you all right?" #normal'
    assert pipeline.validate_call3_output_contract(
        english,
        [7],
        "Masachika, Alisa",
        "영어",
    ) == (True, "")
    assert "영어 대사 출력이므로" in capsys.readouterr().out


def test_call3_roster_leak_recovery_removes_only_violating_entries():
    source = """[Scene slot=44]
Maria: "괜찮은 대사야." #normal
Maria: "Alisa에게 이 말을 전해 줘." #normal
[Scene slot=52] Alisa: (Alisa라면 어떻게 했을까?) #thought_cloud"""

    sanitized, removed = pipeline._remove_call3_roster_leaking_dialogue_entries(
        source,
        "Maria, Alisa",
    )

    assert sanitized == """[Scene slot=44]
Maria: "괜찮은 대사야." #normal
[Scene slot=52]"""
    assert removed == [
        {"entry": 2, "slot": 44, "speaker": "Maria", "names": ["Alisa"]},
        {"entry": 3, "slot": 52, "speaker": "Alisa", "names": ["Alisa"]},
    ]
    assert pipeline._call3_dialogue_roster_leaks(
        sanitized,
        "Maria, Alisa",
    ) == []
    speak_map = pipeline.parse_speak_output(sanitized)
    assert speak_map[44] == 'Maria: "괜찮은 대사야." #normal'
    assert speak_map.get(52, "") == ""


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
        '[Scene slot=2]\n\n[Scene slot=7]\nMinsu: (Wait.)',
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
    assert "leave that Scene block body empty" in manga
    assert "Every supplied selected scene header is mandatory" in manga
    assert "Never output slot -1." in manga


def test_call3_partial_recovery_keeps_safe_dialogue_and_fills_silent_headers(capsys):
    recovered, metadata = pipeline.recover_call3_partial_output(
        '''[Scene slot=11]
Alisa: "정상 대사야." #normal
[Scene slot=99]
Alisa: "예상 밖 슬롯." #normal''',
        [4, 7, 11],
        "Alisa",
        "한국어",
    )

    assert recovered == '''[Scene slot=4]

[Scene slot=7]

[Scene slot=11]
Alisa: "정상 대사야." #normal'''
    assert metadata["missing_headers"] == [4, 7]
    assert metadata["unexpected_headers"] == [99]
    assert metadata["populated_slots"] == [11]
    assert metadata["silent_slots"] == [4, 7]
    assert "슬롯별 부분 복구 성공" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_call3_accepts_header_only_silent_scenes_without_correction(monkeypatch):
    calls = []

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        **kwargs,
    ):
        calls.append(call_name)
        return '''[Scene slot=4]

[Scene slot=7]

[Scene slot=11]
Alisa: "이 장면에는 대사가 있어." #normal'''

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    state = await pipeline._build_call3_dialogue_with_recovery(
        [{"role": "user", "content": "selected scenes"}],
        [4, 7, 11],
        "Alisa",
        "한국어",
    )

    assert calls == ["CALL3"]
    assert state["correction_used"] is False
    assert state["partial_recovery_used"] is False
    assert state["silent_slots"] == [4, 7]
    assert pipeline.parse_speak_output(state["output"]) == {
        11: 'Alisa: "이 장면에는 대사가 있어." #normal'
    }


@pytest.mark.asyncio
async def test_call3_correction_exhaustion_keeps_only_safe_slots(monkeypatch, capsys):
    calls = []

    async def fake_pipeline_call(
        call_name,
        messages,
        stream_notify=None,
        result_validator=None,
        **kwargs,
    ):
        calls.append(call_name)
        if call_name == "CALL3":
            return '[Scene slot=4]\nAlisa: "정상 대사야." #normal'
        raise RuntimeError("교정 라우팅 소진")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_pipeline_call)
    state = await pipeline._build_call3_dialogue_with_recovery(
        [{"role": "user", "content": "selected scenes"}],
        [4, 7],
        "Alisa",
        "한국어",
    )

    assert calls == ["CALL3", "CALL3-CORRECTION"]
    assert state["correction_used"] is True
    assert state["partial_recovery_used"] is True
    assert state["silent_slots"] == [7]
    assert pipeline.parse_speak_output(state["output"]) == {
        4: 'Alisa: "정상 대사야." #normal'
    }
    captured = capsys.readouterr().out
    assert "안전한 슬롯별 대사만 복구" in captured
    assert "슬롯별 부분 복구 성공" in captured


@pytest.mark.asyncio
async def test_call3_total_failure_becomes_silent_without_raising(monkeypatch, capsys):
    async def fail_pipeline_call(*args, **kwargs):
        raise RuntimeError("CALL3 공급자 전체 실패")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fail_pipeline_call)
    state = await pipeline._build_call3_dialogue_with_recovery(
        [{"role": "user", "content": "selected scenes"}],
        [4, 7],
        "Alisa",
        "한국어",
    )

    assert state["partial_recovery_used"] is True
    assert state["silent_slots"] == [4, 7]
    assert state["output"] == "[Scene slot=4]\n\n[Scene slot=7]"
    assert "이미지 파이프라인 계속" in capsys.readouterr().out


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
        if _call_name(task_key) == "CALL2-AUTHORITY-AUDIT":
            return _authority_audit_response(messages)
        return responses[1 if task_key == "illustration_call3" else 0]

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

    assert len(calls) == 3
    assert "manga dialogue writer and balloon-style editor" in calls[2][0]["content"]
    assert "#normal" in calls[2][0]["content"]
    assert result["items"][0]["speak"] == 'Hana: "No way!" #burst'
    assert '[SPEAK]\nHana: "No way!" #burst' in result["items"][0]["raw_positive"]


@pytest.mark.asyncio
async def test_call3_uses_original_narrative_and_only_call2_selected_scene_slots(monkeypatch):
    calls = []
    call_names = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages, kwargs))
        metadata = pipeline.llm_service._stream_metadata_ctx.get({})
        call_name = str(metadata.get("call_name") or task_key)
        call_names.append(call_name)
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
            if call_name == "CALL2-AUTHORITY-AUDIT":
                return _authority_audit_response(messages)
            if call_name == "CALL2-PLAN":
                return json.dumps({
                    "scene_plan": [{
                        "anchor_segment": "C001",
                        "characters": ["hana"],
                        "scene_brief": "first selected moment",
                    }, {
                        "anchor_segment": "C002",
                        "characters": ["hana"],
                        "scene_brief": "second selected moment",
                    }],
                })
            if call_name == "CALL2-KEYVIS":
                return """<lb-xnai>
keyvis:
  camera: portrait
  characters[1]:
    - name: hana
      positive: 1girl, hana, black hair
  scene: poster key visual
scenes: []
</lb-xnai>"""
            assert call_name.startswith("CALL2-DETAIL")
            detail_text = "\n".join(
                str(message.get("content") or "") for message in messages
            )
            plan_match = re.search(
                r"# ASSIGNED GLOBAL SCENE PLAN\s*(\[[\s\S]*?\])\s*\n\nExpand each plan",
                detail_text,
            )
            assert plan_match
            assigned = json.loads(plan_match.group(1))
            return _toon_for_slots([int(item["slot"]) for item in assigned])

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

    task_keys = [task_key for task_key, _messages, _kwargs in calls]
    assert task_keys[0] == "illustration_call1_backtranslate"
    assert task_keys[-1] == "illustration_call3"
    assert task_keys.count("illustration_call2") == 5
    assert "CALL2-PLAN" in call_names
    assert "CALL2-KEYVIS" in call_names
    assert sum(name.startswith("CALL2-DETAIL") for name in call_names) == 2
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
async def test_call3_retries_when_internal_roster_id_leaks_into_korean_dialogue(
    monkeypatch,
    capsys,
):
    call3_attempts = 0

    async def fake_call(task_key, messages, **kwargs):
        nonlocal call3_attempts
        if task_key == "illustration_call2":
            return """<lb-xnai>
scenes[1]:
  - camera: medium shot
    characters[2]:
      - name: Maria
        positive: 1girl, Maria, blonde hair
      - name: Masachika
        positive: 1boy, Masachika, black hair
    scene: Maria checks Masachika's notebook
    slot: 0
</lb-xnai>"""

        assert task_key == "illustration_call3"
        call3_attempts += 1
        if call3_attempts == 1:
            return (
                '[Scene slot=0]\n'
                'Maria: "Masachika 군. 수업이 많이 힘들었나 보네?" #charming'
            )

        correction = messages[-1]["content"]
        assert "Never repeat a roster identifier inside quoted dialogue" in correction
        assert "if uncertain, omit the direct address" in correction
        assert "대사 본문에 내부 발화자 ID 유출" in correction
        validator = kwargs.get("result_validator")
        assert validator is not None
        corrected = (
            '[Scene slot=0]\n'
            'Maria: "마사치카 군. 수업이 많이 힘들었나 보네?" #charming'
        )
        assert validator(corrected) == (True, "")
        return corrected

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    result = await pipeline.build_from_context(
        {
            "session_id": "call3_roster_leak_retry_test",
            "target_slotted": "마리아가 마사치카의 노트를 살핀다.\n\n[Slot 0]",
            "chats": [
                {"role": "user", "data": "오전 수업이 끝났다."},
                {"role": "char", "data": "마리아가 마사치카의 노트를 살핀다."},
            ],
        },
        {
            "call1_enabled": False,
            "call3_enabled": True,
            "speak_enabled": True,
            "call3_prompt_mode": "manga",
            "speak_language": "한국어",
            "key_visual": False,
        },
        (
            "### Maria\n-Appearance: 1girl, blonde hair\n\n"
            "### Masachika\n-Appearance: 1boy, black hair"
        ),
        extra_names="Maria, Masachika",
    )

    assert call3_attempts == 2
    assert result["call3_correction_used"] is True
    assert "Masachika 군" in result["call3_initial_output"]
    assert result["items"][0]["speak"] == (
        'Maria: "마사치카 군. 수업이 많이 힘들었나 보네?" #charming'
    )
    captured = capsys.readouterr().out
    assert "대사 본문에 내부 발화자 ID 유출" in captured
    assert "출력 계약을 위반해" in captured


@pytest.mark.asyncio
async def test_call3_correction_exhaustion_drops_only_leaking_dialogue_and_continues(
    monkeypatch,
    capsys,
):
    call3_attempts = 0
    history_records = []

    async def fake_call(task_key, messages, **kwargs):
        nonlocal call3_attempts
        if task_key == "illustration_call2":
            return """<lb-xnai>
scenes[2]:
  - camera: medium shot
    characters[2]:
      - name: Maria
        positive: 1girl, Maria, blonde hair
      - name: Alisa
        positive: 1girl, Alisa, silver hair
    scene: first conversation
    slot: 0
  - camera: close-up
    characters[1]:
      - name: Alisa
        positive: 1girl, Alisa, silver hair
    scene: second conversation
    slot: 1
</lb-xnai>"""

        assert task_key == "illustration_call3"
        call3_attempts += 1
        if call3_attempts == 1:
            return """[Scene slot=0]
Maria: "괜찮은 대사야." #normal
Maria: "Alisa에게 이 말을 전해 줘." #normal
[Scene slot=1]
Alisa: (Alisa라면 어떻게 했을까?) #thought_cloud"""

        validator = kwargs.get("result_validator")
        assert validator is not None
        still_invalid = """[Scene slot=0]
Maria: "Alisa에게 전해야 해." #normal
[Scene slot=1]
Alisa: (Alisa라면 어떻게 했을까?) #thought_cloud"""
        assert validator(still_invalid)[0] is False
        return "[LLM 실패] illustration_call3 fallback 재시도 소진"

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(
        pipeline.lighbd_service,
        "_log_lighbd_history",
        history_records.append,
    )
    result = await pipeline.build_from_context(
        {
            "session_id": "call3_roster_leak_exhaustion_recovery_test",
            "target_slotted": "첫 대화.\n\n[Slot 0]\n\n둘째 대화.\n\n[Slot 1]",
            "chats": [
                {"role": "user", "data": "계속해."},
                {"role": "char", "data": "첫 대화.\n\n둘째 대화."},
            ],
        },
        {
            "call1_enabled": False,
            "call3_enabled": True,
            "speak_enabled": True,
            "call3_prompt_mode": "manga",
            "speak_language": "한국어",
            "key_visual": False,
        },
        (
            "### Maria\n-Appearance: 1girl, blonde hair\n\n"
            "### Alisa\n-Appearance: 1girl, silver hair"
        ),
        extra_names="Maria, Alisa",
    )

    assert call3_attempts == 2
    assert result["call3_correction_used"] is True
    assert result["call3_dialogue_drop_recovery_used"] is True
    assert result["call3_dropped_dialogue_entries"] == [
        {"entry": 2, "slot": 0, "speaker": "Maria", "names": ["Alisa"]},
        {"entry": 3, "slot": 1, "speaker": "Alisa", "names": ["Alisa"]},
    ]
    assert result["items"][0]["speak"] == 'Maria: "괜찮은 대사야." #normal'
    assert result["items"][1]["speak"] == ""
    assert "Alisa에게" not in result["call3_output"]
    assert "Alisa라면" not in result["call3_output"]
    captured = capsys.readouterr().out
    assert "CALL3 교정/라우팅 폴백 소진" in captured
    assert "부분 복구 성공" in captured
    assert "파이프라인 계속" in captured


@pytest.mark.asyncio
async def test_call3_skips_dialogue_when_call2_selected_only_key_visual(monkeypatch, capsys):
    task_keys = []
    call_names = []

    async def fake_call(task_key, messages, **kwargs):
        task_keys.append(task_key)
        metadata = pipeline.llm_service._stream_metadata_ctx.get({})
        call_name = str(metadata.get("call_name") or task_key)
        call_names.append(call_name)
        assert task_key == "illustration_call2"
        if call_name == "CALL2-PLAN":
            return json.dumps({
                "scene_plan": [{
                    "anchor_segment": "C001",
                    "characters": ["Maria"],
                    "scene_brief": "Maria appears in the poster scene",
                }],
            })
        if call_name == "CALL2-KEYVIS":
            return """<lb-xnai>
keyvis:
  camera: portrait
  characters[1]:
    - name: hana
      positive: 1girl, hana, black hair
  scene: poster key visual
scenes: []
</lb-xnai>"""
        if call_name == "CALL2-AUTHORITY-AUDIT":
            return _authority_audit_response(messages)
        assert call_name.startswith("CALL2-DETAIL")
        return _toon_for_slots([0])

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
        (
            "### hana\n-Appearance: 1girl, black hair\n\n"
            "### Maria\n-Appearance: 1girl, blonde hair\n"
            "-default_outfit\nschool uniform"
        ),
        extra_names="Hana",
        history_plan={
            "state_before": {
                "maria": {
                    "canonical_name": "Maria",
                    "current_wardrobe": {
                        "body_state": "clothed",
                        "worn": ["school uniform"],
                        "removed": [],
                    },
                    "wardrobe_timeline": [],
                },
            },
            "call1_history": [],
            "call2_fallback_history": [],
            "record_before": {"last_pipeline": {}},
            "current_message_id": "msg-keyvis-only",
        },
    )

    assert task_keys == ["illustration_call2"] * 4
    assert "CALL2-PLAN" in call_names
    assert "CALL2-KEYVIS" in call_names
    assert sum(name.startswith("CALL2-DETAIL") for name in call_names) == 1
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
            if _call_name(task_key) == "CALL2-AUTHORITY-AUDIT":
                return _authority_audit_response(messages)
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
        assert events == [
            "illustration_call2",
            "illustration_call2",
            "dispatch",
            "illustration_call3",
        ]
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

    assert events == [
        "illustration_call2",
        "illustration_call2",
        "dispatch",
        "illustration_call3",
    ]
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
        if _call_name(task_key) == "CALL2-AUTHORITY-AUDIT":
            return _authority_audit_response(messages)
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
        and not any(
            "CALL2-AUTHORITY-AUDIT" in str(message.get("content") or "")
            for message in messages
        )
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
            {
                "name": "Right",
                "positive": "green hair",
                "negative": "red hair",
                "position": "on the right",
                "outfit_state": {"worn": ["green jacket"]},
            },
            {
                "name": "Left",
                "positive": "red hair",
                "negative": "green hair",
                "position": "on the left",
                "outfit_state": {"worn": ["red coat"]},
            },
        ],
    }
    calls = []

    async def fake_call(call_name, messages, stream_notify=None, result_validator=None, json_mode=False, history_ids_sink=None):
        calls.append((call_name, messages, json_mode))
        if history_ids_sink is not None:
            history_ids_sink.append("mask-slot4-id")
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
    request_characters = json.loads(calls[0][1][1]["content"])["characters"]
    assert request_characters[0]["validated_positive"] == "green hair"
    assert request_characters[0]["validated_negative"] == "red hair"
    assert request_characters[0]["outfit_state"] == {"worn": ["green jacket"]}
    assert [character["name"] for character in descriptor["characters"]] == ["Left", "Right"]
    assert descriptor["multi_char_layout"]["character_order"] == ["Left", "Right"]
    assert descriptor["multi_char_history_ids"] == ["mask-slot4-id"]
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
    ] == ["red hair", "green hair"]
    assert [
        region["negative"]
        for region in descriptor["multi_char_layout"]["regions"]
    ] == ["green hair", "red hair"]
    assert [
        region["outfit_state"]
        for region in descriptor["multi_char_layout"]["regions"]
    ] == [{"worn": ["red coat"]}, {"worn": ["green jacket"]}]


def test_multi_char_mask_generation_setting_controls_layout_path():
    assert pipeline.merged_toggles({})["multi_char_mask_enabled"] is True
    assert pipeline.should_enable_multi_char_layout(
        {"multi_char_mask_enabled": True, "prompt_format": "v3"},
        "comfy",
    ) is True
    assert pipeline.should_enable_multi_char_layout(
        {"multi_char_mask_enabled": False, "prompt_format": "v3"},
        "comfy",
    ) is False
    assert pipeline.should_enable_multi_char_layout(
        {"multi_char_mask_enabled": True, "prompt_format": "v3"},
        "chansub",
    ) is False


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

    async def fake_call(call_name, messages, stream_notify=None, result_validator=None, json_mode=False, history_ids_sink=None):
        if history_ids_sink is not None:
            history_ids_sink.append("mask-slot5-id")
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
    # 실패 경로에서도 자기 slot의 MULTI-CHAR-MASK 호출 id가 descriptor에 남는다.
    assert descriptor["multi_char_history_ids"] == ["mask-slot5-id"]


@pytest.mark.asyncio
async def test_call_pipeline_llm_history_ids_sink_excludes_global_trace(monkeypatch):
    """MULTI-CHAR-MASK 경로(history_ids_sink)는 전역 trace에 넣지 않고 sink에만 담는다."""
    trace: list[str] = []
    token = pipeline._llm_trace_ctx.set(trace)
    try:
        async def fake_call(task_key, actual_messages, **kwargs):
            return "ok"

        monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
        monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", lambda _r: None)

        sink: list[str] = []
        await pipeline._call_pipeline_llm(
            "MULTI-CHAR-MASK slot=1",
            [{"role": "user", "content": "{}"}],
            history_id="mask-main-id",
            history_ids_sink=sink,
        )

        assert "mask-main-id" in sink
        assert trace == []
    finally:
        pipeline._llm_trace_ctx.reset(token)


@pytest.mark.asyncio
async def test_call_pipeline_llm_without_sink_appends_global_trace(monkeypatch):
    """sink가 없으면 기존대로 전역 trace에 id가 들어간다(CALL1/2/3 회귀)."""
    trace: list[str] = []
    token = pipeline._llm_trace_ctx.set(trace)
    try:
        async def fake_call(task_key, actual_messages, **kwargs):
            return "ok"

        monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
        monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", lambda _r: None)

        await pipeline._call_pipeline_llm(
            "CALL1",
            [{"role": "user", "content": "scene"}],
            history_id="call1-id",
        )

        assert "call1-id" in trace
    finally:
        pipeline._llm_trace_ctx.reset(token)


@pytest.mark.asyncio
async def test_pipeline_llm_records_success_in_lighbd_history(monkeypatch):
    records = []
    events = []
    call_kwargs = {}
    messages = [{"role": "user", "content": "scene"}]

    async def fake_call(task_key, actual_messages, **kwargs):
        assert task_key == "illustration_call1"
        assert actual_messages == messages
        call_kwargs.update(kwargs)
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
    assert records[0]["execution_id"]
    assert records[0]["history_id"] == records[0]["execution_id"]
    assert call_kwargs["execution_id"] == records[0]["execution_id"]
    assert callable(call_kwargs["execution_observer"])
    assert {event["execution_id"] for event in events} == {
        records[0]["execution_id"]
    }


@pytest.mark.asyncio
async def test_call2_keyvis_shares_route_and_has_distinct_queue_live_history(monkeypatch):
    records = []
    events = []
    messages = [{"role": "user", "content": "one key visual"}]

    async def fake_call(task_key, actual_messages, **_kwargs):
        assert task_key == "illustration_call2"
        assert actual_messages == messages
        metadata = pipeline.llm_service._stream_metadata_ctx.get({})
        assert metadata["task_key"] == "illustration_call2"
        assert metadata["call_name"] == "CALL2-KEYVIS"
        assert metadata["execution_id"]
        return "completed key visual"

    async def fake_notify(event):
        events.append(event)

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", records.append)

    result = await pipeline._call_pipeline_llm(
        "CALL2-KEYVIS",
        messages,
        fake_notify,
    )

    assert result == "completed key visual"
    assert [event["type"] for event in events] == ["start", "done"]
    assert {event["call_name"] for event in events} == {"CALL2-KEYVIS"}
    assert {
        event["queue_subtask"]["group_id"] for event in events
    } == {"call2_keyvis"}
    assert len(records) == 1
    assert records[0]["call_name"] == "CALL2-KEYVIS"
    assert records[0]["prompt_id"] == "illustration_context:CALL2-KEYVIS"
    assert records[0]["task_key"] == "illustration_call2"
    assert records[0]["input"] == messages
    assert records[0]["output"] == "completed key visual"


@pytest.mark.asyncio
async def test_pipeline_llm_records_failure_in_lighbd_history(monkeypatch):
    records = []
    events = []
    messages = [{"role": "user", "content": "broken scene"}]

    async def fake_call(task_key, actual_messages, **_kwargs):
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
    backup_path = history_path.parent / "backups" / "lighbd_history.jsonl.bak"
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


def test_call1_compact_output_uses_server_defaults_for_omitted_fields():
    current = "She enters the room."
    _rendered, segments = pipeline._segment_current_context(current)

    analysis = pipeline.parse_call1_analysis(
        json.dumps({
            "reference_assignments": [{
                "segment_id": "C001",
                "surface": "She",
                "canonical_name": "Hana",
            }],
            "history_characters": ["Hana"],
            "current_characters": ["Hana"],
            "wardrobe_events": [],
            "unresolved_references": [],
        }),
        current,
        segments,
        "Hana",
    )

    assert analysis is not None
    assert analysis["current_characters"] == [{"name": "Hana", "confidence": 1.0}]
    assert analysis["reference_assignments"][0]["occurrence"] == 1
    assert analysis["reference_assignments"][0]["replacement"] == "Hana"
    assert analysis["reference_assignments"][0]["confidence"] == 1.0


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


@pytest.mark.parametrize(
    (
        "operation",
        "current_base_hash",
        "previous_base_hash",
        "expected_relation",
        "expected_comparison",
    ),
    [
        ("new", "base-current", "", "NO_PRIOR_REFERENCE", "unavailable"),
        (
            "append",
            "base-current",
            "base-previous",
            "PRIOR_COMMITTED_TURN",
            "different",
        ),
        (
            "duplicate",
            "base-same",
            "base-same",
            "SAME_ACTIVE_TURN_EXACT",
            "same",
        ),
        (
            "reroll",
            "base-same",
            "base-same",
            "SAME_ACTIVE_TURN_REPLACED",
            "same",
        ),
    ],
)
def test_reference_provenance_uses_server_history_operation(
    operation,
    current_base_hash,
    previous_base_hash,
    expected_relation,
    expected_comparison,
):
    provenance = pipeline.build_reference_provenance({
        "history_id": "hist_reference_provenance",
        "operation": operation,
        "base_context_hash": current_base_hash,
        "record_before": {
            "source": {"branch_id": "main"},
            "active_turn": {"base_context_hash": previous_base_hash},
        },
    })

    assert provenance["history_operation"] == operation
    assert provenance["turn_relation"] == expected_relation
    assert provenance["target_comparison"] == expected_comparison
    assert provenance["classification_source"] == "history_alignment"
    if expected_comparison == "same":
        assert (
            provenance["current_turn_target_id"]
            == provenance["previous_turn_target_id"]
        )
    elif expected_comparison == "different":
        assert (
            provenance["current_turn_target_id"]
            != provenance["previous_turn_target_id"]
        )


def test_extract_authoritative_fixed_appearance_supports_current_and_legacy_schema():
    extracted = pipeline.extract_authoritative_fixed_appearance(
        "### Hana\n"
        "-Name\nHana\n"
        "-Appearance\n1girl, black hair, blue eyes\n"
        "-default_outfit\nblue dress\n\n"
        "### Sato\n"
        "-Appearance: 1boy, short black hair\n"
        "-default_outfit\nwhite shirt"
    )

    assert extracted == {
        "Hana": "1girl, black hair, blue eyes",
        "Sato": "1boy, short black hair",
    }


def test_extract_authoritative_default_outfits_keeps_complete_declared_order():
    extracted = pipeline.extract_authoritative_default_outfits(
        "### Elizabella\n"
        "-Appearance\n1girl, blonde hair\n"
        "-default_outfit\n"
        "white dress, halter dress, detached sleeves, white gloves, "
        "white choker, mini crown"
    )

    assert extracted == {
        "Elizabella": [
            "white dress",
            "halter dress",
            "detached sleeves",
            "white gloves",
            "white choker",
            "mini crown",
        ],
    }


def test_sparse_call1_set_keeps_unmentioned_default_outfit_features():
    reference = (
        "### Elizabella\n"
        "-Appearance\n1girl, blonde hair\n"
        "-default_outfit\n"
        "white dress, halter dress, detached sleeves, white gloves, "
        "white choker, mini crown"
    )
    states = pipeline.apply_wardrobe_events(
        {},
        [{"name": "Elizabella"}],
        [{
            "segment_id": "C008",
            "character": "Elizabella",
            "operation": "set",
            "items": ["white royal dress", "mini crown", "long blonde hair down"],
            "state_after": "clothed",
            "evidence": "short sparse description",
        }],
        "msg_current",
        selected_reference=reference,
    )

    worn = states["elizabella"]["current_wardrobe"]["worn"]
    assert worn[:6] == [
        "white dress",
        "halter dress",
        "detached sleeves",
        "white gloves",
        "white choker",
        "mini crown",
    ]
    assert "white royal dress" in worn
    assert "long blonde hair down" in worn


def test_parse_call1_wardrobe_change_schema_preserves_semantic_text():
    # 신규 CALL1은 items 대신 자연어 wardrobe_change만 낸다.
    current = "She changed into a swimsuit."
    _rendered, segments = pipeline._segment_current_context(current)
    analysis = pipeline.parse_call1_analysis(
        json.dumps({
            "reference_assignments": [],
            "history_characters": [],
            "current_characters": ["Hana"],
            "wardrobe_events": [{
                "segment_id": "C001",
                "character": "Hana",
                "operation": "replace",
                "wardrobe_change": "She changed into a swimsuit.",
                "state_after": "clothed",
                "evidence": "She changed into a swimsuit.",
            }],
            "unresolved_references": [],
        }),
        current,
        segments,
        "Hana",
    )

    assert analysis is not None
    event = analysis["wardrobe_events"][0]
    assert event["operation"] == "replace"
    assert event["wardrobe_change"] == "She changed into a swimsuit."
    assert event["items"] == []
    assert event["state_after"] == "clothed"
    assert event["evidence"] == "She changed into a swimsuit."


def test_parse_call1_accepts_literal_multiline_evidence_with_transport_whitespace():
    current = (
        "He undid the fastener.\n"
        "He pulled down his pants and underwear.\n"
        "His penis remained fully visible."
    )
    _rendered, segments = pipeline._segment_current_context(current)
    evidence = (
        "He undid the fastener. He pulled down his pants and underwear. "
        "His penis remained fully visible."
    )

    analysis = pipeline.parse_call1_analysis(
        json.dumps({
            "reference_assignments": [],
            "history_characters": [],
            "current_characters": ["Sato"],
            "wardrobe_events": [{
                "segment_id": "C001",
                "character": "Sato",
                "operation": "open",
                "wardrobe_change": (
                    "He pulled down his pants and underwear, leaving his penis exposed."
                ),
                "state_after": "partial",
                "evidence": evidence,
            }],
            "hairstyle_events": [],
            "unresolved_references": [],
        }),
        current,
        segments,
        "Sato",
    )

    assert analysis is not None
    assert len(analysis["wardrobe_events"]) == 1
    assert analysis["wardrobe_events"][0]["wardrobe_change"].endswith(
        "leaving his penis exposed."
    )
    assert not any(
        "복장 변경 근거 불일치" in warning
        for warning in analysis["validation_warnings"]
    )


def test_parse_call1_legacy_items_event_still_carried_for_backward_compat():
    # 과거 기록/구 출력의 items 형식은 하위 호환을 위해 계속 파싱한다.
    current = "She removed her gloves."
    _rendered, segments = pipeline._segment_current_context(current)
    analysis = pipeline.parse_call1_analysis(
        json.dumps({
            "reference_assignments": [],
            "history_characters": [],
            "current_characters": ["Hana"],
            "wardrobe_events": [{
                "segment_id": "C001",
                "character": "Hana",
                "operation": "remove",
                "items": ["white gloves", "elbow gloves"],
                "state_after": "clothed",
                "evidence": "She removed her gloves.",
            }],
            "unresolved_references": [],
        }),
        current,
        segments,
        "Hana",
    )

    assert analysis is not None
    event = analysis["wardrobe_events"][0]
    assert event["operation"] == "remove"
    assert event["items"] == ["white gloves", "elbow gloves"]
    assert event["wardrobe_change"] == ""


def test_apply_wardrobe_semantic_remove_applies_state_after_only(capsys):
    # TEST 5: items 없는 remove 이벤트. 옷 태그 해석은 CALL2 대기, body_state만 반영.
    reference = (
        "### Sato\n"
        "-Appearance\n1boy, short hair\n"
        "-default_outfit\n"
        "white shirt, blue pants, leather belt, underwear"
    )
    states = pipeline.apply_wardrobe_events(
        {},
        [{"name": "Sato"}],
        [{
            "segment_id": "C001",
            "character": "Sato",
            "operation": "remove",
            "wardrobe_change": (
                "He removed his pants and underwear, "
                "leaving his lower body exposed."
            ),
            "state_after": "bottomless",
            "evidence": "He lowered his pants and underwear.",
        }],
        "msg_current",
        selected_reference=reference,
    )

    wardrobe = states["sato"]["current_wardrobe"]
    assert wardrobe["worn"] == [
        "white shirt", "blue pants", "leather belt", "underwear",
    ]
    assert wardrobe["removed"] == []
    assert wardrobe["body_state"] == "bottomless"
    assert any("semantic event 보류" in line for line in capsys.readouterr().out.splitlines())


def test_apply_wardrobe_semantic_replace_preserves_default_until_call2(capsys):
    # TEST 2: items 없는 replace. swimsuit 태그는 CALL2가 결정하므로 기본 복장 보존.
    reference = (
        "### Hana\n"
        "-Appearance\n1girl, long hair\n"
        "-default_outfit\n"
        "white dress, halter dress, mini crown"
    )
    states = pipeline.apply_wardrobe_events(
        {},
        [{"name": "Hana"}],
        [{
            "segment_id": "C001",
            "character": "Hana",
            "operation": "replace",
            "wardrobe_change": "She changed into a swimsuit.",
            "state_after": "clothed",
            "evidence": "She changed into a swimsuit.",
        }],
        "msg_current",
        selected_reference=reference,
    )

    wardrobe = states["hana"]["current_wardrobe"]
    assert wardrobe["worn"] == ["white dress", "halter dress", "mini crown"]
    assert wardrobe["body_state"] == "clothed"
    assert any("semantic event 보류" in line for line in capsys.readouterr().out.splitlines())


def test_apply_wardrobe_semantic_reset_default_restores_outfit():
    # TEST 7: items 없는 reset_default. 의미상 복귀이므로 기본 복장을 복원한다.
    reference = (
        "### Hana\n"
        "-Appearance\n1girl, long hair\n"
        "-default_outfit\n"
        "white dress, halter dress, mini crown"
    )
    states = pipeline.apply_wardrobe_events(
        {},
        [{"name": "Hana"}],
        [{
            "segment_id": "C001",
            "character": "Hana",
            "operation": "reset_default",
            "wardrobe_change": "She changed back into her usual outfit.",
            "state_after": "clothed",
            "evidence": "She changed back into her usual clothes.",
        }],
        "msg_current",
        selected_reference=reference,
    )

    wardrobe = states["hana"]["current_wardrobe"]
    assert wardrobe["worn"] == ["white dress", "halter dress", "mini crown"]
    assert wardrobe["body_state"] == "clothed"


def test_call2_authority_base_restores_missing_fixed_and_default_tags():
    descriptors = [{
        "kind": "scene",
        "slot": 5,
        "characters": [{
            "name": "Elizabella",
            "positive": "girl, blonde hair, long hair, hair down, white royal dress, mini crown",
            "authority_exceptions": [],
            "outfit_state": {
                "body_state": "clothed",
                "worn": ["white royal dress", "mini crown"],
                "removed": [],
            },
        }],
    }]

    audits = pipeline.apply_call2_authority_base(
        descriptors,
        {
            "Elizabella": (
                "1girl, blonde hair, long hair, hair rings, two side up, "
                "hair between eyes, hair intakes, sidelocks, orange eyes"
            ),
        },
        {
            "Elizabella": [
                "white dress", "halter dress", "detached sleeves",
                "white gloves", "white choker", "mini crown",
            ],
        },
    )

    positive = descriptors[0]["characters"][0]["positive"]
    for required in (
        "hair rings", "two side up", "hair between eyes", "hair intakes",
        "detached sleeves", "white gloves", "white choker",
    ):
        assert required in positive
    assert audits == [{
        "kind": "scene",
        "slot": 5,
        "character": "Elizabella",
        "missing_fixed_added": [
            "hair rings", "two side up", "hair between eyes", "hair intakes",
            "sidelocks", "orange eyes",
        ],
        "missing_wardrobe_added": [
            "white dress", "halter dress", "detached sleeves",
            "white gloves", "white choker",
        ],
        "authority_exceptions": [],
        "forbidden_added_removed": [],
        "conflicts_removed": [],
        "rejected_exceptions": [],
        "semantic_status": "not_run",
    }]


def test_call2_authority_base_allows_only_explicit_exact_temporary_exceptions():
    descriptors = [{
        "kind": "scene",
        "slot": 7,
        "characters": [{
            "name": "Elizabella",
            "positive": "girl, blonde hair, long hair, twintails, two side up, orange eyes",
            "authority_exceptions": ["invented omission"],
            "outfit_state": {
                "body_state": "clothed",
                "worn": ["detached sleeves"],
                "removed": [],
            },
        }],
    }]

    audits = pipeline.apply_call2_authority_base(
        descriptors,
        {
            "Elizabella": (
                "1girl, blonde hair, long hair, hair rings, two side up, "
                "hair between eyes, hair intakes, orange eyes"
            ),
        },
        {"Elizabella": ["white dress", "detached sleeves", "white choker"]},
        {
            ("scene", 7, "elizabella"): {
                "authority_exceptions": ["two side up", "hair rings"],
                "conflicts": [],
            },
        },
        "ok",
    )

    character = descriptors[0]["characters"][0]
    tags = pipeline._split_top_level_authority_tags(character["positive"])
    assert "twintails" in tags
    assert "two side up" not in tags
    assert "hair rings" not in tags
    assert "hair between eyes" in tags
    assert "hair intakes" in tags
    assert "detached sleeves" in tags
    assert "white choker" in tags
    assert "authority_exceptions" not in character
    assert audits[0]["conflicts_removed"] == []
    assert audits[0]["rejected_exceptions"] == ["invented omission"]
    assert audits[0]["semantic_status"] == "ok"


def test_call2_semantic_authority_audit_removes_unsupported_hair_conflict():
    descriptors = [{
        "kind": "scene",
        "slot": 5,
        "anchor_before": "The queen is still on the stairs.",
        "anchor_after": "She enters later.",
        "characters": [{
            "name": "Elizabella",
            "positive": "girl, blonde hair, long hair, hair down, orange eyes",
            "outfit_state": {
                "body_state": "clothed",
                "worn": ["white royal dress", "mini crown"],
                "removed": [],
            },
        }],
    }]
    fixed = {
        "Elizabella": (
            "1girl, blonde hair, long hair, hair rings, two side up, "
            "hair between eyes, hair intakes, orange eyes"
        ),
    }
    defaults = {
        "Elizabella": ["white dress", "detached sleeves", "white choker", "mini crown"],
    }
    entries, entry_keys = pipeline._call2_authority_audit_entries(
        descriptors,
        fixed,
        defaults,
    )
    decisions, reason = pipeline._parse_call2_authority_audit_output(
        '{"entries":[{"id":1,"authority_exceptions":[],"conflicts":["hair down"]}]}',
        entries,
        entry_keys,
    )

    assert reason == ""
    audits = pipeline.apply_call2_authority_base(
        descriptors,
        fixed,
        defaults,
        decisions,
        "ok",
    )
    tags = pipeline._split_top_level_authority_tags(
        descriptors[0]["characters"][0]["positive"]
    )
    assert "hair down" not in tags
    assert "hair rings" in tags
    assert "two side up" in tags
    assert "detached sleeves" in tags
    assert audits[0]["conflicts_removed"] == ["hair down"]


def test_call2_existing_audit_call_repairs_missing_visual_bundle_and_camera():
    descriptors = [{
        "kind": "scene",
        "slot": 4,
        "scene_brief": (
            "Sato's pants and underwear are lowered and his exposed anatomy is visible."
        ),
        "continuity_note": (
            "Sato pulled down his pants and underwear, leaving his penis exposed."
        ),
        "camera": "upper body, straight-on",
        "scene": "1boy, interior, bedroom",
        "supplement": "The man thrusts his hips forward.",
        "characters": [{
            "name": "Sato",
            "positive": "boy, white shirt, blue pants, underwear, hips forward",
            "outfit_state": {
                "body_state": "bottomless",
                "worn": ["white shirt"],
                "removed": ["blue pants", "underwear"],
            },
        }],
    }]
    fixed = {"Sato": "1boy, short black hair, brown eyes"}
    defaults = {"Sato": ["white shirt", "blue pants", "underwear"]}
    entries, entry_keys = pipeline._call2_authority_audit_entries(
        descriptors,
        fixed,
        defaults,
    )

    assert entries[0]["scene_context"]["scene_brief"].startswith("Sato's pants")
    assert "leaving his penis exposed" in entries[0]["scene_context"]["continuity_note"]
    decisions, reason = pipeline._parse_call2_authority_audit_output(
        json.dumps({
            "entries": [{
                "id": 1,
                "authority_exceptions": ["blue pants", "underwear"],
                "forbidden_additions": [],
                "conflicts": [],
                "required_additions": ["bottomless", "penis", "pants down"],
                "scene_additions": ["nsfw"],
                "camera_replacement": "full body, straight-on",
            }],
        }),
        entries,
        entry_keys,
    )
    audits = pipeline.apply_call2_authority_base(
        descriptors,
        fixed,
        defaults,
        decisions,
        "ok",
    )

    assert reason == ""
    assert descriptors[0]["camera"] == "full body, straight-on"
    assert pipeline._split_top_level_authority_tags(descriptors[0]["scene"])[-1] == "nsfw"
    tags = pipeline._split_top_level_authority_tags(
        descriptors[0]["characters"][0]["positive"]
    )
    for required in ("bottomless", "penis", "pants down"):
        assert required in tags
    assert "blue pants" not in tags
    assert "underwear" not in tags
    assert descriptors[0]["characters"][0]["outfit_state"] == {
        "body_state": "bottomless",
        "worn": ["white shirt"],
        "removed": ["blue pants", "underwear"],
    }
    assert audits[0]["required_additions"] == ["bottomless", "penis", "pants down"]
    assert audits[0]["scene_additions"] == ["nsfw"]
    assert audits[0]["camera_replacement"] == "full body, straight-on"


def test_call2_semantic_audit_checks_conflicts_even_when_base_is_complete():
    descriptors = [{
        "kind": "scene",
        "slot": 2,
        "characters": [{
            "name": "Elizabella",
            "positive": (
                "girl, blonde hair, hair rings, two side up, orange eyes, "
                "white dress, detached sleeves, hair down"
            ),
            "outfit_state": {
                "body_state": "clothed",
                "worn": ["white dress", "detached sleeves"],
                "removed": [],
            },
        }],
    }]
    fixed = {
        "Elizabella": "1girl, blonde hair, hair rings, two side up, orange eyes",
    }
    defaults = {
        "Elizabella": ["white dress", "detached sleeves"],
    }

    entries, entry_keys = pipeline._call2_authority_audit_entries(
        descriptors,
        fixed,
        defaults,
    )
    decisions, reason = pipeline._parse_call2_authority_audit_output(
        '{"entries":[{"id":1,"authority_exceptions":[],"conflicts":["hair down"]}]}',
        entries,
        entry_keys,
    )
    pipeline.apply_call2_authority_base(
        descriptors,
        fixed,
        defaults,
        decisions,
        "ok",
    )

    assert reason == ""
    assert len(entries) == 1
    assert "hair down" not in pipeline._split_top_level_authority_tags(
        descriptors[0]["characters"][0]["positive"]
    )


def test_call2_semantic_audit_removes_and_logs_forbidden_identity_additions():
    descriptors = [{
        "kind": "scene",
        "slot": 5,
        "characters": [{
            "name": "Elizabella",
            "positive": (
                "girl, blonde hair, long hair, hair rings, two side up, "
                "orange eyes, white dress, detached sleeves, adult, fair skin, "
                "tsurime, hair down, straight hair"
            ),
            "outfit_state": {
                "body_state": "clothed",
                "worn": ["white dress", "detached sleeves"],
                "removed": [],
            },
        }],
    }]
    fixed = {
        "Elizabella": (
            "1girl, blonde hair, long hair, hair rings, two side up, orange eyes"
        ),
    }
    defaults = {"Elizabella": ["white dress", "detached sleeves"]}
    entries, entry_keys = pipeline._call2_authority_audit_entries(
        descriptors,
        fixed,
        defaults,
    )
    decisions, reason = pipeline._parse_call2_authority_audit_output(
        json.dumps({
            "entries": [{
                "id": 1,
                "authority_exceptions": [],
                "forbidden_additions": [
                    "adult", "fair skin", "tsurime", "hair down",
                ],
                "conflicts": ["straight hair"],
            }],
        }),
        entries,
        entry_keys,
    )

    audits = pipeline.apply_call2_authority_base(
        descriptors,
        fixed,
        defaults,
        decisions,
        "ok",
    )

    assert reason == ""
    tags = pipeline._split_top_level_authority_tags(
        descriptors[0]["characters"][0]["positive"]
    )
    for removed in ("adult", "fair skin", "tsurime", "hair down", "straight hair"):
        assert removed not in tags
    assert audits[0]["forbidden_added_removed"] == [
        "adult", "fair skin", "tsurime", "hair down",
    ]
    assert audits[0]["conflicts_removed"] == ["straight hair"]


def test_call2_semantic_audit_drops_out_of_candidate_value_not_whole_response():
    # 한 값이 후보 밖이면 응답 전체를 거부(degraded)하지 않고 그 값만 버린다.
    # 같은 엔트리의 다른 결정과 다른 엔트리 결정은 살아야 한다.
    descriptors = [
        {
            "kind": "scene",
            "slot": 5,
            "characters": [{
                "name": "Elizabella",
                "positive": (
                    "girl, blonde hair, long hair, hair rings, two side up, "
                    "orange eyes, white dress, detached sleeves, adult, "
                    "fair skin, hair down"
                ),
                "outfit_state": {
                    "body_state": "clothed",
                    "worn": ["white dress", "detached sleeves"],
                    "removed": [],
                },
            }],
        },
        {
            "kind": "scene",
            "slot": 6,
            "characters": [{
                "name": "Elizabella",
                "positive": (
                    "girl, blonde hair, long hair, orange eyes, white dress, "
                    "detached sleeves, tsurime, straight hair"
                ),
                "outfit_state": {
                    "body_state": "clothed",
                    "worn": ["white dress", "detached sleeves"],
                    "removed": [],
                },
            }],
        },
    ]
    fixed = {"Elizabella": "1girl, blonde hair, long hair, orange eyes"}
    defaults = {"Elizabella": ["white dress", "detached sleeves"]}
    entries, entry_keys = pipeline._call2_authority_audit_entries(
        descriptors,
        fixed,
        defaults,
    )
    # id=1: "white royal dress" 는 generated_positive 에 정확히 없으므로 후보 밖.
    #       이 값만 버리고 adult/fair skin/hair down 은 살려야 한다.
    # id=2: 정상 결정.
    decisions, reason = pipeline._parse_call2_authority_audit_output(
        json.dumps({
            "entries": [
                {
                    "id": 1,
                    "authority_exceptions": [],
                    "forbidden_additions": [
                        "white royal dress", "adult", "fair skin", "hair down",
                    ],
                    "conflicts": [],
                },
                {
                    "id": 2,
                    "authority_exceptions": [],
                    "forbidden_additions": ["tsurime"],
                    "conflicts": ["straight hair"],
                },
            ],
        }),
        entries,
        entry_keys,
    )

    # 응답 전체가 거부되지 않는다.
    assert reason == ""
    key1 = ("scene", 5, "elizabella")
    key2 = ("scene", 6, "elizabella")
    assert decisions[key1]["forbidden_additions"] == [
        "adult", "fair skin", "hair down",
    ]
    assert decisions[key2]["forbidden_additions"] == ["tsurime"]
    assert decisions[key2]["conflicts"] == ["straight hair"]

    pipeline.apply_call2_authority_base(
        descriptors,
        fixed,
        defaults,
        decisions,
        "ok",
    )
    tags1 = pipeline._split_top_level_authority_tags(
        descriptors[0]["characters"][0]["positive"]
    )
    tags2 = pipeline._split_top_level_authority_tags(
        descriptors[1]["characters"][0]["positive"]
    )
    for removed in ("adult", "fair skin", "hair down"):
        assert removed not in tags1
    # "white royal dress" 는 원래 positive 에 없었으므로 결과 태그에 없다.
    assert "white royal dress" not in tags1
    assert "tsurime" not in tags2
    assert "straight hair" not in tags2


def test_call2_untrusted_removed_or_nude_state_cannot_skip_default_restore():
    descriptors = [{
        "kind": "scene",
        "slot": 4,
        "characters": [{
            "name": "Elizabella",
            "positive": "girl, blonde hair",
            "authority_exceptions": ["detached sleeves"],
            "outfit_state": {
                "body_state": "nude",
                "worn": [],
                "removed": ["white dress", "detached sleeves"],
            },
        }],
    }]

    audits = pipeline.apply_call2_authority_base(
        descriptors,
        {"Elizabella": "1girl, blonde hair"},
        {"Elizabella": ["white dress", "detached sleeves"]},
        {("scene", 4, "elizabella"): {
            "authority_exceptions": [],
            "conflicts": [],
        }},
        "ok",
    )

    character = descriptors[0]["characters"][0]
    assert character["outfit_state"] == {
        "body_state": "clothed",
        "worn": ["white dress", "detached sleeves"],
        "removed": [],
    }
    assert "authority_exceptions" not in character
    assert audits[0]["rejected_exceptions"] == ["detached sleeves"]


def test_call2_audited_explicit_nude_change_can_except_default_outfit():
    descriptors = [{
        "kind": "scene",
        "slot": 4,
        "characters": [{
            "name": "Elizabella",
            "positive": "girl, blonde hair",
            "outfit_state": {
                "body_state": "nude",
                "worn": [],
                "removed": ["white dress", "detached sleeves"],
            },
        }],
    }]

    pipeline.apply_call2_authority_base(
        descriptors,
        {"Elizabella": "1girl, blonde hair"},
        {"Elizabella": ["white dress", "detached sleeves"]},
        {("scene", 4, "elizabella"): {
            "authority_exceptions": ["white dress", "detached sleeves"],
            "conflicts": [],
        }},
        "ok",
    )

    character = descriptors[0]["characters"][0]
    assert character["outfit_state"] == {
        "body_state": "nude",
        "worn": [],
        "removed": ["white dress", "detached sleeves"],
    }
    assert "authority_exceptions" not in character
    assert character["outfit_state"]["removed"] == [
        "white dress",
        "detached sleeves",
    ]


@pytest.mark.parametrize(
    ("operation", "comparison", "has_visual", "expected_type"),
    [
        ("new", "unavailable", True, "IGNORE"),
        ("append", "different", True, "CONTINUITY"),
        ("duplicate", "same", True, "REROLL"),
        ("reroll", "same", True, "REROLL"),
        ("reroll", "different", True, "SOFT_REFERENCE"),
        ("unknown", "unavailable", True, "SOFT_REFERENCE"),
        ("append", "different", False, "IGNORE"),
    ],
)
def test_last_visual_reference_classification_is_metadata_only(
    operation,
    comparison,
    has_visual,
    expected_type,
):
    visual = (
        {"Hana": {"positive_tags": "words must not affect classification"}}
        if has_visual
        else {}
    )

    result = pipeline.classify_last_visual_reference(
        {
            "history_operation": operation,
            "target_comparison": comparison,
            "turn_relation": "test relation",
        },
        visual,
    )

    assert result["reference_type"] == expected_type
    assert result["character_count"] == (1 if has_visual else 0)


def test_call1_prompt_state_excludes_generated_visual_and_duplicate_default():
    stored = {
        "hana": {
            "canonical_name": "Hana",
            "default_outfit_reference": "duplicate default marker",
            "current_wardrobe": {
                "body_state": "clothed",
                "worn": ["blue dress"],
                "removed": [],
            },
            "wardrobe_timeline": [{"segment_id": "C001", "operation": "wear"}],
            "last_seen_message_id": "msg_previous",
            "last_visual_reference": {
                "source_slot": 7,
                "positive_tags": "generated visual contamination marker",
                "outfit_state": {"body_state": "clothed", "worn": ["blue dress"]},
            },
        }
    }

    prompt_state = pipeline._call1_state_for_prompt(
        stored,
        "### Hana\n-Name\nHana\n-Appearance\n1girl\n-default_outfit\nblue dress",
    )

    assert "last_visual_reference" not in prompt_state["hana"]
    assert "default_outfit_reference" not in prompt_state["hana"]
    assert prompt_state["hana"]["current_wardrobe"]["worn"] == ["blue dress"]
    assert prompt_state["hana"]["wardrobe_timeline"] == [
        {"segment_id": "C001", "operation": "wear"}
    ]
    assert stored["hana"]["last_visual_reference"]["source_slot"] == 7
    assert stored["hana"]["default_outfit_reference"] == "duplicate default marker"


def test_call1_prompt_state_preserves_default_when_lb_extra_character_is_missing():
    stored = {
        "hana": {
            "canonical_name": "Hana",
            "default_outfit_reference": "Hana fallback outfit",
            "current_wardrobe": {"body_state": "clothed", "worn": ["blue dress"]},
            "last_visual_reference": {
                "positive_tags": "generated visual contamination marker"
            },
        },
        "bob": {
            "canonical_name": "Bob",
            "default_outfit_reference": "Bob duplicate outfit",
            "current_wardrobe": {"body_state": "clothed", "worn": ["black suit"]},
        },
    }

    prompt_state = pipeline._call1_state_for_prompt(
        stored,
        "### Bob\n-Name\nBob\n-Appearance\n1boy\n-default_outfit\nblack suit",
    )

    assert "last_visual_reference" not in prompt_state["hana"]
    assert prompt_state["hana"]["default_outfit_reference"] == "Hana fallback outfit"
    assert "default_outfit_reference" not in prompt_state["bob"]
    assert stored["bob"]["default_outfit_reference"] == "Bob duplicate outfit"


@pytest.mark.asyncio
async def test_persistent_history_path_uses_compact_call2_and_updates_wardrobe(monkeypatch):
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages))
        if task_key == "illustration_call1":
            request_text = "\n".join(message["content"] for message in messages)
            assert "generated visual contamination marker" not in request_text
            assert "duplicate default marker" not in request_text
            assert "blue dress" in request_text
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
        if _call_name(task_key) == "CALL2-AUTHORITY-AUDIT":
            return _authority_audit_response(
                messages,
                authority_exceptions=["blue dress"],
            )
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
            "default_outfit_reference": "duplicate default marker",
            "current_wardrobe": {
                "body_state": "clothed",
                "worn": ["blue dress"],
                "removed": [],
            },
            "last_visual_reference": {
                "source_slot": 7,
                "positive_tags": "generated visual contamination marker",
                "outfit_state": {
                    "body_state": "clothed",
                    "worn": ["blue dress"],
                    "removed": [],
                },
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
        "illustration_call2",
    ]
    assert result["balanced_fallback_used"] is False
    assert result["reference_provenance"]["turn_relation"] == "PRIOR_COMMITTED_TURN"
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
        if _call_name(task_key) == "CALL2-AUTHORITY-AUDIT":
            return _authority_audit_response(messages)
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

    assert [task_key for task_key, _messages in calls] == [
        "illustration_call2",
        "illustration_call2",
    ]
    assert result["balanced_fallback_used"] is True
    assert result["enhanced_narrative"] == "She waits by the door."
    state = result["character_states_after"]["hana"]
    assert state["current_wardrobe"]["worn"] == ["white shirt", "black skirt"]
    assert "source" not in state["current_wardrobe"]
    assert state["last_visual_reference"]["outfit_state"]["worn"] == [
        "white shirt", "black skirt",
    ]


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
        if _call_name(task_key) == "CALL2-AUTHORITY-AUDIT":
            return _authority_audit_response(messages)
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
        "illustration_call2",
    ]
    assert result["balanced_fallback_used"] is True
    state = result["character_states_after"]["hana"]
    assert state["current_wardrobe"]["worn"] == ["red cardigan", "pleated skirt"]
    assert "source" not in state["current_wardrobe"]


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
            if _call_name(task_key) == "CALL2-AUTHORITY-AUDIT":
                return _authority_audit_response(messages)
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
            if _call_name(task_key) == "CALL2-AUTHORITY-AUDIT":
                return _authority_audit_response(messages)
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
        "illustration_call2",
        "illustration_call3",
    ]
    assert result["balanced_fallback_used"] is True
    assert result["enhanced_narrative"] == "She waits by the door."
    assert result["items"][0]["speak"] == 'Hana: "Still here." #normal'
