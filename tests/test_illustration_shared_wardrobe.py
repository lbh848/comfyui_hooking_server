"""Shared wardrobe handoff and resolver exclusion regression coverage."""

import json

import pytest

from modes import illustration_context_pipeline as pipeline


@pytest.mark.parametrize(
    "name,narrative,start_state,operation",
    [
        (
            "Hiyori",
            "Hiyori carries her trench coat over her arm, wearing an ivory chiffon blouse and white miniskirt.",
            "Hiyori begins CURRENT wearing the trench coat over her ivory blouse and white miniskirt.",
            "remove",
        ),
        (
            "Mina",
            "Mina drapes her rain jacket across her forearm and keeps walking in her green shirt and linen trousers.",
            "Mina begins CURRENT wearing her rain jacket over a green shirt and linen trousers.",
            "remove",
        ),
        (
            "Ren",
            "Ren lifts the jacket from the chair, puts both arms through its sleeves, and fastens the front.",
            "Ren begins CURRENT in a cream shirt while the brown jacket rests on the chair.",
            "wear",
        ),
    ],
)
def test_call1_resolution_survives_binding_and_public_handoff(
    name,
    narrative,
    start_state,
    operation,
):
    rejected_plan_note = f"{name} wears a PLAN-invented silver costume."
    raw = json.dumps({"scene_plan": [{
        "anchor_segment": "C001", "characters": [name],
        "scene_brief": narrative, "continuity_note": rejected_plan_note,
    }]})
    parsed, reason = pipeline.parse_call2_plan(
        raw,
        pipeline.merged_toggles({"output_count_min": 1, "output_count_max": 1, "key_visual": False}),
        narrative + "\n\n[Slot 0]",
        segment_slot_map={"C001": 0},
    )
    assert parsed is not None, reason
    original = parsed["scene_plan"][0]["scene_brief"]
    bound = pipeline.bind_scene_plan_wardrobes(
        parsed["scene_plan"], ["C001"], {}, [{"name": name}],
        [{"segment_id": "C001", "character": name, "operation": operation,
          "wardrobe_change": narrative, "evidence": narrative, "state_after": "clothed"}],
        "test-message", default_outfits={name: ["old default sweater"]},
        wardrobe_at_start=[{"character": name, "state": start_state}],
    )
    public = pipeline._public_call2_scene_plan(bound[0])
    assert start_state in public["continuity_note"]
    assert narrative in public["continuity_note"]
    assert rejected_plan_note not in public["continuity_note"]
    assert public["scene_brief"] == original
    assert "wardrobe_snapshot" not in public
    assert "old default sweater" not in public["continuity_note"]
    assert public["slot"] == 0
    assert public["characters"] == [name]


@pytest.mark.parametrize(
    "name,notes",
    [
        ("Hiyori", ["Doyoon has no registered entry and is omitted.",
                    "Hoshino is mentioned only hypothetically."]),
        ("Mina", ["The unnamed courier remains an anonymous interaction partner."]),
        ("Ren", []),
    ],
)
def test_resolver_explanations_do_not_discard_usable_current_selection(name, notes):
    narrative = f"{name} waits beside the door."
    _, segments = pipeline._segment_current_context(narrative)
    parsed = pipeline.parse_call1_analysis(
        json.dumps({"wardrobe_events": [], "hairstyle_events": []}),
        narrative, segments, name,
        resolved_characters={
            "current_characters": [{"name": name, "confidence": 1.0}],
            "history_characters": [], "uncertainties": notes,
        },
    )
    assert parsed is not None
    assert parsed["fallback_required"] is False
    assert parsed["current_characters"][0]["name"] == name


def test_missing_required_selection_still_uses_existing_fallback():
    narrative = "An unidentified person waits beside the door."
    _, segments = pipeline._segment_current_context(narrative)
    parsed = pipeline.parse_call1_analysis(
        json.dumps({"wardrobe_events": [], "hairstyle_events": []}),
        narrative, segments, "Mina",
        resolved_characters={"current_characters": [], "uncertainties": ["Identity unknown."]},
    )
    assert parsed["fallback_required"] is True
