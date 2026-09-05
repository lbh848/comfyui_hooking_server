"""Shared wardrobe handoff and resolver exclusion regression coverage."""

import json

import pytest

from modes import illustration_context_pipeline as pipeline


@pytest.mark.parametrize(
    "name,narrative,note,operation",
    [
        (
            "Hiyori",
            "Hiyori carries her trench coat over her arm, wearing an ivory chiffon blouse and white miniskirt.",
            "Hiyori wears the ivory chiffon blouse with a frilled neckline and pink ribbon, and the white miniskirt. The trench coat is carried over her arm, not worn.",
            "remove",
        ),
        (
            "Mina",
            "Mina drapes her rain jacket across her forearm and keeps walking in her green shirt and linen trousers.",
            "Mina wears the same green short-sleeved shirt and linen trousers. Her rain jacket rests across her forearm, not on her body.",
            "remove",
        ),
        (
            "Ren",
            "Ren lifts the jacket from the chair, puts both arms through its sleeves, and fastens the front.",
            "Ren now wears the brown jacket buttoned over his cream shirt. The jacket is no longer on the chair.",
            "wear",
        ),
    ],
)
def test_plan_resolution_survives_binding_and_public_handoff(name, narrative, note, operation):
    raw = json.dumps({"scene_plan": [{
        "anchor_segment": "C001", "characters": [name],
        "scene_brief": narrative, "continuity_note": note,
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
    )
    public = pipeline._public_call2_scene_plan(bound[0])
    assert note in public["continuity_note"]
    assert narrative in public["continuity_note"]
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
