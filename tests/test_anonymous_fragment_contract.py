import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import illustration_context_pipeline as pipeline
from modes.chansub_prompt_builder import ChansubPromptBuilder
from modes.illust_prompt_builder import (
    IllustPromptBuilder,
    allows_anonymous_fragment_negative_relaxation,
)


def _negative_inputs():
    tags = {
        "negative_presets": {
            "scene-negative": [
                "bad anatomy",
                "extra person",
                "foreshortening",
                "extra limbs",
            ],
        },
    }
    settings = {
        "anima_negative_preset": "scene-negative",
        "sdxl_negative_preset": "scene-negative",
        "chansub_workflow_type": "anima",
    }
    return tags, settings


def test_actual_fragment_case_removes_only_conflicting_preset_negatives():
    tags, settings = _negative_inputs()
    bot = {
        "system_prompt_preset": "배포_1차 싱글 V5",
        "characters": [{
            "name": "Hana",
            "character_negative": "wrong identity",
        }],
    }

    allowed = allows_anonymous_fragment_negative_relaxation(bot, True)
    negative = IllustPromptBuilder.build_negative_prompt(
        tags,
        settings,
        ["Hana"],
        bot,
        allow_anonymous_fragment=allowed,
    )

    assert allowed is True
    assert "extra person" not in negative
    assert negative.count("foreshortening") == 2
    assert negative.count("bad anatomy") == 2
    assert negative.count("extra limbs") == 2
    assert negative.count("wrong identity") == 2


def test_isomorphic_fragment_case_uses_same_contract_for_chansub():
    tags, settings = _negative_inputs()
    bot = {"system_prompt_preset": "배포_1차 싱글 V5"}

    negative = ChansubPromptBuilder.build_negative_prompt(
        tags,
        settings,
        allow_anonymous_fragment=allows_anonymous_fragment_negative_relaxation(
            bot,
            True,
        ),
    )

    assert negative == "bad anatomy, foreshortening, extra limbs"


def test_opposite_solo_case_and_other_presets_keep_all_negatives():
    tags, settings = _negative_inputs()
    single_v5 = {"system_prompt_preset": "배포_1차 싱글 V5"}
    another_contract = {"system_prompt_preset": "배포_1차 멀티 V5"}

    assert allows_anonymous_fragment_negative_relaxation(single_v5, False) is False
    assert allows_anonymous_fragment_negative_relaxation(another_contract, True) is False

    negative = IllustPromptBuilder.build_negative_prompt(
        tags,
        settings,
        allow_anonymous_fragment=False,
    )
    assert negative.count("extra person") == 2
    assert negative.count("foreshortening") == 2


def test_plan_boolean_survives_plan_parse_and_detail_assignment():
    plan, reason = pipeline.parse_call2_plan(
        json.dumps({
            "scene_plan": [{
                "anchor_segment": "C001",
                "characters": ["Mina"],
                "scene_brief": (
                    "Mina braces against pressure from a connected off-frame arm."
                ),
                "anonymous_partner_fragment": True,
            }],
        }),
        pipeline.merged_toggles({
            "key_visual": False,
            "scene_mode": "manual",
            "output_count_min": 1,
            "output_count_max": 1,
        }),
        "Mina braces.\n\n[Slot 0]",
        segment_slot_map={"C001": 0},
    )

    assert reason == ""
    selected = pipeline.bind_scene_plan_wardrobes(
        plan["scene_plan"],
        ["C001"],
        {},
        [{"name": "Mina"}],
        [],
        "test-message",
        wardrobe_at_start=[{
            "character": "Mina",
            "state": "Mina still wears the same rumpled jacket.",
        }],
    )[0]
    assert selected["anonymous_partner_fragment"] is True
    assert pipeline._public_call2_scene_plan(selected)[
        "anonymous_partner_fragment"
    ] is True

    detail = """<lb-xnai>
scenes[1]:
  - camera: tight contact close-up
    characters[1]:
      - name: Mina
        positive: girl, red hair, rumpled jacket
        outfit_state:
          body_state: clothed
          worn: [rumpled jacket]
          removed: []
    scene: cropped arm entering from frame edge
    slot: 0
    supplement: The off-frame arm stays connected at the crop boundary.
</lb-xnai>"""
    descriptors, detail_reason = pipeline._parse_call2_detail_output(
        detail,
        pipeline.merged_toggles({"key_visual": False}),
        [0],
        ["S001"],
        "TEST-ANONYMOUS-FRAGMENT-CONTRACT",
        assigned_characters_by_slot={0: ["Mina"]},
        assigned_scene_context_by_slot={
            0: {
                "scene_brief": selected["scene_brief"],
                "continuity_note": selected["continuity_note"],
                "anonymous_partner_fragment": True,
            },
        },
    )
    assert detail_reason == ""
    assert descriptors[0]["anonymous_partner_fragment"] is True
    assert "anonymous_partner_fragment: true" in pipeline.descriptors_to_toon(
        descriptors
    )


def test_soft_reference_withholds_generated_payload_but_continuity_keeps_it():
    visual = {
        "Mina": {
            "positive_tags": "untrusted pose and substitute garment marker",
        },
    }
    soft = pipeline._classified_visual_reference_content(
        {
            "reference_type": "SOFT_REFERENCE",
            "reference_reason": "turn relationship unavailable",
        },
        visual,
    )
    continuity = pipeline._classified_visual_reference_content(
        {
            "reference_type": "CONTINUITY",
            "reference_reason": "earlier committed turn",
        },
        visual,
    )

    assert "generated prompt payload is intentionally withheld" in soft
    assert "untrusted pose and substitute garment marker" not in soft
    assert "untrusted pose and substitute garment marker" in continuity


@pytest.mark.parametrize(
    ("brief", "fragment_required"),
    [
        (
            "Doyun covers an off-frame visitor's mouth while the visitor pulls away.",
            True,
        ),
        (
            "Mara steadies an off-frame guest by both shoulders.",
            True,
        ),
        (
            "An off-frame guard covers Nari's mouth while she recoils.",
            True,
        ),
        (
            "Nari adjusts her own sleeve with nobody else visible.",
            False,
        ),
    ],
)
def test_detail_handoff_preserves_semantic_brief_and_explicit_fragment_decision(
    brief,
    fragment_required,
):
    public = pipeline._public_call2_scene_plan({
        "plan_id": "S001",
        "slot": 4,
        "anchor_segment": "C001",
        "characters": ["Nari"],
        "scene_brief": brief,
        "anonymous_partner_fragment": fragment_required,
    })

    assert public["scene_brief"] == brief
    assert public["anonymous_partner_fragment"] is fragment_required
