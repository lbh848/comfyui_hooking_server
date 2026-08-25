import json
from pathlib import Path

from modes.illust_prompt_builder import IllustPromptBuilder
from modes import illustration_context_pipeline as pipeline
from modes.visual_profiles import cards_to_character_profiles


def _profiles():
    character = cards_to_character_profiles("Adachi", [{
                "id": "civilian",
                "label": "카드 1",
                "selection_guide": "변신 전의 인간 모습.",
                "appearance": ["brown hair", "brown eyes"],
                "default_outfit": ["hoodie", "jeans"],
            }, {
                "id": "despair",
                "label": "카드 2",
                "selection_guide": "몸 자체가 절망 형태로 변한 뒤의 모습.",
                "appearance": ["white hair", "red eyes", "black horns"],
                "default_outfit": ["black armor"],
                "face_tags": "white hair, red eyes",
                "use_profile_embedding": True,
            }])
    return {"Adachi": character}


def _segments():
    first = "Adachi stood in her hoodie."
    second = "Dark light swallowed her, and her whole body changed into her despair form."
    return {
        "C001": {"text": first, "start": 0, "end": len(first)},
        "C002": {
            "text": second,
            "start": len(first) + 1,
            "end": len(first) + 1 + len(second),
        },
    }


def _event():
    return {
        "segment_id": "C002",
        "character": "Adachi",
        "target_visual_profile_id": "despair",
        "visual_change": "Her whole body changed into her despair form.",
        "evidence": "her whole body changed into her despair form",
        "confidence": 0.96,
    }


def test_call1_prompt_dynamically_selects_cards_from_narrative_state():
    prompt = (
        Path(__file__).parents[1] / "prompts" / "lighbd" / "enhance.txt"
    ).read_text(encoding="utf-8")

    assert "Registered character cards" in prompt
    assert "including on first appearance" in prompt
    assert "Use the registered default only when the narrative does not establish another card" in prompt
    assert "no nested outfit choice" in prompt
    assert "fallback when the story and scene context do not call for different attire" in prompt
    assert "not a mandatory outfit" in prompt
    assert "target_outfit_id" not in prompt
    assert "never choose by a fixed keyword list" in prompt


def test_call1_visual_event_requires_literal_evidence_and_registered_ids():
    raw = {
        "reference_assignments": [],
        "history_characters": [],
        "current_characters": ["Adachi"],
        "visual_base_events": [_event()],
        "wardrobe_events": [],
        "hairstyle_events": [],
        "unresolved_references": [],
    }

    parsed = pipeline.parse_call1_analysis(
        json.dumps(raw),
        "\n".join(item["text"] for item in _segments().values()),
        _segments(),
        "Adachi",
        visual_profiles=_profiles(),
    )

    assert parsed is not None
    assert parsed["visual_base_events"][0]["target_visual_profile_id"] == "despair"
    assert "target_outfit_id" not in parsed["visual_base_events"][0]

    raw["visual_base_events"][0]["target_visual_profile_id"] = "invented"
    rejected = pipeline.parse_call1_analysis(
        json.dumps(raw),
        "\n".join(item["text"] for item in _segments().values()),
        _segments(),
        "Adachi",
        visual_profiles=_profiles(),
    )
    assert rejected["visual_base_events"] == []
    assert any("등록되지 않은 외형 프로필" in warning for warning in rejected["validation_warnings"])


def test_visual_event_resets_flat_profile_wardrobe_base_and_removes_legacy_outfit_state():
    before = {
        "adachi": {
            "canonical_name": "Adachi",
            "active_visual_profile_id": "civilian",
            "active_outfit_id": "casual",
            "current_wardrobe": {
                "body_state": "partial",
                "worn": ["hoodie"],
                "removed": ["jeans"],
            },
        }
    }

    after = pipeline.apply_visual_base_events(
        before,
        [{"name": "Adachi"}],
        [_event()],
        "message-2",
        _profiles(),
    )

    state = after["adachi"]
    assert state["active_visual_profile_id"] == "despair"
    assert "active_outfit_id" not in state
    assert state["current_wardrobe"] == {
        "body_state": "clothed",
        "worn": ["black armor"],
        "removed": [],
    }
    assert state["visual_base_timeline"][-1]["message_id"] == "message-2"


def test_scene_plan_binds_different_profiles_before_and_after_transformation():
    plans = [{
        "plan_id": "P001",
        "slot": 1,
        "anchor_segment": "C001",
        "characters": ["Adachi"],
        "scene_brief": "Before the change.",
    }, {
        "plan_id": "P002",
        "slot": 2,
        "anchor_segment": "C002",
        "characters": ["Adachi"],
        "scene_brief": "After the change.",
    }]

    bound = pipeline.bind_scene_plan_wardrobes(
        plans,
        ["C001", "C002"],
        {},
        [{"name": "Adachi"}],
        [],
        "message-2",
        default_outfits={"Adachi": ["hoodie", "jeans"]},
        visual_profiles=_profiles(),
        visual_base_events=[_event()],
    )

    assert bound[0]["visual_base_snapshot"]["Adachi"]["visual_profile_id"] == "civilian"
    assert bound[0]["wardrobe_snapshot"]["Adachi"]["worn"] == ["hoodie", "jeans"]
    assert bound[1]["visual_base_snapshot"]["Adachi"]["visual_profile_id"] == "despair"
    assert bound[1]["wardrobe_snapshot"]["Adachi"]["worn"] == ["black armor"]
    assert "Do not choose another profile" in bound[1]["visual_base_authority"]


def test_authority_repair_uses_descriptor_profile_not_global_default():
    base = pipeline.visual_base_snapshot(
        pipeline.apply_visual_base_events(
            {},
            [{"name": "Adachi"}],
            [_event()],
            "message-2",
            _profiles(),
        ),
        ["Adachi"],
        _profiles(),
    )
    descriptors = [{
        "kind": "scene",
        "slot": 2,
        "camera": "",
        "scene": "",
        "supplement": "",
        "visual_base_snapshot": base,
        "characters": [{
            "name": "Adachi",
            "positive": "smile",
            "outfit_state": {"body_state": "clothed", "worn": [], "removed": []},
        }],
    }]

    pipeline.apply_call2_authority_base(
        descriptors,
        {"Adachi": "brown hair, brown eyes"},
        {"Adachi": ["hoodie", "jeans"]},
        semantic_decisions={},
        semantic_status="degraded",
    )

    positive = descriptors[0]["characters"][0]["positive"]
    assert "white hair" in positive
    assert "red eyes" in positive
    assert "black horns" in positive
    assert "black armor" in positive
    assert "brown hair" not in positive
    assert "hoodie" not in positive


def test_profile_embedding_paths_keep_logical_character_name():
    characters = [{
        "name": "Adachi",
        "_visual_profile_id": "despair",
        "_use_profile_embedding": True,
    }]

    cache = IllustPromptBuilder.build_cache_path(["Adachi"], "demo", characters)
    face = IllustPromptBuilder.build_face_id_dir(
        ["Adachi"],
        "demo",
        {"face_id_str": 0.55},
        characters,
    )

    assert cache["list"][0] == {
        "emb_path": "soya_bot/demo/Adachi/_visual_profiles/despair/cache.pt",
        "CHAR": "Adachi",
    }
    assert face["list"][0]["ipa_path"] == (
        "soya_bot/demo/Adachi/_visual_profiles/despair/cache.ipadpt"
    )
    assert face["list"][0]["CHAR"] == "Adachi"
