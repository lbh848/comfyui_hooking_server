import json
from pathlib import Path

from modes.illust_prompt_builder import IllustPromptBuilder
from modes import illustration_context_pipeline as pipeline
from modes.visual_profiles import cards_to_character_profiles


def _profiles():
    character = cards_to_character_profiles("Adachi", [{
                "id": "civilian",
                "label": "카드 1",
                "aliases": ["Adachi_Civilian"],
                "selection_guide": "변신 전의 인간 모습.",
                "appearance": ["brown hair", "brown eyes"],
                "default_outfit": ["hoodie", "jeans"],
            }, {
                "id": "despair",
                "label": "카드 2",
                "aliases": ["Adachi_Despair"],
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


def _initial_base():
    return {
        "character": "Adachi",
        "target_visual_profile_id": "despair",
        "initial_state": "Adachi is already in her despair form when CURRENT begins.",
        "visual_profile_name": "Adachi_Despair",
        "anchor_segment": "START",
        "evidence": [],
        "confidence": 0.97,
    }


def _reversion_segments():
    first = "Adachi stood in her despair form."
    second = "The darkness dissolved, and she changed back into her civilian form."
    return {
        "C001": {"text": first, "start": 0, "end": len(first)},
        "C002": {
            "text": second,
            "start": len(first) + 1,
            "end": len(first) + 1 + len(second),
        },
    }


def _reversion_event():
    return {
        "segment_id": "C002",
        "character": "Adachi",
        "target_visual_profile_id": "civilian",
        "visual_change": "She changed back into her civilian form.",
        "evidence": "changed back into her civilian form",
        "confidence": 0.98,
    }


def test_call1_prompt_tracks_initial_card_before_sparse_transitions():
    prompt = (
        Path(__file__).parents[1] / "prompts" / "lighbd" / "enhance.txt"
    ).read_text(encoding="utf-8")

    assert "Registered character cards" in prompt
    assert '"profile_events"' in prompt
    assert '"segment_id": "START or one exact Cxxx ID"' in prompt
    assert "correct it at START" in prompt
    assert "later explicit release" in prompt
    assert "use the registered default only when neither narrative evidence nor a previous tracked profile" in prompt
    assert "distinctive eyes or pupils" in prompt
    assert "initial_visual_bases" not in prompt
    assert "target_visual_profile_id" not in prompt
    assert "visual_base_events" not in prompt
    assert "no nested outfit choice" in prompt
    assert "fallback when the story and scene context do not call for different attire" in prompt
    assert "not a mandatory outfit" in prompt
    assert "target_outfit_id" not in prompt
    assert "never from a trigger word or a fixed keyword list" in prompt


def test_call1_profile_event_maps_exact_meaningful_name_to_internal_route():
    raw = {
        "reference_assignments": [],
        "history_characters": [],
        "current_characters": ["Adachi"],
        "profile_events": [{
            "segment_id": "START",
            "character": "Adachi",
            "profile": "Adachi_Civilian",
            "state": "Adachi begins CURRENT in her civilian form.",
        }, {
            "segment_id": "C002",
            "character": "Adachi",
            "profile": "Adachi_Despair",
            "state": "Her whole body changes into her despair form.",
        }],
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
    assert parsed["profile_events"][1]["profile"] == "Adachi_Despair"
    assert parsed["visual_base_events"][0]["target_visual_profile_id"] == "despair"
    assert "target_outfit_id" not in parsed["visual_base_events"][0]

    raw["profile_events"][1]["profile"] = "invented"
    rejected = pipeline.parse_call1_analysis(
        json.dumps(raw),
        "\n".join(item["text"] for item in _segments().values()),
        _segments(),
        "Adachi",
        visual_profiles=_profiles(),
    )
    assert rejected["visual_base_events"] == []
    assert any("등록되지 않은 프로필 의미 이름" in warning for warning in rejected["validation_warnings"])


def test_call1_profile_timeline_keeps_start_and_reversion_as_four_field_events():
    segments = _reversion_segments()
    raw = {
        "reference_assignments": [],
        "history_characters": [],
        "current_characters": ["Adachi"],
        "profile_events": [{
            "segment_id": "START",
            "character": "Adachi",
            "profile": "Adachi_Despair",
            "state": "Adachi is already in her despair form when CURRENT begins.",
        }, {
            "segment_id": "C002",
            "character": "Adachi",
            "profile": "Adachi_Civilian",
            "state": "She changes back into her civilian form.",
        }],
        "wardrobe_events": [],
        "hairstyle_events": [],
        "unresolved_references": [],
    }

    parsed = pipeline.parse_call1_analysis(
        json.dumps(raw),
        "\n".join(item["text"] for item in segments.values()),
        segments,
        "Adachi",
        visual_profiles=_profiles(),
    )

    assert parsed is not None
    assert [set(item) - {"confidence"} for item in parsed["profile_events"]] == [
        {"segment_id", "character", "profile", "state"},
        {"segment_id", "character", "profile", "state"},
    ]
    assert parsed["initial_visual_bases"][0]["target_visual_profile_id"] == "despair"
    assert parsed["initial_visual_bases"][0]["anchor_segment"] == "START"
    assert parsed["visual_base_events"][0]["target_visual_profile_id"] == "civilian"


def test_initial_visual_base_seeds_nondefault_before_reversion_event():
    seeded = pipeline.apply_initial_visual_bases(
        {},
        [_initial_base()],
        "message-1",
        _profiles(),
    )
    assert seeded["adachi"]["active_visual_profile_id"] == "despair"
    assert seeded["adachi"]["current_wardrobe"]["worn"] == ["black armor"]

    plans = [{
        "plan_id": "P001",
        "slot": 1,
        "anchor_segment": "C001",
        "characters": ["Adachi"],
        "scene_brief": "Before release.",
    }, {
        "plan_id": "P002",
        "slot": 2,
        "anchor_segment": "C002",
        "characters": ["Adachi"],
        "scene_brief": "After release.",
    }]
    bound = pipeline.bind_scene_plan_wardrobes(
        plans,
        ["C001", "C002"],
        seeded,
        [{"name": "Adachi"}],
        [],
        "message-1",
        default_outfits={"Adachi": ["hoodie", "jeans"]},
        visual_profiles=_profiles(),
        visual_base_events=[_reversion_event()],
    )

    assert bound[0]["visual_base_snapshot"]["Adachi"]["visual_profile_id"] == "despair"
    assert bound[0]["wardrobe_snapshot"]["Adachi"]["worn"] == ["black armor"]
    assert bound[1]["visual_base_snapshot"]["Adachi"]["visual_profile_id"] == "civilian"
    assert bound[1]["wardrobe_snapshot"]["Adachi"]["worn"] == ["hoodie", "jeans"]


def test_call1_start_correction_overrides_wrong_tracked_profile_and_wardrobe():
    tracked = {
        "adachi": {
            "canonical_name": "Adachi",
            "active_visual_profile_id": "civilian",
            "current_wardrobe": {
                "body_state": "clothed",
                "worn": ["hoodie", "jeans"],
                "removed": [],
            },
        }
    }

    corrected = pipeline.apply_initial_visual_bases(
        tracked,
        [_initial_base()],
        "message-2",
        _profiles(),
    )

    assert corrected["adachi"]["active_visual_profile_id"] == "despair"
    assert corrected["adachi"]["current_wardrobe"]["worn"] == ["black armor"]
    assert corrected["adachi"]["visual_base_timeline"][-1][
        "previous_visual_profile_id"
    ] == "civilian"


def test_call1_prompt_state_replaces_internal_profile_id_with_meaningful_name():
    prompt_state = pipeline._call1_state_for_prompt(
        {
            "adachi": {
                "canonical_name": "Adachi",
                "active_visual_profile_id": "despair",
                "visual_base_timeline": [{"target_visual_profile_id": "despair"}],
            }
        },
        visual_profiles=_profiles(),
    )

    assert prompt_state["adachi"]["active_visual_profile"] == "Adachi_Despair"
    assert "active_visual_profile_id" not in prompt_state["adachi"]
    assert "visual_base_timeline" not in prompt_state["adachi"]


def test_shiho_trace_uses_corrupted_profile_before_c037_and_card1_after_release():
    shiho_profiles = {
        "Shiho": cards_to_character_profiles("Shiho", [{
            "id": "card_1",
            "label": "카드 1",
            "aliases": ["Shiho_Overcome_School"],
            "selection_guide": "변신이 풀린 일반 상태.",
            "appearance": ["brown hair", "red eyes"],
            "default_outfit": ["school uniform", "blazer", "red plaid skirt"],
        }, {
            "id": "card_84d30493bc23",
            "label": "카드 2",
            "aliases": ["Shiho_Corrupted Heart"],
            "selection_guide": "타락한 마법소녀 Corrupted Heart 변신 상태.",
            "appearance": ["ash-blonde hair", "violet eyes"],
            "default_outfit": ["black and violet combat outfit", "gauntlets"],
        }]),
    }
    segment_texts = {
        "C007": "그 중심에서, 이형의 힘을 두른 소꿉친구 소녀가 가쁜 숨을 몰아쉬고 있었다.",
        "C034": "시호야, 변신 풀 수 있겠어?",
        "C037": "그녀의 몸을 감싸고 있던 검정과 보랏빛의 퇴폐적인 전투복이 입자 형태의 검은 안개가 되어 흩어져 갔다.",
        "C038": "그 아래에서 나타난 것은 짙은 남색 블레이저와 붉은 타탄체크 플리츠 스커트였다.",
        "C039": "변신이 풀린 순간, 시호의 몸이 휘청였다.",
    }
    segments = {
        segment_id: {"text": text, "start": index, "end": index + len(text)}
        for index, (segment_id, text) in enumerate(segment_texts.items())
    }
    raw = {
        "reference_assignments": [],
        "history_characters": ["Shiho"],
        "current_characters": ["Shiho"],
        "profile_events": [{
            "segment_id": "START",
            "character": "Shiho",
            "profile": "Shiho_Corrupted Heart",
            "state": "Shiho is already in her Corrupted Heart transformation.",
        }, {
            "segment_id": "C037",
            "character": "Shiho",
            "profile": "Shiho_Overcome_School",
            "state": "Shiho releases her transformation and returns to her school form.",
        }],
        "wardrobe_events": [],
        "hairstyle_events": [],
        "unresolved_references": [],
    }
    parsed = pipeline.parse_call1_analysis(
        json.dumps(raw, ensure_ascii=False),
        "\n".join(segment_texts.values()),
        segments,
        "Shiho",
        visual_profiles=shiho_profiles,
    )
    assert parsed is not None

    seeded = pipeline.apply_initial_visual_bases(
        {},
        parsed["initial_visual_bases"],
        "message-shiho",
        shiho_profiles,
    )
    plans = [{
        "plan_id": "P001",
        "slot": 4,
        "anchor_segment": "C007",
        "characters": ["Shiho"],
        "scene_brief": "Before transformation release.",
    }, {
        "plan_id": "P002",
        "slot": 5,
        "anchor_segment": "C037",
        "characters": ["Shiho"],
        "scene_brief": "Transformation release.",
    }, {
        "plan_id": "P003",
        "slot": 6,
        "anchor_segment": "C039",
        "characters": ["Shiho"],
        "scene_brief": "After transformation release.",
    }]
    bound = pipeline.bind_scene_plan_wardrobes(
        plans,
        list(segment_texts),
        seeded,
        [{"name": "Shiho"}],
        [],
        "message-shiho",
        default_outfits={"Shiho": ["school uniform"]},
        visual_profiles=shiho_profiles,
        visual_base_events=parsed["visual_base_events"],
    )

    assert [
        item["visual_base_snapshot"]["Shiho"]["visual_profile_id"]
        for item in bound
    ] == ["card_84d30493bc23", "card_1", "card_1"]
    assert "Shiho is already in her Corrupted Heart transformation" in bound[0][
        "visual_base_authority"
    ]
    assert "releases her transformation" in bound[1]["visual_base_authority"]
    assert "card_84d30493bc23" not in bound[0]["visual_base_authority"]

    descriptors = [{
        "kind": "scene",
        "slot": item["slot"],
        "visual_base_snapshot": item["visual_base_snapshot"],
        "characters": [{
            "name": "Shiho",
            "positive": "placeholder",
            "outfit_state": item["wardrobe_snapshot"]["Shiho"],
        }],
    } for item in bound]
    audit_entries, _keys = pipeline._call2_authority_audit_entries(
        descriptors,
        {},
        {},
    )
    assert [entry["visual_profile"] for entry in audit_entries] == [
        "Shiho_Corrupted Heart",
        "Shiho_Overcome_School",
        "Shiho_Overcome_School",
    ]
    assert [entry["profile_state"] for entry in audit_entries] == [
        "Shiho is already in her Corrupted Heart transformation.",
        "Shiho releases her transformation and returns to her school form.",
        "Shiho releases her transformation and returns to her school form.",
    ]
    assert all("visual_profile_id" not in entry for entry in audit_entries)


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


def test_authority_audit_entries_expose_each_scene_meaningful_profile_name():
    despair_base = pipeline.visual_base_snapshot(
        pipeline.apply_initial_visual_bases(
            {}, [_initial_base()], "message-1", _profiles()
        ),
        ["Adachi"],
        _profiles(),
    )
    civilian_base = pipeline.visual_base_snapshot(
        pipeline.apply_visual_base_events(
            pipeline.apply_initial_visual_bases(
                {}, [_initial_base()], "message-1", _profiles()
            ),
            [{"name": "Adachi"}],
            [_reversion_event()],
            "message-1",
            _profiles(),
        ),
        ["Adachi"],
        _profiles(),
    )
    descriptors = [{
        "kind": "scene",
        "slot": 1,
        "visual_base_snapshot": despair_base,
        "characters": [{
            "name": "Adachi",
            "positive": "white hair, red eyes, black horns, black armor",
            "outfit_state": {
                "body_state": "clothed",
                "worn": ["black armor"],
                "removed": [],
            },
        }],
    }, {
        "kind": "scene",
        "slot": 2,
        "visual_base_snapshot": civilian_base,
        "characters": [{
            "name": "Adachi",
            "positive": "brown hair, brown eyes, hoodie, jeans",
            "outfit_state": {
                "body_state": "clothed",
                "worn": ["hoodie", "jeans"],
                "removed": [],
            },
        }],
    }]

    entries, _keys = pipeline._call2_authority_audit_entries(
        descriptors,
        {},
        {},
    )

    assert [entry["visual_profile"] for entry in entries] == [
        "Adachi_Despair",
        "Adachi_Civilian",
    ]
    assert all("visual_profile_id" not in entry for entry in entries)
    assert [entry["visual_profile_label"] for entry in entries] == [
        "카드 2",
        "카드 1",
    ]


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
