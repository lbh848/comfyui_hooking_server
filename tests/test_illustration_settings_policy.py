"""Required illustration stages and the single optional dialogue switch."""

from copy import deepcopy
import ast
import json
from pathlib import Path

import pytest

from modes import illustration_context_pipeline as pipeline
from modes.visual_profiles import cards_to_character_profiles


@pytest.mark.parametrize(("saved", "enabled", "limit"), [
    ({}, True, 3),
    ({"call2_parallel_enabled": False, "call2_parallel_max_concurrency": 7}, False, 7),
    ({"call2_parallel_enabled": True, "call2_parallel_max_concurrency": 5}, True, 5),
    ({"call2_parallel_enabled": False, "call2_parallel_max_concurrency": 6,
      "call2_detail_parallel_enabled": True, "call2_detail_parallel_max_concurrency": 2}, True, 2),
    ({"call2_parallel_enabled": True, "call2_parallel_max_concurrency": 9,
      "call2_detail_parallel_enabled": False, "call2_detail_parallel_max_concurrency": 4}, False, 4),
    ({"call2_parallel_max_concurrency": 7, "call2_detail_parallel_enabled": False}, False, 7),
    ({"call2_parallel_enabled": False, "call2_detail_parallel_max_concurrency": 8}, False, 8),
])
def test_detail_parallel_migration_is_initial_only_and_does_not_mutate_saved_values(saved, enabled, limit):
    before = deepcopy(saved)
    normalized = pipeline.merged_toggles(saved)
    assert normalized["call2_detail_parallel_enabled"] is enabled
    assert normalized["call2_detail_parallel_max_concurrency"] == limit
    assert pipeline.merged_toggles(normalized) == normalized
    assert saved == before
    # Once the missing DETAIL fields have been materialized, changing PLAN
    # no longer changes either DETAIL control.
    normalized["call2_parallel_enabled"] = not enabled
    normalized["call2_parallel_max_concurrency"] = 16 if limit != 16 else 1
    changed_plan = pipeline.merged_toggles(normalized)
    assert changed_plan["call2_detail_parallel_enabled"] is enabled
    assert changed_plan["call2_detail_parallel_max_concurrency"] == limit
    changed_plan["call2_detail_parallel_enabled"] = not enabled
    changed_plan["call2_detail_parallel_max_concurrency"] = 1
    changed_detail = pipeline.merged_toggles(changed_plan)
    assert changed_detail["call2_parallel_enabled"] == changed_plan["call2_parallel_enabled"]
    assert changed_detail["call2_parallel_max_concurrency"] == changed_plan["call2_parallel_max_concurrency"]


@pytest.mark.parametrize(("raw", "expected"), [(0, 1), (99, 16), ("5", 5), (None, 3), ("invalid", 3)])
def test_detail_limit_normalization_does_not_reset_plan_settings(raw, expected):
    normalized = pipeline.merged_toggles({
        "call2_parallel_enabled": False, "call2_parallel_max_concurrency": 7,
        "call2_detail_parallel_enabled": True, "call2_detail_parallel_max_concurrency": raw,
    })
    assert normalized["call2_parallel_enabled"] is False
    assert normalized["call2_parallel_max_concurrency"] == 7
    assert normalized["call2_detail_parallel_enabled"] is True
    assert normalized["call2_detail_parallel_max_concurrency"] == expected


def test_server_and_pipeline_register_the_same_independent_parallel_defaults():
    source = Path(__file__).resolve().parents[1] / "server.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DEFAULT_CONFIG" for target in node.targets))
    toggle_node = next(value for key, value in zip(assignment.value.keys, assignment.value.values)
        if isinstance(key, ast.Constant) and key.value == "illustration_context_toggles")
    defaults = ast.literal_eval(toggle_node)
    assert "compat_character_divider" not in defaults
    assert "compat_character_divider" not in pipeline.DEFAULT_TOGGLES
    for key in ("call2_parallel_enabled", "call2_parallel_max_concurrency",
                "call2_detail_parallel_enabled", "call2_detail_parallel_max_concurrency",
                *(f"call2_detail_parallel_slow_retry_{suffix}" for suffix in (
                    "enabled", "remaining", "progress_enabled", "progress_threshold",
                    "tps_enabled", "tps_threshold", "condition_operator"))):
        assert defaults[key] == pipeline.DEFAULT_TOGGLES[key]


def test_detail_slow_retry_defaults_and_saved_values_are_independent_of_plan():
    plan = {f"call2_parallel_slow_retry_{suffix}": value for suffix, value in {
        "enabled": True, "remaining": 4, "progress_enabled": False,
        "progress_threshold": 80, "tps_enabled": True, "tps_threshold": 40.0,
        "condition_operator": "or",
    }.items()}
    normalized = pipeline.merged_toggles(plan)
    assert not normalized["call2_detail_parallel_slow_retry_enabled"]
    for suffix in ("remaining", "progress_enabled", "progress_threshold", "tps_enabled",
                   "tps_threshold", "condition_operator"):
        key = f"call2_detail_parallel_slow_retry_{suffix}"
        assert normalized[key] == pipeline.DEFAULT_TOGGLES[key]
    saved = {**plan, "call2_detail_parallel_slow_retry_enabled": True,
             "call2_detail_parallel_slow_retry_remaining": 2,
             "call2_detail_parallel_slow_retry_progress_threshold": 25,
             "call2_detail_parallel_slow_retry_tps_threshold": 3.5,
             "call2_detail_parallel_slow_retry_condition_operator": "and"}
    before = deepcopy(saved)
    normalized = pipeline.merged_toggles(saved)
    assert saved == before
    assert all(normalized[key] == value for key, value in saved.items())
    assert pipeline.merged_toggles(normalized) == normalized


@pytest.mark.parametrize(("suffix", "raw", "expected"), [
    ("remaining", 0, 1), ("remaining", 99, 16), ("remaining", "bad", 1),
    ("progress_threshold", 100, 99), ("progress_threshold", None, 50),
    ("tps_threshold", 0, 0.1), ("tps_threshold", "7.5", 7.5),
    ("condition_operator", "xor", "and"), ("condition_operator", "OR", "or"),
])
def test_detail_slow_retry_normalization_keeps_plan_values(suffix, raw, expected):
    key = f"call2_detail_parallel_slow_retry_{suffix}"
    result = pipeline.merged_toggles({key: raw, "call2_parallel_max_concurrency": 7,
        "call2_parallel_slow_retry_remaining": 4,
        "call2_parallel_slow_retry_progress_threshold": 80,
        "call2_parallel_slow_retry_tps_threshold": 30})
    assert result[key] == expected
    assert result["call2_parallel_max_concurrency"] == 7
    assert result["call2_parallel_slow_retry_remaining"] == 4
    assert result["call2_parallel_slow_retry_progress_threshold"] == 80
    assert result["call2_parallel_slow_retry_tps_threshold"] == 30


@pytest.mark.parametrize("enabled", [False, True])
def test_retired_stage_switches_do_not_survive_normalization(enabled):
    saved = {"call1_enabled": enabled, "profile_resolve_enabled": enabled,
             "key_visual": False, "call1_backtranslate_enabled": False,
             "multi_char_mask_enabled": False, "direction": "keep this direction"}
    before = deepcopy(saved)
    normalized = pipeline.merged_toggles(saved)
    assert "call1_enabled" not in normalized
    assert "profile_resolve_enabled" not in normalized
    for key in ("key_visual", "call1_backtranslate_enabled", "multi_char_mask_enabled", "direction"):
        assert normalized[key] == saved[key]
    assert pipeline.merged_toggles(normalized) == normalized
    assert saved == before


@pytest.mark.parametrize("prompt_format", ["v1", "v3", "chansub"])
@pytest.mark.parametrize(("characters", "legacy_divider"), [
    ([{"name": "Aster", "positive": "red hair, white coat"},
      {"name": "Beryl", "positive": "blue hair, black jacket"}], "newline"),
    ([{"name": "Ren", "positive": "silver hair, green cloak"},
      {"name": "Nora", "positive": "black hair, orange scarf"}], "pipe"),
    ([{"name": "Mina", "positive": "brown hair, blue shirt"}], "newline"),
])
def test_fixed_raw_character_separator_preserves_downstream_binding(
    prompt_format, characters, legacy_divider,
):
    from modes.chansub_prompt_builder import ChansubPromptBuilder
    from modes.illust_prompt_builder import IllustPromptBuilder

    saved = {"prompt_format": prompt_format,
             "compat_character_divider": legacy_divider,
             "positive_note": "soft lighting", "negative_note": "blur"}
    before = deepcopy(saved)
    normalized = pipeline.merged_toggles(saved)
    assert "compat_character_divider" not in normalized
    assert normalized["positive_note"] == saved["positive_note"]
    assert normalized["negative_note"] == saved["negative_note"]
    assert pipeline.merged_toggles(normalized) == normalized
    assert saved == before

    descriptor = {"characters": characters, "camera": "medium shot",
                  "scene": "at a table", "supplement": "a book rests on the table"}
    prompts = {"call2_preset": "[Positive]\n[NAME]\n{name}\n[SETUP]\n{setup}"
               "\n[CHAR]\n{char}\n[SUPPLEMENT]\n{supplement}\n[Negative]\n"}
    # Also cover direct callers still carrying a retired saved setting.
    raw, negative = pipeline.build_raw_prompt(descriptor, "", prompts, saved)
    assert pipeline.build_raw_prompt(descriptor, "", prompts, normalized) == (raw, negative)
    cards = [{"name": character["name"]} for character in characters]
    bound = IllustPromptBuilder.parse_sections(raw, lb_extra=cards, characters=cards)
    blocks = bound["char"].split(" | ")
    assert len(blocks) == len(characters)
    for block, character in zip(blocks, characters):
        assert block == f"{character['name']}, {character['positive']}"
    assert negative == "blur"

    if prompt_format == "chansub":
        sections = IllustPromptBuilder.parse_sections(raw)
        flat = ChansubPromptBuilder.build_positive_prompt(
            sections["setup"], sections["char"], sections["supplement"], {}, {},
        )
        raw_blocks = sections["char"].split(" | ")
        assert raw_blocks == [character["positive"] for character in characters]
        for block in raw_blocks:
            assert block in flat
        assert flat.index(raw_blocks[0]) <= flat.index(raw_blocks[-1])


@pytest.mark.parametrize(("legacy", "speak", "expected"), [
    (None, None, True), (None, False, False), (None, True, True),
    (False, None, False), (True, None, True),
    (False, False, False), (False, True, False),
    (True, False, False), (True, True, True),
])
def test_dialogue_migration_preserves_effective_state_without_hidden_veto(legacy, speak, expected):
    saved = {}
    if legacy is not None:
        saved["call3_enabled"] = legacy
    if speak is not None:
        saved["speak_enabled"] = speak
    before = deepcopy(saved)
    normalized = pipeline.merged_toggles(saved)
    assert normalized["speak_enabled"] is expected
    assert "call3_enabled" not in normalized
    assert pipeline.merged_toggles(normalized) == normalized
    assert saved == before
    # A normalized OFF value can be re-enabled through the one visible switch.
    normalized["speak_enabled"] = True
    assert pipeline.merged_toggles(normalized)["speak_enabled"] is True
    normalized["speak_enabled"] = False
    assert pipeline.merged_toggles(normalized)["speak_enabled"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Mina", "Ren"])
async def test_profile_resolution_skips_llm_for_single_profile_current_character(monkeypatch, name):
    calls = []

    async def fake_call(stage, _messages, *_args, **_kwargs):
        calls.append(stage)
        assert stage == "CHARACTER-RESOLVE"
        return json.dumps({"characters": [{"name": name}], "uncertainties": []})

    async def unexpected_translation(*_args, **_kwargs):
        raise AssertionError("No current character needs multi-profile resolution")

    monkeypatch.setattr(pipeline, "_call_pipeline_llm", fake_call)
    monkeypatch.setattr(pipeline, "prepare_profile_context_translations", unexpected_translation)
    profiles = {
        name: cards_to_character_profiles(name, [{"id": "only", "appearance": ["black hair"]}]),
        "Absent": cards_to_character_profiles("Absent", [{"id": "daily"}, {"id": "changed"}]),
    }
    _output, result = await pipeline.resolve_profiles_before_generation(
        payload={"chats": [{"role": "char", "data": f"{name} arrives alone."}]},
        toggles={"profile_resolve_enabled": False}, history_plan=None,
        visual_profiles=profiles,
    )
    assert calls == ["CHARACTER-RESOLVE"]
    assert result["current_characters"] == [{"name": name, "confidence": 1.0}]
