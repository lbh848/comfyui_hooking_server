from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALL1_PROMPT = ROOT / "prompts" / "lighbd" / "enhance.txt"


def test_call1_wardrobe_operations_use_replace_without_set_or_contextual_reset():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert (
        '"operation": "wear|add|remove|replace|open|close|adjust|nude|topless|bottomless|reset_default"'
        in prompt
    )
    assert "Do not emit `set` or `contextual_reset`." in prompt


def test_call1_replace_is_a_semantic_full_outfit_transition():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert "`replace` for a complete transition to another outfit" in prompt
    assert "`wear`/`add` for an incremental addition" in prompt
    assert "`remove` for specified removed items" in prompt
    assert "`open`/`close`/`adjust` when the same garment stays worn" in prompt


def test_call1_uses_one_start_state_plus_sparse_current_changes():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert '"wardrobe_at_start"' in prompt
    assert "immediately before C001" in prompt
    assert "not a sequence of complete outfit snapshots" in prompt
    assert "Do not generate Danbooru tags or per-segment full outfit snapshots" in prompt
    assert "repeated description of an existing/default outfit is not a change" in prompt


def test_call1_natural_change_text_outranks_coarse_enum_hints():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert "`wardrobe_at_start[].state` and `wardrobe_change` are the meaning-bearing handoff" in prompt
    assert "must never simplify or contradict the natural language" in prompt
    assert "spans consecutive segments as one event" in prompt
    assert "lowered or displaced clothing may stop covering a region while remaining attached" in prompt


def test_call1_start_state_cannot_mix_terminal_nudity_with_remaining_garments():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert "one physically possible instant" in prompt
    assert "attached, draped, lowered, pooled" in prompt
    assert "make one final semantic choice" in prompt
    assert "only when no garment remains worn or attached" in prompt
    assert "describe the exact exposure without terminal wording" in prompt
    assert "do not also describe it as worn, open, unhooked" in prompt
    assert "summarizes coverage only" in prompt
    assert "never erases an explicitly remaining displaced garment" in prompt


def test_call1_restored_semantic_sections_cover_actual_isomorphic_and_opposite_states():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    # Actual regression class: a terminal state cannot coexist with a garment
    # that remains open, unhooked, lowered, draped, or otherwise attached.
    assert "WARDROBE AT START" in prompt
    assert "make one final semantic choice" in prompt
    assert "Use `fully unclothed`, `wearing nothing`" in prompt
    assert "only when no garment remains worn or attached" in prompt

    # Isomorphic transitions remain separate decisions instead of one coarse
    # state: adding, replacing, opening, and losing coverage are not synonyms.
    assert "CURRENT WARDROBE EVENTS" in prompt
    assert "Putting a shirt over a swimsuit adds the shirt" in prompt
    assert "A replacement ends the previous worn outfit as a whole" in prompt
    assert "the same garment stays worn but its worn state changes" in prompt
    assert "COVERAGE AND PHYSICAL STATE" in prompt
    assert "pooled around the thighs or ankles" in prompt

    # Opposite valid cases keep their meaning: off-camera fabric is not removed,
    # and disordered tied hair is not silently converted to hair-down.
    assert "A garment absent from a camera view is not removed" in prompt
    assert "HAIRSTYLE EVENTS" in prompt
    assert "A disheveled ponytail is still a ponytail" in prompt
    assert "unless the narrative establishes that it was undone" in prompt
    assert "messy, tousled, spread, swaying, or loose-looking hair" in prompt
