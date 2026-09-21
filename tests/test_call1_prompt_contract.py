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

    assert "Use `replace` for a semantically clear transition from the current outfit to a different outfit" in prompt
    assert "Use `add` or `wear` only for an incremental addition that leaves the existing outfit in place" in prompt
    assert "Use `remove` only when specific worn items are taken off while the rest of the current outfit remains in effect" in prompt
    assert "Use `open`/`close`/`adjust` only when the garment stays worn but its worn state changes" in prompt


def test_call1_uses_one_start_state_plus_sparse_current_changes():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert '"wardrobe_at_start"' in prompt
    assert "immediately before C001" in prompt
    assert "not a sequence of complete outfit snapshots" in prompt
    assert "never compute or emit per-segment full outfit snapshots" in prompt
    assert "If it only re-describes an already known/default outfit, emit no event" in prompt


def test_call1_natural_change_text_outranks_coarse_enum_hints():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert "`wardrobe_at_start[].state` and `wardrobe_change` are the meaning-bearing handoff" in prompt
    assert "must never simplify or contradict it" in prompt
    assert "evidence may span consecutive numbered segments" in prompt
    assert "A garment that is merely lowered, pooled around the thighs or ankles, displaced, or only partly removed still counts as lost coverage" in prompt


def test_call1_start_state_cannot_mix_terminal_nudity_with_remaining_garments():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert "Do not combine mutually exclusive intermediate and terminal states" in prompt
    assert "lowered, pooled around the thighs or ankles, displaced" in prompt
    assert "First resolve each CURRENT character's coherent wardrobe, coverage, exposure, and continuity-relevant carried/nearby-garment state" in prompt
    assert "`nude`: no clothing provides coverage of the groin, chest, or other normally covered areas" in prompt
    assert "Do not euphemize an explicit body part or visible result" in prompt
    assert "do not reduce a compound change to only one intermediate gesture" in prompt
    assert "Track nude, topless, bottomless, underwear-only, open, displaced, and partially removed clothing as real wardrobe states" in prompt
    assert "Pick `state_after` by whether the relevant body area is still covered, not by whether the garment came fully off the body" in prompt


def test_call1_restored_semantic_sections_cover_actual_isomorphic_and_opposite_states():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    # Actual regression class: a terminal state cannot coexist with a garment
    # that remains open, unhooked, lowered, draped, or otherwise attached.
    assert '"wardrobe_at_start"' in prompt
    assert "Do not combine mutually exclusive intermediate and terminal states" in prompt
    assert 'An empty outfit is not "unknown" when nudity is established' in prompt
    assert "`nude`: no clothing provides coverage of the groin, chest, or other normally covered areas" in prompt

    # Isomorphic transitions remain separate decisions instead of one coarse
    # state: adding, replacing, opening, and losing coverage are not synonyms.
    assert "`wardrobe_events` is a sparse, evidence-bearing change history" in prompt
    assert "putting a shirt over a swimsuit adds the shirt" in prompt
    assert "A `replace` ends the prior worn outfit as a whole" in prompt
    assert "when the garment stays worn but its worn state changes" in prompt
    assert "Pick `state_after` by whether the relevant body area is still covered" in prompt
    assert "pooled around the thighs or ankles" in prompt

    # Opposite valid cases keep their meaning: off-camera fabric is not removed,
    # and disordered tied hair is not silently converted to hair-down.
    assert "A garment absent from a camera view is not removed" in prompt
    assert "`hairstyle_events` tracks only hairstyle arrangement transitions" in prompt
    assert '"a girl with disheveled long twintails" still has twintails' in prompt
    assert "Emit an event only when the narrative itself establishes that the arrangement was actually undone, released, replaced, added, or restored" in prompt
    assert "messy, disheveled, tousled, spread around the body, or loose-looking" in prompt
