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

    assert "Choose the operation from the meaning of the full sentence" in prompt
    assert "A `replace` ends the prior worn outfit as a whole" in prompt
    assert "do not require separate `remove` events" in prompt
    assert "putting a shirt over a swimsuit adds the shirt" in prompt


def test_call1_remains_a_sparse_change_extractor():
    prompt = CALL1_PROMPT.read_text(encoding="utf-8")

    assert "not a sequence of complete outfit snapshots" in prompt
    assert "Never compute or emit a full current-outfit snapshot" in prompt
    assert "merely repeats an already known state, emit no event" in prompt
