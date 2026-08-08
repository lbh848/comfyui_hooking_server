from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALL2_SYSTEM = ROOT / "prompts" / "lighbd" / "system.txt"
CALL2_THOUGHTS = ROOT / "prompts" / "lighbd" / "thoughts.txt"
PIPELINE_PY = ROOT / "modes" / "illustration_context_pipeline.py"


def test_call2_resolves_wardrobe_change_as_semantic_instruction():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")

    assert "`operation` + `wardrobe_change`" in system
    assert "semantic instructions, never as a ready-made tag list" in system


def test_call2_default_compatible_prose_keeps_authoritative_tags():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")

    assert "keep the authoritative default tags" in system
    assert "the default tags win" in thoughts


def test_call2_remove_targets_semantic_garment_cluster():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")

    assert "every tag describing that same physical garment" in system
    assert (
        "removing one physical garment removes every tag describing that same garment"
        in thoughts
    )


def test_call2_replace_keeps_independent_accessories_and_minimal_new_tags():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")

    assert "Independent accessories" in system
    assert "minimum tags for the new outfit" in system


def test_call2_reset_default_restores_exact_outfit():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")

    assert "restore the exact authoritative `default_outfit`" in system


def test_call2_body_state_suppresses_conflicting_garments():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")

    assert "suppress the conflicting default/current garments" in system


def test_call2_smallest_change_principle():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")

    assert "make the smallest change" in system
    assert "make the smallest change the evidence supports" in thoughts


def test_call2_injection_message_states_items_may_be_empty():
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "`items` may be empty and that is expected" in source


def test_call2_authority_audit_rejects_associated_accessory_removal():
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert (
        "Do not grant an authority exception for an accessory merely because it is"
        in source
    )
    assert "physically associated with a removed garment" in source
    assert "removing a belt does not authorize removing `belt pouch`" in source


def test_call2_builds_one_coherent_explicit_bundle_without_tag_dictionary():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")

    assert "Coherent Explicit Scene Bundle" in system
    assert "at least five complementary, story-supported visual details" in system
    assert "not from a fixed palette" in system
    assert "Do not consult or simulate an external tag dictionary" in system
    assert "camera whose framing actually contains every body part" in system
    assert "source#`/`target#` counterparts symmetrical" in system
    assert "silently assemble and cross-check a coherent scene-specific bundle" in thoughts
    assert "never invent a new act, anatomy, intensity, or garment state" in thoughts


def test_call2_plan_handoff_stays_natural_and_schema_remains_compact():
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "Write scene_brief as natural language, not a field menu or tag list" in source
    assert '"scene_brief": "objective visual moment to expand"' in source
    assert "lower_body_exposure" not in source
    assert '"must_show"' not in source
    assert "required_additions" in source
    assert "camera_replacement" in source
