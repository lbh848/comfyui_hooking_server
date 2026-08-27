import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALL1_ENHANCE = ROOT / "prompts" / "lighbd" / "enhance.txt"
CALL2_SYSTEM = ROOT / "prompts" / "lighbd" / "system.txt"
CALL2_THOUGHTS = ROOT / "prompts" / "lighbd" / "thoughts.txt"
PIPELINE_PY = ROOT / "modes" / "illustration_context_pipeline.py"


def test_model_facing_prompt_files_do_not_assign_internal_call_stage_roles():
    prompt_dir = ROOT / "prompts" / "lighbd"
    for prompt_path in prompt_dir.glob("*.txt"):
        prompt = prompt_path.read_text(encoding="utf-8")
        assert not re.search(
            r"\bCALL[1235](?:-[A-Z-]+)?\b",
            prompt,
            re.IGNORECASE,
        ), prompt_path.name


def test_call2_resolves_wardrobe_change_as_semantic_instruction():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")

    assert "`operation` + `wardrobe_change`" in system
    assert "semantic instructions, never as a ready-made tag list" in system


def test_call2_default_outfit_is_fallback_and_context_can_create_replacement():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")

    assert "fallback visual reference" in system
    assert "create a coherent outfit suited to that context" in system
    assert "explicit garment sentence is not required" in thoughts
    assert "without keyword matching" in thoughts
    assert "Do not mix abandoned default garments" in thoughts


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

    assert "restore the exact `default_outfit` reference" in system


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

    assert "Do not except an accessory merely because it is associated" in source
    assert "the accessory itself is removed" in source
    assert "the coherent outfit is replaced" in source


def test_call2_authority_audit_allows_contextual_outfit_creation():
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "default_outfit is a fallback reference rather than identity" in source
    assert "scene-appropriate replacement outfit may except" in source
    assert "preserve any default item that logically remains" in source
    assert "full narrative, role, activity, occasion, setting, and continuity" in source


def test_call2_fixed_appearance_requires_explicit_narrative_change():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "authoritative identity of the already-selected visual profile" in system
    assert "Only a direct, explicit statement in the actual narrative" in system
    assert "active evidence-bearing history event" in thoughts
    assert "actual narrative or an active hairstyle_history event contains" in source
    assert "literal evidence of a temporary physical change" in source
    assert "assigned scene selection controls the visual beat but has no appearance authority" in source
    assert "without turning appearance wording into a temporary replacement" in source
    assert "generated visual " in source
    assert "state structurally differs" in source
    assert "scene-appropriate replacement outfit may except" in source


def test_call1_dishevelment_does_not_invent_hairstyle_transition():
    enhance = CALL1_ENHANCE.read_text(encoding="utf-8")

    assert "real before-to-after arrangement change" in enhance
    assert '"a girl with disheveled long twintails" still has twintails' in enhance
    assert 'never paraphrase it as "her twintails came undone"' in enhance
    assert "ambiguous between disorder and a true arrangement change" in enhance
    assert "emit no event and preserve the fixed appearance" in enhance


def test_call2_authority_audit_never_leaves_hair_color_unspecified():
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "fixed hair-color exception is truly established" in source
    assert "explicit replacement color in required_additions" in source
    assert "otherwise keep the fixed color" in source


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
    assert "camera_replacement" not in source
    assert "Do not rewrite the scene, camera, composition, dialogue" in source


def test_call2_prioritizes_character_state_over_environment_detail():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")

    assert "Character rendering is the priority" in system
    assert "current pose, action, clothing or exposure state" in system
    assert "Establish the visible characters first" in thoughts
    assert "Environment is last priority" in system


def test_call2_uses_simple_background_without_inventing_scene_detail():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "use `simple background` as the sole environment description" in system
    assert "when no clear or important background exists" in thoughts
    assert "when no clear " in source
    assert "or important background exists" in source
    assert "Establish the world-building, time, and weather" not in system
    assert "Setup lighting with multiple tags" not in system


def test_call2_background_density_has_minimal_and_normal_toggle_branches():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "lb-xnai.background.minimal" in system
    assert "lb-xnai.background.minimal" in thoughts
    assert "Describe the environment at a useful visual density" in system
    assert "do not collapse a specific story-supported setting" in system
    assert "story-supported setting at a useful visual density" in thoughts
    assert 'toggles.get("minimal_background_description", True)' in source
    assert "environment at a useful visual density" in source
    assert "Keep the environment " in source
    assert "to the smallest story-supported cue" in source


def test_call2_keeps_scene_environment_out_of_character_positive():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "belong only in `scene`; never copy them" in system
    assert "Never repeat `scene` environment" in thoughts
    assert "Never repeat scene-wide environment" in source


def test_call2_supplement_does_not_repeat_existing_tags():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")

    assert "Do not restate character appearance" in system
    assert "leave it empty when the tags are sufficient" in system
    assert "supplement does not repeat details already expressed by tags" in thoughts


def test_call2_prompt_forbids_duplicate_character_items_in_solo_scenes():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "exact unique canonical roster" in system
    assert "Never repeat the same canonical name within one image" in system
    assert "physically exactly one `characters[]` item" in system
    assert "declared `characters[n]` count must equal the physical number" in system
    assert "exact unique canonical roster" in source
    assert "never repeat the same " in source
    assert "canonical name within one scene" in source
    assert "physically emit exactly one characters[] list item" in source
    assert "declared characters[n] count must equal the physical number" in source
