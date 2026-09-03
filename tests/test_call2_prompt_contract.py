import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALL1_ENHANCE = ROOT / "prompts" / "lighbd" / "enhance.txt"
CALL2_SYSTEM = ROOT / "prompts" / "lighbd" / "system.txt"
CALL2_THOUGHTS = ROOT / "prompts" / "lighbd" / "thoughts.txt"
CALL2_PRESET = ROOT / "prompts" / "lighbd" / "preset.txt"
PIPELINE_PY = ROOT / "modes" / "illustration_context_pipeline.py"
BUILTIN_PRESETS = ROOT / "prompts" / "bot_system_prompt" / "presets.json"


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


def test_call2_authority_audit_excludes_wardrobe_from_its_scope():
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "Wardrobe, outfit, accessories, coverage, and exposure are fully" in source
    assert "owned by CALL2 and are outside this audit" in source
    assert "Never add, remove, restore, or judge them" in source
    assert 'audit_reasons.append("default_outfit_differs")' not in source


def test_call2_known_outfit_state_bypasses_default_outfit_restore():
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "A known CALL2 outfit_state owns wardrobe as a complete set" in source
    assert "default outfit is restored only when CALL2 did not provide a usable state" in source
    assert "wardrobe_authority = [] if outfit_state_known else default_tags" in source


def test_call2_fixed_appearance_requires_explicit_narrative_change():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "authoritative identity of the already-selected visual profile" in system
    assert "Only a direct, explicit statement in the actual narrative" in system
    assert "active evidence-bearing history event" in thoughts
    assert "narrative or active " in source
    assert "hairstyle_history establishes a temporary physical change" in source
    assert "assigned scene selection controls the visual beat but has no appearance authority" in source
    assert "without turning appearance wording into a temporary replacement" in source
    assert "Audit fixed physical appearance and per-image visibility" in source
    assert "Wardrobe, outfit, accessories, coverage, and exposure" in source


def test_call1_dishevelment_does_not_invent_hairstyle_transition():
    enhance = CALL1_ENHANCE.read_text(encoding="utf-8")

    assert "real before-to-after arrangement change" in enhance
    assert '"a girl with disheveled long twintails" still has twintails' in enhance
    assert 'never paraphrase it as "her twintails came undone"' in enhance
    assert "ambiguous between disorder and a true arrangement change" in enhance
    assert "emit no event and preserve the fixed appearance" in enhance


def test_call2_authority_audit_never_leaves_hair_color_unspecified():
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "explicit replacement color whenever a fixed" in source
    assert "hair-color exception is established" in source
    assert "otherwise keep the fixed color" in source


def test_call2_builds_one_coherent_explicit_bundle_without_tag_dictionary():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")

    assert "Coherent Explicit Scene Bundle" in system
    assert "one minimal scene-specific bundle" in system
    assert "not from a fixed palette or quota" in system
    assert "Do not consult or simulate an external tag dictionary" in system
    assert "camera whose framing contains the body portions" in system
    assert "source#`/`target#` counterparts symmetrical" in system
    assert "silently assemble and cross-check one minimal coherent scene-specific bundle" in thoughts
    assert "never invent a new act, anatomy, intensity, garment state" in thoughts


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
    assert "current pose, action, visible clothing or exposure state" in system
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
    assert "at most two short complete sentences" in system
    assert "Leave it empty when the tags are sufficient" in system
    assert "supplement may use up to two short sentences" in thoughts


def test_call2_prompt_separates_named_roster_from_anonymous_fragment():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "exact unique canonical roster of named, identity-managed character entries" in system
    assert "Never repeat the same canonical name within one image" in system
    assert "must not receive an invented `characters[]` entry or a second complete-person count" in system
    assert "do not add `1boy` or any person-focus/solo tag" in system
    assert "exact unique canonical roster of named, identity-managed" in source
    assert "never repeat the same " in source
    assert "canonical name within one scene" in source
    assert "does not require a second complete-person count tag" in source
    assert "do not add `1boy` or any person-focus" in source
    assert "must never expand into a complete second person" in source


def test_call2_prompt_keeps_cropped_partner_out_of_focused_character_positive():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "does not require a second `1girl` or `1boy` count tag" in system
    assert "omit person-focus tags including `solo`, `solo focus`, `female focus`, and `male focus`" in system
    assert "never place them in the focused named character's `positive`" in system
    assert "Keep an anonymous, unnamed, or unregistered cropped partner's body parts" in system
    assert "never in the named girl's " in source
    assert '"positive. The declared characters[n] count' in source


def test_single_v5_preserves_v4_and_keeps_partner_as_connected_fragment():
    presets = json.loads(BUILTIN_PRESETS.read_text(encoding="utf-8"))

    assert "배포_1차 싱글 V4" in presets
    v5 = presets["배포_1차 싱글 V5"]
    assert "exactly one identifiable named character as the subject" in v5
    assert "does not mean full-body, fully exposed, unobstructed" in v5
    assert "Do not use keyword matching" in v5
    assert "does not add a second `1girl` or `1boy` count" in v5
    assert "Do not add `1boy` merely because the forearms are visible" in v5
    assert "omit every person-focus tag" in v5
    assert "including `solo`, `solo focus`, `female focus`, and `male focus`" in v5
    assert "Never put the partner's body parts or actions in the named subject's `positive`" in v5
    assert "do not expand the fragment into a whole man" in v5
    assert "may naturally occlude large portions of the named subject" in v5
    assert "back or side of the head may enter only when contact with that head" in v5
    assert "At most one face is visible" in v5
    assert "a zero-face contact crop is allowed" in v5
    assert "do not combine `legs together` with thighs framing" in v5
    assert "body-part whitelist" in v5
    assert "No weights or negative tags are invented" in v5
    assert "`1girl, 1boy, female focus`" not in v5
    assert "ALLOWED fragment tags" not in v5
    assert "MANDATORY negative field" not in v5


def test_call2_negative_does_not_block_intentional_partial_body_framing():
    negative = CALL2_PRESET.read_text(encoding="utf-8").split("[Negative]", 1)[1]

    tags = {tag.strip().casefold() for tag in negative.split(",")}
    assert "cropped" not in tags
    assert "head out of frame" not in tags


def test_interaction_contract_does_not_invent_secondary_limb_contact():
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")
    presets = json.loads(BUILTIN_PRESETS.read_text(encoding="utf-8"))
    v5 = presets["배포_1차 싱글 V5"]

    assert "contact by one body region does not authorize a second embrace" in thoughts
    assert "contact point or action for an unmentioned limb" in source
    assert "contact by one body region does not authorize a second embrace" in v5


def test_anima_fragment_uses_one_broad_anchor_and_natural_language_geometry():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")
    presets = json.loads(BUILTIN_PRESETS.read_text(encoding="utf-8"))
    v5 = presets["배포_1차 싱글 V5"]

    assert "anchor an anonymous fragment exactly once in `scene` with one short, familiar body-region" in system
    assert "prefer `cropped male lower body`" in system
    assert "do not atomize the same connected fragment" in thoughts
    assert "do not atomize one connected fragment into a comma chain" in source
    assert "never change a third-person camera to POV" in source
    assert "`cropped male upper torso` is too broad" in system
    assert "use a contact-point `close-up`" in system
    assert "semantically inspect every phrase in each named character positive" in source
    assert "express the partner fragment exactly once with one familiar region/composition phrase" in v5
    assert "over atomizing one connected fragment into a comma chain" in v5
    assert "use a contact-point close-up instead of portrait, cowboy-shot, or full-body framing" in v5
    assert "semantically inspect every phrase in each named character positive" in v5


def test_call2_visibility_contract_does_not_force_hidden_character_details():
    system = CALL2_SYSTEM.read_text(encoding="utf-8")
    thoughts = CALL2_THOUGHTS.read_text(encoding="utf-8")
    source = PIPELINE_PY.read_text(encoding="utf-8")

    assert "fully outside the frame or fully hidden by natural occlusion may be omitted" in system
    assert "Do not widen the camera, move an interacting body aside" in system
    assert "Keep every non-conflicting garment and accessory" in system
    assert "in `outfit_state`" in system
    assert "Put a garment or accessory in `positive` only when it is visible" in system
    assert "a wholly cropped or naturally hidden feature may be absent" in thoughts
    assert "put only visible or coverage-defining garments in " in source
    assert '"positive. Never advance state' in source
    assert "visibility_omissions" in source
