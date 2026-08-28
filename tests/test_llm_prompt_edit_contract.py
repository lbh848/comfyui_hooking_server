from modes import llm_prompt_edit


def test_v3_builtin_requires_natural_language_only_for_supplement():
    template = llm_prompt_edit._load_user_v3_builtin()

    assert (
        "Write scene_setup and scene_char as English Danbooru-style tags"
        in template
    )
    assert (
        "Write scene_supplement in concise, minimal English natural language"
        in template
    )
    assert "Do NOT write it as a comma-separated tag list" in template
    assert "Do not include character names in scene_supplement" in template
    assert "applied only to the ANIMA blocks" in template


def test_v3_fallback_matches_natural_language_supplement_contract():
    template = llm_prompt_edit.DEFAULT_USER_V3_TEMPLATE

    assert (
        "Write scene_supplement in concise, minimal English natural language"
        in template
    )
    assert "Do NOT write it as a comma-separated tag list" in template
    assert "preferably one short sentence" in template


def _assert_plan_describes_executed_scene(template):
    assert "Finalize the scene_* fields first" in template
    assert "faithful summary of the edits that are actually present" in template
    assert "Never claim in plan" in template
    assert "update every affected occurrence" in template

    output_contract = template.split("## Output", 1)[1]
    assert output_contract.index('"scene_char"') < output_contract.index('"plan"')


def test_v3_plan_is_a_summary_of_the_executed_scene_edit():
    _assert_plan_describes_executed_scene(llm_prompt_edit._load_user_v3_builtin())
    _assert_plan_describes_executed_scene(llm_prompt_edit.DEFAULT_USER_V3_TEMPLATE)


def test_identity_contract_does_not_preserve_a_causally_affected_pose():
    bot = {
        "characters": [
            {
                "name": "Hoshino",
                "gender_tag": "1girl",
                "face_tags": "short hair, black hair",
                "eye_tags": "yellow eyes",
            }
        ]
    }

    contract = llm_prompt_edit.character_selection_contract(
        bot,
        ["Hoshino"],
        ["Hoshino"],
    )

    assert "remain unrelated after the causal audit" in contract
    assert "identified as contributing to the unwanted result is affected" in contract
    assert "even when the user's wording does not name it directly" in contract


def _assert_causal_system_contract(prompt):
    assert "causal prompt debugger" in prompt
    assert "desired visual outcome" in prompt
    assert "root cause" in prompt or "root conflicts" in prompt
    assert "physically coherent" in prompt
    assert "symptom-countering tags" in prompt


def _assert_causal_user_contract(template):
    normalized = template.casefold()
    assert "desired visual outcome" in normalized
    assert "causally audit" in normalized
    assert "fix the cause before" in normalized
    assert "physically feasible" in normalized
    assert "when one subject is requested" in normalized
    assert "do not merely" in normalized or "do not solve a failure merely" in normalized
    assert "root-cause diagnosis" in template or "근본 원인 진단" in template


def test_builtin_system_prompts_require_causal_conflict_resolution():
    _assert_causal_system_contract(llm_prompt_edit._load_llm_edit_builtin())
    _assert_causal_system_contract(llm_prompt_edit._load_system_chansub_builtin())


def test_fallback_system_prompts_require_causal_conflict_resolution():
    _assert_causal_system_contract(llm_prompt_edit.DEFAULT_SYSTEM_PROMPT)
    _assert_causal_system_contract(llm_prompt_edit.DEFAULT_SYSTEM_CHANSUB_PROMPT)


def test_builtin_user_templates_fix_root_conflicts_before_tag_reinforcement():
    _assert_causal_user_contract(llm_prompt_edit._load_user_v3_builtin())
    _assert_causal_user_contract(llm_prompt_edit._load_user_v1_builtin())
    _assert_causal_user_contract(llm_prompt_edit._load_user_chansub_builtin())


def test_fallback_user_templates_fix_root_conflicts_before_tag_reinforcement():
    _assert_causal_user_contract(llm_prompt_edit.DEFAULT_USER_V3_TEMPLATE)
    _assert_causal_user_contract(llm_prompt_edit.DEFAULT_USER_V1_TEMPLATE)
    _assert_causal_user_contract(llm_prompt_edit.DEFAULT_USER_CHANSUB_TEMPLATE)
