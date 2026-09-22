import pytest

from modes import llm_prompt_edit


@pytest.mark.parametrize(
    "template",
    [
        pytest.param(llm_prompt_edit._load_user_v3_builtin(), id="builtin"),
        pytest.param(llm_prompt_edit.DEFAULT_USER_V3_TEMPLATE, id="fallback"),
    ],
)
def test_v3_template_keeps_the_machine_consumed_output_contract(template: str) -> None:
    assert "scene_setup and scene_char as English Danbooru-style tags" in template
    assert "scene_supplement in concise, minimal English natural language" in template
    assert "Do NOT write it as a comma-separated tag list" in template

    output_contract = template.split("## Output", 1)[1]
    assert output_contract.index('"plan"') < output_contract.index('"scene_char"')


def test_character_selection_contract_preserves_causal_identity_scope() -> None:
    contract = llm_prompt_edit.character_selection_contract(
        {
            "characters": [
                {
                    "name": "Hoshino",
                    "gender_tag": "1girl",
                    "face_tags": "short hair, black hair",
                    "eye_tags": "yellow eyes",
                }
            ]
        },
        ["Hoshino"],
        ["Hoshino"],
    )

    assert "remain unrelated after the causal audit" in contract
    assert "identified as contributing to the unwanted result is affected" in contract


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(llm_prompt_edit._load_llm_edit_builtin(), id="v3-system"),
        pytest.param(llm_prompt_edit._load_system_chansub_builtin(), id="chansub-system"),
        pytest.param(llm_prompt_edit.DEFAULT_SYSTEM_PROMPT, id="v3-fallback"),
        pytest.param(llm_prompt_edit.DEFAULT_SYSTEM_CHANSUB_PROMPT, id="chansub-fallback"),
    ],
)
def test_system_prompts_keep_one_causal_editing_objective(prompt: str) -> None:
    assert "causal prompt debugger" in prompt
    assert "desired visual outcome" in prompt
    assert "root cause" in prompt or "root conflicts" in prompt
    assert "physically coherent" in prompt
