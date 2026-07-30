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
