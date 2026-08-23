from pathlib import Path


FRONTEND = Path("frontend/index.html").read_text(encoding="utf-8")


def test_illustration_prompt_settings_header_uses_two_rows() -> None:
    block_start = FRONTEND.index('id="illust-prompt-settings-block"')
    block_end = FRONTEND.index('class="illust-workflow-option"', block_start)
    header = FRONTEND[block_start:block_end]

    title_row = header.index('class="illust-prompt-settings-title-row"')
    profile_row = header.index('class="illust-prompt-settings-profile-row"')

    assert title_row < header.index("삽화 프롬프트 설정") < profile_row
    assert title_row < header.index("태그 편집") < profile_row
    assert profile_row < header.index('id="illust-profile-tab-solo"')
    assert profile_row < header.index('id="illust-profile-copy-btn"')


def test_illustration_prompt_settings_header_prevents_control_shrinking() -> None:
    header_selector = "#illust-prompt-settings-block .illust-prompt-settings-header"
    header_start = FRONTEND.index(header_selector)
    header_rule = FRONTEND[header_start : FRONTEND.index("}", header_start)]
    assert "flex-direction: column;" in header_rule

    controls_selector = (
        "#illust-prompt-settings-block .illust-prompt-settings-profile-tabs"
    )
    controls_start = FRONTEND.index(controls_selector)
    controls_rule = FRONTEND[controls_start : FRONTEND.index("}", controls_start)]
    assert "flex: 0 0 auto;" in controls_rule
