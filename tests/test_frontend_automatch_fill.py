from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def test_expression_profile_navigation_is_removed():
    source = _frontend_source()

    assert 'id="at-sidebar-expr-profile"' not in source
    assert "atSwitchView('expr-profile')" not in source


def test_matching_result_contains_locked_worker_and_top_n_fill_controls():
    source = _frontend_source()

    assert 'class="at-result-panel"' in source
    assert 'id="at-fill-expression"' in source
    assert "🔒 잠금" in source
    assert 'id="at-fill-top-n" value="3" min="1" max="12"' in source
    assert 'onclick="atFillEmptySlots()"' in source


def test_automatch_generation_uses_separate_storage_and_source_badges():
    source = _frontend_source()

    assert "storage_group: 'automatch_defaults'" in source
    assert "? '선택 복장'" in source
    assert "source === 'existing_asset' ? '기존 에셋'" in source
    assert "source === 'generated_default' ? '기본 이미지'" in source
    assert "/automatch_compare?outfit=" in source
    assert "&include_existing=${includeExisting}" in source


def test_fill_worker_shows_model_specific_controls_and_existing_asset_option():
    source = _frontend_source()

    assert 'id="at-fill-mode-badge">ILXL<' in source
    assert 'data-at-availability="anima"' in source
    assert 'data-at-availability="sdxl-only"' in source
    assert "ANIMA 전용" in source
    assert "ILXL" in source
    assert 'data-at-availability="ipadapter"' in source
    assert 'id="at-fill-use-existing" checked' in source
    assert "function atApplyFillWorkflowAvailability()" in source


def test_hires_details_group_is_disabled_until_a_hires_mode_is_enabled():
    source = _frontend_source()

    assert 'class="at-model-option-group at-model-control" id="at-fill-hrf-options"' in source
    assert "(capabilities.ilxl && !!atFillSlot.hrf_sdxl)" in source
    assert "(capabilities.anima && !!atFillSlot.hrf_anima)" in source
    assert "container.classList.toggle('is-unavailable', !enabled);" in source
    assert "container.setAttribute('aria-disabled', enabled ? 'false' : 'true');" in source
    assert "control.disabled = !enabled" in source


def test_hires_detailers_and_face_crop_are_separate_worker_groups():
    source = _frontend_source()
    generation_controls = source[
        source.index('<span>생성 옵션</span>') : source.index('<span>프롬프트 미리보기</span>')
    ]
    hires_controls = generation_controls[
        generation_controls.index('<span>HIRES FIX</span>') : generation_controls.index('id="at-fill-hrf-options"')
    ]

    assert '<span>디테일러</span>' in generation_controls
    assert '<span>FACE CROP</span>' in generation_controls
    assert 'id="at-fill-face-crop-top"' in generation_controls
    assert 'id="at-fill-face-crop-bottom"' in generation_controls
    assert 'id="at-fill-fd"' not in hires_controls
    assert 'id="at-fill-anima-fd"' not in hires_controls


def test_fill_worker_previews_prompt_for_first_empty_top_match():
    source = _frontend_source()

    assert 'id="at-fill-preview-target"' in source
    assert 'id="at-fill-positive-preview"' in source
    assert 'id="at-fill-negative-preview"' in source
    assert "function atToggleFillPromptPreview(labelElement)" in source
    assert "const previewSlot = { ...atFillSlot, expression: targets[0] };" in source
    assert "positive.innerHTML = previewBatchSlotPrompt(previewSlot);" in source
    assert "negative.textContent = previewBatchSlotNegativePrompt(previewSlot);" in source
    assert "atUpdateFillPromptPreview();" in source[source.index("function atUpdateFillAvailability()") :]


def test_matching_requests_top_twelve_and_pages_each_match_type_by_four():
    source = _frontend_source()
    render_matches = source[source.index("function atRenderMatches()") : source.index("function atSelectSingleMatch(")]

    assert "const AT_MATCH_RESULT_LIMIT = 12;" in source
    assert "const AT_MATCH_PAGE_SIZE = 4;" in source
    assert "top_n: AT_MATCH_RESULT_LIMIT" in source
    assert "atCreateMatchPager('embedding'" in render_matches
    assert "atCreateMatchPager('tag'" in render_matches
    assert "embGroup.matches.slice(embPager.start, embPager.end)" in render_matches
    assert "tagGroup.matches.slice(tagPager.start, tagPager.end)" in render_matches


def test_matching_cards_wrap_without_per_result_scrollbars():
    source = _frontend_source()
    render_matches = source[source.index("function atRenderMatches()") : source.index("function atSelectSingleMatch(")]

    assert "#at-view-result { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }" in source
    assert "flex: 0 0 auto; display: grid" in source
    assert "grid-template-columns: repeat(auto-fill, minmax(160px, 1fr))" in source
    assert "wrapper.className = 'at-result-item'" in render_matches
    assert "srcCard.className = 'at-card at-card-source at-result-source'" in render_matches
    assert "embRow.className = 'at-match-grid'" in render_matches
    assert "tagRow.className = 'at-match-grid'" in render_matches
    assert "overflow-x:auto" not in render_matches
    assert "flex-shrink:0" not in render_matches
