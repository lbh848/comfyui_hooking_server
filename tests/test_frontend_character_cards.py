from pathlib import Path


FRONTEND = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
    encoding="utf-8"
)


def test_character_cards_are_switched_inline_and_limited_to_ten():
    assert "const VISUAL_CARD_LIMIT = 10" in FRONTEND
    assert "switchVisualCard" in FRONTEND
    assert "addVisualCard" in FRONTEND
    assert "[${index + 1}]" in FRONTEND


def test_card_metadata_and_flat_lb_extra_editor_are_on_the_character_card():
    assert "자연어 선택 기준" in FRONTEND
    assert "작중 별칭" in FRONTEND
    card_metadata = FRONTEND[FRONTEND.index('<div class="card-section" style="display:grid;grid-template-columns:minmax(180px,.65fr) minmax(320px,1.35fr)'):]
    assert card_metadata.index("작중 별칭") < card_metadata.index("자연어 선택 기준")
    assert "lb-xnai.lb.extra 설정" in FRONTEND
    assert "lb-xnai.lb.extra 복장 설정" not in FRONTEND
    assert "openVisualCardLbExtraEditor" in FRONTEND
    assert "_openFocusEditModal(0)" in FRONTEND
    assert 'id="fe-card-controls"' not in FRONTEND
    assert "saveVisualCardFocusEdit" in FRONTEND
    assert "session.profile.default_outfit" in FRONTEND
    assert "session.profile.outfits" not in FRONTEND
    assert "default_outfit_id" not in FRONTEND
    assert "addVisualCardFocusOutfit" not in FRONTEND
    assert "deleteVisualCardFocusOutfit" not in FRONTEND
    assert "visual_card_id: _visualCardFocusSession?.profileId" in FRONTEND
    assert "openVisualOutfitEditor" not in FRONTEND
    assert "visual-outfit-overlay" not in FRONTEND


def test_removed_profile_modal_and_raw_json_editor_do_not_return():
    assert "openVisualProfileEditor" not in FRONTEND
    assert "고급 렌더 오버라이드" not in FRONTEND
    assert "/api/bot_mode/visual_profiles" not in FRONTEND
    assert "visual-profile-overlay" not in FRONTEND


def test_shared_focus_editor_does_not_close_on_backdrop_click():
    modal_start = FRONTEND.index("function _openFocusEditModal(ci)")
    modal_end = FRONTEND.index("function _closeFocusEditModal()", modal_start)
    modal_source = FRONTEND[modal_start:modal_end]
    assert "overlay.onclick" not in modal_source
    assert "overlay.addEventListener('click'" not in modal_source


def test_active_character_card_face_preview_and_prompt_edit_keep_profile_id():
    assert "&visual_card_id=${encodeURIComponent(activeVisualCardId)}" in FRONTEND
    assert "const utilImg = data.images.find(i => i.filename === '_face_image.webp');" in FRONTEND
    assert "(_visualCardSlots[charName] || 0) === 0" not in FRONTEND
    assert 'data-visual-card-id="${escAttr(activeVisualCardId)}"' in FRONTEND

    modal_start = FRONTEND.index("function lv1OpenEditModal(btn)")
    modal_end = FRONTEND.index("function lv1TogglePrompt(textEl)", modal_start)
    modal_source = FRONTEND[modal_start:modal_end]
    assert "const visualCardId = box.dataset.visualCardId || '';" in modal_source
    assert "visual_card_id: visualCardId" in modal_source


def test_lb_extra_batch_refine_targets_every_profile_and_only_syncs_card_one_portable_data():
    assert "프로필 카드 일괄 정제" in FRONTEND
    assert "이식용 데이터 일괄 정제" not in FRONTEND

    target_start = FRONTEND.index("function _lbExtraProfileBatchTargets()")
    target_end = FRONTEND.index("async function _lbExtraBatchRefine()", target_start)
    target_source = FRONTEND[target_start:target_end]
    assert "profiles.forEach((profile, profileIndex)" in target_source
    assert "profile.default_outfit" in target_source
    assert "defaultOutfitId" not in target_source
    assert "rep: repImages[0] || ''" in target_source
    assert "cardData.default_visual_profile_id || profiles[0]?.id" in target_source
    assert "isPortable: String(profile.id || '') === portableProfileId" in target_source

    run_start = FRONTEND.index("async function _lbExtraBatchRefineRun(targets)")
    run_end = FRONTEND.index("let _lbExtraBatchAbort = null", run_start)
    run_source = FRONTEND[run_start:run_end]
    assert "profile.id === target.visualCardId" in run_source
    assert "profile.outfits" not in run_source
    assert "await _loadVisualCardRefineOriginal(" in run_source
    assert "const appearanceTags = (original.appearance || [])" in run_source
    assert "const outfitTags = (original.outfit || [])" in run_source
    assert "const etcTags = (original.uncategorized || [])" in run_source
    assert "etc: etcTags" in run_source
    assert "etc: []" not in run_source
    assert "visual_card_id: target.visualCardId" in run_source
    assert "await _saveVisualCardState(target.character, {quiet:true})" in run_source
    assert "profile.default_outfit = JSON.parse" in run_source
    assert "if (target.isPortable && _lbExtraEdited[ci])" in run_source
    assert "_lbExtraEdited[ci].appearance" in run_source
    assert "_lbExtraEdited[ci].outfit" in run_source


def test_lb_extra_refine_is_registered_in_queue_ui_and_routing_ui():
    assert "bot_lb_extra_refine: 'lb.extra 프로필 정제'" in FRONTEND
    assert "{ key: 'refine_lb_extra'" in FRONTEND


def test_focused_and_batch_refine_share_the_same_representative_prompt_loader():
    assert "async function _loadVisualCardRefineOriginal(" in FRONTEND
    assert FRONTEND.count("await _loadVisualCardRefineOriginal(") == 2
    assert "_loadVisualCardFocusOriginal" not in FRONTEND


def test_manual_restore_draw_selects_a_profile_for_each_character():
    assert 'id="restore-profile-select-1"' in FRONTEND
    assert 'id="restore-profile-select-2"' in FRONTEND
    assert "function _fillRestoreProfileSelect(slot)" in FRONTEND
    assert "character.default_visual_profile_id" in FRONTEND
    assert "visual_profile_ids: visualProfileIds" in FRONTEND
    assert 'id="restore-profile-thumb-1"' in FRONTEND
    assert 'id="restore-profile-thumb-2"' in FRONTEND
    assert "function _updateRestoreProfileThumbnail(slot)" in FRONTEND
    assert "encodeURIComponent(filename)" in FRONTEND
    assert "const filename = String(profile?.rep_image || '').trim();" in FRONTEND
    assert "openRestoreProfileThumbnail(1)" in FRONTEND


def test_easy_edit_selects_and_submits_a_visual_profile_for_each_slot():
    assert "function fillLlmEditProfileSelect(index, preferredProfileId = '')" in FRONTEND
    assert "llm-edit-profile-select-${index}" in FRONTEND
    assert "capability.visual_profile_ids" in FRONTEND
    assert "state.visualProfileIds[character.name] = profile.id" in FRONTEND
    assert "requestBody.visual_profile_ids = selectedVisualProfileIds" in FRONTEND
    assert "requestBody.previous_identity = currentModalIdentityEdit" in FRONTEND
    assert "currentModalIdentityEdit?.visual_profile_ids" in FRONTEND
    assert "profile?.rep_image" in FRONTEND
