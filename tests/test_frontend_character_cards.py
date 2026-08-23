from pathlib import Path


FRONTEND = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
    encoding="utf-8"
)


def test_character_cards_are_switched_inline_and_limited_to_ten():
    assert "const VISUAL_CARD_LIMIT = 10" in FRONTEND
    assert "switchVisualCard" in FRONTEND
    assert "addVisualCard" in FRONTEND
    assert "[${index + 1}]" in FRONTEND


def test_card_metadata_and_outfit_entry_are_on_the_character_card():
    assert "자연어 선택 기준" in FRONTEND
    assert "작중 별칭" in FRONTEND
    card_metadata = FRONTEND[FRONTEND.index('<div class="card-section" style="display:grid;grid-template-columns:minmax(180px,.65fr) minmax(320px,1.35fr)'):]
    assert card_metadata.index("작중 별칭") < card_metadata.index("자연어 선택 기준")
    assert "lb-xnai.lb.extra 설정" in FRONTEND
    assert "lb-xnai.lb.extra 복장 설정" not in FRONTEND
    assert "openVisualCardLbExtraEditor" in FRONTEND
    assert "_openFocusEditModal(0)" in FRONTEND
    assert 'id="fe-card-controls"' in FRONTEND
    assert "saveVisualCardFocusEdit" in FRONTEND
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
