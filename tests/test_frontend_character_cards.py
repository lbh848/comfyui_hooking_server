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
    assert "lb-xnai.lb.extra 설정" in FRONTEND
    assert "lb-xnai.lb.extra 복장 설정" not in FRONTEND
    assert 'id="visual-card-appearance"' in FRONTEND
    assert "target.appearance = JSON.parse" in FRONTEND
    assert "openVisualOutfitEditor" in FRONTEND
    assert 'id="visual-outfit-label"' in FRONTEND
    assert 'id="visual-outfit-aliases"' not in FRONTEND
    assert "복장 이름" not in FRONTEND


def test_removed_profile_modal_and_raw_json_editor_do_not_return():
    assert "openVisualProfileEditor" not in FRONTEND
    assert "고급 렌더 오버라이드" not in FRONTEND
    assert "/api/bot_mode/visual_profiles" not in FRONTEND
    assert "visual-profile-overlay" not in FRONTEND


def test_outfit_editor_does_not_close_on_backdrop_click():
    modal_start = FRONTEND.index("function _visualOutfitModal()")
    modal_end = FRONTEND.index("function _visualOutfitCurrent()", modal_start)
    modal_source = FRONTEND[modal_start:modal_end]
    assert "root.onclick" not in modal_source
    assert "closeVisualOutfitEditor()" not in modal_source
