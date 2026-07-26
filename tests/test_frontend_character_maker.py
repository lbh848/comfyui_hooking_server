from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _html():
    return FRONTEND.read_text(encoding="utf-8")


def test_character_maker_tab_is_immediately_after_automatch():
    html = _html()
    automatch = html.index('id="tab-btn-smart-asset"')
    maker = html.index('id="tab-btn-character-maker"')
    divider = html.index('<span class="tab-divider"></span>', maker)

    assert automatch < maker < divider
    assert "switchTab('character-maker')" in html
    assert 'id="tab-character-maker-content"' in html


def test_character_maker_has_at_a_glance_three_rail_workflow():
    html = _html()

    assert 'class="cm-workspace"' in html
    assert 'class="cm-context-rail"' in html
    assert 'class="cm-stage"' in html
    assert 'class="cm-editor-wall"' in html
    assert 'id="cm-world-context"' in html
    assert 'id="cm-reference-drop"' in html
    assert 'id="cm-chat-log"' in html
    assert 'id="cm-revisions"' in html
    assert 'id="cm-settings-wall" class="cm-settings-wall collapsed"' in html


def test_only_four_fields_have_free_chip_and_text_editors():
    html = _html()

    for field in ("appearance", "outfit", "expression", "composition"):
        assert f'class="cm-field-card" data-field="{field}"' in html
        assert f"cmTextFieldInput('{field}', this.value)" in html
        assert f"cmToggleLock('{field}')" in html

    assert 'id="cm-mode-chip"' in html
    assert 'id="cm-mode-text"' in html
    assert "raw_appearance_tags" in html
    assert "raw_outfit_tags" in html
    assert "raw_expression_tags" in html
    assert "raw_composition_tags" in html


def test_browser_only_persists_server_session_identifier():
    html = _html()

    assert "const CM_SESSION_STORAGE_KEY = 'characterMakerSessionId'" in html
    assert "sessionStorage.setItem(CM_SESSION_STORAGE_KEY, cmSession.id)" in html
    assert "sessionStorage.getItem(CM_SESSION_STORAGE_KEY)" in html
    assert "cmSession.boot_id !== cmCapabilities.boot_id" in html
    assert "localStorage.setItem('characterMaker" not in html


def test_character_maker_apis_and_optional_confirmation_modes_are_wired():
    html = _html()

    for fragment in (
        "/api/character_maker/capabilities",
        "/api/character_maker/session",
        "/revise",
        "/generate",
        "/reference",
        "/confirm",
    ):
        assert fragment in html

    assert '<option value="none">등록하지 않음</option>' in html
    assert '<option value="existing">기존 표정 프리셋 사용</option>' in html
    assert '<option value="new">새 표정 프리셋 등록</option>' in html
    assert "appearance_name" in html
    assert "outfit_name" in html


def test_rag_settings_test_and_external_llm_routes_are_visible():
    html = _html()

    assert "https://github.com/joykst96/danbooru-tag-rag" in html
    assert 'id="setting-character-maker-rag-enabled"' in html
    assert 'id="setting-character-maker-rag-test-btn"' in html
    assert "/api/character_maker/rag/test" in html
    assert "character_maker_draft" in html
    assert "character_maker_feedback" in html
    assert "캐릭터 메이커 이미지 피드백" in html


def test_prompt_builder_accepts_raw_character_maker_fields_and_workflow_type():
    html = _html()

    assert "Array.isArray(slot.raw_appearance_tags)" in html
    assert "Array.isArray(slot.raw_outfit_tags)" in html
    assert "Array.isArray(slot.raw_expression_tags)" in html
    assert "Array.isArray(slot.raw_composition_tags)" in html
    assert "getAssetWorkflowCapabilities(slot.asset_workflow_type)" in html
