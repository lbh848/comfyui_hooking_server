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


def test_editor_wall_places_settings_before_fields_with_one_scroll_container():
    html = _html()

    editor_css = html[html.index(".cm-editor-wall {"):html.index(".cm-card {")]
    field_css = html[html.index(".cm-field-list {"):html.index(".cm-field-card {")]
    settings_css = html[html.index(".cm-settings-wall {"):html.index(".cm-settings-wall.collapsed")]
    settings_body_css = html[html.index(".cm-settings-body {"):html.index(".cm-setting-grid {")]

    assert "overflow-y: auto;" in editor_css
    assert "order: 2;" in field_css
    assert "overflow: visible;" in field_css
    assert "order: 1;" in settings_css
    assert "overflow-y: auto;" not in settings_body_css
    assert "overflow: visible;" in settings_body_css


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


def test_each_field_has_preset_loader_wired_to_load_function():
    html = _html()

    # 에셋 생성과 동일한 커스텀 셀렉트(검색 내장) 디자인을 사용한다.
    # 빈 래퍼 div를 두고 JS가 buildAssetCustomSelect 로 채운다.
    for field in ("appearance", "outfit", "expression", "composition"):
        assert (
            f'class="cm-field-preset-wrap" data-preset-select"\n'
            in html
            or f'class="cm-field-preset-wrap" data-preset-select' in html
        ), f"필드 프리셋 래퍼(cm-field-preset-wrap)이 없습니다"
        assert f'data-field="{field}"' in html, f"필드 {field} 래퍼에 data-field가 없습니다"

    # 필드 → assetTags 프리셋 사전 매핑과 렌더/불러오기 함수가 정의되어 있어야 한다.
    assert "const CM_FIELD_PRESET_KEYS" in html
    assert "appearance: 'appearances'" in html
    assert "outfit: 'outfits'" in html
    assert "expression: 'expressions'" in html
    assert "composition: 'composition_presets'" in html
    assert "function cmRenderFieldPresetSelects()" in html
    assert "function cmLoadFieldPreset(field)" in html
    # cmPopulatePresetOptions() 안에서 필드 프리셋 셀렉트도 채운다.
    assert "cmRenderFieldPresetSelects()" in html

    # 에셋 쪽 빌더/바인더/값 접근 함수를 재사용해 검색 기능 내장 셀렉트를 만든다.
    assert "const CM_FIELD_PRESET_SELECT_IDS" in html
    assert "const CM_FIELD_PRESET_LOAD_FNS" in html
    assert "cm-preset-appearance" in html
    assert "cm-preset-outfit" in html
    assert "cm-preset-expression" in html
    assert "cm-preset-composition" in html
    assert "buildAssetCustomSelect(" in html
    assert "bindAssetCustomSelects()" in html
    # 래퍼: bindAssetCustomSelects 가 window[fn]() 무인자 호출을 하므로 필드별 래퍼 필요.
    for fn in (
        "cmLoadAppearancePreset",
        "cmLoadOutfitPreset",
        "cmLoadExpressionPreset",
        "cmLoadCompositionPreset",
    ):
        assert f"function {fn}(" in html, f"래퍼 함수 {fn} 정의가 없습니다"
    # 값 읽기/되돌리기도 커스텀 셀렉트 API를 경유한다.
    assert "getAssetSelectValue(selectId)" in html
    assert "setAssetSelectValue(selectId, '')" in html


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
    assert 'id="setting-character-maker-rag-runtime-start"' in html
    assert 'id="setting-character-maker-rag-runtime-stop"' in html
    assert "/api/character_maker/rag/test" in html
    assert "/api/character_maker/rag/runtime/start" in html
    assert "/api/character_maker/rag/runtime/stop" in html
    assert "cmStartRagRuntime" in html
    assert "cmStopRagRuntime" in html
    assert "character_maker_draft" in html
    assert "character_maker_feedback" in html
    assert "캐릭터 메이커 이미지 피드백" in html


def test_rag_settings_include_integrated_dataset_converter_and_tidy_cards():
    html = _html()

    assert 'class="cm-rag-settings-page"' in html
    assert 'class="cm-rag-settings-grid"' in html
    assert 'id="setting-character-maker-rag-data-drop"' in html
    assert 'id="setting-character-maker-rag-data-input"' in html
    assert 'id="setting-character-maker-rag-data-search-btn"' in html
    assert 'id="setting-character-maker-rag-repo-path"' in html
    assert 'id="setting-character-maker-rag-data-install-btn"' in html
    assert "auto_complete 자료 검색" in html
    assert "직접 파일 찾기" in html
    assert "변환 및 설치" in html
    assert "/api/character_maker/rag/dataset" in html
    assert "/api/character_maker/rag/install" in html
    assert 'id="setting-character-maker-rag-data-progress"' in html
    assert 'id="setting-character-maker-rag-data-progress-fill"' in html
    assert "handleCmRagInstallProgress" in html
    assert "character_maker_rag_install_progress" in html
    assert "cmFindRagDataset" in html
    assert "form.append('dataset', cmRagDatasetFile" in html
    assert "form.append('source', 'auto_complete')" in html
    assert "form.append('repository', repository)" in html
    assert "anchor.download = 'danbooru-tags.csv'" not in html
    assert "'character-maker-layout'" in html


def test_character_maker_uses_lora_instead_of_generation_ipadapter():
    html = _html()

    assert 'id="cm-setting-lora-enabled"' in html
    assert 'id="cm-setting-lora-panel"' in html
    assert 'id="cm-setting-lora-list"' in html
    assert "function cmOpenLoraPicker" in html
    assert "lora_list: (settings.lora_list || [])" in html
    assert "lora_enabled: !!settings.lora_enabled" in html
    assert 'id="cm-setting-use-refs"' not in html
    assert "use_references_for_generation" not in html


def test_prompt_builder_accepts_raw_character_maker_fields_and_workflow_type():
    html = _html()

    assert "Array.isArray(slot.raw_appearance_tags)" in html
    assert "Array.isArray(slot.raw_outfit_tags)" in html
    assert "Array.isArray(slot.raw_expression_tags)" in html
    assert "Array.isArray(slot.raw_composition_tags)" in html
    assert "getAssetWorkflowCapabilities(slot.asset_workflow_type)" in html
