from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _html():
    return FRONTEND.read_text(encoding="utf-8")


def test_character_maker_tab_is_immediately_after_automatch():
    html = _html()
    automatch = html.index('id="tab-btn-smart-asset"')
    maker = html.index('id="tab-btn-character-maker"')
    automatch_end = html.index("</button>", automatch) + len("</button>")
    maker_start = html.rfind("<button", automatch_end, maker)
    asset_group_end = html.index("</div>", maker)
    manage_group = html.index('class="tab-group tab-group--manage"', maker)

    assert "<button" not in html[automatch_end:maker_start]
    assert automatch < maker < asset_group_end < manage_group
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


def test_user_image_can_open_instance_lora_modal_as_an_uploaded_file():
    html = _html()
    user_pane_start = html.index('id="cm-pane-user"')
    user_pane_end = html.index('id="cm-busy-overlay"', user_pane_start)
    user_pane = html[user_pane_start:user_pane_end]

    assert 'id="cm-send-instance-lora-btn"' in user_pane
    assert "cmSendUserImageToInstanceLora()" in user_pane
    assert "인스턴스 로라로 보내기" in user_pane
    assert ".cm-pane.has-image .cm-instance-lora-btn { display: inline-flex; }" in html

    transfer_start = html.index("async function cmSendUserImageToInstanceLora()")
    transfer_end = html.index("function cmRenderLlmTags()", transfer_start)
    transfer = html[transfer_start:transfer_end]
    assert "cmSession?.active_revision_id" in transfer
    assert "await response.blob()" in transfer
    assert "new File(" in transfer
    assert "await showInstanceLoraCreateModal(uploadFile)" in transfer
    assert "console.error('[CHARACTER_MAKER] 인스턴스 로라 전달 실패:" in transfer

    modal_start = html.index("async function showInstanceLoraCreateModal(initialUploadFile = null)")
    modal_end = html.index("function closeInstanceLoraCreateModal()", modal_start)
    modal = html[modal_start:modal_end]
    assert "const settingsLoaded = await ensureInstanceLoraSettingsLoaded()" in modal
    assert "if (!settingsLoaded)" in modal
    assert modal.index("await ensureInstanceLoraSettingsLoaded()") < modal.index(
        "instance-lora-create-modal"
    )
    assert "switchInstanceCreateBrowser(initialUploadFile ? 'upload' : 'asset')" in modal
    assert "type: 'upload'" in modal
    assert "file: initialUploadFile" in modal
    assert "previewUrl: URL.createObjectURL(initialUploadFile)" in modal
    assert "return true;" in modal

    assert "let _instanceLoraSettingsLoaded = false;" in html
    assert "let _instanceLoraSettingsLoadPromise = null;" in html
    assert "async function ensureInstanceLoraSettingsLoaded(forceReload = false)" in html
    assert "await ensureInstanceLoraSettingsLoaded(true)" in html


def test_character_maker_marks_chat_branch_scope_and_accept_checkpoint():
    html = _html()

    assert 'id="cm-chat-scope"' in html
    assert "cm-message-context-badge" in html
    assert "사용자 기준" in html
    assert "LLM 기준" in html
    assert "accept됨" in html
    assert "폐기됨" in html
    assert "active_chat_branch_id" in html
    assert "user_chat_checkpoint_id" in html
    assert "현재 대화 분기" in html
    assert "사용자 체크포인트로 병합" in html


def test_character_maker_actions_follow_llm_to_user_flow():
    html = _html()
    action_start = html.index('<div class="cm-stage-actions">')
    action_end = html.index('<div id="cm-diff-strip"', action_start)
    actions = html[action_start:action_end]

    assert actions.index('id="cm-accept-btn"') < actions.index('id="cm-generate-btn"')
    assert "사용자에게 적용 →" in actions
    assert "사용자 이미지 생성" in actions
    assert 'class="cm-stage-action-side user"' in actions
    assert "grid-template-columns: minmax(170px, .8fr) minmax(290px, 1.2fr);" in html


def test_editor_wall_keeps_single_scroll_and_fields_order():
    html = _html()

    # 에디터 열은 여전히 단일 스크롤 컨테이너이고 필드 리스트 순서를 유지한다.
    editor_css = html[html.index(".cm-editor-wall {"):html.index(".cm-card {")]
    field_css = html[html.index(".cm-field-list {"):html.index(".cm-field-card {")]
    assert "overflow-y: auto;" in editor_css
    assert "order: 2;" in field_css
    assert "overflow: visible;" in field_css


def test_settings_wall_is_a_fixed_side_drawer_with_handle():
    html = _html()

    # 설정 패널은 에디터 열 플로우를 벗어나 우측에 뜨는 fixed 드로어.
    settings_css = html[html.index(".cm-settings-wall {"):html.index(".cm-settings-panel {")]
    assert "position: fixed;" in settings_css
    assert "right: 0;" in settings_css
    assert "z-index:" in settings_css

    # 항상 보이는 명확한 손잡이 + 슬라이드 패널.
    assert 'id="cm-settings-handle"' in html
    assert "cm-settings-handle-text" in html  # 세로 라벨
    assert "cm-settings-handle-arrow" in html
    assert 'class="cm-settings-panel"' in html

    # 접힘(기본)에서 패널은 우측 바깥으로 슬라이드되어 숨고, 열리면 돌아온다.
    panel_css = html[html.index(".cm-settings-panel {"):html.index(".cm-settings-head {")]
    assert "transform: translateX" in panel_css
    assert ".cm-settings-wall:not(.collapsed) .cm-settings-panel" in html

    # 패널 본체는 자체 스크롤을 갖는다.
    body_css = html[html.index(".cm-settings-body {"):html.index(".cm-settings-handle {")]
    assert "overflow-y: auto;" in body_css


def test_settings_drawer_marks_prompt_composition_as_locked():
    html = _html()

    # 외모·복장·표정·구도는 LLM+사람 협동 영역 → 🔒 잠금(읽기 전용) 표시.
    for field in ("appearance", "outfit", "expression", "composition"):
        assert f'data-cm-lock-summary="{field}"' in html
    assert "at-fill-lock-badge" in html
    assert "🔒 잠금" in html
    assert "function cmRenderPromptLockSummary" in html


def test_settings_drawer_disables_items_by_asset_workflow():
    html = _html()

    # 에셋 워크플로우(ILXL/ANIMA) 기반 가용성 잠금이 프리셋 셀렉트에 반영되어 있다.
    assert 'data-at-availability="sdxl-only"' in html
    assert 'data-at-availability="anima"' in html
    assert "function cmApplySettingsAvailability" in html
    assert "getAssetWorkflowCapabilities" in html
    # 워크플로우 타입은 설정→삽화의 전역값을 따르므로 드로어 셀렉트는 읽기전용(disabled).
    assert 'id="cm-setting-workflow" data-cm-setting="asset_workflow_type" disabled' in html
    assert "설정 → 삽화에서 결정" in html


def test_settings_drawer_marks_only_pose_and_ipadapter_as_locked():
    html = _html()

    # pose·ipadapter(참조 이미지)는 CM 미지원 잠금으로 남겨둔다.
    for key in ("ipadapter", "pose"):
        assert f'data-cm-unsupported="{key}"' in html, f"미지원 표시가 없습니다: {key}"
    assert "cm-unsupported" in html
    # 나머지(hires/detailer/face-crop/style-lora/face-lora)는 잠금을 풀고 실제 컨트롤을 제공한다.
    for key in ("hires", "detailer", "face-crop", "style-lora", "face-lora"):
        assert f'data-cm-unsupported="{key}"' not in html, f"여전히 잠겨 있습니다: {key}"
    # 기능 컨트롤이 마크업에 존재한다(에셋/오토매치와 동일 토큰 세트).
    assert 'id="cm-setting-hrf-sdxl"' in html
    assert 'id="cm-setting-hrf-anima"' in html
    assert 'id="cm-setting-style-lora-enabled"' in html
    assert 'id="cm-setting-face-lora-enabled"' in html
    assert 'id="cm-setting-face-crop-top"' in html
    assert 'id="cm-setting-sdxl-fd"' in html
    assert 'id="cm-setting-anima-ed"' in html
    # 미지원 컨트롤(ipadapter/pose)은 여전히 비활성화되어 있다.
    assert "<input type=\"checkbox\" disabled>" in html or 'type="checkbox" disabled' in html


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


def test_browser_uses_single_persistent_session_and_persists_settings():
    html = _html()

    # 단일 영속 세션: 식별자는 고정 'default'이며 sessionStorage에 저장하지 않는다.
    assert "const CM_SINGLE_SESSION_ID = 'default'" in html
    assert "CM_SESSION_STORAGE_KEY" not in html
    assert "sessionStorage.setItem(CM_SESSION_STORAGE_KEY" not in html
    # 서버 재시작 감지 폐기 분기는 제거되었다(세션이 디스크에 보존되므로).
    assert "cmSession.boot_id !== cmCapabilities.boot_id" not in html
    # 생성 설정은 세션을 갈아치워도 유지되도록 localStorage에 영속화한다.
    assert "const CM_SETTINGS_STORAGE_KEY = 'characterMakerSettings'" in html
    assert "function cmPersistSettings" in html
    assert "function cmMergePersistedSettings" in html
    assert "cmMergePersistedSettings()" in html


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


def test_character_maker_can_add_presets_to_an_existing_character():
    html = _html()

    assert 'id="cm-confirm-registration-mode"' in html
    assert '<option value="existing">기존 캐릭터에 외모·복장 추가</option>' in html
    assert 'id="cm-confirm-existing-character"' in html
    assert "function cmUpdateConfirmRegistrationMode()" in html
    assert "registration_mode: registrationMode" in html
    assert "assetTags?.characters" in html
    assert "현재 기본값 유지" in html


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


def test_rag_settings_install_the_prebuilt_huggingface_index():
    html = _html()

    assert 'class="cm-rag-settings-page"' in html
    assert 'class="cm-rag-settings-grid"' in html
    assert 'id="setting-character-maker-rag-data-install-btn"' in html
    assert "Hugging Face variant-b" in html
    assert "다운로드 및 설치" in html
    assert "크기 + SHA-256" in html
    assert "/api/character_maker/rag/dataset" in html
    assert "/api/character_maker/rag/install" in html
    assert 'id="setting-character-maker-rag-data-progress"' in html
    assert 'id="setting-character-maker-rag-data-progress-fill"' in html
    assert "handleCmRagInstallProgress" in html
    assert "character_maker_rag_install_progress" in html
    assert "cmInstallRagDataset" in html
    assert "cmRefreshRagArtifactStatus" in html
    assert "setting-character-maker-rag-repo-path" not in html
    assert "cmFindRagDataset" not in html
    assert "form.append('dataset'" not in html
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


def test_character_maker_has_free_edit_natural_language_card():
    html = _html()
    # 자유편집영역에 자연어 카드(외모/복장/표정/구도와 동일: preset-row + textarea + 잠금).
    assert 'class="cm-field-card cm-natural-card"' in html
    assert 'data-field="natural_language"' in html
    assert 'data-natural-text' in html
    assert "cmToggleLock('natural_language')" in html
    assert "cmNaturalLanguageInput" in html
    assert "cmRenderNaturalLanguage" in html
    # 자연어 프리셋 "불러오기" 검색 셀렉트가 카드 안에 있다.
    assert "cmLoadNaturalLanguagePreset" in html
    # 설정 패널의 구식 자연어 드롭다운은 제거되었다(settings가 아닌 최상위 natural_language).
    assert 'id="cm-setting-natural"' not in html
    assert 'data-cm-setting="natural_language_preset"' not in html
    # Anima 자연어 가이드가 UI 도움말에 반영되어 있다.
    assert "Anima는 자연어 이해력" in html
