from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_backup_card_has_one_video_button_immediately_before_delete() -> None:
    video_button = FRONTEND.index('class="video-backup-btn"')
    delete_button = FRONTEND.index('class="delete-backup-btn"', video_button)

    assert video_button < delete_button
    assert "🎬 영상화" in FRONTEND[video_button:delete_button]
    assert "영상화에 쓸 대사 없는 원본이 없는 백업" in FRONTEND
    assert "openVideoWorkspace(" in FRONTEND[video_button:delete_button]


def test_animated_cards_expose_existing_video_postprocess_action() -> None:
    backup_button = FRONTEND.index('class="video-reprocess-btn"')
    delete_button = FRONTEND.index('class="delete-backup-btn"', backup_button)
    backup_section = FRONTEND[backup_button:delete_button]
    assert "영상 후처리" in backup_section
    assert "openBackupVideoReprocess(" in backup_section
    assert "b.is_animated" in backup_section

    assert FRONTEND.count('class="asset-video-postprocess-btn"') == 2
    assert FRONTEND.count("openAssetVideoReprocess({") >= 2
    assert "img.is_animated" in FRONTEND


def test_name_mapping_wizard_adds_zip_only_batch_video_postprocess_step() -> None:
    wizard = FRONTEND.split("async function openNameMappingModal()", 1)[1].split(
        "let assetRefPickerSelected", 1
    )[0]

    assert 'data-step="4"' in wizard
    assert 'id="nm-page-4"' in wizard
    assert 'id="nm-video-store"' in wizard
    assert "영상 후처리 설정" in wizard
    assert "일괄 적용" in wizard
    assert "function nmHandleFinalAction()" in wizard
    assert "videoFiles.length" in wizard
    assert "/api/asset_mode/export_video_sessions" in wizard
    assert "export_video_session_id: exportVideoSessionId" in wizard
    assert ".nm-nested-overlay" in FRONTEND
    assert "z-index: 10020" in FRONTEND
    # 기존 카드/백업 단일 후처리는 새 단계와 병행해 그대로 유지한다.
    assert FRONTEND.count('class="asset-video-postprocess-btn"') == 2
    assert "openBackupVideoReprocess(" in FRONTEND


def test_backup_card_disables_regen_and_edit_for_animated_backups() -> None:
    """영상(is_animated) 백업 카드의 재생성/수정 버튼은 회색 disabled로 막힌다.
    에셋 카드의 EDIT 툴 차단과 같은 패턴(b.is_animated 우선 평가)."""
    regen = FRONTEND.index('class="regen-btn"')
    edit = FRONTEND.index('class="edit-prompt-btn"', regen)
    assert regen < edit

    assert "영상 백업에는 재생성을 사용할 수 없습니다" in FRONTEND[regen:edit]
    assert "b.is_animated ?" in FRONTEND[regen:edit]
    # 영상이 아닐 때 기존 '프롬프트 없음' 분기는 유지
    assert 'disabled title="프롬프트 없음"' in FRONTEND[regen:edit]

    assert "영상 백업에는 프롬프트 수정을 사용할 수 없습니다" in FRONTEND[edit:]
    assert "b.is_animated ?" in FRONTEND[edit:]
    assert 'disabled title="프롬프트 없음"' in FRONTEND[edit:]


def test_video_page_uses_modal_entry_with_six_workflow_cards() -> None:
    assert 'id="video-modal"' in FRONTEND
    assert "closeVideoModal()" in FRONTEND
    assert "openVideoWorkspace(" in FRONTEND
    assert 'id="tab-btn-video"' not in FRONTEND
    assert "switchTab('video')" not in FRONTEND
    assert 'id="tab-video-content"' not in FRONTEND
    assert 'id="video-generation-source"' in FRONTEND
    assert "video-generation-overlay" not in FRONTEND
    assert "{ key: 'video_generation', label: '영상화'" in FRONTEND
    assert "key: 'video_t2v', label: '영상화" not in FRONTEND
    assert "key: 'video_i2v', label: '영상화" not in FRONTEND
    assert "key: 'video_first_last', label: '영상화" not in FRONTEND
    assert 'value="t2v"' not in FRONTEND
    assert FRONTEND.count('type="radio" name="video-mode-choice"') == 6
    for value in (
        "i2v:standard", "i2v:fast",
        "first_last:standard", "first_last:fast",
        "ref2v:standard", "ref2v:fast",
    ):
        assert f'value="{value}"' in FRONTEND
    card_positions = [
        FRONTEND.index(f'value="{value}"')
        for value in (
            "i2v:standard", "i2v:fast",
            "first_last:standard", "first_last:fast",
            "ref2v:standard", "ref2v:fast",
        )
    ]
    assert card_positions == sorted(card_positions)
    assert "고속 I2V" in FRONTEND
    assert "고속 FLF2V" in FRONTEND
    assert "고속 REF2V" in FRONTEND
    assert 'value="ref2v:standard" disabled aria-disabled="true"' not in FRONTEND
    assert 'value="ref2v:standard" onchange="selectVideoMode(\'ref2v\', \'standard\')"' in FRONTEND
    assert "Base 20-step · TeaCache 0.10 · 960×544" in FRONTEND
    assert "전용 4-step · 960×544 고정" in FRONTEND
    # 첫·마지막 모드에서 마지막 프레임 미리보기가 존재해야 한다
    assert 'id="video-frame-last"' in FRONTEND
    assert 'id="video-generation-last-preview"' in FRONTEND
    assert "onVideoLastFrameChange()" in FRONTEND
    assert 'id="video-frame-references"' in FRONTEND
    assert 'id="video-generation-ref-2"' in FRONTEND
    assert 'id="video-generation-ref-3"' in FRONTEND
    assert "reference_refs: getSelectedVideoReferenceRefs()" in FRONTEND


def test_video_references_use_shared_lazy_picker_instead_of_dropdowns() -> None:
    modal = FRONTEND.split('id="video-modal"', 1)[1].split(
        '<!-- 기존 animated AVIF/WebP 재후처리 모달 -->', 1
    )[0]

    assert '<select id="video-generation-last"' not in modal
    assert '<select id="video-generation-ref-2"' not in modal
    assert '<select id="video-generation-ref-3"' not in modal
    assert 'id="video-generation-last" type="hidden"' in modal
    assert 'id="video-generation-ref-2" type="hidden"' in modal
    assert 'id="video-generation-ref-3" type="hidden"' in modal
    for target in ("source", "last", "ref-2", "ref-3"):
        assert f"openVideoReferencePicker('{target}')" in modal

    assert 'id="video-reference-picker-modal"' in modal
    assert 'id="video-reference-picker-tab-illustration"' in modal
    assert 'id="video-reference-picker-tab-asset"' in modal
    assert '>삽화</button>' in modal
    assert '>에셋</button>' in modal
    assert "const VIDEO_REFERENCE_PICKER_PAGE_SIZE = 40;" in FRONTEND
    assert "async function loadMoreVideoReferenceOptions(reset = false)" in FRONTEND
    assert "tab: videoReferencePickerState.tab" in FRONTEND
    assert "offset: String(videoReferencePickerState.offset)" in FRONTEND
    assert "limit: String(VIDEO_REFERENCE_PICKER_PAGE_SIZE)" in FRONTEND


def test_video_workspace_loads_only_its_direct_reference_before_picker_opens() -> None:
    loader = FRONTEND.split(
        "async function loadVideoReferenceOptions(preferredReference)", 1
    )[1].split("function applyVideoSecondaryMotionConfig", 1)[0]

    assert "videoReferenceDirectEndpoint(preferredReference)" in loader
    assert "registerVideoReferenceOptions(data.options);" in loader
    assert "/api/asset_mode/characters/" not in loader
    assert "videoReferencePickerState.offset" not in loader


def test_video_asset_picker_preserves_character_and_outfit_expression_hierarchy() -> None:
    modal = FRONTEND.split('id="video-reference-picker-modal"', 1)[1].split(
        '<!-- 기존 animated AVIF/WebP 재후처리 모달 -->', 1
    )[0]
    hierarchy = FRONTEND.split(
        "async function loadVideoAssetPickerLevel()", 1
    )[1].split("async function loadMoreVideoReferenceOptions", 1)[0]

    assert 'id="video-reference-picker-breadcrumb"' in modal
    assert "assetLevel: 'characters'" in FRONTEND
    assert "assetLevel === 'characters'" in hierarchy
    assert "assetLevel === 'combinations'" in hierarchy
    assert "assetLevel === 'images'" in hierarchy
    assert "assetLevel === 'outfits'" not in hierarchy
    assert "assetLevel === 'expressions'" not in hierarchy
    assert "'/api/asset_mode/characters'" in hierarchy
    assert "`/api/asset_mode/characters/${enc(character)}/gallery`" in hierarchy
    assert "/outfits/${enc(outfit)}/expressions/${enc(expression)}/images" in hierarchy
    assert "label: `${reference.outfit} · ${reference.expression}`" in hierarchy
    assert "async function openVideoAssetCharacters()" in FRONTEND
    assert "async function openVideoAssetCharacter(character)" in FRONTEND
    assert "async function openVideoAssetCombination(character, outfit, expression)" in FRONTEND
    assert "await loadVideoAssetPickerLevel();" in FRONTEND


def test_existing_video_postprocess_modal_collects_requested_controls() -> None:
    assert 'id="video-reprocess-modal"' in FRONTEND
    assert 'id="video-reprocess-target-size"' in FRONTEND
    assert 'id="video-reprocess-fps"' in FRONTEND
    assert 'id="video-reprocess-upscale-enabled"' in FRONTEND
    assert 'id="video-reprocess-upscale-model"' in FRONTEND
    assert 'id="video-reprocess-upscale-scale"' in FRONTEND
    assert FRONTEND.count('name="video-reprocess-output-format"') == 2
    assert "/api/video/reprocess/enqueue" in FRONTEND
    assert "target_size_mb: targetSizeMb" in FRONTEND
    assert "fps," in FRONTEND
    assert "원본을 보존하고 새 결과 생성" in FRONTEND


def test_video_prompt_diagnostics_are_rendered_for_backups_and_assets() -> None:
    assert "function renderVideoPromptDetails(record, finalPrompt)" in FRONTEND
    assert "[연출 지시 출처]" in FRONTEND
    assert "[다듬기 전 사용자 원문]" in FRONTEND
    assert "[연출 지시]" in FRONTEND
    assert "Visual Context · 그림 직접 분석" in FRONTEND
    assert "Visual Context · 생성 프롬프트 해석" in FRONTEND
    assert "[최종 H3 프롬프트]" in FRONTEND
    assert "b.is_video_prompt === true" in FRONTEND
    assert "img.is_video_animation === true" in FRONTEND


def test_video_llm_flow_keeps_refine_provenance_and_original_input() -> None:
    assert "let videoInstructionOriginalInput = '';" in FRONTEND
    assert "let videoInstructionLlmTrace = [];" in FRONTEND
    assert "appendVideoInstructionProvenance(target.value, data.llm_trace)" in FRONTEND
    assert "replaceVideoInstructionProvenance(data.llm_trace)" in FRONTEND
    assert "instruction_original: videoInstructionOriginalInput" in FRONTEND
    assert "instruction_llm_trace: [...videoInstructionLlmTrace]" in FRONTEND
    assert "snapshotVideoInstructionProvenance(target.value)" in FRONTEND
    assert "영상화 LLM 흐름 단계" in FRONTEND
    assert "다듬기 전 사용자 원문" in FRONTEND
    assert "최종 H3 프롬프트" in FRONTEND


def test_settings_expose_standard_and_fast_video_workflows_without_card_selection() -> None:
    asset_tab = FRONTEND.index("switchSettingsTab('asset')")
    video_tab = FRONTEND.index("switchSettingsTab('video')", asset_tab)
    rag_tab = FRONTEND.index("switchSettingsTab('character_maker')", video_tab)

    assert asset_tab < video_tab < rag_tab
    assert 'id="settings-tab-video"' in FRONTEND
    assert 'id="setting-video-i2v-workflow-filename"' in FRONTEND
    assert 'id="setting-video-first-last-workflow-filename"' in FRONTEND
    assert 'id="setting-video-ref2v-workflow-filename"' in FRONTEND
    assert 'id="setting-video-i2v-fast-workflow-filename"' in FRONTEND
    assert 'id="setting-video-first-last-fast-workflow-filename"' in FRONTEND
    assert 'id="setting-video-ref2v-fast-workflow-filename"' in FRONTEND
    assert "배포_영상_H3_I2V_v1.json" in FRONTEND
    assert "배포_영상_H3_FLF2V_v1.json" in FRONTEND
    assert "배포_영상_H3_I2V_고속_v1.json" in FRONTEND
    assert "배포_영상_H3_FLF2V_고속_v1.json" in FRONTEND
    assert "배포_영상_H3_REF2V_v1.json" in FRONTEND
    assert "배포_영상_H3_REF2V_고속_v1.json" in FRONTEND
    assert "onVideoWorkflowFilenameInput('i2v', this.value)" in FRONTEND
    assert "onVideoWorkflowFilenameInput('first_last', this.value)" in FRONTEND
    assert "onVideoWorkflowFilenameInput('ref2v', this.value)" in FRONTEND
    assert "onVideoWorkflowFilenameInput('i2v_fast', this.value)" in FRONTEND
    assert "onVideoWorkflowFilenameInput('first_last_fast', this.value)" in FRONTEND
    assert "onVideoWorkflowFilenameInput('ref2v_fast', this.value)" in FRONTEND
    assert 'data-video-workflow-mode="i2v"' in FRONTEND
    assert 'data-video-workflow-mode="first_last"' in FRONTEND
    assert 'data-video-workflow-mode="ref2v"' in FRONTEND
    assert 'data-video-workflow-mode="i2v_fast"' in FRONTEND
    assert 'data-video-workflow-mode="first_last_fast"' in FRONTEND
    assert 'data-video-workflow-mode="ref2v_fast"' in FRONTEND
    assert 'placeholder="공식 전용 비고속판 미지원" disabled aria-disabled="true"' not in FRONTEND
    assert "selectVideoWorkflowFile(list.dataset.videoWorkflowMode, path, filename);" in FRONTEND

    panel = FRONTEND.split('id="settings-tab-video"', 1)[1].split(
        'id="settings-tab-character_maker"', 1
    )[0]
    assert "asset-workflow-selector" not in panel
    assert "option-card" not in panel


def test_video_workflow_settings_load_and_save_only_supported_paths() -> None:
    assert "const videoWorkflowPaths = currentConfig.video_workflow_source_paths || {};" in FRONTEND
    assert "'i2v', 'first_last', 'ref2v'," in FRONTEND
    assert "'i2v_fast', 'first_last_fast', 'ref2v_fast'," in FRONTEND
    assert "video_workflow_source_paths: {" in FRONTEND
    assert "...(currentConfig.video_workflow_source_paths || {})," not in FRONTEND
    assert "i2v: document.getElementById('setting-video-i2v-workflow-source-path').value" in FRONTEND
    assert "first_last: document.getElementById('setting-video-first-last-workflow-source-path').value" in FRONTEND
    assert "ref2v: document.getElementById('setting-video-ref2v-workflow-source-path').value" in FRONTEND
    assert "i2v_fast: document.getElementById('setting-video-i2v-fast-workflow-source-path').value" in FRONTEND
    assert "first_last_fast: document.getElementById('setting-video-first-last-fast-workflow-source-path').value" in FRONTEND
    assert "ref2v_fast: document.getElementById('setting-video-ref2v-fast-workflow-source-path').value" in FRONTEND
    assert "updateVideoWorkflowPath('i2v');" in FRONTEND
    assert "updateVideoWorkflowPath('first_last');" in FRONTEND
    assert "updateVideoWorkflowPath('ref2v');" in FRONTEND
    assert "updateVideoWorkflowPath('i2v_fast');" in FRONTEND
    assert "updateVideoWorkflowPath('first_last_fast');" in FRONTEND
    assert "updateVideoWorkflowPath('ref2v_fast');" in FRONTEND
    assert "[VIDEO_SETTINGS] ${mode} 워크플로우 파일 검색 실패:" in FRONTEND


def test_video_page_separates_fast_aspect_ratio_and_mp_level() -> None:
    assert 'id="video-generation-aspect-ratio"' in FRONTEND
    for value in ("auto", "1:1", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21", "3:2", "2:3", "5:4", "4:5"):
        assert f'<option value="{value}"' in FRONTEND
    assert 'id="video-generation-quality-level"' in FRONTEND
    assert '<option value="low">FAST 저화질 · 0.2 MP</option>' in FRONTEND
    assert '<option value="medium" selected>FAST 기본 · 0.35 MP</option>' in FRONTEND
    assert '<option value="high">FAST 고화질 · 0.5 MP</option>' in FRONTEND
    assert '<option value="native">Native 최대 · 비율별</option>' in FRONTEND
    assert "calculateVideoFastResolution(aspectRatio, qualityLevel)" in FRONTEND
    assert "aspect_ratio: aspectRatio" in FRONTEND
    assert "quality_level: qualityLevel" in FRONTEND
    assert "원본 분석 후 결정" in FRONTEND
    assert "최소 중앙 크롭" in FRONTEND
    assert "VIDEO_FAST_768_ASPECT_KEYS" in FRONTEND
    assert "calculateVideoFast768Resolution(aspectRatio)" in FRONTEND
    assert "calculateVideoFastModeResolution(aspectRatio, qualityLevel, mode = '')" in FRONTEND
    # I2V/FLF2V 고속은 768p/실험 MP 선택을 유지한다.
    assert "VIDEO_FAST_QUALITY_OPTIONS" in FRONTEND
    assert "고속 기본 · 짧은 변 768p" in FRONTEND
    assert "저화질 · 0.2 MP (실험적)" in FRONTEND
    assert "기본 · 0.35 MP (실험적)" in FRONTEND
    assert "고화질 · 0.5 MP (실험적)" in FRONTEND
    # 고속 REF2V는 공식 예제 하나만 지원하므로 두 셀렉트를 잠근다.
    assert "VIDEO_REF_FAST_FIXED_ASPECT_RATIO = '16:9'" in FRONTEND
    assert "VIDEO_REF_FAST_FIXED_WIDTH = 960" in FRONTEND
    assert "VIDEO_REF_FAST_FIXED_HEIGHT = 544" in FRONTEND
    assert "고속 REF 고정 · 960×544" in FRONTEND
    assert "일반 REF 고정 · 960×544" in FRONTEND
    assert "REF 20-step · TeaCache 0.10" in FRONTEND
    assert "16:9 · 960×544 고정" in FRONTEND
    assert "qualitySelect.disabled = true" in FRONTEND
    assert "aspectSelect.disabled = true" in FRONTEND
    assert "fast-ref-fixed-960x544" in FRONTEND
    assert "onVideoQualityLevelChange()" in FRONTEND
    assert "qualitySelect.dataset.activeVariant" in FRONTEND
    assert "workflow_variant: selectedVideoWorkflowVariant()" in FRONTEND
    assert "/api/video/reference-options" in FRONTEND
    assert "/api/video/enqueue" in FRONTEND
    assert FRONTEND.count('type="radio" name="video-output-format"') == 2
    assert 'name="video-output-format" value="avif" checked' in FRONTEND
    assert 'name="video-output-format" value="webp"' in FRONTEND
    assert 'id="video-generation-duration" type="number" min="1" max="15" step="1"' in FRONTEND
    assert 'id="video-generation-upscale-enabled"' in FRONTEND
    assert 'id="video-generation-upscale-scale"' in FRONTEND
    for model in ("realesr-animevideov3", "anime4k-fast-m", "lanczos", "none"):
        assert f'name="video-upscale-model" value="{model}"' in FRONTEND
    assert "upscale_enabled: upscaleEnabled" in FRONTEND
    assert "upscale_scale: upscaleScale" in FRONTEND
    assert "upscale_model: upscaleEnabled ? upscaleModel : ''" in FRONTEND
    assert "output_format: outputFormat" in FRONTEND
    assert "secondary_motion: secondaryMotion" in FRONTEND
    assert "duration," in FRONTEND


def test_video_modal_opens_before_async_loading_and_applies_saved_defaults() -> None:
    modal_open = FRONTEND.split(
        "async function openVideoWorkspaceForReference(reference)", 1
    )[1].split("function closeVideoModal()", 1)[0]

    assert modal_open.index("modal.classList.add('visible');") < modal_open.index(
        "await Promise.all(["
    )
    assert "applyVideoGenerationDefaults(currentConfig);" in modal_open
    assert "applyVideoGenerationDefaults(cfg);" in modal_open
    assert FRONTEND.count('onchange="onVideoUpscaleModelChange()"') == 4
    assert FRONTEND.count('onchange="selectVideoUpscaleScale(this.value, true)"') == 3
    assert "function onVideoUpscaleModelChange()" in FRONTEND
    assert "videoPostprocessSynced = true;" in FRONTEND


def test_video_modal_can_persist_non_content_defaults() -> None:
    modal = FRONTEND.split('id="video-modal"', 1)[1].split(
        '<!-- 기존 animated AVIF/WebP 재후처리 모달 -->', 1
    )[0]
    assert 'id="video-generation-save-settings"' in modal
    assert "saveVideoGenerationDefaults()" in modal
    assert modal.index('id="video-generation-save-settings"') < modal.index(
        'id="video-generation-submit"'
    )

    collector = FRONTEND.split("function collectVideoGenerationDefaults()", 1)[1].split(
        "async function saveVideoGenerationDefaults()", 1
    )[0]
    for field in (
        "mode",
        "duration",
        "aspect_ratio",
        "quality_level",
        "loop",
        "visual_context_source",
        "instruction_language",
        "include_dialogue_context",
        "allow_camera_motion",
        "allow_background_change",
        "upscale_model",
        "upscale_scale",
        "output_format",
    ):
        assert f"{field}:" in collector
    assert "video-generation-instruction')?.value" not in collector
    assert "video-generation-source" not in collector
    assert "video-generation-last" not in collector

    saver = FRONTEND.split("async function saveVideoGenerationDefaults()", 1)[1].split(
        "function normalizedVideoDuration()", 1
    )[0]
    assert "video_generation_defaults: collected.defaults" in saver
    assert "video_secondary_motion: collected.secondaryMotion" in saver
    assert "currentConfigLoadPromise = Promise.resolve(currentConfig);" in saver
    assert "영상화 설정을 영구 저장했습니다." in saver


def test_video_page_generates_editable_direction_draft_in_separate_llm_queue() -> None:
    assert 'id="video-ai-settings-title" class="video-panel-title">AI 연출 문맥</span>' in FRONTEND
    assert 'class="video-ai-subgroup"' in FRONTEND
    assert 'id="video-ai-draft-title" class="video-panel-title">AI에게 맡기기</span>' in FRONTEND
    assert 'id="video-generation-draft-button"' in FRONTEND
    assert "requestVideoInstructionDraft()" in FRONTEND
    assert "/api/video/instruction-draft" in FRONTEND
    assert "LLM 큐 대기 및 참조 이미지 분석 중" in FRONTEND
    assert "instructionInput.value = String(data.draft).trim()" in FRONTEND
    assert "자동 제출되지 않습니다" in FRONTEND
    assert "if (!instruction)" in FRONTEND
    assert "instruction," in FRONTEND
    assert "auto_instruction: autoInstruction" not in FRONTEND
    assert 'id="video-generation-instruction-language"' in FRONTEND
    assert '<option value="ko" selected>한국어</option>' in FRONTEND
    assert '<option value="en">English</option>' in FRONTEND
    assert 'id="video-generation-include-dialogue-context" type="checkbox" checked' in FRONTEND
    assert "include_dialogue_context: isVideoDialogueContextEnabled()" in FRONTEND
    assert "대사·감정 정보 전달" in FRONTEND
    assert 'id="video-generation-allow-camera-motion" type="checkbox" checked' in FRONTEND
    assert "allow_camera_motion: isVideoCameraMotionAllowed()" in FRONTEND
    assert 'id="video-generation-allow-background-change" type="checkbox"' in FRONTEND
    assert "allow_background_change: isVideoBackgroundChangeAllowed()" in FRONTEND


def test_secondary_animation_is_inside_ai_direction_context_panel() -> None:
    modal = FRONTEND.split('id="video-modal"', 1)[1].split(
        '<!-- 기존 animated AVIF/WebP 재후처리 모달 -->', 1
    )[0]
    ai_panel = modal.split('id="video-ai-settings-title"', 1)[1].split(
        '<div class="video-options-layout">', 1
    )[0]

    assert 'id="video-generation-secondary-motion"' in ai_panel
    assert ai_panel.count('id="video-generation-secondary-motion"') == 1
    assert "세컨더리 애니메이션" in ai_panel


def test_video_modal_orders_direction_and_pipeline_before_postprocess() -> None:
    modal = FRONTEND.split('id="video-modal"', 1)[1].split(
        '<!-- 삽화 설정 탭 -->', 1
    )[0]
    last_frame = modal.index('id="video-frame-last"')
    direction = modal.index('id="video-generation-instruction"')
    pipeline = modal.index('id="video-ai-settings-title"')
    upscale = modal.index('id="video-generation-upscale-enabled"')
    output = modal.index('name="video-output-format"')

    assert last_frame < direction < pipeline < upscale < output
    assert 'id="video-generation-visual-context-source"' in modal
    assert '<option value="image" selected>그림 직접 분석</option>' in modal
    assert '<option value="prompt">생성 프롬프트에서 구축</option>' in modal
    assert "visual_context_source: visualContextSource" in FRONTEND


def test_asset_lv1_and_lv2_cards_support_video_and_block_edit_for_animations() -> None:
    upload_lv2 = FRONTEND[
        FRONTEND.index("async function auRenderImages("):
        FRONTEND.index("let auQwenEditState")
    ]
    generation_lv2 = FRONTEND[
        FRONTEND.index("async function loadAssetImages()"):
        FRONTEND.index("async function setAssetRepresentative(")
    ]
    for section in (upload_lv2, generation_lv2):
        edit_button = section.index('class="qwen-edit-btn"')
        video_button = section.index('class="asset-video-btn"', edit_button)
        assert edit_button < video_button
        assert "영상 에셋에는 EDIT 툴을 사용할 수 없습니다" in section
        assert "openAssetVideoWorkspace({" in section
        assert "data-anim-src" in section
        assert "data-poster-src" in section

    assert FRONTEND.count('class="gallery-video-btn"') >= 2
    assert "representative_is_animated" in FRONTEND
    assert "videoReferenceDirectEndpoint(preferredReference)" in FRONTEND
    assert "source_ref: sourceRef" in FRONTEND
    assert "last_ref: lastRef" in FRONTEND
    assert "case 'asset_video_created':" in FRONTEND


def test_asset_cards_show_llm_flow_after_video_actions_and_disable_it_for_stills() -> None:
    upload_lv2 = FRONTEND[
        FRONTEND.index("async function auRenderImages("):
        FRONTEND.index("let auQwenEditState")
    ]
    generation_lv2 = FRONTEND[
        FRONTEND.index("async function loadAssetImages()"):
        FRONTEND.index("async function setAssetRepresentative(")
    ]
    for section in (upload_lv2, generation_lv2):
        video_button = section.index('class="asset-video-btn"')
        postprocess_button = section.index('class="asset-video-postprocess-btn"', video_button)
        llm_flow_button = section.index('class="asset-llm-flow-btn"', postprocess_button)
        assert video_button < postprocess_button < llm_flow_button
        assert "openAssetLlmFlowModal({" in section
        assert 'img.is_animated ? \'title="이 애니메이션 에셋' in section
        assert 'disabled title="LLM 흐름은 애니메이션 에셋에서만' in section

    assert FRONTEND.count('class="asset-llm-flow-btn"') == 2
    assert "/images/${encodeURIComponent(reference.filename)}/llm_trace" in FRONTEND


def test_video_postprocess_shares_renamed_background_lane() -> None:
    assert "{key: 'background', icon: '⚙️', title: '백그라운드 처리'" in FRONTEND
    assert "Modal 다운로드 · 영상 업스케일/AVIF 변환" in FRONTEND
    assert "area === 'modal_download' || area === 'video_postprocess'" in FRONTEND
    assert "video_postprocess: '영상 후처리'" in FRONTEND


def test_every_comfy_task_allocation_can_select_all_three_instances() -> None:
    assert "item.localInstances || [1, 2, 3]" in FRONTEND
    assert "Comfy #3만 실행 중 · 로컬 대상 작업은 #3로 폴백" in FRONTEND
