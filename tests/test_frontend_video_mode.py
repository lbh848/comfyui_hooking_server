from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_backup_card_has_one_video_button_immediately_before_delete() -> None:
    video_button = FRONTEND.index('class="video-backup-btn"')
    delete_button = FRONTEND.index('class="delete-backup-btn"', video_button)

    assert video_button < delete_button
    assert "🎬 영상화" in FRONTEND[video_button:delete_button]
    assert "영상화에 필요한 _raw 원본이 없는 백업" in FRONTEND
    assert "openVideoWorkspace(" in FRONTEND[video_button:delete_button]


def test_video_page_uses_modal_entry_with_two_internal_modes() -> None:
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
    assert FRONTEND.count('type="radio" name="video-mode-choice"') == 2
    for mode in ("i2v", "first_last"):
        assert f'name="video-mode-choice" value="{mode}"' in FRONTEND
    # 첫·마지막 모드에서 마지막 프레임 미리보기가 존재해야 한다
    assert 'id="video-frame-last"' in FRONTEND
    assert 'id="video-generation-last-preview"' in FRONTEND
    assert "onVideoLastFrameChange()" in FRONTEND


def test_settings_expose_i2v_and_flf2v_workflows_without_card_selection() -> None:
    asset_tab = FRONTEND.index("switchSettingsTab('asset')")
    video_tab = FRONTEND.index("switchSettingsTab('video')", asset_tab)
    rag_tab = FRONTEND.index("switchSettingsTab('character_maker')", video_tab)

    assert asset_tab < video_tab < rag_tab
    assert 'id="settings-tab-video"' in FRONTEND
    assert 'id="setting-video-i2v-workflow-filename"' in FRONTEND
    assert 'id="setting-video-first-last-workflow-filename"' in FRONTEND
    assert "배포_영상_H3_I2V_v1.json" in FRONTEND
    assert "배포_영상_H3_FLF2V_v1.json" in FRONTEND
    assert "onVideoWorkflowFilenameInput('i2v', this.value)" in FRONTEND
    assert "onVideoWorkflowFilenameInput('first_last', this.value)" in FRONTEND
    assert 'data-video-workflow-mode="i2v"' in FRONTEND
    assert 'data-video-workflow-mode="first_last"' in FRONTEND
    assert "selectVideoWorkflowFile(list.dataset.videoWorkflowMode, path, filename);" in FRONTEND

    panel = FRONTEND.split('id="settings-tab-video"', 1)[1].split(
        'id="settings-tab-character_maker"', 1
    )[0]
    assert "asset-workflow-selector" not in panel
    assert "option-card" not in panel


def test_video_workflow_settings_load_and_save_only_supported_paths() -> None:
    assert "const videoWorkflowPaths = currentConfig.video_workflow_source_paths || {};" in FRONTEND
    assert "for (const mode of ['i2v', 'first_last'])" in FRONTEND
    assert "video_workflow_source_paths: {" in FRONTEND
    assert "...(currentConfig.video_workflow_source_paths || {})," not in FRONTEND
    assert "i2v: document.getElementById('setting-video-i2v-workflow-source-path').value" in FRONTEND
    assert "first_last: document.getElementById('setting-video-first-last-workflow-source-path').value" in FRONTEND
    assert "updateVideoWorkflowPath('i2v');" in FRONTEND
    assert "updateVideoWorkflowPath('first_last');" in FRONTEND
    assert "[VIDEO_SETTINGS] ${mode} 워크플로우 파일 검색 실패:" in FRONTEND


def test_video_page_exposes_every_fast_resolution_and_queue_endpoint() -> None:
    for value in (
        "512×512",
        "512×384",
        "384×512",
        "672×384",
        "384×672",
        "672×288",
        "288×672",
        "576×384",
        "384×576",
        "480×384",
        "384×480",
    ):
        assert value in FRONTEND
    assert "/api/video/reference-options" in FRONTEND
    assert "/api/video/enqueue" in FRONTEND
    assert FRONTEND.count('name="video-output-format"') == 2
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
    assert "duration," in FRONTEND


def test_video_page_can_delegate_direction_to_ai() -> None:
    assert 'id="video-generation-auto-instruction"' in FRONTEND
    assert 'aria-pressed="false"' in FRONTEND
    assert "AI에게 맡기기" in FRONTEND
    assert "toggleVideoAutoInstruction()" in FRONTEND
    assert "isVideoAutoInstructionEnabled()" in FRONTEND
    assert "if (!autoInstruction && !instruction)" in FRONTEND
    assert "auto_instruction: autoInstruction" in FRONTEND
    assert "instruction: autoInstruction ? '' : instruction" in FRONTEND
    assert "백업에 대사·감정 정보가 있으면 행동과 표정에 함께 반영합니다" in FRONTEND


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
    assert "/video_references`" in FRONTEND
    assert "source_ref: sourceRef" in FRONTEND
    assert "last_ref: lastRef" in FRONTEND
    assert "case 'asset_video_created':" in FRONTEND


def test_video_postprocess_shares_renamed_background_lane() -> None:
    assert "{key: 'background', icon: '⚙️', title: '백그라운드 처리'" in FRONTEND
    assert "Modal 다운로드 · 영상 업스케일/AVIF 변환" in FRONTEND
    assert "area === 'modal_download' || area === 'video_postprocess'" in FRONTEND
    assert "video_postprocess: '영상 후처리'" in FRONTEND


def test_every_comfy_task_allocation_can_select_all_three_instances() -> None:
    assert "item.localInstances || [1, 2, 3]" in FRONTEND
    assert "Comfy #3만 실행 중 · 로컬 대상 작업은 #3로 폴백" in FRONTEND
