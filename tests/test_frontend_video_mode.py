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
    assert 'data-video-mode-tab="t2v"' not in FRONTEND
    assert FRONTEND.count("data-video-mode-tab=") == 2
    for mode in ("i2v", "first_last"):
        assert f'data-video-mode-tab="{mode}"' in FRONTEND
        assert f"selectVideoMode('{mode}')" in FRONTEND
    # 첫·마지막 모드에서 마지막 프레임 미리보기가 존재해야 한다
    assert 'id="video-frame-last"' in FRONTEND
    assert 'id="video-generation-last-preview"' in FRONTEND
    assert "onVideoLastFrameChange()" in FRONTEND


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
    assert "animated AVIF 우선" in FRONTEND
    assert 'id="video-generation-upscale-enabled"' in FRONTEND
    assert 'id="video-generation-upscale-scale"' in FRONTEND
    assert "upscale_enabled: upscaleEnabled" in FRONTEND
    assert "upscale_scale: upscaleScale" in FRONTEND


def test_video_postprocess_shares_renamed_background_lane() -> None:
    assert "{key: 'background', icon: '⚙️', title: '백그라운드 처리'" in FRONTEND
    assert "Modal 다운로드 · 영상 업스케일/AVIF 변환" in FRONTEND
    assert "area === 'modal_download' || area === 'video_postprocess'" in FRONTEND
    assert "video_postprocess: '영상 후처리'" in FRONTEND


def test_every_comfy_task_allocation_can_select_all_three_instances() -> None:
    assert "item.localInstances || [1, 2, 3]" in FRONTEND
    assert "Comfy #3만 실행 중 · 로컬 대상 작업은 #3로 폴백" in FRONTEND
