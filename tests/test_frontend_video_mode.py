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


def test_video_page_uses_one_entry_with_three_internal_tabs() -> None:
    assert 'id="tab-btn-video"' in FRONTEND
    assert "switchTab('video')" in FRONTEND
    assert 'id="tab-video-content" class="tab-content"' in FRONTEND
    assert 'id="video-generation-source"' in FRONTEND
    assert "video-generation-overlay" not in FRONTEND
    assert "{ key: 'video_generation', label: '영상화'" in FRONTEND
    assert "key: 'video_t2v', label: '영상화" not in FRONTEND
    assert "key: 'video_i2v', label: '영상화" not in FRONTEND
    assert "key: 'video_first_last', label: '영상화" not in FRONTEND
    assert FRONTEND.count("data-video-mode-tab=") == 3
    for mode in ("t2v", "i2v", "first_last"):
        assert f'data-video-mode-tab="{mode}"' in FRONTEND
        assert f"selectVideoMode('{mode}')" in FRONTEND


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
    assert "중간 MP4와 오디오는 보관하지 않습니다" in FRONTEND


def test_every_comfy_task_allocation_can_select_all_three_instances() -> None:
    assert "item.localInstances || [1, 2, 3]" in FRONTEND
    assert "Comfy #3만 실행 중 · 로컬 대상 작업은 #3로 폴백" in FRONTEND
