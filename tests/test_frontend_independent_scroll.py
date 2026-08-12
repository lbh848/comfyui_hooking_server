from pathlib import Path


FRONTEND = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(
    encoding="utf-8"
)


def test_three_asset_views_use_independent_scroll_containers():
    assert FRONTEND.count('class="asset-container independent-scroll-container"') == 3
    assert "#tab-bot-content.active > .independent-scroll-container" in FRONTEND
    assert "#tab-asset-content.active > .independent-scroll-container" in FRONTEND
    assert "#tab-asset-upload-content.active > .independent-scroll-container" in FRONTEND


def test_independent_scroll_uses_vertical_accent_scrollbars():
    css_start = FRONTEND.index(".independent-scroll-container {")
    css_end = FRONTEND.index("/* ─── Pose 편집 스타일", css_start)
    css = FRONTEND[css_start:css_end]

    assert "overflow-x: hidden;" in css
    assert "overflow-y: auto;" in css
    assert "scrollbar-color: var(--accent) transparent;" in css
    assert "width: 6px;" in css
    assert "::-webkit-scrollbar-thumb" in css
    assert "height: 6px;" not in css


def test_independent_scroll_height_updates_on_tab_switch_and_resize():
    assert "function syncIndependentScrollHeight()" in FRONTEND
    assert "window.innerHeight - containerTop - mainBottomPadding" in FRONTEND
    assert (
        "window.addEventListener('resize', scheduleIndependentScrollHeightSync);"
        in FRONTEND
    )
    assert FRONTEND.count("scheduleIndependentScrollHeightSync();") >= 3
