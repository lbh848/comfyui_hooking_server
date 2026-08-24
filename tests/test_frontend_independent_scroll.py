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


def _function_source(signature: str, next_signature: str) -> str:
    start = FRONTEND.index(signature)
    end = FRONTEND.index(next_signature, start)
    return FRONTEND[start:end]


def test_asset_generation_restores_lv1_main_scroll_container():
    navigate = _function_source(
        "function navigateToImages(charName, outfit, expression)",
        "function syncAssetSelectFromDirname(selectId, dirname)",
    )
    render = _function_source(
        "async function renderCharacterGallery(charName)",
        "let _assetScrollCache = {};",
    )

    assert "document.getElementById('asset-main')" in navigate
    assert "_assetScrollCache[1] = main.scrollTop;" in navigate
    assert "document.getElementById('asset-main')" in render
    assert "main.scrollTop = savedScroll;" in render
    assert "window.scrollY" not in navigate
    assert "window.scrollTo" not in render


def test_asset_upload_restores_lv1_main_scroll_container():
    navigate = _function_source(
        "async function auNavigateToImages(charName, outfit, expression)",
        "async function auRenderImages(charName, outfit, expression)",
    )
    breadcrumb = _function_source(
        "async function auNavigateBreadcrumb(level)",
        "function auOnCharacterChange()",
    )

    assert "document.getElementById('au-main')" in navigate
    assert "_auScrollCache[1] = main.scrollTop;" in navigate
    assert "document.getElementById('au-main')" in breadcrumb
    assert "main.scrollTop = savedScroll;" in breadcrumb
    assert "window.scrollY" not in navigate
    assert "window.scrollTo" not in breadcrumb


def test_bot_character_detail_restores_main_and_sidebar_scroll():
    open_detail = _function_source(
        "async function openCharacterDetail(charName)",
        "async function closeCharacterDetail()",
    )
    close_detail = _function_source(
        "async function closeCharacterDetail()",
        "let _botCharDetailCache = null;",
    )
    breadcrumb_back = _function_source(
        "async function navToBotChars()",
        "function botNavBack()",
    )

    assert "document.getElementById('bot-main')" in open_detail
    assert "document.getElementById('bot-sidebar')" in open_detail
    assert "main: main ? main.scrollTop : 0" in open_detail
    assert "sidebar: sidebar ? sidebar.scrollTop : 0" in open_detail
    assert "main.scrollTop = 0;" in open_detail
    assert "sidebar.scrollTop = 0;" in open_detail

    assert "await renderBotCharacters();" in close_detail
    assert "main.scrollTop = savedScroll.main;" in close_detail
    assert "sidebar.scrollTop = savedScroll.sidebar;" in close_detail
    assert "requestAnimationFrame(restoreScroll);" in close_detail
    assert "await closeCharacterDetail();" in breadcrumb_back
