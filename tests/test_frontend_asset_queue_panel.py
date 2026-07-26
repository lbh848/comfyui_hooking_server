from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_asset_queue_panel_starts_expanded():
    source = _frontend_source()

    assert "let assetQueueExpanded = true;" in source
    assert source.count("assetQueueExpanded = true;") == 1
    assert 'class="asset-queue-panel visible"' in source
    assert 'aria-expanded="true"' in source
    assert 'id="asset-queue-wall-tab"' in source
    assert 'aria-hidden="true" tabindex="-1"' in source


def test_asset_queue_panel_keeps_user_state_when_queue_becomes_empty():
    source = _frontend_source()
    render_function = _function_source(
        source, "renderAssetQueueUI()", "removeQueueItemBackend(id)"
    )

    assert "if (assetQueue.length === 0) assetQueueExpanded = true;" not in render_function


def test_asset_queue_panel_has_dedicated_collapse_control():
    source = _frontend_source()

    assert 'id="queue-panel-collapse-btn"' in source
    assert 'onclick="collapseAssetQueuePanel()"' in source
    collapse_function = _function_source(
        source, "collapseAssetQueuePanel()", "expandAssetQueuePanel()"
    )
    assert "setAssetQueuePanelExpanded(false);" in collapse_function
    assert "wallTab.focus();" in collapse_function


def test_asset_queue_panel_has_left_wall_expand_tab():
    source = _frontend_source()

    assert 'class="asset-queue-wall-tab"' in source
    assert 'onclick="expandAssetQueuePanel()"' in source
    assert '.asset-queue-container.collapsed .asset-queue-wall-tab' in source
    assert '.asset-queue-container.collapsed .asset-queue-bar' in source
    expand_function = _function_source(
        source, "expandAssetQueuePanel()", "toggleAssetQueuePanel()"
    )
    assert "setAssetQueuePanelExpanded(true);" in expand_function
    assert "collapseBtn.focus();" in expand_function


def test_asset_queue_panel_toggle_updates_visual_and_accessibility_state():
    source = _frontend_source()
    state_function = _function_source(
        source, "setAssetQueuePanelExpanded(expanded)", "collapseAssetQueuePanel()"
    )

    assert "panel.classList.add('visible');" in state_function
    assert "panel.classList.add('hidden');" in state_function
    assert "bar.setAttribute('aria-expanded', 'true');" in state_function
    assert "bar.setAttribute('aria-expanded', 'false');" in state_function
    assert "panel.inert = false;" in state_function
    assert "panel.inert = true;" in state_function
    assert "container.classList.add('collapsed');" in state_function
    assert "container.classList.remove('collapsed');" in state_function
    assert "wallTab.tabIndex = 0;" in state_function
    assert "wallTab.tabIndex = -1;" in state_function
    assert "console.error('[QUEUE] setAssetQueuePanelExpanded" in state_function
