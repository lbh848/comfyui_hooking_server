from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_asset_queue_panel_starts_expanded():
    source = _frontend_source()

    assert "let assetQueueExpanded = true;" in source
    assert 'class="asset-queue-panel visible"' in source
    assert 'aria-expanded="true"' in source


def test_asset_queue_panel_has_dedicated_collapse_control():
    source = _frontend_source()

    assert 'id="queue-panel-collapse-btn"' in source
    assert 'onclick="collapseAssetQueuePanel()"' in source
    collapse_function = _function_source(
        source, "collapseAssetQueuePanel()", "toggleAssetQueuePanel()"
    )
    assert "setAssetQueuePanelExpanded(false);" in collapse_function
    assert "bar.focus();" in collapse_function


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
    assert "console.error('[QUEUE] setAssetQueuePanelExpanded" in state_function
