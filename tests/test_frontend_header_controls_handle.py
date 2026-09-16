from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_header_controls_have_a_clickable_handle_without_attention_animation() -> None:
    assert 'id="header-controls-dock" class="controls"' in FRONTEND
    assert 'id="header-controls-handle" class="header-controls-handle"' in FRONTEND
    assert 'onclick="toggleHeaderControlsDock(event)"' in FRONTEND
    assert 'aria-expanded="false"' in FRONTEND
    assert "--header-dock-peek: 22px;" in FRONTEND
    assert "header-controls-handle-nudge" not in FRONTEND
    assert "header-controls-handle-cue" not in FRONTEND


def test_header_controls_handle_can_pin_and_immediately_close_the_dock() -> None:
    assert "function toggleHeaderControlsDock(event)" in FRONTEND
    assert "dock.classList.toggle('is-pinned', willPin);" in FRONTEND
    assert "dock.classList.toggle('is-hover-suppressed', !willPin);" in FRONTEND
    assert "dock.classList.remove('is-hover-active');" in FRONTEND
    assert "header.app-header .controls.is-pinned" in FRONTEND


def test_header_controls_hover_opens_immediately_and_closes_after_a_grace_period() -> None:
    assert "const HEADER_CONTROLS_HOVER_CLOSE_DELAY_MS = 300;" in FRONTEND
    assert "headerControlsDock.addEventListener('pointerenter'" in FRONTEND
    assert "cancelHeaderControlsHoverClose();" in FRONTEND
    assert "'is-hover-active'," in FRONTEND
    assert "headerControlsDock.addEventListener('pointerleave'" in FRONTEND
    assert "window.setTimeout(() =>" in FRONTEND
    assert "}, HEADER_CONTROLS_HOVER_CLOSE_DELAY_MS);" in FRONTEND
    assert (
        "header.app-header .controls.is-hover-active:not(.is-hover-suppressed),"
        in FRONTEND
    )
    assert "transition-duration: 0.16s, 0.18s;" in FRONTEND


def test_header_controls_handle_respects_reduced_motion() -> None:
    reduced_motion = FRONTEND.split(
        "@media (prefers-reduced-motion: reduce) {", 1
    )[1].split("}", 1)[0]

    assert ".header-controls-handle .ui-icon," in reduced_motion
    assert "transition-duration: 0.01ms !important;" in reduced_motion
