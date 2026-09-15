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
    assert "header.app-header .controls.is-pinned" in FRONTEND
    assert "headerControlsDock.addEventListener('pointerleave'" in FRONTEND


def test_header_controls_handle_respects_reduced_motion() -> None:
    reduced_motion = FRONTEND.split(
        "@media (prefers-reduced-motion: reduce) {", 1
    )[1].split("}", 1)[0]

    assert ".header-controls-handle .ui-icon," in reduced_motion
    assert "transition-duration: 0.01ms;" in reduced_motion
