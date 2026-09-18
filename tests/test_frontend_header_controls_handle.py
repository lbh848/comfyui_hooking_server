from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_header_controls_have_a_clickable_handle_without_attention_animation() -> None:
    assert 'id="header-controls-dock" class="controls"' in FRONTEND
    assert 'id="header-controls-handle" class="header-controls-handle"' in FRONTEND
    assert '<div class="header-controls-scroll">' in FRONTEND
    assert 'onclick="toggleHeaderControlsDock(event)"' in FRONTEND
    assert 'aria-expanded="false"' in FRONTEND
    assert "--header-dock-peek: 22px;" in FRONTEND
    assert 'class="ui-icon header-controls-arrow--down"' in FRONTEND
    assert 'class="ui-icon header-controls-arrow--up"' in FRONTEND
    assert "header-controls-handle-nudge" not in FRONTEND
    assert "header-controls-handle-cue" not in FRONTEND


def test_header_controls_handle_can_pin_and_immediately_close_the_dock() -> None:
    assert "function toggleHeaderControlsDock(event)" in FRONTEND
    assert "dock.classList.toggle('is-pinned', willPin);" in FRONTEND
    assert "header.app-header .controls.is-pinned" in FRONTEND
    assert "handle.setAttribute('aria-expanded', String(willPin));" in FRONTEND


def test_header_controls_open_only_when_the_arrow_is_toggled() -> None:
    assert "is-hover-active" not in FRONTEND
    assert "is-hover-suppressed" not in FRONTEND
    assert "HEADER_CONTROLS_HOVER_CLOSE_DELAY_MS" not in FRONTEND
    assert "headerControlsDock.addEventListener('pointerenter'" not in FRONTEND
    assert "headerControlsDock.addEventListener('pointerleave'" not in FRONTEND
    assert "header.app-header .controls:has(:focus-visible)" not in FRONTEND
    assert "transition-duration: 0.16s, 0.18s;" in FRONTEND


def test_header_controls_handle_is_a_full_width_plain_arrow_grip() -> None:
    handle_rule = FRONTEND.split(
        ".header-controls-handle {", 1
    )[1].split("}", 1)[0]

    assert "width: auto;" in handle_rule
    assert "height: var(--header-dock-peek);" in handle_rule
    assert "border-radius: 0 0 11px 11px;" in handle_rule
    assert "right: 0;" in handle_rule
    assert "left: 0;" in handle_rule
    assert "margin: 0;" in handle_rule
    assert "transform:" not in handle_rule
    assert "border-radius: 999px;" not in handle_rule
    assert "header.app-header .controls::after {" not in FRONTEND
    assert "header.app-header .controls.is-pinned::after {" not in FRONTEND


def test_header_controls_handle_stays_outside_the_horizontal_scroll_area() -> None:
    dock_markup = FRONTEND.split(
        '<div id="header-controls-dock" class="controls">', 1
    )[1].split("</header>", 1)[0]
    handle_index = dock_markup.index('id="header-controls-handle"')
    scroll_index = dock_markup.index('<div class="header-controls-scroll">')
    assert handle_index < scroll_index

    responsive = FRONTEND.split("@media (max-width: 1180px) {", 1)[1].split(
        "@media (max-width: 620px) {", 1
    )[0]
    assert ".header-controls-scroll {" in responsive
    assert "overflow-x: auto;" in responsive
    assert "header.app-header .controls::-webkit-scrollbar" not in responsive
    assert ".header-controls-scroll::-webkit-scrollbar" in responsive


def test_header_controls_arrow_swaps_without_horizontal_motion() -> None:
    assert ".header-controls-arrow--up" in FRONTEND
    assert "header.app-header .controls.is-pinned .header-controls-arrow--down" in FRONTEND
    assert "header.app-header .controls.is-pinned .header-controls-arrow--up" in FRONTEND

    icon_rule = FRONTEND.split(
        ".header-controls-handle .ui-icon {", 1
    )[1].split("}", 1)[0]
    assert "transition:" not in icon_rule
    assert "transform:" not in icon_rule


def test_header_controls_ignores_repeat_clicks_during_one_transition() -> None:
    assert "const HEADER_CONTROLS_TOGGLE_LOCK_MS = 260;" in FRONTEND
    assert "let headerControlsToggleLocked = false;" in FRONTEND
    assert "if (headerControlsToggleLocked) return;" in FRONTEND
    assert "headerControlsToggleLocked = true;" in FRONTEND
    assert "headerControlsToggleLocked = false;" in FRONTEND
    assert "}, HEADER_CONTROLS_TOGGLE_LOCK_MS);" in FRONTEND


def test_header_controls_handle_respects_reduced_motion() -> None:
    reduced_motion = FRONTEND.split(
        "@media (prefers-reduced-motion: reduce) {", 1
    )[1].split("}", 1)[0]

    assert "header.app-header .controls," in reduced_motion
    assert ".header-controls-handle .ui-icon," not in reduced_motion
    assert "transition-duration: 0.01ms !important;" in reduced_motion
