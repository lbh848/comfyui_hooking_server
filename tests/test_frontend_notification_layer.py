from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _notification_reminder_source() -> str:
    return FRONTEND.split("function showNotiReminderBubble(unreadCount) {", 1)[1].split(
        "function checkAndRemindNoti() {", 1
    )[0]


def test_collapsed_header_anchors_notification_to_visible_handle() -> None:
    source = _notification_reminder_source()

    assert "const handle = document.getElementById('header-controls-handle');" in source
    assert "const bellFullyVisible = bellRect.width > 0" in source
    assert "bellRect.top >= 0" in source
    assert "bellRect.bottom <= window.innerHeight" in source
    assert "const anchorRect = bellFullyVisible ? bellRect : handle.getBoundingClientRect();" in source


def test_visible_bell_remains_the_preferred_notification_anchor() -> None:
    source = _notification_reminder_source()

    anchor_choice = (
        "const anchorRect = bellFullyVisible ? bellRect : handle.getBoundingClientRect();"
    )
    assert anchor_choice in source
    assert source.index("const bellRect = bell.getBoundingClientRect();") < source.index(
        anchor_choice
    )


def test_notification_bubble_is_clamped_inside_every_viewport_edge() -> None:
    source = _notification_reminder_source()

    assert "const bubbleRect = bubble.getBoundingClientRect();" in source
    assert "const viewportMargin = 12;" in source
    assert "const safeLeft = Math.min(Math.max(centeredLeft, viewportMargin), maxLeft);" in source
    assert "if (safeTop > maxTop)" in source
    assert "safeTop = Math.min(Math.max(safeTop, viewportMargin), maxTop);" in source
    assert "max-width:calc(100vw - 24px)" in source


def test_notification_surfaces_share_the_existing_top_ui_layer() -> None:
    reminder = _notification_reminder_source()
    board = FRONTEND.split("async function openNotificationBoard() {", 1)[1].split(
        "// ══════════════════════════════════════════════════", 1
    )[0]

    assert "z-index:2147483647" in reminder
    assert "z-index:2147483647" in board
