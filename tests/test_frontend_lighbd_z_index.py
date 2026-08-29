from __future__ import annotations

from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")

MEMO_LEVEL_Z_INDEX = "2147483646"


def test_lighbd_widget_and_related_overlays_use_memo_level_z_index() -> None:
    widget_tag = FRONTEND.split('id="lighbd-llm-widget"', 1)[1].split(">", 1)[0]

    assert f"z-index:{MEMO_LEVEL_Z_INDEX};" in widget_tag
    assert (
        f".lighbd-history-overlay {{ z-index: {MEMO_LEVEL_Z_INDEX}; }}"
        in FRONTEND
    )
    assert (
        f".lighbd-detail-overlay {{ z-index: {MEMO_LEVEL_Z_INDEX}; }}"
        in FRONTEND
    )
    assert (
        f".lighbd-live-overlay {{ z-index: {MEMO_LEVEL_Z_INDEX}; }}"
        in FRONTEND
    )


def test_qwen_edit_does_not_lower_lighbd_layers() -> None:
    qwen_override = FRONTEND.split(
        "body.qwen-edit-open #lighbd-llm-widget,", 1
    )[1].split("}", 1)[0]

    assert f"z-index: {MEMO_LEVEL_Z_INDEX} !important;" in qwen_override
    assert ".asset-queue-container" not in qwen_override
