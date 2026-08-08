from pathlib import Path
import json
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.multi_char_mask import (
    extract_multi_char_prompt_payload,
    layout_fingerprint,
    normalize_multi_char_snapshot,
    prepare_region_mask,
    recover_multi_char_snapshot_from_sessions,
    render_region_mask,
    resolve_mask_directory,
    validate_multi_char_prompt_context,
    validate_multi_char_layout,
)


def _layout():
    return {
        "regions": [
            {"name": "Right", "x": 0.4, "y": 0.0, "width": 0.6, "height": 1.0},
            {"name": "Left", "x": 0.0, "y": 0.0, "width": 0.6, "height": 1.0},
        ]
    }


def _separated_layout():
    layout = _layout()
    layout["background_prompt"] = "wide shot, rooftop, blue-hour city lights"
    layout["composition_prompt"] = (
        "two distinct women, one on the left holding a chart, one on the right pointing upward"
    )
    layout["regions"][0]["character_prompt"] = "black hair, purple eyes, pointing upward"
    layout["regions"][1]["character_prompt"] = "grey hair, aqua eyes, holding a star chart"
    return layout


def _snapshot():
    return normalize_multi_char_snapshot({
        "enable": True,
        "character_order": ["Left", "Right"],
        "layout": _separated_layout(),
        "mask_location": "region_mask",
    })


def _positive_for(snapshot, *, names=None, fingerprint=None):
    payload = {
        "enable": True,
        "char_num": len(names or snapshot["character_order"]),
        "char_name_list": names or snapshot["character_order"],
        "mask_fingerprint": fingerprint or snapshot["mask_fingerprint"],
    }
    return "\n".join([
        "[ANIMA_CONTENT]",
        "scene tags",
        "[MULTI_CHAR]",
        json.dumps(payload, ensure_ascii=False),
        "[HRF_ACTIVATE]",
        "false",
    ])


def test_layout_is_sorted_left_to_right_and_overlap_is_preserved():
    normalized = validate_multi_char_layout(_layout(), ["Left", "Right"])

    assert normalized["character_order"] == ["Left", "Right"]
    assert [region["channel"] for region in normalized["regions"]] == ["R", "G"]

    image = render_region_mask(normalized, width=10, height=4)
    assert image.getpixel((1, 1)) == (255, 0, 0)
    assert image.getpixel((5, 1)) == (255, 255, 0)
    assert image.getpixel((8, 1)) == (0, 255, 0)


def test_layout_prompt_separation_is_required_and_preserved_when_requested():
    separated = _separated_layout()

    normalized = validate_multi_char_layout(
        separated,
        ["Left", "Right"],
        require_prompt_separation=True,
    )

    assert normalized["background_prompt"] == "wide shot, rooftop, blue-hour city lights"
    assert normalized["composition_prompt"].startswith("two distinct women")
    assert [region["name"] for region in normalized["regions"]] == ["Left", "Right"]
    assert [region["character_prompt"] for region in normalized["regions"]] == [
        "grey hair, aqua eyes, holding a star chart",
        "black hair, purple eyes, pointing upward",
    ]

    with pytest.raises(ValueError, match="background_prompt"):
        validate_multi_char_layout(
            _layout(),
            ["Left", "Right"],
            require_prompt_separation=True,
        )

    background_only = _layout()
    background_only["background_prompt"] = "wide shot, rooftop"
    with pytest.raises(ValueError, match="composition_prompt"):
        validate_multi_char_layout(
            background_only,
            ["Left", "Right"],
            require_prompt_separation=True,
        )


def test_layout_accepts_spatial_only_regions_when_character_prompts_are_server_owned():
    layout = _layout()
    layout["background_prompt"] = "wide shot, rooftop, blue-hour city lights"
    layout["composition_prompt"] = "two separate figures standing side by side"

    normalized = validate_multi_char_layout(
        layout,
        ["Left", "Right"],
        require_prompt_separation=True,
        require_character_prompt=False,
        max_pairwise_overlap_ratio=0.60,
    )

    assert normalized["character_order"] == ["Left", "Right"]
    assert [region["character_prompt"] for region in normalized["regions"]] == ["", ""]


def test_layout_rejects_regions_over_pairwise_overlap_limit():
    layout = {
        "background_prompt": "wide shot, rooftop",
        "composition_prompt": "two separate figures",
        "regions": [
            {"name": "Left", "x": 0.0, "y": 0.0, "width": 0.8, "height": 1.0},
            {"name": "Right", "x": 0.1, "y": 0.0, "width": 0.8, "height": 1.0},
        ],
    }

    with pytest.raises(ValueError, match="overlap"):
        validate_multi_char_layout(
            layout,
            ["Left", "Right"],
            require_prompt_separation=True,
            require_character_prompt=False,
            max_pairwise_overlap_ratio=0.60,
        )


def test_layout_fingerprint_changes_when_only_mask_coordinates_change():
    first = validate_multi_char_layout(_layout(), ["Left", "Right"])
    changed_layout = _layout()
    changed_layout["regions"][1]["width"] = 0.5
    second = validate_multi_char_layout(changed_layout, ["Left", "Right"])

    assert len(layout_fingerprint(first)) == 64
    assert layout_fingerprint(first) != layout_fingerprint(second)


def test_prepare_region_mask_clears_folder_only_when_called(tmp_path):
    input_dir = tmp_path / "input"
    mask_dir = input_dir / "region_mask"
    stale_dir = mask_dir / "old-folder"
    stale_dir.mkdir(parents=True)
    (mask_dir / "old.png").write_bytes(b"old")
    (stale_dir / "old.txt").write_text("old", encoding="utf-8")

    # 큐 적재 단계에는 helper를 호출하지 않으므로 기존 파일이 그대로 있다.
    assert (mask_dir / "old.png").exists()

    final_path = prepare_region_mask(
        str(input_dir),
        {
            "enable": True,
            "characters": [{"name": "Left"}, {"name": "Right"}],
            "character_order": ["Left", "Right"],
            "layout": _layout(),
            "mask_location": "region_mask",
        },
    )

    assert Path(final_path).name == "region_mask.png"
    assert [path.name for path in mask_dir.iterdir()] == ["region_mask.png"]
    with Image.open(final_path) as image:
        assert image.mode == "RGB"
        assert image.size == (1024, 1024)


def test_mask_location_cannot_escape_input_directory(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="밖"):
        resolve_mask_directory(str(input_dir), "../outside")

    with pytest.raises(ValueError, match="비어"):
        resolve_mask_directory("", "region_mask")


def test_multi_char_snapshot_and_prompt_control_block_must_match():
    snapshot = _snapshot()
    positive = _positive_for(snapshot)

    payload = extract_multi_char_prompt_payload(positive)
    validated = validate_multi_char_prompt_context(positive, snapshot)

    assert payload["char_name_list"] == ["Left", "Right"]
    assert validated == snapshot
    assert validated["layout"]["character_order"] == ["Left", "Right"]

    with pytest.raises(ValueError, match="캐릭터 순서"):
        validate_multi_char_prompt_context(
            _positive_for(snapshot, names=["Right", "Left"]),
            snapshot,
        )
    with pytest.raises(ValueError, match="지문"):
        validate_multi_char_prompt_context(
            _positive_for(snapshot, fingerprint="0" * 64),
            snapshot,
        )


def test_multi_char_snapshot_rejects_changed_coordinates_with_old_fingerprint():
    snapshot = _snapshot()
    changed = _separated_layout()
    changed["regions"][1]["width"] = 0.5

    with pytest.raises(ValueError, match="지문"):
        normalize_multi_char_snapshot({
            "enable": True,
            "character_order": ["Left", "Right"],
            "layout": changed,
            "mask_fingerprint": snapshot["mask_fingerprint"],
        })


def test_three_character_snapshot_preserves_rgb_order():
    layout = {
        "regions": [
            {"name": "Center", "x": 0.3, "y": 0.0, "width": 0.4, "height": 1.0},
            {"name": "Right", "x": 0.65, "y": 0.0, "width": 0.35, "height": 1.0},
            {"name": "Left", "x": 0.0, "y": 0.0, "width": 0.35, "height": 1.0},
        ]
    }
    snapshot = normalize_multi_char_snapshot({
        "enable": True,
        "character_order": ["Left", "Center", "Right"],
        "layout": layout,
    })
    positive = _positive_for(snapshot)

    validated = validate_multi_char_prompt_context(positive, snapshot)
    image = render_region_mask(validated["layout"], width=12, height=3)

    assert validated["character_order"] == ["Left", "Center", "Right"]
    assert [region["channel"] for region in validated["layout"]["regions"]] == [
        "R",
        "G",
        "B",
    ]
    assert image.getpixel((1, 1)) == (255, 0, 0)
    assert image.getpixel((6, 1))[1] == 255
    assert image.getpixel((10, 1)) == (0, 0, 255)


def test_legacy_snapshot_is_recovered_only_by_matching_session_fingerprint(tmp_path):
    snapshot = _snapshot()
    payload = extract_multi_char_prompt_payload(_positive_for(snapshot))
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    (session_dir / "risu_test.json").write_text(
        json.dumps({
            "items": [
                {"slot": 1, "multi_char_layout": _separated_layout()},
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    recovered = recover_multi_char_snapshot_from_sessions(
        str(session_dir),
        payload,
    )

    assert recovered["mask_fingerprint"] == snapshot["mask_fingerprint"]
    assert recovered["character_order"] == ["Left", "Right"]

    bad_payload = dict(payload)
    bad_payload["mask_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        recover_multi_char_snapshot_from_sessions(str(session_dir), bad_payload)
