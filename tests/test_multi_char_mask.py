from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.multi_char_mask import (
    layout_fingerprint,
    prepare_region_mask,
    render_region_mask,
    resolve_mask_directory,
    validate_multi_char_layout,
)


def _layout():
    return {
        "regions": [
            {"name": "Right", "x": 0.4, "y": 0.0, "width": 0.6, "height": 1.0},
            {"name": "Left", "x": 0.0, "y": 0.0, "width": 0.6, "height": 1.0},
        ]
    }


def test_layout_is_sorted_left_to_right_and_overlap_is_preserved():
    normalized = validate_multi_char_layout(_layout(), ["Left", "Right"])

    assert normalized["character_order"] == ["Left", "Right"]
    assert [region["channel"] for region in normalized["regions"]] == ["R", "G"]

    image = render_region_mask(normalized, width=10, height=4)
    assert image.getpixel((1, 1)) == (255, 0, 0)
    assert image.getpixel((5, 1)) == (255, 255, 0)
    assert image.getpixel((8, 1)) == (0, 255, 0)


def test_layout_prompt_separation_is_required_and_preserved_when_requested():
    separated = _layout()
    separated["background_prompt"] = "wide shot, rooftop, blue-hour city lights"
    separated["regions"][0]["character_prompt"] = "black hair, purple eyes, pointing upward"
    separated["regions"][1]["character_prompt"] = "grey hair, aqua eyes, holding a star chart"

    normalized = validate_multi_char_layout(
        separated,
        ["Left", "Right"],
        require_prompt_separation=True,
    )

    assert normalized["background_prompt"] == "wide shot, rooftop, blue-hour city lights"
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
