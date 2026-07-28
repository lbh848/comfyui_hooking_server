import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modes import qwen_composite_items


def _transparent_item_bytes(size=(24, 20)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 4, 14, 13), fill=(220, 40, 70, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _opaque_item_bytes(size=(12, 8)):
    output = io.BytesIO()
    Image.new("RGBA", size, (30, 60, 90, 255)).save(output, format="PNG")
    return output.getvalue()


def test_file_library_saves_trimmed_png_lists_and_soft_deletes(tmp_path):
    item_dir = tmp_path / "qwen_composite_items"

    saved = qwen_composite_items.save_item(
        _transparent_item_bytes(),
        "검은 리본",
        str(item_dir),
    )

    assert saved["name"] == "검은 리본"
    assert saved["width"] == 10
    assert saved["height"] == 10
    saved_path = item_dir / saved["filename"]
    assert saved_path.is_file()
    with Image.open(saved_path) as image:
        assert image.mode == "RGBA"
        assert image.size == (10, 10)

    listed = qwen_composite_items.list_items(str(item_dir))
    assert [item["filename"] for item in listed] == [saved["filename"]]

    deleted = qwen_composite_items.trash_item(
        saved["filename"],
        str(item_dir),
    )
    assert deleted["recoverable"] is True
    assert not saved_path.exists()
    assert (
        item_dir
        / qwen_composite_items.QWEN_COMPOSITE_ITEM_TRASH_DIRNAME
        / deleted["trash_filename"]
    ).is_file()
    assert qwen_composite_items.list_items(str(item_dir)) == []


def test_file_library_rejects_path_traversal(tmp_path):
    item_dir = tmp_path / "qwen_composite_items"
    item_dir.mkdir()

    with pytest.raises(ValueError, match="파일명"):
        qwen_composite_items.resolve_item_path(
            "../outside.png",
            str(item_dir),
        )


def test_background_remove_combines_onnx_mask_with_existing_alpha(
    tmp_path,
    monkeypatch,
):
    del tmp_path

    def fake_foreground(image, **_kwargs):
        assert image.size == (12, 8)
        mask = np.zeros((8, 12), dtype=np.float32)
        mask[:, 6:] = 1.0
        return mask

    monkeypatch.setattr(
        qwen_composite_items,
        "predict_foreground_mask",
        fake_foreground,
    )

    result = qwen_composite_items.remove_background(_opaque_item_bytes())

    with Image.open(io.BytesIO(result)) as image:
        alpha = np.asarray(image.getchannel("A"))
        assert not alpha[:, :6].any()
        assert np.all(alpha[:, 6:] == 255)


def test_save_rejects_fully_transparent_item(tmp_path):
    output = io.BytesIO()
    Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(output, format="PNG")

    with pytest.raises(ValueError, match="픽셀"):
        qwen_composite_items.save_item(
            output.getvalue(),
            "empty",
            str(tmp_path / "items"),
        )
