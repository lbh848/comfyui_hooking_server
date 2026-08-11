from pathlib import Path

from modes import bot_lora_mode, lora_mode
from modes.lora_export_utils import format_lora_export_filename


def _create_source_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for index in range(1, count + 1):
        (directory / f"source_{index:03d}.png").write_bytes(
            f"image-{index}".encode("utf-8")
        )


def _expected_export_names(count: int) -> list[str]:
    return [f"[{index:05d}].png" for index in range(1, count + 1)]


def test_lora_export_filename_keeps_lexical_order_past_five_digits() -> None:
    indices = (1, 9, 10, 99_999, 100_000)
    names = [
        format_lora_export_filename(index, 100_000, ".png")
        for index in indices
    ]

    assert names == sorted(names)
    assert names == [
        "[000001].png",
        "[000009].png",
        "[000010].png",
        "[099999].png",
        "[100000].png",
    ]


def test_asset_lora_export_uses_zero_padded_names_for_twelve_images(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir = tmp_path / "asset-source"
    comfy_input_dir = tmp_path / "comfy-input"
    _create_source_images(source_dir, 12)
    comfy_input_dir.mkdir()

    monkeypatch.setattr(
        lora_mode,
        "_training_dir",
        lambda _character, _entry="": str(source_dir),
    )
    monkeypatch.setattr(lora_mode, "_load_lora_manage", lambda: {})

    result = lora_mode.export_training_images(
        "character",
        "entry",
        str(comfy_input_dir),
        folder_name_override="asset-export",
    )

    expected = _expected_export_names(12)
    assert result["success"] is True
    assert result["exported"] == expected
    assert sorted(path.name for path in Path(result["target_dir"]).iterdir()) == expected


def test_bot_lora_export_uses_zero_padded_names_for_twelve_images(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir = tmp_path / "bot-source"
    comfy_input_dir = tmp_path / "comfy-input"
    _create_source_images(source_dir, 12)
    comfy_input_dir.mkdir()

    monkeypatch.setattr(
        bot_lora_mode,
        "_bot_project_char_dir",
        lambda _bot, _project, _character: str(source_dir),
    )

    result = bot_lora_mode.export_bot_training_images(
        "bot",
        "project",
        "character",
        str(comfy_input_dir),
        "bot-export",
    )

    expected = _expected_export_names(12)
    assert result["success"] is True
    assert result["exported"] == expected
    assert sorted(path.name for path in Path(result["target_dir"]).iterdir()) == expected
