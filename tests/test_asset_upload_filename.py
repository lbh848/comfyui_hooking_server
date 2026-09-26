import pytest


@pytest.fixture
def asset_mode_module():
    return __import__("modes.asset_mode", fromlist=["AssetMode"])


@pytest.mark.parametrize(
    ("filename", "saved_filename"),
    [("_sample.png", "sample.png"), ("__portrait.webp", "portrait.webp")],
)
def test_asset_upload_removes_leading_underscores(
    monkeypatch,
    tmp_path,
    asset_mode_module,
    filename,
    saved_filename,
):
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(tmp_path))

    result = asset_mode_module.AssetMode().upload_image(
        "alice",
        "uniform",
        "smile",
        filename,
        b"image-bytes",
    )

    assert result == {"success": True, "filename": saved_filename}
    image_dir = tmp_path / "alice" / "uniform" / "smile"
    assert (image_dir / saved_filename).read_bytes() == b"image-bytes"
    assert (image_dir / f"{saved_filename.rsplit('.', 1)[0]}_prompt.json").is_file()
    assert not (image_dir / filename).exists()


def test_asset_upload_removes_reserved_prefix_after_sanitizing(
    monkeypatch,
    tmp_path,
    asset_mode_module,
):
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(tmp_path))

    result = asset_mode_module.AssetMode().upload_image(
        "alice",
        "uniform",
        "smile",
        " !_sample.png",
        b"image-bytes",
    )

    assert result == {"success": True, "filename": "sample.png"}
    assert (tmp_path / "alice" / "uniform" / "smile" / "sample.png").is_file()


def test_asset_upload_suffixes_name_when_prefix_removal_causes_collision(
    monkeypatch,
    tmp_path,
    asset_mode_module,
):
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(tmp_path))
    image_dir = tmp_path / "alice" / "uniform" / "smile"
    image_dir.mkdir(parents=True)
    original_path = image_dir / "sample.png"
    original_path.write_bytes(b"existing-image")

    result = asset_mode_module.AssetMode().upload_image(
        "alice",
        "uniform",
        "smile",
        "_sample.png",
        b"new-image",
    )

    assert result == {"success": True, "filename": "sample_1.png"}
    assert original_path.read_bytes() == b"existing-image"
    assert (image_dir / "sample_1.png").read_bytes() == b"new-image"
    assert (image_dir / "sample_1_prompt.json").is_file()


def test_asset_upload_keeps_underscore_away_from_filename_start(
    monkeypatch,
    tmp_path,
    asset_mode_module,
):
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(tmp_path))

    result = asset_mode_module.AssetMode().upload_image(
        "alice",
        "uniform",
        "smile",
        "sample_portrait.png",
        b"image-bytes",
    )

    assert result == {"success": True, "filename": "sample_portrait.png"}
    image_dir = tmp_path / "alice" / "uniform" / "smile"
    assert (image_dir / "sample_portrait.png").read_bytes() == b"image-bytes"
    assert (image_dir / "sample_portrait_prompt.json").is_file()
