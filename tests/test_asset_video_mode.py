import importlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from modes.asset_mode import AssetMode


asset_mode_module = importlib.import_module("modes.asset_mode")


def _animated_webp_bytes() -> bytes:
    output = io.BytesIO()
    frames = [
        Image.new("RGB", (24, 24), "red"),
        Image.new("RGB", (24, 24), "blue"),
    ]
    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        lossless=True,
    )
    return output.getvalue()


def _static_source(root: Path) -> tuple[Path, dict]:
    expression_dir = root / "alice" / "uniform" / "smile"
    expression_dir.mkdir(parents=True)
    source_path = expression_dir / "source.webp"
    Image.new("RGB", (32, 32), "green").save(source_path, format="WEBP")
    (expression_dir / "source_prompt.json").write_text(
        json.dumps(
            {
                "positive": "1girl, green coat",
                "negative": "",
                "character": "alice",
                "outfit": "uniform",
                "expression": "smile",
            }
        ),
        encoding="utf-8",
    )
    (expression_dir / "_representative.json").write_text(
        json.dumps({"filename": "source.webp"}),
        encoding="utf-8",
    )
    reference = {
        "kind": "asset",
        "character": "alice",
        "outfit": "uniform",
        "expression": "smile",
        "filename": "source.webp",
    }
    return source_path, reference


def _mapping(export_format="png", export_quality=12):
    return {
        "export_name": "hero",
        "outfit_mapping": {"uniform": "school"},
        "expression_mapping": {"smile": "happy"},
        "export_format": export_format,
        "export_quality": export_quality,
        "naming_order": ["character", "outfit", "expression"],
        "naming_enabled": {
            "character": True,
            "outfit": True,
            "expression": True,
        },
    }


def test_asset_video_commit_adds_new_file_without_touching_source_or_representative(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_root = tmp_path / "asset"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()
    source_path, reference = _static_source(asset_root)
    source_before = source_path.read_bytes()

    staging = tmp_path / "staging"
    staging.mkdir()
    main_path = staging / "main.webp"
    raw_path = staging / "raw.webp"
    animated = _animated_webp_bytes()
    main_path.write_bytes(animated)
    raw_path.write_bytes(animated)

    result = mode.commit_video_result(
        reference,
        str(main_path),
        str(raw_path),
        ".webp",
        {
            "base_name": "20260813_120000_newvideo",
            "positive": "official H3 prompt",
            "instruction": "move gently",
            "instruction_source": "llm",
            "auto_instruction": True,
            "visual_context": "visual_context:\nAlice stands by a window.",
            "visual_context_source": "prompt",
            "llm_trace": ["video-trace-1"],
            "mode": "i2v",
            "last_ref": {},
            "duration": 5,
            "fps": 24,
            "video_seed": 123,
            "output_width": 512,
            "output_height": 512,
            "upscale_enabled": True,
            "upscale_scale": 2,
            "upscale_model": "realesr-animevideov3",
            "output_format": "webp",
        },
    )

    expression_dir = source_path.parent
    output_path = expression_dir / result["filename"]
    assert source_path.read_bytes() == source_before
    assert output_path.read_bytes() == animated
    assert (expression_dir / "_raw" / result["filename"]).read_bytes() == animated
    assert json.loads(
        (expression_dir / "_representative.json").read_text(encoding="utf-8")
    ) == {"filename": "source.webp"}

    listing = mode.list_images("alice", "uniform", "smile")
    output_record = next(
        item for item in listing["images"] if item["filename"] == result["filename"]
    )
    assert output_record["is_animated"] is True
    assert output_record["is_video_animation"] is True
    assert output_record["is_representative"] is False
    assert output_record["video_instruction"] == "move gently"
    assert output_record["video_instruction_source"] == "llm"
    assert output_record["video_auto_instruction"] is True
    assert output_record["video_visual_context"] == "visual_context:\nAlice stands by a window."
    assert output_record["video_visual_context_source"] == "prompt"
    prompt = json.loads(
        (expression_dir / "20260813_120000_newvideo_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    assert prompt["video_source_filename"] == "source.webp"
    assert prompt["video_mode"] == "i2v"
    assert prompt["video_instruction"] == "move gently"
    assert prompt["video_instruction_source"] == "llm"
    assert prompt["video_auto_instruction"] is True
    assert prompt["video_visual_context"] == "visual_context:\nAlice stands by a window."
    assert prompt["video_visual_context_source"] == "prompt"
    assert prompt["llm_trace"] == ["video-trace-1"]


def test_representative_animation_is_playable_and_exported_verbatim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_root = tmp_path / "asset"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()
    source_path, reference = _static_source(asset_root)
    expression_dir = source_path.parent
    animation_path = expression_dir / "animation.webp"
    animation_bytes = _animated_webp_bytes()
    animation_path.write_bytes(animation_bytes)
    (expression_dir / "animation_prompt.json").write_text(
        json.dumps(
            {
                "positive": "official H3 prompt",
                "character": "alice",
                "outfit": "uniform",
                "expression": "smile",
                "is_video_animation": True,
                "video_source_ref": reference,
            }
        ),
        encoding="utf-8",
    )
    mode.set_representative("alice", "uniform", "smile", "animation.webp")

    gallery = mode.list_character_gallery("alice")
    assert gallery[0]["representative"] == "animation.webp"
    assert gallery[0]["representative_is_animated"] is True
    poster_path = Path(mode.get_video_poster_path({**reference, "filename": "animation.webp"}))
    assert poster_path.is_file()
    with Image.open(poster_path) as poster:
        assert getattr(poster, "is_animated", False) is False

    plan = mode.build_character_export_plan(
        "alice",
        ["uniform"],
        ["smile"],
        _mapping(export_format="png", export_quality=12),
    )
    assert plan["success"] is True
    assert plan["files"][0]["filename"] == "hero_school_happy.webp"
    assert plan["files"][0]["is_video_animation"] is True

    archive_buffer = mode.export_character_zip(
        "alice",
        ["uniform"],
        ["smile"],
        mapping_override=_mapping(export_format="png", export_quality=12),
        export_plan=plan,
    )
    with zipfile.ZipFile(archive_buffer) as archive:
        assert archive.namelist() == ["hero_school_happy.webp"]
        assert archive.read("hero_school_happy.webp") == animation_bytes


def test_asset_video_reference_list_includes_static_and_animated_media(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_root = tmp_path / "asset"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()
    source_path, reference = _static_source(asset_root)
    animation_path = source_path.parent / "animation.webp"
    animation_path.write_bytes(_animated_webp_bytes())
    (source_path.parent / "animation_prompt.json").write_text(
        json.dumps({"is_video_animation": True}),
        encoding="utf-8",
    )

    references = mode.list_video_references("alice")
    by_filename = {
        item["reference"]["filename"]: item for item in references
    }
    assert by_filename["source.webp"]["is_animated"] is False
    assert by_filename["animation.webp"]["is_animated"] is True
    assert mode.resolve_video_reference(reference)["positive"] == "1girl, green coat"
