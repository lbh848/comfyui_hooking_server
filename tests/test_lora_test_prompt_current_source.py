import json
from pathlib import Path

from modes import bot_lora_mode, lora_mode, style_lora_mode


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_bot_common_test_copy_uses_current_prompt_as_derived_original(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(bot_lora_mode, "BOT_DIR", str(tmp_path / "bot"))
    src_dir = Path(bot_lora_mode._bot_test_dir("작품", "프로젝트"))
    src_dir.mkdir(parents=True)
    (src_dir / "pose.webp").write_bytes(b"image")
    _write_json(
        src_dir / "pose_prompt.json",
        {
            "positive": "user edited pose tags",
            "negative": "user edited negative tags",
            "original_positive": "raw tagger pose tags",
            "original_negative": "raw negative tags",
            "custom": "preserved",
        },
    )

    result = bot_lora_mode.copy_project_test_to_char(
        "작품", "프로젝트", "다른 등장인물", ["pose.webp"]
    )

    assert result["success"] is True
    dst = Path(
        bot_lora_mode._bot_char_test_dir("작품", "프로젝트", "다른 등장인물")
    ) / "pose_prompt.json"
    copied = _read_json(dst)
    assert copied == {
        "positive": "user edited pose tags",
        "negative": "user edited negative tags",
        "original_positive": "user edited pose tags",
        "original_negative": "user edited negative tags",
        "custom": "preserved",
    }


def test_bot_existing_test_setup_replaces_stale_original_with_actual_input(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(bot_lora_mode, "BOT_DIR", str(tmp_path / "bot"))
    prompt_path = Path(
        bot_lora_mode._bot_char_test_dir("봇", "프로젝트", "Akane")
    ) / "1778846574_655520_prompt.json"
    _write_json(
        prompt_path,
        {
            "positive": "previous generated character tags",
            "negative": "kept negative",
            "original_positive": "stale raw source tags",
            "original_negative": "kept original negative",
        },
    )

    result = bot_lora_mode.save_bot_char_test_prompt_positive_only(
        "봇",
        "프로젝트",
        "Akane",
        "1778846574_655520.webp",
        "new generated character tags",
        source_positive="user edited common test tags",
    )

    assert result["success"] is True
    saved = _read_json(prompt_path)
    assert saved["positive"] == "new generated character tags"
    assert saved["original_positive"] == "user edited common test tags"
    assert saved["negative"] == "kept negative"
    assert saved["original_negative"] == "kept original negative"

    bot_lora_mode.save_bot_char_test_prompt_positive_only(
        "봇", "프로젝트", "Akane", "1778846574_655520.webp", "single refine result"
    )
    assert _read_json(prompt_path)["original_positive"] == "user edited common test tags"


def test_asset_existing_test_setup_records_current_input_as_original(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(lora_mode, "ASSET_DIR", str(tmp_path / "asset"))
    prompt_path = Path(lora_mode._test_dir("캐릭터", "복장")) / "pose_prompt.json"
    _write_json(
        prompt_path,
        {
            "positive": "user edited test tags",
            "negative": "negative",
            "original_positive": "raw source tags",
            "original_negative": "original negative",
        },
    )

    result = lora_mode.save_test_prompt_positive_only(
        "캐릭터",
        "복장",
        "pose.webp",
        "generated setup tags",
        source_positive="user edited test tags",
    )

    assert result["success"] is True
    saved = _read_json(prompt_path)
    assert saved["positive"] == "generated setup tags"
    assert saved["original_positive"] == "user edited test tags"
    assert saved["negative"] == "negative"
    assert saved["original_negative"] == "original negative"


def test_style_training_import_uses_current_prompt_as_test_original(
    tmp_path, monkeypatch
):
    data = {"projects": {"화풍": {"images": ["sample.webp"], "test_images": []}}}
    monkeypatch.setattr(style_lora_mode, "STYLE_LORA_DIR", str(tmp_path / "style"))
    monkeypatch.setattr(style_lora_mode, "_load_data", lambda: data)
    monkeypatch.setattr(style_lora_mode, "_save_data", lambda _data: None)
    project_dir = Path(style_lora_mode._project_dir("화풍"))
    project_dir.mkdir(parents=True)
    (project_dir / "sample.webp").write_bytes(b"image")
    _write_json(
        project_dir / "sample_prompt.json",
        {
            "positive": "user edited style tags",
            "negative": "user edited negative",
            "original_positive": "raw style tags",
            "original_negative": "raw negative",
        },
    )

    result = style_lora_mode.add_test_image_from_train("화풍", "sample.webp")

    assert result["success"] is True
    seeded = _read_json(project_dir / "sample_test_prompt.json")
    assert seeded == {
        "positive": "user edited style tags",
        "negative": "user edited negative",
        "original_positive": "user edited style tags",
        "original_negative": "user edited negative",
    }
