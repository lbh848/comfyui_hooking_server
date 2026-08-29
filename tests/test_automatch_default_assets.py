import json
import importlib
import asyncio
import base64
import os

from modes.asset_mode import AUTOMATCH_DEFAULT_OUTFIT_DIR, AssetMode


asset_mode_module = importlib.import_module("modes.asset_mode")


def _write_image(root, character, outfit, expression, filename, *, representative=False):
    image_dir = root / character / outfit / expression
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / filename).write_bytes(b"test-image")
    if representative:
        (image_dir / "_representative.json").write_text(
            json.dumps({"filename": filename}),
            encoding="utf-8",
        )


def _mode_with_expressions(monkeypatch, root):
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(root))
    mode = AssetMode()
    mode._tags = {"expressions": {"smile": [], "sad": [], "angry": []}}
    return mode


def test_automatch_compare_prefers_selected_outfit_direct_image(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "smile", "default.webp", representative=True)
    _write_image(tmp_path, "alice", "casual", "smile", "other.webp", representative=True)
    _write_image(tmp_path, "alice", "uniform", "smile", "direct.webp", representative=True)

    result = mode.list_automatch_compare_images("alice", "uniform", include_existing=True)

    assert result["success"] is True
    assert result["images"]["smile"] == {
        "source": "direct",
        "outfit": "uniform",
        "expression": "smile",
        "filename": "direct.webp",
    }


def test_automatch_compare_falls_back_to_separate_default_bucket(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "sad", "fallback.webp")

    result = mode.list_automatch_compare_images("alice", "uniform")

    assert result["images"]["sad"]["source"] == "generated_default"
    assert result["images"]["sad"]["outfit"] == AUTOMATCH_DEFAULT_OUTFIT_DIR
    assert "smile" not in result["images"]
    assert "angry" not in result["images"]


def test_automatch_compare_can_use_same_expression_from_other_outfit(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", "z_uniform", "smile", "z.webp", representative=True)
    _write_image(tmp_path, "alice", "a_casual", "smile", "a.webp", representative=True)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "smile", "default.webp", representative=True)

    result = mode.list_automatch_compare_images("alice", "missing_outfit", include_existing=True)

    assert result["images"]["smile"] == {
        "source": "existing_asset",
        "outfit": "a_casual",
        "expression": "smile",
        "filename": "a.webp",
    }


def test_automatch_compare_does_not_scan_other_outfits_when_option_is_off(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", "casual", "sad", "existing.webp", representative=True)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "sad", "default.webp", representative=True)

    result = mode.list_automatch_compare_images("alice", "uniform", include_existing=False)

    assert result["images"]["sad"]["source"] == "generated_default"
    assert result["images"]["sad"]["filename"] == "default.webp"


def test_automatch_compare_ignores_stale_representative_file(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    image_dir = tmp_path / "alice" / "uniform" / "smile"
    image_dir.mkdir(parents=True)
    (image_dir / "actual.webp").write_bytes(b"test-image")
    (image_dir / "_representative.json").write_text(
        json.dumps({"filename": "missing.webp"}),
        encoding="utf-8",
    )

    result = mode.list_automatch_compare_images("alice", "uniform")

    assert result["images"]["smile"]["filename"] == "actual.webp"


def test_automatch_compare_rejects_empty_character(monkeypatch, tmp_path, capsys):
    mode = _mode_with_expressions(monkeypatch, tmp_path)

    result = mode.list_automatch_compare_images("", "uniform")

    assert result["success"] is False
    assert result["images"] == {}
    assert "character가 비어있음" in capsys.readouterr().out


def test_generation_saves_automatch_image_in_separate_bucket(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    async def update_workflow():
        mode._asset_api_workflow = {}
        return True

    async def submit_workflow(_workflow, progress_callback=None):
        return png_bytes, None

    monkeypatch.setattr(mode, "update_asset_workflow", update_workflow)
    monkeypatch.setattr(mode, "_save_cached_api", lambda _workflow: None)
    monkeypatch.setattr(mode, "_log", lambda *_args, **_kwargs: None)
    mode.build_prompt_with_workflow_func = lambda _workflow, _positive, _negative: {}
    mode.submit_workflow_func = submit_workflow

    result = asyncio.run(
        mode.generate(
            character="alice",
            appearance="default look",
            outfit="uniform",
            expression="smile",
            positive_prompt="smile",
            negative_prompt="",
            storage_group="automatch_defaults",
        )
    )

    assert result["success"] is True
    assert result["outfit"] == "uniform"
    assert result["storage_outfit"] == AUTOMATCH_DEFAULT_OUTFIT_DIR
    saved = tmp_path / "alice" / AUTOMATCH_DEFAULT_OUTFIT_DIR / "smile" / result["filename"]
    assert saved.is_file()
    assert not (tmp_path / "alice" / "uniform" / "smile" / result["filename"]).exists()


def test_generation_rejects_unknown_storage_group_before_workflow(monkeypatch, tmp_path, capsys):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    calls = {"workflow": 0}

    async def update_workflow():
        calls["workflow"] += 1
        return True

    monkeypatch.setattr(mode, "update_asset_workflow", update_workflow)

    result = asyncio.run(
        mode.generate(
            character="alice",
            outfit="uniform",
            expression="smile",
            storage_group="unknown",
        )
    )

    assert result["success"] is False
    assert calls["workflow"] == 0
    assert "지원하지 않는 에셋 저장 분류" in capsys.readouterr().out


def test_asset_gallery_hides_automatch_default_bucket(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "smile", "default.webp")
    _write_image(tmp_path, "alice", "uniform", "smile", "normal.webp")

    gallery = mode.list_character_gallery("alice")

    assert gallery == [{
        "outfit": "uniform",
        "expression": "smile",
        "representative": "",
        "image_count": 1,
        "local_path": "",
        "modified_at": (tmp_path / "alice" / "uniform" / "smile" / "normal.webp").stat().st_mtime,
    }]


def test_asset_gallery_modified_at_uses_the_newest_image_in_each_combination(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", "uniform", "smile", "older.webp")
    _write_image(tmp_path, "alice", "uniform", "smile", "newer.webp")
    older = tmp_path / "alice" / "uniform" / "smile" / "older.webp"
    newer = tmp_path / "alice" / "uniform" / "smile" / "newer.webp"
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    gallery = mode.list_character_gallery("alice")

    assert gallery[0]["modified_at"] == 200


def test_automatch_manager_lists_all_default_images(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "smile", "first.webp")
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "smile", "second.png")
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "sad", "sad.webp")
    _write_image(tmp_path, "alice", "uniform", "angry", "normal.webp")

    result = mode.list_automatch_default_images("alice")

    assert result["success"] is True
    assert result["image_count"] == 3
    assert result["expression_count"] == 2
    assert {
        (item["expression"], item["filename"])
        for item in result["images"]
    } == {
        ("smile", "first.webp"),
        ("smile", "second.png"),
        ("sad", "sad.webp"),
    }
    assert all(item["modified_at"] > 0 for item in result["images"])


def test_automatch_manager_delete_is_limited_to_default_bucket(monkeypatch, tmp_path):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "smile", "same.webp")
    _write_image(tmp_path, "alice", "uniform", "smile", "same.webp")
    prompt_path = (
        tmp_path / "alice" / AUTOMATCH_DEFAULT_OUTFIT_DIR / "smile" / "same_prompt.json"
    )
    prompt_path.write_text("{}", encoding="utf-8")

    result = mode.delete_automatch_default_image("alice", "smile", "same.webp")

    assert result["success"] is True
    assert not (
        tmp_path / "alice" / AUTOMATCH_DEFAULT_OUTFIT_DIR / "smile" / "same.webp"
    ).exists()
    assert not prompt_path.exists()
    assert (tmp_path / "alice" / "uniform" / "smile" / "same.webp").is_file()


def test_automatch_manager_rejects_unsafe_delete_filename(monkeypatch, tmp_path, capsys):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "smile", "safe.webp")

    result = mode.delete_automatch_default_image("alice", "smile", "../safe.webp")

    assert result["success"] is False
    assert "유효하지 않은 파일명" in result["error"]
    assert (tmp_path / "alice" / AUTOMATCH_DEFAULT_OUTFIT_DIR / "smile" / "safe.webp").is_file()
    assert "안전하지 않은 파일명" in capsys.readouterr().out


def test_automatch_manager_rejects_storage_path_escape(monkeypatch, tmp_path, capsys):
    mode = _mode_with_expressions(monkeypatch, tmp_path)
    _write_image(tmp_path, "alice", AUTOMATCH_DEFAULT_OUTFIT_DIR, "smile", "safe.webp")

    delete_result = mode.delete_automatch_default_image("alice", "..", "safe.webp")
    list_result = mode.list_automatch_default_images("..")

    assert delete_result["success"] is False
    assert list_result["success"] is False
    assert (tmp_path / "alice" / AUTOMATCH_DEFAULT_OUTFIT_DIR / "smile" / "safe.webp").is_file()
    output = capsys.readouterr().out
    assert "이미지 삭제 거부" in output
    assert "이미지 목록 조회 거부" in output
