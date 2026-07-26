import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.asset_mode import AssetMode


asset_mode_module = importlib.import_module("modes.asset_mode")


def _write_representative(root, character, outfit, expression, filename="rep.webp"):
    expression_dir = root / character / outfit / expression
    expression_dir.mkdir(parents=True, exist_ok=True)
    (expression_dir / filename).write_bytes(b"fake-webp-data")
    (expression_dir / "_representative.json").write_text(
        json.dumps({"filename": filename}, ensure_ascii=False),
        encoding="utf-8",
    )


def _mapping(
    *,
    export_name="alice",
    outfits=None,
    expressions=None,
    order=None,
    enabled=None,
):
    return {
        "export_name": export_name,
        "outfit_mapping": outfits or {},
        "expression_mapping": expressions or {},
        "export_format": "webp",
        "export_quality": 90,
        "naming_order": order or ["character", "outfit", "expression"],
        "naming_enabled": enabled or {
            "character": True,
            "outfit": True,
            "expression": True,
        },
    }


@pytest.fixture
def mode(monkeypatch, tmp_path):
    asset_root = tmp_path / "asset"
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    backup_dir = tmp_path / "요구사항"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_BACKUP_DIR", str(backup_dir))
    return AssetMode(), asset_root, mapping_file, backup_dir


def _codes(plan):
    return {issue["code"] for issue in plan["errors"]}


def _warning_codes(plan):
    return {issue["code"] for issue in plan["warnings"]}


def test_unselected_outfit_with_missing_mapping_does_not_block_export(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "current_outfit", "smile")
    _write_representative(root, "alice", "party", "smile")
    mapping = _mapping(
        outfits={"party": "party"},
        expressions={"smile": "smile"},
    )

    plan = asset_mode.build_character_export_plan(
        "alice",
        selected_outfits=["party"],
        selected_expressions=["smile"],
        mapping_override=mapping,
    )

    assert plan["success"] is True
    assert [item["filename"] for item in plan["files"]] == ["alice_party_smile.webp"]
    assert "missing_outfit_mapping" not in _codes(plan)


def test_unselected_stale_duplicate_mapping_is_ignored(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "current_outfit", "smile")
    _write_representative(root, "alice", "party", "smile")
    mapping = _mapping(
        outfits={"current_outfit": "party", "party": "party"},
        expressions={"smile": "smile"},
    )

    plan = asset_mode.build_character_export_plan(
        "alice",
        selected_outfits=["party"],
        selected_expressions=["smile"],
        mapping_override=mapping,
    )

    assert plan["success"] is True
    assert "filename_collision" not in _codes(plan)


def test_same_word_in_different_mapping_categories_is_not_a_collision(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "uniform")
    mapping = _mapping(
        outfits={"uniform": "same"},
        expressions={"uniform": "same"},
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["uniform"], mapping
    )

    assert plan["success"] is True
    assert plan["files"][0]["filename"] == "alice_same_same.webp"


def test_disabled_outfit_block_does_not_require_outfit_mapping(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    mapping = _mapping(
        expressions={"smile": "happy"},
        enabled={"character": True, "outfit": False, "expression": True},
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile"], mapping
    )

    assert plan["success"] is True
    assert plan["files"][0]["filename"] == "alice_happy.webp"


def test_final_filename_collision_reports_every_source_pair(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    _write_representative(root, "alice", "uniform", "grin")
    mapping = _mapping(
        expressions={"smile": "happy", "grin": "happy"},
        enabled={"character": True, "outfit": False, "expression": True},
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile", "grin"], mapping
    )

    assert plan["success"] is False
    collision = next(issue for issue in plan["errors"] if issue["code"] == "filename_collision")
    assert collision["collisions"][0]["filename"] == "alice_happy.webp"
    assert collision["collisions"][0]["sources"] == [
        {"outfit": "uniform", "expression": "grin"},
        {"outfit": "uniform", "expression": "smile"},
    ]
    assert "복장/표정 블록" in collision["resolution"]


def test_case_only_final_filename_collision_is_rejected(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    _write_representative(root, "alice", "uniform", "grin")
    mapping = _mapping(
        expressions={"smile": "Happy", "grin": "happy"},
        enabled={"character": False, "outfit": False, "expression": True},
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile", "grin"], mapping
    )

    assert "filename_collision" in _codes(plan)


@pytest.mark.parametrize("unsafe", ["bad/name", "CON", "trailing.", " space"])
def test_unsafe_filename_mapping_is_rejected_with_specific_message(mode, unsafe):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    mapping = _mapping(
        expressions={"smile": unsafe},
        enabled={"character": True, "outfit": False, "expression": True},
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile"], mapping
    )

    assert "unsafe_mapping_name" in _codes(plan)
    assert any("표정 'smile'" in issue["message"] for issue in plan["errors"])


def test_empty_selection_and_missing_mapping_have_distinct_errors(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")

    empty = asset_mode.build_character_export_plan(
        "alice", [], ["smile"], _mapping(expressions={"smile": "happy"})
    )
    missing = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile"], _mapping(outfits={"uniform": "uniform"})
    )

    assert "empty_outfit_selection" in _codes(empty)
    assert "missing_expression_mapping" in _warning_codes(missing)
    assert "no_mapped_export_files" in _codes(missing)


def test_partially_mapped_selection_exports_only_filled_items(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    _write_representative(root, "alice", "uniform", "sad")
    mapping = _mapping(
        outfits={"uniform": "school"},
        expressions={"smile": "happy"},
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile", "sad"], mapping
    )

    assert plan["success"] is True
    assert [item["filename"] for item in plan["files"]] == ["alice_school_happy.webp"]
    assert "missing_expression_mapping" in _warning_codes(plan)
    warning = next(
        issue for issue in plan["warnings"]
        if issue["code"] == "missing_expression_mapping"
    )
    assert warning["details"] == ["sad"]

    result = asset_mode.export_character_zip(
        "alice",
        ["uniform"],
        ["smile", "sad"],
        mapping_override=mapping,
        export_plan=plan,
    )
    with zipfile.ZipFile(result) as archive:
        assert archive.namelist() == ["alice_school_happy.webp"]


def test_invalid_naming_order_and_non_boolean_toggle_are_rejected(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    mapping = _mapping(
        outfits={"uniform": "school"},
        expressions={"smile": "happy"},
        order=["character", "expression"],
        enabled={"character": True, "outfit": "false", "expression": True},
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile"], mapping
    )

    assert "invalid_naming_order" in _codes(plan)
    assert "invalid_naming_enabled" in _codes(plan)


def test_unhashable_naming_order_value_returns_validation_error_instead_of_crashing(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    mapping = _mapping(
        outfits={"uniform": "school"},
        expressions={"smile": "happy"},
        order=[{"bad": "block"}, {"bad": "block"}, "character"],
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile"], mapping
    )

    assert "invalid_naming_order" in _codes(plan)


def test_missing_representative_is_warning_when_other_files_can_export(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    missing_dir = root / "alice" / "uniform" / "sad"
    missing_dir.mkdir(parents=True)
    (missing_dir / "not-selected.webp").write_bytes(b"image")
    mapping = _mapping(
        outfits={"uniform": "uniform"},
        expressions={"smile": "happy", "sad": "sad"},
    )

    plan = asset_mode.build_character_export_plan(
        "alice", ["uniform"], ["smile", "sad"], mapping
    )

    assert plan["success"] is True
    assert plan["file_count"] == 1
    assert plan["warnings"][0]["code"] == "missing_representative"
    assert plan["warnings"][0]["details"] == ["uniform / sad"]


def test_export_zip_uses_draft_mapping_without_saving(mode):
    asset_mode, root, mapping_file, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    mapping = _mapping(
        export_name="hero",
        outfits={"uniform": "school"},
        expressions={"smile": "happy"},
    )

    result = asset_mode.export_character_zip(
        "alice", ["uniform"], ["smile"], mapping_override=mapping
    )

    assert isinstance(result, io.BytesIO)
    assert not mapping_file.exists()
    with zipfile.ZipFile(result) as archive:
        assert archive.namelist() == ["hero_school_happy.webp"]
        assert archive.read("hero_school_happy.webp") == b"fake-webp-data"


def test_get_export_info_does_not_delete_empty_directories(mode):
    asset_mode, root, _, _ = mode
    empty = root / "alice" / "uniform" / "empty_expression"
    empty.mkdir(parents=True)

    info = asset_mode.get_character_export_info("alice")

    assert info["outfits"] == []
    assert info["expressions"] == []
    assert empty.is_dir()


def test_representative_metadata_cannot_escape_expression_directory(mode, tmp_path):
    asset_mode, root, _, _ = mode
    expression_dir = root / "alice" / "uniform" / "smile"
    expression_dir.mkdir(parents=True)
    outside = tmp_path / "outside.webp"
    outside.write_bytes(b"outside")
    (expression_dir / "_representative.json").write_text(
        json.dumps({"filename": "../../../../outside.webp"}),
        encoding="utf-8",
    )

    plan = asset_mode.build_character_export_plan(
        "alice",
        ["uniform"],
        ["smile"],
        _mapping(outfits={"uniform": "school"}, expressions={"smile": "happy"}),
    )

    assert plan["success"] is False
    assert "no_exportable_representatives" in _codes(plan)
    assert plan["warnings"][0]["code"] == "missing_representative"


def test_mapping_save_backs_up_existing_json_before_utf8_write(mode):
    asset_mode, _, mapping_file, backup_dir = mode
    mapping_file.parent.mkdir(parents=True)
    mapping_file.write_text(
        json.dumps({"alice": {"export_name": "old"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    asset_mode.save_character_name_mapping(
        "alice",
        "새이름",
        {"uniform": "교복"},
        {"smile": "미소"},
    )

    backups = list(backup_dir.glob("name_mapping_*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8"))["alice"]["export_name"] == "old"
    saved = json.loads(mapping_file.read_text(encoding="utf-8"))
    assert saved["alice"]["export_name"] == "새이름"


def test_invalid_mapping_save_is_rejected_without_overwriting_existing_json(mode):
    asset_mode, _, mapping_file, backup_dir = mode
    original = {"alice": {"export_name": "old"}}
    mapping_file.parent.mkdir(parents=True)
    mapping_file.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError, match="금지 문자"):
        asset_mode.save_character_name_mapping(
            "alice",
            "new",
            {"uniform": "bad/name"},
            {"smile": "happy"},
        )

    assert json.loads(mapping_file.read_text(encoding="utf-8")) == original
    assert list(backup_dir.glob("*.json")) == []


def test_backup_failure_stops_mapping_overwrite(mode, monkeypatch):
    asset_mode, _, mapping_file, _ = mode
    original = {"alice": {"export_name": "old"}}
    mapping_file.parent.mkdir(parents=True)
    mapping_file.write_text(json.dumps(original), encoding="utf-8")

    def fail_backup(_source, _target):
        raise OSError("backup unavailable")

    monkeypatch.setattr(asset_mode_module.shutil, "copy2", fail_backup)

    with pytest.raises(RuntimeError, match="백업에 실패"):
        asset_mode.save_character_name_mapping(
            "alice",
            "new",
            {"uniform": "school"},
            {"smile": "happy"},
        )

    assert json.loads(mapping_file.read_text(encoding="utf-8")) == original


def test_public_plan_does_not_expose_local_image_paths(mode):
    asset_mode, root, _, _ = mode
    _write_representative(root, "alice", "uniform", "smile")
    plan = asset_mode.build_character_export_plan(
        "alice",
        ["uniform"],
        ["smile"],
        _mapping(outfits={"uniform": "school"}, expressions={"smile": "happy"}),
    )

    public = asset_mode.public_export_plan(plan)

    assert public["success"] is True
    assert "image_path" not in public["files"][0]
    assert "source_filename" not in public["files"][0]
