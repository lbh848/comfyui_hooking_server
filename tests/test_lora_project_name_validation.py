import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import bot_lora_mode, instance_lora_mode, lora_mode, style_lora_mode
from modes.lora_name_validation import validate_lora_project_name


@pytest.mark.parametrize("name", ["anima-v1.0_1", "애니마 1", "Guild Receptionist"])
def test_lora_project_name_accepts_directory_safe_characters(name: str) -> None:
    assert validate_lora_project_name(name) == ""


@pytest.mark.parametrize("name", ["anima-v1.0_(1)", "anima/test", "anima:test", ".", ".."])
def test_lora_project_name_rejects_characters_changed_by_safe_dirname(name: str) -> None:
    assert validate_lora_project_name(name)


def test_add_lora_entry_rejects_invalid_name_before_data_access(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("invalid names must be rejected before reading production data")

    monkeypatch.setattr(lora_mode, "_load_lora_manage", fail_if_called)

    result = lora_mode.add_lora_entry(
        "anima-v1.0_(1)",
        "Guild Receptionist",
        "Guild Receptionist_1",
        "",
    )

    assert result["success"] is False
    assert "'('" in result["error"]
    assert "')'" in result["error"]


def test_bot_project_rejects_invalid_name_before_data_access(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("invalid names must be rejected before reading production data")

    monkeypatch.setattr(bot_lora_mode, "_load_bot_lora_manage", fail_if_called)

    result = bot_lora_mode.add_project("Bot", "anima_(1)")

    assert result["success"] is False
    assert "'('" in result["error"]


def test_asset_duplicate_rejects_invalid_name_before_data_access(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("invalid names must be rejected before reading production data")

    monkeypatch.setattr(lora_mode, "_load_lora_manage", fail_if_called)

    result = lora_mode.duplicate_lora_entry(
        "Source Character",
        "Source Project",
        "Target Character",
        "anima_(1)",
        "trigger",
        "",
    )

    assert result["success"] is False
    assert "'('" in result["error"]


def test_bot_duplicate_rejects_invalid_name_before_data_access(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("invalid names must be rejected before reading production data")

    monkeypatch.setattr(bot_lora_mode, "_load_bot_lora_manage", fail_if_called)

    result = bot_lora_mode.duplicate_project("Bot", "Source Project", "anima_(1)")

    assert result["success"] is False
    assert "'('" in result["error"]


def test_style_project_rejects_invalid_name_before_data_access(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("invalid names must be rejected before reading production data")

    monkeypatch.setattr(style_lora_mode, "_load_data", fail_if_called)

    result = style_lora_mode.create_project("style_(1)")

    assert result["success"] is False
    assert "'('" in result["error"]


def test_instance_lora_keeps_trigger_separate_from_safe_storage_id(
    monkeypatch, tmp_path
) -> None:
    saved = {}
    monkeypatch.setattr(instance_lora_mode, "_load_data", lambda: {"instance_loras": {}})
    monkeypatch.setattr(instance_lora_mode, "_save_data", lambda data: saved.update(data))
    monkeypatch.setattr(
        instance_lora_mode,
        "_lora_dir",
        lambda lora_id: str(tmp_path / "local" / lora_id),
    )

    result = instance_lora_mode.create_lora("trigger_(1)")

    assert result["success"] is True
    assert re.fullmatch(r"trigger_1-[0-9a-f]{6}", result["id"])
    assert saved["instance_loras"][result["id"]]["trigger"] == "trigger_(1)"


def test_instance_lora_import_also_uses_safe_storage_id(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"test")
    saved = {}
    monkeypatch.setattr(instance_lora_mode, "_load_data", lambda: {"instance_loras": {}})
    monkeypatch.setattr(instance_lora_mode, "_save_data", lambda data: saved.update(data))
    monkeypatch.setattr(
        instance_lora_mode,
        "_lora_dir",
        lambda lora_id: str(tmp_path / "local" / lora_id),
    )

    result = instance_lora_mode.import_uploaded_lora(
        "trigger_(1)",
        "anima",
        str(source),
        source.name,
        str(tmp_path / "trained"),
    )

    assert result["success"] is True
    assert re.fullmatch(r"trigger_1-[0-9a-f]{6}", result["id"])
    assert saved["instance_loras"][result["id"]]["trigger"] == "trigger_(1)"


def test_frontend_validates_new_and_duplicated_lora_project_names() -> None:
    source = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "function validateLoraProjectName(name)" in source
    assert 'id="lora-entry-add-modal"' in source
    assert 'id="style-lora-project-add-modal"' in source
    assert "async function confirmLoraEntryAdd()" in source
    assert "async function confirmStyleLoraCreateProject()" in source
    assert "const nameError = validateLoraProjectName(name);" in source
    assert "const nameError = validateLoraProjectName(targetEntry);" in source
    assert "const nameError = validateLoraProjectName(dstName);" in source
    add_function = source.split("async function addLoraEntry()", 1)[1].split(
        "function closeLoraEntryAddModal()", 1
    )[0]
    assert "prompt(" not in add_function
    style_function = source.split("async function styleLoraCreateProject()", 1)[1].split(
        "function closeStyleLoraProjectAddModal()", 1
    )[0]
    assert "prompt(" not in style_function
