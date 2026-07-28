import json
from pathlib import Path

import pytest

from comfy_installer.configurator import (
    ConfigUpdateError,
    apply_installed_config,
    restore_config_backup,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_config_apply_backs_up_updates_and_restores(tmp_path):
    config_path = tmp_path / "config.json"
    requirements = tmp_path / "요구사항"
    comfy = tmp_path / "comfy"
    workflows = comfy / "user" / "default" / "workflows"
    workflows.mkdir(parents=True)
    first = workflows / "first.json"
    second = workflows / "second.json"
    _write_json(first, {"1": {"class_type": "KSampler"}})
    _write_json(second, {"2": {"class_type": "KSampler"}})
    original = {
        "comfyui_port": 8188,
        "comfy_input_dir": r"E:\old\input",
        "illustration_workflow_source_paths": {
            "v1": r"E:\old\first.json",
        },
        "outfit_mode_enabled": True,
        "outfit_workflow_source_path": r"E:\old\outfit.json",
        "unrelated": "preserve me",
    }
    _write_json(config_path, original)

    result = apply_installed_config(
        config_path=config_path,
        requirements_dir=requirements,
        comfy_root=comfy,
        workflow_bindings={
            "illustration_workflow_source_paths.v1": str(first),
            "asset_workflow_source_path": str(second),
        },
        required_bindings=[
            "illustration_workflow_source_paths.v1",
            "asset_workflow_source_path",
        ],
    )

    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert json.loads(result.backup_path.read_text(encoding="utf-8")) == original
    assert updated["unrelated"] == "preserve me"
    assert updated["illustration_workflow_source_paths"]["v1"] == str(first)
    assert updated["asset_workflow_source_path"] == str(second)
    assert updated["comfy_input_dir"] == str(comfy / "input")
    assert updated["outfit_mode_enabled"] is False
    assert updated["outfit_workflow_source_path"] == ""

    restored = restore_config_backup(
        config_path=config_path,
        requirements_dir=requirements,
        backup_path=result.backup_path,
    )

    assert json.loads(config_path.read_text(encoding="utf-8")) == original
    assert restored["safety_backup"]
    assert Path(restored["safety_backup"]).is_file()
    assert result.backup_path.is_file()


def test_config_apply_rejects_workflow_outside_installed_root(tmp_path):
    config_path = tmp_path / "config.json"
    _write_json(config_path, {})
    comfy = tmp_path / "comfy"
    (comfy / "user" / "default" / "workflows").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    _write_json(outside, {})

    with pytest.raises(ConfigUpdateError, match="설치 폴더 밖"):
        apply_installed_config(
            config_path=config_path,
            requirements_dir=tmp_path / "요구사항",
            comfy_root=comfy,
            workflow_bindings={"comfy_workflow_source_path": str(outside)},
            required_bindings=["comfy_workflow_source_path"],
        )

    assert not (tmp_path / "요구사항").exists()


def test_restore_rejects_arbitrary_file_outside_requirements(tmp_path):
    config_path = tmp_path / "config.json"
    _write_json(config_path, {"safe": True})
    outside = tmp_path / "outside.json"
    _write_json(outside, {"safe": False})

    with pytest.raises(ConfigUpdateError, match="요구사항 폴더 밖"):
        restore_config_backup(
            config_path=config_path,
            requirements_dir=tmp_path / "요구사항",
            backup_path=outside,
        )

    assert json.loads(config_path.read_text(encoding="utf-8")) == {"safe": True}
