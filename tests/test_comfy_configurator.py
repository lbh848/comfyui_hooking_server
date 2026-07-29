import json
from pathlib import Path

import pytest

from comfy_installer.configurator import (
    ConfigUpdateError,
    apply_installed_config,
    backup_current_config,
    retarget_config_to_embedded_comfy,
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


def test_v4_migration_retargets_all_nested_paths_after_verified_backup(tmp_path):
    config_path = tmp_path / "config.json"
    requirements = tmp_path / "요구사항"
    old_comfy = tmp_path / "old-comfy"
    new_comfy = tmp_path / "comfy"
    (old_comfy / "input").mkdir(parents=True)
    (old_comfy / "user" / "default" / "workflows").mkdir(parents=True)
    (new_comfy / ".git").mkdir(parents=True)
    (new_comfy / "input").mkdir(parents=True)
    original = {
        "comfy_input_dir": str(old_comfy / "input"),
        "nested": {
            "workflow": str(
                old_comfy / "user" / "default" / "workflows" / "v4.json"
            ),
            "unrelated": str(tmp_path / "elsewhere"),
        },
        "path_list": [str(old_comfy / "models" / "loras"), "keep"],
        "url": "https://example.com/ComfyUI/info",
    }
    _write_json(config_path, original)
    backup = backup_current_config(
        config_path=config_path,
        backup_dir=requirements,
        reason="comfy_v4_migrate",
    )

    result = retarget_config_to_embedded_comfy(
        config_path=config_path,
        requirements_dir=requirements,
        backup_path=backup["backup_path"],
        old_comfy_root=old_comfy,
        new_comfy_root=new_comfy,
    )

    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert json.loads(result.backup_path.read_text(encoding="utf-8")) == original
    assert updated["comfy_input_dir"] == str(new_comfy / "input")
    assert updated["nested"]["workflow"] == str(
        new_comfy / "user" / "default" / "workflows" / "v4.json"
    )
    assert updated["path_list"][0] == str(new_comfy / "models" / "loras")
    assert updated["nested"]["unrelated"] == str(tmp_path / "elsewhere")
    assert updated["url"] == "https://example.com/ComfyUI/info"
    assert result.updated_paths == (
        "$.comfy_input_dir",
        "$.nested.workflow",
        "$.path_list[0]",
    )
    assert {setting for setting, _target in result.missing_targets} == {
        "$.nested.workflow",
        "$.path_list[0]",
    }
    assert result.already_retargeted is False


def test_v4_migration_accepts_config_already_retargeted_to_embedded_comfy(
    tmp_path,
):
    config_path = tmp_path / "config.json"
    requirements = tmp_path / "요구사항"
    old_comfy = tmp_path / "old-comfy"
    new_comfy = tmp_path / "comfy"
    old_comfy.mkdir()
    (new_comfy / ".git").mkdir(parents=True)
    (new_comfy / "input").mkdir()
    original = {
        "comfy_input_dir": str(new_comfy / "input"),
        "nested": {"unrelated": str(tmp_path / "elsewhere")},
    }
    _write_json(config_path, original)
    original_bytes = config_path.read_bytes()
    backup = backup_current_config(
        config_path=config_path,
        backup_dir=requirements,
        reason="comfy_v4_migrate",
    )

    result = retarget_config_to_embedded_comfy(
        config_path=config_path,
        requirements_dir=requirements,
        backup_path=backup["backup_path"],
        old_comfy_root=old_comfy,
        new_comfy_root=new_comfy,
    )

    assert result.already_retargeted is True
    assert result.updated_paths == ()
    assert result.missing_targets == ()
    assert result.before_sha256 == result.after_sha256
    assert config_path.read_bytes() == original_bytes


def test_v4_migration_still_rejects_unrelated_config_paths(tmp_path):
    config_path = tmp_path / "config.json"
    requirements = tmp_path / "요구사항"
    old_comfy = tmp_path / "old-comfy"
    new_comfy = tmp_path / "comfy"
    old_comfy.mkdir()
    (new_comfy / ".git").mkdir(parents=True)
    _write_json(config_path, {"path": str(tmp_path / "elsewhere")})
    backup = backup_current_config(
        config_path=config_path,
        backup_dir=requirements,
        reason="comfy_v4_migrate",
    )

    with pytest.raises(ConfigUpdateError, match="기존 ComfyUI 아래의 경로"):
        retarget_config_to_embedded_comfy(
            config_path=config_path,
            requirements_dir=requirements,
            backup_path=backup["backup_path"],
            old_comfy_root=old_comfy,
            new_comfy_root=new_comfy,
        )


def test_v4_migration_rejects_stale_backup_without_overwrite(tmp_path):
    config_path = tmp_path / "config.json"
    requirements = tmp_path / "요구사항"
    old_comfy = tmp_path / "old-comfy"
    new_comfy = tmp_path / "comfy"
    old_comfy.mkdir()
    (new_comfy / ".git").mkdir(parents=True)
    _write_json(config_path, {"path": str(old_comfy / "input")})
    backup = backup_current_config(
        config_path=config_path,
        backup_dir=requirements,
        reason="comfy_v4_migrate",
    )
    changed = {"path": str(old_comfy / "changed")}
    _write_json(config_path, changed)

    with pytest.raises(ConfigUpdateError, match="동시 설정 변경"):
        retarget_config_to_embedded_comfy(
            config_path=config_path,
            requirements_dir=requirements,
            backup_path=backup["backup_path"],
            old_comfy_root=old_comfy,
            new_comfy_root=new_comfy,
        )

    assert json.loads(config_path.read_text(encoding="utf-8")) == changed


def test_restore_rejects_arbitrary_file_outside_requirements(tmp_path):
    config_path = tmp_path / "config.json"
    _write_json(config_path, {"safe": True})
    outside = tmp_path / "outside.json"
    _write_json(outside, {"safe": False})

    with pytest.raises(ConfigUpdateError, match="설정 백업 폴더 밖"):
        restore_config_backup(
            config_path=config_path,
            requirements_dir=tmp_path / "요구사항",
            backup_path=outside,
        )

    assert json.loads(config_path.read_text(encoding="utf-8")) == {"safe": True}
