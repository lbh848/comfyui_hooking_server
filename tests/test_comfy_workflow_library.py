from __future__ import annotations

import json
from pathlib import Path

from comfy_installer.crypto import create_workflow_pack
from comfy_installer.manifest import load_install_manifest
from comfy_installer.workflow_library import (
    DISTRIBUTION_LIBRARY_DIRNAME,
    LEGACY_DISTRIBUTION_LIBRARY_DIRNAME,
    LEGACY_USER_WORKFLOW_DIRNAME,
    USER_WORKFLOW_DIRNAME,
    import_user_copies,
    library_status,
    migrate_legacy_workflow_layout,
    selection_requirements,
    unpack_to_library,
)


def _workflow(path: Path, model_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": model_name},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_versioned_library_never_overwrites_edited_user_copy(tmp_path: Path) -> None:
    manifest = load_install_manifest()
    bindings: dict[str, Path] = {}
    workflow_items: list[dict] = []
    for index, fixed in enumerate(
        manifest.workflows["release_dependencies"]["v1"], start=1
    ):
        workflow = tmp_path / f"배포_{index:02d}.json"
        _workflow(workflow, "팩 내용으로 모델을 추측하지 않음")
        for binding in fixed["bindings"]:
            bindings[binding] = workflow
        workflow_items.append(
            {
                "id": fixed["id"],
                "name": workflow.name,
                "archive_name": f"workflows/{workflow.name}",
                "bindings": fixed["bindings"],
                "model_ids": fixed["model_ids"],
            }
        )
    pack = tmp_path / "pack-v1.soyawfp"
    create_workflow_pack(
        bindings,
        pack,
        "pack-key",
        release_version="v1",
        workflow_items=workflow_items,
    )
    library_root = tmp_path / "library"
    unpacked = unpack_to_library(
        pack_path=pack,
        passphrase="pack-key",
        library_root=library_root,
        work_root=tmp_path / "work",
        manifest=manifest,
    )
    assert Path(unpacked["directory"]).parent.name == DISTRIBUTION_LIBRARY_DIRNAME
    item_id = "qwen_edit_workflow_source_path"
    requirements = selection_requirements(
        library_root=library_root,
        release_version="v1",
        selected_item_ids=[item_id],
    )
    assert requirements["model_ids"] == ["qwen-image-edit-rapid-v19"]

    comfy = tmp_path / "comfy"
    first = import_user_copies(
        comfy_root=comfy,
        library_root=library_root,
        release_version="v1",
        selected_item_ids=[item_id],
    )
    first_path = Path(first.user_files[0])
    assert first_path.parent.name == USER_WORKFLOW_DIRNAME
    first_path.write_text('{"user":"edited"}', encoding="utf-8")

    second = import_user_copies(
        comfy_root=comfy,
        library_root=library_root,
        release_version="v1",
        selected_item_ids=[item_id],
    )
    second_path = Path(second.user_files[0])

    assert first_path.read_text(encoding="utf-8") == '{"user":"edited"}'
    assert second_path != first_path
    assert second_path.name.endswith("__v1_2.json")
    status = library_status(comfy, library_root)
    assert len(status["releases"]) == 1
    assert len(status["user_files"]) == 2
    assert Path(status["distributed_root"]).name == DISTRIBUTION_LIBRARY_DIRNAME
    assert Path(status["user_root"]).name == USER_WORKFLOW_DIRNAME


def test_legacy_korean_layout_migrates_to_ascii_without_data_loss(
    tmp_path: Path,
) -> None:
    comfy = tmp_path / "comfy"
    workflows_root = comfy / "user" / "default" / "workflows"
    legacy_user_root = workflows_root / LEGACY_USER_WORKFLOW_DIRNAME
    user_root = workflows_root / USER_WORKFLOW_DIRNAME
    legacy_user_root.mkdir(parents=True)
    user_root.mkdir(parents=True)

    legacy_workflow = legacy_user_root / "main.json"
    legacy_workflow.write_text('{"owner":"legacy-user"}', encoding="utf-8")
    (legacy_user_root / "nested").mkdir()
    legacy_nested = legacy_user_root / "nested" / "asset.json"
    legacy_nested.write_text('{"owner":"legacy-nested"}', encoding="utf-8")
    (user_root / "main.json").write_text(
        '{"owner":"new-user"}',
        encoding="utf-8",
    )

    library_root = tmp_path / "comfy_workflow_library"
    legacy_distribution = (
        library_root / LEGACY_DISTRIBUTION_LIBRARY_DIRNAME / "v1"
    )
    legacy_distribution.mkdir(parents=True)
    (legacy_distribution / ".soya-pack.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "release_version": "v1",
                "items": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (legacy_distribution / "distributed.json").write_text(
        '{"owner":"distribution"}',
        encoding="utf-8",
    )

    config_path = tmp_path / "config.json"
    original_config = {
        "comfy_workflow_source_path": str(legacy_workflow),
        "illustration_workflow_source_paths": {
            "v3": str(legacy_workflow),
            "v3_anima": str(legacy_nested),
        },
        "unrelated": "SOYA_개인 문자열은 경로가 아니므로 유지",
    }
    config_path.write_text(
        json.dumps(original_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    requirements = tmp_path / "요구사항"

    first = migrate_legacy_workflow_layout(
        comfy_root=comfy,
        library_root=library_root,
        config_path=config_path,
        backup_dir=requirements,
    )

    migrated_config = json.loads(config_path.read_text(encoding="utf-8"))
    migrated_main = Path(migrated_config["comfy_workflow_source_path"])
    migrated_nested = Path(
        migrated_config["illustration_workflow_source_paths"]["v3_anima"]
    )
    assert migrated_main.parent == user_root
    assert migrated_main.name == "main__legacy_2.json"
    assert migrated_main.read_text(encoding="utf-8") == (
        '{"owner":"legacy-user"}'
    )
    assert migrated_nested == user_root / "nested" / "asset.json"
    assert migrated_nested.read_text(encoding="utf-8") == (
        '{"owner":"legacy-nested"}'
    )
    assert migrated_config["illustration_workflow_source_paths"]["v3"] == str(
        migrated_main
    )
    assert migrated_config["unrelated"] == original_config["unrelated"]
    assert (user_root / "main.json").read_text(encoding="utf-8") == (
        '{"owner":"new-user"}'
    )
    assert not legacy_user_root.exists()

    distribution_root = library_root / DISTRIBUTION_LIBRARY_DIRNAME / "v1"
    assert (distribution_root / ".soya-pack.json").is_file()
    assert (distribution_root / "distributed.json").read_text(
        encoding="utf-8"
    ) == '{"owner":"distribution"}'
    assert not (
        library_root / LEGACY_DISTRIBUTION_LIBRARY_DIRNAME
    ).exists()
    assert first["user"]["renamed_conflicts"] == 1
    assert first["user"]["legacy_data_preserved"] is True
    assert first["distribution"]["legacy_data_preserved"] is True
    assert first["config"]["updated"] is True

    user_backup = Path(first["user"]["legacy_archive"]["backup"])
    distribution_backup = Path(
        first["distribution"]["legacy_archive"]["backup"]
    )
    assert user_backup.name == "LEGACY_SOYA_USER"
    assert (user_backup / "main.json").read_text(
        encoding="utf-8"
    ) == '{"owner":"legacy-user"}'
    assert (user_backup / "nested" / "asset.json").is_file()
    assert distribution_backup.name == "LEGACY_SOYA_DISTRIBUTION"
    assert (distribution_backup / "v1" / "distributed.json").is_file()

    backups = list(
        requirements.glob("config_before_workflow_ascii_migration_*.json")
    )
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original_config

    second = migrate_legacy_workflow_layout(
        comfy_root=comfy,
        library_root=library_root,
        config_path=config_path,
        backup_dir=requirements,
    )
    assert second["user"]["copied_files"] == 0
    assert second["distribution"]["copied_files"] == 0
    assert second["config"]["updated"] is False
    assert len(list(user_root.glob("main__legacy_*.json"))) == 1
    assert len(
        list(requirements.glob("config_before_workflow_ascii_migration_*.json"))
    ) == 1
