from __future__ import annotations

import json
from pathlib import Path

from comfy_installer.crypto import create_workflow_pack
from comfy_installer.manifest import load_install_manifest
from comfy_installer.workflow_library import (
    import_user_copies,
    library_status,
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
