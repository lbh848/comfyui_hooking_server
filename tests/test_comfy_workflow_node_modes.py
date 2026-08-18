from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_installer.service import ComfyInstallerService
from comfy_installer.workflow_node_modes import (
    PATCH_SAGE_ATTENTION_NODE_TYPE,
    WORKFLOW_MODE_ACTIVE,
    WORKFLOW_MODE_BYPASS,
    WorkflowNodeModeError,
    set_patch_sage_attention_enabled,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_service_bulk_toggles_patch_sage_attention_with_verified_backups(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    service = ComfyInstallerService(
        project_root=tmp_path,
        config_path=config,
        requirements_dir=tmp_path / "requirements",
    )
    workflow_root = (
        tmp_path
        / "comfy"
        / "user"
        / "default"
        / "workflows"
        / "SOYA_USER"
    )
    nested_root = workflow_root / "nested"
    nested_root.mkdir(parents=True)
    first = workflow_root / "이름과_무관.json"
    second = nested_root / "custom-title.json"
    unrelated = workflow_root / "unrelated.json"
    first.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 1,
                        "type": PATCH_SAGE_ATTENTION_NODE_TYPE,
                        "title": "사용자가 바꾼 제목",
                        "mode": WORKFLOW_MODE_ACTIVE,
                    },
                    {"id": 2, "type": "OtherNode", "mode": 0},
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 3,
                        "type": PATCH_SAGE_ATTENTION_NODE_TYPE,
                        "mode": WORKFLOW_MODE_BYPASS,
                    }
                ],
                "definitions": {
                    "subgraphs": [
                        {
                            "nodes": [
                                {
                                    "id": 4,
                                    "type": PATCH_SAGE_ATTENTION_NODE_TYPE,
                                    "mode": WORKFLOW_MODE_ACTIVE,
                                }
                            ]
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    unrelated.write_text('{"nodes":[{"id":5,"type":"OtherNode"}]}', encoding="utf-8")

    disabled = service.set_patch_sage_attention_enabled(False)

    assert disabled["scanned_files"] == 3
    assert disabled["matched_files"] == 2
    assert disabled["matched_nodes"] == 3
    assert disabled["changed_files"] == 2
    assert disabled["changed_nodes"] == 2
    assert _read_json(first)["nodes"][0]["mode"] == WORKFLOW_MODE_BYPASS
    second_data = _read_json(second)
    assert second_data["nodes"][0]["mode"] == WORKFLOW_MODE_BYPASS
    assert (
        second_data["definitions"]["subgraphs"][0]["nodes"][0]["mode"]
        == WORKFLOW_MODE_BYPASS
    )

    backup = Path(disabled["backup_path"])
    assert backup.is_dir()
    assert backup.is_relative_to(
        tmp_path / "backups" / "comfy_workflows" / "patch_sage_attention"
    )
    assert _read_json(backup / first.relative_to(workflow_root))["nodes"][0][
        "mode"
    ] == WORKFLOW_MODE_ACTIVE
    assert (
        _read_json(backup / second.relative_to(workflow_root))["definitions"]
        ["subgraphs"][0]["nodes"][0]["mode"]
        == WORKFLOW_MODE_ACTIVE
    )
    assert (backup / "manifest.json").is_file()
    assert not (backup / unrelated.relative_to(workflow_root)).exists()

    enabled = service.set_patch_sage_attention_enabled(True)

    assert enabled["matched_nodes"] == 3
    assert enabled["changed_files"] == 2
    assert enabled["changed_nodes"] == 3
    assert _read_json(first)["nodes"][0]["mode"] == WORKFLOW_MODE_ACTIVE
    second_data = _read_json(second)
    assert second_data["nodes"][0]["mode"] == WORKFLOW_MODE_ACTIVE
    assert (
        second_data["definitions"]["subgraphs"][0]["nodes"][0]["mode"]
        == WORKFLOW_MODE_ACTIVE
    )

    unchanged = service.set_patch_sage_attention_enabled(True)

    assert unchanged["matched_nodes"] == 3
    assert unchanged["changed_files"] == 0
    assert unchanged["changed_nodes"] == 0
    assert unchanged["backup_path"] is None


def test_bulk_mode_change_parses_every_workflow_before_writing(
    tmp_path: Path,
) -> None:
    workflow_root = tmp_path / "SOYA_USER"
    workflow_root.mkdir()
    valid = workflow_root / "a-valid.json"
    invalid = workflow_root / "z-invalid.json"
    original = json.dumps(
        {
            "nodes": [
                {
                    "id": 1,
                    "type": PATCH_SAGE_ATTENTION_NODE_TYPE,
                    "mode": WORKFLOW_MODE_ACTIVE,
                }
            ]
        },
        separators=(",", ":"),
    )
    valid.write_text(original, encoding="utf-8")
    invalid.write_text("{broken", encoding="utf-8")

    with pytest.raises(WorkflowNodeModeError):
        set_patch_sage_attention_enabled(
            workflow_root=workflow_root,
            backup_root=tmp_path / "backups",
            enabled=False,
        )

    assert valid.read_text(encoding="utf-8") == original
    assert not (tmp_path / "backups").exists()
