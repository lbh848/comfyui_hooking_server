import json
from pathlib import Path

import pytest

from comfy_installer.crypto import (
    PACK_MAGIC,
    WorkflowPackError,
    create_workflow_pack,
    extract_workflow_pack,
)


def _write_workflow(path: Path, node_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                node_id: {
                    "class_type": "KSampler",
                    "inputs": {"seed": 1},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_workflow_pack_round_trip_preserves_bindings_and_hashes(tmp_path):
    first = tmp_path / "배포_삽화.json"
    second = tmp_path / "배포_에셋.json"
    _write_workflow(first, "1")
    _write_workflow(second, "2")
    pack = tmp_path / "workflows.soyawfp"

    created = create_workflow_pack(
        {
            "illustration_workflow_source_paths.v3": first,
            "asset_workflow_source_path": second,
            "comfy_workflow_source_path": first,
        },
        pack,
        "correct key",
    )

    assert pack.read_bytes().startswith(PACK_MAGIC)
    assert created["workflow_count"] == 2
    assert created["binding_count"] == 3
    extracted = extract_workflow_pack(pack, tmp_path / "restored", "correct key")

    assert set(extracted.workflow_bindings) == {
        "illustration_workflow_source_paths.v3",
        "asset_workflow_source_path",
        "comfy_workflow_source_path",
    }
    assert Path(
        extracted.workflow_bindings["illustration_workflow_source_paths.v3"]
    ).read_bytes() == first.read_bytes()
    assert len(extracted.workflow_hashes) == 2
    assert extracted.pack_sha256 == created["sha256"]
    assert extracted.release_version == "v1"
    assert len(extracted.workflow_items) == 2


def test_workflow_pack_records_release_version(tmp_path):
    workflow = tmp_path / "workflow.json"
    _write_workflow(workflow, "1")
    pack = tmp_path / "workflows-v2.soyawfp"

    created = create_workflow_pack(
        {"comfy_workflow_source_path": workflow},
        pack,
        "right",
        release_version="v2",
        workflow_items=[
            {
                "id": "comfy_workflow_source_path",
                "name": workflow.name,
                "archive_name": f"workflows/{workflow.name}",
                "bindings": ["comfy_workflow_source_path"],
                "model_ids": ["fixed-model-b", "fixed-model-a"],
            }
        ],
    )
    extracted = extract_workflow_pack(pack, tmp_path / "restored-v2", "right")

    assert created["release_version"] == "v2"
    assert extracted.release_version == "v2"
    assert extracted.workflow_items[0]["model_ids"] == [
        "fixed-model-a",
        "fixed-model-b",
    ]


def test_workflow_pack_rejects_wrong_key_without_writing_files(tmp_path):
    workflow = tmp_path / "workflow.json"
    _write_workflow(workflow, "1")
    pack = tmp_path / "workflows.soyawfp"
    create_workflow_pack({"comfy_workflow_source_path": workflow}, pack, "right")
    target = tmp_path / "restored"

    with pytest.raises(WorkflowPackError, match="키가 틀렸거나"):
        extract_workflow_pack(pack, target, "wrong")

    assert not target.exists()


def test_workflow_pack_rejects_tampering(tmp_path):
    workflow = tmp_path / "workflow.json"
    _write_workflow(workflow, "1")
    pack = tmp_path / "workflows.soyawfp"
    create_workflow_pack({"comfy_workflow_source_path": workflow}, pack, "right")
    tampered = bytearray(pack.read_bytes())
    tampered[-1] ^= 0x01
    pack.write_bytes(tampered)

    with pytest.raises(WorkflowPackError, match="변조"):
        extract_workflow_pack(pack, tmp_path / "restored", "right")


def test_workflow_pack_rejects_invalid_workflow_json(tmp_path):
    workflow = tmp_path / "invalid.json"
    workflow.write_text("{", encoding="utf-8")

    with pytest.raises(WorkflowPackError, match="JSON"):
        create_workflow_pack(
            {"comfy_workflow_source_path": workflow},
            tmp_path / "bad.soyawfp",
            "key",
        )
