from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
from pathlib import Path

from comfy_installer.manifest import InstallManifest
from comfy_installer.runtime_state import (
    create_runtime_transaction,
    inspect_runtime,
    load_runtime_receipt,
    rollback_runtime_transaction,
    write_runtime_receipt,
)


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _init_repo(path: Path, filename: str = "tracked.txt") -> str:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.email", "runtime-state@example.invalid")
    _git(path, "config", "user.name", "Runtime State Test")
    (path / filename).parent.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text("first\n", encoding="utf-8")
    _git(path, "add", filename)
    _git(path, "commit", "-m", "first")
    return _git(path, "rev-parse", "HEAD").lower()


def _commit(path: Path, filename: str, content: str) -> str:
    (path / filename).write_text(content, encoding="utf-8")
    _git(path, "add", filename)
    _git(path, "commit", "-m", content.strip())
    return _git(path, "rev-parse", "HEAD").lower()


def _manifest(
    tmp_path: Path,
    *,
    comfy_ref: str,
    tracking_repository: str,
) -> InstallManifest:
    data = {
        "comfy": {
            "version": "0.31.0",
            "repository": "https://example.invalid/ComfyUI.git",
            "ref": comfy_ref,
        },
        "python": {
            "version": "3.12.11",
            "compatibility_packages": [],
            "gpu_profiles": [],
        },
        "custom_nodes": [
            {
                "name": "tracking-node",
                "source_type": "git",
                "repository": tracking_repository,
                "tracking_branch": "main",
            },
            {
                "name": "archive-node",
                "source_type": "archive",
                "url": "https://example.invalid/archive.zip",
                "sha256": "a" * 64,
                "size": 1,
            },
        ],
        "models": [],
        "workflows": {},
    }
    return InstallManifest(
        source_path=tmp_path / "manifest.json",
        data=data,
        sha256="manifest-sha",
    )


def _prepare_runtime(tmp_path: Path):
    comfy = tmp_path / "comfy"
    comfy_ref = _init_repo(comfy)
    remote = tmp_path / "tracking-remote"
    tracking_ref = _init_repo(remote)
    _git(remote, "branch", "-M", "main")
    node = comfy / "custom_nodes" / "tracking-node"
    node.parent.mkdir(parents=True)
    _git(node.parent, "clone", str(remote), str(node))
    _git(node, "config", "user.email", "runtime-state@example.invalid")
    _git(node, "config", "user.name", "Runtime State Test")
    assert _git(node, "rev-parse", "HEAD").lower() == tracking_ref
    archive = comfy / "custom_nodes" / "archive-node"
    archive.mkdir()
    (archive / ".comfy-installer-source.json").write_text(
        json.dumps(
            {
                "source_type": "archive",
                "url": "https://example.invalid/archive.zip",
                "sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    venv = comfy / ".venv"
    venv.mkdir()
    (venv / "marker.txt").write_text("old-venv", encoding="utf-8")
    manifest = _manifest(
        tmp_path,
        comfy_ref=comfy_ref,
        tracking_repository=str(remote),
    )
    return comfy, remote, node, manifest


def test_receipt_changes_inventory_from_unverified_to_current(tmp_path: Path) -> None:
    comfy, _remote, _node, manifest = _prepare_runtime(tmp_path)
    first = inspect_runtime(
        comfy_root=comfy,
        manifest=manifest,
        profile_id="nvidia-cu130",
        install_mode="standard",
    )
    assert first["runtime_changed"] is True
    assert "missing_receipt" in first["runtime_change_reasons"]

    write_runtime_receipt(
        comfy_root=comfy,
        manifest=manifest,
        profile_id="nvidia-cu130",
        install_mode="standard",
        workflow_bindings={"workflow": str(tmp_path / "workflow.json")},
        selected_workflow_ids=["workflow"],
        release_version="v1",
    )
    current = inspect_runtime(
        comfy_root=comfy,
        manifest=manifest,
        profile_id="nvidia-cu130",
        install_mode="standard",
    )
    assert current["runtime_changed"] is False
    assert current["runtime_change_reasons"] == []


def test_tracking_main_change_requires_runtime_e2e(tmp_path: Path) -> None:
    comfy, remote, _node, manifest = _prepare_runtime(tmp_path)
    write_runtime_receipt(
        comfy_root=comfy,
        manifest=manifest,
        profile_id="nvidia-cu130",
        install_mode="standard",
        workflow_bindings={},
        selected_workflow_ids=[],
        release_version="v1",
    )
    _commit(remote, "tracked.txt", "second\n")
    inventory = inspect_runtime(
        comfy_root=comfy,
        manifest=manifest,
        profile_id="nvidia-cu130",
        install_mode="standard",
    )
    assert "custom_node_tracking:tracking-node" in inventory[
        "runtime_change_reasons"
    ]


def test_transaction_restores_git_venv_receipt_and_config(tmp_path: Path) -> None:
    comfy, _remote, node, manifest = _prepare_runtime(tmp_path)
    database = comfy / "user" / "comfyui.db"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute("create table marker (value text not null)")
        connection.execute("insert into marker values ('before-update')")
        connection.commit()
    finally:
        connection.close()
    manifest.data["custom_nodes"].append(
        {
            "name": "new-fixed-node",
            "source_type": "git",
            "repository": "https://example.invalid/new-fixed-node.git",
            "ref": "b" * 40,
        }
    )
    old_receipt = write_runtime_receipt(
        comfy_root=comfy,
        manifest=manifest,
        profile_id="nvidia-cu130",
        install_mode="standard",
        workflow_bindings={},
        selected_workflow_ids=[],
        release_version="v1",
    )
    snapshot = create_runtime_transaction(
        comfy_root=comfy,
        manifest=manifest,
        config_backup={"backup_path": str(tmp_path / "config.backup.json")},
    )
    newly_installed = comfy / "custom_nodes" / "new-fixed-node"
    newly_installed.mkdir()
    read_only_file = newly_installed / "installed.txt"
    read_only_file.write_text("new", encoding="utf-8")
    os.chmod(read_only_file, stat.S_IREAD)
    comfy_new_ref = _commit(comfy, "tracked.txt", "new-comfy\n")
    node_new_ref = _commit(node, "tracked.txt", "new-node\n")
    assert comfy_new_ref != snapshot["comfy_ref"]
    assert node_new_ref != snapshot["custom_nodes"]["tracking-node"]["head"]
    (comfy / ".venv" / "marker.txt").write_text("new-venv", encoding="utf-8")
    connection = sqlite3.connect(database)
    try:
        connection.execute("update marker set value = 'after-update'")
        connection.commit()
    finally:
        connection.close()
    write_runtime_receipt(
        comfy_root=comfy,
        manifest=manifest,
        profile_id="changed",
        install_mode="standard",
        workflow_bindings={},
        selected_workflow_ids=[],
        release_version="v1",
    )
    restored_configs = []
    result = rollback_runtime_transaction(
        comfy_root=comfy,
        snapshot=snapshot,
        restore_config=lambda path: restored_configs.append(path) or {"ok": True},
    )
    assert result["status"] == "succeeded"
    assert _git(comfy, "rev-parse", "HEAD").lower() == snapshot["comfy_ref"]
    assert _git(node, "rev-parse", "HEAD").lower() == snapshot[
        "custom_nodes"
    ]["tracking-node"]["head"]
    assert (comfy / ".venv" / "marker.txt").read_text(encoding="utf-8") == (
        "old-venv"
    )
    assert load_runtime_receipt(comfy) == old_receipt
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("select value from marker").fetchone() == (
            "before-update",
        )
    finally:
        connection.close()
    assert "restored_database" in result["restored"]
    assert restored_configs == [str(tmp_path / "config.backup.json")]
    assert not newly_installed.exists()


def test_transaction_preserves_only_the_installer_owned_instant_lora_patch(
    tmp_path: Path,
) -> None:
    comfy = tmp_path / "comfy"
    comfy_ref = _init_repo(comfy)
    (comfy / ".venv").mkdir()
    node = comfy / "custom_nodes" / "comfyui-instant-lora_v_soya"
    node_ref = _init_repo(node, "src/runtime.py")
    runtime = node / "src" / "runtime.py"
    original = runtime.read_text(encoding="utf-8")
    patched = (
        original
        + "# comfy-installer: use the project-managed Python 3.12 runtime\n"
    )
    runtime.write_text(patched, encoding="utf-8")
    manifest = InstallManifest(
        source_path=tmp_path / "manifest.json",
        data={
            "comfy": {
                "version": "0.31.0",
                "repository": "https://example.invalid/ComfyUI.git",
                "ref": comfy_ref,
            },
            "python": {
                "version": "3.12.11",
                "compatibility_packages": [],
                "gpu_profiles": [],
            },
            "custom_nodes": [
                {
                    "name": "comfyui-instant-lora_v_soya",
                    "source_type": "git",
                    "repository": "https://example.invalid/instant-lora.git",
                    "tracking_branch": "main",
                }
            ],
            "models": [],
            "workflows": {},
        },
        sha256="manifest-sha",
    )

    snapshot = create_runtime_transaction(
        comfy_root=comfy,
        manifest=manifest,
        config_backup={},
    )

    saved_node = snapshot["custom_nodes"]["comfyui-instant-lora_v_soya"]
    assert saved_node["head"] == node_ref
    assert "managed_worktree_patch" in saved_node
    _git(node, "checkout", "--", "src/runtime.py")
    _commit(node, "src/runtime.py", "upstream-new\n")

    result = rollback_runtime_transaction(
        comfy_root=comfy,
        snapshot=snapshot,
        restore_config=lambda _path: {"ok": True},
    )

    assert result["status"] == "succeeded"
    assert runtime.read_text(encoding="utf-8") == patched
    assert _git(node, "rev-parse", "HEAD").lower() == node_ref
    assert _git(node, "status", "--porcelain", "--untracked-files=no") == (
        "M src/runtime.py"
    )
