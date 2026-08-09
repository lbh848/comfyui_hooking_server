from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess

import pytest

from modal_backend import custom_nodes as custom_nodes_module
from modal_backend.custom_nodes import (
    inventory_custom_nodes,
    public_custom_node_inventory,
)
from modal_backend.service import ModalService


def _project(tmp_path: Path, manifest_names: list[str] | None = None) -> Path:
    root = tmp_path / "project"
    manifest = root / "comfy_installer" / "resources" / "install_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "custom_nodes": [
                    {"name": name, "source_type": "git"}
                    for name in (manifest_names or [])
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "comfy" / "custom_nodes").mkdir(parents=True)
    return root


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout.strip()


def _clean_git_node(root: Path, name: str = "extra-node") -> tuple[Path, str]:
    node = root / "comfy" / "custom_nodes" / name
    node.mkdir()
    _git(node, "init")
    _git(node, "config", "user.email", "modal-test@example.invalid")
    _git(node, "config", "user.name", "Modal Test")
    (node / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    _git(node, "add", "__init__.py")
    _git(node, "commit", "-m", "initial")
    _git(node, "remote", "add", "origin", "https://example.com/extra-node.git")
    return node, _git(node, "rev-parse", "HEAD")


def test_inventory_adds_clean_git_node_at_exact_local_commit(tmp_path: Path) -> None:
    root = _project(tmp_path, ["known-node"])
    node, commit = _clean_git_node(root)
    cache = node / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"cache")

    inventory = inventory_custom_nodes(root)

    assert inventory["build_nodes"] == [
        {
            "name": "extra-node",
            "source_type": "git",
            "repository": "https://example.com/extra-node.git",
            "ref": commit,
        }
    ]
    assert inventory["summary"]["git"] == 1
    assert inventory["warnings"] == []


def test_inventory_falls_back_to_bounded_local_copy_for_dirty_git(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    node, _commit = _clean_git_node(root)
    (node / "local-change.py").write_text("VALUE = 1\n", encoding="utf-8")
    ignored = node / ".venv"
    ignored.mkdir()
    (ignored / "large.bin").write_bytes(b"x" * 1024)

    inventory = inventory_custom_nodes(root)
    public = public_custom_node_inventory(inventory)

    built = inventory["build_nodes"][0]
    assert built["source_type"] == "local"
    assert built["source_path"] == str(node.resolve())
    assert built["size_bytes"] < 1024
    assert "source_path" not in public["discovered_nodes"][0]
    assert "로컬 변경 사항" in public["warnings"][0]


def test_inventory_reports_oversized_non_git_node_as_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    node = root / "comfy" / "custom_nodes" / "large-node"
    node.mkdir()
    (node / "payload.bin").write_bytes(b"12345")
    monkeypatch.setattr(custom_nodes_module, "LOCAL_NODE_MAX_BYTES", 4)

    inventory = inventory_custom_nodes(root)

    assert inventory["build_nodes"] == []
    assert inventory["summary"]["skipped"] == 1
    assert inventory["skipped"][0]["name"] == "large-node"
    assert "크기 제한 초과" in inventory["skipped"][0]["reason"]


@pytest.mark.asyncio
async def test_custom_node_sync_force_builds_worker_once_then_deploys_web(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    service = ModalService(
        root,
        lambda: {
            "modal_enabled": True,
            "modal_deployment_name": "worker-app",
            "modal_environment": "main",
        },
    )
    inventory = {
        "manifest_nodes": [],
        "build_nodes": [],
        "skipped": [],
        "warnings": [],
        "summary": {
            "manifest": 0,
            "git": 0,
            "local": 0,
            "skipped": 0,
            "warning": 0,
            "local_bytes": 0,
        },
    }
    commands: list[list[str]] = []
    force_values: list[str] = []
    stopped_apps: list[str] = []

    async def connected(_settings) -> bool:
        return True

    async def deploy_inventory() -> dict:
        return inventory

    async def run_command(args, **kwargs):
        commands.append(list(args))
        force_values.append(kwargs["env"]["SOYA_MODAL_FORCE_CUSTOM_NODE_BUILD"])
        return 0, "", ""

    async def stop_web(settings) -> None:
        stopped_apps.append(service._web_app_name(settings))

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_inventory_for_deploy", deploy_inventory)
    monkeypatch.setattr(service, "_run_command", run_command)
    monkeypatch.setattr(service, "_stop_web_app", stop_web)

    state = await service.start_redeploy(force_custom_nodes=True)
    assert state["state"] == "running"
    assert service._deployment_task is not None
    await service._deployment_task

    assert service._deployment_state["state"] == "completed"
    assert service._deployment_state["phase"] == "complete"
    assert service._deployment_state["finished_at"] >= service._deployment_state["started_at"]
    assert any(
        "작업 App 배포 완료" in item["message"]
        for item in service._deployment_state["logs"]
    )
    assert len(commands) == 2
    assert commands[0][1:4] == ["-m", "modal", "deploy"]
    assert commands[1][1:6] == ["-m", "modal", "deploy", "-m", "modal_backend.modal_web_app"]
    assert force_values == ["1", "0"]
    assert stopped_apps == ["worker-app-web"]
    assert service._web_state["state"] == "stopped"
    assert service._web_state["deployed"] is False
    assert service._web_state["num_total_runners"] == 0
    assert any(
        "자동 종료 완료" in item["message"]
        for item in service._deployment_state["logs"]
    )


@pytest.mark.asyncio
async def test_concurrent_web_start_requests_register_only_one_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(
        tmp_path,
        lambda: {"modal_enabled": True, "modal_deployment_name": "worker-app"},
    )
    release = asyncio.Event()
    started = 0

    async def connected(_settings) -> bool:
        await asyncio.sleep(0)
        return True

    async def stopped(**_kwargs) -> dict:
        await asyncio.sleep(0)
        return {"state": "stopped", "deployed": False}

    async def run_start(_settings, _current) -> None:
        nonlocal started
        started += 1
        await release.wait()

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "web_status", stopped)
    monkeypatch.setattr(service, "_run_web_start", run_start)

    first, second = await asyncio.gather(service.start_web(), service.start_web())

    assert first["state"] == "starting"
    assert second["state"] == "starting"
    assert started == 1
    release.set()
    assert service._web_task is not None
    await service._web_task
