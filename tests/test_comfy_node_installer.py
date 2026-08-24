import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from threading import Event

import httpx
import pytest

import comfy_installer.node_installer as node_installer_module
from comfy_installer.downloader import ResumableDownloader
from comfy_installer.node_installer import (
    NodeInstallError,
    install_archive_node,
    install_git_node,
    update_git_node,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_bytes(tmp_path: Path, members: dict[str, bytes]) -> bytes:
    path = tmp_path / "node.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path.read_bytes()


def _downloader(payload: bytes) -> ResumableDownloader:
    return ResumableDownloader(
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=payload)
            )
        )
    )


def test_archive_node_install_is_atomic_and_reusable(tmp_path):
    payload = _zip_bytes(
        tmp_path,
        {
            "__init__.py": b"NODE_CLASS_MAPPINGS = {}\n",
            "requirements.txt": b"numpy\n",
        },
    )
    node = {
        "name": "test-node",
        "source_type": "archive",
        "version": "1.0.0",
        "url": "https://example.test/node.zip",
        "size": len(payload),
        "sha256": _sha(payload),
    }
    custom_root = tmp_path / "comfy" / "custom_nodes"
    cache = tmp_path / "cache"

    first = install_archive_node(
        node=node,
        custom_root=custom_root,
        cache_root=cache,
        downloader=_downloader(payload),
        cancel_event=Event(),
        log=None,
        progress=None,
    )
    second = install_archive_node(
        node=node,
        custom_root=custom_root,
        cache_root=cache,
        downloader=_downloader(payload),
        cancel_event=Event(),
        log=None,
        progress=None,
    )

    assert first == second
    assert (first / "__init__.py").is_file()
    marker = json.loads(
        (first / ".comfy-installer-source.json").read_text(encoding="utf-8")
    )
    assert marker["sha256"] == _sha(payload)


def test_archive_node_rejects_path_traversal(tmp_path):
    payload = _zip_bytes(tmp_path, {"../outside.py": b"bad"})
    node = {
        "name": "bad-node",
        "source_type": "archive",
        "version": "1.0.0",
        "url": "https://example.test/node.zip",
        "size": len(payload),
        "sha256": _sha(payload),
    }

    with pytest.raises(NodeInstallError, match="안전하지 않은 경로"):
        install_archive_node(
            node=node,
            custom_root=tmp_path / "comfy" / "custom_nodes",
            cache_root=tmp_path / "cache",
            downloader=_downloader(payload),
            cancel_event=Event(),
            log=None,
            progress=None,
        )

    assert not (tmp_path / "outside.py").exists()
    assert not (tmp_path / "comfy" / "custom_nodes" / "bad-node").exists()


def test_archive_node_refuses_unmanaged_existing_directory(tmp_path):
    payload = _zip_bytes(tmp_path, {"__init__.py": b""})
    custom_root = tmp_path / "comfy" / "custom_nodes"
    existing = custom_root / "test-node"
    existing.mkdir(parents=True)
    (existing / "user.py").write_text("mine", encoding="utf-8")
    node = {
        "name": "test-node",
        "source_type": "archive",
        "version": "1.0.0",
        "url": "https://example.test/node.zip",
        "size": len(payload),
        "sha256": _sha(payload),
    }

    with pytest.raises(NodeInstallError, match="덮어쓰지 않습니다"):
        install_archive_node(
            node=node,
            custom_root=custom_root,
            cache_root=tmp_path / "cache",
            downloader=_downloader(payload),
            cancel_event=Event(),
            log=None,
            progress=None,
        )

    assert (existing / "user.py").read_text(encoding="utf-8") == "mine"


def test_pinned_git_node_reuses_mismatched_head_with_same_origin_policy(
    tmp_path, monkeypatch, capsys
):
    expected_head = "1" * 40
    actual_head = "2" * 40
    repository = "https://example.test/owned-node.git"
    custom_root = tmp_path / "comfy" / "custom_nodes"
    destination = custom_root / "owned-node"
    (destination / ".git").mkdir(parents=True)
    logs = []

    monkeypatch.setattr(
        node_installer_module,
        "_git_origin",
        lambda _path: repository.removesuffix(".git") + "/",
    )
    monkeypatch.setattr(
        node_installer_module,
        "_git_head",
        lambda _path: actual_head,
    )

    installed = install_git_node(
        node={
            "name": "owned-node",
            "source_type": "git",
            "repository": repository,
            "ref": expected_head,
            "existing_policy": "reuse_if_same_origin",
        },
        custom_root=custom_root,
        cancel_event=Event(),
        log=logs.append,
    )

    assert installed == destination
    assert logs == [
        f"[노드] 기존 Git 설치 정책 재사용: owned-node {actual_head[:12]}"
    ]
    output = capsys.readouterr().out
    assert f"expected={expected_head}" in output
    assert f"actual={actual_head}" in output
    assert f"origin={repository.removesuffix('.git')}/" in output


def test_pinned_git_node_without_policy_still_rejects_mismatched_head(
    tmp_path, monkeypatch
):
    expected_head = "1" * 40
    actual_head = "2" * 40
    repository = "https://example.test/owned-node.git"
    custom_root = tmp_path / "comfy" / "custom_nodes"
    destination = custom_root / "owned-node"
    (destination / ".git").mkdir(parents=True)

    monkeypatch.setattr(
        node_installer_module,
        "_git_origin",
        lambda _path: repository,
    )
    monkeypatch.setattr(
        node_installer_module,
        "_git_head",
        lambda _path: actual_head,
    )

    with pytest.raises(NodeInstallError, match="고정점과 달라"):
        install_git_node(
            node={
                "name": "owned-node",
                "source_type": "git",
                "repository": repository,
                "ref": expected_head,
            },
            custom_root=custom_root,
            cancel_event=Event(),
            log=None,
        )


def test_pinned_git_node_policy_still_rejects_different_origin(
    tmp_path, monkeypatch
):
    expected_head = "1" * 40
    repository = "https://example.test/owned-node.git"
    custom_root = tmp_path / "comfy" / "custom_nodes"
    destination = custom_root / "owned-node"
    (destination / ".git").mkdir(parents=True)

    monkeypatch.setattr(
        node_installer_module,
        "_git_origin",
        lambda _path: "https://example.test/different-node.git",
    )
    monkeypatch.setattr(
        node_installer_module,
        "_git_head",
        lambda _path: "2" * 40,
    )

    with pytest.raises(NodeInstallError, match="원격 저장소가 다릅니다"):
        install_git_node(
            node={
                "name": "owned-node",
                "source_type": "git",
                "repository": repository,
                "ref": expected_head,
                "existing_policy": "reuse_if_same_origin",
            },
            custom_root=custom_root,
            cancel_event=Event(),
            log=None,
        )


def test_update_git_node_policy_reuses_without_status_or_fetch(
    tmp_path, monkeypatch, capsys
):
    expected_head = "1" * 40
    actual_head = "2" * 40
    repository = "https://example.test/owned-node.git"
    comfy_root = tmp_path / "comfy"
    destination = comfy_root / "custom_nodes" / "owned-node"
    (destination / ".git").mkdir(parents=True)
    logs = []
    changed_nodes = []

    monkeypatch.setattr(
        node_installer_module,
        "_git_origin",
        lambda _path: repository,
    )
    monkeypatch.setattr(
        node_installer_module,
        "_git_head",
        lambda _path: actual_head,
    )

    def unexpected_run(command, **_kwargs):
        raise AssertionError(f"정책 재사용 뒤 명령이 실행됐습니다: {command}")

    monkeypatch.setattr(node_installer_module, "run_command", unexpected_run)

    updated = update_git_node(
        node={
            "name": "owned-node",
            "source_type": "git",
            "repository": repository,
            "ref": expected_head,
            "existing_policy": "reuse_if_same_origin",
        },
        comfy_root=comfy_root,
        cancel_event=Event(),
        log=logs.append,
        changed_nodes=changed_nodes,
    )

    assert updated == destination
    assert changed_nodes == []
    assert logs == [
        "[노드 업데이트] 기존 Git 설치 정책 재사용: "
        f"owned-node {actual_head[:12]}"
    ]
    assert "기존 Git 설치 정책 재사용" in capsys.readouterr().out


def test_tracking_git_node_fetches_main_and_checks_out_latest(
    tmp_path, monkeypatch
):
    old_head = "1" * 40
    latest_head = "2" * 40
    repository = "https://example.test/owned-node.git"
    comfy_root = tmp_path / "comfy"
    destination = comfy_root / "custom_nodes" / "owned-node"
    (destination / ".git").mkdir(parents=True)
    commands = []
    changed_nodes = []
    head_values = iter((old_head, latest_head))

    monkeypatch.setattr(
        node_installer_module,
        "_git_origin",
        lambda _path: repository,
    )
    monkeypatch.setattr(
        node_installer_module,
        "_git_head",
        lambda _path: next(head_values),
    )

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command == ["git", "status", "--porcelain", "--untracked-files=no"]:
            return []
        if command == ["git", "rev-parse", "FETCH_HEAD"]:
            return [latest_head]
        return []

    monkeypatch.setattr(node_installer_module, "run_command", fake_run)

    updated = update_git_node(
        node={
            "name": "owned-node",
            "source_type": "git",
            "repository": repository,
            "tracking_branch": "main",
        },
        comfy_root=comfy_root,
        cancel_event=Event(),
        log=None,
        changed_nodes=changed_nodes,
    )

    assert updated == destination
    assert changed_nodes == ["owned-node"]
    assert ["git", "fetch", "--depth", "1", "origin", "main"] in commands
    assert ["git", "checkout", "--detach", "FETCH_HEAD"] in commands


def test_tracking_git_node_fetches_even_when_already_latest(
    tmp_path, monkeypatch
):
    latest_head = "3" * 40
    repository = "https://example.test/owned-node.git"
    comfy_root = tmp_path / "comfy"
    destination = comfy_root / "custom_nodes" / "owned-node"
    (destination / ".git").mkdir(parents=True)
    commands = []
    changed_nodes = []

    monkeypatch.setattr(
        node_installer_module,
        "_git_origin",
        lambda _path: repository,
    )
    monkeypatch.setattr(
        node_installer_module,
        "_git_head",
        lambda _path: latest_head,
    )

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command == ["git", "status", "--porcelain", "--untracked-files=no"]:
            return []
        if command == ["git", "rev-parse", "FETCH_HEAD"]:
            return [latest_head]
        return []

    monkeypatch.setattr(node_installer_module, "run_command", fake_run)

    update_git_node(
        node={
            "name": "owned-node",
            "source_type": "git",
            "repository": repository,
            "tracking_branch": "main",
        },
        comfy_root=comfy_root,
        cancel_event=Event(),
        log=None,
        changed_nodes=changed_nodes,
    )

    assert changed_nodes == []
    assert ["git", "fetch", "--depth", "1", "origin", "main"] in commands
    assert ["git", "checkout", "--detach", "FETCH_HEAD"] not in commands


def test_tracking_git_node_follows_real_main_branch(tmp_path):
    source = tmp_path / "source"
    remote = tmp_path / "remote.git"
    source.mkdir()

    def git(cwd, *arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    git(source, "init", "-b", "main")
    git(source, "config", "user.name", "Comfy Installer Test")
    git(source, "config", "user.email", "comfy-installer@example.test")
    (source / "node.py").write_text("VERSION = 1\n", encoding="utf-8")
    git(source, "add", "node.py")
    git(source, "commit", "-m", "first")
    git(tmp_path, "init", "--bare", str(remote))
    git(source, "remote", "add", "origin", str(remote))
    git(source, "push", "-u", "origin", "main")

    node = {
        "name": "owned-node",
        "source_type": "git",
        "repository": str(remote),
        "tracking_branch": "main",
    }
    custom_root = tmp_path / "comfy" / "custom_nodes"
    installed = install_git_node(
        node=node,
        custom_root=custom_root,
        cancel_event=Event(),
        log=None,
    )
    first_head = git(installed, "rev-parse", "HEAD")
    assert (installed / "node.py").read_text(encoding="utf-8") == "VERSION = 1\n"

    (source / "node.py").write_text("VERSION = 2\n", encoding="utf-8")
    git(source, "add", "node.py")
    git(source, "commit", "-m", "second")
    git(source, "push", "origin", "main")
    latest_head = git(source, "rev-parse", "HEAD")
    changed_nodes = []

    update_git_node(
        node=node,
        comfy_root=tmp_path / "comfy",
        cancel_event=Event(),
        log=None,
        changed_nodes=changed_nodes,
    )

    assert first_head != latest_head
    assert git(installed, "rev-parse", "HEAD") == latest_head
    assert git(installed, "branch", "--show-current") == ""
    assert (installed / "node.py").read_text(encoding="utf-8") == "VERSION = 2\n"
    assert changed_nodes == ["owned-node"]
