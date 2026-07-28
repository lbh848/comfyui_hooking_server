import hashlib
import json
import zipfile
from pathlib import Path
from threading import Event

import httpx
import pytest

from comfy_installer.downloader import ResumableDownloader
from comfy_installer.node_installer import NodeInstallError, install_archive_node


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
