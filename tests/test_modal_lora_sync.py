from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_backend import client_cli
from modal_backend.lora_inventory import (
    merge_remote_lora_catalog,
    public_lora_catalog,
)
from modal_backend.service import ModalService
import modal_backend.service as service_module


FRONTEND = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(
    encoding="utf-8"
)


def _bot_item(tmp_path: Path) -> dict:
    first = tmp_path / "alice.safetensors"
    second = tmp_path / "bob.safetensors"
    first.write_bytes(b"alice")
    second.write_bytes(b"bob")
    return {
        "key": "bot::MyBot",
        "category": "bot",
        "name": "MyBot",
        "subtitle": "프로젝트 1개 · 캐릭터 LoRA 2개",
        "detail": "현재 봇 구성",
        "scopes": ["SOYA_CHAR_LORA/SOYA_BOT_LORA/MyBot"],
        "files": [
            {
                "source_path": str(first),
                "remote_path": "SOYA_CHAR_LORA/SOYA_BOT_LORA/MyBot/Lora/main/Alice/s1/alice.safetensors",
                "name": first.name,
                "size": first.stat().st_size,
                "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            },
            {
                "source_path": str(second),
                "remote_path": "SOYA_CHAR_LORA/SOYA_BOT_LORA/MyBot/Lora/main/Bob/s1/bob.safetensors",
                "name": second.name,
                "size": second.stat().st_size,
                "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
            },
        ],
        "file_count": 2,
        "size_bytes": first.stat().st_size + second.stat().st_size,
        "sync_state": "unchecked",
    }


def test_merge_remote_lora_catalog_marks_bot_extra_and_remote_only(tmp_path: Path) -> None:
    bot = _bot_item(tmp_path)
    first = bot["files"][0]
    second = bot["files"][1]
    remote = {
        "files": [
            {
                "path": first["remote_path"],
                "size": first["size"],
                "manifest_size": first["size"],
                "sha256": first["sha256"],
            },
            {
                "path": second["remote_path"],
                "size": second["size"],
                "manifest_size": second["size"],
                "sha256": "0" * 64,
            },
            {
                "path": "SOYA_CHAR_LORA/SOYA_BOT_LORA/MyBot/Lora/old/Removed/s1/old.safetensors",
                "size": 3,
                "manifest_size": 3,
                "sha256": "1" * 64,
            },
            {
                "path": "SOYA_CHAR_LORA/SOYA_BOT_LORA/RemoteBot/Lora/main/A/s1/a.safetensors",
                "size": 4,
                "manifest_size": 4,
                "sha256": "2" * 64,
            },
        ],
        "errors": [],
    }

    merged = merge_remote_lora_catalog({"items": [bot], "errors": []}, remote)

    local_bot = next(item for item in merged["items"] if item["key"] == "bot::MyBot")
    remote_bot = next(item for item in merged["items"] if item["key"] == "bot::RemoteBot")
    assert local_bot["sync_state"] == "update"
    assert local_bot["different_count"] == 1
    assert local_bot["extra_count"] == 1
    assert remote_bot["sync_state"] == "remote_only"
    assert merged["counts"] == {
        "all": 2,
        "synced": 0,
        "update": 1,
        "local_only": 0,
        "remote_only": 1,
    }
    public = public_lora_catalog(merged)
    assert "scopes" not in public["items"][0]
    assert "source_path" not in public["items"][0]["files"][0]
    assert "sha256" not in public["items"][0]["files"][0]


class _FakeBatch:
    def __init__(self, volume: "_FakeVolume") -> None:
        self.volume = volume

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def put_file(self, source, remote_path: str) -> None:
        if hasattr(source, "read"):
            position = source.tell()
            raw = source.read()
            source.seek(position)
            if remote_path == client_cli.LORA_SYNC_MANIFEST_PATH:
                self.volume.manifest = json.loads(raw.decode("utf-8"))
        else:
            path = str(remote_path).lstrip("/")
            source_path = Path(str(source))
            self.volume.files[path] = source_path.stat().st_size
            self.volume.uploads.append((str(source_path), remote_path))


class _FakeVolume:
    def __init__(self, files: dict[str, int], manifest: dict) -> None:
        self.files = dict(files)
        self.manifest = dict(manifest)
        self.uploads: list[tuple[str, str]] = []
        self.removes: list[tuple[str, bool]] = []

    def read_file(self, path: str):
        if path != client_cli.LORA_SYNC_MANIFEST_PATH:
            raise FileNotFoundError(path)
        return [json.dumps(self.manifest).encode("utf-8")]

    def batch_upload(self, *, force: bool):
        assert force is True
        return _FakeBatch(self)

    def listdir(self, _path: str, *, recursive: bool):
        assert recursive is True
        file_type = SimpleNamespace(name="FILE")
        return [
            SimpleNamespace(path=f"/{path}", size=size, mtime=1, type=file_type)
            for path, size in self.files.items()
        ]

    def remove_file(self, path: str, *, recursive: bool) -> None:
        normalized = path.lstrip("/")
        self.removes.append((path, recursive))
        for existing in list(self.files):
            if existing == normalized or (recursive and existing.startswith(normalized + "/")):
                del self.files[existing]


def test_client_sync_uploads_changed_file_then_removes_bot_extras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "alice.safetensors"
    source.write_bytes(b"new-alice")
    desired = "SOYA_CHAR_LORA/SOYA_BOT_LORA/MyBot/Lora/main/Alice/s1/alice.safetensors"
    extra = "SOYA_CHAR_LORA/SOYA_BOT_LORA/MyBot/Lora/old/Removed/s1/old.safetensors"
    volume = _FakeVolume(
        {desired: 3, extra: 4},
        {
            desired: {"sha256": "0" * 64, "size": 3},
            extra: {"sha256": "1" * 64, "size": 4},
        },
    )
    monkeypatch.setattr(
        client_cli.modal.Volume,
        "from_name",
        lambda *_args, **_kwargs: volume,
    )

    result = client_cli.manage_loras(
        {
            "app_name": "test",
            "environment": "main",
            "mode": "sync",
            "scopes": ["SOYA_CHAR_LORA/SOYA_BOT_LORA/MyBot"],
            "lora_files": [
                {
                    "source_path": str(source),
                    "remote_path": desired,
                    "size": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            ],
        }
    )

    assert result["uploaded"] == 1
    assert result["deleted_count"] == 1
    assert (str(source), f"/{desired}") in volume.uploads
    assert (f"/{extra}", False) in volume.removes
    assert desired in volume.files
    assert extra not in volume.files
    assert set(volume.manifest) == {desired}


@pytest.mark.asyncio
async def test_service_starts_one_bot_as_single_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = _bot_item(tmp_path)
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})

    async def connected(_settings) -> bool:
        return True

    async def list_remote(*_args, **_kwargs) -> dict:
        return {"files": [], "errors": []}

    captured: dict = {}

    async def run_command(_args, *, stdin_payload, **_kwargs):
        captured.update(stdin_payload)
        return (
            0,
            json.dumps(
                {
                    "ok": True,
                    "result": {
                        "mode": "sync",
                        "uploaded": 2,
                        "skipped": 0,
                        "deleted_count": 1,
                    },
                }
            ),
            "",
        )

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_client_action", list_remote)
    monkeypatch.setattr(service, "_run_command", run_command)
    monkeypatch.setattr(
        service_module,
        "build_local_lora_catalog",
        lambda *_args, **_kwargs: {"items": [bot], "errors": []},
    )

    started = await service.start_lora_operation("sync", [bot["key"]])
    assert started["state"] == "running"
    await service._lora_operation_task

    assert captured["action"] == "manage_loras"
    assert captured["mode"] == "sync"
    assert captured["scopes"] == ["SOYA_CHAR_LORA/SOYA_BOT_LORA/MyBot"]
    assert len(captured["lora_files"]) == 2
    assert service.lora_operation_status()["state"] == "completed"


def test_frontend_has_lora_sync_button_modal_status_and_bot_unit_copy() -> None:
    required = (
        'id="modal-lora-sync-open-btn"',
        'id="modal-lora-sync-modal" hidden',
        'id="modal-lora-status-query-btn"',
        'data-lora-category="bot"',
        'data-lora-category="asset"',
        'data-lora-category="instance"',
        'data-lora-category="style"',
        "function modalOpenLoraSync()",
        "function modalLoraRunAction(action, explicitKeys = null)",
        "fetchJSON('/api/modal/loras?remote=1')",
        "봇 전체가 한 단위입니다",
        "원격 Volume에서만 삭제하며 로컬 파일은 그대로 유지합니다",
    )
    for text in required:
        assert text in FRONTEND
