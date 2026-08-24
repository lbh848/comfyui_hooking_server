import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy_allocation import CURRENT_COMFY_EXECUTION_TARGET, MODAL_COMFY_TARGET
from modal_backend import client_cli
from modal_backend.service import DOWNLOAD_PROGRESS_PREFIX, ModalService
from modal_backend.settings import ModalSettings
from queue_manager import QueueItem, QueueManager


def test_modal_function_call_forwards_each_structured_progress_event_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeLogs:
        def __init__(self, call) -> None:
            self.call = call

        def tail(self, *, entries: int):
            assert entries == client_cli.CALL_START_LOG_TAIL_ENTRIES
            events = [
                SimpleNamespace(
                    message=client_cli.CALL_STARTED_LOG_PREFIX + '{"operation":"generate"}',
                    context_ids=["input", "container-1"],
                )
            ]
            if self.call.poll_count >= 1:
                events.append(
                    SimpleNamespace(
                        message=(
                            client_cli.WORKFLOW_PROGRESS_PREFIX
                            + '{"sequence":1,"data":{"phase":"training","step":1,"total":3}}'
                        ),
                        context_ids=["input", "container-1"],
                    )
                )
            if self.call.poll_count >= 2:
                events.append(
                    SimpleNamespace(
                        message=(
                            client_cli.WORKFLOW_PROGRESS_PREFIX
                            + '{"sequence":2,"data":{"phase":"training","step":2,"total":3}}'
                        ),
                        context_ids=["input", "container-1"],
                    )
                )
            return events

    class FakeCall:
        def __init__(self) -> None:
            self.poll_count = 0
            self.logs = FakeLogs(self)

        def get(self, *, timeout: float):
            assert timeout > 0
            self.poll_count += 1
            if self.poll_count < 3:
                raise TimeoutError("running")
            return {"prompt_id": "prompt-1"}

    result = client_cli._wait_for_call_with_start_retry_limit(
        FakeCall(),
        timeout_seconds=30,
        max_retries=2,
        operation="generate",
        stream_progress=True,
    )

    assert result == {"prompt_id": "prompt-1"}
    forwarded = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith(client_cli.WORKFLOW_PROGRESS_PREFIX)
    ]
    assert [
        json.loads(line[len(client_cli.WORKFLOW_PROGRESS_PREFIX) :])["step"]
        for line in forwarded
    ] == [1, 2]


def test_modal_client_downloads_volume_artifact_with_byte_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeVolume:
        def read_file(self, path: str):
            assert path == "/SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors"
            return [b"abc", b"def"]

    monkeypatch.setattr(
        client_cli,
        "_lora_volume",
        lambda _payload, *, create_if_missing: FakeVolume(),
    )

    result = client_cli.download_lora_artifacts(
        {
            "app_name": "test-app",
            "environment": "main",
            "output_dir": str(tmp_path),
            "artifacts": [
                {
                    "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
                    "remote_path": "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors",
                    "size": 6,
                }
            ],
        }
    )

    target = tmp_path / "SOYA_INSTANCE_LORA" / "alice.safetensors"
    assert target.read_bytes() == b"abcdef"
    assert result["artifacts"][0]["path"] == str(target)
    assert result["artifacts"][0]["sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    events = [
        json.loads(line[len(client_cli.DOWNLOAD_PROGRESS_PREFIX) :])
        for line in capsys.readouterr().err.splitlines()
        if line.startswith(client_cli.DOWNLOAD_PROGRESS_PREFIX)
    ]
    assert [event["event"] for event in events] == [
        "batch_start",
        "file_start",
        "chunk",
        "chunk",
        "file_complete",
        "batch_complete",
    ]
    assert events[-1]["downloaded_bytes"] == 6


@pytest.mark.asyncio
async def test_service_download_stores_then_queues_exact_remote_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "local-loras"
    service = ModalService(
        tmp_path,
        lambda: {
            "modal_enabled": True,
            "lora_load_path": str(local_root),
        },
    )
    observed_delete_artifacts: list[dict] = []
    observed_progress: list[dict] = []

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def run_command(*_args, **kwargs):
        payload = kwargs["stdin_payload"]
        output_dir = Path(payload["output_dir"]).resolve()
        assert output_dir.parent == (tmp_path / "runtime" / "temp").resolve()
        target = (
            output_dir
            / "SOYA_INSTANCE_LORA"
            / "alice.safetensors"
        )
        target.parent.mkdir(parents=True)
        target.write_bytes(b"abcdef")
        output_callback = kwargs["output_callback"]
        for event in (
            {
                "event": "batch_start",
                "total_files": 1,
                "total_bytes": 6,
            },
            {
                "event": "chunk",
                "index": 1,
                "total_files": 1,
                "downloaded_bytes": 3,
                "total_bytes": 6,
                "name": "SOYA_INSTANCE_LORA/alice.safetensors",
            },
            {
                "event": "batch_complete",
                "total_files": 1,
                "downloaded_bytes": 6,
                "total_bytes": 6,
            },
        ):
            output_callback(
                "stderr",
                DOWNLOAD_PROGRESS_PREFIX + json.dumps(event),
            )
        return (
            0,
            json.dumps(
                {
                    "ok": True,
                    "result": {
                        "artifacts": [
                            {
                                "path": str(target),
                                "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
                                "remote_path": (
                                    "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors"
                                ),
                                "size": 6,
                                "sha256": hashlib.sha256(b"abcdef").hexdigest(),
                            }
                        ]
                    },
                }
            ),
            "",
        )

    async def enqueue_delete(remote_artifacts: list[dict]) -> None:
        observed_delete_artifacts.extend(dict(item) for item in remote_artifacts)

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", run_command)
    monkeypatch.setattr(service, "enqueue_lora_delete_artifacts", enqueue_delete)

    result = await service.download_lora_artifacts(
        [
            {
                "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
                "remote_path": "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors",
                "size": 6,
            }
        ],
        progress_callback=lambda event: observed_progress.append(dict(event)),
    )

    assert (local_root / "SOYA_INSTANCE_LORA" / "alice.safetensors").read_bytes() == b"abcdef"
    assert result["remote_delete_queued"] == [
        "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors"
    ]
    assert observed_delete_artifacts == [
        {
            "path": observed_delete_artifacts[0]["path"],
            "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
            "remote_path": "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors",
            "size": 6,
            "sha256": hashlib.sha256(b"abcdef").hexdigest(),
        }
    ]
    assert any(event["phase"] == "modal_downloading" for event in observed_progress)
    assert observed_progress[-1]["phase"] == "modal_download_complete"
    assert list((tmp_path / "runtime" / "temp").iterdir()) == []


@pytest.mark.asyncio
async def test_modal_training_returns_before_independent_download_finishes() -> None:
    manager = QueueManager()
    manager.get_config = lambda: {
        "modal_enabled": True,
        "modal_worker_gpu": "L40S",
        "modal_max_concurrency": 1,
    }
    manager.notify_frontend = lambda *_args, **_kwargs: asyncio.sleep(0)
    download_started = asyncio.Event()
    release_download = asyncio.Event()
    completed_callbacks: list[str] = []

    async def run_modal_workflow(_workflow, **kwargs):
        await kwargs["progress_callback"](
            {"phase": "training", "step": 5, "total": 10}
        )
        return {
            "prompt_id": "prompt-1",
            "deferred_artifacts": [
                {
                    "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
                    "remote_path": "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors",
                    "size": 6,
                }
            ],
        }

    async def download_modal_artifacts(_artifacts, *, progress_callback):
        download_started.set()
        await progress_callback(
            {
                "phase": "modal_downloading",
                "percentage": 50,
                "downloaded_bytes": 3,
                "total_bytes": 6,
            }
        )
        await release_download.wait()
        await progress_callback(
            {"phase": "modal_download_complete", "percentage": 100}
        )
        return {
            "artifacts": [{"local_path": "alice.safetensors"}],
            "remote_delete_queued": ["SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors"],
        }

    manager.run_modal_workflow = run_modal_workflow
    manager.download_modal_artifacts = download_modal_artifacts
    source_item = QueueItem(
        id="training-1",
        type="instance_lora_training",
        label="Alice 학습",
        params={},
    )
    manager.items.append(source_item)
    token = CURRENT_COMFY_EXECUTION_TARGET.set(MODAL_COMFY_TARGET)
    try:
        prompt_id, submit_result = await manager._monitor_training_ws(
            source_item,
            {"1": {"class_type": "Test", "inputs": {}}},
            event_type="instance_lora_training_progress",
            on_complete=lambda: completed_callbacks.append("done"),
            modal_input_paths=["input/alice"],
            modal_artifact_prefixes=["SOYA_INSTANCE_LORA/alice"],
        )
    finally:
        CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    await asyncio.wait_for(download_started.wait(), timeout=1)
    download_item = next(
        item for item in manager.items if item.type == "modal_lora_download"
    )
    assert prompt_id == "prompt-1"
    assert submit_result["download_item_id"] == download_item.id
    assert download_item.status == "processing"
    assert manager._item_execution_area(download_item) == (
        "modal_download",
        "modal-volume",
    )
    assert manager.get_status()["modal_download_active"] == 1
    assert completed_callbacks == []

    release_download.set()
    await asyncio.wait_for(download_item.completion_future, timeout=1)
    assert download_item.status == "completed"
    assert completed_callbacks == ["done"]


@pytest.mark.asyncio
async def test_exact_lora_delete_outbox_uses_file_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})
    observed_payloads: list[dict] = []

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def run_command(*_args, **kwargs):
        observed_payloads.append(dict(kwargs["stdin_payload"]))
        return 0, json.dumps({"ok": True, "result": {"deleted_count": 1}}), ""

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", run_command)
    monkeypatch.setattr(service, "_schedule_delete_flush", lambda: None)
    remote_path = "SOYA_CHAR_LORA/SOYA_STYLE_LORA/project/anima/style.safetensors"

    await service.enqueue_lora_delete_paths([remote_path])
    queued = service._load_delete_outbox()
    assert queued[0]["remote_paths"] == [remote_path]

    await service._flush_delete_outbox()
    assert observed_payloads[0]["action"] == "delete_lora_paths"
    assert observed_payloads[0]["remote_paths"] == [remote_path]
    assert service._load_delete_outbox() == []


def test_client_treats_invalid_delete_as_complete_when_target_is_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote_prefix = "SOYA_CHAR_LORA/SOYA_BOT_LORA/project/Lora/anima-v10"

    class FakeVolume:
        def remove_file(self, path: str, *, recursive: bool) -> None:
            assert path == f"/{remote_prefix}"
            assert recursive is True
            raise client_cli.modal.exception.InvalidError(
                "No such file or directory."
            )

        def listdir(self, path: str, *, recursive: bool):
            assert path == "/"
            assert recursive is True
            return [
                SimpleNamespace(
                    path="/SOYA_CHAR_LORA/SOYA_BOT_LORA/another-project/model.safetensors"
                )
            ]

        def read_file(self, path: str):
            assert path == client_cli.LORA_SYNC_MANIFEST_PATH
            raise FileNotFoundError(path)

    result = client_cli._delete_lora_paths(
        FakeVolume(),
        [remote_prefix],
        recursive=True,
    )

    assert result == {"deleted": [], "deleted_count": 0}
    assert "대상이 이미 없는 것을 확인" in capsys.readouterr().err


def test_client_keeps_invalid_delete_failure_when_target_still_exists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote_prefix = "SOYA_CHAR_LORA/SOYA_BOT_LORA/project/Lora/anima-v10"

    class FakeVolume:
        def remove_file(self, path: str, *, recursive: bool) -> None:
            assert path == f"/{remote_prefix}"
            assert recursive is True
            raise client_cli.modal.exception.InvalidError("permission denied")

        def listdir(self, path: str, *, recursive: bool):
            assert path == "/"
            assert recursive is True
            return [
                SimpleNamespace(path=f"/{remote_prefix}/model.safetensors")
            ]

    with pytest.raises(client_cli.modal.exception.InvalidError, match="permission denied"):
        client_cli._delete_lora_paths(
            FakeVolume(),
            [remote_prefix],
            recursive=True,
        )

    assert "오류 후에도 대상이 남아 있음" in capsys.readouterr().err


def test_client_skips_delete_when_remote_lora_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVolume:
        def read_file(self, path: str):
            assert path == "/SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors"
            return [b"new-content"]

    def fail_delete(*_args, **_kwargs):
        raise AssertionError("변경된 원격 LoRA를 삭제하면 안 됩니다.")

    monkeypatch.setattr(
        client_cli,
        "_lora_volume",
        lambda _payload, *, create_if_missing: FakeVolume(),
    )
    monkeypatch.setattr(client_cli, "_delete_lora_paths", fail_delete)
    remote_path = "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors"

    result = client_cli.delete_lora_artifacts(
        {
            "remote_artifacts": [
                {
                    "remote_path": remote_path,
                    "size": 6,
                    "sha256": hashlib.sha256(b"abcdef").hexdigest(),
                }
            ]
        }
    )

    assert result["deleted_count"] == 0
    assert result["skipped_changed"] == [remote_path]


@pytest.mark.asyncio
async def test_verified_lora_delete_outbox_uses_hash_guarded_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})
    observed_payloads: list[dict] = []

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def run_command(*_args, **kwargs):
        observed_payloads.append(dict(kwargs["stdin_payload"]))
        return (
            0,
            json.dumps(
                {
                    "ok": True,
                    "result": {
                        "deleted_count": 1,
                        "skipped_changed": [],
                        "already_missing": [],
                    },
                }
            ),
            "",
        )

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", run_command)
    monkeypatch.setattr(service, "_schedule_delete_flush", lambda: None)
    remote_path = "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/alice.safetensors"
    artifact = {
        "remote_path": remote_path,
        "size": 6,
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }

    await service.enqueue_lora_delete_artifacts([artifact])
    queued = service._load_delete_outbox()
    assert queued[0]["remote_artifacts"] == [artifact]

    await service._flush_delete_outbox()
    assert observed_payloads[0]["action"] == "delete_lora_artifacts"
    assert observed_payloads[0]["remote_artifacts"] == [artifact]
    assert service._load_delete_outbox() == []


def test_modal_client_downloads_and_hash_verifies_video_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_bytes = b"verified-modal-mp4"
    remote_path = "SOYA_VIDEO_OUTPUT/video_job/result.mp4"

    class FakeVolume:
        def read_file(self, path: str):
            assert path == f"/{remote_path}"
            return [video_bytes[:8], video_bytes[8:]]

    monkeypatch.setattr(
        client_cli,
        "_video_volume",
        lambda _payload, *, create_if_missing: FakeVolume(),
    )
    artifact = {
        "remote_path": remote_path,
        "filename": "result.mp4",
        "size": len(video_bytes),
        "sha256": hashlib.sha256(video_bytes).hexdigest(),
    }

    result = client_cli.download_video_artifact(
        {
            "app_name": "test-app",
            "environment": "main",
            "output_dir": str(tmp_path),
            "artifact": artifact,
        }
    )

    target = tmp_path / "result.mp4"
    assert target.read_bytes() == video_bytes
    assert result["artifact"]["path"] == str(target.resolve())
    assert result["artifact"]["sha256"] == artifact["sha256"]


def test_modal_client_rejects_corrupt_video_download_without_completed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVolume:
        def read_file(self, _path: str):
            return [b"corrupt"]

    monkeypatch.setattr(
        client_cli,
        "_video_volume",
        lambda _payload, *, create_if_missing: FakeVolume(),
    )
    with pytest.raises(RuntimeError, match="SHA256"):
        client_cli.download_video_artifact(
            {
                "app_name": "test-app",
                "environment": "main",
                "output_dir": str(tmp_path),
                "artifact": {
                    "remote_path": "SOYA_VIDEO_OUTPUT/video_job/result.mp4",
                    "filename": "result.mp4",
                    "size": len(b"corrupt"),
                    "sha256": hashlib.sha256(b"expected").hexdigest(),
                },
            }
        )

    assert not (tmp_path / "result.mp4").exists()
    assert not (tmp_path / ".result.mp4.part").exists()


def test_modal_client_deletes_only_hash_matching_video_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"verified-modal-mp4"
    remote_path = "SOYA_VIDEO_OUTPUT/video_job/result.mp4"

    class FakeVolume:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, bool]] = []

        def read_file(self, path: str):
            assert path == f"/{remote_path}"
            return [expected]

        def remove_file(self, path: str, *, recursive: bool) -> None:
            self.deleted.append((path, recursive))

    volume = FakeVolume()
    monkeypatch.setattr(
        client_cli,
        "_video_volume",
        lambda _payload, *, create_if_missing: volume,
    )
    result = client_cli.delete_video_artifacts(
        {
            "remote_artifacts": [
                {
                    "remote_path": remote_path,
                    "filename": "result.mp4",
                    "size": len(expected),
                    "sha256": hashlib.sha256(expected).hexdigest(),
                }
            ]
        }
    )

    assert result["deleted"] == [remote_path]
    assert volume.deleted == [(f"/{remote_path}", False)]


def test_modal_client_preserves_changed_remote_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_path = "SOYA_VIDEO_OUTPUT/video_job/result.mp4"

    class FakeVolume:
        def read_file(self, _path: str):
            return [b"changed"]

        def remove_file(self, *_args, **_kwargs) -> None:
            raise AssertionError("해시가 달라진 원격 MP4를 삭제하면 안 됩니다.")

    monkeypatch.setattr(
        client_cli,
        "_video_volume",
        lambda _payload, *, create_if_missing: FakeVolume(),
    )
    result = client_cli.delete_video_artifacts(
        {
            "remote_artifacts": [
                {
                    "remote_path": remote_path,
                    "filename": "result.mp4",
                    "size": len(b"expected"),
                    "sha256": hashlib.sha256(b"expected").hexdigest(),
                }
            ]
        }
    )

    assert result["deleted_count"] == 0
    assert result["skipped_changed"] == [remote_path]


@pytest.mark.asyncio
async def test_service_downloads_video_without_deleting_before_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})
    video_bytes = b"verified-modal-mp4"
    artifact = {
        "remote_path": "SOYA_VIDEO_OUTPUT/video_job/result.mp4",
        "filename": "result.mp4",
        "size": len(video_bytes),
        "sha256": hashlib.sha256(video_bytes).hexdigest(),
        "node_id": "42",
    }
    observed_actions: list[str] = []

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def run_command(*_args, **kwargs):
        payload = kwargs["stdin_payload"]
        observed_actions.append(payload["action"])
        target = Path(payload["output_dir"]) / "result.mp4"
        target.write_bytes(video_bytes)
        return (
            0,
            json.dumps(
                {
                    "ok": True,
                    "result": {
                        "artifact": {**artifact, "path": str(target)},
                    },
                }
            ),
            "",
        )

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", run_command)

    downloaded, descriptor = await service.download_video_artifact(artifact)

    assert downloaded == video_bytes
    assert descriptor == artifact
    assert observed_actions == ["download_video_artifact"]
    assert service._load_video_delete_outbox() == []


@pytest.mark.asyncio
async def test_service_video_delete_records_outbox_before_exact_remote_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})
    observed_payloads: list[dict] = []

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def run_command(*_args, **kwargs):
        observed_payloads.append(dict(kwargs["stdin_payload"]))
        assert service._load_video_delete_outbox()
        return (
            0,
            json.dumps(
                {
                    "ok": True,
                    "result": {
                        "deleted_count": 1,
                        "skipped_changed": [],
                        "already_missing": [],
                    },
                }
            ),
            "",
        )

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", run_command)
    artifact = {
        "remote_path": "SOYA_VIDEO_OUTPUT/video_job/result.mp4",
        "filename": "result.mp4",
        "size": 6,
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }

    assert await service.delete_video_artifacts_after_spool([artifact]) is True
    assert observed_payloads[0]["action"] == "delete_video_artifacts"
    assert observed_payloads[0]["remote_artifacts"] == [
        {**artifact, "node_id": ""}
    ]
    assert service._load_video_delete_outbox() == []


@pytest.mark.asyncio
async def test_service_video_delete_failure_keeps_retry_outbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ModalService(tmp_path, lambda: {"modal_enabled": True})
    scheduled: list[bool] = []

    async def connected(_settings: ModalSettings) -> bool:
        return True

    async def failed_command(*_args, **_kwargs):
        return 1, json.dumps({"ok": False, "error": "network down"}), ""

    monkeypatch.setattr(service, "account_connected", connected)
    monkeypatch.setattr(service, "_run_command", failed_command)
    monkeypatch.setattr(
        service,
        "_schedule_video_delete_flush",
        lambda: scheduled.append(True),
    )
    artifact = {
        "remote_path": "SOYA_VIDEO_OUTPUT/video_job/result.mp4",
        "filename": "result.mp4",
        "size": 6,
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }

    assert await service.delete_video_artifacts_after_spool([artifact]) is False
    assert scheduled == [True]
    queued = service._load_video_delete_outbox()
    assert queued[0]["remote_artifacts"][0]["remote_path"] == artifact["remote_path"]
