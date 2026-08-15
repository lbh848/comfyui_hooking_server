from __future__ import annotations

import asyncio
import errno
import hashlib
import inspect
import json
from pathlib import Path
import socket
import socketserver
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import vast_backend.service as vast_service_module
import vast_backend.ssh_tunnel as ssh_tunnel_module
from vast_backend.client import VastApiError, VastClient, _rate_limit_delay
from vast_backend.favorites import VastMachineFavorites
from vast_backend.image_pull_progress import (
    build_pull_progress,
    parse_daemon_pull_states,
    parse_docker_hub_reference,
)
from vast_backend.model_sources import build_download_plan, save_mapping
from vast_backend.preflight import (
    actual_transfer_result,
    calculate_actual_transfer_estimate,
    calculate_transfer_totals,
)
from vast_backend.service import (
    ACCOUNT_STATUS_CACHE_SECONDS,
    MAX_BUILD_COST_USD,
    MIN_RUNTIME_CUDA_VERSION,
    MODELS_DONE_FLAG,
    NO_PROGRESS_WARNING_SECONDS,
    READY_FLAG,
    SshAuthenticationError,
    WATCHDOG_STATUS_MAX_AGE_SECONDS,
    VastService,
)
from vast_backend.ssh_tunnel import DEFAULT_LOCAL_PORT, ComfySshTunnel


@pytest.mark.asyncio
async def test_vast_lora_download_uses_project_temp_and_handles_cross_device_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "local-loras"
    service = VastService(
        tmp_path,
        lambda: {"lora_load_path": str(local_root)},
    )
    downloaded_paths: list[Path] = []
    removed_remote_paths: list[str] = []
    remote_exists = True

    class FakeSftp:
        def stat(self, remote_path: str):
            assert remote_path.endswith("/SOYA_INSTANCE_LORA/alice.safetensors")
            if not remote_exists:
                raise FileNotFoundError(remote_path)
            return SimpleNamespace(st_size=6, st_mtime=123)

        def get(self, remote_path: str, local_path: str) -> None:
            assert remote_path.endswith("/SOYA_INSTANCE_LORA/alice.safetensors")
            downloaded_paths.append(Path(local_path).resolve())
            Path(local_path).write_bytes(b"abcdef")

        def remove(self, remote_path: str) -> None:
            nonlocal remote_exists
            removed_remote_paths.append(remote_path)
            remote_exists = False

        def close(self) -> None:
            return None

    class FakeSsh:
        def open_sftp(self) -> FakeSftp:
            return FakeSftp()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        service,
        "_require_ssh_endpoint",
        lambda: ("ssh.example", 22, "unused-key"),
    )
    monkeypatch.setattr(service, "_ssh_connect", lambda *_args: FakeSsh())
    monkeypatch.setattr(
        service,
        "_remote_sha256_sync",
        lambda *_args: hashlib.sha256(b"abcdef").hexdigest(),
    )
    real_replace = vast_service_module.os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def replace_with_cross_device_once(source, target) -> None:
        source_path = Path(source).resolve()
        target_path = Path(target).resolve()
        replace_calls.append((source_path, target_path))
        if len(replace_calls) == 1:
            raise OSError(errno.EXDEV, "cross-device link")
        real_replace(source, target)

    monkeypatch.setattr(vast_service_module.os, "replace", replace_with_cross_device_once)
    remote_path = (
        "/root/ComfyUI/models/loras/SOYA_CHAR_LORA/"
        "SOYA_INSTANCE_LORA/alice.safetensors"
    )

    result = await service.download_lora_artifacts(
        [
            {
                "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
                "remote_path": remote_path,
                "size": 6,
            }
        ]
    )

    final_path = local_root / "SOYA_INSTANCE_LORA" / "alice.safetensors"
    assert final_path.read_bytes() == b"abcdef"
    assert result["artifacts"] == [
        {
            "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
            "local_path": str(final_path.resolve()),
            "status": "stored",
        }
    ]
    assert downloaded_paths[0].parents[1] == (tmp_path / "runtime" / "temp").resolve()
    assert replace_calls[0][0] == downloaded_paths[0]
    assert replace_calls[1][0].parent == final_path.parent.resolve()
    assert removed_remote_paths == [remote_path]
    assert list((tmp_path / "runtime" / "temp").iterdir()) == []


@pytest.mark.asyncio
async def test_vast_lora_delete_failure_is_persisted_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "local-loras"
    service = VastService(
        tmp_path,
        lambda: {"lora_load_path": str(local_root)},
    )
    service.launch["instance_id"] = 42
    service.launch["launch_id"] = "launch-42"
    remote_path = (
        "/root/ComfyUI/models/loras/SOYA_CHAR_LORA/"
        "SOYA_INSTANCE_LORA/alice.safetensors"
    )
    remote_exists = True
    allow_remove = False
    remove_attempts = 0

    class FakeSftp:
        def stat(self, path: str):
            assert path == remote_path
            if not remote_exists:
                raise FileNotFoundError(path)
            return SimpleNamespace(st_size=6, st_mtime=123)

        def get(self, path: str, local_path: str) -> None:
            assert path == remote_path
            Path(local_path).write_bytes(b"abcdef")

        def remove(self, path: str) -> None:
            nonlocal remote_exists, remove_attempts
            assert path == remote_path
            remove_attempts += 1
            if not allow_remove:
                raise OSError("temporary delete failure")
            remote_exists = False

        def close(self) -> None:
            return None

    class FakeSsh:
        def open_sftp(self) -> FakeSftp:
            return FakeSftp()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        service,
        "_require_ssh_endpoint",
        lambda: ("ssh.example", 22, "unused-key"),
    )
    monkeypatch.setattr(service, "_ssh_connect", lambda *_args: FakeSsh())
    monkeypatch.setattr(
        service,
        "_remote_sha256_sync",
        lambda *_args: hashlib.sha256(b"abcdef").hexdigest(),
    )

    result = await service.download_lora_artifacts(
        [
            {
                "relative_path": "SOYA_INSTANCE_LORA/alice.safetensors",
                "remote_path": remote_path,
                "size": 6,
                "mtime": 123,
            }
        ]
    )

    assert (local_root / "SOYA_INSTANCE_LORA" / "alice.safetensors").read_bytes() == b"abcdef"
    assert result["remote_delete_queued"] == [remote_path]
    assert service.lora_cleanup_status()["pending_count"] == 1
    assert remote_exists is True

    allow_remove = True
    monkeypatch.setattr(service, "ready_for_queue", lambda: True)
    await asyncio.wait_for(service._flush_lora_delete_outbox(), timeout=1)

    assert remote_exists is False
    assert remove_attempts == 2
    assert service.lora_cleanup_status()["pending_count"] == 0
    payload = json.loads(
        (tmp_path / "runtime" / "vast_lora_delete_outbox.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload == {"version": 1, "items": []}


def test_vast_ssh_keypair_is_created_and_loaded_from_key_directory(
    tmp_path: Path,
) -> None:
    service = VastService(tmp_path, lambda: {})

    private_path_text, first_public_key = service.ensure_ssh_keypair()
    private_path = Path(private_path_text)
    public_path = Path(str(private_path) + ".pub")
    first_private_key = private_path.read_bytes()

    second_private_path, second_public_key = service.ensure_ssh_keypair()

    assert private_path == tmp_path / "key" / "vast_ssh_key"
    assert public_path == tmp_path / "key" / "vast_ssh_key.pub"
    assert private_path.is_file()
    assert public_path.is_file()
    assert second_private_path == str(private_path)
    assert private_path.read_bytes() == first_private_key
    assert second_public_key == first_public_key
    assert not (tmp_path / "vast_ssh_key").exists()
    assert not (tmp_path / "vast_ssh_key.pub").exists()


@pytest.mark.asyncio
async def test_vast_instance_list_uses_v1_and_follows_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = VastClient("test-key")
    requests: list[dict[str, Any]] = []
    responses = [
        {"instances": [{"id": 10}], "next_token": "page-2"},
        {"instances": [{"id": 20}], "next_token": None},
    ]

    async def fake_request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        api_version: str = "v0",
    ) -> dict[str, Any]:
        requests.append(
            {
                "method": method,
                "path": path,
                "json_body": json_body,
                "query": dict(query or {}),
                "api_version": api_version,
            }
        )
        return responses.pop(0)

    monkeypatch.setattr(client, "_request", fake_request)

    assert await client.list_instances() == [{"id": 10}, {"id": 20}]
    assert [request["api_version"] for request in requests] == ["v1", "v1"]
    assert requests[0]["path"] == "/instances/"
    assert "after_token" not in requests[0]["query"]
    assert requests[1]["query"]["after_token"] == "page-2"


@pytest.mark.asyncio
async def test_vast_create_instance_uses_current_api_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = VastClient("test-key")
    observed: dict[str, Any] = {}

    async def fake_request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        api_version: str = "v0",
    ) -> dict[str, Any]:
        observed.update(
            method=method,
            path=path,
            json_body=json_body,
            query=query,
            api_version=api_version,
        )
        return {"success": True, "new_contract": 123}

    monkeypatch.setattr(client, "_request", fake_request)

    result = await client.create_instance(
        ask_id=99,
        image="example/image:latest",
        disk_gb=40,
        onstart_cmd="echo ready",
    )

    assert result["new_contract"] == 123
    assert observed["method"] == "PUT"
    assert observed["path"] == "/asks/99/"
    assert observed["api_version"] == "v0"
    body = observed["json_body"]
    assert body["onstart"] == "echo ready"
    assert "env" not in body
    assert body["runtype"] == "ssh"
    assert "onstart_cmd" not in body
    assert "ports" not in body
    assert "ssh_key" not in body


@pytest.mark.asyncio
async def test_vast_ssh_endpoints_match_current_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = VastClient("test-key")
    requests: list[tuple[str, str, dict[str, Any] | None]] = []

    async def fake_request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        api_version: str = "v0",
    ) -> dict[str, Any]:
        requests.append((method, path, json_body))
        return {"success": True}

    monkeypatch.setattr(client, "_request", fake_request)

    await client.register_account_ssh_key("ssh-rsa test")
    await client.attach_ssh_key(44, "ssh-rsa test")

    assert requests == [
        ("POST", "/ssh/", {"ssh_key": "ssh-rsa test"}),
        ("POST", "/instances/44/ssh/", {"ssh_key": "ssh-rsa test"}),
    ]


@pytest.mark.asyncio
async def test_vast_ssh_recovery_endpoints_list_and_detach_account_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = VastClient("test-key")
    requests: list[tuple[str, str, dict[str, Any] | None]] = []

    async def fake_request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        api_version: str = "v0",
    ) -> Any:
        requests.append((method, path, json_body))
        if method == "GET":
            return [{"id": 12, "public_key": "ssh-rsa test"}]
        return {"success": True}

    monkeypatch.setattr(client, "_request", fake_request)

    assert await client.list_account_ssh_keys() == [
        {"id": 12, "public_key": "ssh-rsa test"}
    ]
    await client.detach_ssh_key(44, 12)

    assert requests == [
        ("GET", "/ssh/", None),
        ("DELETE", "/instances/44/ssh/12/", None),
    ]


@pytest.mark.asyncio
async def test_vast_offer_search_does_not_require_a_direct_port_for_ssh_tunnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = VastClient("test-key")
    observed: dict[str, Any] = {}

    async def fake_request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        api_version: str = "v0",
    ) -> dict[str, Any]:
        observed.update(method=method, path=path, json_body=json_body)
        return {"offers": []}

    monkeypatch.setattr(client, "_request", fake_request)

    assert await client.search_offers() == []
    assert "direct_port_count" not in observed["json_body"]

    assert await client.search_offers(min_direct_port_count=2) == []
    assert observed["json_body"]["direct_port_count"] == {"gte": 2}


@pytest.mark.asyncio
async def test_vast_service_enforces_runtime_cuda_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = VastClient("test-key")
    observed_cuda: list[float] = []

    async def fake_search_offers(**kwargs: Any) -> list[dict[str, Any]]:
        observed_cuda.append(kwargs["min_cuda_version"])
        return []

    monkeypatch.setattr(client, "search_offers", fake_search_offers)
    service._client = client

    default_result = await service.offers()
    lowered_result = await service.offers(min_cuda_version=11.8)
    raised_result = await service.offers(min_cuda_version=13.0)

    assert observed_cuda == [12.8, 12.8, 13.0]
    assert default_result["min_cuda_version"] == MIN_RUNTIME_CUDA_VERSION
    assert lowered_result["min_cuda_version"] == MIN_RUNTIME_CUDA_VERSION
    assert raised_result["min_cuda_version"] == 13.0


@pytest.mark.asyncio
async def test_vast_instance_status_concurrent_callers_share_one_api_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = VastClient("test-key")
    api_calls = 0

    async def fake_get_instance(instance_id: int) -> dict[str, Any]:
        nonlocal api_calls
        api_calls += 1
        await asyncio.sleep(0.01)
        return {
            "id": instance_id,
            "actual_status": "loading",
            "status_msg": "Pulling image",
        }

    monkeypatch.setattr(client, "get_instance", fake_get_instance)
    service._client = client
    service.launch = service._new_launch_state(state="preparing", launch_id="poll")
    service.launch["instance_id"] = 321

    results = await asyncio.gather(
        *(
            service._get_instance_status(
                321, max_age_seconds=WATCHDOG_STATUS_MAX_AGE_SECONDS
            )
            for _ in range(8)
        )
    )

    assert api_calls == 1
    assert sum(1 for _info, refreshed in results if refreshed) == 1
    assert all(info["status_msg"] == "Pulling image" for info, _ in results)
    assert service.launch["instance_status"] == "loading"

    cached_info = service._instance_status_cache[321][1]
    service._instance_status_cache[321] = (
        time.monotonic() - WATCHDOG_STATUS_MAX_AGE_SECONDS - 1,
        cached_info,
    )
    _info, refreshed = await service._get_instance_status(
        321, max_age_seconds=WATCHDOG_STATUS_MAX_AGE_SECONDS
    )

    assert refreshed is True
    assert api_calls == 2


@pytest.mark.asyncio
async def test_vast_account_status_uses_sixty_second_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = VastClient("test-key")
    api_calls = 0

    async def fake_account() -> dict[str, Any]:
        nonlocal api_calls
        api_calls += 1
        await asyncio.sleep(0.01)
        return {"username": "tester", "credit": 12.34}

    monkeypatch.setattr(client, "account", fake_account)
    service._client = client

    results = await asyncio.gather(*(service.account_status() for _ in range(6)))

    assert api_calls == 1
    assert all(result["balance_usd"] == 12.34 for result in results)
    expires_at, payload = service._account_status_cache or (0.0, {})
    remaining = expires_at - time.monotonic()
    assert ACCOUNT_STATUS_CACHE_SECONDS - 1 <= remaining <= ACCOUNT_STATUS_CACHE_SECONDS

    service._account_status_cache = (time.monotonic() - 1, payload)
    await service.account_status()
    assert api_calls == 2


def test_vast_rate_limit_backoff_is_exponential_and_jittered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vast_backend.client.random.uniform", lambda _low, high: high
    )

    assert _rate_limit_delay("API requests too frequent", 0) == 6.0
    assert _rate_limit_delay("API requests too frequent", 1) == 11.0
    assert _rate_limit_delay('{"retry_after": 2}', 0) == 2.4


@pytest.mark.asyncio
async def test_vast_daemon_logs_use_async_result_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = VastClient("test-key")
    observed: dict[str, Any] = {}

    async def fake_request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        api_version: str = "v0",
    ) -> dict[str, Any]:
        observed.update(
            method=method,
            path=path,
            json_body=json_body,
            query=query,
            api_version=api_version,
        )
        return {"result_url": "https://example.test/result"}

    async def fake_poll(result_url: str) -> str:
        observed["result_url"] = result_url
        return "abc123def456: Download complete\n"

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "_poll_result_text", fake_poll)

    result = await client.get_instance_logs(321, daemon_logs=True, tail=250)

    assert result == "abc123def456: Download complete\n"
    assert observed == {
        "method": "PUT",
        "path": "/instances/request_logs/321/",
        "json_body": {"tail": "250", "daemon_logs": "true"},
        "query": None,
        "api_version": "v0",
        "result_url": "https://example.test/result",
    }


def test_vast_pull_progress_is_a_byte_weighted_lower_bound() -> None:
    assert parse_docker_hub_reference(
        "docker.io/bh848/soya-comfy-runtime@sha256:abc"
    ) == ("bh848/soya-comfy-runtime", "sha256:abc")
    logs = """
2026-08-14 15:04:08 UTC: aaaaaaaaaaaa: Already exists
bbbbbbbbbbbb: Pulling fs layer
cccccccccccc: Pulling fs layer
bbbbbbbbbbbb: Download complete
"""
    states = parse_daemon_pull_states(logs)
    progress = build_pull_progress(
        [
            {"digest": "sha256:" + "a" * 64, "size": 100},
            {"digest": "sha256:" + "b" * 64, "size": 300},
            {"digest": "sha256:" + "c" * 64, "size": 600},
        ],
        states,
    )

    assert states == {
        "aaaaaaaaaaaa": "available",
        "bbbbbbbbbbbb": "downloaded",
        "cccccccccccc": "pulling",
    }
    assert progress["total_bytes"] == 1000
    assert progress["confirmed_bytes"] == 400
    assert progress["pending_bytes"] == 600
    assert progress["minimum_percent"] == 40.0
    assert progress["confirmed_layers"] == 2
    assert progress["pending_layers"] == [
        {"id": "cccccccccccc", "size_bytes": 600, "state": "pulling"}
    ]
    assert progress["exact_progress_available"] is False


def test_vast_pull_progress_uses_latest_layer_state_after_retry() -> None:
    states = parse_daemon_pull_states(
        "dddddddddddd: Download complete\n"
        "2026-08-14 15:04:56 UTC: dddddddddddd: Pulling fs layer\n"
    )
    progress = build_pull_progress(
        [{"digest": "sha256:" + "d" * 64, "size": 900}],
        states,
    )

    assert states == {"dddddddddddd": "pulling"}
    assert progress["confirmed_bytes"] == 0
    assert progress["pending_bytes"] == 900
    assert progress["minimum_percent"] == 0.0


@pytest.mark.asyncio
async def test_vast_service_refreshes_image_pull_progress_without_claiming_stuck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = VastClient("test-key")

    async def fake_logs(
        instance_id: int, *, daemon_logs: bool = False, tail: int = 1000
    ) -> str:
        assert instance_id == 321
        assert daemon_logs is True
        assert tail == 1000
        return "aaaaaaaaaaaa: Already exists\nbbbbbbbbbbbb: Pulling fs layer\n"

    async def fake_manifest(
        image_reference: str,
        *,
        architecture: str = "amd64",
        os_name: str = "linux",
    ) -> dict[str, Any]:
        assert image_reference == "docker.io/example/runtime@sha256:index"
        assert architecture == "amd64"
        assert os_name == "linux"
        return {
            "layers": [
                {"digest": "sha256:" + "a" * 64, "size": 400},
                {"digest": "sha256:" + "b" * 64, "size": 600},
            ]
        }

    monkeypatch.setattr(client, "get_instance_logs", fake_logs)
    monkeypatch.setattr(client, "get_docker_hub_manifest_layers", fake_manifest)
    service._client = client
    service.launch = service._new_launch_state(state="preparing", launch_id="pull")
    service.launch["instance_id"] = 321
    service.launch["instance_status"] = "loading"
    service.launch["contract_started_at_epoch"] = time.time() - 300

    await service._refresh_image_pull_progress(
        321,
        {
            "id": 321,
            "actual_status": "loading",
            "image_uuid": "docker.io/example/runtime@sha256:index",
            "cpu_arch": "amd64",
        },
    )

    pull = service.launch["image_pull"]
    assert pull["minimum_percent"] == 40.0
    assert pull["confirmed_bytes"] == 400
    assert pull["pending_bytes"] == 600
    pull["last_observed_progress_at_epoch"] = (
        time.time() - NO_PROGRESS_WARNING_SECONDS - 1
    )
    status = service.launch_status()
    assert status["activity_unobserved"] is True
    assert status["image_pull"]["activity_state"] == "unobserved"
    assert "판별할 수 없습니다" in status["activity_unobserved_reason"]
    assert status["stuck"] is False

def test_vast_launch_status_exposes_cost_deadline_and_stale_warning(
    tmp_path: Path,
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    service.launch = service._new_launch_state(
        state="preparing",
        launch_id="launch-test",
        label="soya-vast-launch-test",
        hourly_price_usd=0.40,
    )
    service.launch["instance_id"] = 123
    service.launch["contract_started_at_epoch"] = time.time() - 300
    service.launch["last_progress_at_epoch"] = (
        time.time() - NO_PROGRESS_WARNING_SECONDS - 1
    )
    service.launch["current_step"] = "ssh"

    status = service.launch_status()

    assert 299 <= status["elapsed_seconds"] <= 301
    assert status["estimated_cost_usd"] == pytest.approx(0.033333, abs=0.00001)
    assert status["activity_unobserved"] is True
    assert "진행 신호" in status["activity_unobserved_reason"]
    assert status["stuck"] is False
    assert status["stuck_reason"] == ""
    assert status["auto_destroy_limit_name"] == "예상 빌드비 상한"
    # 상한 $0.25 ÷ 시간당 $0.40 × 3600s = 2250s, 300s 경과 → 1950s 남음
    assert 1949 <= status["auto_destroy_remaining_seconds"] <= 1951
    assert status["limits"] == {
        "max_build_cost_usd": MAX_BUILD_COST_USD,
        "no_progress_warning_seconds": NO_PROGRESS_WARNING_SECONDS,
    }


def test_vast_onstart_has_no_self_destroy_and_waits_ready_flag(
    tmp_path: Path,
) -> None:
    service = VastService(tmp_path, lambda: {})

    script = service._build_onstart()

    assert READY_FLAG in script
    assert "--request DELETE" not in script
    assert "CONTAINER_API_KEY" not in script
    assert "CONTAINER_ID" not in script


def test_vast_cmd_events_redact_known_api_keys(tmp_path: Path) -> None:
    service = VastService(tmp_path, lambda: {})
    service._log_secrets.add("secret-token-value")

    service._event(
        "remote", "download https://example.test/model?token=secret-token-value"
    )

    message = service.launch["events"][-1]["message"]
    assert "secret-token-value" not in message
    assert "<redacted>" in message


class _DestroyClient:
    def __init__(self, *, remains: bool = False, dph_total: float = 0.40) -> None:
        self.remains = remains
        self.dph_total = dph_total
        self.destroy_calls: list[int] = []

    async def destroy_instance(self, instance_id: int) -> dict[str, Any]:
        self.destroy_calls.append(instance_id)
        return {"success": True}

    async def list_instances(self) -> list[dict[str, Any]]:
        return [{"id": 321}] if self.remains else []

    async def get_instance(self, instance_id: int) -> dict[str, Any]:
        return {
            "id": instance_id,
            "actual_status": "running",
            "status_msg": "container running",
            "dph_total": self.dph_total,
        }


@pytest.mark.asyncio
async def test_vast_manual_destroy_cancels_launch_and_verifies_disappearance(
    tmp_path: Path,
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = _DestroyClient()
    service._client = client  # type: ignore[assignment]
    service.launch = service._new_launch_state(
        state="preparing", launch_id="manual", label="soya-vast-manual"
    )
    service.launch["instance_id"] = 321
    cancel_event = threading.Event()
    service._cancel_events["manual"] = cancel_event
    launch_task = asyncio.create_task(asyncio.Event().wait())
    service._launch_task = launch_task

    result = await service.destroy(321)
    await asyncio.gather(launch_task, return_exceptions=True)

    assert result["ok"] is True
    assert result["verified"] is True
    assert client.destroy_calls == [321]
    assert cancel_event.is_set() is True
    assert launch_task.cancelled() is True
    assert service.launch["state"] == "destroyed"
    assert service.launch["instance_id"] is None
    assert service.launch["destroyed_instance_id"] == 321
    assert any("소멸 확인 완료" in event["message"] for event in service.launch["events"])


@pytest.mark.asyncio
async def test_vast_destroy_failure_keeps_instance_id_and_critical_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = _DestroyClient(remains=True)
    service._client = client  # type: ignore[assignment]
    service.launch = service._new_launch_state(
        state="preparing", launch_id="failure", label="soya-vast-failure"
    )
    service.launch["instance_id"] = 321
    service._cancel_events["failure"] = threading.Event()

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    with pytest.raises(VastApiError, match="파괴를 확인하지 못했습니다"):
        await service.destroy(321)

    assert client.destroy_calls == [321, 321, 321]
    assert service.launch["state"] == "destroy_failed"
    assert service.launch["instance_id"] == 321
    assert service.launch["protection_state"] == "critical"


@pytest.mark.asyncio
async def test_vast_watchdog_auto_destroys_build_over_cost_cap(
    tmp_path: Path,
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = _DestroyClient()
    service._client = client  # type: ignore[assignment]
    service.launch = service._new_launch_state(
        state="preparing",
        launch_id="auto",
        label="soya-vast-auto",
        hourly_price_usd=0.40,
    )
    service.launch["instance_id"] = 321
    # 2400초 × $0.40/hr ≈ $0.27 → 상한 $0.25 초과
    service.launch["contract_started_at_epoch"] = time.time() - 2400
    service.launch["ssh_ready_at_epoch"] = time.time() - 2350
    service._cancel_events["auto"] = threading.Event()

    await service._watchdog_loop("auto")

    assert client.destroy_calls == [321]
    assert service.launch["state"] == "destroyed"
    assert service.launch["destroy_automatic"] is True
    assert "예상 빌드비" in service.launch["destroy_reason"]


@pytest.mark.asyncio
async def test_vast_watchdog_tolerates_long_build_under_cost_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _StopLoop(Exception):
        pass

    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = _DestroyClient(dph_total=0.01)
    service._client = client  # type: ignore[assignment]
    service.launch = service._new_launch_state(
        state="preparing",
        launch_id="long",
        label="soya-vast-long",
        hourly_price_usd=0.01,
    )
    service.launch["instance_id"] = 321
    # 24시간 경과(구 SSH 20분/빌드 60분 제한을 훨씬 초과)해도
    # 예상 빌드비 $0.24 < 상한 $0.25면 파괴하지 않는다.
    service.launch["contract_started_at_epoch"] = time.time() - 86400
    service._cancel_events["long"] = threading.Event()

    sleep_calls = {"count": 0}

    async def stop_after_second_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise _StopLoop

    monkeypatch.setattr(asyncio, "sleep", stop_after_second_sleep)

    with pytest.raises(_StopLoop):
        await service._watchdog_loop("long")

    assert client.destroy_calls == []
    assert service.launch["state"] == "preparing"


def test_vast_frontend_has_manual_destroy_guard_and_cmd_log() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'id="vast-launch-destroy-btn"' in html
    assert 'id="vast-launch-favorite-btn"' in html
    assert "이 머신을 즐겨찾기에 등록" in html
    assert "async function vastFavoriteInstance(instanceId, button = null)" in html
    assert "vastOption.textContent = 'VAST'" in html
    assert 'onclick="vastDestroyActive()"' in html
    assert 'id="vast-launch-terminal"' in html
    assert "launch.instance_status_msg" in html
    assert "launch.auto_destroy_remaining_seconds" in html
    assert 'id="vast-image-pull"' in html
    assert "imagePull.minimum_percent" in html
    assert "launch.activity_unobserved" in html
    assert 'id="vast-search-cuda" min="12.8"' in html
    assert 'value="12.8" title="런타임 이미지 요구사항' in html
    assert 'id="vast-preflight-card"' in html
    assert 'data-vast-preflight-key="ssh"' in html
    assert 'data-vast-preflight-key="download"' in html
    assert 'data-vast-preflight-key="upload"' in html
    assert "function vastRenderPreflight(launch)" in html
    assert "별도 측정용 트래픽은 사용하지 않습니다" in html


def test_vast_machine_favorites_are_persisted_idempotently(tmp_path: Path) -> None:
    favorites = VastMachineFavorites(tmp_path)
    instance = {
        "id": 47724573,
        "machine_id": 114484,
        "host_id": 430738,
        "gpu_name": "RTX 3090",
        "geolocation": "Wyoming, US",
        "reliability2": 0.995,
    }

    first = favorites.add_instance(instance)
    second = favorites.add_instance(instance)

    assert first["added"] is True
    assert second["added"] is False
    assert favorites.machine_ids() == {114484}
    payload = json.loads(favorites.path.read_text(encoding="utf-8"))
    assert payload["machines"][0]["source_instance_id"] == 47724573
    assert favorites.remove(114484)["removed"] is True
    assert favorites.machine_ids() == set()


@pytest.mark.asyncio
async def test_vast_service_favorites_current_instance_and_prioritizes_offers(
    tmp_path: Path,
) -> None:
    class FavoriteClient:
        async def list_instances(self):
            return [
                {
                    "id": 10,
                    "machine_id": 200,
                    "host_id": 20,
                    "gpu_name": "RTX 3090",
                }
            ]

        async def search_offers(self, **_kwargs):
            return [
                {
                    "id": 1,
                    "machine_id": 100,
                    "dph_total": 0.1,
                    "disk_name": "nvme",
                    "disk_bw": 1478.4,
                },
                {"id": 2, "machine_id": 200, "dph_total": 0.2},
            ]

    service = VastService(
        tmp_path,
        lambda: {"vast_enabled": True, "vast_api_key": "test-key"},
    )
    service._client = FavoriteClient()  # type: ignore[assignment]

    added = await service.favorite_instance(10)
    offers = await service.offers()

    assert added["favorite"]["machine_id"] == 200
    assert [offer["machine_id"] for offer in offers["offers"]] == [200, 100]
    assert offers["offers"][0]["favorite"] is True
    # 디스크 종류/대역폭은 원본 그대로 노출된다 (없으면 None/0).
    disk_offer = offers["offers"][1]
    assert disk_offer["disk_name"] == "nvme"
    assert disk_offer["disk_bw_mb_s"] == 1478.4
    assert offers["offers"][0]["disk_name"] is None
    assert offers["offers"][0]["disk_bw_mb_s"] == 0


def test_vast_actual_transfer_eta_uses_parallel_branch_bottleneck() -> None:
    download = actual_transfer_result(
        key="download",
        label="실제 모델 다운로드",
        status="running",
        completed_bytes=200,
        total_bytes=1000,
        total_known=True,
        seconds=4,
        bytes_per_second=100,
        detail="test",
    )
    upload = actual_transfer_result(
        key="upload",
        label="실제 로컬→Vast",
        status="running",
        completed_bytes=200,
        total_bytes=500,
        total_known=True,
        seconds=4,
        bytes_per_second=50,
        detail="test",
    )

    estimate = calculate_actual_transfer_estimate([download, upload])

    assert estimate["available"] is True
    assert estimate["download_seconds"] == 8.0
    assert estimate["upload_seconds"] == 6.0
    assert estimate["remaining_seconds"] == 8
    assert estimate["download_completed_bytes"] == 200
    assert estimate["upload_completed_bytes"] == 200


def test_vast_actual_transfer_eta_waits_for_each_active_branch() -> None:
    download = actual_transfer_result(
        key="download",
        label="실제 모델 다운로드",
        status="running",
        completed_bytes=100,
        total_bytes=1000,
        total_known=True,
        seconds=2,
        bytes_per_second=50,
        detail="test",
    )
    upload = actual_transfer_result(
        key="upload",
        label="실제 로컬→Vast",
        status="waiting",
        completed_bytes=0,
        total_bytes=500,
        total_known=True,
        seconds=0,
        bytes_per_second=0,
        detail="wait",
    )

    estimate = calculate_actual_transfer_estimate([download, upload])

    assert estimate["available"] is False
    assert estimate["remaining_seconds"] is None
    assert "업로드" in estimate["note"]


def test_vast_transfer_totals_route_sources_and_reject_unknown_size() -> None:
    totals = calculate_transfer_totals(
        {
            "models": [
                {"key": "hf", "size_bytes": 100, "source": {"source_type": "hf"}},
                {"key": "local", "size_bytes": 200, "source": {"source_type": "upload"}},
                {"key": "unknown", "size_bytes": 0, "source": {"source_type": "url"}},
            ]
        },
        [{"name": "lora", "size": 50}],
    )

    assert totals["download_bytes"] == 100
    assert totals["upload_bytes"] == 250
    assert totals["download_total_known"] is False
    assert totals["upload_total_known"] is True


def test_vast_actual_transfer_eta_accepts_branch_without_targets() -> None:
    download = actual_transfer_result(
        key="download",
        label="실제 모델 다운로드",
        status="done",
        completed_bytes=1000,
        total_bytes=1000,
        total_known=True,
        seconds=10,
        bytes_per_second=100,
        detail="done",
    )
    upload = actual_transfer_result(
        key="upload",
        label="실제 로컬→Vast",
        status="skipped",
        completed_bytes=0,
        total_bytes=0,
        total_known=True,
        seconds=0,
        bytes_per_second=0,
        detail="none",
    )

    estimate = calculate_actual_transfer_estimate([download, upload])

    assert estimate["available"] is True
    assert estimate["remaining_seconds"] == 0


def test_vast_readiness_check_authenticates_before_actual_transfer_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Channel:
        def recv_exit_status(self) -> int:
            return 0

    class _Stream:
        def __init__(self, value: bytes = b"") -> None:
            self.value = value
            self.channel = _Channel()

        def read(self) -> bytes:
            return self.value

    class _PreflightSsh:
        closed = False

        def exec_command(self, command: str):
            assert command == "printf '__SOYA_SSH_READY__'"
            return _Stream(), _Stream(b"__SOYA_SSH_READY__"), _Stream()

        def close(self) -> None:
            self.closed = True

    service = VastService(tmp_path, lambda: {})
    service.launch = service._new_launch_state(
        state="preparing", launch_id="preflight", label="soya-vast-preflight"
    )
    service.launch["contract_started_at_epoch"] = time.time() - 10
    service.launch["instance_running_at_epoch"] = time.time()
    service.launch["image_pull"].update(total_bytes=10 * 1024**3, observed_layers=10)
    ssh = _PreflightSsh()
    monkeypatch.setattr(service, "_ssh_connect", lambda *_args: ssh)
    model_plan = {
        "models": [
            {
                "filename": "model.safetensors",
                "size_bytes": 1000,
                "source": {
                    "source_type": "hf",
                    "repo_id": "org/model",
                    "hf_filename": "model.safetensors",
                },
            }
        ]
    }

    service._run_preflight("ssh.example", 1234, "unused-key", model_plan, [])

    assert service.launch["preflight"]["state"] == "ready"
    tests = {item["key"]: item for item in service.launch["preflight"]["tests"]}
    assert set(tests) == {"docker", "ssh"}
    assert tests["docker"]["mbps"] == 0
    assert "속도 계산에서 제외" in tests["docker"]["detail"]
    assert tests["ssh"]["status"] == "done"
    assert service.launch["ssh_ready_at_epoch"] > 0
    assert ssh.closed is True

    service._initialize_actual_transfer_tracking(model_plan, [])
    assert service.launch["preflight"]["state"] == "transferring"
    tests = {item["key"]: item for item in service.launch["preflight"]["tests"]}
    assert tests["download"]["status"] == "waiting"
    assert tests["upload"]["status"] == "skipped"

    launch_source = inspect.getsource(VastService._launch_inner)
    assert launch_source.index("self._run_preflight") < launch_source.index("self._initialize_actual_transfer_tracking")
    assert launch_source.index("self._initialize_actual_transfer_tracking") < launch_source.index("upload_task = asyncio.to_thread")


def test_vast_actual_download_rate_uses_first_observation_as_resume_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {})
    model_plan = {
        "models": [
            {
                "key": "checkpoints/model.safetensors",
                "size_bytes": 1000,
                "source": {"source_type": "url", "url": "https://example.test/model"},
            }
        ]
    }
    service._initialize_actual_transfer_tracking(model_plan, [])
    ticks = iter((100.0, 110.0, 120.0))
    monkeypatch.setattr("vast_backend.service.time.monotonic", lambda: next(ticks))

    # 기존 .part 400바이트는 이번 실행에서 받은 속도에 포함하면 안 된다.
    service._update_actual_transfer_progress("download", 400, status="running")
    first = {
        item["key"]: item for item in service.launch["preflight"]["tests"]
    }["download"]
    assert first["bytes_per_second"] == 0
    assert service.launch["preflight"]["estimate"]["available"] is False

    service._update_actual_transfer_progress("download", 600, status="running")
    second = {
        item["key"]: item for item in service.launch["preflight"]["tests"]
    }["download"]
    assert second["bytes_per_second"] == pytest.approx(20)
    assert service.launch["preflight"]["estimate"]["remaining_seconds"] == 20

    service._update_actual_transfer_progress("download", 1000, status="done")
    assert service.launch["preflight"]["state"] == "complete"
    assert service.launch["preflight"]["estimate"]["remaining_seconds"] == 0


def test_vast_readiness_check_does_not_continue_after_ssh_auth_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {})
    service.launch = service._new_launch_state(
        state="preparing", launch_id="auth-fail", label="soya-vast-auth-fail"
    )
    service.launch["contract_started_at_epoch"] = time.time() - 5
    service.launch["instance_running_at_epoch"] = time.time()

    def fail_connect(*_args: Any):
        raise VastApiError("Authentication failed")

    monkeypatch.setattr(service, "_ssh_connect", fail_connect)

    with pytest.raises(VastApiError, match="Authentication failed"):
        service._run_preflight("ssh.example", 1234, "unused-key", {"models": []}, [])

    assert service.launch["preflight"]["state"] == "failed"
    tests = {item["key"]: item for item in service.launch["preflight"]["tests"]}
    assert tests["ssh"]["status"] == "error"
    steps = {item["key"]: item for item in service.launch["steps"]}
    assert steps["preflight"]["state"] == "error"


def test_vast_ssh_connect_classifies_repeated_authentication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import paramiko

    service = VastService(tmp_path, lambda: {})
    connect_options: list[dict[str, Any]] = []

    class _RejectingClient:
        def set_missing_host_key_policy(self, _policy: object) -> None:
            return None

        def connect(self, *_args: object, **kwargs: Any) -> None:
            connect_options.append(dict(kwargs))
            raise paramiko.AuthenticationException("Authentication failed")

        def close(self) -> None:
            return None

    monkeypatch.setattr(paramiko, "SSHClient", _RejectingClient)
    monkeypatch.setattr(service, "_wait_sync", lambda _seconds: None)

    with pytest.raises(SshAuthenticationError, match="4회 연속 거부"):
        service._ssh_connect(
            "ssh2.vast.ai",
            18160,
            "unused-key",
            authentication_attempt_limit=4,
        )

    assert len(connect_options) == 4
    assert all(item["allow_agent"] is False for item in connect_options)
    assert all(item["look_for_keys"] is False for item in connect_options)


@pytest.mark.asyncio
async def test_vast_preflight_auth_failure_repairs_key_once_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {})
    service.launch = service._new_launch_state(
        state="preparing", launch_id="ssh-recovery", label="soya-vast-ssh-recovery"
    )
    service.launch["instance_id"] = 44
    preflight_calls: list[tuple[str, int, dict[str, Any]]] = []
    recoveries: list[tuple[int, str]] = []

    def fake_preflight(
        host: str,
        port: int,
        _private_key_path: str,
        _model_plan: dict[str, Any],
        _lora_files: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        preflight_calls.append((host, port, dict(kwargs)))
        if len(preflight_calls) == 1:
            raise SshAuthenticationError("Authentication failed")

    async def fake_recover(
        _client: object, instance_id: int, public_key: str
    ) -> tuple[str, int]:
        recoveries.append((instance_id, public_key))
        return "direct.example", 22022

    monkeypatch.setattr(service, "_run_preflight", fake_preflight)
    monkeypatch.setattr(service, "_recover_instance_ssh_key", fake_recover)

    endpoint = await service._run_preflight_with_ssh_recovery(
        object(),
        44,
        "ssh2.vast.ai",
        18160,
        "unused-key",
        "ssh-rsa AQIDBA== test",
        {"models": []},
        [],
    )

    assert endpoint == ("direct.example", 22022)
    assert recoveries == [(44, "ssh-rsa AQIDBA== test")]
    assert len(preflight_calls) == 2
    assert preflight_calls[0][2]["authentication_attempt_limit"] == 4
    assert preflight_calls[0][2]["defer_auth_failure"] is True
    assert preflight_calls[1] == ("direct.example", 22022, {})
    assert service._active_ssh_endpoint == (
        "direct.example",
        22022,
        "unused-key",
    )


@pytest.mark.asyncio
async def test_vast_ssh_key_recovery_detaches_and_reattaches_matching_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {})
    service.launch = service._new_launch_state(
        state="preparing", launch_id="key-reset", label="soya-vast-key-reset"
    )
    service.launch["instance_id"] = 44
    actions: list[tuple[Any, ...]] = []
    public_key = "ssh-rsa AQIDBA== soya-vast"

    class _RecoveryClient:
        async def list_account_ssh_keys(self) -> list[dict[str, Any]]:
            actions.append(("list",))
            return [{"id": 12, "public_key": public_key}]

        async def detach_ssh_key(self, instance_id: int, key_id: int) -> dict[str, Any]:
            actions.append(("detach", instance_id, key_id))
            return {"success": True}

        async def attach_ssh_key(self, instance_id: int, key: str) -> dict[str, Any]:
            actions.append(("attach", instance_id, key))
            return {"success": True}

        async def get_instance(self, instance_id: int) -> dict[str, Any]:
            actions.append(("status", instance_id))
            return {
                "id": instance_id,
                "actual_status": "running",
                "ssh_host": "root@ssh2.vast.ai",
                "ssh_port": 18160,
                "machine_id": 99,
                "host_id": 88,
            }

    async def no_sleep(_seconds: float) -> None:
        return None

    client = _RecoveryClient()
    service._client = client  # type: ignore[assignment]
    monkeypatch.setattr(service, "_client_or_raise", lambda: client)
    monkeypatch.setattr("vast_backend.service.asyncio.sleep", no_sleep)

    endpoint = await service._recover_instance_ssh_key(
        client, 44, public_key  # type: ignore[arg-type]
    )

    assert endpoint == ("ssh2.vast.ai", 18160)
    assert actions == [
        ("list",),
        ("status", 44),
        ("detach", 44, 12),
        ("attach", 44, public_key),
        ("status", 44),
    ]


def test_civitai_plan_has_download_url_even_without_key() -> None:
    plan = build_download_plan(
        [
            {
                "kind": "checkpoints",
                "filename": "model.safetensors",
                "size_bytes": 123,
            }
        ],
        {
            "sources": {
                "checkpoints/model.safetensors": {
                    "source_type": "civitai",
                    "civitai_version_id": 456,
                }
            }
        },
    )

    assert plan["items"][0]["source"] == {
        "source_type": "civitai",
        "civitai_version_id": 456,
        "url": "https://civitai.com/api/download/models/456",
    }


def test_download_plan_preserves_source_path_for_upload_fallback(tmp_path: Path) -> None:
    source_path = tmp_path / "models" / "loras" / "fixed.safetensors"
    plan = build_download_plan(
        [
            {
                "kind": "loras",
                "filename": "fixed.safetensors",
                "size_bytes": 123,
                "source_path": str(source_path),
            }
        ],
        {"sources": {}},
    )

    assert plan["items"][0]["source_path"] == str(source_path)
    assert plan["items"][0]["source"] == {"source_type": "upload"}


def test_vast_wizard_plan_includes_workflow_fixed_lora_from_manifest(
    tmp_path: Path,
) -> None:
    lora_path = tmp_path / "comfy" / "models" / "loras" / "fixed.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fixed-lora")
    workflow_path = tmp_path / "fixed-lora-workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {"lora_name": "fixed.safetensors"},
                }
            }
        ),
        encoding="utf-8",
    )
    manifest_path = (
        tmp_path / "comfy_installer" / "resources" / "install_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "relative_path": "models/loras/fixed.safetensors",
                        "url": "https://huggingface.co/example/fixed/resolve/main/fixed.safetensors",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    service = VastService(tmp_path, lambda: {})

    plan = service.wizard_plan(
        workflow_files=[{"path": str(workflow_path), "name": workflow_path.name}],
        lora_files=[],
    )

    assert len(plan["models"]) == 1
    item = plan["models"][0]
    assert item["key"] == "loras/fixed.safetensors"
    assert item["kind"] == "loras"
    assert item["filename"] == "fixed.safetensors"
    assert item["source_path"] == str(lora_path.resolve())
    assert item["source"] == {
        "source_type": "hf",
        "repo_id": "example/fixed",
        "hf_filename": "fixed.safetensors",
    }


def test_explicit_lora_upload_replaces_same_workflow_plan_item(tmp_path: Path) -> None:
    explicit_path = tmp_path / "fixed.safetensors"
    model_plan = {
        "models": [
            {"key": "loras/fixed.safetensors"},
            {"key": "checkpoints/base.safetensors"},
        ]
    }

    deduplicated = VastService._remove_explicit_lora_duplicates(
        model_plan,
        [{"name": "fixed.safetensors", "path": str(explicit_path)}],
    )

    assert deduplicated["models"] == [{"key": "checkpoints/base.safetensors"}]
    assert len(model_plan["models"]) == 2


def test_vast_mapping_backup_uses_deployment_safe_directory(tmp_path: Path) -> None:
    source_path = tmp_path / "vast_model_sources.json"
    source_path.write_text(
        json.dumps({"version": 1, "sources": {"old": {}}}), encoding="utf-8"
    )

    save_mapping(tmp_path, {"sources": {"new": {"source_type": "upload"}}})

    backups = list((tmp_path / "backups" / "vast_model_sources").glob("*.bak"))
    assert len(backups) == 1
    assert not (tmp_path / "요구사항").exists()
    assert json.loads(backups[0].read_text(encoding="utf-8"))["sources"] == {
        "old": {}
    }


class _FakeChannel:
    def recv_exit_status(self) -> int:
        return 0


class _FakeStream:
    def __init__(self, value: bytes = b"") -> None:
        self._value = value
        self.channel = _FakeChannel()

    def read(self) -> bytes:
        return self._value


class _FakeRemoteFile:
    def __init__(self, target: list[str]) -> None:
        self._target = target

    def __enter__(self) -> "_FakeRemoteFile":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def write(self, value: str) -> None:
        self._target.append(value)


class _FakeSftp:
    def __init__(self, script: list[str]) -> None:
        self._script = script

    def open(self, path: str, mode: str) -> _FakeRemoteFile:
        assert path == "/tmp/soya_download.sh"
        assert mode == "w"
        return _FakeRemoteFile(self._script)


class _FakeSsh:
    def __init__(self) -> None:
        self.script: list[str] = []
        self.closed = False

    def open_sftp(self) -> _FakeSftp:
        return _FakeSftp(self.script)

    def exec_command(self, command: str) -> tuple[_FakeStream, _FakeStream, _FakeStream]:
        if "setsid nohup" in command:
            return _FakeStream(), _FakeStream(), _FakeStream()
        return _FakeStream(), _FakeStream(b"__DONE__\n"), _FakeStream()

    def close(self) -> None:
        self.closed = True


def test_remote_download_is_resumable_atomic_and_clears_stale_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = VastService(tmp_path, lambda: {})
    ssh = _FakeSsh()
    monkeypatch.setattr(service, "_ssh_connect", lambda *_args: ssh)
    model_plan = {
        "models": [
            {
                "key": "checkpoints/model name.safetensors",
                "size_bytes": 100,
                "source": {
                    "source_type": "url",
                    "url": "https://example.test/model?id=1&name=test",
                },
            }
        ]
    }
    service._initialize_actual_transfer_tracking(model_plan, [])

    service._run_remote_downloads(
        "ssh.example", 1234, "unused-key", model_plan
    )

    script = "".join(ssh.script)
    assert "set -x" not in script
    assert f"rm -f {MODELS_DONE_FLAG} {MODELS_DONE_FLAG}.fail" in script
    assert "--continue-at -" in script
    assert "model name.safetensors.part" in script
    assert "expected_size=100" in script
    assert "date > /tmp/soya_models_done" in script
    assert ssh.closed is True
    tests = {item["key"]: item for item in service.launch["preflight"]["tests"]}
    assert tests["download"]["status"] == "done"
    assert tests["download"]["bytes"] == 100


def test_remote_download_rejects_missing_civitai_url(tmp_path: Path) -> None:
    service = VastService(tmp_path, lambda: {})
    model_plan = {
        "models": [
            {
                "key": "checkpoints/model.safetensors",
                "size_bytes": 100,
                "source": {
                    "source_type": "civitai",
                    "civitai_version_id": 456,
                },
            }
        ]
    }
    service._initialize_actual_transfer_tracking(model_plan, [])

    with pytest.raises(VastApiError, match="다운로드 URL"):
        service._run_remote_downloads(
            "ssh.example",
            1234,
            "unused-key",
            model_plan,
        )
    tests = {item["key"]: item for item in service.launch["preflight"]["tests"]}
    assert tests["download"]["status"] == "error"


@pytest.mark.asyncio
async def test_comfy_wait_fails_immediately_without_local_ssh_tunnel(
    tmp_path: Path,
) -> None:
    service = VastService(tmp_path, lambda: {})

    with pytest.raises(VastApiError, match="SSH 로컬 터널"):
        await service._start_comfy_and_wait(
            "ssh.example", 1234, "unused-key", ""
        )


class _EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            data = self.request.recv(65536)
            if not data:
                return
            self.request.sendall(data)


class _SocketTransport:
    def __init__(self) -> None:
        self.keepalive = 0

    def is_active(self) -> bool:
        return True

    def set_keepalive(self, seconds: int) -> None:
        self.keepalive = seconds

    def open_channel(
        self,
        _kind: str,
        destination: tuple[str, int],
        _origin: tuple[str, int],
    ) -> socket.socket:
        return socket.create_connection(destination, timeout=5)


class _TunnelSsh:
    def __init__(self, transport: _SocketTransport) -> None:
        self.transport = transport
        self.closed = False

    def get_transport(self) -> _SocketTransport:
        return self.transport

    def close(self) -> None:
        self.closed = True


def test_comfy_ssh_tunnel_forwards_and_closes() -> None:
    echo = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _EchoHandler)
    echo.daemon_threads = True
    echo_thread = threading.Thread(target=echo.serve_forever, daemon=True)
    echo_thread.start()
    transport = _SocketTransport()
    ssh = _TunnelSsh(transport)
    tunnel = ComfySshTunnel(
        ssh,
        remote_host="127.0.0.1",
        remote_port=int(echo.server_address[1]),
    )
    try:
        url = tunnel.start()
        assert tunnel.local_port == DEFAULT_LOCAL_PORT
        assert url == f"http://127.0.0.1:{tunnel.local_port}"
        with socket.create_connection(("127.0.0.1", tunnel.local_port), timeout=5) as client:
            client.sendall(b"tunnel-ok")
            assert client.recv(64) == b"tunnel-ok"
        assert transport.keepalive == 30
    finally:
        tunnel.close()
        echo.shutdown()
        echo.server_close()
        echo_thread.join(timeout=2)

    assert ssh.closed is True


def test_comfy_ssh_tunnel_uses_next_port_when_preferred_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    echo = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _EchoHandler)
    echo.daemon_threads = True
    echo_thread = threading.Thread(target=echo.serve_forever, daemon=True)
    echo_thread.start()
    transport = _SocketTransport()
    ssh = _TunnelSsh(transport)
    original_server = ssh_tunnel_module._TunnelServer
    attempts: list[int] = []

    def busy_once(server_address, *args, **kwargs):
        attempts.append(int(server_address[1]))
        if len(attempts) == 1:
            raise OSError(errno.EADDRINUSE, "test port already in use")
        return original_server(server_address, *args, **kwargs)

    monkeypatch.setattr(ssh_tunnel_module, "_TunnelServer", busy_once)
    tunnel = ComfySshTunnel(
        ssh,
        remote_host="127.0.0.1",
        remote_port=int(echo.server_address[1]),
        local_port=25000,
    )
    try:
        tunnel.start()
        assert attempts == [25000, 25001]
        assert tunnel.local_port == 25001
    finally:
        tunnel.close()
        echo.shutdown()
        echo.server_close()
        echo_thread.join(timeout=2)

    assert ssh.closed is True
