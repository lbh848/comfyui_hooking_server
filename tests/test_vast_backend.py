from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import socket
import socketserver
import threading
import time
from typing import Any

import pytest

from vast_backend.client import VastApiError, VastClient, _rate_limit_delay
from vast_backend.image_pull_progress import (
    build_pull_progress,
    parse_daemon_pull_states,
    parse_docker_hub_reference,
)
from vast_backend.model_sources import build_download_plan, save_mapping
from vast_backend.preflight import (
    calculate_transfer_estimate,
    parse_curl_speed_probe,
    speed_result,
)
from vast_backend.service import (
    ACCOUNT_STATUS_CACHE_SECONDS,
    MAX_BUILD_COST_USD,
    MIN_RUNTIME_CUDA_VERSION,
    MODELS_DONE_FLAG,
    NO_PROGRESS_WARNING_SECONDS,
    READY_FLAG,
    WATCHDOG_STATUS_MAX_AGE_SECONDS,
    VastService,
)
from vast_backend.ssh_tunnel import ComfySshTunnel


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
    assert 'data-vast-preflight-key="cloudflare"' in html
    assert 'data-vast-preflight-key="huggingface"' in html
    assert 'data-vast-preflight-key="upload"' in html
    assert "function vastRenderPreflight(launch)" in html


def test_vast_curl_preflight_accepts_completed_and_timeout_samples() -> None:
    complete = parse_curl_speed_probe(
        "__SOYA_SPEED__:33554432:4.000:206:8388608",
        exit_code=0,
        key="huggingface",
        label="Hugging Face",
        detail="선택 모델",
    )
    partial = parse_curl_speed_probe(
        "__SOYA_SPEED__:10485760:15.000:200:699050",
        exit_code=28,
        key="cloudflare",
        label="Cloudflare",
        detail="제한 시간 도달",
    )

    assert complete["status"] == "done"
    assert complete["mbps"] == pytest.approx(67.11)
    assert partial["status"] == "partial"
    assert partial["bytes"] == 10 * 1024**2
    assert partial["mbps"] == pytest.approx(5.59)


def test_vast_curl_preflight_rejects_http_error() -> None:
    with pytest.raises(ValueError, match="HTTP 오류"):
        parse_curl_speed_probe(
            "__SOYA_SPEED__:123:0.500:403:246",
            exit_code=22,
            key="huggingface",
            label="Hugging Face",
            detail="실패",
        )


def test_vast_transfer_eta_uses_parallel_branch_bottleneck() -> None:
    cloudflare = speed_result(
        key="cloudflare",
        label="Cloudflare",
        transferred_bytes=50 * 1024**2,
        seconds=1,
        detail="test",
    )
    huggingface = speed_result(
        key="huggingface",
        label="Hugging Face",
        transferred_bytes=100 * 1024**2,
        seconds=1,
        detail="test",
    )
    upload = speed_result(
        key="upload",
        label="로컬→Vast",
        transferred_bytes=20 * 1024**2,
        seconds=1,
        detail="test",
    )
    estimate = calculate_transfer_estimate(
        {
            "models": [
                {"size_bytes": 1024**3, "source": {"source_type": "hf"}},
                {"size_bytes": 512 * 1024**2, "source": {"source_type": "url"}},
                {"size_bytes": 2 * 1024**3, "source": {"source_type": "upload"}},
            ]
        },
        [{"size": 1024**3}],
        [cloudflare, huggingface, upload],
    )

    assert estimate["available"] is True
    assert estimate["download_seconds"] == pytest.approx(20.5, abs=0.1)
    assert estimate["upload_seconds"] == pytest.approx(153.6, abs=0.1)
    assert estimate["remaining_seconds"] == 154


def test_vast_preflight_uses_selected_huggingface_model() -> None:
    url, requested_bytes, detail = VastService._preflight_huggingface_target(
        {
            "models": [
                {
                    "filename": "small model.safetensors",
                    "size_bytes": 8 * 1024**2,
                    "source": {
                        "source_type": "hf",
                        "repo_id": "org/model",
                        "hf_filename": "folder/small model.safetensors",
                    },
                }
            ]
        }
    )

    assert url == (
        "https://huggingface.co/org/model/resolve/main/"
        "folder/small%20model.safetensors"
    )
    assert requested_bytes == 8 * 1024**2
    assert "small model.safetensors" in detail


def test_vast_automatic_preflight_runs_before_parallel_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _PreflightSsh:
        closed = False

        def close(self) -> None:
            self.closed = True

    service = VastService(tmp_path, lambda: {})
    service.launch = service._new_launch_state(
        state="preparing", launch_id="preflight", label="soya-vast-preflight"
    )
    service.launch["contract_started_at_epoch"] = time.time() - 10
    service.launch["ssh_ready_at_epoch"] = time.time()
    ssh = _PreflightSsh()
    monkeypatch.setattr(service, "_ssh_connect", lambda *_args: ssh)

    def fake_curl(_ssh: Any, *, key: str, label: str, detail: str, **_kwargs: Any):
        return speed_result(
            key=key,
            label=label,
            transferred_bytes=50 * 1024**2,
            seconds=1,
            detail=detail,
        )

    monkeypatch.setattr(service, "_run_curl_speed_probe", fake_curl)
    monkeypatch.setattr(
        service,
        "_run_sftp_speed_probe",
        lambda _ssh: speed_result(
            key="upload",
            label="로컬→Vast",
            transferred_bytes=25 * 1024**2,
            seconds=1,
            detail="test",
        ),
    )
    model_plan = {
        "models": [
            {
                "filename": "model.safetensors",
                "size_bytes": 500 * 1024**2,
                "source": {
                    "source_type": "hf",
                    "repo_id": "org/model",
                    "hf_filename": "model.safetensors",
                },
            }
        ]
    }

    service._run_preflight("ssh.example", 1234, "unused-key", model_plan, [])

    assert service.launch["preflight"]["state"] == "complete"
    assert service.launch["preflight"]["estimate"]["remaining_seconds"] == 10
    assert ssh.closed is True
    launch_source = inspect.getsource(VastService._launch_inner)
    assert launch_source.index("self._run_preflight") < launch_source.index(
        "upload_task = asyncio.to_thread"
    )


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

    service._run_remote_downloads(
        "ssh.example",
        1234,
        "unused-key",
        {
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
        },
    )

    script = "".join(ssh.script)
    assert "set -x" not in script
    assert f"rm -f {MODELS_DONE_FLAG} {MODELS_DONE_FLAG}.fail" in script
    assert "--continue-at -" in script
    assert "model name.safetensors.part" in script
    assert "expected_size=100" in script
    assert "date > /tmp/soya_models_done" in script
    assert ssh.closed is True


def test_remote_download_rejects_missing_civitai_url(tmp_path: Path) -> None:
    service = VastService(tmp_path, lambda: {})

    with pytest.raises(VastApiError, match="다운로드 URL"):
        service._run_remote_downloads(
            "ssh.example",
            1234,
            "unused-key",
            {
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
            },
        )


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
