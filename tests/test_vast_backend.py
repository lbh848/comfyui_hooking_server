from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import socketserver
import threading
import time
from typing import Any

import pytest

from vast_backend.client import VastApiError, VastClient
from vast_backend.model_sources import build_download_plan, save_mapping
from vast_backend.service import (
    BUILD_COMPLETE_FLAG,
    BUILD_TIMEOUT_SECONDS,
    MAX_BUILD_COST_USD,
    MODELS_DONE_FLAG,
    NO_PROGRESS_WARNING_SECONDS,
    SSH_WAIT_TIMEOUT_SECONDS,
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
    assert body["env"] == {"-p 8188:8188": "1"}
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
async def test_vast_offer_search_requires_an_application_port(
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
    assert observed["json_body"]["direct_port_count"] == {"gte": 1}


def test_vast_comfy_url_uses_public_ip_and_mapped_port(tmp_path: Path) -> None:
    service = VastService(tmp_path, lambda: {})

    assert service._proxy_url(
        {
            "machine_id": 777,
            "public_ipaddr": "203.0.113.10",
            "ports": {
                "8188/tcp": [{"HostIp": "0.0.0.0", "HostPort": "34567"}]
            },
        }
    ) == "http://203.0.113.10:34567"


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
    assert status["stuck"] is True
    assert "진행 변화" in status["stuck_reason"]
    assert status["auto_destroy_limit_name"] == "SSH 준비 제한"
    assert 899 <= status["auto_destroy_remaining_seconds"] <= 901
    assert status["limits"] == {
        "ssh_wait_seconds": SSH_WAIT_TIMEOUT_SECONDS,
        "build_seconds": BUILD_TIMEOUT_SECONDS,
        "max_build_cost_usd": MAX_BUILD_COST_USD,
        "no_progress_warning_seconds": NO_PROGRESS_WARNING_SECONDS,
    }


def test_vast_onstart_has_independent_build_ttl_without_logging_secret(
    tmp_path: Path,
) -> None:
    service = VastService(tmp_path, lambda: {})

    script = service._build_onstart()

    assert str(BUILD_TIMEOUT_SECONDS) in script
    assert BUILD_COMPLETE_FLAG in script
    assert "CONTAINER_API_KEY" in script
    assert "CONTAINER_ID" in script
    assert "--request DELETE" in script
    assert "echo ${CONTAINER_API_KEY}" not in script


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
    def __init__(self, *, remains: bool = False) -> None:
        self.remains = remains
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
            "dph_total": 0.40,
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
async def test_vast_watchdog_auto_destroys_build_over_total_limit(
    tmp_path: Path,
) -> None:
    service = VastService(tmp_path, lambda: {"vast_api_key": "test-key"})
    client = _DestroyClient()
    service._client = client  # type: ignore[assignment]
    service.launch = service._new_launch_state(
        state="preparing",
        launch_id="auto",
        label="soya-vast-auto",
        hourly_price_usd=0.01,
    )
    service.launch["instance_id"] = 321
    service.launch["contract_started_at_epoch"] = (
        time.time() - BUILD_TIMEOUT_SECONDS - 1
    )
    service.launch["ssh_ready_at_epoch"] = time.time() - BUILD_TIMEOUT_SECONDS
    service._cancel_events["auto"] = threading.Event()

    await service._watchdog_loop("auto")

    assert client.destroy_calls == [321]
    assert service.launch["state"] == "destroyed"
    assert service.launch["destroy_automatic"] is True
    assert "전체 빌드 제한" in service.launch["destroy_reason"]


def test_vast_frontend_has_manual_destroy_guard_and_cmd_log() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'id="vast-launch-destroy-btn"' in html
    assert 'onclick="vastDestroyActive()"' in html
    assert 'id="vast-launch-terminal"' in html
    assert "launch.instance_status_msg" in html
    assert "launch.auto_destroy_remaining_seconds" in html


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
async def test_comfy_wait_fails_immediately_without_public_port(tmp_path: Path) -> None:
    service = VastService(tmp_path, lambda: {})

    with pytest.raises(VastApiError, match="8188 외부 포트"):
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
