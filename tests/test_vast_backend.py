from __future__ import annotations

import json
from pathlib import Path
import socket
import socketserver
import threading
from typing import Any

import pytest

from vast_backend.client import VastApiError, VastClient
from vast_backend.model_sources import build_download_plan, save_mapping
from vast_backend.service import MODELS_DONE_FLAG, VastService
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
