import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="Windows SelectorEventLoop subprocess 회귀 테스트",
)


def _run_with_selector(coroutine):
    loop = asyncio.SelectorEventLoop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


def test_tunnel_subprocess_runs_on_proactor_from_selector_loop(monkeypatch):
    worker = server._TunnelSubprocessLoop()
    monkeypatch.setattr(server, "_tunnel_subprocess_loop", worker)

    async def run_child():
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "print('tunnel-proactor-ok')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return (
            proc.returncode,
            out.decode("utf-8").strip(),
            err.decode("utf-8").strip(),
            type(asyncio.get_running_loop()).__name__,
        )

    async def call_from_server_loop():
        assert type(asyncio.get_running_loop()).__name__ == "_WindowsSelectorEventLoop"
        return await server._run_on_tunnel_subprocess_loop(run_child())

    try:
        result = _run_with_selector(call_from_server_loop())
    finally:
        worker.close()

    assert result == (0, "tunnel-proactor-ok", "", "ProactorEventLoop")


def test_tailscale_status_uses_tunnel_subprocess_loop(monkeypatch):
    worker = server._TunnelSubprocessLoop()
    monkeypatch.setattr(server, "_tunnel_subprocess_loop", worker)
    loop_names = []

    async def fake_communicate(ts, *args, timeout):
        loop_names.append(type(asyncio.get_running_loop()).__name__)
        assert ts == "tailscale-test"
        assert args == ("serve", "status", "--json")
        assert timeout == 10.0
        return (
            b'{"TCP":{"8189":{"HTTPS":true}},'
            b'"Web":{"fixed-name.example.ts.net:8189":'
            b'{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8189"}}}}}',
            b"",
            0,
        )

    monkeypatch.setattr(server, "_tailscale_bin", lambda: "tailscale-test")
    monkeypatch.setattr(
        server,
        "_communicate_tailscale_on_subprocess_loop",
        fake_communicate,
    )

    async def call_from_server_loop():
        return await server._tailscale_serve_state()

    try:
        active, url, conflict = _run_with_selector(call_from_server_loop())
    finally:
        worker.close()

    assert active is True
    assert url == "https://fixed-name.example.ts.net:8189"
    assert conflict is None
    assert loop_names == ["ProactorEventLoop"]
