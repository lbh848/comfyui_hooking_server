import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


def _serve_json(*entries):
    web = {}
    tcp = {}
    for authority, target in entries:
        port = authority.rsplit(":", 1)[-1]
        tcp[port] = {"HTTPS": True}
        web[authority] = {"Handlers": {"/": {"Proxy": target}}}
    return {"TCP": tcp, "Web": web}


def test_serve_state_selects_only_manager_8189_mapping() -> None:
    data = _serve_json(
        ("device.example.ts.net:11434", "http://127.0.0.1:11434"),
        ("device.example.ts.net:6001", "http://127.0.0.1:6001"),
        ("device.example.ts.net:8189", "http://127.0.0.1:8189"),
    )

    active, url, conflict = server._parse_tailscale_serve_state(data)

    assert active is True
    assert url == "https://device.example.ts.net:8189"
    assert conflict is None


def test_serve_state_does_not_reuse_an_unrelated_existing_port() -> None:
    data = _serve_json(
        ("device.example.ts.net:11434", "http://127.0.0.1:11434"),
        ("device.example.ts.net:6001", "http://127.0.0.1:6001"),
    )

    assert server._parse_tailscale_serve_state(data) == (False, None, None)


def test_serve_state_reports_8189_conflict_without_claiming_it() -> None:
    data = _serve_json(
        ("device.example.ts.net:8189", "http://127.0.0.1:9000"),
    )

    active, url, conflict = server._parse_tailscale_serve_state(data)

    assert active is False
    assert url is None
    assert "다른 용도" in conflict
    assert "127.0.0.1:9000" in conflict


def test_tailscale_start_uses_exact_8189_serve_command(monkeypatch) -> None:
    states = iter([
        (False, None, None),
        (True, "https://device.example.ts.net:8189", None),
    ])
    calls = []

    async def fake_state():
        return next(states)

    async def fake_communicate(ts, *args, timeout):
        calls.append((ts, args, timeout))
        return b"", b"", 0

    async def passthrough(coroutine):
        return await coroutine

    monkeypatch.setattr(server, "_tailscale_bin", lambda: "tailscale-test")
    monkeypatch.setattr(server, "_tailscale_serve_state", fake_state)
    monkeypatch.setattr(
        server, "_communicate_tailscale_on_subprocess_loop", fake_communicate
    )
    monkeypatch.setattr(server, "_run_on_tunnel_subprocess_loop", passthrough)

    response = asyncio.run(server.handle_api_tailscale_start(None))

    assert response.status == 200
    assert json.loads(response.text)["url"] == "https://device.example.ts.net:8189"
    assert calls == [(
        "tailscale-test",
        ("serve", "--bg", "--https=8189", "http://127.0.0.1:8189"),
        20.0,
    )]


def test_tailscale_stop_only_turns_off_https_8189(monkeypatch) -> None:
    states = iter([
        (True, "https://device.example.ts.net:8189", None),
        (False, None, None),
    ])
    calls = []

    async def fake_state():
        return next(states)

    async def fake_communicate(ts, *args, timeout):
        calls.append((ts, args, timeout))
        return b"", b"", 0

    async def passthrough(coroutine):
        return await coroutine

    monkeypatch.setattr(server, "_tailscale_bin", lambda: "tailscale-test")
    monkeypatch.setattr(server, "_tailscale_serve_state", fake_state)
    monkeypatch.setattr(
        server, "_communicate_tailscale_on_subprocess_loop", fake_communicate
    )
    monkeypatch.setattr(server, "_run_on_tunnel_subprocess_loop", passthrough)

    response = asyncio.run(server.handle_api_tailscale_stop(None))

    assert response.status == 200
    assert json.loads(response.text)["status"] == "stopped"
    assert calls == [(
        "tailscale-test",
        ("serve", "--https=8189", "off"),
        10.0,
    )]
    assert all("reset" not in args for _ts, args, _timeout in calls)


def test_tailscale_stop_does_nothing_when_only_other_ports_exist(monkeypatch) -> None:
    async def fake_state():
        return False, None, None

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("unrelated Tailscale ports must not be changed")

    monkeypatch.setattr(server, "_tailscale_bin", lambda: "tailscale-test")
    monkeypatch.setattr(server, "_tailscale_serve_state", fake_state)
    monkeypatch.setattr(
        server, "_communicate_tailscale_on_subprocess_loop", fail_if_called
    )

    response = asyncio.run(server.handle_api_tailscale_stop(None))

    assert response.status == 200
    assert json.loads(response.text)["status"] == "stopped"


def test_tailscale_start_leaves_conflicting_8189_untouched(monkeypatch) -> None:
    conflict = "HTTPS 8189가 다른 대상으로 설정됨"

    async def fake_state():
        return False, None, conflict

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("conflicting user settings must not be changed")

    monkeypatch.setattr(server, "_tailscale_bin", lambda: "tailscale-test")
    monkeypatch.setattr(server, "_tailscale_serve_state", fake_state)
    monkeypatch.setattr(
        server, "_communicate_tailscale_on_subprocess_loop", fail_if_called
    )

    response = asyncio.run(server.handle_api_tailscale_start(None))

    assert response.status == 409
    payload = json.loads(response.text)
    assert payload["conflict"] is True
    assert payload["error"] == conflict
