import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend_ws_manager import (
    FRONTEND_WS_REPLACED_CLOSE_CODE,
    FrontendWsConnectionManager,
)


FRONTEND_HTML = ROOT / "frontend" / "index.html"
SERVER_SOURCE = ROOT / "server.py"


class FakeWebSocket:
    def __init__(self, *, send_error: Exception | None = None):
        self.closed = False
        self.messages = []
        self.close_calls = []
        self.send_error = send_error

    async def send_json(self, message):
        if self.send_error is not None:
            raise self.send_error
        self.messages.append(message)

    async def close(self, *, code=1000, message=b""):
        self.closed = True
        self.close_calls.append({"code": code, "message": message})


class PausedReadyWebSocket(FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.ready_send_started = asyncio.Event()
        self.allow_ready_send = asyncio.Event()

    async def send_json(self, message):
        if message.get("type") == "connection_ready":
            self.ready_send_started.set()
            await self.allow_ready_send.wait()
        await super().send_json(message)


@pytest.mark.asyncio
async def test_new_frontend_connection_is_acknowledged_and_replaces_old_socket():
    manager = FrontendWsConnectionManager(heartbeat_interval=30, stale_timeout=15)
    old_ws = FakeWebSocket()
    new_ws = FakeWebSocket()

    assert await manager.register("old-client", old_ws) is True
    assert await manager.register("new-client", new_ws) is True

    assert list(manager.connections) == ["new-client"]
    ready = new_ws.messages[0]
    assert ready == {
        "type": "connection_ready",
        "data": {
            "client_id": "new-client",
            "heartbeat_interval_seconds": 30,
            "stale_timeout_seconds": 15,
        },
    }
    assert old_ws.messages[-1]["type"] == "connection_replaced"
    assert old_ws.messages[-1]["data"]["active_client_id"] == "new-client"
    assert old_ws.closed is True
    assert old_ws.close_calls[-1]["code"] == FRONTEND_WS_REPLACED_CLOSE_CODE


@pytest.mark.asyncio
async def test_failed_new_connection_ack_keeps_existing_socket_active(capsys):
    manager = FrontendWsConnectionManager(heartbeat_interval=30, stale_timeout=15)
    old_ws = FakeWebSocket()
    failed_ws = FakeWebSocket(send_error=RuntimeError("ready send failed"))
    await manager.register("old-client", old_ws)

    assert await manager.register("failed-client", failed_ws) is False

    assert list(manager.connections) == ["old-client"]
    assert old_ws.closed is False
    assert failed_ws.closed is True
    assert "connection_ready 송신 실패" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_concurrent_registration_is_serialized_without_ghost_socket():
    manager = FrontendWsConnectionManager(heartbeat_interval=30, stale_timeout=15)
    first_ws = PausedReadyWebSocket()
    second_ws = FakeWebSocket()

    first_task = asyncio.create_task(manager.register("first-client", first_ws))
    await first_ws.ready_send_started.wait()
    second_task = asyncio.create_task(manager.register("second-client", second_ws))
    await asyncio.sleep(0)
    first_ws.allow_ready_send.set()

    assert await first_task is True
    assert await second_task is True
    assert list(manager.connections) == ["second-client"]
    assert first_ws.closed is True
    assert first_ws.close_calls[-1]["code"] == FRONTEND_WS_REPLACED_CLOSE_CODE
    assert second_ws.closed is False


def test_unregistering_replaced_socket_does_not_remove_active_socket():
    manager = FrontendWsConnectionManager(heartbeat_interval=30, stale_timeout=15)
    old_ws = FakeWebSocket()
    active_ws = FakeWebSocket()
    manager.connections["active-client"] = {"ws": active_ws, "last_pong": 0}

    assert manager.unregister("active-client", old_ws) is False
    assert manager.connections["active-client"]["ws"] is active_ws
    assert manager.unregister("active-client", active_ws) is True
    assert manager.connections == {}


def test_frontend_led_uses_server_ack_and_liveness_instead_of_ready_state_only():
    source = FRONTEND_HTML.read_text(encoding="utf-8")

    assert "function isFrontendWsHealthy()" in source
    assert "if (isFrontendWsHealthy())" in source
    assert "case 'connection_ready':" in source
    assert "case 'connection_replaced':" in source
    assert "Date.now() - _wsLastServerActivity <= _wsHealthTimeoutMs" in source
    assert "socket !== frontendWs" in source
    assert "await waitForFrontendWsHealthy(5000)" in source


def test_server_registers_frontend_through_single_connection_manager():
    source = SERVER_SOURCE.read_text(encoding="utf-8")

    assert "frontend_ws_manager.register(client_id, ws)" in source
    assert "frontend_ws_manager.unregister(client_id, ws)" in source
    assert "frontend_ws_connections.clear()" not in source
