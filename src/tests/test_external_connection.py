"""External connection status must not hide one provider when the other fails."""
from __future__ import annotations

import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from external_connection import CONTROL_PREFIX, register_external_connection_control


class FakeServer:
    def __init__(self) -> None:
        self.app = web.Application()
        self.tailscale_available = True
        self.tailscale_error = ""
        self.tunnel_state = {"status": "stopped", "url": None}
        self.tailscale_state = {"status": "stopped", "url": None}

    def _tailscale_bin(self) -> str | None:
        return "tailscale" if self.tailscale_available else None

    async def handle_api_tunnel_status(self, request):
        return web.json_response(dict(self.tunnel_state))

    async def handle_api_tailscale_status(self, request):
        if self.tailscale_error:
            raise RuntimeError(self.tailscale_error)
        return web.json_response(dict(self.tailscale_state))

    async def handle_api_tunnel_start(self, request):
        return web.json_response({"status": "running", "url": "https://tunnel.example"})


class ExternalConnectionStatusTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = FakeServer()
        register_external_connection_control(self.server)
        self.client = TestClient(TestServer(self.server.app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_tailscale_failure_keeps_tunnel_status(self) -> None:
        self.server.tailscale_error = "daemon is not running"
        response = await self.client.get(CONTROL_PREFIX + "/status")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["tunnel"]["status"], "stopped")
        self.assertEqual(payload["tailscale"]["status"], "error")
        self.assertIn("daemon is not running", payload["tailscale"]["error"])

    async def test_missing_tailscale_binary_is_unavailable(self) -> None:
        self.server.tailscale_available = False
        response = await self.client.get(CONTROL_PREFIX + "/status")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["tailscale"]["status"], "unavailable")
        self.assertEqual(payload["tunnel"]["status"], "stopped")

    async def test_start_reports_other_provider_failure(self) -> None:
        self.server.tailscale_error = "daemon is not running"
        response = await self.client.post(CONTROL_PREFIX + "/tunnel/start")
        self.assertEqual(response.status, 503)
        payload = await response.json()
        self.assertIn("tailscale", payload["error"])


if __name__ == "__main__":
    unittest.main()
