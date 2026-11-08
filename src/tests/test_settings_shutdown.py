"""Shutdown requires a deliberate local request and preserves aiohttp cleanup."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from settings_server import local_requests_only
from settings_shutdown import register_program_shutdown


class ProgramShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.exit_server = Mock()
        self.prepare_shutdown = AsyncMock()
        self.app = web.Application(middlewares=[local_requests_only])
        register_program_shutdown(self.app, "2", exit_server=self.exit_server, prepare_shutdown=self.prepare_shutdown)

        async def ok(request):
            return web.json_response({"ok": True})

        self.app.router.add_get("/api/setting/status", ok)
        self.app.router.add_post("/api/work", ok)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.headers = {"Host": "127.0.0.1:8201", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"}
        self.path = "/api/setting/shutdown"

    async def asyncTearDown(self):
        await self.client.close()

    async def test_requires_local_same_origin_request_and_current_schema(self):
        for headers, status in (
            ({"Host": "127.0.0.1:8201"}, 403),
            ({**self.headers, "Host": "outside.example"}, 403),
            ({**self.headers, "Origin": "https://outside.example"}, 403),
            ({**self.headers, "Sec-Fetch-Site": "cross-site"}, 403),
            ({**self.headers, "X-Comfy-Settings-Schema": "1"}, 409),
        ):
            response = await self.client.post(self.path, headers=headers, json={"confirm": True})
            self.assertEqual(response.status, status)
        self.exit_server.assert_not_called()

    async def test_requires_post_and_explicit_json_confirmation(self):
        self.assertEqual(await (await self.client.get(self.path, headers=self.headers)).json(), {"status": "idle"})
        self.prepare_shutdown.assert_not_called()
        self.assertEqual((await self.client.post(self.path, headers=self.headers, data="confirm")).status, 415)
        for body in ({}, {"confirm": False}, {"confirm": 1}, {"confirm": "true"}, [], None):
            response = await self.client.post(self.path, headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(body))
            self.assertEqual(response.status, 400)
        response = await self.client.post(self.path, headers={**self.headers, "Content-Type": "application/json"}, data="{")
        self.assertEqual(response.status, 400)
        self.exit_server.assert_not_called()

    async def test_acknowledges_before_exit_and_only_schedules_once(self):
        for _ in range(2):
            response = await self.client.post(self.path, headers=self.headers, json={"confirm": True})
            self.assertEqual(response.status, 202)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(await response.json(), {"status": "shutting_down"})
        self.exit_server.assert_not_called()
        response = await self.client.get(self.path, headers=self.headers)
        self.assertEqual(await response.json(), {"status": "ready"})
        await asyncio.sleep(.6)
        self.prepare_shutdown.assert_awaited_once_with()
        self.exit_server.assert_called_once_with()

    async def test_does_not_report_ready_or_exit_before_cleanup_finishes(self):
        cleanup_finished = asyncio.Event()
        self.prepare_shutdown.side_effect = cleanup_finished.wait
        await self.client.post(self.path, headers=self.headers, json={"confirm": True})
        await asyncio.sleep(.6)
        response = await self.client.get(self.path, headers=self.headers)
        self.assertEqual(await response.json(), {"status": "shutting_down"})
        self.exit_server.assert_not_called()
        cleanup_finished.set()
        await asyncio.sleep(0)
        self.assertEqual(await (await self.client.get(self.path, headers=self.headers)).json(), {"status": "ready"})
        await asyncio.sleep(.6)
        self.exit_server.assert_called_once_with()

    async def test_failed_cleanup_reports_error_and_can_be_retried(self):
        self.prepare_shutdown.side_effect = RuntimeError("fixture cleanup failure")
        await self.client.post(self.path, headers=self.headers, json={"confirm": True})
        self.assertEqual(await (await self.client.get(self.path, headers=self.headers)).json(), {"status": "error"})
        self.exit_server.assert_not_called()
        self.prepare_shutdown.side_effect = None
        await self.client.post(self.path, headers=self.headers, json={"confirm": True})
        self.assertEqual(await (await self.client.get(self.path, headers=self.headers)).json(), {"status": "ready"})
        self.assertEqual(self.prepare_shutdown.await_count, 2)

    async def test_blocks_new_work_but_keeps_status_available(self):
        self.assertEqual((await self.client.post("/api/work", headers=self.headers)).status, 200)
        await self.client.post(self.path, headers=self.headers, json={"confirm": True})
        self.assertEqual((await self.client.post("/api/work", headers=self.headers)).status, 503)
        self.assertEqual((await self.client.get("/api/setting/status", headers=self.headers)).status, 200)

    async def test_normal_cleanup_cancels_pending_exit(self):
        await self.client.post(self.path, headers=self.headers, json={"confirm": True})
        await self.client.close()
        await asyncio.sleep(.6)
        self.exit_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
