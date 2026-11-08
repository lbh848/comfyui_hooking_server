"""Locked logs from previous workers must not break startup or hide new logs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from settings_internal_lora import InternalLoraRuntime
from settings_program_logs import register_program_logs


class LockedLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_locked_previous_log_is_preserved_and_viewer_follows_new_session(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "src"
            source.mkdir()
            runtime = InternalLoraRuntime(source)
            previous = runtime.log_path
            previous.parent.mkdir(parents=True)
            previous.write_text("previous worker", encoding="utf-8")
            with patch.object(Path, "unlink", side_effect=PermissionError("Windows sharing violation")):
                await runtime.reset_log(None)
            self.assertNotEqual(runtime.log_path, previous)
            self.assertEqual(previous.read_text(), "previous worker")
            runtime.log_path.write_text("새 세션 로그", encoding="utf-8")
            app = web.Application()
            register_program_logs(app, source, "2", Mock(), illustration_runtime=runtime)
            async with TestClient(TestServer(app)) as client:
                headers = {"X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"}
                response = await client.get("/api/setting/program-logs?source=illustration", headers=headers)
                self.assertEqual((await response.json())["text"], "새 세션 로그")
                response = await client.get("/api/setting/program-logs/download?source=illustration", headers=headers)
                self.assertEqual(await response.text(), "새 세션 로그")
            await runtime.close(None)
            self.assertFalse(runtime.log_path.exists())
            self.assertEqual(previous.read_text(), "previous worker")

    async def test_open_log_viewer_does_not_block_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = InternalLoraRuntime(Path(directory) / "src")
            with patch.object(Path, "unlink", side_effect=PermissionError("Viewer holds log")):
                await runtime.close(None)


if __name__ == "__main__":
    unittest.main()
