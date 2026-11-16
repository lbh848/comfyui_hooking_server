"""Concurrent upload staging preserves preflight validation and atomic publication."""
import asyncio
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer


class BotUploadTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        spec = importlib.util.spec_from_file_location("upload_bot_mode", modes / "bot_mode.py")
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modes": package}):
            spec.loader.exec_module(cls.module)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch.object(self.module, "BOT_DIR", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = self.module.BotMode()
        self.directory = self.root / "Bot" / "Alice"
        self.directory.mkdir(parents=True)

    async def preflight(self, files):
        request = SimpleNamespace(json=AsyncMock(return_value={
            "bot": "Bot", "items": [{"character": "Alice", "filename": name,
                "sha256": hashlib.sha256(content).hexdigest()} for name, content in files],
        }))
        response = await self.manager.handle_upload_preflight(request)
        self.assertEqual(response.status, 200)
        return json.loads(response.body)["preflight_id"]

    def upload(self, name, content, preflight_id="", index=0):
        request = SimpleNamespace(post=AsyncMock(return_value={
            "bot": "Bot", "character": "Alice",
            "file": SimpleNamespace(filename=name, file=BytesIO(content)),
            "preflight_id": preflight_id, "preflight_index": str(index),
        }))
        return self.manager.handle_upload_image(request)

    async def test_original_archives_stream_to_one_bot_and_reopen_without_duplicate_files(self):
        import illustration_bot_distribution
        package = ModuleType("modes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "modes")]
        data = {"bots": [{"name": "Bot", "characters": [{"name": "Alice"}]}, {"name": "Other", "characters": []}]}
        app = web.Application(client_max_size=1024, middlewares=[illustration_bot_distribution.bot_distribution_middleware])
        app.router.add_post("/api/bot_mode/archive", self.manager.handle_upload_bot_archive)
        app.router.add_get("/api/bot_mode/archive", self.manager.handle_get_bot_archive)
        app.router.add_get("/api/bot_mode/lore", self.manager.handle_get_character_lore)
        content = b"original archive bytes" * 10000
        entry = {"source": "ignored", "entry_id": "entry-1", "title": "Alice", "content": "Condition", "keys": ["Alice"]}

        async def publish(request, body, handler):
            self.assertEqual(body["bot_name"], "Bot")
            return await handler(request)

        with patch.dict(sys.modules, {"modes": package}), patch.object(self.module, "_load_bot_data", return_value=data), \
             patch.object(illustration_bot_distribution, "_handle_bot_distribution_write", side_effect=publish):
            async with TestClient(TestServer(app)) as client:
                for payload in (content, content, b"updated bytes"):
                    form = FormData()
                    form.add_field("lore_entries", json.dumps([entry]))
                    form.add_field("file", payload, filename="source.charx", content_type="application/octet-stream")
                    response = await client.post("/api/bot_mode/archive?bot_name=Bot&kind=bot", data=form)
                    self.assertEqual(response.status, 200, await response.text())
                    saved = (await response.json())["archive"]
                    self.assertEqual(saved["sha256"], hashlib.sha256(payload).hexdigest())
                    response = await client.get(f"/api/bot_mode/archive?bot_name=Bot&id={saved['id']}")
                    self.assertEqual(await response.read(), payload)
                    self.assertIn("attachment", response.headers["Content-Disposition"])
                    response = await client.get("/api/bot_mode/lore?bot_name=Bot")
                    state = await response.json()
                    self.assertEqual(len(state["archives"]), 1)
                    self.assertEqual(state["archives"][0]["lore_entries"][0]["source"], "봇/source.charx")
                    self.assertEqual(len(list((self.root / "Bot").glob(".bot-archive-*.charx"))), 1)
                response = await client.get("/api/bot_mode/lore?bot_name=Other")
                self.assertEqual((await response.json())["archives"], [])
                response = await client.get(f"/api/bot_mode/archive?bot_name=Other&id={saved['id']}")
                self.assertGreaterEqual(response.status, 400)
        self.assertFalse(list((self.root / "Bot").glob("*.upload")))

    async def test_staging_runs_concurrently_with_a_bound_and_preserves_each_file(self):
        files = [(f"image-{index}.png", bytes([index]) * 4096) for index in range(3)]
        preflight_id = await self.preflight(files)
        self.manager._upload_slots = asyncio.Semaphore(2)
        stage = self.module._stage_image_upload
        loop = asyncio.get_running_loop()
        two_started = asyncio.Event()
        release = threading.Event()
        lock = threading.Lock()
        active = peak = started = 0

        def held_stage(source, path):
            nonlocal active, peak, started
            with lock:
                active += 1
                started += 1
                peak = max(peak, active)
                if started == 2:
                    loop.call_soon_threadsafe(two_started.set)
            try:
                if not release.wait(5):
                    raise TimeoutError("Upload worker was not released")
                return stage(source, path)
            finally:
                with lock:
                    active -= 1

        with patch.object(self.module, "_stage_image_upload", side_effect=held_stage):
            work = asyncio.gather(*(self.upload(name, data, preflight_id, index)
                for index, (name, data) in enumerate(files)))
            try:
                await asyncio.wait_for(two_started.wait(), 3)
                self.assertEqual(started, 2)
                self.assertFalse(self.manager._lock.locked())
                self.assertFalse(work.done())
            finally:
                release.set()
                responses = await work
        self.assertEqual(peak, 2)
        self.assertTrue(all(response.status == 200 for response in responses))
        for name, content in files:
            self.assertEqual((self.directory / name).read_bytes(), content)
        self.assertFalse(list(self.directory.glob("*.upload")))
        self.assertNotIn(preflight_id, self.manager._upload_preflight_sessions)

    async def test_hash_mismatch_keeps_original_and_cleans_staging(self):
        path = self.directory / "image.png"
        path.write_bytes(b"original")
        preflight_id = await self.preflight([("image.png", b"expected")])
        response = await self.upload("image.png", b"different", preflight_id)
        self.assertGreaterEqual(response.status, 400)
        self.assertEqual(path.read_bytes(), b"original")
        self.assertFalse(list(self.directory.glob("*.upload")))
        self.assertIn(0, self.manager._upload_preflight_sessions[preflight_id]["pending"])

    async def test_failed_staging_or_publication_keeps_original_and_cleans_staging(self):
        path = self.directory / "image.png"
        path.write_bytes(b"original")
        preflight_id = await self.preflight([("image.png", b"replacement")])

        def fail_stage(_source, temp_path):
            Path(temp_path).write_bytes(b"partial")
            raise OSError("staging unavailable")

        for target, name, failure in (
            (self.module, "_stage_image_upload", fail_stage),
            (self.module.os, "replace", OSError("publication unavailable")),
        ):
            with self.subTest(operation=name), patch.object(target, name, side_effect=failure):
                response = await self.upload("image.png", b"replacement", preflight_id)
            self.assertGreaterEqual(response.status, 400)
            self.assertEqual(path.read_bytes(), b"original")
            self.assertFalse(list(self.directory.glob("*.upload")))
            self.assertIn(0, self.manager._upload_preflight_sessions[preflight_id]["pending"])

    async def test_direct_duplicate_upload_keeps_existing_filename(self):
        path = self.directory / "original.png"
        path.write_bytes(b"same image")
        response = await self.upload("copy.png", b"same image")
        result = json.loads(response.body)
        self.assertEqual(result["action"], "unchanged")
        self.assertEqual(result["filename"], "original.png")
        self.assertEqual(list(self.directory.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
