"""Backup settings take effect when the next generated image is saved."""
import ast
import asyncio
import copy
from contextvars import ContextVar
import datetime
import glob
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from settings_illustration import allowed_illustration_path, register_illustration_config


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


class BackupRetentionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backups = self.root / "backups"
        self.backups.mkdir()
        self.config = {"backup_max_count": 500, "postprocess_enabled": True}

        async def write_config(request):
            self.config.update(await request.json())
            (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
            return web.json_response({"success": True})

        self.writer = AsyncMock(side_effect=write_config)
        server = SimpleNamespace(
            app=web.Application(), app_config=self.config,
            DEFAULT_CONFIG={"backup_max_count": 500}, CONFIG_FILE=str(self.root / "config.json"),
            handle_api_config=self.writer,
        )
        register_illustration_config(server, model_presets=Mock())
        self.client = TestClient(TestServer(server.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

        # Load the actual save/cleanup functions without starting the generation
        # server or loading its models. All filesystem effects stay in the fixture.
        self.engine = {
            "asyncio": asyncio, "copy": copy, "datetime": datetime, "glob": glob,
            "json": json, "os": os, "traceback": traceback, "Image": Image, "BytesIO": BytesIO,
            "app_config": self.config, "DEFAULT_MAX_BACKUP_IMAGES": 500,
            "WORKFLOW_BACKUP_DIR": str(self.backups), "get_backup_base_dir": lambda: str(self.backups),
            "_BACKUP_SNAPSHOT_UNSET": object(), "HAS_PIEXIF": False, "REMOTE_COMFY_TARGETS": (),
            "CURRENT_ILLUSTRATION_WORKFLOW_SNAPSHOT": ContextVar("test_backup_snapshot", default=None),
            "current_original_workflow": None, "current_api_workflow": None, "current_conversion_info": {},
            "multi_char_mask": SimpleNamespace(normalize_multi_char_snapshot=lambda _: None),
            "_normalize_illustration_visual_states": lambda *args, **kwargs: None,
            "get_backup_webp_quality": lambda: 80, "get_backup_webp_lossless": lambda: False,
            "illustration_flow": SimpleNamespace(record_backup=lambda _: None),
            "_invalidate_backup_filter_cache": Mock(), "notify_frontend": AsyncMock(),
        }
        for name in ("cleanup_backups", "save_backup"):
            node = copy.deepcopy(next(node for node in TREE.body if getattr(node, "name", None) == name))
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), self.engine)

    def seed_backups(self):
        for index in range(5):
            name = f"old_{index}"
            extension = ".webp" if index % 2 == 0 else ".avif"
            for suffix in (extension, ".json", ".txt", "_info.json", "_enhanced.txt", "_chat.txt", "_wildcard.json", "_submitted_prompt.json"):
                path = self.backups / (name + suffix)
                path.write_text("fixture", encoding="utf-8")
                os.utime(path, (100 + index, 100 + index))
            for folder, suffix in (("_raw", extension), ("_posters", ".webp")):
                (self.backups / folder).mkdir(exist_ok=True)
                (self.backups / folder / (name + suffix)).write_bytes(b"fixture")

    async def save_image(self, prompt_id):
        buffer = BytesIO()
        Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
        name, _ = await self.engine["save_backup"](
            buffer.getvalue(), prompt_id, "scene", "", provider="chansub", execution_target="local",
        )
        return name

    async def test_get_uses_default_and_gateway_allows_settings(self):
        self.config.pop("backup_max_count")
        response = await self.client.get("/api/illustration/config")
        self.assertEqual((await response.json())["config"]["backup_max_count"], 500)
        for method in ("GET", "POST"):
            self.assertTrue(allowed_illustration_path("/api/illustration/config", method))

    async def test_rejects_invalid_counts_without_writing(self):
        for value in (0, -1, 10001, 1.5, 2.0, True, False, None, "2", [], {}):
            with self.subTest(value=value):
                response = await self.client.post("/api/illustration/config", json={"backup_max_count": value})
                self.assertEqual(response.status, 400)
                self.assertIn("1~10,000", (await response.json())["error"])
        self.writer.assert_not_awaited()
        self.assertEqual(self.config["backup_max_count"], 500)

    async def test_lower_limit_waits_for_next_save_then_removes_all_excess_and_sidecars(self):
        self.seed_backups()
        before = set(self.backups.rglob("*"))
        response = await self.client.post("/api/illustration/config", json={"backup_max_count": 2})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["config"]["backup_max_count"], 2)
        self.writer.assert_awaited_once()
        self.assertTrue(json.loads((self.root / "config.json").read_text())["postprocess_enabled"])
        self.assertEqual(set(self.backups.rglob("*")), before)

        name = await self.save_image("newimage")
        self.assertEqual({path.stem for ext in ("*.webp", "*.avif") for path in self.backups.glob(ext)}, {"old_4", name})
        for index in range(4):
            self.assertFalse(list(self.backups.rglob(f"old_{index}*")))
        self.assertTrue((self.backups / "_raw/old_4.webp").is_file())
        self.engine["_invalidate_backup_filter_cache"].assert_called_once()
        self.engine["notify_frontend"].assert_awaited_once_with("backup_created", {"name": name})

        response = await self.client.post("/api/illustration/config", json={"backup_max_count": 1})
        self.assertEqual(response.status, 200)
        latest = await self.save_image("nextimage")
        self.assertEqual([path.stem for path in self.backups.glob("*.webp")], [latest])
        self.assertFalse(list(self.backups.rglob("old_4*")))
        self.assertFalse((self.backups / (name + ".json")).exists())

    async def test_increased_limit_preserves_backups_on_next_save(self):
        self.seed_backups()
        response = await self.client.post("/api/illustration/config", json={"backup_max_count": 10000})
        self.assertEqual(response.status, 200)
        await self.save_image("newimage")
        self.assertEqual(sum(len(list(self.backups.glob(ext))) for ext in ("*.webp", "*.avif")), 6)


if __name__ == "__main__":
    unittest.main()
