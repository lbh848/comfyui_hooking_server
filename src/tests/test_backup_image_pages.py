"""Backup search runs before pagination and keeps the server responsive."""
import ast
import asyncio
from contextlib import nullcontext
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aiohttp import web


class BackupImagePagesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        source = Path(__file__).resolve().parents[1] / "server.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        handler = next(node for node in tree.body if getattr(node, "name", None) == "handle_api_backups")
        self.reader = Mock(return_value=False)
        self.engine = {
            "asyncio": asyncio, "os": os, "web": web,
            "reschedule_queue": None,
            "get_backup_base_dir": lambda *args, **kwargs: "/fixture",
            "lighbd_service": SimpleNamespace(_load_lighbd_history=lambda: []),
            "illustration_quality_inspection": SimpleNamespace(TASK_KEY="test"),
            "_backup_image_kind": self.reader, "backup_clean_source_from_info": lambda *args: False,
        }
        exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"), self.engine)
        self.files = [f"/fixture/image_{index:05}.webp" for index in range(9137)]

    async def request(self, query):
        entries = [SimpleNamespace(path=path, name=Path(path).name, is_file=lambda: True,
                                   stat=lambda: SimpleNamespace(st_mtime=1)) for path in self.files]
        with patch.object(os, "scandir", return_value=nullcontext(entries)), \
             patch.object(os.path, "getmtime", return_value=1), \
             patch.object(os.path, "getsize", return_value=1), \
             patch.object(os.path, "exists", return_value=False):
            response = await self.engine["handle_api_backups"](SimpleNamespace(query=query))
        return json.loads(response.text)

    async def test_search_covers_the_whole_index_and_reads_at_most_24_images(self):
        result = await self.request({"search": "IMAGE 091", "offset": "0", "limit": "2000"})
        expected = {Path(name).stem for name in self.files if "091" in name}
        self.assertEqual(result["total"], len(expected))
        self.assertEqual(result["limit"], 24)
        self.assertEqual(self.reader.call_count, 24)
        self.assertTrue({row["name"] for row in result["backups"]} <= expected)
        self.assertTrue(result["has_more"])

    async def test_empty_search_and_negative_offsets_are_safe(self):
        result = await self.request({"search": "missing-file", "offset": "-20", "limit": "0"})
        self.assertEqual(result["offset"], 0)
        self.assertEqual(result["limit"], 1)
        self.assertEqual(result["total"], 0)
        self.assertFalse(result["has_more"])
        self.reader.assert_not_called()

    async def test_directory_scan_does_not_block_the_event_loop(self):
        entered, release = threading.Event(), threading.Event()
        def scan(directory):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("directory scan blocked the event loop")
            return nullcontext([])
        with patch.object(os, "scandir", side_effect=scan):
            task = asyncio.create_task(self.engine["handle_api_backups"](SimpleNamespace(query={})))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            finally:
                release.set()
            await task


if __name__ == "__main__":
    unittest.main()
