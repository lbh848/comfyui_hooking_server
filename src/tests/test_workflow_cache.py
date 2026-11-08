"""Distribution protection stays on originals, never on runtime cache copies."""
import ast
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import tempfile
import traceback
import unittest
from unittest.mock import patch

from aiohttp import web

from workflow_cache import copy_workflow_to_cache


SOURCE = Path(__file__).resolve().parents[1]
WORKFLOW = {"1": {"class_type": "TestNode", "inputs": {"value": "new"}}}


def protect(path):
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


class WorkflowCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "distribution.json"
        self.source.write_text(json.dumps(WORKFLOW), encoding="utf-8")
        protect(self.source)
        self.destination = self.root / "cache" / self.source.name

    def test_copy_is_writable_and_repeated_refresh_keeps_original_protected(self):
        for _ in range(2):
            copy_workflow_to_cache(self.source, self.destination)
            self.assertEqual(self.destination.read_bytes(), self.source.read_bytes())
            self.assertTrue(self.destination.stat().st_mode & stat.S_IWUSR)
            self.assertFalse(self.source.stat().st_mode & stat.S_IWUSR)
            self.destination.write_text("stale cache", encoding="utf-8")
        self.assertEqual(json.loads(self.source.read_text(encoding="utf-8")), WORKFLOW)

    def test_refresh_repairs_an_existing_read_only_copy(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old cache")
        protect(self.destination)
        copy_workflow_to_cache(self.source, self.destination)
        self.assertEqual(self.destination.read_bytes(), self.source.read_bytes())
        self.assertTrue(self.destination.stat().st_mode & stat.S_IWUSR)
        self.assertFalse(self.source.stat().st_mode & stat.S_IWUSR)

    def test_failed_replace_preserves_previous_cache_and_its_attributes(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old cache")
        protect(self.destination)
        with patch("workflow_cache.os.replace", side_effect=PermissionError("file in use")):
            with self.assertRaises(PermissionError):
                copy_workflow_to_cache(self.source, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"old cache")
        self.assertFalse(self.destination.stat().st_mode & stat.S_IWUSR)
        self.assertFalse(self.source.stat().st_mode & stat.S_IWUSR)
        self.assertEqual(list(self.destination.parent.glob("*.part")), [])

    def test_source_in_cache_folder_is_read_without_changing_its_protection(self):
        copy_workflow_to_cache(self.source, self.source)
        self.assertFalse(self.source.stat().st_mode & stat.S_IWUSR)
        self.assertEqual(json.loads(self.source.read_text(encoding="utf-8")), WORKFLOW)


class WorkflowPreparationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.modes = {name: importlib.import_module(f"modes.{name}") for name in ("asset_mode", "asset_tool_mode")}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "distribution.json"
        self.source.write_text(json.dumps(WORKFLOW), encoding="utf-8")
        protect(self.source)
        self.cache = self.root / "mode_workflow"
        self.cache.mkdir()
        self.work = self.root / "current_work"
        self.work.mkdir()

    async def test_asset_generation_and_both_tag_analysis_slots_refresh_protected_cache(self):
        for filename, class_name, fallback in [("asset_mode", "AssetMode", False),
                                               ("asset_tool_mode", "AssetToolMode", False),
                                               ("asset_tool_mode", "AssetToolMode", True)]:
            module = self.modes[filename]
            with self.subTest(mode=filename, fallback=fallback), \
                    patch.object(module, "MODE_WORKFLOW_DIR", str(self.cache)), \
                    patch.object(module, "CURRENT_MODE_WORK_DIR", str(self.work)):
                mode = getattr(module, class_name)()
                mode.workflow_source_path = str(self.source)
                mode.fallback_workflow_source_path = str(self.source)
                destination = self.cache / self.source.name
                if destination.exists():
                    destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
                destination.write_bytes(b"stale cache")
                protect(destination)
                if filename == "asset_mode":
                    self.assertTrue(await mode.update_asset_workflow())
                    self.assertEqual(mode._asset_api_workflow, WORKFLOW)
                else:
                    self.assertTrue(await mode.update_workflow(use_fallback=fallback))
                    self.assertEqual(getattr(mode, "_fallback_api_workflow" if fallback else "_api_workflow"), WORKFLOW)
                self.assertTrue(destination.stat().st_mode & stat.S_IWUSR)
                self.assertFalse(self.source.stat().st_mode & stat.S_IWUSR)

    async def test_illustration_generation_and_reload_read_protected_original_directly(self):
        server = SOURCE / "server.py"
        tree = ast.parse(server.read_text(encoding="utf-8"))
        # Run the actual handlers without importing and starting the server.
        namespace = {"os": os, "json": json, "datetime": datetime, "web": web, "traceback": traceback, "print": lambda *_: None,
                     "CURRENT_WORK_DIR": str(self.work), "WORKFLOW_DIR": str(self.cache),
                     "get_comfy_workflow_source_path": lambda: str(self.source),
                     "get_workflow_file": lambda _profile=None: str(self.source),
                     "compute_file_hash": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     "load_stored_hash": lambda: "old-hash", "save_stored_hash": lambda _hash: None,
                     "is_api_format": lambda value: "class_type" in value["1"]}
        for name in ("update_workflow_if_needed", "handle_api_reload_workflow"):
            node = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(server), "exec"), namespace)
        legacy_copy = self.cache / self.source.name
        legacy_copy.write_bytes(b"old protected copy")
        protect(legacy_copy)
        self.assertTrue(await namespace["update_workflow_if_needed"]())
        for _ in range(2):
            response = await namespace["handle_api_reload_workflow"](None)
            self.assertEqual(response.status, 200)
            self.assertTrue(json.loads(response.body)["success"])
            self.assertEqual(json.loads((self.work / "workflow_api.json").read_text(encoding="utf-8")), WORKFLOW)
            self.assertFalse(self.source.stat().st_mode & stat.S_IWUSR)
            self.assertEqual(legacy_copy.read_bytes(), b"old protected copy")


if __name__ == "__main__":
    unittest.main()
