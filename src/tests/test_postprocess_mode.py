"""Output types select composition despite stale manual choices or UI changes."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import web
from modes.postprocess import get_postprocess_config, get_postprocess_mode


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def server_function(name, namespace):
    node = copy.deepcopy(next(node for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name))
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def config_for(output_mode, manual_mode="vn"):
    return {
        "postprocess_enabled": True,
        "illustration_context_toggles": {"illustration_output_mode": output_mode},
        "postprocess": {
            "mode": manual_mode,
            "subtitle": {"enabled": True, "font_size": 48},
            "manga_text": {"enabled": True, "font_size": 32, "font_id": "system"},
        },
    }


class AutomaticPostprocessTests(unittest.IsolatedAsyncioTestCase):
    def test_output_type_overrides_all_legacy_choices_without_modifying_config(self):
        for output_mode in ("illustration", "both", "original_asset", "composite", "composite_both", "manga", "manga_both"):
            expected = "manga_text" if output_mode.startswith("manga") else "subtitle"
            for manual_mode in ("vn", "bubble", "subtitle", "manga_text", "invalid"):
                with self.subTest(output=output_mode, saved=manual_mode):
                    config = config_for(output_mode, manual_mode)
                    before = copy.deepcopy(config)
                    settings = get_postprocess_config(config)
                    self.assertEqual(settings["mode"], expected)
                    self.assertEqual(settings["subtitle"]["font_size"], 48)
                    self.assertEqual(settings["manga_text"]["font_size"], 32)
                    self.assertEqual(config, before)
        self.assertEqual(get_postprocess_mode({}), "subtitle")

    def test_runtime_uses_captured_type_and_respects_enabled_switches(self):
        for output_mode, current_mode, expected in (
            ("illustration", "manga", "subtitle"),
            ("manga_both", "illustration", "manga_text"),
        ):
            with self.subTest(output=output_mode), patch("modes.font_assets.resolve_font", return_value=("", None)):
                config = config_for(current_mode)
                resolve = server_function("_get_illustration_postprocess_settings", {
                    "app_config": config, "traceback": traceback,
                })
                self.assertEqual(resolve("Bot", output_mode=output_mode)["_mode"], expected)
                config["postprocess_enabled"] = False
                self.assertIsNone(resolve("Bot", output_mode=output_mode))
                config["postprocess_enabled"] = True
                config["postprocess"][expected]["enabled"] = False
                self.assertIsNone(resolve("Bot", output_mode=output_mode))
                self.assertEqual(resolve("Bot", output_mode=output_mode, force=True)["_mode"], expected)

    async def test_settings_api_ignores_manual_mode_and_preserves_other_options(self):
        config = config_for("manga", "subtitle")
        save = Mock()
        handle = server_function("handle_api_postprocess_settings", {
            "web": web, "app_config": config, "save_config": save,
        })
        response = await handle(SimpleNamespace(
            path="/api/postprocess/settings", method="POST",
            json=AsyncMock(return_value={"mode": "subtitle", "manga_text": {"font_size": 42}}),
        ))
        self.assertEqual(response.status, 200)
        saved = json.loads(response.body)
        self.assertEqual(saved["mode"], "manga_text")
        self.assertEqual(saved["subtitle"]["font_size"], 48)
        self.assertEqual(saved["manga_text"]["font_size"], 42)
        save.assert_called_once_with(config)
        config["illustration_context_toggles"]["illustration_output_mode"] = "illustration"
        response = await handle(SimpleNamespace(path="/api/postprocess/settings", method="GET"))
        self.assertEqual(json.loads(response.body)["mode"], "subtitle")
        save.assert_called_once()

    async def test_deferred_image_keeps_its_type_when_global_output_changes(self):
        config = config_for("manga")
        entries = {"pending": {
            "_deferred_image_bytes": b"raw image",
            "_deferred_finalize": {"bot_name": "Bot", "illustration_output_mode": "illustration"},
        }}
        save = AsyncMock(return_value=("backup", b"composed image"))
        namespace = {
            "app_config": config, "traceback": traceback, "prompts": entries,
            "save_backup": save, "ws_connections": {},
        }
        server_function("_get_illustration_postprocess_settings", namespace)
        finalize = server_function("_finalize_deferred_illustration_prompt", namespace)
        self.assertEqual(await finalize("pending", 'Bot: "Hello"'), b"composed image")
        self.assertEqual(save.await_args.kwargs["postprocess_settings"]["_mode"], "subtitle")
        self.assertEqual(entries["pending"]["status"], "completed")
        self.assertNotIn("_deferred_finalize", entries["pending"])
