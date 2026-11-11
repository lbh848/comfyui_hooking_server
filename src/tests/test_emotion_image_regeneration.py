"""Emotion image regeneration re-reads the outfit and keeps the stored scene."""
import asyncio
import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import patch
import unittest


_MODES_PATCH = None


def setUpModule():
    # modes/__init__ starts the application; load the real modules without it.
    global _MODES_PATCH
    package = ModuleType("modes")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "modes")]
    _MODES_PATCH = patch.dict(sys.modules, {"modes": package})
    _MODES_PATCH.start()


def tearDownModule():
    if _MODES_PATCH is not None:
        _MODES_PATCH.stop()


def document():
    return {
        "schema": "soya.illustration.v2",
        "fields": {
            "ANIMA_CONTENT": "1girl, Aria, white dress, smile",
            "SDXL_CONTENT": "masterpiece, 1girl, Aria, white dress, smile",
            "IMG_W": 756,
            "IMG_H": 756,
            "SNAPREF": {"enabled": True, "references": [
                {"CHAR": "Aria", "asset": "representative", "face_asset": "face"}]},
        },
        "assets": {"representative": {"data": "a"}, "face": {"data": "f"}},
    }


class FakeRouter:
    def __init__(self):
        self.routes = {}

    def add_get(self, path, handler):
        self.routes[("GET", path)] = handler

    def add_post(self, path, handler):
        self.routes[("POST", path)] = handler


class FakeRequest:
    def __init__(self, method, body=None, query=None):
        self.method = method
        self._body = body or {}
        self.query = query or {}

    async def json(self):
        return self._body


class RegenerationHandlerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("settings_illustration_regeneration")

    def _server(self, response):
        calls = {}

        async def handle_api_regenerate(request, *, _body=None, _return_queue_result=False):
            calls["body"] = _body
            return dict(response)

        router = FakeRouter()
        server = SimpleNamespace(
            app=SimpleNamespace(router=router, on_cleanup=[]),
            WORKFLOW_BACKUP_DIR=str(self.folder),
            handle_api_regenerate=handle_api_regenerate,
            bot_mode=SimpleNamespace(
                _emotion_image_dir=lambda *args: (str(self.folder), ""),
                _emotion_flat_images=lambda folder: ["1.png"],
            ),
        )
        self.module.register_emotion_image_regeneration(server)
        return server, calls

    async def asyncSetUp(self):
        import tempfile
        self._temp = tempfile.TemporaryDirectory()
        self.folder = Path(self._temp.name)
        (self.folder / "1.png").write_bytes(b"old")
        self._write_patch = patch.object(
            self.module, "_write_emotion_image", lambda path, data: None)
        self._write_patch.start()

    async def asyncTearDown(self):
        self._write_patch.stop()
        self._temp.cleanup()

    async def test_regeneration_applies_current_outfit_tags_to_stored_scene(self):
        server, calls = self._server(
            {"image_bytes": b"\x89PNG\r\n\x1a\nnew", "backup_name": "backup2"})
        recorded = {}
        request = FakeRequest("POST", {
            "bot_name": "Bot", "char_name": "Aria", "profile": "p1",
            "outfit": "white dress", "emotion": "smile",
            "new_tags": ["black dress"],
        })
        with patch.object(self.module, "_find_emotion_asset_backup",
                          return_value="backup1"), \
             patch.object(self.module, "_load_emotion_submitted_prompt",
                          return_value=document()), \
             patch.object(self.module, "_load_emotion_prompt",
                          return_value=("positive", "sidecar", ["white dress"])), \
             patch.object(self.module, "_record_emotion_outfit_tags",
                          side_effect=lambda *args: recorded.update(
                              backup=args[1], tags=args[2])):
            response = await server.app.router.routes[
                ("POST", "/api/bot_mode/regenerate_emotion_image")](request)
            self.assertEqual(response.status, 202)
            for _ in range(20):
                if recorded:
                    break
                await asyncio.sleep(0.01)

        submitted = calls["body"]["illustration_prompt_json"]
        self.assertIn("black dress", submitted["fields"]["ANIMA_CONTENT"])
        self.assertIn("black dress", submitted["fields"]["SDXL_CONTENT"])
        self.assertNotIn("white dress", submitted["fields"]["ANIMA_CONTENT"])
        # Everything except the outfit tag text is replayed unchanged.
        self.assertEqual(submitted["fields"]["IMG_W"], 756)
        self.assertEqual(submitted["assets"], document()["assets"])
        self.assertEqual(submitted["fields"]["SNAPREF"],
                         document()["fields"]["SNAPREF"])
        # The new backup keeps the tag list so the next regeneration swaps the
        # right old tags.
        self.assertEqual(recorded.get("backup"), "backup2")
        self.assertEqual(recorded.get("tags"), ["black dress"])

    async def test_missing_tags_fall_back_to_the_saved_card_outfit(self):
        server, calls = self._server(
            {"image_bytes": b"\x89PNG\r\n\x1a\nnew", "backup_name": "backup2"})
        request = FakeRequest("POST", {
            "bot_name": "Bot", "char_name": "Aria", "profile": "p1",
            "outfit": "white dress", "emotion": "smile",
        })
        with patch.object(self.module, "_find_emotion_asset_backup",
                          return_value="backup1"), \
             patch.object(self.module, "_load_emotion_submitted_prompt",
                          return_value=document()), \
             patch.object(self.module, "_load_emotion_prompt",
                          return_value=("positive", "sidecar", ["white dress"])), \
             patch.object(self.module, "_emotion_card_outfit_tags",
                          return_value=["red dress"]) as read_card, \
             patch.object(self.module, "_record_emotion_outfit_tags"):
            response = await server.app.router.routes[
                ("POST", "/api/bot_mode/regenerate_emotion_image")](request)
            self.assertEqual(response.status, 202)
            for _ in range(20):
                if "body" in calls:
                    break
                await asyncio.sleep(0.01)

        read_card.assert_called_once_with(server, "Bot", "Aria", "p1", "white dress")
        self.assertIn("red dress",
                      calls["body"]["illustration_prompt_json"]["fields"]["ANIMA_CONTENT"])


class CardOutfitTagsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("settings_illustration_regeneration")

    def test_tags_come_from_the_card_and_legacy_label_fallback(self):
        card = {
            "id": "p1",
            "outfit_presets": [
                {"label": "white dress", "tags": [{"tag": "white dress"}]},
            ],
        }
        bot = {"name": "Bot", "characters": [{"name": "Aria", "character_cards": [card]}]}
        with patch.object(self.module.storage, "_load_bot_data",
                          return_value={"bots": [bot]}), \
             patch.object(self.module.storage, "set_bot_scope",
                          return_value="token"), \
             patch.object(self.module.storage, "reset_bot_scope"), \
             patch.object(self.module, "effective_character_cards",
                          side_effect=lambda character, _source: (
                              character.get("character_cards", []), None)):
            self.assertEqual(
                self.module._emotion_card_outfit_tags(
                    SimpleNamespace(), "Bot", "Aria", "p1", "white dress"),
                ["white dress"],
            )
            # A legacy folder whose label is the tag list falls back to it.
            self.assertEqual(
                self.module._emotion_card_outfit_tags(
                    SimpleNamespace(), "Bot", "Aria", "p1", "red dress, boots"),
                ["red dress", "boots"],
            )


if __name__ == "__main__":
    unittest.main()
