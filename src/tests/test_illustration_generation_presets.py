"""Drawing preset persistence, mode isolation and API conflict handling."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from illustration_generation_presets import IllustrationGenerationPresets, register_generation_presets
from settings_illustration import allowed_illustration_path


class GenerationPresetTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        spec = importlib.util.spec_from_file_location("preset_test_bot", modes / "bot_mode.py")
        cls.bot = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modes": package}):
            spec.loader.exec_module(cls.bot)

    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = Path(temporary.name) / "config" / "settings.json"
        self.store = IllustrationGenerationPresets(self.config)
        patcher = patch.dict(sys.modules, {"modes.bot_mode": self.bot})
        patcher.start()
        self.addCleanup(patcher.stop)
        app = web.Application()
        register_generation_presets(app, self.config)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.settings = {
            "solo": {**self.bot.DEFAULT_ILLUST_SETTINGS, "img_w": 864, "seed": 10,
                     "anima_artist_preset": "작가", "snapref_strength": 0,
                     "hrf_activate": True, "face_lora_upscale_size": ""},
            "group": {"img_w": 1200, "img_h": 800, "seed": 20},
        }

    async def get(self, mode="illustration"):
        response = await self.client.get("/api/illustration/generation-presets", params={"generation_mode": mode})
        self.assertEqual(response.status, 200)
        return await response.json()

    async def post(self, action="create", mode="illustration", name="테스트 프리셋", **values):
        body = {"action": action, "generation_mode": mode, "name": name,
                "settings": self.settings, "revision": (await self.get(mode))["revision"], **values}
        return await self.client.post("/api/illustration/generation-presets", json=body)

    async def test_independent_modes_survive_restart_and_automatic_size_is_shared(self):
        self.assertEqual((await self.get())["presets"], [])
        self.assertFalse(self.store.path.exists())
        self.assertEqual((await self.post()).status, 200)
        automatic = deepcopy(self.settings)
        automatic["solo"].update(seed=99, img_w=960, hrf_activate=False)
        self.assertEqual((await self.post(mode="character_auto", settings=automatic)).status, 200)
        restarted = IllustrationGenerationPresets(self.config)
        normal = restarted.snapshot("illustration")["presets"][0]["settings"]
        auto = restarted.snapshot("character_auto")["presets"][0]["settings"]
        self.assertEqual(normal, self.settings)
        self.assertEqual(auto["group"], {key: auto["solo"][key] for key in ("img_w", "img_h", "seed")})
        self.assertEqual(auto["solo"]["seed"], 99)
        self.assertFalse(self.config.exists())

    async def test_overwrite_delete_and_duplicate_name_protection(self):
        await self.post()
        before = self.store.path.read_bytes()
        self.assertEqual((await self.post()).status, 400)
        self.assertEqual(self.store.path.read_bytes(), before)
        replacement = deepcopy(self.settings)
        replacement["solo"].update(seed=-1, hrf_activate=False, anima_artist_preset="")
        response = await self.post("update", settings=replacement)
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["presets"][0]["settings"], replacement)
        await self.post(mode="character_auto")
        response = await self.post("delete")
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["presets"], [])
        self.assertEqual(len((await self.get("character_auto"))["presets"]), 1)
        self.assertEqual((await self.post("delete")).status, 400)

    async def test_stale_revision_cannot_overwrite_or_delete(self):
        stale = (await self.get())["revision"]
        await self.post()
        before = self.store.path.read_bytes()
        for action in ("create", "update", "delete"):
            response = await self.post(action, revision=stale)
            self.assertEqual(response.status, 409)
            self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual((await self.post("update", revision=None)).status, 409)

    async def test_invalid_settings_and_requests_leave_file_unchanged(self):
        await self.post()
        before = self.store.path.read_bytes()
        for field, value in (("img_w", 0), ("seed", "42"), ("seed", -2), ("seed", True),
                             ("snapref_strength", 3), ("hrf_activate", "false"),
                             ("snapref_preprocess", "invalid"), ("face_lora_upscale_size", -1)):
            settings = deepcopy(self.settings)
            settings["solo"][field] = value
            with self.subTest(field=field, value=value):
                self.assertEqual((await self.post("update", settings=settings)).status, 400)
        for values in ({"settings": []}, {"settings": {"solo": {}, "group": {}}}, {"name": " "},
                       {"name": "a" * 101}, {"name": "line\nbreak"}, {"generation_mode": "unknown"}):
            self.assertEqual((await self.post("update", **values)).status, 400)
        response = await self.client.post("/api/illustration/generation-presets", json=[])
        self.assertEqual(response.status, 400)
        self.assertEqual(self.store.path.read_bytes(), before)

    async def test_loading_into_other_bot_preserves_other_mode_and_unknown_fields(self):
        await self.post()
        saved = (await self.get())["presets"][0]["settings"]
        data = {"bots": [{"name": "OtherBot", "custom": "keep",
                          "illust_settings_solo": {"future_setting": "keep"},
                          "character_auto_settings_solo": {"seed": 77}}]}
        with patch.object(self.bot, "_load_bot_data", side_effect=lambda: deepcopy(data)), \
             patch.object(self.bot, "_save_bot_data", side_effect=lambda value: data.update(value)):
            for profile in ("solo", "group"):
                body = {"bot_name": "OtherBot", "generation_mode": "illustration", "profile": profile,
                        "illust_settings": saved[profile]}
                response = await self.bot.handle_update_illust_settings(SimpleNamespace(json=AsyncMock(return_value=body)))
                self.assertEqual(response.status, 200)
        target = data["bots"][0]
        self.assertEqual(target["custom"], "keep")
        self.assertEqual(target["character_auto_settings_solo"], {"seed": 77})
        self.assertEqual(target["illust_settings_solo"]["future_setting"], "keep")
        self.assertEqual(target["illust_settings_group"], saved["group"])
        for key, value in saved["solo"].items():
            self.assertEqual(target["illust_settings_solo"][key], value)

    async def test_gateway_allows_only_expected_methods(self):
        for method in ("GET", "HEAD", "POST"):
            self.assertTrue(allowed_illustration_path("/api/illustration/generation-presets", method))
        self.assertFalse(allowed_illustration_path("/api/illustration/generation-presets", "DELETE"))
        self.assertFalse(allowed_illustration_path("/api/illustration/generation-presets/anything", "POST"))
        response = await self.client.get("/api/illustration/generation-presets?generation_mode=unknown")
        self.assertEqual(response.status, 400)

    async def test_corrupt_file_is_never_replaced_by_a_save(self):
        await self.post()
        self.store.path.write_text('{"version": 999}', encoding="utf-8")
        before = self.store.path.read_bytes()
        response = await self.client.post("/api/illustration/generation-presets", json={
            "action": "create", "generation_mode": "illustration", "name": "new", "settings": self.settings,
        })
        self.assertEqual(response.status, 400)
        self.assertEqual(self.store.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
