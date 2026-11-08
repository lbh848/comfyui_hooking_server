"""Drawing modes, automatic character sizing and background presets."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


class IllustrationGenerationModeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        spec = importlib.util.spec_from_file_location("generation_mode_bot", modes / "bot_mode.py")
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modes": package}):
            spec.loader.exec_module(cls.module)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "bot.json"
        self.original = {
            "name": "TestBot", "custom": "preserve",
            "illust_settings_solo": {"anima_artist_preset": "Original", "img_w": 864, "img_h": 1024, "seed": 11},
            "illust_settings_group": {"img_w": 1200, "img_h": 800, "seed": 22},
        }
        self.write({"bots": [self.original]})
        for target, options in (
            ("_load_bot_data", {"side_effect": self.read}),
            ("_save_bot_data", {"side_effect": self.write}),
        ):
            patcher = patch.object(self.module, target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")

    async def get_settings(self, profile="solo", mode=None):
        query = {"bot_name": "TestBot", "profile": profile}
        if mode is not None:
            query["generation_mode"] = mode
        return await self.module.handle_get_illust_settings(SimpleNamespace(query=query))

    async def save_settings(self, fields, profile="solo", mode=None):
        body = {"bot_name": "TestBot", "profile": profile, "illust_settings": fields}
        if mode is not None:
            body["generation_mode"] = mode
        return await self.module.handle_update_illust_settings(SimpleNamespace(json=AsyncMock(return_value=body)))

    async def test_old_clients_keep_existing_settings_and_legacy_fallback(self):
        response = await self.get_settings()
        self.assertEqual(json.loads(response.body)["img_w"], 864)
        self.assertEqual(json.loads(response.body)["generation_mode"], "illustration")
        await self.save_settings({"seed": 33})
        saved = self.read()["bots"][0]
        self.assertEqual(saved["illust_settings_solo"]["seed"], 33)
        self.assertEqual(saved["illust_settings_group"], self.original["illust_settings_group"])
        self.assertNotIn("character_auto_settings_solo", saved)
        legacy = {"illust_settings": {"seed": 44, "snapref_strength": 0}}
        self.assertEqual(self.module.resolve_illust_settings(legacy)["seed"], 44)
        self.assertEqual(self.module.resolve_illust_settings(legacy)["snapref_strength"], 1)
        self.assertEqual(self.module.resolve_illust_settings(legacy, generation_mode="character_auto")["seed"], -1)

    async def test_automatic_settings_round_trip_without_changing_illustration(self):
        before = self.path.read_bytes()
        response = await self.get_settings(mode="character_auto")
        defaults = json.loads(response.body)
        self.assertEqual(defaults["generation_mode"], "character_auto")
        self.assertEqual(defaults["seed"], -1)
        self.assertEqual(defaults["anima_artist_preset"], "")
        self.assertEqual(self.path.read_bytes(), before)
        result = await self.save_settings({"anima_artist_preset": "Automatic", "img_w": 960, "seed": 55}, mode="character_auto")
        self.assertEqual(result.status, 200)
        loaded = json.loads((await self.get_settings(mode="character_auto")).body)
        self.assertEqual(loaded["anima_artist_preset"], "Automatic")
        self.assertEqual(loaded["img_w"], 960)
        self.assertEqual(loaded["seed"], 55)
        group = json.loads((await self.get_settings("group", "character_auto")).body)
        self.assertEqual(group["anima_artist_preset"], "Automatic")
        self.assertEqual(group["img_w"], 960)
        self.assertEqual(group["seed"], 55)
        saved = self.read()["bots"][0]
        for key, value in self.original.items():
            self.assertEqual(saved[key], value)
        automatic_before = deepcopy(saved)
        await self.save_settings({"seed": 66, "anima_artist_preset": "Updated"})
        saved = self.read()["bots"][0]
        for key in ("character_auto_settings_solo", "character_auto_settings_group"):
            self.assertEqual(saved[key], automatic_before[key])

    async def test_auto_size_is_shared_and_illustration_group_size_is_preserved(self):
        await self.save_settings({"seed": 55, "img_w": 960}, mode="character_auto")
        await self.save_settings({"seed": 77, "img_w": 1280, "hrf_size": 2}, "group", "character_auto")
        solo = json.loads((await self.get_settings(mode="character_auto")).body)
        group = json.loads((await self.get_settings("group", "character_auto")).body)
        self.assertEqual((solo["seed"], solo["img_w"], solo["hrf_size"]), (77, 1280, 2))
        self.assertEqual((group["seed"], group["img_w"], group["hrf_size"]), (77, 1280, 2))
        normal = json.loads((await self.get_settings("group")).body)
        self.assertEqual((normal["seed"], normal["img_w"], normal["hrf_size"]), (22, 1200, 1.5))

    async def test_background_presets_round_trip_and_reach_both_renderers(self):
        await self.save_settings({"anima_background_preset": "Studio", "sdxl_background_preset": "Garden"}, mode="character_auto")
        settings = json.loads((await self.get_settings(mode="character_auto")).body)
        self.assertEqual(settings["anima_background_preset"], "Studio")
        self.assertEqual(settings["sdxl_background_preset"], "Garden")
        await self.save_settings({"anima_background_preset": "Garden", "sdxl_background_preset": "Studio"})
        normal = json.loads((await self.get_settings()).body)
        self.assertEqual(normal["anima_background_preset"], "Garden")
        self.assertEqual(normal["sdxl_background_preset"], "Studio")
        group = json.loads((await self.get_settings("group")).body)
        self.assertEqual(group["anima_background_preset"], "Garden")
        self.assertEqual(group["sdxl_background_preset"], "Studio")
        self.assertEqual(json.loads((await self.get_settings(mode="character_auto")).body), settings)
        old = {"illust_settings_solo": {"anima_background_preset": "Studio"}}
        self.assertEqual(self.module.resolve_illust_settings(old)["anima_background_preset"], "Studio")
        package = ModuleType("modes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "modes")]
        with patch.dict(sys.modules, {"modes": package}):
            from modes.illust_prompt_builder import IllustPromptBuilder
            from modes.chansub_prompt_builder import ChansubPromptBuilder
            from illustration_json import illustration_fields
            tags = {
                "artist_presets": {"Original": ["artist:example"]},
                "background_presets": {"Studio": ["white backdrop"], "Garden": ["flower garden"]},
            }
            for value, anima, sdxl in ((settings, "white backdrop", "flower garden"), (normal, "flower garden", "white backdrop")):
                prompt = IllustPromptBuilder().build_positive_prompt("day", "", "", [], {"characters": []}, tags, value, "TestBot")
                fields = illustration_fields(prompt, "")["fields"]
                self.assertIn(anima, fields["ANIMA_CONTENT"])
                self.assertIn(sdxl, fields["SDXL_CONTENT"])
                self.assertNotIn(sdxl, fields["ANIMA_CONTENT"])
                self.assertNotIn(anima, fields["SDXL_CONTENT"])
                for workflow, expected, other in (("anima", anima, sdxl), ("sdxl", sdxl, anima)):
                    flat = ChansubPromptBuilder().build_positive_prompt("day", "", "", tags, {**value, "chansub_workflow_type": workflow})
                    self.assertIn(expected, flat)
                    self.assertNotIn(other, flat)
        await self.save_settings({"anima_background_preset": "", "sdxl_background_preset": ""})
        cleared = json.loads((await self.get_settings()).body)
        self.assertEqual(cleared["anima_background_preset"], "")
        self.assertEqual(cleared["sdxl_background_preset"], "")
        self.assertEqual(json.loads((await self.get_settings(mode="character_auto")).body), settings)

    async def test_background_mode_settings_are_independent(self):
        before = self.path.read_bytes()
        defaults = json.loads((await self.get_settings(mode="background")).body)
        self.assertEqual(defaults["generation_mode"], "background")
        self.assertEqual(defaults["seed"], -1)
        self.assertEqual(self.path.read_bytes(), before)
        result = await self.save_settings({"img_w": 1280, "seed": 21}, mode="background")
        self.assertEqual(result.status, 200)
        await self.save_settings({"img_w": 1536, "seed": 22}, "group", "background")
        solo = json.loads((await self.get_settings(mode="background")).body)
        group = json.loads((await self.get_settings("group", "background")).body)
        self.assertEqual((solo["img_w"], solo["seed"]), (1280, 21))
        self.assertEqual((group["img_w"], group["seed"]), (1536, 22))
        saved = self.read()["bots"][0]
        for key, value in self.original.items():
            self.assertEqual(saved[key], value)
        self.assertNotIn("character_auto_settings_solo", saved)

    async def test_unknown_mode_is_rejected_without_writing(self):
        before = self.path.read_bytes()
        for response in (await self.get_settings(mode="typo"), await self.save_settings({"seed": 99}, mode="typo")):
            self.assertGreaterEqual(response.status, 400)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
