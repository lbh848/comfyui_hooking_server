"""Bot-owned work settings survive saving and reach every illustration contract."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from modes import illustration_context_pipeline as pipeline
from modes import illustration_prompt_builders as builders
from modes.illustration_prompt_structure import build_prompt_structure

bot_module = importlib.import_module("modes.bot_mode")
PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "lighbd"


class SecondaryGenrePersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data = {"bots": [
            {"name": name, "preset_scope": "local", "system_prompt_preset": "Shared"}
            for name in ("First", "Second")
        ], "system_prompt_presets": {"Shared": "Custom setting"}}
        for name, options in (
            ("_load_bot_data", {"side_effect": lambda: deepcopy(self.data)}),
            ("_save_bot_data", {"side_effect": self.save}),
            ("_load_builtin_presets", {"return_value": {"Builtin": "Shipped setting"}}),
        ):
            patcher = patch.object(bot_module, name, **options)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manager = bot_module.BotMode()

    def save(self, value):
        self.data = deepcopy(value)

    async def write(self, **overrides):
        return await self.manager.handle_save_system_prompt(SimpleNamespace(json=AsyncMock(return_value={
            "bot_name": "First", "preset_name": "Shared", "scope": "local", "text": "Custom setting",
            "genre": "secondary_bot", "genre_work_title": "  Example Work: Part II  ", **overrides,
        })))

    async def read(self, name):
        response = await self.manager.handle_get_system_prompt(SimpleNamespace(query={"bot_name": name}))
        self.assertEqual(response.status, 200, response.text)
        return json.loads(response.text)

    async def test_saved_work_is_owned_by_bot_not_shared_system_preset(self):
        response = await self.write()
        self.assertEqual(response.status, 200, response.text)
        first, second = await self.read("First"), await self.read("Second")
        self.assertEqual((first["genre"], first["genre_work_title"]), ("secondary_bot", "Example Work: Part II"))
        self.assertEqual((second["genre"], second["genre_work_title"]), ("", ""))
        self.assertEqual(self.data["system_prompt_presets"]["Shared"], "Custom setting")

    async def test_builtin_binding_can_save_work_without_editing_preset(self):
        response = await self.write(scope="builtin", preset_name="Builtin", bind_only=True)
        self.assertEqual(response.status, 200, response.text)
        saved = await self.read("First")
        self.assertEqual(saved["genre_work_title"], "Example Work: Part II")
        self.assertEqual(saved["text"], "Shipped setting")

    async def test_missing_work_rejected_before_any_save(self):
        before = deepcopy(self.data)
        for value in ("", "  "):
            response = await self.write(genre_work_title=value)
            self.assertEqual(response.status, 400, response.text)
            self.assertEqual(self.data, before)

    async def test_older_client_does_not_erase_work_when_changing_genre(self):
        await self.write()
        response = await self.manager.handle_save_system_prompt(SimpleNamespace(json=AsyncMock(return_value={
            "bot_name": "First", "preset_name": "Shared", "scope": "local",
            "text": "Custom setting", "genre": "Modern",
        })))
        self.assertEqual(response.status, 200, response.text)
        saved = await self.read("First")
        self.assertEqual(saved["genre"], "Modern")
        self.assertEqual(saved["genre_work_title"], "Example Work: Part II")


class SecondaryGenrePromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = {key: (PROMPTS / filename).read_text(encoding="utf-8")
                       if (PROMPTS / filename).is_file() else ""
                       for key, filename in pipeline.PROMPT_FILES.items()}

    def toggles(self, **overrides):
        return pipeline.merged_toggles({"genre": "secondary_bot", "genre_work_title": "Example Work: Part II",
                                       **overrides}, for_generation=True)

    def test_all_detail_modes_keep_work_and_policy_with_tracking_off(self):
        for model, mode, key in (
            ("anima", "illustration", "call2_detail"),
            ("moss", "illustration", "detail_moss"), ("moss", "manga", "detail_manga"),
            ("anima", "composite", "composite_detail"),
        ):
            with self.subTest(model=model, mode=mode):
                toggles = self.toggles(illustration_model_family=model, illustration_output_mode=mode,
                                       outfit_enabled=False, time_place_enabled=False, call2_plan_enabled=False)
                system = builders.role_context("", "Custom setting", genre=toggles["genre"],
                    genre_work_title=toggles["genre_work_title"], genre_prompt=self.prompts["genre"],
                    genre_secondary_prompt=self.prompts["genre_secondary"]) + "\n\n" + self.prompts[key]
                messages = builders.detail_messages(system, [{"slot": "C001", "characters": []}], ["C001"],
                    toggles=toggles, past_text="Past", segmented="Current scene",
                    canvas_sizes={"solo": (1024, 1024), "group": (1152, 768)})
                self.assertEqual(messages[0]["role"], "system")
                for expected in ("Example Work: Part II", "Custom setting", "EXISTING WORK SETTING"):
                    self.assertIn(expected, messages[0]["content"])
                structure = build_prompt_structure(self.prompts, toggles=toggles)
                detail = next(stage for stage in structure["stages"] if stage["key"] == "detail")
                self.assertIn("EXISTING WORK SETTING", detail["messages"][0]["content"])
                self.assertTrue(any(part["path"].endswith("genre_secondary.txt")
                                    for part in detail["system_parts"]))

    def test_time_place_and_standalone_keyvis_receive_same_work(self):
        toggles = self.toggles(keyvis_in_detail_enabled=False)
        # Current generation combines KEYVIS with DETAIL; exercise the builder's
        # standalone contract explicitly as well.
        toggles["keyvis_in_detail_enabled"] = False
        time_place = builders.time_place_messages(self.prompts, [], {}, "", "", toggles=toggles)
        keyvis = builders.keyvis_context_messages("Key visual", "Custom setting", [], {}, "", "",
            toggles=toggles, genre_prompt=self.prompts["genre"], genre_secondary_prompt=self.prompts["genre_secondary"])
        for messages in (time_place, keyvis):
            self.assertIn("Example Work: Part II", messages[0]["content"])
            self.assertIn("EXISTING WORK SETTING", messages[0]["content"])

    def test_other_genres_ignore_retained_work(self):
        for genre in ("", "Modern", "Custom world"):
            messages = builders.time_place_messages(self.prompts, [], {}, "", "", toggles=self.toggles(genre=genre))
            self.assertNotIn("Example Work: Part II", messages[0]["content"])
            self.assertNotIn("EXISTING WORK SETTING", messages[0]["content"])
            structure = build_prompt_structure(self.prompts, toggles=self.toggles(genre=genre))
            self.assertNotIn("EXISTING WORK SETTING", json.dumps(structure, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
