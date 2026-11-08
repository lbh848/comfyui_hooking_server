"""Extended character types survive persistence and compound prompt handling."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


class BotGenderTagTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        with patch.dict(sys.modules, {"modes": package}):
            for attribute, filename in (("module", "bot_mode"), ("prompts", "illust_prompt_builder")):
                spec = importlib.util.spec_from_file_location(f"gender_test_{filename}", modes / f"{filename}.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                setattr(cls, attribute, module)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "bot.json"
        self.initial = {"bots": [{"name": "TestBot", "characters": [{
            "name": "Character", "gender_tag": "1girl", "visual_cards": [
                {"id": "card_1", "label": "Primary", "gender_tag": "1girl"},
                {"id": "card_2", "label": "Alternate", "gender_tag": "1boy"},
            ],
        }]}]}
        self.write(self.initial)
        for target, options in (
            ("_load_bot_data", {"side_effect": self.read}),
            ("_save_bot_data", {"side_effect": self.write}),
            ("_migrate_system_prompt_preset", {}),
            ("_load_lb_extra", {"return_value": None}),
            ("_save_lb_extra", {}),
        ):
            patcher = patch.object(self.module, target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manager = self.module.BotMode()

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def read(self):
        return self.module._read_bot_data(self.path)

    async def test_legacy_update_survives_reload_and_preserves_other_cards(self):
        for gender in ("1girl", "1boy", "1male", "animal", "1girl, furry", "1boy, furry", "futanari", "monster", "monster girl"):
            with self.subTest(gender=gender):
                self.write(self.initial)
                response = await self.manager._update_char_gender_tag(self.read(), {
                    "bot_name": "TestBot", "char_name": "Character", "gender_tag": gender,
                })
                self.assertEqual(response.status, 200)
                root = self.read()["bots"][0]["characters"][0]
                self.assertEqual(root["gender_tag"], gender)
                self.assertEqual(root["visual_cards"][0]["gender_tag"], gender)
                self.assertEqual(root["visual_cards"][1]["gender_tag"], "1boy")

    async def test_card_editor_save_survives_reload(self):
        for gender in ("animal", "1girl, furry", "1boy, furry", "futanari", "monster", "monster girl"):
            with self.subTest(gender=gender):
                self.write(self.initial)
                root = self.read()["bots"][0]["characters"][0]
                base = self.module._character_card_data(root, None)["character"]
                edited = deepcopy(base)
                edited["profiles"][0]["render_overrides"]["gender_tag"] = gender
                edited["profiles"][1]["render_overrides"]["gender_tag"] = "animal"
                response = await self.manager.handle_save_character_cards(SimpleNamespace(json=AsyncMock(return_value={
                    "bot_name": "TestBot", "character": "Character", "base": base, "data": edited,
                })))
                self.assertEqual(response.status, 200, response.body)
                root = self.read()["bots"][0]["characters"][0]
                self.assertEqual(root["gender_tag"], gender)
                self.assertEqual(root["visual_cards"][0]["gender_tag"], gender)
                self.assertEqual(root["visual_cards"][1]["gender_tag"], "animal")

    def test_compound_gender_keeps_cast_counts_and_deduplicates_face_tags(self):
        builder = self.prompts.IllustPromptBuilder()
        for gender, count in (("1girl, furry", "1girl"), ("1boy, furry", "1boy")):
            with self.subTest(gender=gender):
                character = {"name": "Character", "gender_tag": gender, "face_tags": f"{gender}, blue eyes, FURRY"}
                face = builder.build_char_face_tag_inform(["Character"], [character])
                self.assertEqual(face["list"][0]["FACE_TAGS"], f"{gender}, blue eyes")
                prompt = builder.build_positive_prompt(
                    setup="2girls, garden", char="Character, standing", supplement="",
                    detected_chars=["Character"], bot={"characters": [character]}, tags={},
                    settings={"seed": 1}, bot_name="TestBot",
                )
                content = prompt.split("[ANIMA_CONTENT]\n", 1)[1].split("\n[", 1)[0]
                self.assertIn(count, content.split(", "))
                self.assertNotIn("2girls", content)


if __name__ == "__main__":
    unittest.main()
