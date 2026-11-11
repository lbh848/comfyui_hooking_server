"""EMOTION-RESOLVE receives saved outfit names together with their tags."""
import importlib
import sys
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch


class EmotionOutfitPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = ModuleType("modes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "modes")]
        cls.modules = patch.dict(sys.modules, {"modes": package})
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        cls.pipeline = importlib.import_module("modes.illustration_context_pipeline")

    def test_saved_outfit_prompt_row_includes_tags(self):
        rows = self.pipeline._emotion_anchor_rows(
            [{
                "character": "Aria",
                "anchor": "C001",
                "emotion": "joy",
                "text": "Aria stands by the window.",
            }],
            None,
            {
                "Aria": {
                    "name": "Aria",
                    "default_visual_profile_id": "card_1",
                    "profiles": [{
                        "id": "card_1",
                        "name": "card_1",
                        "appearance": [],
                        "default_outfit": ["white dress"],
                    }],
                },
            },
            {
                "Aria": {
                    "card_1": [{
                        "name": "white dress",
                        "tags": ["white dress", "lace trim"],
                    }],
                },
            },
            None,
            {"C001": 1},
        )
        self.assertEqual(
            rows[0]["saved_outfits"],
            [{"name": "white dress", "tags": ["white dress", "lace trim"]}],
        )


if __name__ == "__main__":
    unittest.main()
