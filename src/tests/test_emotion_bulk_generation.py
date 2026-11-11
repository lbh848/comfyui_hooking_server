"""Emotion bulk generation: preset filtering and outfit tag resolution."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
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


class EmotionBulkHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("settings_illustration_regeneration")

    def test_missing_skips_existing_and_dedupes_folders(self):
        options = [
            {"id": "E1", "rating": "sfw", "name": "neutral", "tags": ["a"]},
            {"id": "E2", "rating": "sfw", "name": "smile", "tags": ["b"]},
            {"id": "E3", "rating": "sfw", "name": "SMile", "tags": ["c"]},
            {"id": "E4", "rating": "sfw", "name": "sad", "tags": []},
        ]
        reuse = {"neutral": "/store/neutral.png"}
        pending = self.module._pending_emotion_presets(
            options, reuse, include_existing=False)
        self.assertEqual([row["name"] for row in pending], ["smile", "sad"])
        self.assertEqual(pending[0]["folder"], "smile")

    def test_all_includes_existing(self):
        options = [
            {"id": "E1", "rating": "sfw", "name": "neutral", "tags": ["a"]},
            {"id": "E2", "rating": "sfw", "name": "smile", "tags": ["b"]},
        ]
        reuse = {"neutral": "/store/neutral.png"}
        pending = self.module._pending_emotion_presets(
            options, reuse, include_existing=True)
        self.assertEqual([row["name"] for row in pending], ["neutral", "smile"])
        self.assertEqual(pending[0]["reuse_path"], "/store/neutral.png")

    def test_outfit_tags_match_normalized_label_and_fall_back_to_folder(self):
        card = {
            "outfit_presets": [
                {"label": "white dress",
                 "tags": [{"tag": "white dress"}, {"tag": "lace trim"}]},
            ],
        }
        self.assertEqual(
            self.module._emotion_bulk_outfit_tags(card, "white_dress"),
            ["white dress", "lace trim"],
        )
        self.assertEqual(
            self.module._emotion_bulk_outfit_tags({"outfit_presets": []}, "bikini, choker"),
            ["bikini", "choker"],
        )


if __name__ == "__main__":
    unittest.main()
