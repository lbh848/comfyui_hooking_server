"""Outfit cover regeneration follows the character_auto drawing settings."""
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
            "ANIMA_CONTENT": "1girl, Aria, white dress",
            "SDXL_CONTENT": "masterpiece, 1girl, Aria, white dress",
            "IMG_W": 756,
            "IMG_H": 756,
            "HRF_SIZE": 1.5,
            "HRF_ACTIVATE": False,
            "FD_ACTIVATE": True,
            "SNAPREF": {
                "enabled": True,
                "secondary_reference": "none",
                "strength": 1.0,
                "references": [
                    {"CHAR": "Aria", "asset": "representative", "face_asset": "face"},
                ],
            },
        },
        "assets": {"representative": {"data": "a"}, "face": {"data": "f"}},
    }


SETTINGS = {
    "img_w": 752,
    "img_h": 1024,
    "hrf_size": 2.0,
    "hrf_activate": True,
    "anima_hrf_activate": True,
    "hrf_restore_size": True,
    "anima_fd_activate": True,
    "anima_hd_activate": True,
    "anima_ed_activate": True,
    "fd_activate": False,
    "hd_activate": False,
    "ed_activate": False,
    "snapref_enabled": True,
    "snapref_preprocess": "lineart",
    "snapref_strength": 0.8,
    "snapref_strength_scale": "linear",
    "snapref_fidelity": 0.9,
    "snapref_secondary_reference": "face_only",
    "snapref_mode": "char",
    "snapref_megapixels": 1.0,
}


class DrawingModeOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("settings_illustration_regeneration")

    def test_overlays_settings_without_touching_prompt_or_references(self):
        original = document()
        updated = self.module._apply_drawing_mode(original, SETTINGS)
        fields = updated["fields"]
        self.assertEqual((fields["IMG_W"], fields["IMG_H"]), (752, 1024))
        self.assertEqual(fields["HRF_SIZE"], 2.0)
        for key in ("HRF_ACTIVATE", "ANIMA_HRF_ACTIVATE", "HRF_RESTORE_SIZE",
                    "ANIMA_FD_ACTIVATE", "ANIMA_HD_ACTIVATE", "ANIMA_ED_ACTIVATE"):
            self.assertIs(fields[key], True, key)
        for key in ("FD_ACTIVATE", "HD_ACTIVATE", "ED_ACTIVATE"):
            self.assertIs(fields[key], False, key)
        snapref = fields["SNAPREF"]
        self.assertEqual(snapref["preprocess"], "lineart")
        self.assertEqual(snapref["secondary_reference"], "face_only")
        self.assertEqual(snapref["strength"], 0.8)
        self.assertEqual(snapref["fidelity"], 0.9)
        # The stored reference ids stay so the workflow node can pick the face.
        self.assertEqual(
            snapref["references"],
            [{"CHAR": "Aria", "asset": "representative", "face_asset": "face"}],
        )
        # Prompt text and embedded images must survive a settings-only overlay.
        self.assertEqual(fields["ANIMA_CONTENT"], "1girl, Aria, white dress")
        self.assertEqual(updated["assets"], original["assets"])
        # The source document is not mutated.
        self.assertEqual(original["fields"]["IMG_W"], 756)
        self.assertEqual(original["fields"]["SNAPREF"]["secondary_reference"], "none")

    def test_missing_snapref_is_created(self):
        source = document()
        source["fields"].pop("SNAPREF")
        updated = self.module._apply_drawing_mode(source, SETTINGS)
        self.assertEqual(
            updated["fields"]["SNAPREF"]["secondary_reference"], "face_only")

    def test_character_auto_settings_are_resolved_for_the_emotion_bot(self):
        bot = {
            "name": "TestBot",
            "character_auto_settings_solo": {
                "img_w": 960,
                "snapref_secondary_reference": "face_only",
            },
        }
        # resolve_illust_settings는 BotMode 인스턴스가 아니라 모듈 함수이므로
        # server.bot_mode에 가짜 메서드를 붙이면 실제 런타임 버그를 가린다.
        server = SimpleNamespace()
        with patch.object(self.module.storage, "_load_bot_data",
                          return_value={"bots": [bot]}), \
             patch.object(self.module.storage, "set_bot_scope",
                          return_value="token"), \
             patch.object(self.module.storage, "reset_bot_scope"):
            settings = self.module._character_auto_settings(server, "TestBot")
        self.assertEqual(settings["img_w"], 960)
        self.assertEqual(settings["snapref_secondary_reference"], "face_only")
        # The emotion asset mode's values must not leak into the cover job.
        self.assertNotEqual(settings["img_w"], 756)

    def test_illustration_settings_are_resolved_for_the_emotion_bot(self):
        bot = {
            "name": "TestBot",
            "illust_settings_solo": {
                "img_w": 832,
                "snapref_secondary_reference": "face_only",
            },
        }
        server = SimpleNamespace()
        with patch.object(self.module.storage, "_load_bot_data",
                          return_value={"bots": [bot]}), \
             patch.object(self.module.storage, "set_bot_scope",
                          return_value="token"), \
             patch.object(self.module.storage, "reset_bot_scope"):
            settings = self.module._illustration_settings(server, "TestBot")
        self.assertEqual(settings["img_w"], 832)
        self.assertEqual(settings["snapref_secondary_reference"], "face_only")

    def test_profile_base_outfit_is_distinguished_from_variations(self):
        card = {
            "id": "p1",
            "label": "기본",
            "default_outfit": ["white dress"],
            "outfit_presets": [
                {"label": "white dress", "tags": ["white dress"]},
                {"label": "red dress", "tags": ["red dress"]},
            ],
        }
        bot = {
            "name": "TestBot",
            "characters": [{"name": "Aria", "character_cards": [card]}],
        }
        server = SimpleNamespace(bot_mode=SimpleNamespace(
            _emotion_outfits=lambda *args: []))
        with patch.object(self.module.storage, "_load_bot_data",
                          return_value={"bots": [bot]}), \
             patch.object(self.module.storage, "set_bot_scope",
                          return_value="token"), \
             patch.object(self.module.storage, "reset_bot_scope"), \
             patch.object(self.module, "effective_character_cards",
                          side_effect=lambda character, _source: (
                              character.get("character_cards", []), None)), \
             patch.object(self.module, "_emotion_store_outfit_names",
                          return_value=["white dress", "red dress"]):
            check = self.module._is_profile_base_outfit
            self.assertTrue(check(server, "TestBot", "Aria", "p1", "white dress"))
            self.assertFalse(check(server, "TestBot", "Aria", "p1", "red dress"))
            # An unknown profile keeps the previous variation behaviour.
            self.assertFalse(check(server, "TestBot", "Aria", "missing", "white dress"))


if __name__ == "__main__":
    unittest.main()
