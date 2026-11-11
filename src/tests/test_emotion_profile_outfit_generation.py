"""Profile base-outfit plans use each card's default outfit tags."""
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


def card(card_id, label, outfit, *, profile="card label"):
    return {
        "id": card_id,
        "label": profile,
        "default_outfit": [{"tag": tag} for tag in outfit],
        "appearance": [{"tag": "black hair"}],
        "gender_tag": "1girl",
        "outfit_presets": [{
            "label": label,
            "tags": [{"tag": tag} for tag in outfit],
        }],
    }


class ProfileOutfitPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("settings_illustration_regeneration")

    def build(self, cards, *, has_images=False, representative=None, toggles=None, folders=None):
        bot_data = {"bots": [{"name": "TestBot", "characters": [{
            "name": "Aria", "visual_cards": cards,
        }]}]}
        server = SimpleNamespace(bot_mode=SimpleNamespace(
            _emotion_outfit_representative=lambda *args: representative,
            _emotion_outfits=lambda *args: [{"name": name} for name in (folders or [])],
        ))
        with patch.object(self.module.storage, "_load_bot_data", return_value=bot_data), \
             patch.object(self.module, "effective_character_cards",
                          lambda character, _extra: (character["visual_cards"], "cards")), \
             patch.object(self.module.pipeline, "_emotion_outfit_has_images",
                          return_value=has_images), \
             patch.object(self.module.pipeline, "_emotion_expression_catalog",
                          return_value={"smile": ["smile", "happy"]}), \
             patch.object(self.module.pipeline, "_emotion_asset_store_paths",
                          side_effect=lambda bot, char, profile, outfit, emotion:
                          (f"/store/{bot}/{char}/{profile}/{outfit}/{emotion}", "")):
            return self.module._profile_outfit_plans(
                server, "TestBot", toggles or {"nsfw": True,
                                               "nsfw_asset_generation_enabled": True})

    def test_plan_uses_card_default_outfit_and_representative(self):
        plans = self.build(
            [card("card_1", "white dress", ["white dress", "lace trim"])],
            representative={"preset": "smile", "name": "기쁨", "rating": "sfw"},
        )
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan["action"], "generate")
        self.assertEqual(plan["outfit"], "white dress")
        self.assertEqual(plan["outfit_folder"], "white dress")
        self.assertEqual(plan["outfit_tags"], ["white dress", "lace trim"])
        self.assertEqual(plan["appearance"], ["black hair"])
        self.assertEqual(plan["emotion_name"], "기쁨")
        self.assertEqual(plan["emotion_tags"], ["smile", "happy"])
        self.assertEqual(plan["target_dir"], "/store/TestBot/Aria/card_1/white dress/기쁨")

    def test_existing_outfit_image_is_skipped(self):
        plans = self.build(
            [card("card_1", "white dress", ["white dress"])], has_images=True)
        self.assertEqual(plans[0]["action"], "skip")
        self.assertIn("이미 복장 이미지", plans[0]["reason"])

    def test_missing_default_outfit_is_skipped(self):
        plans = self.build([card("card_1", "white dress", [])])
        self.assertEqual(plans[0]["action"], "skip")
        self.assertIn("기본 복장 태그", plans[0]["reason"])

    def test_nsfw_representative_requires_nsfw_generation(self):
        plans = self.build(
            [card("card_1", "white dress", ["white dress"])],
            representative={"preset": "explicit", "name": "음란", "rating": "nsfw"},
            toggles={"nsfw": False, "nsfw_asset_generation_enabled": False},
        )
        self.assertEqual(plans[0]["action"], "skip")
        self.assertIn("NSFW", plans[0]["reason"])

    def test_missing_representative_falls_back_to_neutral(self):
        plans = self.build([card("card_1", "white dress", ["white dress"])])
        self.assertEqual(plans[0]["action"], "generate")
        self.assertEqual(plans[0]["emotion_name"], "neutral")
        self.assertEqual(plans[0]["emotion_tags"], [])

    def test_existing_outfit_folder_wins_weak_tag_overlap(self):
        # 'plain work apron' does not match 'work apron' exactly, so the saved
        # folder must be selected by overlap instead of creating a duplicate.
        plans = self.build(
            [card("card_1", "buttoned shirt",
                  ["buttoned shirt", "plain work apron"])],
            folders=["buttoned shirt"], has_images=True,
        )
        self.assertEqual(plans[0]["outfit"], "buttoned shirt")
        self.assertEqual(plans[0]["outfit_folder"], "buttoned shirt")
        self.assertEqual(plans[0]["action"], "skip")
        self.assertIn("이미 복장 이미지", plans[0]["reason"])

    def test_job_uses_card_appearance_and_emotion_workshop(self):
        captured = {}

        def descriptor(asset, slot, prompts, toggles):
            captured["asset"] = asset
            return {"raw_positive": "positive", "raw_negative": "negative",
                    "visual_base_snapshot": {"Aria": {"visual_profile_id": "card_1"}}}

        server = SimpleNamespace(
            prompts={},
            _descriptor_visual_states=lambda value: {"Aria": "card_1"},
        )
        plan = {
            "character": "Aria", "profile_id": "card_1", "profile_label": "카드 1",
            "outfit": "white dress", "outfit_folder": "white dress",
            "outfit_tags": ["white dress", "lace trim"],
            "gender": "1girl", "appearance": ["black hair"],
            "rating": "sfw", "emotion_name": "기쁨",
            "emotion_folder": "기쁨", "emotion_tags": ["smile", "happy"],
            "target_dir": "/store", "reuse_path": "",
        }
        snapshot = {"provider": "comfy", "illustration_workflow_type": "anima"}
        with patch.object(self.module.scene_pipeline, "_emotion_asset_descriptor",
                          side_effect=descriptor), \
             patch.object(self.module.workflow_profiles, "illustration_prompt_format",
                          return_value="v3"):
            job = self.module._profile_outfit_prompt_job(
                server, plan, 1, {}, {}, snapshot)
        asset = captured["asset"]
        self.assertEqual(asset["appearance"], ["black hair"])
        self.assertEqual(asset["outfit_tags"], ["white dress", "lace trim"])
        self.assertEqual(asset["emotion_tags"], ["smile", "happy"])
        self.assertEqual(asset["profile_id"], "card_1")
        self.assertEqual(job["params"]["workshop"], "emotion")
        self.assertIs(
            server.prompts[job["prompt_id"]]["_illustration_runtime_snapshot"], snapshot)

    def test_existing_short_folder_beats_missing_long_preset(self):
        # Both presets share the default outfit tags; the existing 'bikini'
        # folder must be chosen over the missing 'black bikini' folder.
        outfit = ["bikini", "black bikini", "choker", "earrings"]
        same_tags = [{"tag": tag} for tag in outfit]
        plans = self.build(
            [{
                "id": "card_1", "label": "카드 1",
                "default_outfit": same_tags,
                "appearance": [{"tag": "black hair"}],
                "gender_tag": "1girl",
                "outfit_presets": [
                    {"label": "black bikini", "tags": same_tags},
                    {"label": "bikini", "tags": same_tags},
                ],
            }],
            folders=["bikini"], has_images=True,
        )
        self.assertEqual(plans[0]["outfit_folder"], "bikini")
        self.assertEqual(plans[0]["action"], "skip")


if __name__ == "__main__":
    unittest.main()
