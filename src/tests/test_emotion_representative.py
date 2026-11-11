"""Emotion workshop: per-outfit representative expression builds the cover first."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch
import unittest


def asset_rows(representative, *, outfit_label="white dress",
               generation_mode="illustration"):
    return [{
        "anchor": "C001",
        "character": "Aria",
        "profile": "card_1",
        "gender": "1girl",
        "appearance": ["red hair"],
        "default_outfit": ["white dress"],
        "outfit_label": outfit_label,
        "outfit_tags": ["white dress"],
        "outfit_saved": False,
        "saved_outfits": [],
        "generation_mode": generation_mode,
        "decided_emotion": "joy",
        "context": "Aria stands by the window.",
        "representative_emotion": representative,
    }]


def asset_options():
    return [
        {"id": "E1", "rating": "sfw", "name": "기쁨", "tags": ["smile", "happy"]},
        {"id": "E2", "rating": "nsfw", "name": "음란", "tags": ["explicit"]},
    ]


def asset_response():
    return {
        "C001": {
            "character": "Aria",
            "rating": "sfw",
            "emotion": "E1",
            "outfit_tags": ["white dress"],
        },
    }


def representative(preset="smile", name="기쁨", rating="sfw"):
    return {"preset": preset, "name": name, "rating": rating}


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


class RepresentativeAssetTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = importlib.import_module("modes.illustration_context_pipeline")
        cls.visual_profiles = importlib.import_module("modes.visual_profiles")

    def parse(self, rows, *, toggles=None, image_exists=False):
        def store(bot, character, profile, outfit, emotion):
            return (f"/store/{bot}/{character}/{profile}/{outfit}/{emotion}", "")

        with patch.object(self.pipeline, "_emotion_asset_store_paths", store), \
             patch.object(self.pipeline, "_emotion_outfit_has_images",
                          lambda *args: image_exists), \
             patch.object(self.pipeline, "_emotion_expression_catalog",
                          return_value={"smile": ["smile", "happy"], "explicit": ["explicit"]}):
            return self.pipeline._parse_emotion_asset_resolution(
                json.dumps(asset_response()), rows, asset_options(),
                bot_name="test", toggles=toggles or {"nsfw": True},
            )

    def test_new_outfit_injects_representative_first(self):
        result = self.parse(asset_rows(representative()))
        reps = result["representative_assets"]
        self.assertEqual(len(reps), 1)
        self.assertTrue(reps[0]["representative"])
        self.assertEqual(reps[0]["emotion_name"], "기쁨")
        self.assertEqual(reps[0]["emotion_tags"], ["smile", "happy"])
        self.assertEqual(
            reps[0]["target_dir"], "/store/test/Aria/card_1/white dress/기쁨")
        self.assertEqual(len(result["assets"]), 1)

    def test_existing_outfit_skips_representative(self):
        result = self.parse(asset_rows(representative()), image_exists=True)
        self.assertEqual(result["representative_assets"], [])

    def test_nsfw_representative_needs_nsfw_generation(self):
        result = self.parse(
            asset_rows(representative("explicit", "음란", "nsfw")),
            toggles={"nsfw": False, "nsfw_asset_generation_enabled": False},
        )
        self.assertEqual(result["representative_assets"], [])
        self.assertTrue(any("NSFW" in warning for warning in result["validation_warnings"]))

    def test_representative_emotion_keeps_scene_target(self):
        # 씬이 필요로 하는 감정과 대표 표정이 같아도 씬 에셋은 그대로 남고,
        # 선행 생성 단계에서 저장된 이미지를 reuse_path로 재사용한다.
        result = self.parse(asset_rows(representative()))
        self.assertEqual(
            result["assets"][0]["target_dir"],
            result["representative_assets"][0]["target_dir"],
        )

    def test_normalization_preserves_representative(self):
        cards = self.visual_profiles._normalize_outfit_presets([
            {"label": "white dress", "tags": [{"tag": "dress"}],
             "representative_emotion": representative()},
            {"label": "broken", "tags": [],
             "representative_emotion": {"preset": "smile", "name": "", "rating": "sfw"}},
        ], field="card.outfit_presets")
        self.assertEqual(cards[0]["representative_emotion"], representative())
        self.assertNotIn("representative_emotion", cards[1])

    def test_preset_representative_applies_without_outfit_setting(self):
        # 새 NPC처럼 복장별 지정이 없어도 프리셋 설정의 대표 표정을 쓴다.
        rows = self.pipeline._emotion_asset_rows(
            [{"anchor": "C001", "character": "Aria", "emotion": "joy",
              "text": "Aria stands by the window."}],
            {}, {}, {}, {},
            {"emotion_representative": {"preset": "smile", "name": "기쁨", "rating": "sfw"}},
        )
        self.assertEqual(
            rows[0]["representative_emotion"],
            {"preset": "smile", "name": "기쁨", "rating": "sfw"})

    def test_npc_representative_tags_are_emotion_workshop_only(self):
        with patch.object(self.pipeline, "_emotion_expression_catalog",
                          return_value={"smile": ["smile", "happy", "cowboy shot"]}):
            toggles = {"emotion_representative": representative()}
            self.assertEqual(
                self.pipeline.npc_representative_emotion_tags(toggles, "emotion"),
                ["smile", "happy", "cowboy shot"])
            self.assertEqual(
                self.pipeline.npc_representative_emotion_tags(toggles, "illustration"), [])
            self.assertEqual(
                self.pipeline.npc_representative_emotion_tags({}, "emotion"), [])

    def test_npc_representative_tags_require_nsfw_generation(self):
        with patch.object(self.pipeline, "_emotion_expression_catalog",
                          return_value={"explicit": ["explicit"]}):
            toggles = {"emotion_representative": representative("explicit", "음란", "nsfw"),
                       "nsfw": False, "nsfw_asset_generation_enabled": False}
            self.assertEqual(
                self.pipeline.npc_representative_emotion_tags(toggles, "emotion"), [])
            toggles.update(nsfw=True, nsfw_asset_generation_enabled=True)
            self.assertEqual(
                self.pipeline.npc_representative_emotion_tags(toggles, "emotion"),
                ["explicit"])

    def test_representative_keeps_outfit_generation_mode(self):
        # 본 복장은 에셋 생성 모드, 변주 복장은 캐릭터 자동 생성 모드 값을
        # 복장 대표와 씬 에셋 양쪽에 그대로 넘긴다.
        for mode in ("illustration", "character_auto"):
            result = self.parse(asset_rows(representative(), generation_mode=mode))
            self.assertEqual(
                result["representative_assets"][0]["generation_mode"], mode)
            self.assertEqual(result["assets"][0]["generation_mode"], mode)

    def test_descriptor_carries_outfit_generation_mode(self):
        scene = importlib.import_module("modes.illustration_scene_pipeline")
        asset = asset_rows(representative(), generation_mode="character_auto")[0]
        asset.update(profile_id="card_1", outfit_folder="white dress",
                     emotion_folder="기쁨", emotion_name="기쁨",
                     emotion_tags=["smile"], target_dir="/store/x")
        descriptor = scene._emotion_asset_descriptor(asset, 1, {}, {})
        self.assertEqual(descriptor["generation_mode"], "character_auto")

    def test_merged_toggles_normalizes_preset_representative(self):
        toggles = self.pipeline.merged_toggles({
            "emotion_representative": {"preset": " smile ", "name": " 기쁨 ", "rating": "SFW"}})
        self.assertEqual(
            toggles["emotion_representative"],
            {"preset": "smile", "name": "기쁨", "rating": "sfw"})
        invalid = self.pipeline.merged_toggles({
            "emotion_representative": {"preset": "smile", "name": "", "rating": "sfw"}})
        self.assertEqual(invalid["emotion_representative"], {})


class OutfitLabelTests(unittest.TestCase):
    """Emotion store folders are named after outfits, not raw tag lists."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = importlib.import_module("modes.illustration_context_pipeline")

    def test_tag_list_outfit_reuses_saved_name(self):
        base = {
            "outfit": [
                {"tag": "white lab coat"}, {"tag": "brown shirt"},
                {"tag": "lace-up front"}, {"tag": "brown"},
            ],
            "outfit_presets": [
                {"label": "white lab coat", "tags": [
                    {"tag": "white lab coat"}, {"tag": "brown shirt"},
                    {"tag": "lace-up front"}, {"tag": "brown pants"}]},
                {"label": "white lab coat, brown shirt, lace-up front, brown",
                 "tags": [
                     {"tag": "white lab coat"}, {"tag": "brown shirt"},
                     {"tag": "lace-up front"}, {"tag": "brown"}]},
            ],
        }
        self.assertEqual(
            self.pipeline._emotion_outfit_label(
                base, "white lab coat, brown shirt, lace-up front, brown"),
            "white lab coat")

    def test_missing_label_uses_default_outfit_name(self):
        base = {
            "outfit": [{"tag": "white_dress"}, {"tag": "rope_belt"}],
            "outfit_presets": [{
                "label": "white dress",
                "tags": [{"tag": "white_dress"}, {"tag": "rope_belt"}],
            }],
        }
        self.assertEqual(self.pipeline._emotion_outfit_label(base, ""), "white dress")
        self.assertEqual(
            self.pipeline._emotion_outfit_label(base, "white_dress"), "white dress")


class OutfitGenerationModeTests(unittest.TestCase):
    """본 복장은 에셋 생성 모드, 저장된 변주 복장만 캐릭터 자동 생성 모드."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = importlib.import_module("modes.illustration_context_pipeline")

    def rows(self, *, anchor_outfit="", presets=None, saved=None):
        card = {
            "id": "card_1", "label": "카드 1", "selection_guide": "",
            "visual_context": "", "aliases": [],
            "appearance": [{"tag": "black hair"}],
            "default_outfit": [
                {"tag": "white shirt"}, {"tag": "dark-colored work apron"}],
            "outfit_presets": presets or [],
        }
        profiles = {"Doyun": {
            "default_visual_profile_id": "card_1", "profiles": [card]}}
        profile_result = (
            {"anchor_outfits": {"Doyun": {"C001": anchor_outfit}}}
            if anchor_outfit else {})
        catalog = {"Doyun": {"card_1": saved or []}}
        return self.pipeline._emotion_asset_rows(
            [{"anchor": "C001", "character": "Doyun", "emotion": "joy",
              "text": "Doyun stands."}],
            profile_result, profiles, catalog, {"C001": 0}, {})

    def test_first_outfit_without_preset_is_asset_mode(self):
        # 새 NPC는 복장 프리셋이 없어도 카드 기본 복장 태그로 본 복장을 찾는다.
        rows = self.rows(anchor_outfit="white shirt")
        self.assertEqual(rows[0]["outfit_label"], "white shirt")
        self.assertEqual(rows[0]["generation_mode"], "illustration")
        self.assertFalse(rows[0]["outfit_saved"])

    def test_preset_default_outfit_is_asset_mode(self):
        rows = self.rows(
            anchor_outfit="white shirt",
            presets=[{"label": "white shirt", "tags": [
                {"tag": "white shirt"}, {"tag": "buttoned shirt"}]}])
        self.assertEqual(rows[0]["generation_mode"], "illustration")

    def test_saved_variation_outfit_uses_character_auto(self):
        rows = self.rows(
            anchor_outfit="red dress",
            saved=[{"name": "white shirt", "tags": ["white shirt"]},
                   {"name": "red dress", "tags": ["red dress"]}])
        self.assertEqual(rows[0]["generation_mode"], "character_auto")
        self.assertTrue(rows[0]["outfit_saved"])

    def test_unsaved_outfit_does_not_force_face_reference(self):
        # 복장 이미지가 없는 새 복장은 변주 복장으로 강제하지 않는다.
        rows = self.rows(anchor_outfit="blue coat")
        self.assertEqual(rows[0]["generation_mode"], "illustration")


class RepresentativeCoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = importlib.import_module("modes.bot_mode")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bot"
        outfit = self.root / "test" / "Aria" / "card_1" / "white dress"
        (outfit / "기쁨").mkdir(parents=True)
        (outfit / "angry").mkdir(parents=True)
        (outfit / "기쁨" / "cover.png").write_bytes(b"cover")
        (outfit / "angry" / "aaa.png").write_bytes(b"angry")
        self.bot_data = {"bots": [{"name": "test", "characters": [{
            "name": "Aria",
            "visual_cards": [{
                "id": "card_1",
                "label": "카드 1",
                "appearance": [{"tag": "red hair"}],
                "default_outfit": [{"tag": "white dress"}],
                "outfit_presets": [{
                    "label": "white dress",
                    "tags": [{"tag": "white dress"}],
                    "representative_emotion": representative(),
                }],
            }],
        }]}]}
        for field, value in (
            ("BOT_DIR", str(self.root)),
            ("_load_bot_data", lambda: self.bot_data),
        ):
            patcher = patch.object(self.storage, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manager = self.storage.BotMode()

    def test_cover_prefers_representative_emotion(self):
        images = self.manager._emotion_image_files(
            str(self.root / "test" / "Aria" / "card_1" / "white dress"))
        cover = self.manager._emotion_outfit_cover_rel(
            "test", "Aria", "card_1", "white dress", images)
        self.assertEqual(cover, ["기쁨", "cover.png"])

    def test_cover_falls_back_without_representative_image(self):
        self.bot_data["bots"][0]["characters"][0]["visual_cards"][0][
            "outfit_presets"][0]["representative_emotion"] = representative("smile", "없는감정")
        self._patch_preset_representative(None)
        images = self.manager._emotion_image_files(
            str(self.root / "test" / "Aria" / "card_1" / "white dress"))
        cover = self.manager._emotion_outfit_cover_rel(
            "test", "Aria", "card_1", "white dress", images)
        self.assertEqual(cover, ["angry", "aaa.png"])

    def _patch_preset_representative(self, value):
        """프리셋 설정(설정 파일)의 복장 대표 표정을 테스트 값으로 고정한다."""
        config = {"emotion_illustration_context_toggles": {}}
        if value is not None:
            config["emotion_illustration_context_toggles"][
                "emotion_representative"] = value
        for field, patched in (
            ("application_config_path", lambda base: "settings.json"),
            ("read_config", lambda path: config),
        ):
            patcher = patch.object(self.storage, field, patched)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_cover_uses_preset_representative_without_outfit_setting(self):
        # 복장별 지정이 없어도 프리셋 설정에서 별표한 복장 대표 표정을 쓴다.
        self.bot_data["bots"][0]["characters"][0]["visual_cards"][0][
            "outfit_presets"][0].pop("representative_emotion")
        self._patch_preset_representative(representative("smile", "기쁨"))
        images = self.manager._emotion_image_files(
            str(self.root / "test" / "Aria" / "card_1" / "white dress"))
        cover = self.manager._emotion_outfit_cover_rel(
            "test", "Aria", "card_1", "white dress", images)
        self.assertEqual(cover, ["기쁨", "cover.png"])

    def test_card_representative_wins_over_preset_setting(self):
        self._patch_preset_representative(representative("smile", "angry"))
        images = self.manager._emotion_image_files(
            str(self.root / "test" / "Aria" / "card_1" / "white dress"))
        cover = self.manager._emotion_outfit_cover_rel(
            "test", "Aria", "card_1", "white dress", images)
        self.assertEqual(cover, ["기쁨", "cover.png"])

    def test_cover_falls_back_when_preset_folder_missing(self):
        # 별표한 표정의 폴더가 이 복장에 없으면 정렬 순서의 첫 이미지를 쓴다.
        self.bot_data["bots"][0]["characters"][0]["visual_cards"][0][
            "outfit_presets"][0].pop("representative_emotion")
        self._patch_preset_representative(representative("smile", "없는감정"))
        images = self.manager._emotion_image_files(
            str(self.root / "test" / "Aria" / "card_1" / "white dress"))
        cover = self.manager._emotion_outfit_cover_rel(
            "test", "Aria", "card_1", "white dress", images)
        self.assertEqual(cover, ["angry", "aaa.png"])


if __name__ == "__main__":
    unittest.main()
