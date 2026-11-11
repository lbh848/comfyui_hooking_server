"""감정 삽화 복장 폴더의 이름 변경·삭제 동작을 검증한다."""
import asyncio
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class _Request:
    def __init__(self, body=None, method="POST", query=None):
        self.method = method
        self._body = body or {}
        self.query = query or {}

    async def json(self):
        return self._body


class EmotionStoreOutfitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = importlib.import_module("modes.bot_mode")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / "bot"
        outfit = root / "test" / "Aria" / "card_1" / "white dress"
        (outfit / "기쁨").mkdir(parents=True)
        (outfit / "기쁨" / "cover.png").write_bytes(b"cover")
        self.root = root
        self.outfit = outfit

        self.card = {
            "id": "card_1",
            "label": "카드 1",
            "gender_tag": "female",
            "outfit_presets": [{
                "label": "white dress",
                "tags": [{"tag": "white dress"}],
                "representative_emotion": {
                    "preset": "smile", "name": "기쁨", "rating": "sfw"},
            }],
        }
        self.character = {"name": "Aria", "visual_cards": [self.card]}
        self.bot_data = {"bots": [{"name": "test", "characters": [self.character]}]}
        self.saved = []

        def fake_store(root_character, cards):
            root_character["visual_cards"] = cards
            return cards

        for patcher in (
            patch.object(self.storage, "BOT_DIR", str(root)),
            patch.object(self.storage, "_load_bot_data", lambda: self.bot_data),
            patch.object(self.storage, "_save_bot_data", self.saved.append),
            patch.object(self.storage, "store_visual_cards", fake_store),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manager = self.storage.BotMode()

    def _post(self, **body):
        return asyncio.run(self.manager.handle_emotion_store_outfits(
            _Request({"bot": "test", "character": "Aria", "profile": "card_1", **body})))

    def test_rename_moves_folder_and_preset_label(self):
        response = self._post(action="rename", outfit="white dress", new_outfit="blue dress")
        payload = json.loads(response.text)
        self.assertEqual(response.status, 200, payload)
        self.assertTrue((self.outfit.parent / "blue dress" / "기쁨" / "cover.png").is_file())
        self.assertFalse(self.outfit.exists())
        self.assertEqual(
            self.character["visual_cards"][0]["outfit_presets"][0]["label"],
            "blue dress")
        self.assertEqual(
            [row["name"] for row in payload["outfits"]], ["blue dress"])
        self.assertTrue(self.saved)

    def test_rename_rejects_existing_name(self):
        (self.outfit.parent / "taken").mkdir()
        response = self._post(action="rename", outfit="white dress", new_outfit="taken")
        self.assertNotEqual(response.status, 200, response.text)
        self.assertTrue(self.outfit.is_dir())

    def test_rename_rejects_missing_folder(self):
        response = self._post(action="rename", outfit="missing", new_outfit="other")
        self.assertEqual(response.status, 404, response.text)

    def test_rename_rejects_path_segments(self):
        response = self._post(action="rename", outfit="white dress", new_outfit="../escape")
        self.assertNotEqual(response.status, 200, response.text)
        self.assertTrue(self.outfit.is_dir())

    def test_rename_to_same_name_is_noop(self):
        response = self._post(action="rename", outfit="white dress", new_outfit="white dress")
        self.assertEqual(response.status, 200, response.text)
        self.assertTrue(self.outfit.is_dir())
        self.assertFalse(self.saved)

    def test_outfit_rows_mark_the_base_outfit_as_asset_mode(self):
        self.card["default_outfit"] = [{"tag": "red dress"}]
        variation = self.outfit.parent / "red dress" / "기쁨"
        variation.mkdir(parents=True)
        (variation / "cover.png").write_bytes(b"cover")
        response = asyncio.run(self.manager.handle_emotion_store_outfits(
            _Request(method="GET", query={
                "bot": "test", "character": "Aria", "profile": "card_1"})))
        self.assertEqual(response.status, 200, response.text)
        modes = {
            row["name"]: row["generation_mode"]
            for row in json.loads(response.text)["outfits"]
        }
        # 기본 복장만 에셋 생성 모드, 변주 복장은 캐릭터 자동 생성 모드.
        self.assertEqual(modes, {
            "red dress": "illustration",
            "white dress": "character_auto",
        })


if __name__ == "__main__":
    unittest.main()
