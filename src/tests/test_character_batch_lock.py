"""Character batch locks persist independently of card edits and exclusions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
import asyncio
from unittest.mock import AsyncMock, patch


class CharacterBatchLockTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        spec = importlib.util.spec_from_file_location("batch_lock_bot_mode", modes / "bot_mode.py")
        module = importlib.util.module_from_spec(spec)
        # Load real handlers without starting the unrelated generation runtime.
        with patch.dict(sys.modules, {"modes": package}):
            spec.loader.exec_module(module)
        cls.module = module

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "bot.json"
        self.initial = {"bots": [{"name": "TestBot", "characters": [
            {"name": "Alice", "custom_field": "preserve", "visual_cards": [
                {"id": "card_1", "label": "기본", "gender_tag": "1girl", "appearance": [{"tag": "long hair", "desc": "기존 설명"}], "default_outfit": [{"tag": "dress", "desc": ""}]},
                {"id": "card_2", "label": "제외 카드", "gender_tag": "1girl", "skip_lb_extra_batch_refine": True},
            ]},
            {"name": "Bob", "gender_tag": "1boy"},
        ]}]}
        self.path.write_text(json.dumps(self.initial), encoding="utf-8")
        for target, value in (("ASSET_DATA_DIR", self.temporary.name), ("BOT_DATA_FILE", str(self.path))):
            patcher = patch.object(self.module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, options in (("_load_bot_data", {"side_effect": self.read}), ("_load_lb_extra", {"return_value": None}), ("_save_lb_extra", {})):
            patcher = patch.object(self.module, target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manager = self.module.BotMode()

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def root(self):
        return self.read()["bots"][0]["characters"][0]

    def profiles(self):
        return self.module._character_card_data(self.root(), None)["character"]

    def request(self, body):
        return SimpleNamespace(json=AsyncMock(return_value=body))

    async def lock(self, locked=True, **overrides):
        body = {"action": "set_lb_extra_batch_lock", "bot_name": "TestBot", "char_name": "Alice", "locked": locked, **overrides}
        return await self.manager.handle_bot_action(self.request(body))

    async def save(self, base, data, batch=False):
        return await self.manager.handle_save_character_cards(self.request({
            "bot_name": "TestBot", "character": "Alice", "base": base, "data": data,
            "batch_refine": batch, "visual_card_id": "card_1",
        }))

    async def test_imported_lore_persists_per_character_and_survives_stale_card_save(self):
        base = self.profiles()
        entry = {"entry_id": "entry-1", "source": "bot.charx", "title": "Alice lore", "content": "Original text",
                 "keys": ["Alice", "앨리스"], "secondary_keys": ["garden"], "enabled": True,
                 "constant": False, "selective": True, "use_regex": False}
        body = {"action": "import_character_lore", "bot_name": "TestBot", "characters": [{"character": "alice", "entries": [entry]}]}
        response = await self.manager.handle_bot_action(self.request(body))
        self.assertEqual(response.status, 200)
        self.assertEqual(self.root()["imported_lore"], [entry])
        self.assertNotIn("imported_lore", self.read()["bots"][0]["characters"][1])
        self.assertEqual(self.profiles()["imported_lore"], [entry])
        entry["content"] = "Updated source text"
        await self.manager.handle_bot_action(self.request(body))
        self.assertEqual(self.root()["imported_lore"], [entry])
        edited = deepcopy(base)
        edited["profiles"][1]["label"] = "Changed card"
        response = await self.save(base, edited)
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body)["character"]["imported_lore"], [entry])
        self.assertEqual(self.root()["imported_lore"], [entry])
        self.assertTrue(all("imported_lore" not in card for card in self.root()["visual_cards"]))

    async def test_invalid_lore_target_does_not_partially_save(self):
        entry = {"entry_id": "entry-1", "source": "module.risum", "title": "Shared", "content": "Text", "keys": ["Alice"]}
        response = await self.manager.handle_bot_action(self.request({"action": "import_character_lore", "bot_name": "TestBot",
            "characters": [{"character": "Alice", "entries": [entry]}, {"character": "Missing", "entries": [entry]}]}))
        self.assertGreaterEqual(response.status, 400)
        self.assertEqual(self.read(), self.initial)

    async def test_imported_global_note_replaces_only_selected_bots_asset_instruction(self):
        initial = deepcopy(self.initial)
        initial["bots"][0]["asset_output_instruction"] = "Previous instruction"
        initial["bots"].append({"name": "OtherBot", "characters": [], "asset_output_instruction": "Keep this"})
        self.path.write_text(json.dumps(initial), encoding="utf-8")
        note = "  ## Asset output\r\nUse {{char}}_happy.webp.\nKeep <img> tags unchanged.  "
        response = await self.manager.handle_save_asset_output_instruction(self.request({
            "bot_name": "TestBot", "instruction": note,
        }))
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body)["instruction"], note)
        saved = self.read()
        self.assertEqual(saved["bots"][0]["asset_output_instruction"], note)
        self.assertEqual(saved["bots"][0]["characters"], initial["bots"][0]["characters"])
        self.assertEqual(saved["bots"][1], initial["bots"][1])
        response = await self.manager.handle_get_asset_output_instruction(SimpleNamespace(query={"bot_name": "TestBot"}))
        self.assertEqual(json.loads(response.body)["instruction"], note)

    async def test_lock_persists_and_unlock_keeps_card_exclusions(self):
        self.assertFalse(self.profiles()["lb_extra_batch_locked"])
        response = await self.lock(char_name="alice")
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(response.body)["locked"])
        self.assertTrue(self.profiles()["lb_extra_batch_locked"])
        expected = deepcopy(self.initial)
        expected["bots"][0]["characters"][0]["lb_extra_batch_locked"] = True
        self.assertEqual(self.read(), expected)
        await self.lock(False)
        self.assertFalse(self.profiles()["lb_extra_batch_locked"])
        self.assertTrue(self.profiles()["profiles"][1]["skip_lb_extra_batch_refine"])

    async def test_invalid_lock_does_not_write(self):
        for overrides in ({"locked": "false"}, {"char_name": "Missing"}, {"bot_name": "Missing"}):
            with self.subTest(overrides=overrides):
                response = await self.lock(**overrides)
                self.assertGreaterEqual(response.status, 400)
                self.assertEqual(self.read(), self.initial)

    async def test_lock_save_failure_is_reported_without_changing_persisted_state(self):
        with patch.object(self.module, "_save_bot_data", side_effect=OSError("disk unavailable")):
            response = await self.lock()
        self.assertIn("disk unavailable", json.loads(response.body)["error"])
        self.assertEqual(self.read(), self.initial)

    async def test_stale_batch_result_cannot_overwrite_after_lock(self):
        base = self.profiles()
        edited = deepcopy(base)
        edited["profiles"][0]["appearance"] = [{"tag": "refined hair", "desc": ""}]
        await self.lock()
        before = self.path.read_bytes()
        response = await self.save(base, edited, batch=True)
        self.assertGreaterEqual(response.status, 400)
        self.assertIn("잠겨", json.loads(response.body)["error"])
        self.assertEqual(self.path.read_bytes(), before)

    async def test_manual_edit_still_saves_and_keeps_current_lock(self):
        base = self.profiles()
        edited = deepcopy(base)
        edited["profiles"][0]["appearance"] = [{"tag": "manual edit", "desc": "직접 편집"}]
        await self.lock()
        response = await self.save(base, edited)
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(response.body)["character"]["lb_extra_batch_locked"])
        self.assertTrue(self.root()["lb_extra_batch_locked"])
        self.assertEqual(self.profiles()["profiles"][0]["appearance"], edited["profiles"][0]["appearance"])
        self.assertTrue(self.profiles()["profiles"][1]["skip_lb_extra_batch_refine"])
        self.assertEqual(self.root()["custom_field"], "preserve")

    async def test_unlock_allows_batch_save(self):
        await self.lock()
        await self.lock(False)
        base = self.profiles()
        edited = deepcopy(base)
        edited["profiles"][0]["default_outfit"] = [{"tag": "new dress", "desc": "새 복장"}]
        response = await self.save(base, edited, batch=True)
        self.assertEqual(response.status, 200)
        self.assertEqual(self.profiles()["profiles"][0]["default_outfit"], edited["profiles"][0]["default_outfit"])

    async def test_parallel_results_for_different_cards_preserve_both_edits(self):
        base = self.profiles()
        edits = []
        for index, tag in enumerate(("refined first", "refined second")):
            edited = deepcopy(base)
            edited["profiles"][index]["appearance"] = [{"tag": tag, "desc": ""}]
            edits.append(edited)
        responses = await asyncio.gather(*(self.manager.handle_save_character_cards(self.request({
            "bot_name": "TestBot", "character": "Alice", "base": base, "data": edited,
        })) for edited in edits))
        self.assertTrue(all(response.status == 200 for response in responses))
        for index, edited in enumerate(edits):
            self.assertEqual([tag["tag"] for tag in self.profiles()["profiles"][index]["appearance"]],
                             [tag["tag"] for tag in edited["profiles"][index]["appearance"]])

    async def test_data_patch_and_status_use_original_images_without_input(self):
        root = Path(self.temporary.name)
        input_dir = root / "data/comfy/input"
        bot_dir = root / "bots"
        source = bot_dir / "TestBot/Alice/portrait.png"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"test image")
        value = self.read()
        value["bots"][0]["characters"][0]["rep_images"] = [source.name]
        value["bots"][0]["characters"][0]["visual_cards"][0]["rep_images"] = [source.name]
        self.path.write_text(json.dumps(value), encoding="utf-8")
        patcher = self.module.BotDataPatcher()
        with patch.object(self.module, "BASE_DIR", str(root / "src")), \
             patch.object(self.module, "BOT_DIR", str(bot_dir)), \
             patch.object(self.module, "comfy_input_root", side_effect=AssertionError("Must not access Comfy input")), \
             patch.object(self.module, "read_config", side_effect=AssertionError("Input must not depend on configuration")):
            response = await patcher.handle_data_patch(self.request({"bot_name": "TestBot", "char_name": "Alice"}))
            self.assertEqual(response.status, 200, response.body)
            self.assertFalse(input_dir.exists())
            self.assertEqual(source.read_bytes(), b"test image")
            result = json.loads(response.body)
            self.assertEqual(result["created_dirs"], [])
            self.assertEqual(result["copied_files"], [])
            self.assertEqual(result["visual_targets"][0]["rep_images"], [source.name])
            response = await patcher.handle_check_patch_files(SimpleNamespace(query={"bot_name": "TestBot"}))
            self.assertEqual(response.status, 200, response.body)
            self.assertEqual(json.loads(response.body)["Alice"], {
                "face": False,
                "cards": {"card_1": {"face": False}, "card_2": {"face": False}},
            })

            face_image = source.parent / "_face_image.webp"
            face_image.write_bytes(b"test FACE image")
            response = await patcher.handle_check_patch_files(SimpleNamespace(query={"bot_name": "TestBot"}))
            self.assertEqual(response.status, 200, response.body)
            self.assertEqual(json.loads(response.body)["Alice"], {
                "face": True,
                "cards": {"card_1": {"face": True}, "card_2": {"face": False}},
            })

            card_dir = source.parent / "_visual_profiles" / "card_2"
            card_dir.mkdir(parents=True)
            face_image.rename(card_dir / face_image.name)
            response = await patcher.handle_check_patch_files(SimpleNamespace(query={"bot_name": "TestBot"}))
            self.assertEqual(response.status, 200, response.body)
            self.assertEqual(json.loads(response.body)["Alice"], {
                "face": False,
                "cards": {"card_1": {"face": False}, "card_2": {"face": True}},
            })

    async def test_locked_or_excluded_card_never_enters_llm_queue(self):
        queue = SimpleNamespace(add_item=AsyncMock())
        with patch.dict(sys.modules, {"server": SimpleNamespace(queue_manager=queue)}):
            await self.lock()
            for card_id in ("card_1", "card_2"):
                response = await self.module.handle_lb_extra_refine(self.request({
                    "bot": "TestBot", "character": "Alice", "batch_refine": True, "visual_card_id": card_id,
                }))
                self.assertEqual(response.status, 409)
            await self.lock(False)
            response = await self.module.handle_lb_extra_refine(self.request({
                "bot": "TestBot", "character": "Alice", "batch_refine": True, "visual_card_id": "card_2",
            }))
            self.assertEqual(response.status, 409)
        queue.add_item.assert_not_awaited()

    async def test_card_excluded_while_refining_cannot_be_saved(self):
        base = self.profiles()
        edited = deepcopy(base)
        edited["profiles"][0]["appearance"] = [{"tag": "refined hair", "desc": ""}]
        latest = self.read()
        latest["bots"][0]["characters"][0]["visual_cards"][0]["skip_lb_extra_batch_refine"] = True
        self.path.write_text(json.dumps(latest), encoding="utf-8")
        response = await self.save(base, edited, batch=True)
        self.assertGreaterEqual(response.status, 400)
        self.assertEqual(self.read(), latest)

    async def test_face_batch_respects_character_and_profile_locks_before_llm(self):
        with patch.object(self.module, "run_auto_classify_face_tags", new_callable=AsyncMock) as classify:
            await self.lock()
            for card_id in ("card_1", "card_2"):
                response = await self.module.handle_auto_classify_face_tags(self.request({
                    "bot": "TestBot", "character": "Alice", "visual_card_id": card_id, "batch_refine": True,
                }))
                self.assertEqual(response.status, 409)
            await self.lock(False)
            response = await self.module.handle_auto_classify_face_tags(self.request({
                "bot": "TestBot", "character": "Alice", "visual_card_id": "card_2", "batch_refine": True,
            }))
            self.assertEqual(response.status, 409)
            classify.assert_not_awaited()

    async def test_face_batch_rejects_missing_character_and_card(self):
        with patch.object(self.module, "run_auto_classify_face_tags", new_callable=AsyncMock) as classify:
            for character, card_id, status in (("Missing", "card_1", 404), ("Alice", "missing", 409), ("Alice", "", 409)):
                response = await self.module.handle_auto_classify_face_tags(self.request({
                    "bot": "TestBot", "character": character, "visual_card_id": card_id, "batch_refine": True,
                }))
                self.assertEqual(response.status, status)
            classify.assert_not_awaited()

    async def test_face_batch_uses_selected_profile_and_manual_classification_remains_available(self):
        result = {"success": True, "data": {"face": ["long hair"], "eye": ["blue eyes"]}}
        with patch.object(self.module, "run_auto_classify_face_tags", new_callable=AsyncMock, return_value=result) as classify:
            response = await self.module.handle_auto_classify_face_tags(self.request({
                "bot": "TestBot", "character": "Alice", "visual_card_id": "card_1", "batch_refine": True,
            }))
            self.assertEqual(json.loads(response.body), result)
            classify.assert_awaited_once_with("TestBot", "Alice", "card_1")
            classify.reset_mock()
            await self.lock()
            response = await self.module.handle_auto_classify_face_tags(self.request({
                "bot": "TestBot", "character": "Alice", "visual_card_id": "card_2",
            }))
            self.assertEqual(json.loads(response.body), result)
            classify.assert_awaited_once_with("TestBot", "Alice", "card_2")

    async def test_face_batch_result_cannot_be_saved_after_character_lock(self):
        base = self.profiles()
        edited = deepcopy(base)
        edited["profiles"][0]["render_overrides"].update(face_tags="new hair", eye_tags="blue eyes")
        edited["profiles"][0]["appearance"] = [{"tag": "new hair", "desc": ""}, {"tag": "blue eyes", "desc": ""}]
        await self.lock()
        before = self.path.read_bytes()
        response = await self.save(base, edited, batch=True)
        self.assertGreaterEqual(response.status, 400)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
