"""Distribution edits require a session unlock; copies and transfers retain data."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from asset_character_presets import AssetCharacterPresets, register_character_preset_runtime
from asset_style_presets import AssetStylePresets
from json_store import write_json_atomic
from tests.test_asset_style_presets import settings_fixture


class AssetPresetGroupsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.styles = AssetStylePresets(self.root / "settings.json", self.root / "builtin-styles.json")
        self.style = {"id": "shipped", "name": "배포 그림체", "settings": settings_fixture()}
        write_json_atomic(self.styles.builtin_file, {"version": 1, "presets": [self.style]})
        self.characters = AssetCharacterPresets(self.root / "tags.json", self.root / "builtin-characters.json")
        self.character = {"appearance": "외모", "outfit": "복장", "expression": "표정", "group": "old-group",
                          "settings": {"defaults": {"raw_appearance_tags": ["default"]},
                                       "values": {"raw_appearance_tags": ["edited"], "snapref_images": [{"name": "ref.png"}]}}}
        write_json_atomic(self.characters.builtin_file, {"version": 1, "characters": {"배포 캐릭터": self.character}})
        write_json_atomic(self.characters.path, {"characters": {"내 캐릭터": copy.deepcopy(self.character)}, "outfits": {"keep": ["dress"]}})

    def style_change(self, action, **fields):
        return self.styles.change({"action": action, "id": "shipped", "revision": self.styles.catalog()["revision"], **fields})

    def test_existing_items_default_to_user_and_builtin_is_separate(self):
        self.style_change("create", name="내 그림체", settings=settings_fixture())
        self.assertEqual([item["scope"] for item in self.styles.catalog()["presets"]], ["builtin", "local"])
        self.assertEqual(self.characters.catalog()["scopes"], {"배포 캐릭터": "builtin", "내 캐릭터": "local"})
        self.assertEqual(json.loads(self.styles.builtin_file.read_text(encoding="utf-8"))["presets"], [self.style])

    def test_builtin_style_is_locked_but_copy_is_editable(self):
        before = self.styles.builtin_file.read_bytes()
        for action in ("save", "rename", "delete", "scope"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.style_change(action, settings=settings_fixture(), name="changed", confirmed=True, scope="local", builtin_unlock_token="fake")
        result = self.style_change("create", name="내 복사", settings=settings_fixture())
        self.style_change("rename", id=result["saved_id"], name="내 수정")
        self.assertEqual(self.styles.builtin_file.read_bytes(), before)

    def test_style_unlock_move_relock_and_restart(self):
        token = self.style_change("unlock", confirmed=True)["builtin_unlock_token"]
        self.style_change("scope", scope="local", builtin_unlock_token=token)
        self.assertEqual(json.loads(self.styles.builtin_file.read_text())["presets"], [])
        item = self.styles.catalog()["presets"][0]
        self.assertEqual(item, {**self.style, "scope": "local"})
        self.style_change("scope", scope="builtin", builtin_unlock_token=token)
        restarted = AssetStylePresets(self.root / "settings.json", self.styles.builtin_file)
        with self.assertRaises(ValueError):
            restarted.change({"action": "save", "id": "shipped", "revision": restarted.catalog()["revision"], "settings": settings_fixture(), "builtin_unlock_token": token})
        self.style_change("lock", builtin_unlock_token=token)
        with self.assertRaises(ValueError):
            self.style_change("delete", confirmed=True, builtin_unlock_token=token)

    def test_style_transfer_conflict_and_failed_second_write_preserve_both_groups(self):
        self.style_change("create", name=self.style["name"], settings=settings_fixture())
        token = self.style_change("unlock", confirmed=True)["builtin_unlock_token"]
        before = self.styles.catalog()
        with self.assertRaises(ValueError):
            self.style_change("scope", scope="local", builtin_unlock_token=token)
        self.assertEqual(self.styles.catalog(), before)

        local_id = next(item["id"] for item in before["presets"] if item["scope"] == "local")
        self.style_change("rename", id=local_id, name="다른 이름")
        before = self.styles.catalog()
        def fail_local(path, value):
            if path == self.styles.path:
                raise OSError("disk full")
            write_json_atomic(path, value)
        with patch("asset_style_presets.write_json_atomic", side_effect=fail_local), self.assertRaises(OSError):
            self.style_change("scope", scope="local", builtin_unlock_token=token)
        self.assertEqual(self.styles.catalog(), before)

    def test_release_reintroducing_moved_style_keeps_user_id_settings_and_chain_references(self):
        token = self.style_change("unlock", confirmed=True)["builtin_unlock_token"]
        self.style_change("scope", scope="local", builtin_unlock_token=token)
        local_before = self.styles.path.read_bytes()
        updated = copy.deepcopy(self.style)
        updated["settings"]["values"]["img_w"] = 1024
        write_json_atomic(self.styles.builtin_file, {"version": 1, "presets": [updated]})
        catalog = self.styles.catalog()
        local = next(item for item in catalog["presets"] if item["scope"] == "local")
        shipped = next(item for item in catalog["presets"] if item["scope"] == "builtin")
        self.assertEqual(local["id"], "shipped")
        self.assertEqual(local["settings"], self.style["settings"])
        self.assertNotEqual(shipped["id"], local["id"])
        self.assertEqual(shipped["settings"], updated["settings"])
        self.assertEqual(self.styles.path.read_bytes(), local_before)
        self.style_change("rename", id=shipped["id"], name="새 배포 이름", builtin_unlock_token=token)
        self.assertEqual(json.loads(self.styles.builtin_file.read_bytes())["presets"][0]["id"], "shipped")
        self.assertEqual(self.styles.path.read_bytes(), local_before)

    def test_character_readonly_copy_and_autosave_keep_other_tags(self):
        before = self.characters.builtin_file.read_bytes()
        for action in ("save", "delete", "scope"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.characters.change({"action": action, "name": "배포 캐릭터", "fields": {"group": "bad"}, "scope": "local", "revision": self.characters.catalog()["revision"]})
        result = self.characters.change({"action": "duplicate", "name": "Copy", "source": "배포 캐릭터"})
        self.assertEqual(result["characters"]["Copy"], self.character)
        self.assertEqual(result["scopes"]["Copy"], "local")
        self.characters.change({"action": "save", "name": "Copy", "fields": {"settings": {"defaults": {}, "values": {"appearance": "new", "snapref_images": [{"name": "new.png"}]}}}})
        local = json.loads(self.characters.path.read_text(encoding="utf-8"))
        self.assertEqual(local["outfits"], {"keep": ["dress"]})
        self.assertEqual(local["characters"]["Copy"]["appearance"], "new")
        self.assertEqual(self.characters.builtin_file.read_bytes(), before)

    def test_character_group_transfer_retains_all_settings_and_revision_guards(self):
        token = self.characters.change({"action": "unlock", "confirmed": True})["builtin_unlock_token"]
        before = self.characters.catalog()
        self.characters.change({"action": "scope", "name": "내 캐릭터", "scope": "builtin", "revision": before["revision"], "builtin_unlock_token": token})
        result = self.characters.catalog()
        self.assertEqual(result["characters"]["내 캐릭터"], self.character)
        self.assertEqual(result["scopes"]["내 캐릭터"], "builtin")
        self.assertNotIn("내 캐릭터", json.loads(self.characters.path.read_text(encoding="utf-8"))["characters"])
        with self.assertRaises(ValueError):
            self.characters.change({"action": "scope", "name": "내 캐릭터", "scope": "local", "revision": before["revision"], "builtin_unlock_token": token})
        self.characters.change({"action": "scope", "name": "내 캐릭터", "scope": "local", "revision": result["revision"], "builtin_unlock_token": token})
        self.assertEqual(self.characters.catalog()["scopes"]["내 캐릭터"], "local")

    def test_character_failed_transfer_restores_original_catalog(self):
        token = self.characters.change({"action": "unlock", "confirmed": True})["builtin_unlock_token"]
        before = self.characters.catalog()
        def fail_local(path, value):
            if path == self.characters.path:
                raise OSError("disk full")
            write_json_atomic(path, value)
        from asset_preset_scope import write_scope_documents
        with patch("asset_character_presets.write_scope_documents", side_effect=lambda docs: write_scope_documents(docs, writer=fail_local)), self.assertRaises(OSError):
            self.characters.change({"action": "scope", "name": "내 캐릭터", "scope": "builtin", "revision": before["revision"], "builtin_unlock_token": token})
        self.assertEqual(self.characters.catalog(), before)

    def test_asset_mode_uses_provenance_and_rejects_locked_character_edits(self):
        module = importlib.import_module("modes.asset_mode")
        with patch.object(module, "TAGS_FILE", str(self.characters.path)), patch.object(module, "ASSET_DATA_DIR", str(self.root)), patch.object(module, "load_builtin_presets", return_value={"outfits": {"shipped": ["uniform"], "keep": ["old"]}}):
            mode = module.AssetMode()
            mode.character_presets = self.characters
            mode.load_tags()
            tags = mode.get_tags()
            self.assertEqual(tags["preset_scopes"]["characters"]["배포 캐릭터"], "builtin")
            self.assertEqual(tags["preset_scopes"]["outfits"], {"shipped": "builtin", "keep": "local"})
            self.assertFalse(mode.update_character("배포 캐릭터", group="blocked")["success"])
            token = self.characters.change({"action": "unlock", "confirmed": True})["builtin_unlock_token"]
            self.assertTrue(mode.update_character("배포 캐릭터", group="edited", builtin_unlock_token=token)["success"])
            self.assertEqual(mode.get_tags()["characters"]["배포 캐릭터"]["group"], "edited")
            self.assertIn("배포 캐릭터", mode.list_characters())


class CharacterGroupApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_scope_endpoint_shares_lock_with_character_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AssetCharacterPresets(root / "tags.json", root / "builtin.json")
            store.change({"action": "create", "name": "sample"})
            app = web.Application()
            register_character_preset_runtime(SimpleNamespace(app=app, asset_mode=SimpleNamespace(character_presets=store, load_tags=lambda: None)))
            async with TestClient(TestServer(app)) as client:
                catalog = await (await client.get("/api/asset/character-presets")).json()
                body = {"action": "scope", "name": "sample", "scope": "builtin", "revision": catalog["revision"]}
                self.assertEqual((await client.post("/api/asset/character-presets", json=body)).status, 400)
                unlocked = await (await client.post("/api/asset/character-presets", json={"action": "unlock", "confirmed": True})).json()
                body["builtin_unlock_token"] = unlocked["builtin_unlock_token"]
                response = await client.post("/api/asset/character-presets", json=body)
                self.assertEqual(response.status, 200)
                self.assertEqual((await response.json())["scopes"]["sample"], "builtin")
                store.change({"action": "save", "name": "sample", "fields": {"group": "edited"}, "builtin_unlock_token": unlocked["builtin_unlock_token"]})


if __name__ == "__main__":
    unittest.main()
