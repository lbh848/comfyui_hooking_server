"""Settings/preset histories and startup snapshots use isolated installations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app_paths import initialize_layout
from configuration import read_document, write_config
from data_backups import backup_json_file, backup_policy, backup_presets, content_digest
from json_store import write_json_atomic
from settings_store import SettingsStore
from test_llm_settings import document


class DataBackupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "src"
        self.source.mkdir()

    def put(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def history(self, path):
        policy = backup_policy(path)
        return list(policy.directory.glob(policy.pattern))

    def test_settings_form_and_runtime_share_history_and_skip_identical_saves(self):
        store = SettingsStore(self.source)
        store.save(document(), '"missing"')
        write_config(store.path, {"value": 0})
        value, revision = store.read()
        before = store.path.read_bytes()
        count = len(self.history(store.path))
        store.save(value, revision)
        write_config(store.path, {"value": 0})
        self.assertEqual(store.path.read_bytes(), before)
        self.assertEqual(len(self.history(store.path)), count)
        for version in range(1, 106):
            write_config(store.path, {"value": version})
        history = [json.loads(path.read_bytes()) for path in self.history(store.path)]
        self.assertEqual(len(history), 100)
        self.assertEqual({item["appConfig"]["value"] for item in history}, set(range(5, 105)))
        value, revision = store.read()
        value["draft"]["updates"]["channel"] = "preview"
        previous = read_document(store.path)
        store.save(value, revision)
        self.assertIn(previous, [json.loads(path.read_bytes()) for path in self.history(store.path)])
        self.assertEqual(read_document(store.path)["appConfig"], {"value": 105})
        self.assertEqual(len(self.history(store.path)), 100)

    def test_json_key_order_and_revisited_versions_are_deduplicated(self):
        path = self.put("config/asset_style_presets.json", {"a": 1, "b": 2})
        backup_json_file(path)
        first = self.history(path)
        path.write_text('{"b": 2, "a": 1}', encoding="utf-8")
        backup_json_file(path)
        self.assertEqual(self.history(path), first)
        write_json_atomic(path, {"a": 2})
        write_json_atomic(path, {"a": 1, "b": 2})
        backup_json_file(path)
        self.assertEqual(len(self.history(path)), 2)
        self.assertEqual(len({content_digest(item.read_bytes()) for item in self.history(path)}), 2)

    def test_each_preset_and_channel_keeps_100_independent_versions(self):
        paths = [self.put(relative, {"value": 0}) for relative in (
            "src/presets/asset_style_presets.json", "config/asset_style_presets.json",
            "data/asset_data/tags.json", "data/asset_data/hidden_tags.json", "data/asset_data/bot.json",
        )]
        for version in range(1, 103):
            for path in paths:
                write_json_atomic(path, {"value": version})
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(len(self.history(path)), 100)
                self.assertEqual({json.loads(item.read_bytes())["value"] for item in self.history(path)}, set(range(2, 102)))
                self.assertTrue(backup_policy(path).directory.is_relative_to(self.root / "data"))
        self.assertNotEqual(backup_policy(paths[0]).directory, backup_policy(paths[1]).directory)

    def test_startup_covers_all_preset_locations_without_comfy_or_source_changes(self):
        paths = [self.put(relative, {"value": relative}) for relative in (
            "src/presets/presets.json", "src/presets/asset_character_presets.json",
            "src/prompts/bot_system_prompt/presets.json", "src/defaults/illustration/bot.json",
            "config/illustration_generation_presets.json", "config/asset_model_presets.json",
            "data/asset_data/tags.json", "data/asset_data/hidden_tags.json", "data/asset_data/bot.json",
            "data/chain_presets/active.json", "data/chain_presets/hidden/hidden.json",
        )]
        comfy = self.put("data/comfy/models/do-not-back-up.json", {"model": "large"})
        settings = self.put("config/settings.json", {"only": "presets at startup"})
        originals = {path: path.read_bytes() for path in [*paths, comfy, settings]}
        initialize_layout(self.source)
        first = {path: self.history(path) for path in paths}
        result = backup_presets(self.source)
        self.assertFalse(result["errors"])
        for path in paths:
            self.assertEqual(len(first[path]), 1, path)
            self.assertEqual(self.history(path), first[path])
        self.assertIsNone(backup_policy(comfy))
        self.assertFalse(self.history(settings))
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_existing_config_and_tag_histories_are_pruned_without_mixing_sources(self):
        path = self.put("config/settings.json", {"value": 200})
        policy = backup_policy(path)
        policy.directory.mkdir(parents=True)
        for index in range(105):
            (policy.directory / f"config_before_save_20260101_000000_{index:06}.json").write_text(json.dumps({"value": index}))
        (policy.directory / "config_before_save_20260101_000001_000000.json").write_text('{"value":104}')
        keep = policy.directory / "unrelated.json"
        keep.write_text("keep")
        backup_json_file(path)
        history = [json.loads(item.read_bytes())["value"] for item in self.history(path)]
        self.assertEqual(len(history), 100)
        self.assertEqual(set(history), {*range(6, 105), 200})
        self.assertEqual(keep.read_text(), "keep")

    def test_backup_failure_preserves_original_and_chain_timestamp_is_not_an_edit(self):
        path = self.put("data/chain_presets/example.json", {"chains": [1], "saved_at": "old"})
        write_json_atomic(path, {"chains": [1], "saved_at": "new"})
        self.assertFalse(self.history(path))
        before = path.read_bytes()
        with patch("data_backups.shutil.copy2", side_effect=OSError("disk full")), self.assertRaises(OSError):
            write_json_atomic(path, {"chains": [2], "saved_at": "new"})
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
