"""One settings file preserves providers and application settings on every write."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from configuration import config_path, migrate_legacy_config, read_config, read_document, write_config
from json_store import write_json_atomic
from llm_settings import migrate_document, runtime_config
from settings_store import SettingsStore, SettingsConflict
from test_llm_settings import document


class UnifiedConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = config_path(self.root)
        self.store = SettingsStore(self.root)
        self.value = migrate_document(document(), self.path.parent)
        self.value["draft"]["llmProviders"][0]["maxConcurrency"] = 3
        self.value["draft"]["llms"][1]["model"] = "vision-model"
        self.value["apiKeys"]["openai"] = "test-secret"
        self.store.save(self.value, '"missing"')

    def test_migration_preserves_paths_and_new_provider_settings_then_removes_old_file(self):
        old = self.path.with_name("config.json")
        legacy = {"comfy_input_dir": "E:/original/comfy/input", "llm_model2": "",
                  "llm_max_concurrency2": 1, "llm_queue_type_order": ["bot_lb_extra_refine"],
                  "comfy_launch_profiles": {"1": {"auto_start": True}}}
        write_json_atomic(old, legacy)
        migrate_legacy_config(self.path)
        self.assertFalse(old.exists())
        self.assertEqual(self.store.read()[0], self.value)
        expected = {key: value for key, value in legacy.items() if key not in {"llm_model2", "llm_max_concurrency2"}}
        self.assertEqual(read_config(self.path), expected)
        self.assertEqual(runtime_config(self.store.read()[0])["llm_max_concurrency2"], 3)
        before = self.path.read_bytes()
        migrate_legacy_config(self.path)
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(len(list((self.root / "data/backups/config_migration").glob("legacy_config_*.json"))), 1)

    def test_app_save_and_form_save_preserve_each_others_sections(self):
        value, revision = self.store.read()
        write_config(self.path, {"comfyui_port": 8190, "llm_max_concurrency2": 1})
        self.assertEqual(self.store.read()[1], revision)
        value["draft"]["llmProviders"][0]["maxConcurrency"] = 5
        self.store.save(value, revision)
        self.assertEqual(read_config(self.path), {"comfyui_port": 8190})
        self.assertEqual(self.store.read()[0], value)
        write_config(self.path, {"comfyui_port": 8191})
        self.assertEqual(self.store.read()[0], value)
        self.assertEqual(set(read_document(self.path)), {"version", "draft", "apiKeys", "appConfig"})

    def test_stale_form_cannot_overwrite_provider_edits(self):
        value, revision = self.store.read()
        changed = copy.deepcopy(value)
        changed["draft"]["llmProviders"][0]["maxConcurrency"] = 4
        self.store.save(changed, revision)
        value["draft"]["llmProviders"][0]["maxConcurrency"] = 2
        with self.assertRaises(SettingsConflict):
            self.store.save(value, revision)

    def test_first_install_app_save_creates_valid_settings_document(self):
        fresh = config_path(self.root / "fresh")
        write_config(fresh, {"comfyui_port": 8188})
        self.assertIsNotNone(SettingsStore(self.root / "fresh").read()[0])
        self.assertEqual(read_config(fresh), {"comfyui_port": 8188})
        self.assertFalse(fresh.with_name("config.json").exists())

    def test_invalid_legacy_file_is_retained(self):
        legacy = self.path.with_name("config.json")
        legacy.write_text("[]", encoding="utf-8")
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            migrate_legacy_config(self.path)
        self.assertTrue(legacy.exists())
        self.assertEqual(self.path.read_bytes(), before)

    def test_parallel_form_and_runtime_writes_do_not_lose_values(self):
        def save_app():
            for port in range(9000, 9010):
                write_config(self.path, {"comfyui_port": port})

        def save_form():
            for limit in range(1, 11):
                value, revision = self.store.read()
                value["draft"]["llmProviders"][0]["maxConcurrency"] = limit
                self.store.save(value, revision)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(save_app), pool.submit(save_form)]
            for future in futures:
                future.result()
        self.assertEqual(read_config(self.path)["comfyui_port"], 9009)
        self.assertEqual(self.store.read()[0]["draft"]["llmProviders"][0]["maxConcurrency"], 10)

    def test_installer_restore_preserves_current_provider_settings(self):
        from comfy_installer.configurator import backup_current_config, restore_config_backup
        write_config(self.path, {"comfy_input_dir": "before/input"})
        backups = self.root / "backups"
        backup = backup_current_config(config_path=self.path, backup_dir=backups, reason="comfy_install")
        value, revision = self.store.read()
        value["draft"]["llmProviders"][0]["maxConcurrency"] = 7
        self.store.save(value, revision)
        write_config(self.path, {"comfy_input_dir": "after/input"})
        restore_config_backup(config_path=self.path, requirements_dir=backups, backup_path=backup["backup_path"])
        self.assertEqual(read_config(self.path), {"comfy_input_dir": "before/input"})
        self.assertEqual(self.store.read()[0], value)

    def test_installer_can_restore_old_flat_backup_into_unified_file(self):
        from comfy_installer.configurator import restore_config_backup
        backups = self.root / "backups"
        backup = backups / "config_before_comfy_install_old.json"
        write_json_atomic(backup, {"comfyui_port": 9000, "llm_max_concurrency2": 1})
        restore_config_backup(config_path=self.path, requirements_dir=backups, backup_path=backup)
        self.assertEqual(read_config(self.path), {"comfyui_port": 9000})
        self.assertEqual(self.store.read()[0], self.value)

