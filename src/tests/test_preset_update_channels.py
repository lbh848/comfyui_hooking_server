"""Real local Git updates and packaged updates preserve the user channel."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from app_paths import SOURCE_DIR
from comfy_installer import updater
from data_backups import backup_presets
from illustration_bot_distribution import IllustrationBotDistribution
from json_store import write_json_atomic
from update_policy import is_distribution_file


class PresetUpdateChannelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.release = self.root / "release"
        self.release.mkdir()
        self.git(self.release, "init", "-b", "main")
        self.git(self.release, "config", "user.name", "Backup Tests")
        self.git(self.release, "config", "user.email", "backup-tests@example.invalid")
        shutil.copy2(SOURCE_DIR.parent / ".gitignore", self.release / ".gitignore")
        self.catalogs = ["src/presets/presets.json", "src/presets/asset_style_presets.json",
                         "src/presets/asset_character_presets.json", "src/presets/illustration_model_presets.json",
                         "src/presets/asset_model_presets.json", "src/defaults/illustration/bot.json",
                         "src/prompts/bot_system_prompt/presets.json"]
        for name in self.catalogs:
            self.put(self.release / name, {"release": 1})
        self.git(self.release, "add", "--all")
        self.git(self.release, "commit", "-m", "Release one")
        self.client = self.root / "client"
        self.git(self.root, "clone", str(self.release), str(self.client))
        self.user_paths = [self.client / name for name in (
            "config/settings.json", "config/asset_style_presets.json",
            "data/asset_data/tags.json", "data/asset_data/hidden_tags.json", "data/asset_data/bot.json",
            "data/chain_presets/active.json", "data/chain_presets/hidden/hidden.json",
            "data/comfy/models/keep.json",
        )]
        for path in self.user_paths:
            self.put(path, {"user": path.name, "long_term_edits": [1, 2, 3]})
        self.originals = {path: path.read_bytes() for path in self.user_paths}
        for name in self.catalogs:
            self.put(self.client / name, {"local_distribution_edit": True})
        backup_presets(self.client / "src")

    @staticmethod
    def git(root, *args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise AssertionError(f"git {args}: {result.stderr}")
        return result.stdout.strip()

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def next_release(self):
        for name in self.catalogs:
            self.put(self.release / name, {"release": 2})
        (self.release / self.catalogs[2]).unlink()
        self.put(self.release / "src/presets/new_category.json", {"release": 2})
        self.git(self.release, "add", "--all")
        self.git(self.release, "commit", "-m", "Release two")

    def assert_channels(self):
        for name in self.catalogs:
            if name == self.catalogs[2]:
                self.assertFalse((self.client / name).exists())
            else:
                self.assertEqual(json.loads((self.client / name).read_bytes()), {"release": 2})
        self.assertEqual(json.loads((self.client / "src/presets/new_category.json").read_bytes()), {"release": 2})
        self.assertEqual({path: path.read_bytes() for path in self.user_paths}, self.originals)
        self.assertEqual(self.git(self.client, "status", "--porcelain"), "")

    def test_git_add_excludes_user_files_and_all_backups_but_includes_distribution(self):
        self.git(self.client, "add", "--all")
        tracked = self.git(self.client, "ls-files").splitlines()
        self.assertTrue(set(self.catalogs).issubset(tracked))
        self.assertFalse(any(name.startswith(("data/", "config/")) for name in tracked))
        ignored = self.git(self.client, "ls-files", "--others", "--ignored", "--exclude-standard")
        self.assertIn("config/settings.json", ignored)
        self.assertIn("data/asset_data/backup/", ignored)

    def test_embedded_git_updater_replaces_modified_added_deleted_distribution_files(self):
        self.next_release()
        # Exercise staged edits as well as ordinary worktree changes.
        self.git(self.client, "add", self.catalogs[1])
        with patch.object(updater, "HOOKING_REPOSITORY", str(self.release)):
            result = updater.update_hooking_server_main(
                project_root=self.client / "src", config_path=self.user_paths[0],
                backup_dir=self.client / "data/backups/installer", cancel_event=Event())
        self.assertTrue(result["changed"])
        self.assert_channels()

    def test_standalone_git_updater_uses_the_same_channel_policy(self):
        from updater_dev.updater import GitAutoUpdater
        self.next_release()
        instance = GitAutoUpdater.__new__(GitAutoUpdater)
        logs = []
        instance._thread_log = logs.append
        instance._do_update({"name": "Fixture", "path": str(self.client), "url": str(self.release), "branch": "main"})
        self.assert_channels()
        self.assertFalse(any("실패" in line or "오류" in line for line in logs), logs)

    def test_packaged_updater_replaces_distribution_and_never_touches_user_inventory_entries(self):
        spec = importlib.util.spec_from_file_location("fixture_update_payload", SOURCE_DIR / "packaging/macos/update_payload.py")
        packaged = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packaged)
        # The prior inventory must include the deleted shipped file.
        packaged.inventory(self.release)
        prior = json.loads((self.release / packaged.INVENTORY).read_bytes())
        prior = {name: digest for name, digest in prior.items() if not name.startswith(".git/")}
        self.put(self.client / packaged.INVENTORY, prior)
        self.next_release()
        for path in self.user_paths:
            self.put(self.release / path.relative_to(self.client), {"incorrect_user_payload": True})
        payload = self.root / "payload"
        shutil.copytree(self.release, payload, ignore=shutil.ignore_patterns(".git", packaged.INVENTORY))
        packaged.inventory(payload)
        inventory = json.loads((payload / packaged.INVENTORY).read_bytes())
        self.assertFalse(any(name.startswith(("data/", "config/")) for name in inventory))
        # Even a stale/incorrect inventory cannot replace or remove user files.
        inventory["config/settings.json"] = packaged.digest(payload / "config/settings.json")
        self.put(payload / packaged.INVENTORY, inventory)
        prior["data/asset_data/tags.json"] = "old-user-file"
        self.put(self.client / packaged.INVENTORY, prior)
        conflicts = packaged.update(payload, self.client)
        self.assertFalse(any(is_distribution_file(name) for name in conflicts))
        for name in self.catalogs:
            self.assertEqual((self.client / name).exists(), name != self.catalogs[2])
            if name != self.catalogs[2]:
                self.assertEqual(json.loads((self.client / name).read_bytes()), {"release": 2})
        self.assertEqual({path: path.read_bytes() for path in self.user_paths}, self.originals)


class BotChannelTests(unittest.TestCase):
    def test_prompt_migration_preserves_same_named_user_and_distribution_choices(self):
        import importlib
        bot_mode = importlib.import_module("modes.bot_mode")
        data = {"system_prompt_presets": {"same": "user work"}, "bots": [
            {"name": "User", "system_prompt_preset": "same", "preset_scope": "local"},
            {"name": "Shipped", "system_prompt_preset": "same", "preset_scope": "builtin"},
        ]}
        before = json.dumps(data, sort_keys=True)
        with patch.object(bot_mode, "_load_builtin_presets", return_value={"same": "release"}):
            bot_mode._migrate_system_prompt_preset(data)
        self.assertEqual(json.dumps(data, sort_keys=True), before)

    def test_new_release_cannot_overwrite_same_named_user_bot_or_user_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = IllustrationBotDistribution(root / "src")
            user = {"name": "User", "distribution_scope": "local", "system_prompt_preset": "shared", "preset_scope": "local"}
            self.put(store.runtime_path, {"bots": [user], "system_prompt_presets": {"shared": "user work"}})
            document = {"bots": [dict(user, system_prompt="release wants this name"),
                                 {"name": "Shipped", "system_prompt_preset": "shared", "preset_scope": "local"}],
                        "system_prompt_presets": {"shared": "release one"}}
            self.put(store.path, document)
            with patch.object(store, "_save_runtime", side_effect=lambda data: write_json_atomic(store.runtime_path, data)):
                store.sync()
                first = json.loads(store.runtime_path.read_bytes())
                self.assertEqual(first["system_prompt_presets"]["shared"], "user work")
                self.assertEqual(next(bot for bot in first["bots"] if bot["name"] == "User"), user)
                shipped = next(bot for bot in first["bots"] if bot["name"] == "Shipped")
                alias = shipped["system_prompt_preset"]
                self.assertNotEqual(alias, "shared")
                self.assertEqual(first["system_prompt_presets"][alias], "release one")
                document["system_prompt_presets"]["shared"] = "release two"
                self.put(store.path, document)
                store.sync()
                second = json.loads(store.runtime_path.read_bytes())
                self.assertEqual(second["system_prompt_presets"]["shared"], "user work")
                self.assertEqual(second["system_prompt_presets"][alias], "release two")
                self.assertEqual(next(bot for bot in second["bots"] if bot["name"] == "User"), user)

    put = staticmethod(PresetUpdateChannelTests.put)


if __name__ == "__main__":
    unittest.main()
