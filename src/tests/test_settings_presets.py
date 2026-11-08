"""Preset deletion targets exact storage/state and never partially validates a batch."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from settings_presets import SettingsPresets
from settings_store import SettingsConflict


class SettingsPresetDeletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = SettingsPresets(self.root / "src")
        self.write(self.store.storage_dir / "tags.json", {
            "appearances": {"same": ["active"], "keep": ["keep"]},
            "characters": {"person": {"appearance": "same"}},
            "natural_language_presets": {"text": "example"},
        })
        self.write(self.store.storage_dir / "hidden_tags.json", {"appearances": {"same": ["hidden"], "hidden-only": ["tag"]}})
        self.write(self.store.builtin_file, {"version": 1, "presets": {"appearances": {"same": ["builtin"]}}})

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def delete(self, items=None, **fields):
        body = {"category": "appearances", "scope": "local", "confirmed": True,
                "items": items if items is not None else [{"name": "same", "state": "active"}], **fields}
        return self.store.mutate("delete", body, self.store.revision())

    def test_deletes_only_requested_scope_and_state_and_keeps_backup(self):
        result = self.delete()
        catalog = SettingsPresets(self.root / "src").catalog()
        self.assertEqual(result["deleted_count"], 1)
        self.assertNotIn("same", catalog["active"]["appearances"])
        self.assertEqual(catalog["active"]["appearances"]["keep"], ["keep"])
        self.assertEqual(catalog["hidden"]["appearances"]["same"], ["hidden"])
        self.assertEqual(catalog["builtin"]["appearances"]["same"], ["builtin"])
        self.assertEqual(catalog["active"]["characters"]["person"], {"appearance": "same"})
        self.assertTrue(list((self.store.storage_dir / "backup").glob("*")))

    def test_batch_can_delete_active_and_hidden_and_deduplicates(self):
        result = self.delete([{"name": "same", "state": "active"}, {"name": "same", "state": "hidden"}, {"name": "same", "state": "active"}])
        self.assertEqual(result["deleted_count"], 2)
        catalog = self.store.catalog()
        self.assertNotIn("same", catalog["active"]["appearances"])
        self.assertNotIn("same", catalog["hidden"]["appearances"])

    def test_rejects_missing_target_before_deleting_anything(self):
        revision = self.store.revision()
        with self.assertRaises(ValueError):
            self.delete([{"name": "same", "state": "active"}, {"name": "missing", "state": "hidden"}])
        self.assertEqual(self.store.revision(), revision)

    def test_confirmation_and_exact_state_are_required(self):
        revision = self.store.revision()
        for fields in ({"confirmed": False}, {"items": []}, {"items": [{"name": "same"}]},
                       {"items": [{"name": "same", "state": "invalid"}]}, {"scope": "invalid"}, {"category": "invalid"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.delete(**fields)
        self.assertEqual(self.store.revision(), revision)

    def test_stale_revision_cannot_delete(self):
        revision = self.store.revision()
        self.write(self.store.storage_dir / "hidden_tags.json", {})
        with self.assertRaises(SettingsConflict):
            self.store.mutate("delete", {"name": "same", "category": "appearances", "confirmed": True}, revision)
        self.assertIn("same", self.store.catalog()["active"]["appearances"])

    def test_builtin_requires_unlock_and_preserves_same_named_local(self):
        with self.assertRaises(ValueError):
            self.delete(scope="builtin")
        token = self.store.builtin_lock("unlock", {"confirmed": True})["builtin_unlock_token"]
        self.store.mutate("delete", {"category": "appearances", "name": "same", "scope": "builtin",
                                    "confirmed": True, "builtin_unlock_token": token}, self.store.revision())
        catalog = self.store.catalog()
        self.assertNotIn("same", catalog["builtin"]["appearances"])
        self.assertIn("same", catalog["active"]["appearances"])
        self.assertIn("same", catalog["hidden"]["appearances"])

    def test_character_and_text_deletion_preserve_asset_files(self):
        asset = self.root / "data" / "asset" / "person" / "example.png"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"image")
        self.delete([{"name": "person", "state": "active"}], category="characters")
        self.delete([{"name": "text", "state": "active"}], category="natural_language_presets")
        self.assertEqual(asset.read_bytes(), b"image")
        self.assertEqual(self.store.catalog()["active"]["characters"], {})
        self.assertEqual(self.store.catalog()["active"]["natural_language_presets"], {})

    def chain(self, name, hidden=False):
        path = Path(self.store.chains._preset_path(name, hidden=hidden))
        self.write(path, {"name": name, "chains": [], "repeat": 1})
        return path

    def test_chain_deletion_handles_hidden_and_preserves_other_state(self):
        active = self.chain("same")
        hidden = self.chain("same", hidden=True)
        keep = self.chain("keep")
        self.delete([{"name": "same", "state": "hidden"}], category="chain_presets")
        self.assertTrue(active.is_file())
        self.assertFalse(hidden.exists())
        self.assertTrue(keep.is_file())
        from data_backups import backup_policy
        policy = backup_policy(hidden)
        self.assertTrue(list(policy.directory.glob(policy.pattern)))
        self.delete([{"name": "same", "state": "active"}, {"name": "keep", "state": "active"}], category="chain_presets")
        self.assertFalse(active.exists())
        self.assertFalse(keep.exists())

    def test_chain_batch_validation_and_path_traversal_do_not_delete(self):
        active = self.chain("same")
        for name in ("missing", "../same", "..\\same"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.delete([{"name": "same", "state": "active"}, {"name": name, "state": "active"}], category="chain_presets")
            self.assertTrue(active.exists())

    def test_failed_chain_batch_restores_already_removed_files(self):
        first, second = self.chain("first"), self.chain("second")
        original = Path.unlink
        def fail_second(path, *args, **kwargs):
            if path == second:
                raise OSError("simulated deletion failure")
            return original(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_second), self.assertRaises(OSError):
            self.delete([{"name": "first", "state": "active"}, {"name": "second", "state": "active"}], category="chain_presets")
        self.assertEqual(json.loads(first.read_text())["name"], "first")
        self.assertEqual(json.loads(second.read_text())["name"], "second")


if __name__ == "__main__":
    unittest.main()
