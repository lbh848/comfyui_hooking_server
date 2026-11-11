"""Emotion preset groups: distributed groups stay locked until a session unlock."""
import json
from pathlib import Path
import tempfile
import unittest

from emotion_preset_groups import EmotionPresetGroups


class EmotionPresetGroupsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "presets").mkdir()
        self.builtin_rows = [{"preset": "SD-80_smug_v4", "name": "smug"}]
        (self.root / "presets" / "emotion_preset_groups.json").write_text(json.dumps({
            "version": 1,
            "groups": {"sfw": [{"name": "기본", "rows": self.builtin_rows}], "nsfw": []},
        }, ensure_ascii=False), encoding="utf-8")
        self.groups = EmotionPresetGroups(self.root)
        self.groups.files["local"] = self.root / "local.json"

    def test_sfw_and_nsfw_are_saved_separately(self):
        self.groups.mutate("save", {"kind": "sfw", "name": "내 SFW", "rows": self.builtin_rows})
        self.groups.mutate("save", {"kind": "nsfw", "name": "내 NSFW", "rows": []})
        catalog = self.groups.catalog()
        self.assertEqual([group["name"] for group in catalog["local"]["sfw"]], ["내 SFW"])
        self.assertEqual([group["name"] for group in catalog["local"]["nsfw"]], ["내 NSFW"])

    def test_builtin_edit_requires_unlock(self):
        with self.assertRaises(ValueError):
            self.groups.mutate("save", {"kind": "sfw", "scope": "builtin", "name": "기본", "rows": []})
        token = self.groups.mutate("unlock", {"confirmed": True})["builtin_unlock_token"]
        catalog = self.groups.mutate("save", {
            "kind": "sfw", "scope": "builtin", "name": "기본", "rows": [], "builtin_unlock_token": token,
        })
        self.assertEqual(catalog["builtin"]["sfw"][0]["rows"], [])
        self.groups.mutate("lock", {"builtin_unlock_token": token})
        with self.assertRaises(ValueError):
            self.groups.mutate("save", {"kind": "sfw", "scope": "builtin", "name": "기본", "rows": []})

    def test_duplicate_always_lands_in_user_groups(self):
        catalog = self.groups.mutate("duplicate", {
            "kind": "sfw", "source_scope": "builtin", "source_name": "기본", "new_name": "기본 복사",
        })
        copied = catalog["local"]["sfw"][0]
        self.assertEqual(copied, {"name": "기본 복사", "rows": self.builtin_rows})
        with self.assertRaises(ValueError):
            self.groups.mutate("duplicate", {
                "kind": "sfw", "source_scope": "builtin", "source_name": "기본", "new_name": "기본 복사",
            })

    def test_create_starts_empty(self):
        catalog = self.groups.mutate("create", {"kind": "nsfw", "name": "새 그룹"})
        self.assertEqual(catalog["local"]["nsfw"], [{"name": "새 그룹", "rows": []}])

    def test_rename_keeps_rows_and_rejects_duplicates(self):
        self.groups.mutate("create", {"kind": "sfw", "name": "첫 그룹"})
        self.groups.mutate("create", {"kind": "sfw", "name": "둘째 그룹"})
        catalog = self.groups.mutate("rename", {
            "kind": "sfw", "name": "첫 그룹", "new_name": "바뀐 이름",
        })
        self.assertEqual([group["name"] for group in catalog["local"]["sfw"]], ["바뀐 이름", "둘째 그룹"])
        with self.assertRaises(ValueError):
            self.groups.mutate("rename", {"kind": "sfw", "name": "바뀐 이름", "new_name": "둘째 그룹"})

    def test_delete_requires_confirmation_and_unlock_for_builtin(self):
        self.groups.mutate("create", {"kind": "nsfw", "name": "지울 그룹"})
        with self.assertRaises(ValueError):
            self.groups.mutate("delete", {"kind": "nsfw", "name": "지울 그룹"})
        catalog = self.groups.mutate("delete", {"kind": "nsfw", "name": "지울 그룹", "confirmed": True})
        self.assertEqual(catalog["local"]["nsfw"], [])
        with self.assertRaises(ValueError):
            self.groups.mutate("delete", {
                "kind": "sfw", "scope": "builtin", "name": "기본", "confirmed": True,
            })


if __name__ == "__main__":
    unittest.main()
