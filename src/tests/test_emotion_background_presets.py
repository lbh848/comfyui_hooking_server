"""Manual background presets: distribution presets stay locked until a session unlock."""
import json
from pathlib import Path
import tempfile
import unittest

from emotion_background_presets import EmotionBackgroundPresets


def values(**overrides):
    return {
        "text": "",
        "bg_css": ".background{background:#000}",
        "fg_css": "",
        "preview_name": "",
        "preview_url": "",
        **overrides,
    }


class EmotionBackgroundPresetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.builtin = self.root / "presets" / "emotion_background_presets.json"
        self.builtin.parent.mkdir()
        self.builtin.write_text(json.dumps({
            "version": 1,
            "presets": [{"name": "기본", **values(bg_css=".background{background:#fff}")}],
        }, ensure_ascii=False), encoding="utf-8")
        self.store = EmotionBackgroundPresets(self.root / "config.json", self.builtin)

    def unlock(self):
        return self.store.change_lock({"action": "unlock", "confirmed": True})["builtin_unlock_token"]

    def test_builtin_save_requires_confirmed_unlock(self):
        with self.assertRaises(ValueError):
            self.store.save("builtin", "새 배포", values())
        with self.assertRaises(ValueError):
            self.store.change_lock({"action": "unlock"})
        token = self.unlock()
        presets = self.store.save("builtin", "새 배포", values(), token)
        self.assertIn("새 배포", [preset["name"] for preset in presets if preset["scope"] == "builtin"])

    def test_lock_revokes_token(self):
        token = self.unlock()
        self.store.change_lock({"action": "lock", "builtin_unlock_token": token})
        with self.assertRaises(ValueError):
            self.store.save("builtin", "새 배포", values(), token)
        with self.assertRaises(ValueError):
            self.store.rename("기본", "이름", token)

    def test_builtin_rename_delete_and_move_require_unlock(self):
        with self.assertRaises(ValueError):
            self.store.rename("기본", "이름")
        with self.assertRaises(ValueError):
            self.store.delete("기본")
        with self.assertRaises(ValueError):
            self.store.move("기본", "local")
        with self.assertRaises(ValueError):
            self.store.save("local", "사용자", values())
            self.store.move("사용자", "builtin")

        token = self.unlock()
        presets = self.store.rename("기본", "이름", token)
        self.assertIn("이름", [preset["name"] for preset in presets])
        presets = self.store.move("이름", "local", token)
        self.assertEqual(
            [preset["scope"] for preset in presets if preset["name"] == "이름"], ["local"]
        )
        self.store.delete("이름", token)
        self.assertNotIn("이름", [preset["name"] for preset in self.store.list()])

    def test_duplicate_from_builtin_to_user_is_allowed_while_locked(self):
        presets = self.store.duplicate("기본", "기본 복사", "local")
        copied = next(preset for preset in presets if preset["name"] == "기본 복사")
        self.assertEqual(copied["scope"], "local")
        with self.assertRaises(ValueError):
            self.store.duplicate("기본", "배포 복사", "builtin")

    def test_user_preset_edits_do_not_require_unlock(self):
        self.store.save("local", "사용자", values())
        self.store.rename("사용자", "사용자2")
        self.store.duplicate("사용자2", "사용자3", "local")
        self.store.delete("사용자3")
        self.assertEqual([preset["name"] for preset in self.store.list()], ["기본", "사용자2"])

    def test_unknown_token_does_not_unlock(self):
        with self.assertRaises(ValueError):
            self.store.save("builtin", "새 배포", values(), "not-a-token")


if __name__ == "__main__":
    unittest.main()
