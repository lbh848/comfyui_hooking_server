"""범위 분리는 인덱스가 완성되기 전에도 감정 삽화를 삽화 백업에 섞지 않는다."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


class BackupScopeIndexTests(unittest.TestCase):
    def setUp(self):
        engine = {
            "illustration_flow": SimpleNamespace(WORKSHOPS=("illustration", "emotion")),
            "BACKUP_EMOTION_METHOD": "합성 삽화",
        }
        node = next(node for node in TREE.body if getattr(node, "name", None) == "_backup_scope_matches")
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), engine)
        self.matches = engine["_backup_scope_matches"]

    @staticmethod
    def cache(building, workshops=None, methods=None):
        return {
            "building": building,
            "name_to_workshop": workshops or {},
            "name_to_method": methods or {},
        }

    def test_unscanned_backup_is_unknown_while_building(self):
        # Empty cache published before the first scan batch used to treat every
        # backup as illustration, which briefly showed emotion assets in the
        # illustration backup tab.
        cache = self.cache(True, {"scanned": "illustration"}, {"scanned": ""})
        self.assertIsNone(self.matches(cache, "pending", False))
        self.assertIsNone(self.matches(cache, "pending", True))

    def test_scanned_entries_keep_their_workshop_while_building(self):
        cache = self.cache(
            True,
            {"art": "illustration", "asset": "emotion"},
            {"art": "", "asset": "합성 삽화"},
        )
        self.assertFalse(self.matches(cache, "art", True))
        self.assertTrue(self.matches(cache, "art", False))
        self.assertTrue(self.matches(cache, "asset", True))
        self.assertFalse(self.matches(cache, "asset", False))

    def test_legacy_gen_method_still_splits_completed_cache(self):
        cache = self.cache(False, {}, {"legacy": "합성 삽화", "normal": ""})
        self.assertTrue(self.matches(cache, "legacy", True))
        self.assertFalse(self.matches(cache, "legacy", False))
        self.assertTrue(self.matches(cache, "normal", False))
        self.assertFalse(self.matches(cache, "normal", True))

    def test_file_added_after_completed_scan_falls_back_to_legacy_rule(self):
        cache = self.cache(False, {"known": "emotion"}, {"known": ""})
        self.assertTrue(self.matches(cache, "new", False))
        self.assertFalse(self.matches(cache, "new", True))


if __name__ == "__main__":
    unittest.main()
