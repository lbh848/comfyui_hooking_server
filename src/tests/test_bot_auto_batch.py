import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

modes = Path(__file__).resolve().parents[1] / "modes"
package = ModuleType("modes")
package.__path__ = [str(modes)]
with patch.dict(sys.modules, {"modes": package}):
    spec = importlib.util.spec_from_file_location("auto_batch_test_module", modes / "bot_auto_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
candidate_identity = module.candidate_identity


class CandidateIdentityTests(unittest.TestCase):
    def test_matches_user_index_anywhere_in_filename(self):
        for filename in ("HAPPY_A_Casual.png", "A_HAPPY_Casual.png", "A_Casual_HAPPY.png"):
            with self.subTest(filename=filename):
                self.assertEqual(candidate_identity(filename, ["happy"]), ("a_casual", "happy"))
        self.assertEqual(candidate_identity("Adachi_Electric Amber_HAPPY.webp.webp", ["happy"]), ("adachi_electric_amber", "happy"))
        self.assertEqual(candidate_identity("Hibiki_Casual_pouting.webp..webp", ["pouting"]), ("hibiki_casual", "pouting"))

    def test_search_is_literal_and_does_not_infer_meaning_or_boundaries(self):
        self.assertEqual(candidate_identity("A_schooluniform.png", ["school"]), ("a_uniform", "school"))
        self.assertEqual(candidate_identity("A[alt]pose.png", ["[alt]"]), ("apose", "[alt]"))
        self.assertEqual(candidate_identity("교복-캐릭터-정면.png", ["교복"]), ("캐릭터_정면", "교복"))
        self.assertEqual(candidate_identity("A_happy outfit_sad.png", ["happy"]), ("a_outfit_sad", "happy"))
        self.assertIsNone(candidate_identity("A_outfit.png", [".*"]))

    def test_longer_supplied_index_wins_and_missing_index_does_not_match(self):
        self.assertEqual(candidate_identity("A_School_happy tears.png", ["happy", "happy tears"]), ("a_school", "happy tears"))
        self.assertEqual(candidate_identity("A_School_happy tears.png", ["happy"]), ("a_school_tears", "happy"))
        self.assertIsNone(candidate_identity("portrait.png", ["school"]))

    def test_preview_groups_by_remainder_for_different_naming_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            filenames = ["red-A-full.png", "A-blue-full.webp", "A-full-green.jpg"]
            for filename in [*filenames, "_face_image.webp", "unmatched.png"]:
                (directory / filename).touch()
            row = module.character_candidates("TestBot", {"name": "A"}, directory, ["red", "blue", "green"])
            self.assertEqual(len(row["groups"]), 1)
            self.assertEqual(row["groups"][0]["key"], "a_full")
            self.assertFalse(row["groups"][0].get("fallback_first"))
            self.assertCountEqual([image["filename"] for image in row["groups"][0]["candidates"]], filenames)

    def test_normalizes_explicit_search_input_without_a_catalog(self):
        self.assertEqual(module.normalize_indices(" School | [ALT] \n school ; 정면"), ["school", "[alt]", "정면"])
        self.assertEqual(module.normalize_indices(" \n | , ; "), [])
        self.assertEqual(module.normalize_indices([]), [])
        with self.assertRaises(ValueError):
            module.normalize_indices([str(number) for number in range(9)])
        with self.assertRaises(ValueError):
            module.normalize_indices("x" * 81)

    def test_empty_index_shows_all_candidate_images_and_clearing_restores_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            images = ["red-A-full.png", "A-blue-full.webp", "A-full-green.jpg", "portrait.png"]
            for filename in [*images, "_face_image.webp", "notes.txt"]:
                (directory / filename).touch()
            data = {"bots": [{"name": "TestBot", "characters": [{"name": "A"}]}]}
            character_dir = directory / "TestBot" / "A"
            character_dir.mkdir(parents=True)
            for filename in directory.iterdir():
                if filename.is_file():
                    filename.rename(character_dir / filename.name)
            for query, expected in [("", images), ("blue", ["A-blue-full.webp"]), ([], images)]:
                with self.subTest(query=query):
                    result = module.preview(data, "TestBot", directory, query, ["A"])
                    candidates = [candidate for group in result["rows"][0]["groups"] for candidate in group["candidates"]]
                    self.assertCountEqual([candidate["filename"] for candidate in candidates], expected)
                    if not query:
                        self.assertTrue(all(candidate["emotion"] == "" for candidate in candidates))


class FallbackRepresentativeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.character_dir = self.directory / "TestBot" / "A"
        self.character_dir.mkdir(parents=True)
        self.character = {"name": "A", "visual_cards": [
            {"id": "primary", "label": "Primary", "rep_images": ["old.png"]},
            {"id": "secondary", "label": "Secondary", "rep_images": ["b.png"]},
        ]}
        self.data = {"bots": [{"name": "TestBot", "characters": [self.character]}]}

    def preview(self):
        return module.preview(self.data, "TestBot", self.directory, ["unmatched"], ["A"])["rows"][0]

    def body(self, row):
        return {"bot_name": "TestBot", "character": "A", "indices": ["unmatched"], "fingerprint": row["fingerprint"]}

    async def test_no_match_defaults_to_first_image_without_vision_or_candidate_limit(self):
        filenames = ["b.png", "A.webp", *[f"z{index:02}.jpg" for index in range(17)]]
        for filename in [*filenames, "_face_image.webp", "notes.txt"]:
            (self.character_dir / filename).touch()
        (self.character_dir / "0_directory.png").mkdir()
        row = self.preview()
        self.assertEqual(row["error"], "")
        self.assertEqual(len(row["groups"]), 1)
        group = row["groups"][0]
        self.assertTrue(group["fallback_first"])
        self.assertEqual([candidate["filename"] for candidate in group["candidates"]], sorted(filenames, key=str.casefold))
        with patch.dict(sys.modules, {"modes.llm_service": None, "modes.llm_prompt_edit": None}):
            selection = await module.select_candidate(group, self.character_dir)
        self.assertEqual(selection["filename"], "A.webp")
        checked = module.checked_row(self.data, self.directory, self.body(row))
        items = module.apply_items(checked, [{"key": group["key"], **selection}])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["filename"], "A.webp")
        self.assertEqual(items[0]["visual_card_id"], "primary")

    async def test_manual_choice_is_allowed_and_secondary_cards_are_preserved(self):
        for filename in ("a.png", "b.png"):
            (self.character_dir / filename).touch()
        row = self.preview()
        group = row["groups"][0]
        self.assertEqual(row["removed_card_ids"], [])
        self.assertEqual(row["new_count"], 0)
        items = module.apply_items(row, [{"key": group["key"], "filename": "b.png"}])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["visual_card_id"], "primary")
        self.assertEqual(items[0]["filename"], "b.png")
        self.assertFalse(items[0]["create_profile"])
        with self.assertRaises(ValueError):
            module.apply_items(row, [{"key": group["key"], "filename": "missing.png"}])

    async def test_fallback_rejects_stale_images_and_new_search_matches(self):
        for filename in ("a.png", "b.png"):
            (self.character_dir / filename).touch()
        row = self.preview()
        (self.character_dir / "b.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "다시 검색"):
            module.checked_row(self.data, self.directory, self.body(row))
        row = self.preview()
        (self.character_dir / "unmatched.png").touch()
        with self.assertRaisesRegex(ValueError, "다시 검색"):
            module.checked_row(self.data, self.directory, self.body(row))

    async def test_no_images_remains_unavailable(self):
        for filename in ("_face_image.webp", "notes.txt"):
            (self.character_dir / filename).touch()
        row = self.preview()
        self.assertEqual(row["groups"], [])
        self.assertEqual(row["removed_card_ids"], [])
        with self.assertRaisesRegex(ValueError, "이미지가 없습니다"):
            module.checked_row(self.data, self.directory, self.body(row))


if __name__ == "__main__":
    unittest.main()
