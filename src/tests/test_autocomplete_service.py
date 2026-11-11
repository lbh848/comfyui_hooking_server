"""Autocomplete supports both upstream keyword formats and CSV upgrades."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[1]


class AutocompleteTests(unittest.TestCase):
    def setUp(self):
        # Avoid initializing the unrelated modes singletons and their dependencies.
        spec = importlib.util.spec_from_file_location(
            "autocomplete_under_test", SOURCE_DIR / "modes" / "autocomplete_service.py",
        )
        self.service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.service)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_csv(self, filename, rows):
        path = self.directory / filename
        with path.open("w", encoding="utf-8-sig", newline="") as output:
            csv.writer(output).writerows(rows)
        return path

    def names(self, query):
        return [tag["name"] for tag in self.service.search_tags(query)]

    def test_new_keywords_support_plain_and_exact_search(self):
        self.write_csv("danbooru-tags-anima-kr-2026-11-10.csv", [
            ["blue hair", 0, 100, "[패션 > 헤어컬러] 파란색 머리. <키워드> 머리색, 파란 머리, 청발."],
            ["blue eyes", 0, 90, "[인물 > 눈] 파란색 눈동자. <키워드> 눈색, 파란 눈, 벽안."],
        ])
        self.service.load_all_csv(self.directory)
        for query in ("<머리색>", "<청발>", "청발", "파란_머리", "blue_hair"):
            with self.subTest(query=query):
                self.assertEqual(self.names(query), ["blue hair"])
        self.assertEqual(self.names("<눈색>"), ["blue eyes"])
        self.assertEqual(self.names("<키워드>"), [])
        self.assertEqual(self.names("<머리>"), [])
        hair = self.service._tags[0]
        self.assertEqual(hair["body"], "파란색 머리")
        self.assertEqual(self.service._score_tag(hair, "청발", False), 4)

    def test_legacy_keywords_keep_category_and_plain_search_distinct(self):
        self.write_csv("danbooru-tags-anima-kr-2026-10-05-v2.csv", [
            ["blue hair", 0, 100, "[패션 > 헤어컬러] 파란색 머리. 키워드: <머리색>, 파란 머리, 청발"],
        ])
        self.service.load_all_csv(self.directory)
        self.assertEqual(self.names("<머리색>"), ["blue hair"])
        self.assertEqual(self.names("청발"), ["blue hair"])
        self.assertEqual(self.names("<청발>"), [])

    def test_new_marker_without_subcategory_is_a_description(self):
        self.write_csv("custom.csv", [["eye", 0, 100, "[인물] 눈. <키워드> 눈, 안구."]])
        self.service.load_all_csv(self.directory)
        self.assertEqual(self.names("눈"), ["eye"])
        self.assertEqual(self.names("<눈>"), ["eye"])
        self.assertEqual(self.names("<안구>"), ["eye"])
        self.assertTrue(self.service.search_tags("eye")[0]["description"])

    def test_latest_official_csv_replaces_older_rows_and_keeps_custom_csv(self):
        old = self.write_csv("danbooru-tags-anima-kr-2026-10-05-v2.csv", [
            ["blue hair", 0, 100, "[패션 > 헤어컬러] 오래된 설명이 더 길어도 최신 파일을 우선함. 키워드: <머리색>, 옛표현"],
            ["removed tag", 0, 10, "[기타 > 삭제] 최신판에서 삭제된 태그. 키워드: <삭제>, 삭제됨"],
        ])
        new = self.write_csv("danbooru-tags-anima-kr-2026-11-10.csv", [
            ["blue hair", 0, 100, "[패션 > 헤어컬러] 파란색 머리. <키워드> 머리색, 청발."],
        ])
        self.assertGreater(old.stat().st_size, new.stat().st_size)
        self.write_csv("custom.csv", [["custom tag", 0, 10, "my_alias"]])
        self.write_csv("custom_for_anima.csv", [["excluded tag", 0, 10, "excluded_alias"]])
        self.service.load_all_csv(self.directory)
        self.assertEqual(self.names("청발"), ["blue hair"])
        self.assertEqual(self.names("옛표현"), [])
        self.assertEqual(self.names("removed tag"), [])
        self.assertEqual(self.names("my alias"), ["custom tag"])
        self.assertEqual(self.names("excluded tag"), [])
        self.assertTrue(old.exists())

    def test_same_date_uses_highest_numeric_revision(self):
        for revision in (None, 2, 10):
            suffix = f"-v{revision}" if revision is not None else ""
            name = f"revision {revision}"
            self.write_csv(f"danbooru-tags-anima-kr-2026-11-10{suffix}.csv", [
                [name, 0, 1, "[기타] 개정판. <키워드> 개정판."],
            ])
        self.service.load_all_csv(self.directory)
        self.assertEqual(self.names("revision"), ["revision 10"])

    def test_plain_description_and_empty_keyword_list(self):
        self.assertEqual(self.service._parse_description("[인물 > 눈] 눈 설명"), {
            "category": "인물 > 눈", "body": "눈 설명", "angle_kws": [], "plain_kws": [],
        })
        for description in ("", "[인물] 눈. <키워드> , , ."):
            with self.subTest(description=description):
                parsed = self.service._parse_description(description)
                self.assertEqual(parsed["angle_kws"], [])
                self.assertEqual(parsed["plain_kws"], [])


if __name__ == "__main__":
    unittest.main()
