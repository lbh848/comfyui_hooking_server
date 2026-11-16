"""Name suggestions must be grounded in character tags and never save settings."""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

SOURCE = Path(__file__).resolve().parents[1]
package = ModuleType("modes")
package.__path__ = [str(SOURCE / "modes")]
spec = importlib.util.spec_from_file_location("bot_name_tags_under_test", SOURCE / "modes" / "bot_name_tags.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NameTagTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.imports = patch.dict(sys.modules, {"modes": package})
        self.imports.start()
        self.addCleanup(self.imports.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.write_csv("danbooru-tags-anima-kr-2026-11-10.csv", [
            ["hatsune miku", 4, 1000, "[캐릭터 > 하츠네 미쿠] 보컬로이드. <키워드> 하츠네 미쿠, 미쿠."],
            [r"hina \(blue archive\)", 4, 200, "[캐릭터 > 소라사키 히나] 블루 아카이브. <키워드> 소라사키 히나, 히나."],
            [r"hina \(another work\)", 4, 100, "[캐릭터 > 히나] 다른 작품. <키워드> 히나."],
            ["blue hair", 0, 9999, "[외모] 머리색. <키워드> 미쿠."],
            ["vocaloid", 3, 9999, "[작품] 보컬로이드. <키워드> 미쿠."],
            ["miku artist", 1, 9999, "[작가] 작가 이름. <키워드> 미쿠."],
        ])
        stat = self.path.stat()
        self.entries = module._read_dictionary(str(self.path), stat.st_mtime_ns, stat.st_size)
        self.dictionary = (self.path.name, self.entries)

    def write_csv(self, name, rows):
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as output:
            csv.writer(output).writerows(rows)
        return path

    def caller(self, *responses):
        return AsyncMock(side_effect=[json.dumps(response, ensure_ascii=False) for response in responses])

    async def suggest(self, body, caller):
        return await module.suggest_name_tag(body, dictionary=self.dictionary, llm_caller=caller)

    def test_dictionary_uses_latest_numeric_revision_not_custom_files(self):
        self.write_csv("custom.csv", [["fake", 4, 1]])
        self.write_csv("danbooru-tags-anima-kr-2026-10-05-v20.csv", [])
        self.write_csv("danbooru-tags-anima-kr-2026-11-10-v2.csv", [])
        newest = self.write_csv("danbooru-tags-anima-kr-2026-11-10-v10.csv", [])
        self.assertEqual(module.dictionary_path([self.directory]), newest)

    def test_search_filters_categories_and_handles_korean_and_escaped_names(self):
        self.assertEqual([row["tag"] for row in module.find_candidates(self.entries, ["하츠네미쿠"])], ["hatsune miku"])
        self.assertEqual([row["tag"] for row in module.find_candidates(self.entries, ["미쿠"])], ["hatsune miku"])
        rows = module.find_candidates(self.entries, ["hina_(blue_archive)"])
        self.assertEqual(rows[0]["tag"], r"hina \(blue archive\)")
        self.assertEqual(len(self.entries), 3)

    async def test_confirmed_choice_copies_dictionary_tag_ignoring_generated_tag(self):
        caller = self.caller({"candidate_id": 1, "confidence": "high", "reason": "이름 일치", "tag": "invented tag"})
        result = await self.suggest({"character": "하츠네 미쿠"}, caller)
        self.assertEqual(result["tag"], "hatsune miku")
        self.assertEqual(caller.await_count, 1)
        self.assertEqual(caller.call_args.args[0], "bot_name_tag")
        self.assertTrue(caller.call_args.kwargs["json_mode"])

    async def test_ambiguous_choice_does_not_auto_fill(self):
        caller = self.caller({"candidate_id": 1, "confidence": "low", "reason": "동명이인"})
        result = await self.suggest({"character": "히나"}, caller)
        self.assertEqual(result["tag"], "")
        self.assertEqual(len(result["candidates"]), 2)

    async def test_missing_spelling_is_expanded_then_grounded(self):
        caller = self.caller({"queries": ["hatsune_miku"]}, {"candidate_id": 1, "confidence": "high", "reason": "별칭 일치"})
        result = await self.suggest({"character": "初音ミク", "work": "보컬로이드"}, caller)
        self.assertEqual(result["tag"], "hatsune miku")
        self.assertEqual(caller.await_count, 2)
        selection_context = json.loads(caller.call_args.args[1][1]["content"])
        self.assertEqual(selection_context["work"], "보컬로이드")

    async def test_unknown_character_does_not_invent_a_tag(self):
        caller = self.caller({"queries": []})
        result = await self.suggest({"character": "unknown original 123"}, caller)
        self.assertEqual(result["tag"], "")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(caller.await_count, 1)

    async def test_rejects_wrong_work_without_filling(self):
        caller = self.caller({"candidate_id": 0, "confidence": "none", "reason": "작품 불일치"})
        result = await self.suggest({"character": "하츠네 미쿠", "work": "다른 작품"}, caller)
        self.assertEqual(result["tag"], "")

    async def test_failure_preserves_candidates_for_manual_choice(self):
        for caller in (AsyncMock(return_value="not JSON"), AsyncMock(return_value="[LLM 실패] offline"), AsyncMock(side_effect=RuntimeError("offline")), AsyncMock(side_effect=TimeoutError("timeout"))):
            result = await self.suggest({"character": "미쿠"}, caller)
            self.assertEqual(result["tag"], "")
            self.assertEqual(result["candidates"][0]["tag"], "hatsune miku")
            self.assertTrue(result["warning"])

    async def test_out_of_pool_choice_never_auto_fills(self):
        for candidate_id in (999, True, "1", -1):
            caller = self.caller({"candidate_id": candidate_id, "confidence": "high", "reason": "invented"})
            result = await self.suggest({"character": "미쿠"}, caller)
            self.assertEqual(result["tag"], "")
            self.assertTrue(result["warning"])

    async def test_bad_input_rejected_before_llm_call(self):
        caller = AsyncMock()
        for body in ([], {}, {"character": False}, {"character": "x" * 201}, {"character": "miku", "work": []}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                await self.suggest(body, caller)
        caller.assert_not_awaited()

    async def test_http_invalid_input_and_success(self):
        response = await module.handle_suggest_name_tag(SimpleNamespace(json=AsyncMock(return_value=[])))
        self.assertEqual(response.status, 400)
        with patch.object(module, "suggest_name_tag", AsyncMock(return_value={"success": True, "tag": "hatsune miku"})):
            response = await module.handle_suggest_name_tag(SimpleNamespace(json=AsyncMock(return_value={"character": "미쿠"})))
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body)["tag"], "hatsune miku")

    def test_gateway_allows_only_post_for_suggestions(self):
        from settings_illustration import allowed_illustration_path
        self.assertTrue(allowed_illustration_path("/api/bot_mode/suggest_name_tag", "POST"))
        self.assertFalse(allowed_illustration_path("/api/bot_mode/suggest_name_tag", "GET"))


if __name__ == "__main__":
    unittest.main()
