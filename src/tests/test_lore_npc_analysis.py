import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modes.character_lore import analyze_lore_npcs


class LoreNpcResponseTests(unittest.IsolatedAsyncioTestCase):
    async def analyze(self, responses):
        calls = iter(responses)

        async def call(task, messages, *, result_validator, **kwargs):
            # Follow the router contract: each attempted response is validated
            # and the final result carries the last failure reason.
            for reply in next(calls):
                accepted, reason = result_validator(json.dumps(reply))
                if accepted:
                    break
            return SimpleNamespace(accepted=accepted, reason=reason, raw_response=reply, exception=None)

        entries = [{"source": "bot.charx", "entry_id": str(i), "title": name,
                    "content": name * 1000} for i, name in enumerate(("Mira", "Kian"))]
        with patch("modes.llm_service.callLLMTaskResult", side_effect=call):
            return await analyze_lore_npcs(entries, {"name": "Story"}, batch_chars=5000)

    async def test_mixed_rows_and_failed_retry_keep_valid_people_and_process_next_batch(self):
        mira = {"name": "Mira", "description": "Navigator", "anchors": ["C001"]}
        invalid = {"name": "Unknown", "description": "Missing evidence", "anchors": ["C099"]}
        kian = {"name": "Kian", "description": "Scholar", "anchors": ["C002"]}
        result = await self.analyze([
            [{"candidates": [mira, invalid]}, {"unexpected": 1}],
            [{"candidates": [kian]}],
        ])
        self.assertEqual([(row["name"], row["entry_indexes"]) for row in result["candidates"]],
                         [("Mira", [0]), ("Kian", [1])])
        self.assertEqual([warning["batch"] for warning in result["warnings"]], [1])
        self.assertTrue(result["warnings"][0]["issues"])

    async def test_corrected_retry_keeps_earlier_valid_person_without_duplicate(self):
        mira = {"name": "Mira", "description": "Navigator", "anchors": ["C001"]}
        other = {"name": "Ren", "description": "Scholar", "anchors": ["C001"]}
        result = await self.analyze([
            [{"candidates": [mira, {**other, "description": ""}]}, [other]],
            [{"candidates": []}],
        ])
        self.assertEqual([row["name"] for row in result["candidates"]], ["Mira", "Ren"])
        self.assertEqual(result["warnings"], [])
