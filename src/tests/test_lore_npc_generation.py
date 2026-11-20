import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modes.character_lore import generate_lore_npc_cards


class LoreNpcResponseTests(unittest.IsolatedAsyncioTestCase):
    card = {"gender_tag": "1male", "appearance": ["short brown hair"], "default_outfit": ["blue shirt"],
            "face_tags": ["oval face"], "eye_tags": ["brown eyes"], "visual_context": "A brown-haired villager."}

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
            return await generate_lore_npc_cards(entries, {"name": "Story"}, batch_chars=5000)

    async def test_mixed_rows_and_failed_retry_keep_valid_people_and_process_next_batch(self):
        mira = {**self.card, "name": "Mira", "description": "Navigator", "anchors": ["C001"]}
        invalid = {**self.card, "name": "Unknown", "description": "Missing evidence", "anchors": ["C099"]}
        kian = {**self.card, "name": "Kian", "description": "Scholar", "anchors": ["C002"]}
        result = await self.analyze([
            [{"candidates": [mira, invalid]}, {"unexpected": 1}],
            [{"candidates": [kian]}],
        ])
        self.assertEqual([(row["name"], row["entry_indexes"]) for row in result["candidates"]],
                         [("Mira", [0]), ("Kian", [1])])
        self.assertEqual([warning["batch"] for warning in result["warnings"]], [1])
        self.assertTrue(result["warnings"][0]["issues"])
        self.assertEqual(result["candidates"][0]["card"]["appearance"], self.card["appearance"])

    async def test_corrected_retry_keeps_earlier_valid_person_without_duplicate(self):
        mira = {**self.card, "name": "Mira", "description": "Navigator", "anchors": ["C001"]}
        other = {**self.card, "name": "Ren", "description": "Scholar", "anchors": ["C001"]}
        result = await self.analyze([
            [{"characters": [mira, {**other, "appearance": []}]}, [other]],
            [{"candidates": []}],
        ])
        self.assertEqual([row["name"] for row in result["candidates"]], ["Mira", "Ren"])
        self.assertEqual(result["warnings"], [])
