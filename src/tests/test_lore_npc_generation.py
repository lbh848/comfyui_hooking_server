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
        attempts = []

        async def call(task, messages, *, result_validator, **kwargs):
            attempts.append(0)
            for reply in next(calls):
                attempts[-1] += 1
                accepted, reason = result_validator(json.dumps(reply))
                if accepted:
                    break
            return SimpleNamespace(accepted=accepted, reason=reason, raw_response=reply, exception=None)

        entries = [{"source": "bot.charx", "entry_id": str(i), "title": name,
                    "content": name * 1000} for i, name in enumerate(("Mira", "Kian"))]
        with patch("modes.llm_service.callLLMTaskResult", side_effect=call):
            result = await generate_lore_npc_cards(entries, {"name": "Story"}, batch_chars=5000)
        return result, attempts

    async def test_mixed_rows_keep_valid_people_without_retrying(self):
        mira = {**self.card, "name": "Mira", "description": "항해사", "anchors": ["C001"]}
        invalid = {**self.card, "name": "Unknown", "description": "Missing evidence", "anchors": ["C099"]}
        kian = {**self.card, "name": "Kian", "description": "Scholar", "anchors": ["C002"]}
        result, attempts = await self.analyze([
            [{"candidates": [mira, invalid]}, {"unexpected": 1}],
            [{"candidates": [kian]}],
        ])
        self.assertEqual(attempts, [1, 1])
        self.assertEqual([(row["name"], row["entry_indexes"]) for row in result["candidates"]],
                         [("Mira", [0]), ("Kian", [1])])
        self.assertEqual([warning["batch"] for warning in result["warnings"]], [1])
        self.assertTrue(result["warnings"][0]["issues"])
        self.assertEqual(result["candidates"][0]["card"]["appearance"], self.card["appearance"])

    async def test_retry_when_no_card_can_be_recovered(self):
        mira = {**self.card, "name": "Mira", "description": "Navigator", "anchors": ["C001"]}
        other = {**self.card, "name": "Ren", "description": "Scholar", "anchors": ["C001"]}
        result, attempts = await self.analyze([
            [{"characters": [{**other, "appearance": []}]}, [mira, other]],
            [{"candidates": []}, {"unexpected": 1}],
        ])
        self.assertEqual(attempts, [2, 1])
        self.assertEqual([row["name"] for row in result["candidates"]], ["Mira", "Ren"])
        self.assertEqual(result["warnings"], [])
