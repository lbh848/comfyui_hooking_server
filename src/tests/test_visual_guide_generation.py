"""Profile guides remain a coherent character set across ordinary retries."""
import asyncio
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from test_llm_settings import load_engine


class InlineQueue:
    def __init__(self):
        self.items = []
        self.events = []

    async def _notify_progress(self, item, detail, *, snapshot_detail=None):
        item.progress_detail = deepcopy(snapshot_detail or detail)
        self.events.append(deepcopy(detail))

    async def add_item(self, item_type, label, params, *, runtime_handler, **kwargs):
        item = SimpleNamespace(id="guide-job", type=item_type, params=params,
                               progress_detail={}, status="processing",
                               completion_future=asyncio.get_running_loop().create_future())
        self.items.append(item)
        try:
            result = await runtime_handler(item)
        except Exception as error:
            item.status = "failed"
            item.completion_future.set_exception(error)
        else:
            item.result = result
            item.status = "cancelled" if getattr(item, "_runtime_cancelled", False) else "completed"
            item.completion_future.set_result(result)
        return item


def suggestion(key, guide="Applicable state."):
    return {"target_key": key, "aliases": ["Character"], "selection_guide": guide,
            "evidence": "Source evidence.", "confidence": "high"}


class VisualGuideGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        package = ModuleType("modes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "modes")]
        package_patch = patch.dict(sys.modules, {"modes": package})
        package_patch.start()
        self.addCleanup(package_patch.stop)
        self.bot = importlib.import_module("modes.bot_mode")
        self.engine = load_engine()
        package.llm_service = self.engine
        engine_patch = patch.dict(sys.modules, {"modes.llm_service": self.engine})
        engine_patch.start()
        self.addCleanup(engine_patch.stop)
        self.queue = InlineQueue()
        self.manager = self.bot.BotMode()
        self.manager.set_queue_manager(self.queue)
        self.characters = [
            {"name": name, "visual_cards": [
                {"id": f"card_{index + 1}", "label": f"Form {index + 1}",
                 "aliases": [], "selection_guide": "", "appearance": [], "default_outfit": []}
                for index in range(count)
            ]} for name, count in (("A", 7), ("B", 1))
        ]
        self.requests = []
        self.policy = {"max_retries": 1, "retry_delay_sec": 0,
                       "fallback_max_retries": 0, "fallback_retry_delay_sec": 0}
        self.appearance = AsyncMock(return_value={"status": "ok", "visual_context": "Visible traits.",
                                                  "error": "", "image_filename": "portrait.png"})

        def messages(source, targets, language):
            # Synthetic content exercises retry contracts without pinning production prose.
            return [{"role": "system", "content": "Generate profile metadata."},
                    {"role": "user", "content": json.dumps({
                        "character": targets[0]["character"],
                        "target_keys": [target["target_key"] for target in targets],
                        "registered_ids": [profile["id"] for profile in targets[0]["routing_profiles"]],
                    })}]

        for patcher in (
            patch.object(self.bot, "_load_bot_data", return_value={"bots": [{
                "name": "test", "asset_output_instruction": "Use the supplied character states.",
                "characters": self.characters}]}),
            patch.object(self.bot, "_load_lb_extra", return_value=[]),
            patch.object(self.bot, "_build_visual_guide_messages", side_effect=messages),
            patch.object(self.bot, "_log_visual_guide_llm_history"),
            patch.object(self.bot, "_generate_visual_profile_appearance_context", self.appearance),
            patch.object(self.engine, "_routing_for", return_value=("llm2", None)),
            patch.object(self.engine, "_routing_retry_policy", return_value=self.policy),
            patch.object(self.engine, "routing_primary_max_concurrency", return_value=1),
            patch.object(self.engine, "_base_config_get", side_effect=lambda key, default=None: default),
            patch.object(self.engine, "_llm_log"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def generate(self, invoke, *, names=("A", "B"), images=True):
        body = {"bot_name": "test", "analyze_images": images, "targets": [
            {"character": character["name"], "profile_id": card["id"]}
            for name in names for character in self.characters if character["name"] == name
            for card in character["visual_cards"]
        ]}

        async def call(slot, messages, **kwargs):
            self.requests.append(deepcopy(messages))
            return await invoke(json.loads(messages[1]["content"]), messages)

        with patch.object(self.engine, "_call_routed_text_slot", side_effect=call):
            response = await self.manager.handle_suggest_character_card_metadata(
                SimpleNamespace(json=AsyncMock(return_value=body)))
        return response.status, json.loads(response.text)

    def test_limits_and_nonessential_metadata(self):
        target = [{"target_key": "0", "character": "A", "profile": {"id": "card_1"}}]
        row = suggestion("card_1", "x" * 4000)
        row.update(aliases=[str(index) + "x" * 158 for index in range(32)],
                   evidence="x" * 3000, confidence="uncertain")
        result, reason = self.bot._normalize_visual_guide_llm_result({"suggestions": [row]}, target)
        self.assertEqual(reason, "")
        self.assertEqual(result[0]["confidence"], "low")
        self.assertEqual(result[0]["evidence"], row["evidence"])
        for changes in ({"aliases": ["x" * 161]}, {"aliases": [str(i) for i in range(33)]},
                        {"selection_guide": "x" * 4001}, {"selection_guide": ""}):
            with self.subTest(changes=list(changes)):
                result, reason = self.bot._normalize_visual_guide_llm_result(
                    {"suggestions": [{**row, **changes}]}, target)
                self.assertIsNone(result)
                self.assertTrue(reason)
        self.assertIsNotNone(self.bot._normalize_visual_guide_llm_result(
            {"suggestions": [suggestion(0)]}, target)[0])

    async def test_retry_regenerates_all_seven_and_never_merges_attempts(self):
        attempts = 0

        async def invoke(request, messages):
            nonlocal attempts
            attempts += 1
            rows = [suggestion(key, f"Attempt {attempts}") for key in request["target_keys"]]
            if attempts == 1:
                rows[-1]["selection_guide"] = ""
            return json.dumps({"suggestions": rows})

        status, result = await self.generate(invoke, names=("A",), images=False)
        self.assertEqual(status, 200)
        self.assertEqual(attempts, 2)
        self.assertEqual(len(result["suggestions"]), 7)
        self.assertEqual({row["selection_guide"] for row in result["suggestions"]}, {"Attempt 2"})
        for request in self.requests:
            payload = json.loads(request[1]["content"])
            self.assertEqual(payload["target_keys"], [str(i) for i in range(7)])
            self.assertEqual(len(payload["registered_ids"]), 7)
        feedback = json.loads(self.requests[1][-1]["content"].split("\n", 1)[1])
        self.assertIn("6", feedback["validation_error"])
        self.assertFalse(any(event.get("suggestions") for event in self.queue.events
                             if event.get("stage") != "guide_completed"))

    async def test_failed_character_does_not_block_other_character_appearance(self):
        async def invoke(request, messages):
            if request["character"] == "A":
                return json.dumps({"suggestions": [suggestion("0")]})
            return json.dumps({"suggestions": [suggestion("card_1")]})

        status, result = await self.generate(invoke)
        self.assertEqual(status, 200)
        self.assertFalse(result["cancelled"])
        self.assertEqual([row["character"] for row in result["suggestions"]], ["B"])
        self.assertEqual([row["character"] for row in result["failures"]], ["A"])
        self.appearance.assert_awaited_once()
        self.assertEqual(self.appearance.call_args.kwargs["target"]["character"], "B")
        self.assertEqual(result["suggestions"][0]["visual_context_status"], "ok")
        self.assertEqual(self.queue.items[0].progress_detail["failures"], result["failures"])

    async def test_all_failed_is_not_reported_as_success(self):
        async def invoke(request, messages):
            return "invalid JSON"

        status, result = await self.generate(invoke, names=("A",))
        self.assertEqual(status, 422)
        self.assertFalse(result.get("success", False))
        self.assertEqual(self.queue.items[0].progress_detail["failures"][0]["character"], "A")
        self.appearance.assert_not_awaited()

    async def test_cancel_stops_retry_and_keeps_completed_character(self):
        async def invoke(request, messages):
            if request["character"] == "B":
                return json.dumps({"suggestions": [suggestion("0")]})
            self.queue.items[0]._visual_guide_cancel_requested = True
            return json.dumps({"suggestions": [suggestion("0")]})

        status, result = await self.generate(invoke, names=("B", "A"))
        self.assertEqual(status, 200)
        self.assertTrue(result["cancelled"])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual([row["character"] for row in result["suggestions"]], ["B"])
        self.appearance.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
