"""Thinking tokens are a subset of total generated tokens, including estimates."""
import unittest
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from llm_token_usage import output_token_stats
from settings_llm_logs import register_llm_logs
from test_llm_settings import load_engine


class TokenUsageTests(unittest.TestCase):
    def test_thinking_only_stream_has_nonzero_speed(self):
        engine = load_engine()
        record = {"status": "생각 중", "active": False, "started_at": 10,
                  "last_event_at": 12, "_prompt_tokens_estimate": 5,
                  "text": "", "reasoning_text": "생각 " * 10, "completion_tokens": 0}
        state = engine._public_stream_state(record)
        self.assertEqual(state["reasoning_tokens"], 10)
        self.assertEqual(state["completion_tokens"], 10)
        self.assertEqual(state["tps"], 5)
        self.assertNotIn("reasoning_tokens", record)

    def test_exact_total_is_preserved_when_only_breakdown_is_estimated(self):
        stats = output_token_stats("answer", "thinking" * 100, completion_tokens=15)
        self.assertEqual(stats["completion_tokens"], 15)
        self.assertEqual(stats["reasoning_tokens"], 15)
        self.assertTrue(stats["reasoning_tokens_estimated"])
        self.assertFalse(stats["tokens_estimated"])

    def test_explicit_zero_is_preserved(self):
        stats = output_token_stats("answer", "thinking", completion_tokens=2, reasoning_tokens=0)
        self.assertEqual(stats["reasoning_tokens"], 0)
        self.assertFalse(stats["reasoning_tokens_estimated"])

    def test_hidden_thinking_usage_is_included_in_fallback_total(self):
        stats = output_token_stats("answer", reasoning_tokens=80)
        self.assertEqual(stats["completion_tokens"], 82)
        self.assertFalse(stats["reasoning_tokens_estimated"])
        self.assertTrue(stats["tokens_estimated"])

    def test_legacy_character_count_can_supply_estimated_breakdown(self):
        stats = output_token_stats("", completion_tokens=100, reasoning_chars=210)
        self.assertEqual(stats["reasoning_tokens"], 70)
        self.assertEqual(stats["completion_tokens"], 100)

    def test_absent_thinking_usage_remains_unknown(self):
        stats = output_token_stats("answer", completion_tokens=2)
        self.assertNotIn("reasoning_tokens", stats)


class TokenUsageIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_llm_final_usage_includes_thinking_breakdown(self):
        engine = load_engine()
        for total in (None, 100):
            async def upstream(*args, **kwargs):
                yield {"type": "done", "text": "answer", "reasoning_text": "thinking" * 15,
                       "completion_tokens": total, "elapsed": 2}

            token = engine._llm_slot_ctx.set(next(iter(engine.LLM_SP_SLOTS)))
            try:
                with patch("llm_sp_workflow.stream", upstream):
                    events = [event async for event in engine._dispatch_stream_unlimited([], "local", "test")]
            finally:
                engine._llm_slot_ctx.reset(token)
            self.assertEqual(events[-1]["completion_tokens"], total if total is not None else 42)
            self.assertEqual(events[-1]["reasoning_tokens"], 40)
            self.assertTrue(events[-1]["reasoning_tokens_estimated"])
            self.assertEqual(events[-1]["tps"], (total if total is not None else 42) / 2)

    async def test_history_list_and_detail_include_legacy_and_exact_thinking_counts(self):
        records = [
            {"ts": "2026-01-01", "completion_tokens": 100, "reasoning_text": "thinking" * 15, "tps": 50},
            {"ts": "2026-01-02", "completion_tokens": 120, "reasoning_tokens": 80, "tps": 60},
        ]
        history_module = types.ModuleType("modes.lighbd_service")
        history_module.LIGHBD_HISTORY_MAX = 300
        history_module._load_lighbd_history = lambda **kwargs: records if kwargs["history_path"].name == "lighbd_history.jsonl" else []
        with tempfile.TemporaryDirectory() as directory:
            app = web.Application()
            register_llm_logs(app, Path(directory), "2", None)
            async with TestClient(TestServer(app)) as client:
                headers = {"X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"}
                with patch.dict(sys.modules, {"modes.lighbd_service": history_module}):
                    response = await client.get("/api/setting/llm/history", headers=headers)
                    self.assertEqual(response.status, 200)
                    rows = (await response.json())["history"]
                    self.assertEqual(rows[0]["reasoning_tokens"], 80)
                    self.assertEqual(rows[1]["reasoning_tokens"], 40)
                    self.assertTrue(rows[1]["reasoning_tokens_estimated"])
                    for row in rows:
                        detail = await client.get("/api/setting/llm/history/" + row["id"], headers=headers)
                        record = (await detail.json())["record"]
                        self.assertEqual(record["reasoning_tokens"], row["reasoning_tokens"])
                        self.assertEqual(record["completion_tokens"], row["completion_tokens"])
                        self.assertEqual(record["tps"], row["tps"])
        self.assertNotIn("reasoning_tokens", records[0])


if __name__ == "__main__":
    unittest.main()
