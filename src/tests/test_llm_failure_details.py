"""Provider stops remain visible through validation, retries and old history."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from app_paths import SOURCE_DIR
from llm_failure_details import format_llm_failure
from settings_llm_logs import register_llm_logs
from test_llm_settings import load_engine


VALIDATION_ERROR = "CALL2-DETAIL: 완료 컷 보존; 누락 컷만 재시도"


class RoutedFailureDetailsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = load_engine()
        self.metadata = {}
        self.failures = []
        self.policy = {"max_retries": 0, "retry_delay_sec": 0,
                       "fallback_max_retries": 0, "fallback_retry_delay_sec": 0}
        for patcher in (
            patch.object(self.engine, "_routing_for", return_value=("llm7", None)),
            patch.object(self.engine, "_routing_retry_policy", return_value=self.policy),
            patch.object(self.engine, "_base_config_get", side_effect=lambda key, default=None: default),
            patch.object(self.engine, "_llm_log"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_provider_reason_reaches_attempt_final_result_and_observers(self):
        raw = '{"scenes":[{"slot":"C008","instruction":"unfinished'

        async def invoke(*args, **kwargs):
            self.engine._usage_sink_ctx.get().update(
                finish_reason="SAFETY", finish_message="Provider blocked this response.")
            return raw

        with patch.object(self.engine, "_call_routed_text_slot", side_effect=invoke):
            result = await self.engine.callLLMTaskResult(
                "illustration_call2", [], metadata_sink=self.metadata,
                result_validator=lambda text: (False, VALIDATION_ERROR),
                on_attempt_failure=self.failures.append)
        self.assertFalse(result.accepted)
        self.assertEqual(result.raw_response, raw)
        for reason in (result.reason, result.text, self.failures[0]["reason"], result.events[-1]["reason"]):
            self.assertIn(VALIDATION_ERROR, reason)
            self.assertIn("SAFETY", reason)
            self.assertIn("안전 필터", reason)
            self.assertIn("Provider blocked this response.", reason)
        self.assertEqual(len(self.failures), 1)

    async def test_later_timeout_does_not_inherit_previous_safety_reason(self):
        self.policy["max_retries"] = 1
        calls = 0

        async def invoke(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.engine._usage_sink_ctx.get().update(finish_reason="SAFETY", finish_message="blocked")
                return '{"scenes":['
            return "[LLM 실패] vertex stream 최초 무응답 제한 초과: 90초 (partial_len=0)"

        with patch.object(self.engine, "_call_routed_text_slot", side_effect=invoke):
            result = await self.engine.callLLMTaskResult(
                "illustration_call2", [], metadata_sink=self.metadata,
                result_validator=lambda text: (False, VALIDATION_ERROR),
                on_attempt_failure=self.failures.append)
        self.assertEqual(calls, 2)
        self.assertIn("SAFETY", self.failures[0]["reason"])
        self.assertIn("90초", result.text)
        self.assertNotIn("SAFETY", result.reason)
        self.assertNotIn("blocked", result.reason)

    async def test_retry_keeps_completed_scenes_and_uses_only_missing_request(self):
        self.policy["max_retries"] = 1
        messages = [{"role": "user", "content": "C008,C011"}]
        requests, completed = [], []

        async def invoke(slot, request, **kwargs):
            requests.append(copy.deepcopy(request))
            self.engine._usage_sink_ctx.get()["finish_reason"] = "SAFETY" if len(requests) == 1 else "STOP"
            return "C008" if len(requests) == 1 else "C011"

        def validate(text):
            completed.append(text)
            messages[0]["content"] = "C011"
            return (False, VALIDATION_ERROR) if text == "C008" else (True, "")

        with patch.object(self.engine, "_call_routed_text_slot", side_effect=invoke):
            result = await self.engine.callLLMTaskResult(
                "illustration_call2", messages, metadata_sink=self.metadata,
                result_validator=validate, retry_messages=messages,
                on_attempt_failure=self.failures.append)
        self.assertTrue(result.accepted)
        self.assertEqual(result.text, "C011")
        self.assertEqual(completed, ["C008", "C011"])
        self.assertEqual([request[0]["content"] for request in requests], ["C008,C011", "C011"])
        self.assertEqual(len(self.failures), 1)
        self.assertIn("SAFETY", self.failures[0]["reason"])
        self.assertEqual(self.metadata["finish_reason"], "STOP")


class HistoryFailureDetailsTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_history_shows_stop_reason_without_rewriting_or_changing_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data/logs/lighbd_history.jsonl"
            path.parent.mkdir(parents=True)
            record = {"ts": "2026-11-13T20:09:39", "call_name": "DETAIL-MANGA",
                      "status": "error", "error": VALIDATION_ERROR, "output": '{"scenes":[',
                      "finish_reason": "SAFETY", "finish_message": ""}
            original = json.dumps(record, ensure_ascii=False) + "\n"
            path.write_text(original, encoding="utf-8")
            identity = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest() + "-1"
            # Load the actual history reader without initializing mode singletons.
            package = types.ModuleType("modes")
            package.__path__ = [str(SOURCE_DIR / "modes")]
            spec = importlib.util.spec_from_file_location("modes.lighbd_service", SOURCE_DIR / "modes/lighbd_service.py")
            history_module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"modes": package, spec.name: history_module}):
                spec.loader.exec_module(history_module)
                app = web.Application()
                register_llm_logs(app, root, "test", None)
                async with TestClient(TestServer(app)) as client:
                    headers = {"X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "test"}
                    response = await client.get("/api/setting/llm/history", headers=headers)
                    self.assertEqual(response.status, 200)
                    summary = (await response.json())["history"][0]
                    self.assertEqual(summary["id"], identity)
                    self.assertIn("SAFETY", summary["preview"])
                    response = await client.get(f"/api/setting/llm/history/{identity}", headers=headers)
                    self.assertEqual(response.status, 200)
                    detail = (await response.json())["record"]
                    self.assertIn(VALIDATION_ERROR, detail["error"])
                    self.assertIn("SAFETY", detail["error"])
                    self.assertIn("안전 필터", detail["error"])
                    self.assertEqual(detail["output"], record["output"])
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    async def test_limit_unknown_and_repeated_formatting_preserve_provider_details(self):
        for reason, expected in (("MAX_TOKENS", "출력 토큰 한도"), ("length", "출력 토큰 한도"),
                                 ("NEW_PROVIDER_REASON", "NEW_PROVIDER_REASON")):
            with self.subTest(reason=reason):
                metadata = {"finish_reason": reason, "finish_message": "Provider explanation"}
                error = format_llm_failure(VALIDATION_ERROR, metadata)
                self.assertIn(expected, error)
                self.assertIn("Provider explanation", error)
                self.assertEqual(format_llm_failure(error, metadata), error)
        for metadata in ({}, {"finish_reason": "STOP"}, {"finish_reason": "end_turn"}):
            self.assertEqual(format_llm_failure(VALIDATION_ERROR, metadata), VALIDATION_ERROR)


if __name__ == "__main__":
    unittest.main()
