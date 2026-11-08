"""Exercise the actual LLM engine against a local provider, without real keys."""
import asyncio
import base64
from contextlib import aclosing
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from settings_llm import LlmTestRequest, LlmTestService
from settings_server import create_app, SETTINGS_UI_SCHEMA
from settings_store import SettingsStore
from test_llm_settings import document, load_engine


class LlmTestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = load_engine()
        self.engine.LOG_DIR = str(self.root / "logs")
        self.engine.HISTORY_PATH = str(self.root / "logs/history.jsonl")
        self.engine.HISTORY_BACKUP_DIR = str(self.root / "backups")
        self.engine.HISTORY_BACKUP_PATH = str(self.root / "backups/history.jsonl")
        self.requests = []
        self.fail = False
        self.hold = False
        self.reasoning = ""
        self.usage = {"prompt_tokens": 4, "completion_tokens": 6}
        self.disconnected = asyncio.Event()

        async def provider(request):
            body = await request.json()
            self.requests.append((body, dict(request.headers)))
            if self.fail:
                return web.json_response({"error": "Rejected test-only-secret-key"}, status=401)
            if not body.get("stream"):
                return web.json_response({"choices": [{"message": {"content": "Hello from the test model", "reasoning_content": self.reasoning}, "finish_reason": "stop"}], "usage": self.usage})
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            try:
                if self.reasoning:
                    await response.write(('data: ' + json.dumps({"choices": [{"delta": {"reasoning_content": self.reasoning}}]}) + '\n\n').encode())
                await response.write(b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n')
                if self.hold:
                    while request.transport is not None and not request.transport.is_closing():
                        await asyncio.sleep(0.02)
                        await response.write(b': keep-alive\n\n')
                    self.disconnected.set()
                    return response
                await response.write(b'data: {"choices":[{"delta":{"content":"from the test model"}}]}\n\n')
                await response.write(('data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": self.usage}) + '\n\n').encode())
                await response.write(b'data: [DONE]\n\n')
            except ConnectionError:
                self.disconnected.set()
            except asyncio.CancelledError:
                self.disconnected.set()
                raise
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", provider)
        self.provider = TestServer(app)
        await self.provider.start_server()
        self.addAsyncCleanup(self.provider.close)
        self.store = SettingsStore(self.root)
        self.store.save(document(), '"missing"')
        self.document, revision = self.store.read()
        self.document["draft"]["llmProviders"][0]["url"] = str(self.provider.make_url("/v1"))
        self.document["apiKeys"]["openai"] = "test-only-secret-key"
        self.document["draft"]["llms"][9].update(model="test-model-10", temperature=0.4, custom_body='{"top_p":0.6}', custom_headers='{"X-Test":"fixture"}')
        self.revision = self.store.save(self.document, revision)
        self.service = LlmTestService(self.engine)
        self.test = LlmTestRequest(target="llm10", prompt="Reply with a short greeting.")

    async def collect(self):
        return [event async for event in self.service.events(self.document, self.test) if event["type"] != "ping"]

    async def test_selected_slot_uses_shared_provider_and_saved_options(self):
        with patch.object(self.engine._settings_reader, "read", side_effect=AssertionError("Read outside request snapshot")):
            events = await self.collect()
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["completion_tokens"], 6)
        body, headers = self.requests[0]
        self.assertEqual(body["model"], "test-model-10")
        self.assertEqual(body["temperature"], 0.4)
        self.assertEqual(body["top_p"], 0.6)
        self.assertEqual(headers["X-Test"], "fixture")
        self.assertEqual(headers["Authorization"], "Bearer test-only-secret-key")
        self.assertIsNone(self.engine._settings_config_ctx.get())

    async def test_stream_delivers_deltas_and_final_response(self):
        self.document["draft"]["llms"][9]["stream"] = True
        events = await self.collect()
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "delta"), "Hello from the test model")
        self.assertEqual(events[-1]["text"], "Hello from the test model")
        self.assertEqual(events[-1]["completion_tokens"], 6)

    async def test_cancel_closes_provider_request_and_releases_capacity(self):
        self.document["draft"]["llms"][9]["stream"] = True
        self.hold = True
        async with aclosing(self.service.events(self.document, self.test)) as events:
            async for event in events:
                if event["type"] == "delta":
                    break
        await asyncio.wait_for(self.disconnected.wait(), 3)
        with self.engine.settings_context(self.document):
            self.assertTrue(self.engine._request_gate("llm10").has_capacity_now())

    async def test_reasoning_usage_reaches_history_without_double_counting(self):
        self.reasoning = "Thinking about the answer. " * 8
        self.usage = {"prompt_tokens": 4, "completion_tokens": 120,
                      "completion_tokens_details": {"reasoning_tokens": 80}}
        records = []
        self.service.history_writer = records.append
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.document["draft"]["llms"][9]["stream"] = streaming
                events = await self.collect()
                self.assertEqual(events[-1]["completion_tokens"], 120)
                record = records[-1]
                self.assertEqual(record["reasoning_tokens"], 80)
                self.assertEqual(record["reasoning_text"], self.reasoning)
                self.assertFalse(record["reasoning_tokens_estimated"])
                self.assertFalse(record["tokens_estimated"])
                self.assertAlmostEqual(record["tps"], 120 / record["elapsed"])

    async def test_missing_usage_counts_thinking_in_final_speed_and_history(self):
        self.reasoning = "Thinking about the answer. " * 8
        self.usage = {}
        records = []
        self.service.history_writer = records.append
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.document["draft"]["llms"][9]["stream"] = streaming
                events = await self.collect()
                expected_reasoning = self.engine._approx_tokens(self.reasoning)
                expected_total = expected_reasoning + self.engine._approx_tokens("Hello from the test model")
                self.assertEqual(events[-1]["completion_tokens"], expected_total)
                record = records[-1]
                self.assertEqual(record["reasoning_tokens"], expected_reasoning)
                self.assertTrue(record["reasoning_tokens_estimated"])
                self.assertTrue(record["tokens_estimated"])
                self.assertAlmostEqual(record["tps"], expected_total / record["elapsed"])

    async def test_live_thinking_tokens_and_partial_result_are_counted(self):
        self.reasoning = "Thinking before answering. " * 8
        self.document["draft"]["llms"][9]["stream"] = True
        self.hold = True
        records = []
        self.service.history_writer = records.append
        controlled = False
        async with aclosing(self.service.events(self.document, self.test)) as events:
            async for event in events:
                if event["type"] != "delta" or not event.get("text") or controlled:
                    continue
                live = self.engine.get_active_streams()[0]
                thinking = self.engine._approx_tokens(self.reasoning)
                self.assertEqual(live["reasoning_tokens"], thinking)
                self.assertTrue(live["reasoning_tokens_estimated"])
                self.assertEqual(live["completion_tokens"], thinking + self.engine._approx_tokens("Hello "))
                self.assertAlmostEqual(live["tps"], live["completion_tokens"] / live["elapsed"])
                success, _ = self.engine.request_stream_control(live["stream_id"], "use_partial")
                self.assertTrue(success)
                controlled = True
        self.assertTrue(controlled)
        self.assertEqual(records[-1]["reasoning_tokens"], thinking)
        self.assertEqual(records[-1]["completion_tokens"], thinking + self.engine._approx_tokens("Hello "))

    async def test_provider_error_is_not_success_and_key_is_redacted(self):
        self.fail = True
        events = await self.collect()
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("test-only-secret-key", events[-1]["error"])
        self.assertIn("401", events[-1]["error"])

    async def test_image_uses_existing_vision_transport(self):
        image = io.BytesIO()
        Image.new("RGB", (2, 2), "white").save(image, format="PNG")
        self.test = LlmTestRequest(target="llm10", prompt="Describe the image.", images=[{"mime": "image/png", "b64": base64.b64encode(image.getvalue()).decode()}])
        events = await self.collect()
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(any(part.get("type") == "image_url" for message in self.requests[0][0]["messages"] for part in message["content"] if isinstance(part, dict)))

    async def test_http_checks_saved_revision_before_calling_and_streams_result(self):
        dist = self.root / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
        client = TestClient(TestServer(create_app(dist, self.root, llm_tests=self.service)))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        headers = {"Host": "127.0.0.1:8201", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": SETTINGS_UI_SCHEMA, "If-Match": '"stale"'}
        stale = await client.post("/api/setting/llm/test", headers=headers, json=self.test.model_dump())
        self.assertEqual(stale.status, 409)
        self.assertEqual(self.requests, [])
        headers["If-Match"] = self.revision
        invalid = await client.post("/api/setting/llm/test", headers=headers, json={"target": "llm11", "prompt": "Hello"})
        self.assertEqual(invalid.status, 400)
        unconfigured = await client.post("/api/setting/llm/test", headers=headers, json={"target": "llm1", "prompt": "Hello"})
        self.assertEqual(unconfigured.status, 400)
        response = await client.post("/api/setting/llm/test", headers=headers, json=self.test.model_dump())
        self.assertEqual(response.status, 200)
        events = [json.loads(frame.removeprefix("data: ")) for frame in (await response.text()).strip().split("\n\n") if frame.startswith("data:")]
        self.assertEqual(events[-1]["type"], "done")


if __name__ == "__main__":
    unittest.main()
