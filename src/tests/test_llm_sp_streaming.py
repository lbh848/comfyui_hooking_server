"""Exercise the local Comfy WebSocket, including count-only executions before generation."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from llm_settings import LLM_SP
from llm_sp_workflow import analyze


class LocalLiveStreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workflow = self.root / "workflow.json"
        self.workflow.write_text(json.dumps({"nodes": [{"id": 1, "type": LLM_SP["node_type"],
                                                       "widgets_values": ["", 32768]}]}), encoding="utf-8")
        self.sockets = {}
        self.prompt_id = None
        self.finished = False
        self.reasoning = "생각 중입니다. " * 12
        self.answer = "완성된 답변"
        self.thinking_received = asyncio.Event()
        self.received = []
        self.skip_count_terminal = False

        async def socket(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            self.sockets[request.query["clientId"]] = ws
            async for _ in ws:
                pass
            return ws

        async def convert(request):
            return web.json_response({"1": {"class_type": LLM_SP["node_type"], "inputs": {}}})

        async def submit(request):
            body = await request.json()
            self.prompt_id = body["prompt_id"]
            ws = self.sockets[body["client_id"]]

            async def emit(execution, sequence, *, phase="starting", thinking="", terminal=False, started=1000):
                await ws.send_json({"type": "llamacpp.generation", "data": {
                    "prompt_id": self.prompt_id, "node_id": "1", "execution_id": execution,
                    "sequence": sequence, "phase": phase, "terminal": terminal, "started_at_ms": started,
                    "thinking": {"text": thinking, "total_bytes": len(thinking.encode("utf-8"))},
                }})

            await emit("count-1", 0)
            if not self.skip_count_terminal:
                await emit("count-1", 2, phase="complete", terminal=True)
            await emit("generation", 0, started=2000)
            await emit("generation", 1, phase="reasoning", thinking=self.reasoning[:20], started=2000)
            # A late count event must not take over the active generation.
            await emit("count-1", 3, phase="complete", terminal=True)
            await emit("generation", 2, phase="reasoning", thinking=self.reasoning, started=2000)
            await emit("generation", 2, phase="reasoning", thinking=self.reasoning, started=2000)
            return web.json_response({"prompt_id": self.prompt_id})

        async def history(request):
            if not self.finished:
                return web.json_response({})
            return web.json_response({self.prompt_id: {
                "status": {"completed": True},
                "outputs": {"1": {"text": [self.answer], "soya_gemma4": [{
                    "reasoning_text": self.reasoning, "usage": {"completion_tokens": 100, "prompt_tokens": 10},
                }]}},
            }})

        app = web.Application()
        app.router.add_get("/ws", socket)
        app.router.add_post("/workflow/convert", convert)
        app.router.add_post("/prompt", submit)
        app.router.add_get("/history/{prompt_id}", history)
        self.server = TestServer(app)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.config = {"comfyui_port": self.server.port, LLM_SP["workflow_binding"]: str(self.workflow)}

    async def check_thinking_before_completion(self):
        metadata = {}

        async def on_event(event):
            self.received.append(event)
            if "".join(item.get("reasoning", "") for item in self.received) == self.reasoning:
                self.thinking_received.set()

        with patch("configuration.read_config", return_value=self.config), patch.dict("os.environ", {"REAL_COMFY_HOST": "127.0.0.1"}):
            task = asyncio.create_task(analyze([{"role": "user", "content": "Test"}],
                                              project_root=self.root, on_event=on_event, metadata=metadata))
            try:
                await asyncio.wait_for(self.thinking_received.wait(), 2)
                self.assertFalse(task.done(), "Thinking must arrive while generation is still running")
                self.assertEqual(metadata["reasoning_text"], self.reasoning)
                self.finished = True
                self.assertEqual(await asyncio.wait_for(task, 3), self.answer)
                self.assertEqual("".join(item.get("reasoning", "") for item in self.received), self.reasoning)
                self.assertEqual(metadata["completion_tokens"], 100)
            finally:
                if not task.done():
                    self.finished = True
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_thinking_streams_after_input_count_execution(self):
        await self.check_thinking_before_completion()

    async def test_newer_generation_recovers_when_count_terminal_was_missed(self):
        self.skip_count_terminal = True
        await self.check_thinking_before_completion()


if __name__ == "__main__":
    unittest.main()
