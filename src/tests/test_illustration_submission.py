"""Exercise the server's real JSON injection and manual-draw queue handoff."""
import ast
import asyncio
import copy
from contextvars import ContextVar
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import uuid

from aiohttp import web


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def server_function(name, namespace):
    entry = copy.deepcopy(next(node for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name))
    entry.decorator_list = []
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class IllustrationJsonSubmissionTests(unittest.TestCase):
    def test_v2_submission_embeds_assets_and_selected_workflow_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache.pt").write_bytes(b"face embedding fixture")
            (root / "refs").mkdir()
            (root / "refs/representation.png").write_bytes(b"reference fixture")
            (root / "job-mask").mkdir()
            (root / "job-mask/mask.png").write_bytes(b"mask fixture")
            defaults = {"ANIMA_FIRST_SAMPLER": {"steps": 23, "cfg": 4.0}}
            workflow = {
                "9": {"class_type": "PrimitiveStringMultiline", "_meta": {"title": "긍정프롬프트"},
                      "inputs": {"value": json.dumps({"schema": "soya.illustration.v2", "fields": defaults})}},
                "901": {"class_type": "SoyaFirstSamplerV2_mdsoya", "inputs": {
                    "prompt_json": ["9", 0], "sampling_key": "ANIMA_FIRST_SAMPLER"}},
            }
            original = copy.deepcopy(workflow)
            item_context = ContextVar("json_test_item", default=SimpleNamespace(_comfy_mask_location="job-mask"))
            build = server_function("build_prompt", {"copy": copy, "CURRENT_COMFY_QUEUE_ITEM": item_context,
                "app_config": {"comfy_input_dir": directory}, "log_to_file": lambda *args: None})
            draft = ('[ANIMA_CONTENT]\n1girl, garden\n[CHAR_LIST]\nAlice\n[IMG_W]\n512\n'
                     '[CACHE_PATH]\n{"list":[{"CHAR":"Alice","emb_path":"cache.pt"}]}\n'
                     '[FACE_ID_DIR]\n{"list":[{"CHAR":"Alice","ipa_path":"refs/cache.ipadpt"}]}\n'
                     '[SNAPREF]\n{"enabled":true}\n[MULTI_CHAR]\n{"enable":true}\n[END]')
            submitted = build(draft, 'anima negative[SDXL]ilxl negative', workflow_api=workflow)
            payload = json.loads(submitted["9"]["inputs"]["value"])
            fields = payload["fields"]
            self.assertEqual(payload["schema"], "soya.illustration.v2")
            self.assertEqual(fields["ANIMA_FIRST_SAMPLER"], defaults["ANIMA_FIRST_SAMPLER"])
            self.assertEqual(fields["NEGATIVE_ANIMA"], "anima negative")
            self.assertEqual(fields["NEGATIVE_SDXL"], "ilxl negative")
            self.assertEqual(fields["CHAR_LIST"], ["Alice"])
            self.assertEqual(fields["IMG_W"], 512)
            self.assertNotIn("CACHE_PATH", fields)
            self.assertNotIn("FACE_ID_DIR", fields)
            for key in (fields["FACE_EMBEDDINGS"][0]["asset"], fields["SNAPREF"]["references"][0]["asset"], fields["REGION_MASK"]):
                self.assertIn(key, payload["assets"])
            self.assertEqual(workflow, original)


class ManualDrawQueueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        Path(self.temp.name, "test_prompt.py").write_text(
            'async def run():\n    return {"positive": "scene", "negative": "bad"}\n', encoding="utf-8")

    def handler(self, queue, bot):
        return server_function("handle_api_restore_manual_draw", {
            "web": web, "asyncio": asyncio, "os": os, "copy": copy, "time": time, "uuid": uuid,
            "importlib": importlib, "traceback": traceback, "prompts": {}, "queue_manager": queue,
            "app_config": {"restore_prompt_file": "test_prompt.py", "bot_selected": bot},
            "CUSTOMPROMPT_DIR": self.temp.name, "RESTORE_WORKFLOW_PROMPT_LLM": "llm.py",
            "workflow_profiles": SimpleNamespace(normalize_restore_prompt_file=lambda value: value,
                normalize_illustration_workflow_type=lambda value: "v3_anima",
                is_restore_prompt_compatible=lambda *args: True, illustration_providers=lambda value: ["comfy"]),
        })

    async def test_success_waits_for_registration_and_returns_item_id(self):
        for bot, expected_type in (("test-bot", "illustration"), ("", "restore_manual")):
            with self.subTest(bot=bot):
                started, release = asyncio.Event(), asyncio.Event()
                async def add(*args, **kwargs):
                    started.set()
                    await release.wait()
                    return SimpleNamespace(id="registered-job")
                queue = SimpleNamespace(add_item=AsyncMock(side_effect=add))
                task = asyncio.create_task(self.handler(queue, bot)(SimpleNamespace(json=AsyncMock(return_value={}))))
                await asyncio.wait_for(started.wait(), 2)
                try:
                    self.assertFalse(task.done())
                finally:
                    release.set()
                response = await asyncio.wait_for(task, 2)
                self.assertEqual(json.loads(response.text), {"success": True, "queued": True, "item_id": "registered-job"})
                self.assertEqual(queue.add_item.await_args.args[0], expected_type)

    async def test_registration_failure_is_not_reported_as_queued(self):
        queue = SimpleNamespace(add_item=AsyncMock(side_effect=RuntimeError("registration failed")))
        response = await self.handler(queue, "test-bot")(SimpleNamespace(json=AsyncMock(return_value={})))
        self.assertEqual(response.status, 500)
        self.assertEqual(json.loads(response.text), {"error": "registration failed"})


if __name__ == "__main__":
    unittest.main()
