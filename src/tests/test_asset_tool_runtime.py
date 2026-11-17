"""Tag analysis works in the imported settings worker and standalone server."""
import copy
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image


class AssetToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        spec = importlib.util.spec_from_file_location("tested_asset_tool", modes / "asset_tool_mode.py")
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modes": package}):
            spec.loader.exec_module(cls.module)

    async def asyncSetUp(self):
        self.uploads = []
        self.workflows = []
        self.history_tags = ["blue_eyes, long_hair, blue_eyes\nsmile"]
        self.history_pending = False

        async def upload(request):
            data = await request.post()
            self.uploads.append(data["image"].file.read())
            return web.json_response({"name": "uploaded.png"})

        async def websocket(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for _ in ws:
                pass
            return ws

        app = web.Application()
        app.router.add_post("/upload/image", upload)
        app.router.add_get("/ws", websocket)
        self.comfy = TestServer(app)
        await self.comfy.start_server()
        self.addAsyncCleanup(self.comfy.close)

        async def submit(workflow, **kwargs):
            self.workflows.append(copy.deepcopy(workflow))
            return f"prompt-{len(self.workflows)}", {}

        async def history(prompt_id, *, port):
            self.assertEqual(port, self.comfy.port)
            self.assertEqual(prompt_id, f"prompt-{len(self.workflows)}")
            if self.history_pending:
                self.history_pending = False
                return {}
            tags = self.history_tags[min(len(self.workflows) - 1, len(self.history_tags) - 1)]
            outputs = {node_id: {"text": [tags]} for node_id, node in self.workflows[-1].items()
                       if node["class_type"] == "PreviewAny"}
            outputs["unrelated"] = {"text": ["must_not_be_used"]}
            return {prompt_id: {"outputs": outputs}}

        self.runtime = SimpleNamespace(
            REAL_COMFY_HOST="127.0.0.1", PORT=8189,
            resolve_comfy_port=Mock(return_value=self.comfy.port),
            submit_to_real_comfy=AsyncMock(side_effect=submit),
            wait_for_real_comfy=AsyncMock(return_value={"type": "executing"}),
            count_ksampler_total_steps=Mock(return_value=0),
            fetch_real_history=AsyncMock(side_effect=history), aiohttp=aiohttp,
            text_outputs={"WD_TAG_TEXT": {"text": "stale_tags", "timestamp": "old"}},
        )
        self.tool = self.module.AssetToolMode()
        self.tool.update_workflow = AsyncMock(return_value=True)
        self.workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"},
                  "_meta": {"title": "분석이미지로드"}},
            "2": {"class_type": "Tagger", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "SoyaTextSender_mdsoya",
                  "inputs": {"text": ["2", 0], "node_title": "WD_TAG_TEXT"},
                  "_meta": {"title": "WD_TAG_TEXT"}},
        }
        self.tool._api_workflow = copy.deepcopy(self.workflow)
        image = BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        self.image = image.getvalue()

    def modules(self, standalone=False):
        return patch.dict(sys.modules, {
            "__main__": self.runtime if standalone else ModuleType("internal_lora_runtime"),
            "server": None if standalone else self.runtime,
        })

    async def test_imported_worker_reads_tags_without_text_callback(self):
        with self.modules():
            result = await self.tool.analyze_image(self.image, comfy_task_key="bot_lora_training")
        self.assertEqual(result, {"success": True, "tags": ["blue eyes", "long hair", "smile"]})
        self.runtime.resolve_comfy_port.assert_called_once_with("bot_lora_training")
        self.assertEqual(self.uploads, [self.image])
        self.assertEqual(self.workflows[0]["1"]["inputs"]["image"], "uploaded.png")
        self.assertEqual(self.tool._api_workflow, self.workflow)
        self.assertEqual(self.runtime.text_outputs["WD_TAG_TEXT"]["text"], "stale_tags")

    async def test_standalone_server_still_works(self):
        with self.modules(standalone=True):
            result = await self.tool.analyze_image(self.image)
        self.assertTrue(result["success"])
        self.runtime.resolve_comfy_port.assert_called_once_with("tag_analysis")

    async def test_empty_tags_fail_without_retrying_another_workflow(self):
        self.history_tags = ["", "1girl, red_eyes"]
        with self.modules():
            result = await self.tool.analyze_image(self.image)
        self.assertFalse(result["success"])
        self.assertTrue(result["error"])
        self.assertEqual(len(self.workflows), 1)
        self.tool.update_workflow.assert_awaited_once()

    async def test_history_can_arrive_after_websocket_completion(self):
        self.history_pending = True
        with self.modules(), patch.object(self.module.asyncio, "sleep", new_callable=AsyncMock):
            result = await self.tool.analyze_image(self.image)
        self.assertTrue(result["success"])
        self.assertEqual(self.runtime.fetch_real_history.await_count, 2)

    async def test_execution_error_does_not_read_stale_tags(self):
        async def fail(_ws, _prompt, **kwargs):
            kwargs["error_holder"]["detail"] = {
                "node_type": "Tagger", "node_id": "2", "exception_message": "model unavailable",
            }
            return None
        self.runtime.wait_for_real_comfy.side_effect = fail
        with self.modules():
            result = await self.tool.analyze_image(self.image)
        self.assertFalse(result["success"])
        self.assertIn("model unavailable", result["error"])
        self.runtime.fetch_real_history.assert_not_awaited()

    async def test_legacy_text_callback_is_supported_without_history(self):
        del self.runtime.fetch_real_history
        async def finish(*_args, **_kwargs):
            self.runtime.text_outputs["WD_TAG_TEXT"] = {"text": "green_eyes", "timestamp": "new"}
            return {"type": "executing"}
        self.runtime.wait_for_real_comfy.side_effect = finish
        with self.modules():
            result = await self.tool.analyze_image(self.image)
        self.assertEqual(result, {"success": True, "tags": ["green eyes"]})

    async def test_remote_worker_uses_imported_server_input_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.runtime.app_config = {"comfy_input_dir": directory}
            self.tool.run_modal_workflow_func = AsyncMock(return_value={
                "text_outputs": [{"node_title": "WD_TAG_TEXT", "text": "brown_eyes"}],
            })
            context = self.module.CURRENT_COMFY_EXECUTION_TARGET
            token = context.set(self.module.MODAL_COMFY_TARGET)
            try:
                with self.modules():
                    result = await self.tool.analyze_image(self.image)
                input_path = self.tool.run_modal_workflow_func.call_args.kwargs["input_paths"][0]
                self.assertEqual(Path(input_path).parent, Path(directory))
                self.assertTrue(Path(input_path).is_file())
            finally:
                context.reset(token)
        self.assertEqual(result, {"success": True, "tags": ["brown eyes"]})
        self.runtime.submit_to_real_comfy.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
