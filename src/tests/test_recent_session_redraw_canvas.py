"""Recent-session redraw keeps the MOSS model's own canvas size and one flow."""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import traceback
import unittest
from unittest.mock import AsyncMock
import uuid

from aiohttp import web

import illustration_flow

SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def server_function(name, namespace):
    entry = copy.deepcopy(next(node for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name))
    entry.decorator_list = []
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def moss_document(width=None, height=None, workflow_size=None, items=None):
    layout = {
        "summary": {"caption": "1girl, garden"},
        "items": items if items is not None else [{
            "label": "character", "name": "Alice", "caption": "1girl",
            "bbox": [-100, -100, 100, 100],
        }],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    if width is not None and height is not None:
        layout["image_size"] = {"width": width, "height": height}
    workflow = {"model": "moss", "seed": 1}
    if workflow_size is not None:
        workflow["width"], workflow["height"] = workflow_size
    return {"moss": layout, "workflow": workflow}


class RecentSessionRedrawCanvasTests(unittest.TestCase):
    def extract(self, document, info=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name, "backup_submitted_prompt.json")
        path.write_text(json.dumps(document), encoding="utf-8")
        return server_function("_recent_session_generated_prompt", {
            "os": os, "json": json, "copy": copy, "traceback": traceback,
            "_read_backup_info": lambda name: info or {},
            "find_backup_file": lambda name, suffix="", **_kwargs: (
                str(path) if suffix == "_submitted_prompt.json" else ""),
        })("backup")

    def test_layout_image_size_becomes_redraw_canvas(self):
        generated = self.extract(moss_document(width=1152, height=1728,
                                               workflow_size=(1024, 752)))
        self.assertEqual(generated["canvas"], [1152, 1728])
        self.assertEqual(generated["moss_layout"]["image_size"],
                         {"width": 1152, "height": 1728})
        self.assertEqual(generated["family"], "moss")

    def test_workflow_size_is_fallback_when_layout_omits_image_size(self):
        generated = self.extract(moss_document(workflow_size=(1056, 1888)))
        self.assertEqual(generated["canvas"], [1056, 1888])

    def test_invalid_sizes_do_not_override_current_canvas(self):
        generated = self.extract(moss_document(width=0, height=0,
                                               workflow_size=(0, -1)))
        self.assertIsNone(generated["canvas"])

    def test_anima_backup_has_no_canvas_override(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name, "backup.json")
        path.write_text(json.dumps({"fields": {"SETUP": "1girl"}}), encoding="utf-8")
        generated = server_function("_recent_session_generated_prompt", {
            "os": os, "json": json, "copy": copy, "traceback": traceback,
            "_read_backup_info": lambda name: {
                "llm_final_result": {"raw_positive": "[SETUP]\n1girl\n[CHAR]\n[SUPPLEMENT]"},
            },
            "find_backup_file": lambda name, suffix="", **_kwargs: (
                str(path) if suffix == "_submitted_prompt.json" else ""),
        })("backup")
        self.assertEqual(generated["family"], "anima")
        self.assertIsNone(generated["canvas"])


class RecentSessionRedrawGroupingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._reset_flows()
        self.addCleanup(self._reset_flows)

    @staticmethod
    def _reset_flows():
        illustration_flow._histories = {
            workshop: {} for workshop in illustration_flow.WORKSHOPS}
        illustration_flow._history.clear()
        illustration_flow._latest = None

    async def test_all_redrawn_images_share_one_flow_session(self):
        source = SimpleNamespace(
            id="source-item", type="illustration", label="원본",
            params={"illustration_flow_auto_open": False})
        illustration_flow.queue_added(source)
        source_run = source._illustration_flow[0]
        source_run["backup_names"] = ["a", "b", "c"]

        class Queue:
            def __init__(self):
                self.items = []
                self.runs = []

            async def add_item(self, item_type, label, params, priority=0, **_kwargs):
                item = SimpleNamespace(
                    id=f"queued-{len(self.items)}", type=item_type, label=label,
                    params=params, status="pending", completed_at=None)
                illustration_flow.queue_added(item)
                self.items.append(item)
                self.runs.append(item._illustration_flow[0])
                return item

        queue = Queue()
        handler = server_function("handle_api_redraw_recent_sessions", {
            "web": web, "uuid": uuid, "time": time, "traceback": traceback,
            "illustration_flow": illustration_flow, "queue_manager": queue,
            "prompts": {},
            "_capture_illustration_runtime_snapshot": lambda **_kwargs: {
                "bot_name": "test-bot",
                "illustration_context_toggles": {"illustration_model_family": "moss"},
            },
            "_recent_session_generated_prompt": lambda name: {
                "family": "moss", "positive": "[SETUP]\n1girl", "negative": "",
                "multi_char": None, "moss_layout": {"items": []},
                "canvas": [1152, 1728], "visual_states": None,
            },
        })
        response = await handler(SimpleNamespace(
            json=AsyncMock(return_value={"run_id": source_run["id"]})))

        self.assertEqual(response.status, 200)
        self.assertEqual(len(queue.items), 3)
        self.assertTrue(all(run is queue.runs[0] for run in queue.runs))
        self.assertIsNot(queue.runs[0], source_run)
        self.assertEqual(
            len({id(run) for run in illustration_flow.history("illustration")}), 2)


if __name__ == "__main__":
    unittest.main()
