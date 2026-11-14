"""Textless manga discards balloon objects and only locks manga postprocessing."""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import Mock, patch

from illustration_json import build_illustration_json, workflow_json_defaults
from modes.illustration_context_pipeline import merged_toggles
from test_moss_prompt_builder import moss_document, workflow_fixture


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def server_function(name, namespace):
    node = copy.deepcopy(next(node for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class MangaBubblesTests(unittest.TestCase):
    def setUp(self):
        self.document = moss_document()
        self.document["moss"]["items"] += [
            {"label": "object", "caption": "PANEL 1. A room", "bbox": [-512, -512, 512, 512]},
            {"label": "text", "caption": "white speech balloon", "text": "Hello!",
             "bbox": [-200, -200, -100, 100]},
        ]

    def test_default_and_saved_boolean_values(self):
        self.assertFalse(merged_toggles({})["manga_disable_bubbles"])
        for raw, expected in ((True, True), (False, False), ("on", True), ("off", False)):
            self.assertIs(merged_toggles({"manga_disable_bubbles": raw})["manga_disable_bubbles"], expected)

    def test_new_and_saved_documents_only_drop_text_items(self):
        before = copy.deepcopy(self.document)
        defaults = workflow_json_defaults(workflow_fixture())
        for saved in (False, True):
            options = {} if saved else {"moss_layout": self.document["moss"]}
            positive = json.dumps(self.document) if saved else "[ANIMA_CONTENT]\npage\n[END]"
            baseline = json.loads(build_illustration_json(
                positive, "", input_dir=".", workflow_defaults=defaults, **options))
            for disabled in (False, True):
                with self.subTest(saved=saved, disabled=disabled):
                    result = json.loads(build_illustration_json(
                        positive, "", input_dir=".", workflow_defaults=defaults,
                        manga_disable_bubbles=disabled, **options))
                    expected = [item for item in before["moss"]["items"]
                                if not disabled or item["label"] != "text"]
                    self.assertEqual(result["moss"]["items"], expected)
                    expected_document = copy.deepcopy(baseline)
                    expected_document["moss"]["items"] = expected
                    self.assertEqual(result, expected_document)
                    self.assertEqual(self.document, before)

    def test_final_workflow_receives_filtered_layout(self):
        workflow = workflow_fixture()
        raw_body = {"illustration_moss_layout": self.document["moss"],
                    "illustration_manga_disable_bubbles": True}
        build = server_function("build_prompt", {
            "copy": copy, "json": json, "os": os,
            "app_config": {}, "log_to_file": Mock(),
            "CURRENT_COMFY_QUEUE_ITEM": SimpleNamespace(get=lambda: SimpleNamespace(
                params={"raw_body": raw_body})),
        })
        with patch("illustration_flow.record_reference_images"):
            for disabled in (True, False):
                raw_body["illustration_manga_disable_bubbles"] = disabled
                result = build("[ANIMA_CONTENT]\npage\n[END]", "", workflow_api=workflow)
                layout = json.loads(result["1"]["inputs"]["value"])["moss"]
                self.assertEqual(any(item["label"] == "text" for item in layout["items"]), not disabled)
        self.assertEqual(len(self.document["moss"]["items"]), 3)

    def test_postprocess_lock_is_limited_to_manga_and_respects_captured_toggle(self):
        config = {"postprocess_enabled": True,
                  "illustration_context_toggles": {"illustration_output_mode": "manga",
                                                   "manga_disable_bubbles": True},
                  "postprocess": {"subtitle": {"enabled": True}, "manga_text": {"enabled": True}}}
        before = copy.deepcopy(config)
        resolve = server_function("_get_illustration_postprocess_settings", {
            "app_config": config, "traceback": traceback,
        })
        with patch("modes.font_assets.resolve_font", return_value=("", None)):
            for mode in ("manga", "manga_both"):
                for force in (False, True):
                    self.assertIsNone(resolve("Bot", output_mode=mode, force=force))
                self.assertEqual(resolve("Bot", output_mode=mode,
                    manga_disable_bubbles=False)["_mode"], "manga_text")
            for mode in ("illustration", "both", "original_asset"):
                self.assertEqual(resolve("Bot", output_mode=mode)["_mode"], "subtitle")
            self.assertEqual(config, before)
            config["illustration_context_toggles"]["manga_disable_bubbles"] = False
            self.assertIsNone(resolve("Bot", output_mode="manga", manga_disable_bubbles=True))
            config["illustration_context_toggles"].update(
                illustration_output_mode="illustration", manga_disable_bubbles=True)
            self.assertEqual(resolve("Bot")["_mode"], "subtitle")


if __name__ == "__main__":
    unittest.main()
