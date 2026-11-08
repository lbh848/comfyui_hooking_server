"""Original references bypass Comfy input; FACE utility staging is temporary."""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import uuid

from aiohttp import web
from asset_json import prepare_asset_prompt
from illustration_json import build_illustration_json
from modes.bot_mode import build_utility_prompt
from queue_manager import QueueItem, QueueManager
from tests import test_asset_generation_time as timing_tests
from tests.test_asset_references import fixture
from tests.test_illustration_submission import server_function
import workflow_profiles


class AssetOriginalReferenceTests(unittest.IsolatedAsyncioTestCase):
    setUp = timing_tests.AssetGenerationTimeTests.setUp

    def configure(self):
        document, references = fixture(self.root, "face", "grayscale")
        self.mode.anima_only_workflow_source_path = "reference-test.json"
        self.mode.workflow_source_path = "reference-test.json"
        self.mode._asset_api_workflow = {
            "9": {"class_type": "PrimitiveStringMultiline", "_meta": {"title": "긍정프롬프트"},
                  "inputs": {"value": ""}},
            "901": {"class_type": "SoyaFirstSamplerV2_mdsoya", "inputs": {"prompt_json": ["9", 0]}},
        }
        self.mode.build_prompts = Mock(return_value=(json.dumps(document), ""))
        return {**references, "asset_workflow_type": "anima_only", "character": "Alice",
                "outfit": "outfit", "expression": "expression", "positive_prompt": json.dumps(document),
                "negative_prompt": "", "face_id_enabled": True, "style_ref_enabled": True,
                "reference_images": references["snapref_images"], "style_ref_images": references["snapref_images"]}

    def verify_delivery(self):
        workflow = self.mode.submit_workflow_func.await_args.args[0]
        document = json.loads(workflow["9"]["inputs"]["value"])
        refs = document["fields"]["SNAPREF"]["references"]
        self.assertEqual(len(refs), 1)
        self.assertIn("face_asset", refs[0])
        self.assertNotEqual(refs[0]["asset"], refs[0]["face_asset"])
        self.assertNotIn("dir", document["fields"]["SNAPREF"])
        self.assertTrue(base64.b64decode(document["assets"][refs[0]["asset"]]["data"]).startswith(b"\x89PNG"))
        self.assertNotIn("input_paths", self.mode.submit_workflow_func.await_args.kwargs)
        from modal_backend.workflow_assets import resolve_input_files
        with patch.object(Path, "resolve", side_effect=AssertionError("Remote JSON must not access input")):
            self.assertEqual(resolve_input_files(workflow, {"comfy_input_dir": "unavailable/input"}), [])
        for source in self.root.glob("*.png"):
            source.unlink()
        # Serialized delivery remains usable without originals or a Comfy folder.
        delivered = json.loads(build_illustration_json(json.dumps(document), "", input_dir=None,
                                                      include_face_embeddings=False))
        self.assertEqual(delivered["assets"], document["assets"])

    async def test_queued_generation_embeds_originals_without_input_or_ipadapter_copies(self):
        body = self.configure()
        queue = QueueManager()
        queue.get_config = lambda: {"comfy_input_dir": None}
        queue.asset_mode = self.mode
        result = await queue._handle_asset_generation(QueueItem(
            id="direct-ref", type="asset_generation", label="reference", params={"body": body}))
        self.assertTrue(result["success"], result)
        self.verify_delivery()

    async def test_direct_api_embeds_originals_without_input_or_ipadapter_copies(self):
        body = self.configure()
        generate = server_function("handle_api_asset_mode_generate", {
            "web": web, "workflow_profiles": workflow_profiles, "asset_mode": self.mode,
            "app_config": {"comfy_input_dir": None},
        })
        response = await generate(SimpleNamespace(json=AsyncMock(return_value=body)))
        result = json.loads(response.body)
        self.assertTrue(result["success"], result)
        self.verify_delivery()

    def test_saved_snapshot_wins_over_stale_paths(self):
        body = self.configure()
        saved = prepare_asset_prompt(body["positive_prompt"], "", snapref_enabled=True, reference_job=body)
        for source in self.root.glob("*.png"):
            source.unlink()
        delivered = prepare_asset_prompt(saved, "", snapref_enabled=True,
                                         snapref_dir="old/staging", reference_job=body)
        self.assertEqual(json.loads(delivered), json.loads(saved))

    def test_missing_reference_fails_without_creating_input(self):
        body = self.configure()
        body["snapref_images"] = [{"local_path": str(self.root / "missing.png")}]
        with self.assertRaises(ValueError):
            prepare_asset_prompt(body["positive_prompt"], "", snapref_enabled=True, reference_job=body)
        self.assertFalse((self.root / "input").exists())


class FaceUtilityInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.input = self.root / "input"
        self.character = self.root / "bot/Story/Alice"
        self.character.mkdir(parents=True)
        self.source = self.character / "current.png"
        self.source.write_bytes(b"current representative")
        self.target = {"visual_card_id": "card_2", "visual_card_index": 2,
                       "is_primary": False, "rep_images": [self.source.name]}
        self.artifact = self.character / "_visual_profiles/card_2"
        self.workflow = {"9": {"class_type": "PrimitiveStringMultiline",
                              "_meta": {"title": "긍정프롬프트"}, "inputs": {"value": "original"}}}
        bot_module = SimpleNamespace(
            BOT_DIR=str(self.root / "bot"),
            _load_bot_data=lambda: {"bots": [{"name": "Story", "characters": [{"name": "Alice"}]}]},
            _load_patch_settings=lambda name: {"emb_target": "대표만"},
            bot_visual_artifact_dir=lambda *args: str(self.artifact),
            build_utility_prompt=build_utility_prompt,
            resolve_bot_visual_target=lambda *args: self.target,
        )
        modules = patch.dict(sys.modules, {"modes.bot_mode": bot_module})
        modules.start()
        self.addCleanup(modules.stop)
        self.submit = AsyncMock(side_effect=self.receive)
        self.run_utility = server_function("_run_data_patch_utility", {
            "os": os, "Path": Path, "hashlib": hashlib, "json": json, "shutil": shutil,
            "uuid": uuid, "traceback": traceback, "app_config": {"comfy_input_dir": str(self.input)},
            "data_patcher": SimpleNamespace(_load_utility_workflow=AsyncMock(return_value=(self.workflow, None))),
            "submit_workflow_to_comfy": self.submit,
        })
        self.staged = None
        self.outcome = "success"

    async def receive(self, workflow, *, task_key, input_paths):
        self.assertEqual(task_key, "utility_debug")
        self.assertEqual(len(input_paths), 1)
        self.staged = Path(input_paths[0])
        self.assertEqual(self.staged.parent, self.input)
        self.assertEqual((self.staged / "representation.png").read_bytes(), b"current representative")
        self.assertIn(f"[PATH]\n{self.staged.name}\n", workflow["9"]["inputs"]["value"])
        from modal_backend.workflow_assets import resolve_explicit_input_files, resolve_input_files
        remote_files = resolve_explicit_input_files(input_paths, {"comfy_input_dir": str(self.input)})
        self.assertEqual(remote_files[0]["remote_name"], f"{self.staged.name}/representation.png")
        # Real path-based inputs must still be uploaded by the shared resolver.
        image_workflow = {"image": {"class_type": "LoadImage", "inputs": {"image": remote_files[0]["remote_name"]}}}
        self.assertEqual(resolve_input_files(image_workflow, {"comfy_input_dir": str(self.input)}), remote_files)
        if self.outcome == "cancel":
            raise asyncio.CancelledError()
        if self.outcome == "failure":
            return None, "failed"
        self.source.write_bytes(b"changed during execution")
        return b"FACE result", {}

    async def test_success_cleans_staging_and_records_actual_snapshot_for_selected_card(self):
        await self.run_utility("Story", "Alice", "card_2")
        self.assertFalse(self.staged.exists())
        self.assertEqual(list(self.input.iterdir()), [])
        self.assertEqual((self.artifact / "_face_image.webp").read_bytes(), b"FACE result")
        record = json.loads((self.artifact / "_face_image_source.json").read_text(encoding="utf-8"))
        self.assertEqual(record["source_filename"], self.source.name)
        self.assertEqual(record["source_sha256"], hashlib.sha256(b"current representative").hexdigest())
        self.assertEqual(self.workflow["9"]["inputs"]["value"], "original")

    async def test_failure_and_cancellation_clean_staging_and_preserve_previous_face(self):
        self.artifact.mkdir(parents=True)
        face = self.artifact / "_face_image.webp"
        face.write_bytes(b"previous FACE")
        for outcome, error in (("failure", RuntimeError), ("cancel", asyncio.CancelledError)):
            self.outcome = outcome
            with self.subTest(outcome=outcome), self.assertRaises(error):
                await self.run_utility("Story", "Alice", "card_2")
            self.assertFalse(self.staged.exists())
            self.assertEqual(list(self.input.iterdir()), [])
            self.assertEqual(face.read_bytes(), b"previous FACE")


if __name__ == "__main__":
    unittest.main()
