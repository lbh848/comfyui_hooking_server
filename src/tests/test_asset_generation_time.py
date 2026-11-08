"""Generation duration survives gallery reloads without borrowing source timings."""
import io
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image
from queue_manager import QueueItem, QueueManager

asset_mode = importlib.import_module("modes.asset_mode")
qwen_edit_mode = importlib.import_module("modes.qwen_edit_mode")


class AssetGenerationTimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        asset_dir = patch.object(asset_mode, "ASSET_DIR", str(self.root))
        asset_dir.start()
        self.addCleanup(asset_dir.stop)
        self.mode = asset_mode.AssetMode()
        self.mode._asset_api_workflow = {}
        self.mode._save_generated_workflow_cache = False
        self.mode.update_asset_workflow = AsyncMock(return_value=True)
        self.mode.notify_frontend_func = AsyncMock()
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buffer, format="PNG")
        self.image_bytes = buffer.getvalue()
        self.mode.submit_workflow_func = AsyncMock(return_value=(self.image_bytes, None))

    async def generate(self):
        return await self.mode.generate(
            character="timing", outfit="outfit", expression="expression",
            positive_prompt="test", negative_prompt="",
        )

    def metadata(self, filename):
        path = self.root / "timing" / "outfit" / "expression" / filename
        return json.loads(path.with_name(path.stem + "_prompt.json").read_text(encoding="utf-8"))

    async def test_each_image_duration_is_persisted_and_returned_after_reload(self):
        with patch.object(asset_mode.time, "perf_counter", side_effect=[100, 183.25, 400, 402]):
            first = await self.generate()
            second = await self.generate()
        for result, duration in [(first, 83.25), (second, 2)]:
            self.assertTrue(result["success"])
            self.assertEqual(result["generation_time"], duration)
            saved = self.metadata(result["filename"])
            self.assertEqual(saved["generation_time"], duration)
            self.assertGreater(saved["created_at"], 0)
        reloaded = asset_mode.AssetMode().list_images("timing", "outfit", "expression")
        self.assertEqual(sorted(image["generation_time"] for image in reloaded["images"]), [2, 83.25])
        completed = [call.args[1] for call in self.mode.notify_frontend_func.call_args_list
                     if call.args[0] == "asset_generation_completed"]
        self.assertEqual([event["generation_time"] for event in completed], [83.25, 2])

    async def test_failed_generation_does_not_create_a_timed_image(self):
        self.mode.submit_workflow_func = AsyncMock(return_value=(None, "test failure"))
        result = await self.generate()
        self.assertFalse(result["success"])
        self.assertNotIn("generation_time", result)
        self.assertEqual(list(self.root.rglob("*_prompt.json")), [])

    async def test_character_version_images_are_saved_separately_from_legacy_session(self):
        from asset_character_maker import fingerprint
        maker_root = self.root / "maker"
        session_id = f"asset-{fingerprint('sample')}"
        with patch.object(asset_mode, "CHARACTER_MAKER_TEMP_DIR", str(maker_root)):
            for session in ("default", session_id):
                result = await self.mode.generate(character="sample", positive_prompt="test", negative_prompt="",
                                                  storage_group="character_maker", storage_session=session)
                self.assertTrue(result["success"], result)
                self.assertEqual(Path(result["local_path"]).parent, maker_root / session / "images")
                self.assertTrue(Path(result["prompt_record_path"]).is_file())
            for invalid in ("../default", "asset-../default", "asset-wrong", fingerprint('sample')):
                result = await self.mode.generate(character="sample", positive_prompt="test", negative_prompt="",
                                                  storage_group="character_maker", storage_session=invalid)
                self.assertFalse(result["success"])

    async def test_sampling_progress_survives_queue_reload_and_stays_on_its_item(self):
        queue = QueueManager()
        queue.get_config = lambda: {}
        queue.asset_mode = self.mode
        queue.notify_frontend = AsyncMock()
        current = QueueItem(id="drawing", type="asset_generation", label="timing",
                            status="processing", params={"body": {
                                "character": "timing", "outfit": "outfit", "expression": "expression",
                                "positive_prompt": "test", "negative_prompt": "",
                            }})
        waiting = QueueItem(id="next", type="asset_generation", label="next")
        queue.items = [current, waiting]

        async def submit(workflow, *, progress_callback, **kwargs):
            # A second sampler can restart from zero; this is current-stage progress.
            for value, maximum, percentage in [(12, 30, 40), (30, 30, 100), (0, 20, 0), (4, 20, 20), (0, 0, 0)]:
                await progress_callback(value, maximum)
                snapshot = queue.get_status()
                self.assertEqual(snapshot["items"][0]["progress"], percentage)
                self.assertEqual(snapshot["items"][0]["progress_detail"]["value"], value)
                self.assertEqual(snapshot["items"][0]["progress_detail"]["max"], maximum)
                self.assertEqual(snapshot["items"][1]["progress"], 0)
                self.assertEqual(snapshot["pending_count"], 1)
            return self.image_bytes, None

        self.mode.submit_workflow_func = submit
        result = await queue._handle_asset_generation(current)
        self.assertTrue(result["success"])
        events = [call.args[1] for call in queue.notify_frontend.call_args_list if call.args[0] == "queue_progress"]
        self.assertEqual(len(events), 5)
        self.assertTrue(all(event["item_id"] == "drawing" for event in events))
        self.assertEqual(events[0]["detail"]["phase"], "sampling")
        legacy_events = [call for call in self.mode.notify_frontend_func.call_args_list if call.args[0] == "asset_generation_progress"]
        self.assertEqual(len(legacy_events), 5)

    async def test_legacy_image_keeps_unknown_duration_despite_timestamps(self):
        folder = self.root / "timing" / "outfit" / "expression"
        folder.mkdir(parents=True)
        (folder / "1700000000_abcdef.png").write_bytes(self.image_bytes)
        (folder / "1700000000_abcdef_prompt.json").write_text(
            json.dumps({"created_at": 1700000000, "video_duration_seconds": 5}), encoding="utf-8",
        )
        image = self.mode.list_images("timing", "outfit", "expression")["images"][0]
        self.assertIsNone(image["generation_time"])

    async def test_edit_records_its_own_duration_and_timestamp(self):
        source = self.root / "source.png"
        source.write_bytes(self.image_bytes)
        params = {
            "source_path": str(source), "source_filename": source.name,
            "source_prompt": {"positive": "original", "generation_time": 999, "created_at": 1},
        }
        editor = qwen_edit_mode.QwenEditMode()
        with patch.object(qwen_edit_mode.time, "perf_counter", return_value=150.5):
            result = editor._save_result(params, self.image_bytes, generation_started=100)
        saved_path = Path(result["local_path"])
        saved = json.loads(saved_path.with_name(saved_path.stem + "_prompt.json").read_text(encoding="utf-8"))
        self.assertEqual(result["generation_time"], 50.5)
        self.assertEqual(saved["generation_time"], 50.5)
        self.assertGreater(saved["created_at"], 1)
        self.assertEqual(saved["positive"], "original")
        self.assertEqual(params["source_prompt"]["generation_time"], 999)
        self.assertEqual(source.read_bytes(), self.image_bytes)
        unmeasured = editor._save_result(params, self.image_bytes)
        self.assertIsNone(unmeasured["generation_time"])
        unknown_path = Path(unmeasured["local_path"])
        unknown = json.loads(unknown_path.with_name(unknown_path.stem + "_prompt.json").read_text(encoding="utf-8"))
        self.assertIsNone(unknown["generation_time"])


if __name__ == "__main__":
    unittest.main()
