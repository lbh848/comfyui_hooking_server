"""Flow checkpoints reuse stored image/prompt data and restore the complete graph."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import illustration_flow as flow


class IllustrationFlowPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.directories = {workshop: Path(directory) / workshop for workshop in flow.WORKSHOPS}
        for name, value in (
            ("_history_dirs", self.directories), ("_histories", {workshop: {} for workshop in flow.WORKSHOPS}),
            ("_history", {}), ("_history_dirty", set()), ("_history_field_cache", {}),
            ("_history_task", None), ("_latest", None), ("_notify", None), ("HISTORY_SAVE_INTERVAL", 0),
        ):
            self.enterContext(patch.object(flow, name, value))

    def run_record(self, workshop="illustration", run_id="a" * 32):
        node = {
            "id": "image", "label": "Image", "kind": "image", "status": "completed",
            "dependencies": [], "attempts": [], "progress": 50,
            "_preview_image": b"image" * 1024, "_preview_source": b"source" * 1024,
            "_preview_mask": b"mask" * 1024,
            "submitted_prompt": {"assets": {"reference": {"data": "x" * 65536}}},
            "input": {"prompt": "synthetic input"}, "output": "synthetic output",
        }
        return {
            "id": run_id, "kind": "illustration", "workshop": workshop, "label": "Flow",
            "status": "completed", "created_at": 1, "updated_at": 2, "revision": 1,
            "nodes": {"image": node}, "backup_names": [],
        }

    async def checkpoint(self, run):
        flow._history[run["id"]] = run
        flow._histories[run["workshop"]][run["id"]] = run
        flow._history_dirty.add(run["id"])
        await flow._persist_history()
        return json.loads((self.directories[run["workshop"]] / f"{run['id']}.json").read_text(encoding="utf-8"))

    async def test_progress_checkpoint_reuses_images_and_prompt_until_their_data_changes(self):
        run = self.run_record()
        run["status"] = run["nodes"]["image"]["status"] = "processing"
        with patch.object(flow, "_write_bytes_atomic", wraps=flow._write_bytes_atomic) as write:
            first = await self.checkpoint(run)
            self.assertTrue(write.called)
            stored = first["nodes"]["image"]["_stored_fields"]
            self.assertIn("_preview_image", stored)
            self.assertIn("submitted_prompt", stored)
            self.assertNotIn("_preview_image", first["nodes"]["image"])
            self.assertNotIn("submitted_prompt", first["nodes"]["image"])

            write.reset_mock()
            run["nodes"]["image"]["progress"] = 75
            run["revision"] += 1
            second = await self.checkpoint(run)
            write.assert_not_called()
            self.assertEqual(second["nodes"]["image"]["progress"], 75)
            self.assertEqual(second["nodes"]["image"]["_stored_fields"], stored)

            run["nodes"]["image"]["_preview_image"] = b"replacement image"
            third = await self.checkpoint(run)
            self.assertEqual(write.call_count, 1)
            self.assertNotEqual(third["nodes"]["image"]["_stored_fields"]["_preview_image"], stored["_preview_image"])
            self.assertEqual(third["nodes"]["image"]["_stored_fields"]["submitted_prompt"], stored["submitted_prompt"])

    async def test_restart_restores_details_and_images_from_new_and_existing_records(self):
        originals = []
        for workshop in flow.WORKSHOPS:
            run = self.run_record(workshop, ("a" if workshop == "illustration" else "b") * 32)
            originals.append(copy.deepcopy(run))
            await self.checkpoint(run)
        legacy = self.run_record(run_id="c" * 32)
        originals.append(copy.deepcopy(legacy))
        for key in flow._HISTORY_IMAGE_FIELDS:
            legacy["nodes"]["image"][key] = base64.b64encode(legacy["nodes"]["image"][key]).decode("ascii")
        (self.directories["illustration"] / f"{legacy['id']}.json").write_text(json.dumps(legacy), encoding="utf-8")

        flow._history.clear()
        flow._history_field_cache.clear()
        for runs in flow._histories.values():
            runs.clear()
        flow.configure(None)

        for original in originals:
            restored = flow.find_run(original["id"])
            self.assertEqual(restored["nodes"], original["nodes"])
            self.assertIn(original["id"], flow._histories[original["workshop"]])
            self.assertEqual(flow.detail(original["id"], "image")["submitted_prompt"], original["nodes"]["image"]["submitted_prompt"])
            self.assertEqual(flow.preview_image(original["id"], "image"), original["nodes"]["image"]["_preview_image"])
            self.assertEqual(flow.preview_mask(original["id"], "image"), original["nodes"]["image"]["_preview_mask"])


if __name__ == "__main__":
    unittest.main()
