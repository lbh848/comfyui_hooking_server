"""Large style galleries read one page; random filtering stays cheap and explicit."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from modes import bot_lora_mode, image_filter_mode as filtering, style_lora_mode as style
from image_name_search import matches_image_name


class StyleImagePagesTests(unittest.TestCase):
    def setUp(self):
        self.files = [f"image-{index:05}.png" for index in range(9137)]
        self.data = {"projects": {"large": {"images": self.files, "test_images": self.files[:50]}}}
        self.patcher = patch.object(style, "_load_data", return_value=self.data)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_filename_index_never_opens_captions(self):
        with patch.object(style, "get_image_prompt", side_effect=AssertionError("caption read")), \
             patch.object(style.os.path, "isfile", side_effect=AssertionError("caption stat")):
            result = style.get_project_detail("large", include_prompts=False)
        self.assertEqual(result["data"]["images"], self.files)
        self.assertEqual(result["data"]["prompts"], {})

    def test_pages_bound_caption_reads_and_preserve_missing_prompts(self):
        with patch.object(style.os.path, "isfile", return_value=True), \
             patch.object(style, "get_image_prompt", side_effect=lambda project, filename: {"success": True, "data": {"positive": filename}}) as reader:
            result = style.get_image_page("large", 24, 2000)
        self.assertEqual(reader.call_count, 24)
        self.assertEqual([row["filename"] for row in result["images"]], self.files[24:48])
        self.assertEqual(result["total"], 9137)
        self.assertTrue(result["has_more"])
        with patch.object(style.os.path, "isfile", return_value=False):
            last = style.get_image_page("large", 9120)
        self.assertEqual(last["images"], [{"filename": name} for name in self.files[9120:]])
        self.assertFalse(last["has_more"])

    def test_test_page_reads_only_test_captions_and_surfaces_read_errors(self):
        with patch.object(style.os.path, "isfile", return_value=True), \
             patch.object(style, "get_test_image_prompt", return_value={"success": True, "data": {"positive": "test-caption"}}) as reader:
            result = style.get_image_page("large", 48, test=True)
        self.assertEqual(reader.call_count, 2)
        self.assertEqual(result["images"][0]["positive"], "test-caption")
        with patch.object(style.os.path, "isfile", return_value=True), \
             patch.object(style, "get_image_prompt", return_value={"success": False, "error": "unreadable"}):
            self.assertEqual(style.get_image_page("large")["error"], "unreadable")

    def test_search_includes_unloaded_files_without_reading_other_captions(self):
        with patch.object(style.os.path, "isfile", return_value=True), \
             patch.object(style, "get_image_prompt", return_value={"success": True, "data": {}}) as reader:
            result = style.get_image_page("large", search="IMAGE 091", limit=2000)
        expected = [name for name in self.files if "091" in name]
        self.assertEqual(result["total"], len(expected))
        self.assertEqual([row["filename"] for row in result["images"]], expected[:24])
        self.assertEqual([call.args[1] for call in reader.call_args_list], expected[:24])
        self.assertTrue(matches_image_name("캐릭터Ａ_blue_hair.PNG", "캐릭터a BLUE hair"))
        self.assertFalse(matches_image_name("blue_hair.png", "blue red"))
        with patch.object(style, "get_image_prompt", side_effect=AssertionError("empty search read captions")):
            result = style.get_image_page("large", search="no-such-image")
        self.assertEqual(result["images"], [])
        self.assertFalse(result["has_more"])

    def test_bot_picker_index_skips_image_decoding_and_caption_reads(self):
        with patch.object(bot_lora_mode.os.path, "isdir", return_value=True), \
             patch.object(bot_lora_mode.os.path, "isfile", return_value=True), \
             patch.object(bot_lora_mode.os, "listdir", return_value=[*self.files, "ignore.txt", "_face_image.webp"]), \
             patch.object(bot_lora_mode, "_load_image_with_prompt", side_effect=AssertionError("unnecessary image read")):
            rows = bot_lora_mode.list_bot_char_available_images("bot", "character", include_details=False)
        self.assertEqual(len(rows), 9138)
        self.assertEqual(rows[-1]["source"], "face")


class FilterJobTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        filtering._job.clear()
        filtering._cancel.clear()
        self.files = [f"image-{index:05}.png" for index in range(9137)]
        self.data = {"projects": {"large": {"images": self.files}}}
        self.patcher = patch.object(style, "_load_data", return_value=self.data)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(filtering._job.clear)

    async def select(self):
        with patch.object(style, "get_image_prompt", side_effect=AssertionError("random read captions")), \
             patch.object(style.os.path, "isfile", side_effect=AssertionError("random stat captions")), \
             patch.object(filtering, "embed_images", side_effect=AssertionError("random ran inference")):
            result = filtering.start_filter_job("large", "random", 2000)
            self.assertEqual(result["status"], "running")
            self.assertFalse(filtering.start_filter_job("large", "random", 2000)["success"])
            await asyncio.sleep(0)
        state = filtering.get_job_status("large", result["job_id"])
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(set(state["selected"])), 2000)
        self.assertEqual(len(state["to_delete"]), 7137)
        self.assertFalse(set(state["selected"]) & set(state["to_delete"]))
        self.assertEqual(set(state["selected"]) | set(state["to_delete"]), set(self.files))
        return result["job_id"]

    async def test_random_selection_does_not_apply_or_leak_between_projects(self):
        with patch.object(style, "delete_images_bulk", side_effect=AssertionError("selection deleted images")):
            job_id = await self.select()
        self.assertEqual(filtering.get_job_status("other")["status"], "idle")
        self.assertEqual(filtering.get_job_status("large", "old-job")["status"], "idle")
        self.assertFalse((await filtering.apply_filter_job("other", job_id))["success"])
        self.assertEqual(self.data["projects"]["large"]["images"], self.files)

    async def test_apply_is_nonblocking_and_cannot_run_twice_or_start_another_job(self):
        job_id = await self.select()
        entered, release = threading.Event(), threading.Event()
        def delete(project, names):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("event loop was blocked")
            return {"success": True, "deleted": len(names), "failed": []}
        with patch.object(style, "delete_images_bulk", side_effect=delete) as deleter:
            task = asyncio.create_task(filtering.apply_filter_job("large", job_id))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                self.assertEqual(filtering.get_job_status("large")["status"], "applying")
                self.assertFalse((await filtering.apply_filter_job("large", job_id))["success"])
                self.assertFalse(filtering.start_filter_job("large", "random", 20)["success"])
            finally:
                release.set()
            self.assertEqual((await task)["status"], "applied")
            self.assertEqual(deleter.call_count, 1)
            self.assertEqual(len(deleter.call_args.args[1]), 7137)
        self.assertFalse((await filtering.apply_filter_job("large", job_id))["success"])

    async def test_changed_image_pool_rejects_stale_selection(self):
        job_id = await self.select()
        self.data["projects"]["large"]["images"] = self.files[1:]
        with patch.object(style, "delete_images_bulk", side_effect=AssertionError("stale result applied")):
            self.assertFalse((await filtering.apply_filter_job("large", job_id))["success"])

    async def test_apply_updates_the_real_project_and_leaves_source_files_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "images" / "small"
            project.mkdir(parents=True)
            names = [f"image-{index}.png" for index in range(8)]
            for name in names:
                (project / name).write_bytes(b"copied image")
            original = root / "original.png"
            original.write_bytes(b"original")
            metadata = root / "manage.json"
            metadata.write_text(json.dumps({"projects": {"small": {"images": names}}}), encoding="utf-8")
            self.patcher.stop()
            with patch.object(style, "STYLE_LORA_DIR", str(root / "images")), \
                 patch.object(style, "STYLE_LORA_MANAGE_FILE", str(metadata)), \
                 patch.object(style, "BACKUP_DIR", str(root / "backups")):
                result = filtering.start_filter_job("small", "random", 2)
                await asyncio.sleep(0)
                applied = await filtering.apply_filter_job("small", result["job_id"])
                self.assertTrue(applied["success"])
                self.assertEqual(set(style.list_images("small")), set(applied["selected"]))
                self.assertEqual(len(list(project.glob("*.png"))), 2)
                self.assertEqual(original.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
