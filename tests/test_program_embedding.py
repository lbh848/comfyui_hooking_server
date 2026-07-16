import hashlib
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from modes import face_embedder

bot_mode = importlib.import_module("modes.bot_mode")


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ProgramEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.bot_dir = os.path.join(self.root, "bot")
        os.makedirs(self.bot_dir, exist_ok=True)

        self.base_patch = patch.object(bot_mode, "BASE_DIR", self.root)
        self.bot_patch = patch.object(bot_mode, "BOT_DIR", self.bot_dir)
        self.embedder_bot_patch = patch.object(face_embedder, "BOT_DIR", self.bot_dir)
        self.base_patch.start()
        self.bot_patch.start()
        self.embedder_bot_patch.start()
        self.addCleanup(self.base_patch.stop)
        self.addCleanup(self.bot_patch.stop)
        self.addCleanup(self.embedder_bot_patch.stop)

        self.data = {"bots": [{"name": "test-bot", "characters": []}]}
        self.load_patch = patch.object(bot_mode, "_load_bot_data", side_effect=lambda: self.data)
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.patcher = bot_mode.BotDataPatcher()

    def _add_character(self, name, rep_color, existing_face_color=None):
        char_dir = os.path.join(self.bot_dir, "test-bot", name)
        os.makedirs(char_dir, exist_ok=True)
        rep_name = f"{name}_rep.webp"
        Image.new("RGB", (96, 128), rep_color).save(
            os.path.join(char_dir, rep_name), "WEBP", lossless=True
        )
        if existing_face_color is not None:
            Image.new("RGB", (64, 64), existing_face_color).save(
                os.path.join(char_dir, "_face_image.webp"), "WEBP", lossless=True
            )
        self.data["bots"][0]["characters"].append({
            "name": name,
            "rep_images": [rep_name],
        })
        return char_dir

    @staticmethod
    def _fake_embedding(image_path):
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), _file_hash(image_path)

    def test_mixed_selection_extracts_only_missing_face_then_commits(self):
        existing_dir = self._add_character("existing", "gray", existing_face_color="blue")
        missing_dir = self._add_character("missing", "red")
        existing_face = os.path.join(existing_dir, "_face_image.webp")
        existing_before = Path(existing_face).read_bytes()
        existing_cache = os.path.join(existing_dir, "_face_image.l14.npz")
        np.savez(existing_cache, emb=np.zeros(4, dtype=np.float32), sha256=np.array("old"))

        with patch("modes.face_detector.crop_face") as crop_face:
            crop_face.return_value = (Image.new("RGB", (512, 512), "green"), 0.88)
            preview = self.patcher._create_program_embedding_preview({
                "bot_name": "test-bot",
                "char_names": ["existing", "missing"],
                "crop_top": 1.4,
                "crop_bottom": 1.1,
                "confidence": 0.35,
                "overwrite": False,
            })

        self.assertEqual([item["status"] for item in preview["items"]], ["existing", "extracted"])
        self.assertEqual(crop_face.call_count, 1)
        self.assertEqual(Path(existing_face).read_bytes(), existing_before)
        self.assertFalse(os.path.exists(os.path.join(missing_dir, "_face_image.webp")))

        session = self.patcher._program_embedding_get_session(preview["preview_id"])
        missing_preview = session["items"][1]["preview_path"]
        preview_bytes = Path(missing_preview).read_bytes()
        with patch.object(face_embedder, "build_embedding_from_path", side_effect=self._fake_embedding):
            committed = self.patcher._commit_program_embedding_preview(preview["preview_id"])

        self.assertTrue(committed["success"])
        self.assertEqual(committed["success_count"], 2)
        self.assertEqual(committed["face_saved_count"], 1)
        self.assertEqual(Path(existing_face).read_bytes(), existing_before)
        self.assertEqual(
            Path(os.path.join(missing_dir, "_face_image.webp")).read_bytes(),
            preview_bytes,
        )
        self.assertTrue(os.path.isfile(os.path.join(missing_dir, "_face_image.l14.npz")))
        self.assertFalse(os.path.isfile(os.path.join(
            committed["backup_dir"], "test-bot", "existing", "_face_image.l14.npz"
        )))
        self.assertFalse(os.path.isdir(session["session_dir"]))

    def test_overwrite_waits_for_commit_and_backs_up_face_and_prompt_only(self):
        char_dir = self._add_character("alice", "gray", existing_face_color="blue")
        face_path = os.path.join(char_dir, "_face_image.webp")
        prompt_path = os.path.join(char_dir, "_face_image_prompt.json")
        cache_path = os.path.join(char_dir, "_face_image.l14.npz")
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": "old face"}, f, ensure_ascii=False)
        np.savez(cache_path, emb=np.zeros(4, dtype=np.float32), sha256=np.array("old"))
        old_face = Path(face_path).read_bytes()

        with patch("modes.face_detector.crop_face") as crop_face:
            crop_face.return_value = (Image.new("RGB", (512, 512), "yellow"), 0.93)
            preview = self.patcher._create_program_embedding_preview({
                "bot_name": "test-bot",
                "char_names": ["alice"],
                "crop_top": 1.0,
                "crop_bottom": 1.0,
                "confidence": 0.3,
                "overwrite": True,
            })

        self.assertEqual(preview["items"][0]["status"], "extracted")
        self.assertEqual(Path(face_path).read_bytes(), old_face)
        self.assertTrue(os.path.isfile(prompt_path))

        with patch.object(face_embedder, "build_embedding_from_path", side_effect=self._fake_embedding):
            committed = self.patcher._commit_program_embedding_preview(preview["preview_id"])

        self.assertNotEqual(Path(face_path).read_bytes(), old_face)
        self.assertFalse(os.path.exists(prompt_path))
        backup_char_dir = os.path.join(committed["backup_dir"], "test-bot", "alice")
        self.assertEqual(Path(os.path.join(backup_char_dir, "_face_image.webp")).read_bytes(), old_face)
        self.assertTrue(os.path.isfile(os.path.join(backup_char_dir, "_face_image_prompt.json")))
        self.assertFalse(os.path.isfile(os.path.join(backup_char_dir, "_face_image.l14.npz")))

    def test_failed_extraction_without_existing_face_does_not_write_on_commit(self):
        char_dir = self._add_character("nobody", "purple")
        with patch("modes.face_detector.crop_face", return_value=(None, 0.12)):
            preview = self.patcher._create_program_embedding_preview({
                "bot_name": "test-bot",
                "char_names": ["nobody"],
                "crop_top": 1.0,
                "crop_bottom": 1.0,
                "confidence": 0.5,
                "overwrite": False,
            })

        self.assertEqual(preview["ready_count"], 0)
        self.assertEqual(preview["items"][0]["status"], "failed")
        committed = self.patcher._commit_program_embedding_preview(preview["preview_id"])
        self.assertFalse(committed["success"])
        self.assertEqual(committed["failed_count"], 1)
        self.assertFalse(os.path.exists(os.path.join(char_dir, "_face_image.webp")))
        self.assertFalse(os.path.exists(os.path.join(char_dir, "_face_image.l14.npz")))

    def test_face_embedder_does_not_fallback_to_representative_image(self):
        self._add_character("rep-only", "orange")
        path, is_face = face_embedder._char_face_image_path("test-bot", "rep-only")
        self.assertIsNone(path)
        self.assertFalse(is_face)


if __name__ == "__main__":
    unittest.main()
