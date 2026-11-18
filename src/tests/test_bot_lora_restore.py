"""Exercise bot LoRA restoration and deletion using isolated training artifacts."""
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from modes import bot_lora_mode as storage
from modes import bot_lora_restore as restore


class BotLoraRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "src"
        self.source.mkdir()
        self.bot_dir = self.root / "data/bot"
        self.bot_file = self.root / "data/asset_data/bot.json"
        self.manage_file = self.bot_file.with_name("bot_lora_manage.json")
        self.lora_dir = self.root / "data/comfy/models/loras/SOYA_CHAR_LORA/SOYA_BOT_LORA"
        self.project = self.bot_dir / "detached/Lora/portrait"
        self.trained = self.lora_dir / "detached/Lora/portrait"
        self.session = self.trained / "Alice/_visual_profiles/card_1/run"
        self.image = self.project / "Alice/image.png"
        self.model = self.session / "step.safetensors"
        self.put(self.image, b"image fixture")
        self.put(self.model, b"weight fixture")
        self.put(self.session / "step.json", {
            "lora_file": self.model.name, "output_dir": str(self.session),
            "nested": {"input_dir": str(self.image.parent)},
        })
        self.put(self.session / "step.toml", (
            f"output_dir = {json.dumps(str(self.session))}\n"
            f"train_data_dir = '{self.image.parent}'\n"
            'pretrained_model_name_or_path = "models/base.safetensors"\n'
            'learning_rate = 0.0001\n'
            '# output_dir = "SOYA_BOT_LORA/detached/Lora/portrait"\n'
        ).encode())
        self.put(self.project / "Alice/image_prompt.json", {
            "reference_path": str(self.image), "positive": "detached is a synthetic tag",
        })
        self.put(self.bot_file, {"bots": [{"name": "current", "characters": [], "loras": [{
            "lora_path": "SOYA_CHAR_LORA/SOYA_BOT_LORA/detached/Lora/portrait/Alice/_visual_profiles/card_1/run/step.safetensors",
        }]}]})
        self.put(self.manage_file, {"bot_loras": {
            "detached": {"portrait": {
                "training_config": {"lora_save_path": "SOYA_BOT_LORA/detached/Lora/portrait"},
                "characters": {"Alice": {"profiles": {"card_1": {
                    "session_representatives": {"run": json.dumps({"safetensors": "step.safetensors", "preview": "/api/bot_lora/trained/preview/detached/portrait/Alice/run/preview.jpg"})},
                }}}},
            }},
            "current": {"keep": {"characters": {}}},
        }})
        self.keep = self.bot_dir / "current/Lora/keep/retained.png"
        self.put(self.keep, b"retain target data")
        for name, value in {"BASE_DIR": str(self.source), "BOT_DIR": str(self.bot_dir),
                            "BOT_DATA_FILE": str(self.bot_file), "BOT_LORA_MANAGE_FILE": str(self.manage_file)}.items():
            self.enterContext(patch.object(storage, name, value))

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value).encode() if isinstance(value, dict) else value)

    def plan(self):
        return restore.prepare_restore_link("detached", "current", str(self.lora_dir))

    def original_files(self):
        return {path: path.read_bytes() for path in [self.bot_file, self.manage_file, self.keep,
                *self.project.rglob("*"), *self.trained.rglob("*")] if path.is_file()}

    def assert_original_files(self, original):
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        self.assertFalse((self.bot_dir / "current/Lora/portrait").exists())
        self.assertFalse((self.lora_dir / "current/Lora/portrait").exists())

    def test_restore_moves_files_and_updates_json_toml_and_saved_references(self):
        original = self.original_files()
        plan = self.plan()
        self.assert_original_files(original)
        result = restore.restore_link("detached", "current", str(self.lora_dir), plan.revision)
        self.assertTrue(result["success"])
        self.assertFalse(self.project.exists())
        self.assertFalse(self.trained.exists())
        session = self.lora_dir / "current/Lora/portrait/Alice/_visual_profiles/card_1/run"
        self.assertEqual((session / self.model.name).read_bytes(), b"weight fixture")
        image = self.bot_dir / "current/Lora/portrait/Alice/image.png"
        self.assertEqual(image.read_bytes(), b"image fixture")
        self.assertEqual(self.keep.read_bytes(), b"retain target data")
        metadata = json.loads((session / "step.json").read_text())
        self.assertEqual(metadata["output_dir"], str(session))
        self.assertEqual(metadata["nested"]["input_dir"], str(image.parent))
        toml = tomllib.loads((session / "step.toml").read_text())
        self.assertEqual(toml["output_dir"], str(session))
        self.assertEqual(toml["train_data_dir"], str(image.parent))
        self.assertEqual(toml["pretrained_model_name_or_path"], "models/base.safetensors")
        self.assertEqual(toml["learning_rate"], 0.0001)
        prompt = json.loads(image.with_name("image_prompt.json").read_text())
        self.assertEqual(prompt["positive"], "detached is a synthetic tag")
        self.assertEqual(prompt["reference_path"], str(image))
        manage = json.loads(self.manage_file.read_text())["bot_loras"]
        self.assertNotIn("detached", manage)
        self.assertEqual(set(manage["current"]), {"keep", "portrait"})
        self.assertEqual(manage["current"]["portrait"]["training_config"]["lora_save_path"], "SOYA_BOT_LORA/current/Lora/portrait")
        rep = json.loads(manage["current"]["portrait"]["characters"]["Alice"]["profiles"]["card_1"]["session_representatives"]["run"])
        self.assertIn("/preview/current/", rep["preview"])
        selected = json.loads(self.bot_file.read_text())["bots"][0]["loras"][0]["lora_path"]
        self.assertIn("SOYA_BOT_LORA/current/", selected)

    def test_conflicting_project_is_rejected_without_changing_files(self):
        manage = json.loads(self.manage_file.read_text())
        manage["bot_loras"]["current"]["portrait"] = {"characters": {"Existing": {}}}
        self.put(self.manage_file, manage)
        original = self.original_files()
        with self.assertRaises(ValueError):
            self.plan()
        self.assert_original_files(original)

    def test_move_or_save_failure_restores_original_files_and_paths(self):
        for failure in ("move", "save"):
            with self.subTest(failure=failure):
                original = self.original_files()
                plan = self.plan()
                rename, write = Path.rename, restore._write_bytes_atomic
                failed = False

                def failing_rename(path, target):
                    nonlocal failed
                    if path == self.trained and not failed:
                        failed = True
                        raise OSError("simulated move failure")
                    return rename(path, target)

                def failing_write(path, payload):
                    nonlocal failed
                    if path == self.manage_file and not failed:
                        failed = True
                        raise OSError("simulated save failure")
                    return write(path, payload)

                with patch.object(Path, "rename", failing_rename if failure == "move" else rename), \
                     patch.object(restore, "_write_bytes_atomic", failing_write if failure == "save" else write):
                    with self.assertRaises(RuntimeError):
                        restore.restore_link("detached", "current", str(self.lora_dir), plan.revision)
                self.assertTrue(failed)
                self.assert_original_files(original)

    def test_delete_detached_project_removes_training_artifacts_only_for_that_bot(self):
        original_bot = self.bot_file.read_bytes()
        result = storage.remove_project("detached", "portrait", str(self.lora_dir))
        self.assertTrue(result["success"])
        self.assertFalse(self.project.exists())
        self.assertFalse(self.trained.exists())
        self.assertEqual(self.keep.read_bytes(), b"retain target data")
        self.assertEqual(self.bot_file.read_bytes(), original_bot)
        self.assertEqual(json.loads(self.manage_file.read_text())["bot_loras"], {"current": {"keep": {"characters": {}}}})


if __name__ == "__main__":
    unittest.main()
