"""Training, result discovery, and remote transfers share category storage roots."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from comfy_installer.configurator import apply_installed_config
from lora_paths import bot_lora_load_path, lora_artifact_destination
from modes import lora_mode
from modal_backend.lora_inventory import _asset_items, _remote_identity
from modal_backend.service import ModalService


class LoraStoragePathsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.managed = self.root / "models" / "loras" / "SOYA_CHAR_LORA"
        self.config = {
            "lora_load_path": str(self.managed / "SOYA_ASSET_LORA"),
            "bot_lora_load_path": str(self.managed / "SOYA_BOT_LORA"),
            "style_lora_load_path": str(self.managed / "SOYA_STYLE_LORA"),
        }

    def test_install_config_places_all_categories_at_the_same_depth(self):
        config_file = self.root / "fixture.json"
        config_file.write_text("{}", encoding="utf-8")
        comfy = self.root / "comfy"
        comfy.mkdir()
        apply_installed_config(config_path=config_file, requirements_dir=self.root / "backups",
                               comfy_root=comfy, workflow_bindings={}, required_bindings=[])
        result = json.loads(config_file.read_text(encoding="utf-8"))
        base = comfy / "models" / "loras" / "SOYA_CHAR_LORA"
        for key, folder in (("lora_load_path", "SOYA_ASSET_LORA"), ("bot_lora_load_path", "SOYA_BOT_LORA"), ("style_lora_load_path", "SOYA_STYLE_LORA")):
            self.assertEqual(Path(result[key]), base / folder)

    def test_training_output_is_discovered_and_uploaded_from_the_same_folder(self):
        save_path = lora_mode.training_save_path("Character", "Project")
        self.assertEqual(save_path, "SOYA_ASSET_LORA/Character/Lora/Project")
        session = self.managed / save_path / "session"
        session.mkdir(parents=True)
        (session / "step.safetensors").write_bytes(b"fixture")
        (session / "step.png").write_bytes(b"preview")
        (session / "step.json").write_text(json.dumps({"lora_file": "step.safetensors", "previews": ["step.png"]}), encoding="utf-8")
        data = {"loras": {"Character": {"Project": {"trigger": "character"}}}}
        with patch.object(lora_mode, "_load_lora_manage", return_value=data), patch.object(lora_mode, "ASSET_DIR", str(self.root / "assets")):
            sessions = lora_mode.list_trained_sessions(self.config["lora_load_path"], "Character", "Project")
            self.assertEqual([row["name"] for row in sessions], ["session"])
            items = _asset_items(self.config, include_hashes=False, hash_cache=None)
        remote = "SOYA_CHAR_LORA/SOYA_ASSET_LORA/Character/Lora/Project/session/step.safetensors"
        self.assertEqual(items[0]["files"][0]["remote_path"], remote)
        self.assertEqual(Path(items[0]["files"][0]["source_path"]), session / "step.safetensors")
        self.assertEqual(_remote_identity(remote)["key"], items[0]["key"])
        self.assertEqual(_remote_identity(remote)["scopes"], items[0]["scopes"])

    def test_saved_config_cannot_restore_the_asset_level_one_output(self):
        data = {"loras": {"Character": {"Project": {}}}}
        config = {"profile": "anima", "lora_save_path": "Character/Lora/Project"}
        with patch.object(lora_mode, "_load_lora_manage", return_value=data), patch.object(lora_mode, "_save_lora_manage"):
            result = lora_mode.update_lora_entry("Project", "Character", training_config=config)
        self.assertTrue(result["success"])
        self.assertEqual(data["loras"]["Character"]["Project"]["training_config"]["lora_save_path"], "SOYA_ASSET_LORA/Character/Lora/Project")
        self.assertEqual(config["lora_save_path"], "Character/Lora/Project")

    def test_bot_fallback_remains_a_sibling_of_asset_storage(self):
        del self.config["bot_lora_load_path"]
        self.assertEqual(Path(bot_lora_load_path(self.config)), self.managed / "SOYA_BOT_LORA")

    def test_remote_results_round_trip_without_repeating_category_folders(self):
        service = ModalService.__new__(ModalService)
        source = self.root / "download.safetensors"
        source.write_bytes(b"fixture")
        for category in ("SOYA_ASSET_LORA", "SOYA_BOT_LORA", "SOYA_STYLE_LORA"):
            with self.subTest(category=category):
                relative = category + "/Project/session/step.safetensors"
                expected = self.managed / relative
                self.assertEqual(lora_artifact_destination(self.config, relative), expected)
                stored = service._store_modal_artifacts([{"path": str(source), "relative_path": relative}], self.config)
                self.assertEqual(Path(stored[0]["local_path"]), expected)
                self.assertEqual(expected.read_bytes(), b"fixture")
                repeated = service._store_modal_artifacts([{"path": str(source), "relative_path": relative}], self.config)
                self.assertEqual(repeated[0]["status"], "identical")

    def test_explicit_category_roots_are_respected_by_remote_downloads(self):
        custom = self.root / "custom-style"
        self.config["style_lora_load_path"] = str(custom)
        self.assertEqual(lora_artifact_destination(self.config, "SOYA_STYLE_LORA/anima/Project/result.safetensors"), custom / "anima/Project/result.safetensors")


if __name__ == "__main__":
    unittest.main()
