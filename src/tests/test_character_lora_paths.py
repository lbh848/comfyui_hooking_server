"""V6 library selections and legacy character LoRAs use different path roots."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class CharacterLoraPathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        spec = importlib.util.spec_from_file_location("character_lora_builder", modes / "illust_prompt_builder.py")
        module = importlib.util.module_from_spec(spec)
        # Import the real builder without initializing unrelated runtime modes.
        with patch.dict(sys.modules, {"modes": package}):
            spec.loader.exec_module(module)
        cls.builder = module.IllustPromptBuilder

    def test_library_path_is_already_relative_to_lora_root(self):
        for path in ["portraits/model.safetensors", "SOYA_BOT_LORA/A/model.safetensors"]:
            with self.subTest(path=path):
                self.assertEqual(self.builder._resolve_lora_path({"source": "library", "lora_path": path}), path)

    def test_managed_sources_use_category_prefixes(self):
        for source, prefix in [("asset", "SOYA_ASSET_LORA"), ("bot", "SOYA_BOT_LORA"), ("instance", "SOYA_INSTANCE_LORA"), ("style", "SOYA_STYLE_LORA")]:
            with self.subTest(source=source):
                self.assertEqual(self.builder._resolve_lora_path({"source": source, "lora_path": "character/model.safetensors"}), prefix + "\\character/model.safetensors")

    def test_legacy_bot_filename_recovers_training_metadata(self):
        value = {"source": "bot", "lora_path": "model.safetensors", "bot_name": "A", "project_name": "P", "character": "C", "preview_url": "/api/bot_lora/trained/preview/A/P/C/session/image.jpg"}
        self.assertEqual(self.builder._resolve_lora_path(value), "SOYA_BOT_LORA\\A\\Lora\\P\\C\\session\\model.safetensors")


if __name__ == "__main__":
    unittest.main()
