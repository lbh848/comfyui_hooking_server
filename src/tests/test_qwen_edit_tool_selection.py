"""The EDIT modal always uses the ANIMA Inpainting workflow."""
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from modes.qwen_edit_mode import QwenEditMode


def png_bytes(size=(32, 32), mask=False):
    image = Image.new("RGB", size, (30, 60, 90))
    if mask:
        image = Image.new("L", size, 0)
        ImageDraw.Draw(image).rectangle((8, 8, 24, 24), fill=255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class StubEditMode(QwenEditMode):
    def __init__(self, source_path, config):
        super().__init__()
        self._stub_source_path = str(source_path)
        self._stub_config = config

    def _resolve_source_path(self, character, outfit, expression, filename):
        return self._stub_source_path

    def _load_source_prompt(self, source_path):
        return {}

    def _require_config(self):
        return dict(self._stub_config)


class QwenEditToolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        self.source.write_bytes(png_bytes())
        self.config = {
            "comfy_input_dir": str(self.root),
        }

    def stage(self, edit_tool):
        return StubEditMode(self.source, self.config).stage_request(
            character="c",
            outfit="o",
            expression="e",
            filename="source.png",
            mask_data=png_bytes(mask=True),
            edit_prompt="edit",
            edit_tool=edit_tool,
        )

    def test_anima_inpainting_is_the_only_tool(self):
        self.assertEqual(self.stage("anima_inpainting")["edit_tool"], "anima_inpainting")
        self.assertEqual(self.stage(None)["edit_tool"], "anima_inpainting")

    def test_removed_and_unknown_tools_are_rejected(self):
        for tool in ("qwen", "unknown"):
            with self.subTest(tool=tool), self.assertRaises(ValueError):
                self.stage(tool)


if __name__ == "__main__":
    unittest.main()
