"""DETAIL-COMPOSITE scene normalization and background reuse identity."""
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw


class IllustrationCompositeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modes = Path(__file__).resolve().parents[1] / "modes"
        package = ModuleType("modes")
        package.__path__ = [str(modes)]
        spec = importlib.util.spec_from_file_location(
            "illustration_composite", modes / "illustration_composite.py")
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modes": package}):
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.module.clear_background_cache()

    def test_missing_canvas_and_characters_are_repaired(self):
        descriptor = self.module.parse_scene(
            {"slot": "4", "background": {"prompt": "scenery, no humans, room"}},
            {"slot": 4, "characters": []},
        )
        self.assertEqual(descriptor["slot"], 4)
        self.assertEqual(descriptor["canvas"], list(self.module.DEFAULT_CANVAS))
        self.assertEqual(descriptor["scene"], "scenery, no humans, room")
        self.assertEqual(descriptor["characters"], [])
        self.assertEqual(len(descriptor["composite_layers"]), 1)

    def test_scalar_background_and_quoted_name_are_normalized(self):
        descriptor = self.module.parse_scene(
            {
                "slot": 3,
                "canvas": [1152.0, 768],
                "background": "scenery, no humans, station",
                "characters": [{
                    "name": '"Alice"',
                    "positive": ["standing", "cowboy shot", "solo", "1girl",
                                 "transparent background"],
                }],
            },
            {"slot": 3, "characters": [{"name": "Alice"}]},
        )
        self.assertEqual(descriptor["canvas"], [1152, 768])
        self.assertEqual(descriptor["characters"][0]["name"], "Alice")
        self.assertEqual(
            descriptor["characters"][0]["positive"],
            "standing, cowboy shot, solo, 1girl, transparent background",
        )
        self.assertEqual(descriptor["composite_layers"][1]["x"], 0.5)

    def test_unusable_character_entries_are_skipped(self):
        descriptor = self.module.parse_scene(
            {
                "slot": 2,
                "background": "scenery, no humans, room",
                "characters": [
                    {"name": "Alice", "positive": "standing, solo, 1girl"},
                    {"name": "Alice", "positive": "standing, solo, 1girl"},
                    {"name": "Bob"},
                    {"positive": "standing, solo, 1boy"},
                    "broken",
                ],
            },
            {"slot": 2, "characters": [{"name": "Alice"}]},
        )
        self.assertEqual([item["name"] for item in descriptor["characters"]], ["Alice"])

    def test_missing_assigned_character_still_reports_the_name(self):
        with self.assertRaisesRegex(ValueError, "Alice"):
            self.module.parse_scene(
                {"slot": 5, "background": "scenery, no humans, room", "characters": []},
                {"slot": 5, "characters": [{"name": "Alice"}]},
            )

    def test_slot_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "slot"):
            self.module.parse_scene(
                {"slot": 6, "background": "scenery, no humans, room"},
                {"slot": 7, "characters": []},
            )

    def test_background_identity_ignores_formatting_but_not_canvas_or_bot(self):
        first = {"raw_positive": "scenery, no humans,  Room  ",
                 "raw_negative": "", "canvas": [1152, 768]}
        second = {"raw_positive": "SCENERY, NO HUMANS, ROOM",
                  "raw_negative": "", "canvas": [1152.0, 768]}
        key = self.module.background_key(first, bot_name="Bot", workflow_type="w")
        self.assertEqual(
            key,
            self.module.background_key(second, bot_name="Bot", workflow_type="w"),
        )
        self.assertNotEqual(
            key,
            self.module.background_key(first, canvas=[1024, 768],
                                       bot_name="Bot", workflow_type="w"),
        )
        self.assertNotEqual(
            key,
            self.module.background_key(first, bot_name="Other", workflow_type="w"),
        )

    def test_declared_backgrounds_normalize_list_and_mapping(self):
        expected = {
            "B1": "scenery, no humans, medieval fantasy workshop",
            "B2": "scenery, no humans, forest clearing",
        }
        listed = self.module.declared_backgrounds([
            {"id": "B1", "positive": "scenery, no humans, medieval fantasy workshop"},
            {"name": "B2", "prompt": "scenery, no humans, forest clearing"},
            {"id": "", "positive": "ignored"},
            "broken",
        ])
        self.assertEqual(listed, expected)
        mapped = self.module.declared_backgrounds({
            "B1": "scenery, no humans, medieval fantasy workshop",
            "B2": {"positive": "scenery, no humans, forest clearing"},
        })
        self.assertEqual(mapped, expected)

    def test_scene_background_reference_resolves_to_declared_text(self):
        backgrounds = {"B1": "scenery, no humans, medieval fantasy workshop, morning"}
        for reference in ("B1", "b1", {"id": "B1"}):
            descriptor = self.module.parse_scene(
                {"slot": 2, "background": reference, "characters": []},
                {"slot": 2, "characters": []},
                backgrounds=backgrounds,
            )
            self.assertEqual(descriptor["scene"], backgrounds["B1"])
            self.assertEqual(descriptor["composite_layers"][0]["scene"], backgrounds["B1"])

    def test_unknown_background_reference_remains_literal_text(self):
        descriptor = self.module.parse_scene(
            {"slot": 2, "background": "scenery, no humans, literal text", "characters": []},
            {"slot": 2, "characters": []},
            backgrounds={"B1": "scenery, no humans, medieval fantasy workshop"},
        )
        self.assertEqual(descriptor["scene"], "scenery, no humans, literal text")

    def test_unresolved_background_id_is_rejected_instead_of_rendered(self):
        # A split KEYVIS document omits the root backgrounds; without merging
        # them the id itself ("B2") would become the image prompt.
        for background in ("B2", {"id": "B2"}, {"ref": "b2"}):
            with self.assertRaisesRegex(ValueError, "[Bb]2"):
                self.module.parse_scene(
                    {"slot": 2, "background": background, "characters": []},
                    {"slot": 2, "characters": []},
                    backgrounds={"B1": "scenery, no humans, medieval fantasy workshop"},
                )

    def test_background_identity_ignores_per_slot_raw_prompt(self):
        # The raw prompt embeds [SLOT] and [CHAT], which differ per scene even
        # for the same location. Identity must follow the rendered background.
        scene = "scenery, no humans, medieval fantasy workshop, morning"
        first = {"scene": scene,
                 "raw_positive": "[SLOT]\n2\n[CHAT]\nYuki entered the workshop."}
        second = {"scene": scene,
                  "raw_positive": "[SLOT]\n11\n[CHAT]\nDoyun waited by the kiln."}
        self.assertEqual(
            self.module.background_key(first, bot_name="Bot", workflow_type="w"),
            self.module.background_key(second, bot_name="Bot", workflow_type="w"),
        )
        self.assertNotEqual(
            self.module.background_key(
                {**first, "scene": "scenery, no humans, forest clearing, night"},
                bot_name="Bot", workflow_type="w"),
            self.module.background_key(first, bot_name="Bot", workflow_type="w"),
        )

    def test_background_cache_is_bounded(self):
        limit = self.module.BACKGROUND_CACHE_LIMIT
        for index in range(limit + 2):
            self.module.cache_background(f"key-{index}", f"image-{index}".encode())
        self.assertIsNone(self.module.cached_background("key-0"))
        self.assertIsNone(self.module.cached_background("key-1"))
        self.assertEqual(self.module.cached_background("key-2"), b"image-2")
        self.assertEqual(self.module.cached_background(f"key-{limit + 1}"),
                         f"image-{limit + 1}".encode())

    @staticmethod
    def _png(image):
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def test_opaque_character_render_reports_missing_transparency(self):
        opaque = Image.new("RGBA", (16, 16), (40, 60, 80, 255))
        with self.assertRaises(self.module.CharacterTransparencyError):
            self.module.character_rgba(self._png(opaque))

    def test_transparent_character_cutout_is_kept_verbatim(self):
        cutout = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        cutout.putpixel((8, 8), (200, 120, 80, 255))
        data = self._png(cutout)
        self.assertEqual(self.module.character_rgba(data), data)

    def test_near_opaque_character_with_painted_background_is_detected(self):
        # Seen in a real run: the painted background has slightly translucent
        # edge pixels, so min-alpha never reaches 255 even though the frame is full.
        painted = Image.new("RGBA", (32, 32), (30, 40, 60, 200))
        with self.assertRaises(self.module.CharacterTransparencyError):
            self.module.character_rgba(self._png(painted))

    def test_large_but_legitimate_cutout_is_accepted(self):
        cutout = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        ImageDraw.Draw(cutout).rectangle((0, 0, 99, 74), fill=(200, 100, 50, 255))
        data = self._png(cutout)
        self.assertEqual(self.module.character_rgba(data), data)

    def test_fully_transparent_character_is_detected_as_empty_render(self):
        blank = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        with self.assertRaisesRegex(self.module.CharacterTransparencyError, "거의 투명"):
            self.module.character_rgba(self._png(blank))

    def test_faint_haze_at_or_below_alpha_floor_is_not_a_background(self):
        # Only alpha above the floor counts as filled. A faint veil behind a
        # normally opaque character is anti-aliasing/haze, not a backdrop.
        cutout = Image.new("RGBA", (100, 100), (90, 90, 90, 5))
        ImageDraw.Draw(cutout).rectangle((0, 0, 59, 99), fill=(200, 100, 50, 255))
        data = self._png(cutout)
        self.assertEqual(self.module.character_rgba(data), data)

    def test_alpha_floor_boundary_counts_pixels_above_ten(self):
        at_floor = Image.new("RGBA", (16, 16), (40, 60, 80, 10))
        with self.assertRaisesRegex(self.module.CharacterTransparencyError, "거의 투명"):
            self.module.character_rgba(self._png(at_floor))
        above_floor = Image.new("RGBA", (16, 16), (40, 60, 80, 11))
        with self.assertRaisesRegex(self.module.CharacterTransparencyError, "배경"):
            self.module.character_rgba(self._png(above_floor))


if __name__ == "__main__":
    unittest.main()
