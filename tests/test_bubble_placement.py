import unittest

from PIL import Image, ImageDraw

from modes.bubble_layout import choose_layout, choose_scaled_layout
from modes.bubble_predictor import select_candidate
from modes.postprocess import normalize_layout_font_scale
from modes.bubble_render import (
    _bubble_is_above_face,
    _draw_layout_bubble,
    _draw_speech,
    _place_body,
    _protected_face_box,
    _resolve_layout_font_scale,
    _tail_side,
)


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class BubblePlacementTest(unittest.TestCase):
    def test_layout_font_scale_is_user_configurable_and_clamped(self):
        self.assertEqual(normalize_layout_font_scale(1.7), 1.7)
        self.assertEqual(normalize_layout_font_scale(9), 4.0)
        self.assertEqual(normalize_layout_font_scale(0.2), 1.0)
        self.assertEqual(_resolve_layout_font_scale({"layout_font_scale": "3.5"}), 3.5)
        self.assertEqual(_resolve_layout_font_scale({"layout_font_scale": 99}), 4.0)

    def test_face_protection_adds_safety_margin(self):
        protected = _protected_face_box((100, 100, 160, 160), (500, 500))
        self.assertEqual(protected, (92.0, 92.0, 168.0, 168.0))

    def test_layout_onnx_selects_font_wrap_ratio_and_shape(self):
        selected, _ = choose_layout(
            "잠깐… 이게 정말 맞는 선택일까? 조금 더 생각해 보자…",
            (1056, 1536),
        )
        self.assertTrue(selected.fits)
        self.assertEqual(selected.font_size, 29)
        self.assertEqual(selected.shape, "cloud")
        self.assertEqual(len(selected.lines), 3)
        self.assertAlmostEqual(selected.bubble_width / selected.bubble_height, 1.28, places=2)
        self.assertEqual(
            selected.lines,
            ("잠깐…", "이게 정말 맞는 선택일까?", "조금 더 생각해 보자…"),
        )

    def test_scaled_layout_nearly_doubles_font_and_reflows(self):
        selected, _ = choose_scaled_layout(
            "잠깐… 이게 정말 맞는 선택일까? 조금 더 생각해 보자…",
            (1056, 1536),
            font_scale=2.0,
        )
        self.assertTrue(selected.fits)
        self.assertEqual(selected.shape, "cloud")
        self.assertGreaterEqual(selected.font_size, 52)
        self.assertLessEqual(selected.font_size, 64)
        self.assertGreater(len(selected.lines), 3)

    def test_scaled_layout_never_exceeds_user_limit(self):
        text = "잠깐… 이게 정말 맞는 선택일까? 조금 더 생각해 보자…"
        base, _ = choose_layout(text, (1056, 1536))
        selected, _ = choose_scaled_layout(text, (1056, 1536), font_scale=1.5)
        self.assertLessEqual(selected.font_size, int(base.font_size * 1.5))

    def test_candidate_uses_distance_before_confidence(self):
        face = (100, 100, 160, 160)
        candidates = [
            {"center": (350, 350), "anchor": (160, 160), "confidence": 0.9},
            {"center": (130, 75), "anchor": (130, 100), "confidence": 0.1},
        ]
        chosen = select_candidate(
            candidates,
            body_size=(60, 40),
            face_box=face,
            canvas_size=(500, 500),
            forbidden_boxes=[face],
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["center"], (130.0, 75.0))
        self.assertFalse(_overlaps(chosen["rect"], face))

    def test_candidate_covering_face_is_rejected(self):
        face = (100, 100, 160, 160)
        candidates = [
            {"center": (130, 130), "anchor": (130, 100), "confidence": 0.9},
            {"center": (130, 70), "anchor": (130, 100), "confidence": 0.1},
        ]
        chosen = select_candidate(
            candidates,
            body_size=(60, 40),
            face_box=face,
            canvas_size=(500, 500),
            forbidden_boxes=[face],
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["center"], (130.0, 70.0))

    def test_safe_fallback_never_covers_face(self):
        face = (200, 200, 260, 260)
        placed = _place_body(face, 100, 50, [face], 500, 500, 30)
        self.assertIsNotNone(placed)
        rect, _, _ = placed
        self.assertFalse(_overlaps(rect, face))

    def test_tail_only_when_bubble_center_is_above_face_center(self):
        face = (100, 100, 160, 160)
        self.assertTrue(_bubble_is_above_face((80, 20, 180, 80), face))
        self.assertFalse(_bubble_is_above_face((80, 110, 180, 150), face))
        self.assertFalse(_bubble_is_above_face((80, 130, 180, 170), face))

    def test_speech_renderer_obeys_tail_flag(self):
        without_tail = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        _draw_speech(
            ImageDraw.Draw(without_tail), (20, 20, 80, 50), (50, 80), "top",
            (255, 255, 255, 255), (0, 0, 0, 255), 2, with_tail=False,
        )
        self.assertEqual(without_tail.getpixel((50, 65)), (0, 0, 0, 0))

        with_tail = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        _draw_speech(
            ImageDraw.Draw(with_tail), (20, 20, 80, 50), (50, 80), "top",
            (255, 255, 255, 255), (0, 0, 0, 255), 2, with_tail=True,
        )
        self.assertNotEqual(with_tail.getpixel((50, 65)), (0, 0, 0, 0))

    def test_tail_starts_from_edge_nearest_face(self):
        rect = (20, 20, 80, 50)
        self.assertEqual(_tail_side(rect, (50, 90)), "top")
        self.assertEqual(_tail_side(rect, (95, 35)), "left")

    def test_cloud_tail_is_drawn_only_when_requested(self):
        rect = (20, 20, 80, 60)
        anchor = (50, 95)
        without_tail = Image.new("RGBA", (110, 110), (0, 0, 0, 0))
        _draw_layout_bubble(
            without_tail,
            rect,
            anchor,
            "cloud",
            (255, 255, 255, 255),
            (0, 0, 0, 255),
            2,
            12,
            False,
        )
        self.assertEqual(without_tail.getpixel((50, 84)), (0, 0, 0, 0))

        with_tail = Image.new("RGBA", (110, 110), (0, 0, 0, 0))
        _draw_layout_bubble(
            with_tail,
            rect,
            anchor,
            "cloud",
            (255, 255, 255, 255),
            (0, 0, 0, 255),
            2,
            12,
            True,
        )
        self.assertNotEqual(with_tail.getpixel((50, 84)), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
