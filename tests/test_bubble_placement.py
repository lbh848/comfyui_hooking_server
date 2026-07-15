import unittest

from PIL import Image, ImageDraw

from modes.bubble_predictor import select_candidate
from modes.bubble_render import (
    _bubble_is_above_face,
    _draw_speech,
    _draw_thought,
    _place_body,
    _tail_side,
)


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class BubblePlacementTest(unittest.TestCase):
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
            (255, 255, 255, 255), (0, 0, 0, 255), 2, 8, with_tail=False,
        )
        self.assertEqual(without_tail.getpixel((50, 65)), (0, 0, 0, 0))

        with_tail = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        _draw_speech(
            ImageDraw.Draw(with_tail), (20, 20, 80, 50), (50, 80), "top",
            (255, 255, 255, 255), (0, 0, 0, 255), 2, 8, with_tail=True,
        )
        self.assertNotEqual(with_tail.getpixel((50, 65)), (0, 0, 0, 0))

    def test_tail_starts_from_edge_nearest_face(self):
        rect = (20, 20, 80, 50)
        self.assertEqual(_tail_side(rect, (50, 90)), "top")
        self.assertEqual(_tail_side(rect, (95, 35)), "left")

    def test_thought_is_plain_tail_free_rectangle(self):
        image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        _draw_thought(draw, (20, 25, 80, 75), (255, 255, 255, 255), (0, 0, 0, 255), 2)
        self.assertEqual(image.getpixel((20, 25)), (0, 0, 0, 255))
        self.assertEqual(image.getpixel((50, 50)), (255, 255, 255, 255))
        self.assertEqual(image.getpixel((15, 20)), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
