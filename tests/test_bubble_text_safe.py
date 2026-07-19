"""text-safe organic rounded box 단위 테스트(MVP).

텍스트 박스 → content_safe_box → 둥근 skeleton → 외곽 normal 저주파 offset 파이프라인이
placement rect(hard envelope) 안에서 텍스트를 가두는지 검증한다.
만화풍 말풍선 개선 작업서 v2 참고.
"""

import unittest

import cv2
import numpy as np
from PIL import Image

from modes.bubble_shape import (
    build_text_safe_geometry,
    estimate_envelope_extra,
    make_organic_text_box,
)
from modes.bubble_types import TextBoxShapeConfig
from modes.bubble_render import _draw_layout_bubble


def _envelope_rect(text_w, text_h, pad_x, pad_y, border_w, content_w=None, content_h=None):
    """content(text+padding) 에서 envelope(rect) 치수를 산출한다(파이프라인과 동일 식)."""
    cw = content_w if content_w is not None else text_w + 2 * pad_x
    ch = content_h if content_h is not None else text_h + 2 * pad_y
    guard, bm, maxo = estimate_envelope_extra(min(cw, ch), border_w, TextBoxShapeConfig())
    extra = guard + bm + maxo
    return cw + 2 * extra, ch + 2 * extra


def _mask_outside_box_ratio(mask_arr, box):
    h, w = mask_arr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
    total = int(np.count_nonzero(mask_arr > 0))
    if total <= 0:
        return 1.0
    inside_box = np.zeros_like(mask_arr)
    inside_box[y1:y2, x1:x2] = 255
    return int(np.count_nonzero((mask_arr > 0) & (inside_box == 0))) / total


def _box_outside_mask_ratio(box, mask_arr):
    h, w = mask_arr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area <= 0:
        return 1.0
    return max(0.0, 1.0 - np.count_nonzero(mask_arr[y1:y2, x1:x2] > 0) / area)


class TestOrganicTextBoxReproducibility(unittest.TestCase):
    def setUp(self):
        self.geometry = build_text_safe_geometry(
            (100, 80, 520, 300), 300, 80, 32,
            padding_x=35, padding_y=22, border_w=2,
        )

    def test_same_seed_reproducible(self):
        a = make_organic_text_box(self.geometry, seed=42)
        b = make_organic_text_box(self.geometry, seed=42)
        self.assertTrue(np.array_equal(a, b))

    def test_different_seed_differs(self):
        a = make_organic_text_box(self.geometry, seed=42)
        b = make_organic_text_box(self.geometry, seed=43)
        self.assertFalse(np.array_equal(a, b))

    def test_shape_dtype(self):
        c = make_organic_text_box(self.geometry, seed=1)
        self.assertEqual(c.ndim, 2)
        self.assertEqual(c.shape[1], 2)
        self.assertEqual(c.dtype, np.int32)


class TestEnvelopeContainment(unittest.TestCase):
    """body contour 는 항상 rect(hard envelope) 안에 있어야 한다."""

    def _check(self, text_w, text_h, pad_x, pad_y, font_size, line_count):
        ew, eh = _envelope_rect(text_w, text_h, pad_x, pad_y, 2)
        rect = (60, 50, int(round(60 + ew)), int(round(50 + eh)))
        geom = build_text_safe_geometry(
            rect, text_w, text_h, font_size,
            padding_x=pad_x, padding_y=pad_y, border_w=2,
        )
        contour = make_organic_text_box(geom, seed=7)
        canvas_w, canvas_h = int(round(60 + ew)) + 120, int(round(50 + eh)) + 120
        mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        # text_box 가 text 를 담을 만큼 커야 한다.
        self.assertGreaterEqual(geom.text_box[2] - geom.text_box[0], text_w - 1)
        self.assertGreaterEqual(geom.text_box[3] - geom.text_box[1], text_h - 1)
        # body ⊆ rect, ⊆ outer_box
        self.assertLessEqual(_mask_outside_box_ratio(mask, geom.rect), 0.001)
        self.assertLessEqual(_mask_outside_box_ratio(mask, geom.outer_box), 0.001)
        # content_safe_box / text_box ⊆ mask
        self.assertLessEqual(_box_outside_mask_ratio(geom.content_safe_box, mask), 0.001)
        self.assertLessEqual(_box_outside_mask_ratio(geom.text_box, mask), 0.001)

    def test_normal_two_lines(self):
        self._check(300, 80, 35, 22, 32, 2)

    def test_short(self):
        self._check(80, 40, 14, 10, 24, 2)

    def test_wide(self):
        self._check(520, 60, 30, 18, 30, 1)

    def test_tall_five_lines(self):
        self._check(200, 260, 22, 18, 30, 5)

    def test_large_font(self):
        self._check(600, 300, 40, 28, 40, 4)


class TestTailKeepsContainment(unittest.TestCase):
    """꼬리 OR 합성 뒤에도 content_safe_box containment 가 유지된다."""

    def test_tail_does_not_break_safe_box(self):
        ew, eh = _envelope_rect(300, 80, 35, 22, 2)
        rect = (80, 60, int(round(80 + ew)), int(round(60 + eh)))
        geom = build_text_safe_geometry(
            rect, 300, 80, 32, padding_x=35, padding_y=22, border_w=2,
        )
        contour = make_organic_text_box(geom, seed=9)
        canvas_w, canvas_h = int(round(80 + ew)) + 160, int(round(60 + eh)) + 200
        body = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(body, [contour], 255)

        # 꼬리를 모방: body 밖으로 뻗는 삼각형을 OR 합성.
        tail = np.zeros_like(body)
        tip_x = (rect[0] + rect[2]) // 2
        cv2.fillPoly(
            tail,
            [np.array([[tip_x - 20, rect[3] - 4], [tip_x + 20, rect[3] - 4],
                       [tip_x, rect[3] + 90]], dtype=np.int32)],
            255,
        )
        union = ((body > 0) | (tail > 0)).astype(np.uint8) * 255

        self.assertLessEqual(_box_outside_mask_ratio(geom.content_safe_box, union), 0.001)
        self.assertLessEqual(_box_outside_mask_ratio(geom.text_box, union), 0.001)


class TestRenderIntegration(unittest.TestCase):
    def _render(self, text_w, text_h, pad_x, pad_y, font_size, line_count, with_tail):
        ew, eh = _envelope_rect(text_w, text_h, pad_x, pad_y, 2)
        rect = (120, 80, int(round(120 + ew)), int(round(80 + eh)))
        anchor = (rect[0] + (rect[2] - rect[0]) // 2, rect[3] + 80)
        canvas = (int(round(120 + ew)) + 200, int(round(80 + eh)) + 260)
        overlay = Image.new("RGBA", canvas, (0, 0, 0, 0))
        _draw_layout_bubble(
            overlay, rect, anchor, "ellipse",
            (250, 250, 247, 255), (24, 24, 24, 255), 2.0, 18, with_tail,
            organic=True, wobble=0.05, point_count=220, seed=123,
            text_w=text_w, text_h=text_h, font_size=font_size,
            padding_x=pad_x, padding_y=pad_y, line_count=line_count,
        )
        return np.asarray(overlay), rect

    def test_renders_without_fallback(self):
        arr, rect = self._render(300, 80, 35, 22, 32, 2, True)
        # 렌더 결과가 비어 있지 않아야 한다(legacy 폴백/검증 실패 아님).
        self.assertGreater(int((arr[..., 3] > 10).sum()), 5000)

    def test_body_stays_inside_rect(self):
        arr, rect = self._render(300, 80, 35, 22, 32, 2, False)
        alpha = (arr[..., 3] > 10)
        x1, y1, x2, y2 = rect
        inside = np.zeros_like(alpha)
        inside[y1:y2, x1:x2] = True
        # 꼬리가 없을 때는 mask 가 rect 안에 완전히 들어가야 한다.
        self.assertEqual(int((alpha & ~inside).sum()), 0)

    def test_no_tail_also_renders(self):
        arr, _ = self._render(200, 260, 22, 18, 30, 5, False)
        self.assertGreater(int((arr[..., 3] > 10).sum()), 5000)


if __name__ == "__main__":
    unittest.main()
