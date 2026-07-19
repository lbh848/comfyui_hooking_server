"""유기형 말풍선 외곽선/꼬리 단위 테스트(Phase 1)."""

import unittest

import numpy as np
from PIL import Image

from modes.bubble_shape import make_organic_ellipse
from modes.bubble_types import OrganicShapeConfig
from modes.bubble_render import (
    _draw_layout_bubble,
    _resolve_bubble_shape_mode,
    _resolve_tail_width_scale,
    _resolve_organic_wobble,
)


class TestOrganicEllipse(unittest.TestCase):
    def test_reproducible_with_same_seed(self):
        first = make_organic_ellipse((200, 120), (90, 55), seed=42)
        second = make_organic_ellipse((200, 120), (90, 55), seed=42)
        self.assertTrue(np.array_equal(first, second))

    def test_changes_with_different_seed(self):
        first = make_organic_ellipse((200, 120), (90, 55), seed=42)
        second = make_organic_ellipse((200, 120), (90, 55), seed=43)
        self.assertFalse(np.array_equal(first, second))

    def test_shape_and_dtype(self):
        config = OrganicShapeConfig(point_count=160)
        contour = make_organic_ellipse((200, 120), (90, 55), seed=1, config=config)
        self.assertEqual(contour.shape, (160, 2))
        self.assertEqual(contour.dtype, np.int32)

    def test_wobble_keeps_text_safe_radius(self):
        """wobble 이 텍스트 영역을 침범하지 않도록 내부 타원 하한을 유지한다.

        각 외곽선 점은 (rx*min_scale, ry*min_scale) 내부 타원 바깥에 있어야 한다.
        점 (dx,dy) 가 그 타원 바깥 ⇔ (dx/(rx*min))^2 + (dy/(ry*min))^2 >= 1.
        """
        cx, cy, rx, ry = 200, 120, 90, 55
        config = OrganicShapeConfig(point_count=180, wobble=0.075, min_radial_scale=0.90)
        contour = make_organic_ellipse((cx, cy), (rx, ry), seed=7, config=config)
        dx = contour[:, 0] - cx
        dy = contour[:, 1] - cy
        inner = (dx / (rx * config.min_radial_scale)) ** 2 + (
            dy / (ry * config.min_radial_scale)
        ) ** 2
        self.assertGreaterEqual(float(inner.min()), 1.0 - 1e-3)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            make_organic_ellipse((200, 120), (0, 55), seed=1)
        with self.assertRaises(ValueError):
            make_organic_ellipse(
                (200, 120), (90, 55), seed=1,
                config=OrganicShapeConfig(point_count=16),
            )


class TestOrganicTailAdditive(unittest.TestCase):
    """organic 꼬리는 몸통을 파내지(노치) 않고 덧셈으로 붙어야 한다."""

    def _filled_pixels(self, with_tail, scale=1.0):
        overlay = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
        _draw_layout_bubble(
            overlay, (120, 80, 280, 200), (300.0, 300.0), "ellipse",
            (255, 255, 255, 255), (40, 40, 40, 255), 2, 22, with_tail,
            organic=True, tail_width_scale=scale, seed=11,
        )
        return int((np.asarray(overlay)[..., 3] > 0).sum())

    def test_tail_adds_area_without_carving_body(self):
        """꼬리가 있으면 채움 면적이 순증한다(몸통 일부를 깎아내지 않는다)."""
        body_only = self._filled_pixels(False)
        with_tail = self._filled_pixels(True)
        self.assertGreater(with_tail, body_only)

    def test_tail_width_scale_grows_filled_area(self):
        """tail_width_scale 이 클수록 꼬리가 채우는 면적이 커진다."""
        thin = self._filled_pixels(True, scale=0.2)
        thick = self._filled_pixels(True, scale=3.0)
        self.assertGreater(thick, thin)


class TestOrganicRenderIntegration(unittest.TestCase):
    def test_organic_branch_fills_overlay(self):
        """organic=True 분기가 overlay 에 채움을 남기고 예외를 던지지 않는다."""
        overlay = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
        rect = (120, 80, 280, 200)
        anchor = (260.0, 260.0)
        fill = (255, 255, 255, 255)
        border = (40, 40, 40, 255)
        _draw_layout_bubble(
            overlay, rect, anchor, "ellipse", fill, border, 2, 22, True,
            organic=True, tail_width_scale=1.5, wobble=0.055, seed=42,
        )
        arr = np.asarray(overlay)
        # 채움이 전부 비어 있으면 렌더가 실패한 것.
        self.assertGreater(int((arr[..., 3] > 0).sum()), 0)

    def test_organic_falls_back_on_bad_config(self):
        """organic 생성이 예외를 던지면 legacy 렌더로 폴백해 overlay 를 채운다."""
        overlay = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
        rect = (120, 80, 280, 200)
        _draw_layout_bubble(
            overlay, rect, (260.0, 260.0), "ellipse",
            (255, 255, 255, 255), (40, 40, 40, 255), 2, 22, True,
            organic=True, point_count=8, seed=1,  # point_count<32 → ValueError → 폴백
        )
        arr = np.asarray(overlay)
        self.assertGreater(int((arr[..., 3] > 0).sum()), 0)

    def test_organic_works_without_tail(self):
        """unanchored(무꼬리) 경로: with_tail=False 도 유기형 외곽선이 그려진다."""
        overlay = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
        _draw_layout_bubble(
            overlay, (120, 80, 280, 200), (260.0, 260.0), "ellipse",
            (255, 255, 255, 255), (40, 40, 40, 255), 2, 22, False,
            organic=True, seed=5,
        )
        arr = np.asarray(overlay)
        self.assertGreater(int((arr[..., 3] > 0).sum()), 0)


class TestSettingsResolvers(unittest.TestCase):
    def test_bubble_shape_mode(self):
        self.assertEqual(_resolve_bubble_shape_mode({"bubble_shape": "organic"}), "organic")
        self.assertEqual(_resolve_bubble_shape_mode({"bubble_shape": "LEGACY"}), "legacy")
        self.assertEqual(_resolve_bubble_shape_mode({}), "legacy")
        self.assertEqual(_resolve_bubble_shape_mode({"bubble_shape": "weird"}), "legacy")

    def test_tail_width_scale_clamps(self):
        self.assertAlmostEqual(_resolve_tail_width_scale({"tail_width_scale": 1.5}), 1.5)
        self.assertEqual(_resolve_tail_width_scale({"tail_width_scale": 99}), 3.0)
        self.assertEqual(_resolve_tail_width_scale({"tail_width_scale": 0.0}), 0.2)
        self.assertEqual(_resolve_tail_width_scale({}), 1.0)
        self.assertEqual(_resolve_tail_width_scale({"tail_width_scale": "abc"}), 1.0)

    def test_organic_wobble_clamps(self):
        self.assertAlmostEqual(_resolve_organic_wobble({"organic_wobble": 0.06}), 0.06)
        self.assertAlmostEqual(_resolve_organic_wobble({"organic_wobble": 0.3}), 0.3)
        self.assertAlmostEqual(_resolve_organic_wobble({"organic_wobble": 5}), 0.30)
        self.assertAlmostEqual(_resolve_organic_wobble({"organic_wobble": 0}), 0.02)
        self.assertAlmostEqual(_resolve_organic_wobble({}), 0.055)


if __name__ == "__main__":
    unittest.main()
