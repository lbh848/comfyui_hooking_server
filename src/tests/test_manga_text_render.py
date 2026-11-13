"""manga_text_render 단위 테스트 — 검출/인식 주입으로 다운로드 없이 검증."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image, ImageDraw

from modes import manga_text_render as mtr
from modes.postprocess import (
    _default_manga_text,
    get_postprocess_config,
    get_postprocess_mode,
    normalize_postprocess_settings,
)


def _document(items, width=400, height=400):
    return {
        "moss": {
            "items": items,
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
            "image_size": {"width": width, "height": height},
        }
    }


def _text_item(text, pixel_box, width=400, height=400):
    x0, y0, x1, y1 = pixel_box
    half_w, half_h = width / 2.0, height / 2.0
    return {
        "label": "text",
        "text": text,
        "caption": "speech balloon",
        "bbox": [y0 - half_h, x0 - half_w, y1 - half_h, x1 - half_w],
    }


def _png(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _white_image(size=(400, 400)):
    return Image.new("RGB", size, (255, 255, 255))


def _recognizer(*texts):
    def recognize(_image, blocks, _items, _settings):
        return [{**block, "recognized_text": text, "recognition_score": 1.0}
                for block, text in zip(blocks, texts)]
    return recognize


class ExtractTextItemsTest(unittest.TestCase):
    def test_centered_yxyx_to_pixels(self):
        document = _document([_text_item("hello", (0, 100, 100, 300))])
        items = mtr.extract_text_items(document, image_size=(400, 400))
        self.assertEqual(len(items), 1)
        self.assertEqual([round(v) for v in items[0]["bbox"]], [0, 100, 100, 300])

    def test_scales_when_actual_image_differs(self):
        document = _document([_text_item("hello", (0, 100, 100, 300))])
        items = mtr.extract_text_items(document, image_size=(800, 800))
        self.assertEqual([round(v) for v in items[0]["bbox"]], [0, 200, 200, 600])

    def test_skips_non_text_and_empty_text(self):
        document = _document([
            {"label": "character", "name": "A", "text": "ignored", "bbox": [-100, -100, 100, 100]},
            {"label": "text", "text": "   ", "bbox": [-100, -100, 100, 100]},
        ])
        self.assertEqual(mtr.extract_text_items(document, image_size=(400, 400)), [])

    def test_rejects_unknown_coord_space(self):
        document = _document([_text_item("hello", (0, 100, 100, 300))])
        document["moss"]["bbox_coord_space"] = "normalized"
        self.assertEqual(mtr.extract_text_items(document, image_size=(400, 400)), [])


class MatchTest(unittest.TestCase):
    def test_filters_small_noise(self):
        boxes = [[0, 0, 200, 40], [10, 100, 30, 110]]
        kept = mtr.filter_text_boxes(boxes, (400, 400))
        self.assertEqual(kept, [[0, 0, 200, 40]])

    def test_assigns_nearest_and_drops_far_boxes(self):
        items = [
            {"bbox": [0, 0, 100, 50]},
            {"bbox": [300, 300, 400, 360]},
        ]
        boxes = [[10, 10, 90, 40], [310, 310, 390, 350], [180, 180, 220, 210]]
        assigned = mtr.assign_boxes_to_items(boxes, items, max_distance=80)
        self.assertEqual(len(assigned[0]), 1)
        self.assertEqual(len(assigned[1]), 1)
        self.assertNotIn(2, assigned)

    def test_groups_stacked_lines_into_one_block(self):
        boxes = [
            [0, 0, 200, 40],
            [5, 45, 195, 85],
            [10, 90, 190, 130],
            [0, 300, 200, 340],
        ]
        blocks = mtr.group_line_boxes(boxes)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]["boxes"]), 3)
        self.assertEqual(len(blocks[1]["boxes"]), 1)

    def test_content_matches_even_when_coordinates_and_order_disagree(self):
        items = [
            {"text": "Good morning", "bbox": [40, 440, 400, 600]},
            {"text": "See you tomorrow", "bbox": [780, 470, 1110, 570]},
            {"text": "Wait a minute", "bbox": [90, 720, 290, 780]},
        ]
        blocks = [
            {"bbox": [460, 70, 700, 140], "recognized_text": "Wait a minute", "recognition_score": 0.95},
            {"bbox": [780, 470, 1110, 570], "recognized_text": "Good morning", "recognition_score": 0.95},
            {"bbox": [90, 720, 290, 780], "recognized_text": "See you tomorrow", "recognition_score": 0.95},
        ]
        matched = mtr.match_blocks_to_items(blocks, items, max_distance=180)
        self.assertEqual(matched[0]["bbox"], blocks[1]["bbox"])
        self.assertEqual(matched[1]["bbox"], blocks[2]["bbox"])
        self.assertEqual(matched[2]["bbox"], blocks[0]["bbox"])

    def test_extra_balloons_do_not_steal_displaced_dialogue(self):
        # 20261114_031559_41db2451: 원래 문장은 멀리 있고, 잘못 생성된 말풍선은 예상 좌표 근처.
        items = [
            {"text": "…야, 잠깐만.", "bbox": [44, 528, 244, 608]},
            {"text": "진짜… 제멋대로네.", "bbox": [84, 948, 364, 1028]},
        ]
        blocks = [
            {"bbox": [343, 241, 487, 271], "recognized_text": "…야, 잠간만.", "recognition_score": 0.82},
            {"bbox": [238, 550, 308, 581], "recognized_text": "Hiyori", "recognition_score": 0.99},
            {"bbox": [267, 833, 383, 890], "recognized_text": "진까\n제멋대로네.", "recognition_score": 0.98},
            {"bbox": [213, 970, 333, 999], "recognized_text": "제멋대로네.", "recognition_score": 0.99},
        ]
        matched = mtr.match_blocks_to_items(blocks, items, max_distance=180)
        self.assertEqual(matched[0]["bbox"], blocks[0]["bbox"])
        self.assertEqual(matched[1]["bbox"], blocks[2]["bbox"])

    def test_unrecognized_unrelated_and_ambiguous_blocks_are_not_assigned(self):
        item = {"text": "Wait a minute", "bbox": [100, 100, 300, 160]}
        for text, confidence in (("Hiyori", 0.99), ("", 1.0), (item["text"], 0.2)):
            block = {"bbox": item["bbox"], "recognized_text": text, "recognition_score": confidence}
            self.assertEqual(mtr.match_blocks_to_items([block], [item], 180), {})
        self.assertEqual(mtr.match_blocks_to_items([{"bbox": item["bbox"]}], [item], 180), {})
        block = {"bbox": item["bbox"], "recognized_text": item["text"], "recognition_score": 0.99}
        self.assertEqual(mtr.match_blocks_to_items([block, dict(block)], [item], 180), {})

    def test_tall_block_does_not_absorb_next_balloon(self):
        boxes = [[100, y, 300, y + 30] for y in (0, 35, 70, 105, 210)]
        blocks = mtr.group_line_boxes(boxes)
        self.assertEqual([len(block["boxes"]) for block in blocks], [4, 1])

    def test_side_by_side_overlapping_boxes_stay_separate(self):
        blocks = mtr.group_line_boxes([[0, 0, 200, 40], [110, 2, 310, 42]])
        self.assertEqual(len(blocks), 2)


class NormalizeTest(unittest.TestCase):
    def test_auto_size_is_explicit_boolean_and_round_trips(self):
        self.assertFalse(mtr.normalize_manga_text_settings({})["auto_font_size"])
        self.assertFalse(mtr.normalize_manga_text_settings({"auto_font_size": "false"})["auto_font_size"])
        settings = normalize_postprocess_settings("manga_text", {"auto_font_size": True})
        self.assertTrue(settings["auto_font_size"])
        self.assertEqual(normalize_postprocess_settings("manga_text", settings), settings)

    def test_font_weight_default_and_normalize(self):
        self.assertEqual(mtr.DEFAULT_MANGA_TEXT_SETTINGS["font_weight"], 500)
        self.assertEqual(
            mtr.normalize_manga_text_settings({"font_weight": 700})["font_weight"], 700
        )
        # 예전 bold=true 저장값은 700 굵기로 옮겨진다.
        legacy = mtr.normalize_manga_text_settings({"bold": True})
        self.assertEqual(legacy["font_weight"], 700)
        self.assertNotIn("bold", legacy)

    def test_font_weight_is_clamped_and_rounded(self):
        self.assertEqual(
            mtr.normalize_manga_text_settings({"font_weight": 5})["font_weight"], 100
        )
        self.assertEqual(
            mtr.normalize_manga_text_settings({"font_weight": 999})["font_weight"], 900
        )
        self.assertEqual(
            mtr.normalize_manga_text_settings({"font_weight": 660})["font_weight"], 700
        )

    def test_faux_bold_for_non_variable_font(self):
        # 시스템 폰트는 가변 축이 없으므로 외곽선 기반 볼드 두께가 잡힌다.
        self.assertGreater(
            mtr._faux_bold_width(40, {"font_weight": 700, "font_id": "system"}), 0
        )
        self.assertGreater(
            mtr._faux_bold_width(40, {"font_weight": 900, "font_id": "system"}),
            mtr._faux_bold_width(40, {"font_weight": 700, "font_id": "system"}),
        )
        self.assertEqual(
            mtr._faux_bold_width(40, {"font_weight": 500, "font_id": "system"}), 0
        )

    def test_clamps_and_enums(self):
        clean = mtr.normalize_manga_text_settings({
            "font_size": 999,
            "min_font_size": -5,
            "erase_mode": "bogus",
            "write_area": "bogus",
            "text_align": "bogus",
            "text_color": "not-a-color",
            "erase_padding": 999,
        })
        self.assertEqual(clean["font_size"], 200)
        self.assertGreaterEqual(clean["min_font_size"], 6)
        self.assertEqual(clean["erase_mode"], "auto")
        self.assertEqual(clean["write_area"], "detected")
        self.assertEqual(clean["text_align"], "center")
        self.assertEqual(clean["text_color"], "#111111")
        self.assertEqual(clean["erase_padding"], 40)


class PostprocessRegistrationTest(unittest.TestCase):
    def test_mode_is_registered(self):
        self.assertEqual(get_postprocess_mode({"postprocess": {"mode": "manga_text"}}), "manga_text")
        config = get_postprocess_config({"postprocess": {"mode": "manga_text"}})
        self.assertIn("manga_text", config)
        self.assertTrue(config["manga_text"]["enabled"])
        self.assertEqual(normalize_postprocess_settings("manga_text", {}), _default_manga_text())


class TextLayoutTest(unittest.TestCase):
    def settings(self, **overrides):
        return mtr.normalize_manga_text_settings({"font_id": "system", **overrides})

    def test_auto_size_ignores_manual_limits(self):
        a = self.settings(auto_font_size=True, font_size=8, min_font_size=6)
        b = self.settings(auto_font_size=True, font_size=200, min_font_size=180)
        first = mtr._fit_text("Hello world", 280, 130, a)
        second = mtr._fit_text("Hello world", 280, 130, b)
        self.assertEqual(first[1:], second[1:])
        self.assertGreater(first[2], 8)

    def test_auto_size_responds_to_text_length_and_area(self):
        settings = self.settings(auto_font_size=True)
        short = mtr._fit_text("Hello", 240, 100, settings)[2]
        long = mtr._fit_text("Hello world " * 15, 240, 100, settings)[2]
        larger = mtr._fit_text("Hello world " * 15, 480, 200, settings)[2]
        self.assertLess(long, short)
        self.assertGreater(larger, long)
        self.assertLess(long, settings["min_font_size"])

    def test_auto_size_can_use_small_system_font(self):
        settings = self.settings(auto_font_size=True)
        font, _lines, size, _line_h = mtr._fit_text("Hello world " * 20, 100, 60, settings)
        self.assertLess(size, 10)
        self.assertEqual(font.size, size)

    def test_scaled_outlined_text_and_manual_minimum_stay_inside_area(self):
        area = [60, 70, 180, 130]
        for automatic in (False, True):
            for scale in (0.7, 1.2):
                for align in ("left", "center", "right"):
                    with self.subTest(automatic=automatic, scale=scale, align=align):
                        settings = self.settings(
                            auto_font_size=automatic, font_size=100, min_font_size=100,
                            text_width_scale=scale, letter_spacing=0.1, outline_width=8,
                            outline_color="#aa0000", font_weight=900, text_align=align,
                        )
                        result = mtr._draw_text_block(
                            _white_image((260, 200)), area, "VeryLongWord" * 6, settings,
                        )
                        ys, xs = np.where(np.any(np.asarray(result) < 245, axis=2))
                        self.assertGreater(len(xs), 0)
                        self.assertGreaterEqual(xs.min(), area[0] + settings["text_padding"])
                        self.assertLess(xs.max(), area[2] - settings["text_padding"])
                        self.assertGreaterEqual(ys.min(), area[1] + settings["text_padding"])
                        self.assertLess(ys.max(), area[3] - settings["text_padding"])

    def test_negative_spacing_can_fit_at_larger_size(self):
        loose = self.settings(auto_font_size=True, letter_spacing=0.1)
        tight = self.settings(auto_font_size=True, letter_spacing=-0.1)
        self.assertGreaterEqual(
            mtr._fit_text("WWWWWWWW", 230, 65, tight)[2],
            mtr._fit_text("WWWWWWWW", 230, 65, loose)[2],
        )


@unittest.skipUnless(mtr._HAS_CV2, "말풍선 경계 검출에는 OpenCV 필요")
class SpeechInteriorTest(unittest.TestCase):
    def balloon(self):
        image = Image.new("RGB", (360, 220), (75, 85, 100))
        draw = ImageDraw.Draw(image)
        draw.ellipse((30, 20, 330, 200), fill=(225, 226, 230), outline="black", width=4)
        font = mtr._load_font(30, {"font_id": "system"})
        draw.text((180, 95), "OLD TEXT", font=font, fill="black", anchor="mt")
        return image

    def block(self):
        return {"bbox": [25, 70, 335, 150], "boxes": [[25, 70, 335, 150]]}

    def test_trims_empty_horizontal_margins_to_actual_ink(self):
        area, region = mtr._refine_text_region(np.array(self.balloon()), self.block())
        self.assertIsNotNone(region)
        self.assertGreater(area[0], 90)
        self.assertLess(area[2], 270)
        self.assertEqual(area[1::2], [70, 150])

    def test_safe_rectangle_fits_curved_boundary(self):
        image = self.balloon()
        _area, region = mtr._refine_text_region(np.array(image), self.block())
        area = mtr._safe_write_area([25, 20, 335, 200], region)
        self.assertIsNotNone(area)
        self.assertGreater(area[0], 30)
        self.assertLess(area[2], 330)
        self.assertTrue(np.all(mtr._region_mask(region, area) > 0))

    def test_long_text_is_fitted_inside_balloon_without_cutting(self):
        image = self.balloon()
        _area, region = mtr._refine_text_region(np.array(image), self.block())
        before = np.asarray(image)
        for automatic in (True, False):
            with self.subTest(automatic=automatic):
                settings = mtr.normalize_manga_text_settings({
                    "font_id": "system", "auto_font_size": automatic,
                    "font_size": 100, "min_font_size": 100,
                    "text_padding": 2, "text_width_scale": 1.2,
                    "outline_width": 4, "text_color": "#aa0000",
                })
                result = mtr._draw_text_block(image, [25, 20, 335, 200], "A long replacement " * 10, settings, region)
                changed = np.any(np.asarray(result) != before, axis=2)
                allowed = mtr._region_mask(region, (0, 0, image.width, image.height)) > 0
                self.assertTrue(changed.any())
                self.assertFalse(np.any(changed & ~allowed))

    def test_erasing_preserves_outline_and_surrounding_artwork(self):
        image = self.balloon()
        before = np.array(image)
        _area, region = mtr._refine_text_region(before, self.block())
        allowed = mtr._region_mask(region, (0, 0, image.width, image.height)) > 0
        for mode in ("flat", "inpaint"):
            with self.subTest(mode=mode):
                settings = mtr.normalize_manga_text_settings({"erase_padding": 40, "erase_mode": mode})
                result = mtr._erase_in_place(before.copy(), {0: self.block()["boxes"]}, settings, {0: region})
                self.assertTrue(np.array_equal(result[~allowed], before[~allowed]))
                self.assertFalse(np.array_equal(result[allowed], before[allowed]))

    def test_uniform_page_has_no_assumed_balloon_boundary(self):
        image = _white_image((360, 220))
        draw = ImageDraw.Draw(image)
        draw.text((180, 95), "OLD TEXT", font=mtr._load_font(30, {"font_id": "system"}), fill="black", anchor="mt")
        area, region = mtr._refine_text_region(np.array(image), self.block())
        self.assertIsNone(region)
        self.assertGreater(area[0], 90)

    def test_no_opencv_retains_original_detection(self):
        from unittest.mock import patch
        with patch.object(mtr, "_HAS_CV2", False):
            area, region = mtr._refine_text_region(np.array(self.balloon()), self.block())
        self.assertEqual(area, self.block()["bbox"])
        self.assertIsNone(region)

    def test_compose_preserves_boundary_for_detected_and_json_areas(self):
        image = self.balloon()
        before = np.asarray(image)
        _area, region = mtr._refine_text_region(before, self.block())
        allowed = mtr._region_mask(region, (0, 0, image.width, image.height)) > 0
        document = _document([
            _text_item("A much longer replacement " * 5, (25, 20, 335, 200), width=360, height=220),
        ], width=360, height=220)
        for write_area in ("detected", "json_bbox"):
            with self.subTest(write_area=write_area):
                output = mtr.compose_manga_text(
                    _png(image), document,
                    {"font_id": "system", "auto_font_size": True, "erase_padding": 40,
                     "write_area": write_area, "text_color": "#aa0000"},
                    detector=lambda *_: self.block()["boxes"],
                    recognizer=_recognizer("A much longer replacement " * 5),
                )
                result = np.asarray(Image.open(io.BytesIO(output)).convert("RGB"))
                self.assertTrue(np.array_equal(result[~allowed], before[~allowed]))
                self.assertFalse(np.array_equal(result[allowed], before[allowed]))

    def test_no_safe_writing_space_does_not_erase_original_text(self):
        from unittest.mock import patch
        image = self.balloon()
        source = _png(image)
        document = _document([
            _text_item("Replacement", self.block()["bbox"], width=360, height=220),
        ], width=360, height=220)
        with patch.object(mtr, "_safe_write_area", return_value=None):
            output = mtr.compose_manga_text(
                source, document, {"font_id": "system"},
                detector=lambda *_: self.block()["boxes"],
                recognizer=_recognizer("Replacement"),
            )
        self.assertEqual(output, source)


class ComposeTest(unittest.TestCase):
    def test_replaces_old_ink_with_json_text(self):
        image = _white_image()
        draw = ImageDraw.Draw(image)
        draw.rectangle([100, 100, 300, 160], fill=(0, 0, 0))
        source = _png(image)
        document = _document([_text_item("NEW TEXT", (100, 100, 300, 160))])
        calls = []

        def detector(img, settings):
            calls.append(img.size)
            return [[100, 100, 300, 160]]

        output = mtr.compose_manga_text(
            source,
            document,
            {
                "font_id": "system",
                "font_size": 40,
                "min_font_size": 18,
                "erase_mode": "flat",
                "outline_width": 0,
            },
            detector=detector,
            recognizer=_recognizer("NEW TEXT"),
        )
        self.assertEqual(calls, [(400, 400)])
        result = Image.open(io.BytesIO(output)).convert("RGB")
        self.assertEqual(result.size, (400, 400))
        self.assertNotEqual(output, source)
        # 검은 사각형이 지워졌는지: 영역 대부분이 밝아야 한다.
        region = np.asarray(result)[100:160, 100:300]
        self.assertGreater(float(region.mean()), 180.0)

    def test_unmatched_blocks_leave_image_untouched(self):
        image = _white_image()
        source = _png(image)
        # 대사가 확인되지 않은 영역에는 좌표에 상관없이 쓰지 않는다.
        document = _document([
            _text_item("A", (0, 0, 100, 40)),
            _text_item("B", (200, 0, 300, 40)),
        ])

        def detector(img, settings):
            return [[300, 300, 390, 340]]

        output = mtr.compose_manga_text(
            source, document, {"font_id": "system"}, detector=detector,
            recognizer=_recognizer("Unknown text"),
        )
        self.assertEqual(output, source)

    def test_font_weight_renders_without_changing_size(self):
        image = _white_image()
        draw = ImageDraw.Draw(image)
        draw.rectangle([100, 100, 300, 160], fill=(0, 0, 0))
        source = _png(image)
        document = _document([_text_item("NEW TEXT", (100, 100, 300, 160))])

        def detector(img, settings):
            return [[100, 100, 300, 160]]

        output = mtr.compose_manga_text(
            source,
            document,
            {"font_id": "system", "font_size": 40, "min_font_size": 18, "font_weight": 700},
            detector=detector,
            recognizer=_recognizer("NEW TEXT"),
        )
        result = Image.open(io.BytesIO(output)).convert("RGB")
        self.assertEqual(result.size, (400, 400))
        self.assertNotEqual(output, source)

    def test_missing_text_items_returns_original(self):
        image = _white_image()
        source = _png(image)
        output = mtr.compose_manga_text(source, {"moss": {"items": []}}, {})
        self.assertEqual(output, source)

    def test_ocr_failure_preserves_original_in_both_write_modes(self):
        from unittest.mock import Mock
        source = _png(_white_image())
        document = _document([_text_item("Wait a minute", (100, 100, 300, 160))])
        for mode in ("detected", "json_bbox"):
            with self.subTest(mode=mode):
                output = mtr.compose_manga_text(
                    source, document, {"font_id": "system", "write_area": mode},
                    detector=lambda *_: [[100, 100, 300, 160]],
                    recognizer=Mock(side_effect=RuntimeError("OCR unavailable")),
                )
                self.assertEqual(output, source)

    def test_json_bbox_does_not_insert_unmatched_dialogue(self):
        source = _png(_white_image())
        document = _document([
            _text_item("Good morning", (10, 20, 180, 60)),
            _text_item("Wait a minute", (200, 200, 380, 240)),
        ])
        output = mtr.compose_manga_text(
            source, document, {"font_id": "system", "write_area": "json_bbox"},
            detector=lambda *_: [[10, 20, 180, 60], [200, 200, 380, 240]],
            recognizer=_recognizer("Good morning", "Hiyori"),
        )
        before = np.array(Image.open(io.BytesIO(source)))
        after = np.array(Image.open(io.BytesIO(output)))
        self.assertFalse(np.array_equal(before[20:60, 10:180], after[20:60, 10:180]))
        self.assertTrue(np.array_equal(before[190:250, 190:390], after[190:250, 190:390]))

    def test_disabled_returns_original(self):
        image = _white_image()
        source = _png(image)
        document = _document([_text_item("NEW", (0, 0, 100, 40))])
        output = mtr.compose_manga_text(source, document, {"enabled": False})
        self.assertEqual(output, source)


if __name__ == "__main__":
    unittest.main()
