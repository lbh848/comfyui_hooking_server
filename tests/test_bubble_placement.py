import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from modes.background_segmenter import background_ratio
from modes.face_detector import _detect_multi, _letterbox
from modes.face_embedder import (
    appearance_descriptor,
    appearance_similarity,
    expanded_face_box,
    prepare_face_for_embedding,
    standardize_face_image,
)
from modes.bubble_layout import choose_layout, choose_scaled_layout
from modes.bubble_match import (
    _assignment_ambiguity_gap,
    _face_boxes_overlap,
    _optimal_assignment,
    _sequential_overlap_assignment,
    match_speakers_to_faces,
)
from modes.bubble_predictor import (
    _rect_iou,
    generate_grid_candidates,
    select_candidate,
    select_relaxed_candidate,
)
from modes.postprocess import normalize_layout_font_scale
from modes.bubble_render import (
    _apply_unanchored_fallbacks,
    _draw_layout_bubble,
    _draw_preview_debug,
    _face_candidate_limit,
    _face_detection_candidate_limit,
    _filter_nested_face_candidates,
    _is_single_speaker_thought,
    _draw_speech,
    _place_body,
    _place_unanchored_body,
    _protected_face_box,
    _resolve_layout_font_scale,
    _tail_within_threshold,
    _tail_side,
)


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class _FakeFaceSession:
    class _Input:
        name = "images"

    def __init__(self, predictions):
        self.output = np.asarray(predictions, dtype=np.float32).T[None]

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, _feeds):
        return [self.output]


class BubblePlacementTest(unittest.TestCase):
    def test_all_low_confidence_faces_use_unanchored_fallback_without_tails(self):
        matched = [
            {
                "segment": {"speaker": "Maria", "type": "speech"},
                "face_box": (500, 200, 900, 800),
            },
            {
                "segment": {"speaker": "Alisa", "type": "thought"},
                "face_box": (80, 800, 160, 880),
            },
        ]
        faces = [
            {"box": matched[0]["face_box"], "conf": 0.006},
            {"box": matched[1]["face_box"], "conf": 0.0001},
        ]

        _apply_unanchored_fallbacks(matched, faces)

        for item in matched:
            self.assertIsNone(item["face_box"])
            self.assertTrue(item["unanchored_fallback"])
            self.assertEqual(
                item["unmatched_reason"],
                "no_reliable_face_detection",
            )

    def test_only_individually_unmatched_speaker_uses_unanchored_fallback(self):
        matched = [
            {
                "segment": {"speaker": "Maria", "type": "speech"},
                "face_box": (100, 100, 180, 180),
            },
            {
                "segment": {"speaker": "Alisa", "type": "speech"},
                "face_box": None,
            },
        ]
        faces = [{"box": matched[0]["face_box"], "conf": 0.8}]

        _apply_unanchored_fallbacks(matched, faces)

        self.assertEqual(matched[0]["face_box"], (100, 100, 180, 180))
        self.assertNotIn("unanchored_fallback", matched[0])
        self.assertIsNone(matched[1]["face_box"])
        self.assertTrue(matched[1]["unanchored_fallback"])
        self.assertEqual(matched[1]["unmatched_reason"], "face_match_unassigned")

    def test_face_letterbox_matches_ultralytics_opencv_geometry(self):
        import cv2

        pixels = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
        image = Image.fromarray(pixels, mode="RGB")

        tensor, gain, left, top = _letterbox(image, size=12)

        expected_resized = cv2.resize(
            pixels, (12, 7), interpolation=cv2.INTER_LINEAR
        )
        expected = cv2.copyMakeBorder(
            expected_resized,
            2,
            3,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        actual = (tensor[0].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

        self.assertAlmostEqual(gain, 12 / 9)
        self.assertEqual((left, top), (0, 2))
        np.testing.assert_array_equal(actual, expected)

    def test_layout_font_scale_is_user_configurable_and_clamped(self):
        self.assertEqual(normalize_layout_font_scale(1.7), 1.7)
        self.assertEqual(normalize_layout_font_scale(9), 4.0)
        self.assertEqual(normalize_layout_font_scale(0.2), 1.0)
        self.assertEqual(_resolve_layout_font_scale({"layout_font_scale": "3.5"}), 3.5)
        self.assertEqual(_resolve_layout_font_scale({"layout_font_scale": 99}), 4.0)

    def test_face_protection_adds_safety_margin(self):
        protected = _protected_face_box((100, 100, 160, 160), (500, 500))
        self.assertEqual(protected, (92.0, 92.0, 168.0, 168.0))

    def test_face_candidate_limit_tracks_unique_speaker_names(self):
        self.assertEqual(_face_candidate_limit([]), 0)
        self.assertEqual(
            _face_candidate_limit([
                {"speaker": "alice"},
                {"speaker": "alice"},
            ]),
            1,
        )
        self.assertEqual(
            _face_candidate_limit([
                {"speaker": "alice"},
                {"speaker": "bob"},
                {"speaker": "carol"},
            ]),
            3,
        )

    def test_face_candidate_limit_ignores_repeated_speech_and_thought(self):
        self.assertEqual(
            _face_candidate_limit([
                {"speaker": "alice", "type": "speech"},
                {"speaker": "alice", "type": "thought"},
                {"speaker": "bob", "type": "speech"},
            ]),
            2,
        )

    def test_face_detection_candidate_pool_is_wider_than_speaker_count(self):
        self.assertEqual(_face_detection_candidate_limit(0), 0)
        self.assertEqual(_face_detection_candidate_limit(1), 8)
        self.assertEqual(_face_detection_candidate_limit(2), 16)
        self.assertEqual(_face_detection_candidate_limit(5), 40)
        self.assertEqual(_face_detection_candidate_limit(20), 64)
        self.assertEqual(_face_detection_candidate_limit(2, per_character=3), 6)

    def test_nested_low_confidence_giant_face_candidate_is_removed(self):
        real = {"box": (620, 120, 780, 300), "conf": 0.24}
        giant = {"box": (520, 20, 965, 670), "conf": 0.00004}
        side_face = {"box": (430, 200, 555, 357), "conf": 0.00004}
        filtered = _filter_nested_face_candidates([real, giant, side_face])
        self.assertEqual(filtered, [real, side_face])

    def test_nested_same_face_box_surviving_nms_is_removed_by_coverage(self):
        tight = {
            "box": (316.47, 137.37, 461.18, 309.32),
            "conf": 0.000107,
        }
        wide = {
            "box": (259.00, 23.80, 511.83, 310.33),
            "conf": 0.000069,
        }
        filtered = _filter_nested_face_candidates([tight, wide])
        self.assertEqual(filtered, [tight])

    def test_nested_smaller_low_confidence_box_is_also_removed(self):
        wide = {"box": (580, 125, 740, 335), "conf": 0.002}
        tight = {"box": (596, 152, 673, 248), "conf": 0.00002}
        filtered = _filter_nested_face_candidates([wide, tight])
        self.assertEqual(filtered, [wide])

    def test_similar_size_same_face_duplicate_is_removed(self):
        primary = {
            "box": (198.35, 193.51, 323.90, 321.71),
            "conf": 0.07275,
        }
        duplicate = {
            "box": (194.58, 155.55, 309.61, 273.47),
            "conf": 0.0000085,
        }
        filtered = _filter_nested_face_candidates([primary, duplicate])
        self.assertEqual(filtered, [primary])

    def test_attached_low_confidence_sliver_is_removed(self):
        face = {
            "box": (570.50, 297.99, 662.08, 402.45),
            "conf": 0.0002218,
        }
        sliver = {
            "box": (548.91, 315.49, 578.75, 388.83),
            "conf": 0.0000108,
        }
        filtered = _filter_nested_face_candidates([face, sliver])
        self.assertEqual(filtered, [face])

    def test_independent_narrow_side_face_is_kept(self):
        primary = {"box": (570, 298, 662, 402), "conf": 0.2}
        side_face = {"box": (500, 315, 530, 389), "conf": 0.01}
        filtered = _filter_nested_face_candidates([primary, side_face])
        self.assertEqual(filtered, [primary, side_face])

    def test_flat_hair_box_does_not_remove_lower_confidence_full_face(self):
        hair = {"box": (282, 80, 369, 138), "conf": 0.00017}
        full_face = {"box": (246, 8, 478, 276), "conf": 0.00012}
        filtered = _filter_nested_face_candidates([hair, full_face])
        self.assertEqual(filtered, [hair, full_face])

    def test_partially_overlapping_distinct_faces_are_kept(self):
        left = {"box": (100, 100, 220, 240), "conf": 0.8}
        right = {"box": (180, 100, 300, 240), "conf": 0.7}
        filtered = _filter_nested_face_candidates([left, right])
        self.assertEqual(filtered, [left, right])

    def test_face_standardization_pads_without_cropping(self):
        image = Image.new("RGB", (40, 20), "black")
        ImageDraw.Draw(image).rectangle((30, 0, 39, 19), fill="red")
        standardized = standardize_face_image(image)
        self.assertEqual(standardized.size, (40, 40))
        self.assertEqual(standardized.getpixel((35, 15)), (255, 0, 0))

    def test_embedding_preparation_forces_rgb_square_padding(self):
        image = Image.new("RGBA", (20, 40), (255, 0, 0, 128))
        prepared = prepare_face_for_embedding(image)
        self.assertEqual(prepared.mode, "RGB")
        self.assertEqual(prepared.size, (40, 40))
        self.assertEqual(prepared.getpixel((20, 20)), (255, 0, 0))

    def test_face_crop_expands_with_data_patch_top_bottom_rule(self):
        image = Image.new("RGB", (200, 200), "white")
        expanded = expanded_face_box(
            image,
            (80, 80, 120, 120),
            top_mult=2.5,
            bottom_mult=1.0,
        )
        self.assertEqual(expanded, (65.0, 50.0, 135.0, 120.0))

    def test_low_confidence_edge_candidate_is_deprioritized(self):
        session = _FakeFaceSession([
            (20, 100, 40, 60, 0.04),   # 경계에 붙은 저신뢰 후보
            (400, 400, 100, 100, 0.01),  # 내부 후보
        ])
        boxes, confidences = _detect_multi(
            session,
            Image.new("RGB", (960, 960), "white"),
            conf_thres=0.0,
            max_faces=1,
        )
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(confidences[0], 0.01, places=5)
        self.assertGreater(boxes[0][0], 1.0)

    def test_single_speaker_thought_requires_only_thought_segments(self):
        self.assertTrue(_is_single_speaker_thought([
            {"speaker": "alice", "type": "thought"},
            {"speaker": "alice", "type": "thought"},
        ]))
        self.assertFalse(_is_single_speaker_thought([
            {"speaker": "alice", "type": "speech"},
        ]))
        self.assertFalse(_is_single_speaker_thought([
            {"speaker": "alice", "type": "thought"},
            {"speaker": "bob", "type": "thought"},
        ]))

    def test_unanchored_monologue_placement_prefers_background(self):
        protected = np.zeros((200, 200), dtype=np.uint8)
        protected[:, :100] = 1
        placed = _place_unanchored_body(
            60,
            40,
            [],
            200,
            200,
            protected_foreground_mask=protected,
        )
        self.assertIsNotNone(placed)
        rect, _anchor = placed
        self.assertGreaterEqual(rect[0], 100)

    def test_appearance_descriptor_distinguishes_brightness_distribution(self):
        dark = appearance_descriptor(Image.new("RGB", (64, 64), (40, 30, 20)))
        bright = appearance_descriptor(Image.new("RGB", (64, 64), (230, 230, 230)))
        self.assertIsNotNone(dark)
        self.assertIsNotNone(bright)
        self.assertLess(appearance_similarity(dark, bright), 0.8)

    def test_face_detector_discards_outside_boxes_before_limit(self):
        session = _FakeFaceSession([
            (100, 480, 100, 100, 0.99),  # 세로 이미지의 왼쪽 letterbox 패딩
            (480, 400, 160, 160, 0.90),  # 유효 얼굴 1
            (650, 700, 120, 120, 0.80),  # 유효 얼굴 2
        ])
        boxes, confidences = _detect_multi(
            session,
            Image.new("RGB", (100, 200), "white"),
            conf_thres=0.3,
            max_faces=1,
        )
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(confidences), 1)
        self.assertAlmostEqual(confidences[0], 0.9, places=5)
        self.assertTrue(all(value >= 0 for value in boxes[0]))
        self.assertLessEqual(boxes[0][2], 100)
        self.assertLessEqual(boxes[0][3], 200)

    def test_face_detector_rejects_flat_and_heavily_clipped_candidates(self):
        session = _FakeFaceSession([
            (150, 150, 120, 120, 0.95),   # 유효 얼굴 1
            (400, 930, 300, 24, 0.93),    # 이미지 하단의 납작한 오검출
            (600, 1020, 140, 140, 0.92),  # 캔버스 밖으로 대부분 잘린 후보
            (700, 700, 100, 100, 0.80),   # 유효 얼굴 2
        ])
        boxes, confidences = _detect_multi(
            session,
            Image.new("RGB", (960, 960), "white"),
            conf_thres=0.0,
            max_faces=2,
        )
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(confidences), 2)
        self.assertAlmostEqual(confidences[0], 0.95, places=5)
        self.assertAlmostEqual(confidences[1], 0.80, places=5)
        self.assertTrue(all(0.35 <= (b[2] - b[0]) / (b[3] - b[1]) <= 2.86 for b in boxes))

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

    def test_force_shape_cloud_overrides_model_choice(self):
        # force_shape="cloud"면 모델 기본 선택과 무관하게 항상 cloud가 나온다.
        selected, _ = choose_scaled_layout(
            "짧은 한마디",
            (1056, 1536),
            font_scale=1.0,
            force_shape="cloud",
        )
        self.assertEqual(selected.shape, "cloud")

    def test_speech_layout_is_limited_to_ellipse_or_comic_source_shape(self):
        selected, _ = choose_scaled_layout(
            "잠깐… 이게 정말 맞는 선택일까? 조금 더 생각해 보자…",
            (1056, 1536),
            font_scale=1.0,
            allowed_shapes=("ellipse", "rounded"),
        )
        self.assertIn(selected.shape, ("ellipse", "rounded"))

    def test_wrap_never_splits_inside_space_delimited_words(self):
        text = "안녕하세요 반갑습니다 다시 만나요"
        selected, _ = choose_layout(
            text,
            (240, 320),
            min_font_size=20,
            max_font_size=20,
            allowed_shapes=("ellipse",),
        )
        self.assertEqual(" ".join(selected.lines), text)

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

    def test_candidate_on_foreground_is_rejected(self):
        face = (100, 100, 160, 160)
        protected_foreground = np.zeros((500, 500), dtype=np.uint8)
        protected_foreground[:, :250] = 1
        candidates = [
            {"center": (200, 70), "anchor": (130, 100), "confidence": 0.9},
            {"center": (350, 70), "anchor": (160, 100), "confidence": 0.1},
        ]
        chosen = select_candidate(
            candidates,
            body_size=(60, 40),
            face_box=face,
            canvas_size=(500, 500),
            forbidden_boxes=[face],
            protected_foreground_mask=protected_foreground,
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["center"], (350.0, 70.0))
        self.assertEqual(chosen["background_ratio"], 1.0)

    def test_rect_iou(self):
        self.assertEqual(_rect_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(
            _rect_iou((0, 0, 10, 10), (5, 0, 15, 10)),
            1.0 / 3.0,
        )

    def test_relaxed_candidate_prefers_zero_face_iou_over_clean_background(self):
        face = (90, 90, 170, 170)
        candidates = [
            {"center": (130, 130), "confidence": 0.9, "source": "onnx"},
            {"center": (245, 245), "confidence": 0.1, "source": "grid"},
        ]
        protected = np.zeros((300, 300), dtype=np.uint8)
        protected[200:300, 190:300] = 1
        strict = select_candidate(
            candidates,
            body_size=(90, 70),
            face_box=face,
            canvas_size=(300, 300),
            forbidden_boxes=[face],
            protected_foreground_mask=protected,
        )
        relaxed = select_relaxed_candidate(
            candidates,
            body_size=(90, 70),
            face_box=face,
            canvas_size=(300, 300),
            face_boxes=[face],
            protected_foreground_mask=protected,
        )
        self.assertIsNone(strict)
        self.assertIsNotNone(relaxed)
        self.assertEqual(relaxed["source"], "grid")
        self.assertEqual(relaxed["face_iou"], 0.0)
        self.assertGreater(relaxed["foreground_overlap"], 0.0)

    def test_relaxed_candidate_chooses_lowest_face_iou_when_all_overlap(self):
        face = (40, 40, 260, 260)
        candidates = [
            {"center": (130, 130), "confidence": 0.9},
            {"center": (54, 54), "confidence": 0.1},
        ]
        relaxed = select_relaxed_candidate(
            candidates,
            body_size=(100, 100),
            face_box=face,
            canvas_size=(300, 300),
            face_boxes=[face],
        )
        self.assertIsNotNone(relaxed)
        ious = [
            _rect_iou((80, 80, 180, 180), face),
            _rect_iou((4, 4, 104, 104), face),
        ]
        self.assertAlmostEqual(relaxed["face_iou"], min(ious), places=6)

    def test_relaxed_candidate_prefers_zero_bubble_iou_over_clean_background(self):
        candidates = [
            {"center": (55, 50), "confidence": 0.9, "source": "onnx"},
            {"center": (245, 250), "confidence": 0.1, "source": "grid"},
        ]
        protected = np.zeros((300, 300), dtype=np.uint8)
        protected[210:300, 200:300] = 1
        relaxed = select_relaxed_candidate(
            candidates,
            body_size=(80, 60),
            face_box=(130, 120, 170, 160),
            canvas_size=(300, 300),
            occupied_boxes=[(15, 20, 95, 80)],
            protected_foreground_mask=protected,
        )
        self.assertIsNotNone(relaxed)
        self.assertEqual(relaxed["source"], "grid")
        self.assertEqual(relaxed["bubble_iou"], 0.0)
        self.assertGreater(relaxed["foreground_overlap"], 0.0)

    def test_grid_candidates_cover_canvas_edges(self):
        candidates = generate_grid_candidates(
            (80, 60),
            (120, 120, 180, 180),
            (300, 300),
        )
        self.assertGreater(len(candidates), 4)
        xs = [item["center"][0] for item in candidates]
        ys = [item["center"][1] for item in candidates]
        self.assertAlmostEqual(min(xs), 44.0)
        self.assertAlmostEqual(max(xs), 256.0)
        self.assertAlmostEqual(min(ys), 34.0)
        self.assertAlmostEqual(max(ys), 266.0)

    def test_background_ratio_uses_rect_pixels(self):
        protected_foreground = np.zeros((20, 20), dtype=np.uint8)
        protected_foreground[0:10, 0:5] = 1
        self.assertAlmostEqual(
            background_ratio(protected_foreground, (0, 0, 10, 10)),
            0.5,
        )

    def test_safe_fallback_never_covers_face(self):
        face = (200, 200, 260, 260)
        placed = _place_body(face, 100, 50, [face], 500, 500)
        self.assertIsNotNone(placed)
        rect, _, _ = placed
        self.assertFalse(_overlaps(rect, face))

    def test_tail_uses_distance_regardless_of_vertical_direction(self):
        face = (100, 100, 160, 160)
        above = _tail_within_threshold(
            (80, 20, 180, 80), (130, 100), face, 1.0, "ellipse"
        )
        below = _tail_within_threshold(
            (80, 180, 180, 240), (130, 160), face, 1.0, "ellipse"
        )
        far = _tail_within_threshold(
            (80, 300, 180, 360), (130, 160), face, 1.0, "ellipse"
        )
        self.assertTrue(above[0])
        self.assertTrue(below[0])
        self.assertFalse(far[0])

    def test_zero_tail_threshold_disables_tail(self):
        visible, gap, limit = _tail_within_threshold(
            (80, 20, 180, 80), (130, 100), (100, 100, 160, 160),
            0.0, "ellipse",
        )
        self.assertFalse(visible)
        self.assertGreater(gap, 0)
        self.assertEqual(limit, 0)

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

    def test_body_and_curved_tail_have_no_internal_border_seam(self):
        image = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        _draw_layout_bubble(
            image,
            (20, 20, 80, 50),
            (50, 90),
            "ellipse",
            (255, 255, 255, 255),
            (0, 0, 0, 255),
            3,
            12,
            True,
        )
        self.assertEqual(image.getpixel((50, 50)), (255, 255, 255, 255))

    def test_thought_box_is_square_cornered_and_never_draws_tail(self):
        image = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        _draw_layout_bubble(
            image,
            (20, 20, 80, 60),
            (50, 100),
            "box",
            (255, 255, 255, 255),
            (0, 0, 0, 255),
            2,
            20,
            True,
        )
        self.assertNotEqual(image.getpixel((20, 20)), (0, 0, 0, 0))
        self.assertEqual(image.getpixel((50, 85)), (0, 0, 0, 0))

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

    def test_single_name_single_face_still_obeys_embedding_threshold(self):
        segments = [{"speaker": "alice", "text": "hello", "type": "speech"}]
        image = Image.new("RGB", (100, 100), "white")
        faces = [{"box": (20, 20, 60, 60), "conf": 0.9, "image": image}]
        char_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        face_embedding = np.asarray([0.8, 0.6], dtype=np.float32)
        with patch("modes.face_embedder.get_char_embedding", return_value=char_embedding), patch(
            "modes.face_embedder.embed_face_crop", return_value=face_embedding
        ):
            rejected = match_speakers_to_faces(
                segments, faces, "bot", match_thres=0.85
            )
            accepted = match_speakers_to_faces(
                segments, faces, "bot", match_thres=0.75
            )
        self.assertIsNone(rejected[0]["face_box"])
        self.assertEqual(accepted[0]["face_box"], faces[0]["box"])
        self.assertAlmostEqual(accepted[0]["sim"], 0.8, places=5)

    def test_character_missing_from_project_is_marked_for_background_fallback(self):
        segments = [{"speaker": "jihoo", "text": "hello", "type": "thought"}]
        with patch(
            "modes.bubble_match._project_character_name_map",
            return_value={"maria": "Maria", "alisa": "Alisa"},
        ), patch("modes.face_embedder.get_char_embedding") as get_embedding:
            results = match_speakers_to_faces(segments, [], "bot")
        get_embedding.assert_not_called()
        self.assertIsNone(results[0]["face_box"])
        self.assertEqual(
            results[0].get("unmatched_reason"),
            "missing_project_character",
        )

    def test_registered_character_embedding_failure_is_not_background_fallback(self):
        segments = [{"speaker": "maria", "text": "hello", "type": "speech"}]
        with patch(
            "modes.bubble_match._project_character_name_map",
            return_value={"maria": "Maria"},
        ), patch(
            "modes.face_embedder.get_char_embedding",
            return_value=None,
        ) as get_embedding:
            results = match_speakers_to_faces(segments, [], "bot")
        get_embedding.assert_called_once_with("bot", "Maria")
        self.assertIsNone(results[0]["face_box"])
        self.assertNotIn("unmatched_reason", results[0])

    def test_optimal_assignment_avoids_greedy_face_stealing(self):
        assigned = _optimal_assignment(
            [
                [0.90, 0.80],
                [0.85, 0.20],
            ],
            match_thres=0.55,
        )
        self.assertEqual(assigned[0], (1, 0.80))
        self.assertEqual(assigned[1], (0, 0.85))

    def test_sequential_assignment_discards_all_overlapping_face_boxes(self):
        clip = [
            [0.95, 0.94, 0.10],
            [0.93, 0.92, 0.85],
        ]
        boxes = [
            (100, 100, 220, 240),
            (90, 80, 205, 225),
            (400, 100, 520, 240),
        ]
        assigned, steps = _sequential_overlap_assignment(
            clip,
            clip,
            0.55,
            boxes,
            ambiguity_margin=0.01,
        )
        self.assertEqual(assigned[0], (0, 0.95))
        self.assertEqual(assigned[1], (2, 0.85))
        self.assertEqual(steps[0]["removed_faces"], [0, 1])

    def test_sequential_assignment_recomputes_global_match_after_confirmation(self):
        clip = [
            [0.715, 0.769],  # Alisa
            [0.775, 0.803],  # Maria
        ]
        boxes = [
            (100, 100, 220, 240),
            (400, 100, 520, 240),
        ]
        assigned, steps = _sequential_overlap_assignment(
            clip,
            clip,
            0.55,
            boxes,
            ambiguity_margin=0.01,
        )
        self.assertEqual(steps[0]["row"], 1)
        self.assertEqual(assigned[1], (0, 0.775))
        self.assertEqual(assigned[0], (1, 0.769))

    def test_face_box_overlap_includes_containment_and_partial_intersection(self):
        self.assertTrue(_face_boxes_overlap(
            (100, 100, 220, 240),
            (120, 120, 180, 180),
        ))
        self.assertTrue(_face_boxes_overlap(
            (100, 100, 220, 240),
            (200, 200, 300, 300),
        ))
        self.assertFalse(_face_boxes_overlap(
            (100, 100, 220, 240),
            (220, 100, 300, 240),
        ))

    def test_optimal_assignment_maximizes_total_similarity_at_same_count(self):
        assigned = _optimal_assignment(
            [
                [0.90, 0.80],
                [0.85, 0.70],
            ],
            match_thres=0.55,
        )
        self.assertEqual(assigned[0], (1, 0.80))
        self.assertEqual(assigned[1], (0, 0.85))

    def test_optimal_assignment_can_rank_with_combined_appearance_score(self):
        clip = [
            [0.81, 0.80],
            [0.79, 0.78],
        ]
        combined = [
            [0.80, 0.84],
            [0.86, 0.77],
        ]
        assigned = _optimal_assignment(
            clip,
            match_thres=0.55,
            ranking_scores=combined,
        )
        self.assertEqual(assigned[0], (1, 0.80))
        self.assertEqual(assigned[1], (0, 0.79))

    def test_assignment_ambiguity_gap_detects_near_tie(self):
        clip = [[0.80, 0.79], [0.79, 0.80]]
        ranking = [[0.800, 0.799], [0.799, 0.800]]
        best = _optimal_assignment(clip, 0.55, ranking_scores=ranking)
        gap = _assignment_ambiguity_gap(clip, ranking, 0.55, best)
        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap, 0.001, places=6)

    def test_preview_debug_draws_mask_and_candidate_guides(self):
        base = Image.new("RGBA", (100, 100), (40, 40, 40, 255))
        protected = np.zeros((100, 100), dtype=np.uint8)
        protected[:, :50] = 1
        candidates = [{
            "rect": (55, 10, 90, 35),
            "center": (72.5, 22.5),
            "anchor": (50, 40),
            "face_box": (30, 30, 50, 50),
            "valid": True,
            "selected": True,
        }]
        debugged = _draw_preview_debug(base, protected, candidates, True, True)
        self.assertNotEqual(debugged.getpixel((10, 10)), base.getpixel((10, 10)))
        self.assertNotEqual(debugged.getpixel((70, 20)), base.getpixel((70, 20)))


if __name__ == "__main__":
    unittest.main()
