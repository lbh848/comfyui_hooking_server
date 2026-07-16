import io
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw
from unittest.mock import Mock

from modes.postprocess import (
    _default_vn,
    _draw_colorized_text,
    _draw_combined_header,
    _load_font,
    _merge_vn_defaults,
    _paste_diagonal_faces,
    _resolve_vn_theme,
    _select_vn_theme,
    compose_postprocess,
)
from modes import face_detector


def _image_bytes(size=(320, 480), color=(60, 60, 90)):
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _settings(**overrides):
    settings = _default_vn()
    settings.update({
        "placement": "extend",
        "height_mode": "px",
        "height_value": 100,
        "font_size": 22,
        "name_font_size": 26,
        "emotion_font_size": 20,
        "face_enabled": False,
        "theme": "sky",
        "opacity": 100,
    })
    settings.update(overrides)
    return settings


class PostprocessVnLayoutTests(unittest.TestCase):
    def test_face_crop_returns_actual_center_in_final_square(self):
        image = Image.new("RGB", (100, 100), (80, 80, 80))
        with patch("modes.face_detector._preferred_session", return_value=object()), patch(
            "modes.face_detector._detect",
            return_value=((20.0, 10.0, 60.0, 50.0), 0.9),
        ):
            crop, confidence, center = face_detector.crop_face(
                image, top_mult=1.8, bottom_mult=1.0, target_size=80,
                return_conf=True, return_center=True,
            )

        self.assertEqual(crop.size, (80, 80))
        self.assertEqual(confidence, 0.9)
        self.assertAlmostEqual(center[0], 0.5, places=2)
        self.assertAlmostEqual(center[1], 0.6, places=2)

    def test_default_vn_contains_new_layout_and_dialogue_color_settings(self):
        settings = _default_vn()

        self.assertFalse(settings["dialogue_color"])
        self.assertEqual(settings["text_outline_width"], -1)
        self.assertEqual(settings["multi_speaker_layout"], "split")
        self.assertEqual(settings["theme_single"], "sky")
        self.assertEqual(settings["theme_dual"], "sky_diagonal")

    def test_single_and_dual_theme_are_selected_independently(self):
        settings = {
            "theme": "sky",
            "theme_single": "ivory",
            "theme_dual": "black_diagonal",
        }

        self.assertEqual(_select_vn_theme(settings, 1), ("ivory", False))
        self.assertEqual(_select_vn_theme(settings, 2), ("black", True))

    def test_legacy_theme_is_migrated_to_separate_single_and_dual_values(self):
        normal = _merge_vn_defaults({"theme": "lavender"})
        diagonal = _merge_vn_defaults({"theme": "gray_diagonal"})

        self.assertEqual(normal["theme_single"], "lavender")
        self.assertEqual(normal["theme_dual"], "lavender_diagonal")
        self.assertEqual(diagonal["theme_single"], "gray")
        self.assertEqual(diagonal["theme_dual"], "gray_diagonal")

    def test_extend_card_does_not_modify_original_image_area(self):
        source = Image.open(io.BytesIO(_image_bytes())).convert("RGB")
        rendered = Image.open(io.BytesIO(compose_postprocess(
            _image_bytes(),
            'alice: "하단 확장 카드가 원본 위로 올라오면 안 됩니다."',
            _settings(),
        ))).convert("RGB")

        self.assertGreater(rendered.height, source.height + 100)
        original_area = rendered.crop((0, 0, source.width, source.height))
        self.assertIsNone(ImageChops.difference(source, original_area).getbbox())

    def test_stack_layout_grows_with_every_dialogue_box(self):
        short = compose_postprocess(
            _image_bytes(),
            'alice: "첫 번째 대사"\nbob: "두 번째 대사"',
            _settings(multi_speaker_layout="stack"),
        )
        long = compose_postprocess(
            _image_bytes(),
            (
                'alice: "첫 번째 대사"\n'
                'bob: "두 번째 대사"\n'
                'alice: "세 번째 대사"\n'
                'bob: "네 번째 대사"'
            ),
            _settings(multi_speaker_layout="stack"),
        )

        short_image = Image.open(io.BytesIO(short))
        long_image = Image.open(io.BytesIO(long))
        self.assertGreater(long_image.height, short_image.height)
        self.assertGreater(long_image.height, 480 + 100)

    def test_diagonal_thumbnail_contains_both_triangles(self):
        canvas = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        first = Image.new("RGBA", (80, 80), (255, 0, 0, 255))
        second = Image.new("RGBA", (80, 80), (0, 0, 255, 255))

        _paste_diagonal_faces(
            canvas, first, second, (20, 20, 100, 100), None,
        )

        # '/' 대각선 기준 인물 1은 좌상단, 인물 2는 우하단에 배정.
        self.assertEqual(canvas.getpixel((35, 35))[:3], (255, 0, 0))
        self.assertEqual(canvas.getpixel((85, 85))[:3], (0, 0, 255))

    def test_diagonal_thumbnail_moves_each_face_center_to_triangle_centroid(self):
        canvas = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        first = Image.new("RGBA", (80, 80), (40, 40, 40, 255))
        second = Image.new("RGBA", (80, 80), (40, 40, 40, 255))
        # 실제 얼굴 중심이 크롭 이미지 중앙보다 아래에 있는 경우를 재현.
        ImageDraw.Draw(first).ellipse((34, 46, 46, 58), fill=(255, 0, 0, 255))
        ImageDraw.Draw(second).ellipse((34, 46, 46, 58), fill=(0, 0, 255, 255))
        first.info["postprocess_face_center"] = (0.5, 0.65)
        second.info["postprocess_face_center"] = (0.5, 0.65)

        _paste_diagonal_faces(
            canvas, first, second, (20, 20, 100, 100), None,
        )

        # 80px 삼각형 무게중심: 좌상단=(47,47), 우하단=(73,73), 박스 오프셋 포함.
        self.assertGreater(canvas.getpixel((47, 47))[0], 180)
        self.assertGreater(canvas.getpixel((73, 73))[2], 180)

    def test_diagonal_header_uses_slash_separator(self):
        draw = Mock()
        draw.textlength.side_effect = lambda text, font=None: len(text) * 10
        font = _load_font(24)

        _draw_combined_header(
            draw, 10, 10, ["alice", "bob"], {}, False, False, "",
            font, "#ffffff",
        )

        separator_calls = [
            call for call in draw.text.call_args_list
            if len(call.args) >= 2 and call.args[1] == " / "
        ]
        self.assertEqual(len(separator_calls), 1)
        self.assertFalse(any(
            len(call.args) >= 2 and "*" in str(call.args[1])
            for call in draw.text.call_args_list
        ))

    def test_every_palette_supports_diagonal_theme_variant(self):
        for base_theme in ("sky", "ivory", "lavender", "black", "gray", "classic"):
            with self.subTest(theme=base_theme):
                palette_theme, diagonal = _resolve_vn_theme(base_theme + "_diagonal")
                self.assertEqual(palette_theme, base_theme)
                self.assertTrue(diagonal)

    def test_colorized_text_uses_glyph_outline_not_rectangle_background(self):
        draw = Mock()
        draw.textlength.return_value = 100.0
        font = _load_font(24)

        _draw_colorized_text(
            draw, (30, 20), "밝은 이름", font, "#ffffff", True,
        )

        draw.rounded_rectangle.assert_not_called()
        draw.text.assert_called_once()
        kwargs = draw.text.call_args.kwargs
        self.assertGreater(kwargs["stroke_width"], 0)
        self.assertEqual(kwargs["stroke_fill"], (0, 0, 0, 255))

        draw.reset_mock()
        draw.textlength.return_value = 100.0
        _draw_colorized_text(
            draw, (30, 20), "고정 두께", font, "#ffffff", True, 9,
        )
        self.assertEqual(draw.text.call_args.kwargs["stroke_width"], 9)

        draw.reset_mock()
        draw.textlength.return_value = 100.0
        _draw_colorized_text(
            draw, (30, 20), "외곽 없음", font, "#ffffff", True, 0,
        )
        self.assertEqual(draw.text.call_args.kwargs["stroke_width"], 0)

    def test_diagonal_thumbnail_is_placed_on_left_side_of_card(self):
        faces = {
            "alice": Image.new("RGBA", (100, 100), (255, 0, 0, 255)),
            "bob": Image.new("RGBA", (100, 100), (0, 0, 255, 255)),
        }
        with patch("modes.postprocess._prepare_face_images", return_value=faces):
            rendered = Image.open(io.BytesIO(compose_postprocess(
                _image_bytes(),
                'alice: "첫 번째"\nbob: "두 번째"',
                _settings(
                    face_enabled=True,
                    theme="sky_diagonal",
                    multi_speaker_layout="split",
                ),
            ))).convert("RGB")

        appended = rendered.crop((0, 480, rendered.width, rendered.height))
        left = appended.crop((0, 0, appended.width // 2, appended.height))
        right = appended.crop((appended.width // 2, 0, appended.width, appended.height))
        left_colored = sum(
            1 for red, green, blue in left.get_flattened_data()
            if (red > 200 and green < 40 and blue < 40)
            or (blue > 200 and red < 40 and green < 40)
        )
        right_colored = sum(
            1 for red, green, blue in right.get_flattened_data()
            if (red > 200 and green < 40 and blue < 40)
            or (blue > 200 and red < 40 and green < 40)
        )
        self.assertGreater(left_colored, right_colored)

    def test_split_bottom_right_text_block_stays_left_aligned_near_thumbnail(self):
        faces = {
            "alice": Image.new("RGBA", (100, 100), (255, 0, 0, 255)),
            "bob": Image.new("RGBA", (100, 100), (0, 0, 255, 255)),
        }
        with patch("modes.postprocess._prepare_face_images", return_value=faces), patch(
            "modes.postprocess._draw_speaker_header", return_value=24,
        ) as draw_header:
            compose_postprocess(
                _image_bytes(),
                'alice: "위"\nbob: "아래"',
                _settings(
                    face_enabled=True,
                    theme="sky",
                    theme_dual="sky",
                    multi_speaker_layout="split",
                ),
            )

        self.assertEqual(draw_header.call_count, 2)
        first_x = draw_header.call_args_list[0].args[1]
        second_x = draw_header.call_args_list[1].args[1]
        # 하단 글 블록은 좌측 정렬이지만, 짧은 글이면 우측 썸네일 옆으로 이동한다.
        self.assertGreater(second_x, first_x)

    def test_three_speakers_fall_back_from_diagonal_without_clipping(self):
        source = Image.open(io.BytesIO(_image_bytes())).convert("RGB")
        with patch(
            "modes.postprocess.resolve_name_color",
            side_effect=lambda speaker, _bot: {
                "alice": "#ff9ec4", "bob": "#7ab8ff", "carol": "#86e08a",
            }.get(speaker, "#ffffff"),
        ):
            rendered = Image.open(io.BytesIO(compose_postprocess(
                _image_bytes(),
                'alice: "하나"\nbob: "둘"\ncarol: "셋"',
                _settings(
                    theme="ivory_diagonal",
                    multi_speaker_layout="split",
                    name_color=True,
                    dialogue_color=True,
                ),
            ))).convert("RGB")

        self.assertGreater(rendered.height, source.height + 100)
        original_area = rendered.crop((0, 0, source.width, source.height))
        self.assertIsNone(ImageChops.difference(source, original_area).getbbox())


if __name__ == "__main__":
    unittest.main()
