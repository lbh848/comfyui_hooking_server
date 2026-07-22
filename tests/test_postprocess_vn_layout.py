import io
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops
from unittest.mock import Mock

from modes.postprocess import (
    VN_THEMES,
    _default_vn,
    _draw_colorized_text,
    _load_font,
    _merge_vn_defaults,
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
    def test_gfl_theme_uses_tactical_cyan_and_orange_reference_colors(self):
        self.assertEqual(VN_THEMES["gfl"]["accent"], (135, 231, 231))
        self.assertEqual(VN_THEMES["gfl"]["emotion"], (244, 163, 77))
        self.assertEqual(VN_THEMES["gfl"]["body"], (238, 247, 247))

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

    def test_default_vn_contains_new_theme_and_thumbnail_settings(self):
        settings = _default_vn()

        self.assertFalse(settings["dialogue_color"])
        self.assertEqual(settings["text_outline_width"], -1)
        self.assertNotIn("multi_speaker_layout", settings)
        self.assertEqual(settings["multi_face_mode"], "both")
        self.assertEqual(settings["theme_single"], "sky")
        self.assertEqual(settings["theme_dual"], "sky")

    def test_single_and_dual_theme_are_selected_independently(self):
        settings = {
            "theme": "sky",
            "theme_single": "ivory",
            "theme_dual": "gfl_simple",
        }

        self.assertEqual(_select_vn_theme(settings, 1), ("ivory", False))
        self.assertEqual(_select_vn_theme(settings, 2), ("gfl", True))

    def test_legacy_theme_is_migrated_to_separate_single_and_dual_values(self):
        normal = _merge_vn_defaults({"theme": "lavender"})
        diagonal = _merge_vn_defaults({"theme": "gray_diagonal"})

        self.assertEqual(normal["theme_single"], "lavender")
        self.assertEqual(normal["theme_dual"], "lavender")
        self.assertEqual(diagonal["theme_single"], "gray")
        self.assertEqual(diagonal["theme_dual"], "gray")

    def test_legacy_multi_layout_is_removed_during_merge(self):
        for legacy_layout in ("split", "stack", "diagonal"):
            with self.subTest(layout=legacy_layout):
                merged = _merge_vn_defaults({
                    "theme_dual": "black_diagonal",
                    "multi_speaker_layout": legacy_layout,
                })
                self.assertNotIn("multi_speaker_layout", merged)
                self.assertEqual(merged["theme_dual"], "black")

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

    def test_shared_multi_area_grows_to_fit_additional_dialogue(self):
        short = compose_postprocess(
            _image_bytes(),
            'alice: "첫 번째 대사"\nbob: "두 번째 대사"',
            _settings(theme_dual="sky"),
        )
        long = compose_postprocess(
            _image_bytes(),
            (
                'alice: "첫 번째 대사"\n'
                'bob: "두 번째 대사"\n'
                'alice: "세 번째 대사"\n'
                'bob: "네 번째 대사"'
            ),
            _settings(theme_dual="sky"),
        )

        short_image = Image.open(io.BytesIO(short))
        long_image = Image.open(io.BytesIO(long))
        self.assertGreater(long_image.height, short_image.height)
        self.assertGreater(long_image.height, 480 + 100)

    def test_simple_and_legacy_theme_suffixes_are_resolved(self):
        for base_theme in ("classic", "gfl", "devil", "nikke"):
            with self.subTest(theme=base_theme):
                palette_theme, simple = _resolve_vn_theme(base_theme + "_simple")
                self.assertEqual(palette_theme, base_theme)
                self.assertTrue(simple)
        self.assertEqual(_resolve_vn_theme("sky_simple"), ("sky", False))
        for base_theme in ("sky", "ivory", "lavender", "black", "gray", "classic"):
            with self.subTest(legacy_theme=base_theme):
                palette_theme, simple = _resolve_vn_theme(base_theme + "_diagonal")
                self.assertEqual(palette_theme, base_theme)
                self.assertFalse(simple)

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

    def test_two_thumbnail_mode_combines_both_faces_in_one_left_slot(self):
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
                    theme_dual="classic_simple",
                    multi_face_mode="both",
                ),
            ))).convert("RGB")

        appended = rendered.crop((0, 480, rendered.width, rendered.height))
        left = appended.crop((0, 0, appended.width // 2, appended.height))
        right = appended.crop((appended.width // 2, 0, appended.width, appended.height))
        left_red = sum(
            1 for red, green, blue in left.get_flattened_data()
            if red > 200 and green < 40 and blue < 40
        )
        right_blue = sum(
            1 for red, green, blue in right.get_flattened_data()
            if blue > 200 and red < 40 and green < 40
        )
        self.assertGreater(left_red, 0)
        left_blue = sum(
            1 for red, green, blue in left.get_flattened_data()
            if blue > 200 and red < 40 and green < 40
        )
        self.assertGreater(left_blue, 0)
        self.assertEqual(right_blue, 0)

    def test_multi_face_mode_first_renders_only_one_thumbnail(self):
        faces = {
            "alice": Image.new("RGBA", (100, 100), (255, 0, 0, 255)),
            "bob": Image.new("RGBA", (100, 100), (0, 0, 255, 255)),
        }
        with patch("modes.postprocess._prepare_face_images", return_value=faces), patch(
            "modes.postprocess._paste_face_slot",
        ) as paste_face:
            compose_postprocess(
                _image_bytes(),
                'alice: "첫 번째"\nbob: "두 번째"',
                _settings(
                    face_enabled=True,
                    theme_dual="sky",
                    multi_face_mode="first",
                ),
            )

        self.assertEqual(paste_face.call_count, 1)
        self.assertIs(paste_face.call_args.args[1], faces["alice"])

    def test_simple_theme_skips_panel_and_block_themes_use_one_panel(self):
        with patch("modes.postprocess._draw_multi_panel") as draw_panel:
            compose_postprocess(
                _image_bytes(),
                'alice: "첫 번째"\nbob: "두 번째"',
                _settings(theme_dual="gfl_simple"),
            )
            draw_panel.assert_not_called()

            compose_postprocess(
                _image_bytes(),
                'alice: "첫 번째"\nbob: "두 번째"',
                _settings(theme_dual="gfl"),
            )
            self.assertEqual(draw_panel.call_count, 1)

            compose_postprocess(
                _image_bytes(),
                'alice: "첫 번째"\nbob: "두 번째"',
                _settings(theme_dual="sky"),
            )
            self.assertEqual(draw_panel.call_count, 2)

    def test_simple_layout_draws_one_combined_header_and_original_segment_order(self):
        with patch("modes.postprocess._draw_combined_header", return_value=80) as header, patch(
            "modes.postprocess._draw_segment_group",
        ) as body:
            compose_postprocess(
                _image_bytes(),
                'alice: "첫 번째"\nbob: "두 번째"\nalice: "세 번째"',
                _settings(theme_dual="classic_simple"),
            )

        self.assertEqual(header.call_count, 1)
        self.assertEqual(header.call_args.args[3], ["alice", "bob"])
        self.assertEqual(body.call_count, 1)
        self.assertEqual(
            [segment["text"] for segment in body.call_args.args[3]],
            ["첫 번째", "두 번째", "세 번째"],
        )

    def test_every_special_single_theme_supports_face_slot(self):
        face = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        for theme in ("gfl", "devil", "nikke"):
            with self.subTest(theme=theme), patch(
                "modes.postprocess._prepare_face_images", return_value={"alice": face},
            ), patch("modes.postprocess._paste_face_slot") as paste_face:
                compose_postprocess(
                    _image_bytes(),
                    'alice: "한 명 대사"',
                    _settings(face_enabled=True, theme_single=theme),
                )
                self.assertEqual(paste_face.call_count, 1)
                self.assertIs(paste_face.call_args.args[1], face)

    def test_every_special_theme_supports_overlay_without_canvas_growth(self):
        for theme in ("gfl", "devil", "nikke"):
            with self.subTest(theme=theme):
                rendered = Image.open(io.BytesIO(compose_postprocess(
                    _image_bytes(),
                    'alice: "오버레이 대사"',
                    _settings(
                        placement="overlay", face_enabled=False,
                        theme_single=theme,
                    ),
                )))
                self.assertEqual(rendered.size, (320, 480))

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
                    theme_dual="sky",
                ),
            )

        self.assertEqual(draw_header.call_count, 2)
        first_x = draw_header.call_args_list[0].args[1]
        second_x = draw_header.call_args_list[1].args[1]
        # 하단 글 블록은 좌측 정렬이지만, 짧은 글이면 우측 썸네일 옆으로 이동한다.
        self.assertGreater(second_x, first_x)

    def test_three_speakers_share_one_area_without_clipping(self):
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
                    theme_dual="ivory",
                    name_color=True,
                    dialogue_color=True,
                ),
            ))).convert("RGB")

        self.assertGreater(rendered.height, source.height + 100)
        original_area = rendered.crop((0, 0, source.width, source.height))
        self.assertIsNone(ImageChops.difference(source, original_area).getbbox())


if __name__ == "__main__":
    unittest.main()
