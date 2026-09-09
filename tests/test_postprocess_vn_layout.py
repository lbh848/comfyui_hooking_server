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

        self.assertTrue(settings["dialogue_color"])
        self.assertEqual(settings["text_outline_width"], -1)
        self.assertNotIn("multi_speaker_layout", settings)
        self.assertEqual(settings["multi_face_mode"], "both")
        self.assertEqual(settings["theme_single"], "classic")
        self.assertEqual(settings["theme_dual"], "classic_simple")

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

    def test_two_thumbnail_diagonal_slot_uses_no_extra_face_zoom(self):
        first = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
        for x in range(100):
            color = (round(255 * x / 99), 0, 0, 255)
            for y in range(100):
                first.putpixel((x, y), color)
        second = Image.new("RGBA", (100, 100), (0, 0, 255, 255))
        canvas = Image.new("RGBA", (120, 120), (0, 0, 0, 0))

        _paste_diagonal_faces(
            canvas, first, second, (0, 0, 120, 120), None,
        )

        # 1.0배 FACE CROP의 바깥 영역이 유지되고, 중심 이동으로 비는 끝부분은
        # 의도한 슬롯 배경색으로 남아야 한다.
        red, green, blue, alpha = canvas.getpixel((95, 10))
        self.assertGreater(red, 235)
        self.assertLess(green, 10)
        self.assertLess(blue, 10)
        self.assertEqual(alpha, 255)
        self.assertEqual(canvas.getpixel((100, 10)), (58, 62, 82, 255))

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


    def test_classic_overlay_opacity_lightens_bar(self):
        """1인 검정 심플(classic) overlay 모드는 opacity 슬라이더를 반영해야 한다.

        회귀: 예전에는 overlay 알파가 170으로 고정돼 opacity를 무시했다.
        opacity가 낮을수록 바가 얇아져 원본 이미지가 더 비쳐야 한다.
        """
        text = 'alice: "반투명도가 이미지에 반영되어야 합니다."'
        full = Image.open(io.BytesIO(compose_postprocess(
            _image_bytes(),
            text,
            _settings(theme_single="classic", theme="classic", placement="overlay", opacity=100),
        ))).convert("RGB")
        half = Image.open(io.BytesIO(compose_postprocess(
            _image_bytes(),
            text,
            _settings(theme_single="classic", theme="classic", placement="overlay", opacity=50),
        ))).convert("RGB")

        # 바 영역(하단)을 비교: opacity=50이 100보다 원본(밝은 회색)이 더 비쳐 밝아야 한다.
        from PIL import ImageStat
        bar_box = (0, full.height - 80, full.width, full.height)
        full_mean = ImageStat.Stat(full.crop(bar_box).convert("L")).mean[0]
        half_mean = ImageStat.Stat(half.crop(bar_box).convert("L")).mean[0]
        self.assertGreater(half_mean, full_mean,
                            f"opacity=50 바가 100보다 밝아야 함(half={half_mean}, full={full_mean})")

    def test_classic_extend_unchanged_by_opacity(self):
        """1인 검정 심플 extend 모드는 검정 위 검정이라 opacity와 무관하게 동일해야 한다."""
        text = 'alice: "extend는 opacity와 무관합니다."'
        base = compose_postprocess(
            _image_bytes(),
            text,
            _settings(theme_single="classic", theme="classic", placement="extend", opacity=100),
        )
        low = compose_postprocess(
            _image_bytes(),
            text,
            _settings(theme_single="classic", theme="classic", placement="extend", opacity=30),
        )
        self.assertEqual(base, low)


    def test_devil_gradient_bottom_is_darkest(self):
        """소악마(devil) 패널은 상단(10,8,12)→하단(0,0,0) 세로 그라데이션이다.
        따라서 opacity=100일 때 패널 하단이 상단보다 어두워야 한다.

        측정은 반드시 '패널 내부의 텍스트 없는 좌측 열'에서만 해야 한다.
        - 좌측 x<8 은 content_x(=x1+pad) 왼쪽이라 본문/이름/악센트 바가 닿지 않는다.
        - 패널 아래쪽에는 원본 배경(60,60,90 → L=63)이 그대로 남아 있어,
          고정 crop 을 쓰면 그라데이션이 아니라 배경을 재는 사고가 난다.
        그래서 패널 세로 범위를 픽셀에서 직접 찾아낸 뒤 비교한다.
        """
        from PIL import ImageStat
        rendered = Image.open(io.BytesIO(compose_postprocess(
            _image_bytes(),
            'alice: "하단이 상단보다 어두워야 합니다."',
            _settings(theme_single="devil", theme="devil", placement="overlay",
                      face_enabled=False, opacity=100),
        ))).convert("RGB")
        w, h = rendered.size

        # 텍스트가 절대 닿지 않는 좌측 열만 사용한다.
        stripe = rendered.crop((0, 0, 8, h)).convert("L")
        background_l = 63  # _image_bytes() 기본색 (60,60,90) 의 L 값
        row_means = [
            ImageStat.Stat(stripe.crop((0, y, 8, y + 1))).mean[0]
            for y in range(h)
        ]
        # 패널은 배경보다 확연히 어둡다. 배경으로 남은 행을 제외해 패널 범위를 얻는다.
        panel_rows = [y for y, m in enumerate(row_means) if m < background_l - 20]
        self.assertTrue(panel_rows, "devil 패널을 찾지 못했습니다")
        panel_top, panel_bottom = panel_rows[0], panel_rows[-1]
        panel_h = panel_bottom - panel_top + 1
        self.assertGreater(panel_h, 40, f"패널이 너무 얇아 그라데이션 측정 불가: {panel_h}px")

        band = max(4, panel_h // 5)
        top_mean = sum(row_means[panel_top:panel_top + band]) / band
        bot_mean = sum(row_means[panel_bottom - band + 1:panel_bottom + 1]) / band
        self.assertLess(bot_mean, top_mean,
                        f"devil 하단({bot_mean})이 상단({top_mean})보다 어두워야 함 "
                        f"(패널 y={panel_top}~{panel_bottom})")

    def test_devil_opacity_scales_gradient_uniformly(self):
        """사용자 opacity는 devil 그라데이션 전체에 동등 곱해져 하단도 옅어진다."""
        from PIL import ImageStat
        full = Image.open(io.BytesIO(compose_postprocess(
            _image_bytes(),
            'alice: "opacity 100."',
            _settings(theme_single="devil", theme="devil", placement="overlay",
                      face_enabled=False, opacity=100),
        ))).convert("RGB")
        half = Image.open(io.BytesIO(compose_postprocess(
            _image_bytes(),
            'alice: "opacity 50."',
            _settings(theme_single="devil", theme="devil", placement="overlay",
                      face_enabled=False, opacity=50),
        ))).convert("RGB")
        h = full.size[1]
        full_bot = ImageStat.Stat(full.crop((0, h - 25, 40, h)).convert("L")).mean[0]
        half_bot = ImageStat.Stat(half.crop((0, h - 25, 40, h)).convert("L")).mean[0]
        self.assertGreater(half_bot, full_bot,
                           f"opacity=50 하단({half_bot})이 100({full_bot})보다 밝아야(옅어야) 함")


if __name__ == "__main__":
    unittest.main()
