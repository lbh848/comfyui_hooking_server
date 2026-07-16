import io
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw

from modes.postprocess import (
    _default_vn,
    _draw_text_with_backplate,
    _load_font,
    _paste_diagonal_faces,
    compose_postprocess,
)


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
    def test_default_vn_contains_new_layout_and_dialogue_color_settings(self):
        settings = _default_vn()

        self.assertFalse(settings["dialogue_color"])
        self.assertEqual(settings["multi_speaker_layout"], "diagonal")

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

        self.assertEqual(canvas.getpixel((85, 35))[:3], (255, 0, 0))
        self.assertEqual(canvas.getpixel((35, 85))[:3], (0, 0, 255))

    def test_colorized_text_automatically_gets_contrast_background(self):
        image = Image.new("RGBA", (240, 80), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = _load_font(24)

        _draw_text_with_backplate(
            draw, (30, 20), "밝은 이름", font, "#ffffff", True,
        )

        self.assertNotEqual(image.getpixel((27, 27))[:3], (255, 255, 255))

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
                    multi_speaker_layout="diagonal",
                    name_color=True,
                    dialogue_color=True,
                ),
            ))).convert("RGB")

        self.assertGreater(rendered.height, source.height + 100)
        original_area = rendered.crop((0, 0, source.width, source.height))
        self.assertIsNone(ImageChops.difference(source, original_area).getbbox())


if __name__ == "__main__":
    unittest.main()
