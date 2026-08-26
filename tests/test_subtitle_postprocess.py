import copy
import importlib
import io
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import font_assets, postprocess
from modes import subtitle_render
from modes.subtitle_render import compose_subtitle, normalize_subtitle_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
bot_mode_module = importlib.import_module("modes.bot_mode")


def _png_bytes(size=(640, 360), color=(42, 64, 92, 255)) -> bytes:
    image = Image.new("RGBA", size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_subtitle_renderer_hides_speaker_metadata_and_draws_near_bottom(monkeypatch):
    original = _png_bytes()
    drawn_texts = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(self, xy, text, *args, **kwargs):
        drawn_texts.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    monkeypatch.setattr(
        font_assets,
        "load_font",
        lambda _size, _font_id=None, _legacy_path=None: ImageFont.load_default(),
    )

    rendered = compose_subtitle(
        original,
        'INTERNAL_NAME: "Visible subtitle line" #joy',
        {
            "font_id": "system",
            "font_size": 36,
            "min_font_size": 20,
            "outline_width": 2,
            "bottom_margin_ratio": 0.08,
        },
        "test-bot",
    )

    assert rendered != original
    assert drawn_texts == ["Visible subtitle line", "Visible subtitle line"]
    assert all("INTERNAL_NAME" not in text for text in drawn_texts)
    result_image = Image.open(io.BytesIO(rendered)).convert("RGB")
    source_image = Image.open(io.BytesIO(original)).convert("RGB")
    changed = ImageChops.difference(result_image, source_image).getbbox()
    assert changed is not None
    assert changed[1] > source_image.height // 2


def test_subtitle_renderer_respects_disabled_setting():
    original = _png_bytes()

    assert compose_subtitle(
        original,
        'Hana: "표시되지 않아야 해."',
        {"enabled": False},
        "disabled-bot",
    ) == original


def test_single_line_subtitle_ellipsizes_instead_of_creating_two_lines(monkeypatch):
    drawn_texts = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(self, xy, text, *args, **kwargs):
        drawn_texts.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    monkeypatch.setattr(
        font_assets,
        "load_font",
        lambda _size, _font_id=None, _legacy_path=None: ImageFont.load_default(),
    )

    compose_subtitle(
        _png_bytes(size=(180, 120)),
        'Hana: "This sentence is deliberately much too long for one line."',
        {
            "font_id": "system",
            "font_size": 24,
            "min_font_size": 24,
            "max_width_ratio": 0.3,
            "max_lines": 1,
        },
        "one-line-bot",
    )

    assert len(drawn_texts) == 2
    assert all("\n" not in text for text in drawn_texts)
    assert all(text.endswith("…") for text in drawn_texts)


def test_thought_subtitle_uses_subtle_italic_without_visible_parentheses(monkeypatch):
    slants = []
    drawn_texts = []
    original_slant = subtitle_render._slant_layer
    original_text = ImageDraw.ImageDraw.text

    def capture_slant(layer, shear):
        slants.append(shear)
        return original_slant(layer, shear)

    def capture_text(self, xy, text, *args, **kwargs):
        drawn_texts.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(subtitle_render, "_slant_layer", capture_slant)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    monkeypatch.setattr(
        font_assets,
        "load_font",
        lambda _size, _font_id=None, _legacy_path=None: ImageFont.load_default(),
    )

    rendered = compose_subtitle(
        _png_bytes(),
        "Hana: (This is only in my head.)",
        {
            "font_id": "system",
            "thought_italic_enabled": True,
            "thought_italic_shear": 0.10,
        },
        "thought-bot",
    )

    assert rendered
    assert slants == [0.10, 0.10]
    assert drawn_texts == ["This is only in my head.", "This is only in my head."]
    assert all("(" not in text and ")" not in text for text in drawn_texts)


def test_speech_line_stays_upright_when_followed_by_italic_thought(monkeypatch):
    slants = []

    monkeypatch.setattr(
        subtitle_render,
        "_slant_layer",
        lambda layer, shear: slants.append(shear) or layer,
    )
    monkeypatch.setattr(
        font_assets,
        "load_font",
        lambda _size, _font_id=None, _legacy_path=None: ImageFont.load_default(),
    )

    compose_subtitle(
        _png_bytes(),
        'Hana: "I said it aloud."\nHana: (But this stays inside.)',
        {"font_id": "system", "thought_italic_enabled": True},
        "mixed-bot",
    )

    # 그림자와 전경 중 생각 줄에만 각각 한 번씩 적용된다.
    assert slants == [0.10, 0.10]


def test_subtitle_settings_clamp_to_broadcast_safe_bounds():
    normalized = normalize_subtitle_settings({
        "font_size": 8,
        "min_font_size": 999,
        "max_width_ratio": 5,
        "bottom_margin_ratio": -1,
        "outline_width": 99,
        "shadow_opacity": -2,
        "max_lines": 9,
    })

    assert normalized["font_size"] == 12
    assert normalized["min_font_size"] == 12
    assert normalized["max_width_ratio"] == 0.96
    assert normalized["bottom_margin_ratio"] == 0.02
    assert normalized["outline_width"] == 20
    assert normalized["shadow_opacity"] == 0.0
    assert normalized["thought_italic_enabled"] is True
    assert normalized["thought_italic_shear"] == 0.10
    assert normalized["max_lines"] == 2


def test_subtitle_settings_persist_with_mode_without_touching_real_bot_data(monkeypatch):
    state = {"bots": [{"name": "anime-bot", "postprocess_mode": "vn"}]}
    saved = []

    monkeypatch.setattr(bot_mode_module, "_load_bot_data", lambda: state)
    monkeypatch.setattr(
        bot_mode_module,
        "_save_bot_data",
        lambda value: saved.append(copy.deepcopy(value)),
    )

    bot_mode_module._save_postprocess_subtitle(
        "anime-bot",
        {"font_size": 60, "max_lines": 1},
        mode="subtitle",
    )

    assert len(saved) == 1
    stored = saved[0]["bots"][0]
    assert stored["postprocess_mode"] == "subtitle"
    assert stored["postprocess_subtitle"]["font_size"] == 60
    assert stored["postprocess_subtitle"]["max_lines"] == 1


def test_get_subtitle_settings_honors_global_and_bot_toggles(monkeypatch):
    monkeypatch.setattr(
        postprocess,
        "_load_bot_subtitle",
        lambda _bot: {"enabled": False, "font_size": 61},
    )

    assert postprocess.get_subtitle_settings(
        {"postprocess_enabled": True}, bot_name="anime-bot"
    ) is None
    forced = postprocess.get_subtitle_settings(
        {"postprocess_enabled": False}, bot_name="anime-bot", force=True
    )
    assert forced is not None
    assert forced["font_size"] == 61


def test_subtitle_mode_is_registered_in_backend_frontend_and_routes():
    server_source = (PROJECT_ROOT / "server.py").read_text(encoding="utf-8")
    frontend_source = (PROJECT_ROOT / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    assert '"illustration_call3_subtitle": _llm_route_defaults()' in server_source
    assert server_source.count('"/api/bot_mode/postprocess_subtitle"') == 2
    assert "key: 'illustration_call3_subtitle'" in frontend_source
    assert "switchPostprocessTab('subtitle')" in frontend_source
    assert "call3_prompt_mode: 'subtitle'" not in frontend_source
    assert "subtitle: 'subtitle'" in frontend_source
    assert "mode: 'subtitle'" in frontend_source
    assert 'id="pp-subtitle-thought-italic-enabled"' in frontend_source
