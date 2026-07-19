"""text-safe organic 박스 시각 회귀 스크립트.

legacy organic ellipse(_build_organic_body_contour 방식)와 신규 text-safe rounded box를
같은 content/text 조건에서 나란히 렌더해 요구사항/ 폴더에 저장한다.
미리보기/실제 전송이 동일 빌더를 쓰는지만 확인(외부 API 호출 없음).

실행: uv run python tests/visual_bubble_text_safe.py
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modes.bubble_render import _draw_layout_bubble
from modes.bubble_types import TextBoxShapeConfig
from modes.bubble_shape import estimate_envelope_extra


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "요구사항")
os.makedirs(OUT_DIR, exist_ok=True)

# (이름, 텍스트 줄, font_size, padding_x, padding_y)
# padding 은 em 기반이 아니라 px 고정(시각 비교 단순화).
CASES = [
    ("01_short_2line", ["안녕!"], 30, 14, 10),
    ("02_normal", ["그래, 오늘 날씨 좋네.", "산책 갈래?"], 32, 35, 22),
    ("03_wide", ["이건 가로로 아주 긴 한 줄짜리 대사입니다"], 30, 30, 18),
    ("04_tall_5line", ["하나", "둘", "셋", "넷", "다섯 줄까지"], 30, 22, 18),
    ("05_first_long", ["이것은 아주 긴 첫 줄입니다", "짧"], 30, 30, 18),
    ("06_last_short", ["첫줄", "두번째줄", "끝"], 30, 22, 18),
    ("07_mixed_lang", ["Hello 안녕 KL", "moon 달 123"], 30, 26, 18),
    ("08_ellipsis", ["잠깐만...", "이게 맞나...?"], 30, 26, 18),
]


def _measure_text(lines, font):
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    widths = [draw.textlength(line, font=font) for line in lines]
    ascent, descent = font.getmetrics()
    text_w = max(widths) if widths else 0.0
    text_h = (ascent + descent) * len(lines)
    return float(text_w), float(text_h)


def _load_font(size):
    for path in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _render_legacy(canvas, rect, anchor, font, lines, text_w, text_h):
    """구 organic 경로 재현: ellipse 중심+반지름 + 덧셈 꼬리. text-safe 검증 없음."""
    overlay = Image.new("RGBA", canvas, (0, 0, 0, 0))
    fill = (250, 250, 247, 255)
    border = (24, 24, 24, 255)
    _draw_layout_bubble(
        overlay, rect, anchor, "ellipse", fill, border, 2.0, 18, True,
        organic=False,  # legacy 타원 강제(text-safe 경로 사용 안 함)
        tail_width_scale=1.0, wobble=0.05, point_count=180, seed=77,
    )
    _draw_centered_text(overlay, rect, font, lines)
    return overlay


def _render_text_safe(canvas, rect, anchor, font, lines, text_w, text_h, pad_x, pad_y):
    overlay = Image.new("RGBA", canvas, (0, 0, 0, 0))
    fill = (250, 250, 247, 255)
    border = (24, 24, 24, 255)
    _draw_layout_bubble(
        overlay, rect, anchor, "ellipse", fill, border, 2.0, 18, True,
        organic=True, tail_width_scale=1.0, wobble=0.05, point_count=220, seed=77,
        text_w=text_w, text_h=text_h, font_size=font.size if hasattr(font, "size") else 30,
        padding_x=pad_x, padding_y=pad_y, line_count=len(lines),
    )
    _draw_centered_text(overlay, rect, font, lines)
    return overlay


def _draw_centered_text(overlay, rect, font, lines):
    draw = ImageDraw.Draw(overlay)
    text = "\n".join(lines)
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tx = cx - (bbox[0] + bbox[2]) / 2.0
    ty = cy - (bbox[1] + bbox[3]) / 2.0
    draw.multiline_text((tx, ty), text, font=font, fill=(24, 24, 24, 255), align="center")


def main():
    canvas = (1000, 700)
    for name, lines, fs, pad_x, pad_y in CASES:
        font = _load_font(fs)
        text_w, text_h = _measure_text(lines, font)
        # content bubble = text + padding; envelope = content + extra(text-safe 용).
        content_w = text_w + 2 * pad_x
        content_h = text_h + 2 * pad_y
        guard, bm, maxo = estimate_envelope_extra(min(content_w, content_h), 2, TextBoxShapeConfig())
        extra = guard + bm + maxo
        env_w = content_w + 2 * extra
        env_h = content_h + 2 * extra

        rect_ts = (140, 120, int(round(140 + env_w)), int(round(120 + env_h)))
        rect_legacy = (140, 120, int(round(140 + content_w)), int(round(120 + content_h)))
        anchor = (rect_ts[0] + (rect_ts[2] - rect_ts[0]) // 2, rect_ts[3] + 90)

        legacy = _render_legacy(canvas, rect_legacy, anchor, font, lines, text_w, text_h)
        ts = _render_text_safe(canvas, rect_ts, anchor, font, lines, text_w, text_h, pad_x, pad_y)

        # 나란히 합치기
        gap = 40
        combined = Image.new("RGBA", (canvas[0] * 2 + gap, canvas[1]), (235, 235, 232, 255))
        combined.paste(legacy, (0, 0))
        combined.paste(ts, (canvas[0] + gap, 0))
        draw = ImageDraw.Draw(combined)
        draw.text((10, 8), f"{name}  |  LEFT=legacy ellipse   RIGHT=text-safe organic",
                  fill=(20, 20, 20, 255))
        out = os.path.join(OUT_DIR, f"visual_{name}.png")
        combined.convert("RGB").save(out)
        print(f"saved {out}  (text={text_w:.0f}x{text_h:.0f}, env={env_w:.0f}x{env_h:.0f})")


if __name__ == "__main__":
    main()
