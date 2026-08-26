"""애니메이션 방송 자막 스타일의 삽화 후처리 렌더러."""

from __future__ import annotations

import io
import math
import re
import traceback
from typing import Iterable

from PIL import Image, ImageColor, ImageDraw, ImageFilter


DEFAULT_SUBTITLE_SETTINGS = {
    "enabled": True,
    "font_id": "noto-sans-kr-medium",
    "font_path": "",
    "font_size": 52,
    "min_font_size": 28,
    "max_width_ratio": 0.86,
    "bottom_margin_ratio": 0.075,
    "line_spacing_ratio": 0.18,
    "text_color": "#FFFFFF",
    "outline_color": "#101010",
    "outline_width": 4,
    "shadow_color": "#000000",
    "shadow_opacity": 0.82,
    "shadow_offset_x": 2,
    "shadow_offset_y": 3,
    "max_lines": 2,
}


def _number(
    value: object,
    default: float,
    minimum: float,
    maximum: float,
    field: str,
) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError("bool 값은 숫자로 허용되지 않음")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("유한한 숫자가 아님")
        return max(minimum, min(maximum, parsed))
    except (TypeError, ValueError, OverflowError) as exc:
        print(
            f"[SUBTITLE:CONFIG] 숫자 설정 검증 실패, 기본값 사용: "
            f"field={field}, value={value!r}, default={default}, error={exc}"
        )
        traceback.print_exc()
        return float(default)


def _color(value: object, default: str, field: str) -> str:
    candidate = str(value or default).strip() or default
    try:
        ImageColor.getrgb(candidate)
        return candidate
    except Exception as exc:
        print(
            f"[SUBTITLE:CONFIG] 색상 설정 검증 실패, 기본값 사용: "
            f"field={field}, value={value!r}, default={default}, error={exc}"
        )
        traceback.print_exc()
        return default


def normalize_subtitle_settings(raw: object) -> dict:
    """저장값/요청값을 렌더 가능한 자막 설정으로 정규화한다."""
    if raw is None:
        source = {}
    elif isinstance(raw, dict):
        source = raw
    else:
        print(
            "[SUBTITLE:CONFIG] 자막 설정이 객체가 아니어서 기본값 사용: "
            f"type={type(raw).__name__}, value={raw!r}"
        )
        source = {}

    normalized = dict(DEFAULT_SUBTITLE_SETTINGS)
    enabled = source.get("enabled", normalized["enabled"])
    if not isinstance(enabled, bool):
        print(
            "[SUBTITLE:CONFIG] enabled가 bool이 아니어서 기본값 사용: "
            f"value={enabled!r}"
        )
        enabled = normalized["enabled"]
    normalized["enabled"] = enabled
    normalized["font_id"] = str(
        source.get("font_id", normalized["font_id"]) or normalized["font_id"]
    ).strip()
    normalized["font_path"] = str(source.get("font_path", "") or "").strip()
    normalized["font_size"] = int(round(_number(
        source.get("font_size", normalized["font_size"]),
        normalized["font_size"], 12, 400, "font_size",
    )))
    normalized["min_font_size"] = int(round(_number(
        source.get("min_font_size", normalized["min_font_size"]),
        normalized["min_font_size"], 10, 400, "min_font_size",
    )))
    if normalized["min_font_size"] > normalized["font_size"]:
        print(
            "[SUBTITLE:CONFIG] 최소 폰트가 기본 폰트보다 커서 동일 크기로 제한: "
            f"min={normalized['min_font_size']}, font={normalized['font_size']}"
        )
        normalized["min_font_size"] = normalized["font_size"]
    normalized["max_width_ratio"] = _number(
        source.get("max_width_ratio", normalized["max_width_ratio"]),
        normalized["max_width_ratio"], 0.30, 0.96, "max_width_ratio",
    )
    normalized["bottom_margin_ratio"] = _number(
        source.get("bottom_margin_ratio", normalized["bottom_margin_ratio"]),
        normalized["bottom_margin_ratio"], 0.02, 0.30, "bottom_margin_ratio",
    )
    normalized["line_spacing_ratio"] = _number(
        source.get("line_spacing_ratio", normalized["line_spacing_ratio"]),
        normalized["line_spacing_ratio"], 0.0, 0.80, "line_spacing_ratio",
    )
    normalized["outline_width"] = int(round(_number(
        source.get("outline_width", normalized["outline_width"]),
        normalized["outline_width"], 0, 20, "outline_width",
    )))
    normalized["shadow_opacity"] = _number(
        source.get("shadow_opacity", normalized["shadow_opacity"]),
        normalized["shadow_opacity"], 0.0, 1.0, "shadow_opacity",
    )
    normalized["shadow_offset_x"] = int(round(_number(
        source.get("shadow_offset_x", normalized["shadow_offset_x"]),
        normalized["shadow_offset_x"], -30, 30, "shadow_offset_x",
    )))
    normalized["shadow_offset_y"] = int(round(_number(
        source.get("shadow_offset_y", normalized["shadow_offset_y"]),
        normalized["shadow_offset_y"], -30, 30, "shadow_offset_y",
    )))
    normalized["max_lines"] = int(round(_number(
        source.get("max_lines", normalized["max_lines"]),
        normalized["max_lines"], 1, 2, "max_lines",
    )))
    for field in ("text_color", "outline_color", "shadow_color"):
        normalized[field] = _color(
            source.get(field, normalized[field]), normalized[field], field
        )
    return normalized


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    try:
        return float(draw.textlength(text, font=font))
    except Exception as exc:
        print(f"[SUBTITLE] 텍스트 폭 측정 실패: text={text!r}, error={exc}")
        traceback.print_exc()
        box = draw.textbbox((0, 0), text, font=font)
        return float(max(0, box[2] - box[0]))


def _candidate_breaks(text: str) -> list[int]:
    breaks = {
        match.end()
        for match in re.finditer(r"\s+", text)
        if 0 < match.end() < len(text)
    }
    if not breaks:
        breaks.update(range(1, len(text)))
    return sorted(breaks)


def _balanced_two_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: float,
) -> list[str] | None:
    best: tuple[float, list[str]] | None = None
    for position in _candidate_breaks(text):
        left = text[:position].strip()
        right = text[position:].strip()
        if not left or not right:
            continue
        left_width = _text_width(draw, left, font)
        right_width = _text_width(draw, right, font)
        widest = max(left_width, right_width)
        if widest > max_width:
            continue
        score = widest + abs(left_width - right_width) * 0.38
        if best is None or score < best[0]:
            best = (score, [left, right])
    return best[1] if best else None


def _layout_lines(
    draw: ImageDraw.ImageDraw,
    texts: Iterable[str],
    font,
    max_width: float,
    max_lines: int,
) -> list[str] | None:
    values = [re.sub(r"\s+", " ", str(value or "")).strip() for value in texts]
    values = [value for value in values if value]
    if not values:
        return []
    if len(values) > max_lines:
        return None
    if len(values) > 1:
        if all(_text_width(draw, value, font) <= max_width for value in values):
            return values
        return None
    value = values[0]
    if _text_width(draw, value, font) <= max_width:
        return [value]
    if max_lines == 1:
        return None
    return _balanced_two_lines(draw, value, font, max_width)


def _ellipsize(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: float,
) -> str:
    ellipsis = "…"
    value = str(text or "").strip()
    while value and _text_width(draw, value + ellipsis, font) > max_width:
        value = value[:-1].rstrip()
    return value + ellipsis if value else ellipsis


def _fit_subtitle(
    draw: ImageDraw.ImageDraw,
    texts: list[str],
    settings: dict,
    max_width: float,
) -> tuple[object, list[str], int]:
    from modes.font_assets import load_font

    start_size = int(settings["font_size"])
    min_size = int(settings["min_font_size"])
    for size in range(start_size, min_size - 1, -1):
        font = load_font(size, settings["font_id"], settings["font_path"])
        lines = _layout_lines(
            draw, texts, font, max_width, int(settings["max_lines"])
        )
        if lines is not None:
            return font, lines, size

    font = load_font(min_size, settings["font_id"], settings["font_path"])
    joined = " ".join(texts)
    max_lines = int(settings["max_lines"])
    if max_lines == 1:
        lines = [_ellipsize(draw, joined, font, max_width)]
    elif len(texts) > 1:
        # 즉시 이어지는 두 발화는 줄마다 한 발화씩 유지한다. 두 문장을 한 문장처럼
        # 재분할하면 방송 자막의 발화 교대 리듬과 응답 경계가 사라진다.
        lines = [
            value if _text_width(draw, value, font) <= max_width
            else _ellipsize(draw, value, font, max_width)
            for value in texts[:2]
        ]
    else:
        lines = _balanced_two_lines(draw, joined, font, max_width)
    if lines is None:
        midpoint = max(1, len(joined) // 2)
        split = min(
            _candidate_breaks(joined),
            key=lambda value: abs(value - midpoint),
            default=midpoint,
        )
        lines = [joined[:split].strip(), joined[split:].strip()]
    lines = [
        line if _text_width(draw, line, font) <= max_width
        else _ellipsize(draw, line, font, max_width)
        for line in lines[:max_lines]
        if line
    ]
    print(
        "[SUBTITLE] 자막이 안전 폭에 들어가지 않아 최소 폰트/말줄임 적용: "
        f"font_size={min_size}, max_width={max_width:.1f}, text={joined!r}"
    )
    return font, lines, min_size


def compose_subtitle(
    image_bytes: bytes,
    speak_text: str,
    settings: dict | None,
    bot_name: str = "",
) -> bytes:
    """이미지 위에 방송 애니메이션 스타일 자막을 합성해 PNG로 반환한다."""
    source = str(speak_text or "")
    if not source.strip() or source.strip().lower() in {"none", "null", "nil"}:
        print(f"[SUBTITLE] 자막 원문 없음, 합성 스킵: bot={bot_name!r}")
        return image_bytes
    try:
        from modes.postprocess import parse_speak

        effective = normalize_subtitle_settings(settings)
        if not effective["enabled"]:
            print(f"[SUBTITLE] 자막 설정 비활성, 합성 스킵: bot={bot_name!r}")
            return image_bytes
        segments = parse_speak(source, strip_emotion=True)
        if not segments:
            print(
                "[SUBTITLE] 파싱된 대사 없음, 합성 스킵: "
                f"bot={bot_name!r}, speak={source!r}"
            )
            return image_bytes
        if len(segments) > 2:
            print(
                "[SUBTITLE] 화면 자막은 최대 2개 발화만 표시: "
                f"bot={bot_name!r}, segments={len(segments)}"
            )
        texts = [str(item.get("text") or "").strip() for item in segments[:2]]
        texts = [value for value in texts if value]
        if not texts:
            print(f"[SUBTITLE] 표시할 자막 본문 없음, 합성 스킵: bot={bot_name!r}")
            return image_bytes

        base = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        width, height = base.size
        measure = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
        max_width = max(1.0, width * float(effective["max_width_ratio"]))
        font, lines, font_size = _fit_subtitle(
            measure, texts, effective, max_width
        )
        if not lines:
            print(f"[SUBTITLE] 레이아웃 결과가 비어 합성 스킵: bot={bot_name!r}")
            return image_bytes

        text = "\n".join(lines)
        spacing = max(0, int(round(font_size * float(effective["line_spacing_ratio"]))))
        bbox = measure.multiline_textbbox(
            (0, 0), text, font=font, spacing=spacing, align="center",
            stroke_width=int(effective["outline_width"]),
        )
        block_height = max(1, bbox[3] - bbox[1])
        bottom_margin = max(
            int(effective["outline_width"]) + 4,
            int(round(height * float(effective["bottom_margin_ratio"]))),
        )
        x = width / 2.0
        y = max(
            int(effective["outline_width"]) + 2,
            height - bottom_margin - block_height - bbox[1],
        )

        shadow_alpha = int(round(255 * float(effective["shadow_opacity"])))
        shadow_rgb = ImageColor.getrgb(effective["shadow_color"])
        shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.multiline_text(
            (
                x + int(effective["shadow_offset_x"]),
                y + int(effective["shadow_offset_y"]),
            ),
            text,
            font=font,
            fill=(*shadow_rgb, shadow_alpha),
            anchor="ma",
            align="center",
            spacing=spacing,
            stroke_width=int(effective["outline_width"]) + 2,
            stroke_fill=(*shadow_rgb, shadow_alpha),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(max(0.8, font_size * 0.018)))
        composed = Image.alpha_composite(base, shadow)

        foreground = Image.new("RGBA", base.size, (0, 0, 0, 0))
        foreground_draw = ImageDraw.Draw(foreground)
        foreground_draw.multiline_text(
            (x, y),
            text,
            font=font,
            fill=(*ImageColor.getrgb(effective["text_color"]), 255),
            anchor="ma",
            align="center",
            spacing=spacing,
            stroke_width=int(effective["outline_width"]),
            stroke_fill=(*ImageColor.getrgb(effective["outline_color"]), 255),
        )
        composed = Image.alpha_composite(composed, foreground)
        output = io.BytesIO()
        composed.save(output, format="PNG")
        print(
            "[SUBTITLE] 자막 합성 완료: "
            f"bot={bot_name!r}, size={width}x{height}, font={font_size}, "
            f"lines={len(lines)}, text={text!r}"
        )
        return output.getvalue()
    except Exception as exc:
        print(
            "[SUBTITLE] 자막 합성 실패, 원본 반환: "
            f"bot={bot_name!r}, speak={source!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return image_bytes
