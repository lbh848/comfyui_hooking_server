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
    "thought_italic_enabled": True,
    "thought_italic_shear": 0.10,
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
    thought_italic_enabled = source.get(
        "thought_italic_enabled",
        normalized["thought_italic_enabled"],
    )
    if not isinstance(thought_italic_enabled, bool):
        print(
            "[SUBTITLE:CONFIG] thought_italic_enabled가 bool이 아니어서 기본값 사용: "
            f"value={thought_italic_enabled!r}"
        )
        thought_italic_enabled = normalized["thought_italic_enabled"]
    normalized["thought_italic_enabled"] = thought_italic_enabled
    normalized["thought_italic_shear"] = _number(
        source.get("thought_italic_shear", normalized["thought_italic_shear"]),
        normalized["thought_italic_shear"], 0.04, 0.20, "thought_italic_shear",
    )
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
    thought_flags: list[bool],
    settings: dict,
    max_width: float,
) -> tuple[object, list[str], list[bool], int]:
    from modes.font_assets import load_font

    def flags_for(lines: list[str]) -> list[bool]:
        if len(texts) == 1:
            return [bool(thought_flags[0])] * len(lines)
        if len(lines) == len(texts):
            return [bool(value) for value in thought_flags[:len(lines)]]
        # 둘 이상의 발화를 한 줄로 합친 경우 모두 생각일 때만 기울인다.
        return [bool(thought_flags) and all(thought_flags)] * len(lines)

    start_size = int(settings["font_size"])
    min_size = int(settings["min_font_size"])
    for size in range(start_size, min_size - 1, -1):
        font = load_font(size, settings["font_id"], settings["font_path"])
        lines = _layout_lines(
            draw, texts, font, max_width, int(settings["max_lines"])
        )
        if lines is not None:
            return font, lines, flags_for(lines), size

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
    return font, lines, flags_for(lines), min_size


def _slant_layer(layer: Image.Image, shear: float) -> Image.Image:
    """투명 텍스트 레이어의 윗부분을 오른쪽으로 밀어 약한 합성 이탤릭을 만든다."""
    amount = max(0.0, float(shear or 0.0))
    if amount <= 0.0 or layer.height <= 0:
        return layer
    shift = max(1, int(math.ceil(layer.height * amount)))
    return layer.transform(
        (layer.width + shift, layer.height),
        Image.Transform.AFFINE,
        (1.0, amount, -amount * layer.height, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
    )


def _build_line_layers(
    measure: ImageDraw.ImageDraw,
    text: str,
    font,
    settings: dict,
    *,
    italic: bool,
) -> tuple[Image.Image, Image.Image, int, int]:
    """같은 기준점으로 정렬된 그림자/전경 한 줄 레이어를 만든다."""
    outline_width = int(settings["outline_width"])
    outer_stroke = outline_width + 2
    bbox = measure.textbbox(
        (0, 0),
        text,
        font=font,
        stroke_width=outer_stroke,
    )
    logical_width = max(1, int(math.ceil(bbox[2] - bbox[0])))
    logical_height = max(1, int(math.ceil(bbox[3] - bbox[1])))
    pad = max(4, outer_stroke + 2)
    layer_size = (logical_width + pad * 2, logical_height + pad * 2)
    origin = (pad - bbox[0], pad - bbox[1])

    shadow_alpha = int(round(255 * float(settings["shadow_opacity"])))
    shadow_rgb = ImageColor.getrgb(settings["shadow_color"])
    shadow = Image.new("RGBA", layer_size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text(
        origin,
        text,
        font=font,
        fill=(*shadow_rgb, shadow_alpha),
        stroke_width=outer_stroke,
        stroke_fill=(*shadow_rgb, shadow_alpha),
    )

    foreground = Image.new("RGBA", layer_size, (0, 0, 0, 0))
    ImageDraw.Draw(foreground).text(
        origin,
        text,
        font=font,
        fill=(*ImageColor.getrgb(settings["text_color"]), 255),
        stroke_width=outline_width,
        stroke_fill=(*ImageColor.getrgb(settings["outline_color"]), 255),
    )

    if italic:
        shear = float(settings["thought_italic_shear"])
        shadow = _slant_layer(shadow, shear)
        foreground = _slant_layer(foreground, shear)
    return shadow, foreground, logical_height, pad


def compose_subtitle(
    image_bytes: bytes,
    speak_text: str,
    settings: dict | None,
    bot_name: str = "",
) -> bytes:
    """이미지 위에 방송 애니메이션 스타일 자막을 합성해 PNG로 반환한다."""
    source = str(speak_text or "")
    if not source.strip() or source.strip().lower() in {"empty", "none", "null", "nil"}:
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
        entries = [
            {
                "text": str(item.get("text") or "").strip(),
                "thought": str(item.get("type") or "") == "thought",
            }
            for item in segments[:2]
            if str(item.get("text") or "").strip()
        ]
        if not entries:
            print(f"[SUBTITLE] 표시할 자막 본문 없음, 합성 스킵: bot={bot_name!r}")
            return image_bytes
        texts = [str(entry["text"]) for entry in entries]
        thought_flags = [bool(entry["thought"]) for entry in entries]

        base = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        width, height = base.size
        measure = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
        max_width = max(1.0, width * float(effective["max_width_ratio"]))
        font, lines, line_thought_flags, font_size = _fit_subtitle(
            measure, texts, thought_flags, effective, max_width
        )
        if not lines:
            print(f"[SUBTITLE] 레이아웃 결과가 비어 합성 스킵: bot={bot_name!r}")
            return image_bytes

        text = "\n".join(lines)
        spacing = max(0, int(round(font_size * float(effective["line_spacing_ratio"]))))
        line_layers = [
            _build_line_layers(
                measure,
                line,
                font,
                effective,
                italic=(
                    bool(is_thought)
                    and bool(effective["thought_italic_enabled"])
                ),
            )
            for line, is_thought in zip(lines, line_thought_flags)
        ]
        block_height = max(
            1,
            sum(item[2] for item in line_layers)
            + spacing * max(0, len(line_layers) - 1),
        )
        bottom_margin = max(
            int(effective["outline_width"]) + 4,
            int(round(height * float(effective["bottom_margin_ratio"]))),
        )
        y_cursor = max(
            int(effective["outline_width"]) + 2,
            height - bottom_margin - block_height,
        )

        shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
        foreground = Image.new("RGBA", base.size, (0, 0, 0, 0))
        for shadow_line, foreground_line, logical_height, pad in line_layers:
            shadow_x = int(round((width - shadow_line.width) / 2.0))
            foreground_x = int(round((width - foreground_line.width) / 2.0))
            paste_y = int(round(y_cursor - pad))
            shadow.alpha_composite(
                shadow_line,
                (
                    shadow_x + int(effective["shadow_offset_x"]),
                    paste_y + int(effective["shadow_offset_y"]),
                ),
            )
            foreground.alpha_composite(
                foreground_line,
                (foreground_x, paste_y),
            )
            y_cursor += logical_height + spacing

        shadow = shadow.filter(ImageFilter.GaussianBlur(max(0.8, font_size * 0.018)))
        composed = Image.alpha_composite(base, shadow)
        composed = Image.alpha_composite(composed, foreground)
        output = io.BytesIO()
        composed.save(output, format="PNG")
        print(
            "[SUBTITLE] 자막 합성 완료: "
            f"bot={bot_name!r}, size={width}x{height}, font={font_size}, "
            f"lines={len(lines)}, thoughts={sum(line_thought_flags)}, text={text!r}"
        )
        return output.getvalue()
    except Exception as exc:
        print(
            "[SUBTITLE] 자막 합성 실패, 원본 반환: "
            f"bot={bot_name!r}, speak={source!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return image_bytes
