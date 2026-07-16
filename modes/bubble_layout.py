"""Font-aware candidate generation and ONNX ranking for bubble text layouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import traceback
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modes.onnx_execution import (
    cache_session,
    create_session,
    session_cache_key,
    session_uses_gpu,
)


FEATURE_NAMES = (
    "fit_quality",
    "font_scale",
    "text_fill",
    "aspect_match",
    "line_balance",
    "last_line_balance",
    "boundary_quality",
    "punctuation_quality",
    "orphan_quality",
    "line_count_quality",
    "compactness",
    "shape_text_match",
    "width_utilization",
    "height_utilization",
    "shape_hint_match",
    "shape_ellipse",
    "shape_rounded",
    "shape_cloud",
    "text_length",
    "ellipsis_signal",
    "question_signal",
    "exclamation_signal",
    "explicit_break_signal",
    "aspect_normalized",
    "semantic_break_quality",
)

# ONNX가 없거나 로드되지 않을 때의 안전한 규칙 기반 폴백.
SEEDED_WEIGHTS = (
    4.00,
    2.40,
    0.55,
    1.00,
    1.40,
    1.05,
    1.25,
    1.75,
    1.20,
    0.70,
    0.20,
    0.80,
    0.20,
    0.15,
    4.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    1.60,
)

_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "bubble_layout.onnx")
)
_sessions = {}


OPENING_PUNCTUATION = frozenset("([{<‘“「『《〈【〔（［｛")
CLOSING_PUNCTUATION = frozenset(")]}>’”」』》〉】〕）］｝,.;:!?、。，；：！？…")
GOOD_BREAK_PUNCTUATION = frozenset(",.;:!?、。，；：！？…~")


@dataclass(frozen=True)
class BubbleShape:
    name: str
    padding_x_em: float
    padding_y_em: float
    preferred_aspect: float
    min_width_frac: float
    max_width_frac: float
    min_height_frac: float
    max_height_frac: float


DEFAULT_SHAPES = (
    BubbleShape("ellipse", 1.10, 0.88, 1.48, 0.18, 0.48, 0.09, 0.36),
    BubbleShape("rounded", 0.88, 0.70, 1.62, 0.18, 0.50, 0.09, 0.38),
    BubbleShape("cloud", 1.24, 1.00, 1.28, 0.19, 0.46, 0.10, 0.36),
)


@dataclass(frozen=True)
class WrapResult:
    lines: tuple[str, ...]
    boundary_quality: float
    punctuation_quality: float


@dataclass(frozen=True)
class LayoutCandidate:
    lines: tuple[str, ...]
    font_size: int
    spacing: int
    text_width: float
    text_height: float
    bubble_width: float
    bubble_height: float
    shape: str
    features: tuple[float, ...]
    fits: bool
    overflow_ratio: float
    score: float | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def to_dict(self, include_features: bool = False) -> dict:
        payload = {
            "text": self.text,
            "lines": list(self.lines),
            "font_size": self.font_size,
            "spacing": self.spacing,
            "text_size": [round(self.text_width, 2), round(self.text_height, 2)],
            "bubble_size": [round(self.bubble_width, 2), round(self.bubble_height, 2)],
            "aspect_ratio": round(self.bubble_width / max(self.bubble_height, 1e-6), 4),
            "shape": self.shape,
            "fits": self.fits,
            "overflow_ratio": round(self.overflow_ratio, 6),
            "score": None if self.score is None else round(self.score, 6),
        }
        if include_features:
            payload["features"] = {
                name: round(value, 6) for name, value in zip(FEATURE_NAMES, self.features)
            }
        return payload


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def load_font(
    size: int,
    font_path: str | os.PathLike | None = None,
    *,
    font_id: str | None = None,
) -> ImageFont.ImageFont:
    # font_id(드롭박스 식별자)가 주어지면 font_assets 로 로드(번들 자동 다운로드,
    # 변수폰트 variation 적용). 렌더(bubble_render)와 동일 로더를 써야 측정과
    # 그리기가 일치한다(CLAUDE.md: 미리보기=실제 동일 빌더).
    if font_id and font_id != "system":
        try:
            from modes.font_assets import load_font as _fa_load

            font = _fa_load(
                size,
                font_id=font_id,
                legacy_path=str(font_path) if font_path else None,
            )
            if font is not None:
                return font
            print(f"[BUBBLE_LAYOUT] font_assets 로드 결과 없음 → 경로 폴백: {font_id}")
        except Exception as e:
            print(f"[BUBBLE_LAYOUT] font_assets 로드 실패, 경로 폴백: {e}")
            traceback.print_exc()
    candidates = (
        str(font_path) if font_path else None,
        r"C:\Windows\Fonts\malgun.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 compatibility
        return ImageFont.load_default()


def _rendered_width(
    draw: ImageDraw.ImageDraw,
    s: str,
    font: ImageFont.ImageFont,
    tracking_px: float,
    h_scale: float,
) -> float:
    """자간(tracking)·가로축소(h_scale) 적용 후 폭.

    그리기(줄 스트립 → 가로 리사이즈)와 동일 기하: (자연폭 + tracking×(len-1)) × h_scale.
    tracking_px=0, h_scale=1.0 이면 textlength 와 동일(기본값에서 기존 동작 유지).
    """
    if not s:
        return 0.0
    base = float(draw.textlength(s, font=font))
    gaps = tracking_px * max(0, len(s) - 1)
    return (base + gaps) * h_scale


def _line_metrics(
    font: ImageFont.ImageFont, font_size: int, line_height_ratio: float | None
) -> tuple:
    """(spacing, line_advance). line_advance=None 이면 호출자가 기존 multiline_textbbox 를 쓴다.

    line_height_ratio=None → 기존 동작(spacing=font_size×0.27, line_advance=None).
    line_height_ratio 주어지면 줄 전체 높이=font_size×ratio, line_gap=max(0, ratio×fs-(ascent+descent)).
    """
    if line_height_ratio is None:
        return max(2, int(round(font_size * 0.27))), None
    ascent, descent = font.getmetrics()
    natural = float(ascent + descent)
    target = float(font_size) * float(line_height_ratio)
    line_gap = max(0.0, target - natural)
    line_advance = max(natural, target)
    return int(round(line_gap)), line_advance


def _next_non_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _boundary_penalty(text: str, end: int, next_start: int) -> float:
    if next_start >= len(text):
        return 0.0
    previous = text[:end].rstrip()[-1] if text[:end].rstrip() else ""
    following = text[next_start]
    if previous in OPENING_PUNCTUATION or following in CLOSING_PUNCTUATION:
        return 1.0
    if previous in GOOD_BREAK_PUNCTUATION:
        return 0.05
    if (end > 0 and text[end - 1].isspace()) or (end < len(text) and text[end].isspace()):
        return 0.24
    return 0.62


def _punctuation_violations(lines: Iterable[str]) -> int:
    lines = tuple(lines)
    violations = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if index > 0 and stripped[0] in CLOSING_PUNCTUATION:
            violations += 1
        if index + 1 < len(lines) and stripped[-1] in OPENING_PUNCTUATION:
            violations += 1
    return violations


def _wrap_paragraph(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: float,
    max_lines: int,
    width_cache: dict[str, float],
    *,
    tracking_px: float = 0.0,
    h_scale: float = 1.0,
) -> WrapResult | None:
    text = text.strip()
    if not text:
        return WrapResult(("",), 1.0, 1.0)

    n = len(text)

    def width(value: str) -> float:
        cached = width_cache.get(value)
        if cached is None:
            cached = _rendered_width(draw, value, font, tracking_px, h_scale)
            width_cache[value] = cached
        return cached

    # (position, line_count) -> (cost, previous_position, previous_count, line,
    #                            boundary_penalty)
    states: dict[tuple[int, int], tuple[float, int, int, str, float]] = {
        (0, 0): (0.0, -1, -1, "", 0.0)
    }
    for line_count in range(max_lines):
        active = [item for item in states.items() if item[0][1] == line_count]
        for (start, _), (base_cost, *_rest) in active:
            if start >= n:
                continue
            for end in range(start + 1, n + 1):
                raw = text[start:end]
                line = raw.rstrip()
                if not line:
                    continue
                line_width = width(line)
                if line_width > max_width + 1e-6:
                    break
                next_start = _next_non_space(text, end)
                if next_start < n:
                    previous = text[:end].rstrip()[-1] if text[:end].rstrip() else ""
                    at_natural_boundary = (
                        (end > 0 and text[end - 1].isspace())
                        or (end < n and text[end].isspace())
                        or previous in GOOD_BREAK_PUNCTUATION
                    )
                    if not at_natural_boundary:
                        # 띄어쓰기 안쪽에서는 절대 줄을 끊지 않는다. 한 단어가
                        # 현재 폭보다 길면 더 넓은 몸통/더 작은 폰트 후보가 고르게
                        # 하고, 여기서 글자 단위로 잘라 "안/녕하/세요"를 만들지 않는다.
                        continue
                penalty = _boundary_penalty(text, end, next_start)
                slack = max(0.0, 1.0 - line_width / max(max_width, 1.0))
                punctuation = _punctuation_violations((line, text[next_start:]))
                cost = base_cost + penalty + 0.16 * slack * slack + 0.04 + 0.8 * punctuation
                key = (next_start, line_count + 1)
                current = states.get(key)
                if current is None or cost < current[0]:
                    states[key] = (cost, start, line_count, line, penalty)

    finals = []
    for (position, line_count), state in states.items():
        if position != n or line_count <= 0:
            continue
        lines: list[str] = []
        penalties: list[float] = []
        pos, count = position, line_count
        while count > 0:
            item = states[(pos, count)]
            lines.append(item[3])
            penalties.append(item[4])
            pos, count = item[1], item[2]
        lines.reverse()
        penalties.reverse()
        widths = [width(line) for line in lines if line]
        mean_width = sum(widths) / max(len(widths), 1)
        variance = sum((value - mean_width) ** 2 for value in widths) / max(len(widths), 1)
        balance_penalty = math.sqrt(variance) / max(mean_width, 1.0)
        last_ratio = widths[-1] / max(mean_width, 1.0)
        orphan_penalty = max(0.0, 0.48 - last_ratio) * 2.2
        finals.append((state[0] + 0.38 * balance_penalty + orphan_penalty, lines, penalties))

    if not finals:
        return None
    _, lines, penalties = min(finals, key=lambda item: item[0])
    break_penalties = penalties[:-1]
    boundary_quality = 1.0 - sum(break_penalties) / max(len(break_penalties), 1)
    punctuation_quality = 1.0 - _punctuation_violations(lines) / max(len(lines) - 1, 1)
    return WrapResult(
        tuple(lines),
        _clamp(boundary_quality, 0.0, 1.0),
        _clamp(punctuation_quality, 0.0, 1.0),
    )


def _wrap_text(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: float,
    max_lines: int,
    width_cache: dict[str, float],
    *,
    tracking_px: float = 0.0,
    h_scale: float = 1.0,
) -> WrapResult | None:
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Fast impossibility check.  It avoids running the dynamic program for
    # every candidate when a very long text cannot fit within max_lines.
    minimum_lines = 0
    for paragraph in paragraphs:
        paragraph_width = _rendered_width(
            draw, paragraph.strip() or " ", font, tracking_px, h_scale
        )
        minimum_lines += max(1, math.ceil(paragraph_width / max(max_width, 1.0)))
    if minimum_lines > max_lines:
        return None
    all_lines: list[str] = []
    boundary_scores: list[float] = []
    punctuation_scores: list[float] = []
    remaining = max_lines
    for paragraph in paragraphs:
        wrapped = _wrap_paragraph(
            paragraph, draw, font, max_width, remaining, width_cache,
            tracking_px=tracking_px, h_scale=h_scale,
        )
        if wrapped is None:
            return None
        all_lines.extend(wrapped.lines)
        boundary_scores.append(wrapped.boundary_quality)
        punctuation_scores.append(wrapped.punctuation_quality)
        remaining -= len(wrapped.lines)
        if remaining < 0:
            return None
    return WrapResult(
        tuple(all_lines),
        sum(boundary_scores) / max(len(boundary_scores), 1),
        sum(punctuation_scores) / max(len(punctuation_scores), 1),
    )


def _greedy_wrap_text(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: float,
    *,
    tracking_px: float = 0.0,
    h_scale: float = 1.0,
) -> WrapResult:
    """공백 단위 전용 무제한 줄바꿈 폴백.

    한 단어가 max_width보다 길어도 단어 자체를 보존한다. 이 후보는 overflow로
    표시되어 호출자가 몸통 확장/폰트 축소 또는 렌더 스킵을 결정할 수 있다.
    """
    lines: list[str] = []
    penalties: list[float] = []
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if current and _rendered_width(draw, trial, font, tracking_px, h_scale) > max_width:
                lines.append(current)
                penalties.append(0.24)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
            penalties.append(0.0)
    boundary_penalties = penalties[:-1]
    return WrapResult(
        tuple(lines),
        _clamp(
            1.0 - sum(boundary_penalties) / max(len(boundary_penalties), 1),
            0.0,
            1.0,
        ),
        _clamp(
            1.0 - _punctuation_violations(lines) / max(len(lines) - 1, 1),
            0.0,
            1.0,
        ),
    )


def _measure_lines(
    lines: tuple[str, ...],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    spacing: int,
    *,
    tracking_px: float = 0.0,
    h_scale: float = 1.0,
    line_advance: float | None = None,
) -> tuple[list[float], float, float]:
    widths = [
        _rendered_width(draw, line or " ", font, tracking_px, h_scale) for line in lines
    ]
    text_width = max(widths) if widths else 0.0
    if line_advance is not None:
        # 그리기(줄별 y 전진=line_advance)와 동일 기하. 줄 전체 높이=font_size×ratio.
        text_height = line_advance * max(len(lines), 1)
    else:
        text = "\n".join(lines)
        box = draw.multiline_textbbox((0, 0), text or " ", font=font, spacing=spacing, align="center")
        text_width = max(text_width, float(box[2] - box[0]))
        text_height = float(box[3] - box[1])
    return widths, text_width, text_height


def _shape_text_match(shape: str, text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    length_signal = _clamp((len(compact) - 32) / 48.0, 0.0, 1.0)
    thought_signal = _clamp(
        (compact.count("…") + compact.count("...") + compact.count("?") * 0.35) / 2.0,
        0.0,
        1.0,
    )
    if shape == "cloud":
        return 0.22 + 0.78 * thought_signal
    if shape == "rounded":
        return 0.32 + 0.68 * length_signal
    return 1.0 - 0.35 * max(length_signal, thought_signal)


def _semantic_break_quality(lines: tuple[str, ...]) -> float:
    """Score whether visual lines follow sentence/clause boundaries.

    Punctuation that appears in the middle of a line is a signal that a better
    break probably existed.  Conversely, a break after sentence punctuation
    is stronger than an arbitrary whitespace break.  This remains a soft
    feature: geometry and the trained ranker can still choose a different
    layout when the bubble is narrow.
    """
    if len(lines) <= 1:
        return 1.0
    strong = frozenset("!?！？")
    soft = frozenset(".;。；…")
    total = 0.0
    violations = 0.0
    boundary_scores: list[float] = []
    for line_index, line in enumerate(lines):
        stripped = line.rstrip()
        for index, character in enumerate(stripped):
            weight = 1.0 if character in strong else (0.55 if character in soft else 0.0)
            if weight <= 0:
                continue
            total += weight
            if index + 1 < len(stripped):
                violations += weight
        if line_index + 1 < len(lines):
            ending = stripped[-1] if stripped else ""
            if ending in strong:
                boundary_scores.append(1.0)
            elif ending in soft:
                boundary_scores.append(0.82)
            else:
                boundary_scores.append(0.42)
    boundary_quality = sum(boundary_scores) / max(len(boundary_scores), 1)
    punctuation_quality = 1.0 if total <= 0 else 1.0 - violations / total
    return _clamp(0.65 * boundary_quality + 0.35 * punctuation_quality, 0.0, 1.0)


def _features(
    *,
    text: str,
    shape: BubbleShape,
    shape_hint: str | None,
    wrap: WrapResult,
    line_widths: list[float],
    font_size: int,
    min_font_size: int,
    max_font_size: int,
    text_width: float,
    text_height: float,
    bubble_width: float,
    bubble_height: float,
    canvas_size: tuple[int, int],
    overflow_ratio: float,
) -> tuple[float, ...]:
    canvas_width, canvas_height = canvas_size
    max_width = shape.max_width_frac * canvas_width
    max_height = shape.max_height_frac * canvas_height
    fit_quality = _clamp(1.0 - overflow_ratio * 4.0, 0.0, 1.0)
    font_scale = (font_size - min_font_size) / max(max_font_size - min_font_size, 1)
    text_fill = text_width * text_height / max(bubble_width * bubble_height, 1.0)
    aspect = bubble_width / max(bubble_height, 1.0)
    aspect_match = math.exp(-abs(math.log(max(aspect, 1e-6) / shape.preferred_aspect)))
    nonempty_widths = [value for value in line_widths if value > 0]
    mean_width = sum(nonempty_widths) / max(len(nonempty_widths), 1)
    variance = sum((value - mean_width) ** 2 for value in nonempty_widths) / max(
        len(nonempty_widths), 1
    )
    line_balance = 1.0 - math.sqrt(variance) / max(mean_width, 1.0)
    last_line_balance = min(nonempty_widths[-1] / max(mean_width, 1.0), 1.0)
    line_lengths = [len(re.sub(r"\s+", "", line)) for line in wrap.lines if line]
    mean_length = sum(line_lengths) / max(len(line_lengths), 1)
    orphan_quality = min(line_lengths[-1] / max(mean_length * 0.48, 1.0), 1.0)
    compact_len = len(re.sub(r"\s+", "", text))
    ellipsis_signal = _clamp(
        (text.count("…") + text.count("...") + text.count("..") * 0.5) / 2.0,
        0.0,
        1.0,
    )
    target_lines = _clamp(round(math.sqrt(max(compact_len, 1)) * 0.62), 1, 6)
    line_count_quality = math.exp(-abs(len(wrap.lines) - target_lines) / max(target_lines, 1))
    compactness = 1.0 - bubble_width * bubble_height / max(max_width * max_height, 1.0)
    hint_match = 1.0 if shape_hint and shape.name == shape_hint else 0.0
    return tuple(
        float(_clamp(value, 0.0, 1.0))
        for value in (
            fit_quality,
            font_scale,
            text_fill,
            aspect_match,
            line_balance,
            last_line_balance,
            wrap.boundary_quality,
            wrap.punctuation_quality,
            orphan_quality,
            line_count_quality,
            compactness,
            _shape_text_match(shape.name, text),
            text_width / max(bubble_width, 1.0),
            text_height / max(bubble_height, 1.0),
            hint_match,
            1.0 if shape.name == "ellipse" else 0.0,
            1.0 if shape.name == "rounded" else 0.0,
            1.0 if shape.name == "cloud" else 0.0,
            compact_len / 120.0,
            ellipsis_signal,
            text.count("?") / 3.0,
            text.count("!") / 3.0,
            text.count("\n") / 3.0,
            aspect / 2.5,
            _semantic_break_quality(wrap.lines),
        )
    )


def generate_layout_candidates(
    text: str,
    canvas_size: tuple[int, int],
    font_path: str | os.PathLike | None = None,
    *,
    shapes: tuple[BubbleShape, ...] = DEFAULT_SHAPES,
    shape_hint: str | None = None,
    min_font_size: int | None = None,
    max_font_size: int | None = None,
    max_lines: int = 7,
    font_id: str | None = None,
    letter_spacing: float = 0.0,
    text_width_scale: float = 1.0,
    line_height_ratio: float | None = None,
) -> list[LayoutCandidate]:
    """Generate measured candidates; valid and overflow candidates are retained."""
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")
    if not shapes:
        raise ValueError("at least one bubble shape is required")
    shape_names = {shape.name for shape in shapes}
    if shape_hint is not None and shape_hint not in shape_names:
        raise ValueError(f"unknown shape_hint {shape_hint!r}; expected one of {sorted(shape_names)}")

    canvas_width, canvas_height = (int(canvas_size[0]), int(canvas_size[1]))
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(f"invalid canvas_size: {canvas_size}")
    base = min(canvas_width, canvas_height)
    min_font_size = int(min_font_size or max(12, round(base * 0.018)))
    max_font_size = int(max_font_size or min(96, max(min_font_size, round(base * 0.055))))
    if min_font_size <= 0 or max_font_size < min_font_size:
        raise ValueError("font size range is invalid")

    steps = min(9, max(3, max_font_size - min_font_size + 1))
    font_sizes = sorted(
        {int(round(value)) for value in np.linspace(min_font_size, max_font_size, steps)},
        reverse=True,
    )
    probe = Image.new("L", (8, 8))
    draw = ImageDraw.Draw(probe)
    candidates: list[LayoutCandidate] = []
    seen: set[tuple] = set()

    for font_size in font_sizes:
        font = load_font(font_size, font_path, font_id=font_id)
        spacing, line_advance = _line_metrics(font, font_size, line_height_ratio)
        tracking_px = float(font_size) * float(letter_spacing)
        width_cache: dict[str, float] = {}
        for shape in shapes:
            fractions = np.linspace(
                max(shape.min_width_frac, 0.18), shape.max_width_frac, 7
            )
            for body_width_limit in fractions * canvas_width:
                max_text_width = body_width_limit - 2.0 * shape.padding_x_em * font_size
                if max_text_width <= font_size:
                    continue
                wrapped = _wrap_text(
                    text, draw, font, float(max_text_width), max_lines, width_cache,
                    tracking_px=tracking_px, h_scale=text_width_scale,
                )
                if wrapped is None:
                    continue
                line_widths, text_width, text_height = _measure_lines(
                    wrapped.lines, draw, font, spacing,
                    tracking_px=tracking_px, h_scale=text_width_scale,
                    line_advance=line_advance,
                )
                raw_width = text_width + 2.0 * shape.padding_x_em * font_size
                raw_height = text_height + 2.0 * shape.padding_y_em * font_size
                raw_bubble_width = max(raw_width, canvas_width * shape.min_width_frac)
                raw_bubble_height = max(raw_height, canvas_height * shape.min_height_frac)
                preferred_width, preferred_height = raw_bubble_width, raw_bubble_height
                current_aspect = preferred_width / max(preferred_height, 1.0)
                if current_aspect < shape.preferred_aspect:
                    preferred_width = preferred_height * shape.preferred_aspect
                elif current_aspect > shape.preferred_aspect:
                    preferred_height = preferred_width / shape.preferred_aspect

                max_width = shape.max_width_frac * canvas_width
                max_height = shape.max_height_frac * canvas_height
                dimension_variants = [(raw_bubble_width, raw_bubble_height)]
                if (
                    abs(preferred_width - raw_bubble_width) > 1e-6
                    or abs(preferred_height - raw_bubble_height) > 1e-6
                ):
                    dimension_variants.append((preferred_width, preferred_height))
                for bubble_width, bubble_height in dimension_variants:
                    overflow_x = max(0.0, bubble_width - max_width) / max(max_width, 1.0)
                    overflow_y = max(0.0, bubble_height - max_height) / max(max_height, 1.0)
                    overflow = overflow_x + overflow_y
                    fits = overflow <= 1e-6
                    key = (
                        wrapped.lines,
                        font_size,
                        shape.name,
                        round(bubble_width, 2),
                        round(bubble_height, 2),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    features = _features(
                        text=text,
                        shape=shape,
                        shape_hint=shape_hint,
                        wrap=wrapped,
                        line_widths=line_widths,
                        font_size=font_size,
                        min_font_size=min_font_size,
                        max_font_size=max_font_size,
                        text_width=text_width,
                        text_height=text_height,
                        bubble_width=bubble_width,
                        bubble_height=bubble_height,
                        canvas_size=(canvas_width, canvas_height),
                        overflow_ratio=overflow,
                    )
                    candidates.append(
                        LayoutCandidate(
                            lines=wrapped.lines,
                            font_size=font_size,
                            spacing=spacing,
                            text_width=text_width,
                            text_height=text_height,
                            bubble_width=bubble_width,
                            bubble_height=bubble_height,
                            shape=shape.name,
                            features=features,
                            fits=fits,
                            overflow_ratio=overflow,
                        )
                    )
    if not candidates:
        # The text is too long even at the smallest font and widest legal
        # bubble.  Return measured overflow candidates instead of throwing;
        # callers can surface ``fits=False`` and ask for a shorter text.
        font_size = min_font_size
        font = load_font(font_size, font_path, font_id=font_id)
        spacing, line_advance = _line_metrics(font, font_size, line_height_ratio)
        tracking_px = float(font_size) * float(letter_spacing)
        for shape in shapes:
            max_text_width = (
                shape.max_width_frac * canvas_width - 2.0 * shape.padding_x_em * font_size
            )
            wrapped = _greedy_wrap_text(
                text, draw, font, max(max_text_width, 1.0),
                tracking_px=tracking_px, h_scale=text_width_scale,
            )
            line_widths, text_width, text_height = _measure_lines(
                wrapped.lines, draw, font, spacing,
                tracking_px=tracking_px, h_scale=text_width_scale,
                line_advance=line_advance,
            )
            bubble_width = max(
                text_width + 2.0 * shape.padding_x_em * font_size,
                canvas_width * shape.min_width_frac,
            )
            bubble_height = max(
                text_height + 2.0 * shape.padding_y_em * font_size,
                canvas_height * shape.min_height_frac,
            )
            current_aspect = bubble_width / max(bubble_height, 1.0)
            if current_aspect < shape.preferred_aspect:
                bubble_width = bubble_height * shape.preferred_aspect
            elif current_aspect > shape.preferred_aspect:
                bubble_height = bubble_width / shape.preferred_aspect
            max_width = shape.max_width_frac * canvas_width
            max_height = shape.max_height_frac * canvas_height
            overflow = (
                max(0.0, bubble_width - max_width) / max(max_width, 1.0)
                + max(0.0, bubble_height - max_height) / max(max_height, 1.0)
                + max(0, len(wrapped.lines) - max_lines) / max(max_lines, 1)
            )
            candidates.append(
                LayoutCandidate(
                    lines=wrapped.lines,
                    font_size=font_size,
                    spacing=spacing,
                    text_width=text_width,
                    text_height=text_height,
                    bubble_width=bubble_width,
                    bubble_height=bubble_height,
                    shape=shape.name,
                    features=_features(
                        text=text,
                        shape=shape,
                        shape_hint=shape_hint,
                        wrap=wrapped,
                        line_widths=line_widths,
                        font_size=font_size,
                        min_font_size=min_font_size,
                        max_font_size=max_font_size,
                        text_width=text_width,
                        text_height=text_height,
                        bubble_width=bubble_width,
                        bubble_height=bubble_height,
                        canvas_size=(canvas_width, canvas_height),
                        overflow_ratio=overflow,
                    ),
                    fits=False,
                    overflow_ratio=overflow,
                )
            )
    return candidates


def _seeded_scores(features: np.ndarray) -> np.ndarray:
    return features @ np.asarray(SEEDED_WEIGHTS, dtype=np.float32)


def _get_layout_session(onnx_path: str | os.PathLike, device="auto", cpu_threads=0):
    """레이아웃 ONNX 세션을 장치·스레드 조합별로 재사용한다."""
    path = os.path.abspath(str(onnx_path))
    if not os.path.isfile(path):
        print(f"[BUBBLE_LAYOUT] ONNX 모델 없음: {path}")
        return None
    key = session_cache_key(path, device, cpu_threads)
    if key in _sessions:
        return _sessions[key]
    session, _active_device = create_session(
        path,
        device_key=device,
        cpu_threads=cpu_threads,
        log_prefix="BUBBLE_LAYOUT",
    )
    if session is not None:
        cache_session(_sessions, key, session, log_prefix="BUBBLE_LAYOUT")
    return session


def _onnx_scores(
    features: np.ndarray,
    onnx_path: str | os.PathLike,
    device="auto",
    cpu_threads=0,
) -> np.ndarray | None:
    session = _get_layout_session(
        onnx_path,
        device=device,
        cpu_threads=cpu_threads,
    )
    if session is None:
        print("[BUBBLE_LAYOUT] 세션이 없어 초기 규칙 점수로 폴백")
        return None

    try:
        input_name = session.get_inputs()[0].name
        feeds = {input_name: features}
        try:
            output = session.run(None, feeds)[0]
        except Exception as gpu_error:
            if not session_uses_gpu(session):
                raise
            print(f"[BUBBLE_LAYOUT] GPU 추론 실패, CPU 폴백: {gpu_error}")
            traceback.print_exc()
            cpu_session = _get_layout_session(
                onnx_path,
                device="cpu",
                cpu_threads=cpu_threads,
            )
            if cpu_session is None:
                raise RuntimeError("레이아웃 CPU 폴백 세션 생성 실패") from gpu_error
            output = cpu_session.run(None, feeds)[0]
            cache_session(
                _sessions,
                session_cache_key(onnx_path, device, cpu_threads),
                cpu_session,
                log_prefix="BUBBLE_LAYOUT",
            )
        return np.asarray(output, dtype=np.float32)
    except Exception as e:
        print(f"[BUBBLE_LAYOUT] ONNX 추론 실패(features={features.shape}): {e}")
        traceback.print_exc()
        return None


def choose_layout(
    text: str,
    canvas_size: tuple[int, int],
    font_path: str | os.PathLike | None = None,
    *,
    onnx_path: str | os.PathLike | None = None,
    shape_hint: str | None = None,
    allowed_shapes: tuple[str, ...] | None = None,
    min_font_size: int | None = None,
    max_font_size: int | None = None,
    max_lines: int = 7,
    top_k: int = 5,
    onnx_device: str = "auto",
    cpu_threads: int = 0,
    font_id: str | None = None,
    letter_spacing: float = 0.0,
    text_width_scale: float = 1.0,
    line_height_ratio: float | None = None,
) -> tuple[LayoutCandidate, list[LayoutCandidate]]:
    shapes = DEFAULT_SHAPES
    if allowed_shapes is not None:
        allowed = set(allowed_shapes)
        shapes = tuple(shape for shape in DEFAULT_SHAPES if shape.name in allowed)
        unknown = allowed.difference(shape.name for shape in DEFAULT_SHAPES)
        if unknown:
            raise ValueError(f"unknown allowed_shapes: {sorted(unknown)}")
        if not shapes:
            raise ValueError("allowed_shapes must contain at least one known shape")
    candidates = generate_layout_candidates(
        text,
        canvas_size,
        font_path,
        shapes=shapes,
        shape_hint=shape_hint,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        max_lines=max_lines,
        font_id=font_id,
        letter_spacing=letter_spacing,
        text_width_scale=text_width_scale,
        line_height_ratio=line_height_ratio,
    )
    feature_array = np.asarray([candidate.features for candidate in candidates], dtype=np.float32)
    if onnx_path is None:
        default = Path(_MODEL_PATH)
        onnx_path = default if default.is_file() else None
    scores = (
        _onnx_scores(
            feature_array,
            onnx_path,
            device=onnx_device,
            cpu_threads=cpu_threads,
        )
        if onnx_path else None
    )
    if scores is None:
        if onnx_path is None:
            print(f"[BUBBLE_LAYOUT] 기본 ONNX 없음({_MODEL_PATH}) → 초기 규칙 점수 사용")
        scores = _seeded_scores(feature_array)
    scored = [replace(candidate, score=float(score)) for candidate, score in zip(candidates, scores)]
    if shape_hint:
        hinted = [candidate for candidate in scored if candidate.shape == shape_hint]
        if hinted:
            scored = hinted
    valid = [candidate for candidate in scored if candidate.fits]
    if valid:
        ranked = sorted(
            valid,
            key=lambda candidate: (
                candidate.score if candidate.score is not None else -math.inf,
                candidate.font_size,
            ),
            reverse=True,
        )
    else:
        ranked = sorted(
            scored,
            key=lambda candidate: (
                -candidate.overflow_ratio,
                candidate.score if candidate.score is not None else -math.inf,
            ),
            reverse=True,
        )
    return ranked[0], ranked[: max(1, top_k)]


def choose_scaled_layout(
    text: str,
    canvas_size: tuple[int, int],
    font_path: str | os.PathLike | None = None,
    *,
    font_scale: float = 2.0,
    force_shape: str | None = None,
    allowed_shapes: tuple[str, ...] | None = None,
    max_lines: int = 7,
    top_k: int = 5,
    onnx_device: str = "auto",
    cpu_threads: int = 0,
    font_id: str | None = None,
    letter_spacing: float = 0.0,
    text_width_scale: float = 1.0,
    line_height_ratio: float | None = None,
) -> tuple[LayoutCandidate, list[LayoutCandidate]]:
    """모델의 기본 선택을 기준으로 더 큰 글자에서 안전하게 재레이아웃한다.

    글자만 사후 확대하지 않고, 기본 모델이 결정한 버블 종류를 유지한 채 목표
    크기 주변에서 줄바꿈과 몸통 치수를 다시 계산한다. 정확한 배율이 캔버스에
    들어가지 않으면 10%씩 낮춰 가장 큰 ``fits=True`` 후보를 사용한다.

    ``force_shape``를 주면 모델이 고를 형상을 해당 종류(예: "cloud")로 강제한다.
    thought 대사(괄호 ``()`` 감싸짐)를 항상 구름으로 그릴 때 사용한다.
    """
    base, base_top = choose_layout(
        text,
        canvas_size,
        font_path,
        shape_hint=force_shape,
        allowed_shapes=allowed_shapes,
        max_lines=max_lines,
        top_k=top_k,
        onnx_device=onnx_device,
        cpu_threads=cpu_threads,
        font_id=font_id,
        letter_spacing=letter_spacing,
        text_width_scale=text_width_scale,
        line_height_ratio=line_height_ratio,
    )
    scale = max(1.0, float(font_scale))
    if scale <= 1.0 + 1e-6 or not base.fits:
        if not base.fits:
            print("[BUBBLE_LAYOUT] 기본 레이아웃도 맞지 않아 글자 확대를 건너뜀")
        return base, base_top

    # 재레이아웃에도 같은 형상을 유지한다. 강제 형상이 있으면 그것을,
    # 없으면 기본 선택의 형상을 따른다.
    keep_shape = force_shape or base.shape

    # 2.0 요청이면 2.0 → 1.8 → 1.6 → 1.4 → 1.2 순으로 후퇴한다.
    attempts = []
    factor = scale
    while factor > 1.05:
        attempts.append(factor)
        factor -= max(0.1, scale * 0.1)

    for factor in attempts:
        # 사용자 값은 '목표'가 아니라 상한이다. 내림을 사용해 선택 글자가
        # 요청 배율을 소수점 반올림 때문에라도 넘지 않게 한다.
        target_font = max(base.font_size + 1, int(math.floor(base.font_size * factor)))
        min_font = max(base.font_size + 1, int(round(target_font * 0.90)))
        max_font = max(min_font, target_font)
        selected, alternatives = choose_layout(
            text,
            canvas_size,
            font_path,
            shape_hint=keep_shape,
            allowed_shapes=allowed_shapes,
            min_font_size=min_font,
            max_font_size=max_font,
            max_lines=max_lines,
            top_k=top_k,
            onnx_device=onnx_device,
            cpu_threads=cpu_threads,
            font_id=font_id,
            letter_spacing=letter_spacing,
            text_width_scale=text_width_scale,
            line_height_ratio=line_height_ratio,
        )
        if selected.fits:
            actual_scale = selected.font_size / max(base.font_size, 1)
            print(
                f"[BUBBLE_LAYOUT] 글자 확대 재레이아웃: "
                f"{base.font_size}px → {selected.font_size}px "
                f"(요청={scale:.2f}x, 실제={actual_scale:.2f}x, shape={keep_shape})"
            )
            return selected, alternatives
        print(
            f"[BUBBLE_LAYOUT] {factor:.2f}x 후보가 캔버스에 맞지 않아 축소 재시도: "
            f"font={selected.font_size}, overflow={selected.overflow_ratio:.4f}"
        )

    print(
        f"[BUBBLE_LAYOUT] 확대 가능한 적합 후보 없음 → 기본 {base.font_size}px 유지"
    )
    return base, base_top


def layout_metadata() -> dict:
    return {
        "format": "sbp-bubble-layout-ranker",
        "version": 1,
        "features": list(FEATURE_NAMES),
        "shapes": [
            {
                "name": shape.name,
                "padding_x_em": shape.padding_x_em,
                "padding_y_em": shape.padding_y_em,
                "preferred_aspect": shape.preferred_aspect,
                "min_width_frac": shape.min_width_frac,
                "max_width_frac": shape.max_width_frac,
                "min_height_frac": shape.min_height_frac,
                "max_height_frac": shape.max_height_frac,
            }
            for shape in DEFAULT_SHAPES
        ],
        "hard_constraints": ["measured_font_fit", "max_lines", "punctuation_guard"],
        "output": "candidate_score",
    }


def save_layout_json(path: str | os.PathLike, candidate: LayoutCandidate, top: list[LayoutCandidate]) -> None:
    payload = {
        "selected": candidate.to_dict(include_features=True),
        "alternatives": [item.to_dict(include_features=True) for item in top],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
