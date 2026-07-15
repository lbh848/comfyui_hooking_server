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
_session = None
_session_path = None


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


def load_font(size: int, font_path: str | os.PathLike | None = None) -> ImageFont.ImageFont:
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
) -> WrapResult | None:
    text = text.strip()
    if not text:
        return WrapResult(("",), 1.0, 1.0)

    n = len(text)

    def width(value: str) -> float:
        cached = width_cache.get(value)
        if cached is None:
            cached = float(draw.textlength(value, font=font))
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
                        token_start = text.rfind(" ", 0, end) + 1
                        token_end = text.find(" ", end)
                        if token_end < 0:
                            token_end = n
                        # Split inside a word only when that word cannot fit on
                        # an empty line.  This prevents Korean endings such as
                        # "밖으/로" while retaining a fallback for long URLs.
                        if width(text[token_start:token_end]) <= max_width + 1e-6:
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
) -> WrapResult | None:
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Fast impossibility check.  It avoids running the dynamic program for
    # every candidate when a very long text cannot fit within max_lines.
    minimum_lines = 0
    for paragraph in paragraphs:
        paragraph_width = float(draw.textlength(paragraph.strip() or " ", font=font))
        minimum_lines += max(1, math.ceil(paragraph_width / max(max_width, 1.0)))
    if minimum_lines > max_lines:
        return None
    all_lines: list[str] = []
    boundary_scores: list[float] = []
    punctuation_scores: list[float] = []
    remaining = max_lines
    for paragraph in paragraphs:
        wrapped = _wrap_paragraph(paragraph, draw, font, max_width, remaining, width_cache)
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
) -> WrapResult:
    """Unlimited-line fallback used to report a clean ``fits=False`` result."""
    lines: list[str] = []
    penalties: list[float] = []
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        start = 0
        while start < len(paragraph):
            last_fit = start + 1
            preferred_end = None
            for end in range(start + 1, len(paragraph) + 1):
                line = paragraph[start:end].rstrip()
                if float(draw.textlength(line, font=font)) > max_width and end > start + 1:
                    break
                last_fit = end
                next_start = _next_non_space(paragraph, end)
                if _boundary_penalty(paragraph, end, next_start) <= 0.05:
                    preferred_end = end
            end = preferred_end if preferred_end and preferred_end > start else last_fit
            next_start = _next_non_space(paragraph, end)
            line = paragraph[start:end].rstrip()
            lines.append(line)
            penalties.append(_boundary_penalty(paragraph, end, next_start))
            start = next_start
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
) -> tuple[list[float], float, float]:
    widths = [float(draw.textlength(line or " ", font=font)) for line in lines]
    text = "\n".join(lines)
    box = draw.multiline_textbbox((0, 0), text or " ", font=font, spacing=spacing, align="center")
    return widths, float(box[2] - box[0]), float(box[3] - box[1])


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
        font = load_font(font_size, font_path)
        spacing = max(2, int(round(font_size * 0.27)))
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
                    text, draw, font, float(max_text_width), max_lines, width_cache
                )
                if wrapped is None:
                    continue
                line_widths, text_width, text_height = _measure_lines(
                    wrapped.lines, draw, font, spacing
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
        font = load_font(font_size, font_path)
        spacing = max(2, int(round(font_size * 0.27)))
        for shape in shapes:
            max_text_width = (
                shape.max_width_frac * canvas_width - 2.0 * shape.padding_x_em * font_size
            )
            wrapped = _greedy_wrap_text(text, draw, font, max(max_text_width, 1.0))
            line_widths, text_width, text_height = _measure_lines(
                wrapped.lines, draw, font, spacing
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


def _get_layout_session(onnx_path: str | os.PathLike):
    """레이아웃 ONNX 세션을 프로세스 동안 재사용한다."""
    global _session, _session_path
    path = os.path.abspath(str(onnx_path))
    if _session is not None and _session_path == path:
        return _session
    if not os.path.isfile(path):
        print(f"[BUBBLE_LAYOUT] ONNX 모델 없음: {path}")
        return None
    try:
        import onnxruntime as ort

        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _session_path = path
        print(f"[BUBBLE_LAYOUT] ONNX 세션 로드: {path}")
        return _session
    except Exception as e:
        print(f"[BUBBLE_LAYOUT] ONNX 세션 로드 실패({path}): {e}")
        traceback.print_exc()
        return None


def _onnx_scores(features: np.ndarray, onnx_path: str | os.PathLike) -> np.ndarray | None:
    session = _get_layout_session(onnx_path)
    if session is None:
        print("[BUBBLE_LAYOUT] 세션이 없어 초기 규칙 점수로 폴백")
        return None

    try:
        input_name = session.get_inputs()[0].name
        return np.asarray(session.run(None, {input_name: features})[0], dtype=np.float32)
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
    min_font_size: int | None = None,
    max_font_size: int | None = None,
    max_lines: int = 7,
    top_k: int = 5,
) -> tuple[LayoutCandidate, list[LayoutCandidate]]:
    candidates = generate_layout_candidates(
        text,
        canvas_size,
        font_path,
        shape_hint=shape_hint,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        max_lines=max_lines,
    )
    feature_array = np.asarray([candidate.features for candidate in candidates], dtype=np.float32)
    if onnx_path is None:
        default = Path(_MODEL_PATH)
        onnx_path = default if default.is_file() else None
    scores = _onnx_scores(feature_array, onnx_path) if onnx_path else None
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
