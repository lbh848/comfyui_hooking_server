"""Shared character and background composition settings for the emotion workshop."""
from __future__ import annotations

import math


DEFAULT_CHARACTER_POSITIONS = {"R": 75, "M": 50, "L": 25}
DEFAULT_COMPOSITION_SETTINGS = {
    "visible_width": 100,
    "visible_height": 100,
    "visible_center_x": None,
    "visible_center_y": None,
    "character_height": 100,
    "character_align_x": 50,
    "character_align_y": 50,
    "character_offset_x": 0,
    "character_offset_y": 0,
    "character_positions": DEFAULT_CHARACTER_POSITIONS,
}
_PREVIOUS_SETTINGS = {
    "background_fit", "background_color", "background_zoom", "background_crop_x", "background_crop_y",
    "character_scale", "character_x", "character_y",
}


def normalize_composition_settings(value: dict | None = None) -> dict:
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - (DEFAULT_COMPOSITION_SETTINGS.keys() | _PREVIOUS_SETTINGS):
        raise ValueError("합성 설정 형식을 확인하세요.")
    settings = {key: value.get(key, default) for key, default in DEFAULT_COMPOSITION_SETTINGS.items()}
    positions = settings["character_positions"]
    if (not isinstance(positions, dict) or set(positions) != set(DEFAULT_CHARACTER_POSITIONS)
            or any(type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 100
                   for number in positions.values())):
        raise ValueError("R/M/L 각각의 가로 위치를 0~100% 사이의 숫자로 지정하세요.")
    settings["character_positions"] = dict(positions)
    # Retain the saved zoom amount when moving to a background-relative viewport.
    if "background_zoom" in value and "visible_width" not in value and "visible_height" not in value:
        zoom = value["background_zoom"]
        if type(zoom) not in (int, float) or not math.isfinite(zoom) or not 100 <= zoom <= 400:
            raise ValueError("배경 확대 값을 확인하세요.")
        settings.update(visible_width=10000 / zoom, visible_height=10000 / zoom)
    if "character_scale" in value and "character_height" not in value:
        settings["character_height"] = value["character_scale"]
    for key, label, minimum, maximum in (
        ("visible_width", "보일 영역 너비", 1, 100),
        ("visible_height", "보일 영역 높이", 1, 100),
        ("visible_center_x", "보일 영역 가로 중심", 0, 100),
        ("visible_center_y", "보일 영역 세로 중심", 0, 100),
        ("character_height", "인물 높이", 1, 1000),
        ("character_align_x", "인물 가로 배치", 0, 100),
        ("character_align_y", "인물 세로 배치", 0, 100),
        ("character_offset_x", "인물 가로 이동", -10000, 10000),
        ("character_offset_y", "인물 세로 이동", -10000, 10000),
    ):
        number = settings[key]
        if key in ("visible_center_x", "visible_center_y") and number is None:
            continue
        if type(number) not in (int, float) or not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"{label}는 {minimum}~{maximum}% 사이의 숫자여야 합니다.")
    return settings


def normalize_view_center(value=None):
    if value is None:
        return {"x": .5, "y": .5}
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        raise ValueError("배경 보일 영역의 중심 좌표를 확인하세요.")
    if any(type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 1
           for number in value.values()):
        raise ValueError("배경 보일 영역의 중심 좌표는 0~1 사이여야 합니다.")
    return dict(value)


def background_crop_box(source_size, settings, view_center=None):
    """Generation uses the scene's center; manual positions are preview fallbacks."""
    source_w, source_h = source_size
    width = max(1, min(source_w, round(source_w * settings["visible_width"] / 100)))
    height = max(1, min(source_h, round(source_h * settings["visible_height"] / 100)))
    center = normalize_view_center(view_center)
    if view_center is None:
        for axis in ("x", "y"):
            manual = settings.get(f"visible_center_{axis}")
            if manual is not None:
                center[axis] = manual / 100
    left = max(0, min(source_w - width, round(center["x"] * source_w - width / 2)))
    top = max(0, min(source_h - height, round(center["y"] * source_h - height / 2)))
    return left, top, left + width, top + height


def normalize_character_position(value):
    if value not in ("R", "M", "L"):
        raise ValueError("인물 가로 위치는 R, M, L 중 하나여야 합니다.")
    return value


def character_box(character_size, visible_size, settings, *, position=None):
    """Keep the cutout's aspect ratio, including figures taller than the full background."""
    visible_w, visible_h = visible_size
    height = max(1, round(visible_h * settings["character_height"] / 100))
    width = max(1, round(character_size[0] * height / character_size[1]))
    if position is None:
        left = round((visible_w - width) * settings["character_align_x"] / 100
                     + visible_w * settings.get("character_offset_x", 0) / 100)
    else:
        # Each scene refers to one of the user's independently configured center points.
        points = settings.get("character_positions", DEFAULT_CHARACTER_POSITIONS)
        left = round(visible_w * points[normalize_character_position(position)] / 100 - width / 2)
    top = (round((visible_h - height) / 2) if position is not None else
           round((visible_h - height) * settings["character_align_y"] / 100
                 + visible_h * settings.get("character_offset_y", 0) / 100))
    return left, top, width, height
