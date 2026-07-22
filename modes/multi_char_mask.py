"""다중 캐릭터 Regional Conditioning용 프롬프트/레이아웃 검증과 RGB 마스크 생성."""

from __future__ import annotations

import math
import os
import shutil
import traceback
import hashlib
import json
from typing import Iterable

from PIL import Image, ImageDraw


MASK_CHANNELS = ("R", "G", "B")
DEFAULT_MASK_LOCATION = "region_mask"
DEFAULT_MASK_SIZE = 1024


def _normalized_name(value: object) -> str:
    return str(value or "").strip().casefold()


def _finite_number(value: object, field: str, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name!r}의 {field} 값이 bool입니다")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r}의 {field} 값이 숫자가 아닙니다: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name!r}의 {field} 값이 유한수가 아닙니다: {value!r}")
    return number


def _prompt_text(value: object, field: str, *, required: bool) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} 값이 문자열이 아닙니다: {type(value).__name__}")
    prompt = value.strip()
    if required and not prompt:
        raise ValueError(f"{field} 값이 비어 있습니다")
    return prompt


def validate_multi_char_layout(
    layout: object,
    expected_names: Iterable[object],
    *,
    require_prompt_separation: bool = False,
) -> dict:
    """LLM 레이아웃과 선택적 프롬프트 분리를 검증하고 왼쪽→오른쪽으로 정규화한다."""
    names = [str(name or "").strip() for name in expected_names]
    if not 2 <= len(names) <= len(MASK_CHANNELS):
        raise ValueError(f"다중 캐릭터 수는 2~{len(MASK_CHANNELS)}명이어야 합니다: {len(names)}")
    if any(not name for name in names):
        raise ValueError(f"캐릭터 이름이 비어 있습니다: {names!r}")

    normalized_to_original = {}
    for name in names:
        key = _normalized_name(name)
        if key in normalized_to_original:
            raise ValueError(f"캐릭터 이름이 중복됩니다: {name!r}")
        normalized_to_original[key] = name

    if not isinstance(layout, dict):
        raise ValueError(f"레이아웃 루트가 object가 아닙니다: {type(layout).__name__}")
    background_prompt = _prompt_text(
        layout.get("background_prompt"),
        "background_prompt",
        required=require_prompt_separation,
    )
    raw_regions = layout.get("regions")
    if not isinstance(raw_regions, list):
        raise ValueError("레이아웃 regions가 list가 아닙니다")
    if len(raw_regions) != len(names):
        raise ValueError(
            f"레이아웃 캐릭터 수가 다릅니다: expected={len(names)}, actual={len(raw_regions)}"
        )

    seen = set()
    validated = []
    for source_index, raw in enumerate(raw_regions):
        if not isinstance(raw, dict):
            raise ValueError(f"regions[{source_index}]가 object가 아닙니다")
        raw_name = str(raw.get("name") or "").strip()
        key = _normalized_name(raw_name)
        if key not in normalized_to_original:
            raise ValueError(f"예상하지 않은 캐릭터 이름입니다: {raw_name!r}")
        if key in seen:
            raise ValueError(f"레이아웃 캐릭터가 중복됩니다: {raw_name!r}")
        seen.add(key)

        character_prompt = _prompt_text(
            raw.get("character_prompt"),
            f"regions[{source_index}].character_prompt",
            required=require_prompt_separation,
        )
        x = _finite_number(raw.get("x"), "x", raw_name)
        y = _finite_number(raw.get("y"), "y", raw_name)
        width = _finite_number(raw.get("width"), "width", raw_name)
        height = _finite_number(raw.get("height"), "height", raw_name)
        if x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"{raw_name!r} 영역 좌표/크기가 범위를 벗어났습니다: "
                f"x={x}, y={y}, width={width}, height={height}"
            )
        if x + width > 1.0 + 1e-9 or y + height > 1.0 + 1e-9:
            raise ValueError(
                f"{raw_name!r} 영역이 캔버스를 벗어났습니다: "
                f"right={x + width}, bottom={y + height}"
            )
        validated.append({
            "name": normalized_to_original[key],
            "character_prompt": character_prompt,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "_source_index": source_index,
        })

    missing = set(normalized_to_original) - seen
    if missing:
        missing_names = [normalized_to_original[key] for key in missing]
        raise ValueError(f"레이아웃에서 누락된 캐릭터가 있습니다: {missing_names!r}")

    validated.sort(key=lambda region: (
        region["x"] + region["width"] / 2.0,
        region["_source_index"],
    ))
    regions = []
    for index, region in enumerate(validated):
        normalized = {key: value for key, value in region.items() if not key.startswith("_")}
        normalized["channel"] = MASK_CHANNELS[index]
        regions.append(normalized)
    normalized_layout = {
        "mask_width": DEFAULT_MASK_SIZE,
        "mask_height": DEFAULT_MASK_SIZE,
        "character_order": [region["name"] for region in regions],
        "regions": regions,
    }
    if background_prompt or require_prompt_separation:
        normalized_layout["background_prompt"] = background_prompt
    return normalized_layout


def layout_fingerprint(layout: dict) -> str:
    """검증된 레이아웃의 마스크 결과를 식별하는 안정적인 SHA-256을 반환한다."""
    if not isinstance(layout, dict) or not isinstance(layout.get("regions"), list):
        raise ValueError("fingerprint를 만들 검증된 layout.regions가 없습니다")
    canonical = {
        "mask_width": int(layout.get("mask_width") or DEFAULT_MASK_SIZE),
        "mask_height": int(layout.get("mask_height") or DEFAULT_MASK_SIZE),
        "regions": [{
            "name": str(region.get("name") or ""),
            "channel": str(region.get("channel") or ""),
            "x": float(region.get("x")),
            "y": float(region.get("y")),
            "width": float(region.get("width")),
            "height": float(region.get("height")),
        } for region in layout["regions"]],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_mask_directory(comfy_input_dir: str, mask_location: str = DEFAULT_MASK_LOCATION) -> str:
    raw_root = str(comfy_input_dir or "").strip()
    if not raw_root:
        raise ValueError("ComfyUI input 폴더가 비어 있습니다")
    root = os.path.realpath(os.path.abspath(raw_root))
    if not os.path.isdir(root):
        raise ValueError(f"ComfyUI input 폴더가 유효하지 않습니다: {comfy_input_dir!r}")
    relative = str(mask_location or DEFAULT_MASK_LOCATION).strip()
    if not relative:
        relative = DEFAULT_MASK_LOCATION
    target = os.path.realpath(os.path.abspath(os.path.join(root, relative)))
    try:
        inside_root = os.path.commonpath([root, target]) == root
    except ValueError as exc:
        raise ValueError(f"마스크 경로가 ComfyUI input 폴더와 다른 드라이브입니다: {relative!r}") from exc
    if not inside_root or target == root:
        raise ValueError(f"마스크 경로가 ComfyUI input 폴더 밖을 가리킵니다: {relative!r}")
    return target


def _clear_directory(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for entry in os.scandir(directory):
        path = entry.path
        if entry.is_symlink() or entry.is_file(follow_symlinks=False):
            os.remove(path)
        elif entry.is_dir(follow_symlinks=False):
            shutil.rmtree(path)
        else:
            os.remove(path)


def render_region_mask(layout: dict, width: int | None = None, height: int | None = None) -> Image.Image:
    """각 캐릭터 영역을 R/G/B 독립 채널에 그려 겹침도 보존한다."""
    regions = layout.get("regions") if isinstance(layout, dict) else None
    if not isinstance(regions, list) or not 2 <= len(regions) <= len(MASK_CHANNELS):
        raise ValueError("검증된 2~3개 regions가 필요합니다")
    width = int(width or layout.get("mask_width") or DEFAULT_MASK_SIZE)
    height = int(height or layout.get("mask_height") or DEFAULT_MASK_SIZE)
    if width < 1 or height < 1:
        raise ValueError(f"마스크 크기가 올바르지 않습니다: {width}x{height}")

    channels = [Image.new("L", (width, height), 0) for _ in MASK_CHANNELS]
    for index, region in enumerate(regions):
        name = str(region.get("name") or f"character-{index + 1}")
        x = _finite_number(region.get("x"), "x", name)
        y = _finite_number(region.get("y"), "y", name)
        box_width = _finite_number(region.get("width"), "width", name)
        box_height = _finite_number(region.get("height"), "height", name)
        left = max(0, min(width - 1, int(math.floor(x * width))))
        top = max(0, min(height - 1, int(math.floor(y * height))))
        right = max(left, min(width - 1, int(math.ceil((x + box_width) * width)) - 1))
        bottom = max(top, min(height - 1, int(math.ceil((y + box_height) * height)) - 1))
        ImageDraw.Draw(channels[index]).rectangle((left, top, right, bottom), fill=255)
    return Image.merge("RGB", tuple(channels))


def prepare_region_mask(
    comfy_input_dir: str,
    multi_char_context: dict,
    mask_location: str = DEFAULT_MASK_LOCATION,
) -> str:
    """큐 실행 직전에 기존 폴더를 비우고 단 하나의 RGB PNG를 원자적으로 배치한다."""
    try:
        if not isinstance(multi_char_context, dict) or not multi_char_context.get("enable"):
            raise ValueError("활성화된 illustration_multi_char 컨텍스트가 필요합니다")
        layout = multi_char_context.get("layout")
        expected_names = multi_char_context.get("character_order") or [
            char.get("name")
            for char in (multi_char_context.get("characters") or [])
            if isinstance(char, dict)
        ]
        normalized = validate_multi_char_layout(layout, expected_names)
        location = str(multi_char_context.get("mask_location") or mask_location)
        target_dir = resolve_mask_directory(comfy_input_dir, location)
        _clear_directory(target_dir)

        image = render_region_mask(normalized)
        final_path = os.path.join(target_dir, "region_mask.png")
        temporary_path = os.path.join(target_dir, ".region_mask.tmp.png")
        image.save(temporary_path, format="PNG")
        os.replace(temporary_path, final_path)
        print(
            f"[MULTI_CHAR:MASK] 실행 시점 마스크 준비 완료: path={final_path}, "
            f"order={normalized['character_order']}, size={image.width}x{image.height}, "
            f"fingerprint={layout_fingerprint(normalized)[:12]}"
        )
        return final_path
    except Exception as exc:
        print(
            f"[MULTI_CHAR:MASK] 마스크 준비 실패: input={comfy_input_dir!r}, "
            f"location={mask_location!r}, error={exc}"
        )
        traceback.print_exc()
        raise
