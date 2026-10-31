"""다중 캐릭터 Regional Conditioning용 프롬프트/레이아웃 검증과 RGB 마스크 생성."""

from __future__ import annotations

import math
import os
import shutil
import traceback
import hashlib
import json
from copy import deepcopy
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


MASK_CHANNELS = ("R", "G", "B")
DEFAULT_MASK_LOCATION = "region_mask"
DEFAULT_MASK_SIZE = 1024
MULTI_CHAR_SNAPSHOT_VERSION = 1
MASK_RENDER_VERSION = 5
SOLO_MASK_SKIP_AREA_RATIO = 0.30


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
        if required:
            raise ValueError(f"{field} 값이 문자열이 아닙니다: {type(value).__name__}")
        import illustration_flow
        reason = f"{field} 문자열 형식 차이; 제공된 값을 유지"
        print(f"[MULTI_CHAR:MASK] 경고: {reason}, input={value!r}")
        illustration_flow.llm_metadata(validation_warnings=[{
            "stage": "MULTI-CHAR-MASK", "reason": reason, "input": deepcopy(value)}])
        return json.dumps(value, ensure_ascii=False)
    prompt = value.strip()
    if required and not prompt:
        raise ValueError(f"{field} 값이 비어 있습니다")
    return prompt


def validate_multi_char_layout(
    layout: object,
    expected_names: Iterable[object],
    *,
    require_prompt_separation: bool = False,
    require_character_prompt: bool | None = None,
) -> dict:
    """LLM 레이아웃과 선택적 2-pass 프롬프트를 검증해 왼쪽→오른쪽으로 정규화한다."""
    if require_character_prompt is None:
        require_character_prompt = require_prompt_separation
    names = [str(name or "").strip() for name in expected_names]
    if not 1 <= len(names) <= len(MASK_CHANNELS):
        raise ValueError(f"영역 캐릭터 수는 1~{len(MASK_CHANNELS)}명이어야 합니다: {len(names)}")
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
        required=False,
    )
    composition_prompt = _prompt_text(
        layout.get("composition_prompt"),
        "composition_prompt",
        required=False,
    )
    raw_regions = layout.get("regions")
    if not isinstance(raw_regions, list):
        raise ValueError("레이아웃 regions가 list가 아닙니다")
    if len(raw_regions) != len(names):
        raise ValueError(
            f"레이아웃 캐릭터 수가 다릅니다: expected={len(names)}, actual={len(raw_regions)}"
        )

    panels = []
    for index, panel in enumerate(layout.get("panels") or []):
        bounds = {field: _finite_number(panel.get(field), field, f"panel {index + 1}")
                  for field in ("x", "y", "width", "height")}
        if (bounds["x"] < 0 or bounds["y"] < 0 or bounds["width"] <= 0 or bounds["height"] <= 0
                or bounds["x"] + bounds["width"] > 1 + 1e-9
                or bounds["y"] + bounds["height"] > 1 + 1e-9):
            raise ValueError(f"컷 영역이 캔버스를 벗어났습니다: {panel!r}")
        panels.append({"panel": panel.get("panel", index + 1), **bounds})

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
            required=False,
        )
        positive = _prompt_text(
            raw.get("positive"),
            f"regions[{source_index}].positive",
            required=False,
        )
        negative = _prompt_text(
            raw.get("negative"),
            f"regions[{source_index}].negative",
            required=False,
        )
        boxes = []
        for box in raw.get("boxes") or []:
            panel = next((value for value in panels if value["panel"] == box.get("panel")), None)
            if panel is None:
                raise ValueError(f"인물 영역의 컷 좌표가 없습니다: name={raw_name!r}, box={box!r}")
            bx, by, bw, bh = (_finite_number(box.get(field), field, raw_name)
                              for field in ("x", "y", "width", "height"))
            left, top = max(bx, panel["x"]), max(by, panel["y"])
            right = min(bx + bw, panel["x"] + panel["width"])
            bottom = min(by + bh, panel["y"] + panel["height"])
            if right <= left or bottom <= top:
                raise ValueError(f"인물 영역이 해당 컷과 겹치지 않습니다: name={raw_name!r}, box={box!r}")
            if (left, top, right, bottom) != (bx, by, bx + bw, by + bh):
                print(f"[MULTI_CHAR:MASK] 인물 영역을 컷 안으로 제한: name={raw_name!r}, box={box!r}")
            boxes.append({"panel": panel["panel"], "x": left, "y": top,
                          "width": right - left, "height": bottom - top})
        geometry = raw
        if panels and not boxes:
            raise ValueError(f"컷별 인물 영역이 없습니다: name={raw_name!r}, region={raw!r}")
        if boxes:
            left, top = min(box["x"] for box in boxes), min(box["y"] for box in boxes)
            geometry = {"x": left, "y": top,
                        "width": max(box["x"] + box["width"] for box in boxes) - left,
                        "height": max(box["y"] + box["height"] for box in boxes) - top}
        x = _finite_number(geometry.get("x"), "x", raw_name)
        y = _finite_number(geometry.get("y"), "y", raw_name)
        width = _finite_number(geometry.get("width"), "width", raw_name)
        height = _finite_number(geometry.get("height"), "height", raw_name)
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
        normalized_region = {
            "name": normalized_to_original[key],
            "character_prompt": character_prompt,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "_source_index": source_index,
        }
        if boxes:
            normalized_region["boxes"] = boxes
        if "positive" in raw or positive:
            normalized_region["positive"] = positive
        if "negative" in raw or negative:
            normalized_region["negative"] = negative
        if "outfit_state" in raw:
            outfit_state = raw.get("outfit_state")
            if outfit_state is not None and not isinstance(outfit_state, dict):
                print(f"[MULTI_CHAR:MASK] outfit_state 형식 차이; 원문 유지: {outfit_state!r}")
            normalized_region["outfit_state"] = deepcopy(outfit_state)
        validated.append(normalized_region)

    missing = set(normalized_to_original) - seen
    if missing:
        missing_names = [normalized_to_original[key] for key in missing]
        raise ValueError(f"레이아웃에서 누락된 캐릭터가 있습니다: {missing_names!r}")

    if not panels:
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
        "mask_width": int(layout.get("mask_width") or DEFAULT_MASK_SIZE),
        "mask_height": int(layout.get("mask_height") or DEFAULT_MASK_SIZE),
        "mask_render_version": int(layout.get("mask_render_version") or MASK_RENDER_VERSION),
        "character_order": [region["name"] for region in regions],
        "regions": regions,
    }
    if panels:
        normalized_layout["panels"] = panels
    if layout.get("dialogue_mask"):
        normalized_layout["dialogue_mask"] = layout["dialogue_mask"]
        normalized_layout["dialogue"] = deepcopy(layout.get("dialogue") or [])
    if background_prompt or require_prompt_separation:
        normalized_layout["background_prompt"] = background_prompt
    if composition_prompt or require_prompt_separation:
        normalized_layout["composition_prompt"] = composition_prompt
    if (require_prompt_separation and (not background_prompt or not composition_prompt)
            or require_character_prompt and any(not region["character_prompt"] for region in regions)):
        import illustration_flow
        reason = "일부 분리 프롬프트 누락; 제공된 프롬프트와 마스크로 진행"
        print(f"[MULTI_CHAR:MASK] 경고: {reason}, input={layout!r}")
        illustration_flow.llm_metadata(validation_warnings=[{
            "stage": "MULTI-CHAR-MASK", "reason": reason, "input": deepcopy(layout)}])
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
            **({"boxes": region["boxes"]} if region.get("boxes") else {}),
        } for region in layout["regions"]],
    }
    if layout.get("panels"):
        canonical["panels"] = layout["panels"]
    if layout.get("dialogue_mask"):
        canonical["dialogue_mask"] = layout["dialogue_mask"]
    render_version = int(layout.get("mask_render_version") or MASK_RENDER_VERSION)
    if render_version >= 2:
        # Legacy snapshots keep their original geometry-only fingerprint.
        # New masks also identify the pixel ownership algorithm they use.
        canonical["mask_render_version"] = render_version
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_multi_char_prompt_payload(positive: object) -> dict | None:
    """빌드 프롬프트의 구조화된 [MULTI_CHAR] JSON을 읽는다.

    자연어 키워드를 판정하지 않고 프롬프트 빌더가 생성한 제어 블록만 파싱한다.
    블록이 없으면 단일 캐릭터/구형 비다인 프롬프트로 보고 None을 반환한다.
    """
    if not isinstance(positive, str):
        raise ValueError(
            f"[MULTI_CHAR]를 읽을 positive가 문자열이 아닙니다: {type(positive).__name__}"
        )
    header = "[MULTI_CHAR]"
    lines = positive.splitlines()
    indexes = [index for index, line in enumerate(lines) if line.strip() == header]
    if not indexes:
        return None
    if len(indexes) != 1:
        raise ValueError(f"[MULTI_CHAR] 블록이 {len(indexes)}개라 안전하게 해석할 수 없습니다")

    start = indexes[0] + 1
    payload_lines = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        payload_lines.append(line)
    raw_payload = "\n".join(payload_lines).strip()
    if not raw_payload:
        raise ValueError("[MULTI_CHAR] JSON이 비어 있습니다")
    try:
        payload = json.loads(raw_payload)
    except Exception as exc:
        print(f"[MULTI_CHAR:BACKUP] 프롬프트 제어 블록 JSON 파싱 실패: {exc}")
        traceback.print_exc()
        raise ValueError(f"[MULTI_CHAR] JSON 파싱 실패: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"[MULTI_CHAR] JSON 루트가 object가 아닙니다: {type(payload).__name__}"
        )
    return payload


def should_skip_solo_region_mask(layout: object, settings: dict | None = None) -> bool:
    """Skip a single region above the enabled user-selected area threshold."""
    regions = layout.get("regions") if isinstance(layout, dict) else None
    if not isinstance(regions, list) or len(regions) != 1:
        return False
    if layout.get("panels") or layout.get("dialogue_mask"):
        print("[MULTI_CHAR:MASK] 만화의 컷별 인물 영역 유지")
        return False
    settings = settings or {}
    if not settings.get("single_char_mask_skip_enabled", True):
        print("[MULTI_CHAR:MASK] 면적 기준 OFF: 1인 영역의 마스크 유지")
        return False
    region = regions[0]
    name = str(region.get("name") or "")
    threshold = _finite_number(
        settings.get("single_char_mask_skip_area_percent", SOLO_MASK_SKIP_AREA_RATIO * 100),
        "single_char_mask_skip_area_percent", name,
    ) / 100
    area_ratio = (
        _finite_number(region.get("width"), "width", name)
        * _finite_number(region.get("height"), "height", name)
    )
    # Preserve the strict boundary despite multiplication rounding.
    skip = area_ratio > threshold and not math.isclose(
        area_ratio, threshold, rel_tol=0.0, abs_tol=1e-12,
    )
    if skip:
        print(
            f"[MULTI_CHAR:MASK] solo 마스크 생략: name={name!r}, "
            f"area={area_ratio:.2%}, threshold={threshold:.2%}"
        )
    return skip


def apply_solo_region_mask_policy(
    positive: str, context: dict, settings: dict | None = None,
) -> tuple[str, dict | None]:
    """Disable the existing control and backup snapshot together for a large solo region."""
    if not should_skip_solo_region_mask(context.get("layout"), settings):
        return positive, context
    try:
        payload = extract_multi_char_prompt_payload(positive)
        if payload is None:
            raise ValueError("solo 마스크를 생략할 [MULTI_CHAR] 제어 블록이 없습니다")
        payload["enable"] = False
        lines = positive.splitlines(keepends=True)
        start = next(index + 1 for index, line in enumerate(lines)
                     if line.strip() == "[MULTI_CHAR]")
        end = next((index for index in range(start, len(lines))
                    if lines[index].strip().startswith("[")
                    and lines[index].strip().endswith("]")), len(lines))
        last = lines[end - 1]
        ending = "\r\n" if last.endswith("\r\n") else "\n" if last.endswith("\n") else ""
        lines[start:end] = [json.dumps(payload, ensure_ascii=False) + ending]
        return "".join(lines), None
    except Exception as exc:
        print(
            f"[MULTI_CHAR:MASK] solo 마스크 비활성화 실패: "
            f"context={context!r}, positive={positive!r}, error={exc}"
        )
        traceback.print_exc()
        raise


def normalize_multi_char_snapshot(context: object) -> dict | None:
    """큐 컨텍스트를 백업 가능한 최소·정규 형식으로 바꾼다."""
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError(
            f"다중 캐릭터 컨텍스트가 object가 아닙니다: {type(context).__name__}"
        )
    if not context.get("enable"):
        return None

    expected_names = [
        str(name or "").strip()
        for name in (context.get("character_order") or [])
    ]
    if not expected_names:
        expected_names = [
            str(character.get("name") or "").strip()
            for character in (context.get("characters") or [])
            if isinstance(character, dict)
        ]
    source_layout = context.get("layout")
    if isinstance(source_layout, dict):
        source_layout = {"mask_render_version": 1, **source_layout}
    normalized_layout = validate_multi_char_layout(
        source_layout,
        expected_names,
        require_prompt_separation=False,
    )
    normalized_order = list(normalized_layout["character_order"])
    if [name.casefold() for name in expected_names] != [
        name.casefold() for name in normalized_order
    ]:
        import illustration_flow
        reason = f"선언 순서와 실제 마스크 순서 차이: {expected_names!r} -> {normalized_order!r}"
        print(f"[MULTI_CHAR:MASK] 경고: {reason}")
        illustration_flow.llm_metadata(validation_warnings=[{
            "stage": "MULTI-CHAR-MASK", "reason": reason, "input": deepcopy(context)}])

    # 재생성 스냅샷에는 고정 마스크에 필요한 기하 정보만 저장한다. 캐릭터별/배경
    # 프롬프트는 [MULTI_CHAR] 제어 블록이 수정 재생성마다 갱신하므로 중복 저장하지 않는다.
    mask_layout = {
        "mask_width": int(normalized_layout.get("mask_width") or DEFAULT_MASK_SIZE),
        "mask_height": int(normalized_layout.get("mask_height") or DEFAULT_MASK_SIZE),
        "mask_render_version": normalized_layout["mask_render_version"],
        "character_order": normalized_order,
        "regions": [
            {
                "name": str(region.get("name") or ""),
                "x": float(region.get("x")),
                "y": float(region.get("y")),
                "width": float(region.get("width")),
                "height": float(region.get("height")),
                "channel": str(region.get("channel") or ""),
                **({"boxes": deepcopy(region["boxes"])} if region.get("boxes") else {}),
            }
            for region in normalized_layout["regions"]
        ],
    }
    if normalized_layout.get("panels"):
        mask_layout["panels"] = deepcopy(normalized_layout["panels"])
    if normalized_layout.get("dialogue_mask"):
        mask_layout["dialogue_mask"] = normalized_layout["dialogue_mask"]
        mask_layout["dialogue"] = deepcopy(normalized_layout.get("dialogue") or [])
    fingerprint = layout_fingerprint(mask_layout)
    declared_fingerprint = str(context.get("mask_fingerprint") or "").strip()
    if declared_fingerprint and declared_fingerprint != fingerprint:
        import illustration_flow
        reason = f"저장된 마스크 지문 차이; 실제 레이아웃 사용: {declared_fingerprint} -> {fingerprint}"
        print(f"[MULTI_CHAR:MASK] 경고: {reason}")
        illustration_flow.llm_metadata(validation_warnings=[{
            "stage": "MULTI-CHAR-MASK", "reason": reason, "input": deepcopy(context)}])
    return {
        "version": MULTI_CHAR_SNAPSHOT_VERSION,
        "enable": True,
        "char_num": len(normalized_order),
        "character_order": normalized_order,
        "mask_location": str(
            context.get("mask_location") or DEFAULT_MASK_LOCATION
        ).strip() or DEFAULT_MASK_LOCATION,
        "mask_fingerprint": fingerprint,
        "layout": mask_layout,
    }


def validate_multi_char_prompt_context(positive: object, context: object) -> dict:
    """프롬프트 제어 블록과 백업 마스크 스냅샷이 같은 작업인지 검증한다."""
    snapshot = normalize_multi_char_snapshot(context)
    if snapshot is None:
        raise ValueError("활성화된 다중 캐릭터 마스크 스냅샷이 없습니다")
    payload = extract_multi_char_prompt_payload(positive)
    # The snapshot preserves editable geometry. The prompt independently
    # chooses whether to apply it for this generation, including ON/OFF checks.
    if not isinstance(payload, dict) or not isinstance(payload.get("enable"), bool):
        raise ValueError("프롬프트의 [MULTI_CHAR] 제어 블록에 ON/OFF 설정이 없습니다")

    prompt_names = [
        str(name or "").strip()
        for name in (payload.get("char_name_list") or [])
    ]
    snapshot_names = list(snapshot["character_order"])
    if [name.casefold() for name in prompt_names] != [
        name.casefold() for name in snapshot_names
    ]:
        raise ValueError(
            "프롬프트 캐릭터 순서와 마스크 순서가 다릅니다: "
            f"prompt={prompt_names}, mask={snapshot_names}"
        )
    prompt_char_num = payload.get("char_num")
    if isinstance(prompt_char_num, bool) or prompt_char_num != snapshot["char_num"]:
        raise ValueError(
            "프롬프트 캐릭터 수와 마스크 캐릭터 수가 다릅니다: "
            f"prompt={prompt_char_num!r}, mask={snapshot['char_num']}"
        )
    prompt_fingerprint = str(payload.get("mask_fingerprint") or "").strip()
    if prompt_fingerprint != snapshot["mask_fingerprint"]:
        import illustration_flow
        reason = f"프롬프트 마스크 지문 차이; 연결 가능한 실제 레이아웃 사용: {prompt_fingerprint!r}"
        print(f"[MULTI_CHAR:MASK] 경고: {reason}")
        illustration_flow.llm_metadata(validation_warnings=[{
            "stage": "MULTI-CHAR-MASK", "reason": reason, "input": deepcopy(payload)}])
    return snapshot


def remap_multi_char_snapshot(context: object, character_names: object) -> dict:
    """고정 마스크의 기하는 유지하고 슬롯별 캐릭터 이름만 교체한다.

    캐릭터 이름도 레이아웃 지문에 포함되므로 새 이름에 맞춰 지문을 다시 계산한다.
    원본 스냅샷은 수정하지 않는다.
    """
    snapshot = normalize_multi_char_snapshot(context)
    if snapshot is None:
        raise ValueError("재매핑할 다중 캐릭터 마스크 스냅샷이 없습니다")

    names = [str(name or "").strip() for name in (character_names or [])]
    if len(names) != snapshot["char_num"] or any(not name for name in names):
        raise ValueError(
            "마스크 재매핑 캐릭터 수가 기존 슬롯 수와 다릅니다: "
            f"expected={snapshot['char_num']}, actual={len(names)}, names={names}"
        )
    folded = [name.casefold() for name in names]
    if len(set(folded)) != len(folded):
        raise ValueError(f"마스크 재매핑 캐릭터가 중복되었습니다: {names}")

    remapped = deepcopy(snapshot)
    remapped["character_order"] = list(names)
    remapped["mask_fingerprint"] = ""
    layout = remapped.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("마스크 재매핑 레이아웃이 object가 아닙니다")
    layout["character_order"] = list(names)
    regions = layout.get("regions")
    if not isinstance(regions, list) or len(regions) != len(names):
        raise ValueError(
            "마스크 재매핑 영역 수가 캐릭터 수와 다릅니다: "
            f"regions={len(regions) if isinstance(regions, list) else 'invalid'}, names={len(names)}"
        )
    for index, name in enumerate(names):
        if not isinstance(regions[index], dict):
            raise ValueError(f"마스크 재매핑 영역이 object가 아닙니다: index={index}")
        regions[index]["name"] = name

    normalized = normalize_multi_char_snapshot(remapped)
    if normalized is None:
        raise ValueError("다중 캐릭터 마스크 재매핑 결과가 비활성 상태입니다")
    print(
        "[MULTI_CHAR:REMAP] 고정 마스크 캐릭터 재매핑 완료: "
        f"old={snapshot['character_order']}, new={names}, "
        f"fingerprint={normalized['mask_fingerprint'][:12]}"
    )
    return normalized


def recover_multi_char_snapshot_from_sessions(
    session_dir: str,
    prompt_payload: object,
    *,
    mask_location: str = DEFAULT_MASK_LOCATION,
) -> dict:
    """구버전 백업의 지문과 일치하는 레이아웃을 영속 세션에서 복구한다."""
    if not isinstance(prompt_payload, dict) or prompt_payload.get("enable") is not True:
        raise ValueError("레거시 복구에 활성화된 [MULTI_CHAR] payload가 필요합니다")
    expected_names = [
        str(name or "").strip()
        for name in (prompt_payload.get("char_name_list") or [])
    ]
    if not 1 <= len(expected_names) <= len(MASK_CHANNELS) or any(
        not name for name in expected_names
    ):
        raise ValueError(f"레거시 복구 캐릭터 순서가 올바르지 않습니다: {expected_names!r}")
    expected_fingerprint = str(prompt_payload.get("mask_fingerprint") or "").strip()
    if len(expected_fingerprint) != 64:
        raise ValueError(
            f"레거시 복구 마스크 지문이 올바르지 않습니다: {expected_fingerprint!r}"
        )
    root = str(session_dir or "").strip()
    if not root or not os.path.isdir(root):
        raise ValueError(f"삽화 세션 폴더가 유효하지 않습니다: {session_dir!r}")

    for entry in sorted(os.scandir(root), key=lambda item: item.name, reverse=True):
        if not entry.is_file() or not entry.name.lower().endswith(".json"):
            continue
        try:
            with open(entry.path, "r", encoding="utf-8") as fp:
                session = json.load(fp)
        except Exception as exc:
            print(
                f"[MULTI_CHAR:BACKUP] 레거시 세션 읽기 실패: "
                f"file={entry.path}, error={exc}"
            )
            traceback.print_exc()
            continue
        if not isinstance(session, dict):
            print(
                f"[MULTI_CHAR:BACKUP] 레거시 세션 형식 오류로 건너뜀: "
                f"file={entry.path}, type={type(session).__name__}"
            )
            continue
        for item_index, item in enumerate(session.get("items") or []):
            if not isinstance(item, dict):
                print(
                    f"[MULTI_CHAR:BACKUP] 레거시 세션 item 형식 오류로 건너뜀: "
                    f"file={entry.path}, index={item_index}, type={type(item).__name__}"
                )
                continue
            layout = item.get("multi_char_layout")
            if not isinstance(layout, dict):
                continue
            declared_candidate_names = [
                str(name or "").strip()
                for name in (layout.get("character_order") or [])
            ]
            candidate_names = declared_candidate_names
            if not candidate_names:
                candidate_names = [
                    str(region.get("name") or "").strip()
                    for region in (layout.get("regions") or [])
                    if isinstance(region, dict)
                ]
            candidate_keys = [name.casefold() for name in candidate_names]
            expected_keys = [name.casefold() for name in expected_names]
            names_match = (
                candidate_keys == expected_keys
                if declared_candidate_names
                else len(candidate_keys) == len(expected_keys)
                and set(candidate_keys) == set(expected_keys)
            )
            if not names_match:
                continue
            try:
                snapshot = normalize_multi_char_snapshot({
                    "enable": True,
                    "character_order": expected_names,
                    "layout": layout,
                    "mask_location": mask_location,
                })
            except Exception as exc:
                print(
                    f"[MULTI_CHAR:BACKUP] 지문 후보 레이아웃 검증 실패: "
                    f"file={entry.path}, index={item_index}, error={exc}"
                )
                traceback.print_exc()
                continue
            if snapshot and snapshot["mask_fingerprint"] == expected_fingerprint:
                print(
                    f"[MULTI_CHAR:BACKUP] 레거시 레이아웃 복구 완료: "
                    f"file={entry.name}, index={item_index}, "
                    f"order={snapshot['character_order']}, "
                    f"fingerprint={expected_fingerprint[:12]}"
                )
                return snapshot
    raise ValueError(
        "기존 백업의 마스크 레이아웃을 삽화 세션에서 찾지 못했습니다: "
        f"order={expected_names}, fingerprint={expected_fingerprint}"
    )


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


def _overlap_midpoint_owners(coverage, boxes, pixel_boxes, original_union, rows, columns, width, height):
    """Center each shared region's partition without favoring a smaller body box."""
    channel_bits = 1 << np.arange(len(boxes))
    membership = (coverage[:, rows, columns] * channel_bits[:, None]).sum(axis=0)
    extension_bit = 1 << len(boxes)
    membership |= (~original_union[rows, columns]).astype(int) * extension_bit
    centers = np.asarray([(x + w / 2, y + h / 2) for x, y, w, h in boxes])
    owners = np.empty(len(rows), dtype=np.intp)
    for shared_region in np.unique(membership):
        selected = membership == shared_region
        members = np.flatnonzero(shared_region & channel_bits)
        bounds = [pixel_boxes[index] for index in members]
        left, right = max(box[0] for box in bounds), min(box[2] + 1 for box in bounds)
        if shared_region & extension_bit:
            # Added background is separate from original body pixels, so upward
            # extensions cannot shift the partition inside either body.
            top, bottom = 0, min(box[1] for box in bounds)
        else:
            top, bottom = max(box[1] for box in bounds), min(box[3] + 1 for box in bounds)
        midpoint = np.asarray([(left + right) / (2 * width), (top + bottom) / (2 * height)])
        member_centers = centers[members]
        anchors = member_centers - member_centers.mean(axis=0) + midpoint
        points = np.column_stack(((columns[selected] + .5) / width, (rows[selected] + .5) / height))
        # Translating unweighted centers to the overlap midpoint puts a two-box
        # boundary through that midpoint, including diagonal arrangements.
        distances = ((points[None, :, :] - anchors[:, None, :]) ** 2).sum(axis=2)
        chosen = distances.argmin(axis=0)
        remaining = set(range(len(members)))
        while remaining:
            first = min(remaining)
            identical = sorted(index for index in remaining if np.allclose(
                member_centers[index], member_centers[first], rtol=0, atol=1e-12,
            ))
            remaining.difference_update(identical)
            if len(identical) < 2:
                continue
            tied = np.isin(chosen, identical)
            if not tied.any():
                print(f"[MULTI_CHAR:MASK] 중심 동률 영역에 배정된 픽셀 없음: channels={members[identical].tolist()}")
                continue
            tied_x, tied_y = columns[selected][tied], rows[selected][tied]
            if (tied_x.max() - tied_x.min() + 1) / width >= (tied_y.max() - tied_y.min() + 1) / height:
                axis = tied_x
            else:
                axis = tied_y
            relative = (axis - axis.min() + .5) / (axis.max() - axis.min() + 1)
            stripes = np.clip((relative * len(identical)).astype(int), 0, len(identical) - 1)
            chosen[tied] = np.asarray(identical)[stripes]
            print(f"[MULTI_CHAR:MASK] 동일 중심 영역의 균등 분할: channels={members[identical].tolist()}")
        owners[selected] = members[chosen]
    return owners


def render_region_mask(layout: dict, width: int | None = None, height: int | None = None) -> Image.Image:
    """Extend upward into background while preserving original body ownership."""
    if isinstance(layout, dict) and layout.get("dialogue_mask"):
        import base64
        import io
        try:
            pixels = np.array(render_region_mask({key: value for key, value in layout.items()
                                                  if key != "dialogue_mask"}, width, height))
            mask = Image.open(io.BytesIO(base64.b64decode(layout["dialogue_mask"]))).convert("L")
            mask = mask.resize((pixels.shape[1], pixels.shape[0]), Image.Resampling.NEAREST)
            pixels[np.asarray(mask) > 0] = 0
            return Image.fromarray(pixels)
        except Exception as exc:
            print(f"[MANGA:MASK] 대사 영역 제외 실패: size={width}x{height}, error={exc}")
            traceback.print_exc()
            raise
    regions = layout.get("regions") if isinstance(layout, dict) else None
    if not isinstance(regions, list) or not 1 <= len(regions) <= len(MASK_CHANNELS):
        raise ValueError("검증된 1~3개 regions가 필요합니다")
    width = int(width or layout.get("mask_width") or DEFAULT_MASK_SIZE)
    height = int(height or layout.get("mask_height") or DEFAULT_MASK_SIZE)
    if width < 1 or height < 1:
        raise ValueError(f"마스크 크기가 올바르지 않습니다: {width}x{height}")

    if layout.get("panels"):
        # Reuse the existing overlap ownership inside each cut, then merge the
        # disjoint cut images into the identity's fixed RGB channel.
        page = np.zeros((height, width, 3), dtype=np.uint8)
        for panel in layout["panels"]:
            panel_regions, owners = [], []
            for index, region in enumerate(regions):
                bounds = [box for box in region.get("boxes", []) if box["panel"] == panel["panel"]]
                if not bounds:
                    continue
                left, top = min(box["x"] for box in bounds), min(box["y"] for box in bounds)
                panel_regions.append({"name": region["name"], "x": left, "y": top,
                    "width": max(box["x"] + box["width"] for box in bounds) - left,
                    "height": max(box["y"] + box["height"] for box in bounds) - top})
                owners.append(index)
            if not panel_regions:
                print(f"[MULTI_CHAR:MASK] 인물 없는 컷: panel={panel['panel']}")
                continue
            pixels = np.asarray(render_region_mask(
                {"regions": panel_regions, "mask_render_version": MASK_RENDER_VERSION}, width, height))
            left, top = max(0, math.ceil(panel["x"] * width)), max(0, math.ceil(panel["y"] * height))
            right = min(width, math.floor((panel["x"] + panel["width"]) * width))
            bottom = min(height, math.floor((panel["y"] + panel["height"]) * height))
            for source, target in enumerate(owners):
                page[top:bottom, left:right, target] = np.maximum(
                    page[top:bottom, left:right, target], pixels[top:bottom, left:right, source])
        print(f"[MULTI_CHAR:MASK] 컷별 RGB 마스크 생성: panels={len(layout['panels'])}, characters={len(regions)}")
        return Image.fromarray(page)

    channels = [Image.new("L", (width, height), 0) for _ in MASK_CHANNELS]
    boxes = []
    pixel_boxes = []
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
        boxes.append((x, y, box_width, box_height))
        pixel_boxes.append((left, top, right, bottom))

    render_version = int(layout.get("mask_render_version") or MASK_RENDER_VERSION)
    if render_version < 2:
        print("[MULTI_CHAR:MASK] 저장된 이전 버전의 마스크 영역과 겹침을 복원합니다")
        return Image.merge("RGB", tuple(channels))

    coverage = np.stack([np.asarray(channel) != 0 for channel in channels[:len(regions)]])
    original_union = coverage.any(axis=0)
    # V5 starts from body geometry; the sampler extends it around observed faces.
    # Archived V3/V4 layouts retain their unconditional upward extension.
    if 3 <= render_version < 5:
        for index, (left, top, right, _bottom) in enumerate(pixel_boxes):
            added = ~original_union[:top, left:right + 1]
            coverage[index, :top, left:right + 1] |= added
            channels[index] = Image.fromarray(coverage[index].astype(np.uint8) * 255)
        print(f"[MULTI_CHAR:MASK] 상단 확장: size={width}x{height}, "
              f"added_pixels={int(coverage.any(axis=0).sum() - original_union.sum())}; "
              "원래 인물 영역과 하단·좌우 경계 유지")
    rows, columns = np.nonzero(coverage.sum(axis=0) > 1)
    if not len(rows):
        print(f"[MULTI_CHAR:MASK] 겹치는 픽셀 없음: size={width}x{height}")
        return Image.merge("RGB", tuple(channels))

    if render_version >= 4:
        owners = _overlap_midpoint_owners(
            coverage, boxes, pixel_boxes, original_union, rows, columns, width, height,
        )
    else:
        # Stored V2/V3 snapshots retain their original mask and fingerprint.
        px, py = (columns + 0.5) / width, (rows + 0.5) / height
        scores = np.full((len(regions), len(rows)), np.inf)
        for index, (x, y, box_width, box_height) in enumerate(boxes):
            distance = ((px - x - box_width / 2) / (box_width / 2)) ** 2
            distance += ((py - y - box_height / 2) / (box_height / 2)) ** 2
            scores[index] = np.where(coverage[index, rows, columns],
                                     distance + 2 * math.log(box_width * box_height), np.inf)

        for box in dict.fromkeys(boxes):
            identical = [index for index, candidate in enumerate(boxes) if candidate == box]
            if len(identical) < 2:
                continue
            x, y, box_width, box_height = box
            relative = (px - x) / box_width if box_width >= box_height else (py - y) / box_height
            chosen = np.asarray(identical)[np.clip((relative * len(identical)).astype(int), 0, len(identical) - 1)]
            for index in identical:
                scores[index, chosen != index] = np.inf
            print(f"[MULTI_CHAR:MASK] 동일 좌표 영역의 기하적 동률 분할: channels={identical}, bounds={box}")

        owners = scores.argmin(axis=0)
    for index in range(len(regions)):
        coverage[index, rows, columns] = owners == index
        channels[index] = Image.fromarray(coverage[index].astype(np.uint8) * 255)
    print(f"[MULTI_CHAR:MASK] 겹친 픽셀 분할 완료: pixels={len(rows)}, "
          f"areas={[(region['name'], int(coverage[index].sum())) for index, region in enumerate(regions)]}")
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
        if isinstance(layout, dict):
            layout = {"mask_render_version": 1, **layout}
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
