"""삽화 백업 수정용 연속형 Anima 카메라 제어.

브라우저에서 사용자가 직접 움직인 카메라 좌표를 태그로 컴파일한다. 기존
프롬프트의 구도는 단어 목록으로 판별하지 않고 삽화 편집 LLM이 의미적으로
정리하며, Anima 1차 생성과 SDXL Hiresfix가 같은 카메라를 사용하게 한다.
"""

from __future__ import annotations

import json
import math
import re
import traceback
from typing import Any


CAMERA_CONTROL_VERSION = 3
DEFAULT_CAMERA_WEIGHT = 3.0
MIN_CAMERA_WEIGHT = 1.0
MAX_CAMERA_WEIGHT = 5.0
CAMERA_AXES = ("pos_x", "pos_y", "pos_z", "roll")
CAMERA_AXIS_BUDGET_MIN = 1.0
CAMERA_AXIS_BUDGET_MAX = 3.0
CAMERA_AXIS_DEADZONE = 0.2
CAMERA_POLE_GATE_START = 0.9
CAMERA_LORA_PATH = (
    r"SOYA_CHAR_LORA\SOYA_INSTANCE_LORA\anima\bsk-anima-camera-control"
    r"\civitai-3174431\anima_bsk_camera_control_v1.0.safetensors"
)
CAMERA_LORA_MODEL_VERSION_ID = 3174431
MIN_CAMERA_LORA_STRENGTH = 0.6
MAX_CAMERA_LORA_STRENGTH = 1.0


def _normalize_axis(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        print(f"[ILLUST_CAMERA] {name} 좌표 변환 실패: value={value!r}, error={exc}")
        raise ValueError(f"카메라 {name} 좌표는 숫자여야 합니다") from exc
    if not math.isfinite(parsed) or not -1.0 <= parsed <= 1.0:
        print(f"[ILLUST_CAMERA] {name} 좌표 범위 오류: value={parsed!r}")
        raise ValueError(f"카메라 {name} 좌표는 -1~1 범위여야 합니다")
    return round(parsed, 4)


def _legacy_control_to_axes(raw: dict[str, Any]) -> dict[str, Any] | None:
    """초기 드롭다운 실험판 메타데이터를 연속 좌표로 한 번 호환한다."""
    if not any(key in raw for key in ("direction", "elevation", "distance", "roll")):
        return None
    direction = {
        "keep": 0.0,
        "front": 0.0,
        "left": -0.5,
        "right": 0.5,
        "behind": 1.0,
    }
    elevation = {
        "keep": 0.0,
        "overhead": 0.9,
        "high": 0.5,
        "eye": 0.0,
        "low": -0.5,
        "ground": -0.9,
    }
    distance = {
        "keep": 0.0,
        "extreme_close": 0.85,
        "close": 0.45,
        "medium": 0.0,
        "full": -0.45,
        "wide": -0.85,
    }
    roll = {"keep": 0.0, "level": 0.0, "dutch": 0.6}
    values = {
        "pos_x": direction.get(str(raw.get("direction") or "keep")),
        "pos_y": elevation.get(str(raw.get("elevation") or "keep")),
        "pos_z": distance.get(str(raw.get("distance") or "keep")),
        "roll": roll.get(str(raw.get("roll") or "keep")),
        "weight": raw.get("weight", DEFAULT_CAMERA_WEIGHT),
    }
    if any(value is None for key, value in values.items() if key != "weight"):
        print(f"[ILLUST_CAMERA] 이전 구도 메타데이터 값 오류: input={raw!r}")
        raise ValueError("이전 구도 조정 메타데이터를 좌표로 변환하지 못했습니다")
    print("[ILLUST_CAMERA] 이전 드롭다운 구도 메타데이터를 연속 좌표로 변환")
    return values


def normalize_camera_control(raw: Any) -> dict[str, Any]:
    """브라우저 입력을 엄격히 검증해 저장 가능한 연속 카메라 상태로 만든다."""
    if not isinstance(raw, dict):
        print(
            "[ILLUST_CAMERA] 카메라 입력 형식 오류: "
            f"type={type(raw).__name__}, value={raw!r}"
        )
        raise ValueError("카메라 구도 입력은 object 형식이어야 합니다")

    source = raw
    if not all(axis in source for axis in CAMERA_AXES):
        legacy = _legacy_control_to_axes(source)
        if legacy is None:
            print(f"[ILLUST_CAMERA] 카메라 좌표 누락: input={raw!r}")
            raise ValueError("카메라 pos_x·pos_y·pos_z·roll 좌표가 모두 필요합니다")
        source = legacy

    normalized: dict[str, Any] = {"version": CAMERA_CONTROL_VERSION}
    for axis in CAMERA_AXES:
        normalized[axis] = _normalize_axis(axis, source.get(axis))

    raw_weight = source.get("weight", DEFAULT_CAMERA_WEIGHT)
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        print(f"[ILLUST_CAMERA] 카메라 강도 변환 실패: value={raw_weight!r}, error={exc}")
        raise ValueError("카메라 강도는 숫자여야 합니다") from exc
    if not math.isfinite(weight) or not MIN_CAMERA_WEIGHT <= weight <= MAX_CAMERA_WEIGHT:
        print(
            "[ILLUST_CAMERA] 카메라 강도 범위 오류: "
            f"value={weight!r}, range={MIN_CAMERA_WEIGHT}..{MAX_CAMERA_WEIGHT}"
        )
        raise ValueError(
            f"카메라 강도는 {MIN_CAMERA_WEIGHT:g}~{MAX_CAMERA_WEIGHT:g} 범위여야 합니다"
        )
    normalized["weight"] = round(weight, 2)
    return normalized


def compile_camera_lora_strength(control: dict[str, Any]) -> float:
    """UI 강도 1~5를 모델 권장 범위 0.6~1.0에 선형 대응한다."""
    try:
        weight = float(control["weight"])
    except (KeyError, TypeError, ValueError) as exc:
        print(
            "[ILLUST_CAMERA:LORA] 카메라 LoRA 강도 변환 실패: "
            f"control={control!r}, error={exc}"
        )
        raise ValueError("카메라 LoRA 강도를 계산하지 못했습니다") from exc
    if not math.isfinite(weight) or not MIN_CAMERA_WEIGHT <= weight <= MAX_CAMERA_WEIGHT:
        print(
            "[ILLUST_CAMERA:LORA] 카메라 LoRA 강도 입력 범위 오류: "
            f"weight={weight!r}, range={MIN_CAMERA_WEIGHT}..{MAX_CAMERA_WEIGHT}"
        )
        raise ValueError("카메라 LoRA 강도 입력이 1~5 범위를 벗어났습니다")
    ratio = (weight - MIN_CAMERA_WEIGHT) / (MAX_CAMERA_WEIGHT - MIN_CAMERA_WEIGHT)
    strength = MIN_CAMERA_LORA_STRENGTH + ratio * (
        MAX_CAMERA_LORA_STRENGTH - MIN_CAMERA_LORA_STRENGTH
    )
    return round(strength, 2)


def camera_lora_metadata(control: dict[str, Any]) -> dict[str, Any]:
    """백업 메타데이터와 API 응답에 남길 고정 카메라 LoRA 정보."""
    return {
        "model_version_id": CAMERA_LORA_MODEL_VERSION_ID,
        "lora_path": CAMERA_LORA_PATH,
        "strength": compile_camera_lora_strength(control),
        "base": "anima",
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _weighted(tag: str, weight: float) -> str:
    return f"({tag}:{_clamp(weight, 0.1, 10.0):.2f})"


def _prompt_axis_budget(weight: float) -> float:
    """UI 강도 1~5를 안전한 축별 총 예산 1~3으로 선형 축소한다."""
    ratio = (weight - MIN_CAMERA_WEIGHT) / (MAX_CAMERA_WEIGHT - MIN_CAMERA_WEIGHT)
    return CAMERA_AXIS_BUDGET_MIN + ratio * (
        CAMERA_AXIS_BUDGET_MAX - CAMERA_AXIS_BUDGET_MIN
    )


def _pole_gate(pos_y: float) -> float:
    """플러그인처럼 극점 직전까지 방위를 유지하고 마지막 10%에서만 감쇠한다."""
    return _clamp(
        (1.0 - abs(pos_y)) / (1.0 - CAMERA_POLE_GATE_START),
        0.0,
        1.0,
    )


def _distribute_axis_budget(
    tags: tuple[str, ...],
    budget: float,
) -> list[tuple[str, float]]:
    """동의 태그가 축의 총 가중치를 중복하지 않고 나눠 갖게 한다."""
    if not tags or budget <= 0:
        return []
    ranks = [1.0 / (index + 1) for index in range(len(tags))]
    rank_total = sum(ranks)
    return [
        (tag, budget * rank / rank_total)
        for tag, rank in zip(tags, ranks)
    ]


def _azimuth_components(pos_x: float) -> list[tuple[str, float]]:
    angle = pos_x * math.pi
    raw = [
        ("from front", max(0.0, math.cos(angle))),
        ("from behind", max(0.0, -math.cos(angle))),
        ("from right", max(0.0, math.sin(angle))),
        ("from left", max(0.0, -math.sin(angle))),
    ]
    total = sum(value for _tag, value in raw)
    if total <= 0:
        return []
    return [(tag, value / total) for tag, value in raw if value > 0]


def _elevation_target(pos_y: float) -> tuple[str, tuple[str, ...]]:
    if pos_y > 0.7:
        return "수직 탑뷰", ("directly above", "from above", "aerial view")
    if pos_y > 0.2:
        return "하이 앵글", ("high angle", "from above")
    if pos_y >= -0.2:
        return "눈높이", () if abs(pos_y) < 0.05 else ("eye-level",)
    if pos_y >= -0.7:
        return "로우 앵글", ("low angle", "from below")
    return "바닥 시점", ("directly below",)


def _distance_target(pos_z: float) -> tuple[str, tuple[str, ...]]:
    if pos_z > 0.7:
        return "극근접", ("extreme close-up",)
    if pos_z > 0.2:
        return "근접", ("close-up",)
    if pos_z >= -0.2:
        return "중경", ("medium shot",)
    if pos_z >= -0.7:
        return "전신", ("full body",)
    return "원경", ("wide shot",)


def camera_selection_summary(control: dict[str, Any]) -> str:
    """LLM 계획과 프런트 상태 표시에 쓸 한국어 좌표 요약."""
    pos_x = float(control["pos_x"])
    pos_y = float(control["pos_y"])
    pos_z = float(control["pos_z"])
    roll = float(control["roll"])
    degrees = round(abs(pos_x) * 180)
    if abs(pos_x) >= 0.95:
        direction = "후면"
    elif pos_x > 0.05:
        direction = f"오른쪽 {degrees}°"
    elif pos_x < -0.05:
        direction = f"왼쪽 {degrees}°"
    else:
        direction = "정면"
    elevation, _tags = _elevation_target(pos_y)
    distance, _distance_tags = _distance_target(pos_z)
    roll_degrees = round(roll * 45)
    return f"{direction} · {elevation} · {distance} · 롤 {roll_degrees}°"


def compile_camera_prompt(control: dict[str, Any]) -> str:
    """연속 좌표를 축별 총량이 제한된 Anima 카메라 태그로 컴파일한다."""
    pos_x = float(control["pos_x"])
    pos_y = float(control["pos_y"])
    pos_z = float(control["pos_z"])
    roll = float(control["roll"])
    axis_budget = _prompt_axis_budget(float(control["weight"]))
    parts: list[str] = []
    seen: set[str] = set()

    # BSK 플러그인의 극점 처리만 따르되 총 예산은 안전 범위로 제한한다.
    pole_gate = _pole_gate(pos_y)
    for tag, ratio in _azimuth_components(pos_x):
        weighted_strength = ratio * axis_budget * pole_gate
        if weighted_strength < CAMERA_AXIS_DEADZONE or tag in seen:
            continue
        seen.add(tag)
        parts.append(_weighted(tag, weighted_strength))

    _elevation_label, elevation_tags = _elevation_target(pos_y)
    elevation_budget = abs(pos_y) * axis_budget
    elevation_parts = (
        _distribute_axis_budget(elevation_tags, elevation_budget)
        if elevation_budget >= CAMERA_AXIS_DEADZONE
        else []
    )
    for tag, tag_strength in elevation_parts:
        if tag not in seen and tag_strength >= 0.1:
            seen.add(tag)
            parts.append(_weighted(tag, tag_strength))

    _distance_label, distance_tags = _distance_target(pos_z)
    for tag in distance_tags:
        if tag not in seen:
            seen.add(tag)
            parts.append(_weighted(tag, 1.0))

    if abs(roll) >= 0.15:
        parts.append(_weighted("dutch angle", 1.0))
    return ", ".join(parts)


def build_camera_edit_direction(control: dict[str, Any]) -> str:
    """기존 편하게 수정 파이프라인의 direction 필드에 넣을 지시."""
    return (
        "Apply the mandatory camera-only edit contract appended below. "
        f"Target camera position: {camera_selection_summary(control)}. "
        "Preserve all unrelated scene content."
    )


def build_camera_edit_contract(control: dict[str, Any]) -> str:
    """LLM이 모든 기존 카메라 의미를 문맥적으로 정리하도록 강제한다."""
    pos_x = float(control["pos_x"])
    pos_y = float(control["pos_y"])
    pos_z = float(control["pos_z"])
    roll = float(control["roll"])
    elevation, _elevation_tags = _elevation_target(pos_y)
    distance, _distance_tags = _distance_target(pos_z)
    if abs(pos_x) >= 0.95:
        direction = "a rear view from behind the subject"
    elif pos_x > 0.05:
        direction = f"a viewpoint orbiting {abs(pos_x) * 180:.0f} degrees to the subject's right"
    elif pos_x < -0.05:
        direction = f"a viewpoint orbiting {abs(pos_x) * 180:.0f} degrees to the subject's left"
    else:
        direction = "a frontal view"
    roll_instruction = (
        f"a Dutch tilt of about {abs(roll) * 45:.0f} degrees "
        + ("clockwise" if roll > 0 else "counter-clockwise")
        if abs(roll) >= 0.15
        else "a level, untilted camera"
    )
    final_prompt = compile_camera_prompt(control)
    return (
        "\n\n## Mandatory Anima camera-only edit contract\n"
        "Replace the complete previous camera setup semantically. Locate and remove every old "
        "viewpoint, camera height, shot distance, crop, and roll instruction from the returned "
        "scene fields. Do not merely append a second camera setup and do not use literal keyword "
        "substitution; reason from the full scene and supplied image.\n"
        f"- Horizontal orbit: {direction}.\n"
        f"- Camera elevation: {elevation}.\n"
        f"- Shot distance/crop: {distance}.\n"
        f"- Camera roll: {roll_instruction}.\n"
        "Apply this semantic camera cleanup consistently to BOTH the current Anima scene and the "
        "current SDXL scene. Return camera-neutral scene_* fields: do not restate either the old "
        "camera or the target camera in tags, prose, captions, or weighted syntax. The application "
        "is the sole authority that appends the identical final camera to Anima and SDXL.\n"
        "Preserve character identity, count, pose, expression, outfit, objects, background, lighting, "
        "and spatial relationships unless a minimal compatibility adjustment is unavoidable. "
        "Preserve SDXL quality, artist, style, and identity content that is unrelated to the camera. "
        "Return the normal complete scene_* JSON fields with no camera setup. Preserve character "
        "gaze and body pose when they are character actions rather than camera instructions.\n"
        f"Application-owned final synchronized camera tags: {final_prompt}"
    )


def _section_content(positive: str, section: str) -> str:
    pattern = re.compile(rf"^\[{re.escape(section)}\]\r?\n(.*)$", re.MULTILINE)
    matches = pattern.findall(positive)
    if len(matches) != 1:
        print(f"[ILLUST_CAMERA] [{section}] 블록 개수 오류: count={len(matches)}")
        raise ValueError(f"[{section}] 블록이 정확히 하나여야 합니다")
    return matches[0]


def _replace_section(positive: str, section: str, content: str) -> str:
    pattern = re.compile(rf"(^\[{re.escape(section)}\]\r?\n).*$", re.MULTILINE)
    updated, count = pattern.subn(lambda match: match.group(1) + content, positive)
    if count != 1:
        print(f"[ILLUST_CAMERA] [{section}] 교체 실패: count={count}")
        raise ValueError(f"[{section}] 블록을 안전하게 교체하지 못했습니다")
    return updated


def _has_single_character_metadata(positive: str) -> bool:
    """구조화된 MULTI_CHAR 메타데이터로만 단일 인물 여부를 판단한다."""
    pattern = re.compile(r"^\[MULTI_CHAR\]\r?\n(.*)$", re.MULTILINE)
    matches = pattern.findall(positive)
    if not matches:
        print("[ILLUST_CAMERA] 단일 인물 보호 생략: [MULTI_CHAR] 메타데이터 없음")
        return False
    if len(matches) != 1:
        print(f"[ILLUST_CAMERA] [MULTI_CHAR] 블록 개수 오류: count={len(matches)}")
        raise ValueError("[MULTI_CHAR] 블록이 정확히 하나여야 합니다")
    try:
        payload = json.loads(matches[0])
        if not isinstance(payload, dict):
            raise ValueError("[MULTI_CHAR] 값이 object가 아닙니다")
    except Exception as exc:
        print(f"[ILLUST_CAMERA] [MULTI_CHAR] 메타데이터 해석 실패: error={exc}")
        traceback.print_exc()
        raise ValueError("단일 인물 여부를 구조적으로 확인하지 못했습니다") from exc

    enabled = payload.get("enable")
    char_num = payload.get("char_num")
    is_single = enabled is False and type(char_num) is int and char_num == 1
    if is_single:
        print("[ILLUST_CAMERA] 단일 인물 메타데이터 확인: solo 보호 태그 적용")
        return True

    print(
        "[ILLUST_CAMERA] solo 보호 태그 생략: "
        f"MULTI_CHAR.enable={enabled!r}, char_num={char_num!r}"
    )
    return False


def _append_solo_guard(content: str) -> str:
    """단일 인물 프롬프트에 정확한 ``solo`` 태그를 한 번만 보장한다."""
    if re.search(r"(?:^|,\s*)solo(?:\s*,|$)", content):
        return content
    return f"{content.rstrip(' ,')}, solo" if content.strip(" ,") else "solo"


def _normalized_lora_path(value: Any) -> str:
    return str(value or "").replace("/", "\\").strip().casefold()


def _merge_camera_lora(positive: str, control: dict[str, Any]) -> str:
    """기존 활성 전역 LoRA를 보존하며 Anima 카메라 LoRA를 한 번 병합한다."""
    try:
        raw_activation = _section_content(positive, "STYLE_LORA_ACTIVATE").strip().lower()
        if raw_activation not in ("true", "false"):
            print(
                "[ILLUST_CAMERA:LORA] STYLE_LORA_ACTIVATE 값 오류: "
                f"value={raw_activation!r}"
            )
            raise ValueError("[STYLE_LORA_ACTIVATE] 값은 true 또는 false여야 합니다")
        raw_lora_data = _section_content(positive, "STYLE_LORA_DATA")
        payload = json.loads(raw_lora_data)
        if not isinstance(payload, dict):
            raise ValueError("[STYLE_LORA_DATA] 값이 object가 아닙니다")
        lora_list = payload.get("list")
        if not isinstance(lora_list, list):
            raise ValueError("[STYLE_LORA_DATA].list 값이 배열이 아닙니다")
        if raw_activation == "false" and lora_list:
            print(
                "[ILLUST_CAMERA:LORA] 비활성 전역 LoRA 데이터 충돌: "
                f"entries={len(lora_list)}"
            )
            raise ValueError(
                "비활성 STYLE_LORA_DATA에 기존 항목이 남아 있어 카메라 LoRA만 "
                "안전하게 활성화할 수 없습니다"
            )

        strength = compile_camera_lora_strength(control)
        target_key = _normalized_lora_path(CAMERA_LORA_PATH)
        merged_list: list[dict[str, Any]] = []
        camera_entry_added = False
        removed_duplicates = 0
        for index, entry in enumerate(lora_list):
            if not isinstance(entry, dict):
                raise ValueError(f"[STYLE_LORA_DATA].list[{index}] 값이 object가 아닙니다")
            if _normalized_lora_path(entry.get("lora_path")) != target_key:
                merged_list.append(entry)
                continue
            if camera_entry_added:
                removed_duplicates += 1
                continue
            merged_list.append({
                "lora_path": CAMERA_LORA_PATH,
                "str": strength,
                "BASE": "anima",
            })
            camera_entry_added = True

        if not camera_entry_added:
            merged_list.append({
                "lora_path": CAMERA_LORA_PATH,
                "str": strength,
                "BASE": "anima",
            })
        updated_payload = dict(payload)
        updated_payload["list"] = merged_list
        finalized = _replace_section(
            positive,
            "STYLE_LORA_DATA",
            json.dumps(updated_payload, ensure_ascii=False, separators=(",", ":")),
        )
        finalized = _replace_section(finalized, "STYLE_LORA_ACTIVATE", "true")
        print(
            "[ILLUST_CAMERA:LORA] Anima 전역 카메라 LoRA 병합 완료: "
            f"path={CAMERA_LORA_PATH!r}, strength={strength}, "
            f"existing_entries={len(lora_list)}, removed_duplicates={removed_duplicates}"
        )
        return finalized
    except Exception as exc:
        print(f"[ILLUST_CAMERA:LORA] 카메라 LoRA 병합 실패: error={exc}")
        traceback.print_exc()
        raise


def finalize_camera_edit(
    original_positive: str,
    llm_edited_positive: str,
    control: dict[str, Any],
) -> tuple[str, str]:
    """LLM 정리 결과에 동일한 Anima·SDXL 확정 카메라 태그를 추가한다."""
    try:
        finalized = llm_edited_positive
        protect_single_character = _has_single_character_metadata(original_positive)
        # 품질·작가 설정은 카메라 편집 대상이 아니므로 원본을 명시적으로 보존한다.
        for section in ("SDXL_QUALITY", "SDXL_ARTIST"):
            finalized = _replace_section(
                finalized,
                section,
                _section_content(original_positive, section),
            )
        camera_prompt = compile_camera_prompt(control)
        if not camera_prompt:
            print(f"[ILLUST_CAMERA] 카메라 태그 컴파일 결과가 비어 있음: control={control!r}")
            raise ValueError("카메라 좌표에서 생성된 Anima 태그가 비어 있습니다")
        for section in ("ANIMA_CONTENT", "ANIMA_ALL", "SDXL"):
            current = _section_content(finalized, section).rstrip(" ,")
            if protect_single_character:
                current = _append_solo_guard(current)
            combined = f"{current}, {camera_prompt}" if current else camera_prompt
            finalized = _replace_section(finalized, section, combined)
        finalized = _merge_camera_lora(finalized, control)
        print(
            "[ILLUST_CAMERA] Anima·SDXL 카메라 태그 및 Anima LoRA 동기화 완료: "
            f"prompt={camera_prompt!r}, lora={camera_lora_metadata(control)!r}"
        )
        return finalized, camera_prompt
    except Exception as exc:
        print(f"[ILLUST_CAMERA] 카메라 프롬프트 최종 조립 실패: error={exc}")
        traceback.print_exc()
        raise
