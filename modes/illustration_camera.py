"""삽화 백업 수정용 Anima 카메라 제어 계약과 프롬프트 처리.

사용자가 명시적으로 선택한 카메라 차원만 구조화한다. 기존 프롬프트의 구도를
단어 목록으로 판별하지 않으며, 의미 기반 정리는 삽화 편집 LLM이 수행한다.
"""

from __future__ import annotations

import math
import re
import traceback
from typing import Any


CAMERA_CONTROL_VERSION = 1
DEFAULT_CAMERA_WEIGHT = 3.0
MIN_CAMERA_WEIGHT = 1.0
MAX_CAMERA_WEIGHT = 5.0


_DIMENSIONS: dict[str, dict[str, dict[str, Any]]] = {
    "direction": {
        "keep": {
            "label": "기존 방향 유지",
            "instruction": "Preserve the current horizontal viewing direction.",
            "tags": (),
        },
        "front": {
            "label": "정면",
            "instruction": "Use a front-facing view of the subject.",
            "tags": ("from front",),
        },
        "left": {
            "label": "왼쪽에서",
            "instruction": "View the subject from the left side.",
            "tags": ("from left",),
        },
        "right": {
            "label": "오른쪽에서",
            "instruction": "View the subject from the right side.",
            "tags": ("from right",),
        },
        "behind": {
            "label": "뒤에서",
            "instruction": "View the subject from behind.",
            "tags": ("from behind",),
        },
    },
    "elevation": {
        "keep": {
            "label": "기존 높이 유지",
            "instruction": "Preserve the current camera elevation.",
            "tags": (),
        },
        "overhead": {
            "label": "수직 탑뷰",
            "instruction": "Use a directly overhead aerial viewpoint.",
            "tags": ("directly above", "from above", "aerial view"),
        },
        "high": {
            "label": "하이 앵글",
            "instruction": "Use a high-angle view looking down at the subject.",
            "tags": ("high angle", "from above"),
        },
        "eye": {
            "label": "눈높이",
            "instruction": "Use a level eye-height viewpoint.",
            "tags": ("eye-level",),
        },
        "low": {
            "label": "로우 앵글",
            "instruction": "Use a low-angle view looking up at the subject.",
            "tags": ("low angle", "from below"),
        },
        "ground": {
            "label": "바닥 시점",
            "instruction": "Use a directly-below ground-level viewpoint.",
            "tags": ("directly below",),
        },
    },
    "distance": {
        "keep": {
            "label": "기존 거리 유지",
            "instruction": "Preserve the current shot distance and crop.",
            "tags": (),
        },
        "extreme_close": {
            "label": "극근접",
            "instruction": "Use an extreme close-up crop.",
            "tags": ("extreme close-up",),
        },
        "close": {
            "label": "근접",
            "instruction": "Use a close-up shot.",
            "tags": ("close-up",),
        },
        "medium": {
            "label": "중경",
            "instruction": "Use a medium shot.",
            "tags": ("medium shot",),
        },
        "full": {
            "label": "전신",
            "instruction": "Use a full-body shot without cropping the subject.",
            "tags": ("full body",),
        },
        "wide": {
            "label": "원경",
            "instruction": "Use a wide shot that includes the surrounding scene.",
            "tags": ("wide shot",),
        },
    },
    "roll": {
        "keep": {
            "label": "기존 기울기 유지",
            "instruction": "Preserve the current camera roll.",
            "tags": (),
        },
        "level": {
            "label": "수평",
            "instruction": "Use a level camera and remove any rolled or tilted framing.",
            "tags": (),
        },
        "dutch": {
            "label": "더치 앵글",
            "instruction": "Use a visibly tilted Dutch-angle composition.",
            "tags": ("dutch angle",),
        },
    },
}


def normalize_camera_control(raw: Any) -> dict[str, Any]:
    """브라우저 입력을 엄격히 검증해 저장 가능한 카메라 상태로 정규화한다."""
    if not isinstance(raw, dict):
        print(
            "[ILLUST_CAMERA] 카메라 입력 형식 오류: "
            f"type={type(raw).__name__}, value={raw!r}"
        )
        raise ValueError("카메라 구도 입력은 object 형식이어야 합니다")

    normalized: dict[str, Any] = {"version": CAMERA_CONTROL_VERSION}
    changed_dimensions: list[str] = []
    for dimension, choices in _DIMENSIONS.items():
        value = str(raw.get(dimension) or "keep").strip().lower()
        if value not in choices:
            print(
                f"[ILLUST_CAMERA] 지원하지 않는 {dimension} 값: "
                f"value={value!r}, allowed={list(choices)}"
            )
            raise ValueError(f"지원하지 않는 카메라 {dimension} 값입니다: {value!r}")
        normalized[dimension] = value
        if value != "keep":
            changed_dimensions.append(dimension)

    raw_weight = raw.get("weight", DEFAULT_CAMERA_WEIGHT)
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

    if not changed_dimensions:
        print(f"[ILLUST_CAMERA] 변경할 구도 차원이 없음: input={raw!r}")
        raise ValueError("방향·높이·거리·기울기 중 하나 이상을 선택해주세요")
    return normalized


def camera_selection_summary(control: dict[str, Any]) -> str:
    """LLM 계획과 프런트 상태 표시에 쓸 한국어 선택 요약."""
    parts = []
    for dimension, choices in _DIMENSIONS.items():
        value = str(control.get(dimension) or "keep")
        if value == "keep":
            continue
        parts.append(str(choices[value]["label"]))
    return " · ".join(parts)


def compile_camera_prompt(control: dict[str, Any]) -> str:
    """선택한 차원의 최종 Anima 가중치 태그를 만든다."""
    weight = float(control["weight"])
    tags: list[str] = []
    seen: set[str] = set()
    for dimension, choices in _DIMENSIONS.items():
        value = str(control.get(dimension) or "keep")
        for tag in choices[value]["tags"]:
            clean = str(tag).strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            tags.append(f"({clean}:{weight:.2f})")
    return ", ".join(tags)


def build_camera_edit_direction(control: dict[str, Any]) -> str:
    """기존 편하게 수정 파이프라인의 direction 필드에 넣을 간단한 지시."""
    summary = camera_selection_summary(control)
    return (
        "Apply the mandatory camera-only edit contract appended below. "
        f"Target camera adjustment: {summary}. Preserve all unrelated scene content."
    )


def build_camera_edit_contract(control: dict[str, Any]) -> str:
    """LLM이 기존 카메라 의미를 문맥적으로 정리하도록 강제하는 추가 계약."""
    changed_lines = []
    preserved_lines = []
    for dimension, choices in _DIMENSIONS.items():
        value = str(control.get(dimension) or "keep")
        instruction = str(choices[value]["instruction"])
        if value == "keep":
            preserved_lines.append(f"- {dimension}: {instruction}")
        else:
            changed_lines.append(f"- {dimension}: {instruction}")

    final_prompt = compile_camera_prompt(control)
    final_prompt_note = final_prompt or "(no final tag; semantic cleanup only)"
    return (
        "\n\n## Mandatory Anima camera-only edit contract\n"
        "This contract overrides any generic instruction that would keep conflicting camera terms.\n"
        "For every changed dimension below, semantically locate and remove or minimally rewrite all "
        "previous camera, viewpoint, crop, shot-distance, or roll instructions that conflict with the "
        "target. Do not merely append a second camera instruction. Do not use literal keyword "
        "substitution; reason from the full scene and the supplied image.\n"
        "Changed dimensions:\n"
        + "\n".join(changed_lines)
        + "\nUnchanged dimensions:\n"
        + "\n".join(preserved_lines or ["- none"])
        + "\nPreserve character identity, count, pose, expression, outfit, objects, background, lighting, "
        "and spatial relationships unless a minimal compatibility adjustment is unavoidable. "
        "Do not change SDXL-specific content. Return the normal complete scene_* JSON fields. "
        "Do not add weighted camera-control syntax to scene_*; the application appends it after validation.\n"
        f"Application-owned final Anima camera tags: {final_prompt_note}"
    )


def _section_content(positive: str, section: str) -> str:
    pattern = re.compile(rf"^\[{re.escape(section)}\]\r?\n(.*)$", re.MULTILINE)
    matches = pattern.findall(positive)
    if len(matches) != 1:
        print(
            f"[ILLUST_CAMERA] [{section}] 블록 개수 오류: count={len(matches)}"
        )
        raise ValueError(f"[{section}] 블록이 정확히 하나여야 합니다")
    return matches[0]


def _replace_section(positive: str, section: str, content: str) -> str:
    pattern = re.compile(rf"(^\[{re.escape(section)}\]\r?\n).*$", re.MULTILINE)
    updated, count = pattern.subn(lambda match: match.group(1) + content, positive)
    if count != 1:
        print(f"[ILLUST_CAMERA] [{section}] 교체 실패: count={count}")
        raise ValueError(f"[{section}] 블록을 안전하게 교체하지 못했습니다")
    return updated


def finalize_camera_edit(
    original_positive: str,
    llm_edited_positive: str,
    control: dict[str, Any],
) -> tuple[str, str]:
    """SDXL은 원본 그대로 복원하고 두 Anima 장면 블록에 확정 태그를 추가한다."""
    try:
        original_sdxl = _section_content(original_positive, "SDXL")
        finalized = _replace_section(llm_edited_positive, "SDXL", original_sdxl)
        camera_prompt = compile_camera_prompt(control)
        if camera_prompt:
            for section in ("ANIMA_CONTENT", "ANIMA_ALL"):
                current = _section_content(finalized, section).rstrip(" ,")
                combined = f"{current}, {camera_prompt}" if current else camera_prompt
                finalized = _replace_section(finalized, section, combined)
        return finalized, camera_prompt
    except Exception as exc:
        print(f"[ILLUST_CAMERA] 카메라 프롬프트 최종 조립 실패: error={exc}")
        traceback.print_exc()
        raise

