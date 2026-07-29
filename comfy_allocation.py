"""작업 종류별 ComfyUI 실행 인스턴스 배분 설정."""

from __future__ import annotations

import traceback
from typing import Any


COMFY_TASK_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ("illustration", "삽화", "일반 삽화 생성 요청"),
    (
        "restore_regenerate",
        "재생성 / 수동 그리기",
        "삽화 재생성, 수동 그리기, 워크플로우 테스트",
    ),
    ("asset_generation", "에셋 생성", "에셋 이미지 생성"),
    ("qwen_edit", "EDIT", "Qwen Edit 및 ANIMA Inpainting"),
    ("tag_analysis", "태그 분석", "Comfy 워크플로우 기반 이미지 태그 분석"),
    ("outfit", "복장 추출", "복장 추출 워크플로우"),
    ("asset_lora_training", "에셋 LoRA", "에셋 LoRA 학습"),
    ("bot_lora_training", "봇 LoRA", "봇 캐릭터 LoRA 학습"),
    (
        "instance_lora",
        "인스턴스 / 스타일 LoRA",
        "인스턴스·스타일 LoRA 분석 및 학습",
    ),
    ("face_extract", "얼굴 추출", "인스턴스 LoRA 얼굴 추출"),
    (
        "utility_debug",
        "유틸리티 / 디버그",
        "데이터 패치, 봇 유틸리티, 디버그 워크플로우",
    ),
)

COMFY_TASK_KEYS = tuple(item[0] for item in COMFY_TASK_DEFINITIONS)
DEFAULT_COMFY_TASK_ALLOCATIONS = {key: 1 for key in COMFY_TASK_KEYS}


class ComfyTaskAllocationValidationError(ValueError):
    """Comfy 작업 배분 설정 검증 실패."""


def normalize_comfy_task_allocations(
    raw: Any,
    *,
    legacy_illustration_port: Any = None,
) -> dict[str, int]:
    """누락 키를 채우고 각 작업의 인스턴스 번호를 1 또는 2로 검증한다.

    새 배분 설정이 전혀 없는 기존 설정에서는 ``comfyui_port_illustration`` 사용
    여부를 삽화 계열 두 항목의 Comfy #2 선택으로 승계한다.
    """

    if raw is None:
        source: dict[str, Any] = {}
        use_legacy_illustration = legacy_illustration_port is not None
    elif isinstance(raw, dict):
        source = raw
        use_legacy_illustration = False
    else:
        message = (
            "Comfy 작업 배분 설정은 객체여야 합니다: "
            f"type={type(raw).__name__}, value={raw!r}"
        )
        print(f"[COMFY_ALLOCATION] 설정 검증 실패: {message}")
        raise ComfyTaskAllocationValidationError(message)

    normalized = dict(DEFAULT_COMFY_TASK_ALLOCATIONS)
    if use_legacy_illustration:
        normalized["illustration"] = 2
        normalized["restore_regenerate"] = 2

    for key in COMFY_TASK_KEYS:
        if key not in source:
            continue
        value = source.get(key)
        try:
            if isinstance(value, bool):
                raise TypeError("bool은 허용되지 않음")
            parsed = int(value)
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("정수가 아닌 실수는 허용되지 않음")
            if isinstance(value, str) and value.strip() != str(parsed):
                raise ValueError("정수 문자열 형식이 아님")
            if parsed not in (1, 2):
                raise ValueError("1 또는 2가 아님")
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[COMFY_ALLOCATION] 작업 배분 값 검증 실패: "
                f"task={key}, value={value!r}, error={exc}"
            )
            traceback.print_exc()
            raise ComfyTaskAllocationValidationError(
                f"comfy_task_allocations.{key} 값은 1 또는 2여야 합니다."
            ) from exc
        normalized[key] = parsed

    unknown = sorted(str(key) for key in source if key not in COMFY_TASK_KEYS)
    if unknown:
        print(f"[COMFY_ALLOCATION] 알 수 없는 작업 배분 키 무시: {unknown!r}")

    return normalized


def comfy_task_definition_payload() -> list[dict[str, str]]:
    """프런트엔드가 동일한 작업 목록을 렌더링할 수 있는 JSON 안전 구조."""

    return [
        {"key": key, "label": label, "description": description}
        for key, label, description in COMFY_TASK_DEFINITIONS
    ]


def select_comfy_instance(
    allocations: dict[str, int],
    task_key: str,
    running: dict[int, bool],
) -> int:
    """설정 배분을 선택하되 정확히 하나만 실행 중이면 그 인스턴스로 폴백한다."""

    if task_key not in allocations:
        print(
            "[COMFY_ALLOCATION] 인스턴스 선택 실패: "
            f"알 수 없는 task={task_key!r}, supported={tuple(allocations)!r}"
        )
        raise ComfyTaskAllocationValidationError(
            f"알 수 없는 Comfy 작업 배분 키입니다: {task_key}"
        )
    configured = allocations[task_key]
    running_ids = [instance_id for instance_id in (1, 2) if running.get(instance_id)]
    if len(running_ids) == 1:
        return running_ids[0]
    return configured
