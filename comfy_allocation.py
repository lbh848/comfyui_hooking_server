"""작업 종류별 ComfyUI 실행 인스턴스 배분 설정."""

from __future__ import annotations

import traceback
from contextvars import ContextVar
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
    (
        "video_generation",
        "영상화",
        "MiniMax H3 I2V·FLF2V·REF2V 영상화",
    ),
)

COMFY_TASK_KEYS = tuple(item[0] for item in COMFY_TASK_DEFINITIONS)
DEFAULT_COMFY_TASK_ALLOCATIONS = {key: 1 for key in COMFY_TASK_KEYS}
MODAL_COMFY_TARGET = "modal"
VAST_COMFY_TARGET = "vast"
VIDEO_ENGINE_COMFY_TARGET = "video_engine"
MODAL_SUPPORTED_COMFY_TASK_KEYS = frozenset(
    {
        "illustration",
        "restore_regenerate",
        "asset_generation",
        "qwen_edit",
        "asset_lora_training",
        "bot_lora_training",
        "instance_lora",
        "video_generation",
    }
)
VAST_SUPPORTED_COMFY_TASK_KEYS = MODAL_SUPPORTED_COMFY_TASK_KEYS
REMOTE_COMFY_TARGETS = frozenset({MODAL_COMFY_TARGET, VAST_COMFY_TARGET})
VIDEO_ENGINE_SUPPORTED_COMFY_TASK_KEYS = frozenset({"video_generation"})
NONLOCAL_COMFY_TARGETS = frozenset(
    {*REMOTE_COMFY_TARGETS, VIDEO_ENGINE_COMFY_TARGET}
)
DEFAULT_COMFY_TASK_MODAL_PARALLEL = {key: False for key in COMFY_TASK_KEYS}
DEFAULT_COMFY_TASK_VAST_PARALLEL = {key: False for key in COMFY_TASK_KEYS}

# 큐 워커가 claim한 실행 대상을 하위 모드와 server.py의 공통 제출 함수까지 전달한다.
# asyncio Task별 값이 분리되어 로컬 Comfy와 여러 Modal 워커가 동시에 실행돼도 섞이지 않는다.
CURRENT_COMFY_EXECUTION_TARGET: ContextVar[str | None] = ContextVar(
    "current_comfy_execution_target",
    default=None,
)


class ComfyTaskAllocationValidationError(ValueError):
    """Comfy 작업 배분 설정 검증 실패."""


def normalize_comfy_task_allocations(
    raw: Any,
    *,
    legacy_illustration_port: Any = None,
) -> dict[str, int | str]:
    """누락 키를 채우고 각 작업의 로컬 인스턴스 또는 Modal 대상을 검증한다.

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
        normalized_target = value.strip().lower() if isinstance(value, str) else ""
        if normalized_target in REMOTE_COMFY_TARGETS:
            supported_keys = (
                MODAL_SUPPORTED_COMFY_TASK_KEYS
                if normalized_target == MODAL_COMFY_TARGET
                else VAST_SUPPORTED_COMFY_TASK_KEYS
            )
            provider_label = "Modal" if normalized_target == MODAL_COMFY_TARGET else "Vast"
            if key not in supported_keys:
                message = f"{key} 작업은 {provider_label} 배분을 지원하지 않습니다."
                print(
                    f"[COMFY_ALLOCATION] {provider_label} 작업 배분 값 검증 실패: "
                    f"task={key}, value={value!r}, error={message}"
                )
                raise ComfyTaskAllocationValidationError(message)
            normalized[key] = normalized_target
            continue
        if normalized_target == VIDEO_ENGINE_COMFY_TARGET:
            if key not in VIDEO_ENGINE_SUPPORTED_COMFY_TASK_KEYS:
                message = f"{key} 작업은 영상 전용 엔진 배분을 지원하지 않습니다."
                print(
                    "[COMFY_ALLOCATION] 영상 전용 엔진 작업 배분 값 검증 실패: "
                    f"task={key}, value={value!r}, error={message}"
                )
                raise ComfyTaskAllocationValidationError(message)
            normalized[key] = normalized_target
            continue
        try:
            if isinstance(value, bool):
                raise TypeError("bool은 허용되지 않음")
            parsed = int(value)
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("정수가 아닌 실수는 허용되지 않음")
            if isinstance(value, str) and value.strip() != str(parsed):
                raise ValueError("정수 문자열 형식이 아님")
            allowed_instances = (1, 2, 3)
            if parsed not in allowed_instances:
                raise ValueError(f"허용 인스턴스 {allowed_instances!r}에 없음")
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[COMFY_ALLOCATION] 작업 배분 값 검증 실패: "
                f"task={key}, value={value!r}, error={exc}"
            )
            traceback.print_exc()
            raise ComfyTaskAllocationValidationError(
                f"comfy_task_allocations.{key} 값은 1, 2, 3"
                + (
                    " 또는 modal, vast, video_engine이어야 합니다."
                    if key in VIDEO_ENGINE_SUPPORTED_COMFY_TASK_KEYS
                    else " 또는 modal, vast여야 합니다."
                    if key in MODAL_SUPPORTED_COMFY_TASK_KEYS
                    else " 중 하나여야 합니다."
                )
            ) from exc
        normalized[key] = parsed

    unknown = sorted(str(key) for key in source if key not in COMFY_TASK_KEYS)
    if unknown:
        print(f"[COMFY_ALLOCATION] 알 수 없는 작업 배분 키 무시: {unknown!r}")

    return normalized


def _normalize_comfy_task_remote_parallel(
    raw: Any,
    *,
    allocations: dict[str, int | str] | None = None,
    setting_key: str,
    provider_target: str,
    provider_label: str,
    supported_keys: frozenset[str],
) -> dict[str, bool]:
    """원격 공급자 병렬 사용 여부를 검증하고 중복·미지원 조합을 OFF로 보정한다."""

    if raw is None:
        source: dict[str, Any] = {}
    elif isinstance(raw, dict):
        source = raw
    else:
        message = (
            f"Comfy 작업별 {provider_label} 병렬 설정은 객체여야 합니다: "
            f"type={type(raw).__name__}, value={raw!r}"
        )
        print(
            f"[COMFY_ALLOCATION] {provider_label} 병렬 설정 검증 실패: "
            f"{message}"
        )
        raise ComfyTaskAllocationValidationError(message)

    normalized = {key: False for key in COMFY_TASK_KEYS}
    effective_allocations = (
        allocations
        if allocations is not None
        else dict(DEFAULT_COMFY_TASK_ALLOCATIONS)
    )
    for key in COMFY_TASK_KEYS:
        if key not in source:
            continue
        value = source.get(key)
        if not isinstance(value, bool):
            message = (
                f"{setting_key}.{key} 값은 bool이어야 합니다: "
                f"value={value!r}"
            )
            print(
                f"[COMFY_ALLOCATION] {provider_label} 병렬 값 검증 실패: "
                f"{message}"
            )
            try:
                raise TypeError(message)
            except TypeError as exc:
                traceback.print_exc()
                raise ComfyTaskAllocationValidationError(message) from exc
        normalized[key] = value

    for key in COMFY_TASK_KEYS:
        if not normalized[key]:
            continue
        if key not in supported_keys:
            print(
                f"[COMFY_ALLOCATION] {provider_label} 미지원 작업의 병렬 설정을 "
                "OFF로 보정: "
                f"task={key}"
            )
            normalized[key] = False
        elif effective_allocations.get(key) == provider_target:
            print(
                f"[COMFY_ALLOCATION] 기본 대상과 동일한 {provider_label} 병렬 설정을 "
                "OFF로 보정: "
                f"task={key}, target={effective_allocations.get(key)!r}"
            )
            normalized[key] = False

    unknown = sorted(str(key) for key in source if key not in COMFY_TASK_KEYS)
    if unknown:
        print(
            f"[COMFY_ALLOCATION] 알 수 없는 {provider_label} 병렬 설정 키 무시: "
            f"{unknown!r}"
        )
    return normalized


def normalize_comfy_task_modal_parallel(
    raw: Any,
    *,
    allocations: dict[str, int | str] | None = None,
) -> dict[str, bool]:
    """작업별 Modal 추가 병렬 사용 여부를 정규화한다."""

    return _normalize_comfy_task_remote_parallel(
        raw,
        allocations=allocations,
        setting_key="comfy_task_modal_parallel",
        provider_target=MODAL_COMFY_TARGET,
        provider_label="Modal",
        supported_keys=MODAL_SUPPORTED_COMFY_TASK_KEYS,
    )


def normalize_comfy_task_vast_parallel(
    raw: Any,
    *,
    allocations: dict[str, int | str] | None = None,
) -> dict[str, bool]:
    """작업별 Vast 추가 병렬 사용 여부를 정규화한다."""

    return _normalize_comfy_task_remote_parallel(
        raw,
        allocations=allocations,
        setting_key="comfy_task_vast_parallel",
        provider_target=VAST_COMFY_TARGET,
        provider_label="Vast",
        supported_keys=VAST_SUPPORTED_COMFY_TASK_KEYS,
    )


def comfy_task_definition_payload() -> list[dict[str, Any]]:
    """프런트엔드가 동일한 작업 목록을 렌더링할 수 있는 JSON 안전 구조."""

    return [
        {
            "key": key,
            "label": label,
            "description": description,
            "local_instances": [1, 2, 3],
            "default_instance": DEFAULT_COMFY_TASK_ALLOCATIONS[key],
        }
        for key, label, description in COMFY_TASK_DEFINITIONS
    ]


def select_comfy_instance(
    allocations: dict[str, int | str],
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
    if configured in NONLOCAL_COMFY_TARGETS:
        provider_label = (
            "Modal"
            if configured == MODAL_COMFY_TARGET
            else "Vast"
            if configured == VAST_COMFY_TARGET
            else "영상 전용 엔진"
        )
        print(
            "[COMFY_ALLOCATION] 로컬 인스턴스 선택 거부: "
            f"task={task_key}, configured={provider_label}"
        )
        raise ComfyTaskAllocationValidationError(
            f"{task_key} 작업은 {provider_label} 전용으로 배분되어 로컬 포트를 선택할 수 없습니다."
        )
    running_ids = [instance_id for instance_id in (1, 2, 3) if running.get(instance_id)]
    if len(running_ids) == 1:
        return running_ids[0]
    return configured
