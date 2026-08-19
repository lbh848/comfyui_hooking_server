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

# 작업 종류 ↔ 설치 매니페스트 워크플로우 바인딩 id.
#
# 왜 필요한가: 모델을 저장소에서 원격 볼륨으로 직접 받는 구성에서 설치기가
# "로컬에서 실제로 실행되는 작업이 쓰는 모델만" 받으려면, 작업에서 워크플로우
# 바인딩을 거쳐 매니페스트 model_ids 로 내려가는 길이 필요하다. 그 길이 없었다.
#
# 왜 표인가: 파일명이나 작업 이름에서 추론하면 워크플로우 팩이 바뀔 때 조용히
# 어긋난다. 매니페스트의 모든 바인딩이 이 표 또는 UNMAPPED_WORKFLOW_BINDINGS 에
# 있어야 한다는 사실은 테스트로 강제한다.
#
# 하나의 바인딩이 여러 작업에 걸릴 수 있다(LoRA 학습 3종이 같은 워크플로우를
# 공유한다). 그래서 "한 작업이라도 로컬이면 그 바인딩은 로컬 필요"로 합집합을
# 취한다 — 부족한 쪽보다 남는 쪽이 안전하다.
COMFY_TASK_WORKFLOW_BINDINGS: dict[str, tuple[str, ...]] = {
    # 삽화·재생성은 같은 워크플로우 계열을 탄다
    # (queue_manager.py:3233·3296 → generate_image_with_prompt →
    #  server.py:1324 get_comfy_workflow_source_path).
    "illustration": (
        "comfy_workflow_source_path",
        "illustration_workflow_source_paths.v1",
        "illustration_workflow_source_paths.v3",
        "illustration_workflow_source_paths.v3_anima",
    ),
    "restore_regenerate": (
        "comfy_workflow_source_path",
        "illustration_workflow_source_paths.v1",
        "illustration_workflow_source_paths.v3",
        "illustration_workflow_source_paths.v3_anima",
    ),
    "asset_generation": (
        "asset_workflow_source_path",
        "anima_asset_workflow_source_path",
        "anima_only_asset_workflow_source_path",
    ),
    # EDIT 계열 = Qwen Edit + ANIMA Inpainting (COMFY_TASK_DEFINITIONS 참고).
    "qwen_edit": (
        "qwen_edit_workflow_source_path",
        "anima_inpainting_workflow_source_path",
    ),
    "tag_analysis": (
        "tag_analysis_workflow_source_path",
        "asset_tag_analysis_workflow_source_path",
    ),
    # 복장 추출 워크플로우는 배포되지 않는다(설치기 excluded_filenames).
    # 매니페스트에 대응 바인딩이 없어 빈 튜플이 정답이다.
    "outfit": (),
    # lora_training_* 는 에셋·봇·인스턴스 학습이 공유한다
    # (queue_manager.py:3691·3815·4694).
    "asset_lora_training": (
        "lora_training_workflow_source_paths.anima",
        "lora_training_workflow_source_paths.sdxl",
    ),
    "bot_lora_training": (
        "lora_training_workflow_source_paths.anima",
        "lora_training_workflow_source_paths.sdxl",
    ),
    "instance_lora": (
        "lora_training_workflow_source_paths.anima",
        "lora_training_workflow_source_paths.sdxl",
        "style_lora_training_workflow_source_paths.anima",
        "style_lora_training_workflow_source_paths.sdxl",
    ),
    "face_extract": ("face_extract_workflow_source_path",),
    "utility_debug": (
        "utility_workflow_source_path",
        "debug_workflow_source_path",
    ),
    "video_generation": (
        "video_workflow_source_paths.i2v",
        "video_workflow_source_paths.first_last",
        "video_workflow_source_paths.ref2v",
        "video_workflow_source_paths.i2v_fast",
        "video_workflow_source_paths.first_last_fast",
        "video_workflow_source_paths.ref2v_fast",
    ),
}

# 매니페스트에 있으나 어떤 작업에도 매이지 않는 바인딩.
# 지금은 없다. 새 워크플로우가 추가됐는데 어느 작업이 쓰는지 아직 모를 때
# 여기에 넣으면 커버리지 테스트는 통과하되 기록은 남는다.
UNMAPPED_WORKFLOW_BINDINGS: frozenset[str] = frozenset()


def is_remote_allocation(value: Any) -> bool:
    """배분 값이 원격 대상(modal/vast)인지."""

    return isinstance(value, str) and value.strip().lower() in REMOTE_COMFY_TARGETS


def local_comfy_task_keys(allocations: Any) -> tuple[str, ...]:
    """원격이 아닌 대상에 배분된 작업 키들.

    키가 없으면 로컬로 본다 — DEFAULT_COMFY_TASK_ALLOCATIONS 가 로컬 인스턴스이고,
    모르는 것을 원격으로 가정하면 필요한 모델을 안 받게 되기 때문이다.
    """

    source = allocations if isinstance(allocations, dict) else {}
    return tuple(
        key
        for key in COMFY_TASK_KEYS
        if not is_remote_allocation(source.get(key))
    )


def local_required_binding_ids(allocations: Any) -> frozenset[str]:
    """로컬에서 실행되는 작업들이 쓰는 워크플로우 바인딩 id 집합."""

    bindings: set[str] = set()
    for key in local_comfy_task_keys(allocations):
        bindings.update(COMFY_TASK_WORKFLOW_BINDINGS.get(key, ()))
    return frozenset(bindings)

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
