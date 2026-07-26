"""워크플로우 선택값, 경로, 기능 호환성을 한곳에서 관리한다.

프론트엔드는 같은 식별자를 사용하고, 서버의 모든 생성 진입점은 이 모듈을
통해 레거시 설정값을 정규화한다. 채팅/프롬프트 본문을 키워드로 추측하지 않고
사용자가 선택한 명시적 프로필만을 판단 근거로 삼는다.
"""

from __future__ import annotations

import copy
import os
from typing import Iterable


ILLUST_V1 = "v1"
ILLUST_V3 = "v3"
ILLUST_V3_ANIMA = "v3_anima"
ILLUST_CHANSUB = "chansub"
ILLUST_CHANSUB_V3_ANIMA = "chansub_v3_anima"

ILLUSTRATION_WORKFLOW_TYPES = (
    ILLUST_V1,
    ILLUST_V3,
    ILLUST_V3_ANIMA,
    ILLUST_CHANSUB,
    ILLUST_CHANSUB_V3_ANIMA,
)

ILLUSTRATION_LOCAL_FILENAMES = {
    ILLUST_V1: "배포_삽화_v1_1.json",
    ILLUST_V3: "배포_삽화_v3_4.json",
    ILLUST_V3_ANIMA: "배포_삽화(ONLY_ANIMA)_v3_4.json",
}

ILLUSTRATION_PROFILE_LABELS = {
    ILLUST_V1: "V1 계열 삽화 워크플로우",
    ILLUST_V3: "V3 계열 삽화 워크플로우 (ANIMA+ILXL)",
    ILLUST_V3_ANIMA: "V3 계열 삽화 워크플로우 (ONLY ANIMA)",
    ILLUST_CHANSUB: "챈섭",
    ILLUST_CHANSUB_V3_ANIMA: "챈섭 + V3 계열 삽화 워크플로우 (ONLY ANIMA)",
}

ASSET_ILXL = "ilxl"
ASSET_ANIMA_ILXL = "anima_ilxl"
ASSET_ANIMA_ONLY = "anima_only"
ASSET_WORKFLOW_TYPES = (ASSET_ILXL, ASSET_ANIMA_ILXL, ASSET_ANIMA_ONLY)

ASSET_PROFILE_LABELS = {
    ASSET_ILXL: "ILXL 에셋 생성 워크플로우",
    ASSET_ANIMA_ILXL: "ANIMA+ILXL 에셋 생성 워크플로우",
    ASSET_ANIMA_ONLY: "ONLY ANIMA 에셋 생성 워크플로우",
}

_LEGACY_ASSET_TYPES = {
    "regular": ASSET_ILXL,
    "anima": ASSET_ANIMA_ILXL,
}

# 파일 내용을 검색해 형식을 추측하지 않는다. 현재 제공되는 복원 프롬프트의
# 명시적 호환 계약이며, 새 파일은 이 표에 등록한 뒤 선택할 수 있다.
RESTORE_PROMPT_COMPATIBILITY = {
    "restore_workflow_prompt_nikke_style_v2.py": frozenset({"v1"}),
    "restore_workflow_prompt_terminater_style.py": frozenset({"v1"}),
    "restore_workflow_prompt_llm_solo.py": frozenset({"v3", "chansub"}),
    # 이 파일은 설정의 자동 prompt_format을 읽어 V1에서는 [ILXL]/[UPSCALE],
    # V3/챈섭에서는 삽화 섹션 형식을 내보내므로 세 계열 모두 호환된다.
    "restore_workflow_prompt_nikke_style_v3.py": frozenset({"v1", "v3", "chansub"}),
}


def normalize_illustration_workflow_type(value, *, fallback: str = ILLUST_V3) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in ILLUSTRATION_WORKFLOW_TYPES:
        return normalized
    print(
        f"[WORKFLOW_PROFILE] 지원하지 않는 삽화 워크플로우 타입: "
        f"{value!r}; {fallback!r} 사용"
    )
    return fallback


def normalize_asset_workflow_type(value, *, fallback: str = ASSET_ILXL) -> str:
    normalized = str(value or "").strip().lower()
    normalized = _LEGACY_ASSET_TYPES.get(normalized, normalized)
    if normalized in ASSET_WORKFLOW_TYPES:
        return normalized
    print(
        f"[WORKFLOW_PROFILE] 지원하지 않는 에셋 워크플로우 타입: "
        f"{value!r}; {fallback!r} 사용"
    )
    return fallback


def infer_legacy_illustration_workflow_type(config: dict) -> str:
    """새 선택 키가 없는 기존 config를 공급자/저장 포맷에서 이관한다."""
    provider = str(config.get("illustration_provider") or "comfy").strip().lower()
    if provider == "chansub":
        return ILLUST_CHANSUB
    toggles = config.get("illustration_context_toggles") or {}
    prompt_format = str(toggles.get("prompt_format") or "v3").strip().lower()
    return ILLUST_V1 if prompt_format == "v1" else ILLUST_V3


def illustration_provider_mode(workflow_type: str) -> str:
    workflow_type = normalize_illustration_workflow_type(workflow_type)
    if workflow_type == ILLUST_CHANSUB:
        return "chansub"
    if workflow_type == ILLUST_CHANSUB_V3_ANIMA:
        return "hybrid"
    return "comfy"


def illustration_prompt_format(workflow_type: str, provider: str | None = None) -> str:
    """선택 프로필과 실제 실행 공급자에 맞는 후속 빌더 형식을 반환한다."""
    workflow_type = normalize_illustration_workflow_type(workflow_type)
    actual_provider = str(provider or illustration_provider_mode(workflow_type)).strip().lower()
    if actual_provider == "chansub":
        return "chansub"
    if workflow_type == ILLUST_V1:
        return "v1"
    return "v3"


def illustration_providers(workflow_type: str) -> tuple[str, ...]:
    workflow_type = normalize_illustration_workflow_type(workflow_type)
    if workflow_type == ILLUST_CHANSUB:
        return ("chansub",)
    if workflow_type == ILLUST_CHANSUB_V3_ANIMA:
        return ("comfy", "chansub")
    return ("comfy",)


def illustration_provider_for_slot(workflow_type: str, slot_index: int) -> str:
    """하이브리드에서는 1번 로컬, 2번 챈섭 순으로 장면을 교대 분배한다."""
    providers = illustration_providers(workflow_type)
    try:
        index = max(1, int(slot_index))
    except (TypeError, ValueError) as exc:
        print(
            f"[WORKFLOW_PROFILE] 삽화 슬롯 인덱스 변환 실패: "
            f"value={slot_index!r}, error={exc}; 1 사용"
        )
        index = 1
    return providers[(index - 1) % len(providers)]


def illustration_local_profile(workflow_type: str) -> str | None:
    workflow_type = normalize_illustration_workflow_type(workflow_type)
    if workflow_type in ILLUSTRATION_LOCAL_FILENAMES:
        return workflow_type
    if workflow_type == ILLUST_CHANSUB_V3_ANIMA:
        return ILLUST_V3_ANIMA
    return None


def illustration_capabilities(workflow_type: str, chansub_workflow_type: str = "anima") -> dict:
    """UI와 서버 검증이 공유하는 삽화 옵션 기능표."""
    workflow_type = normalize_illustration_workflow_type(workflow_type)
    chansub_family = str(chansub_workflow_type or "anima").strip().lower()
    if chansub_family not in ("anima", "sdxl"):
        print(
            f"[WORKFLOW_PROFILE] 알 수 없는 챈섭 계열: {chansub_workflow_type!r}; "
            "anima 사용"
        )
        chansub_family = "anima"

    if workflow_type == ILLUST_V1:
        return {
            "anima_presets": True,
            "sdxl_presets": False,
            "ipadapter": False,
            "local_advanced": False,
            "chansub_advanced": False,
            "multi_char_mask": False,
            "v1_limited": True,
        }
    if workflow_type == ILLUST_V3:
        return {
            "anima_presets": True,
            "sdxl_presets": True,
            "ipadapter": True,
            "local_advanced": True,
            "chansub_advanced": False,
            "multi_char_mask": True,
            "v1_limited": False,
        }
    if workflow_type == ILLUST_V3_ANIMA:
        return {
            "anima_presets": True,
            "sdxl_presets": False,
            "ipadapter": False,
            "local_advanced": True,
            "chansub_advanced": False,
            "multi_char_mask": True,
            "v1_limited": False,
        }
    if workflow_type == ILLUST_CHANSUB:
        return {
            "anima_presets": chansub_family == "anima",
            "sdxl_presets": chansub_family == "sdxl",
            "ipadapter": False,
            "local_advanced": False,
            "chansub_advanced": True,
            "multi_char_mask": False,
            "v1_limited": False,
        }
    # 하이브리드는 사용자가 승인한 교집합 정책: 챈섭 ANIMA에서 쓸 수 있는
    # 프리셋/이미지 크기만 편집하고 로컬 전용 고급 옵션은 회색 처리한다.
    return {
        "anima_presets": True,
        "sdxl_presets": False,
        "ipadapter": False,
        "local_advanced": False,
        "chansub_advanced": True,
        "multi_char_mask": False,
        "v1_limited": False,
    }


def asset_capabilities(workflow_type: str) -> dict:
    workflow_type = normalize_asset_workflow_type(workflow_type)
    return {
        "ilxl": workflow_type in (ASSET_ILXL, ASSET_ANIMA_ILXL),
        "anima": workflow_type in (ASSET_ANIMA_ILXL, ASSET_ANIMA_ONLY),
        "dual": workflow_type == ASSET_ANIMA_ILXL,
        "pose": workflow_type == ASSET_ILXL,
        "ipadapter": workflow_type in (ASSET_ILXL, ASSET_ANIMA_ILXL),
        "face_lora": workflow_type in (ASSET_ANIMA_ILXL, ASSET_ANIMA_ONLY),
        "seed": workflow_type in (ASSET_ANIMA_ILXL, ASSET_ANIMA_ONLY),
    }


def restore_families(workflow_type: str) -> tuple[str, ...]:
    workflow_type = normalize_illustration_workflow_type(workflow_type)
    if workflow_type == ILLUST_V1:
        return ("v1",)
    if workflow_type == ILLUST_CHANSUB:
        return ("chansub",)
    if workflow_type == ILLUST_CHANSUB_V3_ANIMA:
        # 하이브리드는 실제 빌더가 로컬 V3와 챈섭으로 분리되므로 양쪽에서
        # 유효한 복원 프롬프트만 노출한다.
        return ("v3", "chansub")
    return ("v3",)


def restore_family(workflow_type: str) -> str:
    """API/UI 표시용 복원 계열 문자열을 반환한다."""
    return "+".join(restore_families(workflow_type))


def compatible_restore_prompt_files(workflow_type: str, files: Iterable[str]) -> list[str]:
    required_families = frozenset(restore_families(workflow_type))
    return [
        str(filename)
        for filename in files
        if required_families.issubset(
            RESTORE_PROMPT_COMPATIBILITY.get(str(filename), frozenset())
        )
    ]


def is_restore_prompt_compatible(filename: str, workflow_type: str) -> bool:
    if not str(filename or "").strip():
        return True
    return str(filename) in compatible_restore_prompt_files(workflow_type, [str(filename)])


def _default_illustration_source_paths(config: dict) -> dict:
    raw_paths = config.get("illustration_workflow_source_paths")
    paths = copy.deepcopy(raw_paths) if isinstance(raw_paths, dict) else {}
    legacy_path = str(config.get("comfy_workflow_source_path") or "").strip()
    base_dir = os.path.dirname(legacy_path) if legacy_path else ""
    if not base_dir:
        base_dir = next(
            (
                os.path.dirname(str(path))
                for path in paths.values()
                if str(path or "").strip()
            ),
            "",
        )
    for profile, filename in ILLUSTRATION_LOCAL_FILENAMES.items():
        if not str(paths.get(profile) or "").strip():
            paths[profile] = os.path.join(base_dir, filename) if base_dir else filename
    return {key: str(paths.get(key) or "") for key in ILLUSTRATION_LOCAL_FILENAMES}


def normalize_workflow_config(config: dict) -> dict:
    """설정 dict를 제자리에서 새 프로필 계약으로 정규화한다."""
    raw_type = config.get("illustration_workflow_type")
    if raw_type in (None, ""):
        raw_type = infer_legacy_illustration_workflow_type(config)
    illustration_type = normalize_illustration_workflow_type(raw_type)
    config["illustration_workflow_type"] = illustration_type
    config["illustration_workflow_source_paths"] = _default_illustration_source_paths(config)
    config["illustration_provider"] = illustration_provider_mode(illustration_type)

    if illustration_type == ILLUST_CHANSUB_V3_ANIMA:
        config["chansub_workflow_type"] = "anima"
    elif str(config.get("chansub_workflow_type") or "").strip().lower() not in ("anima", "sdxl"):
        print(
            f"[WORKFLOW_PROFILE] 잘못된 chansub_workflow_type: "
            f"{config.get('chansub_workflow_type')!r}; anima 사용"
        )
        config["chansub_workflow_type"] = "anima"

    toggles = config.get("illustration_context_toggles")
    if not isinstance(toggles, dict):
        print(
            f"[WORKFLOW_PROFILE] illustration_context_toggles가 객체가 아님: "
            f"{type(toggles).__name__}; 새 객체 사용"
        )
        toggles = {}
        config["illustration_context_toggles"] = toggles
    toggles["prompt_format"] = illustration_prompt_format(illustration_type)

    local_profile = illustration_local_profile(illustration_type)
    if local_profile:
        source_path = config["illustration_workflow_source_paths"].get(local_profile, "")
        if source_path:
            config["comfy_workflow_source_path"] = source_path

    config["asset_workflow_type"] = normalize_asset_workflow_type(
        config.get("asset_workflow_type")
    )
    return config


def active_illustration_source_path(config: dict, workflow_type: str | None = None) -> str:
    profile = normalize_illustration_workflow_type(
        workflow_type or config.get("illustration_workflow_type")
    )
    local_profile = illustration_local_profile(profile)
    if not local_profile:
        print(f"[WORKFLOW_PROFILE] 로컬 소스가 없는 삽화 프로필: {profile}")
        return ""
    paths = config.get("illustration_workflow_source_paths") or {}
    if not isinstance(paths, dict):
        print(
            f"[WORKFLOW_PROFILE] illustration_workflow_source_paths가 객체가 아님: "
            f"{type(paths).__name__}"
        )
        return ""
    source_path = str(paths.get(local_profile) or "").strip()
    if not source_path:
        print(f"[WORKFLOW_PROFILE] 삽화 소스 경로가 비어 있음: profile={local_profile}")
    return source_path
