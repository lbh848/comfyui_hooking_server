from __future__ import annotations

import copy
from typing import Any


INSTALL_MODE_STANDARD = "standard"
INSTALL_MODE_NVIDIA_COMPATIBILITY = "nvidia_compatibility"
SUPPORTED_INSTALL_MODES = frozenset(
    {
        INSTALL_MODE_STANDARD,
        INSTALL_MODE_NVIDIA_COMPATIBILITY,
    }
)

NVIDIA_COMPATIBILITY_MINIMUM_COMPUTE = "7.5"
NVIDIA_COMPATIBILITY_WARNING = (
    "호환 설치에는 SageAttention과 전용 Triton을 설치하지 않습니다. "
    "설치 후 사용하는 워크플로우에서 SageAttention 노드를 제거하거나 "
    "해당 노드를 disabled로 설정하세요."
)


def normalize_install_mode(value: Any) -> str:
    mode = INSTALL_MODE_STANDARD if value is None else str(value).strip()
    if mode not in SUPPORTED_INSTALL_MODES:
        print(
            "[COMFY_INSTALL][MODE] 지원하지 않는 설치 모드: "
            f"value={value!r}, supported={sorted(SUPPORTED_INSTALL_MODES)!r}"
        )
        raise ValueError(f"지원하지 않는 ComfyUI 설치 모드입니다: {value!r}")
    return mode


def effective_gpu_profile(
    profile: dict[str, Any],
    install_mode: str,
) -> dict[str, Any]:
    mode = normalize_install_mode(install_mode)
    effective = copy.deepcopy(profile)
    effective["install_mode"] = mode
    if mode == INSTALL_MODE_STANDARD:
        return effective
    if effective.get("kind") != "nvidia":
        print(
            "[COMFY_INSTALL][MODE] NVIDIA 호환 설치에 NVIDIA 프로필이 "
            f"아닌 값이 전달되었습니다: profile={effective.get('id')!r}, "
            f"kind={effective.get('kind')!r}"
        )
        raise ValueError("NVIDIA 호환 설치에는 NVIDIA GPU 프로필이 필요합니다.")

    effective["minimum_compute_capability"] = (
        NVIDIA_COMPATIBILITY_MINIMUM_COMPUTE
    )
    effective["sageattention_required"] = False
    effective.pop("sageattention", None)
    effective.pop("triton_package", None)
    return effective


def compatibility_warning(install_mode: str) -> str | None:
    mode = normalize_install_mode(install_mode)
    if mode == INSTALL_MODE_NVIDIA_COMPATIBILITY:
        return NVIDIA_COMPATIBILITY_WARNING
    return None
