from __future__ import annotations

import copy
from typing import Any


INSTALL_MODE_STANDARD = "standard"
INSTALL_MODE_NVIDIA_COMPATIBILITY = "nvidia_compatibility"
# 클라우드 전용 설치. 로컬에서 모델 추론을 하지 않는다.
#
# 왜 별도 모드인가: "클라우드 전용" 은 감지로 정할 수 없다. GPU 가 없다는 사실은
# 클라우드를 쓸 이유는 되지만 근거는 아니다 — 로컬 CPU 로 돌릴 생각일 수도 있다.
# 반대로 GPU 가 있어도 전부 원격에 맡기고 싶을 수 있다. 그래서 사용자가 고른다.
INSTALL_MODE_CLOUD_ONLY = "cloud_only"
SUPPORTED_INSTALL_MODES = frozenset(
    {
        INSTALL_MODE_STANDARD,
        INSTALL_MODE_NVIDIA_COMPATIBILITY,
        INSTALL_MODE_CLOUD_ONLY,
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
    if mode == INSTALL_MODE_CLOUD_ONLY:
        # 로컬 추론이 없으므로 SageAttention·Triton 을 설치할 이유가 없다.
        # 호환 설치와 달리 최소 compute capability 도 두지 않는다 — 로컬 GPU 가
        # 아예 없는 구성이 정상이라, 하한을 두면 정상 구성이 막힌다.
        effective["sageattention_required"] = False
        effective.pop("sageattention", None)
        effective.pop("triton_package", None)
        return effective
    if mode == INSTALL_MODE_STANDARD or effective.get("kind") != "nvidia":
        return effective

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
