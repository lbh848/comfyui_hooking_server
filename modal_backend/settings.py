from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from remote_comfy_vram import (
    DEFAULT_REMOTE_COMFY_VRAM_MODE,
    normalize_remote_comfy_vram_mode,
)


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")

MODAL_GPU_PROFILES: dict[str, dict[str, str | int | float]] = {
    "L4": {
        "label": "L4",
        "vram_gib": 24,
        "usd_per_second": 0.000222,
        "cuda_arch": "8.9",
    },
    "A10": {
        "label": "A10",
        "vram_gib": 24,
        "usd_per_second": 0.000306,
        "cuda_arch": "8.6",
    },
    "L40S": {
        "label": "L40S",
        "vram_gib": 48,
        "usd_per_second": 0.000542,
        "cuda_arch": "8.9",
    },
    "A100-40GB": {
        "label": "A100 40GB",
        "vram_gib": 40,
        "usd_per_second": 0.000583,
        "cuda_arch": "8.0",
    },
    "RTX-PRO-6000": {
        "label": "RTX PRO 6000",
        "vram_gib": 96,
        "usd_per_second": 0.000842,
        "cuda_arch": "12.0",
    },
}
SUPPORTED_MODAL_GPUS = frozenset(MODAL_GPU_PROFILES)


def normalize_modal_gpu(value: Any, field: str, default: str = "L4") -> str:
    result = str(value or default).strip().upper()
    if result not in SUPPORTED_MODAL_GPUS:
        allowed = ", ".join(MODAL_GPU_PROFILES)
        raise ValueError(f"{field}는 다음 중 하나여야 합니다: {allowed}")
    return result


def _name(value: Any, field: str, default: str) -> str:
    result = str(value or default).strip()
    if not _NAME_RE.fullmatch(result):
        raise ValueError(
            f"{field}는 영문자·숫자로 시작하는 1~63자의 영문자, 숫자, 점, 밑줄, 하이픈만 허용합니다."
        )
    return result


@dataclass(frozen=True)
class ModalSettings:
    enabled: bool = False
    profile: str = "soya-comfy"
    environment: str = "main"
    deployment_name: str = "soya-comfy-worker"
    worker_gpu: str = "L4"
    web_gpu: str = "L4"
    vram_mode: str = DEFAULT_REMOTE_COMFY_VRAM_MODE
    max_concurrency: int = 2
    monthly_credit_usd: float = 30.0
    scaledown_window_seconds: int = 15
    status_refresh_seconds: int = 5
    container_start_max_retries: int = 2
    web_fast: bool = False

    @property
    def gpu(self) -> str:
        """이전 호출자를 위한 작업 워커 GPU alias."""

        return self.worker_gpu

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModalSettings":
        enabled = config.get("modal_enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("modal_enabled는 true/false여야 합니다.")
        web_fast = config.get("modal_web_fast", False)
        if not isinstance(web_fast, bool):
            raise ValueError("modal_web_fast는 true/false여야 합니다.")
        legacy_gpu = config.get("modal_gpu")
        worker_gpu = normalize_modal_gpu(
            config.get("modal_worker_gpu", legacy_gpu),
            "Modal 작업 워커 GPU",
        )
        web_gpu = normalize_modal_gpu(
            config.get("modal_web_gpu", legacy_gpu),
            "Modal 웹 GPU",
        )
        vram_mode = normalize_remote_comfy_vram_mode(
            config.get("modal_vram_mode"),
            "Modal VRAM 모드",
        )
        raw_concurrency = config.get("modal_max_concurrency", 2)
        if isinstance(raw_concurrency, bool):
            raise ValueError("Modal 동시 실행 수는 1~10 사이의 정수여야 합니다.")
        concurrency = int(raw_concurrency)
        if concurrency != float(raw_concurrency) or not 1 <= concurrency <= 10:
            raise ValueError("Modal 동시 실행 수는 1~10 사이의 정수여야 합니다.")
        credit = float(config.get("modal_monthly_credit_usd", 30.0))
        if not math.isfinite(credit) or not 0 < credit <= 100_000:
            raise ValueError("Modal 월간 크레딧은 0보다 큰 유한한 숫자여야 합니다.")
        raw_scaledown = config.get("modal_scaledown_window_seconds", 15)
        if isinstance(raw_scaledown, bool):
            raise ValueError("Modal 유휴 종료 시간은 2~1200초 사이의 정수여야 합니다.")
        scaledown = int(raw_scaledown)
        if scaledown != float(raw_scaledown) or not 2 <= scaledown <= 1200:
            raise ValueError("Modal 유휴 종료 시간은 2~1200초 사이의 정수여야 합니다.")
        raw_refresh = config.get("modal_status_refresh_seconds", 5)
        if isinstance(raw_refresh, bool):
            raise ValueError("Modal 상태 갱신 주기는 2~60초 사이의 정수여야 합니다.")
        refresh = int(raw_refresh)
        if refresh != float(raw_refresh) or not 2 <= refresh <= 60:
            raise ValueError("Modal 상태 갱신 주기는 2~60초 사이의 정수여야 합니다.")
        raw_start_retries = config.get("modal_container_start_max_retries", 2)
        if isinstance(raw_start_retries, bool):
            raise ValueError("Modal 컨테이너 시작 재시도 횟수는 0~10 사이의 정수여야 합니다.")
        start_retries = int(raw_start_retries)
        if (
            start_retries != float(raw_start_retries)
            or not 0 <= start_retries <= 10
        ):
            raise ValueError("Modal 컨테이너 시작 재시도 횟수는 0~10 사이의 정수여야 합니다.")
        return cls(
            enabled=enabled,
            profile=_name(config.get("modal_profile"), "Modal 프로필", "soya-comfy"),
            environment=_name(config.get("modal_environment"), "Modal 환경", "main"),
            deployment_name=_name(
                config.get("modal_deployment_name"),
                "Modal 배포 이름",
                "soya-comfy-worker",
            ),
            worker_gpu=worker_gpu,
            web_gpu=web_gpu,
            vram_mode=vram_mode,
            max_concurrency=concurrency,
            monthly_credit_usd=credit,
            scaledown_window_seconds=scaledown,
            status_refresh_seconds=refresh,
            container_start_max_retries=start_retries,
            web_fast=web_fast,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "profile": self.profile,
            "environment": self.environment,
            "deployment_name": self.deployment_name,
            # gpu는 이전 상태 API 소비자와의 호환을 위한 작업 워커 alias다.
            "gpu": self.worker_gpu,
            "worker_gpu": self.worker_gpu,
            "web_gpu": self.web_gpu,
            "vram_mode": self.vram_mode,
            "gpu_profiles": [
                {
                    "id": gpu_id,
                    **profile,
                    "usd_per_hour": round(
                        float(profile["usd_per_second"]) * 3600,
                        4,
                    ),
                }
                for gpu_id, profile in MODAL_GPU_PROFILES.items()
            ],
            "max_concurrency": self.max_concurrency,
            "monthly_credit_usd": self.monthly_credit_usd,
            "scaledown_window_seconds": self.scaledown_window_seconds,
            "status_refresh_seconds": self.status_refresh_seconds,
            "container_start_max_retries": self.container_start_max_retries,
            "web_fast": self.web_fast,
            "volume_names": {
                "models": f"{self.deployment_name}-models",
                "loras": f"{self.deployment_name}-loras",
                "workflows": f"{self.deployment_name}-workflows",
            },
        }
