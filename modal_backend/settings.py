from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


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
    gpu: str = "L4"
    max_concurrency: int = 2
    monthly_credit_usd: float = 30.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModalSettings":
        enabled = config.get("modal_enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("modal_enabled는 true/false여야 합니다.")
        gpu = str(config.get("modal_gpu") or "L4").strip().upper()
        if gpu != "L4":
            raise ValueError("현재 원격 GPU 프로필은 L4만 지원합니다.")
        raw_concurrency = config.get("modal_max_concurrency", 2)
        if isinstance(raw_concurrency, bool):
            raise ValueError("Modal 동시 실행 수는 1~10 사이의 정수여야 합니다.")
        concurrency = int(raw_concurrency)
        if concurrency != float(raw_concurrency) or not 1 <= concurrency <= 10:
            raise ValueError("Modal 동시 실행 수는 1~10 사이의 정수여야 합니다.")
        credit = float(config.get("modal_monthly_credit_usd", 30.0))
        if not math.isfinite(credit) or not 0 < credit <= 100_000:
            raise ValueError("Modal 월간 크레딧은 0보다 큰 유한한 숫자여야 합니다.")
        return cls(
            enabled=enabled,
            profile=_name(config.get("modal_profile"), "Modal 프로필", "soya-comfy"),
            environment=_name(config.get("modal_environment"), "Modal 환경", "main"),
            deployment_name=_name(
                config.get("modal_deployment_name"),
                "Modal 배포 이름",
                "soya-comfy-worker",
            ),
            gpu=gpu,
            max_concurrency=concurrency,
            monthly_credit_usd=credit,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "profile": self.profile,
            "environment": self.environment,
            "deployment_name": self.deployment_name,
            "gpu": self.gpu,
            "max_concurrency": self.max_concurrency,
            "monthly_credit_usd": self.monthly_credit_usd,
            "volume_names": {
                "models": f"{self.deployment_name}-models",
                "loras": f"{self.deployment_name}-loras",
                "workflows": f"{self.deployment_name}-workflows",
            },
        }
