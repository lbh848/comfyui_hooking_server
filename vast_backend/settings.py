"""Vast.ai 설정 모델 — config.json의 vast_* 키를 검증·변환한다."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

# Modal과 동일한 런타임 바이너리(CUDA 12.8 + PyTorch 2.11 + SageAttention
# sm_80/86/89/120 커널 + ComfyUI)를 쓰기 위해 Docker Hub 공개 이미지를
# digest로 고정한다. modal_app.py의 RUNTIME_IMAGE_REF와 같은 이미지다.
DEFAULT_RUNTIME_IMAGE = (
    "docker.io/bh848/soya-comfy-runtime@"
    "sha256:2f63f258f60614cb15bad285e41bff11643fb46a88b19419b974931bc5e4b135"
)

# 오퍼 브라우저 기본 프리셋. 가격은 USD/시간 상한.
GPU_PRESETS: dict[str, dict[str, Any]] = {
    "budget_3090": {
        "label": "예산형 3090 (24GB)",
        "gpu_names": ["RTX 3090"],
    },
    "fast_4090": {
        "label": "속도형 4090 (24GB)",
        "gpu_names": ["RTX 4090", "RTX 4090D"],
    },
    "beefy_vram": {
        "label": "대용량 VRAM (5090/A100)",
        "gpu_names": ["RTX 5090", "A100_PCIE_80GB", "A100_SXM4_80GB"],
    },
    "custom": {
        "label": "사용자 지정",
        "gpu_names": [],
    },
}


def _bool(config: Mapping[str, Any], key: str, default: bool, label: str) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{label}은 true/false여야 합니다.")
    return value


def load_key_files(project_root: str | Path) -> dict[str, str]:
    """key/ 폴더의 키 파일들을 읽는다. 없으면 빈 문자열."""
    import json
    from pathlib import Path as _Path

    keys = {"vast_api_key": "", "civitai_api_key": ""}
    for name, filename in (
        ("vast_api_key", "vast_key.json"),
        ("civitai_api_key", "civitai_key.json"),
    ):
        path = _Path(project_root).resolve() / "key" / filename
        if not path.is_file():
            print(f"[VAST_KEY] 키 파일 없음: {path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            keys[name] = str(data.get("api_key") or "").strip()
        except (OSError, ValueError) as exc:
            print(f"[VAST_KEY] 키 파일 읽기 실패: {path}, error={exc}")
    return keys


def _int_range(
    config: Mapping[str, Any],
    key: str,
    default: int,
    low: int,
    high: int,
    label: str,
) -> int:
    raw = config.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"{label}은 {low}~{high} 사이의 정수여야 합니다.")
    value = int(raw)
    if value != float(raw) or not low <= value <= high:
        raise ValueError(f"{label}은 {low}~{high} 사이의 정수여야 합니다.")
    return value


def _float_range(
    config: Mapping[str, Any],
    key: str,
    default: float,
    low: float,
    high: float,
    label: str,
) -> float:
    raw = config.get(key, default)
    value = float(raw)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label}은 {low}~{high} 사이의 숫자여야 합니다.")
    return value


@dataclass(frozen=True)
class VastSettings:
    enabled: bool = False
    api_key: str = ""
    civitai_api_key: str = ""
    runtime_image: str = DEFAULT_RUNTIME_IMAGE
    # 안정 모드(verified + On-Demand) / 반값 모드(interruptible)
    verified_only: bool = True
    on_demand: bool = True
    min_cpu_ram_gb: int = 32
    max_price_usd_hr: float = 0.40
    disk_buffer_gb: int = 10
    output_headroom_gb_video: int = 10
    output_headroom_gb_image: int = 3
    status_refresh_seconds: int = 5

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        vast_api_key: str = "",
        civitai_api_key: str = "",
    ) -> "VastSettings":
        # 키는 key/vast_key.json, key/civitai_key.json 우선 (프로젝트 키 보관 관례).
        # config.json의 vast_api_key/vast_civitai_api_key는 폴백으로만 쓴다.
        return cls(
            enabled=_bool(config, "vast_enabled", False, "vast_enabled"),
            api_key=vast_api_key or str(config.get("vast_api_key") or "").strip(),
            civitai_api_key=civitai_api_key
            or str(config.get("vast_civitai_api_key") or "").strip(),
            runtime_image=str(
                config.get("vast_runtime_image") or DEFAULT_RUNTIME_IMAGE
            ).strip(),
            verified_only=_bool(
                config, "vast_verified_only", True, "vast_verified_only"
            ),
            on_demand=_bool(config, "vast_on_demand", True, "vast_on_demand"),
            min_cpu_ram_gb=_int_range(
                config, "vast_min_cpu_ram_gb", 32, 4, 1024, "Vast 최소 시스템 RAM(GB)"
            ),
            max_price_usd_hr=_float_range(
                config, "vast_max_price_usd_hr", 0.40, 0.01, 20.0, "Vast 시간당 가격 상한(USD)"
            ),
            disk_buffer_gb=_int_range(
                config, "vast_disk_buffer_gb", 10, 0, 500, "Vast 디스크 버퍼(GB)"
            ),
            output_headroom_gb_video=_int_range(
                config,
                "vast_output_headroom_gb_video",
                10,
                1,
                200,
                "Vast 영상 출력 여유(GB)",
            ),
            output_headroom_gb_image=_int_range(
                config,
                "vast_output_headroom_gb_image",
                3,
                1,
                200,
                "Vast 이미지 출력 여유(GB)",
            ),
            status_refresh_seconds=_int_range(
                config, "vast_status_refresh_seconds", 5, 2, 60, "Vast 상태 갱신 주기(초)"
            ),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "has_api_key": bool(self.api_key),
            "has_civitai_api_key": bool(self.civitai_api_key),
            "runtime_image": self.runtime_image,
            "verified_only": self.verified_only,
            "on_demand": self.on_demand,
            "min_cpu_ram_gb": self.min_cpu_ram_gb,
            "max_price_usd_hr": self.max_price_usd_hr,
            "disk_buffer_gb": self.disk_buffer_gb,
            "output_headroom_gb_video": self.output_headroom_gb_video,
            "output_headroom_gb_image": self.output_headroom_gb_image,
            "status_refresh_seconds": self.status_refresh_seconds,
            "gpu_presets": GPU_PRESETS,
        }

    def recommend_disk_gb(
        self,
        *,
        image_gb: float = 10.0,
        model_gb: float = 0.0,
        lora_gb: float = 0.0,
        includes_video: bool = False,
    ) -> int:
        """이미지+모델+LoRA+출력 여유+버퍼로 권장 디스크(GB)를 계산한다."""
        headroom = (
            self.output_headroom_gb_video
            if includes_video
            else self.output_headroom_gb_image
        )
        total = image_gb + model_gb + lora_gb + headroom + self.disk_buffer_gb
        return max(10, int(total + 0.999))
