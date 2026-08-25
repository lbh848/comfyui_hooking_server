"""Modal/Vast 원격 ComfyUI 프로세스의 VRAM 모드 설정."""

from __future__ import annotations

from typing import Any


DEFAULT_REMOTE_COMFY_VRAM_MODE = "highvram"
REMOTE_COMFY_VRAM_MODES = (
    "auto",
    "highvram",
    "normalvram",
    "lowvram",
    "novram",
)
SUPPORTED_REMOTE_COMFY_VRAM_MODES = frozenset(REMOTE_COMFY_VRAM_MODES)


def normalize_remote_comfy_vram_mode(value: Any, field: str) -> str:
    normalized = str(value or DEFAULT_REMOTE_COMFY_VRAM_MODE).strip().lower()
    if normalized not in SUPPORTED_REMOTE_COMFY_VRAM_MODES:
        allowed = ", ".join(REMOTE_COMFY_VRAM_MODES)
        raise ValueError(f"{field}는 다음 중 하나여야 합니다: {allowed}")
    return normalized


def remote_comfy_vram_arguments(mode: Any) -> list[str]:
    normalized = normalize_remote_comfy_vram_mode(mode, "원격 ComfyUI VRAM 모드")
    return [] if normalized == "auto" else [f"--{normalized}"]
