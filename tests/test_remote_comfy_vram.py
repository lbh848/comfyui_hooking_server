from __future__ import annotations

import pytest

from remote_comfy_vram import (
    normalize_remote_comfy_vram_mode,
    remote_comfy_vram_arguments,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, ["--highvram"]),
        ("AUTO", []),
        ("highvram", ["--highvram"]),
        ("normalvram", ["--normalvram"]),
        ("lowvram", ["--lowvram"]),
        ("novram", ["--novram"]),
    ],
)
def test_remote_comfy_vram_arguments(mode: str | None, expected: list[str]) -> None:
    assert remote_comfy_vram_arguments(mode) == expected


def test_remote_comfy_vram_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="원격 테스트 VRAM 모드"):
        normalize_remote_comfy_vram_mode("gpu-only", "원격 테스트 VRAM 모드")
