from __future__ import annotations

import pytest

from comfy_installer.manifest import load_install_manifest
from comfy_installer.install_modes import (
    INSTALL_MODE_NVIDIA_COMPATIBILITY,
)
from comfy_installer.system_probe import select_gpu_profile


def _nvidia(
    *,
    driver: str,
    compute: str,
    driver_cuda: str | None = None,
    name: str = "NVIDIA GeForce RTX test",
) -> dict:
    return {
        "available": True,
        "driver_cuda": driver_cuda,
        "gpus": [
            {
                "name": name,
                "driver_version": driver,
                "compute_capability": compute,
            }
        ],
    }


@pytest.mark.parametrize(
    ("driver", "compute", "driver_cuda", "expected_profile"),
    [
        ("591.74", "8.9", "13.0", "nvidia-cu130"),
        ("580.00", "8.6", None, "nvidia-cu130"),
        ("579.99", "8.6", "12.9", "nvidia-cu128"),
        ("570.65", "8.6", "12.4", "nvidia-cu128"),
    ],
)
def test_gpu_profile_uses_driver_and_compute_capability(
    driver: str,
    compute: str,
    driver_cuda: str | None,
    expected_profile: str,
) -> None:
    manifest = load_install_manifest()

    selected = select_gpu_profile(
        manifest,
        _nvidia(
            driver=driver,
            compute=compute,
            driver_cuda=driver_cuda,
        ),
    )

    assert selected["id"] == expected_profile


def test_gpu_profile_does_not_require_nvidia_smi_cuda_summary() -> None:
    manifest = load_install_manifest()

    selected = select_gpu_profile(
        manifest,
        _nvidia(driver="580.00", compute="8.6", driver_cuda=None),
    )

    assert selected["id"] == "nvidia-cu130"


def test_gpu_profile_falls_back_to_cpu_for_old_driver() -> None:
    manifest = load_install_manifest()

    selected = select_gpu_profile(
        manifest,
        _nvidia(driver="570.64", compute="8.6", driver_cuda="12.8"),
    )

    assert selected["id"] == "cpu"
    assert "NVIDIA 그래픽 드라이버" in selected["fallback_reason"]


def test_standard_gpu_profile_falls_back_to_cpu_for_turing() -> None:
    manifest = load_install_manifest()

    selected = select_gpu_profile(
        manifest,
        _nvidia(driver="591.74", compute="7.5", driver_cuda="13.0"),
    )

    assert selected["id"] == "cpu"
    assert "SageAttention" in selected["fallback_reason"]


@pytest.mark.parametrize(
    ("driver", "expected_profile"),
    [
        ("591.74", "nvidia-cu130"),
        ("579.99", "nvidia-cu128"),
    ],
)
def test_nvidia_compatibility_profile_accepts_turing_without_sageattention(
    driver: str,
    expected_profile: str,
) -> None:
    manifest = load_install_manifest()

    selected = select_gpu_profile(
        manifest,
        _nvidia(driver=driver, compute="7.5", driver_cuda="13.0"),
        install_mode=INSTALL_MODE_NVIDIA_COMPATIBILITY,
    )

    assert selected["id"] == expected_profile
    assert selected["minimum_compute_capability"] == "7.5"
    assert selected["sageattention_required"] is False
    assert "sageattention" not in selected
    assert "triton_package" not in selected


def test_nvidia_compatibility_profile_falls_back_for_pre_turing_gpu() -> None:
    manifest = load_install_manifest()

    selected = select_gpu_profile(
        manifest,
        _nvidia(driver="591.74", compute="6.1", driver_cuda="13.0"),
        install_mode=INSTALL_MODE_NVIDIA_COMPATIBILITY,
    )

    assert selected["id"] == "cpu"
    assert "Turing" in selected["fallback_reason"]


def test_nvidia_compatibility_profile_allows_no_gpu_with_cpu_fallback() -> None:
    manifest = load_install_manifest()

    selected = select_gpu_profile(
        manifest,
        {"available": False, "gpus": [], "driver_cuda": None},
        install_mode=INSTALL_MODE_NVIDIA_COMPATIBILITY,
    )

    assert selected["id"] == "cpu"
    assert "GPU 없음" in selected["fallback_reason"]


def test_gpu_profile_uses_cpu_only_when_nvidia_is_absent() -> None:
    manifest = load_install_manifest()

    selected = select_gpu_profile(
        manifest,
        {"available": False, "gpus": [], "driver_cuda": None},
    )

    assert selected["id"] == "cpu"
