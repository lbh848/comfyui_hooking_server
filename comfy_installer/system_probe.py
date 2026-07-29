from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

from .install_modes import (
    INSTALL_MODE_NVIDIA_COMPATIBILITY,
    INSTALL_MODE_STANDARD,
    effective_gpu_profile,
    normalize_install_mode,
)
from .manifest import InstallManifest


class SystemProbeError(RuntimeError):
    """운영체제, GPU, 도구 또는 디스크 검사 실패."""


MINIMUM_UV_VERSION = (0, 11, 8)
_NUMERIC_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){0,3}$")


def _run_probe(command: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PROBE] 명령 실행 실패: "
            f"command={command!r}, error={exc}"
        )
        traceback.print_exc()
        raise SystemProbeError(f"시스템 검사 명령 실패: {command[0]}") from exc


def _tool_version(command: list[str]) -> str:
    result = _run_probe(command)
    if result.returncode != 0:
        print(
            "[COMFY_INSTALL][PROBE] 필수 도구 검사 실패: "
            f"command={command!r}, code={result.returncode}, "
            f"stderr={result.stderr.strip()!r}"
        )
        raise SystemProbeError(f"필수 도구를 사용할 수 없습니다: {command[0]}")
    return (result.stdout or result.stderr).strip().splitlines()[0]


def _require_uv_version(raw_version: str) -> str:
    match = re.search(r"\buv\s+(\d+)\.(\d+)\.(\d+)\b", raw_version)
    if match is None:
        print(
            "[COMFY_INSTALL][PROBE] uv 버전 형식 확인 실패: "
            f"value={raw_version!r}"
        )
        raise SystemProbeError(f"uv 버전을 확인할 수 없습니다: {raw_version}")
    parsed = tuple(int(value) for value in match.groups())
    if parsed < MINIMUM_UV_VERSION:
        required = ".".join(str(value) for value in MINIMUM_UV_VERSION)
        actual = ".".join(str(value) for value in parsed)
        print(
            "[COMFY_INSTALL][PROBE] uv 버전이 너무 오래되었습니다: "
            f"actual={actual}, required={required}"
        )
        raise SystemProbeError(
            "프로젝트 내부 Python 설치에는 더 최신 uv가 필요합니다: "
            f"actual={actual}, required>={required}. run_en.bat로 서버를 실행하세요."
        )
    return raw_version


def _probe_nvidia() -> dict[str, Any]:
    try:
        query = _run_probe(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        )
    except SystemProbeError:
        print(
            "[COMFY_INSTALL][PROBE] nvidia-smi를 실행할 수 없어 "
            "CPU 프로필 후보로 처리합니다."
        )
        return {"available": False, "gpus": [], "driver_cuda": None}
    if query.returncode != 0:
        print(
            "[COMFY_INSTALL][PROBE] NVIDIA GPU가 없거나 nvidia-smi 실행 실패: "
            f"code={query.returncode}, stderr={query.stderr.strip()!r}"
        )
        return {"available": False, "gpus": [], "driver_cuda": None}

    gpus = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            print(f"[COMFY_INSTALL][PROBE] nvidia-smi GPU 행 형식 오류: {line!r}")
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "compute_capability": parts[2],
            }
        )
    if not gpus:
        print("[COMFY_INSTALL][PROBE] nvidia-smi 결과에 GPU 행이 없습니다.")
        return {"available": False, "gpus": [], "driver_cuda": None}

    try:
        summary = _run_probe(["nvidia-smi"])
        combined = f"{summary.stdout}\n{summary.stderr}"
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", combined)
        driver_cuda = match.group(1) if match else None
    except SystemProbeError:
        driver_cuda = None
        print(
            "[COMFY_INSTALL][PROBE] nvidia-smi 요약 조회에 실패했지만 "
            "드라이버 버전 기반 프로필 선택은 계속합니다."
        )
    if driver_cuda is None:
        print("[COMFY_INSTALL][PROBE] NVIDIA 드라이버 CUDA 상한을 찾지 못했습니다.")
    return {"available": True, "gpus": gpus, "driver_cuda": driver_cuda}


def _parse_numeric_version(
    value: Any,
    *,
    label: str,
    width: int = 4,
) -> tuple[int, ...]:
    raw = str(value).strip()
    if not _NUMERIC_VERSION_RE.fullmatch(raw):
        print(
            "[COMFY_INSTALL][PROBE] 숫자 버전 형식 오류: "
            f"label={label}, value={value!r}"
        )
        raise SystemProbeError(f"{label} 버전 형식이 잘못되었습니다: {value!r}")
    parts = [int(part) for part in raw.split(".")]
    if len(parts) > width:
        print(
            "[COMFY_INSTALL][PROBE] 숫자 버전 구성요소 초과: "
            f"label={label}, value={value!r}, width={width}"
        )
        raise SystemProbeError(f"{label} 버전 형식이 너무 깁니다: {value!r}")
    return tuple(parts + [0] * (width - len(parts)))


def select_gpu_profile(
    manifest: InstallManifest,
    nvidia: dict[str, Any],
    *,
    install_mode: str = INSTALL_MODE_STANDARD,
) -> dict[str, Any]:
    mode = normalize_install_mode(install_mode)
    profiles = manifest.python["gpu_profiles"]
    if not nvidia.get("available"):
        if mode == INSTALL_MODE_NVIDIA_COMPATIBILITY:
            message = (
                "NVIDIA 호환 설치를 선택했지만 NVIDIA GPU를 찾지 "
                "못했습니다."
            )
            print(f"[COMFY_INSTALL][PROBE] GPU 프로필 선택 실패: {message}")
            raise SystemProbeError(message)
        for profile in profiles:
            if profile.get("kind") == "cpu":
                print(
                    "[COMFY_INSTALL][PROBE] NVIDIA GPU가 없어 CPU 프로필을 "
                    f"선택합니다: profile={profile.get('id')}"
                )
                return profile
        print("[COMFY_INSTALL][PROBE] CPU 설치 프로필이 없습니다.")
        raise SystemProbeError("NVIDIA GPU가 없고 CPU 설치 프로필도 없습니다.")

    gpus = nvidia.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        print(
            "[COMFY_INSTALL][PROBE] NVIDIA 사용 가능 상태이지만 GPU 정보가 "
            f"비어 있습니다: nvidia={nvidia!r}"
        )
        raise SystemProbeError("NVIDIA GPU 상세 정보를 확인하지 못했습니다.")

    detected: list[tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]] = []
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            print(
                "[COMFY_INSTALL][PROBE] NVIDIA GPU 정보 형식 오류: "
                f"index={index}, value={gpu!r}"
            )
            raise SystemProbeError("NVIDIA GPU 정보 형식이 잘못되었습니다.")
        driver = _parse_numeric_version(
            gpu.get("driver_version"),
            label=f"GPU {index} NVIDIA 드라이버",
        )
        compute = _parse_numeric_version(
            gpu.get("compute_capability"),
            label=f"GPU {index} compute capability",
            width=2,
        )
        detected.append((driver, compute, gpu))

    candidates = []
    for base_profile in profiles:
        if base_profile.get("kind") != "nvidia":
            continue
        profile = effective_gpu_profile(base_profile, mode)
        minimum_driver = _parse_numeric_version(
            profile.get("minimum_driver_version"),
            label=f"{profile.get('id')} 최소 NVIDIA 드라이버",
        )
        minimum_compute = _parse_numeric_version(
            profile.get("minimum_compute_capability"),
            label=f"{profile.get('id')} 최소 compute capability",
            width=2,
        )
        compatible_gpus = [
            gpu
            for driver, compute, gpu in detected
            if driver >= minimum_driver and compute >= minimum_compute
        ]
        if compatible_gpus:
            cuda_priority = _parse_numeric_version(
                profile.get("torch_cuda"),
                label=f"{profile.get('id')} PyTorch CUDA",
                width=2,
            )
            candidates.append(
                (minimum_driver, cuda_priority, profile, compatible_gpus)
            )
    if not candidates:
        nvidia_profiles = [
            effective_gpu_profile(profile, mode)
            for profile in profiles
            if profile.get("kind") == "nvidia"
        ]
        lowest_driver = min(
            (
                _parse_numeric_version(
                    profile.get("minimum_driver_version"),
                    label=f"{profile.get('id')} 최소 NVIDIA 드라이버",
                ),
                str(profile.get("minimum_driver_version")),
            )
            for profile in nvidia_profiles
        )
        lowest_compute = min(
            (
                _parse_numeric_version(
                    profile.get("minimum_compute_capability"),
                    label=f"{profile.get('id')} 최소 compute capability",
                    width=2,
                ),
                str(profile.get("minimum_compute_capability")),
            )
            for profile in nvidia_profiles
        )
        detected_driver = max(item[0] for item in detected)
        detected_compute = max(item[1] for item in detected)
        detected_driver_text = ".".join(str(value) for value in detected_driver[:2])
        detected_compute_text = ".".join(str(value) for value in detected_compute)
        if detected_compute < lowest_compute[0]:
            if mode == INSTALL_MODE_NVIDIA_COMPATIBILITY:
                message = (
                    "감지된 NVIDIA GPU 아키텍처가 호환 설치 기준보다 "
                    "오래되었습니다: "
                    f"detected_compute={detected_compute_text}, "
                    f"required_compute>={lowest_compute[1]}. "
                    "RTX 20(Turing, sm75) 이상 GPU가 필요합니다."
                )
            else:
                message = (
                    "감지된 NVIDIA GPU 아키텍처가 SageAttention 배포 "
                    "기준보다 오래되었습니다: "
                    f"detected_compute={detected_compute_text}, "
                    f"required_compute>={lowest_compute[1]}. "
                    "RTX 30(Ampere, sm86) 이상 GPU가 필요합니다."
                )
        else:
            message = (
                "NVIDIA 드라이버가 고정된 PyTorch CUDA 프로필의 최소 "
                "버전보다 오래되었습니다: "
                f"detected_driver={detected_driver_text}, "
                f"required_driver>={lowest_driver[1]}. "
                "CUDA Toolkit을 설치하지 말고 NVIDIA 그래픽 드라이버를 "
                "업데이트한 뒤 다시 시도하세요."
            )
        print(f"[COMFY_INSTALL][PROBE] GPU 프로필 선택 실패: {message}")
        raise SystemProbeError(message)

    selected = sorted(
        candidates,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )[0]
    profile = selected[2]
    compatible_names = ", ".join(
        str(gpu.get("name", "unknown")) for gpu in selected[3]
    )
    print(
        "[COMFY_INSTALL][PROBE] NVIDIA GPU 프로필 선택 완료: "
        f"profile={profile.get('id')}, "
        f"install_mode={mode}, "
        f"minimum_driver={profile.get('minimum_driver_version')}, "
        f"torch_cuda={profile.get('torch_cuda')}, "
        f"compatible_gpus={compatible_names}"
    )
    return profile


def probe_system(
    install_root: str | os.PathLike[str],
    manifest: InstallManifest,
    *,
    required_bytes: int | None = None,
    require_disk: bool = True,
    install_mode: str = INSTALL_MODE_STANDARD,
) -> dict[str, Any]:
    try:
        mode = normalize_install_mode(install_mode)
        if platform.system() != "Windows":
            raise SystemProbeError(
                f"현재 설치기는 Windows만 지원합니다: {platform.system()}"
            )
        root = Path(install_root).resolve()
        disk_anchor = root if root.exists() else root.parent
        while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
            disk_anchor = disk_anchor.parent
        disk = shutil.disk_usage(disk_anchor)
        required_bytes = (
            int(required_bytes)
            if required_bytes is not None
            else 110 * 1024**3
        )
        if required_bytes <= 0:
            raise SystemProbeError(
                f"필요 디스크 크기가 유효하지 않습니다: {required_bytes}"
            )
        nvidia = _probe_nvidia()
        profile = select_gpu_profile(
            manifest,
            nvidia,
            install_mode=mode,
        )
        uv_version = _require_uv_version(_tool_version(["uv", "--version"]))
        result = {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "install_root": str(root),
            "disk": {
                "anchor": str(disk_anchor),
                "free": disk.free,
                "total": disk.total,
                "required": required_bytes,
                "enough": disk.free >= required_bytes,
            },
            "nvidia": nvidia,
            "gpu_profile": profile["id"],
            "install_mode": mode,
            "uv": uv_version,
            "git": _tool_version(["git", "--version"]),
        }
        if not result["disk"]["enough"] and require_disk:
            raise SystemProbeError(
                "ComfyUI 설치 공간이 부족합니다: "
                f"free={disk.free / 1024**3:.2f} GiB, "
                f"required={required_bytes / 1024**3:.2f} GiB"
            )
        return result
    except SystemProbeError:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][PROBE] 시스템 검사 실패: {exc}")
        traceback.print_exc()
        raise SystemProbeError(f"시스템 검사 실패: {exc}") from exc
