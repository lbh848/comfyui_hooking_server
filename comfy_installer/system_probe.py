from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

from .manifest import InstallManifest


class SystemProbeError(RuntimeError):
    """운영체제, GPU, 도구 또는 디스크 검사 실패."""


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


def _probe_nvidia() -> dict[str, Any]:
    query = _run_probe(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
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

    summary = _run_probe(["nvidia-smi"])
    combined = f"{summary.stdout}\n{summary.stderr}"
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", combined)
    driver_cuda = match.group(1) if match else None
    if driver_cuda is None:
        print("[COMFY_INSTALL][PROBE] NVIDIA 드라이버 CUDA 상한을 찾지 못했습니다.")
    return {"available": True, "gpus": gpus, "driver_cuda": driver_cuda}


def select_gpu_profile(
    manifest: InstallManifest, nvidia: dict[str, Any]
) -> dict[str, Any]:
    profiles = manifest.python["gpu_profiles"]
    if not nvidia.get("available"):
        for profile in profiles:
            if profile.get("kind") == "cpu":
                return profile
        raise SystemProbeError("NVIDIA GPU가 없고 CPU 설치 프로필도 없습니다.")

    raw_cuda = nvidia.get("driver_cuda")
    if raw_cuda is None:
        raise SystemProbeError(
            "NVIDIA 드라이버의 지원 CUDA 버전을 확인하지 못했습니다. "
            "nvidia-smi가 정상 출력되는지 확인하세요."
        )
    try:
        driver_cuda = tuple(int(part) for part in str(raw_cuda).split(".")[:2])
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PROBE] CUDA 버전 파싱 실패: "
            f"value={raw_cuda!r}, error={exc}"
        )
        traceback.print_exc()
        raise SystemProbeError(f"CUDA 버전 형식이 잘못되었습니다: {raw_cuda}") from exc

    candidates = []
    for profile in profiles:
        if profile.get("kind") != "nvidia":
            continue
        minimum = tuple(
            int(part)
            for part in str(profile.get("minimum_driver_cuda", "999.0")).split(".")[:2]
        )
        if driver_cuda >= minimum:
            candidates.append((minimum, profile))
    if not candidates:
        supported = ", ".join(
            str(profile.get("minimum_driver_cuda"))
            for profile in profiles
            if profile.get("kind") == "nvidia"
        )
        raise SystemProbeError(
            "현재 NVIDIA 드라이버가 지원하는 CUDA 상한으로는 고정된 PyTorch "
            f"프로필을 설치할 수 없습니다: detected={raw_cuda}, required={supported}. "
            "NVIDIA 드라이버를 업데이트한 뒤 다시 시도하세요."
        )
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def probe_system(
    install_root: str | os.PathLike[str],
    manifest: InstallManifest,
    *,
    required_bytes: int | None = None,
    require_disk: bool = True,
) -> dict[str, Any]:
    try:
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
        profile = select_gpu_profile(manifest, nvidia)
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
            "uv": _tool_version(["uv", "--version"]),
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
