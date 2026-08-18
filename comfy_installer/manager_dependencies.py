from __future__ import annotations

import re
import traceback
from pathlib import Path
from threading import Event
from typing import Callable

from .operations import run_command


class ManagerDependencyError(RuntimeError):
    """ComfyUI V4 Manager requirement or installation failure."""


LogCallback = Callable[[str], None]
_MANAGER_REQUIREMENT = re.compile(
    r"^comfyui[-_]manager\s*==\s*([^\s;]+)\s*$",
    re.IGNORECASE,
)


def expected_manager_version(comfy_root: str | Path) -> str:
    root = Path(comfy_root).resolve()
    requirements = root / "manager_requirements.txt"
    if not requirements.is_file():
        message = (
            "ComfyUI Manager 요구사항 파일이 없습니다: "
            f"{requirements}"
        )
        print(f"[COMFY_INSTALL][MANAGER] {message}")
        raise ManagerDependencyError(message)
    try:
        active_lines = [
            line.strip()
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except Exception as exc:
        print(
            "[COMFY_INSTALL][MANAGER] 요구사항 파일 읽기 실패: "
            f"path={requirements}, error={exc}"
        )
        traceback.print_exc()
        raise ManagerDependencyError(
            f"ComfyUI Manager 요구사항을 읽지 못했습니다: {requirements}"
        ) from exc

    matches = [
        match
        for line in active_lines
        if (match := _MANAGER_REQUIREMENT.fullmatch(line)) is not None
    ]
    if len(matches) != 1:
        message = (
            "ComfyUI Manager 고정 버전을 정확히 하나 찾지 못했습니다: "
            f"path={requirements}, active_lines={active_lines!r}"
        )
        print(f"[COMFY_INSTALL][MANAGER] {message}")
        raise ManagerDependencyError(message)
    return matches[0].group(1)


def installed_manager_versions(comfy_root: str | Path) -> list[str]:
    root = Path(comfy_root).resolve()
    site_packages = root / ".venv" / "Lib" / "site-packages"
    if not site_packages.is_dir():
        print(
            "[COMFY_INSTALL][MANAGER] Manager 패키지 조사 생략: "
            f"site-packages 폴더가 없습니다: {site_packages}"
        )
        return []

    versions: list[str] = []
    try:
        for metadata_path in site_packages.glob("*.dist-info/METADATA"):
            name = None
            version = None
            for line in metadata_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.startswith("Name:"):
                    name = line.partition(":")[2].strip()
                elif line.startswith("Version:"):
                    version = line.partition(":")[2].strip()
                if name is not None and version is not None:
                    break
            normalized = str(name or "").replace("_", "-").casefold()
            if normalized == "comfyui-manager" and version:
                versions.append(version)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][MANAGER] 설치 버전 조사 실패: "
            f"site_packages={site_packages}, error={exc}"
        )
        traceback.print_exc()
        raise ManagerDependencyError(
            f"설치된 ComfyUI Manager 버전을 조사하지 못했습니다: {site_packages}"
        ) from exc

    unique = sorted(set(versions))
    if not unique:
        print(
            "[COMFY_INSTALL][MANAGER] 신형 Manager 패키지가 없습니다: "
            f"site_packages={site_packages}"
        )
    elif len(unique) > 1:
        print(
            "[COMFY_INSTALL][MANAGER] 신형 Manager 패키지 버전이 중복됩니다: "
            f"versions={unique}, site_packages={site_packages}"
        )
    return unique


def install_manager_dependencies(
    *,
    comfy_root: Path,
    python: Path,
    cancel_event: Event,
    log: LogCallback | None,
) -> dict[str, str]:
    requirements = comfy_root.resolve() / "manager_requirements.txt"
    expected = expected_manager_version(comfy_root)
    try:
        if log:
            log(
                "[Python] ComfyUI 신형 Manager 설치: "
                f"comfyui-manager=={expected}"
            )
        run_command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "-r",
                str(requirements),
            ],
            cwd=comfy_root,
            cancel_event=cancel_event,
            log=log,
            timeout=1800,
        )
        version_lines = run_command(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata as m;"
                    "print(m.version('comfyui-manager'))"
                ),
            ],
            cwd=comfy_root,
            cancel_event=cancel_event,
            log=None,
            timeout=60,
        )
        if not version_lines:
            print(
                "[COMFY_INSTALL][MANAGER] 설치 후 버전 확인 결과가 "
                f"비어 있습니다: python={python}, expected={expected}"
            )
            raise ManagerDependencyError(
                "ComfyUI Manager 설치 후 버전을 확인하지 못했습니다."
            )
        actual = version_lines[-1].strip()
        if actual != expected:
            print(
                "[COMFY_INSTALL][MANAGER] 설치 버전 불일치: "
                f"expected={expected}, actual={actual}, python={python}"
            )
            raise ManagerDependencyError(
                "ComfyUI Manager 설치 버전이 ComfyUI 고정 요구사항과 "
                f"다릅니다: expected={expected}, actual={actual}"
            )
        return {
            "requirements": str(requirements),
            "expected_version": expected,
            "installed_version": actual,
        }
    except ManagerDependencyError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][MANAGER] 신형 Manager 설치 실패: "
            f"requirements={requirements}, python={python}, error={exc}"
        )
        traceback.print_exc()
        raise ManagerDependencyError(
            f"ComfyUI 신형 Manager 설치 실패: {exc}"
        ) from exc
