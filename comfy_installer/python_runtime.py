from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from threading import Event
from typing import Callable

from .operations import isolated_subprocess_env, run_command


class ManagedPythonError(RuntimeError):
    """ComfyUI 프로젝트 내부 Python 준비 또는 이동 복구 실패."""


LogCallback = Callable[[str], None]
MANAGED_PYTHON_DIRNAME = ".python-runtime"
MANAGED_PYTHON_MARKER = "soya-managed-python"


def managed_python_root(comfy_root: str | os.PathLike[str]) -> Path:
    return Path(comfy_root).resolve() / MANAGED_PYTHON_DIRNAME


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _python_candidates(root: Path) -> list[Path]:
    executable_name = "python.exe" if os.name == "nt" else "python"
    candidates: list[Path] = []
    if not root.is_dir():
        return candidates
    for child in sorted(root.iterdir(), key=lambda value: value.name.lower()):
        if not child.is_dir() or not child.name.startswith("cpython-"):
            continue
        candidate = (
            child / executable_name
            if os.name == "nt"
            else child / "bin" / executable_name
        )
        if candidate.is_file():
            candidates.append(candidate.resolve())
    return candidates


def _probe_python(
    python: Path,
    *,
    cwd: Path,
    cancel_event: Event,
    log: LogCallback | None,
) -> dict[str, str]:
    lines = run_command(
        [
            str(python),
            "-c",
            (
                "import json,platform,sys;"
                "print(json.dumps({"
                "'version':platform.python_version(),"
                "'executable':sys.executable,"
                "'base_executable':getattr(sys,'_base_executable',''),"
                "'prefix':sys.prefix,"
                "'base_prefix':sys.base_prefix"
                "},ensure_ascii=False))"
            ),
        ],
        cwd=cwd,
        cancel_event=cancel_event,
        log=log,
        timeout=60,
    )
    if not lines:
        raise ManagedPythonError(f"Python 검사 결과가 비어 있습니다: {python}")
    try:
        result = json.loads(lines[-1])
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] Python 검사 JSON 해석 실패: "
            f"python={python}, output={lines[-1]!r}, error={exc}"
        )
        traceback.print_exc()
        raise ManagedPythonError(
            f"Python 검사 결과를 해석할 수 없습니다: {python}"
        ) from exc
    if not isinstance(result, dict):
        raise ManagedPythonError(f"Python 검사 결과 형식이 잘못되었습니다: {python}")
    return {str(key): str(value) for key, value in result.items()}


def find_managed_python(
    *,
    comfy_root: Path,
    python_version: str,
    cancel_event: Event,
    log: LogCallback | None,
) -> Path | None:
    root = managed_python_root(comfy_root)
    for candidate in _python_candidates(root):
        try:
            result = _probe_python(
                candidate,
                cwd=comfy_root,
                cancel_event=cancel_event,
                log=log,
            )
        except Exception as exc:
            print(
                "[COMFY_INSTALL][PYTHON_RUNTIME] 내부 Python 후보 검사 실패: "
                f"python={candidate}, error={exc}"
            )
            traceback.print_exc()
            continue
        if result.get("version") != python_version:
            if log:
                log(
                    "[Python] 버전이 다른 프로젝트 내부 Python 제외: "
                    f"expected={python_version}, actual={result.get('version')}, "
                    f"path={candidate}"
                )
            continue
        executable = Path(result.get("executable", "")).resolve()
        if executable != candidate or not _is_within(executable, root):
            print(
                "[COMFY_INSTALL][PYTHON_RUNTIME] 내부 Python 경로 검증 실패: "
                f"candidate={candidate}, executable={executable}, root={root}"
            )
            continue
        return candidate
    return None


def ensure_managed_python(
    *,
    comfy_root: Path,
    python_version: str,
    cancel_event: Event,
    log: LogCallback | None,
) -> Path:
    """외부 Python 탐색 없이 comfy/.python-runtime에 CPython을 설치한다."""

    root = managed_python_root(comfy_root)
    try:
        existing = find_managed_python(
            comfy_root=comfy_root,
            python_version=python_version,
            cancel_event=cancel_event,
            log=log,
        )
        if existing is not None:
            if log:
                log(
                    "[Python] 기존 프로젝트 내부 Python 재사용: "
                    f"{existing} (Python {python_version})"
                )
            return existing

        root.mkdir(parents=True, exist_ok=True)
        reinstall = any(root.iterdir())
        command = [
            "uv",
            "python",
            "install",
            "--install-dir",
            str(root),
        ]
        if reinstall:
            command.append("--reinstall")
        command.append(python_version)
        if log:
            log(
                "[Python] 프로젝트 내부 전용 CPython 설치: "
                f"version={python_version}, root={root}"
            )
        run_command(
            command,
            cwd=comfy_root,
            cancel_event=cancel_event,
            log=log,
            timeout=1800,
            env=isolated_subprocess_env(
                {
                    "UV_PYTHON_INSTALL_DIR": str(root),
                    "UV_PYTHON_BIN_DIR": str(root / ".bin"),
                    "UV_PYTHON_INSTALL_BIN": "false",
                    "UV_PYTHON_INSTALL_REGISTRY": "false",
                    "UV_PYTHON_NO_REGISTRY": "1",
                    "UV_MANAGED_PYTHON": "1",
                }
            ),
        )
        installed = find_managed_python(
            comfy_root=comfy_root,
            python_version=python_version,
            cancel_event=cancel_event,
            log=log,
        )
        if installed is None:
            raise ManagedPythonError(
                "uv 설치 후 프로젝트 내부 Python을 찾지 못했습니다: "
                f"version={python_version}, root={root}"
            )
        return installed
    except ManagedPythonError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] 프로젝트 내부 Python 설치 실패: "
            f"version={python_version}, root={root}, error={exc}"
        )
        traceback.print_exc()
        raise ManagedPythonError(
            f"프로젝트 내부 Python {python_version} 설치 실패: {exc}"
        ) from exc


def read_pyvenv_config(config_path: Path) -> dict[str, str]:
    try:
        values: dict[str, str] = {}
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            if "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            values[key.strip()] = value.strip()
        return values
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] pyvenv.cfg 읽기 실패: "
            f"path={config_path}, error={exc}"
        )
        traceback.print_exc()
        raise ManagedPythonError(
            f"ComfyUI Python 설정을 읽을 수 없습니다: {config_path}"
        ) from exc


def _backup_pyvenv_config(config_path: Path, requirements_dir: Path) -> Path:
    requirements_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = requirements_dir / f"comfy_pyvenv_before_relocation_{timestamp}.cfg"
    try:
        shutil.copy2(config_path, backup)
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] pyvenv.cfg 백업 완료: "
            f"source={config_path}, backup={backup}"
        )
        return backup
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] pyvenv.cfg 백업 실패: "
            f"source={config_path}, backup={backup}, error={exc}"
        )
        traceback.print_exc()
        raise ManagedPythonError(
            f"ComfyUI Python 설정 백업 실패: {config_path}"
        ) from exc


def write_managed_venv_config(
    *,
    comfy_root: Path,
    managed_python: Path,
    requirements_dir: Path | None,
    backup_existing: bool,
) -> Path | None:
    """pyvenv.cfg에 프로젝트 내부 Python의 상대 표식을 기록한다."""

    config_path = comfy_root / ".venv" / "pyvenv.cfg"
    values = read_pyvenv_config(config_path)
    try:
        relative_python = managed_python.resolve().relative_to(comfy_root.resolve())
    except ValueError as exc:
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] 관리 Python이 ComfyUI 폴더 밖입니다: "
            f"python={managed_python}, comfy_root={comfy_root}"
        )
        raise ManagedPythonError(
            f"프로젝트 외부 Python은 사용할 수 없습니다: {managed_python}"
        ) from exc

    expected = {
        "home": str(managed_python.parent.resolve()),
        "relocatable": "true",
        MANAGED_PYTHON_MARKER: relative_python.as_posix(),
    }
    if all(values.get(key) == value for key, value in expected.items()):
        return None

    backup: Path | None = None
    if backup_existing:
        if requirements_dir is None:
            raise ManagedPythonError(
                "기존 pyvenv.cfg를 수정하려면 백업 폴더가 필요합니다."
            )
        backup = _backup_pyvenv_config(config_path, requirements_dir)

    lines = config_path.read_text(encoding="utf-8").splitlines()
    remaining = dict(expected)
    updated: list[str] = []
    for raw_line in lines:
        if "=" not in raw_line:
            updated.append(raw_line)
            continue
        key = raw_line.split("=", 1)[0].strip()
        if key in remaining:
            updated.append(f"{key} = {remaining.pop(key)}")
        else:
            updated.append(raw_line)
    updated.extend(f"{key} = {value}" for key, value in remaining.items())
    part_path = config_path.with_suffix(config_path.suffix + ".part")
    try:
        part_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        os.replace(part_path, config_path)
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] 프로젝트 내부 Python 연결 기록 완료: "
            f"config={config_path}, python={managed_python}"
        )
        return backup
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] pyvenv.cfg 원자적 쓰기 실패: "
            f"config={config_path}, part={part_path}, error={exc}"
        )
        traceback.print_exc()
        raise ManagedPythonError(
            f"ComfyUI Python 설정 저장 실패: {config_path}"
        ) from exc


def repair_relocated_managed_venv(
    *,
    comfy_root: Path,
    requirements_dir: Path,
) -> bool:
    """새 설치기가 만든 환경만 프로젝트 이동 후 안전하게 재결합한다."""

    config_path = comfy_root / ".venv" / "pyvenv.cfg"
    if not config_path.is_file():
        return False
    values = read_pyvenv_config(config_path)
    marker = values.get(MANAGED_PYTHON_MARKER, "")
    if not marker:
        return False
    if values.get("relocatable", "").lower() != "true":
        raise ManagedPythonError(
            "프로젝트 내부 Python 표식은 있지만 relocatable 환경이 아닙니다: "
            f"{config_path}"
        )

    managed_python = (comfy_root / Path(marker)).resolve()
    root = managed_python_root(comfy_root)
    if not _is_within(managed_python, root):
        raise ManagedPythonError(
            "pyvenv.cfg의 프로젝트 내부 Python 표식이 안전하지 않습니다: "
            f"marker={marker!r}, root={root}"
        )
    if not managed_python.is_file():
        raise ManagedPythonError(
            f"프로젝트 내부 Python 실행 파일이 없습니다: {managed_python}"
        )
    expected_home = str(managed_python.parent.resolve())
    if values.get("home") == expected_home:
        return False

    backup = write_managed_venv_config(
        comfy_root=comfy_root,
        managed_python=managed_python,
        requirements_dir=requirements_dir,
        backup_existing=True,
    )
    venv_python = (
        comfy_root / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else comfy_root / ".venv" / "bin" / "python"
    )
    try:
        result = subprocess.run(
            [
                str(venv_python),
                "-c",
                "import sys;print(sys.prefix);print(sys._base_executable)",
            ],
            cwd=str(comfy_root),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=isolated_subprocess_env(),
        )
        if result.returncode != 0:
            raise ManagedPythonError(
                "이동된 ComfyUI Python 검증 실패: "
                f"code={result.returncode}, output={(result.stdout + result.stderr).strip()}"
            )
        output = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(output) < 2:
            raise ManagedPythonError(
                f"이동된 ComfyUI Python 검증 출력이 비어 있습니다: {result.stdout!r}"
            )
        if Path(output[-2]).resolve() != (comfy_root / ".venv").resolve():
            raise ManagedPythonError(
                f"이동된 ComfyUI venv 경로 검증 실패: {output[-2]}"
            )
        if Path(output[-1]).resolve() != managed_python:
            raise ManagedPythonError(
                "이동된 ComfyUI 기반 Python 검증 실패: "
                f"expected={managed_python}, actual={output[-1]}"
            )
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] 프로젝트 이동 경로 자동 복구 완료: "
            f"venv={comfy_root / '.venv'}, python={managed_python}"
        )
        return True
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON_RUNTIME] 프로젝트 이동 경로 복구 검증 실패: "
            f"config={config_path}, python={managed_python}, error={exc}"
        )
        traceback.print_exc()
        if backup is not None and backup.is_file():
            try:
                shutil.copy2(backup, config_path)
                print(
                    "[COMFY_INSTALL][PYTHON_RUNTIME] 실패 후 pyvenv.cfg 복원 완료: "
                    f"backup={backup}, config={config_path}"
                )
            except Exception as restore_exc:
                print(
                    "[COMFY_INSTALL][PYTHON_RUNTIME] 실패 후 pyvenv.cfg 복원 실패: "
                    f"backup={backup}, config={config_path}, error={restore_exc}"
                )
                traceback.print_exc()
        if isinstance(exc, ManagedPythonError):
            raise
        raise ManagedPythonError(f"ComfyUI 이동 경로 복구 실패: {exc}") from exc
