from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from threading import Event

import pytest

import comfy_installer.dependency_installer as dependency_module
import comfy_installer.python_runtime as runtime_module
from comfy_installer.system_probe import SystemProbeError, _require_uv_version
from comfy_installer.dependency_installer import (
    DependencyInstallError,
    create_comfy_venv,
)
from comfy_installer.python_runtime import (
    MANAGED_PYTHON_MARKER,
    ensure_managed_python,
    repair_relocated_managed_venv,
)


# uv 가 만드는 배치는 OS 마다 다르다. Windows 만 적어 두면 그 외 OS 에서는
# 프로덕션의 인터프리터 탐색이 한 번도 시험되지 않는다.
MANAGED_TAG = "windows-x86_64" if os.name == "nt" else "macos-aarch64"


def _managed_python(install_root: Path) -> Path:
    base = install_root / f"cpython-3.12.11-{MANAGED_TAG}-none"
    return base / "python.exe" if os.name == "nt" else base / "bin" / "python"


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def test_manager_bootstrap_installs_missing_system_git_before_uv_sync() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "run_en.bat"
    ).read_text(encoding="utf-8")

    assert "call :ensure_git" in source
    assert "where git.exe >nul 2>&1" in source
    assert "winget install --id Git.Git --exact --source winget" in source
    assert "--accept-package-agreements" in source
    assert "--accept-source-agreements" in source
    assert "%LOCALAPPDATA%\\Programs\\Git\\cmd\\git.exe" in source
    assert "%ProgramFiles%\\Git\\cmd\\git.exe" in source
    assert source.index("call :ensure_git") < source.index('"%UV_EXE%" sync')


def test_managed_python_install_is_forced_into_comfy_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comfy_root = tmp_path / "comfy"
    comfy_root.mkdir()
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[:3] == ["uv", "python", "install"]:
            install_root = Path(command[command.index("--install-dir") + 1])
            python = _managed_python(install_root)
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            return ["installed"]
        python = Path(command[0]).resolve()
        return [
            json.dumps(
                {
                    "version": "3.12.11",
                    "executable": str(python),
                    "base_executable": str(python),
                    "prefix": str(python.parent),
                    "base_prefix": str(python.parent),
                }
            )
        ]

    monkeypatch.setattr(runtime_module, "run_command", fake_run)

    python = ensure_managed_python(
        comfy_root=comfy_root,
        python_version="3.12.11",
        cancel_event=Event(),
        log=None,
    )

    runtime_root = (comfy_root / ".python-runtime").resolve()
    assert python.is_relative_to(runtime_root)
    install = next(command for command in commands if command[:3] == ["uv", "python", "install"])
    assert install[install.index("--install-dir") + 1] == str(runtime_root)
    assert install[-1] == "3.12.11"


def test_uv_version_guard_rejects_versions_without_local_registry_controls() -> None:
    assert _require_uv_version("uv 0.11.8 (build)") == "uv 0.11.8 (build)"
    assert _require_uv_version("uv 1.0.0") == "uv 1.0.0"
    with pytest.raises(SystemProbeError, match="required>=0.11.8"):
        _require_uv_version("uv 0.6.9 (old)")


def test_create_venv_uses_explicit_managed_python_and_relocatable_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comfy_root = tmp_path / "comfy"
    comfy_root.mkdir()
    managed_python = _managed_python(comfy_root / ".python-runtime")
    managed_python.parent.mkdir(parents=True)
    managed_python.write_bytes(b"python")
    commands: list[list[str]] = []

    monkeypatch.setattr(
        dependency_module,
        "ensure_managed_python",
        lambda **_kwargs: managed_python.resolve(),
    )

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["uv", "venv"]:
            venv_root = Path(command[2])
            python = _venv_python(venv_root)
            python.parent.mkdir(parents=True)
            python.write_bytes(b"venv-python")
            (venv_root / "pyvenv.cfg").write_text(
                "home = placeholder\n"
                "version_info = 3.12.11\n"
                "relocatable = true\n",
                encoding="utf-8",
            )
            return ["created"]
        raise AssertionError(command)

    monkeypatch.setattr(dependency_module, "run_command", fake_run)

    python = create_comfy_venv(
        comfy_root=comfy_root,
        python_version="3.12.11",
        cancel_event=Event(),
        log=None,
        requirements_dir=tmp_path / "요구사항",
    )

    command = commands[0]
    assert command[:2] == ["uv", "venv"]
    assert command[command.index("--python") + 1] == str(managed_python.resolve())
    assert "--relocatable" in command
    assert python == _venv_python(comfy_root / ".venv")
    config = runtime_module.read_pyvenv_config(comfy_root / ".venv" / "pyvenv.cfg")
    assert config[MANAGED_PYTHON_MARKER] == managed_python.relative_to(comfy_root).as_posix()
    assert config["home"] == str(managed_python.parent.resolve())


def test_existing_external_venv_is_rejected_without_legacy_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comfy_root = tmp_path / "comfy"
    venv_python = _venv_python(comfy_root / ".venv")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"venv-python")
    (comfy_root / ".venv" / "pyvenv.cfg").write_text(
        "home = E:\\legacy\\python\n"
        "version_info = 3.12.11\n",
        encoding="utf-8",
    )
    managed_python = _managed_python(comfy_root / ".python-runtime")
    managed_python.parent.mkdir(parents=True)
    managed_python.write_bytes(b"python")
    monkeypatch.setattr(
        dependency_module,
        "ensure_managed_python",
        lambda **_kwargs: managed_python.resolve(),
    )

    with pytest.raises(DependencyInstallError, match="자동 이관하지 않습니다"):
        create_comfy_venv(
            comfy_root=comfy_root,
            python_version="3.12.11",
            cancel_event=Event(),
            log=None,
            requirements_dir=tmp_path / "요구사항",
        )


def test_repair_relocated_venv_uses_relative_marker_and_backs_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comfy_root = tmp_path / "moved" / "comfy"
    managed_python = _managed_python(comfy_root / ".python-runtime")
    managed_python.parent.mkdir(parents=True)
    managed_python.write_bytes(b"python")
    venv_python = _venv_python(comfy_root / ".venv")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"venv-python")
    marker = managed_python.relative_to(comfy_root).as_posix()
    config_path = comfy_root / ".venv" / "pyvenv.cfg"
    config_path.write_text(
        "home = D:\\old-location\\comfy\\.python-runtime\\cpython\n"
        "version_info = 3.12.11\n"
        "relocatable = true\n"
        f"{MANAGED_PYTHON_MARKER} = {marker}\n",
        encoding="utf-8",
    )

    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=f"{comfy_root / '.venv'}\n{managed_python.resolve()}\n",
        stderr="",
    )
    monkeypatch.setattr(runtime_module.subprocess, "run", lambda *_args, **_kwargs: completed)
    requirements = tmp_path / "요구사항"

    assert repair_relocated_managed_venv(
        comfy_root=comfy_root,
        requirements_dir=requirements,
    ) is True

    config = runtime_module.read_pyvenv_config(config_path)
    assert config["home"] == str(managed_python.parent.resolve())
    backups = list(requirements.glob("comfy_pyvenv_before_relocation_*.cfg"))
    assert len(backups) == 1
    assert "old-location" in backups[0].read_text(encoding="utf-8")
