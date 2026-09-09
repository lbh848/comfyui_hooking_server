"""설치기 E2E 런처가 venv 인터프리터의 심볼릭 링크를 따라가던 문제 회귀 테스트.

무설치 설치 시험(macOS)에서 12단계 startup 이 이렇게 실패했다:

    ModuleNotFoundError: No module named 'sqlalchemy'

sqlalchemy 는 6단계에서 이미 설치돼 있었고(uv pip freeze 로 확인), 같은 명령을
셸에서 직접 실행하면 정상 기동했다. 원인은 실행 대상 인터프리터였다:

    ComfyProcess.__init__ 이 python.resolve() 로 심볼릭 링크를 따라가
    `.venv/bin/python` → `.python-runtime/.../bin/python3.12` (기본 인터프리터)를
    실행했다. 그러면 sys.prefix 가 venv 가 아니라 관리 런타임이 되어 venv 의
    site-packages 가 sys.path 에 없다 — 서드파티 import 가 **전부** 깨진다.
    sqlalchemy 는 main.py 가 가장 먼저 필요로 한 패키지였을 뿐이다.

**Windows 에서는 드러나지 않는다.** uv 가 python.exe 를 복사하므로 resolve() 해도
venv 안에 머문다. POSIX 에서만 심볼릭 링크다. 앱의 런처
comfy_runtime._resolve_python_path() 는 원래부터 링크를 따라가지 않는다.

수정 전 4회 연속 실패, 수정 후 2회 연속 성공(ComfyUI 0.31.0 기동 확인).
"""

import os
import sys
from pathlib import Path
from threading import Event

import pytest

from comfy_installer.e2e import ComfyProcess


def _venv_like(tmp_path: Path) -> tuple[Path, Path]:
    """`.venv/bin/python` 이 관리 런타임을 가리키는 uv POSIX 배치를 흉내낸다."""
    comfy_root = tmp_path / "comfy"
    managed = comfy_root / ".python-runtime" / "cpython-3.12" / "bin" / "python3.12"
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python = comfy_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    return comfy_root, venv_python


def _process(comfy_root: Path, python: Path) -> ComfyProcess:
    return ComfyProcess(
        comfy_root=comfy_root,
        python=python,
        cancel_event=Event(),
        log=None,
        port=61999,
        verify_manager=False,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX 심볼릭 링크 배치 재현")
def test_symlinked_venv_python_is_not_followed(tmp_path):
    """링크를 따라가면 venv 밖 인터프리터가 실행돼 site-packages 를 잃는다."""
    comfy_root, venv_python = _venv_like(tmp_path)
    managed = comfy_root / ".python-runtime" / "cpython-3.12" / "bin" / "python3.12"
    venv_python.symlink_to(managed)

    process = _process(comfy_root, venv_python)

    assert process.python == venv_python.parent.resolve() / venv_python.name
    assert ".venv" in process.python.parts
    assert ".python-runtime" not in process.python.parts


@pytest.mark.skipif(os.name == "nt", reason="POSIX 심볼릭 링크 배치 재현")
def test_launch_command_runs_the_venv_interpreter(tmp_path):
    comfy_root, venv_python = _venv_like(tmp_path)
    managed = comfy_root / ".python-runtime" / "cpython-3.12" / "bin" / "python3.12"
    venv_python.symlink_to(managed)

    command = _process(comfy_root, venv_python).launch_command()

    assert command[0] == str(venv_python.parent.resolve() / venv_python.name)
    assert str(managed) != command[0]
    assert command[2] == str(comfy_root.resolve() / "main.py")


def test_directory_part_is_still_normalized(tmp_path):
    """링크는 따라가지 않되 경로 자체는 정규화돼야 한다(`..` 등)."""
    comfy_root, venv_python = _venv_like(tmp_path)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    messy = venv_python.parent / ".." / "bin" / venv_python.name

    process = _process(comfy_root, messy)

    assert ".." not in process.python.parts
    assert process.python == venv_python.parent.resolve() / venv_python.name


def test_app_launcher_and_installer_agree():
    """앱 런처와 설치기 런처가 다른 인터프리터를 쓰면 한쪽만 동작하게 된다."""
    source = (Path(__file__).resolve().parents[1] / "comfy_runtime.py").read_text(
        encoding="utf-8"
    )
    # comfy_runtime 은 예전부터 링크를 따라가지 않는다. 그 계약을 함께 잠근다.
    assert '.venv" / "bin" / "python"' in source
    assert "python.resolve()" not in source


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX 전용 재현")
def test_regular_file_interpreter_is_unchanged(tmp_path):
    """Windows 처럼 복사본인 경우 동작이 달라지면 안 된다."""
    comfy_root, venv_python = _venv_like(tmp_path)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    process = _process(comfy_root, venv_python)

    assert process.python == venv_python.resolve()
