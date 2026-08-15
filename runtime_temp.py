"""Project-local runtime temporary directory management."""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path


RUNTIME_TEMP_RELATIVE_PATH = Path("runtime") / "temp"


def runtime_temp_root(project_root: str | Path) -> Path:
    """Return and create the validated project-local runtime temp directory."""

    root = Path(project_root).resolve()
    target = (root / RUNTIME_TEMP_RELATIVE_PATH).resolve()
    if target == root or root not in target.parents:
        print(
            "[RUNTIME_TEMP][ERROR] 프로젝트 밖의 임시 디렉터리 경로를 거부합니다: "
            f"project_root={root}, target={target}"
        )
        raise ValueError(f"안전하지 않은 프로젝트 임시 디렉터리 경로입니다: {target}")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(
            "[RUNTIME_TEMP][ERROR] 프로젝트 임시 디렉터리 생성 실패: "
            f"target={target}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    return target


def clear_runtime_temp(project_root: str | Path) -> Path:
    """Remove only the contents of the project-local runtime temp directory."""

    target = runtime_temp_root(project_root)
    removed = 0
    try:
        entries = list(target.iterdir())
        for entry in entries:
            if entry.is_symlink():
                entry.unlink()
            elif getattr(entry, "is_junction", lambda: False)():
                entry.rmdir()
            elif entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
    except Exception as exc:
        print(
            "[RUNTIME_TEMP][ERROR] 프로그램 시작 임시 디렉터리 정리 실패: "
            f"target={target}, entry={entry if 'entry' in locals() else None}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    print(
        "[RUNTIME_TEMP] 프로그램 시작 임시 디렉터리 정리 완료: "
        f"target={target}, removed={removed}"
    )
    return target
