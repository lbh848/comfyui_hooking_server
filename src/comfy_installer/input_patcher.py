from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from typing import Callable


class ComfyInputPatchError(RuntimeError):
    """Comfy input 임시 폴더 리패치 실패."""


LogCallback = Callable[[str], None]


def _clear_folder(folder: Path) -> int:
    removed = 0
    if not folder.is_dir():
        return removed
    for item in folder.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
            removed += 1
        except Exception as exc:
            print(
                "[COMFY_INSTALL][REPATCH] 항목 삭제 실패: "
                f"path={item}, error={exc}"
            )
            traceback.print_exc()
            raise ComfyInputPatchError(f"리패치 항목 삭제 실패: {item}") from exc
    return removed


def patch_comfy_input(
    *,
    comfy_input_dir: str | os.PathLike[str],
    fallback_source: str | os.PathLike[str],
    log: LogCallback | None = None,
) -> dict:
    root = Path(comfy_input_dir).resolve()
    fallback = Path(fallback_source).resolve()
    try:
        if not root.is_dir():
            raise ComfyInputPatchError(f"Comfy input 폴더가 없습니다: {root}")
        folders = [
            root / "soya_char_ref",
            root / "soya_style_ref",
            root / "soya_lora",
            root / "soya_bot",
            root / "soya_char_ref" / "fallback",
            root / "soya_style_ref" / "fallback",
        ]
        created: list[str] = []
        cleared: list[str] = []
        removed_count = 0
        for folder in folders:
            if not folder.is_dir():
                folder.mkdir(parents=True, exist_ok=True)
                created.append(str(folder))
                if log:
                    log(f"[리패치] 폴더 생성: {folder}")
                continue
            if (
                folder.name in {"soya_char_ref", "soya_style_ref", "soya_lora"}
                or folder.name == "fallback"
            ):
                removed_count += _clear_folder(folder)
                cleared.append(str(folder))
                if log:
                    log(f"[리패치] 임시 폴더 비움: {folder}")
            elif log:
                log(f"[리패치] 캐시 폴더 유지: {folder}")

        copied: list[str] = []
        if not fallback.is_dir():
            print(
                "[COMFY_INSTALL][REPATCH] fallback 원본 폴더 없음: "
                f"{fallback}"
            )
        else:
            for source in fallback.iterdir():
                if not source.is_file():
                    continue
                for relative in (
                    Path("soya_char_ref") / "fallback" / source.name,
                    Path("soya_style_ref") / "fallback" / source.name,
                ):
                    destination = root / relative
                    shutil.copy2(source, destination)
                    copied.append(str(destination))
                    if log:
                        log(f"[리패치] fallback 복사: {destination}")
        return {
            "created": created,
            "cleared": cleared,
            "removed_count": removed_count,
            "copied": copied,
        }
    except ComfyInputPatchError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][REPATCH] Comfy input 리패치 실패: "
            f"root={root}, error={exc}"
        )
        traceback.print_exc()
        raise ComfyInputPatchError(f"Comfy input 리패치 실패: {exc}") from exc

