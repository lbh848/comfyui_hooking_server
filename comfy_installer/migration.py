from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from typing import Callable


class ComfyMigrationError(RuntimeError):
    """기존 ComfyUI 사용자 LoRA/봇 캐시 복사 실패."""


LogCallback = Callable[[str], None]


def migrate_user_data(
    *,
    old_comfy_root: str | os.PathLike[str],
    new_comfy_root: str | os.PathLike[str],
    log: LogCallback | None = None,
) -> dict:
    old_root = Path(old_comfy_root).resolve()
    new_root = Path(new_comfy_root).resolve()
    try:
        if not old_root.is_dir():
            raise ComfyMigrationError(f"기존 ComfyUI 폴더가 없습니다: {old_root}")
        if old_root == new_root:
            raise ComfyMigrationError("기존 ComfyUI와 내장 ComfyUI 경로가 같습니다.")
        if not new_root.is_dir() or not (new_root / ".git").is_dir():
            raise ComfyMigrationError(
                "이사 대상인 내장 ComfyUI가 아직 설치되지 않았습니다. "
                "먼저 설치하기를 완료하세요."
            )
        mappings = (
            (
                old_root / "models" / "loras" / "SOYA_CHAR_LORA",
                new_root / "models" / "loras" / "SOYA_CHAR_LORA",
                "LoRA",
            ),
            (
                old_root / "input" / "soya_bot",
                new_root / "input" / "soya_bot",
                "봇 캐시",
            ),
        )
        copied: list[str] = []
        skipped: list[str] = []
        missing: list[str] = []
        failures: list[dict[str, str]] = []
        for source_root, destination_root, label in mappings:
            if not source_root.is_dir():
                message = f"{label} 원본 폴더 없음: {source_root}"
                print(f"[COMFY_INSTALL][MIGRATE] {message}")
                if log:
                    log(f"[이사] {message}")
                missing.append(str(source_root))
                continue
            for source in sorted(source_root.rglob("*")):
                if not source.is_file():
                    continue
                relative = source.relative_to(source_root)
                destination = destination_root / relative
                if destination.exists():
                    skipped.append(str(destination))
                    if log:
                        log(f"[이사] 기존 파일 유지: {destination}")
                    continue
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    copied.append(str(destination))
                    if log:
                        log(f"[이사] {label} 복사: {relative}")
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][MIGRATE] 파일 복사 실패: "
                        f"source={source}, destination={destination}, error={exc}"
                    )
                    traceback.print_exc()
                    failures.append(
                        {
                            "source": str(source),
                            "destination": str(destination),
                            "error": str(exc),
                        }
                    )
        result = {
            "old_comfy_root": str(old_root),
            "new_comfy_root": str(new_root),
            "copied": copied,
            "skipped": skipped,
            "missing": missing,
            "failures": failures,
        }
        if failures:
            raise ComfyMigrationError(
                f"일부 파일 이사 실패: 성공={len(copied)}, 실패={len(failures)}"
            )
        print(
            "[COMFY_INSTALL][MIGRATE] 이사 완료: "
            f"copied={len(copied)}, skipped={len(skipped)}, missing={len(missing)}"
        )
        return result
    except ComfyMigrationError:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][MIGRATE] 이사 실패: {exc}")
        traceback.print_exc()
        raise ComfyMigrationError(f"사용자 데이터 이사 실패: {exc}") from exc
