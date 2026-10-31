from __future__ import annotations

import os
import traceback
import uuid
from pathlib import Path
from threading import Event
from typing import Callable

from .operations import CommandError, run_command
from .source_compatibility import (
    apply_comfy_system_stats_compatibility,
    managed_comfy_system_stats_update,
)


class SourceInstallError(RuntimeError):
    """ComfyUI 소스의 고정 커밋 설치 또는 검증 실패."""


LogCallback = Callable[[str], None]


def _git_value(path: Path, *arguments: str) -> str:
    lines = run_command(["git", *arguments], cwd=path)
    return lines[-1].strip() if lines else ""


def _normalized_git_url(value: str) -> str:
    return value.strip().rstrip("/").removesuffix(".git").casefold()


def _verify_existing(
    destination: Path,
    *,
    repository: str,
    ref: str,
    log: LogCallback | None,
) -> bool:
    if not destination.exists():
        return False
    if not destination.is_dir() or not (destination / ".git").is_dir():
        raise SourceInstallError(
            "관리되지 않는 기존 ComfyUI 경로가 있어 덮어쓰지 않습니다: "
            f"{destination}"
        )
    try:
        actual_ref = _git_value(destination, "rev-parse", "HEAD").lower()
        actual_origin = _git_value(
            destination, "remote", "get-url", "origin"
        )
    except CommandError as exc:
        raise SourceInstallError(
            f"기존 ComfyUI Git 상태를 확인하지 못했습니다: {destination}"
        ) from exc
    if actual_ref != ref.lower():
        raise SourceInstallError(
            "기존 ComfyUI 커밋이 설치 매니페스트와 다릅니다. 자동으로 "
            "덮어쓰지 않습니다: "
            f"expected={ref}, actual={actual_ref}, path={destination}"
        )
    if _normalized_git_url(actual_origin) != _normalized_git_url(repository):
        raise SourceInstallError(
            "기존 ComfyUI 원격 저장소가 설치 매니페스트와 다릅니다: "
            f"expected={repository}, actual={actual_origin}"
        )
    if log:
        log(f"[ComfyUI] 고정 소스 재사용: {actual_ref[:12]}")
    return True


def install_comfy_source(
    *,
    destination: str | os.PathLike[str],
    repository: str,
    ref: str,
    cancel_event: Event,
    log: LogCallback | None = None,
    requirements_dir: Path | None = None,
) -> Path:
    target = Path(destination).resolve()
    backup_root = (
        requirements_dir.resolve()
        if requirements_dir is not None
        else target / ".installer-state" / "backups" / "runtime"
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if _verify_existing(
            target, repository=repository, ref=ref, log=log
        ):
            apply_comfy_system_stats_compatibility(
                comfy_root=target,
                requirements_dir=backup_root,
                log=log,
            )
            return target

        staging = target.parent / (
            f".{target.name}.installing_{uuid.uuid4().hex[:10]}"
        )
        if staging.exists():
            raise SourceInstallError(
                f"ComfyUI 소스 스테이징 경로가 이미 존재합니다: {staging}"
            )
        staging.mkdir()
        try:
            run_command(
                ["git", "init"],
                cwd=staging,
                cancel_event=cancel_event,
                log=log,
            )
            run_command(
                ["git", "remote", "add", "origin", repository],
                cwd=staging,
                cancel_event=cancel_event,
                log=log,
            )
            run_command(
                ["git", "fetch", "--depth", "1", "origin", ref],
                cwd=staging,
                cancel_event=cancel_event,
                log=log,
                timeout=900,
            )
            run_command(
                ["git", "checkout", "--detach", "FETCH_HEAD"],
                cwd=staging,
                cancel_event=cancel_event,
                log=log,
            )
            actual_ref = _git_value(staging, "rev-parse", "HEAD").lower()
            if actual_ref != ref.lower():
                raise SourceInstallError(
                    "ComfyUI 고정 커밋 검증 실패: "
                    f"expected={ref}, actual={actual_ref}"
                )
            os.replace(staging, target)
            apply_comfy_system_stats_compatibility(
                comfy_root=target,
                requirements_dir=backup_root,
                log=log,
            )
            if log:
                log(
                    "[ComfyUI] 매니페스트 고정 소스 설치 완료: "
                    f"{actual_ref[:12]}"
                )
            return target
        except Exception:
            print(
                "[COMFY_INSTALL][SOURCE] 소스 설치 실패, 조사할 수 있도록 "
                f"스테이징을 보존합니다: {staging}"
            )
            raise
    except SourceInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][SOURCE] ComfyUI 소스 설치 실패: "
            f"target={target}, error={exc}"
        )
        traceback.print_exc()
        raise SourceInstallError(f"ComfyUI 소스 설치 실패: {exc}") from exc


def update_comfy_source(
    *,
    destination: str | os.PathLike[str],
    repository: str,
    ref: str,
    cancel_event: Event,
    log: LogCallback | None = None,
    requirements_dir: Path | None = None,
) -> Path:
    """사용자가 명시적으로 업데이트한 경우에만 관리 중인 Comfy 소스를 갱신한다."""

    target = Path(destination).resolve()
    backup_root = (
        requirements_dir.resolve()
        if requirements_dir is not None
        else target / ".installer-state" / "backups" / "runtime"
    )
    try:
        with managed_comfy_system_stats_update(
            comfy_root=target,
            requirements_dir=backup_root,
            log=log,
        ):
            if not target.is_dir() or not (target / ".git").is_dir():
                raise SourceInstallError(
                    f"업데이트할 관리형 ComfyUI Git 폴더가 없습니다: {target}"
                )
            actual_origin = _git_value(target, "remote", "get-url", "origin")
            if _normalized_git_url(actual_origin) != _normalized_git_url(repository):
                raise SourceInstallError(
                    "ComfyUI 원격 저장소가 설치 매니페스트와 달라 업데이트하지 "
                    f"않습니다: expected={repository}, actual={actual_origin}"
                )
            status = run_command(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=target,
            )
            if status:
                raise SourceInstallError(
                    "ComfyUI 소스에 로컬 변경이 있어 업데이트하지 않습니다: "
                    + ", ".join(status[:10])
                )
            actual_ref = _git_value(target, "rev-parse", "HEAD").lower()
            if actual_ref == ref.lower():
                if log:
                    log(f"[ComfyUI 업데이트] 이미 최신 고정점: {actual_ref[:12]}")
                return target
            if log:
                log(
                    "[ComfyUI 업데이트] 새 고정점 가져오기: "
                    f"{actual_ref[:12]} -> {ref[:12]}"
                )
            run_command(
                ["git", "fetch", "--depth", "1", "origin", ref],
                cwd=target,
                cancel_event=cancel_event,
                log=log,
                timeout=900,
            )
            run_command(
                ["git", "checkout", "--detach", "FETCH_HEAD"],
                cwd=target,
                cancel_event=cancel_event,
                log=log,
            )
            updated_ref = _git_value(target, "rev-parse", "HEAD").lower()
            if updated_ref != ref.lower():
                raise SourceInstallError(
                    "ComfyUI 업데이트 고정점 검증 실패: "
                    f"expected={ref}, actual={updated_ref}"
                )
            if log:
                log(f"[ComfyUI 업데이트] 완료: {updated_ref[:12]}")
        return target
    except SourceInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][SOURCE] ComfyUI 업데이트 실패: "
            f"target={target}, error={exc}"
        )
        traceback.print_exc()
        raise SourceInstallError(f"ComfyUI 업데이트 실패: {exc}") from exc
