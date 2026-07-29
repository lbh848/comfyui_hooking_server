from __future__ import annotations

import os
import traceback
from pathlib import Path
from threading import Event
from typing import Callable

from .configurator import backup_current_config
from .operations import CommandError, run_command


class HookingServerUpdateError(RuntimeError):
    """후킹 서버 수동 업데이트 실패."""


LogCallback = Callable[[str], None]
HOOKING_REPOSITORY = "https://github.com/lbh848/comfyui_hooking_server"
HOOKING_LOCAL_BRANCH = "main"
HOOKING_REMOTE_BRANCH = "dev"


def _git_value(root: Path, *arguments: str) -> str:
    lines = run_command(["git", *arguments], cwd=root)
    return lines[-1].strip() if lines else ""


def _normalize_repository(value: str) -> str:
    return value.strip().rstrip("/").removesuffix(".git").casefold()


def update_hooking_server_main(
    *,
    project_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    backup_dir: str | os.PathLike[str],
    cancel_event: Event,
    log: LogCallback | None = None,
    config_backup: dict | None = None,
) -> dict:
    """버튼을 누른 경우에만 원격 추적 브랜치를 fast-forward로 가져온다."""

    root = Path(project_root).resolve()
    try:
        if not (root / ".git").is_dir():
            raise HookingServerUpdateError(
                f"후킹 서버가 Git 설치가 아니어서 업데이트할 수 없습니다: {root}"
            )
        branch = _git_value(root, "branch", "--show-current")
        if branch != HOOKING_LOCAL_BRANCH:
            raise HookingServerUpdateError(
                f"배포 업데이터는 {HOOKING_LOCAL_BRANCH} 브랜치에서만 동작합니다: "
                f"current={branch or '(detached)'}"
            )
        origin = _git_value(root, "remote", "get-url", "origin")
        if _normalize_repository(origin) != _normalize_repository(
            HOOKING_REPOSITORY
        ):
            raise HookingServerUpdateError(
                "후킹 서버 origin이 공식 배포 저장소와 다릅니다: "
                f"actual={origin}"
            )
        changes = run_command(["git", "status", "--porcelain"], cwd=root)
        if changes:
            raise HookingServerUpdateError(
                "후킹 서버 소스에 로컬 변경이 있어 업데이트하지 않습니다: "
                + ", ".join(changes[:12])
            )
        before = _git_value(root, "rev-parse", "HEAD").lower()
        if config_backup is None:
            config_backup = backup_current_config(
                config_path=config_path,
                backup_dir=backup_dir,
                reason="hooking_update",
            )
        if log:
            log(
                "[후킹 서버 업데이트] 사용자가 요청하여 "
                f"origin/{HOOKING_REMOTE_BRANCH} 업데이트 시작: "
                f"현재={before[:12]}"
            )
        run_command(
            ["git", "pull", "--ff-only", "origin", HOOKING_REMOTE_BRANCH],
            cwd=root,
            cancel_event=cancel_event,
            log=log,
            timeout=900,
        )
        after = _git_value(root, "rev-parse", "HEAD").lower()
        if log:
            if before == after:
                log(f"[후킹 서버 업데이트] 이미 최신: {after[:12]}")
            else:
                log(
                    f"[후킹 서버 업데이트] {HOOKING_REMOTE_BRANCH} 적용 완료: "
                    f"{before[:12]} -> {after[:12]}"
                )
        return {
            "local_branch": HOOKING_LOCAL_BRANCH,
            "branch": HOOKING_REMOTE_BRANCH,
            "before": before,
            "after": after,
            "changed": before != after,
            "config_backup": config_backup,
            "restart_required": before != after,
        }
    except HookingServerUpdateError:
        raise
    except CommandError as exc:
        print(f"[COMFY_INSTALL][UPDATE] Git 명령 실패: {exc}")
        traceback.print_exc()
        raise HookingServerUpdateError(f"후킹 서버 업데이트 실패: {exc}") from exc
    except Exception as exc:
        print(f"[COMFY_INSTALL][UPDATE] 후킹 서버 업데이트 실패: {exc}")
        traceback.print_exc()
        raise HookingServerUpdateError(f"후킹 서버 업데이트 실패: {exc}") from exc
