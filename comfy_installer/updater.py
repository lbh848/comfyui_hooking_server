from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from threading import Event
from typing import Callable

from .configurator import backup_current_config
from .operations import CommandError, run_command


class HookingServerUpdateError(RuntimeError):
    """후킹 서버 main 브랜치 수동 업데이트 실패."""


LogCallback = Callable[[str], None]
HOOKING_REPOSITORY = "https://github.com/lbh848/comfyui_hooking_server"
HOOKING_BRANCH = "main"
_QUARANTINE_ROOT_NAME = "comfy-installer-quarantine"
_TRACKED_BACKUP_ROOT_NAME = "update_backup"


@dataclass(frozen=True)
class _UntrackedQuarantine:
    path: Path | None
    entries: tuple[Path, ...]


@dataclass(frozen=True)
class _TrackedChangesBackup:
    path: Path | None
    entries: tuple[Path, ...]
    patch_path: Path | None


def _emit(message: str, log: LogCallback | None) -> None:
    print(f"[COMFY_INSTALL][UPDATE] {message}")
    if log:
        log(f"[후킹 서버 업데이트] {message}")


def _git_creationflags() -> int:
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def _list_untracked_files(root: Path) -> tuple[Path, ...]:
    command = ["git", "ls-files", "--others", "--exclude-standard", "-z", "--"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_git_creationflags(),
        )
        decoded = completed.stdout.decode("utf-8")
    except Exception as exc:
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        print(
            "[COMFY_INSTALL][UPDATE] 미추적 파일 목록 확인 실패: "
            f"root={root}, stderr={stderr!r}, error={exc}"
        )
        traceback.print_exc()
        raise HookingServerUpdateError(
            f"후킹 서버 미추적 파일을 확인하지 못했습니다: {root}"
        ) from exc

    entries: list[Path] = []
    seen: set[Path] = set()
    skipped_update_backups = 0
    for raw_path in decoded.split("\0"):
        if not raw_path:
            continue
        posix_path = PurePosixPath(raw_path)
        if (
            posix_path.is_absolute()
            or not posix_path.parts
            or any(part in {"", ".", ".."} for part in posix_path.parts)
            or posix_path.parts[0].casefold() == ".git"
        ):
            print(
                "[COMFY_INSTALL][UPDATE] 안전하지 않은 미추적 파일 경로 거부: "
                f"root={root}, path={raw_path!r}"
            )
            raise HookingServerUpdateError(
                f"안전하지 않은 미추적 파일 경로를 격리하지 않습니다: {raw_path!r}"
            )
        if posix_path.parts[0].casefold() == _TRACKED_BACKUP_ROOT_NAME.casefold():
            skipped_update_backups += 1
            continue
        relative = Path(*posix_path.parts)
        if relative not in seen:
            entries.append(relative)
            seen.add(relative)
    if skipped_update_backups:
        print(
            "[COMFY_INSTALL][UPDATE] 영구 로컬 수정 백업은 미추적 파일 "
            "임시 격리에서 제외: "
            f"count={skipped_update_backups}, root={root / _TRACKED_BACKUP_ROOT_NAME}"
        )
    return tuple(entries)


def _assert_no_symlink_parent(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            print(
                "[COMFY_INSTALL][UPDATE] 심볼릭 링크 하위 파일은 격리하지 "
                f"않습니다: root={root}, path={relative}, link={current}"
            )
            raise HookingServerUpdateError(
                f"심볼릭 링크 하위의 미추적 파일을 안전하게 격리할 수 없습니다: {relative}"
            )


def _write_quarantine_record(path: Path, payload: dict) -> None:
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][UPDATE] 미추적 파일 격리 기록 저장 실패: "
            f"path={path}, error={exc}"
        )
        traceback.print_exc()
        raise HookingServerUpdateError(
            f"미추적 파일 격리 기록을 저장하지 못했습니다: {path}"
        ) from exc


def _remove_empty_source_parents(root: Path, entries: tuple[Path, ...]) -> None:
    parents: set[Path] = set()
    for relative in entries:
        current = (root / relative).parent
        while current != root and root in current.parents:
            parents.add(current)
            current = current.parent
    for parent in sorted(parents, key=lambda value: len(value.parts), reverse=True):
        try:
            parent.rmdir()
        except OSError:
            continue


def _prepare_untracked_quarantine(
    root: Path,
    *,
    log: LogCallback | None,
) -> _UntrackedQuarantine:
    entries = _list_untracked_files(root)
    if not entries:
        _emit("격리할 미추적 파일 없음", log)
        return _UntrackedQuarantine(path=None, entries=())

    git_root = (root / ".git").resolve()
    quarantine_base = git_root / _QUARANTINE_ROOT_NAME
    quarantine_path = quarantine_base / (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    )
    files_root = quarantine_path / "files"
    moved: list[Path] = []
    try:
        files_root.mkdir(parents=True, exist_ok=False)
        _write_quarantine_record(
            quarantine_path / "plan.json",
            {
                "schema_version": 1,
                "created_at": datetime.datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "project_root": str(root),
                "entries": [entry.as_posix() for entry in entries],
            },
        )
        _emit(
            f"미추적 파일 임시 격리 시작: count={len(entries)}, "
            f"path={quarantine_path}",
            log,
        )
        for relative in entries:
            _assert_no_symlink_parent(root, relative)
            source = root / relative
            if not os.path.lexists(source):
                print(
                    "[COMFY_INSTALL][UPDATE] 격리 직전에 미추적 파일이 "
                    f"사라졌습니다: {source}"
                )
                raise HookingServerUpdateError(
                    f"격리할 미추적 파일이 사라져 업데이트를 중단합니다: {relative}"
                )
            if source.is_dir() and not source.is_symlink():
                print(
                    "[COMFY_INSTALL][UPDATE] 디렉터리 전체 이동은 수행하지 "
                    f"않습니다: {source}"
                )
                raise HookingServerUpdateError(
                    f"예상하지 못한 미추적 디렉터리 항목입니다: {relative}"
                )
            destination = files_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append(relative)
            _emit(f"미추적 파일 격리: {relative.as_posix()}", log)
        _remove_empty_source_parents(root, entries)
        return _UntrackedQuarantine(path=quarantine_path, entries=tuple(moved))
    except Exception as operation_exc:
        print(
            "[COMFY_INSTALL][UPDATE] 미추적 파일 격리 실패: "
            f"root={root}, quarantine={quarantine_path}, error={operation_exc}"
        )
        traceback.print_exc()
        partial = _UntrackedQuarantine(
            path=quarantine_path,
            entries=tuple(moved),
        )
        try:
            _restore_untracked_quarantine(root, partial, log=log)
        except Exception as restore_exc:
            print(
                "[COMFY_INSTALL][UPDATE] 격리 실패 후 원상 복구도 실패: "
                f"quarantine={quarantine_path}, error={restore_exc}"
            )
            traceback.print_exc()
            raise HookingServerUpdateError(
                "미추적 파일 격리와 원상 복구가 모두 실패했습니다. "
                f"보존 위치: {quarantine_path}"
            ) from operation_exc
        if isinstance(operation_exc, HookingServerUpdateError):
            raise operation_exc
        raise HookingServerUpdateError(
            f"미추적 파일 격리에 실패했습니다: {operation_exc}"
        ) from operation_exc


def _safe_restore_parent(root: Path, relative: Path) -> tuple[bool, str | None]:
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink():
                return False, f"복원 상위 경로가 심볼릭 링크입니다: {current}"
            if not current.is_dir():
                return False, f"복원 상위 경로가 디렉터리가 아닙니다: {current}"
            continue
        try:
            current.mkdir()
        except Exception as exc:
            print(
                "[COMFY_INSTALL][UPDATE] 미추적 파일 복원 폴더 생성 실패: "
                f"path={current}, error={exc}"
            )
            traceback.print_exc()
            return False, f"복원 폴더를 만들지 못했습니다: {current}: {exc}"
    return True, None


def _cleanup_quarantine(quarantine_path: Path) -> None:
    try:
        plan = quarantine_path / "plan.json"
        if plan.is_file():
            plan.unlink()
        files_root = quarantine_path / "files"
        if files_root.is_dir():
            directories = [
                path for path in files_root.rglob("*") if path.is_dir()
            ]
            for directory in sorted(
                directories,
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                directory.rmdir()
            files_root.rmdir()
        quarantine_path.rmdir()
        quarantine_base = quarantine_path.parent
        try:
            quarantine_base.rmdir()
        except OSError:
            pass
    except Exception as exc:
        print(
            "[COMFY_INSTALL][UPDATE] 복원 완료 후 빈 격리 폴더 정리 실패: "
            f"path={quarantine_path}, error={exc}"
        )
        traceback.print_exc()


def _restore_untracked_quarantine(
    root: Path,
    quarantine: _UntrackedQuarantine,
    *,
    log: LogCallback | None,
) -> None:
    if quarantine.path is None:
        return
    files_root = quarantine.path / "files"
    conflicts: list[dict[str, str]] = []
    for relative in quarantine.entries:
        source = files_root / relative
        destination = root / relative
        if not os.path.lexists(source):
            message = f"격리본이 없습니다: {source}"
            print(f"[COMFY_INSTALL][UPDATE] 미추적 파일 복원 실패: {message}")
            conflicts.append({"path": relative.as_posix(), "reason": message})
            continue
        if os.path.lexists(destination):
            message = f"업데이트 후 같은 경로가 이미 존재합니다: {destination}"
            print(f"[COMFY_INSTALL][UPDATE] 미추적 파일 복원 충돌: {message}")
            conflicts.append({"path": relative.as_posix(), "reason": message})
            continue
        parent_ok, parent_error = _safe_restore_parent(root, relative)
        if not parent_ok:
            assert parent_error is not None
            print(
                "[COMFY_INSTALL][UPDATE] 미추적 파일 복원 충돌: "
                f"path={relative}, reason={parent_error}"
            )
            conflicts.append(
                {"path": relative.as_posix(), "reason": parent_error}
            )
            continue
        try:
            os.replace(source, destination)
            _emit(f"미추적 파일 복원: {relative.as_posix()}", log)
        except Exception as exc:
            print(
                "[COMFY_INSTALL][UPDATE] 미추적 파일 복원 이동 실패: "
                f"source={source}, destination={destination}, error={exc}"
            )
            traceback.print_exc()
            conflicts.append(
                {"path": relative.as_posix(), "reason": str(exc)}
            )

    if conflicts:
        conflict_record = quarantine.path / "restore-conflict.json"
        _write_quarantine_record(
            conflict_record,
            {
                "schema_version": 1,
                "recorded_at": datetime.datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "project_root": str(root),
                "quarantine_path": str(quarantine.path),
                "conflicts": conflicts,
            },
        )
        _emit(
            "미추적 파일 복원 충돌로 격리본 보존: "
            f"count={len(conflicts)}, path={quarantine.path}",
            log,
        )
        raise HookingServerUpdateError(
            "업데이트 후 사용자 파일을 원위치에 복원하지 못했습니다. "
            f"덮어쓰지 않고 보존했습니다: {quarantine.path}"
        )

    _cleanup_quarantine(quarantine.path)
    _emit(
        f"미추적 파일 원상 복원 완료: count={len(quarantine.entries)}",
        log,
    )


def _run_git_capture(root: Path, *arguments: str) -> bytes:
    command = ["git", *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_git_creationflags(),
        )
        return completed.stdout
    except Exception as exc:
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        print(
            "[COMFY_INSTALL][UPDATE] Git 변경 내용 캡처 실패: "
            f"root={root}, command={command}, stderr={stderr!r}, error={exc}"
        )
        traceback.print_exc()
        raise HookingServerUpdateError(
            f"로컬 수정 백업용 Git 정보를 읽지 못했습니다: {root}"
        ) from exc


def _tracked_change_paths(root: Path) -> tuple[Path, ...]:
    raw_output = _run_git_capture(root, "diff", "--name-only", "-z", "HEAD", "--")
    try:
        decoded = raw_output.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(
            "[COMFY_INSTALL][UPDATE] 추적 변경 파일 경로 UTF-8 해석 실패: "
            f"root={root}, error={exc}"
        )
        traceback.print_exc()
        raise HookingServerUpdateError(
            "로컬 수정 파일 경로를 UTF-8로 해석하지 못해 업데이트를 중단합니다."
        ) from exc

    entries: list[Path] = []
    seen: set[Path] = set()
    for raw_path in decoded.split("\0"):
        if not raw_path:
            continue
        posix_path = PurePosixPath(raw_path)
        if (
            posix_path.is_absolute()
            or not posix_path.parts
            or any(part in {"", ".", ".."} for part in posix_path.parts)
            or posix_path.parts[0].casefold() == ".git"
        ):
            print(
                "[COMFY_INSTALL][UPDATE] 안전하지 않은 추적 변경 경로 거부: "
                f"root={root}, path={raw_path!r}"
            )
            raise HookingServerUpdateError(
                f"안전하지 않은 로컬 수정 경로는 백업하지 않습니다: {raw_path!r}"
            )
        if posix_path.parts[0].casefold() == _TRACKED_BACKUP_ROOT_NAME.casefold():
            print(
                "[COMFY_INSTALL][UPDATE] 영구 백업 폴더가 Git 추적 변경에 "
                f"포함되어 업데이트 중단: path={raw_path!r}"
            )
            raise HookingServerUpdateError(
                f"{_TRACKED_BACKUP_ROOT_NAME}/ 폴더가 Git 추적 대상입니다. "
                "백업 재귀를 막기 위해 업데이트를 중단합니다."
            )
        relative = Path(*posix_path.parts)
        if relative not in seen:
            entries.append(relative)
            seen.add(relative)
    return tuple(entries)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_tracked_backup_path(root: Path) -> Path:
    backup_root = root / _TRACKED_BACKUP_ROOT_NAME
    if backup_root.is_symlink():
        print(
            "[COMFY_INSTALL][UPDATE] 영구 백업 루트가 심볼릭 링크여서 거부: "
            f"path={backup_root}"
        )
        raise HookingServerUpdateError(
            f"로컬 수정 백업 폴더가 심볼릭 링크입니다: {backup_root}"
        )
    try:
        backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
        candidate = backup_root / stamp
        if candidate.exists():
            candidate = backup_root / f"{stamp}_{uuid.uuid4().hex[:8]}"
        candidate.mkdir(parents=False, exist_ok=False)
        return candidate
    except HookingServerUpdateError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][UPDATE] 영구 로컬 수정 백업 폴더 생성 실패: "
            f"root={backup_root}, error={exc}"
        )
        traceback.print_exc()
        raise HookingServerUpdateError(
            f"로컬 수정 백업 폴더를 만들지 못했습니다: {backup_root}"
        ) from exc


def _backup_and_restore_tracked_changes(
    root: Path,
    *,
    changes: list[str],
    head: str,
    log: LogCallback | None,
) -> _TrackedChangesBackup:
    if not changes:
        _emit("영구 백업할 Git 추적 파일의 로컬 수정 없음", log)
        return _TrackedChangesBackup(path=None, entries=(), patch_path=None)

    backup_path: Path | None = None
    entries: tuple[Path, ...] = ()
    try:
        entries = _tracked_change_paths(root)
        if not entries:
            print(
                "[COMFY_INSTALL][UPDATE] Git status에는 추적 변경이 있지만 "
                f"백업할 경로를 찾지 못했습니다: changes={changes!r}"
            )
            raise HookingServerUpdateError(
                "Git 추적 변경 파일 목록을 확인하지 못해 업데이트를 중단합니다."
            )

        backup_path = _create_tracked_backup_path(root)
        files_root = backup_path / "files"
        files_root.mkdir()
        _emit(
            "Git 추적 파일의 로컬 수정 영구 백업 시작: "
            f"count={len(entries)}, path={backup_path}",
            log,
        )

        file_records: list[dict[str, object]] = []
        for relative in entries:
            _assert_no_symlink_parent(root, relative)
            source = root / relative
            destination = files_root / relative
            if not os.path.lexists(source):
                file_records.append(
                    {
                        "path": relative.as_posix(),
                        "state": "deleted",
                        "backup_path": None,
                    }
                )
                _emit(f"삭제된 추적 파일 상태 기록: {relative.as_posix()}", log)
                continue
            if source.is_symlink():
                destination.parent.mkdir(parents=True, exist_ok=True)
                link_target = os.readlink(source)
                destination.symlink_to(link_target, target_is_directory=source.is_dir())
                if not destination.is_symlink() or os.readlink(destination) != link_target:
                    raise HookingServerUpdateError(
                        f"심볼릭 링크 백업 검증에 실패했습니다: {relative}"
                    )
                file_records.append(
                    {
                        "path": relative.as_posix(),
                        "state": "symlink",
                        "backup_path": (Path("files") / relative).as_posix(),
                        "link_target": link_target,
                    }
                )
                _emit(f"수정된 추적 심볼릭 링크 백업: {relative.as_posix()}", log)
                continue
            if source.is_dir():
                print(
                    "[COMFY_INSTALL][UPDATE] 디렉터리 형태의 추적 변경은 "
                    f"자동 백업하지 않음: path={source}"
                )
                raise HookingServerUpdateError(
                    "디렉터리 또는 서브모듈 형태의 Git 추적 변경은 안전하게 "
                    f"자동 백업할 수 없습니다: {relative}"
                )
            if not source.is_file():
                print(
                    "[COMFY_INSTALL][UPDATE] 지원하지 않는 추적 파일 형식: "
                    f"path={source}"
                )
                raise HookingServerUpdateError(
                    f"지원하지 않는 로컬 수정 파일 형식입니다: {relative}"
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_hash = _file_sha256(source)
            backup_hash = _file_sha256(destination)
            if source_hash != backup_hash:
                print(
                    "[COMFY_INSTALL][UPDATE] 추적 파일 백업 해시 불일치: "
                    f"source={source}, backup={destination}, "
                    f"source_sha256={source_hash}, backup_sha256={backup_hash}"
                )
                raise HookingServerUpdateError(
                    f"로컬 수정 파일 백업 검증에 실패했습니다: {relative}"
                )
            file_records.append(
                {
                    "path": relative.as_posix(),
                    "state": "copied",
                    "backup_path": (Path("files") / relative).as_posix(),
                    "size": destination.stat().st_size,
                    "sha256": backup_hash,
                }
            )
            _emit(f"수정된 추적 파일 백업: {relative.as_posix()}", log)

        patch_bytes = _run_git_capture(root, "diff", "--binary", "--full-index", "HEAD", "--")
        patch_text = patch_bytes.decode("utf-8")
        patch_path = backup_path / "local_changes.patch"
        with patch_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(patch_text)

        manifest_path = backup_path / "manifest.json"
        manifest = {
            "schema_version": 1,
            "created_at": datetime.datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "project_root": str(root),
            "head": head,
            "files_root": "files",
            "patch_file": patch_path.name,
            "git_status": changes,
            "files": file_records,
            "automatic_restore": False,
            "restore_note": (
                "업데이트 후 필요한 파일만 files/에서 수동으로 복구하세요. "
                "전체 덮어쓰기는 새 버전 변경을 되돌릴 수 있습니다."
            ),
        }
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

        run_command(
            [
                "git",
                "restore",
                "--source=HEAD",
                "--staged",
                "--worktree",
                "--",
                ".",
            ],
            cwd=root,
        )
        remaining = run_command(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
        )
        if remaining:
            print(
                "[COMFY_INSTALL][UPDATE] 백업 후 Git 추적 파일 원본 복구 "
                f"검증 실패: remaining={remaining!r}, backup={backup_path}"
            )
            raise HookingServerUpdateError(
                "로컬 수정은 백업했지만 Git 원본 복구가 완료되지 않았습니다. "
                f"백업 위치: {backup_path}"
            )
        _emit(
            "Git 추적 파일 원본 복구 완료; 수정본은 자동 복구하지 않음: "
            f"count={len(entries)}, path={backup_path}",
            log,
        )
        return _TrackedChangesBackup(
            path=backup_path,
            entries=entries,
            patch_path=patch_path,
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][UPDATE] Git 추적 파일 로컬 수정 백업/복구 실패: "
            f"root={root}, backup={backup_path}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, HookingServerUpdateError):
            raise
        location = f" 부분 백업 위치: {backup_path}" if backup_path else ""
        raise HookingServerUpdateError(
            f"Git 추적 파일의 로컬 수정 백업에 실패했습니다.{location}"
        ) from exc


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
    """버튼을 누른 경우에만 origin/main을 fast-forward 방식으로 가져온다."""

    root = Path(project_root).resolve()
    try:
        if not (root / ".git").is_dir():
            raise HookingServerUpdateError(
                f"후킹 서버가 Git 설치가 아니어서 업데이트할 수 없습니다: {root}"
            )
        branch = _git_value(root, "branch", "--show-current")
        if branch != HOOKING_BRANCH:
            raise HookingServerUpdateError(
                "배포 업데이터는 main 브랜치에서만 동작합니다: "
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
        changes = run_command(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
        )
        before = _git_value(root, "rev-parse", "HEAD").lower()
        if config_backup is None:
            config_backup = backup_current_config(
                config_path=config_path,
                backup_dir=backup_dir,
                reason="hooking_update",
            )
        tracked_backup = _backup_and_restore_tracked_changes(
            root,
            changes=changes,
            head=before,
            log=log,
        )
        if log:
            log(
                "[후킹 서버 업데이트] 사용자가 요청하여 origin/main 업데이트 시작: "
                f"현재={before[:12]}"
            )
        quarantine = _prepare_untracked_quarantine(root, log=log)
        try:
            run_command(
                ["git", "pull", "--ff-only", "origin", HOOKING_BRANCH],
                cwd=root,
                cancel_event=cancel_event,
                log=log,
                timeout=900,
            )
        except Exception as update_exc:
            try:
                _restore_untracked_quarantine(root, quarantine, log=log)
            except Exception as restore_exc:
                print(
                    "[COMFY_INSTALL][UPDATE] Git 업데이트 실패 후 미추적 "
                    "파일 복원도 실패: "
                    f"update_error={update_exc}, restore_error={restore_exc}, "
                    f"quarantine={quarantine.path}"
                )
                traceback.print_exc()
                raise HookingServerUpdateError(
                    "후킹 서버 업데이트와 사용자 파일 복원이 모두 "
                    "실패했습니다. "
                    f"격리본 위치: {quarantine.path}; "
                    f"추적 파일 백업 위치: {tracked_backup.path or '(없음)'}"
                ) from update_exc
            if tracked_backup.path is not None:
                _emit(
                    "Git 업데이트는 실패했지만 추적 파일 수정본은 영구 "
                    f"백업에 보존됨: {tracked_backup.path}",
                    log,
                )
                raise HookingServerUpdateError(
                    "후킹 서버 업데이트에 실패했습니다. 추적 파일의 로컬 "
                    f"수정본은 보존했습니다: {tracked_backup.path}; "
                    f"원인: {update_exc}"
                ) from update_exc
            raise
        _restore_untracked_quarantine(root, quarantine, log=log)
        after = _git_value(root, "rev-parse", "HEAD").lower()
        if log:
            if before == after:
                log(f"[후킹 서버 업데이트] 이미 최신: {after[:12]}")
            else:
                log(
                    "[후킹 서버 업데이트] main 적용 완료: "
                    f"{before[:12]} -> {after[:12]}"
                )
        return {
            "branch": HOOKING_BRANCH,
            "before": before,
            "after": after,
            "changed": before != after,
            "config_backup": config_backup,
            "restart_required": before != after,
            "quarantined_untracked": [
                entry.as_posix() for entry in quarantine.entries
            ],
            "tracked_changes_backup": (
                {
                    "path": str(tracked_backup.path),
                    "files_root": str(tracked_backup.path / "files"),
                    "patch_path": str(tracked_backup.patch_path),
                    "entries": [
                        entry.as_posix() for entry in tracked_backup.entries
                    ],
                    "automatic_restore": False,
                }
                if tracked_backup.path is not None
                else None
            ),
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
