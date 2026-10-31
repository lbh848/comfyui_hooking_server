from __future__ import annotations

import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from threading import Event
from typing import Callable


class ComfyMigrationError(RuntimeError):
    """기존 ComfyUI 사용자 LoRA/봇 캐시 복사 실패."""


class ComfyMigrationCancelled(ComfyMigrationError):
    """사용자가 진행 중인 데이터 이사를 중단함."""


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict], None]

_ROBOCOPY_THREADS = 8
_PROGRESS_INTERVAL_SECONDS = 0.5


def _emit_progress(progress: ProgressCallback | None, payload: dict) -> None:
    if progress is not None:
        progress(payload)


def _robocopy_command(
    executable: str,
    source_root: Path,
    destination_root: Path,
) -> list[str]:
    # /XC /XN /XO를 함께 사용하면 대상에 이름이 같은 파일이 어떤 상태이든
    # 복사 대상에서 제외된다. /MOV, /MOVE, /MIR, /PURGE는 안전상 사용하지 않는다.
    return [
        executable,
        str(source_root),
        str(destination_root),
        "*.*",
        "/E",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/XC",
        "/XN",
        "/XO",
        "/XJ",
        "/R:1",
        "/W:1",
        f"/MT:{_ROBOCOPY_THREADS}",
        "/J",
        "/BYTES",
        "/NFL",
        "/NDL",
        "/NP",
        "/NJH",
    ]


def _pending_snapshot(
    entries: list[tuple[Path, Path, str, int]],
) -> tuple[int, int, str]:
    transferred = 0
    completed = 0
    latest_item = ""
    latest_mtime_ns = -1
    for _source, destination, _label, expected_size in entries:
        try:
            stat = destination.stat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(
                "[COMFY_INSTALL][MIGRATE] 진행률 파일 확인 실패: "
                f"path={destination}, error={exc}"
            )
            traceback.print_exc()
            continue
        actual_size = max(int(stat.st_size), 0)
        transferred += min(actual_size, expected_size)
        if actual_size == expected_size:
            completed += 1
        if stat.st_mtime_ns > latest_mtime_ns:
            latest_mtime_ns = stat.st_mtime_ns
            latest_item = destination.name
    return transferred, completed, latest_item


def _progress_payload(
    *,
    engine: str,
    entries: list[tuple[Path, Path, str, int]],
    completed_bytes_before: int,
    completed_files_before: int,
    total_bytes: int,
    total_files: int,
    started_at: float,
) -> dict:
    mapping_bytes, mapping_files, latest_item = _pending_snapshot(entries)
    downloaded = min(completed_bytes_before + mapping_bytes, total_bytes)
    current = min(completed_files_before + mapping_files, total_files)
    elapsed = max(time.monotonic() - started_at, 0.001)
    bytes_per_second = downloaded / elapsed if downloaded > 0 else 0.0
    remaining = max(total_bytes - downloaded, 0)
    eta_seconds = (
        remaining / bytes_per_second if bytes_per_second > 0 else None
    )
    return {
        "event": "migration_copy",
        "engine": engine,
        "item": latest_item,
        "current": current,
        "total": total_files,
        "overall_downloaded": downloaded,
        "overall_total": total_bytes,
        "bytes_per_second": round(bytes_per_second, 3),
        "eta_seconds": (
            round(eta_seconds, 1) if eta_seconds is not None else None
        ),
    }


def _terminate_robocopy(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][MIGRATE] robocopy 정상 종료 실패, 강제 종료 시도: "
            f"pid={process.pid}, error={exc}"
        )
        traceback.print_exc()
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception as kill_exc:
            print(
                "[COMFY_INSTALL][MIGRATE] robocopy 강제 종료 실패: "
                f"pid={process.pid}, error={kill_exc}"
            )
            traceback.print_exc()


def _cleanup_partial_files(
    entries: list[tuple[Path, Path, str, int]],
    *,
    log: LogCallback | None,
) -> None:
    for _source, destination, _label, expected_size in entries:
        try:
            if not destination.exists():
                continue
            if destination.stat().st_size == expected_size:
                continue
            destination.unlink()
            message = f"중단된 부분 파일 정리: {destination}"
            print(f"[COMFY_INSTALL][MIGRATE] {message}")
            if log:
                log(f"[이사] {message}")
        except Exception as exc:
            print(
                "[COMFY_INSTALL][MIGRATE] 중단된 부분 파일 정리 실패: "
                f"path={destination}, error={exc}"
            )
            traceback.print_exc()


def _copy_with_robocopy(
    *,
    executable: str,
    source_root: Path,
    destination_root: Path,
    entries: list[tuple[Path, Path, str, int]],
    completed_bytes_before: int,
    completed_files_before: int,
    total_bytes: int,
    total_files: int,
    started_at: float,
    cancel_event: Event | None,
    progress: ProgressCallback | None,
    log: LogCallback | None,
) -> int:
    destination_root.mkdir(parents=True, exist_ok=True)
    command = _robocopy_command(executable, source_root, destination_root)
    creationflags = (
        getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    )
    if log:
        log(
            "[이사] robocopy 병렬 복사 시작: "
            f"source={source_root}, files={len(entries)}, "
            f"threads={_ROBOCOPY_THREADS}"
        )
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][MIGRATE] robocopy 시작 실패: "
            f"command={command}, error={exc}"
        )
        traceback.print_exc()
        raise ComfyMigrationError(f"robocopy 시작 실패: {exc}") from exc

    try:
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                print(
                    "[COMFY_INSTALL][MIGRATE] 사용자 요청으로 robocopy 중단: "
                    f"pid={process.pid}, source={source_root}"
                )
                _terminate_robocopy(process)
                _cleanup_partial_files(entries, log=log)
                raise ComfyMigrationCancelled("사용자 데이터 이사를 중단했습니다.")
            _emit_progress(
                progress,
                _progress_payload(
                    engine="robocopy",
                    entries=entries,
                    completed_bytes_before=completed_bytes_before,
                    completed_files_before=completed_files_before,
                    total_bytes=total_bytes,
                    total_files=total_files,
                    started_at=started_at,
                ),
            )
            time.sleep(_PROGRESS_INTERVAL_SECONDS)
        return_code = int(process.returncode or 0)
        _emit_progress(
            progress,
            _progress_payload(
                engine="robocopy",
                entries=entries,
                completed_bytes_before=completed_bytes_before,
                completed_files_before=completed_files_before,
                total_bytes=total_bytes,
                total_files=total_files,
                started_at=started_at,
            ),
        )
        if return_code >= 8:
            print(
                "[COMFY_INSTALL][MIGRATE] robocopy 실패: "
                f"returncode={return_code}, source={source_root}, "
                f"destination={destination_root}"
            )
            raise ComfyMigrationError(
                f"robocopy 복사 실패(returncode={return_code}): {source_root}"
            )
        if log:
            log(
                "[이사] robocopy 병렬 복사 완료: "
                f"source={source_root}, returncode={return_code}"
            )
        return return_code
    except BaseException:
        if process.poll() is None:
            _terminate_robocopy(process)
        raise


def _copy_with_python(
    *,
    entries: list[tuple[Path, Path, str, int]],
    total_bytes: int,
    total_files: int,
    cancel_event: Event | None,
    progress: ProgressCallback | None,
    log: LogCallback | None,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    copied: list[str] = []
    skipped: list[str] = []
    failures: list[dict[str, str]] = []
    copied_bytes = 0
    started_at = time.monotonic()
    for source, destination, label, size in entries:
        if cancel_event is not None and cancel_event.is_set():
            print(
                "[COMFY_INSTALL][MIGRATE] 사용자 요청으로 Python 복사 중단: "
                f"source={source}, destination={destination}"
            )
            raise ComfyMigrationCancelled("사용자 데이터 이사를 중단했습니다.")
        if destination.exists():
            skipped.append(str(destination))
            copied_bytes += size
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(str(destination))
            copied_bytes += size
            elapsed = max(time.monotonic() - started_at, 0.001)
            speed = copied_bytes / elapsed
            remaining = max(total_bytes - copied_bytes, 0)
            _emit_progress(
                progress,
                {
                    "event": "migration_copy",
                    "engine": "python",
                    "item": destination.name,
                    "current": len(copied) + len(skipped),
                    "total": total_files,
                    "overall_downloaded": min(copied_bytes, total_bytes),
                    "overall_total": total_bytes,
                    "bytes_per_second": round(speed, 3),
                    "eta_seconds": (
                        round(remaining / speed, 1) if speed > 0 else None
                    ),
                },
            )
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
                    "label": label,
                }
            )
    if log:
        log(
            "[이사] Python 안전 복사 완료: "
            f"copied={len(copied)}, skipped={len(skipped)}"
        )
    return copied, skipped, failures


def migrate_user_data(
    *,
    old_comfy_root: str | os.PathLike[str],
    new_comfy_root: str | os.PathLike[str],
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
    copy_engine: str = "auto",
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
        if copy_engine not in {"auto", "robocopy", "python"}:
            raise ComfyMigrationError(f"지원하지 않는 복사 엔진: {copy_engine}")

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
        skipped: list[str] = []
        missing: list[str] = []
        failures: list[dict[str, str]] = []
        pending_by_mapping: list[
            tuple[Path, Path, str, list[tuple[Path, Path, str, int]]]
        ] = []
        scanned_files = 0
        pending_bytes = 0

        _emit_progress(
            progress,
            {
                "event": "migration_scan",
                "current": 0,
                "total": 0,
                "item": "이사할 파일 목록 확인 중",
            },
        )
        for source_root, destination_root, label in mappings:
            if not source_root.is_dir():
                message = f"{label} 원본 폴더 없음: {source_root}"
                print(f"[COMFY_INSTALL][MIGRATE] {message}")
                if log:
                    log(f"[이사] {message}")
                missing.append(str(source_root))
                continue
            entries: list[tuple[Path, Path, str, int]] = []
            for source in sorted(source_root.rglob("*")):
                if cancel_event is not None and cancel_event.is_set():
                    print(
                        "[COMFY_INSTALL][MIGRATE] 파일 목록 확인 중 사용자 중단: "
                        f"source={source_root}, scanned={scanned_files}"
                    )
                    raise ComfyMigrationCancelled(
                        "사용자 데이터 이사를 중단했습니다."
                    )
                if not source.is_file():
                    continue
                try:
                    relative = source.relative_to(source_root)
                    destination = destination_root / relative
                    size = max(int(source.stat().st_size), 0)
                    scanned_files += 1
                    if destination.exists():
                        skipped.append(str(destination))
                    else:
                        entries.append((source, destination, label, size))
                        pending_bytes += size
                    if scanned_files % 100 == 0:
                        _emit_progress(
                            progress,
                            {
                                "event": "migration_scan",
                                "current": scanned_files,
                                "total": 0,
                                "item": relative.name,
                                "pending_files": sum(
                                    len(item[3]) for item in pending_by_mapping
                                )
                                + len(entries),
                                "pending_bytes": pending_bytes,
                                "skipped_files": len(skipped),
                            },
                        )
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][MIGRATE] 이사 파일 확인 실패: "
                        f"source={source}, error={exc}"
                    )
                    traceback.print_exc()
                    failures.append(
                        {
                            "source": str(source),
                            "destination": "",
                            "error": str(exc),
                            "label": label,
                        }
                    )
            pending_by_mapping.append(
                (source_root, destination_root, label, entries)
            )

        pending_files = sum(len(item[3]) for item in pending_by_mapping)
        robocopy = shutil.which("robocopy") if os.name == "nt" else None
        engine = copy_engine
        if engine == "auto":
            engine = "robocopy" if robocopy else "python"
        if engine == "robocopy" and not robocopy:
            print(
                "[COMFY_INSTALL][MIGRATE] robocopy를 찾지 못해 Python 복사로 대체합니다."
            )
            if log:
                log("[이사] robocopy 없음: Python 안전 복사 사용")
            engine = "python"
        _emit_progress(
            progress,
            {
                "event": "migration_copy",
                "engine": engine,
                "current": 0,
                "total": pending_files,
                "overall_downloaded": 0,
                "overall_total": pending_bytes,
                "bytes_per_second": 0,
                "eta_seconds": None,
                "item": "복사 준비",
                "scanned_files": scanned_files,
                "skipped_files": len(skipped),
            },
        )
        if log:
            log(
                "[이사] 파일 확인 완료: "
                f"scanned={scanned_files}, pending={pending_files}, "
                f"pending_bytes={pending_bytes}, skipped={len(skipped)}, "
                f"engine={engine}"
            )

        copied: list[str] = []
        if engine == "robocopy":
            copy_started = time.monotonic()
            completed_bytes = 0
            completed_files = 0
            for source_root, destination_root, label, entries in pending_by_mapping:
                if not entries:
                    continue
                _copy_with_robocopy(
                    executable=str(robocopy),
                    source_root=source_root,
                    destination_root=destination_root,
                    entries=entries,
                    completed_bytes_before=completed_bytes,
                    completed_files_before=completed_files,
                    total_bytes=pending_bytes,
                    total_files=pending_files,
                    started_at=copy_started,
                    cancel_event=cancel_event,
                    progress=progress,
                    log=log,
                )
                for source, destination, _entry_label, size in entries:
                    try:
                        if destination.is_file() and destination.stat().st_size == size:
                            copied.append(str(destination))
                            completed_bytes += size
                            completed_files += 1
                        else:
                            raise ComfyMigrationError(
                                "robocopy 후 대상 파일 크기가 일치하지 않습니다."
                            )
                    except Exception as exc:
                        print(
                            "[COMFY_INSTALL][MIGRATE] robocopy 결과 검증 실패: "
                            f"source={source}, destination={destination}, error={exc}"
                        )
                        traceback.print_exc()
                        failures.append(
                            {
                                "source": str(source),
                                "destination": str(destination),
                                "error": str(exc),
                                "label": label,
                            }
                        )
        else:
            all_entries = [
                entry
                for _source_root, _destination_root, _label, entries
                in pending_by_mapping
                for entry in entries
            ]
            python_copied, late_skipped, python_failures = _copy_with_python(
                entries=all_entries,
                total_bytes=pending_bytes,
                total_files=pending_files,
                cancel_event=cancel_event,
                progress=progress,
                log=log,
            )
            copied.extend(python_copied)
            skipped.extend(late_skipped)
            failures.extend(python_failures)

        result = {
            "old_comfy_root": str(old_root),
            "new_comfy_root": str(new_root),
            "copy_engine": engine,
            "scanned_files": scanned_files,
            "pending_bytes": pending_bytes,
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
            f"engine={engine}, copied={len(copied)}, "
            f"skipped={len(skipped)}, missing={len(missing)}"
        )
        return result
    except ComfyMigrationError:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][MIGRATE] 이사 실패: {exc}")
        traceback.print_exc()
        raise ComfyMigrationError(f"사용자 데이터 이사 실패: {exc}") from exc
