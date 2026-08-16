from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


class SourceCompatibilityError(RuntimeError):
    """설치기가 관리하는 ComfyUI 소스 호환 패치 실패."""


LogCallback = Callable[[str], None]

_SERVER_RELATIVE_PATH = Path("server.py")
_PATCH_MARKER = "# comfy-installer: keep system_stats available when GPU telemetry fails"
_DEVICE_STATS_ANCHOR = """            device_entries = []
            for d in torch_devices:
                vram_total, torch_vram_total = comfy.model_management.get_total_memory(d, torch_total_too=True)
                vram_free, torch_vram_free = comfy.model_management.get_free_memory(d, torch_free_too=True)
                device_entries.append({
                    "name": comfy.model_management.get_torch_device_name(d),
                    "type": d.type,
                    "index": d.index,
                    "vram_total": vram_total,
                    "vram_free": vram_free,
                    "torch_vram_total": torch_vram_total,
                    "torch_vram_free": torch_vram_free,
                })
"""
_PATCHED_DEVICE_STATS = f"""            device_entries = []
            for d in torch_devices:
                try:
                    vram_total, torch_vram_total = comfy.model_management.get_total_memory(d, torch_total_too=True)
                    vram_free, torch_vram_free = comfy.model_management.get_free_memory(d, torch_free_too=True)
                    device_name = comfy.model_management.get_torch_device_name(d)
                except Exception as exc:
                    {_PATCH_MARKER}
                    print(
                        "[SYSTEM_STATS] GPU telemetry failed; returning zeroed "
                        f"values for {{d}}: {{type(exc).__name__}}: {{exc}}"
                    )
                    traceback.print_exc()
                    vram_total = 0
                    vram_free = 0
                    torch_vram_total = 0
                    torch_vram_free = 0
                    device_name = str(d)
                device_entries.append({{
                    "name": device_name,
                    "type": d.type,
                    "index": d.index,
                    "vram_total": vram_total,
                    "vram_free": vram_free,
                    "torch_vram_total": torch_vram_total,
                    "torch_vram_free": torch_vram_free,
                }})
"""


def _emit(message: str, log: LogCallback | None) -> None:
    print(f"[COMFY_INSTALL][SOURCE_COMPAT] {message}")
    if log:
        log(f"[ComfyUI 호환] {message}")


def _server_path(comfy_root: Path) -> Path:
    return comfy_root / _SERVER_RELATIVE_PATH


def _git_head_server_source(
    source_path: Path,
    *,
    log: LogCallback | None,
) -> str | None:
    comfy_root = source_path.parent
    git_metadata = comfy_root / ".git"
    if not git_metadata.exists():
        _emit(
            "Git 저장소가 아니므로 HEAD 기반 system_stats 패치 판별을 "
            f"생략합니다: {comfy_root}",
            log,
        )
        return None

    relative_path = source_path.relative_to(comfy_root).as_posix()
    command = ["git", "show", f"HEAD:{relative_path}"]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command,
            cwd=comfy_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        return completed.stdout.decode("utf-8").replace("\r\n", "\n")
    except Exception as exc:
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        print(
            "[COMFY_INSTALL][SOURCE_COMPAT] Git HEAD의 ComfyUI 서버 소스 "
            "확인 실패: "
            f"comfy_root={comfy_root}, path={relative_path}, "
            f"stderr={stderr!r}, error={exc}"
        )
        traceback.print_exc()
        raise SourceCompatibilityError(
            "Git HEAD의 ComfyUI 서버 소스를 확인하지 못해 system_stats "
            f"패치를 안전하게 해제할 수 없습니다: {source_path}"
        ) from exc


def _read_utf8_normalized(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    crlf_count = text.count("\r\n")
    bare_lf_count = text.count("\n") - crlf_count
    without_crlf = text.replace("\r\n", "")
    if "\r" in without_crlf:
        raise SourceCompatibilityError(
            f"지원하지 않는 CR 줄바꿈이 포함되어 있습니다: {path}"
        )
    if crlf_count and bare_lf_count:
        raise SourceCompatibilityError(
            f"줄바꿈이 CRLF와 LF로 혼합되어 자동 패치하지 않습니다: {path}"
        )
    newline = "\r\n" if crlf_count else "\n"
    return text.replace("\r\n", "\n"), newline


def _atomic_write_utf8(path: Path, content: str, *, newline: str) -> None:
    temporary = path.with_name(f".{path.name}.compat-{uuid.uuid4().hex}.tmp")
    try:
        payload = content.replace("\n", newline).encode("utf-8")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        print(
            "[COMFY_INSTALL][SOURCE_COMPAT] UTF-8 원자적 쓰기 실패: "
            f"target={path}, temporary={temporary}"
        )
        traceback.print_exc()
        try:
            if temporary.exists():
                temporary.unlink()
        except Exception as cleanup_exc:
            print(
                "[COMFY_INSTALL][SOURCE_COMPAT] 실패한 임시 파일 정리 실패: "
                f"path={temporary}, error={cleanup_exc}"
            )
            traceback.print_exc()
        raise


def _backup_before_write(
    source_path: Path,
    *,
    requirements_dir: Path,
    operation: str,
    log: LogCallback | None,
) -> Path:
    payload = source_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    backup_root = requirements_dir / "comfy-source-compatibility"
    backup = backup_root / f"server_{operation}_{digest[:16]}.py"
    backup_root.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if backup.read_bytes() != payload:
            raise SourceCompatibilityError(
                "호환 패치 백업 파일의 내용이 원본과 다릅니다: "
                f"backup={backup}, source={source_path}"
            )
        _emit(f"기존 백업 재사용: {backup}", log)
        return backup
    shutil.copy2(source_path, backup)
    _emit(f"원본 백업 완료: {source_path} -> {backup}", log)
    return backup


def apply_comfy_system_stats_compatibility(
    *,
    comfy_root: Path,
    requirements_dir: Path,
    log: LogCallback | None = None,
) -> dict[str, str | bool | None]:
    source_path = _server_path(comfy_root)
    try:
        if not source_path.is_file():
            raise SourceCompatibilityError(
                f"ComfyUI 서버 파일이 없습니다: {source_path}"
            )
        source, newline = _read_utf8_normalized(source_path)
        if _PATCHED_DEVICE_STATS in source:
            _emit(f"system_stats GPU 폴백 패치 재사용: {source_path}", log)
            return {
                "status": "reused",
                "path": str(source_path),
                "backup": None,
                "changed": False,
            }
        if _PATCH_MARKER in source:
            raise SourceCompatibilityError(
                "system_stats GPU 폴백 패치 표식은 있으나 본문이 다릅니다: "
                f"{source_path}"
            )
        if source.count(_DEVICE_STATS_ANCHOR) != 1:
            raise SourceCompatibilityError(
                "ComfyUI system_stats 코드가 검증된 형식과 달라 자동 패치하지 "
                f"않습니다: {source_path}"
            )

        backup = _backup_before_write(
            source_path,
            requirements_dir=requirements_dir,
            operation="before-patch",
            log=log,
        )
        patched = source.replace(
            _DEVICE_STATS_ANCHOR,
            _PATCHED_DEVICE_STATS,
            1,
        )
        compile(patched, str(source_path), "exec")
        _atomic_write_utf8(source_path, patched, newline=newline)
        _emit(f"system_stats GPU 폴백 패치 적용 완료: {source_path}", log)
        return {
            "status": "patched",
            "path": str(source_path),
            "backup": str(backup),
            "changed": True,
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][SOURCE_COMPAT] system_stats 호환 패치 실패: "
            f"path={source_path}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, SourceCompatibilityError):
            raise
        raise SourceCompatibilityError(
            f"ComfyUI system_stats 호환 패치 실패: {exc}"
        ) from exc


def remove_comfy_system_stats_compatibility(
    *,
    comfy_root: Path,
    requirements_dir: Path,
    log: LogCallback | None = None,
    allow_missing: bool = False,
) -> dict[str, str | bool | None]:
    source_path = _server_path(comfy_root)
    try:
        if not source_path.is_file():
            message = f"ComfyUI 서버 파일이 없습니다: {source_path}"
            if allow_missing:
                _emit(f"패치 해제 생략 — {message}", log)
                return {
                    "status": "missing",
                    "path": str(source_path),
                    "backup": None,
                    "changed": False,
                }
            raise SourceCompatibilityError(message)

        source, newline = _read_utf8_normalized(source_path)
        if _PATCHED_DEVICE_STATS not in source:
            if _PATCH_MARKER in source:
                raise SourceCompatibilityError(
                    "system_stats GPU 폴백 패치 표식은 있으나 본문이 달라 "
                    f"해제하지 않습니다: {source_path}"
                )
            _emit(f"해제할 system_stats GPU 폴백 패치 없음: {source_path}", log)
            return {
                "status": "unpatched",
                "path": str(source_path),
                "backup": None,
                "changed": False,
            }

        head_source = _git_head_server_source(source_path, log=log)
        if head_source is not None and _PATCHED_DEVICE_STATS in head_source:
            _emit(
                "Git HEAD에 system_stats GPU 폴백이 이미 포함되어 정식 "
                f"소스를 패치 해제하지 않습니다: {source_path}",
                log,
            )
            return {
                "status": "upstream-compatible",
                "path": str(source_path),
                "backup": None,
                "changed": False,
            }

        backup = _backup_before_write(
            source_path,
            requirements_dir=requirements_dir,
            operation="before-unpatch",
            log=log,
        )
        restored = source.replace(
            _PATCHED_DEVICE_STATS,
            _DEVICE_STATS_ANCHOR,
            1,
        )
        compile(restored, str(source_path), "exec")
        _atomic_write_utf8(source_path, restored, newline=newline)
        _emit(f"Git 갱신 전 system_stats 패치 해제 완료: {source_path}", log)
        return {
            "status": "removed",
            "path": str(source_path),
            "backup": str(backup),
            "changed": True,
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][SOURCE_COMPAT] system_stats 패치 해제 실패: "
            f"path={source_path}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, SourceCompatibilityError):
            raise
        raise SourceCompatibilityError(
            f"ComfyUI system_stats 호환 패치 해제 실패: {exc}"
        ) from exc


@contextmanager
def managed_comfy_system_stats_update(
    *,
    comfy_root: Path,
    requirements_dir: Path,
    log: LogCallback | None = None,
) -> Iterator[None]:
    source_path = _server_path(comfy_root)
    remove_comfy_system_stats_compatibility(
        comfy_root=comfy_root,
        requirements_dir=requirements_dir,
        log=log,
        allow_missing=True,
    )
    try:
        yield
    except Exception as operation_exc:
        if source_path.is_file():
            try:
                apply_comfy_system_stats_compatibility(
                    comfy_root=comfy_root,
                    requirements_dir=requirements_dir,
                    log=log,
                )
            except Exception as restore_exc:
                print(
                    "[COMFY_INSTALL][SOURCE_COMPAT] ComfyUI 소스 작업 실패 후 "
                    "system_stats 패치 복구도 실패: "
                    f"operation_error={operation_exc}, restore_error={restore_exc}"
                )
                traceback.print_exc()
                raise SourceCompatibilityError(
                    "ComfyUI 소스 작업과 system_stats 패치 복구가 모두 "
                    f"실패했습니다: operation={operation_exc}, "
                    f"restore={restore_exc}"
                ) from operation_exc
        else:
            print(
                "[COMFY_INSTALL][SOURCE_COMPAT] ComfyUI 소스 작업 실패 후 "
                f"복구할 서버 파일이 없습니다: {source_path}"
            )
        raise

    apply_comfy_system_stats_compatibility(
        comfy_root=comfy_root,
        requirements_dir=requirements_dir,
        log=log,
    )
