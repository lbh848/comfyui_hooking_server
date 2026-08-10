"""Modal SDK 동작을 별도 프로세스에서 실행한다.

프로필 선택은 MODAL_PROFILE 환경변수에만 적용되어 메인 서버 프로세스나 다른
사용자의 활성 프로필을 바꾸지 않는다. 입력에는 API 키가 포함될 수 있으므로 stdin만
사용하고 명령행이나 파일에 남기지 않는다.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import io
from pathlib import Path, PurePosixPath
import sys
import time
import traceback

import modal
from modal._logs import LogsFilters, tail_logs
from modal._server import _Server
from modal.client import _Client
from modal_proto import api_pb2

from modal_backend.settings import normalize_modal_gpu


MODEL_SYNC_MANIFEST_PATH = "/.soya-local-model-sync-manifest.json"
# 기존 사용자 LoRA Volume의 동기화 기록을 그대로 이어받는다.
LORA_SYNC_MANIFEST_PATH = "/.soya-sync-manifest.json"
INSTALL_PROGRESS_PREFIX = "@@SOYA_MODAL_PROGRESS@@"
CALL_STARTED_LOG_PREFIX = "@@SOYA_MODAL_CALL_STARTED@@"
CALL_START_POLL_SECONDS = 3.0
CALL_START_LOG_TAIL_ENTRIES = 1000


class ModalContainerStartRetryLimitError(RuntimeError):
    """배포 App의 컨테이너 시작 반복이 사용자 설정 한도를 넘었다."""


def _container_start_max_retries(payload: dict) -> int:
    raw_value = payload.get("container_start_max_retries", 2)
    if isinstance(raw_value, bool):
        raise ValueError("Modal 컨테이너 시작 재시도 횟수는 0~10 사이의 정수여야 합니다.")
    retries = int(raw_value)
    if retries != float(raw_value) or not 0 <= retries <= 10:
        raise ValueError("Modal 컨테이너 시작 재시도 횟수는 0~10 사이의 정수여야 합니다.")
    return retries


def _call_start_observations(call) -> tuple[bool, set[str]]:
    """FunctionCall 로그의 구조화 컨텍스트에서 시작 완료와 컨테이너를 읽는다."""
    started = False
    container_ids: set[str] = set()
    for entry in call.logs.tail(entries=CALL_START_LOG_TAIL_ENTRIES):
        message = str(getattr(entry, "message", "") or "")
        if message.startswith(CALL_STARTED_LOG_PREFIX):
            started = True
        context_ids = list(getattr(entry, "context_ids", None) or [])
        # Modal 1.5 FunctionCall 로그 컨텍스트는 [input_id, container_id]다.
        # container_id가 없는 레코드에서 input_id를 컨테이너로 오인하지 않는다.
        if len(context_ids) < 2:
            continue
        container_id = str(context_ids[-1] or "").strip()
        if container_id:
            container_ids.add(container_id)
    return started, container_ids


def _cancel_function_call(
    call,
    *,
    operation: str,
    terminate_containers: bool,
) -> None:
    try:
        call.cancel(terminate_containers=terminate_containers)
    except Exception as cancel_exc:
        print(
            f"[MODAL_CLIENT] {operation} 호출 취소 실패: "
            f"terminate_containers={terminate_containers}, "
            f"error={type(cancel_exc).__name__}: {cancel_exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)


def _wait_for_call_with_start_retry_limit(
    call,
    *,
    timeout_seconds: int,
    max_retries: int,
    operation: str,
):
    """컨테이너 시작 반복을 감시하다 원격 메서드 진입 후 일반 대기로 전환한다."""
    allowed_attempts = max_retries + 1
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    observed_container_ids: set[str] = set()
    monitoring_start = True

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"[MODAL_CLIENT] {operation} 결과 대기 시간 초과: "
                f"timeout_seconds={timeout_seconds}, "
                f"observed_container_starts={len(observed_container_ids)}",
                file=sys.stderr,
            )
            raise TimeoutError(
                f"Modal {operation} 결과가 {timeout_seconds}초 안에 도착하지 않았습니다."
            )

        wait_seconds = (
            min(CALL_START_POLL_SECONDS, remaining)
            if monitoring_start
            else remaining
        )
        try:
            return call.get(timeout=wait_seconds)
        # Modal 1.5의 FunctionCall.get(timeout=...) 폴링은 내장 TimeoutError를
        # 사용하지만 일부 SDK 경로는 modal.exception.TimeoutError를 사용한다.
        except (TimeoutError, modal.exception.TimeoutError):
            if not monitoring_start:
                continue
        except Exception:
            raise

        try:
            started, container_ids = _call_start_observations(call)
        except Exception as monitor_exc:
            print(
                f"[MODAL_CLIENT] {operation} 컨테이너 시작 감시 실패로 호출 취소: "
                f"error={type(monitor_exc).__name__}: {monitor_exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            _cancel_function_call(
                call,
                operation=operation,
                terminate_containers=True,
            )
            raise RuntimeError(
                f"Modal {operation} 컨테이너 시작 횟수를 확인하지 못해 안전을 위해 취소했습니다."
            ) from monitor_exc

        previous_count = len(observed_container_ids)
        observed_container_ids.update(container_ids)
        observed_count = len(observed_container_ids)
        if observed_count != previous_count:
            print(
                f"[MODAL_CLIENT] {operation} 컨테이너 시작 감지: "
                f"attempt={observed_count}/{allowed_attempts}, "
                f"max_retries={max_retries}",
                file=sys.stderr,
            )
        if observed_count > allowed_attempts:
            print(
                f"[MODAL_CLIENT] {operation} 컨테이너 시작 재시도 한도 초과: "
                f"observed={observed_count}, allowed={allowed_attempts}, "
                "원격 호출과 실행 컨테이너를 취소합니다.",
                file=sys.stderr,
            )
            _cancel_function_call(
                call,
                operation=operation,
                terminate_containers=True,
            )
            raise ModalContainerStartRetryLimitError(
                f"Modal {operation} 컨테이너 시작이 최초 1회와 추가 재시도 "
                f"{max_retries}회를 초과해 취소되었습니다."
            )
        if started:
            print(
                f"[MODAL_CLIENT] {operation} 원격 메서드 진입 확인: "
                f"container_starts={observed_count}, 시작 재시도 감시 종료",
                file=sys.stderr,
            )
            monitoring_start = False


def _emit_install_progress(event: str, **payload) -> None:
    """설치 subprocess의 stderr로만 전달하는 기계 판독용 진행 이벤트."""
    print(
        INSTALL_PROGRESS_PREFIX
        + json.dumps({"event": event, **payload}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )


def _error_reason(exc: Exception) -> str:
    """Map concrete SDK failures to stable reasons consumed by the UI."""
    if isinstance(exc, modal.exception.NotFoundError):
        return "app_not_deployed"
    if isinstance(
        exc,
        (
            modal.exception.ConnectionError,
            modal.exception.ServiceError,
            modal.exception.TimeoutError,
            ConnectionError,
            TimeoutError,
        ),
    ):
        return "network_unavailable"
    return "runtime_unavailable"


def _remote_function(payload: dict, function_name: str) -> modal.Function:
    return modal.Function.from_name(
        str(payload["app_name"]),
        function_name,
        environment_name=str(payload["environment"]),
    )


def _worker_gpu(payload: dict) -> str:
    return normalize_modal_gpu(
        payload.get("worker_gpu"),
        "Modal 동적 작업 워커 GPU",
    )


def _worker_cls(payload: dict) -> modal.Cls:
    worker_cls = modal.Cls.from_name(
        str(payload["app_name"]),
        "ComfyWorker",
        environment_name=str(payload["environment"]),
    )
    return worker_cls.with_options(gpu=_worker_gpu(payload))


def _dynamic_worker_function(payload: dict, function_name: str) -> modal.Function:
    return _remote_function(payload, function_name).with_options(
        gpu=_worker_gpu(payload)
    )


def _web_app_name(payload: dict) -> str:
    configured = str(payload.get("web_app_name") or "").strip()
    return configured or f"{str(payload['app_name'])}-web"


def _web_server(payload: dict) -> modal.Server:
    return modal.Server.from_name(
        _web_app_name(payload),
        "comfy_web_server",
        environment_name=str(payload["environment"]),
    )


def _read_payload() -> dict:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Modal 클라이언트 입력은 JSON 객체여야 합니다.")
    return payload


def install(payload: dict) -> dict:
    app_name = str(payload["app_name"])
    environment = str(payload["environment"])
    workflow_volume = modal.Volume.from_name(
        f"{app_name}-workflows",
        environment_name=environment,
        create_if_missing=True,
    )
    workflow_files = payload.get("workflow_files") or []
    workflow_bytes = sum(
        Path(str(item["source_path"])).stat().st_size for item in workflow_files
    )
    _emit_install_progress(
        "batch_start",
        label="워크플로우",
        total_files=len(workflow_files),
        total_bytes=workflow_bytes,
    )
    with workflow_volume.batch_upload(force=True) as batch:
        for item in workflow_files:
            batch.put_file(item["source_path"], f"/{item['remote_name']}")
            _emit_install_progress(
                "file_queued",
                label="워크플로우",
                name=str(item["remote_name"]),
            )
    _emit_install_progress(
        "batch_complete",
        label="워크플로우",
        processed_files=len(workflow_files),
        processed_bytes=workflow_bytes,
        uploaded_files=len(workflow_files),
        skipped_files=0,
    )
    sync = _sync_environment(payload)
    return {
        "uploaded_workflows": len(workflow_files),
        **sync,
    }


def _read_sync_manifest(
    volume: modal.Volume,
    manifest_path: str,
    label: str,
) -> dict:
    try:
        raw = b"".join(volume.read_file(manifest_path))
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, modal.exception.NotFoundError):
        return {}
    except Exception as exc:
        print(
            f"[MODAL_CLIENT] {label} 동기화 명세 읽기 실패: "
            f"path={manifest_path}, error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return {}


def _safe_remote_path(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        print(f"[MODAL_CLIENT] 안전하지 않은 Volume 경로 거부: {value!r}", file=sys.stderr)
        raise ValueError(f"안전하지 않은 Modal Volume 경로입니다: {value!r}")
    return path.as_posix()


def _safe_managed_lora_path(value: str) -> str:
    path = _safe_remote_path(value).rstrip("/")
    if path != "SOYA_CHAR_LORA" and not path.startswith("SOYA_CHAR_LORA/"):
        print(
            f"[MODAL_CLIENT] 관리 대상 밖의 LoRA 경로 거부: {value!r}",
            file=sys.stderr,
        )
        raise ValueError(f"SOYA_CHAR_LORA 밖의 원격 LoRA는 관리할 수 없습니다: {value!r}")
    return path


def _safe_workflow_name(value: str) -> str:
    raw = str(value or "").strip()
    path = PurePosixPath(raw.replace("\\", "/"))
    if (
        not raw
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != raw
        or path.suffix.casefold() != ".json"
    ):
        print(
            f"[MODAL_CLIENT] 안전하지 않은 원격 워크플로우 이름 거부: {value!r}",
            file=sys.stderr,
        )
        raise ValueError(f"원격 워크플로우 이름은 .json 파일명만 허용합니다: {value!r}")
    return path.name


def list_workflows(payload: dict) -> dict:
    app_name = str(payload["app_name"])
    environment = str(payload["environment"])
    volume = modal.Volume.from_name(
        f"{app_name}-workflows",
        environment_name=environment,
    )
    try:
        entries = volume.listdir("/", recursive=False)
    except Exception as exc:
        print(
            "[MODAL_CLIENT] 원격 워크플로우 목록 조회 실패: "
            f"app={app_name}, environment={environment}, "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        raise

    workflows: list[dict] = []
    errors: list[dict[str, str]] = []
    for entry in entries:
        entry_type = getattr(getattr(entry, "type", None), "name", "")
        if entry_type and entry_type != "FILE":
            continue
        relative = str(getattr(entry, "path", "") or "").replace("\\", "/").lstrip("/")
        path = PurePosixPath(relative)
        if len(path.parts) != 1 or path.suffix.casefold() != ".json":
            continue
        name = path.name
        record = {
            "name": name,
            "size": max(0, int(getattr(entry, "size", 0) or 0)),
            "mtime": int(getattr(entry, "mtime", 0) or 0),
            "valid": False,
        }
        try:
            raw = b"".join(volume.read_file(f"/{name}"))
            workflow = json.loads(raw.decode("utf-8"))
            if not isinstance(workflow, dict) or not workflow:
                raise ValueError("워크플로우 JSON 객체가 비어 있거나 객체가 아닙니다.")
            record.update(
                size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                valid=True,
            )
        except Exception as exc:
            print(
                "[MODAL_CLIENT] 원격 워크플로우 파일 검사 실패: "
                f"name={name}, error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append({"name": name, "error": record["error"]})
        workflows.append(record)
    workflows.sort(key=lambda item: str(item["name"]).casefold())
    return {"workflows": workflows, "errors": errors}


def read_workflow(payload: dict) -> dict:
    app_name = str(payload["app_name"])
    environment = str(payload["environment"])
    name = _safe_workflow_name(str(payload.get("workflow_name") or ""))
    volume = modal.Volume.from_name(
        f"{app_name}-workflows",
        environment_name=environment,
    )
    try:
        raw = b"".join(volume.read_file(f"/{name}"))
        workflow = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        print(
            "[MODAL_CLIENT] 원격 워크플로우 읽기 실패: "
            f"name={name}, error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        raise
    if not isinstance(workflow, dict) or not workflow:
        print(
            "[MODAL_CLIENT] 원격 워크플로우 JSON 객체가 비어 있습니다: "
            f"name={name}, type={type(workflow).__name__}",
            file=sys.stderr,
        )
        raise ValueError(f"원격 워크플로우 JSON 객체가 비어 있습니다: {name}")
    return {
        "name": name,
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "workflow": workflow,
    }


def _sync_files(
    volume: modal.Volume,
    files: list[dict],
    *,
    manifest_path: str,
    label: str,
) -> dict:
    manifest = _read_sync_manifest(volume, manifest_path, label)
    uploads = []
    skipped = 0
    processed_bytes = 0
    for item in files:
        source_path = Path(str(item.get("source_path") or ""))
        remote_path = _safe_remote_path(str(item.get("remote_path") or ""))
        sha256 = str(item.get("sha256") or "").strip().lower()
        size = int(item.get("size") or 0)
        if not source_path.is_file():
            print(
                f"[MODAL_CLIENT] {label} 로컬 원본 파일 없음: {source_path}",
                file=sys.stderr,
            )
            raise FileNotFoundError(f"Modal에 올릴 로컬 파일이 없습니다: {source_path}")
        actual_size = source_path.stat().st_size
        if not sha256 or size < 0 or actual_size != size:
            print(
                f"[MODAL_CLIENT] {label} 로컬 파일 명세 오류: source={source_path}, "
                f"sha256={sha256!r}, expected_size={size}, actual_size={actual_size}",
                file=sys.stderr,
            )
            raise ValueError(f"Modal에 올릴 로컬 파일의 검증 정보가 올바르지 않습니다: {source_path}")
        processed_bytes += actual_size
        expected = {"sha256": sha256, "size": size}
        if manifest.get(remote_path) == expected:
            skipped += 1
            continue
        uploads.append((item, remote_path, expected))
    _emit_install_progress(
        "batch_start",
        label=label,
        total_files=len(files),
        total_bytes=processed_bytes,
    )
    if uploads:
        try:
            with volume.batch_upload(force=True) as batch:
                for item, remote_path, expected in uploads:
                    batch.put_file(item["source_path"], f"/{remote_path}")
                    _emit_install_progress(
                        "file_queued",
                        label=label,
                        name=remote_path,
                    )
                    manifest[remote_path] = expected
                encoded = (
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
                ).encode("utf-8")
                batch.put_file(io.BytesIO(encoded), manifest_path)
        except Exception as exc:
            print(
                f"[MODAL_CLIENT] {label} 로컬 파일 업로드 실패: "
                f"uploads={len(uploads)}, error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            raise
    _emit_install_progress(
        "batch_complete",
        label=label,
        processed_files=len(files),
        processed_bytes=processed_bytes,
        uploaded_files=len(uploads),
        skipped_files=skipped,
    )
    return {"uploaded": len(uploads), "skipped": skipped}


def _sync_environment(payload: dict) -> dict:
    app_name = str(payload["app_name"])
    environment = str(payload["environment"])
    models_volume = modal.Volume.from_name(
        f"{app_name}-models",
        environment_name=environment,
        create_if_missing=True,
    )
    loras_volume = modal.Volume.from_name(
        f"{app_name}-loras",
        environment_name=environment,
        create_if_missing=True,
    )
    model_sync = _sync_files(
        models_volume,
        list(payload.get("model_files") or []),
        manifest_path=MODEL_SYNC_MANIFEST_PATH,
        label="모델",
    )
    lora_sync = _sync_files(
        loras_volume,
        list(payload.get("lora_files") or []),
        manifest_path=LORA_SYNC_MANIFEST_PATH,
        label="LoRA",
    )
    return {"model_sync": model_sync, "lora_sync": lora_sync}


def _write_sync_manifest(
    volume: modal.Volume,
    manifest_path: str,
    manifest: dict,
) -> None:
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(encoded), manifest_path)
    except Exception as exc:
        print(
            f"[MODAL_CLIENT] LoRA 동기화 명세 저장 실패: path={manifest_path}, "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        raise


def _lora_volume(payload: dict, *, create_if_missing: bool) -> modal.Volume:
    return modal.Volume.from_name(
        f"{str(payload['app_name'])}-loras",
        environment_name=str(payload["environment"]),
        create_if_missing=create_if_missing,
    )


def _list_lora_volume(volume: modal.Volume) -> dict:
    manifest = _read_sync_manifest(volume, LORA_SYNC_MANIFEST_PATH, "LoRA")
    try:
        entries = volume.listdir("/", recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        entries = []
    except Exception as exc:
        print(
            f"[MODAL_CLIENT] 원격 LoRA 파일 목록 조회 실패: "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        raise

    files: list[dict] = []
    actual_paths: set[str] = set()
    for entry in entries:
        entry_type = str(getattr(getattr(entry, "type", None), "name", "") or "")
        if entry_type and entry_type != "FILE":
            continue
        relative = str(getattr(entry, "path", "") or "").replace("\\", "/").lstrip("/")
        if not relative or f"/{relative}" == LORA_SYNC_MANIFEST_PATH:
            continue
        try:
            safe_path = _safe_remote_path(relative)
        except ValueError as exc:
            print(
                f"[MODAL_CLIENT] 안전하지 않은 원격 LoRA 항목 제외: "
                f"path={relative!r}, error={exc}",
                file=sys.stderr,
            )
            continue
        actual_paths.add(safe_path)
        manifest_entry = manifest.get(safe_path)
        if not isinstance(manifest_entry, dict):
            manifest_entry = {}
        files.append(
            {
                "path": safe_path,
                "size": max(0, int(getattr(entry, "size", 0) or 0)),
                "mtime": int(getattr(entry, "mtime", 0) or 0),
                "sha256": str(manifest_entry.get("sha256") or ""),
                "manifest_size": int(manifest_entry.get("size") or -1),
                "tracked": bool(manifest_entry),
            }
        )
    errors = [
        {
            "path": str(path),
            "error": "동기화 명세에는 있으나 실제 원격 파일이 없습니다.",
        }
        for path in manifest
        if path not in actual_paths
    ]
    files.sort(key=lambda item: str(item["path"]).casefold())
    return {
        "files": files,
        "errors": errors,
        "file_count": len(files),
        "tracked_count": sum(1 for item in files if item["tracked"]),
    }


def list_loras(payload: dict) -> dict:
    return _list_lora_volume(_lora_volume(payload, create_if_missing=True))


def _path_in_scopes(path: str, scopes: list[str]) -> bool:
    return any(path == scope or path.startswith(scope + "/") for scope in scopes)


def _delete_lora_paths(
    volume: modal.Volume,
    paths: list[str],
    *,
    recursive: bool,
) -> dict:
    normalized = list(dict.fromkeys(_safe_managed_lora_path(path) for path in paths))
    deleted: list[str] = []
    for path in normalized:
        try:
            volume.remove_file(f"/{path}", recursive=recursive)
            deleted.append(path)
        except (FileNotFoundError, modal.exception.NotFoundError):
            print(f"[MODAL_CLIENT] 원격 LoRA 삭제 대상 없음: {path}", file=sys.stderr)
        except Exception as exc:
            print(
                f"[MODAL_CLIENT] 원격 LoRA 삭제 실패: path={path}, "
                f"recursive={recursive}, error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            raise

    manifest = _read_sync_manifest(volume, LORA_SYNC_MANIFEST_PATH, "LoRA")
    filtered = {
        path: value
        for path, value in manifest.items()
        if not any(path == target or (recursive and path.startswith(target + "/")) for target in normalized)
    }
    if filtered != manifest:
        _write_sync_manifest(volume, LORA_SYNC_MANIFEST_PATH, filtered)
    return {"deleted": deleted, "deleted_count": len(deleted)}


def manage_loras(payload: dict) -> dict:
    mode = str(payload.get("mode") or "").strip().lower()
    if mode not in {"upload", "sync", "delete"}:
        print(f"[MODAL_CLIENT] 지원하지 않는 LoRA 관리 모드: {mode!r}", file=sys.stderr)
        raise ValueError(f"지원하지 않는 LoRA 관리 모드입니다: {mode!r}")
    scopes = list(
        dict.fromkeys(
            _safe_managed_lora_path(str(scope))
            for scope in (payload.get("scopes") or [])
        )
    )
    if not scopes:
        print("[MODAL_CLIENT] LoRA 관리 범위가 비어 있습니다.", file=sys.stderr)
        raise ValueError("원격 LoRA 관리 범위가 비어 있습니다.")

    volume = _lora_volume(payload, create_if_missing=True)
    if mode == "delete":
        current = _list_lora_volume(volume)
        deleted_file_count = sum(
            1
            for item in current["files"]
            if _path_in_scopes(str(item["path"]), scopes)
        )
        deleted = _delete_lora_paths(volume, scopes, recursive=True)
        deleted["deleted_scopes"] = int(deleted.get("deleted_count") or 0)
        deleted["deleted_count"] = deleted_file_count
        return {"mode": mode, "uploaded": 0, "skipped": 0, **deleted}

    files = list(payload.get("lora_files") or [])
    if not files:
        print(f"[MODAL_CLIENT] {mode}할 현재 사용 LoRA 파일이 없습니다.", file=sys.stderr)
        raise ValueError("업로드할 현재 사용 LoRA 파일이 없습니다.")
    for item in files:
        remote_path = _safe_managed_lora_path(str(item.get("remote_path") or ""))
        if not _path_in_scopes(remote_path, scopes):
            print(
                f"[MODAL_CLIENT] 선택 범위 밖의 LoRA 업로드 거부: "
                f"remote={remote_path}, scopes={scopes}",
                file=sys.stderr,
            )
            raise ValueError(f"선택한 관리 범위 밖의 LoRA 파일입니다: {remote_path}")
    sync_result = _sync_files(
        volume,
        files,
        manifest_path=LORA_SYNC_MANIFEST_PATH,
        label="LoRA",
    )
    deleted = {"deleted": [], "deleted_count": 0}
    if mode == "sync":
        desired = {
            _safe_managed_lora_path(str(item.get("remote_path") or ""))
            for item in files
        }
        current = _list_lora_volume(volume)
        extras = [
            str(item["path"])
            for item in current["files"]
            if _path_in_scopes(str(item["path"]), scopes)
            and str(item["path"]) not in desired
        ]
        deleted = _delete_lora_paths(volume, extras, recursive=False) if extras else deleted
    return {"mode": mode, **sync_result, **deleted}


def generate(payload: dict) -> dict:
    sync = _sync_environment(payload)
    input_files = {
        item["remote_name"]: Path(item["source_path"]).read_bytes()
        for item in (payload.get("input_files") or [])
    }
    worker_cls = _worker_cls(payload)
    call = worker_cls().generate.spawn(
        payload["workflow"],
        input_files,
        int(payload.get("timeout_seconds") or 3300),
        list(payload.get("artifact_prefixes") or []),
        bool(payload.get("require_images", True)),
    )
    try:
        remote_result = _wait_for_call_with_start_retry_limit(
            call,
            timeout_seconds=int(payload.get("timeout_seconds") or 3300) + 120,
            max_retries=_container_start_max_retries(payload),
            operation="generate",
        )
    except ModalContainerStartRetryLimitError:
        raise
    except Exception:
        _cancel_function_call(
            call,
            operation="generate",
            terminate_containers=False,
        )
        raise

    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, image in enumerate(remote_result.get("images") or []):
        target = output_dir / f"{index:03d}-{Path(image['filename']).name}"
        target.write_bytes(image["bytes"])
        outputs.append(
            {
                "path": str(target),
                "filename": image["filename"],
                "content_type": image.get("content_type", "application/octet-stream"),
                "node_id": image.get("node_id", ""),
            }
        )
    if bool(payload.get("require_images", True)) and not outputs:
        raise RuntimeError("Modal ComfyUI가 출력 이미지를 반환하지 않았습니다.")
    artifacts = []
    artifact_root = output_dir / "artifacts"
    for artifact in remote_result.get("artifacts") or []:
        relative = Path(str(artifact.get("relative_path") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(
                f"Modal이 안전하지 않은 LoRA 결과 경로를 반환했습니다: {relative!s}"
            )
        target = artifact_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact["bytes"])
        artifacts.append(
            {
                "path": str(target),
                "relative_path": relative.as_posix(),
                "size": target.stat().st_size,
            }
        )
    return {
        "prompt_id": remote_result.get("prompt_id"),
        "outputs": outputs,
        "artifacts": artifacts,
        "text_outputs": list(remote_result.get("text_outputs") or []),
        **sync,
    }


def convert_workflow(payload: dict) -> dict:
    _sync_environment(payload)
    worker_cls = _worker_cls(payload)
    timeout_seconds = max(30, min(int(payload.get("timeout_seconds") or 600), 900))
    call = worker_cls().convert.spawn(payload["workflow"])
    try:
        return _wait_for_call_with_start_retry_limit(
            call,
            timeout_seconds=timeout_seconds,
            max_retries=_container_start_max_retries(payload),
            operation="convert_workflow",
        )
    except ModalContainerStartRetryLimitError:
        raise
    except Exception:
        _cancel_function_call(
            call,
            operation="convert_workflow",
            terminate_containers=False,
        )
        raise


def update_autoscaler(payload: dict) -> dict:
    max_containers = int(payload["max_containers"])
    scaledown_window = int(payload["scaledown_window_seconds"])
    if not 1 <= max_containers <= 10:
        raise ValueError("Modal 최대 컨테이너 수는 1~10 사이여야 합니다.")
    if not 2 <= scaledown_window <= 1200:
        raise ValueError("Modal 유휴 종료 시간은 2~1200초 사이여야 합니다.")
    autoscaler_options = {
        "min_containers": 0,
        "max_containers": max_containers,
        "scaledown_window": scaledown_window,
    }
    probe = _dynamic_worker_function(payload, "gpu_probe")
    probe.update_autoscaler(**autoscaler_options)

    worker = _worker_cls(payload)()
    worker.update_autoscaler(**autoscaler_options)
    return {
        "updated": ["gpu_probe", "ComfyWorker"],
        "min_containers": 0,
        "max_containers": max_containers,
        "scaledown_window_seconds": scaledown_window,
    }


def runtime_stats(payload: dict) -> dict:
    worker_cls = _worker_cls(payload)
    stats = worker_cls().generate.get_current_stats()
    return {
        "backlog": int(stats.backlog),
        "num_total_runners": int(stats.num_total_runners),
        "num_running_inputs": int(stats.num_running_inputs),
        "input_headroom": int(stats.input_headroom),
    }


def gpu_probe(payload: dict) -> dict:
    probe = _dynamic_worker_function(payload, "gpu_probe")
    call = probe.spawn()
    try:
        return _wait_for_call_with_start_retry_limit(
            call,
            timeout_seconds=900,
            max_retries=_container_start_max_retries(payload),
            operation="gpu_probe",
        )
    except ModalContainerStartRetryLimitError:
        raise
    except Exception:
        _cancel_function_call(
            call,
            operation="gpu_probe",
            terminate_containers=False,
        )
        raise


def web_url(payload: dict) -> dict:
    """ComfyUI 웹 UI 공개 URL을 Modal에서 조회한다.

    comfy_web_server가 아직 배포되지 않았거나 Server 엔드포인트가 아니면
    get_url()이 None 을 반환하거나 NotFoundError 를 일으킨다(후자는
    _error_reason 이 app_not_deployed 로 매핑).
    """
    server = _web_server(payload)
    return {"url": server.get_url()}


async def _try_app_list_runners(
    client: "_Client",
    app_name: str,
    environment: str,
) -> int | None:
    """best-effort AppList probe. 성공 시 n_running_tasks, 실패/미조회 시 None.

    IMPORTANT: ``n_running_tasks``는 APP-level 통계이지 comfy_web_server 특정
    값이 아니다. 이 값이 WebUI 실행 여부를 의미하려면 SOYA_MODAL_WEB_APP_NAME가
    WebUI 전용 App이어야 한다(나중에 다른 작업을 같은 App에 넣으면 오판한다).

    Modal은 공개 Python API로 Server의 현재 replica 수를 제공하지 않는다
    (``modal.Server`` 공개 API는 get_url/autoscaler/logs 정도). 따라서 공식
    CLI(``modal app list``)가 내부적으로 쓰는 동일한 AppList gRPC를 직접
    호출한다. 이 경로는 공개·안정 API가 아니므로 Modal SDK 버전업 시 깨질 수
    있고, 그 경우 이 probe만 None 처리해 호출자가 안전 퇴각하도록 격리했다.
    향후 Modal이 Server용 공개 stats API를 제공하면 이 함수만 교체하면 된다.
    """
    try:
        response = await client.stub.AppList(
            api_pb2.AppListRequest(environment_name=environment)
        )
    except Exception as exc:
        print(
            f"[MODAL_CLIENT] AppList 호출 실패(runner probe 사용 불가): "
            f"app={app_name}, error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return None
    for app_stats in response.apps:
        if (
            str(app_stats.description) == app_name
            and int(app_stats.state) == api_pb2.APP_STATE_DEPLOYED
        ):
            return int(app_stats.n_running_tasks)
    print(
        f"[MODAL_CLIENT] 배포된 웹 Server App 통계 누락(미추적): "
        f"app={app_name}, environment={environment}",
        file=sys.stderr,
    )
    return None


async def _web_server_status_async(payload: dict) -> dict:
    """한 Modal 클라이언트에서 Server URL(필수)과 컨테이너 수(best-effort)를 읽는다.

    공개 ``modal.Server.get_url()``을 진실의 기본값으로 먼저 조회해 url/deployed를
    확정한다. 그 뒤 AppList probe로 running 컨테이너 수를 보강하되, probe는
    best-effort라 실패해도 url/deployed를 버리지 않고 runners=None으로 돌려준다.
    """
    app_name = _web_app_name(payload)
    environment = str(payload["environment"])
    client = await _Client.from_env()
    # 공개 ``modal.Server``는 동기 래퍼라 같은 프로세스에서 별도 asyncio.run과
    # 섞으면 SDK 실행 루프가 교착될 수 있다. 이미 생성한 비동기 클라이언트를
    # 내부 Server 객체에도 전달해 URL과 App 통계를 단일 루프에서 조회한다.
    server = _Server.from_name(
        app_name,
        "comfy_web_server",
        environment_name=environment,
        client=client,
    )
    # 1. get_url (공개 API, 필수) — 미배포면 None 또는 NotFoundError.
    try:
        url = await server.get_url()
    except modal.exception.NotFoundError as exc:
        # 웹 App은 사용자가 명시적으로 시작하기 전이나 종료한 뒤에는 존재하지
        # 않는 것이 정상 상태다. subprocess 자체를 실패시키면 짧은 상태 폴링마다
        # 같은 traceback이 상위 서버 로그에 반복 노출되므로 조용히 stopped로 돌려준다.
        print(
            f"[MODAL_CLIENT] 웹 App 미배포: app={app_name}, "
            f"environment={environment}, error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return {
            "url": None,
            "deployed": False,
            "runners": None,
            "app_name": app_name,
        }
    if not url:
        # get_url이 None을 반환하는 미배포 케이스.
        return {
            "url": None,
            "deployed": False,
            "runners": None,
            "app_name": app_name,
        }
    # 2. AppList best-effort probe — 실패해도 url/deployed는 살린다.
    runners = await _try_app_list_runners(client, app_name, environment)
    return {
        "url": url,
        "deployed": True,
        "runners": runners,
        "app_name": app_name,
    }


def _web_server_status(payload: dict) -> dict:
    return asyncio.run(_web_server_status_async(payload))


def web_status(payload: dict) -> dict:
    """웹 전용 App을 기동하지 않고 URL과 현재 Server task 수를 조회한다.

    ``deployed``/``runners``(int|None)가 새 필드이고, ``num_total_runners`` 등
    레거시 필드는 기존 _remote_web_status 호환을 위해 runners 기반으로 채운다.
    runners가 None이면 레거시 통계는 0으로 내려주되, 새 필드로 None임을 구분한다.
    """
    app_name = _web_app_name(payload)
    # Modal Server의 내부 Function에는 get_current_stats()가 노출되어 있어도
    # 호출하면 ConflictError가 난다. App task 수로 GPU 컨테이너 상태를 읽는다.
    stats = _web_server_status(payload)
    url = stats.get("url")
    deployed = bool(stats.get("deployed"))
    runners = stats.get("runners")
    runners_value = int(runners) if runners is not None else 0
    return {
        "url": url,
        "deployed": deployed,
        "runners": runners,
        "app_name": app_name,
        # 레거시 필드(_remote_web_status 호환). Server는 Function input 큐를
        # 쓰지 않으므로 running_inputs/backlog는 항상 0이다.
        "backlog": 0,
        "num_total_runners": runners_value,
        "num_running_inputs": 0,
    }


def _log_source(file_descriptor: int) -> str:
    if file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
        return "stdout"
    if file_descriptor == api_pb2.FILE_DESCRIPTOR_STDERR:
        return "stderr"
    return "system"


async def _tail_deployed_app_logs(
    client: _Client,
    *,
    app_name: str,
    environment: str,
    app_role: str,
    entries: int,
) -> list[dict]:
    deployed = await client.stub.AppGetByDeploymentName(
        api_pb2.AppGetByDeploymentNameRequest(
            name=app_name,
            environment_name=environment,
        )
    )
    app_id = str(deployed.app_id or "")
    if not app_id:
        raise modal.exception.NotFoundError(
            f"Modal App이 배포되어 있지 않습니다: app={app_name}"
        )

    function_names: dict[str, str] = {}
    try:
        layout = await client.stub.AppGetLayout(
            api_pb2.AppGetLayoutRequest(app_id=app_id)
        )
        function_names = {
            str(function_id): str(name)
            for name, function_id in layout.app_layout.function_ids.items()
        }
    except Exception as exc:
        print(
            f"[MODAL_CLIENT] 로그 함수 이름 조회 실패: app={app_name}, "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)

    records: list[dict] = []
    async for batch in tail_logs(
        client,
        app_id,
        entries,
        filters=LogsFilters(),
    ):
        function_id = str(batch.function_id or batch.root_function_id or "")
        function_name = function_names.get(function_id, "")
        for item in batch.items:
            message = str(item.data or "").replace("\x00", "").rstrip("\r\n")
            if not message:
                continue
            timestamp = (
                float(item.timestamp_ns) / 1_000_000_000
                if item.timestamp_ns
                else float(item.timestamp or 0.0)
            )
            source = _log_source(int(item.file_descriptor))
            if app_role == "web" or function_name == "comfy_web_server":
                category = "web"
            elif function_name == "gpu_probe":
                category = "diagnostic"
            elif not function_id and source == "system":
                category = "deployment"
            else:
                category = "jobs"
            records.append(
                {
                    "time": timestamp,
                    "timestamp": datetime.fromtimestamp(
                        timestamp or 0.0,
                        timezone.utc,
                    ).isoformat(),
                    "source": source,
                    "category": category,
                    "app_role": app_role,
                    "app_name": app_name,
                    "app_id": app_id,
                    "function_id": function_id,
                    "function_name": function_name,
                    "container_id": str(item.container_id or batch.task_id or ""),
                    "function_call_id": str(item.function_call_id or ""),
                    "message": message,
                }
            )
    return records


async def _runtime_logs_async(payload: dict) -> dict:
    environment = str(payload["environment"])
    requested = max(20, min(int(payload.get("entries") or 300), 1000))
    client = await _Client.from_env()
    app_specs = (
        (str(payload["app_name"]), "worker"),
        (_web_app_name(payload), "web"),
    )
    logs: list[dict] = []
    errors: list[dict] = []
    for app_name, app_role in app_specs:
        try:
            logs.extend(
                await _tail_deployed_app_logs(
                    client,
                    app_name=app_name,
                    environment=environment,
                    app_role=app_role,
                    entries=requested,
                )
            )
        except modal.exception.NotFoundError as exc:
            errors.append(
                {
                    "app_name": app_name,
                    "app_role": app_role,
                    "reason": "app_not_deployed",
                    "error": str(exc),
                }
            )
        except Exception as exc:
            print(
                f"[MODAL_CLIENT] App 로그 조회 실패: app={app_name}, "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            errors.append(
                {
                    "app_name": app_name,
                    "app_role": app_role,
                    "reason": _error_reason(exc),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    logs.sort(key=lambda item: (float(item.get("time") or 0.0), item.get("app_name", "")))
    if len(logs) > requested:
        logs = logs[-requested:]
    return {"logs": logs, "errors": errors, "limit": requested}


def runtime_logs(payload: dict) -> dict:
    return asyncio.run(_runtime_logs_async(payload))


def delete_lora_prefix(payload: dict) -> dict:
    prefix = _safe_managed_lora_path(payload["remote_prefix"])
    volume = _lora_volume(payload, create_if_missing=True)
    _delete_lora_paths(volume, [prefix], recursive=True)
    return {"deleted_prefix": prefix}


def main() -> int:
    try:
        payload = _read_payload()
        action = str(payload.get("action") or "")
        if action == "install":
            result = install(payload)
        elif action == "list_workflows":
            result = list_workflows(payload)
        elif action == "read_workflow":
            result = read_workflow(payload)
        elif action == "generate":
            result = generate(payload)
        elif action == "convert_workflow":
            result = convert_workflow(payload)
        elif action == "update_autoscaler":
            result = update_autoscaler(payload)
        elif action == "runtime_stats":
            result = runtime_stats(payload)
        elif action == "gpu_probe":
            result = gpu_probe(payload)
        elif action == "web_url":
            result = web_url(payload)
        elif action == "web_status":
            result = web_status(payload)
        elif action == "runtime_logs":
            result = runtime_logs(payload)
        elif action == "list_loras":
            result = list_loras(payload)
        elif action == "manage_loras":
            result = manage_loras(payload)
        elif action == "delete_lora_prefix":
            result = delete_lora_prefix(payload)
        else:
            raise ValueError(f"지원하지 않는 Modal 클라이언트 동작입니다: {action}")
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": _error_reason(exc),
                    "error_type": type(exc).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
        )
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
