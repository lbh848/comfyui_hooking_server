from __future__ import annotations

import asyncio
import json
import os
import re
import traceback
import uuid
from pathlib import Path

from aiohttp import web

from .crypto import PACK_MAGIC
from .service import ComfyInstallerService, InstallerServiceError


MAX_PACK_BYTES = 32 * 1024 * 1024
_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
APP_SERVICE_KEY = web.AppKey(
    "comfy_installer_service", ComfyInstallerService
)


def _json_error(message: str, *, status: int = 400) -> web.Response:
    print(
        f"[COMFY_INSTALL][API] 요청 실패: status={status}, error={message}"
    )
    return web.json_response({"ok": False, "error": message}, status=status)


def _pack_path(service: ComfyInstallerService, upload_id: str) -> Path:
    if not _UPLOAD_ID.fullmatch(upload_id):
        raise InstallerServiceError("워크플로우 팩 업로드 ID 형식이 잘못되었습니다.")
    path = (service.upload_root / f"{upload_id}.soyawfp").resolve()
    if path.parent != service.upload_root.resolve():
        raise InstallerServiceError("워크플로우 팩 업로드 경로가 안전하지 않습니다.")
    return path


async def handle_status(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        since = int(request.query.get("since", "0"))
        if since < 0:
            raise ValueError("negative")
        return web.json_response(
            {"ok": True, **service.status(since=since)}
        )
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 상태 조회 실패: {exc}")
        traceback.print_exc()
        return _json_error("설치 상태를 조회하지 못했습니다.", status=500)


async def handle_preflight(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        result = await asyncio.to_thread(service.preflight)
        return web.json_response({"ok": True, "preflight": result})
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 사전 검사 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=400)


async def handle_pack_upload(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    upload_id = uuid.uuid4().hex
    destination = _pack_path(service, upload_id)
    part = destination.with_name(f"{destination.name}.part")
    try:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "pack":
            return _json_error(
                "multipart의 pack 파일 필드가 필요합니다."
            )
        service.upload_root.mkdir(parents=True, exist_ok=True)
        total = 0
        prefix = b""
        with part.open("xb") as stream:
            while True:
                chunk = await field.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_PACK_BYTES:
                    raise InstallerServiceError(
                        "워크플로우 팩이 허용 크기(32 MiB)를 초과했습니다."
                    )
                if len(prefix) < len(PACK_MAGIC):
                    prefix = (prefix + chunk)[: len(PACK_MAGIC)]
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if total <= len(PACK_MAGIC) or prefix != PACK_MAGIC:
            raise InstallerServiceError(
                "SOYAWFP1 워크플로우 팩 형식이 아닙니다."
            )
        os.replace(part, destination)
        print(
            "[COMFY_INSTALL][API] 암호화 워크플로우 팩 업로드 완료: "
            f"id={upload_id}, size={total}"
        )
        return web.json_response(
            {
                "ok": True,
                "upload_id": upload_id,
                "filename": Path(field.filename or "workflow.soyawfp").name,
                "size": total,
            }
        )
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] 팩 업로드 검증 실패: {exc}")
        traceback.print_exc()
        if part.exists():
            invalid = part.with_name(f"{part.name}.invalid")
            suffix = 1
            while invalid.exists():
                invalid = part.with_name(f"{part.name}.invalid_{suffix}")
                suffix += 1
            os.replace(part, invalid)
            print(
                "[COMFY_INSTALL][API] 실패한 업로드 part 보존: "
                f"{invalid}"
            )
        return _json_error(str(exc))
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 팩 업로드 실패: {exc}")
        traceback.print_exc()
        return _json_error("워크플로우 팩 업로드에 실패했습니다.", status=500)


async def _read_json_object(request: web.Request) -> dict:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        print(f"[COMFY_INSTALL][API] JSON 본문 디코딩 실패: {exc}")
        traceback.print_exc()
        raise InstallerServiceError("JSON 요청 본문이 잘못되었습니다.") from exc
    if not isinstance(body, dict):
        raise InstallerServiceError("JSON 요청 본문은 객체여야 합니다.")
    return body


async def handle_start(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    workflow_key = ""
    civitai_key = ""
    try:
        body = await _read_json_object(request)
        upload_id = str(body.get("upload_id", ""))
        workflow_key = str(body.get("workflow_key", ""))
        civitai_key = str(body.get("civitai_key", ""))
        restore_after_success = body.get(
            "restore_config_after_success", False
        )
        if not isinstance(restore_after_success, bool):
            raise InstallerServiceError(
                "restore_config_after_success는 boolean이어야 합니다."
            )
        pack = _pack_path(service, upload_id)
        result = service.start(
            workflow_pack=pack,
            workflow_key=workflow_key,
            civitai_key=civitai_key,
            restore_config_after_success=restore_after_success,
        )
        return web.json_response({"ok": True, **result})
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] 설치 시작 거부: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=409)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 설치 시작 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)
    finally:
        workflow_key = ""
        civitai_key = ""


async def handle_cancel(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        return web.json_response({"ok": True, **service.cancel()})
    except InstallerServiceError as exc:
        return _json_error(str(exc), status=409)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 설치 중단 요청 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


async def handle_restore_config(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        body = await _read_json_object(request)
        backup_path = body.get("backup_path")
        if not isinstance(backup_path, str) or not backup_path:
            raise InstallerServiceError("복원할 backup_path가 비어 있습니다.")
        result = service.restore_backup(backup_path)
        return web.json_response({"ok": True, "restore": result})
    except InstallerServiceError as exc:
        return _json_error(str(exc), status=400)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] config.json 복원 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


def register_comfy_installer_routes(
    app: web.Application,
    *,
    project_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    requirements_dir: str | os.PathLike[str],
) -> ComfyInstallerService:
    service = ComfyInstallerService(
        project_root=project_root,
        config_path=config_path,
        requirements_dir=requirements_dir,
    )
    app[APP_SERVICE_KEY] = service
    app.router.add_get("/api/comfy-installer/status", handle_status)
    app.router.add_post("/api/comfy-installer/preflight", handle_preflight)
    app.router.add_post(
        "/api/comfy-installer/workflow-pack", handle_pack_upload
    )
    app.router.add_post("/api/comfy-installer/start", handle_start)
    app.router.add_post("/api/comfy-installer/cancel", handle_cancel)
    app.router.add_post(
        "/api/comfy-installer/restore-config", handle_restore_config
    )
    return service
