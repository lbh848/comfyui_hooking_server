from __future__ import annotations

import asyncio
import json
import os
import re
import traceback
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aiohttp import web

from .crypto import PACK_MAGIC
from .service import ComfyInstallerService, InstallerServiceError


MAX_PACK_BYTES = 32 * 1024 * 1024
_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
APP_SERVICE_KEY = web.AppKey(
    "comfy_installer_service", ComfyInstallerService
)
ShutdownAfterUpdateCallback = Callable[[], Awaitable[dict[str, Any]]]
PauseManagedComfyCallback = Callable[[], Any]
ResumeManagedComfyCallback = Callable[[Any], Any]


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
        body = await _read_json_object(request) if request.can_read_body else {}
        release_version = body.get("release_version")
        selected_item_ids = body.get("selected_item_ids")
        install_mode = body.get("install_mode", "standard")
        if not isinstance(install_mode, str):
            raise InstallerServiceError("install_mode는 문자열이어야 합니다.")
        if release_version is None and selected_item_ids is None:
            result = await asyncio.to_thread(
                service.preflight,
                require_disk=False,
                install_mode=install_mode,
            )
        else:
            if not isinstance(release_version, str) or not isinstance(
                selected_item_ids, list
            ) or not all(isinstance(value, str) for value in selected_item_ids):
                raise InstallerServiceError(
                    "선택 검사는 release_version 문자열과 "
                    "selected_item_ids 문자열 배열이 필요합니다."
                )
            result = await asyncio.to_thread(
                service.preflight_selection,
                release_version=release_version,
                selected_item_ids=selected_item_ids,
                install_mode=install_mode,
            )
        return web.json_response({"ok": True, "preflight": result})
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 사전 검사 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=400)


async def handle_workflow_library(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        result = await asyncio.to_thread(service.workflow_library_status)
        return web.json_response({"ok": True, "library": result})
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 워크플로우 라이브러리 조회 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


async def handle_e2e_catalog(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        result = await asyncio.to_thread(service.e2e_workflow_catalog)
        return web.json_response({"ok": True, "catalog": result})
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] E2E 목록 조회 거부: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=409)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] E2E 목록 조회 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


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
    try:
        body = await _read_json_object(request)
        release_version = body.get("release_version")
        selected_item_ids = body.get("selected_item_ids")
        install_mode = body.get("install_mode", "standard")
        if not isinstance(release_version, str):
            raise InstallerServiceError("release_version은 문자열이어야 합니다.")
        if not isinstance(selected_item_ids, list) or not all(
            isinstance(value, str) for value in selected_item_ids
        ):
            raise InstallerServiceError("selected_item_ids는 문자열 배열이어야 합니다.")
        if not isinstance(install_mode, str):
            raise InstallerServiceError("install_mode는 문자열이어야 합니다.")
        result = service.start_install(
            release_version=release_version,
            selected_item_ids=selected_item_ids,
            install_mode=install_mode,
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


async def handle_e2e_start(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        body = await _read_json_object(request)
        release_version = body.get("release_version")
        selected_item_ids = body.get("selected_item_ids")
        if not isinstance(release_version, str):
            raise InstallerServiceError("release_version은 문자열이어야 합니다.")
        if not isinstance(selected_item_ids, list) or not all(
            isinstance(value, str) for value in selected_item_ids
        ):
            raise InstallerServiceError(
                "selected_item_ids는 문자열 배열이어야 합니다."
            )
        result = service.start_e2e(
            release_version=release_version,
            selected_item_ids=selected_item_ids,
        )
        return web.json_response({"ok": True, **result})
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] E2E 시작 거부: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=409)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] E2E 시작 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


async def handle_unpack_workflow_pack(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    workflow_key = ""
    try:
        body = await _read_json_object(request)
        upload_id = str(body.get("upload_id", ""))
        workflow_key = str(body.get("workflow_key", ""))
        pack = _pack_path(service, upload_id)
        result = await asyncio.to_thread(
            service.unpack_workflow_pack,
            workflow_pack=pack,
            workflow_key=workflow_key,
        )
        return web.json_response({"ok": True, **result})
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] 워크플로우 팩 풀기 거부: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=409)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 워크플로우 팩 풀기 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=400)
    finally:
        workflow_key = ""


async def handle_civitai_key(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        if request.method == "POST":
            body = await _read_json_object(request)
            api_key = body.get("api_key", "")
            if not isinstance(api_key, str):
                raise InstallerServiceError("api_key는 문자열이어야 합니다.")
            result = await asyncio.to_thread(service.set_civitai_key, api_key)
            return web.json_response({"ok": True, **result})
        api_key = await asyncio.to_thread(service.get_civitai_key)
        return web.json_response({"ok": True, "api_key": api_key})
    except InstallerServiceError as exc:
        return _json_error(str(exc), status=400)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] Civitai 키 처리 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


async def handle_replace_lora_manager_civitai_key(
    request: web.Request,
) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        body = await _read_json_object(request)
        api_key = body.get("api_key", "")
        if not isinstance(api_key, str):
            raise InstallerServiceError("api_key는 문자열이어야 합니다.")
        result = await asyncio.to_thread(
            service.replace_lora_manager_civitai_key,
            api_key,
        )
        return web.json_response({"ok": True, **result})
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] LoRA Manager Civitai 키 교체 거부: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=400)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] LoRA Manager Civitai 키 교체 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


async def handle_update(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        result = service.start_update()
        return web.json_response({"ok": True, **result})
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] 업데이트 시작 거부: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=409)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 업데이트 시작 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


async def handle_migrate(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        body = await _read_json_object(request)
        old_comfy_root = body.get("old_comfy_root")
        if not isinstance(old_comfy_root, str) or not old_comfy_root.strip():
            raise InstallerServiceError("기존 ComfyUI 경로가 비어 있습니다.")
        result = service.start_migration(old_comfy_root)
        return web.json_response({"ok": True, **result})
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] 사용자 데이터 이사 시작 거부: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=409)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] 사용자 데이터 이사 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


async def handle_retarget_config(request: web.Request) -> web.Response:
    service = request.app[APP_SERVICE_KEY]
    try:
        result = await asyncio.to_thread(service.retarget_config_to_embedded)
        return web.json_response({"ok": True, **result})
    except InstallerServiceError as exc:
        print(f"[COMFY_INSTALL][API] config.json 경로 전환 거부: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=409)
    except Exception as exc:
        print(f"[COMFY_INSTALL][API] config.json 경로 전환 실패: {exc}")
        traceback.print_exc()
        return _json_error(str(exc), status=500)


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
    authorize_shutdown: Callable[[web.Request], bool] | None = None,
    shutdown_after_update: ShutdownAfterUpdateCallback | None = None,
    pause_managed_comfy: PauseManagedComfyCallback | None = None,
    resume_managed_comfy: ResumeManagedComfyCallback | None = None,
) -> ComfyInstallerService:
    service = ComfyInstallerService(
        project_root=project_root,
        config_path=config_path,
        requirements_dir=requirements_dir,
        pause_managed_comfy=pause_managed_comfy,
        resume_managed_comfy=resume_managed_comfy,
    )
    app[APP_SERVICE_KEY] = service
    shutdown_requested = False

    async def handle_shutdown_after_update(
        request: web.Request,
    ) -> web.Response:
        nonlocal shutdown_requested
        if authorize_shutdown is not None:
            try:
                authorized = bool(authorize_shutdown(request))
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][API] 업데이트 후 종료 인증 확인 실패: "
                    f"remote={request.remote}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                return _json_error(
                    "업데이트 후 종료 인증을 확인하지 못했습니다.", status=500
                )
            if not authorized:
                print(
                    "[COMFY_INSTALL][API] 인증되지 않은 업데이트 후 종료 요청 거부: "
                    f"remote={request.remote}"
                )
                return _json_error("대시보드 로그인이 필요합니다.", status=401)

        status = service.status()
        if status.get("state") != "succeeded" or status.get("operation") != "update":
            return _json_error(
                "재시작이 필요한 업데이트 성공 후에만 종료할 수 있습니다.",
                status=409,
            )
        result = status.get("result")
        if not isinstance(result, dict) or result.get("restart_required") is not True:
            return _json_error(
                "현재 업데이트는 매니저 재시작이 필요하지 않습니다.",
                status=409,
            )
        if shutdown_after_update is None:
            return _json_error(
                "업데이트 후 종료 기능이 연결되지 않았습니다.", status=503
            )
        if shutdown_requested:
            return _json_error("업데이트 후 종료가 이미 요청되었습니다.", status=409)

        shutdown_requested = True
        try:
            shutdown_result = await shutdown_after_update()
            return web.json_response(
                {
                    "ok": True,
                    "message": (
                        "업데이트가 완료되었습니다. "
                        "매니저를 재시작해주세요."
                    ),
                    "shutdown": shutdown_result,
                }
            )
        except Exception as exc:
            shutdown_requested = False
            print(
                "[COMFY_INSTALL][API] 업데이트 후 종료 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return _json_error(str(exc), status=500)

    app.router.add_get("/api/comfy-installer/status", handle_status)
    app.router.add_post("/api/comfy-installer/preflight", handle_preflight)
    app.router.add_get(
        "/api/comfy-installer/workflow-library", handle_workflow_library
    )
    app.router.add_get(
        "/api/comfy-installer/e2e-catalog", handle_e2e_catalog
    )
    app.router.add_post(
        "/api/comfy-installer/workflow-pack", handle_pack_upload
    )
    app.router.add_post("/api/comfy-installer/start", handle_start)
    app.router.add_post("/api/comfy-installer/e2e", handle_e2e_start)
    app.router.add_post(
        "/api/comfy-installer/unpack-workflow-pack",
        handle_unpack_workflow_pack,
    )
    app.router.add_get("/api/comfy-installer/civitai-key", handle_civitai_key)
    app.router.add_post("/api/comfy-installer/civitai-key", handle_civitai_key)
    app.router.add_post(
        "/api/comfy-installer/troubleshooting/civitai-key",
        handle_replace_lora_manager_civitai_key,
    )
    app.router.add_post("/api/comfy-installer/update", handle_update)
    app.router.add_post(
        "/api/comfy-installer/shutdown-after-update",
        handle_shutdown_after_update,
    )
    app.router.add_post("/api/comfy-installer/migrate", handle_migrate)
    app.router.add_post(
        "/api/comfy-installer/retarget-config", handle_retarget_config
    )
    app.router.add_post("/api/comfy-installer/cancel", handle_cancel)
    app.router.add_post(
        "/api/comfy-installer/restore-config", handle_restore_config
    )
    return service
