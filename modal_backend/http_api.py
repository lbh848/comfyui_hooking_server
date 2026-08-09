from __future__ import annotations

import traceback

from aiohttp import web

from .service import ModalService


def register_modal_routes(
    app: web.Application,
    *,
    project_root: str,
    get_config,
) -> ModalService:
    service = ModalService(project_root, get_config)

    async def status(_request: web.Request) -> web.Response:
        try:
            return web.json_response(await service.status())
        except Exception as exc:
            print(f"[MODAL_API] 상태 조회 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def connect(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            return web.json_response({"ok": True, "auth": await service.start_auth(body.get("profile", ""))})
        except (ValueError, RuntimeError) as exc:
            print(f"[MODAL_API] 계정 연결 요청 거부: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 계정 연결 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def workflows(_request: web.Request) -> web.Response:
        try:
            return web.json_response({"ok": True, "workflows": service.workflows()})
        except Exception as exc:
            print(f"[MODAL_API] 워크플로우 목록 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def billing(_request: web.Request) -> web.Response:
        try:
            return web.json_response(await service.billing())
        except RuntimeError as exc:
            print(f"[MODAL_API] 비용 조회 요청 실패: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 비용 조회 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def install(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            selected = body.get("workflow_ids") or []
            if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
                raise ValueError("workflow_ids는 문자열 배열이어야 합니다.")
            state = await service.start_install(selected)
            return web.json_response({"ok": True, "install": state})
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"[MODAL_API] 설치 요청 거부: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 설치 시작 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    app.router.add_get("/api/modal/status", status)
    app.router.add_post("/api/modal/connect", connect)
    app.router.add_get("/api/modal/workflows", workflows)
    app.router.add_get("/api/modal/billing", billing)
    app.router.add_post("/api/modal/install", install)
    return service
