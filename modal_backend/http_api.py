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

    async def status(request: web.Request) -> web.Response:
        try:
            include_runtime = request.query.get("runtime", "").strip() in {"1", "true"}
            return web.json_response(
                await service.status(include_runtime=include_runtime)
            )
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

    async def billing(request: web.Request) -> web.Response:
        try:
            force_refresh = request.query.get("refresh", "").strip() in {"1", "true"}
            return web.json_response(
                await service.billing(force_refresh=force_refresh)
            )
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

    async def apply_autoscaler(_request: web.Request) -> web.Response:
        try:
            return web.json_response(
                {"ok": True, "autoscaler": await service.apply_autoscaler()}
            )
        except (ValueError, RuntimeError) as exc:
            print(f"[MODAL_API] Autoscaler 적용 요청 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] Autoscaler 적용 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def probe(_request: web.Request) -> web.Response:
        try:
            return web.json_response(
                {"ok": True, "probe": await service.start_probe()}
            )
        except (ValueError, RuntimeError) as exc:
            print(f"[MODAL_API] L4 연결 테스트 요청 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] L4 연결 테스트 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def run_workflow(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            workflow_id = body.get("workflow_id", "")
            if not isinstance(workflow_id, str) or not workflow_id.strip():
                raise ValueError("workflow_id는 비어 있지 않은 문자열이어야 합니다.")
            run = await service.start_workflow_run(workflow_id)
            return web.json_response({"ok": True, "run": run})
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"[MODAL_API] 워크플로우 실행 요청 거부: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 워크플로우 실행 시작 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def workflow_run_status(request: web.Request) -> web.Response:
        try:
            run = service.workflow_run_status(request.match_info["job_id"])
            return web.json_response({"ok": True, "run": run})
        except KeyError as exc:
            print(f"[MODAL_API] 워크플로우 실행 상태 조회 실패: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            print(f"[MODAL_API] 워크플로우 실행 상태 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def workflow_run_image(request: web.Request) -> web.Response:
        try:
            image_bytes, content_type = service.workflow_run_image(
                request.match_info["job_id"]
            )
            return web.Response(body=image_bytes, content_type=content_type)
        except KeyError as exc:
            print(f"[MODAL_API] 워크플로우 결과 이미지 조회 실패: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except RuntimeError as exc:
            print(f"[MODAL_API] 워크플로우 결과 이미지 미준비: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=409)
        except Exception as exc:
            print(f"[MODAL_API] 워크플로우 결과 이미지 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    app.router.add_get("/api/modal/status", status)
    app.router.add_post("/api/modal/connect", connect)
    app.router.add_get("/api/modal/workflows", workflows)
    app.router.add_get("/api/modal/billing", billing)
    app.router.add_post("/api/modal/install", install)
    app.router.add_post("/api/modal/autoscaler", apply_autoscaler)
    app.router.add_post("/api/modal/probe", probe)
    app.router.add_post("/api/modal/workflow/run", run_workflow)
    app.router.add_get("/api/modal/workflow/runs/{job_id}", workflow_run_status)
    app.router.add_get(
        "/api/modal/workflow/runs/{job_id}/image",
        workflow_run_image,
    )
    return service
