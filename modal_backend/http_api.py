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

    async def worker_status(_request: web.Request) -> web.Response:
        try:
            return web.json_response(await service.worker_status())
        except Exception as exc:
            print(f"[MODAL_API] Worker 상태 조회 실패: {type(exc).__name__}: {exc}")
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
            payload = service.workflows()
            return web.json_response(
                {"ok": True, "workflows": payload["workflows"], "errors": payload["errors"]}
            )
        except Exception as exc:
            print(f"[MODAL_API] 워크플로우 목록 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def remote_workflows(_request: web.Request) -> web.Response:
        try:
            payload = await service.remote_workflows()
            return web.json_response({"ok": True, **payload})
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(
                "[MODAL_API] 원격 워크플로우 조회 요청 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(
                "[MODAL_API] 원격 워크플로우 조회 예외: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def loras(request: web.Request) -> web.Response:
        try:
            include_remote = request.query.get("remote", "").strip().lower() in {
                "1",
                "true",
            }
            item_keys = [
                value.strip()
                for value in request.query.getall("item_key", [])
                if value.strip()
            ]
            return web.json_response(
                await service.lora_catalog(
                    include_remote=include_remote,
                    item_keys=item_keys or None,
                )
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(
                "[MODAL_API] LoRA 카탈로그 요청 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(
                "[MODAL_API] LoRA 카탈로그 조회 예외: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def lora_operation(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            action = body.get("action", "")
            item_keys = body.get("item_keys") or []
            if not isinstance(action, str):
                raise ValueError("action은 문자열이어야 합니다.")
            if not isinstance(item_keys, list) or not all(
                isinstance(item, str) for item in item_keys
            ):
                raise ValueError("item_keys는 문자열 배열이어야 합니다.")
            operation = await service.start_lora_operation(action, item_keys)
            return web.json_response({"ok": True, "operation": operation})
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(
                "[MODAL_API] LoRA 작업 요청 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(
                "[MODAL_API] LoRA 작업 시작 예외: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def lora_operation_status(_request: web.Request) -> web.Response:
        try:
            return web.json_response(
                {"ok": True, "operation": service.lora_operation_status()}
            )
        except Exception as exc:
            print(
                "[MODAL_API] LoRA 작업 상태 조회 예외: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def custom_nodes(_request: web.Request) -> web.Response:
        try:
            return web.json_response(await service.custom_nodes())
        except Exception as exc:
            print(
                "[MODAL_API] custom node 인벤토리 조회 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def redeploy(_request: web.Request) -> web.Response:
        try:
            deployment = await service.start_redeploy(force_custom_nodes=False)
            return web.json_response({"ok": True, "deployment": deployment})
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"[MODAL_API] 재배포 요청 거부: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 재배포 시작 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def sync_custom_nodes(_request: web.Request) -> web.Response:
        try:
            deployment = await service.start_redeploy(force_custom_nodes=True)
            return web.json_response({"ok": True, "deployment": deployment})
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(
                "[MODAL_API] custom node 동기화 요청 거부: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(
                "[MODAL_API] custom node 동기화 시작 실패: "
                f"{type(exc).__name__}: {exc}"
            )
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
            print(f"[MODAL_API] 동기화 요청 거부: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 동기화 시작 실패: {type(exc).__name__}: {exc}")
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
            print(f"[MODAL_API] GPU 연결 테스트 요청 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] GPU 연결 테스트 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def web_url(_request: web.Request) -> web.Response:
        try:
            return web.json_response({"ok": True, "web": await service.web_url()})
        except Exception as exc:
            print(f"[MODAL_API] 웹 URL 조회 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def web_start(_request: web.Request) -> web.Response:
        try:
            return web.json_response({"ok": True, "web": await service.start_web()})
        except (ValueError, RuntimeError) as exc:
            print(f"[MODAL_API] 웹 App 시작 요청 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 웹 App 시작 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def web_stop(_request: web.Request) -> web.Response:
        try:
            return web.json_response({"ok": True, "web": await service.stop_web()})
        except (ValueError, RuntimeError) as exc:
            print(f"[MODAL_API] 웹 App 종료 요청 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 웹 App 종료 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def runtime_logs(request: web.Request) -> web.Response:
        try:
            raw_entries = request.query.get("entries", "500").strip()
            entries = int(raw_entries)
            if not 20 <= entries <= 1000:
                raise ValueError("entries는 20~1000 사이여야 합니다.")
            return web.json_response(await service.runtime_logs(entries=entries))
        except (ValueError, RuntimeError) as exc:
            print(f"[MODAL_API] 런타임 로그 요청 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(f"[MODAL_API] 런타임 로그 조회 예외: {type(exc).__name__}: {exc}")
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
    app.router.add_get("/api/modal/worker-status", worker_status)
    app.router.add_post("/api/modal/connect", connect)
    app.router.add_get("/api/modal/workflows", workflows)
    app.router.add_get("/api/modal/workflows/remote", remote_workflows)
    app.router.add_get("/api/modal/loras", loras)
    app.router.add_post("/api/modal/loras/operation", lora_operation)
    app.router.add_get("/api/modal/loras/operation", lora_operation_status)
    app.router.add_get("/api/modal/custom-nodes", custom_nodes)
    app.router.add_post("/api/modal/redeploy", redeploy)
    app.router.add_post("/api/modal/custom-nodes/sync", sync_custom_nodes)
    app.router.add_get("/api/modal/billing", billing)
    app.router.add_post("/api/modal/install", install)
    app.router.add_post("/api/modal/autoscaler", apply_autoscaler)
    app.router.add_post("/api/modal/probe", probe)
    app.router.add_get("/api/modal/web-url", web_url)
    app.router.add_post("/api/modal/web/start", web_start)
    app.router.add_post("/api/modal/web/stop", web_stop)
    app.router.add_get("/api/modal/runtime/logs", runtime_logs)
    app.router.add_post("/api/modal/workflow/run", run_workflow)
    app.router.add_get("/api/modal/workflow/runs/{job_id}", workflow_run_status)
    app.router.add_get(
        "/api/modal/workflow/runs/{job_id}/image",
        workflow_run_image,
    )
    return service
