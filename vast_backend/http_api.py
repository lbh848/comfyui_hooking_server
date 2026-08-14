"""/api/vast/* 라우트 등록 (Modal http_api 패턴 준수)."""
from __future__ import annotations

import asyncio
import traceback
from typing import Any

from aiohttp import web

from .model_sources import (
    defaults_from_manifest,
    load_mapping,
    normalize_source_key,
    save_mapping,
    validate_source,
)
from .service import VastService


def register_vast_routes(
    app: web.Application,
    *,
    project_root: str,
    get_config,
) -> VastService:
    service = VastService(project_root, get_config)

    def _fail(prefix: str, exc: Exception, status: int = 500) -> web.Response:
        print(f"[VAST_API] {prefix}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return web.json_response({"ok": False, "error": str(exc)}, status=status)

    async def status(_request: web.Request) -> web.Response:
        try:
            settings = service.settings()
            account = await service.account_status()
            return web.json_response(
                {
                    "ok": True,
                    "settings": settings.public_dict(),
                    "account": account,
                    "launch": service.launch_status(),
                }
            )
        except Exception as exc:
            return _fail("상태 조회 실패", exc)

    async def offers(request: web.Request) -> web.Response:
        try:
            q = request.query
            gpu_names = [
                v.strip() for v in q.get("gpu_names", "").split(",") if v.strip()
            ] or None

            def _opt_int(key):
                raw = q.get(key)
                if not raw:
                    return None
                return int(raw)

            def _opt_float(key):
                raw = q.get(key)
                if not raw:
                    return None
                return float(raw)

            limit_raw = q.get("limit")
            limit = int(limit_raw) if limit_raw else 60
            payload = await service.offers(
                gpu_names=gpu_names,
                min_disk_gb=int(q.get("min_disk_gb", "0") or 0),
                min_cpu_ram_gb=_opt_int("min_cpu_ram_gb"),
                max_price_usd_hr=_opt_float("max_price_usd_hr"),
                min_gpu_ram_gb=_opt_int("min_gpu_ram_gb"),
                inet_down_min_mbps=_opt_int("inet_down_min_mbps"),
                inet_up_min_mbps=_opt_float("inet_up_min_mbps"),
                min_direct_port_count=_opt_int("min_direct_port_count"),
                min_reliability=_opt_float("min_reliability"),
                min_cuda_version=_opt_float("min_cuda_version"),
                verified_only=(
                    q.get("verified_only") in {"1", "true"}
                    if q.get("verified_only")
                    else None
                ),
                on_demand=(
                    q.get("on_demand") in {"1", "true"}
                    if q.get("on_demand")
                    else None
                ),
                limit=limit,
            )
            return web.json_response(payload)
        except ValueError as exc:
            return _fail("오퍼 조회 요청 거부", exc, status=400)
        except Exception as exc:
            return _fail("오퍼 조회 실패", exc)

    async def plan(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            payload = service.wizard_plan(
                workflow_files=body.get("workflows") or [],
                lora_files=body.get("loras") or [],
            )
            return web.json_response(payload)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return _fail("준비 계획 요청 거부", exc, status=400)
        except Exception as exc:
            return _fail("준비 계획 실패", exc)

    async def launch(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            ask_id = int(body.get("ask_id") or 0)
            disk_gb = int(body.get("disk_gb") or 0)
            hourly_price_usd = float(body.get("hourly_price_usd") or 0.0)
            if ask_id <= 0:
                raise ValueError("ask_id(오퍼 ID)가 필요합니다.")
            if disk_gb < 10:
                raise ValueError("disk_gb는 10GB 이상이어야 합니다.")
            if not 0.0 <= hourly_price_usd <= 20.0:
                raise ValueError("hourly_price_usd는 0~20 사이여야 합니다.")
            install_payload = await asyncio.get_running_loop().run_in_executor(
                None, service.prepare_install_payload
            )
            state = await service.start_launch(
                ask_id=ask_id,
                disk_gb=disk_gb,
                model_plan=body.get("plan") or {},
                lora_files=body.get("loras") or [],
                install_payload=install_payload,
                hourly_price_usd=hourly_price_usd,
            )
            return web.json_response({"ok": True, "launch": state})
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return _fail("인스턴스 생성 요청 거부", exc, status=400)
        except Exception as exc:
            return _fail("인스턴스 생성 시작 실패", exc)

    async def launch_status(_request: web.Request) -> web.Response:
        try:
            return web.json_response({"ok": True, "launch": service.launch_status()})
        except Exception as exc:
            return _fail("생성 상태 조회 실패", exc)

    async def instances(_request: web.Request) -> web.Response:
        try:
            return web.json_response(await service.instances())
        except Exception as exc:
            return _fail("인스턴스 목록 조회 실패", exc)

    async def destroy(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            raw = body.get("instance_id")
            instance_id = int(raw) if raw else None
            return web.json_response(await service.destroy(instance_id))
        except (ValueError, RuntimeError) as exc:
            return _fail("인스턴스 파괴 요청 거부", exc, status=400)
        except Exception as exc:
            return _fail("인스턴스 파괴 실패", exc)

    async def model_sources_get(_request: web.Request) -> web.Response:
        try:
            return web.json_response(
                {
                    "ok": True,
                    "mapping": load_mapping(project_root),
                    "manifest_defaults": defaults_from_manifest(project_root),
                }
            )
        except Exception as exc:
            return _fail("모델 소스 조회 실패", exc)

    async def model_sources_put(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            entries = body.get("sources")
            if not isinstance(entries, dict):
                raise ValueError("sources는 {키: 소스} 객체여야 합니다.")
            current = load_mapping(project_root)
            for raw_key, raw_source in entries.items():
                key = normalize_source_key(*raw_key.split("/", 1))
                current["sources"][key] = validate_source(raw_source)
            path = save_mapping(project_root, current)
            return web.json_response({"ok": True, "path": str(path)})
        except (ValueError, RuntimeError, OSError) as exc:
            return _fail("모델 소스 저장 요청 거부", exc, status=400)
        except Exception as exc:
            return _fail("모델 소스 저장 실패", exc)

    async def run_workflow(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            workflow_api = body.get("workflow_api")
            if not isinstance(workflow_api, dict):
                raise ValueError("workflow_api(API 형식 워크플로우 객체)가 필요합니다.")
            return web.json_response(await service.run_workflow(workflow_api))
        except (ValueError, RuntimeError) as exc:
            return _fail("워크플로우 실행 요청 거부", exc, status=400)
        except Exception as exc:
            return _fail("워크플로우 실행 실패", exc)

    async def cleanup(_app: web.Application) -> None:
        try:
            await service.close()
            print("[VAST_API] 서비스와 SSH 터널 종료 완료")
        except Exception as exc:
            print(
                "[VAST_API] 서비스 종료 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    async def startup(_app: web.Application) -> None:
        try:
            await service.startup()
        except Exception as exc:
            print(
                "[VAST_API] 시작 시 비용 보호 복구 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    app.router.add_get("/api/vast/status", status)
    app.router.add_get("/api/vast/offers", offers)
    app.router.add_post("/api/vast/plan", plan)
    app.router.add_post("/api/vast/launch", launch)
    app.router.add_get("/api/vast/launch", launch_status)
    app.router.add_get("/api/vast/instances", instances)
    app.router.add_post("/api/vast/destroy", destroy)
    app.router.add_get("/api/vast/model-sources", model_sources_get)
    app.router.add_post("/api/vast/model-sources", model_sources_put)
    app.router.add_post("/api/vast/workflow/run", run_workflow)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return service
