from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import modal

from comfy_installer.credentials import load_civitai_key

from .manifest import selected_install_plan, workflow_catalog
from .settings import ModalSettings
from .workflow_assets import (
    resolve_explicit_input_files,
    resolve_input_files,
    resolve_lora_files,
)


L4_USD_PER_SECOND = 0.000222
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
RUNTIME_CPU_CORES = 4
RUNTIME_MEMORY_GIB = 16
BILLING_CACHE_SECONDS = 60


def cost_summary(settings: ModalSettings) -> dict[str, Any]:
    gpu_hour = L4_USD_PER_SECOND * 3600
    cpu_hour = CPU_USD_PER_CORE_SECOND * RUNTIME_CPU_CORES * 3600
    memory_hour = MEMORY_USD_PER_GIB_SECOND * RUNTIME_MEMORY_GIB * 3600
    container_hour = gpu_hour + cpu_hour + memory_hour
    return {
        "currency": "USD",
        "monthly_credit": settings.monthly_credit_usd,
        "l4_gpu_per_hour": round(gpu_hour, 4),
        "estimated_container_per_hour": round(container_hour, 4),
        "estimated_container_hours": round(settings.monthly_credit_usd / container_hour, 2),
        "estimated_wall_hours_at_max_concurrency": round(
            settings.monthly_credit_usd / container_hour / settings.max_concurrency,
            2,
        ),
        "assumptions": {
            "cpu_cores": RUNTIME_CPU_CORES,
            "memory_gib": RUNTIME_MEMORY_GIB,
            "min_containers": 0,
            "scaledown_window_seconds": settings.scaledown_window_seconds,
            "region_multiplier": 1.0,
        },
    }


class ModalService:
    def __init__(self, project_root: str | Path, get_config):
        self.project_root = Path(project_root).resolve()
        self.get_config = get_config
        self._auth_task: asyncio.Task | None = None
        self._auth_state: dict[str, Any] = {
            "state": "idle",
            "message": "Modal 계정 연결을 기다리고 있습니다.",
        }
        self._install_task: asyncio.Task | None = None
        self._install_state: dict[str, Any] = {
            "state": "idle",
            "message": "Modal 설치를 기다리고 있습니다.",
        }
        self._autoscaler_state: dict[str, Any] = {
            "state": "idle",
            "message": "저장된 자동 종료 설정이 다음 배포에 적용됩니다.",
        }
        self._probe_task: asyncio.Task | None = None
        self._probe_state: dict[str, Any] = {
            "state": "idle",
            "message": "L4 연결 테스트를 기다리고 있습니다.",
        }
        self._workflow_runs: dict[str, dict[str, Any]] = {}
        self._workflow_run_tasks: dict[str, asyncio.Task] = {}
        self._delete_outbox_path = self.project_root / "modal_lora_delete_outbox.json"
        self._delete_lock = asyncio.Lock()
        self._delete_flush_task: asyncio.Task | None = None
        self._billing_lock = asyncio.Lock()
        self._billing_cache: dict[str, Any] | None = None

    @staticmethod
    def _subprocess_env(profile: str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env["MODAL_PROFILE"] = profile
        if extra:
            env.update(extra)
        return env

    @staticmethod
    async def _run_command(
        args: list[str],
        *,
        env: Mapping[str, str],
        stdin_payload: dict | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        input_text = None
        if stdin_payload is not None:
            input_text = json.dumps(stdin_payload, ensure_ascii=False)

        def run_blocking() -> subprocess.CompletedProcess[str]:
            kwargs: dict[str, Any] = {}
            if os.name == "nt":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            return subprocess.run(
                args,
                input=input_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=dict(env),
                timeout=timeout,
                check=False,
                **kwargs,
            )

        try:
            completed = await asyncio.to_thread(run_blocking)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"명령 제한 시간을 초과했습니다: {args[0]} {args[1]}"
            ) from exc
        return (
            int(completed.returncode or 0),
            completed.stdout,
            completed.stderr,
        )

    async def account_connected(self, settings: ModalSettings) -> bool:
        try:
            code, _stdout, _stderr = await self._run_command(
                [sys.executable, "-m", "modal", "token", "info"],
                env=self._subprocess_env(settings.profile),
                timeout=20,
            )
            if code != 0:
                print(
                    f"[MODAL] 계정 상태 확인 실패: profile={settings.profile}, exit_code={code}"
                )
            return code == 0
        except Exception as exc:
            print(f"[MODAL] 계정 상태 확인 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return False

    async def _run_client_action(
        self,
        settings: ModalSettings,
        action: str,
        *,
        timeout: float,
        **payload: Any,
    ) -> dict[str, Any]:
        request_payload = {
            "action": action,
            "app_name": settings.deployment_name,
            "environment": settings.environment,
            **payload,
        }
        code, stdout, stderr = await self._run_command(
            [sys.executable, "-m", "modal_backend.client_cli"],
            env=self._subprocess_env(settings.profile),
            stdin_payload=request_payload,
            timeout=timeout,
        )
        try:
            response = json.loads(stdout) if stdout.strip() else {}
        except json.JSONDecodeError as exc:
            print(
                f"[MODAL] {action} 응답 JSON 파싱 실패: exit_code={code}, "
                f"stdout_length={len(stdout)}, stderr={stderr[-1000:]}"
            )
            traceback.print_exc()
            raise RuntimeError(f"Modal {action} 응답 형식이 올바르지 않습니다.") from exc
        if code != 0 or not response.get("ok"):
            error = str(response.get("error") or f"Modal client exit_code={code}")
            print(
                f"[MODAL] {action} 실패: app={settings.deployment_name}, "
                f"environment={settings.environment}, error={error}, stderr={stderr[-1000:]}"
            )
            raise RuntimeError(error)
        result = response.get("result")
        if not isinstance(result, dict):
            print(f"[MODAL] {action} 결과 객체 누락: type={type(result).__name__}")
            raise RuntimeError(f"Modal {action} 결과 객체가 없습니다.")
        return result

    async def status(self, *, include_runtime: bool = False) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        # 토큰(연결) 존재 여부는 modal_enabled와 무관하게 항상 조회한다.
        # 기능이 꺼져 있어도 "방금 인증한 토큰이 등록됐는지" 확인할 수 있어야 한다.
        connection_checked = True
        connected = await self.account_connected(settings)
        pending_deletes = await asyncio.to_thread(self._delete_outbox_count)
        if settings.enabled and connected and pending_deletes:
            self._schedule_delete_flush()
        runtime: dict[str, Any] | None = None
        if include_runtime:
            if not settings.enabled:
                runtime = {"available": False, "reason": "disabled"}
            elif not connected:
                print("[MODAL] 런타임 통계 조회 생략: Modal 계정이 연결되지 않았습니다.")
                runtime = {"available": False, "reason": "account_not_connected"}
            else:
                try:
                    stats = await self._run_client_action(
                        settings,
                        "runtime_stats",
                        timeout=30,
                    )
                    runtime = {"available": True, **stats}
                except Exception as exc:
                    print(f"[MODAL] 런타임 통계 조회 실패: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    runtime = {
                        "available": False,
                        "reason": "deployment_unavailable",
                        "error": str(exc),
                    }
        billing: dict[str, Any]
        if not settings.enabled:
            billing = {
                "available": False,
                "reason": "disabled",
                "cache_seconds": BILLING_CACHE_SECONDS,
            }
        elif not connected:
            print("[MODAL] 청구 자동 조회 생략: Modal 계정이 연결되지 않았습니다.")
            billing = {
                "available": False,
                "reason": "account_not_connected",
                "cache_seconds": BILLING_CACHE_SECONDS,
            }
        else:
            try:
                billing = {
                    "available": True,
                    **await self._billing_for_settings(settings),
                }
            except Exception as exc:
                print(f"[MODAL] 청구 자동 조회 실패: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                billing = {
                    "available": False,
                    "reason": "billing_unavailable",
                    "error": str(exc),
                    "cache_seconds": BILLING_CACHE_SECONDS,
                }
        return {
            "ok": True,
            "connected": connected,
            "connection_checked": connection_checked,
            "sdk_version": modal.__version__,
            "settings": settings.public_dict(),
            "auth": dict(self._auth_state),
            "install": dict(self._install_state),
            "autoscaler": dict(self._autoscaler_state),
            "probe": dict(self._probe_state),
            "cost": cost_summary(settings),
            "billing": billing,
            "pending_lora_deletes": pending_deletes,
            "runtime": runtime,
            "workflow_runs": self.recent_workflow_runs(),
        }

    async def start_auth(self, profile: str) -> dict[str, Any]:
        settings = ModalSettings.from_mapping({"modal_profile": profile})
        if self._auth_task and not self._auth_task.done():
            # 이미 브라우저 인증이 진행 중이면 에러 대신 현재 상태를 그대로 반환.
            # 멱등 처리: 프론트의 반복 클릭/재시도가 400 스팸 무한 루프를 만들지 않도록 한다.
            self._auth_state = {
                **self._auth_state,
                "state": "running",
                "message": self._auth_state.get("message") or "브라우저에서 Modal 로그인과 Workspace 선택을 완료하세요.",
                "profile": settings.profile,
            }
            return dict(self._auth_state)
        self._auth_state = {
            "state": "running",
            "message": "브라우저에서 Modal 로그인과 Workspace 선택을 완료하세요.",
            "profile": settings.profile,
        }
        self._auth_task = asyncio.create_task(self._run_auth(settings.profile))
        return dict(self._auth_state)

    async def _run_auth(self, profile: str) -> None:
        try:
            code, _stdout, _stderr = await self._run_command(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "token",
                    "new",
                    "--profile",
                    profile,
                    "--no-activate",
                    "--verify",
                ],
                env=self._subprocess_env(profile),
                timeout=600,
            )
            if code != 0:
                print(f"[MODAL] 브라우저 계정 연결 실패: profile={profile}, exit_code={code}")
                self._auth_state = {
                    "state": "failed",
                    "message": "Modal 계정 연결에 실패했습니다. 브라우저 인증을 다시 시도하세요.",
                    "profile": profile,
                }
                return
            print(f"[MODAL] 브라우저 계정 연결 완료: profile={profile}")
            self._auth_state = {
                "state": "completed",
                "message": "Modal 계정 연결이 완료되었습니다.",
                "profile": profile,
            }
        except Exception as exc:
            print(f"[MODAL] 브라우저 계정 연결 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._auth_state = {
                "state": "failed",
                "message": f"Modal 계정 연결 실패: {type(exc).__name__}: {exc}",
                "profile": profile,
            }

    def workflows(self) -> list[dict[str, Any]]:
        config = self.get_config()
        result: list[dict[str, Any]] = []
        for entry in workflow_catalog(self.project_root):
            item = dict(entry)
            try:
                plan = selected_install_plan(
                    self.project_root,
                    [str(entry["id"])],
                    config,
                )
                source = plan["workflow_files"][0]
                item.update(
                    configured=True,
                    source_name=Path(source["source_path"]).name,
                    binding=source["binding"],
                )
            except (ValueError, FileNotFoundError) as exc:
                print(
                    f"[MODAL] 워크플로우 실행 경로 미설정: "
                    f"workflow_id={entry.get('id')}, reason={exc}"
                )
                item.update(configured=False, source_name="", binding="")
            result.append(item)
        return result

    @staticmethod
    def _parse_billing_summary(raw_summary: Any) -> dict[str, Any]:
        try:
            if not isinstance(raw_summary, dict):
                raise TypeError(
                    f"청구 요약이 객체가 아닙니다: {type(raw_summary).__name__}"
                )

            def decimal_value(value: Any, field: str) -> Decimal:
                try:
                    parsed = Decimal(str(value))
                except (InvalidOperation, ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{field} 값이 유효한 금액이 아닙니다: {value!r}"
                    ) from exc
                if not parsed.is_finite():
                    raise ValueError(f"{field} 값이 유한한 금액이 아닙니다: {value!r}")
                return parsed

            metered_cost = decimal_value(raw_summary["metered_cost"], "metered_cost")
            billed_cost = decimal_value(raw_summary["billed_cost"], "billed_cost")
            raw_adjustments = raw_summary.get("adjustments", {})
            raw_breakdown = raw_summary.get("metered_cost_breakdown", {})
            if not isinstance(raw_adjustments, dict):
                raise TypeError("adjustments가 객체가 아닙니다.")
            if not isinstance(raw_breakdown, dict):
                raise TypeError("metered_cost_breakdown이 객체가 아닙니다.")
            adjustments = {
                str(key): decimal_value(value, f"adjustments.{key}")
                for key, value in raw_adjustments.items()
            }
            breakdown = {
                str(key): decimal_value(value, f"metered_cost_breakdown.{key}")
                for key, value in raw_breakdown.items()
            }
            return {
                "metered_cost": format(metered_cost, "f"),
                "billed_cost": format(billed_cost, "f"),
                "adjustments": {
                    key: format(value, "f") for key, value in adjustments.items()
                },
                "adjustment_total": format(sum(adjustments.values(), Decimal("0")), "f"),
                "metered_cost_breakdown": {
                    key: format(value, "f") for key, value in breakdown.items()
                },
            }
        except Exception as exc:
            print(f"[MODAL] 청구 요약 검증 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise RuntimeError("Modal 비용 응답 형식이 올바르지 않습니다.") from exc

    @staticmethod
    def _billing_result(
        cache: dict[str, Any],
        settings: ModalSettings,
        *,
        cached: bool,
        cache_age_seconds: float,
    ) -> dict[str, Any]:
        summary = dict(cache["summary"])
        summary["adjustments"] = dict(summary["adjustments"])
        summary["metered_cost_breakdown"] = dict(
            summary["metered_cost_breakdown"]
        )
        configured_credit = Decimal(str(settings.monthly_credit_usd))
        metered_cost = Decimal(summary["metered_cost"])
        remaining_credit = max(Decimal("0"), configured_credit - metered_cost)
        summary.update(
            configured_credit=format(configured_credit, "f"),
            remaining_credit_estimate=format(remaining_credit, "f"),
            fetched_at=cache["fetched_at"],
        )
        return {
            "ok": True,
            "summary": summary,
            "cached": cached,
            "cache_age_seconds": round(max(0.0, cache_age_seconds), 1),
            "cache_seconds": BILLING_CACHE_SECONDS,
        }

    async def _billing_for_settings(
        self,
        settings: ModalSettings,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        async with self._billing_lock:
            now = time.monotonic()
            cache = self._billing_cache
            cache_age = (
                now - float(cache["stored_at_monotonic"])
                if cache is not None
                else float("inf")
            )
            if (
                not force_refresh
                and cache is not None
                and cache.get("profile") == settings.profile
                and cache_age < BILLING_CACHE_SECONDS
            ):
                return self._billing_result(
                    cache,
                    settings,
                    cached=True,
                    cache_age_seconds=cache_age,
                )

            code, stdout, stderr = await self._run_command(
                [sys.executable, "-m", "modal", "billing", "summary", "--json"],
                env=self._subprocess_env(settings.profile),
                timeout=30,
            )
            if code != 0:
                print(
                    f"[MODAL] 비용 조회 실패: profile={settings.profile}, "
                    f"exit_code={code}, stderr={stderr[-1000:]}"
                )
                raise RuntimeError("Modal 비용 정보를 조회하지 못했습니다.")
            try:
                raw_summary = json.loads(stdout)
            except json.JSONDecodeError as exc:
                print(f"[MODAL] 비용 응답 JSON 파싱 실패: {exc}")
                traceback.print_exc()
                raise RuntimeError("Modal 비용 응답 형식이 올바르지 않습니다.") from exc

            normalized = self._parse_billing_summary(raw_summary)
            fetched_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._billing_cache = {
                "profile": settings.profile,
                "summary": normalized,
                "fetched_at": fetched_at,
                "stored_at_monotonic": time.monotonic(),
            }
            return self._billing_result(
                self._billing_cache,
                settings,
                cached=False,
                cache_age_seconds=0.0,
            )

    async def billing(self, *, force_refresh: bool = False) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print(
                f"[MODAL] 비용 조회 생략: Modal이 비활성화되어 있습니다. "
                f"profile={settings.profile}"
            )
            raise RuntimeError("외부 API 설정에서 Modal 사용을 먼저 켜고 저장하세요.")
        if not await self.account_connected(settings):
            print(
                f"[MODAL] 비용 조회 생략: 계정이 연결되지 않았습니다. "
                f"profile={settings.profile}"
            )
            raise RuntimeError("Modal 계정을 먼저 연결하세요.")
        return await self._billing_for_settings(
            settings,
            force_refresh=force_refresh,
        )

    async def start_install(self, selected_ids: list[str]) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            raise RuntimeError("외부 API 설정에서 Modal 사용을 먼저 켜고 저장하세요.")
        if not await self.account_connected(settings):
            raise RuntimeError("Modal 계정을 먼저 연결하세요.")
        if self._install_task and not self._install_task.done():
            raise RuntimeError("Modal 설치 또는 업데이트가 이미 진행 중입니다.")
        plan = selected_install_plan(self.project_root, selected_ids, self.get_config())
        self._install_state = {
            "state": "running",
            "phase": "deploy",
            "message": "Modal ComfyUI 런타임을 배포하고 있습니다.",
            "workflow_ids": plan["workflow_ids"],
            "size_gib": plan["size_gib"],
        }
        self._install_task = asyncio.create_task(self._run_install(settings, plan))
        return dict(self._install_state)

    async def _run_install(self, settings: ModalSettings, plan: dict[str, Any]) -> None:
        try:
            app_path = self.project_root / "modal_backend" / "modal_app.py"
            deploy_env = self._subprocess_env(
                settings.profile,
                {
                    "SOYA_MODAL_APP_NAME": settings.deployment_name,
                    "SOYA_MODAL_MAX_CONTAINERS": str(settings.max_concurrency),
                    "SOYA_MODAL_SCALEDOWN_WINDOW": str(
                        settings.scaledown_window_seconds
                    ),
                },
            )
            code, _stdout, _stderr = await self._run_command(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "deploy",
                    str(app_path),
                    "--env",
                    settings.environment,
                ],
                env=deploy_env,
                timeout=3600,
            )
            if code != 0:
                print(
                    f"[MODAL] 앱 배포 실패: app={settings.deployment_name}, "
                    f"env={settings.environment}, exit_code={code}"
                )
                raise RuntimeError("Modal 앱 배포에 실패했습니다. 서버 로그를 확인하세요.")

            self._install_state.update(
                phase="models",
                message="워크플로우를 업로드하고 필요한 모델을 설치하고 있습니다.",
            )
            client_payload = {
                "action": "install",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "workflow_files": plan["workflow_files"],
                "model_ids": plan["model_ids"],
                "civitai_key": load_civitai_key(self.project_root),
            }
            code, stdout, _stderr = await self._run_command(
                [sys.executable, "-m", "modal_backend.client_cli"],
                env=self._subprocess_env(settings.profile),
                stdin_payload=client_payload,
                timeout=86_400,
            )
            if code != 0:
                print(
                    f"[MODAL] 워크플로우/모델 설치 실패: app={settings.deployment_name}, "
                    f"exit_code={code}"
                )
                raise RuntimeError("Modal 워크플로우 또는 모델 설치에 실패했습니다.")
            response = json.loads(stdout)
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "Modal 원격 설치 실패"))
            print(
                f"[MODAL] 설치 완료: app={settings.deployment_name}, "
                f"workflows={len(plan['workflow_ids'])}, models={plan['model_count']}, "
                f"size_gib={plan['size_gib']}"
            )
            self._install_state = {
                "state": "completed",
                "phase": "complete",
                "message": "Modal 워크플로우와 모델 설치가 완료되었습니다.",
                "workflow_ids": plan["workflow_ids"],
                "model_count": plan["model_count"],
                "size_gib": plan["size_gib"],
            }
        except Exception as exc:
            print(f"[MODAL] 설치 작업 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._install_state = {
                "state": "failed",
                "phase": self._install_state.get("phase", "unknown"),
                "message": f"Modal 설치 실패: {type(exc).__name__}: {exc}",
                "workflow_ids": plan.get("workflow_ids", []),
            }

    async def apply_autoscaler(self) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print("[MODAL] Autoscaler 즉시 적용 생략: Modal이 비활성화되어 있습니다.")
            self._autoscaler_state = {
                "state": "waiting",
                "message": "Modal을 켜면 저장된 자동 종료 설정을 적용합니다.",
            }
            return dict(self._autoscaler_state)
        if not await self.account_connected(settings):
            print("[MODAL] Autoscaler 즉시 적용 생략: Modal 계정이 연결되지 않았습니다.")
            self._autoscaler_state = {
                "state": "waiting",
                "message": "계정 연결 후 설치하거나 설정을 다시 저장하면 적용됩니다.",
            }
            return dict(self._autoscaler_state)
        self._autoscaler_state = {
            "state": "running",
            "message": "배포된 Modal autoscaler에 설정을 적용하고 있습니다.",
        }
        try:
            result = await self._run_client_action(
                settings,
                "update_autoscaler",
                timeout=60,
                max_containers=settings.max_concurrency,
                scaledown_window_seconds=settings.scaledown_window_seconds,
            )
            self._autoscaler_state = {
                "state": "completed",
                "message": (
                    f"최대 {settings.max_concurrency}개 · 유휴 "
                    f"{settings.scaledown_window_seconds}초로 적용되었습니다."
                ),
                **result,
            }
        except Exception as exc:
            print(f"[MODAL] Autoscaler 적용 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._autoscaler_state = {
                "state": "failed",
                "message": f"Autoscaler 적용 실패: {type(exc).__name__}: {exc}",
            }
        return dict(self._autoscaler_state)

    async def start_probe(self) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            raise RuntimeError("Modal 사용을 켜고 설정을 저장하세요.")
        if not await self.account_connected(settings):
            raise RuntimeError("Modal 계정을 먼저 연결하세요.")
        if self._probe_task and not self._probe_task.done():
            raise RuntimeError("L4 연결 테스트가 이미 진행 중입니다.")
        self._probe_state = {
            "state": "running",
            "message": "L4 컨테이너를 깨우고 CUDA를 확인하고 있습니다.",
        }
        self._probe_task = asyncio.create_task(self._run_probe(settings))
        return dict(self._probe_state)

    async def _run_probe(self, settings: ModalSettings) -> None:
        try:
            result = await self._run_client_action(
                settings,
                "gpu_probe",
                timeout=960,
            )
            vram_gib = round(int(result.get("vram_bytes") or 0) / 1024**3, 1)
            self._probe_state = {
                "state": "completed",
                "message": (
                    f"{result.get('device') or 'L4'} · VRAM {vram_gib} GiB · "
                    f"CUDA {result.get('cuda') or '-'} 연결 확인"
                ),
                **result,
            }
            print(f"[MODAL] L4 연결 테스트 완료: {self._probe_state['message']}")
        except Exception as exc:
            print(f"[MODAL] L4 연결 테스트 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._probe_state = {
                "state": "failed",
                "message": f"L4 연결 테스트 실패: {type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _is_api_workflow(workflow: Mapping[str, Any]) -> bool:
        if "nodes" in workflow and "links" in workflow:
            return False
        return any(
            isinstance(node, Mapping) and "class_type" in node
            for node in workflow.values()
        )

    async def convert_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print("[MODAL] 워크플로우 변환 실패: Modal이 비활성화되어 있습니다.")
            raise RuntimeError("Modal 원격 생성이 비활성화되어 있습니다.")
        if not await self.account_connected(settings):
            print("[MODAL] 워크플로우 변환 실패: Modal 계정이 연결되어 있지 않습니다.")
            raise RuntimeError("Modal 계정이 연결되어 있지 않습니다.")
        converted = await self._run_client_action(
            settings,
            "convert_workflow",
            timeout=960,
            workflow=workflow,
            timeout_seconds=900,
        )
        if not isinstance(converted, dict) or not self._is_api_workflow(converted):
            print(
                "[MODAL] 워크플로우 변환 결과 검증 실패: "
                f"type={type(converted).__name__}"
            )
            raise RuntimeError("Modal ComfyUI 워크플로우 변환 결과가 올바르지 않습니다.")
        return converted

    def _workflow_run_public(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in state.items()
            if key not in {"image_bytes"}
        }

    def recent_workflow_runs(self) -> list[dict[str, Any]]:
        runs = sorted(
            self._workflow_runs.values(),
            key=lambda item: str(item.get("created_at") or ""),
            reverse=True,
        )
        return [self._workflow_run_public(item) for item in runs[:20]]

    async def start_workflow_run(self, workflow_id: str) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            raise RuntimeError("Modal 사용을 켜고 설정을 저장하세요.")
        if not await self.account_connected(settings):
            raise RuntimeError("Modal 계정을 먼저 연결하세요.")
        normalized_id = str(workflow_id or "").strip()
        plan = selected_install_plan(
            self.project_root,
            [normalized_id],
            self.get_config(),
        )
        active_count = sum(
            1
            for state in self._workflow_runs.values()
            if state.get("state") in {"queued", "running"}
        )
        if active_count >= settings.max_concurrency:
            raise RuntimeError(
                f"Modal 워크플로우가 이미 {active_count}개 실행 중입니다. "
                "완료 후 다시 시도하세요."
            )
        job_id = uuid.uuid4().hex
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state = {
            "job_id": job_id,
            "workflow_id": normalized_id,
            "source_name": Path(plan["workflow_files"][0]["source_path"]).name,
            "state": "queued",
            "phase": "queued",
            "message": "Modal 워크플로우 실행을 준비하고 있습니다.",
            "created_at": now,
            "result_available": False,
        }
        self._workflow_runs[job_id] = state
        while len(self._workflow_runs) > 20:
            oldest_id = min(
                self._workflow_runs,
                key=lambda key: str(self._workflow_runs[key].get("created_at") or ""),
            )
            if self._workflow_runs[oldest_id].get("state") in {"queued", "running"}:
                break
            self._workflow_runs.pop(oldest_id, None)
            self._workflow_run_tasks.pop(oldest_id, None)
        task = asyncio.create_task(self._run_saved_workflow(settings, plan, state))
        self._workflow_run_tasks[job_id] = task
        return self._workflow_run_public(state)

    async def _run_saved_workflow(
        self,
        settings: ModalSettings,
        plan: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        job_id = str(state["job_id"])
        try:
            source_path = Path(plan["workflow_files"][0]["source_path"])
            state.update(
                state="running",
                phase="loading",
                message="로컬 워크플로우 JSON을 읽고 있습니다.",
            )
            workflow = await asyncio.to_thread(
                lambda: json.loads(source_path.read_text(encoding="utf-8"))
            )
            if not isinstance(workflow, dict) or not workflow:
                raise ValueError("워크플로우 JSON 객체가 비어 있습니다.")
            if not self._is_api_workflow(workflow):
                state.update(
                    phase="converting",
                    message="원격 ComfyUI에서 워크플로우를 API 형식으로 변환하고 있습니다.",
                )
                workflow = await self._run_client_action(
                    settings,
                    "convert_workflow",
                    timeout=960,
                    workflow=workflow,
                    timeout_seconds=900,
                )
            state.update(
                phase="generating",
                message="LoRA와 입력 이미지를 동기화하고 L4에서 실행하고 있습니다.",
            )
            image_bytes, metadata = await self.generate(workflow)
            state.update(
                state="completed",
                phase="completed",
                message="Modal 워크플로우 실행이 완료되었습니다.",
                completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                prompt_id=metadata.get("prompt_id"),
                content_type=metadata.get("content_type") or "image/png",
                lora_sync=metadata.get("lora_sync") or {},
                result_available=True,
                image_bytes=image_bytes,
            )
            print(
                f"[MODAL] 관리 탭 워크플로우 완료: job_id={job_id}, "
                f"workflow_id={state.get('workflow_id')}, bytes={len(image_bytes)}"
            )
        except Exception as exc:
            print(
                f"[MODAL] 관리 탭 워크플로우 실패: job_id={job_id}, "
                f"workflow_id={state.get('workflow_id')}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            state.update(
                state="failed",
                phase="failed",
                message=f"Modal 워크플로우 실패: {type(exc).__name__}: {exc}",
                completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                result_available=False,
            )

    def workflow_run_status(self, job_id: str) -> dict[str, Any]:
        state = self._workflow_runs.get(str(job_id))
        if state is None:
            print(f"[MODAL] 워크플로우 실행 상태 없음: job_id={job_id}")
            raise KeyError("Modal 워크플로우 실행 기록을 찾을 수 없습니다.")
        return self._workflow_run_public(state)

    def workflow_run_image(self, job_id: str) -> tuple[bytes, str]:
        state = self._workflow_runs.get(str(job_id))
        if state is None:
            print(f"[MODAL] 워크플로우 결과 없음: job_id={job_id}")
            raise KeyError("Modal 워크플로우 실행 기록을 찾을 수 없습니다.")
        image_bytes = state.get("image_bytes")
        if not isinstance(image_bytes, bytes) or not image_bytes:
            print(
                f"[MODAL] 워크플로우 결과 이미지 미준비: job_id={job_id}, "
                f"state={state.get('state')}"
            )
            raise RuntimeError("Modal 워크플로우 결과 이미지가 아직 준비되지 않았습니다.")
        content_type = str(state.get("content_type") or "image/png").split(";", 1)[0].strip()
        if "/" not in content_type:
            print(
                f"[MODAL] 워크플로우 결과 Content-Type 보정: "
                f"job_id={job_id}, value={content_type!r}"
            )
            content_type = "application/octet-stream"
        return image_bytes, content_type

    @staticmethod
    def _merge_input_files(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for group in groups:
            for item in group:
                remote_name = str(item.get("remote_name") or "").replace("\\", "/")
                if not remote_name:
                    print(f"[MODAL_SYNC] remote_name 없는 입력 파일 거부: {item!r}")
                    raise ValueError("Modal 입력 파일의 원격 이름이 비어 있습니다.")
                merged[remote_name] = item
        return list(merged.values())

    def _store_modal_artifacts(
        self,
        artifacts: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """원격 LoRA 결과를 기존 파일을 덮어쓰지 않는 방식으로 로컬에 복귀시킨다."""

        if not artifacts:
            return []
        local_root_raw = str(config.get("lora_load_path") or "").strip()
        if not local_root_raw:
            print("[MODAL_SYNC] LoRA 결과 저장 실패: lora_load_path 설정이 비어 있습니다.")
            raise ValueError("Modal LoRA 결과를 저장할 로컬 LoRA 경로가 비어 있습니다.")
        local_root = Path(local_root_raw).resolve()
        local_root.mkdir(parents=True, exist_ok=True)
        stored: list[dict[str, Any]] = []
        for item in artifacts:
            source = Path(str(item.get("path") or ""))
            relative = Path(str(item.get("relative_path") or ""))
            if not source.is_file():
                print(f"[MODAL_SYNC] LoRA 결과 임시 파일 없음: {source}")
                raise FileNotFoundError(f"Modal LoRA 결과 임시 파일이 없습니다: {source}")
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                print(f"[MODAL_SYNC] 안전하지 않은 LoRA 결과 상대 경로: {relative!s}")
                raise ValueError(f"안전하지 않은 Modal LoRA 결과 경로입니다: {relative!s}")
            target = local_root.joinpath(*relative.parts).resolve()
            if local_root != target and local_root not in target.parents:
                print(
                    "[MODAL_SYNC] LoRA 결과 저장 경로 거부: 로컬 LoRA 루트 밖입니다. "
                    f"root={local_root}, target={target}"
                )
                raise ValueError(f"로컬 LoRA 폴더 밖에는 결과를 저장할 수 없습니다: {target}")

            final_target = target
            status = "stored"
            if target.exists():
                if not target.is_file():
                    print(
                        "[MODAL_SYNC] LoRA 결과 경로 충돌: 기존 대상이 파일이 아님. "
                        f"target={target}, type={'directory' if target.is_dir() else 'other'}"
                    )
                    raise IsADirectoryError(
                        f"Modal LoRA 결과 대상이 파일이 아닙니다: {target}"
                    )
                source_hash = self._sha256_file(source)
                target_hash = self._sha256_file(target)
                if source_hash == target_hash:
                    print(f"[MODAL_SYNC] 동일한 로컬 LoRA 결과 저장 생략: {target}")
                    stored.append(
                        {
                            "relative_path": relative.as_posix(),
                            "local_path": str(target),
                            "status": "identical",
                        }
                    )
                    continue
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                final_target = target.with_name(
                    f"{target.stem}.modal-{stamp}{target.suffix}"
                )
                status = "conflict_copy"
                print(
                    "[MODAL_SYNC] 기존 로컬 LoRA 보존, 충돌 사본으로 저장: "
                    f"existing={target}, new={final_target}"
                )
            final_target.parent.mkdir(parents=True, exist_ok=True)
            temp_target = final_target.with_name(
                f".{final_target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            )
            try:
                shutil.copy2(source, temp_target)
                os.replace(temp_target, final_target)
            except Exception as exc:
                temp_target.unlink(missing_ok=True)
                print(
                    "[MODAL_SYNC] LoRA 결과 로컬 저장 실패: "
                    f"source={source}, target={final_target}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise
            stored.append(
                {
                    "relative_path": relative.as_posix(),
                    "local_path": str(final_target),
                    "status": status,
                }
            )
        return stored

    @staticmethod
    def _sha256_file(path: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    async def run_workflow(
        self,
        workflow: dict[str, Any],
        *,
        timeout_seconds: int = 3_300,
        input_paths: list[str] | tuple[str, ...] | None = None,
        artifact_prefixes: list[str] | tuple[str, ...] | None = None,
        require_images: bool = True,
    ) -> dict[str, Any]:
        config = self.get_config()
        settings = ModalSettings.from_mapping(config)
        if not settings.enabled:
            print("[MODAL] 원격 워크플로우 실행 실패: Modal이 비활성화되어 있습니다.")
            raise RuntimeError("Modal 원격 생성이 비활성화되어 있습니다.")
        if not await self.account_connected(settings):
            print(
                "[MODAL] 원격 워크플로우 실행 실패: "
                f"Modal 계정이 연결되지 않았습니다. profile={settings.profile}"
            )
            raise RuntimeError("Modal 계정이 연결되어 있지 않습니다.")
        lora_files, workflow_input_files, explicit_input_files = await asyncio.gather(
            asyncio.to_thread(resolve_lora_files, workflow, config),
            asyncio.to_thread(resolve_input_files, workflow, config),
            asyncio.to_thread(
                resolve_explicit_input_files,
                input_paths or [],
                config,
            ) if input_paths else asyncio.sleep(0, result=[]),
        )
        input_files = self._merge_input_files(
            workflow_input_files,
            explicit_input_files,
        )
        with tempfile.TemporaryDirectory(prefix="soya-modal-output-") as output_dir:
            payload = {
                "action": "generate",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "workflow": workflow,
                "lora_files": lora_files,
                "input_files": input_files,
                "artifact_prefixes": list(artifact_prefixes or []),
                "require_images": bool(require_images),
                "timeout_seconds": max(30, min(int(timeout_seconds), 3_300)),
                "output_dir": output_dir,
            }
            code, stdout, stderr = await self._run_command(
                [sys.executable, "-m", "modal_backend.client_cli"],
                env=self._subprocess_env(settings.profile),
                stdin_payload=payload,
                timeout=payload["timeout_seconds"] + 180,
            )
            if code != 0:
                print(
                    f"[MODAL] 원격 생성 실패: app={settings.deployment_name}, "
                    f"exit_code={code}, loras={len(lora_files)}, inputs={len(input_files)}, "
                    f"stderr={stderr[-2000:]}"
                )
                raise RuntimeError("Modal 원격 이미지 생성에 실패했습니다. 서버 로그를 확인하세요.")
            try:
                response = json.loads(stdout)
            except json.JSONDecodeError as exc:
                print(f"[MODAL] 원격 생성 응답 파싱 실패: {exc}")
                traceback.print_exc()
                raise RuntimeError("Modal 원격 생성 응답 형식이 올바르지 않습니다.") from exc
            if not response.get("ok"):
                error = str(response.get("error") or "Modal 원격 생성 실패")
                print(
                    "[MODAL] 원격 워크플로우 응답 실패: "
                    f"app={settings.deployment_name}, error={error}, "
                    f"stderr={stderr[-2000:]}"
                )
                raise RuntimeError(error)
            result = response["result"]
            outputs = result.get("outputs") or []
            if require_images and not outputs:
                print(
                    "[MODAL] 원격 워크플로우 이미지 결과 없음: "
                    f"prompt_id={result.get('prompt_id')}, result_keys={list(result)}"
                )
                raise RuntimeError("Modal 원격 생성 결과 이미지가 없습니다.")
            images: list[dict[str, Any]] = []
            for output in outputs:
                output_path = Path(str(output.get("path") or ""))
                if not output_path.is_file():
                    print(
                        "[MODAL] 원격 결과 임시 파일 없음: "
                        f"prompt_id={result.get('prompt_id')}, path={output_path}"
                    )
                    raise FileNotFoundError(f"Modal 결과 임시 파일이 없습니다: {output_path}")
                images.append(
                    {
                        "bytes": output_path.read_bytes(),
                        "filename": output.get("filename"),
                        "content_type": output.get("content_type"),
                        "node_id": output.get("node_id"),
                    }
                )
            stored_artifacts = self._store_modal_artifacts(
                list(result.get("artifacts") or []),
                config,
            )
            print(
                f"[MODAL] 원격 워크플로우 완료: app={settings.deployment_name}, "
                f"prompt_id={result.get('prompt_id')}, images={len(images)}, "
                f"artifacts={len(stored_artifacts)}, "
                f"lora_sync={result.get('lora_sync')}"
            )
            return {
                "prompt_id": result.get("prompt_id"),
                "lora_sync": result.get("lora_sync") or {},
                "images": images,
                "artifacts": stored_artifacts,
                "text_outputs": list(result.get("text_outputs") or []),
            }

    async def generate(
        self,
        workflow: dict[str, Any],
        *,
        timeout_seconds: int = 3_300,
        input_paths: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        result = await self.run_workflow(
            workflow,
            timeout_seconds=timeout_seconds,
            input_paths=input_paths,
            require_images=True,
        )
        images = result.get("images") or []
        if not images:
            print("[MODAL] 원격 이미지 생성 결과가 비어 있습니다.")
            raise RuntimeError("Modal 원격 이미지 생성 결과가 없습니다.")
        first = images[0]
        return first["bytes"], {
            "prompt_id": result.get("prompt_id"),
            "lora_sync": result.get("lora_sync") or {},
            "content_type": first.get("content_type"),
        }

    def _load_delete_outbox(self) -> list[dict[str, Any]]:
        if not self._delete_outbox_path.is_file():
            return []
        try:
            data = json.loads(self._delete_outbox_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("삭제 outbox 루트는 배열이어야 합니다.")
            return [item for item in data if isinstance(item, dict)]
        except Exception as exc:
            print(f"[MODAL_SYNC] 삭제 outbox 읽기 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return []

    def _delete_outbox_count(self) -> int:
        return len(self._load_delete_outbox())

    def _save_delete_outbox(self, items: list[dict[str, Any]]) -> None:
        target = self._delete_outbox_path
        if target.exists():
            backup_root = self.project_root / "backups" / "modal"
            backup_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = backup_root / f"modal_lora_delete_outbox_before_save_{stamp}.json"
            shutil.copy2(target, backup)
            print(f"[MODAL_SYNC] 삭제 outbox 백업: {backup}")
        temp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            temp_path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    async def enqueue_lora_delete(self, remote_prefix: str) -> None:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print(
                f"[MODAL_SYNC] Modal이 비활성화되어 원격 LoRA 삭제 예약 생략: "
                f"{remote_prefix}"
            )
            return
        normalized = str(remote_prefix or "").strip().replace("\\", "/").strip("/")
        parts = normalized.split("/") if normalized else []
        if not parts or any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"안전하지 않은 Modal LoRA 삭제 경로입니다: {remote_prefix!r}")
        async with self._delete_lock:
            items = await asyncio.to_thread(self._load_delete_outbox)
            if not any(item.get("remote_prefix") == normalized for item in items):
                items.append(
                    {
                        "remote_prefix": normalized,
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "attempts": 0,
                    }
                )
                await asyncio.to_thread(self._save_delete_outbox, items)
                print(f"[MODAL_SYNC] 원격 LoRA 삭제 예약: {normalized}")
        self._schedule_delete_flush()

    def _schedule_delete_flush(self) -> None:
        if self._delete_flush_task and not self._delete_flush_task.done():
            return
        self._delete_flush_task = asyncio.create_task(self._flush_delete_outbox())

    async def _flush_delete_outbox(self) -> None:
        config = self.get_config()
        settings = ModalSettings.from_mapping(config)
        if not settings.enabled:
            print("[MODAL_SYNC] Modal이 비활성화되어 LoRA 삭제 outbox 전송을 보류합니다.")
            return
        if not await self.account_connected(settings):
            print("[MODAL_SYNC] Modal 계정이 연결되지 않아 LoRA 삭제 outbox 전송을 보류합니다.")
            return
        while True:
            async with self._delete_lock:
                items = await asyncio.to_thread(self._load_delete_outbox)
                if not items:
                    return
                item = dict(items[0])
            payload = {
                "action": "delete_lora_prefix",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "remote_prefix": item["remote_prefix"],
            }
            try:
                code, stdout, _stderr = await self._run_command(
                    [sys.executable, "-m", "modal_backend.client_cli"],
                    env=self._subprocess_env(settings.profile),
                    stdin_payload=payload,
                    timeout=120,
                )
                response = json.loads(stdout) if stdout.strip() else {}
                if code != 0 or not response.get("ok"):
                    raise RuntimeError(
                        str(response.get("error") or f"Modal client exit_code={code}")
                    )
                async with self._delete_lock:
                    current = await asyncio.to_thread(self._load_delete_outbox)
                    current = [
                        queued
                        for queued in current
                        if queued.get("remote_prefix") != item["remote_prefix"]
                    ]
                    await asyncio.to_thread(self._save_delete_outbox, current)
                print(f"[MODAL_SYNC] 원격 LoRA 삭제 완료: {item['remote_prefix']}")
            except Exception as exc:
                print(
                    f"[MODAL_SYNC] 원격 LoRA 삭제 실패, outbox 유지: "
                    f"path={item.get('remote_prefix')}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                async with self._delete_lock:
                    current = await asyncio.to_thread(self._load_delete_outbox)
                    for queued in current:
                        if queued.get("remote_prefix") == item.get("remote_prefix"):
                            queued["attempts"] = int(queued.get("attempts") or 0) + 1
                            queued["last_error"] = f"{type(exc).__name__}: {exc}"
                    await asyncio.to_thread(self._save_delete_outbox, current)
                return
