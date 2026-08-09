from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping

import modal

from comfy_installer.credentials import load_civitai_key

from .manifest import selected_install_plan, workflow_catalog
from .settings import ModalSettings
from .workflow_assets import resolve_input_files, resolve_lora_files


L4_USD_PER_SECOND = 0.000222
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
RUNTIME_CPU_CORES = 4
RUNTIME_MEMORY_GIB = 16


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
            "scaledown_window_seconds": 15,
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
        self._delete_outbox_path = self.project_root / "modal_lora_delete_outbox.json"
        self._delete_lock = asyncio.Lock()
        self._delete_flush_task: asyncio.Task | None = None

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

    async def status(self) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        connected = await self.account_connected(settings)
        pending_deletes = await asyncio.to_thread(self._delete_outbox_count)
        if settings.enabled and connected and pending_deletes:
            self._schedule_delete_flush()
        return {
            "ok": True,
            "connected": connected,
            "sdk_version": modal.__version__,
            "settings": settings.public_dict(),
            "auth": dict(self._auth_state),
            "install": dict(self._install_state),
            "cost": cost_summary(settings),
            "pending_lora_deletes": pending_deletes,
        }

    async def start_auth(self, profile: str) -> dict[str, Any]:
        settings = ModalSettings.from_mapping({"modal_profile": profile})
        if self._auth_task and not self._auth_task.done():
            raise RuntimeError("Modal 계정 연결이 이미 진행 중입니다.")
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
        return workflow_catalog(self.project_root)

    async def billing(self) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not await self.account_connected(settings):
            raise RuntimeError("Modal 계정을 먼저 연결하세요.")
        code, stdout, _stderr = await self._run_command(
            [sys.executable, "-m", "modal", "billing", "summary", "--json"],
            env=self._subprocess_env(settings.profile),
            timeout=30,
        )
        if code != 0:
            print(f"[MODAL] 비용 조회 실패: profile={settings.profile}, exit_code={code}")
            raise RuntimeError("Modal 비용 정보를 조회하지 못했습니다.")
        try:
            return {"ok": True, "summary": json.loads(stdout)}
        except json.JSONDecodeError as exc:
            print(f"[MODAL] 비용 응답 JSON 파싱 실패: {exc}")
            traceback.print_exc()
            raise RuntimeError("Modal 비용 응답 형식이 올바르지 않습니다.") from exc

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

    async def generate(
        self,
        workflow: dict[str, Any],
        *,
        timeout_seconds: int = 3_300,
    ) -> tuple[bytes, dict[str, Any]]:
        config = self.get_config()
        settings = ModalSettings.from_mapping(config)
        if not settings.enabled:
            raise RuntimeError("Modal 원격 생성이 비활성화되어 있습니다.")
        if not await self.account_connected(settings):
            raise RuntimeError("Modal 계정이 연결되어 있지 않습니다.")
        lora_files, input_files = await asyncio.gather(
            asyncio.to_thread(resolve_lora_files, workflow, config),
            asyncio.to_thread(resolve_input_files, workflow, config),
        )
        with tempfile.TemporaryDirectory(prefix="soya-modal-output-") as output_dir:
            payload = {
                "action": "generate",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "workflow": workflow,
                "lora_files": lora_files,
                "input_files": input_files,
                "timeout_seconds": max(30, min(int(timeout_seconds), 3_300)),
                "output_dir": output_dir,
            }
            code, stdout, _stderr = await self._run_command(
                [sys.executable, "-m", "modal_backend.client_cli"],
                env=self._subprocess_env(settings.profile),
                stdin_payload=payload,
                timeout=payload["timeout_seconds"] + 180,
            )
            if code != 0:
                print(
                    f"[MODAL] 원격 생성 실패: app={settings.deployment_name}, "
                    f"exit_code={code}, loras={len(lora_files)}, inputs={len(input_files)}"
                )
                raise RuntimeError("Modal 원격 이미지 생성에 실패했습니다. 서버 로그를 확인하세요.")
            try:
                response = json.loads(stdout)
            except json.JSONDecodeError as exc:
                print(f"[MODAL] 원격 생성 응답 파싱 실패: {exc}")
                traceback.print_exc()
                raise RuntimeError("Modal 원격 생성 응답 형식이 올바르지 않습니다.") from exc
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "Modal 원격 생성 실패"))
            result = response["result"]
            outputs = result.get("outputs") or []
            if not outputs:
                raise RuntimeError("Modal 원격 생성 결과 이미지가 없습니다.")
            output_path = Path(outputs[0]["path"])
            if not output_path.is_file():
                raise FileNotFoundError(f"Modal 결과 임시 파일이 없습니다: {output_path}")
            image_bytes = output_path.read_bytes()
            print(
                f"[MODAL] 원격 생성 완료: app={settings.deployment_name}, "
                f"prompt_id={result.get('prompt_id')}, bytes={len(image_bytes)}, "
                f"lora_sync={result.get('lora_sync')}"
            )
            return image_bytes, {
                "prompt_id": result.get("prompt_id"),
                "lora_sync": result.get("lora_sync") or {},
                "content_type": outputs[0].get("content_type"),
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
            backup_root = self.project_root / "요구사항"
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
