from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping

import modal

from runtime_temp import runtime_temp_root

from .custom_nodes import (
    deploy_custom_nodes_json,
    inventory_custom_nodes,
    public_custom_node_inventory,
)
from .manifest import (
    list_soya_user_workflows,
    model_ids_for_workflow_files,
    plan_from_soya_user_names,
)
from .lora_inventory import (
    build_local_lora_catalog,
    merge_remote_lora_catalog,
    public_lora_catalog,
)
from .settings import (
    MODAL_GPU_PROFILES,
    MODEL_SOURCE_CLOUD_DIRECT,
    ModalSettings,
)
from .workflow_assets import (
    build_local_model_index,
    resolve_explicit_input_files,
    resolve_input_files,
    resolve_workflow_model_files,
)


CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
RUNTIME_CPU_CORES = 4
RUNTIME_MEMORY_GIB = 16
BILLING_CACHE_SECONDS = 60
# WebUI control-plane 조회(get_url/AppList) 캐시 주기. 브라우저는 5초 폴링하지만
# WebUI 상태가 초 단위로 움직일 필요는 없으므로 Modal 왕복·rate-limit 부담을 절반
# 이하로 줄인다(10~15초 범위).
WEB_REMOTE_CACHE_SECONDS = 12
INSTALL_LOG_LIMIT = 180
INSTALL_LOG_LINE_LIMIT = 1_200
RUNTIME_LOG_LIMIT = 500
WEB_APP_SUFFIX = "-web"
INSTALL_PROGRESS_PREFIX = "@@SOYA_MODAL_PROGRESS@@"
WORKFLOW_PROGRESS_PREFIX = "@@SOYA_MODAL_WORKFLOW_PROGRESS@@"
DOWNLOAD_PROGRESS_PREFIX = "@@SOYA_MODAL_DOWNLOAD_PROGRESS@@"
INSTALL_PHASE_LABELS = {
    "assets": "자산 분석",
    "upload": "파일 업로드",
    "complete": "완료",
}
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUNTIME_FAILURE_REASONS = frozenset(
    {"app_not_deployed", "network_unavailable", "runtime_unavailable"}
)


class ModalClientActionError(RuntimeError):
    def __init__(self, message: str, *, reason: str, error_type: str) -> None:
        super().__init__(message)
        self.reason = (
            reason if reason in RUNTIME_FAILURE_REASONS else "runtime_unavailable"
        )
        self.error_type = error_type


class WebStartCancelled(RuntimeError):
    """사용자가 Modal 웹 Server 준비 대기를 취소했다."""


class _ModalSubprocessLoop:
    """Modal 자식 프로세스 I/O를 전담하는 상주 asyncio 루프.

    Windows에서 서버·pytest가 SelectorEventLoop를 사용하면 asyncio subprocess를
    직접 만들 수 없다. 전용 ProactorEventLoop를 daemon 스레드에 한 번만 띄우고
    모든 Modal 명령을 그 루프에 제출하면, 명령 수만큼 기본 executor 스레드를
    점유하지 않으면서 여러 자식 프로세스를 동시에 기다릴 수 있다.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._startup_error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = (
                asyncio.ProactorEventLoop()
                if os.name == "nt"
                else asyncio.new_event_loop()
            )
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            print(
                "[MODAL] 비동기 subprocess 전용 루프 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            self._ready.set()
        finally:
            if loop is not None and not loop.is_closed():
                loop.close()

    def submit(self, coroutine):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._ready.clear()
                self._startup_error = None
                self._loop = None
                self._thread = threading.Thread(
                    target=self._run,
                    name="modal-async-subprocess-loop",
                    daemon=True,
                )
                self._thread.start()
            if not self._ready.wait(timeout=10):
                coroutine.close()
                print("[MODAL] 비동기 subprocess 전용 루프 시작 시간 초과")
                raise RuntimeError("Modal 명령 실행 루프를 시작하지 못했습니다.")
            if self._startup_error is not None or self._loop is None:
                coroutine.close()
                error = self._startup_error
                print(
                    "[MODAL] 비동기 subprocess 전용 루프를 사용할 수 없습니다: "
                    f"error={type(error).__name__ if error else 'unknown'}: {error}"
                )
                raise RuntimeError("Modal 명령 실행 루프가 준비되지 않았습니다.") from error
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop)


_MODAL_SUBPROCESS_LOOP = _ModalSubprocessLoop()


def _runtime_failure_reason(exc: Exception) -> str:
    if isinstance(exc, ModalClientActionError):
        return exc.reason
    if isinstance(exc, TimeoutError):
        return "network_unavailable"
    return "runtime_unavailable"


def cost_summary(settings: ModalSettings) -> dict[str, Any]:
    cpu_hour = CPU_USD_PER_CORE_SECOND * RUNTIME_CPU_CORES * 3600
    memory_hour = MEMORY_USD_PER_GIB_SECOND * RUNTIME_MEMORY_GIB * 3600
    support_hour = cpu_hour + memory_hour

    def gpu_cost(gpu_id: str) -> dict[str, Any]:
        profile = MODAL_GPU_PROFILES[gpu_id]
        gpu_hour = float(profile["usd_per_second"]) * 3600
        container_hour = gpu_hour + support_hour
        return {
            "gpu": gpu_id,
            "label": str(profile["label"]),
            "vram_gib": int(profile["vram_gib"]),
            "gpu_per_hour": round(gpu_hour, 4),
            "cpu_memory_per_hour": round(support_hour, 4),
            "container_per_hour": round(container_hour, 4),
        }

    worker = gpu_cost(settings.worker_gpu)
    web = gpu_cost(settings.web_gpu)
    worker_container_hour = float(worker["container_per_hour"])
    return {
        "currency": "USD",
        "monthly_credit": settings.monthly_credit_usd,
        "worker": worker,
        "web": web,
        "combined_container_per_hour": round(
            float(worker["container_per_hour"]) + float(web["container_per_hour"]),
            4,
        ),
        # 기존 비용 UI가 작업 워커 기준 필드를 소비할 수 있도록 일반화된 alias를 둔다.
        "gpu_per_hour": worker["gpu_per_hour"],
        "estimated_container_per_hour": worker["container_per_hour"],
        "estimated_container_hours": round(
            settings.monthly_credit_usd / worker_container_hour,
            2,
        ),
        "estimated_wall_hours_at_max_concurrency": round(
            settings.monthly_credit_usd
            / worker_container_hour
            / settings.max_concurrency,
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
        # 관리 화면의 CMD는 현재 서버 프로세스에서 일어난 일만 보여준다.
        # Modal 원격 로그는 배포 수명 전체를 반환하므로 이 기준 시각 이전 기록은
        # runtime_logs()에서 제거한다.
        self._runtime_log_session_started_at = time.time()
        self._auth_task: asyncio.Task | None = None
        self._auth_state: dict[str, Any] = {
            "state": "idle",
            "message": "Modal 계정 연결을 기다리고 있습니다.",
        }
        self._install_task: asyncio.Task | None = None
        self._install_state: dict[str, Any] = {
            "state": "idle",
            "message": "Modal 동기화를 기다리고 있습니다.",
            "logs": [],
        }
        self._lora_operation_task: asyncio.Task | None = None
        self._lora_operation_state: dict[str, Any] = {
            "state": "idle",
            "action": "",
            "message": "LoRA 관리 작업을 기다리고 있습니다.",
            "logs": [],
        }
        self._autoscaler_state: dict[str, Any] = {
            "state": "idle",
            "message": "설정 저장 시 선택한 동적 GPU 워커에 자동 종료 설정을 적용합니다.",
        }
        self._probe_task: asyncio.Task | None = None
        self._probe_state: dict[str, Any] = {
            "state": "idle",
            "message": "작업 워커 GPU 연결 테스트를 기다리고 있습니다.",
            "updated_at": time.time(),
        }
        self._web_task: asyncio.Task | None = None
        self._web_start_cancel_event = threading.Event()
        self._web_state: dict[str, Any] = {
            "state": "stopped",
            "message": "Modal ComfyUI 웹 GPU가 꺼져 있습니다.",
            "updated_at": time.time(),
        }
        self._deployment_task: asyncio.Task | None = None
        self._deployment_state: dict[str, Any] = {
            "state": "idle",
            "kind": "",
            "message": "Modal 재배포 요청을 기다리고 있습니다.",
            "logs": [],
        }
        self._runtime_log_cache: list[dict[str, Any]] = []
        self._workflow_runs: dict[str, dict[str, Any]] = {}
        self._workflow_run_tasks: dict[str, asyncio.Task] = {}
        self._delete_outbox_path = self.project_root / "modal_lora_delete_outbox.json"
        self._delete_lock = asyncio.Lock()
        self._delete_flush_task: asyncio.Task | None = None
        self._video_delete_outbox_path = (
            self.project_root / "modal_video_delete_outbox.json"
        )
        self._video_delete_lock = asyncio.Lock()
        self._video_delete_flush_task: asyncio.Task | None = None
        self._billing_lock = asyncio.Lock()
        self._billing_cache: dict[str, Any] | None = None
        self._model_sync_lock = asyncio.Lock()
        self._model_hash_cache: dict[str, tuple[int, int, str]] = {}
        self._deploy_lock = asyncio.Lock()
        self._deployment_action_lock = asyncio.Lock()
        # 웹 시작/종료와 App 재배포의 승인 구간을 하나의 락으로 묶어,
        # 상태 조회 await 사이에 서로 다른 작업이 동시에 등록되지 않게 한다.
        self._web_action_lock = self._deployment_action_lock
        # 좌측 위젯용 WebUI control-plane 결과(get_url/AppList) 캐시.
        # _billing_cache 패턴과 동일: stored_at_monotonic 기반 TTL.
        self._web_remote_lock = asyncio.Lock()
        self._web_remote_cache: dict[str, Any] | None = None
        # 상태 위젯과 설정 화면이 동시에 갱신되어도 같은 Modal control-plane
        # 명령을 중복 실행하지 않는다. 서로 다른 상태 명령은 최대 2개까지 별도
        # 비동기 subprocess로 실행해 생성 작업과 메인 이벤트 루프를 점유하지 않는다.
        self._status_command_semaphore = asyncio.Semaphore(2)
        self._status_action_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._account_check_tasks: dict[str, asyncio.Task[bool]] = {}
        # 삽화 LLM 처리와 Modal ComfyWorker 기동을 겹치기 위한 임시 warm pool.
        # 여러 삽화 요청이 겹쳐도 마지막 lease가 끝날 때만 scale-to-zero로 복구한다.
        self._worker_warm_lease_lock = asyncio.Lock()
        self._worker_warm_leases: dict[str, str] = {}
        self._worker_warm_pool_applied_min = 0
        self._worker_warm_reset_task: asyncio.Task | None = None

    @staticmethod
    def _subprocess_env(profile: str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env["MODAL_PROFILE"] = profile
        # Windows 호스트의 기본 Python 출력 인코딩이 CP949여도 Modal CLI가
        # ✓ 같은 Unicode 상태 문자를 stdout/stderr에 안전하게 출력해야 한다.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if extra:
            env.update(extra)
        return env

    @staticmethod
    def _web_app_name(settings: ModalSettings) -> str:
        return f"{settings.deployment_name}{WEB_APP_SUFFIX}"

    def _modal_deploy_env(
        self,
        settings: ModalSettings,
        *,
        custom_node_inventory: Mapping[str, Any] | None = None,
        force_custom_node_build: bool = False,
    ) -> dict[str, str]:
        return self._subprocess_env(
            settings.profile,
            {
                "SOYA_MODAL_APP_NAME": settings.deployment_name,
                "SOYA_MODAL_WEB_APP_NAME": self._web_app_name(settings),
                "SOYA_MODAL_WORKER_GPU": settings.worker_gpu,
                "SOYA_MODAL_WEB_GPU": settings.web_gpu,
                "SOYA_MODAL_VRAM_MODE": settings.vram_mode,
                "SOYA_MODAL_MAX_CONTAINERS": str(settings.max_concurrency),
                "SOYA_MODAL_SCALEDOWN_WINDOW": str(
                    settings.scaledown_window_seconds
                ),
                "SOYA_MODAL_WEB_FAST": "1" if settings.web_fast else "0",
                "SOYA_MODAL_EXTRA_CUSTOM_NODES": deploy_custom_nodes_json(
                    custom_node_inventory or {}
                ),
                "SOYA_MODAL_FORCE_CUSTOM_NODE_BUILD": (
                    "1" if force_custom_node_build else "0"
                ),
            },
        )

    @staticmethod
    async def _run_command(
        args: list[str],
        *,
        env: Mapping[str, str],
        cwd: str | Path | None = None,
        stdin_payload: dict | None = None,
        timeout: float | None = None,
        output_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[int, str, str]:
        caller_loop = asyncio.get_running_loop()
        output_queue: asyncio.Queue[tuple[str, str] | None] | None = (
            asyncio.Queue() if output_callback is not None else None
        )

        def forward_output(source: str, line: str) -> None:
            if output_queue is None:
                return
            try:
                caller_loop.call_soon_threadsafe(
                    output_queue.put_nowait,
                    (source, line),
                )
            except Exception as exc:
                print(
                    "[MODAL] 명령 출력의 메인 루프 전달 실패: "
                    f"source={source}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        async def execute_on_subprocess_loop() -> tuple[int, str, str]:
            try:
                return await ModalService._run_command_async(
                    args,
                    env=env,
                    cwd=cwd,
                    stdin_payload=stdin_payload,
                    timeout=timeout,
                    output_callback=(forward_output if output_queue is not None else None),
                )
            finally:
                if output_queue is not None:
                    try:
                        caller_loop.call_soon_threadsafe(output_queue.put_nowait, None)
                    except Exception as exc:
                        print(
                            "[MODAL] 명령 출력 종료 신호 전달 실패: "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()

        async def consume_output() -> None:
            if output_queue is None or output_callback is None:
                return
            while True:
                entry = await output_queue.get()
                if entry is None:
                    return
                source, line = entry
                try:
                    output_callback(source, line)
                except Exception as callback_exc:
                    print(
                        "[MODAL] 실시간 명령 출력 처리 실패: "
                        f"source={source}, "
                        f"error={type(callback_exc).__name__}: {callback_exc}"
                    )
                    traceback.print_exc()

        concurrent_future = _MODAL_SUBPROCESS_LOOP.submit(
            execute_on_subprocess_loop()
        )
        wrapped_future = asyncio.wrap_future(concurrent_future)
        consumer_task = (
            asyncio.create_task(consume_output())
            if output_queue is not None
            else None
        )
        try:
            result = await wrapped_future
            if consumer_task is not None:
                await consumer_task
            return result
        except asyncio.CancelledError:
            concurrent_future.cancel()
            if consumer_task is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(consumer_task),
                        timeout=10,
                    )
                except Exception as exc:
                    print(
                        "[MODAL] 취소된 명령 출력 소비 정리 실패: "
                        f"command={args!r}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    consumer_task.cancel()
            raise
        except Exception:
            if consumer_task is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(consumer_task),
                        timeout=10,
                    )
                except Exception as exc:
                    print(
                        "[MODAL] 실패한 명령 출력 소비 정리 실패: "
                        f"command={args!r}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    consumer_task.cancel()
            raise

    @staticmethod
    async def _run_command_async(
        args: list[str],
        *,
        env: Mapping[str, str],
        cwd: str | Path | None = None,
        stdin_payload: dict | None = None,
        timeout: float | None = None,
        output_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[int, str, str]:
        if not args:
            print("[MODAL] 실행할 명령 인자가 비어 있습니다.")
            raise ValueError("Modal 명령 인자가 비어 있습니다.")

        input_bytes = (
            json.dumps(stdin_payload, ensure_ascii=False).encode("utf-8")
            if stdin_payload is not None
            else None
        )
        process_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            process_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=(
                    asyncio.subprocess.PIPE
                    if input_bytes is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(env),
                cwd=str(cwd) if cwd is not None else None,
                limit=8 * 1024 * 1024,
                **process_kwargs,
            )
        except Exception as exc:
            print(
                "[MODAL] 비동기 명령 프로세스 시작 실패: "
                f"command={args!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

        async def stop_process(reason: str) -> None:
            if process.returncode is not None:
                return
            print(
                f"[MODAL] {reason} 명령 프로세스 종료: "
                f"pid={process.pid}, command={args!r}"
            )
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=10)
            except ProcessLookupError:
                return
            except Exception as exc:
                print(
                    "[MODAL] 명령 프로세스 종료 실패: "
                    f"pid={process.pid}, command={args!r}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        if output_callback is None:
            communicate_task = asyncio.create_task(process.communicate(input_bytes))
            try:
                if timeout is None:
                    stdout_bytes, stderr_bytes = await communicate_task
                else:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        asyncio.shield(communicate_task),
                        timeout=timeout,
                    )
            except asyncio.TimeoutError as exc:
                print(
                    "[MODAL] 명령 제한 시간 초과: "
                    f"command={args[0]} {args[1] if len(args) > 1 else ''}, "
                    f"timeout={timeout}"
                )
                traceback.print_exc()
                await stop_process("제한 시간 초과")
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        asyncio.shield(communicate_task),
                        timeout=10,
                    )
                except Exception:
                    communicate_task.cancel()
                raise TimeoutError(
                    f"명령 제한 시간을 초과했습니다: {args[0]}"
                ) from exc
            except asyncio.CancelledError:
                await stop_process("취소된")
                try:
                    await asyncio.wait_for(
                        asyncio.shield(communicate_task),
                        timeout=10,
                    )
                except Exception as cleanup_exc:
                    print(
                        "[MODAL] 취소된 명령 출력 정리 실패: "
                        f"command={args!r}, "
                        f"error={type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                    traceback.print_exc()
                    communicate_task.cancel()
                raise
            return (
                int(process.returncode or 0),
                stdout_bytes.decode("utf-8", errors="replace")
                .replace("\r\n", "\n")
                .replace("\r", "\n"),
                stderr_bytes.decode("utf-8", errors="replace")
                .replace("\r\n", "\n")
                .replace("\r", "\n"),
            )

        if process.stdout is None or process.stderr is None:
            print(f"[MODAL] 비동기 스트리밍 출력 파이프 생성 실패: command={args!r}")
            await stop_process("출력 파이프 생성 실패")
            raise RuntimeError("Modal 명령 출력 파이프를 만들지 못했습니다.")

        captured: dict[str, list[str]] = {"stdout": [], "stderr": []}

        async def read_stream(
            stream: asyncio.StreamReader,
            source: str,
        ) -> None:
            try:
                while True:
                    raw_line = await stream.readline()
                    if not raw_line:
                        return
                    line = (
                        raw_line.decode("utf-8", errors="replace")
                        .replace("\r\n", "\n")
                        .replace("\r", "\n")
                    )
                    captured[source].append(line)
                    try:
                        output_callback(source, line.rstrip("\r\n"))
                    except Exception as callback_exc:
                        print(
                            "[MODAL] 실시간 명령 출력 처리 실패: "
                            f"source={source}, "
                            f"error={type(callback_exc).__name__}: {callback_exc}"
                        )
                        traceback.print_exc()
            except Exception as exc:
                print(
                    "[MODAL] 비동기 명령 출력 읽기 실패: "
                    f"source={source}, command={args!r}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise

        reader_tasks = [
            asyncio.create_task(read_stream(process.stdout, "stdout")),
            asyncio.create_task(read_stream(process.stderr, "stderr")),
        ]
        if input_bytes is not None:
            if process.stdin is None:
                print(f"[MODAL] 비동기 스트리밍 stdin 생성 실패: command={args!r}")
                await stop_process("stdin 생성 실패")
                await asyncio.gather(*reader_tasks, return_exceptions=True)
                raise RuntimeError("Modal 명령 입력 파이프를 만들지 못했습니다.")
            process.stdin.write(input_bytes)
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()

        wait_task = asyncio.create_task(process.wait())
        try:
            if timeout is None:
                await wait_task
            else:
                await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
            reader_results = await asyncio.gather(
                *reader_tasks,
                return_exceptions=True,
            )
            for reader_result in reader_results:
                if isinstance(reader_result, Exception):
                    raise reader_result
        except asyncio.TimeoutError as exc:
            print(
                "[MODAL] 스트리밍 명령 제한 시간 초과: "
                f"command={args[0]} {args[1] if len(args) > 1 else ''}, "
                f"timeout={timeout}"
            )
            traceback.print_exc()
            await stop_process("제한 시간 초과된 스트리밍")
            await asyncio.gather(*reader_tasks, return_exceptions=True)
            raise TimeoutError(
                f"명령 제한 시간을 초과했습니다: {args[0]}"
            ) from exc
        except asyncio.CancelledError:
            await stop_process("취소된 스트리밍")
            await asyncio.gather(*reader_tasks, return_exceptions=True)
            raise
        except Exception:
            await stop_process("실패한 스트리밍")
            await asyncio.gather(*reader_tasks, return_exceptions=True)
            raise

        return (
            int(process.returncode or 0),
            "".join(captured["stdout"]),
            "".join(captured["stderr"]),
        )

    def _install_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._install_state)
        snapshot["logs"] = [dict(item) for item in self._install_state.get("logs", [])]
        progress = self._install_state.get("progress")
        if isinstance(progress, dict):
            snapshot["progress"] = dict(progress)
        started_at = float(snapshot.get("started_at") or 0.0)
        if started_at > 0:
            finished_at = float(snapshot.get("finished_at") or 0.0)
            snapshot["elapsed_seconds"] = round(
                max(0.0, (finished_at or time.time()) - started_at),
                1,
            )
        return snapshot

    def _deployment_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._deployment_state)
        snapshot["logs"] = [
            dict(item) for item in self._deployment_state.get("logs", [])
        ]
        inventory = self._deployment_state.get("inventory")
        if isinstance(inventory, Mapping):
            snapshot["inventory"] = {
                key: (
                    [dict(item) if isinstance(item, Mapping) else item for item in value]
                    if isinstance(value, list)
                    else dict(value) if isinstance(value, Mapping) else value
                )
                for key, value in inventory.items()
            }
        started_at = float(snapshot.get("started_at") or 0.0)
        if started_at > 0:
            finished_at = float(snapshot.get("finished_at") or 0.0)
            snapshot["elapsed_seconds"] = round(
                max(0.0, (finished_at or time.time()) - started_at),
                1,
            )
        return snapshot

    def _deployment_running(self) -> bool:
        return bool(self._deployment_task and not self._deployment_task.done())

    def _append_deployment_log(self, source: str, line: str) -> None:
        cleaned = _ANSI_ESCAPE_RE.sub("", str(line)).replace("\x00", "").strip()
        if not cleaned:
            return
        if len(cleaned) > INSTALL_LOG_LINE_LIMIT:
            cleaned = cleaned[: INSTALL_LOG_LINE_LIMIT - 1] + "…"
        logs = self._deployment_state.setdefault("logs", [])
        if not isinstance(logs, list):
            print(
                "[MODAL] 재배포 로그 상태 형식 오류: "
                f"type={type(logs).__name__}; 빈 로그로 복구합니다."
            )
            logs = []
            self._deployment_state["logs"] = logs
        logs.append(
            {
                "time": time.time(),
                "source": str(source or "system"),
                "message": cleaned,
            }
        )
        if len(logs) > INSTALL_LOG_LIMIT:
            del logs[: len(logs) - INSTALL_LOG_LIMIT]
        self._deployment_state["updated_at"] = time.time()

    def _set_install_phase(
        self,
        phase: str,
        message: str,
        *,
        progress_mode: str,
    ) -> None:
        phase_order = tuple(INSTALL_PHASE_LABELS)
        if phase not in INSTALL_PHASE_LABELS:
            print(f"[MODAL] 알 수 없는 설치 단계 거부: phase={phase!r}")
            raise ValueError(f"알 수 없는 Modal 설치 단계입니다: {phase}")
        progress = dict(self._install_state.get("progress") or {})
        progress["mode"] = progress_mode
        self._install_state.update(
            phase=phase,
            phase_label=INSTALL_PHASE_LABELS[phase],
            phase_index=phase_order.index(phase),
            phase_count=len(phase_order),
            message=message,
            progress=progress,
            updated_at=time.time(),
        )

    def _append_install_log(self, source: str, line: str) -> None:
        cleaned = _ANSI_ESCAPE_RE.sub("", str(line)).replace("\x00", "").strip()
        if not cleaned:
            return
        if len(cleaned) > INSTALL_LOG_LINE_LIMIT:
            cleaned = cleaned[: INSTALL_LOG_LINE_LIMIT - 1] + "…"
        logs = self._install_state.setdefault("logs", [])
        if not isinstance(logs, list):
            print(
                "[MODAL] 설치 로그 상태 형식 오류: "
                f"type={type(logs).__name__}; 빈 로그로 복구합니다."
            )
            logs = []
            self._install_state["logs"] = logs
        logs.append(
            {
                "time": time.time(),
                "source": str(source or "system"),
                "message": cleaned,
            }
        )
        if len(logs) > INSTALL_LOG_LIMIT:
            del logs[: len(logs) - INSTALL_LOG_LIMIT]
        self._install_state["updated_at"] = time.time()

    def _handle_install_client_output(self, source: str, line: str) -> None:
        if source != "stderr" or not line.startswith(INSTALL_PROGRESS_PREFIX):
            self._append_install_log(source, line)
            return
        raw_event = line[len(INSTALL_PROGRESS_PREFIX) :]
        try:
            event = json.loads(raw_event)
            if not isinstance(event, dict):
                raise TypeError("진행 이벤트는 JSON 객체여야 합니다.")
        except Exception as exc:
            print(
                "[MODAL] 업로드 진행 이벤트 파싱 실패: "
                f"error={type(exc).__name__}: {exc}, payload={raw_event[:500]!r}"
            )
            traceback.print_exc()
            self._append_install_log("stderr", line)
            return

        event_name = str(event.get("event") or "")
        label = str(event.get("label") or "파일")
        progress = self._install_state.setdefault("progress", {})
        if event_name == "batch_start":
            progress["current_label"] = label
            progress["current_item"] = ""
            self._install_state["message"] = f"{label} 동기화를 준비하고 있습니다."
            self._append_install_log(
                "upload",
                f"{label}: {int(event.get('total_files') or 0)}개 파일 확인",
            )
        elif event_name == "file_queued":
            progress["current_label"] = label
            progress["current_item"] = str(event.get("name") or "")
            self._install_state["message"] = (
                f"{label} 전송 준비 중: {progress['current_item']}"
            )
            self._install_state["updated_at"] = time.time()
        elif event_name == "batch_complete":
            processed_files = max(0, int(event.get("processed_files") or 0))
            processed_bytes = max(0, int(event.get("processed_bytes") or 0))
            progress["completed_files"] = min(
                int(progress.get("total_files") or processed_files),
                int(progress.get("completed_files") or 0) + processed_files,
            )
            progress["completed_bytes"] = min(
                int(progress.get("total_bytes") or processed_bytes),
                int(progress.get("completed_bytes") or 0) + processed_bytes,
            )
            progress["uploaded_files"] = int(progress.get("uploaded_files") or 0) + max(
                0, int(event.get("uploaded_files") or 0)
            )
            progress["skipped_files"] = int(progress.get("skipped_files") or 0) + max(
                0, int(event.get("skipped_files") or 0)
            )
            progress["current_label"] = label
            progress["current_item"] = ""
            self._install_state["message"] = (
                f"{label} 동기화 완료 · 전체 "
                f"{progress['completed_files']}/{progress.get('total_files', 0)}개"
            )
            self._append_install_log(
                "upload",
                f"{label} 완료: 업로드 {int(event.get('uploaded_files') or 0)}개 · "
                f"기존 파일 {int(event.get('skipped_files') or 0)}개",
            )
        else:
            print(
                "[MODAL] 알 수 없는 업로드 진행 이벤트: "
                f"event={event_name!r}, payload={event!r}"
            )
            self._append_install_log("stderr", line)

    def _lora_operation_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._lora_operation_state)
        snapshot["logs"] = [
            dict(item) for item in self._lora_operation_state.get("logs", [])
        ]
        progress = self._lora_operation_state.get("progress")
        if isinstance(progress, Mapping):
            snapshot["progress"] = dict(progress)
        started_at = float(snapshot.get("started_at") or 0.0)
        if started_at > 0:
            finished_at = float(snapshot.get("finished_at") or 0.0)
            snapshot["elapsed_seconds"] = round(
                max(0.0, (finished_at or time.time()) - started_at),
                1,
            )
        return snapshot

    def _lora_operation_running(self) -> bool:
        return bool(
            self._lora_operation_task and not self._lora_operation_task.done()
        )

    def _append_lora_operation_log(self, source: str, line: str) -> None:
        cleaned = _ANSI_ESCAPE_RE.sub("", str(line)).replace("\x00", "").strip()
        if not cleaned:
            return
        if len(cleaned) > INSTALL_LOG_LINE_LIMIT:
            cleaned = cleaned[: INSTALL_LOG_LINE_LIMIT - 1] + "…"
        logs = self._lora_operation_state.setdefault("logs", [])
        if not isinstance(logs, list):
            print(
                "[MODAL_LORA] 작업 로그 상태 형식 오류: "
                f"type={type(logs).__name__}; 빈 로그로 복구합니다."
            )
            logs = []
            self._lora_operation_state["logs"] = logs
        logs.append(
            {
                "time": time.time(),
                "source": str(source or "system"),
                "message": cleaned,
            }
        )
        if len(logs) > INSTALL_LOG_LIMIT:
            del logs[: len(logs) - INSTALL_LOG_LIMIT]
        self._lora_operation_state["updated_at"] = time.time()

    def _handle_lora_client_output(self, source: str, line: str) -> None:
        if source != "stderr" or not line.startswith(INSTALL_PROGRESS_PREFIX):
            self._append_lora_operation_log(source, line)
            return
        raw_event = line[len(INSTALL_PROGRESS_PREFIX) :]
        try:
            event = json.loads(raw_event)
            if not isinstance(event, dict):
                raise TypeError("진행 이벤트는 JSON 객체여야 합니다.")
        except Exception as exc:
            print(
                "[MODAL_LORA] 업로드 진행 이벤트 파싱 실패: "
                f"error={type(exc).__name__}: {exc}, payload={raw_event[:500]!r}"
            )
            traceback.print_exc()
            self._append_lora_operation_log("stderr", line)
            return
        progress = self._lora_operation_state.setdefault("progress", {})
        event_name = str(event.get("event") or "")
        if event_name == "batch_start":
            progress["current_item"] = ""
            self._lora_operation_state["message"] = "현재 사용 LoRA를 확인하고 있습니다."
            self._append_lora_operation_log(
                "upload",
                f"LoRA {int(event.get('total_files') or 0)}개 전송 확인",
            )
        elif event_name == "file_queued":
            current = str(event.get("name") or "")
            progress["current_item"] = current
            self._lora_operation_state.update(
                message=f"LoRA 전송 준비 중: {current}",
                updated_at=time.time(),
            )
        elif event_name == "batch_complete":
            processed_files = max(0, int(event.get("processed_files") or 0))
            processed_bytes = max(0, int(event.get("processed_bytes") or 0))
            progress.update(
                completed_files=processed_files,
                completed_bytes=processed_bytes,
                uploaded_files=max(0, int(event.get("uploaded_files") or 0)),
                skipped_files=max(0, int(event.get("skipped_files") or 0)),
                current_item="",
            )
            self._lora_operation_state["message"] = (
                f"LoRA 업로드 완료 · 신규/갱신 {progress['uploaded_files']}개 · "
                f"동일 {progress['skipped_files']}개"
            )
            self._append_lora_operation_log(
                "upload",
                f"LoRA 업로드 완료: 신규/갱신 {progress['uploaded_files']}개 · "
                f"동일 {progress['skipped_files']}개",
            )
        else:
            print(
                "[MODAL_LORA] 알 수 없는 업로드 진행 이벤트: "
                f"event={event_name!r}, payload={event!r}"
            )
            self._append_lora_operation_log("stderr", line)

    async def _account_connected_once(self, settings: ModalSettings) -> bool:
        try:
            async with self._status_command_semaphore:
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

    async def account_connected(self, settings: ModalSettings) -> bool:
        """동시에 들어온 계정 확인 요청은 하나의 ``modal token info``를 공유한다."""

        profile = settings.profile
        task = self._account_check_tasks.get(profile)
        if task is None or task.done():
            task = asyncio.create_task(self._account_connected_once(settings))
            self._account_check_tasks[profile] = task

            def clear_account_task(
                completed: asyncio.Task[bool],
                *,
                key: str = profile,
            ) -> None:
                if self._account_check_tasks.get(key) is completed:
                    self._account_check_tasks.pop(key, None)
                if completed.cancelled():
                    return
                try:
                    completed.exception()
                except Exception as exc:
                    print(
                        "[MODAL] 계정 확인 백그라운드 작업 상태 회수 실패: "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

            task.add_done_callback(clear_account_task)
        return await asyncio.shield(task)

    async def _run_client_action_once(
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
            # 배포 정의에는 GPU를 고정하지 않고 모든 작업 호출에서 이 값을
            # with_options(gpu=...)로 적용한다.
            "worker_gpu": settings.worker_gpu,
            "vram_mode": settings.vram_mode,
            "container_start_max_retries": (
                settings.container_start_max_retries
            ),
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
            reason = str(response.get("reason") or "runtime_unavailable")
            if reason not in RUNTIME_FAILURE_REASONS:
                reason = "runtime_unavailable"
            error_type = str(response.get("error_type") or "ModalClientError")
            print(
                f"[MODAL] {action} 실패: app={settings.deployment_name}, "
                f"environment={settings.environment}, reason={reason}, "
                f"error_type={error_type}, error={error}, stderr={stderr[-1000:]}"
            )
            raise ModalClientActionError(
                error,
                reason=reason,
                error_type=error_type,
            )
        result = response.get("result")
        if not isinstance(result, dict):
            print(f"[MODAL] {action} 결과 객체 누락: type={type(result).__name__}")
            raise RuntimeError(f"Modal {action} 결과 객체가 없습니다.")
        return result

    async def _run_client_action(
        self,
        settings: ModalSettings,
        action: str,
        *,
        timeout: float,
        **payload: Any,
    ) -> dict[str, Any]:
        # 생성·변환·설치 등 쓰기 작업은 호출마다 반드시 독립 실행한다. 상태 UI가
        # 사용하는 읽기 전용 명령만 동일 payload 기준 single-flight로 합친다.
        shared_actions = {"runtime_stats", "web_status", "runtime_logs", "list_loras"}
        if action not in shared_actions:
            return await self._run_client_action_once(
                settings,
                action,
                timeout=timeout,
                **payload,
            )

        action_key = json.dumps(
            {
                "action": action,
                "profile": settings.profile,
                "environment": settings.environment,
                "app_name": settings.deployment_name,
                "worker_gpu": settings.worker_gpu,
                "vram_mode": settings.vram_mode,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        task = self._status_action_tasks.get(action_key)
        if task is None or task.done():
            async def run_limited() -> dict[str, Any]:
                async with self._status_command_semaphore:
                    return await self._run_client_action_once(
                        settings,
                        action,
                        timeout=timeout,
                        **payload,
                    )

            task = asyncio.create_task(run_limited())
            self._status_action_tasks[action_key] = task

            def clear_status_task(
                completed: asyncio.Task[dict[str, Any]],
                *,
                key: str = action_key,
            ) -> None:
                if self._status_action_tasks.get(key) is completed:
                    self._status_action_tasks.pop(key, None)
                if completed.cancelled():
                    return
                try:
                    completed.exception()
                except Exception as exc:
                    print(
                        "[MODAL] 상태 명령 작업 상태 회수 실패: "
                        f"action={action}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

            task.add_done_callback(clear_status_task)
        return await asyncio.shield(task)

    async def custom_nodes(self) -> dict[str, Any]:
        try:
            inventory = await asyncio.to_thread(
                inventory_custom_nodes,
                self.project_root,
            )
        except Exception as exc:
            print(
                "[MODAL] custom node 인벤토리 생성 실패: "
                f"error={type(exc).__name__}: {exc}, project_root={self.project_root}"
            )
            traceback.print_exc()
            raise
        return {
            "ok": True,
            "custom_nodes": public_custom_node_inventory(inventory),
        }

    async def _inventory_for_deploy(self) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                inventory_custom_nodes,
                self.project_root,
            )
        except Exception as exc:
            print(
                "[MODAL] 배포용 custom node 인벤토리 생성 실패: "
                f"error={type(exc).__name__}: {exc}, project_root={self.project_root}"
            )
            traceback.print_exc()
            raise

    async def status(self, *, include_runtime: bool = False) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        # 토큰(연결) 존재 여부는 modal_enabled와 무관하게 항상 조회한다.
        # 기능이 꺼져 있어도 "방금 인증한 토큰이 등록됐는지" 확인할 수 있어야 한다.
        connection_checked = True
        connected = await self.account_connected(settings)
        pending_deletes, pending_video_deletes = await asyncio.gather(
            asyncio.to_thread(self._delete_outbox_count),
            asyncio.to_thread(self._video_delete_outbox_count),
        )
        if settings.enabled and connected and pending_deletes:
            self._schedule_delete_flush()
        if settings.enabled and connected and pending_video_deletes:
            self._schedule_video_delete_flush()
        runtime: dict[str, Any] | None = None
        web_runtime: dict[str, Any] | None = None
        if include_runtime:
            if not settings.enabled:
                runtime = {"available": False, "reason": "disabled"}
            elif not connected:
                print("[MODAL] 런타임 통계 조회 생략: Modal 계정이 연결되지 않았습니다.")
                runtime = {"available": False, "reason": "account_not_connected"}
            elif self._deployment_running():
                runtime = {
                    "available": False,
                    "reason": "deployment_in_progress",
                    "error": "",
                }
            elif (
                self._install_state.get("state") == "running"
                and self._install_state.get("phase") in {"assets", "deploy"}
            ):
                runtime = {
                    "available": False,
                    "reason": "deployment_in_progress",
                    "error": "",
                }
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
                    reason = _runtime_failure_reason(exc)
                    installing = self._install_state.get("state") == "running"
                    if reason == "app_not_deployed" and installing:
                        reason = "deployment_in_progress"
                    runtime = {
                        "available": False,
                        "reason": reason,
                        "error": "" if reason == "deployment_in_progress" else str(exc),
                    }
            web_runtime = await self.web_status(
                settings=settings,
                connected=connected,
            )
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
            "install": self._install_snapshot(),
            "lora_operation": self._lora_operation_snapshot(),
            "deployment": self._deployment_snapshot(),
            "autoscaler": dict(self._autoscaler_state),
            "probe": dict(self._probe_state),
            "cost": cost_summary(settings),
            "billing": billing,
            "pending_lora_deletes": pending_deletes,
            "pending_video_deletes": pending_video_deletes,
            "runtime": runtime,
            "web": web_runtime,
            "workflow_runs": self.recent_workflow_runs(),
        }

    async def worker_status(self) -> dict[str, Any]:
        """좌측 always-on 위젯용 Modal 작업자 + WebUI 상태 스냅샷.

        worker 블록과 web 블록은 서로 독립적으로 계산된다. 한쪽이 실패해도 다른
        쪽은 정상 응답하도록 예외를 격리한다(Z안). 최상위 필드(ok/enabled/gpu/
        refresh_seconds/checked_at)는 프론트엔드 기능 게이트·폴링 주기에 그대로 사용.
        """
        settings = ModalSettings.from_mapping(self.get_config())
        checked_at = time.time()
        top = {
            "ok": True,
            "enabled": settings.enabled,
            "gpu": settings.worker_gpu,
            "worker_gpu": settings.worker_gpu,
            "web_gpu": settings.web_gpu,
            "vram_mode": settings.vram_mode,
            "refresh_seconds": settings.status_refresh_seconds,
            "checked_at": checked_at,
        }

        # Worker App과 WebUI App은 독립된 control-plane 조회다. 두 작업을 함께
        # 시작하고 실제 Modal 명령 동시성은 _status_command_semaphore가 제한한다.
        worker_task = asyncio.create_task(self._worker_status_block(settings))
        web_task = asyncio.create_task(self._dock_web_status(settings))

        worker: dict[str, Any]
        try:
            worker = await worker_task
        except asyncio.CancelledError:
            web_task.cancel()
            await asyncio.gather(web_task, return_exceptions=True)
            raise
        except Exception as exc:
            # worker 블록 계산 자체가 예외를 던지면 web 블록은 살린다(양방향 격리).
            print(
                f"[MODAL_WORKER] 작업자 상태 블록 계산 실패: "
                f"app={settings.deployment_name}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            worker = {
                "state": "error",
                "available": False,
                "reason": "runtime_unavailable",
                "gpu_on": False,
                "workers": 0,
                "generating": 0,
                "queued": 0,
                "message": None,
                "install_phase": None,
                "error": str(exc),
            }

        web: dict[str, Any]
        try:
            web = await web_task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # web 블록 계산 자체가 예외를 던지면 worker 블록은 살린다.
            print(
                f"[MODAL_WORKER] WebUI 상태 블록 계산 실패: "
                f"app={self._web_app_name(settings)}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            web = {
                "state": "unknown",
                "deployed": None,
                "url": "",
                "runners": None,
                "message": "WebUI 상태를 확인하지 못했습니다.",
                "error": str(exc),
            }
        return {**top, "worker": worker, "web": web}

    async def _worker_status_block(self, settings: ModalSettings) -> dict[str, Any]:
        """GPU 작업자(작업 App) 상태를 worker 서브오브젝트로 계산.

        state: disabled / deploying / running / stopped / error.
        """
        if not settings.enabled:
            print(
                "[MODAL_WORKER] 상태 조회 생략: Modal 사용 설정이 OFF입니다. "
                f"app={settings.deployment_name}, environment={settings.environment}"
            )
            return {
                "state": "disabled",
                "available": False,
                "reason": "disabled",
                "gpu_on": False,
                "workers": 0,
                "generating": 0,
                "queued": 0,
                "message": None,
                "install_phase": None,
                "error": None,
            }

        if self._deployment_running():
            deployment = self._deployment_snapshot()
            return {
                "state": "deploying",
                "available": False,
                "reason": "deployment_in_progress",
                "gpu_on": False,
                "workers": 0,
                "generating": 0,
                "queued": 0,
                "message": deployment.get("message") or "Modal App을 재배포하고 있습니다.",
                "install_phase": deployment.get("phase"),
                "error": None,
            }

        try:
            stats = await self._run_client_action(
                settings,
                "runtime_stats",
                timeout=30,
            )
            runners = int(stats["num_total_runners"])
            running_inputs = int(stats["num_running_inputs"])
            backlog = int(stats["backlog"])
            if runners < 0 or running_inputs < 0 or backlog < 0:
                raise ValueError(
                    "Modal worker 통계에는 음수가 올 수 없습니다: "
                    f"runners={runners}, running_inputs={running_inputs}, backlog={backlog}"
                )
        except Exception as exc:
            reason = _runtime_failure_reason(exc)
            print(
                "[MODAL_WORKER] 상태 조회 실패: "
                f"app={settings.deployment_name}, environment={settings.environment}, "
                f"reason={reason}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return {
                "state": "error",
                "available": False,
                "reason": reason,
                "gpu_on": False,
                "workers": 0,
                "generating": 0,
                "queued": 0,
                "message": None,
                "install_phase": None,
                "error": str(exc),
            }

        gpu_on = runners > 0
        return {
            "state": "running" if gpu_on else "stopped",
            "available": True,
            "reason": None,
            "gpu_on": gpu_on,
            "workers": runners,
            "generating": running_inputs,
            "queued": backlog,
            "message": None,
            "install_phase": None,
            "error": None,
        }

    async def _dock_web_status(self, settings: ModalSettings) -> dict[str, Any]:
        """좌측 위젯용 WebUI 상태.

        공개 ``get_url()``(필수, 배포/URL) + ``AppList``(best-effort, running 보강)
        + 로컬 ``_web_state``(이 프로세스가 아는 전이 상태)을 조합해 Z안 우선순위로
        state를 정한다. 이 경로에서 WebUI URL로 HTTP 요청은 절대 하지 않는다
        (scale-to-zero 서버를 깨워 과금을 유발한다).
        """
        if not settings.enabled:
            return {
                "state": "stopped",
                "gpu": settings.web_gpu,
                "deployed": False,
                "url": "",
                "runners": None,
                "message": "Modal 사용 설정이 꺼져 있습니다.",
                "error": None,
                "reason": "disabled",
            }

        # 1. 로컬 전이 상태(starting/stopping)는 이 프로세스가 직접 수행 중일 때만
        #    신뢰한다. 작업이 끝났는데 전이 상태가 남아 있으면 stale이므로 무시.
        web_task_active = self._web_task is not None and not self._web_task.done()
        local_state = self._web_state.get("state")
        if local_state in ("starting", "stopping") and not web_task_active:
            local_state = None

        # 2. remote control-plane probe(get_url + AppList) — WEB_REMOTE_CACHE_SECONDS 캐시.
        remote = await self._cached_web_remote(settings)
        deployed = remote.get("deployed")  # True / False / None(조회 불가)
        runners = remote.get("runners")  # int | None
        remote_error = remote.get("error")
        url = remote.get("url") or ""

        # 3. state 우선순위(Z안):
        #    로컬 전이(starting/stopping/failed) > AppList 보강(deployed+runners)
        #    > 로컬 running(이 프로세스가 시작) > deployed(미추적) > stopped/unknown
        if local_state in ("starting", "stopping", "failed"):
            state = local_state
        elif deployed is True and runners is not None:
            state = "running" if runners > 0 else "stopped"
        elif local_state == "running":
            state = "running"
        elif deployed is True:
            # 배포됨이 확인됐으나 running 여부를 추적 못 함 → stopped 거짓말 금지.
            state = "unknown"
        elif deployed is False:
            state = "stopped"
        else:
            # probe 자체가 실패해 deployed를 모름 → 역시 stopped 단정 금지.
            state = "unknown"

        if state == "unknown" and remote_error:
            message = f"WebUI 상태 조회 실패: {remote_error}"
            error = remote_error
        elif state == "unknown":
            message = "WebUI App은 배포되어 있으나 실행 여부를 추적하지 못했습니다."
            error = None
        elif state == "running":
            message = "WebUI가 실행 중입니다."
            error = None
        elif state == "stopped":
            message = "WebUI가 꺼져 있습니다."
            error = None
        elif state in ("starting", "stopping", "failed"):
            message = self._web_state.get("message") or ""
            error = self._web_state.get("error")
        else:
            message = remote.get("message") or ""
            error = None

        return {
            "state": state,
            "gpu": settings.web_gpu,
            "deployed": deployed,
            "url": url,
            "runners": runners,
            "message": message,
            "error": error,
        }

    async def _cached_web_remote(self, settings: ModalSettings) -> dict[str, Any]:
        """get_url + AppList 결과를 WEB_REMOTE_CACHE_SECONDS(기본 12초) 캐시.

        캐시 만료 시 ``web_status`` client action 1회 subprocess로 갱신한다. subprocess
        전체가 실패해도 캐시에 error로 저장해 호출자가 unknown 퇴각하도록 한다.
        _billing_cache 패턴(stored_at_monotonic 기반 TTL)을 따른다.
        """
        async with self._web_remote_lock:
            now = time.monotonic()
            cache = self._web_remote_cache
            cache_age = (
                now - float(cache["stored_at_monotonic"])
                if cache is not None
                else float("inf")
            )
            if (
                cache is not None
                and cache.get("environment") == settings.environment
                and cache.get("profile") == settings.profile
                and cache_age < WEB_REMOTE_CACHE_SECONDS
            ):
                return dict(cache["result"])
            try:
                result_raw = await self._run_client_action(
                    settings,
                    "web_status",
                    timeout=30,
                    web_app_name=self._web_app_name(settings),
                )
                result = {
                    "deployed": bool(result_raw.get("deployed")),
                    "url": str(result_raw.get("url") or ""),
                    "runners": result_raw.get("runners"),  # int | None
                    "message": result_raw.get("message"),
                    "error": None,
                }
            except Exception as exc:
                print(
                    f"[MODAL_WORKER] WebUI remote probe 실패(캐시를 error로 저장): "
                    f"app={self._web_app_name(settings)}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                result = {
                    "deployed": None,
                    "url": "",
                    "runners": None,
                    "message": None,
                    "error": str(exc),
                }
            self._web_remote_cache = {
                "profile": settings.profile,
                "environment": settings.environment,
                "stored_at_monotonic": now,
                "result": result,
            }
            return dict(result)

    async def _remote_web_status(
        self,
        settings: ModalSettings,
    ) -> dict[str, Any]:
        result = await self._run_client_action(
            settings,
            "web_status",
            timeout=30,
            web_app_name=self._web_app_name(settings),
        )
        runners = max(0, int(result.get("num_total_runners") or 0))
        running_inputs = max(0, int(result.get("num_running_inputs") or 0))
        deployed = bool(result.get("url"))
        return {
            "available": True,
            "deployed": deployed,
            "state": "running" if runners > 0 else "stopped",
            "gpu": settings.web_gpu,
            **({"reason": "app_not_deployed"} if not deployed else {}),
            "message": (
                f"Modal ComfyUI 웹 {settings.web_gpu}가 실행 중입니다."
                if runners > 0
                else (
                    f"웹 App은 준비되어 있고 {settings.web_gpu}는 꺼져 있습니다."
                    if deployed
                    else "웹 전용 App이 중지되어 있습니다."
                )
            ),
            "url": str(result.get("url") or ""),
            "app_name": self._web_app_name(settings),
            "num_total_runners": runners,
            "num_running_inputs": running_inputs,
            "backlog": max(0, int(result.get("backlog") or 0)),
            "updated_at": time.time(),
        }

    async def web_status(
        self,
        *,
        settings: ModalSettings | None = None,
        connected: bool | None = None,
    ) -> dict[str, Any]:
        settings = settings or ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print("[MODAL] 웹 상태 조회 생략: Modal 사용 설정이 OFF입니다.")
            return {
                "available": False,
                "state": "stopped",
                "reason": "disabled",
                "message": "Modal 사용 설정이 꺼져 있습니다.",
            }
        if connected is None:
            connected = await self.account_connected(settings)
        if not connected:
            print(
                "[MODAL] 웹 상태 조회 생략: Modal 계정이 연결되지 않았습니다. "
                f"profile={settings.profile}"
            )
            return {
                "available": False,
                "state": "stopped",
                "reason": "account_not_connected",
                "message": "Modal 계정을 먼저 연결하세요.",
            }
        if self._web_task and not self._web_task.done():
            return dict(self._web_state)
        try:
            status = await self._remote_web_status(settings)
            self._web_state = dict(status)
            return status
        except ModalClientActionError as exc:
            if exc.reason == "app_not_deployed":
                status = {
                    "available": True,
                    "deployed": False,
                    "state": "stopped",
                    "reason": "app_not_deployed",
                    "message": "웹 전용 App이 중지되어 있습니다.",
                    "error": str(exc),
                    "app_name": self._web_app_name(settings),
                    "num_total_runners": 0,
                    "num_running_inputs": 0,
                    "backlog": 0,
                    "updated_at": time.time(),
                }
                self._web_state = dict(status)
                return status
            print(
                f"[MODAL] 웹 상태 조회 실패: app={self._web_app_name(settings)}, "
                f"reason={exc.reason}, error_type={exc.error_type}, error={exc}"
            )
            traceback.print_exc()
            return {
                "available": False,
                "state": "failed",
                "reason": exc.reason,
                "error": str(exc),
                "message": f"웹 상태 조회 실패: {exc}",
            }
        except Exception as exc:
            print(
                f"[MODAL] 웹 상태 조회 예외: app={self._web_app_name(settings)}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return {
                "available": False,
                "state": "failed",
                "reason": "runtime_unavailable",
                "error": str(exc),
                "message": f"웹 상태 조회 실패: {type(exc).__name__}: {exc}",
            }

    async def web_url(self) -> dict[str, Any]:
        """실행 중인 웹 전용 App의 URL을 반환하며 GPU를 새로 켜지는 않는다."""
        status = await self.web_status()
        if status.get("state") != "running" or not status.get("url"):
            return {
                **status,
                "available": False,
                "reason": status.get("reason") or "web_not_running",
            }
        return {
            "available": True,
            "state": "running",
            "url": str(status["url"]),
            "app_name": status.get("app_name"),
        }

    async def start_web(self) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            raise RuntimeError("Modal 사용을 켜고 설정을 저장하세요.")
        async with self._web_action_lock:
            if self._deployment_running():
                raise RuntimeError("Modal 재배포가 진행 중입니다. 완료 후 시작하세요.")
            if self._web_task and not self._web_task.done():
                return dict(self._web_state)
            if not await self.account_connected(settings):
                raise RuntimeError("Modal 계정을 먼저 연결하세요.")
            current = await self.web_status(settings=settings, connected=True)
            if current.get("state") == "running":
                return current
            now = time.time()
            self._web_state = {
                "available": True,
                "deployed": bool(current.get("deployed")),
                "state": "starting",
                "gpu": settings.web_gpu,
                "message": (
                    "웹 전용 App을 준비하고 ComfyUI "
                    f"{settings.web_gpu}를 시작하고 있습니다."
                ),
                "app_name": self._web_app_name(settings),
                "num_total_runners": 0,
                "num_running_inputs": 0,
                "backlog": 0,
                "updated_at": now,
            }
            self._web_start_cancel_event = threading.Event()
            self._web_task = asyncio.create_task(
                self._run_web_start(settings, current)
            )
            return dict(self._web_state)

    @staticmethod
    def _warm_web_url(
        url: str,
        cancel_event: threading.Event | None = None,
    ) -> int:
        deadline = time.monotonic() + 650
        attempts = 0

        def raise_if_cancelled() -> None:
            if cancel_event is None or not cancel_event.is_set():
                return
            print(
                f"[MODAL] 웹 Server 준비 대기 취소: attempts={attempts}, url={url}"
            )
            raise WebStartCancelled("Modal ComfyUI 시작이 사용자에 의해 취소되었습니다.")

        while True:
            raise_if_cancelled()
            attempts += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    raise TimeoutError(
                        "Modal ComfyUI Server가 650초 안에 준비되지 않았습니다."
                    )
                except TimeoutError:
                    print(
                        f"[MODAL] 웹 Server 준비 시간 초과: "
                        f"attempts={attempts - 1}, url={url}"
                    )
                    traceback.print_exc()
                    raise
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SOYA-Comfy-Manager/1.0"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=min(30, max(1, remaining)),
                ) as response:
                    response.read(1)
                    status = int(response.status or 0)
                    if 200 <= status < 400:
                        raise_if_cancelled()
                        return status
                    raise RuntimeError(
                        f"Modal ComfyUI 준비 응답이 HTTP {status}입니다."
                    )
            except urllib.error.HTTPError as exc:
                # Modal Server는 scale-to-zero 콜드 스타트 동안 503을 즉시 반환하며,
                # 그 요청 자체가 컨테이너 시작을 트리거한다.
                if exc.code == 503 and time.monotonic() < deadline:
                    if attempts == 1 or attempts % 10 == 0:
                        print(
                            f"[MODAL] 웹 Server 콜드 스타트 대기: "
                            f"status=503, attempt={attempts}, url={url}"
                        )
                    if cancel_event is None:
                        time.sleep(1)
                    elif cancel_event.wait(1):
                        raise_if_cancelled()
                    continue
                print(
                    f"[MODAL] 웹 Server 준비 요청 실패: status={exc.code}, "
                    f"attempt={attempts}, url={url}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise
            except Exception as exc:
                raise_if_cancelled()
                print(
                    f"[MODAL] 웹 Server 준비 요청 예외: attempt={attempts}, "
                    f"url={url}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise

    async def _deploy_worker_app(
        self,
        settings: ModalSettings,
        *,
        custom_node_inventory: Mapping[str, Any] | None = None,
        force_custom_node_build: bool = False,
        output_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        inventory = (
            custom_node_inventory
            if custom_node_inventory is not None
            else await self._inventory_for_deploy()
        )
        async with self._deploy_lock:
            code, _stdout, stderr = await self._run_command(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "deploy",
                    "-m",
                    "modal_backend.modal_app",
                    "--env",
                    settings.environment,
                ],
                env=self._modal_deploy_env(
                    settings,
                    custom_node_inventory=inventory,
                    force_custom_node_build=force_custom_node_build,
                ),
                cwd=self.project_root,
                timeout=3600,
                output_callback=output_callback,
            )
        if code != 0:
            print(
                f"[MODAL] 작업 App 배포 실패: app={settings.deployment_name}, "
                f"env={settings.environment}, exit_code={code}, stderr={stderr[-1200:]}"
            )
            raise RuntimeError("Modal 작업 App 배포에 실패했습니다.")

    async def _deploy_web_app(
        self,
        settings: ModalSettings,
        *,
        custom_node_inventory: Mapping[str, Any] | None = None,
        force_custom_node_build: bool = False,
        output_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        inventory = (
            custom_node_inventory
            if custom_node_inventory is not None
            else await self._inventory_for_deploy()
        )
        if output_callback is None:
            def output_callback(source: str, line: str) -> None:
                if not line:
                    return
                print(f"[MODAL_WEB_DEPLOY][{source}] {line}")

        async with self._deploy_lock:
            code, _stdout, stderr = await self._run_command(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "deploy",
                    "-m",
                    "modal_backend.modal_web_app",
                    "--env",
                    settings.environment,
                ],
                env=self._modal_deploy_env(
                    settings,
                    custom_node_inventory=inventory,
                    force_custom_node_build=force_custom_node_build,
                ),
                cwd=self.project_root,
                timeout=3600,
                output_callback=output_callback,
            )
        if code != 0:
            print(
                f"[MODAL] 웹 App 배포 실패: app={self._web_app_name(settings)}, "
                f"env={settings.environment}, exit_code={code}, stderr={stderr[-1200:]}"
            )
            raise RuntimeError("Modal ComfyUI 웹 App 배포에 실패했습니다.")

    async def _stop_web_app(self, settings: ModalSettings) -> None:
        code, _stdout, stderr = await self._run_command(
            [
                sys.executable,
                "-m",
                "modal",
                "app",
                "stop",
                self._web_app_name(settings),
                "--yes",
                "--env",
                settings.environment,
            ],
            env=self._subprocess_env(settings.profile),
            timeout=120,
        )
        if code != 0:
            print(
                f"[MODAL] 웹 App 종료 실패: app={self._web_app_name(settings)}, "
                f"exit_code={code}, stderr={stderr[-1200:]}"
            )
            raise RuntimeError("Modal ComfyUI 웹 App 종료에 실패했습니다.")

    async def _run_web_start(
        self,
        settings: ModalSettings,
        current: Mapping[str, Any],
    ) -> None:
        try:
            self._web_state.update(
                message=(
                    "웹 전용 App을 선택한 GPU로 재배포하고 있습니다. "
                    "GPU 아키텍처는 공통 CUDA 이미지에 포함되어 있습니다."
                    if current.get("deployed")
                    else (
                        "중지된 웹 전용 App을 다시 배포하고 있습니다. "
                        "공통 이미지가 없거나 코드·Custom Node가 바뀐 경우에만 "
                        "이미지 빌드가 진행됩니다."
                    )
                ),
                updated_at=time.time(),
            )
            await self._deploy_web_app(settings)
            remote = await self._remote_web_status(settings)
            url = str(remote.get("url") or "")
            if not url:
                print(
                    f"[MODAL] 웹 App 시작 URL 누락: app={self._web_app_name(settings)}"
                )
                raise RuntimeError("Modal ComfyUI 웹 URL을 찾지 못했습니다.")
            self._web_state.update(
                deployed=True,
                url=url,
                message=(
                    f"{settings.web_gpu} 컨테이너를 시작하고 "
                    "ComfyUI 준비를 기다리고 있습니다."
                ),
                updated_at=time.time(),
            )
            status_code = await asyncio.to_thread(
                self._warm_web_url,
                url,
                self._web_start_cancel_event,
            )
            if status_code < 200 or status_code >= 400:
                print(
                    f"[MODAL] 웹 App 준비 응답 오류: app={self._web_app_name(settings)}, "
                    f"status={status_code}, url={url}"
                )
                raise RuntimeError(f"Modal ComfyUI 준비 응답이 HTTP {status_code}입니다.")
            remote = await self._remote_web_status(settings)
            self._web_state = {
                **remote,
                "state": "running",
                "message": f"Modal ComfyUI 웹 {settings.web_gpu}가 실행 중입니다.",
                "updated_at": time.time(),
            }
            print(
                f"[MODAL] 웹 App 시작 완료: app={self._web_app_name(settings)}, "
                f"url={url}"
            )
        except (asyncio.CancelledError, WebStartCancelled):
            self._web_start_cancel_event.set()
            print(
                f"[MODAL] 웹 App 시작 작업 취소: app={self._web_app_name(settings)}"
            )
            raise
        except Exception as exc:
            print(f"[MODAL] 웹 App 시작 실패: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            cleanup_error = ""
            try:
                await self._stop_web_app(settings)
                print(
                    f"[MODAL] 시작 실패 웹 App 자동 종료 완료: "
                    f"app={self._web_app_name(settings)}"
                )
            except Exception as cleanup_exc:
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                print(
                    f"[MODAL] 시작 실패 웹 App 자동 종료도 실패: "
                    f"app={self._web_app_name(settings)}, error={cleanup_error}"
                )
                traceback.print_exc()
            stopped = not cleanup_error
            self._web_state = {
                **self._web_state,
                "available": False,
                "deployed": False if stopped else bool(self._web_state.get("deployed")),
                "state": "failed",
                "reason": "runtime_unavailable",
                "message": (
                    f"웹 App 시작 실패로 자동 종료했습니다: {type(exc).__name__}: {exc}"
                    if stopped
                    else (
                        f"웹 App 시작 실패 후 자동 종료도 실패했습니다: "
                        f"{type(exc).__name__}: {exc}"
                    )
                ),
                "error": str(exc),
                **({"cleanup_error": cleanup_error} if cleanup_error else {}),
                "url": "" if stopped else str(self._web_state.get("url") or ""),
                "num_total_runners": 0 if stopped else int(
                    self._web_state.get("num_total_runners") or 0
                ),
                "num_running_inputs": 0 if stopped else int(
                    self._web_state.get("num_running_inputs") or 0
                ),
                "backlog": 0 if stopped else int(self._web_state.get("backlog") or 0),
                "updated_at": time.time(),
            }

    async def stop_web(self) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            raise RuntimeError("Modal 사용을 켜고 설정을 저장하세요.")
        async with self._web_action_lock:
            if self._deployment_running():
                raise RuntimeError("Modal 재배포가 진행 중입니다. 완료 후 종료하세요.")
            if self._web_task and not self._web_task.done():
                if self._web_state.get("state") == "stopping":
                    return dict(self._web_state)
                if self._web_state.get("state") != "starting":
                    raise RuntimeError("Modal ComfyUI 웹 작업이 이미 진행 중입니다.")
                start_task = self._web_task
                self._web_start_cancel_event.set()
                start_task.cancel()
                self._web_state = {
                    **self._web_state,
                    "state": "stopping",
                    "message": (
                        f"ComfyUI 시작을 취소하고 웹 App과 {settings.web_gpu} 컨테이너를 "
                        "완전히 종료하고 있습니다."
                    ),
                    "updated_at": time.time(),
                }
                self._web_task = asyncio.create_task(
                    self._run_web_stop(settings, start_task=start_task)
                )
                return dict(self._web_state)
            if not await self.account_connected(settings):
                raise RuntimeError("Modal 계정을 먼저 연결하세요.")
            current = await self.web_status(settings=settings, connected=True)
            if current.get("state") == "stopped" and not current.get("deployed"):
                return current
            self._web_state = {
                **current,
                "state": "stopping",
                "message": (
                    f"웹 전용 App과 {settings.web_gpu} 컨테이너를 "
                    "완전히 종료하고 있습니다."
                ),
                "updated_at": time.time(),
            }
            self._web_task = asyncio.create_task(self._run_web_stop(settings))
            return dict(self._web_state)

    async def _run_web_stop(
        self,
        settings: ModalSettings,
        *,
        start_task: asyncio.Task | None = None,
    ) -> None:
        try:
            if start_task is not None:
                try:
                    await start_task
                except (asyncio.CancelledError, WebStartCancelled):
                    print(
                        f"[MODAL] 취소된 웹 시작 작업 정리 완료: "
                        f"app={self._web_app_name(settings)}"
                    )
                except Exception as exc:
                    print(
                        f"[MODAL] 취소 중 웹 시작 작업 예외: "
                        f"app={self._web_app_name(settings)}, "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
            # 비용이 발생할 수 있는 원격 App 중지를 최우선으로 처리한다. 로그 조회가
            # 느리거나 실패해도 GPU 종료가 지연되어서는 안 된다.
            await self._stop_web_app(settings)
            self._web_state = {
                "available": True,
                "deployed": False,
                "state": "stopped",
                "reason": "app_not_deployed",
                "message": (
                    f"Modal ComfyUI 웹 App과 {settings.web_gpu}가 "
                    "완전히 종료되었습니다."
                ),
                "app_name": self._web_app_name(settings),
                "num_total_runners": 0,
                "num_running_inputs": 0,
                "backlog": 0,
                "updated_at": time.time(),
            }
            print(f"[MODAL] 웹 App 종료 완료: app={self._web_app_name(settings)}")
            try:
                await self.runtime_logs(entries=RUNTIME_LOG_LIMIT)
            except Exception as log_exc:
                print(
                    "[MODAL] 웹 App 종료 후 로그 조회 실패(종료 상태 유지): "
                    f"app={self._web_app_name(settings)}, "
                    f"error={type(log_exc).__name__}: {log_exc}"
                )
                traceback.print_exc()
        except Exception as exc:
            print(f"[MODAL] 웹 App 종료 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._web_state = {
                **self._web_state,
                "state": "failed",
                "message": f"웹 App 종료 실패: {type(exc).__name__}: {exc}",
                "error": str(exc),
                "updated_at": time.time(),
            }

    async def runtime_logs(self, *, entries: int = RUNTIME_LOG_LIMIT) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        requested = max(20, min(int(entries), 1000))
        remote_errors: list[dict[str, Any]] = []
        if not settings.enabled:
            print("[MODAL] 런타임 로그 원격 조회 생략: Modal 사용 설정이 OFF입니다.")
        elif not await self.account_connected(settings):
            print("[MODAL] 런타임 로그 원격 조회 생략: Modal 계정이 연결되지 않았습니다.")
            remote_errors.append({"reason": "account_not_connected"})
        else:
            try:
                result = await self._run_client_action(
                    settings,
                    "runtime_logs",
                    timeout=45,
                    web_app_name=self._web_app_name(settings),
                    entries=requested,
                )
                fetched = result.get("logs")
                if isinstance(fetched, list):
                    self._runtime_log_cache = [
                        dict(item) for item in fetched if isinstance(item, Mapping)
                    ][-requested:]
                else:
                    print(
                        "[MODAL] 런타임 로그 결과 형식 오류: "
                        f"type={type(fetched).__name__}"
                    )
                errors = result.get("errors")
                if isinstance(errors, list):
                    remote_errors.extend(
                        dict(item) for item in errors if isinstance(item, Mapping)
                    )
            except Exception as exc:
                print(f"[MODAL] 런타임 로그 조회 실패: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                remote_errors.append(
                    {
                        "reason": _runtime_failure_reason(exc),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        logs = [dict(item) for item in self._runtime_log_cache]
        for item in self._install_state.get("logs", []):
            if not isinstance(item, Mapping):
                continue
            logs.append(
                {
                    "time": float(item.get("time") or 0.0),
                    "source": str(item.get("source") or "system"),
                    "category": "sync",
                    "app_role": "local",
                    "app_name": "SOYA Modal client",
                    "function_name": "sync/install",
                    "container_id": "",
                    "message": str(item.get("message") or ""),
                }
            )
        for item in self._deployment_state.get("logs", []):
            if not isinstance(item, Mapping):
                continue
            logs.append(
                {
                    "time": float(item.get("time") or 0.0),
                    "source": str(item.get("source") or "system"),
                    "category": "deployment",
                    "app_role": "local",
                    "app_name": "SOYA Modal client",
                    "function_name": "redeploy/custom-nodes",
                    "container_id": "",
                    "message": str(item.get("message") or ""),
                }
            )
        probe_state = str(self._probe_state.get("state") or "")
        probe_updated_at = float(self._probe_state.get("updated_at") or 0.0)
        probe_message = str(self._probe_state.get("message") or "")
        if probe_state != "idle" and probe_updated_at and probe_message:
            logs.append(
                {
                    "time": probe_updated_at,
                    "source": "system",
                    "category": "diagnostic",
                    "app_role": "local",
                    "app_name": "SOYA Modal client",
                    "function_name": "gpu_probe",
                    "container_id": "",
                    "message": probe_message,
                }
            )
        logs = [
            item
            for item in logs
            if float(item.get("time") or 0.0) >= self._runtime_log_session_started_at
        ]
        logs.sort(key=lambda item: float(item.get("time") or 0.0))
        if len(logs) > requested:
            logs = logs[-requested:]
        return {
            "ok": True,
            "logs": logs,
            "errors": remote_errors,
            "limit": requested,
            "cached": bool(remote_errors),
            "checked_at": time.time(),
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

    def workflows(self) -> dict[str, Any]:
        """``SOYA_USER`` 폴더의 실제 워크플로우 파일을 동적 나열한다.

        config.json 바인딩(루트를 가리켜 전부 거부되던 문제)에 의존하지 않는다.
        각 파일은 개별 try/except로 처리해 한 파일의 깨진 JSON·모델 해석 실패가
        전체 응답을 터뜨리지 않게 한다. 파싱에 실패한 파일은 목록에서 제외되고
        ``errors``에 이름/사유로 모여 프론트에 전달된다.

        반환: ``{"workflows": [...], "errors": [...]}``
        - workflows 항목은 모두 ``configured=True`` (실패 파일은 제외됨).
          ``configured=False``는 이 카탈로그에서는 의미가 없으므로 API 호환을
          위해 항상 True로 내려보낸다.
        - ``id``/``source_name``은 파일명(확장자 포함, 예: ``foo.json``)으로 고정.
        """
        result: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        model_index = build_local_model_index(self.project_root / "comfy")
        for entry in list_soya_user_workflows(self.project_root):
            name = entry["name"]
            source_path = entry["source_path"]
            try:
                workflow = json.loads(Path(source_path).read_text(encoding="utf-8"))
                if not isinstance(workflow, dict) or not workflow:
                    raise ValueError(
                        f"워크플로우 JSON 객체가 비어 있거나 객체가 아닙니다: {name}"
                    )
                assets = resolve_workflow_model_files(
                    [workflow],
                    model_index,
                    include_hashes=False,
                )
            except Exception as exc:
                print(
                    "[MODAL] SOYA_USER 워크플로우 파싱/모델 해석 실패(제외): "
                    f"name={name}, path={source_path}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                errors.append(
                    {"name": name, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            result.append(
                {
                    "id": name,
                    "bindings": [],
                    "binding": "",
                    "configured": True,
                    "source_name": name,
                    "model_count": assets["model_count"],
                    "size_bytes": assets["size_bytes"],
                    "size_gib": assets["size_gib"],
                }
            )
        if errors:
            print(
                "[MODAL] SOYA_USER 워크플로우 처리 실패 요약: "
                f"failed={len(errors)}, names={[item['name'] for item in errors]}"
            )
        return {"workflows": result, "errors": errors}

    async def remote_workflows(self) -> dict[str, Any]:
        """로컬 SOYA_USER 목록을 원격 workflows Volume의 실제 파일과 비교한다."""

        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print("[MODAL] 원격 워크플로우 조회 실패: Modal이 비활성화되어 있습니다.")
            raise RuntimeError("Modal 사용을 켜고 설정을 저장하세요.")
        if not await self.account_connected(settings):
            print(
                "[MODAL] 원격 워크플로우 조회 실패: "
                f"Modal 계정이 연결되지 않았습니다. profile={settings.profile}"
            )
            raise RuntimeError("Modal 계정을 먼저 연결하세요.")

        local_payload, remote_payload = await asyncio.gather(
            asyncio.to_thread(self.workflows),
            self._run_client_action(
                settings,
                "list_workflows",
                timeout=120,
            ),
        )
        remote_items = remote_payload.get("workflows")
        if not isinstance(remote_items, list):
            print(
                "[MODAL] 원격 워크플로우 조회 결과 형식 오류: "
                f"type={type(remote_items).__name__}"
            )
            raise RuntimeError("Modal 원격 워크플로우 목록 형식이 올바르지 않습니다.")

        remote_by_name: dict[str, dict[str, Any]] = {}
        for item in remote_items:
            if not isinstance(item, Mapping):
                print(
                    "[MODAL] 원격 워크플로우 항목 형식 오류로 제외: "
                    f"type={type(item).__name__}, value={item!r}"
                )
                continue
            name = str(item.get("name") or "")
            if not name:
                print(f"[MODAL] 이름 없는 원격 워크플로우 항목 제외: {item!r}")
                continue
            remote_by_name[name] = dict(item)

        local_paths = {
            str(item["name"]): Path(str(item["source_path"]))
            for item in list_soya_user_workflows(self.project_root)
        }
        compared: list[dict[str, Any]] = []
        for local in local_payload["workflows"]:
            item = dict(local)
            source_path = local_paths.get(str(item.get("id") or ""))
            if source_path is None:
                print(
                    "[MODAL] 원격 비교 중 로컬 SOYA_USER 경로를 다시 찾지 못했습니다: "
                    f"workflow={item.get('id')!r}"
                )
                raise FileNotFoundError(
                    f"로컬 SOYA_USER 워크플로우 경로를 찾을 수 없습니다: {item.get('id')}"
                )
            try:
                local_sha256 = await asyncio.to_thread(self._sha256_file, source_path)
            except Exception as exc:
                print(
                    "[MODAL] 로컬 워크플로우 해시 계산 실패: "
                    f"path={source_path}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise
            remote = remote_by_name.get(str(item["id"]))
            item["local_sha256"] = local_sha256
            if remote is None:
                item.update(
                    remote_exists=False,
                    remote_available=False,
                    remote_sha256="",
                    remote_size=0,
                    sync_state="missing",
                )
            elif remote.get("valid") is not True:
                item.update(
                    remote_exists=True,
                    remote_available=False,
                    remote_sha256=str(remote.get("sha256") or ""),
                    remote_size=max(0, int(remote.get("size") or 0)),
                    remote_error=str(remote.get("error") or "원격 JSON 검증 실패"),
                    sync_state="invalid",
                )
            else:
                remote_sha256 = str(remote.get("sha256") or "")
                item.update(
                    remote_exists=True,
                    remote_available=True,
                    remote_sha256=remote_sha256,
                    remote_size=max(0, int(remote.get("size") or 0)),
                    remote_mtime=int(remote.get("mtime") or 0),
                    sync_state=(
                        "synced" if remote_sha256 == local_sha256 else "different"
                    ),
                )
            compared.append(item)

        counts = {
            state: sum(1 for item in compared if item.get("sync_state") == state)
            for state in ("synced", "different", "missing", "invalid")
        }
        print(
            "[MODAL] 원격 워크플로우 조회 완료: "
            f"local={len(compared)}, remote={len(remote_by_name)}, states={counts}"
        )
        return {
            "workflows": compared,
            "errors": list(local_payload.get("errors") or []),
            "remote_errors": list(remote_payload.get("errors") or []),
            "counts": counts,
            "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    async def _build_lora_catalog(
        self,
        config: Mapping[str, Any],
        *,
        include_hashes: bool,
        item_keys: list[str] | None = None,
        allow_missing_item_keys: bool = False,
    ) -> dict[str, Any]:
        async with self._model_sync_lock:
            return await asyncio.to_thread(
                build_local_lora_catalog,
                config,
                include_hashes=include_hashes,
                hash_cache=self._model_hash_cache,
                item_keys=item_keys,
                allow_missing_item_keys=allow_missing_item_keys,
            )

    async def lora_catalog(
        self,
        *,
        include_remote: bool = False,
        item_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        config = self.get_config()
        normalized_keys = list(
            dict.fromkeys(
                str(key).strip() for key in (item_keys or []) if str(key).strip()
            )
        )
        if len(normalized_keys) > 500:
            print(f"[MODAL_LORA] 상태 조회 선택 항목 수 초과: count={len(normalized_keys)}")
            raise ValueError("한 번에 조회할 수 있는 LoRA 항목은 최대 500개입니다.")
        settings: ModalSettings | None = None
        if include_remote:
            settings = ModalSettings.from_mapping(config)
            if not settings.enabled:
                print("[MODAL_LORA] 원격 상태 조회 실패: Modal이 비활성화되어 있습니다.")
                raise RuntimeError("Modal 사용을 켜고 설정을 저장하세요.")
            if not await self.account_connected(settings):
                print(
                    "[MODAL_LORA] 원격 상태 조회 실패: "
                    f"계정이 연결되지 않았습니다. profile={settings.profile}"
                )
                raise RuntimeError("Modal 계정을 먼저 연결하세요.")
        try:
            local_payload = await self._build_lora_catalog(
                config,
                include_hashes=include_remote,
                item_keys=normalized_keys or None,
                allow_missing_item_keys=bool(normalized_keys),
            )
        except Exception as exc:
            print(
                f"[MODAL_LORA] 로컬 카탈로그 조회 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
        payload: dict[str, Any] = dict(local_payload)
        checked_at = ""
        if include_remote:
            assert settings is not None
            remote_payload = await self._run_client_action(
                settings,
                "list_loras",
                timeout=180,
            )
            payload = merge_remote_lora_catalog(
                local_payload,
                remote_payload,
                item_keys=normalized_keys or None,
            )
            if normalized_keys:
                returned_keys = {
                    str(item.get("key") or "") for item in payload.get("items") or []
                }
                missing_keys = [key for key in normalized_keys if key not in returned_keys]
                if missing_keys:
                    print(
                        "[MODAL_LORA] 선택 상태 조회 항목이 로컬과 원격에 없습니다: "
                        f"keys={missing_keys}"
                    )
                    raise ValueError("선택한 LoRA 항목이 최신 목록에 없습니다.")
            checked_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            print(
                "[MODAL_LORA] 로컬/원격 상태 조회 완료: "
                f"scope={'selected' if normalized_keys else 'all'}, "
                f"items={len(payload.get('items') or [])}, counts={payload.get('counts')}"
            )
        public = public_lora_catalog(payload)
        return {
            "ok": True,
            **public,
            "checked_at": checked_at,
            "partial": bool(normalized_keys),
            "queried_item_keys": normalized_keys,
        }

    async def start_lora_operation(
        self,
        action: str,
        item_keys: list[str],
    ) -> dict[str, Any]:
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in {"upload", "sync", "delete"}:
            print(f"[MODAL_LORA] 지원하지 않는 작업 요청: action={action!r}")
            raise ValueError(f"지원하지 않는 LoRA 작업입니다: {action!r}")
        normalized_keys = list(dict.fromkeys(str(key).strip() for key in item_keys if str(key).strip()))
        if not normalized_keys:
            print("[MODAL_LORA] 선택 항목이 없는 작업 요청 거부")
            raise ValueError("LoRA 항목을 하나 이상 선택하세요.")
        if len(normalized_keys) > 500:
            print(f"[MODAL_LORA] 선택 항목 수 초과: count={len(normalized_keys)}")
            raise ValueError("한 번에 관리할 수 있는 LoRA 항목은 최대 500개입니다.")

        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print("[MODAL_LORA] 작업 시작 실패: Modal이 비활성화되어 있습니다.")
            raise RuntimeError("Modal 사용을 켜고 설정을 저장하세요.")
        async with self._deployment_action_lock:
            if self._lora_operation_running():
                raise RuntimeError("다른 LoRA 관리 작업이 이미 진행 중입니다.")
            if self._install_task and not self._install_task.done():
                raise RuntimeError("워크플로우·모델 동기화가 진행 중입니다.")
            if self._deployment_running():
                raise RuntimeError("Modal 앱 재배포가 진행 중입니다.")
            if not await self.account_connected(settings):
                raise RuntimeError("Modal 계정을 먼저 연결하세요.")

            config = self.get_config()
            local_payload, remote_payload = await asyncio.gather(
                self._build_lora_catalog(
                    config,
                    include_hashes=normalized_action in {"upload", "sync"},
                    item_keys=(
                        normalized_keys
                        if normalized_action in {"upload", "sync"}
                        else None
                    ),
                ),
                self._run_client_action(
                    settings,
                    "list_loras",
                    timeout=180,
                ),
            )
            catalog = merge_remote_lora_catalog(
                local_payload,
                remote_payload,
                item_keys=normalized_keys,
            )
            by_key = {str(item["key"]): item for item in catalog.get("items") or []}
            missing_keys = [key for key in normalized_keys if key not in by_key]
            if missing_keys:
                print(f"[MODAL_LORA] 최신 카탈로그에 없는 선택 항목: keys={missing_keys}")
                raise ValueError("선택한 LoRA 항목이 최신 목록에 없습니다. 상태를 다시 조회하세요.")
            selected = [by_key[key] for key in normalized_keys]
            if normalized_action in {"upload", "sync"}:
                remote_only = [item["name"] for item in selected if not item.get("files")]
                if remote_only:
                    print(
                        f"[MODAL_LORA] 로컬 파일 없는 {normalized_action} 요청 거부: "
                        f"items={remote_only}"
                    )
                    raise ValueError(
                        "원격에만 있는 항목은 업로드하거나 동기화할 수 없습니다: "
                        + ", ".join(remote_only)
                    )

            files_by_remote: dict[str, dict[str, Any]] = {}
            scopes: list[str] = []
            for item in selected:
                scopes.extend(str(scope) for scope in item.get("scopes") or [])
                for file_item in item.get("files") or []:
                    remote_path = str(file_item.get("remote_path") or "")
                    existing = files_by_remote.get(remote_path)
                    if existing and existing.get("source_path") != file_item.get("source_path"):
                        print(
                            "[MODAL_LORA] 선택 항목 간 원격 경로 충돌: "
                            f"remote={remote_path}, first={existing.get('source_path')}, "
                            f"second={file_item.get('source_path')}"
                        )
                        raise ValueError(f"선택한 LoRA의 원격 경로가 충돌합니다: {remote_path}")
                    files_by_remote[remote_path] = dict(file_item)
            files = list(files_by_remote.values())
            total_bytes = sum(max(0, int(item.get("size") or 0)) for item in files)
            action_labels = {"upload": "업로드", "sync": "동기화", "delete": "원격 삭제"}
            started_at = time.time()
            self._lora_operation_state = {
                "state": "running",
                "action": normalized_action,
                "action_label": action_labels[normalized_action],
                "message": f"선택한 LoRA {len(selected)}개 항목의 {action_labels[normalized_action]}를 준비하고 있습니다.",
                "item_keys": normalized_keys,
                "item_names": [str(item["name"]) for item in selected],
                "started_at": started_at,
                "updated_at": started_at,
                "progress": {
                    "completed_files": 0,
                    "total_files": len(files),
                    "completed_bytes": 0,
                    "total_bytes": total_bytes,
                    "uploaded_files": 0,
                    "skipped_files": 0,
                    "deleted_files": 0,
                    "current_item": "",
                },
                "logs": [],
            }
            self._append_lora_operation_log(
                "system",
                f"{action_labels[normalized_action]} 시작: 항목 {len(selected)}개 · "
                f"현재 사용 파일 {len(files)}개 · {total_bytes / 1024 ** 3:.2f} GiB",
            )
            self._lora_operation_task = asyncio.create_task(
                self._run_lora_operation(
                    settings,
                    normalized_action,
                    files,
                    list(dict.fromkeys(scopes)),
                )
            )
            return self._lora_operation_snapshot()

    async def _run_lora_operation(
        self,
        settings: ModalSettings,
        action: str,
        files: list[dict[str, Any]],
        scopes: list[str],
    ) -> None:
        try:
            request_payload = {
                "action": "manage_loras",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "mode": action,
                "lora_files": files,
                "scopes": scopes,
            }
            code, stdout, stderr = await self._run_command(
                [sys.executable, "-m", "modal_backend.client_cli"],
                env=self._subprocess_env(settings.profile),
                stdin_payload=request_payload,
                timeout=86_400,
                output_callback=self._handle_lora_client_output,
            )
            try:
                response = json.loads(stdout) if stdout.strip() else {}
            except json.JSONDecodeError as exc:
                print(
                    "[MODAL_LORA] 작업 응답 JSON 파싱 실패: "
                    f"action={action}, exit_code={code}, stderr={stderr[-1000:]}"
                )
                traceback.print_exc()
                raise RuntimeError("Modal LoRA 작업 응답 형식이 올바르지 않습니다.") from exc
            if code != 0 or not response.get("ok"):
                raise RuntimeError(
                    str(response.get("error") or f"Modal client exit_code={code}")
                )
            result = response.get("result")
            if not isinstance(result, Mapping):
                print(
                    "[MODAL_LORA] 작업 결과 객체 누락: "
                    f"action={action}, type={type(result).__name__}"
                )
                raise RuntimeError("Modal LoRA 작업 결과 객체가 없습니다.")
            progress = dict(self._lora_operation_state.get("progress") or {})
            progress.update(
                completed_files=int(progress.get("total_files") or 0),
                completed_bytes=int(progress.get("total_bytes") or 0),
                uploaded_files=max(0, int(result.get("uploaded") or 0)),
                skipped_files=max(0, int(result.get("skipped") or 0)),
                deleted_files=max(0, int(result.get("deleted_count") or 0)),
                current_item="",
            )
            finished_at = time.time()
            action_label = str(self._lora_operation_state.get("action_label") or action)
            self._lora_operation_state.update(
                state="completed",
                message=(
                    f"LoRA {action_label} 완료 · 업로드/갱신 {progress['uploaded_files']}개 · "
                    f"동일 {progress['skipped_files']}개 · 원격 삭제 {progress['deleted_files']}개"
                ),
                result=dict(result),
                progress=progress,
                finished_at=finished_at,
                updated_at=finished_at,
            )
            self._append_lora_operation_log("system", self._lora_operation_state["message"])
            print(
                "[MODAL_LORA] 작업 완료: "
                f"action={action}, items={len(self._lora_operation_state.get('item_keys') or [])}, "
                f"result={dict(result)}"
            )
        except Exception as exc:
            print(f"[MODAL_LORA] 작업 실패: action={action}, error={type(exc).__name__}: {exc}")
            traceback.print_exc()
            finished_at = time.time()
            self._lora_operation_state.update(
                state="failed",
                message=f"LoRA 작업 실패: {type(exc).__name__}: {exc}",
                error=str(exc),
                finished_at=finished_at,
                updated_at=finished_at,
            )
            self._append_lora_operation_log("error", f"{type(exc).__name__}: {exc}")

    def lora_operation_status(self) -> dict[str, Any]:
        return self._lora_operation_snapshot()

    async def _read_remote_workflow(
        self,
        settings: ModalSettings,
        workflow_name: str,
    ) -> tuple[dict[str, Any], str]:
        try:
            result = await self._run_client_action(
                settings,
                "read_workflow",
                timeout=120,
                workflow_name=workflow_name,
            )
        except ModalClientActionError as exc:
            if exc.error_type == "FileNotFoundError":
                print(
                    "[MODAL] 원격에 동기화되지 않은 워크플로우 실행 거부: "
                    f"workflow={workflow_name}"
                )
                raise FileNotFoundError(
                    f"{workflow_name}은(는) Modal에 동기화되지 않았습니다. "
                    "원격 조회 후 워크플로우를 먼저 동기화하세요."
                ) from exc
            raise
        workflow = result.get("workflow")
        if not isinstance(workflow, dict) or not workflow:
            print(
                "[MODAL] 원격 워크플로우 읽기 결과 검증 실패: "
                f"workflow={workflow_name}, type={type(workflow).__name__}"
            )
            raise RuntimeError(
                f"Modal 원격 워크플로우 JSON 객체가 올바르지 않습니다: {workflow_name}"
            )
        return workflow, str(result.get("sha256") or "")

    @staticmethod
    def _load_workflow_files(workflow_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        workflows: list[dict[str, Any]] = []
        for item in workflow_files:
            source = Path(str(item.get("source_path") or ""))
            try:
                workflow = json.loads(source.read_text(encoding="utf-8"))
            except Exception as exc:
                print(
                    "[MODAL] 사용자 워크플로우 읽기 실패: "
                    f"path={source}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise
            if not isinstance(workflow, dict) or not workflow:
                print(
                    "[MODAL] 사용자 워크플로우 JSON 객체가 비어 있습니다: "
                    f"path={source}, type={type(workflow).__name__}"
                )
                raise ValueError(f"사용자 워크플로우 JSON 객체가 비어 있습니다: {source}")
            workflows.append(workflow)
        if not workflows:
            print("[MODAL] 모델 동기화 대상 사용자 워크플로우가 없습니다.")
            raise ValueError("Modal에 동기화할 사용자 워크플로우가 없습니다.")
        return workflows

    async def _resolve_local_workflow_assets(
        self,
        workflows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        async with self._model_sync_lock:
            def resolve() -> dict[str, Any]:
                model_index = build_local_model_index(self.project_root / "comfy")
                return resolve_workflow_model_files(
                    workflows,
                    model_index,
                    hash_cache=self._model_hash_cache,
                    include_hashes=True,
                )

            assets = await asyncio.to_thread(resolve)
        print(
            "[MODAL_SYNC] 로컬 워크플로우 모델 해석 완료: "
            f"models={len(assets['model_files'])}, loras={len(assets['lora_files'])}, "
            f"size_gib={assets['size_gib']}"
        )
        return assets

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

            async with self._status_command_semaphore:
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

    async def _sync_models_direct(
        self,
        settings: ModalSettings,
        workflow_ids: list[str],
    ) -> None:
        """워커가 저장소에서 Modal Volume 으로 모델을 직접 받게 한다 (cloud_direct).

        로컬 디스크를 거치지 않으므로 로컬에서 생성하지 않는 구성에서 같은 바이트를
        두 번 옮기지 않는다. 무결성은 워커가 매니페스트 sha256 으로 검증한다.
        """

        model_ids = await asyncio.to_thread(
            model_ids_for_workflow_files,
            self.project_root,
            list(workflow_ids),
        )
        if not model_ids:
            self._append_install_log("system", "cloud_direct: 필요한 모델이 없습니다.")
            return
        self._set_install_phase(
            "upload",
            f"저장소에서 Modal Volume 으로 모델 {len(model_ids)}개를 직접 받는 중입니다.",
            progress_mode="indeterminate",
        )
        self._append_install_log(
            "system",
            f"cloud_direct 직접 다운로드 시작: 모델 {len(model_ids)}개",
        )
        client_payload = {
            "action": "sync_models_direct",
            "app_name": settings.deployment_name,
            "environment": settings.environment,
            "model_ids": model_ids,
        }
        code, stdout, _stderr = await self._run_command(
            [sys.executable, "-m", "modal_backend.client_cli"],
            env=self._subprocess_env(settings.profile),
            stdin_payload=client_payload,
            timeout=86_400,
            output_callback=self._handle_install_client_output,
        )
        if code != 0:
            raise RuntimeError("Modal 원격 모델 직접 다운로드에 실패했습니다.")
        response = json.loads(stdout)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "원격 모델 다운로드 실패"))
        results = (response.get("result") or {}).get("results") or []
        downloaded = [r for r in results if r.get("state") == "downloaded"]
        present = [r for r in results if r.get("state") == "already_present"]
        problems = [
            r for r in results
            if r.get("state") not in {"downloaded", "already_present"}
        ]
        summary = (
            f"cloud_direct 완료: 신규 {len(downloaded)}개 · 기존 {len(present)}개"
        )
        if problems:
            summary += f" · 문제 {len(problems)}개"
        self._append_install_log("system", summary)
        for item in problems:
            self._append_install_log(
                "system",
                f"⚠ 모델 처리 실패: id={item.get('id')}, state={item.get('state')}"
                + (f", error={item.get('error')}" if item.get("error") else ""),
            )
        if problems:
            raise RuntimeError(
                f"원격 모델 {len(problems)}개를 확보하지 못했습니다: "
                + ", ".join(str(item.get("id")) for item in problems[:5])
            )

    async def start_install(self, selected_names: list[str]) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            raise RuntimeError("외부 API 설정에서 Modal 사용을 먼저 켜고 저장하세요.")
        async with self._deployment_action_lock:
            if not await self.account_connected(settings):
                raise RuntimeError("Modal 계정을 먼저 연결하세요.")
            if self._install_task and not self._install_task.done():
                raise RuntimeError("Modal 동기화가 이미 진행 중입니다.")
            if self._lora_operation_running():
                raise RuntimeError("Modal LoRA 관리 작업이 진행 중입니다.")
            if self._deployment_running():
                raise RuntimeError("Modal 재배포가 진행 중입니다. 완료 후 동기화하세요.")
            # selected_names: SOYA_USER 파일명(확장자 포함, 예: foo.json). config 바인딩
            # 아님. plan_from_soya_user_names가 파일명 전용 강제 + SOYA_USER 하위 검증.
            plan = plan_from_soya_user_names(self.project_root, selected_names)
            started_at = time.time()
            self._install_state = {
                "state": "running",
                "phase": "assets",
                "phase_label": INSTALL_PHASE_LABELS["assets"],
                "phase_index": 0,
                "phase_count": len(INSTALL_PHASE_LABELS),
                "message": "SOYA_USER 워크플로우의 현재 로컬 모델을 확인하고 있습니다.",
                "workflow_ids": plan["workflow_ids"],
                "size_gib": 0.0,
                "size_bytes": 0,
                "started_at": started_at,
                "updated_at": started_at,
                "progress": {
                    "mode": "indeterminate",
                    "completed_files": 0,
                    "total_files": len(plan["workflow_files"]),
                    "completed_bytes": 0,
                    "total_bytes": 0,
                    "uploaded_files": 0,
                    "skipped_files": 0,
                    "current_label": "워크플로우와 모델 확인",
                    "current_item": "",
                },
                "logs": [],
            }
            self._append_install_log(
                "system",
                f"동기화 시작: 워크플로우 {len(plan['workflow_ids'])}개",
            )
            self._install_task = asyncio.create_task(self._run_install(settings, plan))
            return self._install_snapshot()

    async def _run_install(self, settings: ModalSettings, plan: dict[str, Any]) -> None:
        try:
            workflows = await asyncio.to_thread(
                self._load_workflow_files,
                plan["workflow_files"],
            )
            cloud_direct = settings.model_source == MODEL_SOURCE_CLOUD_DIRECT
            if cloud_direct:
                # 로컬을 원본으로 쓰지 않는다. 매니페스트만 보고 필요한 모델을 정하고
                # 워커가 저장소에서 볼륨으로 직접 받는다. 로컬에 모델이 하나도 없어도 된다.
                assets = {
                    "model_files": [],
                    "lora_files": [],
                    "model_count": 0,
                    "size_bytes": 0,
                    "size_gib": 0.0,
                }
                self._append_install_log(
                    "system",
                    "모델 취득 경로: cloud_direct — 로컬을 거치지 않고 저장소에서 "
                    "Modal Volume 으로 직접 받습니다.",
                )
            else:
                assets = await self._resolve_local_workflow_assets(workflows)
            plan.update(assets)
            workflow_bytes = sum(
                Path(str(item["source_path"])).stat().st_size
                for item in plan["workflow_files"]
            )
            total_files = (
                len(plan["workflow_files"])
                + len(plan["model_files"])
                + len(plan["lora_files"])
            )
            total_bytes = workflow_bytes + sum(
                max(0, int(item.get("size") or 0))
                for item in plan["model_files"] + plan["lora_files"]
            )
            self._install_state.update(
                model_count=plan["model_count"],
                size_gib=plan["size_gib"],
                size_bytes=total_bytes,
                progress={
                    "mode": "indeterminate",
                    "completed_files": 0,
                    "total_files": total_files,
                    "completed_bytes": 0,
                    "total_bytes": total_bytes,
                    "uploaded_files": 0,
                    "skipped_files": 0,
                    "current_label": "동기화 준비",
                    "current_item": "",
                },
            )
            self._append_install_log(
                "system",
                f"자산 분석 완료: 전송 대상 {total_files}개 · {total_bytes / 1024 ** 3:.2f} GiB",
            )
            self._set_install_phase(
                "upload",
                "사용자 워크플로우와 로컬 모델을 Modal Volume에 동기화하고 있습니다.",
                progress_mode="determinate",
            )
            self._append_install_log(
                "system",
                "앱 재배포 없이 Modal Volume 파일 동기화 시작",
            )
            client_payload = {
                "action": "install",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "workflow_files": plan["workflow_files"],
                "model_files": plan["model_files"],
                "lora_files": plan["lora_files"],
            }
            code, stdout, _stderr = await self._run_command(
                [sys.executable, "-m", "modal_backend.client_cli"],
                env=self._subprocess_env(settings.profile),
                stdin_payload=client_payload,
                timeout=86_400,
                output_callback=self._handle_install_client_output,
            )
            if code != 0:
                print(
                    f"[MODAL] 워크플로우/모델 동기화 실패: app={settings.deployment_name}, "
                    f"exit_code={code}"
                )
                raise RuntimeError("Modal 워크플로우 또는 모델 동기화에 실패했습니다.")
            response = json.loads(stdout)
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "Modal 원격 동기화 실패"))
            if cloud_direct:
                await self._sync_models_direct(settings, plan["workflow_ids"])
            progress = dict(self._install_state.get("progress") or {})
            progress.update(
                mode="complete",
                completed_files=int(progress.get("total_files") or total_files),
                completed_bytes=int(progress.get("total_bytes") or total_bytes),
                current_label="동기화 완료",
                current_item="",
            )
            self._install_state["progress"] = progress
            self._set_install_phase(
                "complete",
                "사용자 워크플로우와 로컬 모델의 Modal 업로드가 완료되었습니다.",
                progress_mode="complete",
            )
            finished_at = time.time()
            self._install_state.update(
                state="completed",
                finished_at=finished_at,
                updated_at=finished_at,
            )
            self._append_install_log("system", "Modal 동기화가 완료되었습니다.")
            print(
                f"[MODAL] 동기화 완료: app={settings.deployment_name}, "
                f"workflows={len(plan['workflow_ids'])}, models={plan['model_count']}, "
                f"size_gib={plan['size_gib']}"
            )
        except Exception as exc:
            print(f"[MODAL] 동기화 작업 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            finished_at = time.time()
            self._install_state.update(
                state="failed",
                message=f"Modal 동기화 실패: {type(exc).__name__}: {exc}",
                workflow_ids=plan.get("workflow_ids", []),
                finished_at=finished_at,
                updated_at=finished_at,
            )
            self._append_install_log(
                "error",
                f"{type(exc).__name__}: {exc}",
            )

    async def start_redeploy(self, *, force_custom_nodes: bool = False) -> dict[str, Any]:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            raise RuntimeError("외부 API 설정에서 Modal 사용을 먼저 켜고 저장하세요.")
        async with self._deployment_action_lock:
            requested_at = time.time()
            if self._deployment_running():
                return self._deployment_snapshot()
            if self._install_task and not self._install_task.done():
                raise RuntimeError("Modal 워크플로우 동기화가 진행 중입니다.")
            if self._lora_operation_running():
                raise RuntimeError("Modal LoRA 관리 작업이 진행 중입니다.")
            if self._web_task and not self._web_task.done():
                raise RuntimeError("Modal ComfyUI 시작 또는 종료가 진행 중입니다.")
            if not await self.account_connected(settings):
                raise RuntimeError("Modal 계정을 먼저 연결하세요.")
            inventory = await self._inventory_for_deploy()
            public_inventory = public_custom_node_inventory(inventory)
            kind = "custom_nodes" if force_custom_nodes else "redeploy"
            message = (
                "로컬 custom node를 반영할 Modal 이미지 강제 빌드를 시작합니다."
                if force_custom_nodes
                else "Modal 작업 App과 웹 App 재배포를 시작합니다."
            )
            self._deployment_state = {
                "state": "running",
                "kind": kind,
                "phase": "worker",
                "message": message,
                "inventory": public_inventory,
                "started_at": requested_at,
                "updated_at": time.time(),
                "logs": [],
            }
            summary = public_inventory.get("summary") or {}
            self._append_deployment_log(
                "system",
                "custom node 인벤토리: "
                f"manifest {int(summary.get('manifest') or 0)}개 · "
                f"추가 Git {int(summary.get('git') or 0)}개 · "
                f"로컬 복사 {int(summary.get('local') or 0)}개 · "
                f"제외 {int(summary.get('skipped') or 0)}개",
            )
            for warning in public_inventory.get("warnings", []):
                self._append_deployment_log("warning", str(warning))
            for skipped in public_inventory.get("skipped", []):
                if isinstance(skipped, Mapping):
                    self._append_deployment_log(
                        "warning",
                        f"제외: {skipped.get('name')} · {skipped.get('reason')}",
                    )
            self._deployment_task = asyncio.create_task(
                self._run_redeploy(
                    settings,
                    inventory,
                    force_custom_nodes=force_custom_nodes,
                )
            )
            return self._deployment_snapshot()

    async def _run_redeploy(
        self,
        settings: ModalSettings,
        inventory: Mapping[str, Any],
        *,
        force_custom_nodes: bool,
    ) -> None:
        try:
            self._append_deployment_log(
                "system",
                "Modal 작업 App 이미지 빌드·배포를 시작합니다.",
            )
            await self._deploy_worker_app(
                settings,
                custom_node_inventory=inventory,
                force_custom_node_build=force_custom_nodes,
                output_callback=self._append_deployment_log,
            )
            self._deployment_state.update(
                phase="web",
                message="작업 App 배포 완료 · 웹 App을 재배포하고 있습니다.",
                updated_at=time.time(),
            )
            self._append_deployment_log(
                "system",
                "Modal 작업 App 배포 완료 · 웹 App 배포를 시작합니다.",
            )
            # 작업 App의 강제 빌드 결과가 동일한 이미지 캐시에 기록되므로 웹 App은
            # 같은 인벤토리를 사용하되 두 번째 강제 빌드는 반복하지 않는다.
            await self._deploy_web_app(
                settings,
                custom_node_inventory=inventory,
                force_custom_node_build=False,
                output_callback=self._append_deployment_log,
            )
            self._deployment_state.update(
                phase="shutdown",
                message="웹 App 배포 완료 · 비용 안전을 위해 웹 App을 종료하고 있습니다.",
                updated_at=time.time(),
            )
            self._append_deployment_log(
                "system",
                f"Modal ComfyUI 웹 App과 {settings.web_gpu} 자동 종료를 시작합니다.",
            )
            await self._stop_web_app(settings)
            stopped_at = time.time()
            self._web_state = {
                "available": True,
                "deployed": False,
                "state": "stopped",
                "reason": "app_not_deployed",
                "message": (
                    "재배포 완료 후 Modal ComfyUI 웹 App과 "
                    f"{settings.web_gpu}를 자동 종료했습니다."
                ),
                "app_name": self._web_app_name(settings),
                "num_total_runners": 0,
                "num_running_inputs": 0,
                "backlog": 0,
                "updated_at": stopped_at,
            }
            self._append_deployment_log(
                "system",
                (
                    f"Modal ComfyUI 웹 App과 {settings.web_gpu} "
                    f"자동 종료 완료 · {settings.web_gpu} 0개"
                ),
            )
            finished_at = time.time()
            self._deployment_state.update(
                state="completed",
                phase="complete",
                message=(
                    "Custom node 동기화와 두 Modal App 재배포를 완료하고 웹 App을 자동 종료했습니다."
                    if force_custom_nodes
                    else "Modal 작업 App과 웹 App 재배포를 완료하고 웹 App을 자동 종료했습니다."
                ),
                finished_at=finished_at,
                updated_at=finished_at,
            )
            self._append_deployment_log("system", self._deployment_state["message"])
            print(
                "[MODAL] 재배포 완료: "
                f"app={settings.deployment_name}, force_custom_nodes={force_custom_nodes}"
            )
        except Exception as exc:
            print(f"[MODAL] 재배포 작업 예외: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            finished_at = time.time()
            self._deployment_state.update(
                state="failed",
                message=f"Modal 재배포 실패: {type(exc).__name__}: {exc}",
                error=str(exc),
                finished_at=finished_at,
                updated_at=finished_at,
            )
            self._append_deployment_log(
                "error",
                f"{type(exc).__name__}: {exc}",
            )

    async def _apply_worker_autoscaler(
        self,
        settings: ModalSettings,
        *,
        min_containers: int,
    ) -> dict[str, Any]:
        return await self._run_client_action(
            settings,
            "update_autoscaler",
            timeout=60,
            max_containers=settings.max_concurrency,
            worker_min_containers=min_containers,
            scaledown_window_seconds=settings.scaledown_window_seconds,
        )

    async def acquire_worker_warm_lease(
        self,
        *,
        reason: str = "illustration_llm_build",
    ) -> str | None:
        """최대 병렬 수만큼 ComfyWorker를 예열하고 공유 lease 토큰을 반환한다."""

        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print(
                "[MODAL_WARM] lease 획득 생략: Modal이 비활성화되어 있습니다. "
                f"reason={reason}"
            )
            return None

        token = uuid.uuid4().hex
        async with self._worker_warm_lease_lock:
            if not await self.account_connected(settings):
                print(
                    "[MODAL_WARM] lease 획득 생략: Modal 계정이 연결되지 않았습니다. "
                    f"profile={settings.profile}, reason={reason}"
                )
                return None

            target_min = settings.max_concurrency
            if (
                not self._worker_warm_leases
                or self._worker_warm_pool_applied_min != target_min
            ):
                result = await self._apply_worker_autoscaler(
                    settings,
                    min_containers=target_min,
                )
                self._worker_warm_pool_applied_min = target_min
                print(
                    "[MODAL_WARM] ComfyWorker 예열 시작: "
                    f"min={target_min}, max={settings.max_concurrency}, "
                    f"gpu={settings.worker_gpu}, reason={reason}, result={result}"
                )

            self._worker_warm_leases[token] = reason
            active_count = len(self._worker_warm_leases)

        print(
            "[MODAL_WARM] lease 획득: "
            f"token={token}, active={active_count}, min={target_min}, reason={reason}"
        )
        return token

    def _schedule_worker_warm_pool_reset(self) -> None:
        if self._worker_warm_reset_task and not self._worker_warm_reset_task.done():
            print("[MODAL_WARM] scale-to-zero 재시도가 이미 예약되어 있습니다.")
            return

        task = asyncio.create_task(self._retry_worker_warm_pool_reset())
        self._worker_warm_reset_task = task

        def clear_reset_task(completed: asyncio.Task) -> None:
            if self._worker_warm_reset_task is completed:
                self._worker_warm_reset_task = None
            if completed.cancelled():
                print("[MODAL_WARM] scale-to-zero 재시도 작업이 취소되었습니다.")
                return
            try:
                completed.exception()
            except Exception as exc:
                print(
                    "[MODAL_WARM] scale-to-zero 재시도 작업 상태 회수 실패: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        task.add_done_callback(clear_reset_task)

    async def _retry_worker_warm_pool_reset(self) -> None:
        for attempt, delay in enumerate((1.0, 5.0, 15.0), start=1):
            await asyncio.sleep(delay)
            async with self._worker_warm_lease_lock:
                if self._worker_warm_leases:
                    print(
                        "[MODAL_WARM] scale-to-zero 재시도 중단: "
                        f"새 lease {len(self._worker_warm_leases)}개가 활성 상태입니다."
                    )
                    return
                try:
                    settings = ModalSettings.from_mapping(self.get_config())
                    if not await self.account_connected(settings):
                        raise RuntimeError("Modal 계정이 연결되지 않았습니다")
                    result = await self._apply_worker_autoscaler(
                        settings,
                        min_containers=0,
                    )
                    self._worker_warm_pool_applied_min = 0
                    print(
                        "[MODAL_WARM] scale-to-zero 재시도 성공: "
                        f"attempt={attempt}, result={result}"
                    )
                    return
                except Exception as exc:
                    print(
                        "[MODAL_WARM] scale-to-zero 재시도 실패: "
                        f"attempt={attempt}/3, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
        print(
            "[MODAL_WARM] scale-to-zero 복구 최종 실패: "
            "Modal autoscaler의 min_containers 상태를 관리 화면에서 확인하세요."
        )

    async def release_worker_warm_lease(
        self,
        token: str,
        *,
        reason: str = "illustration_llm_build",
    ) -> bool:
        """lease를 해제하고 마지막 요청이면 ComfyWorker를 scale-to-zero로 복구한다."""

        reset_required = False
        async with self._worker_warm_lease_lock:
            stored_reason = self._worker_warm_leases.pop(token, None)
            if stored_reason is None:
                print(
                    "[MODAL_WARM] lease 해제 실패: 알 수 없거나 이미 해제된 토큰 "
                    f"token={token!r}, reason={reason}"
                )
                return False

            active_count = len(self._worker_warm_leases)
            if active_count:
                print(
                    "[MODAL_WARM] lease 해제, warm pool 유지: "
                    f"token={token}, active={active_count}, reason={stored_reason}"
                )
                return True

            try:
                settings = ModalSettings.from_mapping(self.get_config())
                if not await self.account_connected(settings):
                    raise RuntimeError("Modal 계정이 연결되지 않았습니다")
                result = await self._apply_worker_autoscaler(
                    settings,
                    min_containers=0,
                )
                self._worker_warm_pool_applied_min = 0
                print(
                    "[MODAL_WARM] 마지막 lease 해제 · scale-to-zero 복구: "
                    f"token={token}, reason={stored_reason}, result={result}"
                )
            except Exception as exc:
                reset_required = True
                self._worker_warm_pool_applied_min = -1
                print(
                    "[MODAL_WARM] 마지막 lease 해제 후 scale-to-zero 복구 실패: "
                    f"token={token}, reason={stored_reason}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        if reset_required:
            self._schedule_worker_warm_pool_reset()
            return False
        return True

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
            async with self._worker_warm_lease_lock:
                worker_min = (
                    settings.max_concurrency
                    if self._worker_warm_leases
                    else 0
                )
                result = await self._apply_worker_autoscaler(
                    settings,
                    min_containers=worker_min,
                )
                self._worker_warm_pool_applied_min = worker_min
            self._autoscaler_state = {
                "state": "completed",
                "message": (
                    f"최대 {settings.max_concurrency}개 · 유휴 "
                    f"{settings.scaledown_window_seconds}초 · 임시 최소 "
                    f"{worker_min}개로 적용되었습니다."
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
            raise RuntimeError(
                f"{settings.worker_gpu} 연결 테스트가 이미 진행 중입니다."
            )
        self._probe_state = {
            "state": "running",
            "message": (
                f"{settings.worker_gpu} 컨테이너를 깨우고 CUDA를 확인하고 있습니다."
            ),
            "updated_at": time.time(),
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
            sageattention = result.get("sageattention")
            sage_version = (
                str(sageattention.get("version") or "확인됨")
                if isinstance(sageattention, Mapping)
                else "확인 실패"
            )
            self._probe_state = {
                "state": "completed",
                "message": (
                    f"{result.get('device') or settings.worker_gpu} · "
                    f"VRAM {vram_gib} GiB · "
                    f"CUDA {result.get('cuda') or '-'} · "
                    f"SageAttention {sage_version} 실제 커널 확인"
                ),
                "updated_at": time.time(),
                **result,
            }
            print(
                f"[MODAL] {settings.worker_gpu} 연결 테스트 완료: "
                f"{self._probe_state['message']}"
            )
        except Exception as exc:
            print(
                f"[MODAL] {settings.worker_gpu} 연결 테스트 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            self._probe_state = {
                "state": "failed",
                "message": (
                    f"{settings.worker_gpu} 연결 테스트 실패: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "updated_at": time.time(),
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
        # 변환은 로컬 모델 자산 해석·동기화 없이 원격 ComfyUI에 바로 맡긴다.
        # 모델 동기화는 사용자가 수동 install로만 수행한다.
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
        # workflow_id: SOYA_USER 파일명(확장자 포함). config 바인딩 아님.
        plan = plan_from_soya_user_names(self.project_root, [normalized_id])
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
        remote_workflow, remote_sha256 = await self._read_remote_workflow(
            settings,
            normalized_id,
        )
        # 실행은 항상 Volume 의 원격 사본으로 한다. 로컬 JSON 을 고쳐도
        # 재동기화(/api/modal/install) 전에는 반영되지 않는데, 지금까지는
        # 아무 표시 없이 옛 사본으로 돌아 원인을 찾기 어려웠다.
        # 그래서 실행 전에 로컬 파일 해시와 비교해 상태에 남긴다.
        local_source = Path(plan["workflow_files"][0]["source_path"])
        local_sha256 = ""
        stale = False
        try:
            local_sha256 = await asyncio.to_thread(self._sha256_file, local_source)
            stale = bool(local_sha256) and bool(remote_sha256) and local_sha256 != remote_sha256
        except Exception as exc:
            # 해시 비교는 진단용이다. 실패해도 실행을 막지 않는다.
            print(
                "[MODAL] 로컬 워크플로우 해시 계산 실패(동기화 비교 생략): "
                f"path={local_source}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
        if stale:
            print(
                "[MODAL] ⚠ 원격 워크플로우가 로컬과 다릅니다 — 옛 사본으로 실행됩니다: "
                f"workflow={normalized_id}, local_sha256={local_sha256[:12]}, "
                f"remote_sha256={remote_sha256[:12]}. "
                "최신 내용으로 실행하려면 워크플로우를 다시 동기화하세요."
            )
        job_id = uuid.uuid4().hex
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state = {
            "job_id": job_id,
            "workflow_id": normalized_id,
            "source_name": local_source.name,
            "remote_sha256": remote_sha256,
            "local_sha256": local_sha256,
            "workflow_stale": stale,
            "state": "queued",
            "phase": "queued",
            "message": (
                "⚠ 원격 사본이 로컬 파일과 다릅니다. 옛 사본으로 실행합니다 — "
                "재동기화가 필요할 수 있습니다."
                if stale
                else "Modal 워크플로우 실행을 준비하고 있습니다."
            ),
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
        task = asyncio.create_task(
            self._run_saved_workflow(settings, state, remote_workflow)
        )
        self._workflow_run_tasks[job_id] = task
        return self._workflow_run_public(state)

    async def _run_saved_workflow(
        self,
        settings: ModalSettings,
        state: dict[str, Any],
        workflow: dict[str, Any],
    ) -> None:
        job_id = str(state["job_id"])
        try:
            state.update(
                state="running",
                phase="loading",
                message="Modal Volume의 원격 워크플로우 JSON을 준비하고 있습니다.",
            )
            if not self._is_api_workflow(workflow):
                state.update(
                    phase="converting",
                    message="원격 ComfyUI에서 워크플로우를 API 형식으로 변환하고 있습니다.",
                )
                workflow = await self.convert_workflow(workflow)
            state.update(
                phase="generating",
                message=(
                    "로컬 모델과 입력 이미지를 동기화하고 "
                    f"{settings.worker_gpu}에서 실행하고 있습니다."
                ),
            )
            image_bytes, metadata = await self.generate(workflow)
            state.update(
                state="completed",
                phase="completed",
                message="Modal 워크플로우 실행이 완료되었습니다.",
                completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                prompt_id=metadata.get("prompt_id"),
                content_type=metadata.get("content_type") or "image/png",
                model_sync=metadata.get("model_sync") or {},
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
        video_job_id: str | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
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
        # 모델/LoRA 동기화는 수동 install에서만 수행한다. 실행 경로에서는 로컬
        # 모델 색인 스캔과 해시 계산을 건너뛰고 입력 파일 해석만 한다.
        workflow_input_files, explicit_input_files = await asyncio.gather(
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
        deferred_artifacts = bool(artifact_prefixes and not require_images)

        progress_queue: asyncio.Queue[dict[str, Any] | None] | None = (
            asyncio.Queue() if progress_callback is not None else None
        )

        async def consume_progress() -> None:
            if progress_queue is None or progress_callback is None:
                return
            while True:
                event = await progress_queue.get()
                if event is None:
                    return
                try:
                    callback_result = progress_callback(event)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                except Exception as exc:
                    print(
                        "[MODAL] 워크플로우 진행 콜백 실패: "
                        f"event={event!r}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

        def handle_client_output(source: str, line: str) -> None:
            if source != "stderr" or progress_queue is None:
                return
            event: dict[str, Any] | None = None
            raw_event = ""
            try:
                if line.startswith(WORKFLOW_PROGRESS_PREFIX):
                    raw_event = line[len(WORKFLOW_PROGRESS_PREFIX) :]
                    parsed = json.loads(raw_event)
                    if not isinstance(parsed, dict):
                        raise TypeError("Modal 학습 진행 이벤트는 객체여야 합니다.")
                    event = parsed
                if event is not None:
                    progress_queue.put_nowait(event)
            except Exception as exc:
                print(
                    "[MODAL] 워크플로우 실시간 이벤트 파싱 실패: "
                    f"source={source}, error={type(exc).__name__}: {exc}, "
                    f"payload={raw_event[:500]!r}"
                )
                traceback.print_exc()

        progress_consumer = (
            asyncio.create_task(consume_progress())
            if progress_queue is not None
            else None
        )
        with tempfile.TemporaryDirectory(prefix="soya-modal-output-") as output_dir:
            payload = {
                "action": "generate",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "worker_gpu": settings.worker_gpu,
                "vram_mode": settings.vram_mode,
                "workflow": workflow,
                "input_files": input_files,
                "artifact_prefixes": list(artifact_prefixes or []),
                "require_images": bool(require_images),
                "defer_artifacts": deferred_artifacts,
                "video_job_id": video_job_id,
                "timeout_seconds": max(30, min(int(timeout_seconds), 3_300)),
                "container_start_max_retries": (
                    settings.container_start_max_retries
                ),
                "output_dir": output_dir,
            }
            try:
                command_kwargs: dict[str, Any] = {
                    "env": self._subprocess_env(settings.profile),
                    "stdin_payload": payload,
                    "timeout": payload["timeout_seconds"] + 180,
                }
                if progress_queue is not None:
                    command_kwargs["output_callback"] = handle_client_output
                code, stdout, stderr = await self._run_command(
                    [sys.executable, "-m", "modal_backend.client_cli"],
                    **command_kwargs,
                )
            finally:
                if progress_queue is not None:
                    progress_queue.put_nowait(None)
                if progress_consumer is not None:
                    await progress_consumer
            if code != 0:
                failure_response: dict[str, Any] = {}
                if stdout.strip():
                    try:
                        parsed_failure = json.loads(stdout)
                        if isinstance(parsed_failure, dict):
                            failure_response = parsed_failure
                        else:
                            print(
                                "[MODAL] 실패 응답 JSON 루트가 객체가 아님: "
                                f"type={type(parsed_failure).__name__}"
                            )
                    except json.JSONDecodeError as exc:
                        print(
                            f"[MODAL] 실패 응답 JSON 파싱 실패: {exc}, "
                            f"stdout_length={len(stdout)}"
                        )
                        traceback.print_exc()
                failure_error = str(
                    failure_response.get("error")
                    or "Modal 원격 이미지 생성에 실패했습니다. 서버 로그를 확인하세요."
                )
                print(
                    f"[MODAL] 원격 생성 실패: app={settings.deployment_name}, "
                    f"exit_code={code}, inputs={len(input_files)}, "
                    f"error={failure_error}, "
                    f"stderr={stderr[-2000:]}"
                )
                raise RuntimeError(failure_error)
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
            raw_artifacts = list(result.get("artifacts") or [])
            video_artifacts = list(result.get("video_artifacts") or [])
            if video_job_id and len(video_artifacts) != 1:
                print(
                    "[MODAL:VIDEO] 원격 MP4 artifact 수 검증 실패: "
                    f"prompt_id={result.get('prompt_id')}, job={video_job_id!r}, "
                    f"count={len(video_artifacts)}, artifacts={video_artifacts!r}"
                )
                raise RuntimeError(
                    "Modal 영상 생성 결과 MP4 artifact가 정확히 하나가 아닙니다."
                )
            stored_artifacts = (
                []
                if deferred_artifacts
                else self._store_modal_artifacts(raw_artifacts, config)
            )
            print(
                f"[MODAL] 원격 워크플로우 완료: app={settings.deployment_name}, "
                f"prompt_id={result.get('prompt_id')}, images={len(images)}, "
                f"artifacts={len(stored_artifacts)}, "
                f"video_artifacts={len(video_artifacts)}, "
                f"model_sync={result.get('model_sync')}, "
                f"lora_sync={result.get('lora_sync')}"
            )
            return {
                "prompt_id": result.get("prompt_id"),
                "model_sync": result.get("model_sync") or {},
                "lora_sync": result.get("lora_sync") or {},
                "images": images,
                "artifacts": stored_artifacts,
                "deferred_artifacts": raw_artifacts if deferred_artifacts else [],
                "video_artifacts": video_artifacts,
                "text_outputs": list(result.get("text_outputs") or []),
            }

    @staticmethod
    def _normalize_video_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(artifact, Mapping):
            print(
                "[MODAL:VIDEO] 영상 artifact 형식 오류: "
                f"type={type(artifact).__name__}, value={artifact!r}"
            )
            raise TypeError("Modal 영상 artifact는 객체여야 합니다.")
        remote_path = (
            str(artifact.get("remote_path") or "")
            .strip()
            .replace("\\", "/")
            .strip("/")
        )
        parts = remote_path.split("/") if remote_path else []
        filename = Path(str(artifact.get("filename") or "")).name
        sha256 = str(artifact.get("sha256") or "").strip().lower()
        try:
            size = int(artifact.get("size"))
        except (TypeError, ValueError) as exc:
            print(
                "[MODAL:VIDEO] 영상 artifact 크기 형식 오류: "
                f"remote={remote_path!r}, size={artifact.get('size')!r}"
            )
            traceback.print_exc()
            raise ValueError("Modal 영상 artifact 크기가 올바르지 않습니다.") from exc
        if (
            len(parts) < 3
            or parts[0] != "SOYA_VIDEO_OUTPUT"
            or any(part in ("", ".", "..") for part in parts)
            or not remote_path.casefold().endswith(".mp4")
            or not filename.casefold().endswith(".mp4")
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or size <= 0
        ):
            print(
                "[MODAL:VIDEO] 안전하지 않은 영상 artifact: "
                f"remote={remote_path!r}, filename={filename!r}, "
                f"size={size}, sha256={sha256!r}"
            )
            raise ValueError("Modal 영상 artifact 메타데이터가 올바르지 않습니다.")
        return {
            "remote_path": remote_path,
            "filename": filename,
            "size": size,
            "sha256": sha256,
            "node_id": str(artifact.get("node_id") or ""),
        }

    async def download_video_artifact(
        self,
        artifact: Mapping[str, Any],
    ) -> tuple[bytes, dict[str, Any]]:
        """Video Volume의 MP4를 크기·SHA256 검증 후 로컬 메모리로 가져온다."""

        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print("[MODAL:VIDEO] MP4 다운로드 실패: Modal이 비활성화되어 있습니다.")
            raise RuntimeError("Modal 영상 다운로드 중 Modal이 비활성화되었습니다.")
        if not await self.account_connected(settings):
            print(
                "[MODAL:VIDEO] MP4 다운로드 실패: Modal 계정이 연결되지 않았습니다. "
                f"profile={settings.profile}"
            )
            raise RuntimeError("Modal 계정이 연결되어 있지 않습니다.")
        normalized = self._normalize_video_artifact(artifact)
        with tempfile.TemporaryDirectory(prefix="soya-modal-video-download-") as output_dir:
            payload = {
                "action": "download_video_artifact",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "artifact": normalized,
                "output_dir": output_dir,
            }
            code, stdout, stderr = await self._run_command(
                [sys.executable, "-m", "modal_backend.client_cli"],
                env=self._subprocess_env(settings.profile),
                stdin_payload=payload,
                timeout=3_600,
            )
            if code != 0:
                failure: dict[str, Any] = {}
                if stdout.strip():
                    try:
                        parsed = json.loads(stdout)
                        if isinstance(parsed, dict):
                            failure = parsed
                        else:
                            print(
                                "[MODAL:VIDEO] 다운로드 실패 응답 형식 오류: "
                                f"type={type(parsed).__name__}"
                            )
                    except Exception as exc:
                        print(
                            "[MODAL:VIDEO] 다운로드 실패 응답 파싱 실패: "
                            f"error={type(exc).__name__}: {exc}, "
                            f"stdout_length={len(stdout)}"
                        )
                        traceback.print_exc()
                error = str(
                    failure.get("error")
                    or "Modal Video Volume MP4 다운로드에 실패했습니다."
                )
                print(
                    "[MODAL:VIDEO] MP4 다운로드 client 실패: "
                    f"exit_code={code}, remote={normalized['remote_path']!r}, "
                    f"error={error}, stderr={stderr[-2000:]}"
                )
                raise RuntimeError(error)
            try:
                response = json.loads(stdout)
            except Exception as exc:
                print(
                    "[MODAL:VIDEO] 다운로드 응답 파싱 실패: "
                    f"error={type(exc).__name__}: {exc}, stdout_length={len(stdout)}"
                )
                traceback.print_exc()
                raise RuntimeError("Modal MP4 다운로드 응답 형식이 올바르지 않습니다.") from exc
            if not isinstance(response, dict):
                print(
                    "[MODAL:VIDEO] 다운로드 응답 루트 형식 오류: "
                    f"type={type(response).__name__}, value={response!r}"
                )
                raise RuntimeError("Modal MP4 다운로드 응답 형식이 올바르지 않습니다.")
            if not response.get("ok"):
                error = str(response.get("error") or "Modal MP4 다운로드 실패")
                print(f"[MODAL:VIDEO] MP4 다운로드 응답 실패: error={error}")
                raise RuntimeError(error)
            result_payload = response.get("result")
            if not isinstance(result_payload, Mapping):
                print(
                    "[MODAL:VIDEO] 다운로드 result 형식 오류: "
                    f"type={type(result_payload).__name__}, value={result_payload!r}"
                )
                raise RuntimeError("Modal MP4 다운로드 결과 형식이 올바르지 않습니다.")
            downloaded = result_payload.get("artifact")
            if not isinstance(downloaded, Mapping):
                print(
                    "[MODAL:VIDEO] 다운로드 artifact 응답 누락: "
                    f"response={response!r}"
                )
                raise RuntimeError("Modal MP4 다운로드 결과가 없습니다.")
            local_path = Path(str(downloaded.get("path") or ""))
            if not local_path.is_file():
                print(
                    "[MODAL:VIDEO] 다운로드된 로컬 MP4 없음: "
                    f"path={local_path}, remote={normalized['remote_path']!r}"
                )
                raise FileNotFoundError("다운로드된 Modal MP4 임시 파일이 없습니다.")
            actual_size = local_path.stat().st_size
            actual_sha256 = await asyncio.to_thread(self._sha256_file, local_path)
            if (
                actual_size != normalized["size"]
                or actual_sha256 != normalized["sha256"]
            ):
                print(
                    "[MODAL:VIDEO] 로컬 MP4 재검증 실패: "
                    f"path={local_path}, expected_size={normalized['size']}, "
                    f"actual_size={actual_size}, expected_sha256={normalized['sha256']}, "
                    f"actual_sha256={actual_sha256}"
                )
                raise RuntimeError("다운로드된 Modal MP4의 검증 정보가 일치하지 않습니다.")
            video_bytes = local_path.read_bytes()
            if len(video_bytes) != actual_size:
                print(
                    "[MODAL:VIDEO] MP4 메모리 로드 크기 불일치: "
                    f"path={local_path}, expected={actual_size}, actual={len(video_bytes)}"
                )
                raise RuntimeError("다운로드된 Modal MP4를 완전하게 읽지 못했습니다.")
            print(
                "[MODAL:VIDEO] MP4 다운로드 및 이중 검증 완료: "
                f"remote={normalized['remote_path']!r}, bytes={actual_size:,}, "
                f"sha256={actual_sha256}"
            )
            return video_bytes, dict(normalized)

    async def generate_video(
        self,
        workflow: dict[str, Any],
        *,
        timeout_seconds: int = 3_300,
        input_paths: list[str] | tuple[str, ...] | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        video_job_id = f"video_{uuid.uuid4().hex}"
        result = await self.run_workflow(
            workflow,
            timeout_seconds=timeout_seconds,
            input_paths=input_paths,
            require_images=False,
            video_job_id=video_job_id,
            progress_callback=progress_callback,
        )
        artifacts = result.get("video_artifacts") or []
        if len(artifacts) != 1:
            print(
                "[MODAL:VIDEO] 생성 결과 MP4 artifact 누락: "
                f"prompt_id={result.get('prompt_id')}, artifacts={artifacts!r}"
            )
            raise RuntimeError("Modal 영상 생성 결과 MP4 artifact가 없습니다.")
        video_bytes, artifact = await self.download_video_artifact(artifacts[0])
        return video_bytes, {
            "execution_source": "modal",
            "prompt_id": result.get("prompt_id"),
            "model_sync": result.get("model_sync") or {},
            "lora_sync": result.get("lora_sync") or {},
            "artifact": artifact,
            "filename": artifact["filename"],
            "type": "output",
        }

    async def download_lora_artifacts(
        self,
        artifacts: list[dict[str, Any]],
        *,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        """GPU 호출과 분리된 Volume 다운로드 후 로컬 저장·원격 삭제를 예약한다."""
        config = self.get_config()
        settings = ModalSettings.from_mapping(config)
        if not settings.enabled:
            print("[MODAL_DOWNLOAD] 다운로드 실패: Modal이 비활성화되어 있습니다.")
            raise RuntimeError("Modal LoRA 다운로드 중 Modal이 비활성화되었습니다.")
        if not await self.account_connected(settings):
            print(
                "[MODAL_DOWNLOAD] 다운로드 실패: Modal 계정이 연결되지 않았습니다. "
                f"profile={settings.profile}"
            )
            raise RuntimeError("Modal 계정이 연결되어 있지 않습니다.")
        if not isinstance(artifacts, list) or not artifacts:
            print(f"[MODAL_DOWNLOAD] 다운로드 artifact 없음: artifacts={artifacts!r}")
            raise ValueError("다운로드할 Modal LoRA artifact가 없습니다.")

        normalized: list[dict[str, Any]] = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                print(
                    "[MODAL_DOWNLOAD] artifact 형식 오류: "
                    f"type={type(artifact).__name__}, value={artifact!r}"
                )
                raise TypeError("Modal LoRA artifact는 객체여야 합니다.")
            relative = Path(str(artifact.get("relative_path") or ""))
            remote_path = str(artifact.get("remote_path") or "").replace("\\", "/")
            remote_parts = Path(remote_path).parts
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or not remote_path.startswith("SOYA_CHAR_LORA/")
                or ".." in remote_parts
            ):
                print(
                    "[MODAL_DOWNLOAD] 안전하지 않은 artifact 경로 거부: "
                    f"relative={relative!s}, remote={remote_path!r}"
                )
                raise ValueError("안전하지 않은 Modal LoRA artifact 경로입니다.")
            normalized.append(
                {
                    "relative_path": relative.as_posix(),
                    "remote_path": remote_path.strip("/"),
                    "size": max(0, int(artifact.get("size") or 0)),
                }
            )

        progress_queue: asyncio.Queue[dict[str, Any] | None] | None = (
            asyncio.Queue() if progress_callback is not None else None
        )

        async def consume_progress() -> None:
            if progress_queue is None or progress_callback is None:
                return
            while True:
                event = await progress_queue.get()
                if event is None:
                    return
                try:
                    callback_result = progress_callback(event)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                except Exception as exc:
                    print(
                        "[MODAL_DOWNLOAD] 진행 콜백 실패: "
                        f"event={event!r}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

        def handle_download_output(source: str, line: str) -> None:
            if (
                source != "stderr"
                or progress_queue is None
                or not line.startswith(DOWNLOAD_PROGRESS_PREFIX)
            ):
                return
            raw_event = line[len(DOWNLOAD_PROGRESS_PREFIX) :]
            try:
                event = json.loads(raw_event)
                if not isinstance(event, dict):
                    raise TypeError("Modal LoRA 다운로드 이벤트는 객체여야 합니다.")
                total_bytes = max(0, int(event.get("total_bytes") or 0))
                downloaded_bytes = max(0, int(event.get("downloaded_bytes") or 0))
                total_files = max(0, int(event.get("total_files") or 0))
                index = max(0, int(event.get("index") or 0))
                if total_bytes > 0:
                    percentage = min(99.0, downloaded_bytes / total_bytes * 100.0)
                elif total_files > 0:
                    percentage = min(99.0, index / total_files * 100.0)
                else:
                    percentage = 0.0
                if str(event.get("event") or "") == "batch_complete":
                    percentage = 99.0
                progress_queue.put_nowait(
                    {
                        "phase": "modal_downloading",
                        "percentage": percentage,
                        **event,
                    }
                )
            except Exception as exc:
                print(
                    "[MODAL_DOWNLOAD] 진행 이벤트 파싱 실패: "
                    f"error={type(exc).__name__}: {exc}, payload={raw_event[:500]!r}"
                )
                traceback.print_exc()

        consumer_task = (
            asyncio.create_task(consume_progress())
            if progress_queue is not None
            else None
        )
        with tempfile.TemporaryDirectory(
            prefix="soya-modal-lora-download-",
            dir=runtime_temp_root(self.project_root),
        ) as output_dir:
            payload = {
                "action": "download_lora_artifacts",
                "app_name": settings.deployment_name,
                "environment": settings.environment,
                "artifacts": normalized,
                "output_dir": output_dir,
            }
            try:
                code, stdout, stderr = await self._run_command(
                    [sys.executable, "-m", "modal_backend.client_cli"],
                    env=self._subprocess_env(settings.profile),
                    stdin_payload=payload,
                    timeout=3_600,
                    output_callback=(
                        handle_download_output if progress_queue is not None else None
                    ),
                )
            finally:
                if progress_queue is not None:
                    progress_queue.put_nowait(None)
                if consumer_task is not None:
                    await consumer_task
            if code != 0:
                failure: dict[str, Any] = {}
                if stdout.strip():
                    try:
                        parsed = json.loads(stdout)
                        if isinstance(parsed, dict):
                            failure = parsed
                        else:
                            print(
                                "[MODAL_DOWNLOAD] 실패 응답 루트 형식 오류: "
                                f"type={type(parsed).__name__}"
                            )
                    except Exception as exc:
                        print(
                            "[MODAL_DOWNLOAD] 실패 응답 파싱 실패: "
                            f"error={type(exc).__name__}: {exc}, "
                            f"stdout_length={len(stdout)}"
                        )
                        traceback.print_exc()
                error = str(
                    failure.get("error")
                    or "Modal LoRA Volume 다운로드에 실패했습니다."
                )
                print(
                    "[MODAL_DOWNLOAD] client 실패: "
                    f"exit_code={code}, artifacts={len(normalized)}, error={error}, "
                    f"stderr={stderr[-2000:]}"
                )
                raise RuntimeError(error)
            try:
                response = json.loads(stdout)
            except Exception as exc:
                print(
                    "[MODAL_DOWNLOAD] client 응답 파싱 실패: "
                    f"error={type(exc).__name__}: {exc}, stdout_length={len(stdout)}"
                )
                traceback.print_exc()
                raise RuntimeError("Modal LoRA 다운로드 응답 형식이 올바르지 않습니다.") from exc
            if not response.get("ok"):
                error = str(response.get("error") or "Modal LoRA 다운로드 실패")
                print(f"[MODAL_DOWNLOAD] client 응답 실패: error={error}")
                raise RuntimeError(error)
            result = response.get("result") or {}
            downloaded = list(result.get("artifacts") or [])
            if len(downloaded) != len(normalized):
                print(
                    "[MODAL_DOWNLOAD] 다운로드 파일 수 검증 실패: "
                    f"expected={len(normalized)}, actual={len(downloaded)}"
                )
                raise RuntimeError("Modal LoRA 다운로드 파일 수가 일치하지 않습니다.")
            stored = self._store_modal_artifacts(downloaded, config)
            await self.enqueue_lora_delete_artifacts(downloaded)
            remote_paths = [str(item["remote_path"]) for item in downloaded]
            if progress_callback is not None:
                try:
                    callback_result = progress_callback(
                        {
                            "phase": "modal_download_complete",
                            "percentage": 100.0,
                            "total_files": len(stored),
                            "remote_delete_queued": len(remote_paths),
                        }
                    )
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                except Exception as exc:
                    print(
                        "[MODAL_DOWNLOAD] 완료 진행 콜백 실패: "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
            print(
                "[MODAL_DOWNLOAD] LoRA 로컬 저장 및 원격 삭제 예약 완료: "
                f"stored={len(stored)}, delete_paths={len(remote_paths)}"
            )
            return {
                "artifacts": stored,
                "remote_delete_queued": remote_paths,
            }

    async def generate(
        self,
        workflow: dict[str, Any],
        *,
        timeout_seconds: int = 3_300,
        input_paths: list[str] | tuple[str, ...] | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        result = await self.run_workflow(
            workflow,
            timeout_seconds=timeout_seconds,
            input_paths=input_paths,
            require_images=True,
            progress_callback=progress_callback,
        )
        images = result.get("images") or []
        if not images:
            print("[MODAL] 원격 이미지 생성 결과가 비어 있습니다.")
            raise RuntimeError("Modal 원격 이미지 생성 결과가 없습니다.")
        first = images[0]
        return first["bytes"], {
            "prompt_id": result.get("prompt_id"),
            "model_sync": result.get("model_sync") or {},
            "lora_sync": result.get("lora_sync") or {},
            "content_type": first.get("content_type"),
        }

    def _load_video_delete_outbox(self) -> list[dict[str, Any]]:
        if not self._video_delete_outbox_path.is_file():
            return []
        try:
            data = json.loads(
                self._video_delete_outbox_path.read_text(encoding="utf-8")
            )
            if not isinstance(data, list):
                raise ValueError("영상 삭제 outbox 루트는 배열이어야 합니다.")
            return [item for item in data if isinstance(item, dict)]
        except Exception as exc:
            print(
                "[MODAL:VIDEO] 삭제 outbox 읽기 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return []

    def _video_delete_outbox_count(self) -> int:
        return len(self._load_video_delete_outbox())

    def _save_video_delete_outbox(self, items: list[dict[str, Any]]) -> None:
        target = self._video_delete_outbox_path
        if target.exists():
            backup_root = self.project_root / "backups" / "modal"
            backup_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = backup_root / f"modal_video_delete_outbox_before_save_{stamp}.json"
            shutil.copy2(target, backup)
            print(f"[MODAL:VIDEO] 삭제 outbox 백업: {backup}")
        temp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            temp_path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, target)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            print(
                "[MODAL:VIDEO] 삭제 outbox 저장 실패: "
                f"path={target}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

    @staticmethod
    def _video_delete_item_key(item: Mapping[str, Any]) -> tuple[tuple[str, str, int], ...]:
        artifacts = item.get("remote_artifacts")
        if not isinstance(artifacts, list):
            return ()
        return tuple(
            sorted(
                (
                    str(artifact.get("remote_path") or ""),
                    str(artifact.get("sha256") or ""),
                    int(artifact.get("size") or 0),
                )
                for artifact in artifacts
                if isinstance(artifact, Mapping)
            )
        )

    async def _send_video_delete(
        self,
        settings: ModalSettings,
        artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "action": "delete_video_artifacts",
            "app_name": settings.deployment_name,
            "environment": settings.environment,
            "remote_artifacts": artifacts,
        }
        code, stdout, stderr = await self._run_command(
            [sys.executable, "-m", "modal_backend.client_cli"],
            env=self._subprocess_env(settings.profile),
            stdin_payload=payload,
            timeout=120,
        )
        try:
            response = json.loads(stdout) if stdout.strip() else {}
        except Exception as exc:
            print(
                "[MODAL:VIDEO] 삭제 응답 파싱 실패: "
                f"error={type(exc).__name__}: {exc}, stdout_length={len(stdout)}"
            )
            traceback.print_exc()
            raise RuntimeError("Modal 영상 삭제 응답 형식이 올바르지 않습니다.") from exc
        if not isinstance(response, dict):
            print(
                "[MODAL:VIDEO] 삭제 응답 루트 형식 오류: "
                f"type={type(response).__name__}, value={response!r}"
            )
            raise RuntimeError("Modal 영상 삭제 응답 형식이 올바르지 않습니다.")
        if code != 0 or not response.get("ok"):
            error = str(
                response.get("error")
                or f"Modal client exit_code={code}, stderr={stderr[-1000:]}"
            )
            print(
                "[MODAL:VIDEO] 원격 MP4 검증 삭제 client 실패: "
                f"count={len(artifacts)}, error={error}, stderr={stderr[-2000:]}"
            )
            raise RuntimeError(error)
        result = response.get("result") or {}
        if not isinstance(result, dict):
            print(
                "[MODAL:VIDEO] 삭제 결과 형식 오류: "
                f"type={type(result).__name__}, value={result!r}"
            )
            raise RuntimeError("Modal 영상 삭제 결과가 올바르지 않습니다.")
        return result

    async def _remove_video_delete_outbox_item(
        self,
        item_key: tuple[tuple[str, str, int], ...],
    ) -> None:
        async with self._video_delete_lock:
            current = await asyncio.to_thread(self._load_video_delete_outbox)
            filtered = [
                item
                for item in current
                if self._video_delete_item_key(item) != item_key
            ]
            await asyncio.to_thread(self._save_video_delete_outbox, filtered)

    def _schedule_video_delete_flush(self) -> None:
        if (
            self._video_delete_flush_task
            and not self._video_delete_flush_task.done()
        ):
            return
        self._video_delete_flush_task = asyncio.create_task(
            self._flush_video_delete_outbox()
        )

    async def delete_video_artifacts_after_spool(
        self,
        remote_artifacts: list[dict[str, Any]],
    ) -> bool:
        """스풀 저장 뒤 원격 MP4를 즉시 검증 삭제하고 실패는 outbox에 남긴다."""

        if not isinstance(remote_artifacts, list) or not remote_artifacts:
            print(
                "[MODAL:VIDEO] 삭제할 영상 artifact가 비어 있음: "
                f"artifacts={remote_artifacts!r}"
            )
            raise ValueError("삭제할 Modal 영상 artifact가 없습니다.")
        normalized = [
            self._normalize_video_artifact(artifact)
            for artifact in remote_artifacts
        ]
        unique = {
            (item["remote_path"], item["sha256"], item["size"]): item
            for item in normalized
        }
        normalized = [unique[key] for key in sorted(unique)]
        queued_item = {
            "remote_artifacts": normalized,
            "created_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "attempts": 0,
        }
        item_key = self._video_delete_item_key(queued_item)
        async with self._video_delete_lock:
            current = await asyncio.to_thread(self._load_video_delete_outbox)
            if not any(self._video_delete_item_key(item) == item_key for item in current):
                current.append(queued_item)
                await asyncio.to_thread(self._save_video_delete_outbox, current)
                print(
                    "[MODAL:VIDEO] 원격 MP4 검증 삭제 outbox 기록: "
                    f"paths={[item['remote_path'] for item in normalized]!r}"
                )
            else:
                print(
                    "[MODAL:VIDEO] 원격 MP4 삭제가 이미 outbox에 기록됨: "
                    f"paths={[item['remote_path'] for item in normalized]!r}"
                )

        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print(
                "[MODAL:VIDEO] Modal 비활성화로 원격 MP4 즉시 삭제 보류: "
                f"paths={[item['remote_path'] for item in normalized]!r}"
            )
            return False
        if not await self.account_connected(settings):
            print(
                "[MODAL:VIDEO] 계정 미연결로 원격 MP4 즉시 삭제 보류: "
                f"profile={settings.profile}"
            )
            return False
        try:
            result = await self._send_video_delete(settings, normalized)
            await self._remove_video_delete_outbox_item(item_key)
            skipped = list(result.get("skipped_changed") or [])
            if skipped:
                print(
                    "[MODAL:VIDEO] 원격 MP4 변경 감지로 삭제하지 않음: "
                    f"paths={skipped!r}"
                )
                return False
            print(
                "[MODAL:VIDEO] 다운로드·스풀 확인 후 원격 MP4 삭제 완료: "
                f"deleted={result.get('deleted_count', 0)}, "
                f"already_missing={result.get('already_missing') or []}"
            )
            return True
        except Exception as exc:
            print(
                "[MODAL:VIDEO] 원격 MP4 즉시 삭제 실패, outbox 재시도 예약: "
                f"paths={[item['remote_path'] for item in normalized]!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            self._schedule_video_delete_flush()
            return False

    async def _flush_video_delete_outbox(self) -> None:
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print("[MODAL:VIDEO] Modal 비활성화로 영상 삭제 outbox 전송을 보류합니다.")
            return
        if not await self.account_connected(settings):
            print("[MODAL:VIDEO] 계정 미연결로 영상 삭제 outbox 전송을 보류합니다.")
            return
        while True:
            async with self._video_delete_lock:
                items = await asyncio.to_thread(self._load_video_delete_outbox)
                if not items:
                    return
                item = dict(items[0])
            artifacts = item.get("remote_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                print(f"[MODAL:VIDEO] 삭제 outbox artifact 누락: item={item!r}")
                async with self._video_delete_lock:
                    current = await asyncio.to_thread(self._load_video_delete_outbox)
                    await asyncio.to_thread(self._save_video_delete_outbox, current[1:])
                continue
            try:
                normalized = [
                    self._normalize_video_artifact(artifact)
                    for artifact in artifacts
                ]
                item_key = self._video_delete_item_key(item)
                result = await self._send_video_delete(settings, normalized)
                await self._remove_video_delete_outbox_item(item_key)
                print(
                    "[MODAL:VIDEO] outbox 원격 MP4 삭제 처리 완료: "
                    f"deleted={result.get('deleted_count', 0)}, "
                    f"skipped={result.get('skipped_changed') or []}, "
                    f"already_missing={result.get('already_missing') or []}"
                )
            except Exception as exc:
                print(
                    "[MODAL:VIDEO] 원격 MP4 삭제 재시도 실패, outbox 유지: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                attempts = int(item.get("attempts") or 0) + 1
                async with self._video_delete_lock:
                    current = await asyncio.to_thread(self._load_video_delete_outbox)
                    item_key = self._video_delete_item_key(item)
                    for queued in current:
                        if self._video_delete_item_key(queued) == item_key:
                            queued["attempts"] = attempts
                            queued["last_error"] = f"{type(exc).__name__}: {exc}"
                    await asyncio.to_thread(self._save_video_delete_outbox, current)
                retry_seconds = min(60.0, 2.0 ** min(attempts, 6))
                print(
                    "[MODAL:VIDEO] 원격 MP4 삭제 재시도 대기: "
                    f"attempts={attempts}, retry_seconds={retry_seconds:.0f}"
                )
                await asyncio.sleep(retry_seconds)

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

    async def enqueue_lora_delete_paths(self, remote_paths: list[str]) -> None:
        """로컬 저장이 확인된 정확한 원격 LoRA 파일만 삭제 outbox에 넣는다."""
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print(
                "[MODAL_SYNC] Modal이 비활성화되어 원격 LoRA 파일 삭제 예약 생략: "
                f"paths={remote_paths!r}"
            )
            return
        if not isinstance(remote_paths, list) or not remote_paths:
            print(f"[MODAL_SYNC] 원격 LoRA 삭제 파일 목록이 비어 있음: {remote_paths!r}")
            raise ValueError("원격 LoRA 삭제 파일 목록이 비어 있습니다.")
        normalized_paths: list[str] = []
        for remote_path in remote_paths:
            normalized = str(remote_path or "").strip().replace("\\", "/").strip("/")
            parts = normalized.split("/") if normalized else []
            if (
                len(parts) < 2
                or parts[0] != "SOYA_CHAR_LORA"
                or any(part in ("", ".", "..") for part in parts)
            ):
                print(
                    "[MODAL_SYNC] 안전하지 않은 원격 LoRA 삭제 파일 경로: "
                    f"path={remote_path!r}"
                )
                raise ValueError(
                    f"안전하지 않은 Modal LoRA 삭제 파일 경로입니다: {remote_path!r}"
                )
            normalized_paths.append(normalized)
        normalized_paths = sorted(set(normalized_paths), key=str.casefold)
        async with self._delete_lock:
            items = await asyncio.to_thread(self._load_delete_outbox)
            already_queued = {
                str(path)
                for item in items
                for path in (
                    item.get("remote_paths")
                    if isinstance(item.get("remote_paths"), list)
                    else []
                )
            }
            already_queued.update(
                str(artifact.get("remote_path") or "")
                for item in items
                for artifact in (
                    item.get("remote_artifacts")
                    if isinstance(item.get("remote_artifacts"), list)
                    else []
                )
                if isinstance(artifact, Mapping)
            )
            new_paths = [path for path in normalized_paths if path not in already_queued]
            if new_paths:
                items.append(
                    {
                        "remote_paths": new_paths,
                        "created_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "attempts": 0,
                    }
                )
                await asyncio.to_thread(self._save_delete_outbox, items)
                print(
                    "[MODAL_SYNC] 원격 LoRA 파일 삭제 예약: "
                    f"count={len(new_paths)}, paths={new_paths!r}"
                )
            else:
                print(
                    "[MODAL_SYNC] 원격 LoRA 파일 삭제가 이미 예약됨: "
                    f"paths={normalized_paths!r}"
                )
        self._schedule_delete_flush()

    async def enqueue_lora_delete_artifacts(
        self,
        remote_artifacts: list[dict[str, Any]],
    ) -> None:
        """다운로드한 해시와 일치할 때만 삭제하도록 artifact를 outbox에 넣는다."""
        settings = ModalSettings.from_mapping(self.get_config())
        if not settings.enabled:
            print(
                "[MODAL_SYNC] Modal이 비활성화되어 원격 LoRA 검증 삭제 예약 생략: "
                f"artifacts={remote_artifacts!r}"
            )
            return
        if not isinstance(remote_artifacts, list) or not remote_artifacts:
            print(
                "[MODAL_SYNC] 원격 LoRA 검증 삭제 artifact가 비어 있음: "
                f"artifacts={remote_artifacts!r}"
            )
            raise ValueError("원격 LoRA 검증 삭제 artifact가 비어 있습니다.")

        normalized_artifacts: list[dict[str, Any]] = []
        for artifact in remote_artifacts:
            if not isinstance(artifact, Mapping):
                print(
                    "[MODAL_SYNC] 원격 LoRA 검증 삭제 artifact 형식 오류: "
                    f"type={type(artifact).__name__}, value={artifact!r}"
                )
                raise TypeError("원격 LoRA 검증 삭제 artifact는 객체여야 합니다.")
            remote_path = (
                str(artifact.get("remote_path") or "")
                .strip()
                .replace("\\", "/")
                .strip("/")
            )
            parts = remote_path.split("/") if remote_path else []
            sha256 = str(artifact.get("sha256") or "").strip().lower()
            try:
                size = int(artifact.get("size"))
            except (TypeError, ValueError) as exc:
                print(
                    "[MODAL_SYNC] 원격 LoRA 검증 삭제 크기 형식 오류: "
                    f"path={remote_path!r}, size={artifact.get('size')!r}"
                )
                traceback.print_exc()
                raise ValueError("원격 LoRA 검증 삭제 크기가 올바르지 않습니다.") from exc
            if (
                len(parts) < 2
                or parts[0] != "SOYA_CHAR_LORA"
                or any(part in ("", ".", "..") for part in parts)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
                or size < 0
            ):
                print(
                    "[MODAL_SYNC] 안전하지 않은 원격 LoRA 검증 삭제 artifact: "
                    f"path={remote_path!r}, sha256={sha256!r}, size={size}"
                )
                raise ValueError("안전하지 않은 Modal LoRA 검증 삭제 artifact입니다.")
            normalized_artifacts.append(
                {"remote_path": remote_path, "sha256": sha256, "size": size}
            )

        unique_artifacts = {
            (artifact["remote_path"], artifact["sha256"], artifact["size"]): artifact
            for artifact in normalized_artifacts
        }
        normalized_artifacts = [
            unique_artifacts[key]
            for key in sorted(unique_artifacts, key=lambda value: value[0].casefold())
        ]
        async with self._delete_lock:
            items = await asyncio.to_thread(self._load_delete_outbox)
            already_queued = {
                (
                    str(artifact.get("remote_path") or ""),
                    str(artifact.get("sha256") or ""),
                    str(artifact.get("size") or ""),
                )
                for item in items
                for artifact in (
                    item.get("remote_artifacts")
                    if isinstance(item.get("remote_artifacts"), list)
                    else []
                )
                if isinstance(artifact, Mapping)
            }
            new_artifacts = [
                artifact
                for artifact in normalized_artifacts
                if (
                    artifact["remote_path"],
                    artifact["sha256"],
                    str(artifact["size"]),
                )
                not in already_queued
            ]
            if new_artifacts:
                items.append(
                    {
                        "remote_artifacts": new_artifacts,
                        "created_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "attempts": 0,
                    }
                )
                await asyncio.to_thread(self._save_delete_outbox, items)
                print(
                    "[MODAL_SYNC] 원격 LoRA 검증 삭제 예약: "
                    f"count={len(new_artifacts)}, "
                    f"paths={[item['remote_path'] for item in new_artifacts]!r}"
                )
            else:
                print(
                    "[MODAL_SYNC] 원격 LoRA 검증 삭제가 이미 예약됨: "
                    f"paths={[item['remote_path'] for item in normalized_artifacts]!r}"
                )
        self._schedule_delete_flush()

    @staticmethod
    def _delete_outbox_item_key(item: Mapping[str, Any]) -> tuple[str, Any]:
        remote_artifacts = item.get("remote_artifacts")
        if isinstance(remote_artifacts, list) and remote_artifacts:
            return (
                "artifacts",
                tuple(
                    sorted(
                        (
                            str(artifact.get("remote_path") or ""),
                            str(artifact.get("sha256") or ""),
                            str(artifact.get("size") or ""),
                        )
                        for artifact in remote_artifacts
                        if isinstance(artifact, Mapping)
                    )
                ),
            )
        remote_paths = item.get("remote_paths")
        if isinstance(remote_paths, list) and remote_paths:
            return ("paths", tuple(sorted(str(path) for path in remote_paths)))
        return ("prefix", str(item.get("remote_prefix") or ""))

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
            remote_artifacts = item.get("remote_artifacts")
            remote_paths = item.get("remote_paths")
            if isinstance(remote_artifacts, list) and remote_artifacts:
                item_key = self._delete_outbox_item_key(item)
                payload = {
                    "action": "delete_lora_artifacts",
                    "app_name": settings.deployment_name,
                    "environment": settings.environment,
                    "remote_artifacts": list(remote_artifacts),
                }
                target_label = f"verified_files={len(remote_artifacts)}"
            elif isinstance(remote_paths, list) and remote_paths:
                item_key = self._delete_outbox_item_key(item)
                payload = {
                    "action": "delete_lora_paths",
                    "app_name": settings.deployment_name,
                    "environment": settings.environment,
                    "remote_paths": list(remote_paths),
                }
                target_label = f"files={len(remote_paths)}"
            else:
                remote_prefix = str(item.get("remote_prefix") or "")
                if not remote_prefix:
                    print(f"[MODAL_SYNC] 삭제 outbox 항목 경로 누락: item={item!r}")
                    async with self._delete_lock:
                        current = await asyncio.to_thread(self._load_delete_outbox)
                        current = current[1:]
                        await asyncio.to_thread(self._save_delete_outbox, current)
                    continue
                item_key = ("prefix", remote_prefix)
                payload = {
                    "action": "delete_lora_prefix",
                    "app_name": settings.deployment_name,
                    "environment": settings.environment,
                    "remote_prefix": remote_prefix,
                }
                target_label = f"prefix={remote_prefix}"
            try:
                code, stdout, stderr = await self._run_command(
                    [sys.executable, "-m", "modal_backend.client_cli"],
                    env=self._subprocess_env(settings.profile),
                    stdin_payload=payload,
                    timeout=120,
                )
                response = json.loads(stdout) if stdout.strip() else {}
                if code != 0 or not response.get("ok"):
                    raise RuntimeError(
                        str(
                            response.get("error")
                            or f"Modal client exit_code={code}, stderr={stderr[-1000:]}"
                        )
                    )
                async with self._delete_lock:
                    current = await asyncio.to_thread(self._load_delete_outbox)
                    filtered = []
                    for queued in current:
                        queued_key = self._delete_outbox_item_key(queued)
                        if queued_key != item_key:
                            filtered.append(queued)
                    current = filtered
                    await asyncio.to_thread(self._save_delete_outbox, current)
                print(f"[MODAL_SYNC] 원격 LoRA 삭제 완료: {target_label}")
            except Exception as exc:
                print(
                    f"[MODAL_SYNC] 원격 LoRA 삭제 실패, outbox 유지: "
                    f"target={target_label}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                attempts = int(item.get("attempts") or 0) + 1
                async with self._delete_lock:
                    current = await asyncio.to_thread(self._load_delete_outbox)
                    for queued in current:
                        queued_key = self._delete_outbox_item_key(queued)
                        if queued_key == item_key:
                            queued["attempts"] = attempts
                            queued["last_error"] = f"{type(exc).__name__}: {exc}"
                    await asyncio.to_thread(self._save_delete_outbox, current)
                retry_seconds = min(60.0, 2.0 ** min(attempts, 6))
                print(
                    "[MODAL_SYNC] 원격 LoRA 삭제 재시도 대기: "
                    f"target={target_label}, attempts={attempts}, "
                    f"retry_seconds={retry_seconds:.0f}"
                )
                await asyncio.sleep(retry_seconds)
