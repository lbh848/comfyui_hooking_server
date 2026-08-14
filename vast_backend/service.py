"""VastService — 인스턴스 라이프사이클/모델 준비/원격 ComfyUI 실행 총괄.

준비 흐름(마법사 ④단계):
  1. 인스턴스 생성 (이미지: Modal과 동일한 bh848/soya-comfy-runtime, onstart 대기 스크립트)
  2. SSH 키 부착 → paramiko 접속
  3. 짧은 실사용 전송 프리플라이트 + 워크플로우 기준 남은 전송 ETA
  4. 병렬 A: sftp 업로드 — custom_nodes 압축본 + 선택 LoRA + 'upload' 배정 모델
     병렬 B: 원격 다운로드 스크립트 — HF/Civitai/URL 모델
  5. SSH 로컬 터널 생성 + /tmp/soya_ready 터치 → ComfyUI(8188) 기동
  6. 로컬 터널 헬스체크 → 'ready'
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import aiohttp

from .client import VastApiError, VastClient
from .image_pull_progress import build_pull_progress, parse_daemon_pull_states
from .model_sources import (
    build_download_plan,
    defaults_from_manifest,
    load_mapping,
)
from .preflight import (
    calculate_transfer_estimate,
    empty_preflight_state,
    failed_result,
    informational_result,
    parse_curl_speed_probe,
    speed_result,
)
from .settings import VastSettings, load_key_files
from .ssh_tunnel import ComfySshTunnel

COMFY_ROOT_REMOTE = "/root/ComfyUI"
READY_FLAG = "/tmp/soya_ready"
MODELS_DONE_FLAG = "/tmp/soya_models_done"
BUILD_COMPLETE_FLAG = "/tmp/soya_build_complete"
# 비용 보호 기본값. 시간 제한은 없고 예상 빌드비 상한만 존재한다.
MAX_BUILD_COST_USD = 0.25
NO_PROGRESS_WARNING_SECONDS = 180
WATCHDOG_POLL_SECONDS = 10
SSH_CONNECT_TIMEOUT_SECONDS = 60
SSH_STATUS_POLL_SECONDS = 10
WATCHDOG_STATUS_MAX_AGE_SECONDS = 25
ACCOUNT_STATUS_CACHE_SECONDS = 60
ACCOUNT_STATUS_ERROR_CACHE_SECONDS = 15
IMAGE_PULL_POLL_SECONDS = 20
IMAGE_PULL_LOG_TAIL = 1000
PREFLIGHT_DOWNLOAD_BYTES = 32 * 1024 * 1024
PREFLIGHT_UPLOAD_BYTES = 16 * 1024 * 1024
PREFLIGHT_CURL_MAX_SECONDS = 15
PREFLIGHT_CLOUDFLARE_URL = "https://speed.cloudflare.com/__down"
PREFLIGHT_HF_FALLBACK_URL = (
    "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
)
# 고정 런타임 이미지는 CUDA 12.8 바이너리를 사용한다. 오퍼 검색에서 이보다
# 낮은 cuda_max_good 머신을 제외해 호환성 문제로 유료 빌드가 실패하지 않게 한다.
MIN_RUNTIME_CUDA_VERSION = 12.8
SERVICE_LABEL_PREFIX = "soya-vast-"
_LAUNCH_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "vast_launch_id", default=""
)


class LaunchCancelled(RuntimeError):
    """사용자 파괴/서버 종료로 중단된 launch 작업."""


def _log(message: str) -> None:
    print(f"[VAST] {message}")


class VastService:
    def __init__(self, project_root: str | Path, get_config: Callable[[], dict]) -> None:
        self.project_root = Path(project_root).resolve()
        self._get_config = get_config
        self._client: VastClient | None = None
        self._comfy_tunnel: ComfySshTunnel | None = None
        self._launch_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._watchdog_launch_id = ""
        self._launch_lock = asyncio.Lock()
        self._destroy_lock = asyncio.Lock()
        self._instance_status_lock = asyncio.Lock()
        self._instance_status_cache: dict[
            int, tuple[float, dict[str, Any]]
        ] = {}
        self._account_status_lock = asyncio.Lock()
        self._account_status_cache: tuple[float, dict[str, Any]] | None = None
        self._image_pull_lock = asyncio.Lock()
        self._image_pull_last_poll_monotonic = 0.0
        self._image_manifest_cache: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._log_secrets: set[str] = set()
        self._state_lock = threading.RLock()
        self._guard_write_lock = threading.Lock()
        self._guard_path = self.project_root / "runtime" / "vast_launch_guard.json"
        # 생성 진행 상태(단일 인스턴스 운영 가정 — 파괴 후 재생성)
        self.launch: dict[str, Any] = self._new_launch_state()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _new_launch_state(
        self,
        *,
        state: str = "idle",
        launch_id: str = "",
        label: str = "",
        hourly_price_usd: float = 0.0,
    ) -> dict[str, Any]:
        now = time.time()
        return {
            "state": state,
            "launch_id": launch_id,
            "label": label,
            "instance_id": None,
            "destroyed_instance_id": None,
            "error": "",
            "steps": [],
            "current_step": "",
            "comfy_base_url": "",
            "started_at": self._utc_now() if state != "idle" else "",
            "started_at_epoch": now if state != "idle" else 0.0,
            "contract_started_at_epoch": 0.0,
            "updated_at": self._utc_now(),
            "last_progress_at_epoch": now if state != "idle" else 0.0,
            "instance_status": "",
            "instance_status_msg": "",
            "image_pull": {
                "available": False,
                "exact_progress_available": False,
                "phase": "waiting",
                "image": "",
                "total_bytes": 0,
                "confirmed_bytes": 0,
                "pending_bytes": 0,
                "minimum_percent": 0.0,
                "total_layers": 0,
                "confirmed_layers": 0,
                "pending_layers": [],
                "observed_layers": 0,
                "complete": False,
                "last_checked_at": "",
                "last_observed_progress_at_epoch": 0.0,
                "error": "",
                "_signature": "",
            },
            "preflight": empty_preflight_state(),
            "hourly_price_usd": max(0.0, float(hourly_price_usd or 0.0)),
            "status_history": [],
            "events": [],
            "last_watchdog_log_at_epoch": 0.0,
            "orphan_instance_ids": [],
            "recovered_was_ready": False,
            "ssh_ready_at_epoch": 0.0,
            "protection_state": "armed" if state != "idle" else "off",
            "protection_reason": "",
            "destroy_reason": "",
            "destroy_automatic": False,
            "limits": {
                "max_build_cost_usd": MAX_BUILD_COST_USD,
                "no_progress_warning_seconds": NO_PROGRESS_WARNING_SECONDS,
            },
        }

    # ── 설정/클라이언트 ─────────────────────────────────────

    def settings(self) -> VastSettings:
        return VastSettings.from_mapping(
            self._get_config(), **load_key_files(self.project_root)
        )

    def _client_or_raise(self) -> VastClient:
        config = self.settings()
        if not config.api_key:
            raise VastApiError(
                "Vast API 키가 설정되지 않았습니다. 설정에서 API 키를 먼저 입력하세요."
            )
        self._log_secrets.update(
            value
            for value in (config.api_key, config.civitai_api_key)
            if value
        )
        if self._client is None:
            self._client = VastClient(config.api_key)
        return self._client

    def _load_guard_state(self) -> dict[str, Any]:
        if not self._guard_path.is_file():
            print(f"[VAST_GUARD] 복구 상태 파일 없음: {self._guard_path}")
            return {}
        try:
            data = json.loads(self._guard_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(
                    f"복구 상태 최상위 값은 객체여야 합니다: {type(data).__name__}"
                )
            return data
        except (OSError, ValueError) as exc:
            print(
                "[VAST_GUARD][ERROR] 복구 상태 읽기 실패: "
                f"path={self._guard_path}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return {}

    def _persist_guard_state(self) -> None:
        """재시작 복구에 필요한 launch 상태를 UTF-8/원자 교체로 저장한다."""
        snapshot = dict(self.launch)
        try:
            with self._guard_write_lock:
                self._guard_path.parent.mkdir(parents=True, exist_ok=True)
                if self._guard_path.is_file():
                    backup_dir = self.project_root / "backups" / "vast_launch_guard"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = backup_dir / f"vast_launch_guard_{time.time_ns()}.json"
                    shutil.copy2(self._guard_path, backup_path)
                    backups = sorted(
                        backup_dir.glob("vast_launch_guard_*.json"),
                        key=lambda path: path.stat().st_mtime,
                    )
                    for old in backups[:-5]:
                        try:
                            old.unlink()
                        except OSError as exc:
                            print(
                                "[VAST_GUARD][ERROR] 오래된 백업 정리 실패: "
                                f"path={old}, error={type(exc).__name__}: {exc}"
                            )
                            traceback.print_exc()
                temp_path = self._guard_path.with_suffix(".json.tmp")
                temp_path.write_text(
                    json.dumps(snapshot, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(temp_path, self._guard_path)
        except (OSError, TypeError, ValueError) as exc:
            print(
                "[VAST_GUARD][ERROR] 복구 상태 저장 실패(재시작 보호 약화): "
                f"path={self._guard_path}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    async def startup(self) -> None:
        """서버 재시작 뒤 서비스 소유 인스턴스를 찾아 비용 감시를 복구한다."""
        try:
            cfg = self.settings()
            if not cfg.enabled:
                print("[VAST_GUARD] Vast 기능이 꺼져 있어 시작 복구를 건너뜁니다.")
                return
            if not cfg.api_key:
                print("[VAST_GUARD][ERROR] API 키가 없어 시작 복구를 수행할 수 없습니다.")
                return
            client = self._client_or_raise()
            rows = await client.list_instances()
            owned = [
                row
                for row in rows
                if str(row.get("label") or "").startswith(SERVICE_LABEL_PREFIX)
                or str(row.get("label") or "") == "soya-vast"
            ]
            saved = self._load_guard_state()
            saved_id = int(saved.get("instance_id") or 0)
            target = next(
                (row for row in owned if int(row.get("id") or 0) == saved_id),
                None,
            )
            if target is None and owned:
                target = max(
                    owned,
                    key=lambda row: float(row.get("start_date") or 0.0),
                )
            if target is None:
                print("[VAST_GUARD] 복구할 서비스 소유 인스턴스가 없습니다.")
                if saved_id:
                    self.launch = self._new_launch_state(state="destroyed")
                    self.launch["destroyed_instance_id"] = saved_id
                    self.launch["destroy_reason"] = "서버 시작 시 Vast 목록에서 이미 소멸 확인"
                    self.launch["protection_state"] = "complete"
                    self._persist_guard_state()
                return

            instance_id = int(target.get("id") or 0)
            label = str(target.get("label") or f"{SERVICE_LABEL_PREFIX}recovered")
            launch_id = str(saved.get("launch_id") or label.removeprefix(SERVICE_LABEL_PREFIX))
            has_saved_match = saved_id == instance_id
            # 복구 파일과 연결되지 않는 과거 인스턴스는 자동 파괴하지 않는다.
            # UI/CMD에 경고하고 사용자가 직접 확인·파괴하게 한다.
            was_ready = not has_saved_match or saved.get("state") == "ready"
            self.launch = self._new_launch_state(
                state="recovered",
                launch_id=launch_id,
                label=label,
                hourly_price_usd=float(
                    target.get("dph_total") or saved.get("hourly_price_usd") or 0.0
                ),
            )
            self.launch["instance_id"] = instance_id
            self.launch["contract_started_at_epoch"] = float(
                target.get("start_date")
                or saved.get("contract_started_at_epoch")
                or time.time()
            )
            self.launch["started_at_epoch"] = float(
                saved.get("started_at_epoch")
                or self.launch["contract_started_at_epoch"]
            )
            self.launch["started_at"] = str(saved.get("started_at") or self._utc_now())
            self.launch["steps"] = list(saved.get("steps") or [])
            self.launch["events"] = list(saved.get("events") or [])[-200:]
            self.launch["current_step"] = "recovered"
            self.launch["recovered_was_ready"] = bool(was_ready)
            self.launch["orphan_instance_ids"] = [
                int(row.get("id") or 0)
                for row in owned
                if int(row.get("id") or 0) != instance_id
            ]
            self._update_instance_status(target)
            self._set_step_unchecked(
                "recovered",
                "running",
                "서버 재시작 후 인스턴스 감시 복구 — 필요하면 즉시 파괴하세요",
            )
            if was_ready:
                self.launch["protection_state"] = "manual_required"
                self.launch["protection_reason"] = (
                    "준비 완료 인스턴스였지만 SSH 터널이 끊겨 수동 파괴 또는 재빌드가 필요합니다."
                )
            self._cancel_events[launch_id] = threading.Event()
            self._persist_guard_state()
            self._ensure_watchdog(launch_id)
            print(
                "[VAST_GUARD] 인스턴스 감시 복구 완료: "
                f"instance={instance_id}, label={label}, was_ready={was_ready}, "
                f"other_owned={self.launch['orphan_instance_ids']}"
            )
        except Exception as exc:
            print(
                "[VAST_GUARD][ERROR] 시작 복구 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    async def close(self) -> None:
        launch_id = str(self.launch.get("launch_id") or "")
        cancel_event = self._cancel_events.get(launch_id)
        if cancel_event is not None:
            cancel_event.set()
        tasks = [
            task
            for task in (self._launch_task, self._watchdog_task)
            if task is not None and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._close_comfy_tunnel()
        if self._client:
            await self._client.close()
            self._client = None

    def _current_cancel_event(self) -> threading.Event | None:
        launch_id = _LAUNCH_CONTEXT.get()
        return self._cancel_events.get(launch_id) if launch_id else None

    def _check_cancelled(self) -> None:
        launch_id = _LAUNCH_CONTEXT.get()
        if not launch_id:
            return
        cancel_event = self._cancel_events.get(launch_id)
        if (
            launch_id != str(self.launch.get("launch_id") or "")
            or cancel_event is None
            or cancel_event.is_set()
        ):
            raise LaunchCancelled(f"Vast launch 작업 취소됨: launch_id={launch_id}")

    def _wait_sync(self, seconds: float) -> None:
        cancel_event = self._current_cancel_event()
        if cancel_event is not None and cancel_event.wait(seconds):
            self._check_cancelled()
        elif cancel_event is None:
            time.sleep(seconds)

    def _set_step_unchecked(self, key: str, state: str, detail: str = "") -> None:
        with self._state_lock:
            now = time.time()
            steps = {s["key"]: s for s in self.launch["steps"]}
            previous = steps.get(key) or {}
            changed = previous.get("state") != state or previous.get("detail") != detail
            steps[key] = {
                "key": key,
                "state": state,
                "detail": detail,
                "updated_at": self._utc_now(),
            }
            self.launch["steps"] = list(steps.values())
            if state == "running":
                self.launch["current_step"] = key
            if changed:
                self.launch["last_progress_at_epoch"] = now
                self.launch["updated_at"] = self._utc_now()
                self._event("step", f"{key}: {state} — {detail or '(상세 없음)'}")

    def _event(self, level: str, message: str) -> None:
        """서버 CMD와 UI 터미널에 동일한 빌드 이벤트를 남긴다."""
        with self._state_lock:
            safe_message = "".join(
                char
                for char in str(message)
                if char in {"\n", "\t"} or ord(char) >= 32
            )[-4000:]
            for secret in tuple(self._log_secrets):
                safe_message = safe_message.replace(secret, "<redacted>")
            now_iso = self._utc_now()
            events = list(self.launch.get("events") or [])
            events.append({"at": now_iso, "level": str(level), "message": safe_message})
            self.launch["events"] = events[-200:]
            print(f"[VAST][BUILD][{str(level).upper()}] {safe_message}")

    def _update_instance_status(self, info: dict[str, Any]) -> None:
        status = str(info.get("actual_status") or "").lower()
        status_msg = str(info.get("status_msg") or "")
        previous = (
            str(self.launch.get("instance_status") or ""),
            str(self.launch.get("instance_status_msg") or ""),
        )
        current = (status, status_msg)
        self.launch["instance_status"] = status
        self.launch["instance_status_msg"] = status_msg
        try:
            price = float(info.get("dph_total") or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        if price > 0:
            self.launch["hourly_price_usd"] = price
        try:
            start_date = float(info.get("start_date") or 0.0)
        except (TypeError, ValueError):
            start_date = 0.0
        current_contract_at = float(
            self.launch.get("contract_started_at_epoch") or 0.0
        )
        if start_date > 0 and (
            current_contract_at <= 0 or start_date < current_contract_at
        ):
            self.launch["contract_started_at_epoch"] = start_date
        if current != previous:
            now = time.time()
            self.launch["last_progress_at_epoch"] = now
            self.launch["updated_at"] = self._utc_now()
            history = list(self.launch.get("status_history") or [])
            history.append(
                {
                    "at": self._utc_now(),
                    "status": status,
                    "message": status_msg,
                }
            )
            self.launch["status_history"] = history[-20:]
            self._event(
                "vast",
                f"instance={self.launch.get('instance_id')}, "
                f"status={status or '(없음)'}, message={status_msg or '(없음)'}"
            )

    async def _get_instance_status(
        self,
        instance_id: int,
        *,
        max_age_seconds: float,
    ) -> tuple[dict[str, Any], bool]:
        """단일 Vast 상태 조회 경로.

        SSH 대기와 비용 watchdog가 같은 캐시와 lock을 공유한다. 반환값의 두 번째
        항목은 이번 호출이 실제 Vast API를 갱신했는지를 나타낸다.
        """
        target = int(instance_id)
        if target <= 0:
            print(f"[VAST_STATUS][ERROR] 잘못된 인스턴스 ID: {instance_id!r}")
            raise VastApiError(f"잘못된 Vast 인스턴스 ID: {instance_id!r}")

        def cached_status() -> dict[str, Any] | None:
            cached = self._instance_status_cache.get(target)
            if cached is None:
                return None
            fetched_at, info = cached
            age = max(0.0, time.monotonic() - fetched_at)
            if age >= max(0.0, float(max_age_seconds)):
                return None
            return dict(info)

        cached = cached_status()
        if cached is not None:
            return cached, False

        async with self._instance_status_lock:
            cached = cached_status()
            if cached is not None:
                return cached, False
            info = await self._client_or_raise().get_instance(target)
            snapshot = dict(info) if isinstance(info, dict) else {}
            self._instance_status_cache[target] = (time.monotonic(), snapshot)
            if int(self.launch.get("instance_id") or 0) == target:
                self._update_instance_status(snapshot)
            return dict(snapshot), True

    async def _refresh_image_pull_progress(
        self, instance_id: int, info: dict[str, Any]
    ) -> None:
        """loading 동안 daemon log와 manifest를 결합해 최소 완료량을 갱신한다."""
        target = int(instance_id)
        if target <= 0:
            print(f"[VAST_PULL][ERROR] 잘못된 인스턴스 ID: {instance_id!r}")
            return
        status = str(info.get("actual_status") or "").lower()
        current = dict(self.launch.get("image_pull") or {})
        if status != "loading":
            if status == "running" and current.get("available") and not current.get("complete"):
                now = time.time()
                current.update(
                    phase="complete",
                    complete=True,
                    confirmed_bytes=int(current.get("total_bytes") or 0),
                    pending_bytes=0,
                    minimum_percent=100.0,
                    confirmed_layers=int(current.get("total_layers") or 0),
                    pending_layers=[],
                    last_checked_at=self._utc_now(),
                    last_observed_progress_at_epoch=now,
                    error="",
                )
                self.launch["image_pull"] = current
                self.launch["last_progress_at_epoch"] = now
                self.launch["updated_at"] = self._utc_now()
            return

        now_monotonic = time.monotonic()
        if (
            now_monotonic - self._image_pull_last_poll_monotonic
            < IMAGE_PULL_POLL_SECONDS
        ):
            return

        async with self._image_pull_lock:
            now_monotonic = time.monotonic()
            if (
                now_monotonic - self._image_pull_last_poll_monotonic
                < IMAGE_PULL_POLL_SECONDS
            ):
                return
            self._image_pull_last_poll_monotonic = now_monotonic
            image_reference = str(
                info.get("image_uuid") or self.settings().runtime_image or ""
            ).strip()
            architecture = str(info.get("cpu_arch") or "amd64").strip().lower()
            if not image_reference:
                print(
                    "[VAST_PULL][ERROR] 이미지 pull 진행률을 계산할 이미지 참조 없음: "
                    f"instance={target}"
                )
                current.update(
                    phase="unavailable",
                    last_checked_at=self._utc_now(),
                    error="인스턴스 응답에 이미지 참조가 없습니다.",
                )
                self.launch["image_pull"] = current
                return
            try:
                client = self._client_or_raise()
                daemon_log = await client.get_instance_logs(
                    target, daemon_logs=True, tail=IMAGE_PULL_LOG_TAIL
                )
                manifest_key = f"{image_reference}|{architecture}"
                manifest = self._image_manifest_cache.get(manifest_key)
                if manifest is None:
                    manifest = await client.get_docker_hub_manifest_layers(
                        image_reference, architecture=architecture, os_name="linux"
                    )
                    self._image_manifest_cache[manifest_key] = dict(manifest)
                layer_states = parse_daemon_pull_states(daemon_log)
                progress = build_pull_progress(
                    list(manifest.get("layers") or []), layer_states
                )
            except Exception as exc:
                print(
                    "[VAST_PULL][ERROR] 이미지 pull 관측 갱신 실패: "
                    f"instance={target}, image={image_reference}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                current.update(
                    phase="unavailable" if not current.get("available") else current.get("phase", "tracking"),
                    image=image_reference,
                    last_checked_at=self._utc_now(),
                    error=str(exc),
                )
                self.launch["image_pull"] = current
                return

            if int(self.launch.get("instance_id") or 0) != target:
                print(
                    "[VAST_PULL] 관측 중 대상 인스턴스가 변경되어 결과를 폐기합니다: "
                    f"observed={target}, current={self.launch.get('instance_id')}"
                )
                return

            now = time.time()
            previous_signature = str(current.get("_signature") or "")
            signature = str(progress.get("_signature") or "")
            observed_progress = bool(signature and signature != previous_signature)
            last_observed = float(
                current.get("last_observed_progress_at_epoch") or 0.0
            )
            if observed_progress:
                last_observed = now
                self.launch["last_progress_at_epoch"] = now
                self.launch["updated_at"] = self._utc_now()

            progress.update(
                phase="complete" if progress.get("complete") else "tracking",
                image=image_reference,
                last_checked_at=self._utc_now(),
                last_observed_progress_at_epoch=last_observed,
                error="",
            )
            self.launch["image_pull"] = progress
            if observed_progress:
                self._event(
                    "pull",
                    f"#{target} Docker 이미지 최소 진행률 "
                    f"{float(progress.get('minimum_percent') or 0.0):.1f}% "
                    f"({int(progress.get('confirmed_layers') or 0)}/"
                    f"{int(progress.get('total_layers') or 0)} layers, "
                    f"confirmed={int(progress.get('confirmed_bytes') or 0)} bytes)",
                )

    def _ensure_watchdog(self, launch_id: str) -> None:
        if self._watchdog_task is not None and not self._watchdog_task.done():
            if self._watchdog_launch_id == launch_id:
                return
            self._watchdog_task.cancel()
        self._watchdog_launch_id = launch_id
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(launch_id), name=f"vast-watchdog-{launch_id}"
        )

    async def _watchdog_loop(self, launch_id: str) -> None:
        missing_count = 0
        while launch_id == str(self.launch.get("launch_id") or ""):
            state = str(self.launch.get("state") or "")
            if state in {"idle", "destroyed"}:
                return
            instance_id = int(self.launch.get("instance_id") or 0)
            if instance_id:
                try:
                    info, refreshed = await self._get_instance_status(
                        instance_id,
                        max_age_seconds=WATCHDOG_STATUS_MAX_AGE_SECONDS,
                    )
                    if info.get("id"):
                        missing_count = 0
                        await self._refresh_image_pull_progress(instance_id, info)
                    elif refreshed:
                        missing_count += 1
                        print(
                            "[VAST_GUARD] 단일 상태 응답에 인스턴스가 없음: "
                            f"instance={instance_id}, count={missing_count}"
                        )
                    if missing_count >= 2:
                        rows = await self._client_or_raise().list_instances()
                        if not any(int(row.get("id") or 0) == instance_id for row in rows):
                            self.launch["state"] = "destroyed"
                            self.launch["destroyed_instance_id"] = instance_id
                            self.launch["instance_id"] = None
                            self.launch["destroy_reason"] = "Vast 목록에서 인스턴스 소멸 확인"
                            self.launch["protection_state"] = "complete"
                            self._persist_guard_state()
                            print(
                                "[VAST_GUARD] 외부 파괴 확인: "
                                f"instance={instance_id}"
                            )
                            return
                        missing_count = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(
                        "[VAST_GUARD][ERROR] 인스턴스 감시 조회 실패: "
                        f"instance={instance_id}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

                if str(self.launch.get("state") or "") in {
                    "destroying",
                    "destroyed",
                }:
                    return
                status = str(self.launch.get("instance_status") or "").lower()
                if status in {"exited", "unknown", "offline"}:
                    try:
                        await self.destroy(
                            instance_id,
                            reason=f"Vast 비정상 상태 자동 정리: {status}",
                            automatic=True,
                        )
                    except Exception as exc:
                        print(
                            "[VAST_GUARD][ERROR] 비정상 상태 자동 파괴 실패: "
                            f"instance={instance_id}, error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()
                    return

                if state in {"creating", "preparing", "recovered"} and not bool(
                    self.launch.get("recovered_was_ready")
                ):
                    now = time.time()
                    contract_at = float(
                        self.launch.get("contract_started_at_epoch") or now
                    )
                    elapsed = max(0.0, now - contract_at)
                    reason = ""
                    hourly = float(self.launch.get("hourly_price_usd") or 0.0)
                    estimated = hourly * elapsed / 3600 if hourly > 0 else 0.0
                    if estimated >= MAX_BUILD_COST_USD:
                        reason = (
                            f"예상 빌드비 ${estimated:.3f}가 "
                            f"상한 ${MAX_BUILD_COST_USD:.2f}에 도달"
                        )
                    if reason:
                        print(
                            "[VAST_GUARD] 비용 보호 자동 파괴 시작: "
                            f"instance={instance_id}, reason={reason}"
                        )
                        try:
                            await self.destroy(
                                instance_id, reason=reason, automatic=True
                            )
                        except Exception as exc:
                            print(
                                "[VAST_GUARD][ERROR] 비용 보호 자동 파괴 실패: "
                                f"instance={instance_id}, error={type(exc).__name__}: {exc}"
                            )
                            traceback.print_exc()
                        return
                now = time.time()
                last_log = float(self.launch.get("last_watchdog_log_at_epoch") or 0.0)
                if now - last_log >= 30:
                    contract_at = float(
                        self.launch.get("contract_started_at_epoch") or now
                    )
                    elapsed = max(0.0, now - contract_at)
                    hourly = float(self.launch.get("hourly_price_usd") or 0.0)
                    estimated = hourly * elapsed / 3600 if hourly > 0 else 0.0
                    self.launch["last_watchdog_log_at_epoch"] = now
                    self._event(
                        "watch",
                        f"#{instance_id} step={self.launch.get('current_step') or '-'} "
                        f"vast={self.launch.get('instance_status') or '-'} "
                        f"elapsed={int(elapsed)}s estimated=${estimated:.4f} "
                        f"cost_limit_left=${max(0.0, MAX_BUILD_COST_USD - estimated):.4f} "
                        f"status_msg={str(self.launch.get('instance_status_msg') or '(없음)')[-600:]}",
                    )
            await asyncio.sleep(WATCHDOG_POLL_SECONDS)

    def reset_client(self) -> None:
        """API 키 변경 후 다음 조회에서 새 키로 클라이언트를 재생성하도록 캐시를 비운다."""
        if self._client is not None:
            try:
                asyncio.get_running_loop().create_task(self._client.close())
            except RuntimeError:
                # 실행 중인 이벤트 루프가 없으면 동기적으로 닫을 수 없다 — GC에 맡긴다.
                pass
        self._client = None
        self._instance_status_cache.clear()
        self._account_status_cache = None
        self._image_manifest_cache.clear()
        self._image_pull_last_poll_monotonic = 0.0

    def _close_comfy_tunnel(self) -> None:
        tunnel = self._comfy_tunnel
        self._comfy_tunnel = None
        if tunnel is None:
            return
        try:
            tunnel.close()
        except Exception as exc:
            print(
                "[VAST][TUNNEL][ERROR] 서비스 터널 종료 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    def _open_comfy_tunnel(
        self, host: str, port: int, private_key_path: str
    ) -> str:
        self._close_comfy_tunnel()
        ssh = self._ssh_connect(host, port, private_key_path)
        tunnel = ComfySshTunnel(ssh, remote_port=8188)
        try:
            url = tunnel.start()
        except Exception:
            try:
                ssh.close()
            except Exception as close_exc:
                print(
                    "[VAST][TUNNEL][ERROR] 시작 실패 후 SSH 닫기 실패: "
                    f"error={type(close_exc).__name__}: {close_exc}"
                )
                traceback.print_exc()
            raise
        self._comfy_tunnel = tunnel
        return url

    # ── 계정/오퍼 ───────────────────────────────────────────

    async def account_status(self) -> dict[str, Any]:
        def cached_status() -> dict[str, Any] | None:
            cached = self._account_status_cache
            if cached is None:
                return None
            expires_at, payload = cached
            if time.monotonic() >= expires_at:
                return None
            return dict(payload)

        cached = cached_status()
        if cached is not None:
            return cached

        async with self._account_status_lock:
            cached = cached_status()
            if cached is not None:
                return cached
            try:
                client = self._client_or_raise()
                data = await client.account()
                result = {
                    "ok": True,
                    "username": data.get("username"),
                    # 잔액 필드는 credit이다 (balance는 별도 의미).
                    "balance_usd": data.get("credit") or 0.0,
                    "api_key_valid": True,
                }
                ttl = ACCOUNT_STATUS_CACHE_SECONDS
            except VastApiError as exc:
                _log(f"계정 확인 실패: {exc}")
                traceback.print_exc()
                result = {
                    "ok": False,
                    "error": str(exc),
                    "api_key_valid": False,
                }
                ttl = ACCOUNT_STATUS_ERROR_CACHE_SECONDS
            self._account_status_cache = (time.monotonic() + ttl, result)
            return dict(result)

    async def offers(
        self,
        *,
        gpu_names: list[str] | None = None,
        min_cpu_ram_gb: int | None = None,
        min_disk_gb: int = 0,
        max_price_usd_hr: float | None = None,
        verified_only: bool | None = None,
        on_demand: bool | None = None,
        min_gpu_ram_gb: int | None = None,
        inet_down_min_mbps: int | None = None,
        inet_up_min_mbps: float | None = None,
        min_direct_port_count: int | None = None,
        min_reliability: float | None = None,
        min_cuda_version: float | None = None,
        limit: int = 60,
    ) -> dict[str, Any]:
        cfg = self.settings()
        client = self._client_or_raise()
        requested_min_cuda = (
            float(min_cuda_version) if min_cuda_version is not None else 0.0
        )
        applied_min_cuda = max(MIN_RUNTIME_CUDA_VERSION, requested_min_cuda)
        offers = await client.search_offers(
            gpu_names=gpu_names,
            min_cpu_ram_gb=cfg.min_cpu_ram_gb if min_cpu_ram_gb is None else min_cpu_ram_gb,
            min_disk_gb=min_disk_gb,
            max_price_usd_hr=cfg.max_price_usd_hr if max_price_usd_hr is None else max_price_usd_hr,
            verified_only=cfg.verified_only if verified_only is None else verified_only,
            on_demand=cfg.on_demand if on_demand is None else on_demand,
            min_gpu_ram_gb=min_gpu_ram_gb if min_gpu_ram_gb is not None else 0,
            inet_down_min_mbps=inet_down_min_mbps if inet_down_min_mbps is not None else 1000,
            inet_up_min_mbps=inet_up_min_mbps if inet_up_min_mbps is not None else 0,
            min_direct_port_count=min_direct_port_count if min_direct_port_count is not None else 0,
            min_reliability=min_reliability if min_reliability is not None else 0.0,
            min_cuda_version=applied_min_cuda,
            limit=limit,
        )
        return {
            "ok": True,
            "min_cuda_version": applied_min_cuda,
            "offers": [
                {
                    "id": o.get("id"),
                    "gpu_name": o.get("gpu_name"),
                    "num_gpus": o.get("num_gpus"),
                    "cpu_ram_gb": round(float(o.get("cpu_ram") or 0) / 1024, 1),
                    "gpu_ram_gb": round(float(o.get("gpu_ram") or 0) / 1024, 1),
                    "disk_gb": float(o.get("disk_space") or 0),
                    "dph_total": float(o.get("dph_total") or 0),
                    # dph_total은 Vast 기본 디스크 할당(약 8GB) 기준이라
                    # 실제 요금 예측엔 GPU 단가(dph_base) + 저장 단가가 필요하다.
                    # storage_cost는 $/GB·월, 저장료는 이를 720hr/월으로 나눠 과금.
                    "dph_base": float(o.get("dph_base") or 0),
                    "storage_cost_usd_per_gb_month": float(o.get("storage_cost") or 0),
                    "inet_cost_usd_per_tb_down": float(o.get("internet_down_cost_per_tb") or 0),
                    "inet_cost_usd_per_tb_up": float(o.get("internet_up_cost_per_tb") or 0),
                    "inet_down_mbps": float(o.get("inet_down") or 0),
                    "inet_up_mbps": float(o.get("inet_up") or 0),
                    "direct_port_count": int(o.get("direct_port_count") or 0),
                    "reliability": float(o.get("reliability2") or o.get("reliability") or 0),
                    "cuda_max_good": o.get("cuda_max_good"),
                    "geolocation": o.get("geolocation"),
                    "verified": str(o.get("verification") or "").lower() == "verified",
                }
                for o in offers
            ],
        }

    # ── 마법사 계획 (②단계) ─────────────────────────────────

    def wizard_plan(
        self,
        *,
        workflow_files: list[dict[str, Any]],
        lora_files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """선택 워크플로우의 모델 참조 + LoRA 체크 목록으로 준비 계획을 만든다."""
        import json as _json

        from modal_backend.workflow_assets import (
            build_local_model_index,
            resolve_workflow_model_files,
        )

        config = self._get_config()
        settings = self.settings()
        mapping = load_mapping(self.project_root)
        defaults = defaults_from_manifest(self.project_root)
        workflows: list[dict[str, Any]] = []
        for wf in workflow_files:
            path = Path(str(wf.get("path") or ""))
            if not path.is_file():
                print(f"[VAST] 워크플로우 파일 없음(건너뜀): {path}")
                continue
            try:
                workflows.append(_json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                print(f"[VAST] 워크플로우 해석 실패(건너뜀): {path}, error={exc}")
        model_files: list[dict[str, Any]] = []
        if workflows:
            model_index = build_local_model_index(self.project_root / "comfy")
            resolved = resolve_workflow_model_files(
                workflows, model_index, include_hashes=False
            )
            for item in resolved.get("model_files") or []:
                kind, _, filename = str(item.get("remote_path") or "").partition("/")
                model_files.append(
                    {
                        "kind": kind,
                        "filename": filename,
                        "size_bytes": item.get("size") or 0,
                        "source_path": item.get("source_path") or "",
                    }
                )
        plan = build_download_plan(
            model_files,
            mapping,
            manifest_defaults=defaults,
            civitai_api_key=settings.civitai_api_key,
        )
        lora_gb = sum(
            int(f.get("size_bytes") or f.get("size") or 0) for f in lora_files
        ) / 1024**3
        includes_video = any("영상" in str(wf.get("name", "")) for wf in workflow_files)
        return {
            "ok": True,
            "models": plan["items"],
            "totals": {
                **plan["totals"],
                "lora_upload_gb": round(lora_gb, 2),
                "recommended_disk_gb": settings.recommend_disk_gb(
                    model_gb=plan["totals"]["download_gb"] + plan["totals"]["upload_gb"],
                    lora_gb=lora_gb,
                    includes_video=includes_video,
                ),
            },
        }

    # ── SSH 키 관리 ─────────────────────────────────────────

    def _ssh_key_paths(self) -> tuple[Path, Path]:
        private = self.project_root / "key" / "vast_ssh_key"
        return private, Path(str(private) + ".pub")

    def ensure_ssh_keypair(self) -> tuple[str, str]:
        """로컬 키페어(없으면 생성). 반환: (개인키 경로, 공개키 문자열)."""
        import paramiko

        private_path, public_path = self._ssh_key_paths()
        if not private_path.exists():
            if public_path.exists():
                message = (
                    "Vast SSH 공개키만 있고 개인키가 없습니다. 기존 공개키를 "
                    f"덮어쓰지 않습니다: public={public_path}, private={private_path}"
                )
                print(f"[VAST][SSH_KEY][ERROR] {message}")
                raise VastApiError(message)
            try:
                private_path.parent.mkdir(parents=True, exist_ok=True)
                _log(f"SSH 키페어 생성: {private_path}")
                key = paramiko.RSAKey.generate(2048)
                key.write_private_key_file(str(private_path))
                with open(str(public_path), "w", encoding="utf-8") as fh:
                    fh.write(f"{key.get_name()} {key.get_base64()} soya-vast\n")
            except Exception as exc:
                print(
                    "[VAST][SSH_KEY][ERROR] SSH 키페어 생성 실패: "
                    f"private={private_path}, public={public_path}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise VastApiError(f"Vast SSH 키페어를 생성할 수 없습니다: {exc}") from exc
        try:
            key = paramiko.RSAKey.from_private_key_file(str(private_path))
            expected = f"{key.get_name()} {key.get_base64()} soya-vast"
            if not public_path.exists():
                print(f"[VAST][SSH_KEY] 누락된 공개키 복구: {public_path}")
                public_path.write_text(expected + "\n", encoding="utf-8")
            public = public_path.read_text(encoding="utf-8").strip()
            fields = public.split()
            if len(fields) < 2 or fields[0] != key.get_name() or fields[1] != key.get_base64():
                message = (
                    "Vast SSH 개인키와 공개키가 서로 일치하지 않습니다. "
                    f"private={private_path}, public={public_path}"
                )
                print(f"[VAST][SSH_KEY][ERROR] {message}")
                raise VastApiError(message)
        except VastApiError:
            raise
        except (OSError, paramiko.SSHException, ValueError) as exc:
            print(
                "[VAST][SSH_KEY][ERROR] SSH 키페어 검증 실패: "
                f"private={private_path}, public={public_path}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise VastApiError(f"Vast SSH 키페어를 읽을 수 없습니다: {exc}") from exc
        return str(private_path), public

    # ── 생성 (④단계, 백그라운드) ────────────────────────────

    def _set_step(self, key: str, state: str, detail: str = "") -> None:
        self._check_cancelled()
        self._set_step_unchecked(key, state, detail)

    def _build_onstart(self) -> str:
        """ComfyUI 시작 스크립트. 빌드 시간 제한은 없다(비용 상한은 서버 watchdog이 담당)."""
        return (
            "#!/bin/bash\n"
            "echo '[onstart] 빌드 완료 신호 대기 중' >> /tmp/soya_onstart.log\n"
            f"while [ ! -f {READY_FLAG} ]; do sleep 2; done\n"
            f"cd {COMFY_ROOT_REMOTE}\n"
            "exec python main.py --listen 0.0.0.0 --port 8188"
        )

    async def start_launch(
        self,
        *,
        ask_id: int,
        disk_gb: int,
        model_plan: dict[str, Any],
        lora_files: list[dict[str, Any]],
        install_payload: dict[str, Any],
        adopt_instance_id: int | None = None,
        hourly_price_usd: float = 0.0,
    ) -> dict[str, Any]:
        async with self._launch_lock:
            return await self._start_launch_locked(
                ask_id=ask_id,
                disk_gb=disk_gb,
                model_plan=model_plan,
                lora_files=lora_files,
                install_payload=install_payload,
                adopt_instance_id=adopt_instance_id,
                hourly_price_usd=hourly_price_usd,
            )

    async def _start_launch_locked(
        self,
        *,
        ask_id: int,
        disk_gb: int,
        model_plan: dict[str, Any],
        lora_files: list[dict[str, Any]],
        install_payload: dict[str, Any],
        adopt_instance_id: int | None = None,
        hourly_price_usd: float = 0.0,
    ) -> dict[str, Any]:
        """인스턴스를 생성(또는 adopt_instance_id로 기존 인스턴스 재활용)해 준비한다."""
        if self.launch["state"] in {"creating", "preparing", "destroying"}:
            raise VastApiError(
                f"이미 생성/준비 진행 중입니다(instance_id={self.launch['instance_id']})."
            )
        client = self._client_or_raise()
        if not adopt_instance_id:
            rows = await client.list_instances()
            owned_ids = [
                int(row.get("id") or 0)
                for row in rows
                if str(row.get("label") or "").startswith(SERVICE_LABEL_PREFIX)
                or str(row.get("label") or "") == "soya-vast"
            ]
            if owned_ids:
                print(
                    "[VAST_GUARD] 중복 생성 차단 — 기존 서비스 인스턴스 존재: "
                    f"ids={owned_ids}"
                )
                raise VastApiError(
                    "기존 Soya Vast 인스턴스가 남아 있습니다. "
                    f"먼저 파괴하거나 확인하세요: {owned_ids}"
                )
        self._close_comfy_tunnel()
        self._instance_status_cache.clear()
        self._image_pull_last_poll_monotonic = 0.0
        launch_id = uuid.uuid4().hex[:12]
        label = f"{SERVICE_LABEL_PREFIX}{launch_id}"
        self.launch = self._new_launch_state(
            state="creating",
            launch_id=launch_id,
            label=label,
            hourly_price_usd=hourly_price_usd,
        )
        self._event(
            "start",
            f"launch={launch_id} ask={ask_id} disk={disk_gb}GB "
            f"rate=${float(hourly_price_usd or 0.0):.4f}/hr, "
            f"limits=cost:${MAX_BUILD_COST_USD:.2f}",
        )
        self._cancel_events[launch_id] = threading.Event()
        task = asyncio.create_task(
            self._launch(
                ask_id,
                disk_gb,
                model_plan,
                lora_files,
                install_payload,
                adopt_instance_id=adopt_instance_id,
                launch_id=launch_id,
                label=label,
            ),
            name=f"vast-launch-{launch_id}",
        )
        self._launch_task = task
        task.add_done_callback(self._launch_done)
        self._ensure_watchdog(launch_id)
        return self.launch_status()

    def _launch_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            print(f"[VAST] 생성 태스크 취소 완료: task={task.get_name()}")
            return
        exc = task.exception()
        if isinstance(exc, LaunchCancelled):
            print(f"[VAST] 생성 태스크 협력 취소 완료: {exc}")
            return
        if exc is not None:
            _log(f"생성 태스크 실패: {type(exc).__name__}: {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            self._close_comfy_tunnel()
            if self.launch.get("state") in {"destroying", "destroyed"}:
                return
            self.launch["state"] = "error"
            self.launch["error"] = str(exc)
            self.launch["protection_state"] = "triggered"
            self.launch["protection_reason"] = f"빌드 실패 자동 정리: {exc}"
            self._persist_guard_state()
            # 실패 시 남은 인스턴스는 그대로 과금되므로 즉시 파괴한다.
            instance_id = self.launch.get("instance_id")
            if instance_id:
                asyncio.create_task(
                    self._destroy_quietly(
                        int(instance_id), reason=f"빌드 실패 자동 정리: {exc}"
                    )
                )

    async def _destroy_quietly(self, instance_id: int, *, reason: str) -> None:
        try:
            await self.destroy(instance_id, reason=reason, automatic=True)
        except Exception as exc:
            _log(
                f"실패 정리용 인스턴스 파괴 실패(수동 확인 필요): "
                f"id={instance_id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    async def _launch(
        self,
        ask_id: int,
        disk_gb: int,
        model_plan: dict[str, Any],
        lora_files: list[dict[str, Any]],
        install_payload: dict[str, Any],
        *,
        adopt_instance_id: int | None = None,
        launch_id: str,
        label: str,
    ) -> None:
        context_token = _LAUNCH_CONTEXT.set(launch_id)
        try:
            await self._launch_inner(
                ask_id,
                disk_gb,
                model_plan,
                lora_files,
                install_payload,
                adopt_instance_id=adopt_instance_id,
                label=label,
            )
        finally:
            _LAUNCH_CONTEXT.reset(context_token)

    async def _launch_inner(
        self,
        ask_id: int,
        disk_gb: int,
        model_plan: dict[str, Any],
        lora_files: list[dict[str, Any]],
        install_payload: dict[str, Any],
        *,
        adopt_instance_id: int | None,
        label: str,
    ) -> None:
        self._check_cancelled()
        cfg = self.settings()
        client = self._client_or_raise()
        private_key_path, public_key = self.ensure_ssh_keypair()

        if adopt_instance_id:
            instance_id = int(adopt_instance_id)
            self.launch["instance_id"] = instance_id
            self.launch["state"] = "preparing"
            self.launch["contract_started_at_epoch"] = time.time()
            self._set_step("instance", "done", f"기존 인스턴스 재활용: {instance_id}")
            self._persist_guard_state()
        else:
            # 계정 수준 키 등록(실패해도 진행) — 생성되는 인스턴스에 자동 적용.
            try:
                await client.register_account_ssh_key(public_key)
                self._event("ssh", "계정 SSH 공개키 등록 완료")
            except VastApiError as exc:
                if "already" in str(exc).lower():
                    self._event("ssh", "계정 SSH 공개키가 이미 등록되어 있음")
                else:
                    print(
                        "[VAST] 계정 SSH 키 등록 실패(인스턴스 부착으로 대체): "
                        f"{exc}"
                    )
            self._set_step("instance", "running", "인스턴스 생성 요청")
            created = await client.create_instance(
                ask_id=ask_id,
                image=cfg.runtime_image,
                disk_gb=disk_gb,
                onstart_cmd=self._build_onstart(),
                label=label,
            )
            instance_id = int(created["new_contract"])
            self.launch["instance_id"] = instance_id
            self.launch["state"] = "preparing"
            self.launch["contract_started_at_epoch"] = time.time()
            self._set_step("instance", "done", f"인스턴스 #{instance_id} 생성 완료")
            self._persist_guard_state()

        self._set_step("ssh", "running", "SSH 대기")
        ssh_host, ssh_port = await self._wait_ssh(instance_id)
        self.launch["ssh_ready_at_epoch"] = time.time()
        # 생성 요청의 ssh_key 필드는 무시되므로(검증됨) running 후 부착 API로 등록한다.
        await self._attach_key_with_retry(client, instance_id, public_key)
        self._set_step("ssh", "done", f"{ssh_host}:{ssh_port}")

        # Docker/SSH 준비 뒤 실제 원격 경로와 로컬 SFTP를 짧게 측정한다.
        # 측정 실패는 결과에 남기되 본 빌드를 차단하지 않는다.
        await asyncio.to_thread(
            self._run_preflight,
            ssh_host,
            ssh_port,
            private_key_path,
            model_plan,
            lora_files,
        )

        upload_task = asyncio.to_thread(
            self._upload_all,
            ssh_host, ssh_port, private_key_path, install_payload, lora_files, model_plan,
        )
        download_task = asyncio.to_thread(
            self._run_remote_downloads, ssh_host, ssh_port, private_key_path, model_plan
        )
        await asyncio.gather(upload_task, download_task)

        self._set_step("tunnel", "running", "SSH 로컬 포워더 생성")
        comfy_url = await asyncio.to_thread(
            self._open_comfy_tunnel, ssh_host, ssh_port, private_key_path
        )
        self._set_step("tunnel", "done", comfy_url)
        self._set_step("comfy", "running", "ComfyUI 기동 대기")
        await self._start_comfy_and_wait(ssh_host, ssh_port, private_key_path, comfy_url)
        self.launch["comfy_base_url"] = comfy_url
        self.launch["state"] = "ready"
        self.launch["protection_state"] = "ready"
        self.launch["protection_reason"] = "빌드 완료 — 비용 보호 해제"
        self._set_step("comfy", "done", comfy_url)
        self._persist_guard_state()

    async def _wait_ssh(self, instance_id: int) -> tuple[str, int]:
        # 시간 제한 없이 running+SSH 정보를 기다린다. 비용 상한은 watchdog이 담당.
        last_status = ""
        while True:
            self._check_cancelled()
            info, _refreshed = await self._get_instance_status(
                instance_id,
                max_age_seconds=max(1.0, SSH_STATUS_POLL_SECONDS - 1.0),
            )
            status = str(info.get("actual_status") or "").lower()
            status_msg = str(info.get("status_msg") or "")
            status_signature = f"{status}|{status_msg}"
            if status_signature != last_status:
                print(
                    f"[VAST] 인스턴스 준비 상태: instance={instance_id}, "
                    f"status={status or '(없음)'}, message={status_msg or '(없음)'}"
                )
                last_status = status_signature
            if status in {"exited", "unknown", "offline"}:
                raise VastApiError(
                    "Vast 인스턴스가 SSH 준비 전에 종료 상태가 되었습니다: "
                    f"instance={instance_id}, status={status}, message={status_msg}"
                )
            ssh_host = str(info.get("ssh_host") or "").split("@")[-1]
            try:
                ssh_port = int(info.get("ssh_port") or 0)
            except (TypeError, ValueError):
                ssh_port = 0
            if status == "running" and ssh_host and ssh_port:
                return ssh_host, ssh_port
            await asyncio.sleep(SSH_STATUS_POLL_SECONDS)

    async def _attach_key_with_retry(
        self, client: VastClient, instance_id: int, public_key: str
    ) -> None:
        """running 직후 부착 API는 일시적 서버 오류를 낼 수 있어 재시도한다."""
        import time

        last_error: Exception | None = None
        for attempt in range(6):
            try:
                await client.attach_ssh_key(instance_id, public_key)
                self._event(
                    "ssh",
                    f"SSH 키 부착 성공(시도 {attempt + 1}): instance={instance_id}",
                )
                return
            except VastApiError as exc:
                if "already associated" in str(exc):
                    self._event(
                        "ssh", f"SSH 키 이미 등록됨(성공으로 간주): {instance_id}"
                    )
                    return
                last_error = exc
                print(
                    f"[VAST] SSH 키 부착 재시도({attempt + 1}/6): "
                    f"instance={instance_id}, error={exc}"
                )
                self._event(
                    "ssh-retry",
                    f"SSH 키 부착 재시도 {attempt + 1}/6: "
                    f"instance={instance_id}, error={exc}",
                )
                await asyncio.sleep(10)
        raise VastApiError(
            f"SSH 키 부착 실패: instance={instance_id}, last={last_error}"
        )

    def _ssh_connect(self, host: str, port: int, private_key_path: str):
        """SSH 접속 — 키 부착 직후 데몬 재시작으로 일시 거부될 수 있어 재시도한다."""
        import time

        import paramiko

        last_error: Exception | None = None
        for attempt in range(10):
            self._check_cancelled()
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    host,
                    port=port,
                    username="root",
                    key_filename=private_key_path,
                    timeout=SSH_CONNECT_TIMEOUT_SECONDS,
                    banner_timeout=SSH_CONNECT_TIMEOUT_SECONDS,
                    auth_timeout=SSH_CONNECT_TIMEOUT_SECONDS,
                )
                if attempt:
                    self._event(
                        "ssh", f"SSH 접속 성공(시도 {attempt + 1}): {host}:{port}"
                    )
                return client
            except (paramiko.ssh_exception.SSHException, OSError) as exc:
                last_error = exc
                print(
                    f"[VAST] SSH 접속 재시도({attempt + 1}/10): {host}:{port}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                self._event(
                    "ssh-retry",
                    f"SSH 접속 재시도 {attempt + 1}/10: {host}:{port} — "
                    f"{type(exc).__name__}: {exc}",
                )
                try:
                    client.close()
                except Exception:
                    pass
                self._wait_sync(10)
        raise VastApiError(
            f"SSH 접속 실패: {host}:{port}, last={type(last_error).__name__}: {last_error}"
        )

    def _store_preflight(self, **updates: Any) -> None:
        with self._state_lock:
            current = dict(self.launch.get("preflight") or empty_preflight_state())
            current.update(updates)
            self.launch["preflight"] = current
            self.launch["last_progress_at_epoch"] = time.time()
            self.launch["updated_at"] = self._utc_now()

    def _record_preflight_test(self, result: dict[str, Any]) -> None:
        key = str(result.get("key") or "")
        with self._state_lock:
            current = dict(self.launch.get("preflight") or empty_preflight_state())
            tests_by_key = {
                str(item.get("key") or ""): dict(item)
                for item in current.get("tests") or []
                if isinstance(item, dict)
            }
            tests_by_key[key] = dict(result)
            order = ("docker", "cloudflare", "huggingface", "upload")
            current["tests"] = [
                tests_by_key[item_key]
                for item_key in order
                if item_key in tests_by_key
            ] + [
                value
                for item_key, value in tests_by_key.items()
                if item_key not in order
            ]
            self.launch["preflight"] = current
            self.launch["last_progress_at_epoch"] = time.time()
            self.launch["updated_at"] = self._utc_now()
        speed = float(result.get("mbps") or 0.0)
        speed_text = f"{speed:.1f} Mbps" if speed > 0 else str(result.get("detail") or "-")
        self._event(
            "preflight",
            f"{result.get('label') or key}: {result.get('status') or '?'} — {speed_text}",
        )

    def _docker_preflight_result(self) -> dict[str, Any]:
        contract_at = float(self.launch.get("contract_started_at_epoch") or 0.0)
        ssh_ready_at = float(self.launch.get("ssh_ready_at_epoch") or time.time())
        pull_started_at = contract_at
        for item in self.launch.get("status_history") or []:
            if str(item.get("status") or "").lower() != "loading":
                continue
            try:
                observed_at = datetime.fromisoformat(str(item.get("at") or "")).timestamp()
            except (TypeError, ValueError):
                print(
                    "[VAST][PREFLIGHT][ERROR] Docker loading 시각 해석 실패: "
                    f"item={item!r}"
                )
                traceback.print_exc()
                continue
            if observed_at > 0:
                pull_started_at = observed_at
                break
        seconds = max(0.001, ssh_ready_at - pull_started_at) if pull_started_at else 0.0
        pull = self.launch.get("image_pull") or {}
        try:
            total_bytes = max(0, int(pull.get("total_bytes") or 0))
            observed_layers = max(0, int(pull.get("observed_layers") or 0))
        except (TypeError, ValueError):
            print(
                "[VAST][PREFLIGHT][ERROR] Docker pull 수치 해석 실패: "
                f"total={pull.get('total_bytes')!r}, observed={pull.get('observed_layers')!r}"
            )
            traceback.print_exc()
            total_bytes = 0
            observed_layers = 0
        if total_bytes > 0 and observed_layers > 0 and seconds > 0:
            return speed_result(
                key="docker",
                label="Docker 준비",
                transferred_bytes=total_bytes,
                seconds=seconds,
                detail="Vast daemon에서 관측된 압축 레이어 기준 실효 속도",
            )
        return informational_result(
            key="docker",
            label="Docker 준비",
            seconds=seconds,
            detail=(
                f"런타임 준비 {seconds:.1f}초 · 캐시/레이어 바이트 미관측으로 속도 계산 제외"
            ),
        )

    @staticmethod
    def _preflight_huggingface_target(
        model_plan: dict[str, Any],
    ) -> tuple[str, int, str]:
        from urllib.parse import quote

        candidates: list[tuple[int, str, str]] = []
        for item in model_plan.get("models") or []:
            if not isinstance(item, dict):
                continue
            source = item.get("source") or {}
            if str(source.get("source_type") or "") != "hf":
                continue
            repo_id = str(source.get("repo_id") or "").strip()
            filename = str(source.get("hf_filename") or "").strip()
            if not repo_id or not filename:
                print(
                    "[VAST][PREFLIGHT][ERROR] HF 측정 대상 필드 누락: "
                    f"key={item.get('key')!r}, repo_id={repo_id!r}, filename={filename!r}"
                )
                continue
            try:
                size_bytes = max(0, int(item.get("size_bytes") or 0))
            except (TypeError, ValueError):
                print(
                    "[VAST][PREFLIGHT][ERROR] HF 측정 대상 크기 해석 실패: "
                    f"key={item.get('key')!r}, size={item.get('size_bytes')!r}"
                )
                traceback.print_exc()
                size_bytes = 0
            url = (
                f"https://huggingface.co/{quote(repo_id, safe='/')}"
                f"/resolve/main/{quote(filename, safe='/')}"
            )
            candidates.append((size_bytes, url, str(item.get("filename") or filename)))
        if candidates:
            size_bytes, url, filename = max(candidates, key=lambda item: item[0])
            requested_bytes = min(
                PREFLIGHT_DOWNLOAD_BYTES,
                size_bytes if size_bytes > 0 else PREFLIGHT_DOWNLOAD_BYTES,
            )
            return url, max(1, requested_bytes), f"선택 모델 구간 측정: {filename}"
        print(
            "[VAST][PREFLIGHT] HF 직접 다운로드 모델이 없어 공개 기준 파일로 측정합니다."
        )
        return (
            PREFLIGHT_HF_FALLBACK_URL,
            PREFLIGHT_DOWNLOAD_BYTES,
            "공개 기준 파일 구간 측정",
        )

    def _run_curl_speed_probe(
        self,
        ssh,
        *,
        key: str,
        label: str,
        url: str,
        requested_bytes: int,
        detail: str,
    ) -> dict[str, Any]:
        import shlex

        self._check_cancelled()
        upper_byte = max(0, int(requested_bytes) - 1)
        write_out = (
            "\\n__SOYA_SPEED__:%{size_download}:%{time_total}:"
            "%{http_code}:%{speed_download}\\n"
        )
        command = (
            "curl --location --silent --show-error --output /dev/null "
            "--connect-timeout 8 "
            f"--max-time {PREFLIGHT_CURL_MAX_SECONDS} "
            f"--max-filesize {max(1, int(requested_bytes) * 2)} "
            f"--range 0-{upper_byte} "
            "--user-agent soya-vast-preflight/1 "
            f"--write-out {shlex.quote(write_out)} {shlex.quote(url)}"
        )
        _stdin, stdout, stderr = ssh.exec_command(command)
        stdout_text = stdout.read().decode("utf-8", "replace")
        stderr_text = stderr.read().decode("utf-8", "replace").strip()
        exit_code = stdout.channel.recv_exit_status()
        if stderr_text:
            print(
                f"[VAST][PREFLIGHT] {label} curl stderr: "
                f"exit={exit_code}, detail={stderr_text[-800:]}"
            )
        self._check_cancelled()
        return parse_curl_speed_probe(
            stdout_text,
            exit_code=exit_code,
            key=key,
            label=label,
            detail=detail,
        )

    def _run_sftp_speed_probe(self, ssh) -> dict[str, Any]:
        import io

        self._check_cancelled()
        remote_path = f"/tmp/soya_preflight_upload_{self.launch.get('launch_id') or 'test'}.bin"
        payload = io.BytesIO(os.urandom(PREFLIGHT_UPLOAD_BYTES))
        sftp = ssh.open_sftp()
        try:
            channel = sftp.get_channel()
            channel.settimeout(PREFLIGHT_CURL_MAX_SECONDS + 10)
            started = time.perf_counter()
            sftp.putfo(
                payload,
                remote_path,
                file_size=PREFLIGHT_UPLOAD_BYTES,
                confirm=True,
            )
            seconds = time.perf_counter() - started
            self._check_cancelled()
            return speed_result(
                key="upload",
                label="로컬→Vast",
                transferred_bytes=PREFLIGHT_UPLOAD_BYTES,
                seconds=seconds,
                detail="임시 SFTP 업로드",
            )
        finally:
            try:
                sftp.remove(remote_path)
            except OSError as exc:
                print(
                    "[VAST][PREFLIGHT][ERROR] 임시 SFTP 파일 삭제 실패: "
                    f"path={remote_path}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
            try:
                sftp.close()
            except Exception as exc:
                print(
                    "[VAST][PREFLIGHT][ERROR] SFTP 측정 채널 종료 실패: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

    def _run_preflight(
        self,
        host: str,
        port: int,
        private_key_path: str,
        model_plan: dict[str, Any],
        lora_files: list[dict[str, Any]],
    ) -> None:
        started_at_epoch = time.time()
        state = empty_preflight_state()
        state.update(state="running", started_at=self._utc_now())
        self._store_preflight(**state)
        self._set_step("preflight", "running", "실사용 전송 속도 측정")
        self._event(
            "preflight",
            f"자동 프리플라이트 시작: ssh={host}:{port}, "
            f"download_sample={PREFLIGHT_DOWNLOAD_BYTES}, upload_sample={PREFLIGHT_UPLOAD_BYTES}",
        )

        ssh = None
        tests: list[dict[str, Any]] = []

        def run_one(key: str, label: str, callback) -> None:
            self._check_cancelled()
            try:
                result = callback()
            except LaunchCancelled:
                raise
            except Exception as exc:
                print(
                    f"[VAST][PREFLIGHT][ERROR] {label} 측정 실패: "
                    f"key={key}, host={host}, port={port}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                result = failed_result(
                    key=key,
                    label=label,
                    detail=f"{type(exc).__name__}: {str(exc)[:240]}",
                )
            tests.append(result)
            self._record_preflight_test(result)

        try:
            run_one("docker", "Docker 준비", self._docker_preflight_result)
            ssh = self._ssh_connect(host, port, private_key_path)
            run_one(
                "cloudflare",
                "Cloudflare",
                lambda: self._run_curl_speed_probe(
                    ssh,
                    key="cloudflare",
                    label="Cloudflare",
                    url=f"{PREFLIGHT_CLOUDFLARE_URL}?bytes={PREFLIGHT_DOWNLOAD_BYTES}",
                    requested_bytes=PREFLIGHT_DOWNLOAD_BYTES,
                    detail="Cloudflare edge 제한 구간 측정",
                ),
            )
            hf_url, hf_bytes, hf_detail = self._preflight_huggingface_target(model_plan)
            run_one(
                "huggingface",
                "Hugging Face",
                lambda: self._run_curl_speed_probe(
                    ssh,
                    key="huggingface",
                    label="Hugging Face",
                    url=hf_url,
                    requested_bytes=hf_bytes,
                    detail=hf_detail,
                ),
            )
            run_one("upload", "로컬→Vast", lambda: self._run_sftp_speed_probe(ssh))

            estimate = calculate_transfer_estimate(model_plan, lora_files, tests)
            if not estimate.get("available"):
                print(
                    "[VAST][PREFLIGHT][ERROR] 전송 ETA 계산 불가: "
                    f"download_bytes={estimate.get('download_bytes')}, "
                    f"upload_bytes={estimate.get('upload_bytes')}, "
                    f"note={estimate.get('note')}"
                )
            has_failure = any(item.get("status") == "error" for item in tests)
            final_state = "partial" if has_failure else "complete"
            elapsed = max(0.0, time.time() - started_at_epoch)
            self._store_preflight(
                state=final_state,
                completed_at=self._utc_now(),
                elapsed_seconds=round(elapsed, 1),
                tests=tests,
                estimate=estimate,
                error="",
            )
            remaining = estimate.get("remaining_seconds")
            if remaining is None:
                step_detail = "일부 측정 실패 · ETA 없음 · 빌드 자동 계속"
            else:
                step_detail = f"예상 남은 전송 {int(remaining)}초 · 빌드 자동 계속"
            self._set_step("preflight", "done", step_detail)
            self._event("preflight", step_detail)
        except LaunchCancelled:
            self._store_preflight(
                state="cancelled",
                completed_at=self._utc_now(),
                elapsed_seconds=round(max(0.0, time.time() - started_at_epoch), 1),
                error="사용자 요청으로 취소됨",
            )
            raise
        except Exception as exc:
            print(
                "[VAST][PREFLIGHT][ERROR] 자동 프리플라이트 전체 실패(빌드 계속): "
                f"host={host}, port={port}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            estimate = calculate_transfer_estimate(model_plan, lora_files, tests)
            self._store_preflight(
                state="failed",
                completed_at=self._utc_now(),
                elapsed_seconds=round(max(0.0, time.time() - started_at_epoch), 1),
                tests=tests,
                estimate=estimate,
                error=f"{type(exc).__name__}: {str(exc)[:500]}",
            )
            self._set_step("preflight", "done", "측정 실패 · 빌드 자동 계속")
            self._event("preflight-error", f"측정 실패 · 빌드 자동 계속: {exc}")
        finally:
            if ssh is not None:
                try:
                    ssh.close()
                except Exception as exc:
                    print(
                        "[VAST][PREFLIGHT][ERROR] SSH 측정 연결 종료 실패: "
                        f"host={host}, port={port}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

    def _upload_all(
        self,
        host: str,
        port: int,
        private_key_path: str,
        install_payload: dict[str, Any],
        lora_files: list[dict[str, Any]],
        model_plan: dict[str, Any],
    ) -> None:
        """병렬 A: 노드 설치(Modal image_install 방식) + LoRA/'upload' 모델 sftp."""
        self._set_step("upload", "running", "sftp 연결")
        ssh = self._ssh_connect(host, port, private_key_path)
        try:
            self._check_cancelled()
            sftp = ssh.open_sftp()
            ssh.exec_command("mkdir -p /root/ComfyUI/models/loras")

            # Modal이 이미지 빌드에서 하던 노드 설치를 그대로 실행.
            self._upload_local_nodes(sftp, ssh, install_payload)
            self._run_image_install(ssh, install_payload)

            total = len(lora_files) + sum(
                1
                for m in model_plan.get("models") or []
                if m["source"]["source_type"] == "upload"
            )
            done = 0
            for item in lora_files:
                remote = f"{COMFY_ROOT_REMOTE}/models/loras/{Path(item['name']).name}"
                self._sftp_put_progress(sftp, str(item["path"]), remote, "lora", item["name"])
                done += 1
                self._set_step("upload_loras", "running", f"{done}/{total}")
            for m in model_plan.get("models") or []:
                if m["source"]["source_type"] != "upload":
                    continue
                remote = f"{COMFY_ROOT_REMOTE}/models/{m['key']}"
                self._sftp_put_progress(
                    sftp, str(m.get("source_path") or ""), remote, "model", m["filename"]
                )
                done += 1
                self._set_step("upload_models", "running", f"{done}/{total}")
            self._set_step("upload", "done", f"{done}개 전송 완료")
        finally:
            ssh.close()

    def _sftp_put_progress(
        self, sftp, local: str, remote: str, label: str, name: str
    ) -> None:
        import os

        if not local or not Path(local).is_file():
            print(f"[VAST] 업로드 로컬 파일 없음({label}): {local} ({name})")
            raise FileNotFoundError(f"업로드할 로컬 파일이 없습니다: {local}")
        size = os.path.getsize(local)
        sent = 0

        def cb(transferred: int, _total: int) -> None:
            nonlocal sent
            self._check_cancelled()
            if transferred - sent > 50 * 1024 * 1024 or transferred == _total:
                sent = transferred
                self._set_step(
                    f"upload_{label}",
                    "running",
                    f"{name} {transferred / 1024**3:.1f}/{size / 1024**3:.1f}GB",
                )

        sftp.put(local, remote, callback=cb)

    def _run_remote_downloads(
        self, host: str, port: int, private_key_path: str, model_plan: dict[str, Any]
    ) -> None:
        """병렬 B: HF/Civitai/URL 모델을 인스턴스에서 직접 다운로드."""
        import shlex

        downloads = [
            m
            for m in model_plan.get("models") or []
            if m["source"]["source_type"] in {"hf", "civitai", "url"}
        ]
        if not downloads:
            self._set_step("download", "done", "원격 다운로드 대상 없음")
            return
        self._set_step("download", "running", f"{len(downloads)}개 스크립트 생성")
        lines = [
            "#!/bin/bash",
            "set -u",
            f"rm -f {shlex.quote(MODELS_DONE_FLAG)} {shlex.quote(MODELS_DONE_FLAG + '.fail')}",
            f"echo start > {shlex.quote(MODELS_DONE_FLAG + '.log')}",
        ]
        for m in downloads:
            src = m["source"]
            if src["source_type"] == "hf":
                url = (
                    f"https://huggingface.co/{src['repo_id']}"
                    f"/resolve/main/{src['hf_filename']}"
                )
            else:
                url = str(src.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                print(
                    "[VAST][DOWNLOAD][ERROR] 모델 다운로드 URL이 없습니다: "
                    f"key={m.get('key')!r}, source_type={src.get('source_type')!r}"
                )
                raise VastApiError(
                    f"모델 다운로드 URL이 없습니다: {m.get('key')} "
                    f"({src.get('source_type')})"
                )
            dest = f"{COMFY_ROOT_REMOTE}/models/{m['key']}"
            part = f"{dest}.part"
            expected = int(m.get("size_bytes") or 0)
            q_url = shlex.quote(url)
            q_dest = shlex.quote(dest)
            q_part = shlex.quote(part)
            q_dir = shlex.quote(str(Path(dest).parent).replace("\\", "/"))
            q_key = shlex.quote(str(m.get("key") or "(unknown)"))
            q_curl_error = shlex.quote(f"FAIL {m.get('key')}: curl")
            lines.append(f"mkdir -p {q_dir}")
            if expected > 0:
                lines.append(
                    f"if [ -f {q_dest} ] && "
                    f"[ \"$(stat -c%s {q_dest} 2>/dev/null || echo 0)\" = \"{expected}\" ]; "
                    f"then echo SKIP {q_key}; else"
                )
            else:
                lines.append(f"if false; then :; else")
            lines.extend(
                [
                    f"  if [ -f {q_dest} ] && [ ! -f {q_part} ]; then mv -f {q_dest} {q_part}; fi",
                    "  if ! curl --location --fail --show-error --retry 5 "
                    "--retry-delay 5 --retry-all-errors --continue-at - "
                    f"--output {q_part} {q_url}; then",
                    f"    printf '%s\\n' {q_curl_error} >> {shlex.quote(MODELS_DONE_FLAG + '.fail')}",
                    "  else",
                ]
            )
            if expected > 0:
                q_size_error = shlex.quote(
                    f"FAIL {m.get('key')}: expected_size={expected}"
                )
                lines.extend(
                    [
                        f"    actual_size=$(stat -c%s {q_part} 2>/dev/null || echo 0)",
                        f"    if [ \"$actual_size\" != \"{expected}\" ]; then",
                        f"      printf '%s actual_size=%s\\n' {q_size_error} \"$actual_size\" >> {shlex.quote(MODELS_DONE_FLAG + '.fail')}",
                        "    else",
                        f"      mv -f {q_part} {q_dest}",
                        "    fi",
                    ]
                )
            else:
                lines.append(f"    mv -f {q_part} {q_dest}")
            lines.extend(["  fi", "fi"])
        lines.extend(
            [
                f"if [ -s {shlex.quote(MODELS_DONE_FLAG + '.fail')} ]; then exit 1; fi",
                f"date > {shlex.quote(MODELS_DONE_FLAG)}",
            ]
        )
        script = "\n".join(lines) + "\n"

        ssh = self._ssh_connect(host, port, private_key_path)
        try:
            sftp = ssh.open_sftp()
            with sftp.open("/tmp/soya_download.sh", "w") as fh:
                fh.write(script)
            # setsid+리다이렉트로 SSH 채널 종료와 프로세스 생명주기를 분리한다.
            _stdin, launch_stdout, launch_stderr = ssh.exec_command(
                "chmod +x /tmp/soya_download.sh && "
                "setsid nohup /tmp/soya_download.sh "
                "> /tmp/soya_download.log 2>&1 < /dev/null &"
            )
            launch_code = launch_stdout.channel.recv_exit_status()
            if launch_code != 0:
                launch_error = launch_stderr.read().decode("utf-8", "replace")[-1000:]
                print(
                    "[VAST][DOWNLOAD][ERROR] 다운로드 프로세스 시작 실패: "
                    f"exit={launch_code}, stderr={launch_error}"
                )
                raise VastApiError(
                    f"원격 모델 다운로드 프로세스 시작 실패: exit={launch_code}"
                )
            # 시간 제한 없이 다운로드 완료를 기다린다. 비용 상한은 watchdog이 담당.
            download_paths = [
                path
                for item in downloads
                for path in (
                    f"{COMFY_ROOT_REMOTE}/models/{item['key']}",
                    f"{COMFY_ROOT_REMOTE}/models/{item['key']}.part",
                )
            ]
            quoted_paths = " ".join(shlex.quote(path) for path in download_paths)
            last_downloaded_bytes = -1
            while True:
                self._check_cancelled()
                _stdin, stdout, stderr = ssh.exec_command(
                    f"if [ -s {MODELS_DONE_FLAG}.fail ]; then "
                    f"echo __FAIL__; cat {MODELS_DONE_FLAG}.fail; "
                    f"elif [ -f {MODELS_DONE_FLAG} ]; then echo __DONE__; "
                    "elif pgrep -f '[s]oya_download.sh' >/dev/null; then "
                    f"echo __RUNNING__; total=0; for f in {quoted_paths}; do "
                    "n=$(stat -c%s \"$f\" 2>/dev/null || echo 0); total=$((total+n)); "
                    "done; echo __BYTES__:$total; "
                    "tail -c 400 /tmp/soya_download.log 2>/dev/null | tr '\\r' '\\n' | tail -n 3; "
                    "else echo __STOPPED__; tail -n 30 /tmp/soya_download.log 2>/dev/null; fi"
                )
                out = stdout.read().decode("utf-8", "replace")
                err = stderr.read().decode("utf-8", "replace").strip()
                if err:
                    print(f"[VAST][DOWNLOAD] 상태 조회 stderr: {err[-1000:]}")
                if "__FAIL__" in out:
                    failed = [ln for ln in out.splitlines() if ln.startswith("FAIL")]
                    self._set_step("download", "error", f"실패: {failed}")
                    raise VastApiError(f"원격 모델 다운로드 실패: {failed}")
                if "__DONE__" in out:
                    self._set_step("download", "done", f"{len(downloads)}개 완료")
                    return
                if "__STOPPED__" in out:
                    detail = out.split("__STOPPED__", 1)[1].strip()[-1500:]
                    print(
                        "[VAST][DOWNLOAD][ERROR] 완료 표시 없이 다운로드 프로세스 종료: "
                        f"{detail}"
                    )
                    self._set_step("download", "error", "프로세스 비정상 종료")
                    raise VastApiError(
                        "원격 모델 다운로드 프로세스가 완료 표시 없이 종료되었습니다: "
                        f"{detail[-500:]}"
                    )
                for line in out.splitlines():
                    if line.startswith("__BYTES__:"):
                        try:
                            downloaded_bytes = int(line.split(":", 1)[1])
                        except ValueError:
                            print(
                                "[VAST][DOWNLOAD][ERROR] 다운로드 바이트 해석 실패: "
                                f"line={line!r}"
                            )
                            traceback.print_exc()
                            continue
                        if downloaded_bytes != last_downloaded_bytes:
                            last_downloaded_bytes = downloaded_bytes
                            expected_total = sum(
                                int(item.get("size_bytes") or 0) for item in downloads
                            )
                            detail = f"{downloaded_bytes / 1024**3:.2f}GB"
                            if expected_total > 0:
                                detail += f"/{expected_total / 1024**3:.2f}GB"
                            self._set_step("download", "running", detail)
                visible_tail = [
                    line.strip()
                    for line in out.splitlines()
                    if line.strip() and not line.startswith("__")
                ]
                if visible_tail:
                    self._event("remote", "download: " + " | ".join(visible_tail[-3:])[-1000:])
                self._wait_sync(10)
        finally:
            ssh.close()

    async def _start_comfy_and_wait(
        self, host: str, port: int, private_key_path: str, comfy_url: str
    ) -> None:
        import time

        if not comfy_url:
            print(
                "[VAST][COMFY][ERROR] SSH 로컬 터널 URL이 없어 ComfyUI 상태를 확인할 수 "
                f"없습니다: ssh={host}:{port}"
            )
            raise VastApiError(
                "ComfyUI SSH 로컬 터널을 만들지 못했습니다. "
                "Vast SSH 상태와 로컬 터널 로그를 확인하세요."
            )
        def touch_ready() -> None:
            self._check_cancelled()
            ssh = self._ssh_connect(host, port, private_key_path)
            try:
                ssh.exec_command(f"touch {READY_FLAG}")
            finally:
                ssh.close()

        def mark_build_complete() -> None:
            self._check_cancelled()
            ssh = self._ssh_connect(host, port, private_key_path)
            try:
                _stdin, stdout, stderr = ssh.exec_command(f"touch {BUILD_COMPLETE_FLAG}")
                code = stdout.channel.recv_exit_status()
                if code != 0:
                    error_text = stderr.read().decode("utf-8", "replace")[-500:]
                    print(
                        "[VAST][COMFY][ERROR] 빌드 완료 플래그 기록 실패: "
                        f"exit={code}, stderr={error_text}"
                    )
                    raise VastApiError(
                        f"컨테이너 빌드 완료 플래그 기록 실패: exit={code}"
                    )
            finally:
                ssh.close()

        await asyncio.to_thread(touch_ready)
        self._event("remote", f"{READY_FLAG} 생성 — ComfyUI 시작 요청")
        # 시간 제한 없이 ComfyUI 기동을 기다린다. 비용 상한은 watchdog이 담당.
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            last_error = ""
            last_log_at = 0.0
            while True:
                self._check_cancelled()
                try:
                    async with session.get(f"{comfy_url}/system_stats") as resp:
                        if resp.status == 200:
                            self._event("health", f"ComfyUI health HTTP 200: {comfy_url}")
                            await asyncio.to_thread(mark_build_complete)
                            return
                        last_error = f"HTTP {resp.status}"
                except aiohttp.ClientError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                now = time.time()
                if now - last_log_at >= 60:
                    print(
                        "[VAST][COMFY] 기동 대기 중: "
                        f"url={comfy_url}, last={last_error or '(응답 없음)'}"
                    )
                    self._event(
                        "health",
                        f"ComfyUI 기동 대기: {comfy_url}, "
                        f"last={last_error or '(응답 없음)'}",
                    )
                    last_log_at = now
                await asyncio.sleep(5)

    # ── 상태/제어 ───────────────────────────────────────────

    def launch_status(self) -> dict[str, Any]:
        result = dict(self.launch)
        result["steps"] = [dict(step) for step in self.launch.get("steps") or []]
        result["events"] = [dict(event) for event in self.launch.get("events") or []]
        result["status_history"] = [
            dict(item) for item in self.launch.get("status_history") or []
        ]
        image_pull = dict(self.launch.get("image_pull") or {})
        image_pull["pending_layers"] = [
            dict(layer) for layer in image_pull.get("pending_layers") or []
        ]
        image_pull.pop("_signature", None)
        result["image_pull"] = image_pull
        preflight = dict(self.launch.get("preflight") or empty_preflight_state())
        preflight["tests"] = [
            dict(item) for item in preflight.get("tests") or [] if isinstance(item, dict)
        ]
        preflight["estimate"] = dict(preflight.get("estimate") or {})
        result["preflight"] = preflight
        now = time.time()
        contract_at = float(self.launch.get("contract_started_at_epoch") or 0.0)
        elapsed = max(0.0, now - contract_at) if contract_at else 0.0
        hourly = float(self.launch.get("hourly_price_usd") or 0.0)
        result["elapsed_seconds"] = int(elapsed)
        result["estimated_cost_usd"] = round(
            hourly * elapsed / 3600 if hourly > 0 else 0.0, 6
        )
        last_progress = float(self.launch.get("last_progress_at_epoch") or 0.0)
        stale_seconds = max(0.0, now - last_progress) if last_progress else 0.0
        result["progress_stale_seconds"] = int(stale_seconds)
        active_build = result.get("state") in {"creating", "preparing", "recovered"}
        active_build = active_build and not bool(result.get("recovered_was_ready"))
        pull_last_observed = float(
            image_pull.get("last_observed_progress_at_epoch") or contract_at or 0.0
        )
        pull_stale_seconds = (
            max(0.0, now - pull_last_observed) if pull_last_observed else 0.0
        )
        pull_active = active_build and str(result.get("instance_status") or "").lower() == "loading"
        pull_unobserved = bool(
            pull_active and pull_stale_seconds >= NO_PROGRESS_WARNING_SECONDS
        )
        image_pull["activity_stale_seconds"] = int(pull_stale_seconds)
        image_pull["activity_unobserved"] = pull_unobserved
        if image_pull.get("complete"):
            image_pull["activity_state"] = "complete"
        elif image_pull.get("error") and not image_pull.get("available"):
            image_pull["activity_state"] = "unavailable"
        elif pull_unobserved:
            image_pull["activity_state"] = "unobserved"
        elif image_pull.get("available"):
            image_pull["activity_state"] = "observed"
        else:
            image_pull["activity_state"] = "waiting"

        general_unobserved = bool(
            active_build
            and not pull_active
            and stale_seconds >= NO_PROGRESS_WARNING_SECONDS
        )
        result["activity_unobserved"] = pull_unobserved or general_unobserved
        result["activity_unobserved_reason"] = (
            f"관찰 가능한 진행 신호가 {int(pull_stale_seconds)}초 동안 없습니다. "
            "Vast가 현재 레이어의 수신 바이트를 제공하지 않아 다운로드 중인지 "
            "정지했는지는 판별할 수 없습니다."
            if pull_unobserved
            else (
                f"관찰 가능한 진행 신호가 {int(stale_seconds)}초 동안 없습니다. "
                "현재 단계의 상세 상태를 확인하세요."
                if general_unobserved
                else ""
            )
        )
        # 과거 API 필드는 호환을 위해 남기되 관측 공백을 실제 정지로 단정하지 않는다.
        result["stuck"] = False
        result["stuck_reason"] = ""
        deadlines: list[tuple[str, float]] = []
        if active_build and contract_at and hourly > 0:
            deadlines.append(
                (
                    "예상 빌드비 상한",
                    contract_at + (MAX_BUILD_COST_USD / hourly * 3600),
                )
            )
        if deadlines:
            deadline_name, deadline_at = min(deadlines, key=lambda item: item[1])
            result["auto_destroy_deadline"] = datetime.fromtimestamp(
                deadline_at, timezone.utc
            ).isoformat()
            result["auto_destroy_remaining_seconds"] = max(
                0, int(deadline_at - now)
            )
            result["auto_destroy_limit_name"] = deadline_name
        else:
            result["auto_destroy_deadline"] = ""
            result["auto_destroy_remaining_seconds"] = None
            result["auto_destroy_limit_name"] = ""
        return result

    async def instances(self) -> dict[str, Any]:
        client = self._client_or_raise()
        rows = await client.list_instances()
        return {
            "ok": True,
            "instances": [
                {
                    "id": i.get("id"),
                    "label": i.get("label"),
                    "actual_status": i.get("actual_status"),
                    "gpu_name": i.get("gpu_name"),
                    "dph_total": float(i.get("dph_total") or 0),
                    "cur_state": i.get("cur_state"),
                    "status_msg": i.get("status_msg"),
                    "start_date": i.get("start_date"),
                }
                for i in rows
            ],
        }

    async def destroy(
        self,
        instance_id: int | None = None,
        *,
        reason: str = "사용자 즉시 파괴 요청",
        automatic: bool = False,
    ) -> dict[str, Any]:
        async with self._destroy_lock:
            client = self._client_or_raise()
            target = int(instance_id or self.launch.get("instance_id") or 0)
            if not target:
                print("[VAST_DESTROY][ERROR] 파괴할 인스턴스 ID가 없습니다.")
                raise VastApiError("파괴할 인스턴스 ID가 없습니다.")
            owns_current = int(self.launch.get("instance_id") or 0) == target
            if owns_current:
                launch_id = str(self.launch.get("launch_id") or "")
                cancel_event = self._cancel_events.get(launch_id)
                if cancel_event is not None:
                    cancel_event.set()
                self.launch["state"] = "destroying"
                self.launch["destroy_reason"] = reason
                self.launch["destroy_automatic"] = bool(automatic)
                self.launch["protection_state"] = "destroying"
                self.launch["protection_reason"] = reason
                self._event(
                    "destroy",
                    f"#{target} {'자동' if automatic else '사용자'} 파괴 시작 — {reason}",
                )
                self._close_comfy_tunnel()
                launch_task = self._launch_task
                if (
                    launch_task is not None
                    and not launch_task.done()
                    and launch_task is not asyncio.current_task()
                ):
                    launch_task.cancel()

            verified = False
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    if owns_current:
                        self._event(
                            "destroy", f"#{target} Vast DELETE 요청 {attempt}/3"
                        )
                    else:
                        print(f"[VAST_DESTROY] #{target} DELETE 요청 {attempt}/3")
                    await client.destroy_instance(target)
                    last_error = None
                except Exception as exc:
                    last_error = exc
                    print(
                        "[VAST_DESTROY][ERROR] DELETE 요청 실패: "
                        f"instance={target}, attempt={attempt}/3, "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

                for verify_attempt in range(1, 5):
                    try:
                        rows = await client.list_instances()
                        still_exists = any(
                            int(row.get("id") or 0) == target for row in rows
                        )
                        if owns_current:
                            self._event(
                                "verify",
                                f"#{target} 소멸 확인 {verify_attempt}/4 — "
                                f"{'아직 존재' if still_exists else '목록에서 제거됨'}",
                            )
                        if not still_exists:
                            verified = True
                            break
                    except Exception as exc:
                        last_error = exc
                        print(
                            "[VAST_DESTROY][ERROR] 소멸 확인 조회 실패: "
                            f"instance={target}, verify={verify_attempt}/4, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()
                    if verify_attempt < 4:
                        await asyncio.sleep(1)
                if verified:
                    break
                if attempt < 3:
                    await asyncio.sleep(2)

            if not verified:
                message = (
                    f"Vast 인스턴스 #{target} 파괴를 확인하지 못했습니다. "
                    "과금이 계속될 수 있으므로 UI에서 재시도하거나 Vast 콘솔을 확인하세요."
                )
                if last_error is not None:
                    message += f" 마지막 오류: {type(last_error).__name__}: {last_error}"
                print(f"[VAST_DESTROY][CRITICAL] {message}")
                if owns_current:
                    self.launch["state"] = "destroy_failed"
                    self.launch["error"] = message
                    self.launch["protection_state"] = "critical"
                    self.launch["protection_reason"] = message
                    self._event("critical", message)
                    self._persist_guard_state()
                raise VastApiError(message)

            if owns_current and int(self.launch.get("instance_id") or 0) == target:
                self._event("destroy", f"#{target} 소멸 확인 완료 — 과금 차단")
                self.launch["state"] = "destroyed"
                self.launch["destroyed_instance_id"] = target
                self.launch["instance_id"] = None
                self.launch["comfy_base_url"] = ""
                self.launch["current_step"] = ""
                self.launch["protection_state"] = "complete"
                self.launch["protection_reason"] = reason
                self.launch["updated_at"] = self._utc_now()
                self._persist_guard_state()
                watchdog_task = self._watchdog_task
                if (
                    watchdog_task is not None
                    and not watchdog_task.done()
                    and watchdog_task is not asyncio.current_task()
                ):
                    watchdog_task.cancel()
            self._instance_status_cache.pop(target, None)
            _log(f"인스턴스 파괴 및 소멸 확인 완료: {target}")
            return {
                "ok": True,
                "destroyed": target,
                "verified": True,
                "launch": self.launch_status() if owns_current else None,
            }

    # ── custom node 설치 (Modal image_install 메커니즘 그대로) ──

    def prepare_install_payload(self) -> dict[str, Any]:
        """Modal이 이미지 빌드에 쓰는 재료를 그대로 준비한다.

        - install_manifest.json + image_install.py → 인스턴스 /opt/soya/ 로 업로드
        - 공개 노드: manifest의 CDN 아카이브(git ref + sha256 고정)에서 설치
        - 로컬 soya 노드: sftp로 /opt/soya/local_custom_nodes/ 에 전송 후
          image_install.py가 bundled_path로 복사 (Modal과 동일한 env 계약)
        """
        from modal_backend.custom_nodes import (
            deploy_custom_nodes_json,
            inventory_custom_nodes,
        )

        inventory = inventory_custom_nodes(self.project_root)
        env_json = deploy_custom_nodes_json(inventory)
        local_nodes: list[dict[str, Any]] = []
        for node in inventory.get("build_nodes") or []:
            if str(node.get("source_type")) != "local":
                continue
            local_nodes.append(
                {
                    "name": str(node.get("name") or ""),
                    "source_path": str(node.get("source_path") or ""),
                }
            )
        manifest_path = (
            self.project_root / "comfy_installer" / "resources" / "install_manifest.json"
        )
        script_path = self.project_root / "modal_backend" / "image_install.py"
        for path, label in ((manifest_path, "install_manifest"), (script_path, "image_install")):
            if not path.is_file():
                print(f"[VAST] {label} 파일 없음: {path}")
                raise FileNotFoundError(f"Vast 노드 설치에 필요한 파일이 없습니다: {path}")
        return {
            "manifest_bytes": manifest_path.read_bytes(),
            "script_bytes": script_path.read_bytes(),
            "env_json": env_json,
            "local_nodes": local_nodes,
        }

    _NODE_IGNORE_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".venv", "venv", "node_modules", "runtime"}
    _NODE_IGNORE_EXT = {".pyc", ".pyo", ".dll", ".pyd", ".whl", ".safetensors", ".ckpt", ".pt", ".bin", ".onnx", ".gguf", ".exe"}

    def _upload_local_nodes(self, sftp, ssh, payload: dict[str, Any]) -> None:
        """로컬 soya 노드를 코드만 골라 sftp 업로드한다.

        runtime/(Windows venv·아티팩트)는 Linux에서 재구성되므로 제외한다.
        """
        self._set_step("nodes", "running", f"로컬 노드 {len(payload['local_nodes'])}개")
        ssh.exec_command("mkdir -p /opt/soya/local_custom_nodes")
        total = 0
        for node in payload["local_nodes"]:
            self._check_cancelled()
            src = Path(node["source_path"])
            if not src.is_dir():
                print(f"[VAST] 로컬 노드 원본 없음(건너뜀): {node['name']} -> {src}")
                raise FileNotFoundError(f"로컬 custom node 폴더가 없습니다: {src}")
            count = 0
            for file in sorted(src.rglob("*")):
                self._check_cancelled()
                if not file.is_file():
                    continue
                rel = file.relative_to(src)
                if any(part in self._NODE_IGNORE_DIRS for part in rel.parts[:-1]):
                    continue
                if file.suffix.lower() in self._NODE_IGNORE_EXT or file.stat().st_size > 100 * 1024 * 1024:
                    continue
                remote = f"/opt/soya/local_custom_nodes/{node['name']}/{rel.as_posix()}"
                ssh.exec_command(f'mkdir -p "$(dirname "{remote}")"')
                sftp.put(str(file), remote)
                count += 1
                total += 1
            self._set_step("nodes", "running", f"{node['name']} {count}파일")
        self._set_step("nodes", "done", f"총 {total}파일 전송")

    def _run_image_install(self, ssh, payload: dict[str, Any]) -> None:
        """/opt/soya/image_install.py 실행 (Modal 이미지 빌드 단계 재현)."""
        self._set_step("nodes_install", "running", "공개 노드 CDN 설치 + pip")
        ssh.exec_command("mkdir -p /opt/soya")
        sftp = ssh.open_sftp()
        with sftp.open("/opt/soya/install_manifest.json", "w") as fh:
            fh.write(payload["manifest_bytes"].decode("utf-8"))
        with sftp.open("/opt/soya/image_install.py", "w") as fh:
            fh.write(payload["script_bytes"].decode("utf-8"))
        with sftp.open("/opt/soya/extra_nodes.json", "w") as fh:
            fh.write(payload["env_json"])
        stdin, stdout, stderr = ssh.exec_command(
            'SOYA_MODAL_IMAGE_CUSTOM_NODES="$(cat /opt/soya/extra_nodes.json)" '
            "python /opt/soya/image_install.py"
        )
        channel = stdout.channel
        stderr_chunks: list[str] = []
        if all(
            hasattr(channel, name)
            for name in (
                "exit_status_ready",
                "recv_ready",
                "recv",
                "recv_stderr_ready",
                "recv_stderr",
            )
        ):
            while not channel.exit_status_ready():
                self._check_cancelled()
                emitted = False
                while channel.recv_ready():
                    emitted = True
                    text = channel.recv(65536).decode("utf-8", "replace")
                    for line in text.replace("\r", "\n").splitlines():
                        if line.strip():
                            self._event("remote", f"image_install: {line.strip()[-1500:]}")
                while channel.recv_stderr_ready():
                    emitted = True
                    text = channel.recv_stderr(65536).decode("utf-8", "replace")
                    stderr_chunks.append(text)
                    for line in text.replace("\r", "\n").splitlines():
                        if line.strip():
                            self._event("remote-error", f"image_install: {line.strip()[-1500:]}")
                if not emitted:
                    self._wait_sync(0.25)
        code = channel.recv_exit_status()
        remaining_out = stdout.read().decode("utf-8", "replace")
        remaining_err = stderr.read().decode("utf-8", "replace")
        for line in remaining_out.replace("\r", "\n").splitlines():
            if line.strip():
                self._event("remote", f"image_install: {line.strip()[-1500:]}")
        if remaining_err:
            stderr_chunks.append(remaining_err)
        err_text = "".join(stderr_chunks)[-2000:]
        if code != 0:
            print(f"[VAST] image_install 실패: exit={code} stderr={err_text[-800:]}")
            raise VastApiError(f"custom node 설치 실패(원격): exit={code} {err_text[-400:]}")
        self._set_step("nodes_install", "done", "노드 설치 완료")

    # ── 원격 ComfyUI 실행 ───────────────────────────────────

    def _require_ready(self) -> str:
        base = str(self.launch.get("comfy_base_url") or "")
        if self.launch.get("state") != "ready" or not base:
            raise VastApiError(
                "Vast 인스턴스가 준비되지 않았습니다. 먼저 생성 마법사를 완료하세요."
            )
        return base

    async def run_workflow(self, workflow_api: dict[str, Any]) -> dict[str, Any]:
        """준비된 인스턴스의 ComfyUI /prompt 로 API 워크플로우를 실행한다."""
        base = self._require_ready()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.post(
                f"{base}/prompt", json={"prompt": workflow_api}
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    print(f"[VAST] 프롬프트 전송 실패: http={resp.status} body={text[:300]}")
                    raise VastApiError(f"ComfyUI 프롬프트 전송 실패: HTTP {resp.status} {text[:200]}")
                try:
                    prompt_id = json.loads(text).get("prompt_id")
                except ValueError as exc:
                    raise VastApiError("ComfyUI 프롬프트 응답 해석 실패") from exc
                if not prompt_id:
                    raise VastApiError(f"ComfyUI 프롬프트 ID 없음: {text[:200]}")
        return {"ok": True, "prompt_id": prompt_id, "base_url": base}
