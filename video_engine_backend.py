"""Local MiniMax H3 video-engine client and 4080 ownership transitions."""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from aiohttp import web

from comfy_allocation import VIDEO_ENGINE_COMFY_TARGET
from video_engine_runtime import (
    VideoEngineRuntimeError,
    VideoEngineRuntimeManager,
    VideoEngineRuntimeValidationError,
)


VIDEO_ENGINE_DEFAULT_PORT = 8093
VIDEO_ENGINE_TARGET = VIDEO_ENGINE_COMFY_TARGET
VIDEO_ENGINE_MODES = frozenset({"i2v", "first_last", "ref2v"})
_ACTIVE_ENGINE_STATES = frozenset(
    {"starting", "warming", "busy", "rewarming", "cooling"}
)


class VideoEngineError(RuntimeError):
    """The video engine could not complete a requested operation."""


class VideoEngineUnavailableError(VideoEngineError):
    """The configured local video-engine port is not reachable."""


def normalize_video_engine_port(value: Any) -> int:
    if isinstance(value, bool):
        print(f"[VIDEO_ENGINE] 포트 검증 실패: bool은 허용되지 않음 value={value!r}")
        raise ValueError("영상 전용 엔진 포트는 1~65535 사이 정수여야 합니다.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        print(
            "[VIDEO_ENGINE] 포트 검증 실패: "
            f"value={value!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise ValueError("영상 전용 엔진 포트는 1~65535 사이 정수여야 합니다.") from exc
    if isinstance(value, float) and not value.is_integer():
        print(f"[VIDEO_ENGINE] 포트 검증 실패: 정수가 아닌 실수 value={value!r}")
        raise ValueError("영상 전용 엔진 포트는 1~65535 사이 정수여야 합니다.")
    if isinstance(value, str) and value.strip() != str(parsed):
        print(f"[VIDEO_ENGINE] 포트 검증 실패: 정수 문자열 아님 value={value!r}")
        raise ValueError("영상 전용 엔진 포트는 1~65535 사이 정수여야 합니다.")
    if not 1 <= parsed <= 65535:
        print(f"[VIDEO_ENGINE] 포트 검증 실패: 범위 벗어남 value={parsed}")
        raise ValueError("영상 전용 엔진 포트는 1~65535 사이 정수여야 합니다.")
    return parsed


class VideoEngineService:
    def __init__(
        self,
        *,
        get_config: Callable[[], dict[str, Any]],
        get_comfy_ports: Callable[[], list[tuple[int, int]]],
    ) -> None:
        self.get_config = get_config
        self.get_comfy_ports = get_comfy_ports
        self._last_connection_error = ""

    def port(self) -> int:
        config = self.get_config()
        return normalize_video_engine_port(
            config.get("video_engine_port", VIDEO_ENGINE_DEFAULT_PORT)
        )

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port()}"

    def _report_connection_error(self, operation: str, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if message == self._last_connection_error:
            print(
                "[VIDEO_ENGINE] 연결 실패 반복: "
                f"operation={operation}, url={self.base_url()}, error={message}"
            )
            return
        self._last_connection_error = message
        print(
            "[VIDEO_ENGINE] 연결 실패: "
            f"operation={operation}, url={self.base_url()}, error={message}"
        )
        traceback.print_exc()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout_seconds: float = 15.0,
    ) -> dict[str, Any]:
        url = f"{self.base_url()}{path}"
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, url, json=body) as response:
                    text = await response.text()
                    try:
                        payload = json.loads(text) if text else {}
                    except json.JSONDecodeError as exc:
                        print(
                            "[VIDEO_ENGINE] JSON 응답 파싱 실패: "
                            f"method={method}, url={url}, status={response.status}, "
                            f"response={text[:500]!r}, error={exc}"
                        )
                        traceback.print_exc()
                        raise VideoEngineError(
                            f"영상 전용 엔진이 JSON이 아닌 응답을 반환했습니다. (HTTP {response.status})"
                        ) from exc
                    if not 200 <= response.status < 300:
                        detail = payload.get("detail") if isinstance(payload, dict) else None
                        print(
                            "[VIDEO_ENGINE] API 요청 실패: "
                            f"method={method}, url={url}, status={response.status}, "
                            f"response={text[:1000]!r}"
                        )
                        raise VideoEngineError(
                            str(detail or f"영상 전용 엔진 HTTP {response.status}")
                        )
                    if not isinstance(payload, dict):
                        print(
                            "[VIDEO_ENGINE] 객체가 아닌 API 응답: "
                            f"method={method}, url={url}, payload={payload!r}"
                        )
                        raise VideoEngineError("영상 전용 엔진 응답 형식이 올바르지 않습니다.")
                    self._last_connection_error = ""
                    return payload
        except VideoEngineError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            self._report_connection_error(f"{method} {path}", exc)
            raise VideoEngineUnavailableError(
                f"영상 전용 엔진(127.0.0.1:{self.port()})에 연결할 수 없습니다."
            ) from exc

    async def _request_bytes(self, path: str, *, timeout_seconds: float = 120.0) -> bytes:
        url = f"{self.base_url()}{path}"
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    payload = await response.read()
                    if not 200 <= response.status < 300:
                        print(
                            "[VIDEO_ENGINE] 파일 다운로드 실패: "
                            f"url={url}, status={response.status}, "
                            f"response={payload[:500]!r}"
                        )
                        raise VideoEngineError(
                            f"영상 전용 엔진 MP4 다운로드 실패 (HTTP {response.status})"
                        )
                    if not payload:
                        print(f"[VIDEO_ENGINE] 빈 MP4 다운로드: url={url}")
                        raise VideoEngineError("영상 전용 엔진 MP4가 비어 있습니다.")
                    self._last_connection_error = ""
                    return payload
        except VideoEngineError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            self._report_connection_error(f"GET {path}", exc)
            raise VideoEngineUnavailableError(
                f"영상 전용 엔진(127.0.0.1:{self.port()}) MP4를 받을 수 없습니다."
            ) from exc

    async def status(self) -> dict[str, Any]:
        payload = await self._request_json("GET", "/api/status", timeout_seconds=5.0)
        return {"reachable": True, "port": self.port(), **payload}

    @staticmethod
    def _comfy_queue_busy(payload: Any) -> bool:
        if not isinstance(payload, dict):
            print(
                "[VIDEO_ENGINE] Comfy /queue 응답 형식 오류: "
                f"type={type(payload).__name__}, value={payload!r}"
            )
            raise VideoEngineError("ComfyUI /queue 응답 형식이 올바르지 않습니다.")
        running = payload.get("queue_running")
        pending = payload.get("queue_pending")
        if not isinstance(running, list) or not isinstance(pending, list):
            print(
                "[VIDEO_ENGINE] Comfy /queue 실행·대기 목록 누락: "
                f"payload={payload!r}"
            )
            raise VideoEngineError("ComfyUI /queue 응답에 실행·대기 목록이 없습니다.")
        return bool(running or pending)

    async def _free_one_comfy(self, instance_id: int, port: int) -> dict[str, Any]:
        queue_url = f"http://127.0.0.1:{port}/queue"
        timeout = aiohttp.ClientTimeout(total=5.0)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(queue_url) as response:
                    if response.status != 200:
                        text = await response.text()
                        print(
                            "[VIDEO_ENGINE] Comfy 큐 조회 실패: "
                            f"instance={instance_id}, port={port}, status={response.status}, "
                            f"response={text[:500]!r}"
                        )
                        raise VideoEngineError(
                            f"Comfy #{instance_id} 큐 상태를 확인할 수 없습니다."
                        )
                    queue_payload = await response.json()
        except VideoEngineError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            print(
                "[VIDEO_ENGINE] 실행되지 않은 Comfy 정리 생략: "
                f"instance={instance_id}, port={port}, "
                f"reason={type(exc).__name__}: {exc}"
            )
            return {"instance_id": instance_id, "port": port, "running": False}

        if self._comfy_queue_busy(queue_payload):
            print(
                "[VIDEO_ENGINE] Comfy 내부 큐가 비지 않아 영상 전환 대기: "
                f"instance={instance_id}, port={port}"
            )
            deadline = asyncio.get_running_loop().time() + 1800.0
            while self._comfy_queue_busy(queue_payload):
                if asyncio.get_running_loop().time() >= deadline:
                    print(
                        "[VIDEO_ENGINE] Comfy 내부 큐 대기 시간 초과: "
                        f"instance={instance_id}, port={port}, queue={queue_payload!r}"
                    )
                    raise VideoEngineError(
                        f"Comfy #{instance_id} 작업이 끝나지 않아 영상 엔진으로 전환할 수 없습니다."
                    )
                await asyncio.sleep(0.5)
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(queue_url) as response:
                            if response.status != 200:
                                text = await response.text()
                                print(
                                    "[VIDEO_ENGINE] Comfy 큐 재조회 실패: "
                                    f"instance={instance_id}, port={port}, "
                                    f"status={response.status}, response={text[:500]!r}"
                                )
                                raise VideoEngineError(
                                    f"Comfy #{instance_id} 큐 상태 재확인에 실패했습니다."
                                )
                            queue_payload = await response.json()
                except VideoEngineError:
                    raise
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                    print(
                        "[VIDEO_ENGINE] Comfy 큐 재조회 연결 실패: "
                        f"instance={instance_id}, port={port}, "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    raise VideoEngineError(
                        f"Comfy #{instance_id} 큐 상태 재확인에 실패했습니다."
                    ) from exc

        free_url = f"http://127.0.0.1:{port}/free"
        body = {"unload_models": True, "free_memory": True}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(free_url, json=body) as response:
                    text = await response.text()
                    if not 200 <= response.status < 300:
                        print(
                            "[VIDEO_ENGINE] Comfy VRAM/RAM 정리 실패: "
                            f"instance={instance_id}, port={port}, status={response.status}, "
                            f"response={text[:500]!r}"
                        )
                        raise VideoEngineError(
                            f"Comfy #{instance_id} VRAM/RAM 정리에 실패했습니다."
                        )
        except VideoEngineError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            print(
                "[VIDEO_ENGINE] Comfy VRAM/RAM 정리 연결 실패: "
                f"instance={instance_id}, port={port}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise VideoEngineError(
                f"Comfy #{instance_id} VRAM/RAM 정리 요청에 실패했습니다."
            ) from exc
        print(
            "[VIDEO_ENGINE] Comfy VRAM/RAM 정리 요청 완료: "
            f"instance={instance_id}, port={port}"
        )
        return {"instance_id": instance_id, "port": port, "running": True}

    async def free_comfy_memory(self) -> list[dict[str, Any]]:
        unique: dict[int, int] = {}
        for instance_id, port in self.get_comfy_ports():
            unique.setdefault(int(port), int(instance_id))
        if not unique:
            print("[VIDEO_ENGINE] Comfy VRAM/RAM 정리 생략: 설정된 포트 없음")
            return []
        results = []
        for port, instance_id in unique.items():
            results.append(await self._free_one_comfy(instance_id, port))
        if any(item.get("running") for item in results):
            # /free는 Comfy 실행 루프에 플래그를 전달한다. 플래그 소비와 gc가
            # 시작될 시간을 준 뒤 외부 엔진의 자체 12 GiB headroom 검증으로 확인한다.
            await asyncio.sleep(0.5)
        return results

    async def _set_warmup(self, enabled: bool, *, mode: str) -> dict[str, Any]:
        normalized_mode = "ref2v" if mode == "ref2v" else "i2v"
        deadline = asyncio.get_running_loop().time() + 600.0
        requested = False
        while True:
            status = await self.status()
            state = str(status.get("status") or "")
            residency = status.get("residency") if isinstance(status.get("residency"), dict) else {}
            active_mode = str(residency.get("model_mode") or "")
            if enabled and state == "ready" and active_mode == normalized_mode:
                return status
            if not enabled and state == "cold":
                return status
            if state == "error" and requested:
                print(
                    "[VIDEO_ENGINE] 외부 엔진 자원 전환 실패 상태: "
                    f"enabled={enabled}, mode={normalized_mode}, status={status!r}"
                )
                raise VideoEngineError(
                    str(status.get("error") or "영상 전용 엔진 자원 전환에 실패했습니다.")
                )
            if asyncio.get_running_loop().time() >= deadline:
                print(
                    "[VIDEO_ENGINE] 자원 전환 시간 초과: "
                    f"enabled={enabled}, mode={normalized_mode}, status={status!r}"
                )
                raise VideoEngineError("영상 전용 엔진 자원 전환 시간이 초과되었습니다.")
            if state in _ACTIVE_ENGINE_STATES:
                await asyncio.sleep(0.5)
                continue
            if requested:
                await asyncio.sleep(0.5)
                continue
            try:
                await self._request_json(
                    "PUT",
                    "/api/warmup",
                    body={"enabled": enabled, "mode": normalized_mode},
                    timeout_seconds=15.0,
                )
                requested = True
            except VideoEngineError as exc:
                print(
                    "[VIDEO_ENGINE] 자원 전환 요청 실패: "
                    f"enabled={enabled}, mode={normalized_mode}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise

    async def prepare_video(self, *, mode: str) -> dict[str, Any]:
        if mode not in VIDEO_ENGINE_MODES:
            print(f"[VIDEO_ENGINE] 영상 모드 검증 실패: mode={mode!r}")
            raise VideoEngineError(f"지원하지 않는 영상 전용 엔진 모드입니다: {mode}")
        await self.free_comfy_memory()
        try:
            return await self._set_warmup(True, mode=mode)
        except VideoEngineError:
            # Comfy /free 플래그 소비가 늦은 경우 headroom 오류가 먼저 올 수 있다.
            # 한 번 더 정리한 뒤 동일 전환을 재시도한다.
            print(
                "[VIDEO_ENGINE] 첫 WARMUP 실패 후 Comfy 정리 재확인: "
                f"mode={mode}"
            )
            await self.free_comfy_memory()
            return await self._set_warmup(True, mode=mode)

    async def ensure_cold_for_comfy(self) -> dict[str, Any]:
        try:
            return await self._set_warmup(False, mode="i2v")
        except VideoEngineUnavailableError as exc:
            print(
                "[VIDEO_ENGINE] 외부 엔진 미실행으로 내리기 생략: "
                f"port={self.port()}, reason={exc}"
            )
            return {
                "reachable": False,
                "port": self.port(),
                "status": "offline",
                "released": True,
            }

    async def set_warmup(self, enabled: bool, *, mode: str = "i2v") -> dict[str, Any]:
        if enabled:
            return await self.prepare_video(mode=mode)
        return await self.ensure_cold_for_comfy()

    async def generate_video(
        self,
        payload: dict[str, Any],
        *,
        progress_callback: Callable[[int, int], Awaitable[None] | None] | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        mode = str(payload.get("mode") or "i2v")
        await self.prepare_video(mode=mode)
        created = await self._request_json(
            "POST",
            "/api/generate",
            body=payload,
            timeout_seconds=30.0,
        )
        job_id = str(created.get("job_id") or "")
        if not job_id:
            print(f"[VIDEO_ENGINE] 작업 등록 응답에 job_id 없음: response={created!r}")
            raise VideoEngineError("영상 전용 엔진 작업 ID가 없습니다.")
        print(f"[VIDEO_ENGINE] 생성 작업 등록: job={job_id}, mode={mode}")
        deadline = asyncio.get_running_loop().time() + 1800.0
        last_progress = -1
        while True:
            job = await self._request_json(
                "GET",
                f"/api/jobs/{job_id}",
                timeout_seconds=10.0,
            )
            try:
                progress = max(0, min(1000, round(float(job.get("progress") or 0) * 1000)))
            except (TypeError, ValueError, OverflowError) as exc:
                print(
                    "[VIDEO_ENGINE] 작업 진행률 파싱 실패: "
                    f"job={job_id}, value={job.get('progress')!r}, error={exc}"
                )
                traceback.print_exc()
                progress = last_progress if last_progress >= 0 else 0
            if progress_callback is not None and progress != last_progress:
                callback_result = progress_callback(progress, 1000)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
                last_progress = progress
            job_status = str(job.get("status") or "")
            if job_status == "completed":
                break
            if job_status in {"failed", "interrupted", "cancelled"}:
                print(
                    "[VIDEO_ENGINE] 생성 작업 실패: "
                    f"job={job_id}, status={job_status}, error={job.get('error')!r}, "
                    f"traceback={job.get('traceback')!r}"
                )
                raise VideoEngineError(
                    str(job.get("error") or f"영상 전용 엔진 작업 {job_status}")
                )
            if asyncio.get_running_loop().time() >= deadline:
                print(f"[VIDEO_ENGINE] 생성 작업 시간 초과: job={job_id}, state={job!r}")
                raise VideoEngineError("영상 전용 엔진 생성 시간이 초과되었습니다.")
            await asyncio.sleep(0.5)

        video_bytes = await self._request_bytes(f"/api/jobs/{job_id}/file")
        if progress_callback is not None and last_progress != 1000:
            callback_result = progress_callback(1000, 1000)
            if asyncio.iscoroutine(callback_result):
                await callback_result
        descriptor = {
            "execution_source": VIDEO_ENGINE_TARGET,
            "job_id": job_id,
            "port": self.port(),
            "result": job.get("result") if isinstance(job.get("result"), dict) else {},
        }
        print(
            "[VIDEO_ENGINE] MP4 수신 완료: "
            f"job={job_id}, mode={mode}, bytes={len(video_bytes):,}"
        )
        return video_bytes, descriptor

    async def delete_video_output(self, descriptor: dict[str, Any]) -> bool:
        job_id = str(descriptor.get("job_id") or "")
        if not job_id:
            print(f"[VIDEO_ENGINE] MP4 정리 실패: job_id 없음 descriptor={descriptor!r}")
            return False
        try:
            result = await self._request_json(
                "DELETE",
                f"/api/jobs/{job_id}/file",
                timeout_seconds=15.0,
            )
            return bool(result.get("deleted") or result.get("job_id") == job_id)
        except VideoEngineError as exc:
            print(
                "[VIDEO_ENGINE] MP4 정리 실패: "
                f"job={job_id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return False


def register_video_engine_routes(
    app: web.Application,
    *,
    get_config: Callable[[], dict[str, Any]],
    get_comfy_ports: Callable[[], list[tuple[int, int]]],
    runtime_manager: VideoEngineRuntimeManager | None = None,
    authorize: Callable[[web.Request], bool] | None = None,
) -> VideoEngineService:
    service = VideoEngineService(
        get_config=get_config,
        get_comfy_ports=get_comfy_ports,
    )

    def require_authorized(request: web.Request) -> web.Response | None:
        if authorize is None:
            return None
        try:
            allowed = bool(authorize(request))
        except Exception as exc:
            print(
                "[VIDEO_ENGINE_API] 인증 확인 예외: "
                f"method={request.method}, path={request.path}, remote={request.remote}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response(
                {"ok": False, "error": "실행 관리 인증을 확인하지 못했습니다."},
                status=500,
            )
        if allowed:
            return None
        print(
            "[VIDEO_ENGINE_API] 인증되지 않은 요청 거부: "
            f"method={request.method}, path={request.path}, remote={request.remote}"
        )
        return web.json_response(
            {"ok": False, "error": "대시보드 로그인이 필요합니다."}, status=401
        )

    def runtime_status(*, after: Any, reachable: bool) -> dict[str, Any] | None:
        if runtime_manager is None:
            return None
        payload = runtime_manager.status(after=after)
        payload["external"] = bool(reachable and not payload.get("running"))
        return payload

    async def handle_status(request: web.Request) -> web.Response:
        denied = require_authorized(request)
        if denied is not None:
            return denied
        after = request.query.get("after", "0")
        try:
            payload = await service.status()
            runtime = runtime_status(after=after, reachable=True)
            if runtime is not None:
                payload["runtime"] = runtime
            return web.json_response(payload)
        except VideoEngineUnavailableError as exc:
            payload = {
                "reachable": False,
                "port": service.port(),
                "status": "offline",
                "error": str(exc),
            }
            runtime = runtime_status(after=after, reachable=False)
            if runtime is not None:
                payload["runtime"] = runtime
            return web.json_response(payload)
        except VideoEngineRuntimeValidationError as exc:
            print(
                "[VIDEO_ENGINE_API] 상태 로그 위치 검증 실패: "
                f"after={after!r}, error={exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(
                "[VIDEO_ENGINE_API] 상태 조회 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            payload = {
                "reachable": False,
                "port": service.port(),
                "status": "error",
                "error": str(exc),
            }
            try:
                runtime = runtime_status(after=after, reachable=False)
                if runtime is not None:
                    payload["runtime"] = runtime
            except Exception as runtime_exc:
                print(
                    "[VIDEO_ENGINE_API] 상태 오류 응답용 런타임 조회 실패: "
                    f"error={type(runtime_exc).__name__}: {runtime_exc}"
                )
                traceback.print_exc()
            return web.json_response(payload, status=502)

    async def handle_resources(request: web.Request) -> web.Response:
        denied = require_authorized(request)
        if denied is not None:
            return denied
        body: Any = None
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("영상 전용 엔진 자원 요청은 객체여야 합니다.")
            enabled = body.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError("영상 전용 엔진 enabled 값은 true/false여야 합니다.")
            mode = str(body.get("mode") or "i2v").strip().lower()
            if mode not in VIDEO_ENGINE_MODES:
                raise ValueError("영상 전용 엔진 mode 값이 올바르지 않습니다.")
            return web.json_response(await service.set_warmup(enabled, mode=mode))
        except ValueError as exc:
            print(f"[VIDEO_ENGINE_API] 자원 요청 검증 실패: body={body!r}, error={exc}")
            traceback.print_exc()
            return web.json_response({"error": str(exc)}, status=400)
        except VideoEngineError as exc:
            print(
                "[VIDEO_ENGINE_API] 자원 전환 실패: "
                f"body={body!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"error": str(exc)}, status=409)
        except Exception as exc:
            print(
                "[VIDEO_ENGINE_API] 자원 전환 예외: "
                f"body={body!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_start(request: web.Request) -> web.Response:
        denied = require_authorized(request)
        if denied is not None:
            return denied
        if runtime_manager is None:
            print("[VIDEO_ENGINE_API] 실행 요청 실패: 런타임 관리자가 등록되지 않음")
            return web.json_response(
                {"ok": False, "error": "영상 전용 엔진 런타임 관리자가 없습니다."},
                status=501,
            )
        try:
            config = get_config()
            payload = await asyncio.to_thread(
                runtime_manager.start,
                project_path=config.get("video_engine_project_path", ""),
                port=config.get("video_engine_port", VIDEO_ENGINE_DEFAULT_PORT),
            )
            return web.json_response({"ok": True, "runtime": payload})
        except VideoEngineRuntimeValidationError as exc:
            print(f"[VIDEO_ENGINE_API] 실행 설정 검증 실패: error={exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except VideoEngineRuntimeError as exc:
            print(f"[VIDEO_ENGINE_API] 실행 요청 실패: error={exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=409)
        except Exception as exc:
            print(
                "[VIDEO_ENGINE_API] 실행 처리 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_stop(request: web.Request) -> web.Response:
        denied = require_authorized(request)
        if denied is not None:
            return denied
        if runtime_manager is None:
            print("[VIDEO_ENGINE_API] 종료 요청 실패: 런타임 관리자가 등록되지 않음")
            return web.json_response(
                {"ok": False, "error": "영상 전용 엔진 런타임 관리자가 없습니다."},
                status=501,
            )
        try:
            try:
                engine = await service.status()
            except VideoEngineUnavailableError as exc:
                print(
                    "[VIDEO_ENGINE_API] 종료 전 데몬 상태 확인 생략: "
                    f"프로세스 API 미응답 reason={exc}"
                )
                engine = None
            if isinstance(engine, dict):
                engine_state = str(engine.get("status") or "")
                queue_size = int(engine.get("queue_size") or 0)
                if engine_state in _ACTIVE_ENGINE_STATES or queue_size > 0:
                    message = (
                        "영상 생성 또는 자원 전환이 진행 중이어서 데몬을 종료할 수 "
                        "없습니다. 작업 완료 후 다시 시도하세요."
                    )
                    print(
                        "[VIDEO_ENGINE_API] 실행 중 종료 거부: "
                        f"status={engine_state}, queue_size={queue_size}"
                    )
                    raise VideoEngineRuntimeError(message)
            payload = await asyncio.to_thread(runtime_manager.stop)
            return web.json_response({"ok": True, "runtime": payload})
        except VideoEngineRuntimeError as exc:
            print(f"[VIDEO_ENGINE_API] 종료 요청 실패: error={exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=409)
        except VideoEngineError as exc:
            print(
                "[VIDEO_ENGINE_API] 종료 전 상태 확인 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response(
                {
                    "ok": False,
                    "error": "영상 전용 엔진 상태를 확인하지 못해 안전 종료를 중단했습니다.",
                },
                status=502,
            )
        except Exception as exc:
            print(
                "[VIDEO_ENGINE_API] 종료 처리 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def cleanup_runtime(_app: web.Application) -> None:
        if runtime_manager is None:
            return
        try:
            await asyncio.to_thread(runtime_manager.stop_if_running)
        except Exception as exc:
            print(
                "[VIDEO_ENGINE_RUNTIME] 서버 종료 정리 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    app.router.add_get("/api/video-engine/status", handle_status)
    app.router.add_put("/api/video-engine/resources", handle_resources)
    app.router.add_post("/api/video-engine/start", handle_start)
    app.router.add_post("/api/video-engine/stop", handle_stop)
    app.on_cleanup.append(cleanup_runtime)
    return service
