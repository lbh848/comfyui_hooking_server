"""Local settings gateway to the original LoRA management and training engine."""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys

import aiohttp
import psutil
from aiohttp import web

from app_paths import data_root


API_PREFIX = "/api/setting/internal-lora"
SHARED_PATHS = {
    "/api/internal-lora/config", "/api/internal-lora/status", "/api/internal-lora/preview",
    "/api/instance_lora/auto_lora_prompt", "/api/instance_lora/auto_refine_enqueue",
    "/api/instance_lora/bot_test_setup_prompt", "/api/instance_lora/resolve_gender_tag",
    "/api/negative_presets", "/api/queue/status", "/api/queue/cancel",
    "/api/queue/cancel_all", "/api/queue/pause", "/api/queue/remove",
    "/api/llm/streams",
}


def allowed_path(path: str) -> bool:
    return path in SHARED_PATHS or bool(re.fullmatch(r"/api/llm/streams/[^/]+/control", path)) or path.startswith(("/api/lora/", "/api/bot_lora/", "/api/style_lora/")) or bool(re.fullmatch(
        r"/api/asset_mode/characters/[^/]+/(gallery|outfits/[^/]+/expressions/[^/]+/images(?:/[^/]+)?)", path,
    ))


class InternalLoraRuntime:
    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        self.process: subprocess.Popen | None = None
        self.session: aiohttp.ClientSession | None = None
        self.lock = asyncio.Lock()
        self.url = ""
        self.token = secrets.token_urlsafe(32)
        self.log_path = data_root(self.root) / "logs" / "internal-lora.log"

    async def reset_log(self, app: web.Application) -> None:
        # Discard output left by an earlier server session, including crashes.
        try:
            self.log_path.unlink(missing_ok=True)
        except PermissionError:
            # A previous worker or log viewer may still hold this file on Windows.
            # Keep its log and give this session its own file instead of failing startup.
            self.log_path = self.log_path.with_name(f"internal-lora-{os.getpid()}-{secrets.token_hex(4)}.log")

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            parent = psutil.Process(self.process.pid)
            processes = parent.children(recursive=True) + [parent]
            for process in reversed(processes):
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(processes, timeout=5)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)
        except psutil.NoSuchProcess:
            pass
        self.process = None

    async def start(self) -> None:
        async with self.lock:
            if self.session is None:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
            if self.process is not None and self.process.poll() is None:
                return
            venv_python = self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            script = str(self.root / "internal_lora_runtime.py")
            if venv_python.is_file():
                command = [str(venv_python), script]
            else:
                uv = shutil.which("uv")
                if not uv:
                    uv = next((str(path) for path in sorted((self.root / ".tools").glob("uv-*/uv.exe"), reverse=True)), None)
                command = ([uv, "run", "--script", script] if uv else [str(Path(sys.executable)), script])
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "NO_BROWSER": "1", "SOYA_INTERNAL_LORA_TOKEN": self.token,
                   "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            with self.log_path.open("ab") as log:
                self.process = subprocess.Popen(
                    [*command, "--port", str(port)],
                    cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            self.url = f"http://127.0.0.1:{port}"
            try:
                for _ in range(1200):
                    if self.process.poll() is not None:
                        raise RuntimeError("내부 LoRA 실행에 실패했습니다. data/logs/internal-lora.log를 확인하세요.")
                    try:
                        async with self.session.get(self.url + "/api/internal-lora/status", headers={"X-Internal-Lora": self.token}, timeout=aiohttp.ClientTimeout(total=1)) as response:
                            if response.status == 200:
                                return
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass
                    await asyncio.sleep(.25)
                raise RuntimeError("내부 LoRA 준비 시간이 초과되었습니다. 다시 시도하세요.")
            except BaseException:
                await asyncio.to_thread(self.stop)
                raise

    async def request(self, request: web.Request, *, illustration: bool = False, external: bool = False, asset: bool = False) -> web.StreamResponse:
        path = request.path if external else "/api/" + request.match_info["path"]
        from settings_illustration import allowed_illustration_path
        from settings_assets import allowed_asset_path
        from external_connection import allowed_external_path
        allowed = (allowed_external_path(path, request.method) if external else
                   allowed_asset_path(path, request.method) if asset else
                   allowed_illustration_path(path, request.method) if illustration else allowed_path(path))
        if not allowed:
            raise web.HTTPNotFound()
        if not asset and path.startswith("/api/asset_mode/") and request.method != "GET":
            raise web.HTTPMethodNotAllowed(request.method, ["GET"])
        if (request.method == "GET" and path == "/api/queue/status"
                and request.query.get("monitor") == "1"
                and (self.process is None or self.process.poll() is not None)):
            # Sidebar monitoring must not launch an idle worker just to count jobs.
            return web.json_response(
                {"items": [], "pending_count": 0, "processing": False},
                headers={"Cache-Control": "no-store"},
            )
        try:
            await self.start()
            assert self.session is not None
            headers = {"X-Internal-Lora": self.token}
            if illustration and request.headers.get("X-Bot-Distribution-Token"):
                headers["X-Bot-Distribution-Token"] = request.headers["X-Bot-Distribution-Token"]
            if request.content_type:
                headers["Content-Type"] = request.headers.get("Content-Type", "application/json")
            timeout = (aiohttp.ClientTimeout(total=None, sock_connect=30)
                       if external or path in {"/api/bot_mode/character_cards/suggest_metadata",
                                               "/api/bot_mode/auto_batch", "/api/bot_lora/allocate",
                                               "/api/bot_mode/character_lore_auto_link",
                                               "/api/bot_mode/lore_npcs/analyze", "/api/bot_mode/lore_npcs/create",
                                               "/api/bot_mode/lore_npcs/generate", "/api/bot_mode/lore_npcs/register",
                                               "/api/bot_mode/archive",
                                               "/api/asset_mode/qwen_edit/translate"} else self.session.timeout)
            async with self.session.request(request.method, self.url + path, params=request.query,
                                            headers=headers, data=request.content if request.can_read_body else None,
                                            timeout=timeout, allow_redirects=False) as upstream:
                cacheable_media = (illustration and request.method == "GET" and (
                    path.startswith("/api/backup_poster/") and request.query.get("thumbnail") == "1"
                    or path.startswith("/api/backup_image/") and bool(request.query.get("v"))
                    or path == "/api/illustration_flow" and request.query.get("image") == "1"
                    and request.query.get("thumbnail") == "1" and bool(request.query.get("v"))
                ))
                response = web.StreamResponse(status=upstream.status, headers={
                    "Content-Type": upstream.headers.get("Content-Type", "application/json"),
                    **({"Content-Disposition": upstream.headers["Content-Disposition"]} if "Content-Disposition" in upstream.headers else {}),
                    "Cache-Control": upstream.headers.get("Cache-Control", "no-store") if cacheable_media else "no-store",
                })
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(256 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (RuntimeError, OSError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return web.json_response({"success": False, "error": "삽화 서버에 연결할 수 없습니다." if external else str(exc) or "작업 서버에 연결할 수 없습니다."}, status=503)

    async def close(self, app: web.Application) -> None:
        await asyncio.to_thread(self.stop)
        if self.session is not None:
            await self.session.close()
        try:
            self.log_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # A log viewer must not prevent server shutdown.

    async def llm_streams(self, stream_id: str = "", action: str = "", *, monitor: bool = False) -> dict:
        # Monitoring must not start a worker or restart one with stale stream IDs.
        if self.process is None or self.process.poll() is not None:
            if stream_id:
                raise ValueError("활성 스트림을 찾을 수 없습니다.")
            return {"streams": []}
        if self.session is None:
            raise RuntimeError("라이브 연결을 준비 중입니다.")
        from urllib.parse import quote
        path = "/api/llm/streams"
        if stream_id:
            path += f"/{quote(stream_id, safe='')}/control"
        elif monitor:
            path += "?monitor=1"
        async with self.session.request(
            "POST" if stream_id else "GET", self.url + path,
            headers={"X-Internal-Lora": self.token},
            **({"json": {"action": action}} if stream_id else {}),
            timeout=aiohttp.ClientTimeout(total=5),
        ) as response:
            result = await response.json()
            if response.status >= 400 or result.get("success") is False:
                raise ValueError(result.get("error") or "라이브 요청을 처리할 수 없습니다.")
            return result


def register_internal_lora(app: web.Application, project_root: Path, schema: str) -> InternalLoraRuntime:
    from lora_training_presets import register_training_presets

    register_training_presets(app, project_root, schema)
    runtime = InternalLoraRuntime(project_root)

    async def gateway(request: web.Request) -> web.StreamResponse:
        # Images cannot supply fetch headers; the loopback Host/Origin guard still applies.
        image = request.method == "GET" and any(part in request.path for part in (
            "/image/", "/images/", "/training_image/", "/test_image/", "/char_test_image/", "/char_image/", "/entry_image/", "/trained/preview/",
        ))
        if not image:
            if request.headers.get("X-Comfy-Settings") != "1":
                raise web.HTTPForbidden()
            if request.headers.get("X-Comfy-Settings-Schema") != schema:
                return web.json_response({"error": "설정 화면을 새로고침해 주세요."}, status=409)
        return await runtime.request(request)

    app.router.add_route("GET", API_PREFIX + "/{path:.+}", gateway)
    app.router.add_route("POST", API_PREFIX + "/{path:.+}", gateway)
    app.on_startup.append(runtime.reset_log)
    app.on_cleanup.append(runtime.close)
    return runtime
