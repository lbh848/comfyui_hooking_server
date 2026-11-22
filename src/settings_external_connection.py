"""8201 local controls and a separate, allowlisted 8202 illustration gateway."""
from __future__ import annotations

import asyncio
from pathlib import Path

import aiohttp
from aiohttp import web

from configuration import config_path, read_config, settings_lock, write_config
from external_connection import CONTROL_PREFIX, EXTERNAL_PORT, EXTERNAL_ROUTES, allowed_external_path


def register_external_connection(app, runtime, schema):
    state = {"port": EXTERNAL_PORT, "status": "stopped", "error": ""}
    sockets = set()
    connection_lock = asyncio.Lock()
    path = config_path(runtime.root)
    auto_start_task: asyncio.Task | None = None
    auto_starting = False
    auto_start_error = ""
    auto_start_enable_url = ""

    def auto_start_settings():
        return {"tailscale_auto_start": read_config(path).get("tailscale_auto_start") is True}

    def save_auto_start(enabled):
        with settings_lock(path):
            config = read_config(path)
            config["tailscale_auto_start"] = enabled
            write_config(path, config)
        return {"tailscale_auto_start": enabled}

    async def connection_request(method, connection_path):
        await runtime.start()
        async with runtime.session.request(
            method, runtime.url + CONTROL_PREFIX + "/" + connection_path,
            headers={"X-Internal-Lora": runtime.token}, timeout=aiohttp.ClientTimeout(total=180),
        ) as response:
            result = await response.json()
            if not isinstance(result, dict):
                raise ValueError("외부 연결 응답 형식이 올바르지 않습니다.")
            return result, response.status

    async def auto_start():
        nonlocal auto_starting, auto_start_error, auto_start_enable_url
        try:
            settings = await asyncio.to_thread(auto_start_settings)
            if not settings["tailscale_auto_start"]:
                return
            auto_starting = True
            async with connection_lock:
                if state["status"] != "running":
                    return
                result, status = await connection_request("POST", "tailscale/start")
                if status >= 400 or result.get("error"):
                    auto_start_enable_url = str(result.get("enable_url") or "")
                    raise RuntimeError(result.get("error") or "고정 주소를 발급하지 못했습니다.")
            auto_start_error = ""
            auto_start_enable_url = ""
            print("[EXTERNAL] 시작 시 고정 주소 자동 발급 완료")
        except (RuntimeError, OSError, ValueError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            auto_start_error = f"시작 시 고정 주소 자동 발급 실패: {exc}"
            suffix = f" ({auto_start_enable_url})" if auto_start_enable_url else ""
            print(f"[EXTERNAL] {auto_start_error}{suffix}")
        finally:
            auto_starting = False

    @web.middleware
    async def external_requests(request, handler):
        method = request.headers.get("Access-Control-Request-Method", "GET") if request.method == "OPTIONS" else request.method
        if not allowed_external_path(request.path, method):
            raise web.HTTPNotFound()
        if state["status"] != "running":
            return web.json_response({"error": "외부 연결 서버를 종료하고 있습니다."}, status=503)
        if request.method == "OPTIONS":
            response = web.Response(status=204)
        else:
            response = await handler(request)
        return response

    async def external_headers(request, response):
        # RisuAI uses cross-origin fetches, and Tailscale Serve resolves to
        # private CGNAT addresses subject to browser Private Network Access.
        # No cookies or management credentials cross this gateway, including
        # when callers forge internal headers.
        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Private-Network": "true",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        })

    async def gateway(request):
        if request.path != "/ws":
            return await runtime.request(request, external=True)
        try:
            await runtime.start()
            upstream = await runtime.session.ws_connect(
                runtime.url + "/ws", params=request.query,
                headers={"X-Internal-Lora": runtime.token}, heartbeat=30,
            )
        except (RuntimeError, OSError, aiohttp.ClientError, asyncio.TimeoutError):
            return web.json_response({"error": "삽화 서버에 연결할 수 없습니다."}, status=503)
        socket = web.WebSocketResponse(heartbeat=30)
        tasks = []
        try:
            await socket.prepare(request)
            sockets.add(socket)

            async def relay(source, destination):
                async for message in source:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        await destination.send_str(message.data)
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        await destination.send_bytes(message.data)

            tasks = [asyncio.create_task(relay(upstream, socket)), asyncio.create_task(relay(socket, upstream))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            sockets.discard(socket)
            await upstream.close()
            await socket.close()
        return socket

    async def controls(request):
        nonlocal auto_start_error, auto_start_enable_url
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            return web.json_response({"error": "삽화 화면을 새로고침해 주세요."}, status=409)
        connection_path = request.match_info["path"]
        if connection_path == "tailscale/auto-start":
            try:
                body = await request.json()
                if not isinstance(body, dict) or type(body.get("tailscale_auto_start")) is not bool:
                    raise ValueError("시작 시 자동 발급 설정은 켜기 또는 끄기로 지정하세요.")
            except (ValueError, UnicodeDecodeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)
            try:
                result = await asyncio.to_thread(save_auto_start, body["tailscale_auto_start"])
                return web.json_response(result, headers={"Cache-Control": "no-store"})
            except (OSError, ValueError):
                return web.json_response({"error": "자동 발급 설정을 저장할 수 없습니다. 다시 시도하세요."}, status=500)
        if request.method == "POST" and not connection_path.endswith("/stop") and state["status"] != "running":
            return web.json_response({"error": state["error"] or "외부 연결 서버를 시작할 수 없습니다."}, status=503)
        try:
            if request.method == "GET":
                # Keep status queries responsive while auto_start or a slow start runs.
                result, status = await connection_request(request.method, connection_path)
                result["gateway"] = dict(state)
                result.update(await asyncio.to_thread(auto_start_settings))
                tailscale = result.get("tailscale")
                if isinstance(tailscale, dict):
                    if auto_starting and tailscale.get("status") != "running":
                        tailscale["auto_starting"] = True
                    if auto_start_error and tailscale.get("status") != "running" and not tailscale.get("error"):
                        tailscale["error"] = auto_start_error
                        if auto_start_enable_url:
                            tailscale["enable_url"] = auto_start_enable_url
                return web.json_response(result, status=status, headers={"Cache-Control": "no-store"})
            async with connection_lock:
                result, status = await connection_request(request.method, connection_path)
                if connection_path.startswith("tailscale/"):
                    auto_start_error = ""
                    auto_start_enable_url = ""
                return web.json_response(result, status=status, headers={"Cache-Control": "no-store"})
        except (RuntimeError, OSError, ValueError, aiohttp.ClientError, asyncio.TimeoutError):
            return web.json_response({"error": "외부 연결 상태를 확인할 수 없습니다. 다시 시도하세요."}, status=503)

    async def stop_connections():
        state["status"] = "stopped"
        if auto_start_task is not None and not auto_start_task.done():
            auto_start_task.cancel()
            await asyncio.gather(auto_start_task, return_exceptions=True)
        async with connection_lock:
            # Never launch a worker just to stop it.
            if runtime.session and not runtime.session.closed and runtime.process and runtime.process.poll() is None:
                for kind in ("tunnel", "tailscale"):
                    try:
                        async with runtime.session.post(
                            runtime.url + CONTROL_PREFIX + f"/{kind}/stop",
                            headers={"X-Internal-Lora": runtime.token},
                            timeout=aiohttp.ClientTimeout(total=35 if kind == "tailscale" else 15),
                        ) as response:
                            await response.read()
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass

    async def lifecycle(_app):
        nonlocal auto_start_task
        public = web.Application(middlewares=[external_requests], client_max_size=200 * 1024 * 1024)
        public.on_response_prepare.append(external_headers)
        for method, path in EXTERNAL_ROUTES:
            public.router.add_route(method, path, gateway)
            if method == "GET":
                public.router.add_route("HEAD", path, gateway)
        public.router.add_route("OPTIONS", "/{path:.*}", gateway)
        runner = web.AppRunner(public, access_log=None, shutdown_timeout=5)
        await runner.setup()
        try:
            # Both tunnel providers forward only to this listener, never 8201.
            await web.TCPSite(runner, "0.0.0.0", EXTERNAL_PORT).start()
            state.update(status="running", error="")
        except OSError:
            state.update(status="error", error=f"외부 연결 포트 {EXTERNAL_PORT}을 사용할 수 없습니다. 이 포트를 사용하는 프로그램을 종료하고 공방을 다시 시작하세요.")
        if state["status"] == "running":
            auto_start_task = asyncio.create_task(auto_start())
        try:
            yield
        finally:
            for socket in list(sockets):
                await socket.close(code=1001, message=b"Server shutdown")
            await runner.cleanup()
            state["status"] = "stopped"
            # Stop only our 8202 mapping; the original project's 8189 mapping
            # and any other Tailscale services must remain intact.
            await stop_connections()

    async def guide_image(request):
        return web.FileResponse(Path(runtime.root) / "frontend" / "guide" / request.match_info["name"])

    prefix = "/api/illust/external-connection/"
    app.router.add_get(prefix + "{path:status}", controls)
    app.router.add_post(prefix + "{path:tailscale/auto-start}", controls)
    app.router.add_post(prefix + "{path:(?:tunnel|tailscale)/(?:start|stop|restart)}", controls)
    app.router.add_get("/api/illust/connection-guide/{name:img[12]\\.jpg}", guide_image)
    app.cleanup_ctx.append(lifecycle)
    return stop_connections
