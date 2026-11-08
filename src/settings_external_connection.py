"""8201 local controls and a separate, allowlisted 8202 illustration gateway."""
from __future__ import annotations

import asyncio
from pathlib import Path

import aiohttp
from aiohttp import web

from external_connection import CONTROL_PREFIX, EXTERNAL_PORT, EXTERNAL_ROUTES, allowed_external_path


def register_external_connection(app, runtime, schema):
    state = {"port": EXTERNAL_PORT, "status": "stopped", "error": ""}
    sockets = set()
    connection_lock = asyncio.Lock()

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
        # RisuAI uses cross-origin fetches. No cookies or management credentials
        # cross this gateway, including when callers forge internal headers.
        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
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
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            return web.json_response({"error": "삽화 화면을 새로고침해 주세요."}, status=409)
        path = CONTROL_PREFIX + "/" + request.match_info["path"]
        if request.method == "POST" and not path.endswith("/stop") and state["status"] != "running":
            return web.json_response({"error": state["error"] or "외부 연결 서버를 시작할 수 없습니다."}, status=503)
        try:
            async with connection_lock:
                await runtime.start()
                async with runtime.session.request(
                    request.method, runtime.url + path, headers={"X-Internal-Lora": runtime.token},
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as response:
                    result = await response.json()
                    if not isinstance(result, dict):
                        raise ValueError("외부 연결 응답 형식이 올바르지 않습니다.")
                    if request.method == "GET":
                        result["gateway"] = dict(state)
                    return web.json_response(result, status=response.status, headers={"Cache-Control": "no-store"})
        except (RuntimeError, OSError, ValueError, aiohttp.ClientError, asyncio.TimeoutError):
            return web.json_response({"error": "외부 연결 상태를 확인할 수 없습니다. 다시 시도하세요."}, status=503)

    async def stop_connections():
        state["status"] = "stopped"
        async with connection_lock:
            # Never launch a worker just to stop it.
            if runtime.session and not runtime.session.closed and runtime.process and runtime.process.poll() is None:
                for kind in ("tunnel", "tailscale"):
                    try:
                        async with runtime.session.post(
                            runtime.url + CONTROL_PREFIX + f"/{kind}/stop",
                            headers={"X-Internal-Lora": runtime.token}, timeout=aiohttp.ClientTimeout(total=15),
                        ) as response:
                            await response.read()
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass

    async def lifecycle(_app):
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
    app.router.add_post(prefix + "{path:(?:tunnel|tailscale)/(?:start|stop|restart)}", controls)
    app.router.add_get("/api/illust/connection-guide/{name:img[12]\\.jpg}", guide_image)
    app.cleanup_ctx.append(lifecycle)
    return stop_connections
