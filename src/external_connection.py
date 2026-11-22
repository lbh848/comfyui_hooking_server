"""External illustration protocol and local connection controls.

Only the routes listed here can cross the public gateway. Management APIs,
credentials, static files and the dashboard are deliberately absent.
"""
from __future__ import annotations

import asyncio
import json
import re

from aiohttp import web

EXTERNAL_PORT = 8202
EXTERNAL_TARGET = f"http://127.0.0.1:{EXTERNAL_PORT}"
CONTROL_PREFIX = "/api/external-connection"
EXTERNAL_ROUTES = (
    ("POST", "/prompt"),
    ("GET", "/history"),
    ("GET", "/history/{prompt_id}"),
    ("GET", "/view"),
    ("GET", "/ws"),
    ("GET", "/queue"),
    ("GET", "/object_info"),
    ("GET", "/object_info/{node_class}"),
    ("GET", "/system_stats"),
    ("GET", "/embeddings"),
    ("GET", "/extensions"),
    ("POST", "/api/lighbd/enqueue"),
    ("GET", "/api/lighbd/session/{sid}"),
    ("GET", "/api/lighbd/image/{pid}"),
    ("POST", "/api/lighbd/reroll"),
    ("GET", "/api/illustration_context/session/{sid}/manifest"),
    ("GET", "/s/{key}"),
    ("GET", "/api/illustration_context/bridge/health"),
    ("GET", "/api/illustration_context/bridge/bots"),
    ("POST", "/api/illustration_context/bridge/bots"),
    ("GET", "/api/illustration_context/bridge/sessions"),
    ("POST", "/api/illustration_context/bridge/client-log"),
    ("GET", "/api/illustration_context/bridge/session/{sid}"),
    ("GET", "/api/illustration_context/bridge/session/{sid}/image/{slot}"),
)
_EXTERNAL_PATTERNS = tuple(
    (method, re.compile(re.sub(r"\{[^}]+\}", "[^/]+", path)))
    for method, path in EXTERNAL_ROUTES
)


def allowed_external_path(path: str, method: str) -> bool:
    method = "GET" if method == "HEAD" else method
    return any(method == allowed and pattern.fullmatch(path) for allowed, pattern in _EXTERNAL_PATTERNS)


def allowed_external_control(path: str, method: str) -> bool:
    return (method == "GET" and path == CONTROL_PREFIX + "/status") or (
        method == "POST" and bool(re.fullmatch(
            re.escape(CONTROL_PREFIX) + r"/(?:tunnel|tailscale)/(?:start|stop|restart)", path,
        ))
    )


def register_external_connection_control(server) -> None:
    """Reuse the existing tunnel lifecycle in the shared worker, targeting 8202."""
    server._TUNNEL_TARGET = EXTERNAL_TARGET
    server._TAILSCALE_SERVE_HTTPS_PORT = EXTERNAL_PORT
    server._TAILSCALE_SERVE_TARGET = EXTERNAL_TARGET
    lock = asyncio.Lock()

    async def connection_status(kind):
        if kind == "tailscale" and not server._tailscale_bin():
            return {"status": "unavailable", "url": None,
                    "error": "Tailscale을 설치하고 로그인한 뒤 다시 시도하세요."}
        response = await getattr(server, f"handle_api_{kind}_status")(None)
        result = json.loads(response.body)
        if not isinstance(result, dict):
            raise ValueError(f"{kind} 상태 응답이 객체가 아닙니다.")
        return result

    async def safe_connection_status(kind):
        # One provider being unreachable (for example Tailscale's daemon still
        # starting) must not hide the other provider's state from the UI.
        try:
            return await connection_status(kind)
        except Exception as exc:
            print(
                f"[EXTERNAL] {kind} 상태 조회 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            return {"status": "error", "url": None,
                    "error": str(exc) or f"{kind} 연결 상태를 확인할 수 없습니다."}

    async def status(request):
        # Status reads must not block behind a long-running start/stop lock.
        tunnel, tailscale = await asyncio.gather(
            safe_connection_status("tunnel"), safe_connection_status("tailscale")
        )
        return web.json_response({"tunnel": tunnel, "tailscale": tailscale})

    async def change(request):
        kind, action = request.match_info["kind"], request.match_info["action"]
        async with lock:
            if action != "stop":
                other = "tailscale" if kind == "tunnel" else "tunnel"
                try:
                    state = await connection_status(other)
                except Exception as exc:
                    print(
                        f"[EXTERNAL] {other} 상태 조회 실패: "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    return web.json_response(
                        {"error": f"{other} 연결 상태를 확인할 수 없습니다: {exc}"},
                        status=503,
                    )
                if state.get("status") == "running":
                    return web.json_response({"error": "사용 중인 다른 주소의 연결을 먼저 종료하세요."}, status=409)
                if state.get("status") == "error":
                    return web.json_response({"error": state.get("error") or "다른 주소의 연결 상태를 확인할 수 없습니다."}, status=503)
            if action == "restart":
                response = await getattr(server, f"handle_api_{kind}_stop")(request)
                if response.status >= 400:
                    return response
                action = "start"
            return await getattr(server, f"handle_api_{kind}_{action}")(request)

    server.app.router.add_get(CONTROL_PREFIX + "/status", status)
    server.app.router.add_post(CONTROL_PREFIX + "/{kind:tunnel|tailscale}/{action:start|stop|restart}", change)
