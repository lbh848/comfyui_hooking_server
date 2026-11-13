"""Request the normal aiohttp shutdown path from the local workshop UI."""
from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable

from aiohttp import web


def _exit_server() -> None:
    # web.run_app catches this and runs cleanup contexts and on_cleanup hooks.
    raise web.GracefulExit()


def register_program_shutdown(
    app: web.Application, schema: str, *, exit_server: Callable[[], None] = _exit_server,
    prepare_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> None:
    scheduled: asyncio.TimerHandle | None = None
    preparation: asyncio.Task | None = None
    phase = "idle"

    def schedule_exit(delay: float) -> None:
        nonlocal scheduled
        if scheduled is not None:
            scheduled.cancel()
        scheduled = asyncio.get_running_loop().call_later(delay, exit_server)

    async def prepare() -> None:
        nonlocal phase
        try:
            if prepare_shutdown is not None:
                await prepare_shutdown()
            phase = "ready"
            # Allow the browser to observe cleanup completion before exiting.
            # Still exit if the browser tab was already closed.
            schedule_exit(30)
        except Exception as exc:
            phase = "error"
            print(f"[SETTINGS_SHUTDOWN] 종료 준비 실패: {exc}")
            traceback.print_exc()

    @web.middleware
    async def reject_new_work(request: web.Request, handler):
        if (preparation is not None and request.path.startswith("/api/")
                and request.path not in {"/api/setting/status", "/api/setting/shutdown"}):
            return web.json_response({"error": "프로그램을 종료하고 있습니다."}, status=503,
                                     headers={"Cache-Control": "no-store"})
        return await handler(request)

    async def shutdown(request: web.Request) -> web.Response:
        nonlocal preparation, phase
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            return web.json_response({"error": "화면을 새로고침한 뒤 다시 종료해 주세요."}, status=409, headers=headers)
        if request.method == "GET":
            if phase == "ready":
                # A single acknowledgement timer: repeated polls must not delay exit.
                if scheduled is not None and scheduled.when() - asyncio.get_running_loop().time() > 1:
                    schedule_exit(0.5)
            return web.json_response({"status": phase}, headers=headers)
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict) or body.get("confirm") is not True:
            return web.json_response({"error": "프로그램 종료를 확인해 주세요."}, status=400, headers=headers)
        if preparation is None or phase == "error":
            phase = "shutting_down"
            preparation = asyncio.create_task(prepare())
        return web.json_response({"status": "shutting_down"}, status=202, headers=headers)

    async def cancel_scheduled_exit(app: web.Application) -> None:
        if preparation is not None and not preparation.done():
            preparation.cancel()
            await asyncio.gather(preparation, return_exceptions=True)
        if scheduled is not None:
            scheduled.cancel()

    app.middlewares.append(reject_new_work)
    app.router.add_get("/api/setting/shutdown", shutdown)
    app.router.add_post("/api/setting/shutdown", shutdown)
    app.on_cleanup.append(cancel_scheduled_exit)
