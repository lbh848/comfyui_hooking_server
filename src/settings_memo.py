"""Server-side memo pad shared by the work workspaces.

The text lives in the installation's data folder so it survives server
restarts while remaining separate from versioned settings.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web

from app_paths import data_root


MEMO_MAX_LENGTH = 100_000


def register_memo(app: web.Application, project_root: Path, schema: str) -> None:
    path = data_root(project_root) / "memo.txt"
    lock = asyncio.Lock()

    def guard(request: web.Request) -> None:
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            raise web.HTTPConflict(
                text=json.dumps({"error": "설정 화면을 새로고침해 주세요."}, ensure_ascii=False),
                content_type="application/json",
            )

    def response(value: dict, status: int = 200) -> web.Response:
        return web.json_response(value, status=status, headers={"Cache-Control": "no-store"})

    def read_memo() -> str:
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as source:
                return source.read()
        except FileNotFoundError:
            return ""

    def write_memo(memo: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            target.write(memo)
        temporary.replace(path)

    async def memo(request: web.Request) -> web.Response:
        guard(request)
        if request.method == "GET":
            async with lock:
                return response({"memo": await asyncio.to_thread(read_memo)})

        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            body = await request.json()
        except ValueError:
            return response({"error": "메모 요청 형식을 확인하세요."}, 400)
        if not isinstance(body, dict) or not isinstance(body.get("memo"), str):
            return response({"error": "메모 내용은 문자열이어야 합니다."}, 400)
        value = body["memo"]
        if len(value) > MEMO_MAX_LENGTH:
            return response(
                {"error": f"메모는 {MEMO_MAX_LENGTH:,}자까지 저장할 수 있습니다."}, 400
            )
        async with lock:
            await asyncio.to_thread(write_memo, value)
        return response({"memo": value})

    app.router.add_get("/api/setting/memo", memo)
    app.router.add_put("/api/setting/memo", memo)
