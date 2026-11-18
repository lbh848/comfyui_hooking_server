"""Local settings endpoints for selecting resources and following large copies."""
from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event
import traceback
from uuid import uuid4

from aiohttp import web

from app_paths import install_root
from project_migration import ProjectMigration


def register_project_migration(app, project_root: Path, schema: str):
    install = install_root(project_root)
    candidates = (parent / "comfyui_hooking_server" for parent in install.parents)
    default = next((str(p) for p in candidates if p != install and (p / "asset_data").is_dir()), "")
    job = {"state": "idle", "results": [], "total": 0, "bytes": 0, "current": ""}
    task = None
    cancel = Event()

    def progress(size):
        if cancel.is_set():
            raise InterruptedError("가져오기를 중단했습니다.")
        job["bytes"] += size

    async def run(migration, selected):
        try:
            for item in selected:
                if cancel.is_set():
                    break
                job["current"] = item.label
                try:
                    message = await asyncio.to_thread(migration.import_item, item, progress)
                    state = "skipped" if item.imported else "done"
                except Exception as exc:
                    if cancel.is_set():
                        break
                    traceback.print_exc()
                    state, message = "error", str(exc)
                job["results"].append({"id": item.id, "label": item.label, "target": item.target,
                                       "state": state, "message": message})
        finally:
            job.update(state="stopped" if cancel.is_set() else "finished", current="")

    async def handle(request):
        nonlocal task
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            return web.json_response({"error": "설정 화면을 새로고침해 주세요."}, status=409, headers=headers)
        if request.method == "GET":
            return web.json_response({"default_source": default, "job": job}, headers=headers)
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        action = request.match_info["action"]
        if action == "stop":
            cancel.set()
            return web.json_response({"job": job}, headers=headers)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("원본 폴더와 가져올 항목을 확인하세요.")
            migration = await asyncio.to_thread(ProjectMigration, project_root, body.get("source"))
            if action == "scan":
                return web.json_response(await asyncio.to_thread(migration.preview), headers=headers)
            if task is not None and not task.done():
                return web.json_response({"error": "다른 가져오기가 진행 중입니다."}, status=409, headers=headers)
            selection = body.get("selection")
            if not isinstance(selection, dict) or not selection:
                raise ValueError("가져올 항목을 선택하세요.")
            selected = [i for i in migration.resources.values() if i.id in selection]
            if len(selected) != len(selection) or any(selection[i.id] != i.target for i in selected):
                raise ValueError("목록이 변경되었습니다. 원본을 다시 조회하세요.")
            cancel.clear()
            job.update(id=uuid4().hex, state="running", source=str(migration.source), current="", results=[], bytes=0, total=len(selected))
            task = asyncio.create_task(run(migration, selected))
            return web.json_response({"job": job}, status=202, headers=headers)
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)

    async def close(_app):
        cancel.set()
        if task is not None:
            await task

    app.router.add_get("/api/setting/migration", handle)
    app.router.add_post("/api/setting/migration/{action:scan|start|stop}", handle)
    app.on_cleanup.append(close)
