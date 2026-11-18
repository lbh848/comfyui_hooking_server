"""GitHub Markdown announcements shared by the settings and dashboard servers."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
import time
from urllib.parse import quote

import requests
from aiohttp import web

from app_paths import data_root
from configuration import settings_lock
from json_store import write_json_atomic


REPOSITORY = "lbh848/comfyui_hooking_server_notification"
API_URL = f"https://api.github.com/repos/{REPOSITORY}"
RAW_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/main/"
REFRESH_INTERVAL = 20 * 60


class NotificationBoard:
    def __init__(self, project_root):
        self.path = data_root(project_root) / "notification_cache.json"
        self.refresh_lock = threading.Lock()

    def cache(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[공지] 캐시를 다시 가져옵니다: {exc}")
        return {"items": [], "read": []}

    @staticmethod
    def listing(cache):
        return {key: cache[key] for key in ("items", "read", "checked_at") if key in cache}

    def refresh(self, force=False):
        with self.refresh_lock:
            cache = self.cache()
            if not force and time.time() - cache.get("checked_at", 0) < REFRESH_INTERVAL:
                return self.listing(cache), []
            try:
                response = requests.get(f"{API_URL}/contents", params={"ref": "main"}, timeout=10)
                response.raise_for_status()
                files = response.json()
                if not isinstance(files, list):
                    raise ValueError("Invalid GitHub contents response")
                items = [
                    {"name": f["name"], "sha": f["sha"], "download_url": RAW_URL + quote(f["name"])}
                    for f in files if f["type"] == "file" and f["name"].lower().endswith(".md")
                ]
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                print(f"[공지] GitHub 목록 갱신 실패: {exc}")
                cache = self.cache()
                warning = "GitHub에서 공지 목록을 가져오지 못했습니다."
                if cache["items"] or cache.get("checked_at"):
                    warning += " 마지막으로 저장한 목록을 표시합니다."
                return {**self.listing(cache), "warning": warning}, []

            with settings_lock(self.path):
                # Another tab or process may have opened an article during the fetch.
                cache = self.cache()
                previous = {item["name"]: item.get("sha") for item in cache["items"]}
                added = [item for item in items if item["name"] not in previous]
                bodies = cache.get("bodies", {})
                cache["bodies"] = {item["name"]: bodies[item["name"]] for item in items
                                   if item["name"] in bodies and previous.get(item["name"]) == item["sha"]}
                cache.update(items=items, checked_at=int(time.time()))
                write_json_atomic(self.path, cache)
                return self.listing(cache), added

    def mark_read(self, names):
        with settings_lock(self.path):
            cache = self.cache()
            known = {item["name"] for item in cache["items"]}
            read = sorted(set(cache["read"]) | known.intersection(names))
            if read != cache["read"]:
                cache["read"] = read
                write_json_atomic(self.path, cache)
            return cache["read"]

    def content(self, name):
        cache = self.cache()
        item = next((item for item in cache["items"] if item["name"] == name), None)
        if item is None:
            raise web.HTTPNotFound(text=json.dumps({"error": "공지 목록을 새로고침해 주세요."}),
                                   content_type="application/json")
        if name in cache.get("bodies", {}):
            return cache["bodies"][name]

        # Address the listed blob so an edit on main cannot be cached under an old SHA.
        url = f"{API_URL}/git/blobs/{item['sha']}" if item.get("sha") else RAW_URL + quote(name)
        response = requests.get(url, headers={"Accept": "application/vnd.github.raw+json"}, timeout=10)
        response.raise_for_status()
        content = response.content.decode("utf-8-sig")
        with settings_lock(self.path):
            cache = self.cache()
            if item in cache["items"]:
                cache.setdefault("bodies", {})[name] = content
                write_json_atomic(self.path, cache)
        return content


def register_notifications(app: web.Application, project_root: Path | str, *, schema=None, on_new=None):
    board = NotificationBoard(project_root)
    prefix = "/api/setting/notifications" if schema else "/api/notifications"
    headers = {"Cache-Control": "no-store"}

    async def refresh(force=False):
        listing, added = await asyncio.to_thread(board.refresh, force)
        if added and on_new:
            await on_new("notification_new", {"count": len(added), "items": added})
        return listing

    async def handle(request):
        if schema:
            if request.headers.get("X-Comfy-Settings") != "1":
                raise web.HTTPForbidden()
            if request.headers.get("X-Comfy-Settings-Schema") != schema:
                return web.json_response({"error": "설정 화면을 새로고침해 주세요."}, status=409, headers=headers)
        try:
            if request.method == "POST":
                if request.content_type != "application/json":
                    raise web.HTTPUnsupportedMediaType()
                try:
                    body = await request.json()
                except ValueError:
                    body = None
                names = body.get("names") if isinstance(body, dict) else None
                if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                    return web.json_response({"error": "읽음 처리할 공지 목록을 확인하세요."}, status=400, headers=headers)
                result = {"success": True, "read": await asyncio.to_thread(board.mark_read, names)}
            elif request.path.endswith("/content"):
                content = await asyncio.to_thread(board.content, request.query.get("name", ""))
                result = {"content": content}
            elif schema:
                result = await refresh(request.query.get("refresh") == "1")
            else:
                result = board.listing(await asyncio.to_thread(board.cache))
        except (requests.RequestException, UnicodeDecodeError):
            return web.json_response({"error": "GitHub에서 공지 본문을 가져오지 못했습니다."}, status=502, headers=headers)
        except (OSError, ValueError):
            return web.json_response({"error": "공지 캐시 파일을 읽거나 저장할 수 없습니다."}, status=500, headers=headers)
        return web.json_response(result, headers=headers)

    app.router.add_get(prefix, handle)
    app.router.add_post(prefix + "/read", handle)
    if schema:
        app.router.add_get(prefix + "/content", handle)
    return refresh
