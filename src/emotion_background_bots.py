"""Automatic background bots, separate from character bots and manual backgrounds."""
from __future__ import annotations

import json
from pathlib import Path
import threading

from aiohttp import web

from asset_preset_scope import AssetPresetLock, write_scope_documents
from preset_locations import PRESET_LOCATIONS


SCOPES = ("builtin", "local")


def validate_name(value) -> str:
    name = value.strip() if isinstance(value, str) else ""
    if not name or len(name) > 100 or not name.isprintable():
        raise ValueError("봇 이름은 1~100자로 입력하세요.")
    return name


class EmotionBackgroundBots:
    def __init__(self, config_file, builtin_file=None):
        self.local_file = PRESET_LOCATIONS["emotion_background_bot"].local_path(config_file=config_file)
        self.builtin_file = Path(builtin_file) if builtin_file else (
            PRESET_LOCATIONS["emotion_background_bot"].builtin_path()
        )
        self._lock = threading.RLock()
        self.scope_lock = AssetPresetLock()

    @staticmethod
    def _read(path: Path) -> dict:
        if not path.is_file():
            return {"version": 1, "bots": []}
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if (not isinstance(document, dict) or document.get("version") != 1
                or not isinstance(document.get("bots"), list)):
            raise ValueError(f"{path.name} 파일 형식을 확인하세요.")
        for bot in document["bots"]:
            if not isinstance(bot, dict):
                raise ValueError(f"{path.name} 봇 형식을 확인하세요.")
            bot["name"] = validate_name(bot.get("name"))
        return document

    def _load(self):
        local = self._read(self.local_file)
        bots = [
            {**bot, "scope": scope}
            for scope, document in (("builtin", self._read(self.builtin_file)), ("local", local))
            for bot in document["bots"]
        ]
        names = [bot["name"] for bot in bots]
        if len(set(names)) != len(names):
            raise ValueError("같은 이름의 배경 봇이 있습니다.")
        selected = local.get("bot_selected", "")
        if selected not in names:
            selected = names[0] if names else ""
        return bots, selected

    def _commit(self, bots, selected):
        documents = {}
        for scope, path in (("builtin", self.builtin_file), ("local", self.local_file)):
            document = {
                "version": 1,
                "bots": [{key: value for key, value in bot.items() if key != "scope"}
                         for bot in bots if bot["scope"] == scope],
            }
            if scope == "local":
                document["bot_selected"] = selected
            # Selecting or editing user bots must never rewrite shipped bots.
            if document == self._read(path) or (scope == "builtin" and not document["bots"] and not path.exists()):
                continue
            documents[path] = document
        write_scope_documents(documents)
        return {"bots": bots, "bot_selected": selected}

    def snapshot(self):
        with self._lock:
            bots, selected = self._load()
            return {"bots": bots, "bot_selected": selected}

    def change(self, body):
        with self._lock:
            bots, selected = self._load()
            action = body.get("action")
            if action in ("unlock", "lock"):
                return {"bots": bots, "bot_selected": selected, **self.scope_lock.change(body)}
            if action not in ("create", "duplicate", "rename", "delete", "scope", "select"):
                raise ValueError("지원하지 않는 배경 봇 작업입니다.")
            target = None
            if action != "create":
                target = next((bot for bot in bots if bot["name"] == body.get("name")), None)
                if target is None:
                    raise KeyError("배경 봇을 찾을 수 없습니다.")
            if action == "select":
                return self._commit(bots, target["name"])
            scope = body.get("scope", "local") if action in ("create", "duplicate", "scope") else target["scope"]
            if scope not in SCOPES:
                raise ValueError("봇 구분을 확인하세요.")
            if scope == "builtin" or (target and target["scope"] == "builtin" and action != "duplicate"):
                self.scope_lock.require(body)
            if action in ("create", "duplicate", "rename"):
                name = validate_name(body.get("name") if action == "create" else body.get("new_name"))
                if any(bot["name"] == name and not (action == "rename" and bot is target) for bot in bots):
                    raise ValueError("같은 이름의 배경 봇이 있습니다.")
                if action == "rename":
                    from emotion_background_cards import background_bot_key
                    target.setdefault("background_storage_id", background_bot_key(target))
                    if selected == target["name"]:
                        selected = name
                    target["name"] = name
                else:
                    bots.append({**(target or {}), "name": name, "scope": scope})
                    if action == "duplicate":
                        bots[-1].pop("background_storage_id", None)
                    selected = name
            elif action == "delete":
                bots.remove(target)
                if selected == target["name"]:
                    selected = bots[0]["name"] if bots else ""
            elif action == "scope":
                target["scope"] = scope
            return self._commit(bots, selected)


def register_background_bots(server, builtin_file=None):
    store = EmotionBackgroundBots(server.CONFIG_FILE, builtin_file)

    async def background_bots(request):
        if str(request.query.get("scope") or "").strip().lower() != "emotion":
            return web.json_response({"error": "감정 삽화 워크숍에서만 사용할 수 있습니다."}, status=400)
        try:
            if request.method == "GET":
                result = store.snapshot()
            else:
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("배경 봇 요청 형식을 확인하세요.")
                result = store.change(body)
        except KeyError as error:
            return web.json_response({"error": str(error.args[0])}, status=404)
        except (ValueError, OSError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    server.app.router.add_get("/api/illustration/background-bots", background_bots)
    server.app.router.add_post("/api/illustration/background-bots", background_bots)
    from emotion_background_cards import register_background_cards
    register_background_cards(server, store)
