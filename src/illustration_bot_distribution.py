"""Editable shipped illustration bots, mirrored into the engine's bot store."""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import threading

from aiohttp import web

from app_paths import SOURCE_DIR, data_root
from json_store import JsonFileCache, json_file_signature, write_json_atomic
from preset_locations import PRESET_LOCATIONS


# 워크숍 스코프별 배포 봇 저장소. 삽화와 감정 삽화는 봇 데이터를 공유하지 않는다.
_BOT_STORES = {
    "illust": {"defaults": "illustration", "data": "bot.json", "dir": "bot"},
    "emotion": {"defaults": "emotion", "data": "emotion_bot.json", "dir": "emotion_bot"},
}


class IllustrationBotDistribution:
    def __init__(self, source: Path = SOURCE_DIR, scope: str = "illust"):
        store = _BOT_STORES.get(scope)
        if store is None:
            raise ValueError("봇 배포 저장소 스코프를 확인하세요.")
        self.scope = scope
        location = PRESET_LOCATIONS[f"{store['defaults']}_bot"]
        self.path = location.builtin_path(source)
        self.runtime_path = location.local_path(source)
        self.bot_dir = data_root(source) / store["dir"]
        self.tokens: set[str] = set()
        self.lock = threading.RLock()
        self.request_lock = asyncio.Lock()
        self._document_cache = JsonFileCache()
        self._runtime_cache = JsonFileCache()
        self._synced_signature = None

    def read(self) -> dict:
        document = self._document_cache.read(self.path)
        if not isinstance(document, dict) or not isinstance(document.get("bots"), list):
            raise ValueError("배포 봇 파일을 확인하세요.")
        names = [bot["name"] for bot in document["bots"]]
        if len(set(names)) != len(names):
            raise ValueError("배포 봇 이름이 중복되었습니다.")
        for name in names:
            self.directory(name)
        return document

    @staticmethod
    def revision(document: dict) -> str:
        return hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def directory(self, name: str) -> Path:
        if not isinstance(name, str) or not name or name in {".", ".."} or any(c in name for c in '/\\:'):
            raise ValueError("봇 이름을 확인하세요.")
        path = (self.bot_dir / name).resolve()
        path.relative_to(self.bot_dir.resolve())
        return path

    def file(self, name: str, relative: str) -> Path:
        root = self.directory(name)
        path = (root / relative).resolve()
        path.relative_to(root)
        if path == root:
            raise ValueError("봇 파일 경로를 확인하세요.")
        return path

    def _runtime(self) -> dict:
        return self._runtime_cache.read(self.runtime_path) if self.runtime_path.exists() else {"bots": []}

    def _save_runtime(self, data: dict) -> None:
        # Reuse the engine's atomic writer and user-data backup policy.
        # Select the scope explicitly so an emotion sync never writes the illust store.
        from modes.bot_mode import _save_bot_data, reset_bot_scope, set_bot_scope
        token = set_bot_scope(self.scope)
        try:
            _save_bot_data(data)
        finally:
            reset_bot_scope(token)

    def sync(self) -> None:
        """Apply a changed distribution without replacing any user bots."""
        if not self.path.exists():
            self._synced_signature = None
            return
        with self.lock:
            signature = (json_file_signature(self.path), json_file_signature(self.runtime_path))
            if signature == self._synced_signature:
                return
            document = self.read()
            revision = self.revision(document)
            data = self._runtime()
            if data.get("_distribution_revision") == revision:
                self._synced_signature = signature
                return
            # Bot names are the engine's identities. An explicitly demoted
            # user bot must keep that identity even if an update ships it again.
            protected = set(data.get("_distribution_local_names", [])) & {bot["name"] for bot in data.get("bots", [])}
            previous = set(data.get("_distribution_bot_names", []))
            shipped = {bot["name"]: bot for bot in document["bots"]}
            protected.update(bot["name"] for bot in data.get("bots", [])
                             if bot.get("distribution_scope") == "local" or
                             bot["name"] not in previous and bot["name"] in shipped and bot != shipped[bot["name"]])
            data["_distribution_local_names"] = sorted(protected)
            names = set(shipped) - protected
            local = [bot for bot in data.get("bots", []) if bot["name"] in protected or bot["name"] not in names | previous]
            data["bots"] = [*(deepcopy(bot) for bot in document["bots"] if bot["name"] in names), *local]
            for bot in data["bots"]:
                bot["distribution_scope"] = "builtin" if bot["name"] in names else "local"
            self._sync_prompt_presets(data, document, names)
            files = document.get("files", {})
            # Only remove assets previously installed by this distribution.
            for name, paths in data.get("_distribution_files", {}).items():
                if name in protected:
                    continue
                for relative in paths:
                    if relative not in files.get(name, {}):
                        self.file(name, relative).unlink(missing_ok=True)
            for name, entries in files.items():
                if name in protected:
                    continue
                if name not in names:
                    raise ValueError("배포 이미지의 봇을 확인하세요.")
                for relative, encoded in entries.items():
                    target = self.file(name, relative)
                    content = base64.b64decode(encoded, validate=True)
                    if not target.exists() or target.read_bytes() != content:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
            self._mark(data, document)
            data.pop("files", None)
            self._save_runtime(data)

    @staticmethod
    def _sync_prompt_presets(data: dict, document: dict, names: set[str]) -> None:
        """Mirror shipped prompts under noncolliding runtime names; keep user values."""
        presets = data.setdefault("system_prompt_presets", {})
        previous = set(data.get("_distribution_prompt_names", []))
        local_references = {bot.get("system_prompt_preset") for bot in data.get("bots", [])
                            if bot["name"] not in names and bot.get("preset_scope", "local") == "local"}
        reserved = (set(presets) - previous) | local_references
        aliases = {}
        for bot in data.get("bots", []):
            if bot["name"] not in names or bot.get("preset_scope", "local") != "local":
                continue
            name = bot.get("system_prompt_preset")
            if name not in document.get("system_prompt_presets", {}):
                continue
            if name not in aliases:
                alias, suffix = name, 1
                while alias in reserved:
                    alias = f"{name} (배포{'' if suffix == 1 else ' ' + str(suffix)})"
                    suffix += 1
                aliases[name] = alias
                reserved.add(alias)
            alias = aliases[name]
            presets[alias] = document["system_prompt_presets"][name]
            bot["system_prompt_preset"] = alias
            bot["system_prompt"] = presets[alias]
        for name in previous - set(aliases.values()) - local_references:
            presets.pop(name, None)
        data["_distribution_prompt_names"] = sorted(aliases.values())

    def _mark(self, data: dict, document: dict) -> None:
        data["_distribution_revision"] = self.revision(document)
        protected = set(data.get("_distribution_local_names", [])) & {bot["name"] for bot in data.get("bots", [])}
        data["_distribution_bot_names"] = [bot["name"] for bot in document["bots"] if bot["name"] not in protected]
        data["_distribution_files"] = {name: list(files) for name, files in document.get("files", {}).items() if name not in protected}

    def view(self) -> dict:
        self.sync()
        data = self._runtime()
        names = set(data.get("_distribution_bot_names", []))
        bots = [{"name": bot["name"], "scope": "builtin" if bot["name"] in names else "local"} for bot in data.get("bots", [])]
        return {"bots": bots, "revision": self.revision(bots)}

    def require_unlocked(self, token: str) -> None:
        if not isinstance(token, str) or token not in self.tokens:
            raise ValueError("배포 봇은 읽기 전용입니다. 배포 풀기로 잠금을 해제하세요.")

    def _capture(self, document: dict, data: dict, name: str) -> None:
        bot = next((deepcopy(bot) for bot in data["bots"] if bot["name"] == name), None)
        document["bots"] = [item for item in document["bots"] if item["name"] != name]
        document.setdefault("files", {}).pop(name, None)
        if bot is None:
            return
        bot.pop("distribution_scope", None)
        for character in bot.get("characters", []):
            character.pop("image_count", None)
        # Give each shipped bot its own prompt so editing a shared user preset
        # cannot change a locked distribution bot.
        if bot.get("preset_scope", "local") == "local":
            text = data.get("system_prompt_presets", {}).get(bot.get("system_prompt_preset"), bot.get("system_prompt", ""))
            preset = f"{name} 배포 지침"
            bot.update(system_prompt_preset=preset, system_prompt=text)
            document.setdefault("system_prompt_presets", {})[preset] = text
        document["bots"].append(bot)
        directory = self.directory(name)
        document["files"][name] = {
            path.relative_to(directory).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
            for path in sorted(directory.rglob("*"))
            if path.is_file() and not path.is_symlink() and not any(part.startswith(".") or part in {"backups", "__pycache__"} for part in path.relative_to(directory).parts)
        } if directory.exists() else {}

    def _commit(self, document: dict, data: dict) -> None:
        names = {bot["name"] for bot in document["bots"]} - set(data.get("_distribution_local_names", []))
        shipped = {bot["name"]: bot for bot in document["bots"] if bot["name"] in names}
        presets = {bot.get("system_prompt_preset") for bot in document["bots"] if bot.get("preset_scope", "local") == "local"}
        document["system_prompt_presets"] = {name: text for name, text in document.get("system_prompt_presets", {}).items() if name in presets}
        for bot in data["bots"]:
            if bot["name"] in shipped:
                bot.update(deepcopy(shipped[bot["name"]]))
            bot["distribution_scope"] = "builtin" if bot["name"] in names else "local"
        self._sync_prompt_presets(data, document, names)
        write_json_atomic(self.path, document)
        self._mark(data, document)
        self._save_runtime(data)

    def publish(self, name: str) -> None:
        with self.lock:
            document, data = self.read(), self._runtime()
            self._capture(document, data, name)
            self._commit(document, data)

    def change(self, body: dict) -> dict:
        with self.lock:
            action = body.get("action")
            token = body.get("builtin_unlock_token", "")
            if action == "lock":
                self.tokens.discard(token)
                return self.view()
            view = self.view()
            if body.get("revision") != view["revision"]:
                raise ValueError("봇 목록이 변경되었습니다. 새로고침한 뒤 다시 시도하세요.")
            if action == "unlock":
                if body.get("confirmed") is not True:
                    raise ValueError("배포 풀기 경고를 먼저 확인하세요.")
                token = secrets.token_urlsafe(32)
                self.tokens.add(token)
                return {**view, "builtin_unlock_token": token}
            name = body.get("bot_name", "")
            data = self._runtime()
            bot = next((item for item in data["bots"] if item["name"] == name), None)
            if bot is None:
                raise ValueError("봇을 선택하세요.")
            if action == "scope":
                self.require_unlocked(token)
                if body.get("scope") not in {"local", "builtin"}:
                    raise ValueError("배포 또는 사용자 봇을 선택하세요.")
                document = self.read()
                protected = set(data.get("_distribution_local_names", []))
                if body["scope"] == "builtin":
                    protected.discard(name)
                    self._capture(document, data, name)
                else:
                    protected.add(name)
                    document["bots"] = [item for item in document["bots"] if item["name"] != name]
                    document.setdefault("files", {}).pop(name, None)
                data["_distribution_local_names"] = sorted(protected)
                self._commit(document, data)
            elif action == "duplicate":
                from modes.bot_mode import validate_bot_storage_name
                new_name = body.get("name", "")
                error = validate_bot_storage_name(new_name, "새 봇 이름")
                if error:
                    raise ValueError(error)
                if any(item["name"].casefold() == new_name.casefold() for item in data["bots"]) or self.directory(new_name).exists():
                    raise ValueError("이미 존재하는 봇 이름입니다.")
                copied = deepcopy(bot)
                copied.update(name=new_name, distribution_scope="local")
                if copied.get("preset_scope", "local") == "local":
                    preset = f"{new_name} 기본"
                    data.setdefault("system_prompt_presets", {})[preset] = data["system_prompt_presets"].get(copied.get("system_prompt_preset"), "")
                    copied["system_prompt_preset"] = preset
                if self.directory(name).exists():
                    shutil.copytree(self.directory(name), self.directory(new_name))
                data["bots"].append(copied)
                self._save_runtime(data)
            else:
                raise ValueError("지원하지 않는 봇 배포 작업입니다.")
            return self.view()


bot_distributions = {scope: IllustrationBotDistribution(scope=scope) for scope in _BOT_STORES}
# 기존 호출부 호환: 삽화 워크숍 배포 저장소.
bot_distribution = bot_distributions["illust"]


def _distribution_for(request: web.Request) -> IllustrationBotDistribution:
    return bot_distributions["emotion" if request.query.get("scope") == "emotion" else "illust"]


async def handle_bot_distribution(request: web.Request) -> web.Response:
    try:
        distribution = _distribution_for(request)
        async with distribution.request_lock:
            result = distribution.change(await request.json()) if request.method == "POST" else distribution.view()
        return web.json_response(result, headers={"Cache-Control": "no-store"})
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _handle_bot_distribution_write(request: web.Request, body, handler):
    """Authorize and publish a bot mutation under the distribution lock."""
    try:
        distribution = _distribution_for(request)
        async with distribution.request_lock:
            distribution.sync()
            data = distribution._runtime()
            names = set(data.get("_distribution_bot_names", []))
            name = body.get("bot_name") or body.get("bot") or (body.get("name") if body.get("action") in {"add_bot", "remove_bot"} else "")
            affected = {name} & names
            if request.path.endswith(("/system_prompt_presets", "/system_prompt_preset_delete", "/system_prompt")):
                preset = body.get("preset_name") or body.get("name")
                bind_only = request.path.endswith("/system_prompt") and body.get("bind_only") is True and body.get("action", "save") == "save"
                if not bind_only:
                    preset_scope = body.get("scope", "local") if request.method == "DELETE" else "local"
                    affected.update(bot["name"] for bot in data.get("bots", []) if bot["name"] in names and bot.get("preset_scope") == preset_scope and bot.get("system_prompt_preset") == preset)
            if affected:
                from modes.bot_mode import _system_prompt_unlock_tokens
                system_token = body.get("builtin_unlock_token")
                system_unlocked = request.path.endswith(("/system_prompt", "/system_prompt_presets")) and isinstance(system_token, str) and system_token in _system_prompt_unlock_tokens
                if not system_unlocked:
                    distribution.require_unlocked(request.headers.get("X-Bot-Distribution-Token", ""))
            # Suggestions do not write bot data. Release the write lock while the
            # LLM runs so its cancellation request can reach the queue immediately.
            if request.path not in {"/api/bot_mode/suggest_name_tag", "/api/bot_mode/character_lore_auto_link", "/api/bot_mode/lore_npcs/analyze", "/api/bot_mode/lore_npcs/generate", "/api/bot_mode/lore_npcs/classify_images", "/api/bot_mode/auto_batch", "/api/bot_mode/character_cards/suggest_metadata", "/api/bot_mode/character_cards/suggest_metadata/cancel"}:
                response = await handler(request)
                result = json.loads(response.body) if isinstance(response, web.Response) and response.content_type == "application/json" else {}
                if response.status < 400 and not result.get("error") and result.get("success") is not False:
                    for name in affected:
                        distribution.publish(name)
                return response
        return await handler(request)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


@web.middleware
async def bot_distribution_middleware(request: web.Request, handler):
    """Enforce the same lock on both the legacy UI and the settings gateway."""
    if request.method not in {"POST", "DELETE"} or not request.path.startswith("/api/bot_mode/") or request.path == "/api/bot_mode/distribution" or request.path.startswith("/api/bot_mode/postprocess_"):
        return await handler(request)
    try:
        if request.path == "/api/bot_mode/archive":
            # Large original modules stream directly to their bot's staging file.
            request["bot_archive_publish"] = lambda publish: _handle_bot_distribution_write(request, request.query, publish)
            return await handler(request)
        body = await request.post() if request.content_type.startswith("multipart/") else await request.json()
        if not hasattr(body, "get"):
            raise ValueError("봇 요청을 확인하세요.")
        if request.path == "/api/bot_mode/upload":
            # Copy/hash independent uploads concurrently. The upload handler calls
            # this guard before classification and publication, so permission and
            # distribution changes remain serialized with the final file replace.
            request["bot_upload_publish"] = lambda publish: _handle_bot_distribution_write(request, body, publish)
            return await handler(request)
        return await _handle_bot_distribution_write(request, body, handler)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
