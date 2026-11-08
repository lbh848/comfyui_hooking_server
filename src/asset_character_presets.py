"""Character definitions split between shipped presets and user tag storage."""
import copy
import hashlib
import json
from pathlib import Path

from asset_preset_scope import AssetPresetLock, write_scope_documents
from configuration import settings_lock
from lora_paths import validate_lora_storage_name


class AssetCharacterPresets:
    def __init__(self, tags_file, builtin_file=None):
        self.path = Path(tags_file)
        self.builtin_file = Path(builtin_file) if builtin_file else Path(__file__).parent / "presets/asset_character_presets.json"
        self.lock = AssetPresetLock()

    @staticmethod
    def _read_file(path):
        value = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {"characters": {}}
        if (not isinstance(value, dict) or not isinstance(value.get("characters", {}), dict)
                or any(not isinstance(item, dict) for item in value.get("characters", {}).values())):
            raise ValueError("캐릭터 프리셋 파일을 확인하세요.")
        value.setdefault("characters", {})
        return value

    def builtin(self):
        return self._read_file(self.builtin_file)["characters"]

    def catalog(self):
        with settings_lock(self.path):
            local, builtin = self._read_file(self.path)["characters"], self.builtin()
            scopes = {**dict.fromkeys(builtin, "builtin"), **dict.fromkeys(local, "local")}
            revision = hashlib.sha256(json.dumps([builtin, local], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            return {"characters": {**builtin, **local}, "scopes": scopes, "revision": revision}

    def change(self, body):
        if not isinstance(body, dict):
            raise ValueError("캐릭터 프리셋 요청을 확인하세요.")
        with settings_lock(self.path):
            catalog = self.catalog()
            action = body.get("action")
            if action in {"unlock", "lock"}:
                return {**catalog, **self.lock.change(body)}
            if ("revision" in body or action == "scope") and body.get("revision") != catalog["revision"]:
                raise ValueError("캐릭터가 변경되었습니다. 목록을 새로고침하세요.")
            if action not in {"create", "duplicate", "save", "delete", "scope"}:
                raise ValueError("지원하지 않는 캐릭터 요청입니다.")
            name = body.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("캐릭터 이름을 입력하세요.")
            local = self._read_file(self.path)
            builtin = self._read_file(self.builtin_file)
            builtin["version"] = 1
            documents = {"local": local, "builtin": builtin}
            if action in {"create", "duplicate"}:
                name_error = validate_lora_storage_name(name, "캐릭터 이름")
                if name_error:
                    raise ValueError(name_error)
                if name in catalog["characters"]:
                    raise ValueError("이미 존재하는 캐릭터입니다.")
                if action == "duplicate":
                    source = catalog["characters"].get(body.get("source"))
                    if source is None:
                        raise ValueError("원본 캐릭터를 찾을 수 없습니다.")
                    character = copy.deepcopy(source)
                else:
                    character = {"appearance": "", "outfit": "", "expression": ""}
                local["characters"][name] = character
                scopes = {"local"}
            else:
                if name not in catalog["characters"]:
                    raise ValueError("캐릭터를 찾을 수 없습니다.")
                scope = catalog["scopes"][name]
                if scope == "builtin" or action == "scope":
                    self.lock.require(body)
                characters = documents[scope]["characters"]
                scopes = {scope}
                if action == "scope":
                    target = body.get("scope")
                    if target not in documents:
                        raise ValueError("배포 또는 사용자 그룹을 선택하세요.")
                    if target != scope:
                        if name in documents[target]["characters"]:
                            raise ValueError("대상 그룹에 같은 이름의 캐릭터가 있습니다.")
                        documents[target]["characters"][name] = characters.pop(name)
                        scopes.add(target)
                elif action == "delete":
                    del characters[name]
                else:
                    fields = body.get("fields")
                    if not isinstance(fields, dict) or set(fields) - {"appearance", "outfit", "expression", "group", "settings"}:
                        raise ValueError("캐릭터 설정을 확인하세요.")
                    settings = fields.get("settings")
                    if "settings" in fields and (not isinstance(settings, dict) or set(settings) != {"defaults", "values"}
                            or not all(isinstance(value, dict) for value in settings.values())):
                        raise ValueError("캐릭터 기본값과 적용값을 확인하세요.")
                    if any(not isinstance(value, str) for key, value in fields.items() if key != "settings"):
                        raise ValueError("캐릭터 설정 값을 확인하세요.")
                    character = characters[name]
                    character.update(copy.deepcopy(fields))
                    if settings is not None:
                        character.update({key: settings["values"].get(key, "") for key in ("appearance", "outfit", "expression")})
                    if "group" in fields:
                        character["group"] = fields["group"].strip()
            write_scope_documents({self.builtin_file if scope == "builtin" else self.path: documents[scope]
                                   for scope in ("builtin", "local") if scope in scopes})
            return {"success": True, **self.catalog(), "name": name}


def register_character_preset_runtime(server):
    from aiohttp import web

    async def characters(request):
        try:
            store = server.asset_mode.character_presets
            if request.method == "POST":
                body = await request.json()
                if not isinstance(body, dict) or body.get("action") not in {"scope", "unlock", "lock"}:
                    raise ValueError("캐릭터 그룹 요청을 확인하세요.")
                result = store.change(body)
                server.asset_mode.load_tags()
            else:
                result = store.catalog()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    server.app.router.add_get("/api/asset/character-presets", characters)
    server.app.router.add_post("/api/asset/character-presets", characters)
