"""Named snapshots of asset style defaults and applied values."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import uuid

from configuration import settings_lock
from json_store import write_json_atomic
from asset_preset_scope import AssetPresetLock, write_scope_documents
from preset_locations import PRESET_LOCATIONS


STYLE_TAG_KEYS = {"background_preset"} | {
    f"{prefix}{kind}_preset"
    for prefix in ("", "anima_") for kind in ("artist", "quality", "negative")
}
STYLE_RAW_KEYS = {"raw_background_tags"} | {
    f"raw_{prefix}{kind}_tags"
    for prefix in ("", "anima_") for kind in ("artist", "quality", "negative")
}
STYLE_BOOLEAN_KEYS = {
    "hrf_anima", "hrf_sdxl", "hrf_restore_size", "style_lora_enabled",
    *(f"{model}_{part}_enabled" for model in ("anima", "sdxl") for part in ("fd", "hd", "ed")),
}
STYLE_NUMBER_RANGES = {
    "img_w": (64, 8192), "img_h": (64, 8192), "seed": (-1, 9007199254740991), "hrf_size": (1, 8),
}
STYLE_STRING_KEYS = STYLE_TAG_KEYS | {"asset_model_preset_id", "asset_workflow_type"}
STYLE_KEYS = STYLE_STRING_KEYS | STYLE_RAW_KEYS | STYLE_BOOLEAN_KEYS | STYLE_NUMBER_RANGES.keys() | {"style_lora_list"}


class StylePresetConflict(ValueError):
    pass


def validate_style_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {"defaults", "values"}:
        raise ValueError("그림체 기본값과 적용값을 확인하세요.")
    for slot in settings.values():
        if not isinstance(slot, dict) or set(slot) - STYLE_KEYS:
            raise ValueError("그림체 설정에 포함할 수 없는 항목이 있습니다.")
        for key, value in slot.items():
            valid = True
            if key in STYLE_STRING_KEYS:
                valid = isinstance(value, str)
            elif key in STYLE_RAW_KEYS:
                valid = isinstance(value, list) and all(isinstance(tag, str) for tag in value)
            elif key in STYLE_BOOLEAN_KEYS:
                valid = isinstance(value, bool)
            elif key in STYLE_NUMBER_RANGES:
                low, high = STYLE_NUMBER_RANGES[key]
                valid = (type(value) in (int, float) and math.isfinite(value) and low <= value <= high
                         and (key == "hrf_size" or value == int(value)))
            elif key == "style_lora_list":
                valid = isinstance(value, list) and all(isinstance(item, dict) for item in value)
            if not valid:
                raise ValueError(f"그림체 설정 값을 확인하세요: {key}")
        if "asset_workflow_type" in slot and slot["asset_workflow_type"] not in {"anima_only", "anima_ilxl"}:
            raise ValueError("ANIMA 또는 ANIMA+ILXL 그림 모델을 선택하세요.")
    json.dumps(settings, allow_nan=False)
    return copy.deepcopy(settings)


class AssetStylePresets:
    def __init__(self, config_file: str | Path, builtin_file: str | Path | None = None):
        self.path = PRESET_LOCATIONS["asset_style"].local_path(config_file=config_file)
        self.builtin_file = Path(builtin_file) if builtin_file else PRESET_LOCATIONS["asset_style"].builtin_path()
        self.lock = AssetPresetLock()

    def _read_file(self, path):
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            document = {"version": 1, "presets": []}
        if not isinstance(document, dict) or document.get("version") != 1 or not isinstance(document.get("presets"), list):
            raise ValueError("그림체 프리셋 파일을 확인하세요.")
        identities, names = set(), set()
        for preset in document["presets"]:
            if (not isinstance(preset, dict) or not isinstance(preset.get("id"), str) or not preset["id"]
                    or preset["id"] in identities):
                raise ValueError("그림체 프리셋 파일을 확인하세요.")
            name = self._name(preset.get("name"))
            if name.casefold() in names:
                raise ValueError("같은 이름의 그림체 프리셋이 있습니다.")
            identities.add(preset["id"])
            names.add(name.casefold())
            validate_style_settings(preset.get("settings"))
        return document

    def read(self):
        with settings_lock(self.path):
            presets = [{**item, "scope": scope} for scope, path in (("builtin", self.builtin_file), ("local", self.path))
                       for item in self._read_file(path)["presets"]]
            # A release can reintroduce an item previously moved to the user
            # channel. Preserve the user's ID (also used by saved chains), and
            # distinguish only the shipped item's presentation identity.
            identities = {item["id"] for item in presets if item["scope"] == "local"}
            for item in presets:
                if item["scope"] != "builtin":
                    continue
                if item["id"] in identities:
                    item["source_id"] = item["id"]
                    while item["id"] in identities:
                        item["id"] = "builtin:" + item["id"]
                identities.add(item["id"])
            return {"version": 1, "presets": presets}

    @staticmethod
    def _stored_preset(item):
        stored = {key: value for key, value in item.items() if key not in {"scope", "source_id"}}
        if item["scope"] == "builtin":
            stored["id"] = item.get("source_id", item["id"])
        return stored

    @staticmethod
    def _name(value):
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 80:
            raise ValueError("프리셋 이름을 1~80자로 입력하세요.")
        return value.strip()

    def catalog(self):
        document = self.read()
        revision = hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return {**document, "revision": revision}

    def change(self, body):
        if not isinstance(body, dict) or set(body) - {"action", "id", "name", "settings", "revision", "confirmed", "scope", "builtin_unlock_token"}:
            raise ValueError("그림체 프리셋 요청을 확인하세요.")
        with settings_lock(self.path):
            catalog = self.catalog()
            if body.get("action") in {"unlock", "lock"}:
                return {**catalog, **self.lock.change(body)}
            if body.get("revision") != catalog["revision"]:
                raise StylePresetConflict("그림체 프리셋이 변경되었습니다. 목록을 새로고침한 뒤 다시 저장하세요.")
            action = body.get("action")
            if action not in {"create", "save", "rename", "delete", "scope"}:
                raise ValueError("지원하지 않는 그림체 프리셋 요청입니다.")
            presets = catalog["presets"]
            preset = next((item for item in presets if item["id"] == body.get("id")), None)
            if action != "create" and preset is None:
                raise ValueError("그림체 프리셋을 찾을 수 없습니다.")
            original_scope = preset["scope"] if preset and action != "create" else "local"
            if action != "create" and (original_scope == "builtin" or action == "scope"):
                self.lock.require(body)
            if action in {"create", "rename"}:
                name = self._name(body.get("name"))
                if any(item["scope"] == original_scope and item["name"].casefold() == name.casefold() and (action == "create" or item is not preset) for item in presets):
                    raise ValueError("같은 이름의 그림체 프리셋이 있습니다.")
            if action == "create":
                preset = {"id": uuid.uuid4().hex, "name": name, "settings": validate_style_settings(body.get("settings")), "scope": "local"}
                presets.append(preset)
            elif action == "save":
                preset["settings"] = validate_style_settings(body.get("settings"))
            elif action == "rename":
                preset["name"] = name
            elif action == "scope":
                scope = body.get("scope")
                if scope not in {"builtin", "local"}:
                    raise ValueError("배포 또는 사용자 그룹을 선택하세요.")
                if any(item is not preset and item["scope"] == scope and item["name"].casefold() == preset["name"].casefold() for item in presets):
                    raise ValueError("대상 그룹에 같은 이름의 그림체가 있습니다.")
                preset["scope"] = scope
            else:
                if body.get("confirmed") is not True:
                    raise ValueError("삭제할 그림체 프리셋을 확인하세요.")
                presets.remove(preset)
            scopes = {original_scope, preset["scope"]}
            documents = {path: {"version": 1, "presets": [self._stored_preset(item)
                         for item in presets if item["scope"] == scope]}
                         for scope, path in (("builtin", self.builtin_file), ("local", self.path)) if scope in scopes}
            write_scope_documents(documents, writer=write_json_atomic)
            return {**self.catalog(), "saved_id": preset["id"] if action != "delete" else ""}


def register_style_preset_runtime(server):
    from aiohttp import web

    store = AssetStylePresets(server.CONFIG_FILE)

    async def style_presets(request):
        try:
            result = store.change(await request.json()) if request.method == "POST" else store.catalog()
            return web.json_response({**result, "reference_modes_version": 1}, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=409 if isinstance(exc, StylePresetConflict) else 400)

    server.app.router.add_get("/api/asset/style-presets", style_presets)
    server.app.router.add_post("/api/asset/style-presets", style_presets)
