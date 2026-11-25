"""Background asset nested theme/place/time catalog and isolated AssetMode storage."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Optional

from app_paths import data_root
from asset_preset_scope import AssetPresetLock, write_scope_documents
from configuration import settings_lock
from json_store import write_json_atomic
from modes.asset_mode import AssetMode
from preset_locations import PRESET_LOCATIONS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_ASSET_DATA_DIR = os.path.join(data_root(BASE_DIR), "bg_asset_data")
BG_ASSET_DIR = os.path.join(data_root(BASE_DIR), "bg_asset")
BG_TAGS_FILE = os.path.join(BG_ASSET_DATA_DIR, "tags.json")
BG_HIDDEN_TAGS_FILE = os.path.join(BG_ASSET_DATA_DIR, "hidden_tags.json")
BG_NAME_MAPPING_FILE = os.path.join(BG_ASSET_DATA_DIR, "name_mapping.json")

SHARED_STYLE_TAG_CATEGORIES = ("artist_presets", "quality_presets", "negative_presets")
BG_DEFAULT_TAGS = {"themes": {}}

_THEME_FIELDS = {"group", "common_tags", "negative_tags", "lora_enabled", "lora_list", "places"}
_PLACE_FIELDS = {"tags", "natural_language", "times"}
_TIME_FIELDS = {"tags", "natural_language"}


def _normalize_tag_list(value: object, label: str, *, strict: bool) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                if strict:
                    raise ValueError(f"{label} 항목 형식을 확인하세요.")
                continue
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
        return result
    if strict:
        raise ValueError(f"{label} 형식을 확인하세요.")
    return []


def _normalize_text(value: object, label: str, *, strict: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if strict:
        raise ValueError(f"{label} 형식을 확인하세요.")
    return ""


_WINDOWS_RESERVED_STEMS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _validate_folder_segment(name: object, label: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label}을(를) 입력하세요.")
    clean = name.strip()
    if len(clean) > 80:
        raise ValueError(f"{label}은(는) 80자 이하로 입력하세요.")
    invalid = list(dict.fromkeys(
        ch for ch in clean if not (ch.isalnum() or ch in (" ", "_", "-", "."))
    ))
    if invalid:
        bad = ", ".join(repr(ch) for ch in invalid)
        raise ValueError(f"{label}에 사용할 수 없는 문자가 있습니다: {bad}. 한글·영문·숫자와 공백, _, -, .만 사용할 수 있습니다.")
    if clean in {".", ".."} or clean.endswith("."):
        raise ValueError(f"{label}은(는) 마침표로만 구성하거나 마침표로 끝낼 수 없습니다.")
    if clean.split(".", 1)[0].upper() in _WINDOWS_RESERVED_STEMS:
        raise ValueError(f"{label}에 Windows 예약 이름 {clean!r}을(를) 사용할 수 없습니다.")
    return clean


def _normalize_lora_list(value: object, *, strict: bool) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        if strict:
            raise ValueError("배경 LoRA 목록 형식을 확인하세요.")
        return []
    result: list[dict] = []
    for item in value:
        if isinstance(item, dict):
            result.append(copy.deepcopy(item))
        elif strict:
            raise ValueError("배경 LoRA 항목 형식을 확인하세요.")
    return result


def normalize_bg_theme(raw: object, *, strict: bool = False) -> dict:
    """Validate and normalize a theme object with nested places and time-of-day entries."""
    if not isinstance(raw, dict):
        if strict:
            raise ValueError("배경 테마 설정 형식을 확인하세요.")
        raw = {}

    if strict:
        unknown = set(raw) - _THEME_FIELDS
        if unknown:
            raise ValueError("지원하지 않는 배경 테마 설정 항목이 포함되어 있습니다.")

    group_raw = raw.get("group", "")
    if strict and not isinstance(group_raw, str):
        raise ValueError("테마 그룹 형식을 확인하세요.")
    group = group_raw.strip() if isinstance(group_raw, str) else ""

    common_tags = _normalize_tag_list(raw.get("common_tags"), "테마 공통 태그", strict=strict)
    negative_tags = _normalize_tag_list(raw.get("negative_tags"), "배경 부정 태그", strict=strict)

    lora_enabled_raw = raw.get("lora_enabled", False)
    if strict and not isinstance(lora_enabled_raw, bool):
        raise ValueError("배경 LoRA 사용 여부 형식을 확인하세요.")
    lora_enabled = bool(lora_enabled_raw)
    lora_list = _normalize_lora_list(raw.get("lora_list"), strict=strict)

    places_raw = raw.get("places", {})
    if places_raw is None:
        places_raw = {}
    if not isinstance(places_raw, dict):
        if strict:
            raise ValueError("장소 목록 형식을 확인하세요.")
        places_raw = {}

    places: dict[str, dict] = {}
    for place_key, place_val in places_raw.items():
        if strict:
            place_name = _validate_folder_segment(place_key, "장소 이름")
        else:
            if not isinstance(place_key, str) or not place_key.strip():
                continue
            place_name = place_key.strip()
        if not isinstance(place_val, dict):
            if strict:
                raise ValueError(f"장소 '{place_name}' 설정 형식을 확인하세요.")
            place_val = {}
        if strict:
            unknown_place = set(place_val) - _PLACE_FIELDS
            if unknown_place:
                raise ValueError(f"장소 '{place_name}'에 지원하지 않는 항목이 있습니다.")

        place_tags = _normalize_tag_list(place_val.get("tags"), f"장소 '{place_name}' 태그", strict=strict)
        place_nl = _normalize_text(place_val.get("natural_language"), f"장소 '{place_name}' 자연어", strict=strict)

        times_raw = place_val.get("times", {})
        if times_raw is None:
            times_raw = {}
        if not isinstance(times_raw, dict):
            if strict:
                raise ValueError(f"장소 '{place_name}'의 시간대 목록 형식을 확인하세요.")
            times_raw = {}

        times: dict[str, dict] = {}
        for time_key, time_val in times_raw.items():
            if strict:
                time_name = _validate_folder_segment(time_key, "시간대 이름")
            else:
                if not isinstance(time_key, str) or not time_key.strip():
                    continue
                time_name = time_key.strip()
            if not isinstance(time_val, dict):
                if strict:
                    raise ValueError(f"시간대 '{place_name} / {time_name}' 설정 형식을 확인하세요.")
                time_val = {}
            if strict:
                unknown_time = set(time_val) - _TIME_FIELDS
                if unknown_time:
                    raise ValueError(f"시간대 '{place_name} / {time_name}'에 지원하지 않는 항목이 있습니다.")
            times[time_name] = {
                "tags": _normalize_tag_list(time_val.get("tags"), f"시간대 '{time_name}' 태그", strict=strict),
                "natural_language": _normalize_text(
                    time_val.get("natural_language"), f"시간대 '{time_name}' 자연어", strict=strict
                ),
            }

        places[place_name] = {
            "tags": place_tags,
            "natural_language": place_nl,
            "times": times,
        }

    return {
        "group": group,
        "common_tags": common_tags,
        "negative_tags": negative_tags,
        "lora_enabled": lora_enabled,
        "lora_list": lora_list,
        "places": places,
    }


def empty_bg_theme() -> dict:
    return normalize_bg_theme({})


def _bg_catalog_only(tags: dict) -> dict:
    raw_themes = tags.get("themes", {}) if isinstance(tags, dict) else {}
    if not isinstance(raw_themes, dict):
        raw_themes = {}
    return {
        "themes": {
            name: normalize_bg_theme(theme, strict=False)
            for name, theme in raw_themes.items()
            if isinstance(name, str) and name.strip()
        }
    }


class BgAssetThemePresets:
    """Theme definitions split between shipped presets and user background tag storage."""

    def __init__(self, tags_file: str | Path, builtin_file: str | Path | None = None):
        self.path = Path(tags_file)
        self.builtin_file = Path(builtin_file) if builtin_file else PRESET_LOCATIONS["bg_asset_theme"].builtin_path()
        self.lock = AssetPresetLock()

    @staticmethod
    def _read_file(path: Path) -> dict:
        value = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {"themes": {}}
        if (not isinstance(value, dict) or not isinstance(value.get("themes", {}), dict)
                or any(not isinstance(item, dict) for item in value.get("themes", {}).values())):
            raise ValueError("배경 테마 프리셋 파일을 확인하세요.")
        normalized_themes = {
            name: normalize_bg_theme(item, strict=False)
            for name, item in value.get("themes", {}).items()
            if isinstance(name, str) and name.strip()
        }
        return {"themes": normalized_themes}

    def builtin(self) -> dict:
        return self._read_file(self.builtin_file)["themes"]

    def catalog(self) -> dict:
        with settings_lock(self.path):
            local, builtin = self._read_file(self.path)["themes"], self.builtin()
            scopes = {**dict.fromkeys(builtin, "builtin"), **dict.fromkeys(local, "local")}
            revision = hashlib.sha256(
                json.dumps([builtin, local], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
            ).hexdigest()
            return {"themes": {**builtin, **local}, "scopes": scopes, "revision": revision}

    def change(self, body: dict) -> dict:
        if not isinstance(body, dict):
            raise ValueError("배경 테마 프리셋 요청을 확인하세요.")
        with settings_lock(self.path):
            catalog = self.catalog()
            action = body.get("action")
            if action in {"unlock", "lock"}:
                return {**catalog, **self.lock.change(body)}
            if ("revision" in body or action == "scope") and body.get("revision") != catalog["revision"]:
                raise ValueError("테마 목록이 변경되었습니다. 새로고침 후 다시 시도하세요.")
            if action not in {"create", "duplicate", "save", "delete", "scope"}:
                raise ValueError("지원하지 않는 테마 요청입니다.")
            name = body.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("테마 이름을 입력하세요.")
            name = name.strip()
            local = self._read_file(self.path)
            builtin = self._read_file(self.builtin_file)
            builtin["version"] = 1
            documents = {"local": local, "builtin": builtin}
            if action in {"create", "duplicate"}:
                name = _validate_folder_segment(name, "테마 이름")
                if name in catalog["themes"]:
                    raise ValueError("이미 존재하는 테마입니다.")
                if action == "duplicate":
                    source = catalog["themes"].get(body.get("source"))
                    if source is None:
                        raise ValueError("원본 테마를 찾을 수 없습니다.")
                    theme = normalize_bg_theme(copy.deepcopy(source), strict=False)
                else:
                    theme = empty_bg_theme()
                local["themes"][name] = theme
                scopes = {"local"}
            else:
                if name not in catalog["themes"]:
                    raise ValueError("테마를 찾을 수 없습니다.")
                scope = catalog["scopes"][name]
                if scope == "builtin" or action == "scope":
                    self.lock.require(body)
                themes = documents[scope]["themes"]
                scopes = {scope}
                if action == "scope":
                    target = body.get("scope")
                    if target not in documents:
                        raise ValueError("배포 또는 사용자 그룹을 선택하세요.")
                    if target != scope:
                        if name in documents[target]["themes"]:
                            raise ValueError("대상 그룹에 같은 이름의 테마가 있습니다.")
                        documents[target]["themes"][name] = themes.pop(name)
                        scopes.add(target)
                elif action == "delete":
                    del themes[name]
                else:
                    fields = body.get("fields")
                    if not isinstance(fields, dict):
                        raise ValueError("배경 테마 설정을 확인하세요.")
                    unknown = set(fields) - _THEME_FIELDS
                    if unknown:
                        raise ValueError("배경 테마 설정 항목을 확인하세요.")
                    merged = {**normalize_bg_theme(themes[name], strict=False), **copy.deepcopy(fields)}
                    themes[name] = normalize_bg_theme(merged, strict=True)
            write_scope_documents({
                self.builtin_file if scope == "builtin" else self.path: documents[scope]
                for scope in ("builtin", "local") if scope in scopes
            })
            return {"success": True, **self.catalog(), "name": name}


class BgAssetMode(AssetMode):
    """Isolated 3-level (theme / place / time) background asset manager reusing AssetMode."""

    def __init__(
        self,
        *,
        asset_dir: str = BG_ASSET_DIR,
        asset_data_dir: str = BG_ASSET_DATA_DIR,
        tags_file: str = BG_TAGS_FILE,
        hidden_tags_file: str = BG_HIDDEN_TAGS_FILE,
        name_mapping_file: str = BG_NAME_MAPPING_FILE,
        theme_builtin_file: str | Path | None = None,
        shared_style_tags_func: Optional[Callable[[], dict]] = None,
    ):
        theme_presets = BgAssetThemePresets(tags_file, builtin_file=theme_builtin_file)
        super().__init__(
            asset_dir=asset_dir,
            asset_data_dir=asset_data_dir,
            tags_file=tags_file,
            hidden_tags_file=hidden_tags_file,
            name_mapping_file=name_mapping_file,
            character_presets=theme_presets,
            sync_lora_manage=False,
            asset_workspace="bg_asset",
            entity_labels={"entity": "테마", "level2": "장소", "level3": "시간대"},
        )
        self.theme_presets = theme_presets
        self.shared_style_tags_func = shared_style_tags_func
        self._tags = copy.deepcopy(BG_DEFAULT_TAGS)

    def load_tags(self):
        os.makedirs(self.asset_data_dir, exist_ok=True)
        if os.path.isfile(self.tags_file):
            try:
                with open(self.tags_file, "r", encoding="utf-8-sig") as f:
                    stored = json.load(f)
                self._tags = _bg_catalog_only(stored if isinstance(stored, dict) else {})
                self._tags_loaded = True
            except Exception as exc:
                self._log("bg_tags_load_error", {"error": str(exc)})
                self._tags = copy.deepcopy(BG_DEFAULT_TAGS)
                self._tags_loaded = True
        else:
            self._tags = copy.deepcopy(BG_DEFAULT_TAGS)
            self._tags_loaded = True
        self._tags_file_stamp = self._current_tags_file_stamp()

    def save_tags(self):
        if not self._tags_loaded:
            return
        os.makedirs(self.asset_data_dir, exist_ok=True)
        self._tags = _bg_catalog_only(self._tags)
        write_json_atomic(Path(self.tags_file), self._tags)
        self._tags_file_stamp = self._current_tags_file_stamp()

    def get_local_tags(self) -> dict:
        if not self._tags_loaded or self._tags_file_stamp != self._current_tags_file_stamp():
            self.load_tags()
        return copy.deepcopy(_bg_catalog_only(self._tags))

    def get_tags(self) -> dict:
        tags = self.get_local_tags()
        builtin_themes = self.theme_presets.builtin()
        local_themes = tags.get("themes", {})
        scopes: dict[str, dict[str, str]] = {
            "themes": {**dict.fromkeys(builtin_themes, "builtin"), **dict.fromkeys(local_themes, "local")},
        }
        tags["themes"] = {**builtin_themes, **local_themes}
        if callable(self.shared_style_tags_func):
            shared = self.shared_style_tags_func() or {}
            shared_scopes = shared.get("preset_scopes") or {}
            for category in SHARED_STYLE_TAG_CATEGORIES:
                entries = shared.get(category, {})
                tags[category] = copy.deepcopy(entries) if isinstance(entries, dict) else {}
                cat_scopes = shared_scopes.get(category, {})
                scopes[category] = copy.deepcopy(cat_scopes) if isinstance(cat_scopes, dict) else {}
        else:
            for category in SHARED_STYLE_TAG_CATEGORIES:
                tags.setdefault(category, {})
                scopes.setdefault(category, {})
        tags["preset_scopes"] = scopes
        return tags

    def list_themes(self) -> list[str]:
        return list(self.get_tags().get("themes", {}).keys())

    def list_characters(self) -> list[str]:
        return self.list_themes()

    def update_theme(
        self,
        name: str,
        *,
        fields: Optional[dict] = None,
        group: Optional[str] = None,
        builtin_unlock_token=None,
    ) -> dict:
        merged_fields = dict(fields) if isinstance(fields, dict) else {}
        if group is not None:
            merged_fields["group"] = group
        try:
            self.theme_presets.change({
                "action": "save",
                "name": name,
                "fields": merged_fields,
                "builtin_unlock_token": builtin_unlock_token,
            })
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        self.load_tags()
        return {"success": True}


def register_bg_theme_preset_runtime(server):
    from aiohttp import web

    async def themes(request):
        try:
            store = server.bg_asset_mode.theme_presets
            if request.method == "POST":
                body = await request.json()
                if not isinstance(body, dict) or body.get("action") not in {"scope", "unlock", "lock"}:
                    raise ValueError("테마 그룹 요청을 확인하세요.")
                result = store.change(body)
                server.bg_asset_mode.load_tags()
            else:
                result = store.catalog()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    server.app.router.add_get("/api/bg-asset/theme-presets", themes)
    server.app.router.add_post("/api/bg-asset/theme-presets", themes)
