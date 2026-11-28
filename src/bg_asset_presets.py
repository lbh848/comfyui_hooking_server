"""Background asset nested theme/place/time catalog and isolated AssetMode storage."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

from app_paths import data_root
from asset_preset_scope import AssetPresetLock, without_local_reference_images, write_scope_documents
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
_PLACE_FIELDS = {"tags", "natural_language", "i2i_enabled", "i2i_denoise", "i2i_images", "times"}
_TIME_FIELDS = {"tags", "natural_language"}
BG_I2I_DEFAULT_DENOISE = 0.6
BG_I2I_MIN_DENOISE = 0.05
BG_I2I_MAX_DENOISE = 1.0
BG_I2I_STATIC_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg", ".bmp"}


def build_bg_i2i_image_entry(
    theme_name: str,
    place_name: str,
    time_name: str,
    filename: str,
    local_path: str,
) -> dict:
    """Build canonical i2i reference image metadata pointing to a gallery background image."""
    clean_filename = str(filename or "").strip() or Path(local_path).name
    return {
        "name": clean_filename,
        "local_path": str(Path(local_path).resolve()),
        "orig_filename": clean_filename,
        "preview_url": (
            f"/api/bg-asset/themes/{quote(theme_name, safe='')}"
            f"/places/{quote(place_name, safe='')}"
            f"/atmospheres/{quote(time_name, safe='')}"
            f"/images/{quote(clean_filename, safe='')}"
        ),
    }


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


def _normalize_i2i_denoise(value: object, *, strict: bool) -> float:
    if value is None:
        return BG_I2I_DEFAULT_DENOISE
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if strict:
            raise ValueError("i2i 변화 강도(denoise)는 숫자여야 합니다.")
        return BG_I2I_DEFAULT_DENOISE
    number = float(value)
    if not math.isfinite(number):
        if strict:
            raise ValueError("i2i 변화 강도(denoise) 값을 확인하세요.")
        return BG_I2I_DEFAULT_DENOISE
    if number < BG_I2I_MIN_DENOISE or number > BG_I2I_MAX_DENOISE:
        if strict:
            raise ValueError(f"i2i 변화 강도(denoise)는 {BG_I2I_MIN_DENOISE}~{BG_I2I_MAX_DENOISE} 사이여야 합니다.")
        number = max(BG_I2I_MIN_DENOISE, min(BG_I2I_MAX_DENOISE, number))
    return round(number, 4)


def _normalize_i2i_images(value: object, label: str, *, strict: bool) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        if strict:
            raise ValueError(f"{label} 형식을 확인하세요.")
        return []
    if strict and len(value) > 1:
        raise ValueError(f"{label}는 한 장만 등록할 수 있습니다.")
    result: list[dict] = []
    for item in value[:1]:
        if not isinstance(item, dict):
            if strict:
                raise ValueError(f"{label} 항목 형식을 확인하세요.")
            continue
        local_path = item.get("local_path")
        if not isinstance(local_path, str) or not local_path.strip():
            if strict:
                raise ValueError(f"{label} 경로를 확인하세요.")
            continue
        name = str(item.get("name") or Path(local_path).name).strip()
        orig_filename = str(item.get("orig_filename") or name).strip()
        preview_url = str(item.get("preview_url") or "").strip()
        if preview_url and not preview_url.startswith("/api/bg-asset/"):
            preview_url = ""
        result.append({
            "name": name,
            "local_path": local_path.strip(),
            "orig_filename": orig_filename,
            "preview_url": preview_url,
        })
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
        i2i_enabled_raw = place_val.get("i2i_enabled", False)
        if strict and not isinstance(i2i_enabled_raw, bool):
            raise ValueError(f"장소 '{place_name}'의 i2i 사용 여부 형식을 확인하세요.")
        i2i_enabled = bool(i2i_enabled_raw)
        i2i_denoise = _normalize_i2i_denoise(place_val.get("i2i_denoise"), strict=strict)
        i2i_images = _normalize_i2i_images(place_val.get("i2i_images"), f"장소 '{place_name}' i2i 기준 이미지", strict=strict)

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
            "i2i_enabled": i2i_enabled,
            "i2i_denoise": i2i_denoise,
            "i2i_images": i2i_images,
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
        return without_local_reference_images(self._read_file(self.builtin_file)["themes"])

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
                self.builtin_file if scope == "builtin" else self.path: (
                    without_local_reference_images(documents[scope]) if scope == "builtin" else documents[scope]
                )
                for scope in ("builtin", "local") if scope in scopes
            })
            return {"success": True, **self.catalog(), "name": name}

    def save_lore_theme(
        self,
        *,
        mode: str,
        name: object,
        common_tags: object = None,
        negative_tags: object = None,
        places: object = None,
        merge_common_tags: bool = True,
        overwrite_existing_places: bool = False,
        builtin_unlock_token: object = None,
    ) -> dict:
        """Atomically create a new theme or merge lore-generated places into an existing theme."""
        if mode not in {"create", "merge"}:
            raise ValueError("저장 방식을 확인하세요. (새 테마 만들기 또는 기존 테마에 추가)")
        clean_name = _validate_folder_segment(name, "테마 이름")
        incoming = normalize_bg_theme(
            {
                "group": "",
                "common_tags": common_tags if common_tags is not None else [],
                "negative_tags": negative_tags if negative_tags is not None else [],
                "lora_enabled": False,
                "lora_list": [],
                "places": places if places is not None else {},
            },
            strict=True,
        )
        if not incoming["places"]:
            raise ValueError("저장할 장소를 하나 이상 선택하세요.")

        def _merge_tags(existing: list[str], additions: list[str]) -> list[str]:
            seen = {tag.casefold() for tag in existing}
            result = list(existing)
            for tag in additions:
                key = tag.casefold()
                if key not in seen:
                    seen.add(key)
                    result.append(tag)
            return result

        with settings_lock(self.path):
            catalog = self.catalog()
            local = self._read_file(self.path)
            builtin = self._read_file(self.builtin_file)
            builtin["version"] = 1
            documents = {"local": local, "builtin": builtin}
            added_places: list[str] = []
            updated_places: list[str] = []
            skipped_places: list[str] = []

            if mode == "create":
                if clean_name in catalog["themes"]:
                    raise ValueError(f"이미 존재하는 테마입니다: {clean_name}")
                local["themes"][clean_name] = incoming
                added_places = list(incoming["places"].keys())
                target_scope = "local"
                total_places = len(incoming["places"])
                created_theme = True
            else:
                if clean_name not in catalog["themes"]:
                    raise ValueError("대상 테마를 찾을 수 없습니다. 목록을 새로고침하세요.")
                target_scope = catalog["scopes"][clean_name]
                if target_scope == "builtin":
                    self.lock.require({"builtin_unlock_token": builtin_unlock_token})
                themes = documents[target_scope]["themes"]
                merged = normalize_bg_theme(copy.deepcopy(themes[clean_name]), strict=False)
                if merge_common_tags:
                    merged["common_tags"] = _merge_tags(merged["common_tags"], incoming["common_tags"])
                    merged["negative_tags"] = _merge_tags(merged["negative_tags"], incoming["negative_tags"])
                for place_name, place_data in incoming["places"].items():
                    if place_name in merged["places"]:
                        if overwrite_existing_places:
                            existing_place = merged["places"][place_name]
                            merged["places"][place_name] = {
                                **copy.deepcopy(place_data),
                                "i2i_enabled": existing_place.get("i2i_enabled", False),
                                "i2i_denoise": existing_place.get("i2i_denoise", BG_I2I_DEFAULT_DENOISE),
                                "i2i_images": copy.deepcopy(existing_place.get("i2i_images") or []),
                            }
                            updated_places.append(place_name)
                        else:
                            skipped_places.append(place_name)
                    else:
                        merged["places"][place_name] = copy.deepcopy(place_data)
                        added_places.append(place_name)
                if not added_places and not updated_places:
                    raise ValueError("선택한 장소가 모두 대상 테마에 이미 있습니다. 장소 이름을 바꾸거나 덮어쓰기를 켜세요.")
                themes[clean_name] = normalize_bg_theme(merged, strict=True)
                total_places = len(themes[clean_name]["places"])
                created_theme = False

            target_path = self.builtin_file if target_scope == "builtin" else self.path
            target_doc = (
                without_local_reference_images(documents[target_scope])
                if target_scope == "builtin"
                else documents[target_scope]
            )
            write_scope_documents({target_path: target_doc})
            return {
                "success": True,
                "theme_name": clean_name,
                "created_theme": created_theme,
                "added_places": added_places,
                "updated_places": updated_places,
                "skipped_places": skipped_places,
                "total_places": total_places,
            }

    def attach_place_i2i_image(
        self,
        theme_name: str,
        place_name: str,
        image_entry: dict,
        *,
        enable_i2i: bool = True,
        denoise: float | None = None,
    ) -> dict:
        """Atomically register an i2i reference image on a local theme place."""
        clean_theme = _validate_folder_segment(theme_name, "테마 이름")
        clean_place = _validate_folder_segment(place_name, "장소 이름")
        normalized_images = _normalize_i2i_images(
            [image_entry], f"장소 '{clean_place}' i2i 기준 이미지", strict=True
        )
        if not normalized_images:
            raise ValueError(f"장소 '{clean_place}' i2i 기준 이미지를 확인하세요.")
        normalized_denoise = (
            _normalize_i2i_denoise(denoise, strict=True) if denoise is not None else None
        )

        with settings_lock(self.path):
            catalog = self.catalog()
            if clean_theme not in catalog["themes"]:
                raise ValueError(f"테마를 찾을 수 없습니다: {clean_theme}")
            if catalog["scopes"].get(clean_theme) == "builtin":
                raise ValueError("배포 테마에는 로컬 I2I 기준 이미지를 저장할 수 없습니다.")
            local = self._read_file(self.path)
            theme = normalize_bg_theme(copy.deepcopy(local["themes"][clean_theme]), strict=False)
            if clean_place not in theme["places"]:
                raise ValueError(f"장소를 찾을 수 없습니다: {clean_theme} / {clean_place}")
            place = theme["places"][clean_place]
            place["i2i_enabled"] = bool(enable_i2i)
            place["i2i_images"] = normalized_images
            if normalized_denoise is not None:
                place["i2i_denoise"] = normalized_denoise
            local["themes"][clean_theme] = normalize_bg_theme(theme, strict=True)
            write_scope_documents({self.path: local})
            return {
                "success": True,
                "theme_name": clean_theme,
                "place_name": clean_place,
                "i2i_enabled": place["i2i_enabled"],
                "i2i_denoise": place["i2i_denoise"],
                "i2i_images": copy.deepcopy(place["i2i_images"]),
            }

    def bulk_update_theme(
        self,
        theme_name: str,
        *,
        places: object = None,
        add_times: object = None,
        remove_times: object = None,
        i2i_mode: str = "none",
        i2i_denoise: object = BG_I2I_DEFAULT_DENOISE,
        i2i_place_images: dict[str, dict | None] | None = None,
        builtin_unlock_token: object = None,
    ) -> dict:
        """Atomically add/remove time slots and/or link i2i reference images across places."""
        clean_theme = _validate_folder_segment(theme_name, "테마 이름")
        if i2i_mode not in {"none", "missing_only", "overwrite", "denoise_only"}:
            raise ValueError("I2I 일괄 연결 방식을 확인하세요.")

        normalized_adds: list[tuple[str, dict]] = []
        seen_add_keys: set[str] = set()
        if add_times is not None:
            if not isinstance(add_times, list):
                raise ValueError("추가할 시간대 목록 형식을 확인하세요.")
            for entry in add_times:
                if not isinstance(entry, dict):
                    raise ValueError("추가할 시간대 항목 형식을 확인하세요.")
                time_name = _validate_folder_segment(entry.get("name"), "시간대 이름")
                time_key = time_name.casefold()
                if time_key in seen_add_keys:
                    raise ValueError(f"추가할 시간대 이름이 중복되었습니다: {time_name}")
                seen_add_keys.add(time_key)
                normalized_adds.append((
                    time_name,
                    {
                        "tags": _normalize_tag_list(entry.get("tags"), f"시간대 '{time_name}' 태그", strict=True),
                        "natural_language": _normalize_text(
                            entry.get("natural_language"), f"시간대 '{time_name}' 자연어", strict=True
                        ).strip(),
                    },
                ))

        normalized_removes: list[str] = []
        seen_remove_keys: set[str] = set()
        if remove_times is not None:
            if not isinstance(remove_times, list):
                raise ValueError("제거할 시간대 목록 형식을 확인하세요.")
            for item in remove_times:
                time_name = _validate_folder_segment(item, "제거할 시간대 이름")
                time_key = time_name.casefold()
                if time_key in seen_add_keys:
                    raise ValueError(f"추가와 제거 목록에 같은 시간대가 있습니다: {time_name}")
                if time_key not in seen_remove_keys:
                    seen_remove_keys.add(time_key)
                    normalized_removes.append(time_name)

        if not normalized_adds and not normalized_removes and i2i_mode == "none":
            raise ValueError("적용할 시간대 또는 I2I 변경 사항을 선택하세요.")

        normalized_denoise = (
            _normalize_i2i_denoise(i2i_denoise, strict=True)
            if i2i_mode != "none"
            else BG_I2I_DEFAULT_DENOISE
        )

        requested_places: list[str] | None = None
        if places is not None:
            if not isinstance(places, list) or not places:
                raise ValueError("적용할 장소를 하나 이상 선택하세요.")
            requested_places = []
            seen_places: set[str] = set()
            for item in places:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("장소 이름을 확인하세요.")
                clean_place = item.strip()
                if clean_place not in seen_places:
                    seen_places.add(clean_place)
                    requested_places.append(clean_place)

        normalized_i2i_entries: dict[str, list[dict] | None] = {}
        if i2i_mode in {"missing_only", "overwrite"}:
            if not isinstance(i2i_place_images, dict):
                raise ValueError("I2I 기준 이미지 정보를 확인하세요.")
            for place_name, entry in i2i_place_images.items():
                clean_place = _validate_folder_segment(place_name, "장소 이름")
                if entry is None:
                    normalized_i2i_entries[clean_place] = None
                else:
                    images = _normalize_i2i_images(
                        [entry], f"장소 '{clean_place}' i2i 기준 이미지", strict=True
                    )
                    normalized_i2i_entries[clean_place] = images if images else None

        with settings_lock(self.path):
            catalog = self.catalog()
            if clean_theme not in catalog["themes"]:
                raise ValueError("테마를 찾을 수 없습니다. 목록을 새로고침하세요.")
            scope = catalog["scopes"][clean_theme]
            if scope == "builtin":
                if i2i_mode != "none":
                    raise ValueError(
                        "배포 테마에는 로컬 I2I 기준 이미지를 저장할 수 없습니다. 사용자 테마로 복제하거나 그룹을 변경해 주세요."
                    )
                self.lock.require({"builtin_unlock_token": builtin_unlock_token})

            local = self._read_file(self.path)
            builtin = self._read_file(self.builtin_file)
            builtin["version"] = 1
            documents = {"local": local, "builtin": builtin}
            themes = documents[scope]["themes"]
            theme = normalize_bg_theme(copy.deepcopy(themes[clean_theme]), strict=False)

            if requested_places is None:
                target_places = list(theme["places"].keys())
            else:
                missing = [place_name for place_name in requested_places if place_name not in theme["places"]]
                if missing:
                    raise ValueError(f"장소를 찾을 수 없습니다: {', '.join(missing[:3])}")
                target_places = requested_places

            if not target_places:
                raise ValueError("적용할 장소가 없습니다. 먼저 장소를 추가하세요.")

            time_updated_places: list[str] = []
            added_combinations: list[dict[str, str]] = []
            removed_combinations: list[dict[str, str]] = []
            skipped_add_count = 0

            linked_places: list[str] = []
            overwritten_places: list[str] = []
            denoise_updated_places: list[str] = []
            i2i_skipped_places: list[str] = []
            i2i_missing_places: list[str] = []
            changed = False

            for place_name in target_places:
                place = theme["places"][place_name]
                place_time_changed = False

                if seen_remove_keys:
                    keys_to_delete = [
                        existing for existing in list(place["times"].keys())
                        if existing.casefold() in seen_remove_keys
                    ]
                    for existing_key in keys_to_delete:
                        del place["times"][existing_key]
                        removed_combinations.append({"place": place_name, "time": existing_key})
                        place_time_changed = True
                        changed = True

                if normalized_adds:
                    existing_keys = {existing.casefold() for existing in place["times"]}
                    for time_name, time_payload in normalized_adds:
                        time_key = time_name.casefold()
                        if time_key in existing_keys:
                            skipped_add_count += 1
                            continue
                        place["times"][time_name] = copy.deepcopy(time_payload)
                        existing_keys.add(time_key)
                        added_combinations.append({"place": place_name, "time": time_name})
                        place_time_changed = True
                        changed = True

                if place_time_changed:
                    time_updated_places.append(place_name)

                if i2i_mode != "none":
                    existing_images = place.get("i2i_images") or []
                    existing_valid = bool(
                        place.get("i2i_enabled")
                        and existing_images
                        and os.path.isfile(str(existing_images[0].get("local_path") or ""))
                    )
                    resolved_images = normalized_i2i_entries.get(place_name)
                    if i2i_mode == "denoise_only":
                        if existing_valid:
                            if place.get("i2i_denoise") != normalized_denoise:
                                place["i2i_denoise"] = normalized_denoise
                                denoise_updated_places.append(place_name)
                                changed = True
                            else:
                                i2i_skipped_places.append(place_name)
                        else:
                            i2i_skipped_places.append(place_name)
                    elif existing_valid:
                        if i2i_mode == "overwrite":
                            if resolved_images:
                                place["i2i_enabled"] = True
                                place["i2i_images"] = copy.deepcopy(resolved_images)
                                place["i2i_denoise"] = normalized_denoise
                                overwritten_places.append(place_name)
                                changed = True
                            else:
                                i2i_missing_places.append(place_name)
                        else:
                            i2i_skipped_places.append(place_name)
                    else:
                        if resolved_images:
                            place["i2i_enabled"] = True
                            place["i2i_images"] = copy.deepcopy(resolved_images)
                            place["i2i_denoise"] = normalized_denoise
                            linked_places.append(place_name)
                            changed = True
                        else:
                            i2i_missing_places.append(place_name)

            normalized = normalize_bg_theme(theme, strict=True)
            if changed:
                themes[clean_theme] = normalized
                target_path = self.builtin_file if scope == "builtin" else self.path
                target_doc = (
                    without_local_reference_images(documents[scope])
                    if scope == "builtin"
                    else documents[scope]
                )
                write_scope_documents({target_path: target_doc})

            return {
                "success": True,
                "theme_name": clean_theme,
                "target_place_count": len(target_places),
                "times": {
                    "added_count": len(added_combinations),
                    "removed_count": len(removed_combinations),
                    "skipped_add_count": skipped_add_count,
                    "updated_places": time_updated_places,
                    "added_combinations": added_combinations,
                    "removed_combinations": removed_combinations,
                },
                "i2i": {
                    "mode": i2i_mode,
                    "denoise": normalized_denoise,
                    "linked_places": linked_places,
                    "overwritten_places": overwritten_places,
                    "denoise_updated_places": denoise_updated_places,
                    "skipped_places": i2i_skipped_places,
                    "missing_places": i2i_missing_places,
                    "linked_count": len(linked_places),
                    "overwritten_count": len(overwritten_places),
                    "denoise_updated_count": len(denoise_updated_places),
                    "skipped_count": len(i2i_skipped_places),
                    "missing_count": len(i2i_missing_places),
                },
                "theme": normalized,
            }


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

    def delete_place(self, theme: str, place: str) -> dict:
        """Delete a place folder and all time-of-day subfolders/images inside it."""
        try:
            place_dir = self._asset_delete_path(theme, place)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        if not os.path.isdir(place_dir):
            return {"success": False, "error": "존재하지 않는 장소 폴더입니다."}
        shutil.rmtree(place_dir)
        self._log("place_deleted", {"theme": theme, "place": place})
        return {"success": True}

    def resolve_place_i2i_image(self, theme_name: str, place_name: str, time_name: str) -> dict | None:
        """Pick the representative (or latest static) generated image for a theme/place/time folder."""
        listing = self.list_images(theme_name, place_name, time_name)
        images = listing.get("images") if isinstance(listing, dict) else None
        if not isinstance(images, list) or not images:
            return None

        candidates = []
        for item in images:
            if not isinstance(item, dict):
                continue
            if item.get("is_video_animation"):
                continue
            filename = str(item.get("filename") or "").strip()
            local_path = str(item.get("local_path") or "").strip()
            if not filename or not local_path or not os.path.isfile(local_path):
                continue
            if os.path.splitext(filename)[1].lower() not in BG_I2I_STATIC_EXTENSIONS:
                continue
            candidates.append(item)

        if not candidates:
            return None

        chosen = next((item for item in candidates if item.get("is_representative")), None)
        if chosen is None:
            def _sort_key(item: dict) -> tuple[float, float, str]:
                raw_created = item.get("created_at")
                try:
                    created = float(raw_created) if raw_created is not None else 0.0
                except (TypeError, ValueError):
                    created = 0.0
                try:
                    mtime = float(item.get("mtime") or 0.0)
                except (TypeError, ValueError):
                    mtime = 0.0
                return (created, mtime, str(item.get("filename") or ""))

            chosen = max(candidates, key=_sort_key)

        return build_bg_i2i_image_entry(
            theme_name,
            place_name,
            time_name,
            str(chosen["filename"]),
            str(chosen["local_path"]),
        )


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
