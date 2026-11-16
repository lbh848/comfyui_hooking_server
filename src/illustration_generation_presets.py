"""Reusable drawing settings, stored separately for each generation mode."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading

from aiohttp import web

from asset_preset_scope import AssetPresetLock, write_scope_documents
from json_store import write_json_atomic
from manga_page_sizes import parse_manga_page_sizes
from configuration import settings_lock
from preset_locations import PRESET_LOCATIONS


MODES = ("illustration", "character_auto", "background")
SCOPES = ("builtin", "local")
SIZE_FIELDS = ("img_w", "img_h", "seed")


class GenerationPresetConflict(ValueError):
    pass


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError("지원하지 않는 그리기 설정 모드입니다.")
    return mode


def normalize_settings(value, mode):
    # Load only when needed: importing the modes package starts engine services.
    from modes.bot_mode import DEFAULT_ILLUST_SETTINGS

    if not isinstance(value, dict) or not isinstance(value.get("solo"), dict):
        raise ValueError("그리기 설정 형식을 확인하세요.")
    solo = {key: value["solo"].get(key, default) for key, default in DEFAULT_ILLUST_SETTINGS.items()}
    group = solo if mode == "character_auto" else value.get("group")
    if not isinstance(group, dict) or any(key not in group for key in SIZE_FIELDS):
        raise ValueError("2인 이상 이미지 크기·시드를 확인하세요.")
    group = {key: group[key] for key in SIZE_FIELDS}
    enums = {
        "snapref_preprocess": {"none", "lineart", "lineart_color", "grayscale"},
        "snapref_secondary_reference": {"none", "face_only", "face", "representative", "stage"},
        "snapref_primary_reference": {"", "none", "face_only", "face", "representative", "llm"},
        "snapref_detailer_reference": {"", "none", "face_only", "face", "representative", "llm"},
        "snapref_mode": {"char", "cs"},
        "snapref_strength_scale": {"linear"},
    }
    ranges = {
        "img_w": (256, 4096), "img_h": (256, 4096), "seed": (-1, 9007199254740991),
        "manga_scale": (0.2, 2.0), "manga_seed": (-1, 9007199254740991),
        "hrf_size": (1, 3), "snapref_strength": (0, 2), "snapref_fidelity": (0, 1),
        "snapref_megapixels": (.25, 2), "face_id_str": (0, 1),
    }
    for settings in (solo, group):
        # Keep the manga resolution selection canonical and non-empty; the
        # model must always have at least one page size to choose from.
        if "manga_page_sizes" in settings:
            selected_sizes = parse_manga_page_sizes(settings["manga_page_sizes"])
            if not selected_sizes:
                raise ValueError("그리기 설정 값을 확인하세요: manga_page_sizes")
            settings["manga_page_sizes"] = ",".join(
                f"{width}x{height}" for width, height in selected_sizes)
        for key, item in settings.items():
            default = DEFAULT_ILLUST_SETTINGS[key]
            valid = True
            if key == "face_lora_upscale_size":
                valid = item == "" or type(item) is int and item >= 0
            elif type(default) is bool:
                valid = type(item) is bool
            elif type(default) in (int, float):
                valid = type(item) in (int, float) and math.isfinite(item)
                if valid and type(default) is int:
                    valid = item == int(item)
                if valid and key in ranges:
                    valid = ranges[key][0] <= item <= ranges[key][1]
            else:
                valid = isinstance(item, str) and len(item) <= 500
                if valid and key in enums:
                    valid = item in enums[key]
            if not valid:
                raise ValueError(f"그리기 설정 값을 확인하세요: {key}")
    return {"solo": solo, "group": group}


def _empty_document():
    return {"version": 1, "modes": {mode: {} for mode in MODES}}


class IllustrationGenerationPresets:
    def __init__(self, config_file: str | Path, builtin_file: str | Path | None = None, *, workshop: str = "illustration"):
        if workshop not in {"illustration", "emotion"}:
            raise ValueError("그리기 설정 공방을 확인하세요.")
        self.workshop = workshop
        location = PRESET_LOCATIONS[f"{workshop}_generation"]
        self.path = location.local_path(config_file=config_file)
        self.builtin_file = Path(builtin_file) if builtin_file else location.builtin_path()
        self.guard = threading.RLock()
        self.scope_lock = AssetPresetLock()
        if workshop == "emotion":
            # Both workshops previously used the illustration file. Preserve
            # those user presets once, then keep all future edits independent.
            # An empty file also marks a new installation as already split.
            legacy = PRESET_LOCATIONS["illustration_generation"].local_path(config_file=config_file)
            with settings_lock(legacy), settings_lock(self.path):
                if not self.path.exists():
                    write_json_atomic(self.path, self._read_file(legacy, "그리기 설정 프리셋"))

    @staticmethod
    def _validate_name(value):
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100 or not value.strip().isprintable():
            raise ValueError("프리셋 이름은 1~100자로 입력하세요.")
        return value.strip()

    @staticmethod
    def _read_file(path, label):
        if not path.exists():
            return _empty_document()
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if (not isinstance(document, dict) or document.get("version") != 1
                or not isinstance(document.get("modes"), dict)):
            raise ValueError(f"{label} 파일 형식을 확인하세요.")
        # Older files predate newer generation modes; treat a missing mode as empty
        # so adding a mode does not invalidate every existing preset document.
        for mode in MODES:
            document["modes"].setdefault(mode, {})
        if any(not isinstance(document["modes"][mode], dict) for mode in MODES):
            raise ValueError(f"{label} 파일 형식을 확인하세요.")
        for presets in document["modes"].values():
            if any(
                not isinstance(value, dict) or not isinstance(value.get("solo"), dict)
                or not isinstance(value.get("group"), dict) for value in presets.values()
            ):
                raise ValueError(f"{label} 내용을 확인하세요.")
        return document

    def _read(self):
        return self._read_file(self.path, "그리기 설정 프리셋")

    def _builtin(self):
        return self._read_file(self.builtin_file, "배포 그리기 설정 프리셋")

    @staticmethod
    def _entries(document, mode, scope):
        return [
            {"name": name, "scope": scope,
             "settings": normalize_settings(settings, mode) if scope == "builtin" else deepcopy(settings)}
            for name, settings in sorted(document["modes"][mode].items())
        ]

    def _snapshot(self, document, mode, builtin=None):
        builtin = self._builtin() if builtin is None else builtin
        presets = [*self._entries(builtin, mode, "builtin"), *self._entries(document, mode, "local")]
        revision = hashlib.sha256(json.dumps([self.workshop, presets], ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return {"workshop": self.workshop, "generation_mode": mode, "revision": revision, "presets": presets}

    def snapshot(self, mode):
        validate_mode(mode)
        with self.guard:
            return self._snapshot(self._read(), mode)

    def mutate(self, body):
        if not isinstance(body, dict):
            raise ValueError("그리기 설정 프리셋 요청 형식을 확인하세요.")
        mode = validate_mode(body.get("generation_mode"))
        action = body.get("action")
        if action in ("unlock", "lock"):
            with self.guard:
                return {**self._snapshot(self._read(), mode), **self.scope_lock.change(body)}
        if action not in ("create", "update", "delete", "scope"):
            raise ValueError("지원하지 않는 프리셋 요청입니다.")
        name = self._validate_name(body.get("name"))
        scope = body.get("scope", "local")
        if scope not in SCOPES:
            raise ValueError("배포 또는 사용자 프리셋을 선택하세요.")
        with self.guard:
            local = self._read()
            builtin = self._builtin()
            if body.get("revision") != self._snapshot(local, mode, builtin)["revision"]:
                raise GenerationPresetConflict("다른 창에서 프리셋이 변경되었습니다. 목록을 새로고침한 뒤 다시 시도하세요.")
            documents = {"builtin": builtin, "local": local}
            if action == "create":
                presets = local["modes"][mode]
                if name in presets:
                    raise ValueError("같은 이름의 프리셋이 있습니다. 다른 이름으로 저장하거나 덮어쓰기를 선택하세요.")
                presets[name] = normalize_settings(body.get("settings"), mode)
                changed = {"local"}
            else:
                if scope == "builtin":
                    self.scope_lock.require(body)
                presets = documents[scope]["modes"][mode]
                if name not in presets:
                    raise ValueError("선택한 프리셋을 찾을 수 없습니다. 목록을 새로고침하세요.")
                changed = {scope}
                if action == "update":
                    presets[name] = normalize_settings(body.get("settings"), mode)
                elif action == "delete":
                    del presets[name]
                else:
                    target = body.get("target_scope")
                    if target not in SCOPES:
                        raise ValueError("이동할 배포 또는 사용자 프리셋 그룹을 선택하세요.")
                    if target != scope:
                        if target == "builtin":
                            self.scope_lock.require(body)
                        target_presets = documents[target]["modes"][mode]
                        if name in target_presets:
                            raise ValueError("대상 그룹에 같은 이름의 프리셋이 있습니다.")
                        target_presets[name] = presets.pop(name)
                        changed.add(target)
            if changed == {"local"}:
                write_json_atomic(self.path, local)
            else:
                write_scope_documents({
                    self.builtin_file if item == "builtin" else self.path: documents[item] for item in changed
                })
            return self._snapshot(documents["local"], mode, documents["builtin"])


def register_generation_presets(app: web.Application, config_file: str | Path, *, builtin_files: dict[str, Path] | None = None):
    stores = {workshop: IllustrationGenerationPresets(config_file, (builtin_files or {}).get(workshop), workshop=workshop)
              for workshop in ("illustration", "emotion")}

    async def handle(request):
        headers = {"Cache-Control": "no-store"}
        try:
            scope = request.query.get("scope", "illust")
            if scope not in {"illust", "illustration", "emotion"}:
                raise ValueError("그리기 설정 공방을 확인하세요.")
            store = stores["emotion" if scope == "emotion" else "illustration"]
            if request.method == "GET":
                result = await asyncio.to_thread(store.snapshot, request.query.get("generation_mode"))
            else:
                body = await request.json()
                result = await asyncio.to_thread(store.mutate, body)
            return web.json_response(result, headers=headers)
        except GenerationPresetConflict as exc:
            return web.json_response({"error": str(exc)}, status=409, headers=headers)
        except (ValueError, TypeError, OverflowError) as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)
        except OSError:
            return web.json_response({"error": "프리셋 파일을 읽거나 저장할 수 없습니다."}, status=500, headers=headers)

    app.router.add_get("/api/illustration/generation-presets", handle)
    app.router.add_post("/api/illustration/generation-presets", handle)
