"""감정 삽화 수동 배경 프리셋 저장소.

배포 프리셋은 설치본의 ``presets/emotion_background_presets.json``에,
사용자 프리셋은 설정 파일 옆의 ``emotion_background_presets.json``에 둔다.
봇 데이터와는 분리된 공방 공통 저장소이며, 두 파일 모두 같은 문서 형식을 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading

from asset_preset_scope import AssetPresetLock
from json_store import write_json_atomic
from preset_locations import PRESET_LOCATIONS


SCOPES = ("builtin", "local")
FIELDS = ("text", "bg_css", "fg_css", "preview_name", "preview_url")
LIMITS = {"text": 20000, "bg_css": 20000, "fg_css": 20000, "preview_name": 500, "preview_url": 1000}


def _validate_name(value) -> str:
    name = value.strip() if isinstance(value, str) else ""
    if not name or len(name) > 100 or not name.isprintable():
        raise ValueError("프리셋 이름은 1~100자로 입력하세요.")
    return name


def _validate_scope(value) -> str:
    if value not in SCOPES:
        raise ValueError("프리셋 구분을 확인하세요.")
    return value


def _validate_values(values) -> dict:
    if not isinstance(values, dict):
        raise ValueError("프리셋 값을 확인하세요.")
    preset = {}
    for key in FIELDS:
        value = values.get(key, "")
        if not isinstance(value, str) or len(value) > LIMITS[key]:
            raise ValueError(f"프리셋 값을 확인하세요: {key}")
        preset[key] = value
    if preset["preview_url"] and not preset["preview_url"].startswith("/api/"):
        raise ValueError("미리보기 이미지 경로를 확인하세요.")
    return preset


class EmotionBackgroundPresets:
    def __init__(self, config_file: str | Path, builtin_file: str | Path | None = None):
        self.local_file = PRESET_LOCATIONS["emotion_background"].local_path(config_file=config_file)
        self.builtin_file = (
            Path(builtin_file) if builtin_file
            else PRESET_LOCATIONS["emotion_background"].builtin_path()
        )
        self._lock = threading.RLock()
        # 배포 프리셋은 세션 잠금을 해제한 경우에만 편집할 수 있다.
        self.scope_lock = AssetPresetLock()

    @staticmethod
    def _read_file(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if (
            not isinstance(document, dict)
            or document.get("version") != 1
            or not isinstance(document.get("presets"), list)
        ):
            raise ValueError(f"{path.name} 파일 형식을 확인하세요.")
        presets = []
        for item in document["presets"]:
            if not isinstance(item, dict):
                raise ValueError(f"{path.name} 내용을 확인하세요.")
            presets.append({
                "name": _validate_name(item.get("name")),
                **_validate_values(item),
            })
        return presets

    @staticmethod
    def _write_file(path: Path, presets: list[dict]) -> None:
        # 빈 배포 파일을 소스 트리에 새로 만들지 않는다.
        if not presets and not path.is_file():
            return
        document = {"version": 1, "presets": presets}
        if path.is_file():
            try:
                if json.loads(path.read_text(encoding="utf-8-sig")) == document:
                    return
            except (OSError, ValueError):
                pass
        write_json_atomic(path, document)

    def _load(self) -> list[dict]:
        builtin = [{"scope": "builtin", **preset} for preset in self._read_file(self.builtin_file)]
        local = [{"scope": "local", **preset} for preset in self._read_file(self.local_file)]
        return builtin + local

    def _commit(self, presets: list[dict]) -> list[dict]:
        groups = {
            scope: [
                {key: value for key, value in preset.items() if key != "scope"}
                for preset in presets if preset["scope"] == scope
            ]
            for scope in SCOPES
        }
        self._write_file(self.builtin_file, groups["builtin"])
        self._write_file(self.local_file, groups["local"])
        return presets

    @staticmethod
    def _insert(presets: list[dict], preset: dict) -> list[dict]:
        # 같은 그룹의 맨 앞에 넣어 최근 저장한 프리셋이 먼저 보이게 한다.
        index = next(
            (index for index, item in enumerate(presets) if item["scope"] == preset["scope"]),
            len(presets),
        )
        return [*presets[:index], preset, *presets[index:]]

    @staticmethod
    def _find(presets: list[dict], name) -> dict:
        target = next((item for item in presets if item["name"] == name), None)
        if target is None:
            raise KeyError("프리셋을 찾을 수 없습니다.")
        return target

    def list(self) -> list[dict]:
        with self._lock:
            return self._load()

    def change_lock(self, body: dict) -> dict:
        """배포 잠금을 세션 토큰으로 풀거나 다시 잠근다."""
        with self._lock:
            return {"presets": self._load(), **self.scope_lock.change(body)}

    def _require_builtin(self, token) -> None:
        self.scope_lock.require({"builtin_unlock_token": token})

    def save(self, scope, name, values, token=None) -> list[dict]:
        scope = _validate_scope(scope)
        if scope == "builtin":
            self._require_builtin(token)
        preset = {"name": _validate_name(name), "scope": scope, **_validate_values(values)}
        with self._lock:
            presets = self._load()
            if any(item["name"] == preset["name"] for item in presets):
                raise ValueError("같은 이름의 프리셋이 있습니다.")
            return self._commit(self._insert(presets, preset))

    def duplicate(self, name, new_name, scope, token=None) -> list[dict]:
        scope = _validate_scope(scope)
        if scope == "builtin":
            self._require_builtin(token)
        with self._lock:
            presets = self._load()
            source = self._find(presets, name)
            preset = {**source, "name": _validate_name(new_name), "scope": scope}
            if any(item["name"] == preset["name"] for item in presets):
                raise ValueError("같은 이름의 프리셋이 있습니다.")
            return self._commit(self._insert(presets, preset))

    def rename(self, name, new_name, token=None) -> list[dict]:
        new_name = _validate_name(new_name)
        with self._lock:
            presets = self._load()
            target = self._find(presets, name)
            if target["scope"] == "builtin":
                self._require_builtin(token)
            if new_name != target["name"] and any(item["name"] == new_name for item in presets):
                raise ValueError("같은 이름의 프리셋이 있습니다.")
            target["name"] = new_name
            return self._commit(presets)

    def delete(self, name, token=None) -> list[dict]:
        with self._lock:
            presets = self._load()
            target = self._find(presets, name)
            if target["scope"] == "builtin":
                self._require_builtin(token)
            return self._commit([item for item in presets if item["name"] != name])

    def move(self, name, scope, token=None) -> list[dict]:
        scope = _validate_scope(scope)
        with self._lock:
            presets = self._load()
            target = self._find(presets, name)
            if target["scope"] == "builtin" or scope == "builtin":
                self._require_builtin(token)
            if target["scope"] == scope:
                return presets
            target["scope"] = scope
            presets = [item for item in presets if item["name"] != name]
            return self._commit(self._insert(presets, target))

    def migrate(self, values) -> None:
        """이전 버전이 config.json에 저장한 프리셋을 사용자 프리셋으로 옮긴다."""
        if not isinstance(values, list) or not values:
            return
        with self._lock:
            if self.local_file.exists():
                return
            presets = []
            for item in values:
                if not isinstance(item, dict):
                    continue
                try:
                    preset = {
                        "name": _validate_name(item.get("name")),
                        "scope": "local",
                        **_validate_values(item),
                    }
                except ValueError:
                    continue
                if any(existing["name"] == preset["name"] for existing in presets):
                    continue
                presets.append(preset)
            if presets:
                self._commit(presets)
