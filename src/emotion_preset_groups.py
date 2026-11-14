"""Named emotion preset groups for the emotion illustration workshop.

A group is a named list of {preset, name} rows for one scene kind (SFW or
NSFW). Distribution groups ship with the installation under src/presets;
user groups live beside the other editable preset data. Distributed groups
are read-only until the settings UI unlocks them with a warning.
"""
from __future__ import annotations

import copy
import json
import os
import secrets
import tempfile
from pathlib import Path

from app_paths import data_root
from preset_locations import PRESET_LOCATIONS

KINDS = ("sfw", "nsfw")
SCOPES = ("builtin", "local")
MAX_GROUPS = 200
MAX_ROWS = 500
PRESET_MAX_CHARS = 200
NAME_MAX_CHARS = 120
GROUP_NAME_MAX_CHARS = 80


def _empty_document() -> dict:
    return {"version": 1, "groups": {kind: [] for kind in KINDS}}


def _validate_row(row) -> dict:
    if not isinstance(row, dict) or set(row) - {"preset", "name"}:
        raise ValueError("감정 프리셋 행 형식을 확인하세요.")
    preset, name = row.get("preset", ""), row.get("name", "")
    if not isinstance(preset, str) or not isinstance(name, str):
        raise ValueError("감정 프리셋 이름 형식을 확인하세요.")
    if len(preset) > PRESET_MAX_CHARS or len(name) > NAME_MAX_CHARS:
        raise ValueError("감정 프리셋 이름이 너무 깁니다.")
    return {"preset": preset, "name": name}


def _validate_name(value) -> str:
    if not isinstance(value, str):
        raise ValueError("프리셋 그룹 이름을 입력하세요.")
    name = value.strip()
    if not name or len(name) > GROUP_NAME_MAX_CHARS:
        raise ValueError(f"프리셋 그룹 이름은 1~{GROUP_NAME_MAX_CHARS}자로 입력하세요.")
    return name


def _validate_document(document) -> dict:
    if not isinstance(document, dict) or set(document) - {"version", "groups"}:
        raise ValueError("감정 프리셋 그룹 파일 형식을 확인하세요.")
    groups = document.get("groups")
    if not isinstance(groups, dict) or set(groups) - set(KINDS):
        raise ValueError("감정 프리셋 그룹 구분을 확인하세요.")
    checked = _empty_document()
    for kind in KINDS:
        entries = groups.get(kind, [])
        if not isinstance(entries, list) or len(entries) > MAX_GROUPS:
            raise ValueError(f"{kind.upper()} 감정 프리셋 그룹 개수를 확인하세요.")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"name", "rows"}:
                raise ValueError("감정 프리셋 그룹 형식을 확인하세요.")
            name = _validate_name(entry["name"])
            if name in seen:
                raise ValueError(f"{kind.upper()} 감정 프리셋 그룹 이름이 중복되었습니다: {name}")
            seen.add(name)
            rows = entry["rows"]
            if not isinstance(rows, list) or len(rows) > MAX_ROWS:
                raise ValueError(f"감정 프리셋 행은 {MAX_ROWS}개 이하로 저장하세요.")
            checked["groups"][kind].append({"name": name, "rows": [_validate_row(row) for row in rows]})
    return checked


class EmotionPresetGroups:
    """Atomically read and write the two emotion preset group stores."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.files = {
            "builtin": PRESET_LOCATIONS["emotion_groups"].builtin_path(self.project_root),
            "local": PRESET_LOCATIONS["emotion_groups"].local_path(self.project_root),
        }
        self._unlock_tokens: set[str] = set()

    def _read(self, scope: str) -> dict:
        path = self.files[scope]
        if not path.is_file():
            return _empty_document()
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 형식을 확인하세요.") from exc
        return _validate_document(document)

    @staticmethod
    def _write(path: Path, document: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        handle, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise

    def catalog(self) -> dict:
        result = {scope: self._read(scope)["groups"] for scope in SCOPES}
        result["unlocked"] = bool(self._unlock_tokens)
        return result

    def _require_unlock(self, body: dict) -> None:
        token = body.get("builtin_unlock_token")
        if not isinstance(token, str) or token not in self._unlock_tokens:
            raise ValueError("배포 풀기로 잠금을 먼저 해제하세요.")

    @staticmethod
    def _kind(body: dict) -> str:
        kind = body.get("kind")
        if kind not in KINDS:
            raise ValueError("SFW 또는 NSFW를 선택하세요.")
        return kind

    @staticmethod
    def _scope(body: dict) -> str:
        scope = body.get("scope", "local")
        if scope not in SCOPES:
            raise ValueError("배포 또는 사용자 그룹을 선택하세요.")
        return scope

    def mutate(self, action: str, body: dict) -> dict:
        if action == "unlock":
            if body.get("confirmed") is not True:
                raise ValueError("배포 풀기 경고를 먼저 확인하세요.")
            token = secrets.token_urlsafe(32)
            self._unlock_tokens.add(token)
            return {**self.catalog(), "builtin_unlock_token": token}
        if action == "lock":
            token = body.get("builtin_unlock_token")
            if isinstance(token, str):
                self._unlock_tokens.discard(token)
            return self.catalog()

        kind = self._kind(body)
        if action == "create":
            scope = self._scope(body)
            if scope == "builtin":
                self._require_unlock(body)
            name = _validate_name(body.get("name"))
            document = self._read(scope)
            entries = document["groups"][kind]
            if any(entry["name"] == name for entry in entries):
                raise ValueError(f"같은 그룹에 '{name}' 프리셋이 이미 있습니다.")
            if len(entries) >= MAX_GROUPS:
                raise ValueError(f"감정 프리셋 그룹은 {MAX_GROUPS}개 이하로 저장하세요.")
            entries.append({"name": name, "rows": []})
            self._write(self.files[scope], document)
            return self.catalog()

        if action == "save":
            scope = self._scope(body)
            if scope == "builtin":
                self._require_unlock(body)
            name = _validate_name(body.get("name"))
            rows = body.get("rows")
            if not isinstance(rows, list) or len(rows) > MAX_ROWS:
                raise ValueError(f"감정 프리셋 행은 {MAX_ROWS}개 이하로 저장하세요.")
            checked_rows = [_validate_row(row) for row in rows]
            document = self._read(scope)
            entries = document["groups"][kind]
            group = next((entry for entry in entries if entry["name"] == name), None)
            if group is None:
                if len(entries) >= MAX_GROUPS:
                    raise ValueError(f"감정 프리셋 그룹은 {MAX_GROUPS}개 이하로 저장하세요.")
                entries.append({"name": name, "rows": checked_rows})
            else:
                group["rows"] = checked_rows
            self._write(self.files[scope], document)
            return self.catalog()

        if action in {"rename", "delete"}:
            scope = self._scope(body)
            if scope == "builtin":
                self._require_unlock(body)
            name = _validate_name(body.get("name"))
            document = self._read(scope)
            entries = document["groups"][kind]
            index = next((position for position, entry in enumerate(entries) if entry["name"] == name), None)
            if index is None:
                raise ValueError("변경할 감정 프리셋 그룹을 찾을 수 없습니다.")
            if action == "delete":
                if body.get("confirmed") is not True:
                    raise ValueError("프리셋 그룹 삭제를 먼저 확인하세요.")
                del entries[index]
            else:
                new_name = _validate_name(body.get("new_name"))
                if any(entry["name"] == new_name for position, entry in enumerate(entries) if position != index):
                    raise ValueError(f"같은 그룹에 '{new_name}' 프리셋이 이미 있습니다.")
                entries[index]["name"] = new_name
            self._write(self.files[scope], document)
            return self.catalog()

        if action == "duplicate":
            source_scope = self._scope({**body, "scope": body.get("source_scope", "local")})
            source_name = _validate_name(body.get("source_name"))
            new_name = _validate_name(body.get("new_name"))
            source = next((entry for entry in self._read(source_scope)["groups"][kind]
                           if entry["name"] == source_name), None)
            if source is None:
                raise ValueError("복제할 감정 프리셋 그룹을 찾을 수 없습니다.")
            document = self._read("local")
            entries = document["groups"][kind]
            if any(entry["name"] == new_name for entry in entries):
                raise ValueError(f"사용자 그룹에 '{new_name}' 프리셋이 이미 있습니다.")
            if len(entries) >= MAX_GROUPS:
                raise ValueError(f"감정 프리셋 그룹은 {MAX_GROUPS}개 이하로 저장하세요.")
            entries.append({"name": new_name, "rows": copy.deepcopy(source["rows"])})
            self._write(self.files["local"], document)
            return self.catalog()

        raise ValueError("지원하지 않는 감정 프리셋 그룹 작업입니다.")
