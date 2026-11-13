"""Bot-independent illustration model presets and generation snapshots."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import uuid

from illustration_json import enabled_lora_fields, workflow_json_defaults, update_workflow_json_settings
from json_store import write_json_atomic
from configuration import read_config
import workflow_profiles

CONFIG_KEYS = (
    "illustration_workflow_type",
    "illustration_workflow_source_paths",
    "chansub_workflow_type",
    "illustration_model_family",
)
# The illustration and emotion workshops keep separate preset files, so a
# preset created or selected in one never appears in or changes the other.
WORKSHOPS = ("illustration", "emotion")
_DUAL_VAE_PATTERN = re.compile(r"dual[_\s-]?vae", re.IGNORECASE)


def uses_dual_vae(preset: dict) -> bool:
    paths = (preset.get("config") or {}).get("illustration_workflow_source_paths") or {}
    return any(_DUAL_VAE_PATTERN.search(str(path)) for path in paths.values())


class PresetConflict(ValueError):
    pass


def revision(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def selected_config(config: dict) -> dict:
    normalized = workflow_profiles.normalize_workflow_config({"asset_workflow_type": "ilxl", "illustration_context_toggles": {}, **copy.deepcopy(config)})
    return {key: normalized.get(key, "anima" if key == "chansub_workflow_type" else "") for key in CONFIG_KEYS}


def read_workflow(source: str) -> tuple[dict, str]:
    if not source:
        raise ValueError("그림 모델의 워크플로우를 먼저 선택하세요.")
    content = Path(source).read_bytes()
    workflow = json.loads(content.decode("utf-8-sig"))
    if not isinstance(workflow, dict):
        raise ValueError("워크플로우 JSON 객체를 확인하세요.")
    return workflow, hashlib.sha256(content).hexdigest()


class IllustrationModelPresets:
    # Only the illustration store keeps one preset file per workshop; the asset
    # store reuses this class for its own single preset list.
    separates_workshops = True
    type_key = "illustration_workflow_type"
    paths_key = "illustration_workflow_source_paths"
    config_keys = CONFIG_KEYS
    local_profiles = workflow_profiles.ILLUSTRATION_LOCAL_PROFILES
    selected_config = staticmethod(selected_config)
    local_profile = staticmethod(workflow_profiles.illustration_local_profile)

    def validate_config(self, candidate):
        paths = candidate[self.paths_key]
        if (candidate[self.type_key] not in workflow_profiles.ILLUSTRATION_WORKFLOW_TYPES
                or candidate["chansub_workflow_type"] not in {"anima", "sdxl"}
                or str(candidate.get("illustration_model_family", "")) not in {"", *workflow_profiles.MODEL_FAMILIES}
                or not isinstance(paths, dict) or set(paths) - set(self.local_profiles)
                or not all(isinstance(path, str) for path in paths.values())):
            raise ValueError("그림 모델 설정 값을 확인하세요.")

    def __init__(self, config_file: str | Path, builtin_file: str | Path | None = None):
        self.config_file = Path(config_file)
        self.path = Path(config_file).with_name("illustration_model_presets.json")
        self.emotion_path = self.path.with_name("emotion_model_presets.json")
        self.builtin_file = Path(builtin_file) if builtin_file else Path(__file__).parent / "presets" / "illustration_model_presets.json"
        self._builtin_unlock_tokens: set[str] = set()

    def _builtin(self) -> dict:
        document = json.loads(self.builtin_file.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict) or document.get("version") != 1 or not isinstance(document.get("presets"), list):
            raise ValueError("배포 그림 모델 프리셋 파일을 확인하세요.")
        return document

    def _resolve_builtin(self, preset: dict, config: dict) -> dict:
        preset = copy.deepcopy(preset)
        paths = preset["config"][self.paths_key]
        for profile, path in paths.items():
            if path.startswith("@configured:"):
                paths[profile] = self.selected_config(config)[self.paths_key].get(path.removeprefix("@configured:"), "")
            elif path and not Path(path).is_absolute():
                paths[profile] = str((self.config_file.parent.parent / path).resolve())
        for profile, entry in preset["workflows"].items():
            entry["path"] = paths.get(profile, "")
        preset["scope"] = "builtin"
        return preset

    def resolved_config(self, config: dict) -> dict:
        """프리셋 하나의 그림 모델 계열을 확정한다.

        계열 선택값이 없는 예전 프리셋은 워크플로우 경로에서 이관하고, 저장된
        선택값이 있으면 그 값을 쓴다. 다른 공방(감정 삽화)의 설정에서 넘어온
        계열 값이 남지 않도록 프리셋 자신의 config로 항상 덮어쓴다.
        """
        config = copy.deepcopy(config)
        if self.type_key != "illustration_workflow_type":
            return config
        profile = workflow_profiles.illustration_local_profile(config.get(self.type_key))
        paths = config.get(self.paths_key)
        source = str((paths or {}).get(profile) or "") if profile and isinstance(paths, dict) else ""
        config["illustration_model_family"] = workflow_profiles.illustration_model_family(
            config.get(self.type_key), source, config.get("illustration_model_family", ""))
        return config

    def _workshop_path(self, workshop: str | None) -> Path:
        return self.emotion_path if workshop == "emotion" else self.path

    def _migrate_workshop_files(self) -> None:
        """Split the former shared preset file into one file per workshop."""
        if self.emotion_path.exists() or not self.path.exists():
            return
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("presets"), list):
            return
        emotion = [item for item in document["presets"] if uses_dual_vae(item)]
        if not emotion:
            return
        illustration = [item for item in document["presets"] if not uses_dual_vae(item)]
        stored = document.get("active_ids") if isinstance(document.get("active_ids"), dict) else {}
        emotion_ids = {item["id"] for item in emotion}
        emotion_active = stored.get("emotion")
        if emotion_active not in emotion_ids:
            emotion_active = emotion[0]["id"]
        # The illustration side may point at a shipped preset, so keep whatever
        # was selected and only substitute a local preset when nothing was.
        illustration_active = stored.get("illustration") or document.get("active_id") or ""
        if not illustration_active:
            illustration_active = illustration[0]["id"] if illustration else ""
        # Write the emotion file first so a crash cannot lose the DUAL_VAE presets.
        write_json_atomic(self.emotion_path, {"version": 1, "active_id": emotion_active,
                                              "presets": copy.deepcopy(emotion)})
        write_json_atomic(self.path, {"version": 1, "active_id": illustration_active,
                                      "presets": illustration})

    def read(self, config: dict | None = None, workshop: str | None = None) -> dict | None:
        if self.separates_workshops:
            self._migrate_workshop_files()
        path = self._workshop_path(workshop)
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(document, dict) or document.get("version") != 1
                    or not isinstance(document.get("presets"), list)):
                raise ValueError("그림 모델 프리셋 파일을 확인하세요.")
        else:
            document = {"version": 1, "active_id": "", "presets": []}
        # Bind shipped workflow references to this installation's configured
        # files, never to the preceding preset's temporary runtime overlay.
        base = read_config(self.config_file) if self.config_file.exists() else config or {}
        # Shipped presets belong to the illustration workshop; the emotion
        # workshop keeps only the presets created for it.
        builtin = ([self._resolve_builtin(item, base) for item in self._builtin()["presets"]]
                   if workshop != "emotion" else [])
        local = [{**item, "scope": "local"} for item in document["presets"]]
        document["presets"] = [*builtin, *local]
        if not any(item["id"] == document["active_id"] for item in document["presets"]):
            document["active_id"] = document["presets"][0]["id"] if document["presets"] else ""
        return document

    def save(self, document: dict, workshop: str | None = None) -> None:
        # User presets and selection survive updates. Shipped presets live in
        # src/presets and follow the existing replace-without-backup policy.
        local = [{key: value for key, value in item.items() if key != "scope"}
                 for item in document["presets"] if item.get("scope", "local") == "local"]
        write_json_atomic(self._workshop_path(workshop),
                          {"version": 1, "active_id": document["active_id"], "presets": local})

    def _save_builtin(self, preset: dict | None, identity: str) -> None:
        document = self._builtin()
        original = next((item for item in document["presets"] if item["id"] == identity), None)
        if preset is None:
            if original:
                document["presets"].remove(original)
        else:
            saved = copy.deepcopy(preset)
            saved.pop("scope", None)
            base = read_config(self.config_file) if self.config_file.exists() else {}
            resolved = self._resolve_builtin(original, base) if original else None
            for profile, path in saved["config"][self.paths_key].items():
                if resolved and path == resolved["config"][self.paths_key].get(profile):
                    saved["config"][self.paths_key][profile] = original["config"][self.paths_key][profile]
                elif path and Path(path).is_absolute():
                    try:
                        saved["config"][self.paths_key][profile] = Path(path).relative_to(self.config_file.parent.parent).as_posix()
                    except ValueError:
                        pass
            for entry in saved["workflows"].values():
                entry.pop("path", None)
            if original:
                document["presets"][document["presets"].index(original)] = saved
            else:
                document["presets"].append(saved)
        write_json_atomic(self.builtin_file, document)

    def _require_editable(self, preset: dict, body: dict) -> None:
        if preset.get("scope") == "builtin" and body.get("builtin_unlock_token") not in self._builtin_unlock_tokens:
            raise ValueError("배포 프리셋은 읽기 전용입니다. 복제하거나 배포 풀기로 잠금을 해제하세요.")

    @staticmethod
    def active(document: dict) -> dict:
        return next((item for item in document["presets"] if item["id"] == document["active_id"]), {})

    def view(self, document: dict, workshop: str | None = None) -> dict:
        # Field autosaves have separate revisions, so saving one profile cannot
        # invalidate another profile's draft or the model-selection form.
        metadata = {"active_id": document["active_id"], "presets": [
            {key: item.get(key, "") for key in ("id", "name", "config", "scope", "description")} for item in document["presets"]]}
        return {**metadata, "revision": revision(metadata),
                "config": self.resolved_config(self.active(document).get("config", {})),
                "saved_at": self.saved_at(self.active(document), workshop)}

    def saved_at(self, preset: dict, workshop: str | None = None) -> str | None:
        # Use the actual on-disk timestamp, including after a browser/server
        # restart. Keep it outside revisions: field saves must not conflict
        # with another profile or the model-selection form.
        if not preset:
            return None
        path = self.builtin_file if preset.get("scope") == "builtin" else self._workshop_path(workshop)
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        except FileNotFoundError:
            return None

    def _carry_compatible(self, defaults: dict, previous: dict) -> dict:
        """Keep values the user already tuned when the workflow path changes."""
        fields = copy.deepcopy(defaults)
        for key, value in (previous or {}).items():
            if key not in fields:
                continue
            if isinstance(fields[key], dict) and isinstance(value, dict):
                merged = dict(fields[key])
                for field, field_value in value.items():
                    if field in merged:
                        merged[field] = copy.deepcopy(field_value)
                fields[key] = merged
            else:
                fields[key] = copy.deepcopy(value)
        return fields

    def _migrate_workflow(self, preset: dict, profile: str, source: str) -> None:
        previous = preset["workflows"].get(profile) or {}
        preset["workflows"].pop(profile, None)
        if not source or not previous.get("fields"):
            return
        try:
            workflow, _revision = read_workflow(source)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        fields = self._carry_compatible(workflow_json_defaults(workflow), previous["fields"])
        preset["workflows"][profile] = {"path": source, "fields": fields}

    def change(self, config: dict, body: dict, workshop: str | None = None) -> dict:
        document = self.read(config, workshop)
        if body.get("action") == "lock":
            token = body.get("builtin_unlock_token")
            if isinstance(token, str):
                self._builtin_unlock_tokens.discard(token)
            return self.view(document, workshop)
        if body.get("revision") != self.view(document, workshop)["revision"] or body.get("preset_id") != document["active_id"]:
            raise PresetConflict("프리셋이 변경되었습니다. 목록을 다시 불러오세요.")
        active = self.active(document)
        action = body.get("action")
        if action == "unlock":
            if workshop == "emotion":
                raise ValueError("감정 삽화 프리셋은 배포 프리셋을 지원하지 않습니다.")
            if body.get("confirmed") is not True:
                raise ValueError("배포 풀기 경고를 먼저 확인하세요.")
            token = secrets.token_urlsafe(32)
            self._builtin_unlock_tokens.add(token)
            return {**self.view(document, workshop), "builtin_unlock_token": token}
        if not active and action != "create":
            raise ValueError("그림 모델 프리셋을 먼저 만드세요.")
        if action in {"rename", "delete", "config"}:
            self._require_editable(active, body)
        if action in {"create", "duplicate", "rename"}:
            name = body.get("name")
            if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
                raise ValueError("프리셋 이름을 1~80자로 입력하세요.")
            name = name.strip()
            if any(item["name"].casefold() == name.casefold() and (action != "rename" or item["id"] != active["id"]) for item in document["presets"]):
                raise ValueError("같은 이름의 프리셋이 있습니다.")
            if action in {"create", "duplicate"}:
                # A new emotion preset starts from what the emotion workshop is
                # currently using, not from the illustration workshop's config.
                base = active["config"] if workshop == "emotion" and active else self.selected_config(config)
                source = {"config": copy.deepcopy(base), "workflows": {}} if action == "create" else active
                active = copy.deepcopy(source)
                active["id"] = uuid.uuid4().hex
                active["scope"] = "local"
                active.pop("description", None)
                document["presets"].append(active)
                document["active_id"] = active["id"]
            active["name"] = name
        elif action == "select":
            target = next((item for item in document["presets"] if item["id"] == body.get("target_id")), None)
            if target is None:
                raise ValueError("프리셋을 찾을 수 없습니다.")
            if workshop == "illustration" and uses_dual_vae(target):
                raise ValueError("이 프리셋은 감정 삽화 워크숍에서 선택하세요.")
            document["active_id"] = target["id"]
        elif action == "scope":
            if workshop == "emotion":
                raise ValueError("감정 삽화 프리셋은 사용자 프리셋으로만 관리합니다.")
            self._require_editable({"scope": "builtin"}, body)
            scope = body.get("scope")
            if scope not in {"local", "builtin"}:
                raise ValueError("배포 또는 사용자 프리셋을 선택하세요.")
            if scope != active.get("scope"):
                if scope == "builtin":
                    self._save_builtin(active, active["id"])
                else:
                    self._save_builtin(None, active["id"])
                    # An update can restore the old shipped ID. Give the user
                    # copy its own identity so it cannot be shadowed then.
                    active["id"] = uuid.uuid4().hex
                    document["active_id"] = active["id"]
                active["scope"] = scope
        elif action == "delete":
            remaining = [item for item in document["presets"] if item["id"] != active["id"]]
            if not remaining and workshop != "emotion":
                raise ValueError("프리셋은 하나 이상 남겨 두세요.")
            document["presets"].remove(active)
            document["active_id"] = remaining[0]["id"] if remaining else ""
        elif action == "config":
            changes = body.get("config")
            if not isinstance(changes, dict) or set(changes) - set(self.config_keys):
                raise ValueError("그림 모델 설정 항목을 확인하세요.")
            candidate = {**active["config"], **changes}
            self.validate_config(candidate)
            paths = candidate[self.paths_key]
            active["config"] = self.selected_config(candidate)
            # Keep the user's tuned sampling values when only the workflow file
            # changes; only fields the new workflow actually exposes carry over.
            for profile in list(active["workflows"]):
                if active["workflows"][profile].get("path") != paths.get(profile):
                    self._migrate_workflow(active, profile, paths.get(profile))
        else:
            raise ValueError("지원하지 않는 프리셋 작업입니다.")
        if active.get("scope") == "builtin" and action in {"rename", "delete", "config"}:
            self._save_builtin(None if action == "delete" else active, active["id"])
        self.save(document, workshop)
        return self.view(document, workshop)

    def workflow(self, config: dict, profile: str, preset_id: str, body: dict | None = None,
                 workshop: str | None = None) -> dict:
        if profile not in self.local_profiles:
            raise ValueError("로컬 그림 모델을 선택하세요.")
        document = self.read(config, workshop)
        if document["active_id"] != preset_id:
            raise PresetConflict("선택한 프리셋이 변경되었습니다. 프리셋을 다시 불러오세요.")
        preset = self.active(document)
        if body is not None:
            self._require_editable(preset, body)
        source = preset["config"][self.paths_key].get(profile, "")
        workflow, source_revision = read_workflow(source)
        entry = preset["workflows"].get(profile)
        defaults = workflow_json_defaults(workflow)
        if entry and entry.get("path") == source:
            fields = copy.deepcopy(entry["fields"])
        elif entry and entry.get("fields"):
            # A workflow path changed outside the preset editor; keep values
            # that the new workflow still exposes instead of resetting them.
            fields = self._carry_compatible(defaults, entry["fields"])
        else:
            fields = defaults
        fields = workflow_json_defaults(update_workflow_json_settings(workflow, fields))
        token = revision([preset_id, source, source_revision, fields])
        if body is not None:
            if body.get("path") != source or body.get("revision") != token:
                raise PresetConflict("프리셋 설정이나 워크플로우가 변경되었습니다. 다시 불러온 뒤 저장하세요.")
            fields = body.get("fields")
            fields = workflow_json_defaults(update_workflow_json_settings(workflow, fields))
        saved = {"path": source, "fields": fields}
        if entry != saved:
            preset["workflows"][profile] = saved
            if preset.get("scope") == "builtin":
                if body is not None:
                    self._save_builtin(preset, preset["id"])
            else:
                self.save(document, workshop)
        return {"preset_id": preset_id, "profile": profile, "path": source, "fields": fields,
                "revision": revision([preset_id, source, source_revision, fields]), "saved_at": self.saved_at(preset, workshop)}

    def snapshot(self, config: dict | None = None, preset_id: str | None = None,
                 workshop: str | None = None) -> dict:
        document = self.read(config, workshop)
        if not document or not document["presets"]:
            return {}
        selected = next((item for item in document["presets"] if item["id"] == preset_id), None) if preset_id else self.active(document)
        if selected is None:
            raise ValueError("그림 모델 프리셋을 찾을 수 없습니다.")
        preset = copy.deepcopy(selected)
        preset["config"] = self.resolved_config(preset["config"])
        profile = self.local_profile(preset["config"][self.type_key])
        if profile:
            source = preset["config"][self.paths_key].get(profile, "")
            entry = preset["workflows"].get(profile)
            if source and (not entry or entry.get("path") != source):
                fields = workflow_json_defaults(read_workflow(source)[0])
                if entry and entry.get("fields"):
                    fields = self._carry_compatible(fields, entry["fields"])
                if fields:
                    preset["workflows"][profile] = {"path": source, "fields": fields}
        return preset


def apply_active_model_preset(config: dict, config_file: str | Path, builtin_file: str | Path | None = None,
                              workshop: str | None = "illustration") -> dict:
    store = IllustrationModelPresets(config_file, builtin_file)
    document = store.read(workshop=workshop)
    if document and document["presets"]:
        config.update(store.resolved_config(store.active(document)["config"]))
    return workflow_profiles.normalize_workflow_config(config)


def apply_preset_workflow(workflow: dict, preset: dict, profile: str) -> dict:
    entry = preset.get("workflows", {}).get(workflow_profiles.illustration_local_profile(profile))
    # Switched-off LoRAs stay in the preset for later editing but must not reach
    # the ComfyUI graph. Fields themselves keep them so the editor can toggle
    # them back on.
    return update_workflow_json_settings(workflow, enabled_lora_fields(entry["fields"])) if entry else copy.deepcopy(workflow)


def register_model_preset_runtime(server, prefix, presets, on_change=None, workshop: str | None = None):
    import asyncio
    from aiohttp import web

    def requested_workshop(request, body: dict | None = None) -> str | None:
        # Illustration and emotion each read and write their own preset file;
        # asset presets keep one selection as before.
        if workshop not in WORKSHOPS:
            return None
        name = (body or {}).get("workshop", request.query.get("workshop"))
        return name if name in WORKSHOPS else workshop

    workflow_lock = asyncio.Lock()
    async def model_presets(request: web.Request) -> web.Response:
        try:
            async with workflow_lock:
                if request.method == "POST":
                    body = await request.json()
                    if not isinstance(body, dict) or set(body) - {"action", "preset_id", "target_id", "revision", "name", "config", "scope", "confirmed", "builtin_unlock_token", "workshop"}:
                        raise ValueError("프리셋 요청을 확인하세요.")
                    result = presets.change(server.app_config, body, requested_workshop(request, body))
                else:
                    selected_workshop = requested_workshop(request)
                    result = presets.view(presets.read(server.app_config, selected_workshop), selected_workshop)
                if on_change:
                    on_change()
                return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            return web.json_response({"error": str(exc)}, status=409 if isinstance(exc, PresetConflict) else 400)

    server.app.router.add_get(f"/api/{prefix}/presets", model_presets)
    server.app.router.add_post(f"/api/{prefix}/presets", model_presets)

    async def workflow(request: web.Request) -> web.Response:
        try:
            body = await request.json() if request.method == "POST" else {}
            if not isinstance(body, dict) or set(body) - {"preset_id", "profile", "path", "revision", "fields", "builtin_unlock_token", "workshop"}:
                raise ValueError("워크플로우 설정 요청을 확인하세요.")
            profile = body.get("profile", request.query.get("profile"))
            async with workflow_lock:
                result = presets.workflow(server.app_config, profile, body.get("preset_id", request.query.get("preset_id")),
                                          body if request.method == "POST" else None,
                                          requested_workshop(request, body))
                return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            return web.json_response({"error": str(exc)}, status=409 if isinstance(exc, PresetConflict) else 400)

    server.app.router.add_get(f"/api/{prefix}/workflow", workflow)
    server.app.router.add_post(f"/api/{prefix}/workflow", workflow)

