"""Model training snapshots with separate release and user storage."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import uuid

from app_paths import config_root, install_root, resolve_install_path
from asset_preset_scope import AssetPresetLock, write_scope_documents
from configuration import settings_lock
from lora_training_steps import training_step_settings
from lora_training_workflow import training_optimizer_type
from preset_locations import PRESET_LOCATIONS


PROFILES = {"anima", "sdxl", "moss"}
NUMBER_MINIMUMS = {
    "step_per_image": 1, "il_rate": .000001, "save_per_step": 1, "save_after": 0,
    "gen_w": 64, "gen_h": 64, "resolution": 64, "dim": 1, "alpha": 1,
}


class TrainingPresetConflict(ValueError):
    pass


def validate_training_settings(value, profile):
    keys = {*NUMBER_MINIMUMS, "profile", "multi_img_folder_name", "upscale"}
    if (not isinstance(value, dict) or not keys <= set(value) or set(value) - keys - {"step_mode", "total_steps", "max_train_epochs", "workflow_source_path", "optimizer_type"}
            or value.get("profile") != profile):
        raise ValueError("학습 프로필과 설정 항목을 확인하세요.")
    if value.get("step_mode") == "epochs" and profile != "anima":
        raise ValueError("Epochs 학습은 ANIMA 프로필에서 선택하세요.")
    for key, minimum in NUMBER_MINIMUMS.items():
        number = value[key]
        if (type(number) not in (int, float) or number < minimum or number > 9007199254740991
                or not math.isfinite(number) or key != "il_rate" and number != int(number)):
            raise ValueError(f"학습 설정 값을 확인하세요: {key}")
    folder = value["multi_img_folder_name"]
    if (not isinstance(folder, str) or not folder.strip() or folder in {".", ".."}
            or any(char in folder for char in '/\\:*?"<>|') or any(ord(char) < 32 for char in folder)):
        raise ValueError("학습 이미지 폴더 이름을 확인하세요.")
    if not isinstance(value["upscale"], bool):
        raise ValueError("업스케일 설정을 확인하세요.")
    workflow_source_path = value.get("workflow_source_path", "")
    if (not isinstance(workflow_source_path, str) or "\0" in workflow_source_path
            or workflow_source_path.strip() and not workflow_source_path.strip().lower().endswith(".json")):
        raise ValueError("학습 워크플로우 JSON 파일을 선택하세요.")
    return {**copy.deepcopy(value), **training_step_settings(value), "workflow_source_path": workflow_source_path.strip(),
            "optimizer_type": training_optimizer_type(value.get("optimizer_type", "default"))}


class LoraTrainingPresets:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.path = PRESET_LOCATIONS["lora_training"].local_path(project_root)
        self.builtin_file = PRESET_LOCATIONS["lora_training"].builtin_path(project_root)
        self.lock = AssetPresetLock()

    @staticmethod
    def _name(value):
        if (not isinstance(value, str) or not 1 <= len(value.strip()) <= 80
                or any(ord(char) < 32 for char in value)):
            raise ValueError("프리셋 이름을 1~80자로 입력하세요.")
        return value.strip()

    def _read(self, path):
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            document = {"version": 1, "presets": []}
        if not isinstance(document, dict) or document.get("version") != 1 or not isinstance(document.get("presets"), list):
            raise ValueError("학습 프로필 프리셋 파일을 확인하세요.")
        identities, names = set(), set()
        for preset in document["presets"]:
            if (not isinstance(preset, dict) or not isinstance(preset.get("id"), str) or not preset["id"]
                    or preset["id"] in identities or preset.get("profile") not in PROFILES):
                raise ValueError("학습 프로필 프리셋 파일을 확인하세요.")
            name = (preset["profile"], self._name(preset.get("name")).casefold())
            if name in names:
                raise ValueError("같은 이름의 학습 프로필 프리셋이 있습니다.")
            identities.add(preset["id"])
            names.add(name)
            preset["settings"] = validate_training_settings(preset.get("settings"), preset["profile"])
            if path == self.builtin_file and preset["settings"]["workflow_source_path"]:
                preset["settings"]["workflow_source_path"] = str(resolve_install_path(
                    preset["settings"]["workflow_source_path"], self.project_root))
        return document

    def _catalog(self):
        presets = [{**item, "scope": scope} for scope, path in (("builtin", self.builtin_file), ("local", self.path))
                   for item in self._read(path)["presets"]]
        # Releases can restore a builtin previously moved to user storage.
        identities = {item["id"] for item in presets if item["scope"] == "local"}
        for item in presets:
            if item["scope"] == "builtin":
                item["source_id"] = item["id"]
                while item["id"] in identities:
                    item["id"] = "builtin:" + item["id"]
                identities.add(item["id"])
        revision = hashlib.sha256(json.dumps(presets, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        return {"presets": presets, "revision": revision}

    @staticmethod
    def _profile(profile):
        if not isinstance(profile, str) or profile not in PROFILES:
            raise ValueError("ANIMA, ILXL 또는 MOSS 학습 프로필을 선택하세요.")
        return profile

    def catalog(self, profile):
        profile = self._profile(profile)
        with settings_lock(self.path):
            catalog = self._catalog()
            return {**catalog, "profile": profile, "presets": [item for item in catalog["presets"] if item["profile"] == profile]}

    def change(self, body):
        if not isinstance(body, dict) or set(body) - {"action", "profile", "id", "name", "settings", "revision", "confirmed", "scope", "builtin_unlock_token"}:
            raise ValueError("학습 프로필 프리셋 요청을 확인하세요.")
        profile = self._profile(body.get("profile"))
        with settings_lock(self.path):
            catalog = self._catalog()
            action = body.get("action")
            if action in {"unlock", "lock"}:
                return {**self.catalog(profile), **self.lock.change(body)}
            if body.get("revision") != catalog["revision"]:
                raise TrainingPresetConflict("학습 프로필 프리셋이 변경되었습니다. 새로고침한 뒤 다시 저장하세요.")
            if action not in {"create", "save", "rename", "delete", "scope"}:
                raise ValueError("지원하지 않는 학습 프로필 프리셋 요청입니다.")
            presets = catalog["presets"]
            preset = next((item for item in presets if item["id"] == body.get("id") and item["profile"] == profile), None)
            if action != "create" and preset is None:
                raise ValueError("학습 프로필 프리셋을 찾을 수 없습니다.")
            original_scope = preset["scope"] if action != "create" else body.get("scope", "local")
            target_scope = body.get("scope", original_scope) if action == "scope" else original_scope
            if target_scope not in {"builtin", "local"}:
                raise ValueError("배포 또는 사용자 프리셋을 선택하세요.")
            if original_scope == "builtin" or target_scope == "builtin" or action == "scope":
                self.lock.require(body)
            name = self._name(body.get("name")) if action in {"create", "rename"} else preset["name"]
            if action in {"create", "rename", "scope"} and any(
                (action == "create" or item is not preset) and item["profile"] == profile and item["scope"] == target_scope
                and item["name"].casefold() == name.casefold() for item in presets
            ):
                raise ValueError("같은 이름의 학습 프로필 프리셋이 있습니다.")
            if action == "create":
                preset = {"id": uuid.uuid4().hex, "profile": profile, "scope": target_scope, "name": name,
                          "settings": validate_training_settings(body.get("settings"), profile)}
                presets.append(preset)
            elif action == "save":
                preset["settings"] = validate_training_settings(body.get("settings"), profile)
            elif action == "rename":
                preset["name"] = name
            elif action == "scope":
                preset["scope"] = target_scope
            else:
                if body.get("confirmed") is not True:
                    raise ValueError("삭제할 학습 프로필 프리셋을 확인하세요.")
                presets.remove(preset)
            documents = {}
            for scope, path in (("builtin", self.builtin_file), ("local", self.path)):
                if scope not in {original_scope, target_scope}:
                    continue
                entries = []
                for item in presets:
                    if item["scope"] == scope:
                        stored = copy.deepcopy({key: value for key, value in item.items() if key not in {"scope", "source_id"}})
                        if scope == "builtin":
                            stored["id"] = item.get("source_id", item["id"])
                            workflow_source_path = stored["settings"]["workflow_source_path"]
                            if workflow_source_path:
                                try:
                                    stored["settings"]["workflow_source_path"] = resolve_install_path(
                                        workflow_source_path, self.project_root).resolve().relative_to(
                                            install_root(self.project_root)).as_posix()
                                except ValueError as exc:
                                    raise ValueError("배포 학습 프리셋에는 공방 설치 폴더 안의 워크플로우를 선택하세요.") from exc
                        entries.append(stored)
                documents[path] = {"version": 1, "presets": entries}
            write_scope_documents(documents)
            return {**self.catalog(profile), "saved_id": preset["id"] if action != "delete" else ""}


def register_training_presets(app, project_root, schema):
    from aiohttp import web

    store = LoraTrainingPresets(project_root)

    async def presets(request):
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != schema:
            return web.json_response({"error": "설정 화면을 새로고침해 주세요."}, status=409)
        try:
            result = store.change(await request.json()) if request.method == "POST" else store.catalog(request.query.get("profile"))
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=409 if isinstance(exc, TrainingPresetConflict) else 400)

    app.router.add_get("/api/setting/internal-lora/training-presets", presets)
    app.router.add_post("/api/setting/internal-lora/training-presets", presets)
