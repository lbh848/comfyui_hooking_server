"""Settings adapter for the existing distribution workflow validation engine."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app_paths import config_root
from json_store import write_json_atomic
from settings_store import SettingsConflict


# Use the original application's explicit workflow bindings, not filename guesses.
WORKFLOW_GROUPS = {
    "illustration_workflow_source_paths.v1": ("삽화", "V1 계열"),
    "illustration_workflow_source_paths.v3": ("삽화", "ANIMA+ILXL"),
    "illustration_workflow_source_paths.v3_anima": ("삽화", "ANIMA"),
    "workflow_pack_extra_paths.illustration_anima_fast": ("삽화", "ANIMA"),
    "workflow_pack_extra_paths.illustration_anima_fast_8gb": ("삽화", "ANIMA"),
    "asset_workflow_source_path": ("에셋", "ILXL"),
    "anima_asset_workflow_source_path": ("에셋", "ANIMA+ILXL"),
    "anima_only_asset_workflow_source_path": ("에셋", "ANIMA"),
    "workflow_pack_extra_paths.anima_rgba_asset": ("에셋", "ANIMA"),
    "anima_inpainting_workflow_source_path": ("이미지 편집", "ANIMA 인페인팅"),
    "lora_training_workflow_source_paths.anima": ("LoRA 학습", "ANIMA"),
    "lora_training_workflow_source_paths.sdxl": ("LoRA 학습", "ILXL"),
    "style_lora_training_workflow_source_paths.anima": ("LoRA 학습", "ANIMA"),
    "style_lora_training_workflow_source_paths.sdxl": ("LoRA 학습", "ILXL"),
    "workflow_pack_extra_paths.lora_training_anima_v2": ("LoRA 학습", "ANIMA"),
    "workflow_pack_extra_paths.style_lora_training_anima_v2": ("LoRA 학습", "ANIMA"),
    "face_extract_workflow_source_path": ("공통", "얼굴 추출"),
    "video_workflow_source_paths.i2v": ("영상", "H3 · I2V"),
    "video_workflow_source_paths.i2v_fast": ("영상", "H3 · I2V"),
    "video_workflow_source_paths.first_last": ("영상", "H3 · FLF2V"),
    "video_workflow_source_paths.first_last_fast": ("영상", "H3 · FLF2V"),
    "video_workflow_source_paths.ref2v": ("영상", "H3 · REF2V"),
    "video_workflow_source_paths.ref2v_fast": ("영상", "H3 · REF2V"),
    "tag_analysis_workflow_source_path": ("공통", "얼굴 태그 분석"),
    "asset_tag_analysis_workflow_source_path": ("공통", "범용 태그 분석"),
    "utility_workflow_source_path": ("공통", "삽화 유틸리티"),
    "debug_workflow_source_path": ("공통", "디버깅"),
    "gemma4_workflow_source_path": ("공통", "Gemma4 이미지 분석"),
}


def default_workflow_groups(items: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for item in items:
        paths = ({("미분류", "사용자 워크플로우")} if item.get("source") == "local" else
                 {WORKFLOW_GROUPS[binding] for binding in item.get("bindings", []) if binding in WORKFLOW_GROUPS})
        if not paths:
            paths = {("삽화", "기본")} if "comfy_workflow_source_path" in item.get("bindings", []) else {("기타", "미분류")}
        for path in sorted(paths):
            grouped.setdefault(path, []).append(item["id"])
    order = {name: index for index, name in enumerate(("삽화", "에셋", "이미지 편집", "LoRA 학습", "영상", "공통", "기타", "미분류"))}
    return [{"id": "default-" + hashlib.sha256((category + "/" + name).encode()).hexdigest()[:16],
             "category": category, "name": name, "itemIds": identities}
            for (category, name), identities in sorted(grouped.items(), key=lambda item: (order[item[0][0]], item[0][1]))]


class WorkflowGroups:
    def __init__(self, project_root: Path):
        self.path = config_root(project_root) / "workflow-groups.json"

    @staticmethod
    def validate(value: dict) -> dict:
        if not isinstance(value, dict) or set(value) != {"groups"} or not isinstance(value["groups"], list) or len(value["groups"]) > 100:
            raise ValueError("워크플로우 그룹 형식을 확인하세요. 그룹은 최대 100개까지 만들 수 있습니다.")
        groups, identities, names = [], set(), set()
        for group in value["groups"]:
            if (not isinstance(group, dict) or not {"id", "name", "itemIds"} <= set(group) <= {"id", "category", "name", "itemIds"}
                    or not isinstance(group["id"], str) or not 1 <= len(group["id"]) <= 100
                    or not isinstance(group["name"], str) or not 1 <= len(group["name"].strip()) <= 80
                    or not isinstance(group["itemIds"], list) or not 1 <= len(group["itemIds"]) <= 1000
                    or not all(isinstance(item, str) and 0 < len(item) <= 200 for item in group["itemIds"])):
                raise ValueError("그룹 이름과 포함할 워크플로우를 확인하세요.")
            category = group.get("category", "사용자 그룹")
            if not isinstance(category, str) or not 1 <= len(category.strip()) <= 80:
                raise ValueError("상위 그룹 이름은 1~80자로 입력하세요.")
            category, name = category.strip(), group["name"].strip()
            group_name = (category.casefold(), name.casefold())
            if group["id"] in identities or group_name in names:
                raise ValueError("같은 상위 그룹 안에 동일한 하위 그룹 이름이 있습니다.")
            identities.add(group["id"])
            names.add(group_name)
            groups.append({"id": group["id"], "category": category, "name": name, "itemIds": list(dict.fromkeys(group["itemIds"]))})
        return {"groups": groups}

    def read(self) -> tuple[dict, str]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return {"groups": []}, '"empty"'
        value = self.validate(json.loads(raw))
        return value, '"' + hashlib.sha256(raw).hexdigest() + '"'

    def save(self, value: dict, revision: str) -> tuple[dict, str]:
        value = self.validate(value)
        _, current = self.read()
        if revision != current:
            raise SettingsConflict
        write_json_atomic(self.path, value)
        return self.read()


class WorkflowGroupMoveBlocked(ValueError):
    """A group update attempted to move a distributed workflow."""


def _group_key(group: dict) -> tuple[str, str]:
    aliases = {"이미지 편집": "편집", "LoRA 학습": "학습", "공통": "유틸리티", "에셋 및 수정": "에셋"}
    return aliases.get(group["category"], group["category"]).casefold(), group["name"].casefold()


def effective_workflow_groups(catalog: dict, saved: list[dict]) -> list[dict]:
    """Keep pack classifications authoritative and overlay only local memberships."""
    pack_ids = {item["id"] for item in catalog["items"] if item["source"] == "pack"}
    result = []
    by_key = {}
    assigned_local = {identity for group in saved for identity in group["itemIds"] if identity.startswith("local:")}
    for group in catalog["defaultGroups"]:
        identities = [identity for identity in group["itemIds"] if identity in pack_ids or identity not in assigned_local]
        if identities:
            value = {**group, "itemIds": identities}
            result.append(value)
            by_key[_group_key(value)] = value
    for group in saved:
        identities = [identity for identity in group["itemIds"] if identity.startswith("local:")]
        if not identities:
            continue
        target = by_key.get(_group_key(group))
        if target is None:
            # Saved IDs may collide with a now-restored default group.
            identity = group["id"]
            if any(value["id"] == identity for value in result):
                identity = "local-group-" + hashlib.sha256(json.dumps(_group_key(group)).encode()).hexdigest()
            target = {**group, "id": identity, "itemIds": []}
            result.append(target)
            by_key[_group_key(group)] = target
        target["itemIds"] = list(dict.fromkeys([*target["itemIds"], *identities]))
    return result


def validate_workflow_group_move(catalog: dict, value: dict, previous: dict) -> dict:
    value = WorkflowGroups.validate(value)
    def memberships(groups, identity):
        return {_group_key(group) for group in groups if identity in group["itemIds"]}
    for item in catalog["items"]:
        if item["source"] == "pack" and memberships(value["groups"], item["id"]) != memberships(catalog["defaultGroups"], item["id"]):
            raise WorkflowGroupMoveBlocked("배포 워크플로우의 그룹은 변경할 수 없습니다. 사용자 워크플로우만 이동하세요.")
    known = {item["id"] for item in catalog["items"]}
    for group in value["groups"]:
        for identity in group["itemIds"]:
            if identity not in known and (not identity.startswith("local:") or memberships(value["groups"], identity) != memberships(previous["groups"], identity)):
                raise ValueError("이동할 사용자 워크플로우를 찾을 수 없습니다. 목록을 새로고침하세요.")
    return {"groups": effective_workflow_groups(catalog, value["groups"])}


def _latest_release(library: dict) -> dict:
    releases = library.get("releases", [])
    releases = [release for release in releases if isinstance(release.get("release_version"), str)
                and release["release_version"].startswith("v") and release["release_version"][1:].isdigit()]
    return max(releases, key=lambda item: int(item["release_version"][1:])) if releases else {"release_version": "", "items": []}


def _local_workflow_id(relative: str) -> str:
    return "local:" + hashlib.sha256(relative.encode("utf-8")).hexdigest()


def delete_local_workflow(service, *, identity: str, revision: str) -> dict:
    """Delete one catalogued user JSON, without inspecting or changing models."""
    from comfy_installer.workflow_library import WorkflowLibraryError

    if not isinstance(identity, str) or not identity.startswith("local:"):
        raise ValueError("삭제할 사용자 워크플로우를 선택하세요. 배포 워크플로우는 정리를 이용하세요.")
    if not isinstance(revision, str) or not revision:
        raise ValueError("삭제할 워크플로우를 먼저 확인하세요. 목록을 새로고침하세요.")
    library = service.workflow_library_status()
    source = next((file for file in library.get("workflow_files", [])
                   if _local_workflow_id(file["relative_path"]) == identity), None)
    if source is None:
        raise WorkflowLibraryError("삭제할 사용자 워크플로우를 찾을 수 없습니다. 목록을 새로고침하세요.")
    if any(item.get("sha256") == source["sha256"] for item in _latest_release(library).get("items", [])):
        raise WorkflowLibraryError("배포 워크플로우는 삭제할 수 없습니다. 정리를 이용하세요.")
    root = Path(library["user_root"]).resolve()
    target = root / source["relative_path"]
    if (not root.is_relative_to(service.comfy_root.resolve())
            or not target.resolve().is_relative_to(root) or target.suffix.casefold() != ".json"):
        raise WorkflowLibraryError("워크플로우 폴더 밖의 파일은 삭제할 수 없습니다.")
    if source["sha256"] != revision or hashlib.sha256(target.read_bytes()).hexdigest() != revision:
        raise WorkflowLibraryError("워크플로우가 변경되었습니다. 목록을 새로고침한 뒤 다시 확인하세요.")
    target.unlink()
    return {"deleted": identity}


def workflow_catalog(service) -> dict:
    library = service.workflow_library_status()
    release = _latest_release(library)
    available, skipped, error = {}, {}, ""
    from comfy_installer.service import InstallerServiceError
    if release["release_version"]:
        try:
            catalog = service.e2e_workflow_catalog()
            if catalog["release_version"] != release["release_version"]:
                raise InstallerServiceError("배포 버전이 변경되었습니다. 목록을 새로고침하세요.")
            available = {item["id"]: item for item in catalog["items"]}
            skipped = {item["id"]: item["reason"] for item in catalog.get("skipped", [])}
        except InstallerServiceError as exc:
            error = str(exc)
    items = []
    originals_by_hash = {}
    models_by_id = {}
    model_error = ""
    if any(isinstance(item.get("model_ids"), list) for item in release.get("items", [])):
        from comfy_installer.workflow_library import release_install_manifest, WorkflowLibraryError
        try:
            manifest = release_install_manifest(library_root=service.workflow_library_root, release_version=release["release_version"])
            models_by_id = {model["id"]: model for model in manifest.models}
        except (WorkflowLibraryError, OSError) as exc:
            model_error = str(exc)
    for item in release.get("items", []):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        identity = item["id"]
        installation = {"modelsInstalled": False, "missingModelCount": None, "missingModelBytes": None,
                        "installBlockedReason": model_error or "필수 모델 목록을 확인할 수 없습니다."}
        model_ids = item.get("model_ids")
        if not model_error and isinstance(model_ids, list) and all(model_id in models_by_id for model_id in model_ids):
            from comfy_installer.model_installer import missing_models
            missing = missing_models(models=[models_by_id[model_id] for model_id in model_ids], comfy_root=service.comfy_root)
            installation.update(modelsInstalled=not missing, missingModelCount=len(missing),
                                missingModelBytes=sum(model["size"] for model in missing), installBlockedReason="")
        items.append({
            **installation,
            "id": identity, "name": item.get("name") or item.get("filename") or identity,
            "filename": item.get("filename", ""), "bindings": item.get("bindings", []),
            "modelCount": item.get("model_count", len(item.get("model_ids", []))),
            "modelBytes": item.get("model_bytes", 0),
            "source": "pack", "localPaths": [], "integrityBlockedReason": "",
            "e2eProfiles": available.get(identity, {}).get("e2e_profiles", []),
            "e2eBlockedReason": error or skipped.get(identity, "") or ("" if identity in available else "E2E 검사 방법이 등록되지 않았습니다."),
        })
        if item.get("sha256"):
            originals_by_hash.setdefault(item["sha256"], items[-1])
    for file in library.get("workflow_files", []):
        relative = file["relative_path"]
        original = originals_by_hash.get(file["sha256"])
        if original is not None:
            original["localPaths"].append(relative)
            continue
        items.append({
            "id": _local_workflow_id(relative),
            "name": file["name"], "filename": relative, "bindings": [],
            "source": "local", "localPaths": [relative], "fileBytes": file["size"], "fileRevision": file["sha256"],
            "modelCount": None, "modelBytes": None, "e2eProfiles": [],
            "e2eBlockedReason": "사용자 워크플로우는 자동 E2E 검증 대상이 아닙니다.",
            "integrityBlockedReason": "사용자 워크플로우는 비교할 팩 원본이 없어 무결성 검증 대상이 아닙니다.",
            "modelsInstalled": False, "missingModelCount": None, "missingModelBytes": None,
            "installBlockedReason": "사용자 워크플로우에는 자동 설치할 필수 모델 목록이 없습니다.",
        })
    cleanup = None
    cleanup_error = model_error or "필수 모델 목록을 확인할 수 없습니다."
    if models_by_id:
        from comfy_installer.workflow_cleanup import WorkflowModelCleanup
        try:
            cleanup = WorkflowModelCleanup(comfy_root=service.comfy_root, library_root=service.workflow_library_root, library=library)
            cleanup_error = ""
        except (WorkflowLibraryError, OSError, ValueError) as exc:
            cleanup_error = str(exc)
    for item in items:
        item["workflowInstalled"] = bool(item["localPaths"])
        item.update(cleanupFileCount=0, cleanupBytes=0, cleanupBlockedReason=cleanup_error)
        if item["source"] == "local":
            item["cleanupBlockedReason"] = "사용자 워크플로우에는 정리할 필수 모델 목록이 없습니다."
        elif not item["workflowInstalled"]:
            item["cleanupBlockedReason"] = "설치된 워크플로우만 모델을 정리할 수 있습니다."
        elif cleanup is not None:
            try:
                plan = cleanup.plan(release_version=release["release_version"], selected_item_ids=[item["id"]])
                item.update(cleanupFileCount=len(plan["removable"]), cleanupBytes=plan["removableBytes"],
                            cleanupBlockedReason="" if plan["removable"] else "정리할 모델이 없습니다. 공유 모델은 유지합니다.")
            except (WorkflowLibraryError, OSError) as exc:
                item["cleanupBlockedReason"] = str(exc)
    return {"releaseVersion": release["release_version"], "items": items, "defaultGroups": default_workflow_groups(items), "e2eError": error,
            "workflowFolder": library.get("user_root", ""), "folderError": library.get("workflow_files_error", "")}


def workflow_selection(body: dict) -> tuple[str, list[str]]:
    release, selected = body.get("releaseVersion"), body.get("selectedItemIds")
    if (not isinstance(release, str) or not release
            or not isinstance(selected, list) or not 1 <= len(selected) <= 1000
            or not all(isinstance(value, str) and 0 < len(value) <= 200 for value in selected)):
        raise ValueError("검사할 배포 워크플로우를 하나 이상 선택하세요.")
    if any(identity.startswith("local:") for identity in selected):
        raise ValueError("사용자 워크플로우는 팩 검증 대상이 아닙니다. 팩 워크플로우를 선택하세요.")
    return release, list(dict.fromkeys(selected))
