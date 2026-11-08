"""Plan and remove workflow models while preserving installed consumers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .crypto import WorkflowPackError
from .model_installer import ModelInstallError, _safe_model_target
from .pack_cli import _workflow_model_ids
from .workflow_library import WorkflowLibraryError, library_status, release_install_manifest


class WorkflowModelCleanup:
    def __init__(self, *, comfy_root: Path, library_root: Path, library: dict | None = None):
        self.comfy_root = comfy_root.resolve()
        self.library = library if library is not None else library_status(comfy_root, library_root)
        if self.library.get("workflow_files_error"):
            raise WorkflowLibraryError("다른 워크플로우의 모델 사용 여부를 확인할 수 없습니다. " + self.library["workflow_files_error"])
        self.workflows = []
        self.local_usage: dict[Path, set[str]] = {}
        manifests = []
        installed_hashes = {source["sha256"] for source in self.library.get("workflow_files", [])}
        for release in self.library.get("releases", []):
            version = release["release_version"]
            manifest = release_install_manifest(library_root=library_root, release_version=version)
            manifests.append(manifest)
            by_id = {model["id"]: model for model in manifest.models}
            for item in release.get("items", []):
                model_ids = item.get("model_ids")
                if not isinstance(model_ids, list) or any(identity not in by_id for identity in model_ids):
                    raise WorkflowLibraryError("워크플로우의 필수 모델 목록을 확인할 수 없어 정리할 수 없습니다.")
                models = [by_id[identity] for identity in model_ids]
                self.workflows.append({
                    "release": version, "id": item["id"], "sha256": item.get("sha256"),
                    "name": item.get("name") or item.get("filename") or item["id"],
                    "models": models, "paths": {self._target(model["relative_path"]) for model in models},
                    "installed": item.get("sha256") in installed_hashes,
                })
        known_hashes = {item["sha256"] for item in self.workflows if item["sha256"]}
        for source in self.library.get("workflow_files", []):
            if source["sha256"] in known_hashes:
                continue
            # Personal or modified workflows have no complete dependency manifest.
            # Preserve every known model they reference, including partial setups.
            for manifest in manifests:
                try:
                    model_ids = set(_workflow_model_ids(Path(source["path"]), manifest))
                except WorkflowPackError as exc:
                    raise WorkflowLibraryError("사용자 워크플로우의 모델 사용 여부를 확인하지 못했습니다.") from exc
                for model in manifest.models:
                    if model["id"] in model_ids:
                        path = self._target(model["relative_path"])
                        self.local_usage.setdefault(path, set()).add(source["relative_path"])

    def _target(self, relative_path: str) -> Path:
        try:
            target = _safe_model_target(self.comfy_root, relative_path)
        except ModelInstallError as exc:
            raise WorkflowLibraryError(str(exc)) from exc
        if not target.is_relative_to(self.comfy_root / "models"):
            raise WorkflowLibraryError("내장 ComfyUI 모델 폴더 밖의 파일은 정리할 수 없습니다.")
        return target

    def plan(self, *, release_version: str, selected_item_ids: list[str]) -> dict:
        selected = [item for item in self.workflows if item["release"] == release_version and item["id"] in selected_item_ids]
        if not selected or {item["id"] for item in selected} != set(selected_item_ids):
            raise WorkflowLibraryError("정리할 워크플로우를 찾을 수 없습니다. 목록을 새로고침하세요.")
        if any(not item["installed"] for item in selected):
            raise WorkflowLibraryError("설치된 워크플로우만 모델을 정리할 수 있습니다.")
        used_by = {path: set(names) for path, names in self.local_usage.items()}
        for item in self.workflows:
            if item in selected or not item["installed"]:
                continue
            for path in item["paths"]:
                used_by.setdefault(path, set()).add(item["name"])
        models = {self._target(model["relative_path"]): model for item in selected for model in item["models"]}
        removable, retained = [], []
        for target, model in sorted(models.items(), key=lambda pair: str(pair[0])):
            # Include resumable download fragments, but never other files in the folder.
            for path in (target, target.with_name(target.name + ".part")):
                relative = path.relative_to(self.comfy_root).as_posix()
                path = self._target(relative)
                try:
                    if not path.is_file():
                        continue
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                entry = {"id": model["id"], "path": relative, "size": stat.st_size,
                         "mtimeNs": str(stat.st_mtime_ns), "fileId": str(stat.st_ino),
                         "usedBy": sorted(used_by.get(target, set()))}
                (retained if entry["usedBy"] else removable).append(entry)
        plan = {"releaseVersion": release_version, "selectedItemIds": sorted(selected_item_ids),
                "removable": removable, "retained": retained, "removableBytes": sum(item["size"] for item in removable)}
        plan["revision"] = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return plan

    def delete(self, plan: dict) -> dict:
        deleted, failed = [], []
        for item in plan["removable"]:
            try:
                target = self._target(item["path"])
                stat = target.stat()
                if (stat.st_size, str(stat.st_mtime_ns), str(stat.st_ino)) != (item["size"], item["mtimeNs"], item["fileId"]):
                    failed.append(f'{item["path"]}: 파일이 변경되어 유지했습니다.')
                    continue
                target.unlink()
                deleted.append(item)
            except FileNotFoundError:
                continue
            except (OSError, WorkflowLibraryError):
                failed.append(f'{item["path"]}: 파일 사용 여부와 폴더 접근 권한을 확인하세요.')
        return {"deletedCount": len(deleted), "deletedBytes": sum(item["size"] for item in deleted),
                "retainedCount": len(plan["retained"]), "failed": failed}
