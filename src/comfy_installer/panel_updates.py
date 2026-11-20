"""Update-panel orchestration using the installer's existing recovery flow.

Original-project fixes retained here:
* 778c4cd / b76c870: preserve untracked files and verified tracked-change backups.
* 7aeebb5 / 0ac7e4d: receipt-based updates and transactional runtime rollback.
* 39cd99f: accept workflow hotfixes without incrementing the pack release.
* 801ea1e: retain bounded cleanup of old transaction environments.
* d8a57d4: keep workflow generation tests separate from ordinary updates.

Distribution presets and bundled workflows now follow application releases.
The panel follows the installed application branch and has permanent conflict backups and separate actions
for the three Soya repositories and the application's pinned Comfy runtime.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
from pathlib import Path

from app_paths import data_root, install_root
from json_store import write_json_atomic
from .manifest import InstallManifest, default_install_model_ids, load_install_manifest
from .model_installer import missing_models
from .configurator import apply_repaired_workflow_bindings
from .execution_profile import custom_nodes_for_profile
from .operations import run_command, uv_python_path
from .runtime_state import (
    _tree_logical_file_bytes, _remove_tree_force_writable, git_head,
    collect_custom_node_state, desired_python_signature, load_runtime_receipt,
)
from .updater import (
    HOOKING_REPOSITORY, _normalize_repository, _run_git_capture,
    _prepare_untracked_quarantine, _restore_untracked_quarantine,
    fetch_hooking_commit, hooking_update_branch, update_hooking_server,
)
from .workflow_library import (
    basic_workflow_item_ids, install_managed_workflows, unpack_to_library,
)


SOYA_NODE_NAMES = (
    "comfyui-instant-lora_v_soya", "comfyui-soya-custom-nodes",
    "comfyui-workflow-to-api-converter-endpoint",
)


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class WorkshopUpdates:
    def __init__(self, service, *, stop_comfy):
        self.service = service
        self.stop_comfy = stop_comfy
        self.root = install_root(service.project_root)
        self.backup_root = data_root(self.root) / "update_backup"
        self.cache_root = data_root(self.root) / "update_cache"
        self.state_path = self.cache_root / "checked.json"
        self.checked = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else {}

    @staticmethod
    def _head(path: Path) -> str | None:
        return git_head(path) if (path / ".git").is_dir() else None

    def _git(self, path: Path, *args: str) -> str:
        return _run_git_capture(path, *args).decode("utf-8").strip()

    def _verify_origin(self, path: Path, repository: str) -> None:
        if _normalize_repository(self._git(path, "remote", "get-url", "origin")) != _normalize_repository(repository):
            raise ValueError(f"업데이트 저장소 주소가 다릅니다: {path.name}")

    def _save(self) -> None:
        write_json_atomic(self.state_path, self.checked)

    def _checked_for_branch(self, branch: str) -> dict:
        # Checks saved before branch selection always came from origin/main.
        return self.checked if self.checked.get("branch", "main") == branch else {}

    def backup_status(self) -> dict:
        # Never traverse a user-created junction outside this installation.
        if self.backup_root.is_symlink() or self.backup_root.is_junction() or self.backup_root.resolve() != self.root / "data" / "update_backup":
            raise ValueError("BACKUP 경로를 확인하세요.")
        return {"path": str(self.backup_root), "bytes": _tree_logical_file_bytes(self.backup_root) if self.backup_root.is_dir() else 0}

    def _missing_default_models(self, manifest: InstallManifest) -> list[dict] | None:
        comfy_root = self.service.comfy_root
        if not (comfy_root / "main.py").is_file() or not uv_python_path(comfy_root / ".venv").is_file():
            return None
        default_ids = default_install_model_ids(manifest.models)
        return missing_models(models=[model for model in manifest.models if model["id"] in default_ids], comfy_root=comfy_root)

    def status(self) -> dict:
        current = self._head(self.root)
        checked = self._checked_for_branch(self._git(self.root, "branch", "--show-current")) if self.checked and current else {}
        target = checked.get("application")
        ahead = behind = 0
        if target and current:
            ahead, behind = map(int, self._git(self.root, "rev-list", "--left-right", "--count", f"{current}...{target}").split())
        manifest = load_install_manifest(self.service.manifest.source_path)
        missing = self._missing_default_models(manifest)
        desired = checked.get("comfy", manifest.comfy)
        nodes = []
        for name in SOYA_NODE_NAMES:
            node = next(node for node in manifest.custom_nodes if node["name"] == name)
            head = self._head(self.service.comfy_root / "custom_nodes" / name)
            node_target = checked.get("nodes", {}).get(name)
            nodes.append({"name": name, "current": head, "target": node_target,
                          "required_ref": node.get("ref"), "update_needed": bool(node_target and head != node_target)})
        comfy_head = self._head(self.service.comfy_root)
        desired_data = checked.get("manifest", manifest.data)
        desired_manifest = InstallManifest(manifest.source_path, desired_data, manifest.sha256)
        changes = []
        if comfy_head and comfy_head != desired["ref"]:
            changes.append("ComfyUI 버전")
        if comfy_head:
            receipt = load_runtime_receipt(self.service.comfy_root) or {}
            python = receipt.get("python", {})
            profile_id = python.get("profile_id")
            if not profile_id or python.get("signature") != desired_python_signature(
                desired_manifest, profile_id=profile_id, install_mode=self.service._installed_install_mode(),
            ):
                changes.append("Python 실행 환경")
            dependency_nodes = custom_nodes_for_profile(
                [node for node in desired_manifest.custom_nodes if node["name"] not in SOYA_NODE_NAMES],
                cpu_only=python.get("profile_kind") == "cpu",
            )
            states = collect_custom_node_state(self.service.comfy_root, dependency_nodes)
            for node in dependency_nodes:
                actual = states[node["name"]]
                if (not actual.get("exists")
                    or node.get("source_type") == "git" and node.get("existing_policy") != "reuse_if_same_origin" and node.get("ref") and actual.get("head") != node["ref"]
                    or node.get("source_type") == "archive" and actual.get("archive_sha256") != node["sha256"]):
                    changes.append("의존 커스텀 노드")
                    break
        restart_required = bool(current and self.service._startup_project_head and current != self.service._startup_project_head)
        return {
            "checked_at": checked.get("checked_at"), "job": self.service.status(),
            "application": {"current": current, "target": target, "ahead": ahead, "behind": behind,
                            "update_needed": bool(behind or checked.get("workflows_pending")),
                            "commits": checked.get("commits", []), "restart_required": restart_required},
            "nodes": nodes,
            "missing_models": None if missing is None else [
                {"path": model["relative_path"], "size": model["size"]} for model in missing
            ],
            "comfy": {"current": comfy_head, "version": desired["version"], "target": desired["ref"],
                      "required_version": manifest.comfy["version"], "required_ref": manifest.comfy["ref"],
                      "update_needed": bool(changes), "changes": changes},
            "backup": self.backup_status(),
        }

    def clear_backups(self) -> dict:
        with self.service._lock:
            if self.service._thread is not None and self.service._thread.is_alive():
                raise ValueError("진행 중인 작업이 끝난 뒤 BACKUP을 비우세요.")
            self.backup_status()
            if self.backup_root.exists():
                _remove_tree_force_writable(self.backup_root)
        return self.status()

    def start_missing_models(self, civitai_key: str) -> dict:
        missing = self._missing_default_models(load_install_manifest(self.service.manifest.source_path))
        if missing is None:
            raise ValueError("내장 ComfyUI를 먼저 설치하세요.")
        if not missing:
            return self.status()
        self.service.start_missing_models(models=missing, civitai_key=civitai_key, operation="panel_missing-models")
        return self.status()

    def start(self, action: str, *, workflow_key: str = "") -> dict:
        from .service import _UPDATE_PHASES
        with self.service._lock:
            if self.service._thread is not None and self.service._thread.is_alive():
                raise ValueError("이미 설치 또는 업데이트 작업이 진행 중입니다.")
            if action != "fetch":
                state = self.status()
                if not state["checked_at"]:
                    raise ValueError("먼저 커밋 가져오기를 실행하세요.")
                if state["application"]["restart_required"]:
                    raise ValueError("공방을 다시 실행한 뒤 업데이트를 계속하세요.")
                if action != "application":
                    if state["application"]["update_needed"]:
                        raise ValueError("공방 업데이트를 먼저 실행하세요.")
                    if not uv_python_path(self.service.comfy_root / ".venv").is_file():
                        raise ValueError("내장 ComfyUI를 먼저 설치하세요.")
                elif self.service.comfy_root.is_dir() and not workflow_key:
                    raise ValueError("초기 설치에 사용할 암호키를 입력하고 저장하세요.")
                if action == "application" and state["application"]["ahead"] and state["application"]["behind"]:
                    raise ValueError("로컬과 원격의 커밋 이력이 갈라져 있습니다. 저장소 이력을 확인하세요.")
            self.service._start_operation(
                operation=f"panel_{action}", phases=_UPDATE_PHASES if action in {"soya", "comfy"} else ((action, {"fetch": "커밋 가져오기", "application": "공방과 내장 워크플로우 업데이트"}[action]),),
                target=self._run, kwargs={"action": action, "workflow_key": workflow_key},
            )
        return self.status()

    def _fetch(self) -> dict:
        self._verify_origin(self.root, HOOKING_REPOSITORY)
        branch = hooking_update_branch(self.root)
        target = fetch_hooking_commit(self.root, branch, cancel_event=self.service._cancel, log=self.service._log)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_root / "install_manifest.json"
        manifest_path.write_bytes(_run_git_capture(self.root, "show", f"{target}:src/comfy_installer/resources/install_manifest.json"))
        manifest = load_install_manifest(manifest_path)
        nodes = {}
        for name in SOYA_NODE_NAMES:
            node = next(node for node in manifest.custom_nodes if node["name"] == name)
            path = self.service.comfy_root / "custom_nodes" / name
            if not (path / ".git").is_dir():
                path = self.cache_root / name
                if not (path / ".git").is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    run_command(["git", "init"], cwd=path)
                    run_command(["git", "remote", "add", "origin", node["repository"]], cwd=path)
            self._verify_origin(path, node["repository"])
            run_command(["git", "fetch", "origin", node.get("tracking_branch", "main")], cwd=path, cancel_event=self.service._cancel, log=self.service._log, timeout=900)
            nodes[name] = self._git(path, "rev-parse", "FETCH_HEAD")
        commits = self._git(self.root, "log", "-20", "--format=%h%x09%s", f"HEAD..{target}").splitlines()
        self.checked = {"checked_at": _now(), "branch": branch, "application": target, "comfy": manifest.comfy, "manifest": manifest.data,
                        "nodes": nodes, "commits": commits,
                        "workflows_pending": self._checked_for_branch(branch).get("workflows_pending", False)}
        self._save()
        return {"message": "커밋을 가져왔습니다."}

    def _update_application(self, workflow_key: str) -> dict:
        self.checked["workflows_pending"] = True
        self._save()
        target = self.checked["application"]
        state = self.status()["application"]
        result = {"changed": False, "restart_required": False}
        if state["behind"]:
            result = update_hooking_server(
                project_root=self.service.project_root, config_path=self.service.config_path,
                backup_dir=self.backup_root / "config", cancel_event=self.service._cancel,
                log=self.service._log, target_ref=target, target_branch=self.checked.get("branch", "main"),
            )
            # The newly fetched release can require a newer pack reader. Resume
            # the bundled-workflow phase after the new application has started.
            if result.get("restart_required"):
                return {**result, "workflow_pending": True,
                        "message": "공방을 다시 실행한 뒤 공방 업데이트를 눌러 내장 워크플로우 적용을 완료하세요."}
        if self.service.comfy_root.is_dir():
            manifest = load_install_manifest(self.service.manifest.source_path)
            unpacked = unpack_to_library(
                pack_path=self.service.project_root / "comfy_installer" / "resources" / "workflows.soyawfp",
                passphrase=workflow_key, library_root=self.service.workflow_library_root,
                work_root=self.service.work_root, manifest=manifest, log=self.service._log,
            )
            release = unpacked["release_version"]
            managed_path = self.service.comfy_root / ".installer-state" / "managed-workflows.json"
            managed = json.loads(managed_path.read_text(encoding="utf-8")) if managed_path.is_file() else {}
            selected = {entry["item_id"] for entry in managed.get("files", {}).values()}
            available = {item["id"] for item in manifest.workflow_items}
            retired_item_ids = selected - available
            selected = sorted(selected & available) or basic_workflow_item_ids(self.service.workflow_library_root, release)
            if self.service._cancel.is_set():
                raise ValueError("공방 업데이트를 중단했습니다.")
            selection = install_managed_workflows(
                comfy_root=self.service.comfy_root, library_root=self.service.workflow_library_root,
                release_version=release, selected_item_ids=selected, log=self.service._log,
                allow_missing_models=True,
                backup_dir=self.backup_root / "workflows" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                retired_item_ids=retired_item_ids,
            )
            if selection.workflow_bindings:
                apply_repaired_workflow_bindings(
                    config_path=self.service.config_path, backup_dir=self.backup_root / "config",
                    comfy_root=self.service.comfy_root, workflow_bindings=selection.workflow_bindings,
                )
        self.checked["workflows_pending"] = False
        self._save()
        return {**result, "message": "공방 업데이트를 완료했습니다.", "workflow": True}

    def _run(self, *, action: str, workflow_key: str) -> None:
        quarantines = []
        try:
            if action != "fetch":
                self.service._log("업데이트를 준비합니다. 공방에서 실행한 Comfy를 종료합니다.")
                self.stop_comfy()
                if self.service._cancel.is_set():
                    raise ValueError("업데이트를 중단했습니다.")
                self.service._require_comfy_not_in_use()
            if action in {"fetch", "application"}:
                self.service._set_phase(action)
            if action == "fetch":
                result = self._fetch()
            elif action == "application":
                result = self._update_application(workflow_key)
            else:
                manifest = load_install_manifest(self.service.manifest.source_path)
                data = copy.deepcopy(manifest.data)
                for node in data["custom_nodes"]:
                    if node["name"] in SOYA_NODE_NAMES:
                        node.pop("tracking_branch", None)
                        node["ref"] = (self.checked["nodes"][node["name"]] if action == "soya"
                                       else self._head(self.service.comfy_root / "custom_nodes" / node["name"]) or node.get("ref") or self.checked["nodes"][node["name"]])
                if action == "soya":
                    data["comfy"]["ref"] = self._head(self.service.comfy_root)
                active = InstallManifest(manifest.source_path, data, hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest())
                paths = ([self.service.comfy_root] if action == "comfy" else []) + [
                    self.service.comfy_root / "custom_nodes" / node["name"] for node in data["custom_nodes"]
                    if node["source_type"] == "git" and ((node["name"] in SOYA_NODE_NAMES) == (action == "soya"))
                ]
                for path in paths:
                    if (path / ".git").is_dir():
                        node = next((node for node in data["custom_nodes"] if node["name"] == path.name), None)
                        target = node["ref"] if node is not None else data["comfy"]["ref"]
                        repository = node["repository"] if node is not None else data["comfy"]["repository"]
                        self._verify_origin(path, repository)
                        if self._head(path) != target:
                            run_command(["git", "fetch", "origin", target], cwd=path, cancel_event=self.service._cancel, log=self.service._log, timeout=900)
                        quarantines.append((path, _prepare_untracked_quarantine(path, log=self.service._log, backup_root=self.backup_root / "untracked" / path.name, target_ref=target)))
                self.service._run_update(install_mode=self.service._installed_install_mode(), component=action, manifest_override=active)
                return
            with self.service._lock:
                self.service._state.update(state="succeeded", finished_at=_now(), result=result, error=None)
            self.service._log(result["message"])
        except Exception as exc:
            self.service._log(str(exc), "error")
            with self.service._lock:
                self.service._state.update(state="cancelled" if self.service._cancel.is_set() else "failed", finished_at=_now(), error=str(exc))
        finally:
            for path, quarantine in reversed(quarantines):
                try:
                    _restore_untracked_quarantine(path, quarantine, log=self.service._log, keep_conflicts=True)
                except Exception as exc:
                    self.service._log(str(exc), "error")
                    with self.service._lock:
                        self.service._state.update(state="failed", error=str(exc))
