from __future__ import annotations

import datetime
import json
import os
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from threading import Event, RLock
from typing import Any, Callable, Mapping

import httpx

from comfy_runtime import (
    comfy_launch_profile_extra_args,
    normalize_comfy_launch_profiles,
)

from .configurator import (
    ConfigUpdateResult,
    apply_installed_config,
    backup_current_config,
    retarget_config_to_embedded_comfy,
    restore_config_backup,
    restore_verified_config_snapshot,
)
from .credentials import (
    CredentialStoreError,
    load_civitai_key,
    save_civitai_key,
    save_lora_manager_civitai_key,
)
from .crypto import ExtractedWorkflowPack
from .dependency_installer import (
    create_comfy_venv,
    install_node_dependencies,
    install_python_dependencies,
    verify_isolated_runtime,
)
from .downloader import DownloadCancelled, ResumableDownloader
from .e2e import (
    E2E_PROFILE_BY_BINDING,
    bypass_sageattention_nodes,
    ComfyE2ECancelled,
    ComfyE2EError,
    ComfyProcess,
    execute_prompt,
    make_e2e_prompt,
    protected_e2e_fixtures,
    promote_generated_fixture,
    validate_all_workflows,
)
from .install_modes import (
    INSTALL_MODE_NVIDIA_COMPATIBILITY,
    INSTALL_MODE_STANDARD,
    compatibility_warning,
    effective_gpu_profile,
    normalize_install_mode,
)
from .manifest import InstallManifest, load_install_manifest
from .input_patcher import patch_comfy_input
from .migration import ComfyMigrationCancelled, migrate_user_data
from .model_installer import install_models
from .node_installer import install_custom_nodes, update_custom_nodes
from .node_compatibility import validate_instant_lora_export_order
from .operations import uv_python_path
from .pack_cli import collect_workflow_bindings
from .runtime_state import (
    RuntimeStateError,
    complete_runtime_transaction,
    create_runtime_transaction,
    git_head,
    inspect_runtime,
    rollback_runtime_transaction,
    write_runtime_receipt,
)
from .source_installer import install_comfy_source, update_comfy_source
from .system_probe import probe_system
from .updater import update_hooking_server_main
from .workflow_library import (
    WorkflowSelection,
    WorkflowLibraryError,
    distribution_e2e_catalog,
    embedded_workflow_base_dir,
    import_default_user_copies,
    import_user_copies,
    library_status,
    latest_release_version,
    migrate_legacy_workflow_layout,
    resolve_distribution_selection,
    selection_requirements,
    unpack_to_library,
)


class InstallerServiceError(RuntimeError):
    """설치 서비스 상태 또는 입력 검증 실패."""


_INSTALL_PHASES = (
    ("preflight", "Windows/GPU/디스크/도구 검사"),
    ("source", "매니페스트 고정 ComfyUI 소스 설치"),
    ("workflows", "선택 워크플로우 사용자 사본 생성"),
    ("credentials", "Civitai 인증 사전 검증"),
    ("venv", "comfy/.venv Python 3.12.11 생성"),
    ("core_dependencies", "PyTorch 및 선택 가속/Comfy 의존성 설치"),
    ("custom_nodes", "고정 커스텀 노드 설치"),
    ("node_dependencies", "커스텀 노드 Python 의존성 설치"),
    ("runtime_isolation", "GPU 및 독립 환경 검증"),
    ("models", "선택 워크플로우 모델 다운로드·검증"),
    ("repatch", "Comfy input 설치 리패치"),
    ("startup", "독립 ComfyUI 기동·노드 로드 확인"),
    ("config", "config.json 백업·설치 경로 적용"),
    ("complete", "설치 결과 기록"),
)

_UPDATE_PHASES = (
    ("config_backup", "config.json 업데이트 전 백업"),
    ("hooking_server", "후킹 서버 origin/main 수동 업데이트"),
    ("manifest", "새 설치 매니페스트 로드"),
    ("inventory", "실제 Comfy/Python/노드 설치 상태 조사"),
    ("runtime_pause", "실행 중인 관리 Comfy 안전 정지"),
    ("transaction", "승격 전 런타임 복원 상태 보관"),
    ("venv", "프로젝트 내부 ComfyUI Python 검증"),
    ("source", "변경된 ComfyUI 고정 소스 업데이트"),
    ("core_dependencies", "변경된 ComfyUI Python/선택 가속 의존성 적용"),
    ("custom_nodes", "변경된 커스텀 노드 업데이트"),
    ("node_dependencies", "변경된 노드 Python 의존성 적용"),
    ("runtime_isolation", "독립 Python/GPU 빠른 검증"),
    ("repatch", "Comfy input 설치 리패치"),
    ("startup", "ComfyUI 기동·노드 로드 확인"),
    ("config", "검증된 내장 Comfy 경로 적용"),
    ("runtime_resume", "업데이트 전 실행 중이던 관리 Comfy 복구"),
    ("complete", "업데이트 결과 기록"),
)

_E2E_PHASES = (
    ("selection", "읽기 전용 배포 원본 선택 확인"),
    ("startup", "독립 ComfyUI 기동·노드 로드 확인"),
    ("e2e_static", "선택 워크플로우 변환·구조 검증"),
    ("e2e_runtime", "선택 레거시 워크플로우 실제 실행"),
    ("e2e_video_runtime", "선택 MiniMax H3 워크플로우 별도 실행"),
    ("complete", "선택 E2E 결과 기록"),
)

_MIGRATE_PHASES = (
    ("migration_backup", "config.json 이사 전 백업"),
    ("migration_scan", "기존 사용자 데이터 확인"),
    ("migration_copy", "LoRA와 봇 캐시 병렬 복사"),
    ("migration_config", "설정 경로를 내장 Comfy로 전환"),
)

_WINDOWS_E2E_EXTRA_ARGS = ("--disable-pinned-memory",)


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class ComfyInstallerService:
    def __init__(
        self,
        *,
        project_root: str | os.PathLike[str],
        config_path: str | os.PathLike[str] | None = None,
        requirements_dir: str | os.PathLike[str] | None = None,
        manifest: InstallManifest | None = None,
        downloader: ResumableDownloader | None = None,
        pause_managed_comfy: Callable[[], Any] | None = None,
        resume_managed_comfy: Callable[[Any], Any] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.comfy_root = self.project_root / "comfy"
        self.config_path = (
            Path(config_path).resolve()
            if config_path is not None
            else self.project_root / "config.json"
        )
        self.requirements_dir = (
            Path(requirements_dir).resolve()
            if requirements_dir is not None
            else self.project_root / "요구사항"
        )
        self.work_root = self.project_root / ".work" / "comfy-installer"
        self.upload_root = self.work_root / "uploads"
        self.workflow_library_root = (
            self.project_root / "comfy_workflow_library"
        )
        self.config_backup_dir = (
            self.comfy_root / ".installer-state" / "backups" / "config"
        )
        self.runtime_backup_dir = (
            self.comfy_root / ".installer-state" / "backups" / "runtime"
        )
        self.manifest = manifest or load_install_manifest()
        self.downloader = downloader or ResumableDownloader(max_retries=4)
        if (pause_managed_comfy is None) != (resume_managed_comfy is None):
            raise InstallerServiceError(
                "관리 Comfy 일시정지/복구 콜백은 반드시 함께 제공해야 합니다."
            )
        self.pause_managed_comfy = pause_managed_comfy
        self.resume_managed_comfy = resume_managed_comfy
        self._lock = RLock()
        self._cancel = Event()
        self._thread: threading.Thread | None = None
        self._log_sequence = 0
        self._logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self._phases = _INSTALL_PHASES
        self._state: dict[str, Any] = {
            "state": "idle",
            "operation": None,
            "install_mode": None,
            "phase": None,
            "phase_index": 0,
            "phase_count": len(self._phases),
            "phase_label": "",
            "started_at": None,
            "finished_at": None,
            "progress": {},
            "error": None,
            "result": None,
            "manifest": {
                "sha256": self.manifest.sha256,
                "comfy_version": self.manifest.comfy["version"],
                "model_count": len(self.manifest.models),
                "model_bytes": sum(
                    int(model["size"]) for model in self.manifest.models
                ),
                "custom_node_count": len(self.manifest.custom_nodes),
                "workflow_count": self.manifest.workflows["expected_count"],
            },
        }
        try:
            self._state["workflow_path_migration"] = (
                migrate_legacy_workflow_layout(
                    comfy_root=self.comfy_root,
                    library_root=self.workflow_library_root,
                    config_path=self.config_path,
                    backup_dir=self.config_backup_dir,
                )
            )
        except Exception as exc:
            print(
                "[COMFY_INSTALL][SERVICE] 워크플로우 ASCII 경로 "
                f"마이그레이션 실패: {exc}"
            )
            traceback.print_exc()
            self._state["workflow_path_migration"] = {
                "error": str(exc),
            }

    def _log(
        self,
        message: str,
        level: str = "info",
        *,
        echo: bool = True,
    ) -> None:
        clean_message = str(message).replace("\x00", "")
        with self._lock:
            self._log_sequence += 1
            entry = {
                "seq": self._log_sequence,
                "time": _now_iso(),
                "level": level,
                "message": clean_message,
            }
            self._logs.append(entry)
        if echo:
            print(
                f"[COMFY_INSTALL][SERVICE][{level.upper()}] "
                f"{clean_message}"
            )

    def _log_comfy(self, message: str) -> None:
        clean_message = str(message).replace("\x00", "")
        level = (
            "warning"
            if clean_message.startswith("[Comfy][WARNING]")
            else "info"
        )
        self._log(clean_message, level, echo=False)

    def _set_phase(self, phase_id: str) -> None:
        for index, (candidate_id, label) in enumerate(self._phases, 1):
            if candidate_id == phase_id:
                with self._lock:
                    self._state.update(
                        {
                            "phase": phase_id,
                            "phase_index": index,
                            "phase_label": label,
                            "progress": {},
                        }
                    )
                self._log(
                    f"[단계 {index}/{len(self._phases)}] {label}"
                )
                return
        raise InstallerServiceError(
            f"정의되지 않은 설치 단계입니다: {phase_id}"
        )

    def _set_progress(self, payload: dict) -> None:
        safe_payload = {
            str(key): value
            for key, value in payload.items()
            if str(key).casefold()
            not in {"authorization", "civitai_key", "workflow_key"}
        }
        with self._lock:
            self._state["progress"] = safe_payload

    def status(self, *, since: int = 0) -> dict:
        with self._lock:
            snapshot = json.loads(
                json.dumps(self._state, ensure_ascii=False)
            )
            snapshot["logs"] = [
                dict(entry)
                for entry in self._logs
                if int(entry["seq"]) > int(since)
            ]
            snapshot["last_log_seq"] = self._log_sequence
            snapshot["cancel_requested"] = self._cancel.is_set()
            snapshot["install_root"] = str(self.comfy_root)
            return snapshot

    def preflight(
        self,
        *,
        selected_model_bytes: int | None = None,
        require_disk: bool = True,
        install_mode: str = INSTALL_MODE_STANDARD,
    ) -> dict:
        try:
            mode = normalize_install_mode(install_mode)
        except ValueError as exc:
            raise InstallerServiceError(str(exc)) from exc
        runtime_and_buffer = 30 * 1024**3
        required_bytes = runtime_and_buffer + max(
            int(selected_model_bytes or 0), 0
        )
        result = probe_system(
            self.comfy_root,
            self.manifest,
            required_bytes=required_bytes,
            require_disk=require_disk,
            install_mode=mode,
        )
        return {
            **result,
            "manifest": dict(self._state["manifest"]),
        }

    def preflight_selection(
        self,
        *,
        release_version: str,
        selected_item_ids: list[str],
        install_mode: str = INSTALL_MODE_STANDARD,
    ) -> dict:
        requirements = selection_requirements(
            library_root=self.workflow_library_root,
            release_version=release_version,
            selected_item_ids=selected_item_ids,
        )
        return {
            **self.preflight(
                selected_model_bytes=int(requirements["model_bytes"]),
                install_mode=install_mode,
            ),
            "selection": requirements,
        }

    def workflow_library_status(self) -> dict:
        return library_status(
            self.comfy_root,
            self.workflow_library_root,
        )

    def e2e_workflow_catalog(self) -> dict:
        python = uv_python_path(self.comfy_root / ".venv")
        if not python.is_file():
            print(
                "[COMFY_INSTALL][SERVICE] E2E 목록 조회 실패: "
                f"내장 Comfy Python 없음: {python}"
            )
            raise InstallerServiceError(
                "내장 ComfyUI가 설치되지 않았습니다. 먼저 설치를 완료하세요."
            )
        try:
            release_version = latest_release_version(
                self.workflow_library_root
            )
            catalog = distribution_e2e_catalog(
                library_root=self.workflow_library_root,
                release_version=release_version,
                profile_by_binding=E2E_PROFILE_BY_BINDING,
                excluded_filenames=self.manifest.workflows[
                    "excluded_filenames"
                ],
            )
            if not catalog["items"]:
                raise InstallerServiceError(
                    "읽기 전용 배포 원본 중 실행 방법이 등록된 E2E "
                    "워크플로우가 없습니다."
                )
            return catalog
        except WorkflowLibraryError as exc:
            print(f"[COMFY_INSTALL][SERVICE] E2E 원본 목록 조회 실패: {exc}")
            traceback.print_exc()
            raise InstallerServiceError(
                f"E2E 배포 원본 목록을 조회하지 못했습니다: {exc}"
            ) from exc
        except InstallerServiceError as exc:
            print(f"[COMFY_INSTALL][SERVICE] E2E 목록 조회 실패: {exc}")
            raise
        except Exception as exc:
            print(
                "[COMFY_INSTALL][SERVICE] E2E 목록 조회 중 예상하지 못한 "
                f"실패: {type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise InstallerServiceError(
                f"E2E 워크플로우 목록을 조회하지 못했습니다: {exc}"
            ) from exc

    def _prepare_default_workflows(
        self,
        *,
        release_version: str | None = None,
        required: bool = False,
    ) -> WorkflowSelection | None:
        try:
            selection = import_default_user_copies(
                comfy_root=self.comfy_root,
                library_root=self.workflow_library_root,
                release_version=release_version,
                log=self._log,
            )
            self._log(
                "[설정] 기본 워크플로우 사용자 사본 준비 완료: "
                f"release={selection.release_version}, "
                f"bindings={len(selection.workflow_bindings)}"
            )
            return selection
        except WorkflowLibraryError as exc:
            print(
                "[COMFY_INSTALL][SERVICE] 기본 워크플로우를 준비하지 "
                f"못했습니다: release={release_version!r}, error={exc}"
            )
            traceback.print_exc()
            if required:
                raise
            self._log(
                "[설정] 워크플로우 라이브러리가 없어 빈 기본 경로 "
                f"채우기를 건너뜁니다: {exc}",
                "warning",
            )
            return None

    def _embedded_workflow_base_dir(self) -> Path:
        if not self.comfy_root.is_dir() or not (
            self.comfy_root / ".git"
        ).is_dir():
            print(
                "[COMFY_INSTALL][SERVICE] 워크플로우 베이스 경로를 "
                f"설정할 내장 Comfy가 없습니다: {self.comfy_root}"
            )
            raise InstallerServiceError(
                f"내장 Comfy가 설치되지 않았습니다: {self.comfy_root}"
            )
        base_dir = embedded_workflow_base_dir(self.comfy_root)
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(
                "[COMFY_INSTALL][SERVICE] 내장 워크플로우 베이스 폴더 "
                f"생성 실패: path={base_dir}, error={exc}"
            )
            traceback.print_exc()
            raise InstallerServiceError(
                f"내장 워크플로우 베이스 폴더 생성 실패: {exc}"
            ) from exc
        return base_dir

    def unpack_workflow_pack(
        self,
        *,
        workflow_pack: str | os.PathLike[str],
        workflow_key: str,
    ) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise InstallerServiceError(
                    "설치 또는 업데이트 중에는 워크플로우 팩을 풀 수 없습니다."
                )
        if not workflow_key:
            raise InstallerServiceError("워크플로우 팩 키가 비어 있습니다.")
        result = unpack_to_library(
            pack_path=workflow_pack,
            passphrase=workflow_key,
            library_root=self.workflow_library_root,
            work_root=self.work_root,
            manifest=self.manifest,
            validate=self._validate_extracted_pack,
            log=self._log,
        )
        return {
            "unpacked": result,
            "library": self.workflow_library_status(),
        }

    def get_civitai_key(self) -> str:
        return load_civitai_key(self.project_root)

    def set_civitai_key(self, api_key: str) -> dict:
        return save_civitai_key(
            self.project_root,
            api_key,
        )

    def replace_lora_manager_civitai_key(self, api_key: str) -> dict:
        if not isinstance(api_key, str):
            raise InstallerServiceError("api_key는 문자열이어야 합니다.")
        key = api_key.strip()
        if not key:
            print("[COMFY_INSTALL][KEY] 문제 해결 키 교체 거부: 빈 API 키")
            raise InstallerServiceError("교체할 Civitai API 키가 비어 있습니다.")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                print(
                    "[COMFY_INSTALL][KEY] 문제 해결 키 교체 거부: "
                    "설치 또는 업데이트 작업 실행 중"
                )
                raise InstallerServiceError(
                    "설치 또는 업데이트가 끝난 뒤 Civitai 키를 교체하세요."
                )

        try:
            lora_manager = save_lora_manager_civitai_key(self.comfy_root, key)
            installer = save_civitai_key(self.project_root, key)
        except CredentialStoreError as exc:
            print(f"[COMFY_INSTALL][KEY] 문제 해결 키 교체 실패: {exc}")
            traceback.print_exc()
            raise InstallerServiceError(str(exc)) from exc

        self._log(
            "[문제 해결] Civitai 키 교체 완료: "
            "설치기와 LoRA Manager 설정 동기화, ComfyUI 재시작 필요"
        )
        return {
            "key_set": True,
            "installer_path": installer["path"],
            "lora_manager_path": lora_manager["path"],
            "backup_paths": {
                "installer": installer["backup_path"],
                "lora_manager": lora_manager["backup_path"],
            },
            "restart_required": True,
        }

    def migrate_from_existing_comfy(
        self, old_comfy_root: str | os.PathLike[str]
    ) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise InstallerServiceError(
                    "다른 ComfyUI 작업 중에는 사용자 데이터를 이사할 수 없습니다."
                )
        return self._perform_migration(old_comfy_root)

    def _perform_migration(
        self,
        old_comfy_root: str | os.PathLike[str],
        *,
        progress: Callable[[dict], None] | None = None,
        cancel_event: Event | None = None,
        set_phase: Callable[[str], None] | None = None,
    ) -> dict:
        copy_phase_started = False

        def advance(phase_id: str) -> None:
            if set_phase is not None:
                set_phase(phase_id)

        def migration_progress(payload: dict) -> None:
            nonlocal copy_phase_started
            if (
                payload.get("event") == "migration_copy"
                and not copy_phase_started
            ):
                copy_phase_started = True
                advance("migration_copy")
            if progress is not None:
                progress(payload)

        try:
            advance("migration_backup")
            config_backup = backup_current_config(
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                reason="comfy_v4_migrate",
            )
            self._log(
                "[이사] config.json 백업 완료: "
                f"{config_backup['backup_path']}"
            )
            if cancel_event is not None and cancel_event.is_set():
                print(
                    "[COMFY_INSTALL][SERVICE] config 백업 후 사용자 데이터 이사 중단"
                )
                raise ComfyMigrationCancelled(
                    "사용자 데이터 이사를 중단했습니다."
                )
            advance("migration_scan")
            migration = migrate_user_data(
                old_comfy_root=old_comfy_root,
                new_comfy_root=self.comfy_root,
                log=self._log,
                progress=migration_progress,
                cancel_event=cancel_event,
            )
            if cancel_event is not None and cancel_event.is_set():
                print(
                    "[COMFY_INSTALL][SERVICE] 설정 경로 전환 전 사용자 데이터 이사 중단"
                )
                raise ComfyMigrationCancelled(
                    "사용자 데이터 이사를 중단했습니다."
                )
            advance("migration_config")
            default_workflows = self._prepare_default_workflows()
            workflow_base_dir = self._embedded_workflow_base_dir()
            config_update = retarget_config_to_embedded_comfy(
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                backup_path=config_backup["backup_path"],
                old_comfy_root=old_comfy_root,
                new_comfy_root=self.comfy_root,
                default_workflow_bindings=(
                    default_workflows.workflow_bindings
                    if default_workflows is not None
                    else None
                ),
                workflow_base_dir=workflow_base_dir,
            )
            if config_update.already_retargeted:
                self._log("[이사] config.json은 이미 내장 Comfy 경로입니다.")
            else:
                self._log(
                    "[이사] config.json 내장 Comfy 경로 전환 완료: "
                    f"{len(config_update.updated_paths)}개"
                )
            return {
                **migration,
                "config": self._config_retarget_payload(config_update),
            }
        except ComfyMigrationCancelled:
            raise
        except Exception as exc:
            print(f"[COMFY_INSTALL][SERVICE] V4 사용자 이사 실패: {exc}")
            traceback.print_exc()
            self._log(f"[이사 실패] {exc}", "error")
            if isinstance(exc, InstallerServiceError):
                raise
            raise InstallerServiceError(f"V4 사용자 이사 실패: {exc}") from exc

    @staticmethod
    def _config_retarget_payload(config_update) -> dict:
        return {
            "backup_path": str(config_update.backup_path),
            "before_sha256": config_update.before_sha256,
            "after_sha256": config_update.after_sha256,
            "updated_paths": list(config_update.updated_paths),
            "already_retargeted": config_update.already_retargeted,
            "missing_targets": [
                {"setting": setting, "target": target}
                for setting, target in config_update.missing_targets
            ],
        }

    def retarget_config_to_embedded(self) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise InstallerServiceError(
                    "다른 ComfyUI 작업 중에는 config.json 경로를 변경할 수 없습니다."
                )
        try:
            default_workflows = self._prepare_default_workflows()
            workflow_base_dir = self._embedded_workflow_base_dir()
            config_backup = backup_current_config(
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                reason="comfy_embedded_retarget",
            )
            config_update = retarget_config_to_embedded_comfy(
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                backup_path=config_backup["backup_path"],
                old_comfy_root=None,
                new_comfy_root=self.comfy_root,
                default_workflow_bindings=(
                    default_workflows.workflow_bindings
                    if default_workflows is not None
                    else None
                ),
                workflow_base_dir=workflow_base_dir,
            )
            if config_update.already_retargeted:
                self._log("[설정] config.json은 이미 내장 Comfy 경로입니다.")
            else:
                self._log(
                    "[설정] config.json 내장 Comfy 경로 전환 완료: "
                    f"{len(config_update.updated_paths)}개"
                )
            return {"config": self._config_retarget_payload(config_update)}
        except Exception as exc:
            print(
                "[COMFY_INSTALL][SERVICE] config.json 내장 Comfy 경로 전환 실패: "
                f"{exc}"
            )
            traceback.print_exc()
            self._log(f"[설정 전환 실패] {exc}", "error")
            if isinstance(exc, InstallerServiceError):
                raise
            raise InstallerServiceError(
                f"config.json 내장 Comfy 경로 전환 실패: {exc}"
            ) from exc

    def _validate_civitai_access(
        self, civitai_key: str, models: list[dict] | None = None
    ) -> None:
        candidates = models if models is not None else self.manifest.models
        sample = next(
            (
                model
                for model in candidates
                if model.get("auth") == "civitai"
            ),
            None,
        )
        if sample is None:
            self._log(
                "[인증] Civitai 인증 대상이 없어 키 검사를 건너뜁니다."
            )
            return
        if not civitai_key.strip():
            print(
                "[COMFY_INSTALL][AUTH] Civitai API 키가 비어 있습니다."
            )
            raise InstallerServiceError("Civitai API 키가 비어 있습니다.")
        try:
            with httpx.Client(
                timeout=httpx.Timeout(30, connect=15),
                follow_redirects=True,
                headers={
                    "User-Agent": "comfyui-hooking-server-installer/1.0",
                    "Authorization": f"Bearer {civitai_key.strip()}",
                    "Range": "bytes=0-0",
                },
            ) as client:
                with client.stream("GET", str(sample["url"])) as response:
                    if response.status_code not in {200, 206}:
                        body = response.read()[:1000].decode(
                            "utf-8", errors="replace"
                        )
                        raise InstallerServiceError(
                            "Civitai API 키/다운로드 권한 검증 실패: "
                            f"status={response.status_code}, body={body}"
                        )
                    next(response.iter_bytes(1), b"")
            self._log("[인증] Civitai 모델 다운로드 권한 확인 완료")
        except InstallerServiceError:
            raise
        except Exception as exc:
            print(
                "[COMFY_INSTALL][AUTH] Civitai 인증 사전 검사 실패: "
                f"error={exc}"
            )
            traceback.print_exc()
            raise InstallerServiceError(
                f"Civitai 인증 사전 검사 실패: {exc}"
            ) from exc

    def _validate_extracted_pack(
        self, extracted: ExtractedWorkflowPack
    ) -> None:
        required = set(self.manifest.workflows["required_bindings"])
        required.update(self.manifest.workflows.get("optional_bindings", []))
        actual = set(extracted.workflow_bindings)
        if actual != required:
            raise InstallerServiceError(
                "워크플로우 팩 바인딩이 매니페스트와 다릅니다: "
                f"missing={sorted(required - actual)}, "
                f"extra={sorted(actual - required)}"
            )
        unique = {
            Path(path).resolve()
            for path in extracted.workflow_bindings.values()
        }
        expected_count = int(self.manifest.workflows["expected_count"])
        if len(unique) != expected_count:
            raise InstallerServiceError(
                "복호화 워크플로우 고유 파일 수가 다릅니다: "
                f"expected={expected_count}, actual={len(unique)}"
            )
        excluded = {
            str(name).casefold()
            for name in self.manifest.workflows["excluded_filenames"]
        }
        invalid = [
            path.name for path in unique if path.name.casefold() in excluded
        ]
        if invalid:
            raise InstallerServiceError(
                "배포 제외 워크플로우가 팩에 포함되었습니다: "
                + ", ".join(invalid)
            )

    def _video_e2e_extra_args(self) -> tuple[str, ...]:
        try:
            config: dict[str, Any] = {}
            if self.config_path.is_file():
                raw_config = json.loads(
                    self.config_path.read_text(encoding="utf-8")
                )
                if not isinstance(raw_config, dict):
                    raise InstallerServiceError(
                        "config.json 최상위 값이 객체가 아닙니다."
                    )
                config = raw_config
            profiles = normalize_comfy_launch_profiles(
                config.get("comfy_launch_profiles")
            )
            arguments = comfy_launch_profile_extra_args(profiles["3"])
            self._log(
                "[H3 E2E] Comfy #3 독립 실행 옵션 적용: "
                + (" ".join(arguments) if arguments else "기본 DynamicVRAM")
            )
            return arguments
        except Exception as exc:
            print(
                "[COMFY_INSTALL][SERVICE] Comfy #3 H3 E2E 옵션 읽기 실패: "
                f"config={self.config_path}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            if isinstance(exc, InstallerServiceError):
                raise
            raise InstallerServiceError(
                f"Comfy #3 H3 E2E 옵션 읽기 실패: {exc}"
            ) from exc

    def _partition_runtime_validations(
        self,
        validations: list,
    ) -> tuple[list, list]:
        profile = self.manifest.validation_profiles.get("minimax_h3", {})
        video_bindings = set(profile.get("workflow_bindings", []))
        legacy: list = []
        video: list = []
        for validation in validations:
            binding_keys = set(validation.binding_keys)
            matched = binding_keys.intersection(video_bindings)
            if not matched:
                legacy.append(validation)
                continue
            unexpected = binding_keys - video_bindings
            if unexpected:
                raise InstallerServiceError(
                    "H3 워크플로우와 레거시 바인딩이 한 파일에 섞였습니다: "
                    f"filename={validation.filename}, "
                    f"bindings={sorted(binding_keys)}"
                )
            video.append(validation)
        return legacy, video

    @staticmethod
    def _runtime_order(validation) -> tuple[int, str]:
        keys = set(validation.binding_keys)
        priorities = {
            "illustration_workflow_source_paths.v1": 10,
            "comfy_workflow_source_path": 20,
            "illustration_workflow_source_paths.v3": 30,
            "asset_workflow_source_path": 40,
            "anima_asset_workflow_source_path": 50,
            "anima_only_asset_workflow_source_path": 60,
            "qwen_edit_workflow_source_path": 70,
            "anima_inpainting_workflow_source_path": 80,
            "tag_analysis_workflow_source_path": 90,
            "asset_tag_analysis_workflow_source_path": 100,
            "utility_workflow_source_path": 110,
            "face_extract_workflow_source_path": 120,
            "lora_training_workflow_source_paths.anima": 130,
            "lora_training_workflow_source_paths.sdxl": 140,
            "style_lora_training_workflow_source_paths.anima": 150,
            "style_lora_training_workflow_source_paths.sdxl": 160,
            "debug_workflow_source_path": 170,
            "video_workflow_source_paths.i2v": 180,
            "video_workflow_source_paths.first_last": 190,
            "video_workflow_source_paths.i2v_fast": 200,
            "video_workflow_source_paths.first_last_fast": 210,
        }
        return (
            min((priorities.get(key, 999) for key in keys), default=999),
            validation.filename.casefold(),
        )

    def _start_e2e_process(
        self,
        *,
        python: Path,
        extra_args: tuple[str, ...] = (),
    ) -> ComfyProcess:
        process = ComfyProcess(
            comfy_root=self.comfy_root,
            python=python,
            cancel_event=self._cancel,
            log=self._log_comfy,
            extra_args=extra_args,
        )
        try:
            process.start(timeout=900)
            return process
        except Exception as exc:
            print(
                "[COMFY_INSTALL][SERVICE] 격리 E2E ComfyUI 기동 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            process.stop()
            raise

    def _run_runtime_e2e(
        self,
        *,
        process: ComfyProcess,
        validations: list,
        fixtures: Mapping[str, str],
        bypass_sageattention: bool = False,
        process_factory: Callable[[], ComfyProcess] | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        failures: list[dict[str, str]] = []
        promoted = False
        active_process = process
        ordered = sorted(validations, key=self._runtime_order)
        self._log(
            "[E2E] 설치기 전용 입력 픽스처 준비: "
            f"training={fixtures['training']}, "
            f"face_source={fixtures['face_source']}"
        )
        for index, validation in enumerate(ordered, 1):
            if process_factory is not None and index > 1:
                active_process = process_factory()
                self._log(
                    "[E2E 격리] 새 ComfyUI 프로세스로 워크플로 실행: "
                    f"{validation.filename}"
                )
            if self._cancel.is_set():
                raise ComfyE2ECancelled(
                    "실제 워크플로우 E2E 중 중단 요청을 받았습니다."
                )
            self._set_progress(
                {
                    "event": "workflow_execution",
                    "current": index,
                    "total": len(ordered),
                    "filename": validation.filename,
                }
            )
            self._log(
                f"[E2E 실행 {index}/{len(ordered)}] "
                f"{validation.filename}"
            )
            workflow_started = time.monotonic()
            h3_profile = self.manifest.validation_profiles.get(
                "minimax_h3",
                {},
            )
            h3_bindings = set(h3_profile.get("workflow_bindings", []))
            fast_h3_bindings = set(
                h3_profile.get("fast_workflow_bindings", [])
            )
            is_h3_validation = bool(
                set(validation.binding_keys).intersection(h3_bindings)
            )
            try:
                try:
                    if is_h3_validation:
                        is_fast_h3_validation = bool(
                            set(validation.binding_keys).intersection(
                                fast_h3_bindings
                            )
                        )
                        defaults_key = (
                            "fast_defaults"
                            if is_fast_h3_validation
                            else "defaults"
                        )
                        h3_defaults = h3_profile.get(defaults_key, {})
                        prompt = make_e2e_prompt(
                            validation,
                            sample_steps=int(h3_defaults["steps"]),
                            sample_width=int(h3_defaults["width"]),
                            sample_height=int(h3_defaults["height"]),
                        )
                    else:
                        prompt = make_e2e_prompt(validation)
                    bypassed_nodes: list[dict[str, str]] = []
                    if bypass_sageattention:
                        prompt, bypassed_nodes = bypass_sageattention_nodes(
                            prompt,
                            filename=validation.filename,
                        )
                        if bypassed_nodes:
                            self._log(
                                "[E2E 호환] SageAttention 노드를 검증 사본에서 "
                                f"우회: filename={validation.filename}, "
                                f"count={len(bypassed_nodes)}"
                            )
                    timeout = (
                        7200
                        if any(
                            "training_workflow" in key
                            for key in validation.binding_keys
                        )
                        else 3600
                    )
                    result = execute_prompt(
                        base_url=active_process.base_url,
                        prompt=prompt,
                        workflow=validation.workflow,
                        filename=validation.filename,
                        cancel_event=self._cancel,
                        log=self._log,
                        timeout=timeout,
                        fatal_error=getattr(active_process, "fatal_error", None),
                    )
                    if bypassed_nodes:
                        result["compatibility_bypassed_nodes"] = bypassed_nodes
                    result["duration_seconds"] = round(
                        time.monotonic() - workflow_started,
                        3,
                    )
                except ComfyE2ECancelled:
                    raise
                except ComfyE2EError as exc:
                    detail = str(exc)
                    if len(detail) > 4000:
                        detail = detail[:4000] + "... (상세 오류 생략)"
                    failures.append(
                        {
                            "filename": validation.filename,
                            "error": detail,
                        }
                    )
                    self._log(
                        "[E2E 실행] 실패 기록 후 다음 워크플로우 계속: "
                        f"{validation.filename}: {detail}",
                        "error",
                    )
                    if process_factory is None:
                        child = active_process.process
                        if child is None or child.poll() is not None:
                            raise ComfyE2EError(
                                "워크플로우 실패 후 ComfyUI 프로세스가 종료되어 "
                                "남은 E2E를 계속할 수 없습니다: "
                                f"{validation.filename}"
                            ) from exc
                    continue
                results.append(
                    {
                        key: value
                        for key, value in result.items()
                        if key != "output_data"
                    }
                )
                if is_h3_validation:
                    self._log(
                        "[H3 E2E] 영상 출력은 이미지 픽스처 승격 대상이 "
                        f"아니므로 생략: {validation.filename}"
                    )
                elif not promoted:
                    promoted_path = promote_generated_fixture(
                        base_url=active_process.base_url,
                        execution_result=result,
                        comfy_root=self.comfy_root,
                    )
                    if promoted_path:
                        promoted = True
                        self._log(
                            "[E2E] 첫 생성 이미지를 후속 태그/편집/학습 "
                            f"픽스처로 적용: {promoted_path}"
                        )
            finally:
                if process_factory is not None:
                    active_process.stop()
        self._set_progress(
            {
                "event": "workflow_execution",
                "current": len(ordered),
                "total": len(ordered),
                "filename": "전체 검사 완료",
                "succeeded": len(results),
                "failed": len(failures),
                "failed_filenames": [
                    failure["filename"] for failure in failures
                ],
            }
        )
        if failures:
            failed_names = ", ".join(
                failure["filename"] for failure in failures
            )
            raise ComfyE2EError(
                "워크플로우 실제 실행 전체 검사 완료: "
                f"성공 {len(results)}/{len(ordered)}, "
                f"실패 {len(failures)}/{len(ordered)}; "
                f"실패 파일={failed_names}. "
                "각 파일의 상세 원인은 E2E 실패 로그를 확인하세요."
            )
        if len(results) != len(validations):
            raise ComfyE2EError(
                "선택 워크플로우 실제 E2E 성공 수가 다릅니다: "
                f"expected={len(validations)}, actual={len(results)}"
            )
        return results

    def _start_operation(
        self,
        *,
        operation: str,
        phases: tuple[tuple[str, str], ...],
        target,
        kwargs: dict,
    ) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise InstallerServiceError(
                    "ComfyUI 설치·업데이트·E2E·이사 작업이 이미 진행 중입니다."
                )
            self._cancel = Event()
            self._logs.clear()
            self._log_sequence = 0
            self._phases = phases
            self._state.update(
                {
                    "state": "running",
                    "operation": operation,
                    "install_mode": kwargs.get("install_mode"),
                    "selected_item_ids": (
                        list(kwargs.get("selected_item_ids", []))
                        if operation == "e2e"
                        else None
                    ),
                    "phase": None,
                    "phase_index": 0,
                    "phase_count": len(phases),
                    "phase_label": "",
                    "started_at": _now_iso(),
                    "finished_at": None,
                    "progress": {},
                    "error": None,
                    "result": None,
                }
            )
            self._thread = threading.Thread(
                target=target,
                kwargs=kwargs,
                name=f"comfy-{operation}",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def start_install(
        self,
        *,
        release_version: str,
        selected_item_ids: list[str],
        install_mode: str = INSTALL_MODE_STANDARD,
    ) -> dict:
        if not isinstance(release_version, str) or not release_version:
            raise InstallerServiceError("설치할 워크플로우 팩 버전이 비어 있습니다.")
        if not isinstance(selected_item_ids, list) or not selected_item_ids:
            raise InstallerServiceError("설치할 워크플로우를 하나 이상 선택하세요.")
        try:
            mode = normalize_install_mode(install_mode)
        except ValueError as exc:
            raise InstallerServiceError(str(exc)) from exc
        return self._start_operation(
            operation="install",
            phases=_INSTALL_PHASES,
            target=self._run_install,
            kwargs={
                "release_version": release_version,
                "selected_item_ids": [str(value) for value in selected_item_ids],
                "install_mode": mode,
            },
        )

    def start_update(self) -> dict:
        python = uv_python_path(self.comfy_root / ".venv")
        if not python.is_file():
            raise InstallerServiceError(
                f"내장 ComfyUI가 설치되지 않았습니다. 먼저 설치하기를 사용하세요: {python}"
            )
        install_mode = self._installed_install_mode()
        return self._start_operation(
            operation="update",
            phases=_UPDATE_PHASES,
            target=self._run_update,
            kwargs={"install_mode": install_mode},
        )

    def start_e2e(
        self,
        *,
        release_version: str,
        selected_item_ids: list[str],
    ) -> dict:
        if not isinstance(release_version, str) or not release_version:
            raise InstallerServiceError("E2E 배포 버전이 비어 있습니다.")
        if not isinstance(selected_item_ids, list) or not selected_item_ids:
            raise InstallerServiceError(
                "E2E 검사할 배포 워크플로우를 하나 이상 선택하세요."
            )
        if not all(
            isinstance(value, str) and value for value in selected_item_ids
        ):
            raise InstallerServiceError(
                "E2E selected_item_ids는 비어 있지 않은 문자열 배열이어야 합니다."
            )
        catalog = self.e2e_workflow_catalog()
        if catalog["release_version"] != release_version:
            raise InstallerServiceError(
                "현재 읽기 전용 배포 원본 버전과 E2E 요청 버전이 "
                "다릅니다: "
                f"available={catalog['release_version']}, "
                f"requested={release_version}"
            )
        available_ids = {str(item["id"]) for item in catalog["items"]}
        requested = list(dict.fromkeys(selected_item_ids))
        unavailable = [value for value in requested if value not in available_ids]
        if unavailable:
            raise InstallerServiceError(
                "원본 파일 또는 등록된 검사 방법이 없는 워크플로우는 "
                "E2E 검사할 수 없습니다: "
                + ", ".join(unavailable)
            )
        return self._start_operation(
            operation="e2e",
            phases=_E2E_PHASES,
            target=self._run_e2e,
            kwargs={
                "release_version": release_version,
                "selected_item_ids": requested,
            },
        )

    def start_migration(
        self, old_comfy_root: str | os.PathLike[str]
    ) -> dict:
        old_root = str(old_comfy_root).strip()
        if not old_root:
            raise InstallerServiceError("기존 ComfyUI 경로가 비어 있습니다.")
        return self._start_operation(
            operation="migrate",
            phases=_MIGRATE_PHASES,
            target=self._run_migration,
            kwargs={"old_comfy_root": old_root},
        )

    def _run_e2e(
        self,
        *,
        release_version: str,
        selected_item_ids: list[str],
    ) -> None:
        process: ComfyProcess | None = None
        legacy_runtime_e2e: list[dict] = []
        video_runtime_e2e: list[dict] = []
        started_monotonic = time.monotonic()
        try:
            self._set_phase("selection")
            selection = resolve_distribution_selection(
                library_root=self.workflow_library_root,
                release_version=release_version,
                selected_item_ids=selected_item_ids,
            )
            self._log(
                "[E2E] 설치된 배포 원본 선택 완료: "
                f"release={release_version}, selected="
                f"{len(selection.selected_item_ids)}"
            )
            python = uv_python_path(self.comfy_root / ".venv")
            if not python.is_file():
                raise InstallerServiceError(
                    f"E2E를 실행할 내장 Comfy Python이 없습니다: {python}"
                )
            install_mode = self._installed_install_mode()

            with protected_e2e_fixtures(
                comfy_root=self.comfy_root,
                requirements_dir=self.runtime_backup_dir / "e2e-fixtures",
            ) as fixtures:
                try:
                    self._set_phase("startup")
                    process = ComfyProcess(
                        comfy_root=self.comfy_root,
                        python=python,
                        cancel_event=self._cancel,
                        log=self._log_comfy,
                        extra_args=_WINDOWS_E2E_EXTRA_ARGS,
                    )
                    stats = process.start(timeout=900)
                    actual_version = (
                        stats.get("system", {}).get("comfyui_version")
                        if isinstance(stats, dict)
                        else None
                    )
                    if actual_version != self.manifest.comfy["version"]:
                        raise ComfyE2EError(
                            "E2E ComfyUI 버전이 매니페스트와 다릅니다: "
                            f"expected={self.manifest.comfy['version']}, "
                            f"actual={actual_version}"
                        )
                    actual_ref = git_head(self.comfy_root)
                    expected_ref = str(self.manifest.comfy["ref"]).lower()
                    if actual_ref != expected_ref:
                        raise ComfyE2EError(
                            "E2E ComfyUI Git SHA가 매니페스트와 다릅니다: "
                            f"expected={expected_ref}, actual={actual_ref}"
                        )

                    self._set_phase("e2e_static")
                    validations, _ = validate_all_workflows(
                        base_url=process.base_url,
                        workflow_bindings=selection.workflow_bindings,
                        expected_count=len(selection.selected_item_ids),
                        excluded_filenames=list(
                            self.manifest.workflows["excluded_filenames"]
                        ),
                        cancel_event=self._cancel,
                        log=self._log,
                        progress=self._set_progress,
                    )
                    legacy_validations, video_validations = (
                        self._partition_runtime_validations(validations)
                    )
                    if legacy_validations:
                        self._set_phase("e2e_runtime")
                        legacy_runtime_e2e = self._run_runtime_e2e(
                            process=process,
                            validations=legacy_validations,
                            fixtures=fixtures,
                            bypass_sageattention=(
                                install_mode
                                == INSTALL_MODE_NVIDIA_COMPATIBILITY
                            ),
                            process_factory=lambda: self._start_e2e_process(
                                python=python,
                                extra_args=_WINDOWS_E2E_EXTRA_ARGS,
                            ),
                        )
                    else:
                        self._log(
                            "[E2E] 선택된 레거시 워크플로우가 없어 실제 "
                            "실행을 건너뜁니다."
                        )

                    if video_validations:
                        process.stop()
                        process = self._start_e2e_process(
                            python=python,
                            extra_args=(
                                *_WINDOWS_E2E_EXTRA_ARGS,
                                *self._video_e2e_extra_args(),
                            ),
                        )
                        self._set_phase("e2e_video_runtime")
                        video_runtime_e2e = self._run_runtime_e2e(
                            process=process,
                            validations=video_validations,
                            fixtures=fixtures,
                            bypass_sageattention=(
                                install_mode
                                == INSTALL_MODE_NVIDIA_COMPATIBILITY
                            ),
                            process_factory=lambda: self._start_e2e_process(
                                python=python,
                                extra_args=(
                                    *_WINDOWS_E2E_EXTRA_ARGS,
                                    *self._video_e2e_extra_args(),
                                ),
                            ),
                        )
                    else:
                        self._log(
                            "[H3 E2E] 선택된 영상 워크플로우가 없어 실제 "
                            "실행을 건너뜁니다."
                        )
                finally:
                    if process is not None:
                        process.stop()
                        process = None

            runtime_e2e = legacy_runtime_e2e + video_runtime_e2e
            self._set_phase("complete")
            result = {
                "operation": "e2e",
                "release_version": selection.release_version,
                "selected_workflow_ids": list(selection.selected_item_ids),
                "install_mode": install_mode,
                "completed_at": _now_iso(),
                "duration_seconds": round(
                    time.monotonic() - started_monotonic, 3
                ),
                "comfy_version": actual_version,
                "comfy_ref": actual_ref,
                "workflow_static": [
                    validation.public_result() for validation in validations
                ],
                "workflow_runtime": runtime_e2e,
                "workflow_runtime_sections": {
                    "legacy": legacy_runtime_e2e,
                    "minimax_h3": video_runtime_e2e,
                },
            }
            result_path = self._write_result(result, prefix="e2e-result")
            result["result_path"] = str(result_path)
            with self._lock:
                self._state.update(
                    {
                        "state": "succeeded",
                        "finished_at": _now_iso(),
                        "progress": {
                            "event": "complete",
                            "current": len(runtime_e2e),
                            "total": len(selection.selected_item_ids),
                        },
                        "error": None,
                        "result": result,
                    }
                )
            self._log(
                "[완료] 선택한 배포 워크플로우 E2E 성공: "
                f"{len(runtime_e2e)}개, {result_path}"
            )
        except ComfyE2ECancelled as exc:
            self._log(f"[중단] {exc}", "warning")
            with self._lock:
                self._state.update(
                    {
                        "state": "cancelled",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                    }
                )
        except Exception as exc:
            print(f"[COMFY_INSTALL][SERVICE] 독립 E2E 검사 실패: {exc}")
            traceback.print_exc()
            self._log(f"[실패] {exc}", "error")
            with self._lock:
                self._state.update(
                    {
                        "state": "failed",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                    }
                )
        finally:
            if process is not None:
                process.stop()

    def _run_migration(self, *, old_comfy_root: str) -> None:
        started_monotonic = time.monotonic()
        try:
            migration = self._perform_migration(
                old_comfy_root,
                progress=self._set_progress,
                cancel_event=self._cancel,
                set_phase=self._set_phase,
            )
            result = {
                "operation": "migrate",
                "completed_at": _now_iso(),
                "duration_seconds": round(
                    time.monotonic() - started_monotonic, 3
                ),
                **migration,
            }
            copied_count = len(result.get("copied", []))
            skipped_count = len(result.get("skipped", []))
            copied_bytes = int(result.get("pending_bytes", 0))
            with self._lock:
                self._state.update(
                    {
                        "state": "succeeded",
                        "finished_at": _now_iso(),
                        "progress": {
                            "event": "complete",
                            "engine": result.get("copy_engine"),
                            "current": copied_count,
                            "total": copied_count,
                            "overall_downloaded": copied_bytes,
                            "overall_total": copied_bytes,
                            "bytes_per_second": 0,
                            "eta_seconds": 0,
                        },
                        "error": None,
                        "result": result,
                    }
                )
            self._log(
                "[완료] V4 사용자 데이터 이사 성공: "
                f"engine={result.get('copy_engine')}, "
                f"copied={copied_count}, skipped={skipped_count}"
            )
        except ComfyMigrationCancelled as exc:
            self._log(f"[중단] {exc}", "warning")
            with self._lock:
                self._state.update(
                    {
                        "state": "cancelled",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                    }
                )
        except Exception as exc:
            print(f"[COMFY_INSTALL][SERVICE] 사용자 데이터 이사 실패: {exc}")
            traceback.print_exc()
            self._log(f"[실패] {exc}", "error")
            with self._lock:
                self._state.update(
                    {
                        "state": "failed",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                    }
                )

    def cancel(self) -> dict:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                print(
                    "[COMFY_INSTALL][SERVICE] 중단 요청을 받았지만 진행 중인 "
                    "설치·업데이트·이사 작업이 없습니다."
                )
                raise InstallerServiceError("진행 중인 ComfyUI 작업이 없습니다.")
            self._cancel.set()
        self._log("[중단] 안전한 중단을 요청했습니다.", "warning")
        return self.status()

    def _write_result(
        self,
        result: dict,
        *,
        prefix: str = "install-result",
    ) -> Path:
        state_root = self.comfy_root / ".installer-state"
        state_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        result_path = state_root / f"{prefix}-{stamp}.json"
        payload = (
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        part = result_path.with_name(f"{result_path.name}.part")
        try:
            with part.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(part, result_path)
            return result_path
        except Exception as exc:
            print(
                "[COMFY_INSTALL][SERVICE] 설치 결과 기록 실패: "
                f"path={result_path}, error={exc}"
            )
            traceback.print_exc()
            raise InstallerServiceError(
                f"설치 결과 기록 실패: {exc}"
            ) from exc

    def _run_install(
        self,
        *,
        release_version: str,
        selected_item_ids: list[str],
        install_mode: str,
    ) -> None:
        process: ComfyProcess | None = None
        config_update: ConfigUpdateResult | None = None
        selection: WorkflowSelection | None = None
        civitai_key = ""
        started_monotonic = time.monotonic()
        try:
            warning = compatibility_warning(install_mode)
            if warning:
                self._log(f"[호환 설치 안내] {warning}", "warning")
            selection_info = selection_requirements(
                library_root=self.workflow_library_root,
                release_version=release_version,
                selected_item_ids=selected_item_ids,
            )
            self._set_phase("preflight")
            system = self.preflight(
                selected_model_bytes=int(selection_info["model_bytes"]),
                install_mode=install_mode,
            )
            self._log(
                "[검사] GPU 프로필 선택: "
                f"{system['gpu_profile']}, "
                f"free={system['disk']['free'] / 1024**3:.2f} GiB, "
                f"selected_models={len(selection_info['model_ids'])}, "
                f"required={system['disk']['required'] / 1024**3:.2f} GiB"
            )

            self._set_phase("source")
            install_comfy_source(
                destination=self.comfy_root,
                repository=str(self.manifest.comfy["repository"]),
                ref=str(self.manifest.comfy["ref"]),
                cancel_event=self._cancel,
                log=self._log,
                requirements_dir=self.runtime_backup_dir,
            )

            self._set_phase("workflows")
            selection = import_user_copies(
                comfy_root=self.comfy_root,
                library_root=self.workflow_library_root,
                release_version=release_version,
                selected_item_ids=selected_item_ids,
                log=self._log,
            )
            self._log(
                "[워크플로우] 사용자 사본 준비 완료: "
                f"release={selection.release_version}, "
                f"selected={len(selection.selected_item_ids)}"
            )

            models_by_id = {
                str(model["id"]): model for model in self.manifest.models
            }
            missing_model_ids = [
                model_id
                for model_id in selection.model_ids
                if model_id not in models_by_id
            ]
            if missing_model_ids:
                raise InstallerServiceError(
                    "선택 워크플로우 모델이 설치 매니페스트에 없습니다: "
                    + ", ".join(missing_model_ids)
                )
            selected_models = [
                models_by_id[model_id] for model_id in selection.model_ids
            ]

            self._set_phase("credentials")
            civitai_key = self.get_civitai_key()
            self._validate_civitai_access(civitai_key, selected_models)

            self._set_phase("venv")
            python = create_comfy_venv(
                comfy_root=self.comfy_root,
                python_version=str(self.manifest.python["version"]),
                cancel_event=self._cancel,
                log=self._log,
                requirements_dir=self.runtime_backup_dir,
            )

            base_profile = next(
                profile
                for profile in self.manifest.python["gpu_profiles"]
                if profile["id"] == system["gpu_profile"]
            )
            profile = effective_gpu_profile(base_profile, install_mode)
            self._set_phase("core_dependencies")
            python_result = install_python_dependencies(
                comfy_root=self.comfy_root,
                python=python,
                python_manifest=self.manifest.python,
                gpu_profile=profile,
                downloader=self.downloader,
                cancel_event=self._cancel,
                cache_root=self.comfy_root / ".installer-cache",
                log=self._log,
                progress=self._set_progress,
            )

            self._set_phase("custom_nodes")
            node_paths = install_custom_nodes(
                nodes=self.manifest.custom_nodes,
                comfy_root=self.comfy_root,
                downloader=self.downloader,
                cancel_event=self._cancel,
                log=self._log,
                progress=self._set_progress,
                requirements_dir=self.runtime_backup_dir,
            )

            self._set_phase("node_dependencies")
            node_requirements = install_node_dependencies(
                comfy_root=self.comfy_root,
                python=python,
                node_paths=node_paths,
                compatibility_packages=list(
                    self.manifest.python["compatibility_packages"]
                ),
                cancel_event=self._cancel,
                log=self._log,
            )

            self._set_phase("runtime_isolation")
            runtime = verify_isolated_runtime(
                comfy_root=self.comfy_root,
                python=python,
                gpu_profile=profile,
                cancel_event=self._cancel,
                log=self._log,
            )

            lora_export_order = validate_instant_lora_export_order(
                comfy_root=self.comfy_root,
                python=python,
                cancel_event=self._cancel,
                log=self._log,
            )

            self._set_phase("models")
            model_results = install_models(
                models=selected_models,
                comfy_root=self.comfy_root,
                downloader=self.downloader,
                civitai_key=civitai_key,
                cancel_event=self._cancel,
                log=self._log,
                progress=self._set_progress,
            )

            self._set_phase("repatch")
            repatch = patch_comfy_input(
                comfy_input_dir=self.comfy_root / "input",
                fallback_source=self.project_root / "modes" / "fallback_img",
                log=self._log,
            )

            self._set_phase("startup")
            try:
                process = ComfyProcess(
                    comfy_root=self.comfy_root,
                    python=python,
                    cancel_event=self._cancel,
                    log=self._log_comfy,
                    extra_args=_WINDOWS_E2E_EXTRA_ARGS,
                )
                stats = process.start(timeout=900)
                actual_version = (
                    stats.get("system", {}).get("comfyui_version")
                    if isinstance(stats, dict)
                    else None
                )
                if actual_version != self.manifest.comfy["version"]:
                    raise ComfyE2EError(
                        "기동된 ComfyUI 버전이 고정값과 다릅니다: "
                        f"expected={self.manifest.comfy['version']}, "
                        f"actual={actual_version}"
                    )
                actual_ref = git_head(self.comfy_root)
                expected_ref = str(self.manifest.comfy["ref"]).lower()
                if actual_ref != expected_ref:
                    raise ComfyE2EError(
                        "기동된 ComfyUI Git SHA가 고정값과 다릅니다: "
                        f"expected={expected_ref}, actual={actual_ref}"
                    )
                self._log(
                    "[기동 검사] ComfyUI 버전/SHA 확인 완료: "
                    f"version={actual_version}, ref={actual_ref}"
                )
            finally:
                if process is not None:
                    process.stop()
                    process = None

            self._set_phase("config")
            workflow_base_dir = self._embedded_workflow_base_dir()
            config_update = apply_installed_config(
                config_path=self.config_path,
                requirements_dir=self.config_backup_dir,
                comfy_root=self.comfy_root,
                workflow_bindings=selection.workflow_bindings,
                required_bindings=selection.workflow_bindings.keys(),
                default_workflow_bindings=None,
                workflow_base_dir=workflow_base_dir,
            )

            runtime_receipt = write_runtime_receipt(
                comfy_root=self.comfy_root,
                manifest=self.manifest,
                profile_id=str(system["gpu_profile"]),
                install_mode=install_mode,
                workflow_bindings=dict(selection.workflow_bindings),
                selected_workflow_ids=selection.selected_item_ids,
                release_version=selection.release_version,
            )

            self._set_phase("complete")
            result = {
                "operation": "install",
                "install_mode": install_mode,
                "compatibility_warning": warning,
                "completed_at": _now_iso(),
                "duration_seconds": round(
                    time.monotonic() - started_monotonic, 3
                ),
                "manifest_sha256": self.manifest.sha256,
                "comfy_root": str(self.comfy_root),
                "comfy_version": actual_version,
                "system": system,
                "python": python_result,
                "runtime": runtime,
                "lora_export_order": lora_export_order,
                "custom_nodes": [str(path) for path in node_paths],
                "node_requirements": node_requirements,
                "models": model_results,
                "workflow_release_version": selection.release_version,
                "selected_workflow_ids": list(selection.selected_item_ids),
                "selected_model_ids": list(selection.model_ids),
                "user_workflow_files": list(selection.user_files),
                "workflow_e2e_deferred": True,
                "workflow_static": None,
                "workflow_runtime": None,
                "config": {
                    "backup_path": str(config_update.backup_path),
                    "before_sha256": config_update.before_sha256,
                    "after_sha256": config_update.after_sha256,
                    "restored_after_success": False,
                },
                "repatch": repatch,
                "runtime_receipt": runtime_receipt,
            }
            result_path = self._write_result(result)
            result["result_path"] = str(result_path)
            with self._lock:
                self._state.update(
                    {
                        "state": "succeeded",
                        "finished_at": _now_iso(),
                        "progress": {
                            "event": "complete",
                            "current": len(selection.selected_item_ids),
                            "total": len(selection.selected_item_ids),
                        },
                        "error": None,
                        "result": result,
                    }
                )
            self._log(
                "[완료] ComfyUI 설치 성공. 워크플로우 E2E는 별도 "
                "검사 버튼에서 선택 실행할 수 있습니다: "
                f"{result_path}"
            )
            if warning:
                self._log(f"[호환 설치 완료 안내] {warning}", "warning")
        except (
            ComfyE2ECancelled,
            DownloadCancelled,
        ) as exc:
            self._log(f"[중단] {exc}", "warning")
            with self._lock:
                self._state.update(
                    {
                        "state": "cancelled",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                    }
                )
        except Exception as exc:
            print(f"[COMFY_INSTALL][SERVICE] 설치 실패: {exc}")
            traceback.print_exc()
            self._log(f"[실패] {exc}", "error")
            with self._lock:
                self._state.update(
                    {
                        "state": "failed",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                    }
                )
        finally:
            if process is not None:
                process.stop()
            civitai_key = ""

    def _installed_gpu_profile_id(
        self,
        *,
        install_mode: str = INSTALL_MODE_STANDARD,
    ) -> str:
        state_root = self.comfy_root / ".installer-state"
        if state_root.is_dir():
            for path in sorted(
                state_root.glob("install-result-*.json"), reverse=True
            ):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    python_result = (
                        value.get("python") if isinstance(value, dict) else None
                    )
                    profile_id = (
                        python_result.get("profile")
                        if isinstance(python_result, dict)
                        else None
                    )
                    if isinstance(profile_id, str) and profile_id:
                        return profile_id
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][UPDATE] 기존 설치 결과 읽기 실패: "
                        f"path={path}, error={exc}"
                    )
                    traceback.print_exc()
        print(
            "[COMFY_INSTALL][UPDATE] 설치 결과에서 GPU 프로필을 찾지 못해 "
            "시스템 검사를 다시 실행합니다."
        )
        return str(
            self.preflight(
                require_disk=False,
                install_mode=install_mode,
            )["gpu_profile"]
        )

    def _installed_install_mode(self) -> str:
        state_root = self.comfy_root / ".installer-state"
        if state_root.is_dir():
            for path in sorted(
                state_root.glob("install-result-*.json"), reverse=True
            ):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict):
                        print(
                            "[COMFY_INSTALL][UPDATE] 설치 결과가 객체가 "
                            f"아닙니다: path={path}"
                        )
                        continue
                    raw_mode = value.get("install_mode")
                    if raw_mode is None:
                        return INSTALL_MODE_STANDARD
                    return normalize_install_mode(raw_mode)
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][UPDATE] 기존 설치 모드 읽기 실패: "
                        f"path={path}, error={exc}"
                    )
                    traceback.print_exc()
        print(
            "[COMFY_INSTALL][UPDATE] 설치 결과에서 설치 모드를 찾지 못해 "
            "표준 설치 모드를 사용합니다."
        )
        return INSTALL_MODE_STANDARD

    def _run_update_legacy(self, *, install_mode: str) -> None:
        process: ComfyProcess | None = None
        started_monotonic = time.monotonic()
        old_manifest = self.manifest
        try:
            mode = normalize_install_mode(install_mode)
            warning = compatibility_warning(mode)
            if warning:
                self._log(
                    "[업데이트][호환 설치 유지] SageAttention을 다시 "
                    f"설치하지 않습니다. {warning}",
                    "warning",
                )
            self._set_phase("config_backup")
            config_backup = backup_current_config(
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                reason="hooking_update",
            )

            self._set_phase("hooking_server")
            hooking_result = update_hooking_server_main(
                project_root=self.project_root,
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                cancel_event=self._cancel,
                log=self._log,
                config_backup=config_backup,
            )

            self._set_phase("manifest")
            new_manifest = load_install_manifest()
            self.manifest = new_manifest
            with self._lock:
                self._state["manifest"] = {
                    "sha256": new_manifest.sha256,
                    "comfy_version": new_manifest.comfy["version"],
                    "model_count": len(new_manifest.models),
                    "model_bytes": sum(
                        int(model["size"]) for model in new_manifest.models
                    ),
                    "custom_node_count": len(new_manifest.custom_nodes),
                    "workflow_count": new_manifest.workflows["expected_count"],
                }
            source_changed = old_manifest.comfy != new_manifest.comfy
            python_changed = old_manifest.python != new_manifest.python
            nodes_changed = old_manifest.custom_nodes != new_manifest.custom_nodes
            self._log(
                "[업데이트] 매니페스트 변경 요약: "
                f"comfy={source_changed}, python={python_changed}, "
                f"nodes={nodes_changed}"
            )

            self._set_phase("venv")
            python = create_comfy_venv(
                comfy_root=self.comfy_root,
                python_version=str(new_manifest.python["version"]),
                cancel_event=self._cancel,
                log=self._log,
                requirements_dir=self.runtime_backup_dir,
            )
            installed_profile_id = self._installed_gpu_profile_id(
                install_mode=mode
            )
            current_system = self.preflight(
                require_disk=False,
                install_mode=mode,
            )
            profile_id = str(current_system["gpu_profile"])
            profile_changed = profile_id != installed_profile_id
            base_profile = next(
                (
                    value
                    for value in new_manifest.python["gpu_profiles"]
                    if value["id"] == profile_id
                ),
                None,
            )
            if base_profile is None:
                raise InstallerServiceError(
                    f"현재 시스템용 GPU 프로필이 새 매니페스트에 없습니다: "
                    f"{profile_id}"
                )
            profile = effective_gpu_profile(base_profile, mode)
            self._log(
                "[업데이트] 현재 PC 기준 GPU 프로필: "
                f"installed={installed_profile_id}, selected={profile_id}, "
                f"changed={profile_changed}"
            )

            self._set_phase("source")
            update_comfy_source(
                destination=self.comfy_root,
                repository=str(new_manifest.comfy["repository"]),
                ref=str(new_manifest.comfy["ref"]),
                cancel_event=self._cancel,
                log=self._log,
                requirements_dir=self.runtime_backup_dir,
            )

            self._set_phase("core_dependencies")
            python_result = None
            if source_changed or python_changed or profile_changed:
                python_result = install_python_dependencies(
                    comfy_root=self.comfy_root,
                    python=python,
                    python_manifest=new_manifest.python,
                    gpu_profile=profile,
                    downloader=self.downloader,
                    cancel_event=self._cancel,
                    cache_root=self.comfy_root / ".installer-cache",
                    log=self._log,
                    progress=self._set_progress,
                )
            else:
                self._log("[업데이트] Comfy/Python 변경 없음: 핵심 의존성 생략")
                python_result = {
                    "python": str(python),
                    "profile": profile_id,
                    "reused": True,
                    "preinstall_wheels": [],
                    "sageattention_wheel": None,
                    "excluded_acceleration_packages": (
                        ["sageattention", "triton-windows", "triton"]
                        if mode == INSTALL_MODE_NVIDIA_COMPATIBILITY
                        else []
                    ),
                }

            self._set_phase("custom_nodes")
            updated_node_names: list[str] = []
            node_paths = update_custom_nodes(
                nodes=new_manifest.custom_nodes,
                comfy_root=self.comfy_root,
                downloader=self.downloader,
                cancel_event=self._cancel,
                log=self._log,
                progress=self._set_progress,
                changed_nodes=updated_node_names,
                requirements_dir=self.runtime_backup_dir,
            )
            if updated_node_names:
                self._log(
                    "[업데이트] 실제 변경된 커스텀 노드: "
                    + ", ".join(updated_node_names)
                )

            self._set_phase("node_dependencies")
            node_requirements: list[str] = []
            if (
                source_changed
                or python_changed
                or nodes_changed
                or updated_node_names
            ):
                node_requirements = install_node_dependencies(
                    comfy_root=self.comfy_root,
                    python=python,
                    node_paths=node_paths,
                    compatibility_packages=list(
                        new_manifest.python["compatibility_packages"]
                    ),
                    cancel_event=self._cancel,
                    log=self._log,
                )
            else:
                self._log("[업데이트] 노드 변경 없음: Python 의존성 설치 생략")

            self._set_phase("runtime_isolation")
            runtime = verify_isolated_runtime(
                comfy_root=self.comfy_root,
                python=python,
                gpu_profile=profile,
                cancel_event=self._cancel,
                log=self._log,
            )

            self._set_phase("startup")
            process = ComfyProcess(
                comfy_root=self.comfy_root,
                python=python,
                cancel_event=self._cancel,
                log=self._log_comfy,
            )
            stats = process.start(timeout=900)
            actual_version = (
                stats.get("system", {}).get("comfyui_version")
                if isinstance(stats, dict)
                else None
            )
            if actual_version != new_manifest.comfy["version"]:
                raise ComfyE2EError(
                    "업데이트 후 ComfyUI 버전이 매니페스트와 다릅니다: "
                    f"expected={new_manifest.comfy['version']}, actual={actual_version}"
                )
            process.stop()
            process = None

            self._set_phase("complete")
            result = {
                "operation": "update",
                "install_mode": mode,
                "compatibility_warning": warning,
                "completed_at": _now_iso(),
                "duration_seconds": round(
                    time.monotonic() - started_monotonic, 3
                ),
                "hooking_server": hooking_result,
                "manifest_before": old_manifest.sha256,
                "manifest_after": new_manifest.sha256,
                "changes": {
                    "comfy": source_changed,
                    "python": python_changed,
                    "gpu_profile": profile_changed,
                    "custom_nodes": bool(nodes_changed or updated_node_names),
                },
                "system": current_system,
                "updated_custom_nodes": updated_node_names,
                "python": python_result,
                "runtime": runtime,
                "node_requirements": node_requirements,
                "comfy_version": actual_version,
                "repatch": None,
                "workflow_e2e": None,
                "restart_required": bool(
                    hooking_result.get("restart_required")
                ),
            }
            result_path = self._write_result(result)
            result["result_path"] = str(result_path)
            with self._lock:
                self._state.update(
                    {
                        "state": "succeeded",
                        "finished_at": _now_iso(),
                        "progress": {
                            "event": "complete",
                            "current": 1,
                            "total": 1,
                        },
                        "error": None,
                        "result": result,
                    }
                )
            self._log(
                "[완료] 빠른 업데이트 성공 (리패치/E2E 생략): "
                f"{result_path}"
            )
        except DownloadCancelled as exc:
            self._log(f"[중단] {exc}", "warning")
            with self._lock:
                self._state.update(
                    {
                        "state": "cancelled",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                    }
                )
        except Exception as exc:
            print(f"[COMFY_INSTALL][SERVICE] 업데이트 실패: {exc}")
            traceback.print_exc()
            self._log(f"[실패] {exc}", "error")
            with self._lock:
                self._state.update(
                    {
                        "state": "failed",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                    }
                )
        finally:
            if process is not None:
                process.stop()

    def _run_update(self, *, install_mode: str) -> None:
        process: ComfyProcess | None = None
        transaction: dict[str, Any] | None = None
        rollback_result: dict[str, Any] | None = None
        runtime_pause_token: Any = None
        runtime_was_paused = False
        managed_runtime_active = False
        runtime_resume_result: Any = None
        started_monotonic = time.monotonic()
        old_manifest = self.manifest
        mode = normalize_install_mode(install_mode)

        def restore_transaction_config(backup_path: str) -> dict[str, Any]:
            config_backup = (
                transaction.get("config_backup")
                if isinstance(transaction, dict)
                else None
            )
            expected_sha256 = (
                config_backup.get("sha256")
                if isinstance(config_backup, dict)
                else None
            )
            if not isinstance(expected_sha256, str):
                raise RuntimeStateError(
                    "롤백 트랜잭션에 config snapshot SHA-256이 없습니다."
                )
            return restore_verified_config_snapshot(
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                backup_path=backup_path,
                expected_sha256=expected_sha256,
            )

        def resume_managed_runtime() -> Any:
            nonlocal managed_runtime_active, runtime_was_paused
            nonlocal runtime_resume_result
            if not runtime_was_paused:
                return runtime_resume_result
            if self.resume_managed_comfy is None:
                runtime_was_paused = False
                runtime_resume_result = {"status": "not-managed"}
                return runtime_resume_result
            try:
                runtime_resume_result = self.resume_managed_comfy(
                    runtime_pause_token
                )
                runtime_was_paused = False
                managed_runtime_active = True
                return runtime_resume_result
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][SERVICE] 관리 Comfy 복구 실패: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise

        def rollback_after_failure() -> dict[str, Any] | None:
            nonlocal managed_runtime_active, process, runtime_pause_token
            nonlocal runtime_was_paused
            if managed_runtime_active and self.pause_managed_comfy is not None:
                runtime_pause_token = self.pause_managed_comfy()
                managed_runtime_active = False
                runtime_was_paused = True
            if transaction is None:
                if runtime_was_paused:
                    resume_managed_runtime()
                return None
            if process is not None:
                process.stop()
                process = None
            restored = rollback_runtime_transaction(
                comfy_root=self.comfy_root,
                snapshot=transaction,
                restore_config=restore_transaction_config,
                log=self._log,
            )
            rollback_python = uv_python_path(self.comfy_root / ".venv")
            verification = ComfyProcess(
                comfy_root=self.comfy_root,
                python=rollback_python,
                cancel_event=Event(),
                log=self._log_comfy,
            )
            try:
                stats = verification.start(timeout=900)
                restored["verification"] = {
                    "comfy_ref": git_head(self.comfy_root),
                    "comfy_version": (
                        stats.get("system", {}).get("comfyui_version")
                        if isinstance(stats, dict)
                        else None
                    ),
                    "started": True,
                }
            finally:
                verification.stop()
            restored["managed_comfy"] = resume_managed_runtime()
            return restored

        try:
            warning = compatibility_warning(mode)
            if warning:
                self._log(
                    "[업데이트][호환 설치 유지] SageAttention을 다시 "
                    f"설치하지 않습니다. {warning}",
                    "warning",
                )

            self._set_phase("config_backup")
            config_backup = backup_current_config(
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                reason="hooking_update",
            )

            self._set_phase("hooking_server")
            hooking_result = update_hooking_server_main(
                project_root=self.project_root,
                config_path=self.config_path,
                backup_dir=self.config_backup_dir,
                cancel_event=self._cancel,
                log=self._log,
                config_backup=config_backup,
            )
            if hooking_result.get("changed"):
                self._set_phase("complete")
                result = {
                    "operation": "update",
                    "update_kind": "application_bootstrap",
                    "install_mode": mode,
                    "compatibility_warning": warning,
                    "completed_at": _now_iso(),
                    "duration_seconds": round(
                        time.monotonic() - started_monotonic, 3
                    ),
                    "hooking_server": hooking_result,
                    "manifest_before": old_manifest.sha256,
                    "manifest_after": None,
                    "changes": {
                        "hooking_server": True,
                        "runtime": False,
                    },
                    "restart_required": True,
                    "message": (
                        "새 업데이터를 적용했습니다. 앱 재시작 후 업데이트를 "
                        "한 번 더 실행하면 Comfy 호환성 승격을 시작합니다."
                    ),
                }
                result_path = self._write_result(result)
                result["result_path"] = str(result_path)
                with self._lock:
                    self._state.update(
                        {
                            "state": "succeeded",
                            "finished_at": _now_iso(),
                            "progress": {
                                "event": "complete",
                                "current": 1,
                                "total": 1,
                            },
                            "error": None,
                            "result": result,
                        }
                    )
                self._log(
                    "[완료] 업데이터 부트스트랩 성공. 재시작 후 두 번째 "
                    f"업데이트가 필요합니다: {result_path}"
                )
                return

            self._set_phase("manifest")
            new_manifest = load_install_manifest()
            self.manifest = new_manifest
            with self._lock:
                self._state["manifest"] = {
                    "sha256": new_manifest.sha256,
                    "comfy_version": new_manifest.comfy["version"],
                    "model_count": len(new_manifest.models),
                    "model_bytes": sum(
                        int(model["size"]) for model in new_manifest.models
                    ),
                    "custom_node_count": len(new_manifest.custom_nodes),
                    "workflow_count": new_manifest.workflows["expected_count"],
                }

            current_system = self.preflight(
                require_disk=False,
                install_mode=mode,
            )
            profile_id = str(current_system["gpu_profile"])
            installed_profile_id = self._installed_gpu_profile_id(
                install_mode=mode
            )
            base_profile = next(
                (
                    value
                    for value in new_manifest.python["gpu_profiles"]
                    if value["id"] == profile_id
                ),
                None,
            )
            if base_profile is None:
                raise InstallerServiceError(
                    f"현재 시스템용 GPU 프로필이 새 매니페스트에 없습니다: {profile_id}"
                )
            profile = effective_gpu_profile(base_profile, mode)

            self._set_phase("inventory")
            inventory = inspect_runtime(
                comfy_root=self.comfy_root,
                manifest=new_manifest,
                profile_id=profile_id,
                install_mode=mode,
            )
            profile_changed = profile_id != installed_profile_id
            if profile_changed and "gpu_profile" not in inventory["runtime_change_reasons"]:
                inventory["runtime_change_reasons"].append("gpu_profile")
                inventory["runtime_changed"] = True
            self._log(
                "[업데이트] 실제 런타임 변경 사유: "
                + (
                    ", ".join(inventory["runtime_change_reasons"])
                    if inventory["runtime_change_reasons"]
                    else "없음"
                )
            )

            if not inventory["runtime_changed"]:
                self._set_phase("complete")
                result = {
                    "operation": "update",
                    "update_kind": "already_current",
                    "install_mode": mode,
                    "compatibility_warning": warning,
                    "completed_at": _now_iso(),
                    "duration_seconds": round(
                        time.monotonic() - started_monotonic, 3
                    ),
                    "hooking_server": hooking_result,
                    "manifest_before": old_manifest.sha256,
                    "manifest_after": new_manifest.sha256,
                    "inventory": inventory,
                    "changes": {
                        "hooking_server": False,
                        "runtime": False,
                    },
                    "comfy_version": new_manifest.comfy["version"],
                    "comfy_ref": inventory["actual_comfy_ref"],
                    "repatch": None,
                    "workflow_static": None,
                    "workflow_runtime": None,
                    "transaction": None,
                    "rollback": None,
                    "restart_required": False,
                }
                result_path = self._write_result(result)
                result["result_path"] = str(result_path)
                with self._lock:
                    self._state.update(
                        {
                            "state": "succeeded",
                            "finished_at": _now_iso(),
                            "progress": {
                                "event": "complete",
                                "current": 1,
                                "total": 1,
                            },
                            "error": None,
                            "result": result,
                        }
                    )
                self._log(f"[완료] Comfy 런타임이 이미 검증된 최신 상태입니다: {result_path}")
                return

            workflow_paths = collect_workflow_bindings(
                self.config_path,
                new_manifest,
                include_optional=False,
            )
            workflow_bindings = {
                key: str(path) for key, path in workflow_paths.items()
            }

            self._set_phase("runtime_pause")
            if self.pause_managed_comfy is not None:
                try:
                    runtime_pause_token = self.pause_managed_comfy()
                    runtime_was_paused = True
                    managed_runtime_active = False
                    self._log("[업데이트] 실행 중이던 관리 Comfy 안전 정지 완료")
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][SERVICE] 관리 Comfy 안전 정지 실패: "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    raise
            else:
                runtime_pause_token = {"status": "not-managed"}

            self._set_phase("transaction")
            transaction = create_runtime_transaction(
                comfy_root=self.comfy_root,
                manifest=new_manifest,
                config_backup=config_backup,
                log=self._log,
            )

            reasons = set(inventory["runtime_change_reasons"])
            source_changed = "comfy_ref" in reasons
            python_changed = bool(
                {"missing_receipt", "python_profile", "gpu_profile"}.intersection(
                    reasons
                )
            )
            nodes_changed = any(
                reason.startswith("custom_node_") for reason in reasons
            )

            self._set_phase("venv")
            python = create_comfy_venv(
                comfy_root=self.comfy_root,
                python_version=str(new_manifest.python["version"]),
                cancel_event=self._cancel,
                log=self._log,
                requirements_dir=self.runtime_backup_dir,
            )

            self._set_phase("source")
            update_comfy_source(
                destination=self.comfy_root,
                repository=str(new_manifest.comfy["repository"]),
                ref=str(new_manifest.comfy["ref"]),
                cancel_event=self._cancel,
                log=self._log,
                requirements_dir=self.runtime_backup_dir,
            )

            self._set_phase("core_dependencies")
            if source_changed or python_changed:
                python_result = install_python_dependencies(
                    comfy_root=self.comfy_root,
                    python=python,
                    python_manifest=new_manifest.python,
                    gpu_profile=profile,
                    downloader=self.downloader,
                    cancel_event=self._cancel,
                    cache_root=self.comfy_root / ".installer-cache",
                    log=self._log,
                    progress=self._set_progress,
                )
            else:
                self._log("[업데이트] Comfy/Python 변경 없음: 핵심 의존성 재사용")
                python_result = {
                    "python": str(python),
                    "profile": profile_id,
                    "reused": True,
                    "preinstall_wheels": [],
                    "sageattention_wheel": None,
                    "excluded_acceleration_packages": (
                        ["sageattention", "triton-windows", "triton"]
                        if mode == INSTALL_MODE_NVIDIA_COMPATIBILITY
                        else []
                    ),
                }

            self._set_phase("custom_nodes")
            updated_node_names: list[str] = []
            node_paths = update_custom_nodes(
                nodes=new_manifest.custom_nodes,
                comfy_root=self.comfy_root,
                downloader=self.downloader,
                cancel_event=self._cancel,
                log=self._log,
                progress=self._set_progress,
                changed_nodes=updated_node_names,
                requirements_dir=self.runtime_backup_dir,
            )
            if updated_node_names:
                nodes_changed = True
                self._log(
                    "[업데이트] 실제 변경된 커스텀 노드: "
                    + ", ".join(updated_node_names)
                )

            self._set_phase("node_dependencies")
            node_requirements: list[str] = []
            if source_changed or python_changed or nodes_changed:
                node_requirements = install_node_dependencies(
                    comfy_root=self.comfy_root,
                    python=python,
                    node_paths=node_paths,
                    compatibility_packages=list(
                        new_manifest.python["compatibility_packages"]
                    ),
                    cancel_event=self._cancel,
                    log=self._log,
                )
            else:
                self._log("[업데이트] 노드 변경 없음: 노드 Python 의존성 재사용")

            self._set_phase("runtime_isolation")
            runtime = verify_isolated_runtime(
                comfy_root=self.comfy_root,
                python=python,
                gpu_profile=profile,
                cancel_event=self._cancel,
                log=self._log,
            )

            lora_export_order = validate_instant_lora_export_order(
                comfy_root=self.comfy_root,
                python=python,
                cancel_event=self._cancel,
                log=self._log,
            )

            self._set_phase("repatch")
            repatch = patch_comfy_input(
                comfy_input_dir=self.comfy_root / "input",
                fallback_source=self.project_root / "modes" / "fallback_img",
                log=self._log,
            )

            self._set_phase("startup")
            try:
                process = ComfyProcess(
                    comfy_root=self.comfy_root,
                    python=python,
                    cancel_event=self._cancel,
                    log=self._log_comfy,
                    extra_args=_WINDOWS_E2E_EXTRA_ARGS,
                )
                stats = process.start(timeout=900)
                actual_version = (
                    stats.get("system", {}).get("comfyui_version")
                    if isinstance(stats, dict)
                    else None
                )
                actual_ref = git_head(self.comfy_root)
                expected_ref = str(new_manifest.comfy["ref"]).lower()
                if actual_version != new_manifest.comfy["version"]:
                    raise ComfyE2EError(
                        "업데이트 후 ComfyUI 버전이 매니페스트와 다릅니다: "
                        f"expected={new_manifest.comfy['version']}, "
                        f"actual={actual_version}"
                    )
                if actual_ref != expected_ref:
                    raise ComfyE2EError(
                        "업데이트 후 ComfyUI Git SHA가 매니페스트와 다릅니다: "
                        f"expected={expected_ref}, actual={actual_ref}"
                    )
                self._log(
                    "[기동 검사] 업데이트 ComfyUI 버전/SHA 확인 완료: "
                    f"version={actual_version}, ref={actual_ref}"
                )
            finally:
                if process is not None:
                    process.stop()
                    process = None

            self._set_phase("config")
            config_update = apply_installed_config(
                config_path=self.config_path,
                requirements_dir=self.config_backup_dir,
                comfy_root=self.comfy_root,
                workflow_bindings=workflow_bindings,
                required_bindings=workflow_bindings.keys(),
                default_workflow_bindings=None,
                workflow_base_dir=self._embedded_workflow_base_dir(),
            )

            release_version = None
            previous_receipt = inventory.get("receipt")
            if isinstance(previous_receipt, dict):
                previous_workflows = previous_receipt.get("workflows")
                if isinstance(previous_workflows, dict):
                    release_version = previous_workflows.get("release_version")
            if not isinstance(release_version, str) or not release_version:
                release_version = sorted(
                    new_manifest.workflows["release_dependencies"]
                )[-1]
            active_binding_keys = set(workflow_bindings)
            selected_workflow_ids = [
                str(entry["id"])
                for entry in new_manifest.workflows["release_dependencies"][
                    release_version
                ]
                if set(entry["bindings"]).issubset(active_binding_keys)
            ]
            runtime_receipt = write_runtime_receipt(
                comfy_root=self.comfy_root,
                manifest=new_manifest,
                profile_id=profile_id,
                install_mode=mode,
                workflow_bindings=workflow_bindings,
                selected_workflow_ids=selected_workflow_ids,
                release_version=release_version,
            )

            self._set_phase("runtime_resume")
            runtime_resume_result = resume_managed_runtime()
            transaction_result = complete_runtime_transaction(
                comfy_root=self.comfy_root,
                snapshot=transaction,
                log=self._log,
            )

            self._set_phase("complete")
            result = {
                "operation": "update",
                "update_kind": "runtime_compatibility",
                "install_mode": mode,
                "compatibility_warning": warning,
                "completed_at": _now_iso(),
                "duration_seconds": round(
                    time.monotonic() - started_monotonic, 3
                ),
                "hooking_server": hooking_result,
                "manifest_before": old_manifest.sha256,
                "manifest_after": new_manifest.sha256,
                "inventory": inventory,
                "changes": {
                    "hooking_server": False,
                    "runtime": True,
                    "comfy": source_changed,
                    "python": python_changed,
                    "gpu_profile": profile_changed,
                    "custom_nodes": bool(nodes_changed or updated_node_names),
                },
                "system": current_system,
                "updated_custom_nodes": updated_node_names,
                "python": python_result,
                "runtime": runtime,
                "lora_export_order": lora_export_order,
                "node_requirements": node_requirements,
                "comfy_version": actual_version,
                "comfy_ref": actual_ref,
                "repatch": repatch,
                "workflow_e2e_deferred": True,
                "workflow_static": None,
                "workflow_runtime": None,
                "runtime_receipt": runtime_receipt,
                "managed_comfy": {
                    "pause": runtime_pause_token,
                    "resume": runtime_resume_result,
                },
                "config": {
                    "backup_path": str(config_update.backup_path),
                    "before_sha256": config_update.before_sha256,
                    "after_sha256": config_update.after_sha256,
                    "restored_after_success": False,
                },
                "transaction": transaction_result,
                "rollback": None,
                "restart_required": False,
            }
            result_path = self._write_result(result)
            result["result_path"] = str(result_path)
            with self._lock:
                self._state.update(
                    {
                        "state": "succeeded",
                        "finished_at": _now_iso(),
                        "progress": {
                            "event": "complete",
                            "current": 1,
                            "total": 1,
                        },
                        "error": None,
                        "result": result,
                    }
                )
            self._log(
                "[완료] Comfy 업데이트 성공. 워크플로우 E2E는 별도 "
                f"검사 버튼에서 선택 실행할 수 있습니다: {result_path}"
            )
        except (ComfyE2ECancelled, DownloadCancelled) as exc:
            self._log(f"[중단] {exc}", "warning")
            try:
                rollback_result = rollback_after_failure()
            except Exception as rollback_exc:
                print(
                    "[COMFY_INSTALL][SERVICE] 중단 후 자동 롤백 실패: "
                    f"operation_error={exc}, rollback_error={rollback_exc}"
                )
                traceback.print_exc()
            with self._lock:
                self._state.update(
                    {
                        "state": "cancelled",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                        "result": {"rollback": rollback_result},
                    }
                )
        except Exception as exc:
            print(f"[COMFY_INSTALL][SERVICE] 호환성 업데이트 실패: {exc}")
            traceback.print_exc()
            self._log(f"[실패] {exc}", "error")
            rollback_error = None
            try:
                rollback_result = rollback_after_failure()
            except Exception as rollback_exc:
                rollback_error = str(rollback_exc)
                print(
                    "[COMFY_INSTALL][SERVICE] 업데이트 실패 후 자동 롤백 실패: "
                    f"operation_error={exc}, rollback_error={rollback_exc}"
                )
                traceback.print_exc()
                self._log(f"[롤백 실패] {rollback_exc}", "error")
            with self._lock:
                self._state.update(
                    {
                        "state": "failed",
                        "finished_at": _now_iso(),
                        "error": str(exc),
                        "result": {
                            "rollback": rollback_result,
                            "rollback_error": rollback_error,
                        },
                    }
                )
        finally:
            if process is not None:
                process.stop()

    def restore_backup(self, backup_path: str | os.PathLike[str]) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise InstallerServiceError(
                    "설치 또는 업데이트 중에는 config.json을 복원할 수 없습니다."
                )
        result = restore_config_backup(
            config_path=self.config_path,
            requirements_dir=self.config_backup_dir,
            backup_path=backup_path,
        )
        self._log(
            f"[설정] 사용자가 선택한 config.json 백업 복원: {backup_path}"
        )
        return result
