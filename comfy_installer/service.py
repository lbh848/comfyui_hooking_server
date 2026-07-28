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
from typing import Any

import httpx

from .configurator import (
    ConfigUpdateResult,
    apply_installed_config,
    restore_config_backup,
)
from .crypto import ExtractedWorkflowPack, extract_workflow_pack
from .dependency_installer import (
    create_comfy_venv,
    install_node_dependencies,
    install_python_dependencies,
    verify_isolated_runtime,
)
from .downloader import DownloadCancelled, ResumableDownloader
from .e2e import (
    ComfyE2ECancelled,
    ComfyE2EError,
    ComfyProcess,
    execute_prompt,
    make_e2e_prompt,
    protected_e2e_fixtures,
    promote_generated_fixture,
    validate_all_workflows,
)
from .manifest import InstallManifest, load_install_manifest
from .model_installer import install_models
from .node_installer import install_custom_nodes
from .source_installer import install_comfy_source
from .system_probe import probe_system


class InstallerServiceError(RuntimeError):
    """설치 서비스 상태 또는 입력 검증 실패."""


_PHASES = (
    ("preflight", "Windows/GPU/디스크/도구 검사"),
    ("credentials", "워크플로우 팩·Civitai 인증 사전 검증"),
    ("source", "ComfyUI v0.20.1 고정 소스 설치"),
    ("workflows", "암호화 워크플로우 17개 복원"),
    ("venv", "comfy/.venv Python 3.12.11 생성"),
    ("core_dependencies", "PyTorch/Triton/SageAttention/Comfy 의존성 설치"),
    ("custom_nodes", "고정 커스텀 노드 설치"),
    ("node_dependencies", "커스텀 노드 Python 의존성 설치"),
    ("runtime_isolation", "GPU 및 독립 환경 검증"),
    ("models", "고정 모델 다운로드·SHA-256 검증"),
    ("startup", "독립 ComfyUI 기동·노드 로드 확인"),
    ("e2e_static", "17개 워크플로우 변환·구조 검증"),
    ("e2e_runtime", "17개 워크플로우 실제 실행"),
    ("config", "config.json 백업·설치 경로 적용"),
    ("complete", "설치 결과 기록"),
)


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
        self.manifest = manifest or load_install_manifest()
        self.downloader = downloader or ResumableDownloader(max_retries=4)
        self._lock = RLock()
        self._cancel = Event()
        self._thread: threading.Thread | None = None
        self._log_sequence = 0
        self._logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self._state: dict[str, Any] = {
            "state": "idle",
            "phase": None,
            "phase_index": 0,
            "phase_count": len(_PHASES),
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
        for index, (candidate_id, label) in enumerate(_PHASES, 1):
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
                    f"[단계 {index}/{len(_PHASES)}] {label}"
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

    def preflight(self) -> dict:
        result = probe_system(self.comfy_root, self.manifest)
        return {
            **result,
            "manifest": dict(self._state["manifest"]),
        }

    def _validate_civitai_access(self, civitai_key: str) -> None:
        sample = next(
            (
                model
                for model in self.manifest.models
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
        }
        return (
            min((priorities.get(key, 999) for key in keys), default=999),
            validation.filename.casefold(),
        )

    def _run_runtime_e2e(
        self,
        *,
        process: ComfyProcess,
        validations: list,
    ) -> list[dict]:
        results: list[dict] = []
        failures: list[dict[str, str]] = []
        promoted = False
        ordered = sorted(validations, key=self._runtime_order)
        with protected_e2e_fixtures(
            comfy_root=self.comfy_root,
            requirements_dir=self.requirements_dir,
        ) as fixtures:
            self._log(
                "[E2E] 설치기 전용 입력 픽스처 준비: "
                f"{fixtures['training']}"
            )
            for index, validation in enumerate(ordered, 1):
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
                try:
                    prompt = make_e2e_prompt(validation)
                    timeout = (
                        7200
                        if any(
                            "training_workflow" in key
                            for key in validation.binding_keys
                        )
                        else 3600
                    )
                    result = execute_prompt(
                        base_url=process.base_url,
                        prompt=prompt,
                        workflow=validation.workflow,
                        filename=validation.filename,
                        cancel_event=self._cancel,
                        log=self._log,
                        timeout=timeout,
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
                    child = process.process
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
                if not promoted:
                    promoted_path = promote_generated_fixture(
                        base_url=process.base_url,
                        execution_result=result,
                        comfy_root=self.comfy_root,
                    )
                    if promoted_path:
                        promoted = True
                        self._log(
                            "[E2E] 첫 생성 이미지를 후속 태그/편집/학습 "
                            f"픽스처로 적용: {promoted_path}"
                        )
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
        if len(results) != int(self.manifest.workflows["expected_count"]):
            raise ComfyE2EError(
                "실제 E2E 성공 수가 17개가 아닙니다: "
                f"actual={len(results)}"
            )
        return results

    def start(
        self,
        *,
        workflow_pack: str | os.PathLike[str],
        workflow_key: str,
        civitai_key: str,
        restore_config_after_success: bool = False,
    ) -> dict:
        pack_path = Path(workflow_pack).resolve()
        if not pack_path.is_file():
            raise InstallerServiceError(
                f"업로드된 워크플로우 팩이 없습니다: {pack_path}"
            )
        if not workflow_key:
            raise InstallerServiceError("워크플로우 팩 키가 비어 있습니다.")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise InstallerServiceError("ComfyUI 설치가 이미 진행 중입니다.")
            self._cancel = Event()
            self._logs.clear()
            self._log_sequence = 0
            self._state.update(
                {
                    "state": "running",
                    "phase": None,
                    "phase_index": 0,
                    "phase_label": "",
                    "started_at": _now_iso(),
                    "finished_at": None,
                    "progress": {},
                    "error": None,
                    "result": None,
                }
            )
            self._thread = threading.Thread(
                target=self._run_install,
                kwargs={
                    "pack_path": pack_path,
                    "workflow_key": workflow_key,
                    "civitai_key": civitai_key,
                    "restore_config_after_success": bool(
                        restore_config_after_success
                    ),
                },
                name="comfy-installer",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def cancel(self) -> dict:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                print(
                    "[COMFY_INSTALL][SERVICE] 중단 요청을 받았지만 진행 중인 "
                    "설치가 없습니다."
                )
                raise InstallerServiceError("진행 중인 설치가 없습니다.")
            self._cancel.set()
        self._log("[중단] 안전한 중단을 요청했습니다.", "warning")
        return self.status()

    def _write_result(self, result: dict) -> Path:
        state_root = self.comfy_root / ".installer-state"
        state_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        result_path = state_root / f"install-result-{stamp}.json"
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
        pack_path: Path,
        workflow_key: str,
        civitai_key: str,
        restore_config_after_success: bool,
    ) -> None:
        process: ComfyProcess | None = None
        config_update: ConfigUpdateResult | None = None
        started_monotonic = time.monotonic()
        try:
            self._set_phase("preflight")
            system = self.preflight()
            self._log(
                "[검사] GPU 프로필 선택: "
                f"{system['gpu_profile']}, "
                f"free={system['disk']['free'] / 1024**3:.2f} GiB"
            )

            self._set_phase("credentials")
            self._validate_civitai_access(civitai_key)

            self._set_phase("source")
            install_comfy_source(
                destination=self.comfy_root,
                repository=str(self.manifest.comfy["repository"]),
                ref=str(self.manifest.comfy["ref"]),
                cancel_event=self._cancel,
                log=self._log,
            )

            self._set_phase("workflows")
            workflows_root = (
                self.comfy_root / "user" / "default" / "workflows"
            )
            extracted = extract_workflow_pack(
                pack_path,
                workflows_root,
                workflow_key,
            )
            self._validate_extracted_pack(extracted)
            self._log(
                "[워크플로우] 복호화·해시 검증 완료: "
                f"17개, pack_sha256={extracted.pack_sha256}"
            )

            self._set_phase("venv")
            python = create_comfy_venv(
                comfy_root=self.comfy_root,
                python_version=str(self.manifest.python["version"]),
                cancel_event=self._cancel,
                log=self._log,
            )

            profile = next(
                profile
                for profile in self.manifest.python["gpu_profiles"]
                if profile["id"] == system["gpu_profile"]
            )
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

            self._set_phase("models")
            model_results = install_models(
                models=self.manifest.models,
                comfy_root=self.comfy_root,
                downloader=self.downloader,
                civitai_key=civitai_key,
                cancel_event=self._cancel,
                log=self._log,
                progress=self._set_progress,
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
            if actual_version != self.manifest.comfy["version"]:
                raise ComfyE2EError(
                    "기동된 ComfyUI 버전이 고정값과 다릅니다: "
                    f"expected={self.manifest.comfy['version']}, "
                    f"actual={actual_version}"
                )
            self._log(
                f"[E2E] ComfyUI 버전 확인 완료: {actual_version}"
            )

            self._set_phase("e2e_static")
            validations, _ = validate_all_workflows(
                base_url=process.base_url,
                workflow_bindings=extracted.workflow_bindings,
                expected_count=int(
                    self.manifest.workflows["expected_count"]
                ),
                excluded_filenames=list(
                    self.manifest.workflows["excluded_filenames"]
                ),
                cancel_event=self._cancel,
                log=self._log,
                progress=self._set_progress,
            )

            self._set_phase("e2e_runtime")
            runtime_e2e = self._run_runtime_e2e(
                process=process,
                validations=validations,
            )
            process.stop()
            process = None

            self._set_phase("config")
            config_update = apply_installed_config(
                config_path=self.config_path,
                requirements_dir=self.requirements_dir,
                comfy_root=self.comfy_root,
                workflow_bindings=extracted.workflow_bindings,
                required_bindings=self.manifest.workflows[
                    "required_bindings"
                ],
            )
            restore_result = None
            if restore_config_after_success:
                restore_result = restore_config_backup(
                    config_path=self.config_path,
                    requirements_dir=self.requirements_dir,
                    backup_path=config_update.backup_path,
                )
                self._log(
                    "[설정] 설치 성공 검증 후 요청대로 기존 config.json을 "
                    "복원했습니다."
                )

            self._set_phase("complete")
            result = {
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
                "custom_nodes": [str(path) for path in node_paths],
                "node_requirements": node_requirements,
                "models": model_results,
                "workflow_pack_sha256": extracted.pack_sha256,
                "workflow_static": [
                    validation.public_result()
                    for validation in validations
                ],
                "workflow_runtime": runtime_e2e,
                "config": {
                    "backup_path": str(config_update.backup_path),
                    "before_sha256": config_update.before_sha256,
                    "after_sha256": config_update.after_sha256,
                    "restored_after_success": restore_config_after_success,
                    "restore": restore_result,
                },
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
                            "current": 17,
                            "total": 17,
                        },
                        "error": None,
                        "result": result,
                    }
                )
            self._log(
                "[완료] ComfyUI 설치 및 17/17 실제 E2E 성공: "
                f"{result_path}"
            )
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
            workflow_key = ""
            civitai_key = ""

    def restore_backup(self, backup_path: str | os.PathLike[str]) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise InstallerServiceError(
                    "설치 진행 중에는 config.json을 복원할 수 없습니다."
                )
        result = restore_config_backup(
            config_path=self.config_path,
            requirements_dir=self.requirements_dir,
            backup_path=backup_path,
        )
        self._log(
            f"[설정] 사용자가 선택한 config.json 백업 복원: {backup_path}"
        )
        return result
