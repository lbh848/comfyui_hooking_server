from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import stat
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .crypto import ExtractedWorkflowPack, extract_workflow_pack
from .manifest import InstallManifest


class WorkflowLibraryError(RuntimeError):
    """배포 워크플로우 보관 또는 사용자 사본 생성 실패."""


LogCallback = Callable[[str], None]
_RELEASE_RE = re.compile(r"^v[1-9][0-9]*$")
_STATE_FILENAME = ".soya-pack.json"
USER_WORKFLOW_DIRNAME = "SOYA_USER"
DISTRIBUTION_LIBRARY_DIRNAME = "SOYA_DISTRIBUTION"
LEGACY_USER_WORKFLOW_DIRNAME = "SOYA_개인"
LEGACY_DISTRIBUTION_LIBRARY_DIRNAME = "SOYA_배포"


@dataclass(frozen=True)
class WorkflowSelection:
    release_version: str
    selected_item_ids: tuple[str, ...]
    workflow_bindings: dict[str, str]
    model_ids: tuple[str, ...]
    user_files: tuple[str, ...]


def embedded_workflow_base_dir(
    comfy_root: str | os.PathLike[str],
) -> Path:
    """Canonical folder used by embedded Comfy workflow config paths."""
    return (
        Path(comfy_root).resolve()
        / "user"
        / "default"
        / "workflows"
        / USER_WORKFLOW_DIRNAME
    ).resolve()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bytes_new(path: Path, payload: bytes) -> None:
    part = path.with_name(f"{path.name}.part-{uuid.uuid4().hex[:8]}")
    try:
        with part.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, path)
    except Exception:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY] 새 파일 저장 실패: "
            f"target={path}, part={part}"
        )
        traceback.print_exc()
        raise


def _write_json_new(path: Path, value: dict) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2) + os.linesep
    ).encode("utf-8")
    _write_bytes_new(path, payload)


def _files_identical(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    return _sha256_bytes(first.read_bytes()) == _sha256_bytes(second.read_bytes())


def _mark_distribution_files_read_only(release_root: Path) -> None:
    """Protect immutable distribution originals from accidental edits."""

    current_path: Path | None = None
    try:
        for path in sorted(
            release_root.rglob("*"), key=lambda value: str(value).casefold()
        ):
            current_path = path
            if path.is_symlink():
                raise WorkflowLibraryError(
                    f"배포 원본 폴더에 심볼릭 링크가 있습니다: {path}"
                )
            if not path.is_file():
                continue
            try:
                path.resolve().relative_to(release_root.resolve())
            except ValueError as exc:
                raise WorkflowLibraryError(
                    f"배포 원본 파일이 원본 폴더 밖을 가리킵니다: {path}"
                ) from exc
            writable_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            path.chmod(path.stat().st_mode & ~writable_bits)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY] 배포 원본 읽기 전용 "
            "설정 실패: "
            f"root={release_root}, path={current_path}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowLibraryError(
            f"배포 원본을 읽기 전용으로 보호하지 못했습니다: {exc}"
        ) from exc


def _is_read_only_file(path: Path) -> bool:
    writable_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    return not bool(path.stat().st_mode & writable_bits)


def _legacy_collision_destination(source: Path, destination: Path) -> Path:
    if not destination.exists() or _files_identical(source, destination):
        return destination
    suffix = 2
    while True:
        candidate = destination.with_name(
            f"{destination.stem}__legacy_{suffix}{destination.suffix}"
        )
        if not candidate.exists() or _files_identical(source, candidate):
            return candidate
        suffix += 1


def _copy_legacy_tree(
    *,
    source_root: Path,
    destination_root: Path,
    label: str,
    log: LogCallback | None,
) -> dict:
    if not source_root.is_dir():
        return {
            "found": False,
            "source": str(source_root),
            "destination": str(destination_root),
            "copied_files": 0,
            "reused_files": 0,
            "renamed_conflicts": 0,
            "path_map": {},
            "legacy_data_preserved": True,
        }

    copied_files = 0
    reused_files = 0
    renamed_conflicts = 0
    path_map: dict[str, str] = {}
    try:
        destination_root.mkdir(parents=True, exist_ok=True)
        for source in sorted(
            source_root.rglob("*"),
            key=lambda path: str(path).casefold(),
        ):
            relative = source.relative_to(source_root)
            desired = destination_root / relative
            if source.is_symlink():
                raise WorkflowLibraryError(
                    f"{label} 레거시 폴더의 심볼릭 링크는 자동 복사하지 않습니다: "
                    f"{source}"
                )
            if source.is_dir():
                desired.mkdir(parents=True, exist_ok=True)
                continue
            if not source.is_file():
                print(
                    "[COMFY_INSTALL][WORKFLOW_LIBRARY][ASCII_MIGRATE] "
                    f"일반 파일이 아닌 항목 건너뜀: label={label}, path={source}"
                )
                continue

            desired.parent.mkdir(parents=True, exist_ok=True)
            destination = _legacy_collision_destination(source, desired)
            if destination.exists():
                reused_files += 1
            else:
                _write_bytes_new(destination, source.read_bytes())
                copied_files += 1
                if destination != desired:
                    renamed_conflicts += 1
            path_map[str(source.resolve())] = str(destination.resolve())

        message = (
            f"[워크플로우 경로 마이그레이션] {label}: "
            f"copied={copied_files}, reused={reused_files}, "
            f"conflicts={renamed_conflicts}, legacy_preserved={source_root}"
        )
        if log:
            log(message)
        else:
            print(f"[COMFY_INSTALL][WORKFLOW_LIBRARY][ASCII_MIGRATE] {message}")
        return {
            "found": True,
            "source": str(source_root),
            "destination": str(destination_root),
            "copied_files": copied_files,
            "reused_files": reused_files,
            "renamed_conflicts": renamed_conflicts,
            "path_map": path_map,
            "legacy_data_preserved": True,
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY][ASCII_MIGRATE] "
            f"{label} 마이그레이션 실패: source={source_root}, "
            f"destination={destination_root}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, WorkflowLibraryError):
            raise
        raise WorkflowLibraryError(
            f"{label} ASCII 폴더 마이그레이션 실패: {exc}"
        ) from exc


def _archive_legacy_tree(
    *,
    source_root: Path,
    expected_parent: Path,
    archive_root: Path,
    archive_name: str,
    label: str,
    log: LogCallback | None,
) -> dict:
    if not source_root.exists():
        return {
            "archived": False,
            "source": str(source_root),
            "backup": None,
        }
    try:
        resolved_source = source_root.resolve()
        resolved_parent = expected_parent.resolve()
        if resolved_source.parent != resolved_parent:
            raise WorkflowLibraryError(
                f"{label} 레거시 폴더의 부모 경로가 예상과 다릅니다: "
                f"source={resolved_source}, expected_parent={resolved_parent}"
            )
        archive_base = archive_root.resolve()
        archive_base.mkdir(parents=True, exist_ok=True)
        destination = (archive_base / archive_name).resolve()
        try:
            destination.relative_to(archive_base)
        except ValueError as exc:
            raise WorkflowLibraryError(
                f"{label} 백업 경로가 백업 루트 밖입니다: {destination}"
            ) from exc
        if destination.exists():
            raise WorkflowLibraryError(
                f"{label} 백업 대상이 이미 존재합니다: {destination}"
            )
        os.replace(resolved_source, destination)
        message = (
            f"[워크플로우 경로 마이그레이션] {label} 레거시 폴더 백업 이동: "
            f"{destination}"
        )
        if log:
            log(message)
        else:
            print(
                "[COMFY_INSTALL][WORKFLOW_LIBRARY][ASCII_MIGRATE] "
                f"{message}"
            )
        return {
            "archived": True,
            "source": str(resolved_source),
            "backup": str(destination),
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY][ASCII_MIGRATE] "
            f"{label} 레거시 폴더 백업 이동 실패; 원본을 유지합니다: "
            f"source={source_root}, archive_root={archive_root}, error={exc}"
        )
        traceback.print_exc()
        return {
            "archived": False,
            "source": str(source_root),
            "backup": None,
            "error": str(exc),
        }


def migrate_legacy_workflow_layout(
    *,
    comfy_root: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    backup_dir: str | os.PathLike[str],
    log: LogCallback | None = None,
) -> dict:
    """한글 레거시 폴더를 ASCII 폴더로 비파괴 복사하고 설정을 전환한다."""
    comfy = Path(comfy_root).resolve()
    library = Path(library_root).resolve()
    workflows_root = comfy / "user" / "default" / "workflows"
    legacy_user_root = workflows_root / LEGACY_USER_WORKFLOW_DIRNAME
    user_root = workflows_root / USER_WORKFLOW_DIRNAME
    legacy_distribution_root = (
        library / LEGACY_DISTRIBUTION_LIBRARY_DIRNAME
    )
    distribution_root = library / DISTRIBUTION_LIBRARY_DIRNAME
    try:
        distribution = _copy_legacy_tree(
            source_root=legacy_distribution_root,
            destination_root=distribution_root,
            label="배포 라이브러리",
            log=log,
        )
        user = _copy_legacy_tree(
            source_root=legacy_user_root,
            destination_root=user_root,
            label="사용자 워크플로우",
            log=log,
        )

        from .configurator import retarget_legacy_workflow_paths

        config = retarget_legacy_workflow_paths(
            config_path=config_path,
            backup_dir=backup_dir,
            legacy_user_root=legacy_user_root,
            user_root=user_root,
            path_map=user["path_map"],
        )
        archive_root = (
            Path(backup_dir).resolve()
            / "workflow_ascii_migration"
            / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        user_archive = (
            _archive_legacy_tree(
                source_root=legacy_user_root,
                expected_parent=workflows_root,
                archive_root=archive_root,
                archive_name="LEGACY_SOYA_USER",
                label="사용자 워크플로우",
                log=log,
            )
            if user["found"]
            else {
                "archived": False,
                "source": str(legacy_user_root),
                "backup": None,
            }
        )
        distribution_archive = (
            _archive_legacy_tree(
                source_root=legacy_distribution_root,
                expected_parent=library,
                archive_root=archive_root,
                archive_name="LEGACY_SOYA_DISTRIBUTION",
                label="배포 라이브러리",
                log=log,
            )
            if distribution["found"]
            else {
                "archived": False,
                "source": str(legacy_distribution_root),
                "backup": None,
            }
        )
        return {
            "user": {
                **{key: value for key, value in user.items() if key != "path_map"},
                "legacy_archive": user_archive,
            },
            "distribution": {
                **{
                    key: value
                    for key, value in distribution.items()
                    if key != "path_map"
                },
                "legacy_archive": distribution_archive,
            },
            "config": config,
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY][ASCII_MIGRATE] "
            f"워크플로우 레이아웃 마이그레이션 실패: {exc}"
        )
        traceback.print_exc()
        if isinstance(exc, WorkflowLibraryError):
            raise
        raise WorkflowLibraryError(
            f"워크플로우 ASCII 레이아웃 마이그레이션 실패: {exc}"
        ) from exc


def _read_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[COMFY_INSTALL][WORKFLOW_LIBRARY] {label} 읽기 실패: "
            f"path={path}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowLibraryError(f"{label}을 읽을 수 없습니다: {path}") from exc
    if not isinstance(value, dict):
        raise WorkflowLibraryError(f"{label} 최상위 값이 객체가 아닙니다: {path}")
    return value


def _catalog_for_extracted(
    extracted: ExtractedWorkflowPack,
    manifest: InstallManifest,
) -> list[dict]:
    models_by_id = {str(model["id"]): model for model in manifest.models}
    raw_releases = manifest.workflows.get("release_dependencies", {})
    fixed_entries = raw_releases.get(extracted.release_version)
    if not isinstance(fixed_entries, list):
        raise WorkflowLibraryError(
            "고정 모델 목록이 등록되지 않은 워크플로우 배포 버전입니다: "
            f"{extracted.release_version}"
        )
    fixed_by_id = {str(entry["id"]): entry for entry in fixed_entries}
    catalog: list[dict] = []
    seen_ids: set[str] = set()
    for item in extracted.workflow_items:
        item_id = str(item["id"])
        fixed = fixed_by_id.get(item_id)
        if fixed is None:
            raise WorkflowLibraryError(
                f"배포 명세에 없는 워크플로우가 팩에 포함되었습니다: {item_id}"
            )
        item_bindings = sorted(str(value) for value in item["bindings"])
        fixed_bindings = sorted(str(value) for value in fixed["bindings"])
        if item_bindings != fixed_bindings:
            raise WorkflowLibraryError(
                "워크플로우 설정 바인딩이 고정 배포 명세와 다릅니다: "
                f"item={item_id}"
            )
        model_ids = tuple(sorted(str(value) for value in fixed["model_ids"]))
        if "model_ids" in item:
            embedded_model_ids = sorted(str(value) for value in item["model_ids"])
            if embedded_model_ids != list(model_ids):
                raise WorkflowLibraryError(
                    "팩의 모델 목록이 고정 배포 명세와 다릅니다: "
                    f"item={item_id}, release={extracted.release_version}"
                )
        unknown_models = [
            model_id for model_id in model_ids if model_id not in models_by_id
        ]
        if unknown_models:
            raise WorkflowLibraryError(
                "고정 배포 명세에 등록되지 않은 모델 ID가 있습니다: "
                f"item={item_id}, models={unknown_models}"
            )
        archive_name = str(item["archive_name"])
        source_path = extracted.target_dir / Path(archive_name).name
        if not source_path.is_file():
            raise WorkflowLibraryError(
                f"복호화한 워크플로우 파일이 없습니다: {source_path}"
            )
        model_bytes = sum(
            int(models_by_id[model_id]["size"]) for model_id in model_ids
        )
        catalog.append(
            {
                "id": item_id,
                "name": str(item["name"]),
                "filename": source_path.name,
                "bindings": item_bindings,
                "sha256": extracted.workflow_hashes[str(source_path)],
                "model_ids": list(model_ids),
                "model_sizes": {
                    model_id: int(models_by_id[model_id]["size"])
                    for model_id in model_ids
                },
                "model_count": len(model_ids),
                "model_bytes": model_bytes,
            }
        )
        seen_ids.add(item_id)
    missing_ids = sorted(set(fixed_by_id) - seen_ids)
    if missing_ids:
        raise WorkflowLibraryError(
            "팩에 고정 배포 워크플로우가 빠져 있습니다: " + ", ".join(missing_ids)
        )
    return sorted(catalog, key=lambda item: item["name"].casefold())


def _safe_remove_stage(stage: Path, work_root: Path) -> None:
    resolved = stage.resolve()
    root = work_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkflowLibraryError(
            f"워크플로우 스테이징 경로가 작업 루트 밖입니다: {resolved}"
        ) from exc
    if resolved == root:
        raise WorkflowLibraryError("워크플로우 작업 루트 자체는 제거할 수 없습니다.")
    if resolved.exists():
        shutil.rmtree(resolved)


def unpack_to_library(
    *,
    pack_path: str | os.PathLike[str],
    passphrase: str,
    library_root: str | os.PathLike[str],
    work_root: str | os.PathLike[str],
    manifest: InstallManifest,
    validate: Callable[[ExtractedWorkflowPack], None] | None = None,
    log: LogCallback | None = None,
) -> dict:
    work = Path(work_root).resolve() / "workflow-unpack"
    stage = work / f"stage-{uuid.uuid4().hex}"
    distributed_root = (
        Path(library_root).resolve() / DISTRIBUTION_LIBRARY_DIRNAME
    ).resolve()
    try:
        work.mkdir(parents=True, exist_ok=True)
        extracted = extract_workflow_pack(pack_path, stage, passphrase)
        if validate is not None:
            validate(extracted)
        release = extracted.release_version
        if not _RELEASE_RE.fullmatch(release):
            raise WorkflowLibraryError(f"잘못된 워크플로우 배포 버전입니다: {release}")
        catalog = _catalog_for_extracted(extracted, manifest)
        state = {
            "schema_version": 2,
            "release_version": release,
            "pack_sha256": extracted.pack_sha256,
            "workflow_count": len(catalog),
            "binding_count": len(extracted.workflow_bindings),
            "items": catalog,
        }
        _write_json_new(stage / _STATE_FILENAME, state)
        destination = distributed_root / release
        distributed_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing_state_path = destination / _STATE_FILENAME
            if not existing_state_path.is_file():
                raise WorkflowLibraryError(
                    "동일 배포 버전 폴더가 있지만 팩 상태 파일이 없습니다: "
                    f"{destination}"
                )
            existing = _read_json_object(existing_state_path, "기존 팩 상태")
            if existing.get("pack_sha256") != extracted.pack_sha256:
                raise WorkflowLibraryError(
                    "동일한 배포 버전에 다른 팩이 이미 풀려 있습니다. "
                    f"기존 파일을 덮어쓰지 않습니다: {release}"
                )
            _safe_remove_stage(stage, work)
            state = existing
            if log:
                log(f"[워크플로우] 동일한 배포 팩 재사용: {release}")
        else:
            os.replace(stage, destination)
            if log:
                log(
                    f"[워크플로우] 배포 원본 풀기 완료: {release}, "
                    f"{len(catalog)}개"
                )
        _mark_distribution_files_read_only(destination)
        if log:
            log(
                "[워크플로우] 배포 원본 읽기 전용 보호 완료: "
                f"{destination}"
            )
        return {
            **state,
            "directory": str(destination),
            "read_only": True,
        }
    except WorkflowLibraryError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY] 팩 풀기 실패: "
            f"pack={pack_path}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowLibraryError(f"워크플로우 팩 풀기 실패: {exc}") from exc


def _load_release(library_root: Path, release_version: str) -> tuple[Path, dict]:
    if not _RELEASE_RE.fullmatch(release_version):
        raise WorkflowLibraryError(
            f"워크플로우 배포 버전 형식이 잘못되었습니다: {release_version!r}"
        )
    root = (
        library_root / DISTRIBUTION_LIBRARY_DIRNAME / release_version
    ).resolve()
    state_path = root / _STATE_FILENAME
    if not state_path.is_file():
        raise WorkflowLibraryError(
            f"먼저 워크플로우 팩을 풀어야 합니다: {release_version}"
        )
    state = _read_json_object(state_path, "워크플로우 팩 상태")
    if state.get("release_version") != release_version:
        raise WorkflowLibraryError(
            f"워크플로우 팩 폴더와 상태 버전이 다릅니다: {root}"
        )
    return root, state


def distribution_e2e_catalog(
    *,
    library_root: str | os.PathLike[str],
    release_version: str,
    profile_by_binding: Mapping[str, str],
    excluded_filenames: Iterable[str] = (),
) -> dict:
    """List intact distribution originals with an explicit E2E method."""

    release_root, state = _load_release(
        Path(library_root).resolve(), release_version
    )
    _mark_distribution_files_read_only(release_root)
    items = state.get("items")
    if not isinstance(items, list):
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY] E2E 원본 목록 조회 실패: "
            f"release={release_version}, items 형식 오류"
        )
        raise WorkflowLibraryError("워크플로우 팩 항목 목록이 손상되었습니다.")

    excluded = {str(value).casefold() for value in excluded_filenames}
    available: list[dict] = []
    skipped: list[dict[str, str]] = []
    for raw_item in items:
        item_id = (
            str(raw_item.get("id", ""))
            if isinstance(raw_item, dict)
            else ""
        )
        reason = ""
        source: Path | None = None
        profiles: list[str] = []
        bindings: list[str] = []
        try:
            if not isinstance(raw_item, dict) or not item_id:
                reason = "배포 메타데이터 형식이 올바르지 않습니다."
            else:
                raw_filename = str(raw_item.get("filename", ""))
                filename = Path(raw_filename).name
                raw_bindings = raw_item.get("bindings")
                if not filename or raw_filename != filename:
                    reason = "배포 원본 파일명이 올바르지 않습니다."
                elif filename.casefold() in excluded:
                    reason = "E2E 제외 워크플로우입니다."
                elif not isinstance(raw_bindings, list) or not raw_bindings:
                    reason = "배포 바인딩 정보가 없습니다."
                elif not all(
                    isinstance(value, str) and value for value in raw_bindings
                ):
                    reason = "배포 바인딩 형식이 올바르지 않습니다."
                else:
                    bindings = [str(value) for value in raw_bindings]
                    unsupported = [
                        value
                        for value in bindings
                        if value not in profile_by_binding
                    ]
                    if unsupported:
                        reason = (
                            "등록된 E2E 검사 방법이 없는 바인딩입니다: "
                            + ", ".join(unsupported)
                        )
                    else:
                        profiles = list(
                            dict.fromkeys(
                                str(profile_by_binding[value])
                                for value in bindings
                            )
                        )
                        source = release_root / filename
                        if source.is_symlink() or not source.is_file():
                            reason = "배포 원본 파일이 실제로 존재하지 않습니다."
                        else:
                            expected_hash = str(raw_item.get("sha256", ""))
                            if not expected_hash:
                                reason = "배포 원본 SHA-256 기록이 없습니다."
                            elif _sha256_bytes(source.read_bytes()) != expected_hash:
                                reason = "배포 원본 SHA-256이 일치하지 않습니다."
            if reason:
                print(
                    "[COMFY_INSTALL][WORKFLOW_LIBRARY] E2E 대상 제외: "
                    f"release={release_version}, item={item_id or '<unknown>'}, "
                    f"reason={reason}"
                )
                skipped.append({"id": item_id, "reason": reason})
                continue

            assert isinstance(raw_item, dict)
            assert source is not None
            model_ids = raw_item.get("model_ids", [])
            available.append(
                {
                    "id": item_id,
                    "name": str(
                        raw_item.get("name")
                        or raw_item.get("filename")
                        or item_id
                    ),
                    "filename": source.name,
                    "bindings": bindings,
                    "e2e_profiles": profiles,
                    "model_count": int(
                        raw_item.get("model_count")
                        or (
                            len(model_ids)
                            if isinstance(model_ids, list)
                            else 0
                        )
                    ),
                    "model_bytes": int(raw_item.get("model_bytes") or 0),
                    "read_only": _is_read_only_file(source),
                }
            )
        except Exception as exc:
            print(
                "[COMFY_INSTALL][WORKFLOW_LIBRARY] E2E 원본 항목 검사 실패: "
                f"release={release_version}, item={item_id or '<unknown>'}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            skipped.append(
                {
                    "id": item_id,
                    "reason": f"원본 검사 오류: {exc}",
                }
            )

    return {
        "release_version": release_version,
        "directory": str(release_root),
        "source_kind": "read_only_distribution_original",
        "items": available,
        "available_count": len(available),
        "skipped": skipped,
        "skipped_count": len(skipped),
    }


def latest_release_version(
    library_root: str | os.PathLike[str],
) -> str:
    """Return the newest unpacked workflow-library release."""
    distributed_root = (
        Path(library_root).resolve() / DISTRIBUTION_LIBRARY_DIRNAME
    )
    candidates: list[tuple[int, str]] = []
    try:
        if not distributed_root.is_dir():
            print(
                "[COMFY_INSTALL][WORKFLOW_LIBRARY] 기본 워크플로우 배포 "
                f"폴더가 없습니다: {distributed_root}"
            )
            raise WorkflowLibraryError(
                "기본 경로를 설정할 워크플로우 팩이 없습니다. "
                "먼저 워크플로우 팩을 풀어주세요."
            )
        for child in distributed_root.iterdir():
            if (
                child.is_dir()
                and _RELEASE_RE.fullmatch(child.name)
                and (child / _STATE_FILENAME).is_file()
            ):
                candidates.append((int(child.name[1:]), child.name))
        if not candidates:
            print(
                "[COMFY_INSTALL][WORKFLOW_LIBRARY] 사용할 수 있는 기본 "
                f"워크플로우 배포 버전이 없습니다: {distributed_root}"
            )
            raise WorkflowLibraryError(
                "기본 경로를 설정할 워크플로우 배포 버전이 없습니다."
            )
        return max(candidates)[1]
    except WorkflowLibraryError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY] 최신 기본 워크플로우 "
            f"버전 확인 실패: root={distributed_root}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowLibraryError(
            f"최신 기본 워크플로우 버전 확인 실패: {exc}"
        ) from exc


def import_user_copies(
    *,
    comfy_root: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
    release_version: str,
    selected_item_ids: Iterable[str],
    log: LogCallback | None = None,
) -> WorkflowSelection:
    comfy = Path(comfy_root).resolve()
    release_root, state = _load_release(
        Path(library_root).resolve(), release_version
    )
    requested = tuple(dict.fromkeys(str(value) for value in selected_item_ids))
    if not requested:
        raise WorkflowLibraryError("가져올 워크플로우를 하나 이상 선택하세요.")
    items = state.get("items")
    if not isinstance(items, list):
        raise WorkflowLibraryError("워크플로우 팩 항목 목록이 손상되었습니다.")
    items_by_id = {
        str(item.get("id")): item for item in items if isinstance(item, dict)
    }
    missing = [item_id for item_id in requested if item_id not in items_by_id]
    if missing:
        raise WorkflowLibraryError(
            "선택한 워크플로우가 팩에 없습니다: " + ", ".join(missing)
        )
    user_root = embedded_workflow_base_dir(comfy)
    user_root.mkdir(parents=True, exist_ok=True)
    bindings: dict[str, str] = {}
    model_ids: set[str] = set()
    user_files: list[str] = []
    for item_id in requested:
        item = items_by_id[item_id]
        filename = Path(str(item.get("filename", ""))).name
        source = release_root / filename
        if not source.is_file():
            raise WorkflowLibraryError(
                f"배포 워크플로우 원본이 없습니다: {source}"
            )
        payload = source.read_bytes()
        expected_hash = str(item.get("sha256", ""))
        if _sha256_bytes(payload) != expected_hash:
            raise WorkflowLibraryError(
                f"배포 워크플로우 원본 해시가 손상되었습니다: {source}"
            )
        base_name = f"{source.stem}__{release_version}{source.suffix}"
        destination = user_root / base_name
        suffix = 2
        while destination.exists() and destination.read_bytes() != payload:
            destination = user_root / (
                f"{source.stem}__{release_version}_{suffix}{source.suffix}"
            )
            suffix += 1
        if destination.exists():
            if log:
                log(f"[워크플로우] 동일한 사용자 사본 재사용: {destination.name}")
        else:
            _write_bytes_new(destination, payload)
            if log:
                log(f"[워크플로우] 새 사용자 사본 생성: {destination.name}")
        for binding_key in item.get("bindings", []):
            bindings[str(binding_key)] = str(destination)
        model_ids.update(str(value) for value in item.get("model_ids", []))
        user_files.append(str(destination))
    return WorkflowSelection(
        release_version=release_version,
        selected_item_ids=requested,
        workflow_bindings=bindings,
        model_ids=tuple(sorted(model_ids)),
        user_files=tuple(user_files),
    )


def resolve_distribution_selection(
    *,
    library_root: str | os.PathLike[str],
    release_version: str,
    selected_item_ids: Iterable[str],
) -> WorkflowSelection:
    """Resolve immutable distribution originals without creating user copies."""
    try:
        release_root, state = _load_release(
            Path(library_root).resolve(), release_version
        )
        requested = tuple(
            dict.fromkeys(str(value) for value in selected_item_ids)
        )
        if not requested:
            raise WorkflowLibraryError(
                "E2E 검사할 배포 워크플로우를 하나 이상 선택하세요."
            )
        items = state.get("items")
        if not isinstance(items, list):
            raise WorkflowLibraryError(
                "워크플로우 팩 항목 목록이 손상되었습니다."
            )
        items_by_id = {
            str(item.get("id")): item
            for item in items
            if isinstance(item, dict)
        }
        missing = [
            item_id for item_id in requested if item_id not in items_by_id
        ]
        if missing:
            raise WorkflowLibraryError(
                "선택한 E2E 워크플로우가 배포 팩에 없습니다: "
                + ", ".join(missing)
            )

        bindings: dict[str, str] = {}
        model_ids: set[str] = set()
        source_files: list[str] = []
        for item_id in requested:
            item = items_by_id[item_id]
            filename = Path(str(item.get("filename", ""))).name
            source = release_root / filename
            if not source.is_file():
                raise WorkflowLibraryError(
                    f"E2E 배포 워크플로우 원본이 없습니다: {source}"
                )
            payload = source.read_bytes()
            expected_hash = str(item.get("sha256", ""))
            if _sha256_bytes(payload) != expected_hash:
                raise WorkflowLibraryError(
                    f"E2E 배포 워크플로우 원본 해시가 손상되었습니다: {source}"
                )
            raw_bindings = item.get("bindings")
            if not isinstance(raw_bindings, list) or not raw_bindings:
                raise WorkflowLibraryError(
                    "E2E 배포 워크플로우 바인딩이 비어 있습니다: "
                    f"{item_id}"
                )
            for binding_key in raw_bindings:
                bindings[str(binding_key)] = str(source)
            model_ids.update(str(value) for value in item.get("model_ids", []))
            source_files.append(str(source))
        return WorkflowSelection(
            release_version=release_version,
            selected_item_ids=requested,
            workflow_bindings=bindings,
            model_ids=tuple(sorted(model_ids)),
            user_files=tuple(source_files),
        )
    except WorkflowLibraryError as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY] E2E 배포 원본 선택 실패: "
            f"release={release_version!r}, selected={list(selected_item_ids)!r}, "
            f"error={exc}"
        )
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY] E2E 배포 원본 선택 중 "
            "예상하지 못한 실패: "
            f"release={release_version!r}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowLibraryError(
            f"E2E 배포 워크플로우 선택 실패: {exc}"
        ) from exc


def import_default_user_copies(
    *,
    comfy_root: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
    release_version: str | None = None,
    log: LogCallback | None = None,
) -> WorkflowSelection:
    """Ensure every workflow in one release has an embedded user copy.

    The release metadata is the sole source of truth for default config
    bindings.  A future pack can add or rename workflows without adding a
    second filename map to the installer or server config.
    """
    release = release_version or latest_release_version(library_root)
    try:
        _, state = _load_release(Path(library_root).resolve(), release)
        items = state.get("items")
        if not isinstance(items, list) or not items:
            print(
                "[COMFY_INSTALL][WORKFLOW_LIBRARY] 기본 워크플로우 항목이 "
                f"비어 있습니다: release={release}, items={items!r}"
            )
            raise WorkflowLibraryError(
                f"기본 워크플로우 배포 항목이 비어 있습니다: {release}"
            )
        item_ids: list[str] = []
        expected_bindings: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                print(
                    "[COMFY_INSTALL][WORKFLOW_LIBRARY] 기본 워크플로우 "
                    f"항목 형식 오류: release={release}, item={item!r}"
                )
                raise WorkflowLibraryError(
                    f"기본 워크플로우 항목 형식이 잘못되었습니다: {release}"
                )
            item_id = str(item.get("id") or "").strip()
            bindings = item.get("bindings")
            if not item_id or not isinstance(bindings, list) or not bindings:
                print(
                    "[COMFY_INSTALL][WORKFLOW_LIBRARY] 기본 워크플로우 "
                    "ID/바인딩 누락: "
                    f"release={release}, item={item!r}"
                )
                raise WorkflowLibraryError(
                    f"기본 워크플로우 ID/바인딩이 누락되었습니다: {release}"
                )
            item_ids.append(item_id)
            expected_bindings.update(str(value) for value in bindings)

        selection = import_user_copies(
            comfy_root=comfy_root,
            library_root=library_root,
            release_version=release,
            selected_item_ids=item_ids,
            log=log,
        )
        actual_bindings = set(selection.workflow_bindings)
        if actual_bindings != expected_bindings:
            print(
                "[COMFY_INSTALL][WORKFLOW_LIBRARY] 기본 워크플로우 바인딩 "
                "검증 실패: "
                f"release={release}, missing="
                f"{sorted(expected_bindings - actual_bindings)}, extra="
                f"{sorted(actual_bindings - expected_bindings)}"
            )
            raise WorkflowLibraryError(
                f"기본 워크플로우 바인딩 검증에 실패했습니다: {release}"
            )
        return selection
    except WorkflowLibraryError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_LIBRARY] 기본 워크플로우 사용자 "
            f"사본 준비 실패: release={release}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowLibraryError(
            f"기본 워크플로우 사용자 사본 준비 실패: {exc}"
        ) from exc


def selection_requirements(
    *,
    library_root: str | os.PathLike[str],
    release_version: str,
    selected_item_ids: Iterable[str],
) -> dict:
    _, state = _load_release(Path(library_root).resolve(), release_version)
    requested = tuple(dict.fromkeys(str(value) for value in selected_item_ids))
    if not requested:
        raise WorkflowLibraryError("가져올 워크플로우를 하나 이상 선택하세요.")
    items = state.get("items")
    if not isinstance(items, list):
        raise WorkflowLibraryError("워크플로우 팩 항목 목록이 손상되었습니다.")
    items_by_id = {
        str(item.get("id")): item for item in items if isinstance(item, dict)
    }
    missing = [item_id for item_id in requested if item_id not in items_by_id]
    if missing:
        raise WorkflowLibraryError(
            "선택한 워크플로우가 팩에 없습니다: " + ", ".join(missing)
        )
    model_sizes: dict[str, int] = {}
    for item_id in requested:
        item = items_by_id[item_id]
        raw_sizes = item.get("model_sizes", {})
        if isinstance(raw_sizes, dict):
            for model_id, size in raw_sizes.items():
                model_sizes[str(model_id)] = int(size)
        else:
            print(
                "[COMFY_INSTALL][WORKFLOW_LIBRARY] model_sizes 형식 오류: "
                f"release={release_version}, item={item_id}"
            )
    return {
        "release_version": release_version,
        "selected_item_ids": list(requested),
        "model_ids": sorted(model_sizes),
        "model_bytes": sum(model_sizes.values()),
    }


def library_status(
    comfy_root: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
) -> dict:
    comfy = Path(comfy_root).resolve()
    workflows_root = comfy / "user" / "default" / "workflows"
    distributed_root = (
        Path(library_root).resolve() / DISTRIBUTION_LIBRARY_DIRNAME
    )
    user_root = embedded_workflow_base_dir(comfy)
    releases: list[dict] = []
    if distributed_root.is_dir():
        for child in sorted(
            distributed_root.iterdir(), key=lambda path: path.name.casefold()
        ):
            state_path = child / _STATE_FILENAME
            if not child.is_dir() or not state_path.is_file():
                continue
            try:
                state = _read_json_object(state_path, "워크플로우 팩 상태")
                releases.append({**state, "directory": str(child)})
            except WorkflowLibraryError as exc:
                print(
                    "[COMFY_INSTALL][WORKFLOW_LIBRARY] 손상된 배포 버전 제외: "
                    f"path={child}, error={exc}"
                )
                traceback.print_exc()
    user_files = []
    if user_root.is_dir():
        user_files = [
            {
                "name": path.name,
                "path": str(path),
                "size": path.stat().st_size,
            }
            for path in sorted(user_root.glob("*.json"), key=lambda p: p.name.casefold())
            if path.is_file()
        ]
    return {
        "releases": releases,
        "user_files": user_files,
        "distributed_root": str(distributed_root),
        "user_root": str(user_root),
    }
