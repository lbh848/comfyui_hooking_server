from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

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
        return {
            **state,
            "directory": str(destination),
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
    user_root = (
        comfy / "user" / "default" / "workflows" / USER_WORKFLOW_DIRNAME
    ).resolve()
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
    user_root = workflows_root / USER_WORKFLOW_DIRNAME
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
