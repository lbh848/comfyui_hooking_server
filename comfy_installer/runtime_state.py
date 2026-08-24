from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .manager_dependencies import (
    ManagerDependencyError,
    expected_manager_version,
    installed_manager_versions,
)
from .manifest import InstallManifest


class RuntimeStateError(RuntimeError):
    """Comfy 런타임 receipt 또는 승격 트랜잭션 처리 실패."""


LogCallback = Callable[[str], None]
RECEIPT_SCHEMA_VERSION = 1
_COMFY_SERVER = Path("server.py")
_COMFY_SERVER_PATCH_MARKER = (
    "# comfy-installer: keep system_stats available when GPU telemetry fails"
)
_INSTANT_LORA_NODE_NAME = "comfyui-instant-lora_v_soya"
_INSTANT_LORA_RUNTIME = Path("src") / "runtime.py"
_INSTANT_LORA_PATCH_MARKER = (
    "# comfy-installer: use the project-managed Python 3.12 runtime"
)
_REUSE_IF_SAME_ORIGIN = "reuse_if_same_origin"


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_comfy_database(
    root: Path,
    transaction_root: Path,
) -> dict[str, Any]:
    database = (root / "user" / "comfyui.db").resolve()
    expected_database = (root / "user" / "comfyui.db").resolve()
    if database != expected_database:
        raise RuntimeStateError(f"안전하지 않은 Comfy DB 경로입니다: {database}")
    state: dict[str, Any] = {
        "path": str(database),
        "existed": database.is_file(),
        "backup_path": None,
        "sha256": None,
        "previous_backup_path": None,
    }
    if not database.is_file():
        return state

    backup_root = transaction_root / "database"
    backup_root.mkdir(parents=True, exist_ok=False)
    backup = backup_root / "comfyui.db"
    source_connection = None
    destination_connection = None
    try:
        source_connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro",
            uri=True,
        )
        destination_connection = sqlite3.connect(str(backup))
        source_connection.backup(destination_connection)
        destination_connection.commit()
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] Comfy SQLite 백업 실패: "
            f"source={database}, target={backup}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(f"Comfy SQLite 백업 실패: {database}") from exc
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if source_connection is not None:
            source_connection.close()

    previous_backup = database.with_name(f"{database.name}.bkp")
    if previous_backup.is_file():
        saved_previous_backup = backup_root / previous_backup.name
        shutil.copy2(previous_backup, saved_previous_backup)
        state["previous_backup_path"] = str(saved_previous_backup)
    state.update(
        {
            "backup_path": str(backup),
            "sha256": _file_sha256(backup),
        }
    )
    return state


def _restore_comfy_database(
    root: Path,
    transaction_root: Path,
    state: dict[str, Any],
) -> str:
    database = (root / "user" / "comfyui.db").resolve()
    if Path(str(state.get("path") or "")).resolve() != database:
        raise RuntimeStateError(
            f"롤백 snapshot의 Comfy DB 경로가 안전하지 않습니다: {state.get('path')}"
        )
    database.parent.mkdir(parents=True, exist_ok=True)
    related = (
        database,
        database.with_name(f"{database.name}-wal"),
        database.with_name(f"{database.name}-shm"),
        database.with_name(f"{database.name}.bkp"),
    )
    for path in related:
        if path.exists():
            if not path.is_file() and not path.is_symlink():
                raise RuntimeStateError(
                    f"지원하지 않는 Comfy DB 롤백 경로 형식입니다: {path}"
                )
            path.unlink()

    if not state.get("existed"):
        return "removed_new_database"

    backup = Path(str(state.get("backup_path") or "")).resolve()
    expected_backup = (transaction_root / "database" / "comfyui.db").resolve()
    if backup != expected_backup or not backup.is_file():
        raise RuntimeStateError(f"Comfy DB 백업 파일이 안전하지 않습니다: {backup}")
    expected_hash = str(state.get("sha256") or "")
    if not expected_hash or _file_sha256(backup) != expected_hash:
        raise RuntimeStateError(f"Comfy DB 백업 해시가 손상되었습니다: {backup}")
    temporary = database.with_name(f".{database.name}.restore_{uuid.uuid4().hex[:8]}")
    try:
        shutil.copy2(backup, temporary)
        os.replace(temporary, database)
    finally:
        if temporary.exists():
            temporary.unlink()

    previous_backup_raw = state.get("previous_backup_path")
    if previous_backup_raw:
        previous_backup = Path(str(previous_backup_raw)).resolve()
        expected_previous = (
            transaction_root / "database" / "comfyui.db.bkp"
        ).resolve()
        if previous_backup != expected_previous or not previous_backup.is_file():
            raise RuntimeStateError(
                f"기존 Comfy DB 백업 파일이 안전하지 않습니다: {previous_backup}"
            )
        shutil.copy2(previous_backup, database.with_name(f"{database.name}.bkp"))
    return "restored_database"


def _git_value(path: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] Git 상태 확인 실패: "
            f"path={path}, arguments={arguments!r}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(
            f"Git 상태를 확인하지 못했습니다: {path}"
        ) from exc


def git_head(path: str | os.PathLike[str]) -> str | None:
    root = Path(path).resolve()
    if not (root / ".git").is_dir():
        return None
    return _git_value(root, "rev-parse", "HEAD").lower()


def _normalize_git_repository(repository: str) -> str:
    return repository.rstrip("/").removesuffix(".git").casefold()


def _capture_managed_comfy_patch(path: Path) -> str | None:
    status = _git_value(path, "status", "--porcelain", "--untracked-files=no")
    if not status:
        return None
    expected_status = f"M {_COMFY_SERVER.as_posix()}"
    normalized = [line.strip() for line in status.splitlines() if line.strip()]
    server_path = path / _COMFY_SERVER
    if (
        normalized != [expected_status]
        or not server_path.is_file()
        or _COMFY_SERVER_PATCH_MARKER
        not in server_path.read_text(encoding="utf-8")
    ):
        raise RuntimeStateError(
            "ComfyUI에 설치기 소유가 아닌 추적 변경이 있어 승격하지 "
            "않습니다: "
            + ", ".join(normalized[:10])
        )
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary", "--", _COMFY_SERVER.as_posix()],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        patch = completed.stdout
        if not patch.strip():
            raise RuntimeStateError("ComfyUI 관리 패치 diff가 비어 있습니다.")
        return patch
    except RuntimeStateError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] 설치기 소유 ComfyUI 패치 "
            f"캡처 실패: path={path}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(
            "ComfyUI 관리 패치를 보관하지 못했습니다."
        ) from exc


def _capture_managed_custom_node_patch(
    path: Path,
    *,
    name: str,
) -> str | None:
    status = _git_value(path, "status", "--porcelain", "--untracked-files=no")
    if not status:
        return None
    expected_status = f"M {_INSTANT_LORA_RUNTIME.as_posix()}"
    normalized = [line.strip() for line in status.splitlines() if line.strip()]
    runtime_path = path / _INSTANT_LORA_RUNTIME
    if (
        name != _INSTANT_LORA_NODE_NAME
        or normalized != [expected_status]
        or not runtime_path.is_file()
        or _INSTANT_LORA_PATCH_MARKER
        not in runtime_path.read_text(encoding="utf-8")
    ):
        raise RuntimeStateError(
            f"커스텀 노드 {name}에 설치기 소유가 아닌 추적 변경이 있어 "
            "승격하지 않습니다: "
            + ", ".join(normalized[:10])
        )
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary", "--", _INSTANT_LORA_RUNTIME.as_posix()],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        patch = completed.stdout
        if not patch.strip():
            raise RuntimeStateError(
                f"커스텀 노드 {name} 관리 패치 diff가 비어 있습니다."
            )
        return patch
    except RuntimeStateError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] 설치기 소유 노드 패치 캡처 실패: "
            f"name={name}, path={path}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(
            f"커스텀 노드 {name} 관리 패치를 보관하지 못했습니다."
        ) from exc


def receipt_path(comfy_root: str | os.PathLike[str]) -> Path:
    return Path(comfy_root).resolve() / ".installer-state" / "runtime-receipt.json"


def load_runtime_receipt(comfy_root: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = receipt_path(comfy_root)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeStateError(f"runtime receipt가 객체가 아닙니다: {path}")
        if value.get("schema_version") != RECEIPT_SCHEMA_VERSION:
            raise RuntimeStateError(
                "지원하지 않는 runtime receipt 버전입니다: "
                f"{value.get('schema_version')!r}"
            )
        return value
    except RuntimeStateError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] runtime receipt 읽기 실패: "
            f"path={path}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(f"runtime receipt 읽기 실패: {path}") from exc


def desired_python_signature(
    manifest: InstallManifest,
    *,
    profile_id: str,
    install_mode: str,
) -> str:
    return _json_hash(
        {
            "python": manifest.python,
            "profile_id": str(profile_id),
            "install_mode": str(install_mode),
        }
    )


def collect_custom_node_state(
    comfy_root: str | os.PathLike[str],
    nodes: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    root = Path(comfy_root).resolve() / "custom_nodes"
    result: dict[str, dict[str, Any]] = {}
    for node in nodes:
        name = str(node.get("name") or "").strip()
        if not name:
            raise RuntimeStateError("이름이 없는 커스텀 노드는 조사할 수 없습니다.")
        destination = root / name
        source_type = str(node.get("source_type") or "")
        state: dict[str, Any] = {
            "source_type": source_type,
            "path": str(destination),
            "exists": destination.is_dir(),
        }
        if source_type == "git" and destination.is_dir():
            state["head"] = git_head(destination)
            state["tracking_branch"] = node.get("tracking_branch")
            state["expected_ref"] = node.get("ref")
            state["repository"] = node.get("repository")
            state["existing_policy"] = node.get("existing_policy")
            if node.get("existing_policy") == _REUSE_IF_SAME_ORIGIN:
                state["actual_repository"] = _git_value(
                    destination,
                    "remote",
                    "get-url",
                    "origin",
                )
        elif source_type == "archive" and destination.is_dir():
            marker_path = destination / ".comfy-installer-source.json"
            marker = None
            if marker_path.is_file():
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][RUNTIME_STATE] 노드 표식 읽기 실패: "
                        f"path={marker_path}, error={exc}"
                    )
                    traceback.print_exc()
            state["archive_sha256"] = (
                marker.get("sha256") if isinstance(marker, dict) else None
            )
            state["expected_sha256"] = node.get("sha256")
        result[name] = state
    return result


def collect_manager_state(
    comfy_root: str | os.PathLike[str],
) -> dict[str, Any]:
    root = Path(comfy_root).resolve()
    try:
        expected = expected_manager_version(root)
        installed = installed_manager_versions(root)
    except ManagerDependencyError as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] Manager 상태 조사 실패: "
            f"root={root}, error={exc}"
        )
        traceback.print_exc()
        return {
            "requirements_path": str(root / "manager_requirements.txt"),
            "expected_version": None,
            "installed_versions": [],
            "status": "invalid",
            "error": str(exc),
        }

    if not installed:
        status = "missing"
    elif installed != [expected]:
        status = "version_mismatch"
    else:
        status = "current"
    return {
        "requirements_path": str(root / "manager_requirements.txt"),
        "expected_version": expected,
        "installed_versions": installed,
        "status": status,
        "error": None,
    }


def inspect_runtime(
    *,
    comfy_root: str | os.PathLike[str],
    manifest: InstallManifest,
    profile_id: str,
    install_mode: str,
) -> dict[str, Any]:
    root = Path(comfy_root).resolve()
    receipt = load_runtime_receipt(root)
    actual_ref = git_head(root)
    python_signature = desired_python_signature(
        manifest,
        profile_id=profile_id,
        install_mode=install_mode,
    )
    manager = collect_manager_state(root)
    nodes = collect_custom_node_state(root, manifest.custom_nodes)
    receipt_python = (
        receipt.get("python", {}) if isinstance(receipt, dict) else {}
    )
    receipt_nodes = (
        receipt.get("custom_nodes", {}) if isinstance(receipt, dict) else {}
    )
    node_manifest_signature = _json_hash(manifest.custom_nodes)
    reasons: list[str] = []
    if actual_ref != str(manifest.comfy["ref"]).lower():
        reasons.append("comfy_ref")
    if not isinstance(receipt, dict):
        reasons.append("missing_receipt")
    if receipt_python.get("signature") != python_signature:
        reasons.append("python_profile")
    manager_status = manager.get("status")
    if manager_status == "invalid":
        reasons.append("manager_requirements")
    elif manager_status == "missing":
        reasons.append("manager_missing")
    elif manager_status == "version_mismatch":
        reasons.append("manager_version")
    receipt_node_signature = (
        receipt.get("custom_node_manifest_signature")
        if isinstance(receipt, dict)
        else None
    )
    if (
        not isinstance(receipt_nodes, dict)
        or receipt_node_signature != node_manifest_signature
    ):
        reasons.append("custom_node_manifest")
    for name, state in nodes.items():
        if not state.get("exists"):
            reasons.append(f"custom_node_missing:{name}")
            continue
        same_origin_policy = False
        if (
            state.get("source_type") == "git"
            and state.get("existing_policy") == _REUSE_IF_SAME_ORIGIN
        ):
            expected_repository = str(state.get("repository") or "")
            actual_repository = str(state.get("actual_repository") or "")
            same_origin_policy = _normalize_git_repository(
                actual_repository
            ) == _normalize_git_repository(expected_repository)
            if not same_origin_policy:
                reasons.append(f"custom_node_repository:{name}")
        if state.get("source_type") == "git" and state.get("expected_ref"):
            if state.get("head") != str(state["expected_ref"]).lower():
                if same_origin_policy:
                    print(
                        "[COMFY_INSTALL][RUNTIME_STATE] 기존 Git 설치 정책으로 "
                        f"고정점 차이를 허용합니다: name={name}, "
                        f"expected={state['expected_ref']}, "
                        f"actual={state.get('head')}, "
                        f"origin={state.get('actual_repository')}"
                    )
                else:
                    reasons.append(f"custom_node_ref:{name}")
        if state.get("source_type") == "git" and state.get("tracking_branch"):
            repository = str(state.get("repository") or "")
            branch = str(state.get("tracking_branch") or "")
            try:
                remote = subprocess.run(
                    ["git", "ls-remote", repository, f"refs/heads/{branch}"],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                ).stdout.strip()
                remote_head = remote.split()[0].lower() if remote else None
                state["remote_head"] = remote_head
                if remote_head is None:
                    raise RuntimeStateError(
                        f"추적 브랜치 HEAD가 비어 있습니다: {name} {branch}"
                    )
                if state.get("head") != remote_head:
                    reasons.append(f"custom_node_tracking:{name}")
            except RuntimeStateError:
                raise
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][RUNTIME_STATE] 추적 노드 원격 HEAD 확인 실패: "
                    f"name={name}, repository={repository}, branch={branch}, error={exc}"
                )
                traceback.print_exc()
                raise RuntimeStateError(
                    f"추적 커스텀 노드 원격 HEAD 확인 실패: {name}"
                ) from exc
        if state.get("source_type") == "archive":
            if state.get("archive_sha256") != state.get("expected_sha256"):
                reasons.append(f"custom_node_archive:{name}")
    return {
        "actual_comfy_ref": actual_ref,
        "expected_comfy_ref": str(manifest.comfy["ref"]).lower(),
        "python_signature": python_signature,
        "manager": manager,
        "custom_node_manifest_signature": node_manifest_signature,
        "custom_nodes": nodes,
        "receipt": receipt,
        "runtime_change_reasons": list(dict.fromkeys(reasons)),
        "runtime_changed": bool(reasons),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing_{uuid.uuid4().hex[:8]}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] JSON 저장 실패: "
            f"path={path}, temporary={temporary}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(f"런타임 상태 저장 실패: {path}") from exc


def write_runtime_receipt(
    *,
    comfy_root: str | os.PathLike[str],
    manifest: InstallManifest,
    profile_id: str,
    install_mode: str,
    workflow_bindings: dict[str, str],
    selected_workflow_ids: Iterable[str],
    release_version: str | None,
) -> dict[str, Any]:
    root = Path(comfy_root).resolve()
    value = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "written_at": _now_iso(),
        "manifest_sha256": manifest.sha256,
        "comfy": {
            "version": str(manifest.comfy["version"]),
            "expected_ref": str(manifest.comfy["ref"]).lower(),
            "actual_ref": git_head(root),
        },
        "python": {
            "version": str(manifest.python["version"]),
            "profile_id": str(profile_id),
            "install_mode": str(install_mode),
            "signature": desired_python_signature(
                manifest,
                profile_id=profile_id,
                install_mode=install_mode,
            ),
        },
        "manager": collect_manager_state(root),
        "custom_node_manifest_signature": _json_hash(manifest.custom_nodes),
        "custom_nodes": collect_custom_node_state(root, manifest.custom_nodes),
        "workflows": {
            "release_version": release_version,
            "selected_item_ids": [str(value) for value in selected_workflow_ids],
            "bindings": {
                str(key): str(path) for key, path in workflow_bindings.items()
            },
        },
    }
    _write_json_atomic(receipt_path(root), value)
    return value


def create_runtime_transaction(
    *,
    comfy_root: str | os.PathLike[str],
    manifest: InstallManifest,
    config_backup: dict[str, Any],
    log: LogCallback | None = None,
) -> dict[str, Any]:
    root = Path(comfy_root).resolve()
    transaction_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    transaction_root = root / ".installer-state" / "transactions" / transaction_id
    venv = root / ".venv"
    venv_backup = transaction_root / "venv"
    try:
        if not (root / ".git").is_dir():
            raise RuntimeStateError(f"ComfyUI Git 저장소가 없습니다: {root}")
        managed_comfy_patch = _capture_managed_comfy_patch(root)
        node_state = collect_custom_node_state(root, manifest.custom_nodes)
        for name, state in node_state.items():
            path = Path(str(state["path"]))
            if state.get("source_type") == "git" and path.is_dir():
                managed_patch = _capture_managed_custom_node_patch(
                    path,
                    name=name,
                )
                if managed_patch is not None:
                    state["managed_worktree_patch"] = managed_patch
        transaction_root.mkdir(parents=True, exist_ok=False)
        database = _backup_comfy_database(root, transaction_root)
        if venv.is_dir():
            if log:
                log(f"[트랜잭션] Comfy .venv 백업 시작: {venv_backup}")
            shutil.copytree(venv, venv_backup, symlinks=True)
        receipt = load_runtime_receipt(root)
        snapshot = {
            "schema_version": 2,
            "transaction_id": transaction_id,
            "created_at": _now_iso(),
            "comfy_root": str(root),
            "comfy_ref": git_head(root),
            "comfy_managed_worktree_patch": managed_comfy_patch,
            "custom_nodes": node_state,
            "config_backup": config_backup,
            "receipt": receipt,
            "venv_backup": str(venv_backup) if venv_backup.is_dir() else None,
            "database": database,
        }
        _write_json_atomic(transaction_root / "snapshot.json", snapshot)
        if log:
            log(f"[트랜잭션] 승격 전 상태 보관 완료: {transaction_id}")
        return snapshot
    except RuntimeStateError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] 트랜잭션 생성 실패: "
            f"root={transaction_root}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(f"승격 트랜잭션 생성 실패: {exc}") from exc


def _checkout_exact(path: Path, ref: str, label: str) -> None:
    try:
        subprocess.run(
            ["git", "checkout", "--detach", "--force", ref],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        actual = git_head(path)
        if actual != ref.lower():
            raise RuntimeStateError(
                f"{label} 롤백 HEAD가 다릅니다: expected={ref}, actual={actual}"
            )
    except RuntimeStateError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] Git 롤백 실패: "
            f"label={label}, path={path}, ref={ref}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(f"{label} Git 롤백 실패: {ref}") from exc


def _remove_tree_force_writable(path: Path) -> None:
    def make_writable_and_retry(function, target, exc_info) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            function(target)
        except Exception as retry_exc:
            print(
                "[COMFY_INSTALL][RUNTIME_STATE] 읽기 전용 롤백 파일 삭제 실패: "
                f"path={target}, original={exc_info}, retry={retry_exc}"
            )
            traceback.print_exc()
            raise

    shutil.rmtree(path, onexc=make_writable_and_retry)


def _restore_managed_worktree_patch(
    path: Path,
    patch: str,
    *,
    label: str,
) -> None:
    try:
        completed = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=path,
            input=patch,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.stderr.strip() and "warning" not in completed.stderr.lower():
            raise RuntimeStateError(
                f"{label} 관리 패치 복원 중 예상하지 못한 stderr: "
                f"{completed.stderr.strip()}"
            )
    except RuntimeStateError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] 설치기 소유 노드 패치 복원 실패: "
            f"label={label}, path={path}, error={exc}"
        )
        traceback.print_exc()
        raise RuntimeStateError(f"{label} 관리 패치 복원 실패") from exc


def rollback_runtime_transaction(
    *,
    comfy_root: str | os.PathLike[str],
    snapshot: dict[str, Any],
    restore_config: Callable[[str], dict[str, Any]],
    log: LogCallback | None = None,
) -> dict[str, Any]:
    root = Path(comfy_root).resolve()
    if Path(str(snapshot.get("comfy_root") or "")).resolve() != root:
        raise RuntimeStateError("롤백 snapshot의 Comfy 경로가 현재 대상과 다릅니다.")
    transaction_id = str(snapshot.get("transaction_id") or "")
    result: dict[str, Any] = {
        "transaction_id": transaction_id,
        "status": "running",
        "restored": [],
    }
    try:
        transaction_root = (
            root / ".installer-state" / "transactions" / transaction_id
        ).resolve()
        comfy_ref = str(snapshot.get("comfy_ref") or "")
        if comfy_ref:
            _checkout_exact(root, comfy_ref, "ComfyUI")
            result["restored"].append("comfy_ref")
            managed_comfy_patch = snapshot.get("comfy_managed_worktree_patch")
            if isinstance(managed_comfy_patch, str) and managed_comfy_patch:
                _restore_managed_worktree_patch(
                    root,
                    managed_comfy_patch,
                    label="ComfyUI",
                )
                result["restored"].append("comfy_managed_patch")
        for name, state in dict(snapshot.get("custom_nodes") or {}).items():
            if not isinstance(state, dict):
                continue
            node_ref = str(state.get("head") or "")
            node_path = Path(str(state.get("path") or "")).resolve()
            expected_node_root = (root / "custom_nodes").resolve()
            if not state.get("exists") and node_path.exists():
                if (
                    node_path.parent != expected_node_root
                    or node_path.name != str(name)
                ):
                    raise RuntimeStateError(
                        f"안전하지 않은 신규 커스텀 노드 롤백 대상입니다: {node_path}"
                    )
                if node_path.is_symlink() or node_path.is_file():
                    node_path.unlink()
                elif node_path.is_dir():
                    _remove_tree_force_writable(node_path)
                else:
                    raise RuntimeStateError(
                        f"지원하지 않는 신규 노드 경로 형식입니다: {node_path}"
                    )
                result["restored"].append(f"custom_node_removed:{name}")
                continue
            if state.get("source_type") != "git":
                continue
            if node_ref and node_path.is_dir():
                _checkout_exact(node_path, node_ref, f"커스텀 노드 {name}")
                result["restored"].append(f"custom_node:{name}")
                managed_patch = state.get("managed_worktree_patch")
                if isinstance(managed_patch, str) and managed_patch:
                    _restore_managed_worktree_patch(
                        node_path,
                        managed_patch,
                        label=f"커스텀 노드 {name}",
                    )
                    result["restored"].append(
                        f"custom_node_managed_patch:{name}"
                    )

        venv = root / ".venv"
        venv_backup_raw = snapshot.get("venv_backup")
        if venv_backup_raw:
            venv_backup = Path(str(venv_backup_raw)).resolve()
            expected_parent = root / ".installer-state" / "transactions" / transaction_id
            if venv_backup.parent != expected_parent.resolve() or venv_backup.name != "venv":
                raise RuntimeStateError(
                    f"안전하지 않은 .venv 백업 경로입니다: {venv_backup}"
                )
            if venv.exists():
                if venv.resolve().parent != root or venv.name != ".venv":
                    raise RuntimeStateError(f"안전하지 않은 .venv 삭제 대상입니다: {venv}")
                shutil.rmtree(venv)
            shutil.copytree(venv_backup, venv, symlinks=True)
            result["restored"].append("venv")

        database_state = snapshot.get("database")
        if isinstance(database_state, dict):
            database_result = _restore_comfy_database(
                root,
                transaction_root,
                database_state,
            )
            result["restored"].append(database_result)

        prior_receipt = snapshot.get("receipt")
        current_receipt = receipt_path(root)
        if isinstance(prior_receipt, dict):
            _write_json_atomic(current_receipt, prior_receipt)
        elif current_receipt.exists():
            current_receipt.unlink()
        result["restored"].append("runtime_receipt")

        config_backup = snapshot.get("config_backup")
        backup_path = (
            config_backup.get("backup_path")
            if isinstance(config_backup, dict)
            else None
        )
        if backup_path:
            restore_config(str(backup_path))
            result["restored"].append("config")
        result["status"] = "succeeded"
        result["completed_at"] = _now_iso()
        if log:
            log(f"[롤백] 이전 Comfy 런타임 복원 완료: {transaction_id}")
        return result
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] 자동 롤백 실패: "
            f"transaction_id={transaction_id}, error={exc}"
        )
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(exc)
        result["completed_at"] = _now_iso()
        raise RuntimeStateError(f"자동 롤백 실패: {exc}") from exc


def complete_runtime_transaction(
    *,
    comfy_root: str | os.PathLike[str],
    snapshot: dict[str, Any],
    log: LogCallback | None = None,
) -> dict[str, Any]:
    root = Path(comfy_root).resolve()
    transaction_id = str(snapshot.get("transaction_id") or "")
    transaction_root = root / ".installer-state" / "transactions" / transaction_id
    result = {
        "transaction_id": transaction_id,
        "status": "committed",
        "completed_at": _now_iso(),
    }
    _write_json_atomic(transaction_root / "result.json", result)
    venv_backup = transaction_root / "venv"
    database_backup = transaction_root / "database"
    try:
        if venv_backup.is_dir():
            shutil.rmtree(venv_backup)
        if database_backup.is_dir():
            shutil.rmtree(database_backup)
        if log:
            log(f"[트랜잭션] 승격 확정 완료: {transaction_id}")
        return result
    except Exception as exc:
        print(
            "[COMFY_INSTALL][RUNTIME_STATE] 성공한 트랜잭션 백업 정리 실패: "
            f"path={venv_backup}, error={exc}"
        )
        traceback.print_exc()
        result["cleanup_warning"] = str(exc)
        return result
