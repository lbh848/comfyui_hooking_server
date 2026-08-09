from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping
from urllib.parse import urlsplit

from .manifest import load_manifest


LOCAL_NODE_MAX_BYTES = 256 * 1024 * 1024
LOCAL_NODES_TOTAL_MAX_BYTES = 512 * 1024 * 1024
LOCAL_COPY_IGNORE_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)
LOCAL_COPY_IGNORE_SUFFIXES = frozenset({".pyc", ".pyo"})
LOCAL_COPY_IGNORE_PATTERNS = (
    ".git",
    ".git/**",
    "**/.git/**",
    ".mypy_cache",
    ".mypy_cache/**",
    "**/.mypy_cache/**",
    ".pytest_cache",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
    ".ruff_cache",
    ".ruff_cache/**",
    "**/.ruff_cache/**",
    ".venv",
    ".venv/**",
    "**/.venv/**",
    "venv",
    "venv/**",
    "**/venv/**",
    "__pycache__",
    "__pycache__/**",
    "**/__pycache__/**",
    "node_modules",
    "node_modules/**",
    "**/node_modules/**",
    "**/*.pyc",
    "**/*.pyo",
)


def _is_safe_node_name(name: str) -> bool:
    raw = str(name or "")
    return bool(raw and raw not in {".", ".."} and Path(raw).name == raw)


def _manifest_node_names(project_root: Path) -> tuple[list[str], set[str]]:
    manifest = load_manifest(project_root)
    names: list[str] = []
    normalized: set[str] = set()
    for raw in manifest.get("custom_nodes", []):
        if not isinstance(raw, Mapping):
            print(
                "[MODAL_CUSTOM_NODES] manifest custom node 형식 오류: "
                f"type={type(raw).__name__}, value={raw!r}"
            )
            continue
        name = str(raw.get("name") or "").strip()
        if not _is_safe_node_name(name):
            print(
                "[MODAL_CUSTOM_NODES] manifest custom node 이름 제외: "
                f"name={name!r}"
            )
            continue
        names.append(name)
        normalized.add(name.casefold())
    return names, normalized


def _git_output(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout).strip()
        print(
            "[MODAL_CUSTOM_NODES] Git 정보 조회 실패: "
            f"path={path}, args={args!r}, exit_code={completed.returncode}, "
            f"error={error[-1000:]}"
        )
        raise RuntimeError(error or f"git exit code {completed.returncode}")
    return completed.stdout.strip()


def _public_git_remote(remote: str) -> bool:
    raw = str(remote or "").strip()
    if not raw:
        return False
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        print(
            "[MODAL_CUSTOM_NODES] Git remote URL 파싱 실패: "
            f"remote={raw!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return False
    return (
        parsed.scheme in {"http", "https", "git"}
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
    )


def _local_tree_size(path: Path) -> tuple[int, int, list[str]]:
    total_bytes = 0
    total_files = 0
    warnings: list[str] = []
    try:
        walker = os.walk(path, topdown=True, followlinks=False)
        for current, directories, files in walker:
            current_path = Path(current)
            kept_directories: list[str] = []
            for name in directories:
                candidate = current_path / name
                if name in LOCAL_COPY_IGNORE_NAMES:
                    continue
                if candidate.is_symlink():
                    print(
                        "[MODAL_CUSTOM_NODES] 로컬 노드 심볼릭 링크 폴더 제외: "
                        f"node={path.name}, path={candidate}"
                    )
                    warnings.append(f"심볼릭 링크 폴더 제외: {candidate.relative_to(path)}")
                    continue
                kept_directories.append(name)
            directories[:] = kept_directories
            for name in files:
                candidate = current_path / name
                if candidate.suffix.casefold() in LOCAL_COPY_IGNORE_SUFFIXES:
                    continue
                if candidate.is_symlink():
                    print(
                        "[MODAL_CUSTOM_NODES] 로컬 노드 심볼릭 링크 파일 제외: "
                        f"node={path.name}, path={candidate}"
                    )
                    warnings.append(f"심볼릭 링크 파일 제외: {candidate.relative_to(path)}")
                    continue
                try:
                    size = candidate.stat().st_size
                except OSError as exc:
                    print(
                        "[MODAL_CUSTOM_NODES] 로컬 노드 파일 크기 조회 실패: "
                        f"path={candidate}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    warnings.append(
                        f"읽을 수 없는 파일 제외: {candidate.relative_to(path)}"
                    )
                    continue
                total_bytes += max(0, int(size))
                total_files += 1
    except OSError as exc:
        print(
            "[MODAL_CUSTOM_NODES] 로컬 노드 탐색 실패: "
            f"path={path}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    return total_bytes, total_files, warnings


def _local_build_node(path: Path, name: str) -> tuple[dict[str, Any], list[str]]:
    size_bytes, file_count, warnings = _local_tree_size(path)
    return (
        {
            "name": name,
            "source_type": "local",
            "source_path": str(path.resolve()),
            "size_bytes": size_bytes,
            "file_count": file_count,
        },
        warnings,
    )


def _ignored_relative_path(value: str) -> bool:
    path = Path(str(value or "").replace("\\", "/"))
    return any(part in LOCAL_COPY_IGNORE_NAMES for part in path.parts) or (
        path.suffix.casefold() in LOCAL_COPY_IGNORE_SUFFIXES
    )


def _git_has_local_changes(path: Path) -> bool:
    tracked = _git_output(path, "status", "--porcelain", "--untracked-files=no")
    if tracked:
        return True
    untracked = _git_output(path, "ls-files", "--others", "--exclude-standard")
    return any(
        not _ignored_relative_path(item)
        for item in untracked.splitlines()
        if item.strip()
    )


def inventory_custom_nodes(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    custom_root = root / "comfy" / "custom_nodes"
    manifest_names, manifest_normalized = _manifest_node_names(root)
    build_nodes: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped: list[dict[str, str]] = []
    local_total_bytes = 0

    if not custom_root.is_dir():
        message = f"로컬 custom_nodes 폴더가 없습니다: {custom_root}"
        print(f"[MODAL_CUSTOM_NODES] 인벤토리 조회 생략: {message}")
        warnings.append(message)
        return {
            "manifest_nodes": manifest_names,
            "build_nodes": [],
            "skipped": skipped,
            "warnings": warnings,
            "summary": {
                "manifest": len(manifest_names),
                "git": 0,
                "local": 0,
                "skipped": 0,
                "warning": len(warnings),
                "local_bytes": 0,
            },
        }

    for path in sorted(custom_root.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_dir() or path.is_symlink():
            if path.is_symlink():
                reason = "최상위 심볼릭 링크 custom node는 동기화하지 않습니다."
                print(
                    "[MODAL_CUSTOM_NODES] 심볼릭 링크 노드 제외: "
                    f"path={path}, reason={reason}"
                )
                skipped.append({"name": path.name, "reason": reason})
            continue
        name = path.name
        if name in LOCAL_COPY_IGNORE_NAMES or name.startswith("."):
            continue
        if not _is_safe_node_name(name):
            reason = "안전하지 않은 custom node 폴더 이름입니다."
            print(
                "[MODAL_CUSTOM_NODES] custom node 이름 제외: "
                f"name={name!r}, path={path}"
            )
            skipped.append({"name": name, "reason": reason})
            continue
        if name.casefold() in manifest_normalized:
            continue

        git_marker = path / ".git"
        if git_marker.exists():
            try:
                remote = _git_output(path, "remote", "get-url", "origin")
                commit = _git_output(path, "rev-parse", "HEAD")
                dirty = _git_has_local_changes(path)
                if _public_git_remote(remote) and not dirty:
                    build_nodes.append(
                        {
                            "name": name,
                            "source_type": "git",
                            "repository": remote,
                            "ref": commit,
                        }
                    )
                    continue
                fallback_reason = (
                    "로컬 변경 사항이 있어 현재 파일을 복사합니다."
                    if dirty
                    else "Modal에서 직접 접근할 수 없는 Git remote라 현재 파일을 복사합니다."
                )
                print(
                    "[MODAL_CUSTOM_NODES] Git 노드를 로컬 복사로 전환: "
                    f"name={name}, remote={remote!r}, reason={fallback_reason}"
                )
                warnings.append(f"{name}: {fallback_reason}")
            except Exception as exc:
                print(
                    "[MODAL_CUSTOM_NODES] Git 노드 분석 실패 후 로컬 복사 시도: "
                    f"name={name}, path={path}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                warnings.append(
                    f"{name}: Git 정보를 읽지 못해 현재 파일 복사를 시도합니다."
                )

        try:
            node, node_warnings = _local_build_node(path, name)
        except OSError as exc:
            reason = f"로컬 파일을 읽지 못했습니다: {type(exc).__name__}: {exc}"
            skipped.append({"name": name, "reason": reason})
            continue
        size_bytes = int(node["size_bytes"])
        for warning in node_warnings:
            warnings.append(f"{name}: {warning}")
        if size_bytes > LOCAL_NODE_MAX_BYTES:
            reason = (
                f"로컬 복사 크기 제한 초과: {size_bytes / 1024 ** 2:.1f} MiB > "
                f"{LOCAL_NODE_MAX_BYTES / 1024 ** 2:.0f} MiB"
            )
            print(
                "[MODAL_CUSTOM_NODES] 대용량 로컬 노드 제외: "
                f"name={name}, size_bytes={size_bytes}, limit={LOCAL_NODE_MAX_BYTES}"
            )
            skipped.append({"name": name, "reason": reason})
            continue
        if local_total_bytes + size_bytes > LOCAL_NODES_TOTAL_MAX_BYTES:
            reason = (
                "로컬 custom node 전체 복사 크기 제한을 초과합니다: "
                f"{(local_total_bytes + size_bytes) / 1024 ** 2:.1f} MiB > "
                f"{LOCAL_NODES_TOTAL_MAX_BYTES / 1024 ** 2:.0f} MiB"
            )
            print(
                "[MODAL_CUSTOM_NODES] 로컬 노드 전체 크기 제한으로 제외: "
                f"name={name}, accumulated_bytes={local_total_bytes}, "
                f"node_bytes={size_bytes}, limit={LOCAL_NODES_TOTAL_MAX_BYTES}"
            )
            skipped.append({"name": name, "reason": reason})
            continue
        local_total_bytes += size_bytes
        build_nodes.append(node)

    git_count = sum(1 for item in build_nodes if item["source_type"] == "git")
    local_count = sum(1 for item in build_nodes if item["source_type"] == "local")
    return {
        "manifest_nodes": manifest_names,
        "build_nodes": build_nodes,
        "skipped": skipped,
        "warnings": warnings,
        "summary": {
            "manifest": len(manifest_names),
            "git": git_count,
            "local": local_count,
            "skipped": len(skipped),
            "warning": len(warnings),
            "local_bytes": local_total_bytes,
        },
    }


def public_custom_node_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for raw in inventory.get("build_nodes", []):
        if not isinstance(raw, Mapping):
            continue
        item = {
            "name": str(raw.get("name") or ""),
            "source_type": str(raw.get("source_type") or ""),
        }
        if item["source_type"] == "git":
            item["repository"] = str(raw.get("repository") or "")
            item["ref"] = str(raw.get("ref") or "")
        else:
            item["size_bytes"] = max(0, int(raw.get("size_bytes") or 0))
            item["file_count"] = max(0, int(raw.get("file_count") or 0))
        nodes.append(item)
    return {
        "manifest_nodes": [str(name) for name in inventory.get("manifest_nodes", [])],
        "discovered_nodes": nodes,
        "skipped": [
            {
                "name": str(item.get("name") or ""),
                "reason": str(item.get("reason") or ""),
            }
            for item in inventory.get("skipped", [])
            if isinstance(item, Mapping)
        ],
        "warnings": [str(item) for item in inventory.get("warnings", [])],
        "summary": dict(inventory.get("summary") or {}),
    }


def deploy_custom_nodes_json(inventory: Mapping[str, Any]) -> str:
    result: list[dict[str, Any]] = []
    for raw in inventory.get("build_nodes", []):
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "")
        source_type = str(raw.get("source_type") or "")
        if not _is_safe_node_name(name) or source_type not in {"git", "local"}:
            print(
                "[MODAL_CUSTOM_NODES] 배포 인벤토리 항목 제외: "
                f"name={name!r}, source_type={source_type!r}"
            )
            continue
        if source_type == "git":
            result.append(
                {
                    "name": name,
                    "source_type": "git",
                    "repository": str(raw.get("repository") or ""),
                    "ref": str(raw.get("ref") or ""),
                }
            )
        else:
            result.append(
                {
                    "name": name,
                    "source_type": "local",
                    "source_path": str(raw.get("source_path") or ""),
                }
            )
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
