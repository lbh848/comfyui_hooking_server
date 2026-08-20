from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import stat
import traceback
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from threading import Event
from typing import Callable, Iterator

from .downloader import ResumableDownloader
from .node_compatibility import (
    INSTANT_LORA_NODE_NAME,
    MINIMAX_H3_TEACACHE_NODE_NAME,
    apply_instant_lora_python_compatibility,
    apply_minimax_h3_teacache_reset_compatibility,
    remove_instant_lora_python_compatibility,
    remove_minimax_h3_teacache_reset_compatibility,
)
from .operations import CommandError, run_command


class NodeInstallError(RuntimeError):
    """커스텀 노드 소스 설치 실패."""


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict], None]
_SAFE_NODE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _assert_direct_child(path: Path, parent: Path, label: str) -> None:
    resolved = path.resolve()
    root = parent.resolve()
    if resolved.parent != root:
        raise NodeInstallError(
            f"{label} 경로가 커스텀 노드 루트의 직접 자식이 아닙니다: {resolved}"
        )


def _validate_node_name(name: str) -> None:
    if not _SAFE_NODE_NAME.fullmatch(name):
        raise NodeInstallError(f"커스텀 노드 폴더명이 안전하지 않습니다: {name!r}")


def _prepare_staging(custom_root: Path, node_name: str) -> Path:
    staging = custom_root / f".installing_{node_name}_{uuid.uuid4().hex[:8]}"
    _assert_direct_child(staging, custom_root, "스테이징")
    return staging


@contextmanager
def _managed_node_compatibility_update(
    *,
    node_name: str,
    comfy_root: Path,
    requirements_dir: Path | None,
    log: LogCallback | None,
) -> Iterator[None]:
    if node_name == INSTANT_LORA_NODE_NAME:
        compatibility_label = "Instant LoRA managed Python"
        compatibility_path = (
            comfy_root
            / "custom_nodes"
            / INSTANT_LORA_NODE_NAME
            / "src"
            / "runtime.py"
        )
        remove_compatibility = remove_instant_lora_python_compatibility
        apply_compatibility = apply_instant_lora_python_compatibility
    elif node_name == MINIMAX_H3_TEACACHE_NODE_NAME:
        compatibility_label = "MiniMax H3 TeaCache sample reset"
        compatibility_path = (
            comfy_root
            / "custom_nodes"
            / MINIMAX_H3_TEACACHE_NODE_NAME
            / "nodes.py"
        )
        remove_compatibility = remove_minimax_h3_teacache_reset_compatibility
        apply_compatibility = apply_minimax_h3_teacache_reset_compatibility
    else:
        yield
        return

    backup_root = (
        requirements_dir.resolve()
        if requirements_dir is not None
        else (
            comfy_root
            / ".installer-state"
            / "backups"
            / "node-compatibility"
        ).resolve()
    )
    remove_compatibility(
        comfy_root=comfy_root,
        requirements_dir=backup_root,
        log=log,
        allow_missing=True,
    )
    try:
        yield
    except Exception as operation_exc:
        if compatibility_path.is_file():
            try:
                apply_compatibility(
                    comfy_root=comfy_root,
                    requirements_dir=backup_root,
                    log=log,
                )
            except Exception as restore_exc:
                print(
                    "[COMFY_INSTALL][NODE] custom-node operation and "
                    "compatibility restore both failed: "
                    f"node={node_name}, operation_error={operation_exc}, "
                    f"restore_error={restore_exc}"
                )
                traceback.print_exc()
                raise NodeInstallError(
                    f"{compatibility_label} restore failed after node operation: "
                    f"operation={operation_exc}, "
                    f"restore={restore_exc}"
                ) from operation_exc
        else:
            print(
                "[COMFY_INSTALL][NODE] compatibility target missing after node "
                f"operation failure: node={node_name}, path={compatibility_path}"
            )
        raise
    try:
        apply_compatibility(
            comfy_root=comfy_root,
            requirements_dir=backup_root,
            log=log,
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][NODE] compatibility apply failed: "
            f"node={node_name}, compatibility={compatibility_label}, error={exc}"
        )
        traceback.print_exc()
        raise NodeInstallError(
            f"{compatibility_label} compatibility apply failed: {exc}"
        ) from exc


def _safe_zip_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise NodeInstallError(f"노드 압축 파일에 안전하지 않은 경로가 있습니다: {name!r}")
    return path


def _extract_zip_safely(archive_path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                member = _safe_zip_member(info.filename)
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(unix_mode):
                    raise NodeInstallError(
                        f"노드 압축 파일의 심볼릭 링크를 허용하지 않습니다: {info.filename}"
                    )
                target = destination.joinpath(*member.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except NodeInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][NODE] 노드 압축 해제 실패: "
            f"archive={archive_path}, destination={destination}, error={exc}"
        )
        traceback.print_exc()
        raise NodeInstallError(f"노드 압축 해제 실패: {archive_path.name}") from exc


def _archive_marker_matches(destination: Path, node: dict) -> bool:
    marker = destination / ".comfy-installer-source.json"
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            "[COMFY_INSTALL][NODE] 기존 노드 설치 표식 읽기 실패: "
            f"path={marker}, error={exc}"
        )
        traceback.print_exc()
        return False
    return (
        data.get("source_type") == "archive"
        and data.get("sha256") == node.get("sha256")
        and data.get("url") == node.get("url")
    )


def install_archive_node(
    *,
    node: dict,
    custom_root: Path,
    cache_root: Path,
    downloader: ResumableDownloader,
    cancel_event: Event,
    log: LogCallback | None,
    progress: ProgressCallback | None,
) -> Path:
    name = str(node["name"])
    _validate_node_name(name)
    custom_root.mkdir(parents=True, exist_ok=True)
    destination = custom_root / name
    _assert_direct_child(destination, custom_root, "설치 대상")
    if destination.exists():
        if destination.is_dir() and _archive_marker_matches(destination, node):
            if log:
                log(f"[노드] 기존 설치 재사용: {name} {node.get('version', '')}")
            return destination
        raise NodeInstallError(
            f"관리되지 않는 기존 커스텀 노드 폴더가 있어 덮어쓰지 않습니다: {destination}"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    archive_name = f"{name}-{node.get('version', 'pinned')}.archive"
    archive_path = cache_root / archive_name
    downloader.download(
        url=node["url"],
        target=archive_path,
        expected_size=int(node["size"]),
        expected_sha256=node["sha256"],
        cancel_event=cancel_event,
        progress=progress,
    )
    staging = _prepare_staging(custom_root, name)
    try:
        staging.mkdir(parents=False)
        _extract_zip_safely(archive_path, staging)
        marker = {
            "source_type": "archive",
            "url": node["url"],
            "version": node.get("version"),
            "sha256": node["sha256"],
        }
        (staging / ".comfy-installer-source.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        if log:
            log(f"[노드] 설치 완료: {name} {node.get('version', '')}")
        return destination
    except Exception:
        print(
            "[COMFY_INSTALL][NODE] 압축 노드 설치 실패, 스테이징 보존: "
            f"{staging}"
        )
        raise


def _git_origin(path: Path) -> str:
    lines = run_command(["git", "remote", "get-url", "origin"], cwd=path)
    return lines[-1].strip() if lines else ""


def _git_head(path: Path) -> str:
    lines = run_command(["git", "rev-parse", "HEAD"], cwd=path)
    return lines[-1].strip() if lines else ""


def _git_source_target(node: dict) -> tuple[str, bool]:
    tracking_branch = node.get("tracking_branch")
    if isinstance(tracking_branch, str) and tracking_branch.strip():
        return tracking_branch.strip(), True
    return str(node["ref"]).lower(), False


def _fetch_git_target(
    *,
    path: Path,
    target: str,
    cancel_event: Event,
    log: LogCallback | None,
) -> str:
    run_command(
        ["git", "fetch", "--depth", "1", "origin", target],
        cwd=path,
        cancel_event=cancel_event,
        log=log,
        timeout=600,
    )
    lines = run_command(["git", "rev-parse", "FETCH_HEAD"], cwd=path)
    fetched = lines[-1].strip().lower() if lines else ""
    if not re.fullmatch(r"[0-9a-f]{40}", fetched):
        raise NodeInstallError(
            f"Git fetch 결과 커밋을 확인하지 못했습니다: target={target!r}, "
            f"actual={fetched!r}"
        )
    return fetched


def _verify_pinned_fetch(*, target: str, fetched: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-fA-F]{40}", target) and fetched != target.lower():
        raise NodeInstallError(
            f"Git 노드 고정점 fetch 검증 실패: name={name}, "
            f"expected={target.lower()}, actual={fetched}"
        )


def install_git_node(
    *,
    node: dict,
    custom_root: Path,
    cancel_event: Event,
    log: LogCallback | None,
) -> Path:
    name = str(node["name"])
    target, tracks_branch = _git_source_target(node)
    repository = str(node["repository"])
    _validate_node_name(name)
    custom_root.mkdir(parents=True, exist_ok=True)
    destination = custom_root / name
    _assert_direct_child(destination, custom_root, "설치 대상")
    if destination.exists():
        if not (destination / ".git").exists():
            raise NodeInstallError(
                f"관리되지 않는 기존 커스텀 노드 폴더가 있어 덮어쓰지 않습니다: "
                f"{destination}"
            )
        try:
            head = _git_head(destination).lower()
            origin = _git_origin(destination)
        except CommandError as exc:
            raise NodeInstallError(
                f"기존 커스텀 노드 Git 상태를 확인하지 못했습니다: {destination}"
            ) from exc
        origin_matches = origin.rstrip("/").removesuffix(".git").casefold() == (
            repository.rstrip("/").removesuffix(".git").casefold()
        )
        if not origin_matches:
            raise NodeInstallError(
                "기존 Git 커스텀 노드의 원격 저장소가 다릅니다: "
                f"name={name}, expected={repository}, actual={origin}"
            )
        if not tracks_branch:
            if head == target.lower():
                if log:
                    log(f"[노드] 기존 Git 설치 재사용: {name} {target[:12]}")
                return destination
            raise NodeInstallError(
                "기존 Git 커스텀 노드가 고정점과 달라 덮어쓰지 않습니다: "
                f"name={name}, expected={target}, actual={head}, origin={origin}"
            )

        status = run_command(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=destination,
        )
        if status:
            raise NodeInstallError(
                f"노드에 로컬 변경이 있어 업데이트하지 않습니다: {name}: "
                + ", ".join(status[:10])
            )
        fetched = _fetch_git_target(
            path=destination,
            target=target,
            cancel_event=cancel_event,
            log=log,
        )
        if head != fetched:
            run_command(
                ["git", "checkout", "--detach", "FETCH_HEAD"],
                cwd=destination,
                cancel_event=cancel_event,
                log=log,
            )
            actual = _git_head(destination).lower()
            if actual != fetched:
                raise NodeInstallError(
                    f"Git 노드 브랜치 검증 실패: name={name}, "
                    f"branch={target}, expected={fetched}, actual={actual}"
                )
        if log:
            log(f"[노드] 기존 Git 설치 갱신: {name} origin/{target} {fetched[:12]}")
        return destination

    staging = _prepare_staging(custom_root, name)
    try:
        staging.mkdir(parents=False)
        run_command(
            ["git", "init"],
            cwd=staging,
            cancel_event=cancel_event,
            log=log,
        )
        run_command(
            ["git", "remote", "add", "origin", repository],
            cwd=staging,
            cancel_event=cancel_event,
            log=log,
        )
        fetched = _fetch_git_target(
            path=staging,
            target=target,
            cancel_event=cancel_event,
            log=log,
        )
        if not tracks_branch:
            _verify_pinned_fetch(target=target, fetched=fetched, name=name)
        run_command(
            ["git", "checkout", "--detach", "FETCH_HEAD"],
            cwd=staging,
            cancel_event=cancel_event,
            log=log,
        )
        actual = _git_head(staging).lower()
        if actual != fetched:
            raise NodeInstallError(
                f"Git 노드 설치 커밋 검증 실패: name={name}, "
                f"expected={fetched}, actual={actual}"
            )
        os.replace(staging, destination)
        if log:
            source_label = f"origin/{target}" if tracks_branch else target[:12]
            log(f"[노드] Git 설치 완료: {name} {source_label} {actual[:12]}")
        return destination
    except Exception:
        print(
            "[COMFY_INSTALL][NODE] Git 노드 설치 실패, 스테이징 보존: "
            f"{staging}"
        )
        raise


def install_custom_nodes(
    *,
    nodes: list[dict],
    comfy_root: Path,
    downloader: ResumableDownloader,
    cancel_event: Event,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
    requirements_dir: Path | None = None,
) -> list[Path]:
    custom_root = comfy_root / "custom_nodes"
    cache_root = comfy_root / ".installer-cache" / "custom_nodes"
    installed: list[Path] = []
    for index, node in enumerate(nodes, 1):
        if cancel_event.is_set():
            raise NodeInstallError("커스텀 노드 설치 중 중단 요청을 받았습니다.")
        if log:
            log(f"[노드 {index}/{len(nodes)}] {node['name']} 설치")
        with _managed_node_compatibility_update(
            node_name=str(node["name"]),
            comfy_root=comfy_root,
            requirements_dir=requirements_dir,
            log=log,
        ):
            if node["source_type"] == "archive":
                path = install_archive_node(
                    node=node,
                    custom_root=custom_root,
                    cache_root=cache_root,
                    downloader=downloader,
                    cancel_event=cancel_event,
                    log=log,
                    progress=progress,
                )
            elif node["source_type"] == "git":
                path = install_git_node(
                    node=node,
                    custom_root=custom_root,
                    cancel_event=cancel_event,
                    log=log,
                )
            else:
                raise NodeInstallError(
                    f"지원하지 않는 노드 소스 형식: {node.get('source_type')!r}"
                )
        installed.append(path)
    return installed


def update_archive_node(
    *,
    node: dict,
    comfy_root: Path,
    downloader: ResumableDownloader,
    cancel_event: Event,
    log: LogCallback | None,
    progress: ProgressCallback | None,
    changed_nodes: list[str] | None = None,
) -> Path:
    name = str(node["name"])
    _validate_node_name(name)
    custom_root = comfy_root / "custom_nodes"
    cache_root = comfy_root / ".installer-cache" / "custom_nodes"
    destination = custom_root / name
    _assert_direct_child(destination, custom_root, "업데이트 대상")
    if not destination.exists():
        installed = install_archive_node(
            node=node,
            custom_root=custom_root,
            cache_root=cache_root,
            downloader=downloader,
            cancel_event=cancel_event,
            log=log,
            progress=progress,
        )
        if changed_nodes is not None:
            changed_nodes.append(name)
        return installed
    if not destination.is_dir():
        raise NodeInstallError(
            f"아카이브 노드 업데이트 대상이 폴더가 아닙니다: {destination}"
        )
    marker = destination / ".comfy-installer-source.json"
    if not marker.is_file():
        raise NodeInstallError(
            f"관리되지 않는 기존 노드는 업데이트하지 않습니다: {destination}"
        )
    if _archive_marker_matches(destination, node):
        if log:
            log(f"[노드 업데이트] 이미 최신: {name}")
        return destination

    cache_root.mkdir(parents=True, exist_ok=True)
    archive_name = f"{name}-{node.get('version', 'pinned')}.archive"
    archive_path = cache_root / archive_name
    downloader.download(
        url=node["url"],
        target=archive_path,
        expected_size=int(node["size"]),
        expected_sha256=node["sha256"],
        cancel_event=cancel_event,
        progress=progress,
    )
    staging = _prepare_staging(custom_root, name)
    backup_root = comfy_root / ".installer-state" / "backups" / "custom_nodes"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = backup_root / f"{name}-{stamp}"
    try:
        staging.mkdir(parents=False)
        _extract_zip_safely(archive_path, staging)
        marker_value = {
            "source_type": "archive",
            "url": node["url"],
            "version": node.get("version"),
            "sha256": node["sha256"],
        }
        (staging / ".comfy-installer-source.json").write_text(
            json.dumps(marker_value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        backup_root.mkdir(parents=True, exist_ok=True)
        os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            print(
                "[COMFY_INSTALL][NODE] 새 아카이브 노드 배치 실패, "
                f"기존 폴더 복원: {backup} -> {destination}"
            )
            traceback.print_exc()
            os.replace(backup, destination)
            raise
        if log:
            log(f"[노드 업데이트] 완료: {name} (기존 백업: {backup})")
        if changed_nodes is not None:
            changed_nodes.append(name)
        return destination
    except Exception as exc:
        print(
            "[COMFY_INSTALL][NODE] 아카이브 노드 업데이트 실패: "
            f"name={name}, error={exc}, staging={staging}"
        )
        traceback.print_exc()
        if isinstance(exc, NodeInstallError):
            raise
        raise NodeInstallError(f"아카이브 노드 업데이트 실패: {name}") from exc


def update_git_node(
    *,
    node: dict,
    comfy_root: Path,
    cancel_event: Event,
    log: LogCallback | None,
    changed_nodes: list[str] | None = None,
) -> Path:
    name = str(node["name"])
    target, tracks_branch = _git_source_target(node)
    repository = str(node["repository"])
    _validate_node_name(name)
    custom_root = comfy_root / "custom_nodes"
    destination = custom_root / name
    _assert_direct_child(destination, custom_root, "업데이트 대상")
    if not destination.exists():
        installed = install_git_node(
            node=node,
            custom_root=custom_root,
            cancel_event=cancel_event,
            log=log,
        )
        if changed_nodes is not None:
            changed_nodes.append(name)
        return installed
    if not destination.is_dir() or not (destination / ".git").is_dir():
        raise NodeInstallError(
            f"관리되지 않는 기존 Git 노드는 업데이트하지 않습니다: {destination}"
        )
    try:
        origin = _git_origin(destination)
        normalized_origin = origin.rstrip("/").removesuffix(".git").casefold()
        normalized_expected = repository.rstrip("/").removesuffix(".git").casefold()
        if normalized_origin != normalized_expected:
            raise NodeInstallError(
                f"노드 원격 저장소 불일치: {name}, actual={origin}"
            )
        status = run_command(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=destination,
        )
        if status:
            raise NodeInstallError(
                f"노드에 로컬 변경이 있어 업데이트하지 않습니다: {name}: "
                + ", ".join(status[:10])
            )
        head = _git_head(destination).lower()
        if not tracks_branch and head == target.lower():
            if log:
                log(f"[노드 업데이트] 이미 최신: {name} {head[:12]}")
            return destination
        fetched = _fetch_git_target(
            path=destination,
            target=target,
            cancel_event=cancel_event,
            log=log,
        )
        if not tracks_branch:
            _verify_pinned_fetch(target=target, fetched=fetched, name=name)
        if head == fetched:
            if log:
                source_label = f"origin/{target}" if tracks_branch else target[:12]
                log(f"[노드 업데이트] 이미 최신: {name} {source_label} {head[:12]}")
            return destination
        run_command(
            ["git", "checkout", "--detach", "FETCH_HEAD"],
            cwd=destination,
            cancel_event=cancel_event,
            log=log,
        )
        actual = _git_head(destination).lower()
        if actual != fetched:
            raise NodeInstallError(
                f"노드 업데이트 커밋 검증 실패: {name}, "
                f"expected={fetched}, actual={actual}"
            )
        if log:
            source_label = f"origin/{target}" if tracks_branch else target[:12]
            log(f"[노드 업데이트] 완료: {name} {source_label} {actual[:12]}")
        if changed_nodes is not None:
            changed_nodes.append(name)
        return destination
    except NodeInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][NODE] Git 노드 업데이트 실패: "
            f"name={name}, path={destination}, error={exc}"
        )
        traceback.print_exc()
        raise NodeInstallError(f"Git 노드 업데이트 실패: {name}: {exc}") from exc


def update_custom_nodes(
    *,
    nodes: list[dict],
    comfy_root: Path,
    downloader: ResumableDownloader,
    cancel_event: Event,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
    changed_nodes: list[str] | None = None,
    requirements_dir: Path | None = None,
) -> list[Path]:
    updated: list[Path] = []
    for index, node in enumerate(nodes, 1):
        if cancel_event.is_set():
            raise NodeInstallError("커스텀 노드 업데이트 중 중단 요청을 받았습니다.")
        if log:
            log(f"[노드 업데이트 {index}/{len(nodes)}] {node['name']}")
        with _managed_node_compatibility_update(
            node_name=str(node["name"]),
            comfy_root=comfy_root,
            requirements_dir=requirements_dir,
            log=log,
        ):
            if node["source_type"] == "archive":
                path = update_archive_node(
                    node=node,
                    comfy_root=comfy_root,
                    downloader=downloader,
                    cancel_event=cancel_event,
                    log=log,
                    progress=progress,
                    changed_nodes=changed_nodes,
                )
            elif node["source_type"] == "git":
                path = update_git_node(
                    node=node,
                    comfy_root=comfy_root,
                    cancel_event=cancel_event,
                    log=log,
                    changed_nodes=changed_nodes,
                )
            else:
                raise NodeInstallError(
                    f"지원하지 않는 노드 소스 형식: {node.get('source_type')!r}"
                )
        updated.append(path)
    return updated
