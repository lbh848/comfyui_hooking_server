from __future__ import annotations

import json
import os
import re
import shutil
import stat
import traceback
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from threading import Event
from typing import Callable

from .downloader import ResumableDownloader
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


def install_git_node(
    *,
    node: dict,
    custom_root: Path,
    cancel_event: Event,
    log: LogCallback | None,
) -> Path:
    name = str(node["name"])
    ref = str(node["ref"]).lower()
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
        if head == ref and origin.rstrip("/").removesuffix(".git").casefold() == (
            repository.rstrip("/").removesuffix(".git").casefold()
        ):
            if log:
                log(f"[노드] 기존 Git 설치 재사용: {name} {ref[:12]}")
            return destination
        raise NodeInstallError(
            "기존 Git 커스텀 노드가 고정점과 달라 덮어쓰지 않습니다: "
            f"name={name}, expected={ref}, actual={head}, origin={origin}"
        )

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
        run_command(
            ["git", "fetch", "--depth", "1", "origin", ref],
            cwd=staging,
            cancel_event=cancel_event,
            log=log,
            timeout=600,
        )
        run_command(
            ["git", "checkout", "--detach", "FETCH_HEAD"],
            cwd=staging,
            cancel_event=cancel_event,
            log=log,
        )
        actual = _git_head(staging).lower()
        if actual != ref:
            raise NodeInstallError(
                f"Git 노드 고정점 검증 실패: name={name}, "
                f"expected={ref}, actual={actual}"
            )
        os.replace(staging, destination)
        if log:
            log(f"[노드] Git 설치 완료: {name} {ref[:12]}")
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
) -> list[Path]:
    custom_root = comfy_root / "custom_nodes"
    cache_root = comfy_root / ".installer-cache" / "custom_nodes"
    installed: list[Path] = []
    for index, node in enumerate(nodes, 1):
        if cancel_event.is_set():
            raise NodeInstallError("커스텀 노드 설치 중 중단 요청을 받았습니다.")
        if log:
            log(f"[노드 {index}/{len(nodes)}] {node['name']} 설치")
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
