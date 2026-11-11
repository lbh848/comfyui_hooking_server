"""Modal Image 빌드 중 고정 ComfyUI custom node를 설치한다."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request
import zipfile
from pathlib import Path


COMFY_ROOT = Path("/root/ComfyUI")
MANIFEST_PATH = Path("/opt/soya/install_manifest.json")


def _run(args: list[str], cwd: Path | None = None) -> None:
    print(f"[MODAL_IMAGE] 실행: {' '.join(args[:3])}")
    subprocess.run(args, cwd=cwd, check=True)


def _download(node: dict, target: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(
        str(node["url"]),
        headers={"User-Agent": "soya-comfy-modal-image/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    expected_size = int(node.get("size") or 0)
    expected_sha = str(node.get("sha256") or "").lower()
    if expected_size and size != expected_size:
        raise RuntimeError(
            f"{node['name']} 아카이브 용량 불일치: expected={expected_size}, actual={size}"
        )
    if expected_sha and digest.hexdigest() != expected_sha:
        raise RuntimeError(
            f"{node['name']} 아카이브 SHA-256 불일치: "
            f"expected={expected_sha}, actual={digest.hexdigest()}"
        )


def _install_archive(node: dict, destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="soya-modal-node-") as temp_dir:
        archive_path = Path(temp_dir) / "node.zip"
        _download(node, archive_path)
        destination.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive_path, "r") as archive:
            root = destination.resolve()
            for info in archive.infolist():
                target = (destination / info.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(
                        f"{node['name']} 아카이브에 안전하지 않은 경로가 있습니다: {info.filename}"
                    )
            archive.extractall(destination)


def _install_git(node: dict, destination: Path) -> None:
    target = str(node.get("tracking_branch") or node.get("ref") or "main")
    _run(["git", "init", str(destination)])
    _run(["git", "remote", "add", "origin", str(node["repository"])], destination)
    _run(["git", "fetch", "--depth", "1", "origin", target], destination)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], destination)


def _install_local(node: dict, destination: Path) -> None:
    bundled_path = Path(str(node.get("bundled_path") or ""))
    if not bundled_path.is_dir():
        raise FileNotFoundError(
            f"{node['name']}의 패키징된 로컬 소스가 없습니다: {bundled_path}"
        )
    shutil.copytree(bundled_path, destination)


def _extra_nodes() -> list[dict]:
    raw = os.environ.get("SOYA_MODAL_IMAGE_CUSTOM_NODES", "[]")
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise TypeError("추가 custom node 인벤토리는 배열이어야 합니다.")
    result: list[dict] = []
    for node in parsed:
        if not isinstance(node, dict):
            raise TypeError("추가 custom node 항목은 객체여야 합니다.")
        name = str(node.get("name") or "")
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"안전하지 않은 custom node 이름입니다: {name!r}")
        result.append(node)
    return result


def install() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    custom_root = COMFY_ROOT / "custom_nodes"
    custom_root.mkdir(parents=True, exist_ok=True)
    nodes = list(manifest.get("custom_nodes", []))
    known_names = {str(node.get("name") or "").casefold() for node in nodes}
    for node in _extra_nodes():
        normalized = str(node.get("name") or "").casefold()
        if normalized in known_names:
            print(f"[MODAL_IMAGE] manifest 중복 추가 노드 제외: {node.get('name')}")
            continue
        known_names.add(normalized)
        nodes.append(node)
    for node in nodes:
        destination = custom_root / str(node["name"])
        if destination.exists():
            print(f"[MODAL_IMAGE] 기존 노드 재사용: {node['name']}")
            continue
        print(f"[MODAL_IMAGE] 노드 설치: {node['name']}")
        if node.get("source_type") == "archive":
            _install_archive(node, destination)
        elif node.get("source_type") == "git":
            _install_git(node, destination)
        elif node.get("source_type") == "local":
            _install_local(node, destination)
        else:
            raise RuntimeError(
                f"지원하지 않는 custom node 소스입니다: {node.get('source_type')!r}"
            )

    requirement_files = sorted(custom_root.glob("*/requirements.txt"))
    for requirements in requirement_files:
        print(f"[MODAL_IMAGE] 의존성 설치: {requirements.parent.name}")
        _run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)]
        )


if __name__ == "__main__":
    try:
        install()
    except Exception as exc:
        print(f"[MODAL_IMAGE] custom node 설치 실패: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
