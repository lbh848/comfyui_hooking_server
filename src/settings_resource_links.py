"""Explicit, removable model-folder links to another local ComfyUI installation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from app_paths import config_root
from json_store import write_json_atomic


class ResourceLinks:
    def __init__(self, project_root: Path, model_root: Path):
        self.root = model_root.absolute()
        self.state_path = config_root(project_root) / "resource_links.json"

    def read(self) -> list[dict]:
        from settings_resources import safe_relative
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if not isinstance(value, dict) or not isinstance(value.get("links"), list):
            raise ValueError("모델 연결 정보를 읽지 못했습니다. resource_links.json을 확인하세요.")
        links = value["links"]
        for link in links:
            if not isinstance(link, dict) or not all(isinstance(link.get(key), str) for key in ("id", "path", "source")):
                raise ValueError("모델 연결 정보를 확인하세요.")
            relative = Path(safe_relative(link["path"]))
            if len(relative.parts) != 3 or relative.parts[1] != "LINKED" or not Path(link["source"]).is_absolute():
                raise ValueError("모델 연결 경로를 확인하세요.")
        return links

    def match(self, path: Path, links: list[dict]) -> dict | None:
        for link in links:
            mount = self.root / link["path"]
            if not path.absolute().is_relative_to(mount):
                continue
            source = Path(link["source"])
            if ((mount.is_symlink() or mount.is_junction()) and mount.resolve() == source
                    and path.resolve().is_relative_to(source)):
                return link
        return None

    @staticmethod
    def files(source: Path):
        from settings_resources import MODEL_EXTENSIONS
        visited = set()

        def fail(error):
            raise error

        for directory, children, filenames in os.walk(source, followlinks=True, onerror=fail):
            current = Path(directory)
            resolved = current.resolve()
            if resolved in visited or not resolved.is_relative_to(source):
                children[:] = []
                continue
            visited.add(resolved)
            children[:] = sorted(name for name in children if not name.startswith("."))
            for name in sorted(filenames):
                path = current / name
                if path.suffix.lower() in MODEL_EXTENSIONS and path.resolve().is_relative_to(source) and path.is_file():
                    stat = path.stat()
                    if stat.st_size:
                        yield path, stat.st_size

    def status(self, links: list[dict] | None = None) -> list[dict]:
        result = []
        for link in self.read() if links is None else links:
            mount = self.root / link["path"]
            available = self.match(mount, [link]) is not None and mount.is_dir()
            result.append({**link, "folder": Path(link["path"]).parts[0], "available": available})
        return result

    def inspect(self, source: str, kind: str) -> dict:
        from settings_resources import MODEL_KINDS, safe_relative
        if kind not in ("models", "loras") or not isinstance(source, str) or not source.strip():
            raise ValueError("외부 ComfyUI 폴더 또는 models 폴더의 전체 경로를 입력하세요.")
        path = Path(source.strip().strip('"'))
        if not path.is_absolute():
            raise ValueError("외부 ComfyUI 폴더의 전체 경로를 입력하세요.")
        candidates = [path / "models", path / "ComfyUI" / "models"]
        if path.name.casefold() == "models":
            candidates.insert(0, path)
        model_root = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
        if model_root is None:
            raise ValueError("models 폴더를 찾지 못했습니다. 외부 ComfyUI 폴더 또는 models 폴더를 지정하세요.")
        current_root = self.root.resolve()
        if model_root.is_relative_to(current_root) or current_root.is_relative_to(model_root):
            raise ValueError("이 프로그램 밖에 설치된 ComfyUI의 models 폴더를 선택하세요.")
        registered = {Path(link["source"]) for link in self.read()}
        folders = []
        for child in sorted(model_root.iterdir(), key=lambda item: item.name.casefold()):
            if not child.is_dir() or child.name.startswith(".") or (child.name == "loras") != (kind == "loras"):
                continue
            safe_relative(child.name, filename=True)
            source_folder = child.resolve()
            if source_folder.is_relative_to(current_root) or current_root.is_relative_to(source_folder):
                continue
            count, size = 0, 0
            for _, file_size in self.files(source_folder):
                count += 1
                size += file_size
            if count:
                folders.append({"folder": child.name, "label": "LoRA" if kind == "loras" else MODEL_KINDS.get(child.name, "Other"),
                                "source": str(source_folder), "count": count, "size": str(size),
                                "connected": source_folder in registered})
        return {"source": str(model_root), "folders": folders}

    def connect(self, source: str, kind: str, folders: list[str]) -> dict:
        if not isinstance(folders, list) or not folders or not all(isinstance(folder, str) for folder in folders):
            raise ValueError("연결할 모델 폴더를 선택하세요.")
        inspection = self.inspect(source, kind)
        selected = [item for item in inspection["folders"] if item["folder"] in folders]
        if {item["folder"] for item in selected} != set(folders):
            raise ValueError("연결할 폴더가 변경되었습니다. 폴더를 다시 확인하세요.")
        links = self.read()
        created = []
        try:
            for item in selected:
                if item["connected"]:
                    continue
                source_folder = Path(item["source"])
                identity = hashlib.sha256(str(source_folder).encode("utf-8")).hexdigest()[:16]
                # A dedicated namespace prevents same-name local files being replaced.
                relative = Path(item["folder"]) / "LINKED" / identity
                mount = self.root / relative
                if not mount.parent.resolve().is_relative_to(self.root):
                    raise ValueError("외부로 연결된 폴더 안에는 새 연결을 만들 수 없습니다.")
                if mount.exists() or mount.is_symlink() or mount.is_junction():
                    raise FileExistsError("같은 연결 위치가 이미 있습니다. 기존 연결을 확인하세요.")
                mount.parent.mkdir(parents=True, exist_ok=True)
                if os.name == "nt":
                    import _winapi
                    _winapi.CreateJunction(str(source_folder), str(mount))
                else:
                    mount.symlink_to(source_folder, target_is_directory=True)
                created.append(mount)
                links.append({"id": identity, "path": relative.as_posix(), "source": str(source_folder)})
            if created:
                write_json_atomic(self.state_path, {"links": links})
        except BaseException:
            for mount in reversed(created):
                if mount.is_symlink():
                    mount.unlink()
                else:
                    mount.rmdir()
            raise
        return {"connected": len(created), "links": self.status(links)}

    def disconnect(self, identity: str) -> dict:
        links = self.read()
        link = next((item for item in links if item["id"] == identity), None)
        if link is None:
            raise FileNotFoundError("연결이 없습니다. 목록을 새로고침하세요.")
        mount = self.root / link["path"]
        if not mount.parent.resolve().is_relative_to(self.root):
            raise ValueError("연결 위치를 확인하세요. 외부 폴더는 변경하지 않았습니다.")
        # Never enumerate or recursively delete a junction's contents.
        if mount.is_symlink():
            mount.unlink()
        elif mount.is_junction():
            mount.rmdir()
        elif mount.exists():
            raise ValueError("연결 위치에 일반 폴더가 있어 해제하지 않았습니다.")
        write_json_atomic(self.state_path, {"links": [item for item in links if item["id"] != identity]})
        return {"disconnected": identity}
