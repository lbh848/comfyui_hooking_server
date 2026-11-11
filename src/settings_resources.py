"""Read the embedded ComfyUI's local model files without importing ComfyUI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from threading import Event
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from app_paths import SOURCE_DIR, data_root
from comfy_installer.manifest import ManifestError, load_install_manifest
from lora_paths import MANAGED_LORA_ROOT
from comfy_installer.model_previews import PREVIEW_EXTENSIONS, preview_path, public_url
from comfy_installer.model_metadata import METADATA_MAX_SIZE, METADATA_SUFFIXES, complete_model_metadata, download_metadata, enrich_civitai_version, metadata_details, metadata_paths, read_model_metadata
from comfy_installer.operations import uv_python_path
from comfy_installer.runtime_state import load_runtime_receipt, RuntimeStateError
from json_store import write_json_atomic
from settings_resource_details import civitai_version, resource_details, resource_source_url


MODEL_EXTENSIONS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".pkl", ".torchscript"}
MODEL_KINDS = {
    "checkpoints": "Checkpoint", "diffusion_models": "Diffusion", "unet": "Diffusion",
    "vae": "VAE", "vae_approx": "VAE", "text_encoders": "Text encoder", "clip": "Text encoder",
    "upscale_models": "Upscaler", "controlnet": "ControlNet", "clip_vision": "CLIP vision",
    "embeddings": "Embedding",
}
ALLOWED_FOLDERS = {*MODEL_KINDS, "loras", "hypernetworks"}
MANIFEST_DELETE_BLOCKED = "설치 명세에 등록된 파일은 삭제할 수 없습니다."
MANIFEST_UNAVAILABLE = "설치 명세를 확인할 수 없어 삭제할 수 없습니다. 명세를 복구한 뒤 새로고침하세요."
MANIFEST_MOVE_BLOCKED = "설치 명세에 등록된 파일은 이동할 수 없습니다."
MANIFEST_MOVE_UNAVAILABLE = "설치 명세를 확인할 수 없어 이동할 수 없습니다. 명세를 복구한 뒤 새로고침하세요."


def safe_relative(value: str, *, filename=False) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or re.search(r'[<>:"|?*\x00-\x1f]', value)
            or (filename and len(path.parts) != 1)
            or any(part.endswith((".", " ")) for part in path.parts)):
        raise ValueError("파일 이름과 프로젝트 내부 폴더 경로를 확인하세요.")
    for part in path.parts:
        if part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            raise ValueError("사용할 수 없는 파일 이름입니다.")
    return path.as_posix()


def public_download_url(value: object) -> str:
    """Expose stable download selectors, never embedded authentication tokens."""
    if not isinstance(value, str):
        return ""
    try:
        url = urlsplit(public_url(value))
    except ValueError:
        return ""
    query = urlencode([(key, value) for key, value in parse_qsl(url.query)
                       if key.lower() in {"fileid", "modelversionid", "type", "format", "size", "fp", "download"}])
    return urlunsplit((url.scheme, url.netloc, url.path, query, ""))


class ResourceDeleteBlockedError(RuntimeError):
    """The current install manifest does not permit deleting this file."""


class ResourceMoveBlockedError(RuntimeError):
    """The current install manifest does not permit relocating this file."""


class ResourceLibrary:
    def __init__(self, project_root: Path = SOURCE_DIR):
        self.project_root = Path(project_root).resolve()
        self.root = data_root(project_root) / "comfy" / "models"
        self._previews: dict[str, tuple[Path, Path]] = {}

    def _inside(self, path: Path) -> bool:
        # Do not follow symlinks or Windows junctions outside this model folder.
        return path.resolve().is_relative_to(self.root.absolute())

    def installation_blockers(self) -> list[str]:
        comfy_root = self.root.parent
        try:
            receipt = load_runtime_receipt(comfy_root)
        except RuntimeStateError:
            receipt = None
        if (not (comfy_root / "main.py").is_file() or not (comfy_root / "folder_paths.py").is_file()
                or not (comfy_root / "comfy").is_dir() or not uv_python_path(comfy_root / ".venv").is_file() or receipt is None):
            return ["ComfyUI가 설치되지 않았습니다. 초기 설치를 먼저 완료하세요."]
        return []

    def target(self, folder: str, subfolder: str, name: str) -> Path:
        if folder not in ALLOWED_FOLDERS:
            raise ValueError("설치할 모델 폴더를 선택하세요.")
        name = safe_relative(name, filename=True)
        subfolder = safe_relative(subfolder) if subfolder else ""
        target = self.root / folder / subfolder / name
        for path in (target, target.with_name(target.name + ".part"), target.with_suffix(".metadata.json")):
            if not self._inside(path):
                raise ValueError("프로젝트 외부로 연결된 모델 경로에는 설치할 수 없습니다.")
        return target

    def _metadata(self, model: Path) -> dict:
        return read_model_metadata(model, self.root)

    def preview_path(self, model: Path) -> Path | None:
        return preview_path(model, self.root)

    def scan(self) -> dict:
        result: dict[str, list] = {"models": [], "loras": []}
        previews: dict[str, tuple[Path, Path]] = {}
        # Even the ComfyUI/models directory itself may be an external junction.
        if not self.root.resolve().is_relative_to(data_root(self.project_root).absolute()) or not self.root.is_dir():
            self._previews = {}
            return result
        # Reload the same manifest used by the installer on every inventory/delete.
        # Resolve aliases as well as normal paths so an internal junction cannot
        # expose a managed file as an unprotected entry.
        manifest_models: dict[Path, dict] = {}
        manifest_error = ""
        try:
            manifest_models = {
                (self.root.parent / model["relative_path"].replace("\\", "/")).resolve(): model
                for model in load_install_manifest().models
            }
        except (ManifestError, OSError, RuntimeError):
            manifest_error = MANIFEST_UNAVAILABLE
        visited: set[Path] = set()
        seen: set[Path] = set()
        # Internal LoRAs are listed by their own screens, never as external models.
        managed_lora_root = (self.root / "loras" / MANAGED_LORA_ROOT).resolve()

        def is_managed_lora(path: Path) -> bool:
            return path == managed_lora_root or managed_lora_root in path.parents

        def walk_error(error: OSError):
            raise error

        for directory, children, filenames in os.walk(self.root, followlinks=True, onerror=walk_error):
            current = Path(directory)
            resolved = current.resolve()
            if resolved in visited or not self._inside(current):
                children[:] = []
                continue
            visited.add(resolved)
            children[:] = sorted(name for name in children
                                 if not name.startswith(".") and self._inside(current / name)
                                 and not is_managed_lora((current / name).resolve()))
            for name in sorted(filenames):
                path = current / name
                if path.suffix.lower() not in MODEL_EXTENSIONS or not self._inside(path):
                    continue
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue  # A download or a user may remove a file while scanning.
                resolved_path = path.resolve()
                if not path.is_file() or not stat.st_size or resolved_path in seen or is_managed_lora(resolved_path):
                    continue
                seen.add(resolved_path)
                relative = path.relative_to(self.root)
                folder = relative.parts[0] if len(relative.parts) > 1 else ""
                identity = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()
                metadata = self._metadata(path)
                civitai = metadata.get("civitai") if isinstance(metadata.get("civitai"), dict) else {}
                download_url = public_download_url(manifest_models.get(path.resolve(), {}).get("url"))
                if not download_url:
                    download_url = public_download_url(metadata.get("download_url") or metadata.get("downloadUrl") or civitai.get("downloadUrl"))
                version_id = metadata.get("version_id") if metadata.get("source") == "civitai" else civitai.get("id")
                if not download_url and str(version_id).isdecimal():
                    download_url = f"https://civitai.com/api/download/models/{version_id}"
                if not download_url:
                    download_url = public_download_url(metadata.get("hf_url"))
                family = metadata.get("base_model") or "미확인"
                words = metadata.get("trigger") or ""
                preview = self.preview_path(path)
                if preview:
                    previews[identity] = (path, preview)
                item = {
                    "id": identity, "name": path.stem, "path": relative.as_posix(), "size": str(stat.st_size),
                    "family": family if isinstance(family, str) else "미확인",
                    "trigger": words if isinstance(words, str) else "",
                    "thumbnail": f"/api/setting/resources/thumbnail/{identity}" if preview else "",
                    "downloadUrl": download_url,
                    "sourceUrl": resource_source_url(metadata, download_url),
                    "deleteBlockedReason": manifest_error or (MANIFEST_DELETE_BLOCKED if path.resolve() in manifest_models else ""),
                    "moveBlockedReason": MANIFEST_MOVE_UNAVAILABLE if manifest_error else (MANIFEST_MOVE_BLOCKED if path.resolve() in manifest_models else ""),
                }
                if folder == "loras":
                    result["loras"].append({**item, "category": "LoRA", "weight": 0.7, "enabled": False})
                else:
                    result["models"].append({**item, "kind": MODEL_KINDS.get(folder, "Other"), "planned": False})
        self._previews = previews
        return result

    def resolve(self, identity: str) -> tuple[dict, Path]:
        # Resolve only an ID from a fresh inventory, never a client-supplied path.
        inventory = self.scan()
        entry = next((item for items in inventory.values() for item in items if item["id"] == identity), None)
        if entry is None:
            raise FileNotFoundError("파일이 없습니다. 목록을 새로고침하세요.")
        model = self.root / entry["path"]
        if not self._inside(model) or not model.is_file():
            raise FileNotFoundError("파일이 없습니다. 목록을 새로고침하세요.")
        return entry, model

    def details(self, identity: str) -> dict:
        entry, model = self.resolve(identity)
        return resource_details(entry, self._metadata(model))

    def refresh_metadata(self, identity: str, civitai_key: str) -> dict:
        entry, model = self.resolve(identity)
        metadata = self._metadata(model)
        version_id = resource_details(entry, metadata)["civitaiVersionId"]
        if not version_id:
            raise ValueError("Civitai 버전 정보를 찾을 수 없습니다.")
        target = model.with_suffix(".metadata.json")
        if not self._inside(target):
            raise ValueError("메타데이터 저장 경로를 확인하세요.")
        headers = {"Authorization": f"Bearer {civitai_key.strip()}"} if civitai_key.strip() else {}
        options = {"client_factory": lambda: httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "ComfyWorkshop/1.0"}),
                   "headers_for_url": lambda url: headers if urlsplit(url).hostname == "civitai.com" else {}, "cancel_event": Event()}
        payload, _ = download_metadata(urls=[f"https://civitai.com/api/v1/model-versions/{version_id}"], metadata={}, **options)
        sources = payload.get("metadata_sources", [])
        if not sources or str(sources[0].get("id")) != version_id:
            raise ValueError("Civitai 정보를 가져오지 못했습니다. 연결 상태와 기타 API 관리의 Civitai 키를 확인하세요.")
        previous = civitai_version(metadata)
        version = {**previous, **sources[0]}
        version["model"] = {**(previous.get("model") or {}), **(version.get("model") or {})}
        version, warning = enrich_civitai_version(version, **options, force=True)
        metadata.update({"civitai": version, "version_id": version_id, **metadata_details(version)})
        metadata = complete_model_metadata(model, metadata)
        if len(json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")) > METADATA_MAX_SIZE:
            raise ValueError("메타데이터가 너무 커서 저장하지 못했습니다.")
        write_json_atomic(target, metadata)
        return {**resource_details(entry, metadata), "warning": warning}

    def move(self, identity: str, folder: str, subfolder: str = "") -> dict:
        entry, model = self.resolve(identity)
        if entry["moveBlockedReason"]:
            raise ResourceMoveBlockedError(entry["moveBlockedReason"])
        if model.is_symlink():
            raise ValueError("심볼릭 링크 파일은 이동할 수 없습니다. 원본 모델 파일을 이동하세요.")
        if not isinstance(folder, str) or not isinstance(subfolder, str) or len(subfolder) > 240:
            raise ValueError("이동할 폴더를 확인하세요.")
        if ("category" in entry) != (folder == "loras"):
            raise ValueError("LoRA는 loras 안에서, 나머지 모델은 모델 폴더 사이에서 이동하세요.")
        if any(part.startswith(".") for part in PurePosixPath(subfolder).parts):
            raise ValueError("숨김 폴더에는 이동할 수 없습니다.")
        target = self.target(folder, subfolder, model.name)
        if target.resolve() == model.resolve():
            raise ValueError("현재 위치와 다른 폴더를 선택하세요.")
        # Do not fill a managed install location with a different model, either.
        try:
            managed = {(self.root.parent / item["relative_path"].replace("\\", "/")).resolve()
                       for item in load_install_manifest().models}
        except (ManifestError, OSError, RuntimeError):
            raise ResourceMoveBlockedError(MANIFEST_MOVE_UNAVAILABLE) from None
        if model.resolve() in managed or target.resolve() in managed:
            raise ResourceMoveBlockedError(MANIFEST_MOVE_BLOCKED)

        def previews(path: Path) -> list[Path]:
            return [path.with_suffix(prefix + extension)
                    for prefix in (".preview", "") for extension in PREVIEW_EXTENSIONS]
        occupied = [target, target.with_name(target.name + ".part"), *metadata_paths(target), *previews(target),
                    *(target.with_suffix(extension) for extension in MODEL_EXTENSIONS)]
        for path in occupied:
            if not self._inside(path):
                raise ValueError("프로젝트 외부로 연결된 모델 경로에는 이동할 수 없습니다.")
        occupied_names = {path.name.casefold() for path in occupied}
        if target.parent.is_dir() and any(path.name.casefold() in occupied_names for path in target.parent.iterdir()):
            raise FileExistsError("같은 이름의 모델 또는 미리보기·메타데이터가 이미 있습니다.")

        shared = any(sibling != model and sibling.stem.casefold() == model.stem.casefold()
                     and sibling.suffix.lower() in MODEL_EXTENSIONS and sibling.is_file()
                     for sibling in model.parent.iterdir())
        sidecars = []
        for path in [*metadata_paths(model), *previews(model)]:
            if not self._inside(path):
                raise ValueError("프로젝트 외부로 연결된 미리보기·메타데이터는 이동할 수 없습니다.")
            if path.is_file():
                sidecars.append(path)

        # Publish with exclusive creation, keeping every original until the model
        # is ready. Hard links avoid copying multi-GB weights on the same volume.
        created: list[Path] = []
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            for source in [*sidecars, model]:
                destination = target.parent / source.name
                if not self._inside(source) or not self._inside(destination):
                    raise ValueError("프로젝트 내부의 모델 경로를 선택하세요.")
                if shared and source != model and not source.name.startswith(model.name + "."):
                    with source.open("rb") as incoming, destination.open("xb") as outgoing:
                        created.append(destination)
                        shutil.copyfileobj(incoming, outgoing)
                else:
                    os.link(source.resolve(), destination)
                    created.append(destination)
            model.unlink()
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise

        failed = []
        for path in sidecars:
            if shared and not path.name.startswith(model.name + "."):
                continue
            try:
                path.unlink()
            except OSError:
                failed.append(path.name)
        relative = target.relative_to(self.root).as_posix()
        new_identity = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        preview = self._previews.pop(identity, None)
        if preview:
            self._previews[new_identity] = (target, target.parent / preview[1].name)
        moved = {**entry, "id": new_identity, "path": relative,
                 "thumbnail": f"/api/setting/resources/thumbnail/{new_identity}" if preview else ""}
        if "kind" in moved:
            moved["kind"] = MODEL_KINDS.get(folder, "Other")
        warning = "이동은 완료했지만 원래 폴더의 일부 미리보기·메타데이터를 정리하지 못했습니다: " + ", ".join(failed) if failed else ""
        return {"moved": identity, "item": moved, "warning": warning}

    def delete(self, identity: str) -> dict:
        entry, model = self.resolve(identity)
        if entry["deleteBlockedReason"]:
            raise ResourceDeleteBlockedError(entry["deleteBlockedReason"])
        # Different model formats can share the same thumbnail and metadata.
        shared = any(
            sibling != model and sibling.stem.casefold() == model.stem.casefold()
            and sibling.suffix.lower() in MODEL_EXTENSIONS and sibling.is_file()
            for sibling in model.parent.iterdir()
        )
        sidecars = [path for path in metadata_paths(model) if not shared or path.name.startswith(model.name + ".")]
        if not shared:
            sidecars.extend(model.with_suffix(prefix + extension) for prefix in (".preview", "") for extension in PREVIEW_EXTENSIONS)
        model.unlink()
        self._previews.pop(identity, None)
        failed = []
        for path in sidecars:
            try:
                if self._inside(path) and path.is_file():
                    path.unlink()
            except OSError:
                failed.append(path.name)
        warning = "모델 파일은 삭제했지만 일부 썸네일·메타데이터를 삭제하지 못했습니다: " + ", ".join(failed) if failed else ""
        return {"deleted": identity, "warning": warning}

    def thumbnail(self, identity: str) -> Path | None:
        pair = self._previews.get(identity)
        if pair:
            model, preview = pair
            if self._inside(model) and self._inside(preview) and model.is_file() and preview.is_file():
                return preview
        return None
