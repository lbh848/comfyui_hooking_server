"""Stream local model uploads and publish complete files into the library."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile

from aiohttp import BodyPartReader, web
from PIL import Image, UnidentifiedImageError

from comfy_installer.model_metadata import complete_model_metadata
from settings_resources import METADATA_SUFFIXES, MODEL_EXTENSIONS, PREVIEW_EXTENSIONS, ResourceLibrary, safe_relative


PREVIEW_MAX_SIZE = 10 * 1024 * 1024
FIELD_LIMITS = {"kind": 16, "folder": 64, "subfolder": 240, "family": 80, "trigger": 16384, "size": 24}


def publish_upload(library: ResourceLibrary, fields: dict[str, str], files: dict[str, tuple[Path, str, int]]) -> dict:
    if fields.get("kind") not in {"models", "loras"}:
        raise ValueError("모델 또는 LoRA 업로드를 선택하세요.")
    if "file" not in files:
        raise ValueError("업로드할 파일을 선택하세요.")
    model, filename, size = files["file"]
    if not fields.get("size", "").isdecimal() or int(fields["size"]) != size:
        raise ValueError("파일이 모두 전송되지 않았습니다. 다시 업로드하세요.")
    folder = fields.get("folder", "")
    if (fields["kind"] == "loras") != (folder == "loras"):
        raise ValueError("LoRA는 외부 LoRA 관리에서, 나머지 모델은 모델 관리에서 업로드하세요.")
    subfolder = fields.get("subfolder", "")
    if any(part.startswith(".") for part in Path(subfolder).parts):
        raise ValueError("숨김 폴더에는 업로드할 수 없습니다. 다른 하위 폴더를 입력하세요.")
    target = library.target(folder, subfolder, filename)
    metadata = target.with_suffix(".metadata.json")
    destinations = [(model, target)]
    if "preview" in files:
        preview, _, _ = files["preview"]
        try:
            with Image.open(preview) as image:
                extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(image.format)
                if not extension:
                    raise ValueError("미리보기는 PNG, JPEG, WebP 이미지를 선택하세요.")
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
            raise ValueError("미리보기 이미지 파일을 확인하세요.") from None
        destinations.append((preview, target.with_suffix(".preview" + extension)))
    # Sidecars are shared by files with the same stem, even across model formats.
    occupied = [target, target.with_name(target.name + ".part"), *(
        target.with_suffix(suffix) for suffix in (
            *MODEL_EXTENSIONS, *METADATA_SUFFIXES,
            *(prefix + extension for prefix in (".preview", "") for extension in PREVIEW_EXTENSIONS),
        )
    )]
    if any(path.exists() or path.is_symlink() for path in occupied):
        raise FileExistsError("같은 이름의 파일 또는 미리보기·메타데이터가 이미 있습니다. 다른 하위 폴더를 지정하세요.")
    staged_metadata = model.parent / "metadata.json"
    staged_metadata.write_text(json.dumps(complete_model_metadata(target, {
        "source": "upload", "source_filename": filename, "size": size,
        "base_model": fields.get("family") or "미확인", "trigger": fields.get("trigger", ""),
    }, source=model, preview=destinations[1][1] if len(destinations) > 1 else None),
        ensure_ascii=False, indent=2), encoding="utf-8")
    # Publish the model last so inventory scans only see complete entries.
    destinations = [(staged_metadata, metadata), *destinations[1:], destinations[0]]
    created: list[Path] = []
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        for source, destination in destinations:
            if not library._inside(destination):
                raise ValueError("프로젝트 외부로 연결된 모델 경로에는 업로드할 수 없습니다.")
            # Same-volume hard links are atomic, avoid a second multi-GB copy,
            # and fail if another writer has already created the destination.
            os.link(source, destination)
            created.append(destination)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {"name": filename, "path": target.relative_to(library.root).as_posix(), "size": size}


async def receive_upload(request: web.Request, library: ResourceLibrary, resource_lock: asyncio.Lock) -> dict:
    # Multipart chunks bypass aiohttp's small JSON-body limit without buffering
    # an entire model in memory. Temporary directories are hidden from inventory.
    library.target("loras", "", "upload.safetensors")
    library.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".upload-", dir=library.root) as temporary:
        fields: dict[str, str] = {}
        files: dict[str, tuple[Path, str, int]] = {}
        reader = await request.multipart()
        async for part in reader:
            if not isinstance(part, BodyPartReader):
                raise ValueError("업로드 형식을 확인하세요.")
            name = part.name
            if name in fields or name in files:
                raise ValueError("파일과 입력 항목은 하나씩 전송하세요.")
            if name in {"file", "preview"}:
                filename = safe_relative(part.filename or "", filename=True)
                extensions = MODEL_EXTENSIONS if name == "file" else PREVIEW_EXTENSIONS
                if Path(filename).suffix.lower() not in extensions:
                    raise ValueError("지원하지 않는 파일 형식입니다." if name == "file" else "미리보기는 PNG, JPEG, WebP 이미지를 선택하세요.")
                path = Path(temporary) / name
                size = 0
                with path.open("xb") as output:
                    while chunk := await part.read_chunk(1024 * 1024):
                        size += len(chunk)
                        if name == "preview" and size > PREVIEW_MAX_SIZE:
                            raise ValueError("미리보기 이미지는 10 MB 이하로 선택하세요.")
                        await asyncio.to_thread(output.write, chunk)
                if not size:
                    raise ValueError("빈 파일은 업로드할 수 없습니다.")
                files[name] = (path, filename, size)
            elif name in FIELD_LIMITS and part.filename is None:
                content = bytearray()
                while chunk := await part.read_chunk():
                    content.extend(chunk)
                    if len(content) > FIELD_LIMITS[name] * 4:
                        raise ValueError("업로드 입력 항목이 너무 깁니다.")
                value = content.decode("utf-8").strip()
                if len(value) > FIELD_LIMITS[name]:
                    raise ValueError("업로드 입력 항목이 너무 깁니다.")
                fields[name] = value
            else:
                raise ValueError("업로드 입력 항목을 확인하세요.")
        async with resource_lock:
            return publish_upload(library, fields, files)
