"""Shared model previews for managed installs and user-selected downloads."""
from __future__ import annotations

from io import BytesIO
import ipaddress
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from threading import Event
from typing import Callable
from urllib.parse import quote, unquote, urlsplit

import httpx
from PIL import Image

from .downloader import ClientFactory
from .model_metadata import download_metadata, metadata_details, metadata_paths, save_model_metadata


PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def public_url(value: str) -> str:
    url = urlsplit(value)
    host = url.hostname or ""
    if url.scheme != "https" or not host or url.username or url.password or url.port not in (None, 443):
        raise ValueError("HTTPS 다운로드 주소를 확인하세요.")
    if host == "localhost" or host.endswith((".localhost", ".local")):
        raise ValueError("로컬 주소는 다운로드 출처로 사용할 수 없습니다.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host:
            raise ValueError("다운로드 주소를 확인하세요.") from None
    else:
        if not address.is_global:
            raise ValueError("로컬 주소는 다운로드 출처로 사용할 수 없습니다.")
    return value


def preview_path(model: Path, root: Path) -> Path | None:
    for prefix in (".preview", ""):
        for extension in PREVIEW_EXTENSIONS:
            path = model.with_suffix(prefix + extension)
            if path.resolve().is_relative_to(root.absolute()) and path.is_file():
                return path
    return None


def civitai_preview_urls(version: dict, model: dict) -> list[str]:
    urls = []
    for image in version.get("images") or model.get("images") or []:
        if not isinstance(image, dict) or image.get("type") == "video" or not isinstance(image.get("url"), str):
            continue
        try:
            urls.append(public_url(image["url"]))
        except ValueError:
            continue
    return list(dict.fromkeys(urls))[:3]


def huggingface_preview_urls(path: PurePosixPath, preview_paths: list[str], prefix: str) -> list[str]:
    candidates = []
    for preview in dict.fromkeys(preview_paths):
        image = PurePosixPath(preview)
        stem = image.stem.casefold()
        if image.parent == path.parent and stem in {path.stem.casefold(), f"{path.stem}.preview".casefold()}:
            priority = 0
        elif any(word in stem for word in ("preview", "thumbnail", "cover", "example", "sample")):
            priority = 1
        else:
            continue
        candidates.append((priority, preview.casefold(), preview))
    return [prefix + quote(preview, safe="/") for _, _, preview in sorted(candidates)[:3]]


def download_preview(
    *, urls: list[str], target: Path, root: Path, client_factory: ClientFactory,
    headers_for_url: Callable[[str], dict], cancel_event: Event,
) -> str:
    if preview_path(target, root) or not urls:
        return ""
    for url in urls:
        if cancel_event.is_set():
            return "모델 파일은 저장했지만 썸네일 다운로드가 중단되었습니다."
        temporary = None
        try:
            public_url(url)
            content = bytearray()
            with client_factory() as client:
                with client.stream("GET", url, headers=headers_for_url(url)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if cancel_event.is_set():
                            return "모델 파일은 저장했지만 썸네일 다운로드가 중단되었습니다."
                        content.extend(chunk)
                        if len(content) > 10 * 1024 * 1024:
                            raise ValueError("썸네일 파일이 너무 큽니다.")
            with Image.open(BytesIO(content)) as image:
                extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(image.format)
                if not extension:
                    raise ValueError("지원하지 않는 썸네일 형식입니다.")
                image.verify()
            preview = target.with_suffix(".preview" + extension)
            if not preview.resolve().is_relative_to(root.absolute()):
                raise ValueError("썸네일 저장 경로를 확인하세요.")
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            try:
                os.link(temporary, preview)
            except FileExistsError:
                pass
            return ""
        except Exception:
            continue
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return "모델 파일은 저장했지만 썸네일을 가져오지 못했습니다."


def install_model_sidecars(
    *, url: str, target: Path, root: Path, civitai_key: str, cancel_event: Event,
    client_factory: ClientFactory | None = None,
    sha256: str | None = None,
) -> str:
    """Fetch metadata and a preview without modifying an installed model."""
    parsed = urlsplit(url)
    if parsed.hostname not in {"civitai.com", "huggingface.co"}:
        return ""
    try:
        metadata_path = target.with_suffix(".metadata.json")
        if (cancel_event.is_set() or not target.is_file()
                or not target.resolve().is_relative_to(root.absolute())):
            return ""
        if not metadata_path.resolve().is_relative_to(root.absolute()):
            return "프로젝트 외부로 연결된 메타데이터 경로에는 저장할 수 없습니다."
        client_factory = client_factory or (lambda: httpx.Client(
            timeout=30, follow_redirects=True, headers={"User-Agent": "ComfyWorkshop/1.0"},
        ))
        headers = {"Authorization": f"Bearer {civitai_key.strip()}"} if civitai_key.strip() else {}
        metadata_urls = []
        metadata = {"download_url": parsed._replace(query="", fragment="").geturl(), "size": target.stat().st_size}
        if sha256:
            metadata["sha256"] = sha256
        if parsed.hostname == "civitai.com":
            version = re.fullmatch(r"/api/download/models/(\d+)/?", parsed.path)
            if not version:
                return ""
            with client_factory() as client:
                response = client.get(f"https://civitai.com/api/v1/model-versions/{version[1]}", headers=headers)
                response.raise_for_status()
                data = response.json()
            metadata.update({"source": "civitai", "version_id": data.get("id"), "civitai": data, **metadata_details(data)})
            urls = civitai_preview_urls(data, data.get("model") or {})
        else:
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
                return ""
            repo = "/".join(parts[:2])
            revision = quote(unquote(parts[3]), safe="")
            path = PurePosixPath(unquote("/".join(parts[4:])))
            prefix = f"https://huggingface.co/{repo}/resolve/{revision}/"
            page_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=false&limit=1000"
            preview_paths = []
            sidecar_paths = set()
            with client_factory() as client:
                for _ in range(100):
                    if cancel_event.is_set():
                        return "모델 파일은 저장했지만 썸네일 다운로드가 중단되었습니다."
                    public_url(page_url)
                    if urlsplit(page_url).hostname != "huggingface.co":
                        raise ValueError("Hugging Face 파일 목록 주소를 확인하세요.")
                    response = client.get(page_url)
                    response.raise_for_status()
                    entries = response.json()
                    preview_paths.extend(entry["path"] for entry in entries
                                         if entry.get("type") == "file" and Path(entry.get("path", "")).suffix.lower() in PREVIEW_EXTENSIONS)
                    sidecar_paths.update(entry["path"] for entry in entries if entry.get("type") == "file" and entry.get("path", "").endswith((".json", ".civitai.info")))
                    page_url = response.links.get("next", {}).get("url")
                    if not page_url:
                        break
            urls = huggingface_preview_urls(path, preview_paths, prefix)
            metadata.update({"source": "huggingface", "hf_url": f"https://huggingface.co/{repo}"})
            metadata_urls = [prefix + quote(candidate.as_posix(), safe='/') for candidate in metadata_paths(path) if candidate.as_posix() in sidecar_paths]
        warning = download_preview(
            urls=urls, target=target, root=root, client_factory=client_factory,
            headers_for_url=lambda value: headers if urlsplit(value).hostname == "civitai.com" else {},
            cancel_event=cancel_event,
        )
        metadata, metadata_warning = download_metadata(urls=metadata_urls, metadata=metadata, client_factory=client_factory,
            headers_for_url=lambda value: headers if urlsplit(value).hostname == "civitai.com" else {}, cancel_event=cancel_event)
        save_model_metadata(target, metadata, root)
        if not urls and not preview_path(target, root):
            warning = "다운로드 출처에서 자동으로 선택할 썸네일을 찾지 못했습니다."
        return " · ".join(value for value in (metadata_warning, warning) if value)
    except Exception:
        return "모델 파일은 저장했지만 메타데이터 또는 썸네일을 가져오지 못했습니다."
