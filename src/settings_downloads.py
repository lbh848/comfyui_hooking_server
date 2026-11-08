"""User-selected model downloads, stored only in this project's ComfyUI folder."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
from threading import Event, Lock, Thread
from urllib.parse import parse_qs, quote, unquote, urlsplit

import httpx

from app_paths import SOURCE_DIR
from comfy_installer.downloader import DownloadCancelled, ResumableDownloader, _parse_content_range, sha256_file
from comfy_installer.model_previews import civitai_preview_urls, download_preview, huggingface_preview_urls, public_url
from comfy_installer.model_metadata import download_metadata, metadata_details, metadata_paths, save_model_metadata
from settings_resources import ALLOWED_FOLDERS, MODEL_EXTENSIONS, PREVIEW_EXTENSIONS, ResourceLibrary, public_download_url, safe_relative


TYPE_FOLDERS = {
    "checkpoint": "checkpoints", "lora": "loras", "locon": "loras", "dora": "loras",
    "vae": "vae", "controlnet": "controlnet", "upscaler": "upscale_models",
    "textualinversion": "embeddings", "hypernetwork": "hypernetworks",
    "diffusionmodel": "diffusion_models", "diffusion model": "diffusion_models", "unet": "diffusion_models",
}


class DownloadKindError(ValueError):
    def __init__(self, required_kind: str):
        self.required_kind = required_kind
        label = "LoRA" if required_kind == "loras" else "모델"
        super().__init__(f"이 주소는 {label} 설치에서 설치할 수 있습니다. 아래 버튼으로 전환하세요.")


class ResourceDownloadService:
    def __init__(self, project_root: Path = SOURCE_DIR, *, client_factory=None, downloader=None):
        self.library = ResourceLibrary(project_root)
        self.root = self.library.root
        self.client_factory = client_factory or (lambda: httpx.Client(timeout=45, follow_redirects=True, headers={"User-Agent": "ComfyWorkshop/1.0", "Accept-Encoding": "identity"}))
        self.downloader = downloader or ResumableDownloader(client_factory=self.client_factory)
        self._lock = Lock()
        self._cancel = Event()
        self._files: dict[str, dict] = {}
        self._jobs: list[dict] = []
        self._keys: dict[str, str] = {}
        self._state = "idle"
        self._requested = "cancelled"

    @staticmethod
    def _headers(url: str, keys: dict) -> dict:
        host = urlsplit(url).hostname
        key = keys.get("other-civitai", "") if host == "civitai.com" else keys.get("other-huggingface", "") if host == "huggingface.co" else ""
        if host == "civitai.com" and not key.strip():
            raise ValueError("Civitai API 키가 입력되지 않았습니다. 기타 API 관리에서 키를 입력하세요.")
        return {"Authorization": f"Bearer {key.strip()}"} if key.strip() else {}

    def installation_blockers(self) -> list[str]:
        return self.library.installation_blockers()

    def _require_installation(self) -> None:
        blockers = self.installation_blockers()
        if blockers:
            raise ValueError(" · ".join(blockers))

    def _json(self, url: str, keys: dict):
        public_url(url)
        try:
            with self.client_factory() as client:
                response = client.get(url, headers=self._headers(url, keys))
                response.raise_for_status()
                return response.json(), response.links.get("next", {}).get("url")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise ValueError("기타 API 관리의 API 키와 해당 파일의 다운로드 권한을 확인하세요.") from None
            raise ValueError("파일 정보를 불러오지 못했습니다. 주소와 공개 여부를 확인하세요.") from None
        except (httpx.HTTPError, ValueError):
            raise ValueError("파일 정보를 불러오지 못했습니다. 주소와 연결을 확인하세요.") from None

    def _file(self, *, source: str, name: str, remote_path: str, size: int, digest: str | None, urls: list[str], folder: str | None, metadata: dict, preview_urls: list[str] | None = None, metadata_urls: list[str] | None = None) -> dict:
        safe_relative(name, filename=True)
        if Path(name).suffix.lower() not in MODEL_EXTENSIONS:
            raise ValueError("지원하지 않는 모델 파일 형식입니다.")
        urls = list(dict.fromkeys(public_url(url) for url in urls))
        if not urls:
            raise ValueError("사용 가능한 다운로드 주소가 없습니다.")
        digest = digest.lower() if digest else None
        if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("파일의 SHA-256 정보가 올바르지 않습니다.")
        identity = hashlib.sha256(f"{source}:{urls[0]}:{digest or ''}:{remote_path}".encode()).hexdigest()
        item = {"id": identity, "source": source, "name": name, "remotePath": remote_path, "size": size, "folder": folder,
                "sha256": digest, "urls": urls, "metadata": metadata, "preview_urls": preview_urls or [], "metadata_urls": metadata_urls or []}
        with self._lock:
            self._files[identity] = item
        return {key: value for key, value in item.items() if key not in {"urls", "metadata", "sha256", "preview_urls", "metadata_urls"}}

    def _civitai(self, value: str, kind: str, keys: dict, requested_version: str | None) -> dict:
        parsed = urlsplit(value)
        source = "civarchive" if parsed.hostname in {"civarchive.com", "www.civarchive.com", "civitaiarchive.com"} else "civitai"
        if source == "civitai":
            self._headers("https://civitai.com", keys)
        model_match = re.match(r"^/models/(\d+)(?:/|$)", parsed.path)
        version_match = re.match(r"^/(?:api/v1/model-versions|api/download/models)/(\d+)(?:/|$)", parsed.path)
        version_id = requested_version or parse_qs(parsed.query).get("modelVersionId", [None])[0] or (version_match[1] if version_match else None)
        if value.isdecimal():
            version_id = value
        if version_id and (not str(version_id).isdecimal() or int(version_id) <= 0):
            raise ValueError("모델 버전 ID를 확인하세요.")
        if source == "civarchive":
            if not model_match:
                raise ValueError("CivArchive 모델 주소를 입력하세요.")
            url = f"https://civarchive.com/api/models/{model_match[1]}"
            payload, _ = self._json(url + (f"?modelVersionId={version_id}" if version_id else ""), keys)
            data = payload.get("data", payload)
            model = {**data, **(data.get("model") or {})}
            version = data.get("version") or model.get("version") or {}
            versions = data.get("versions") or [version]
            if not version:
                raise ValueError("CivArchive에서 설치할 수 있는 버전을 찾지 못했습니다.")
            version = {**version, "files": version.get("files") or data.get("files") or model.get("files") or []}
        elif version_id:
            version, _ = self._json(f"https://civitai.com/api/v1/model-versions/{version_id}", keys)
            model = version.get("model") or {}
            versions = [version]
        else:
            if not model_match:
                raise ValueError("Civitai 모델 주소 또는 버전 ID를 입력하세요.")
            model, _ = self._json(f"https://civitai.com/api/v1/models/{model_match[1]}", keys)
            versions = model.get("modelVersions") or []
            if not versions:
                raise ValueError("설치할 수 있는 모델 버전이 없습니다.")
            version = versions[0]
        model_type = str(model.get("type") or "").lower()
        folder = TYPE_FOLDERS.get(model_type)
        if not folder:
            raise ValueError("지원하지 않는 모델 유형입니다.")
        if (kind == "loras") != (folder == "loras"):
            raise DownloadKindError("loras" if folder == "loras" else "models")
        metadata = {"model_name": str(model.get("name") or ""), "source": source, "version_id": version.get("id"),
                    "civitai": {**version, "model": {key: val for key, val in model.items() if key != "modelVersions"}},
                    **metadata_details(version, model)}
        preview_urls = civitai_preview_urls(version, model)
        files = []
        for item in version.get("files") or []:
            try:
                mirrors = [mirror.get("url") for mirror in item.get("mirrors", []) if isinstance(mirror, dict) and mirror.get("deletedAt") is None and mirror.get("url")]
                urls = mirrors or [item.get("downloadUrl")]
                if source == "civarchive":
                    urls.sort(key=lambda url: urlsplit(url or "").hostname == "civitai.com")
                if not urls[0]:
                    continue
                name = item.get("name") or next((mirror.get("filename") for mirror in item.get("mirrors", []) if mirror.get("filename")), "")
                file_folder = TYPE_FOLDERS.get(str(item.get("type") or "").lower(), folder)
                if kind == "loras" and file_folder != "loras":
                    continue
                digest = (item.get("hashes") or {}).get("SHA256") or item.get("sha256")
                if not digest:
                    continue
                files.append(self._file(source=source, name=name, remote_path=name, size=max(0, round(float(item.get("sizeKB") or 0) * 1024)), digest=digest,
                                        urls=urls, folder=file_folder, metadata=metadata, preview_urls=preview_urls))
            except (ValueError, TypeError):
                continue
        return {"url": value, "source": source, "title": str(model.get("name") or ""), "versions": [{"id": str(v["id"]), "name": str(v.get("name") or v["id"])} for v in versions if v.get("id")],
                "versionId": str(version.get("id") or ""), "files": files}

    def _huggingface(self, value: str, kind: str, keys: dict) -> dict:
        parsed = urlsplit(value if "://" in value else f"https://huggingface.co/{value}")
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."} for part in parts[:2]):
            raise ValueError("Hugging Face 저장소 또는 파일 주소를 입력하세요.")
        repo = "/".join(parts[:2])
        revision = unquote(parts[3]) if len(parts) >= 4 and parts[2] in {"resolve", "blob", "tree"} else "main"
        selected_path = unquote("/".join(parts[4:])) if len(parts) >= 5 else ""
        file_only = len(parts) >= 5 and parts[2] in {"resolve", "blob"}
        url = f"https://huggingface.co/api/models/{repo}/tree/{quote(revision, safe='')}?recursive=true&expand=false&limit=1000"
        metadata = {"base_model": "미확인", "hf_url": f"https://huggingface.co/{repo}", "source": "huggingface"}
        candidates = []
        preview_paths = []
        sidecar_paths = set()
        pages = 0
        while url:
            if urlsplit(url).hostname != "huggingface.co":
                raise ValueError("Hugging Face 파일 목록 주소가 올바르지 않습니다.")
            entries, url = self._json(url, keys)
            if not isinstance(entries, list):
                raise ValueError("Hugging Face 파일 목록 형식이 올바르지 않습니다.")
            for entry in entries:
                path = entry.get("path", "")
                if entry.get("type") != "file":
                    continue
                if path.endswith((".json", ".civitai.info")):
                    sidecar_paths.add(safe_relative(path))
                    continue
                if Path(path).suffix.lower() in PREVIEW_EXTENSIONS:
                    preview_paths.append(safe_relative(path))
                    continue
                if selected_path and (path != selected_path if file_only else not path.startswith(selected_path.rstrip("/") + "/")):
                    continue
                if Path(path).suffix.lower() not in MODEL_EXTENSIONS:
                    continue
                safe_relative(path)
                candidates.append(entry)
            pages += 1
            if pages > 100 or len(candidates) > 10000:
                raise ValueError("파일 수가 많습니다. 저장소의 하위 폴더나 파일 주소를 입력하세요.")
        files = []
        for entry in candidates:
            path = PurePosixPath(entry["path"])
            lfs = entry.get("lfs") or {}
            digest = lfs.get("oid") or lfs.get("sha256")
            folder = "loras" if kind == "loras" else next((part for part in path.parts[:-1] if part in ALLOWED_FOLDERS and part != "loras"), None)
            prefix = f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/"
            previews = huggingface_preview_urls(path, preview_paths, prefix)
            metadata_urls = [prefix + quote(candidate.as_posix(), safe='/') for candidate in metadata_paths(path) if candidate.as_posix() in sidecar_paths]
            files.append(self._file(source="huggingface", name=path.name, remote_path=path.as_posix(),
                                    size=int(lfs.get("size") or entry.get("size") or 0), digest=digest,
                                    urls=[prefix + quote(path.as_posix(), safe='/')], folder=folder, metadata=metadata,
                                    preview_urls=previews, metadata_urls=metadata_urls))
        return {"url": value, "source": "huggingface", "title": repo, "versions": [], "versionId": revision, "files": files}

    def resolve(self, urls: list[str], kind: str, keys: dict, version_id: str | None = None) -> dict:
        if kind not in {"models", "loras"} or not 1 <= len(urls) <= 50:
            raise ValueError("모델 주소를 1~50개 입력하세요.")
        groups = []
        for value in urls:
            try:
                host = urlsplit(value).hostname
                if host == "huggingface.co" or (not host and not value.isdecimal() and "://" not in value):
                    group = self._huggingface(value, kind, keys)
                elif value.isdecimal() or host in {"civitai.com", "www.civitai.com", "civitai.red", "civitai.green", "civarchive.com", "www.civarchive.com", "civitaiarchive.com"}:
                    if not value.isdecimal():
                        public_url(value)
                    group = self._civitai(value, kind, keys, version_id)
                else:
                    raise ValueError("Civitai, Hugging Face 또는 CivArchive 주소를 입력하세요.")
                if not group["files"]:
                    raise ValueError("설치할 수 있는 모델 파일이 없습니다.")
                groups.append(group)
            except (ValueError, TypeError, KeyError) as error:
                group = {"url": value, "error": str(error) if isinstance(error, ValueError) else "파일 정보를 확인할 수 없습니다.", "files": [], "versions": []}
                if isinstance(error, DownloadKindError):
                    group["requiredKind"] = error.required_kind
                groups.append(group)
        return {"groups": groups}

    def status(self) -> dict:
        with self._lock:
            items = [{k: v for k, v in job.items() if k not in {"file", "target"}} for job in self._jobs]
            return {"state": self._state, "items": items, "completed": sum(item["state"] == "completed" for item in items), "total": len(items),
                    "installBlockers": self.installation_blockers()}

    def _target(self, folder: str, subfolder: str, name: str) -> Path:
        return self.library.target(folder, subfolder, name)

    def start(self, identities: list[str], kind: str, folder: str, subfolder: str, keys: dict) -> dict:
        self._require_installation()
        if kind not in {"models", "loras"} or not 1 <= len(identities) <= 200:
            raise ValueError("설치할 파일을 1~200개 선택하세요.")
        jobs = []
        targets = set()
        with self._lock:
            if self._state == "running":
                raise ValueError("진행 중인 설치가 끝난 뒤 다시 시도하세요.")
            for identity in dict.fromkeys(identities):
                file = self._files.get(identity)
                if not file:
                    raise ValueError("먼저 모델 정보를 다시 불러오세요.")
                if file["source"] == "civitai":
                    self._headers("https://civitai.com", keys)
                if kind == "models" and file["folder"] == "loras":
                    raise ValueError("외부 LoRA 관리에서 설치하세요.")
                selected_folder = "loras" if kind == "loras" else file["folder"] if folder == "auto" else folder
                if not selected_folder:
                    raise ValueError("Hugging Face 파일의 설치 폴더를 선택하세요.")
                if kind == "models" and selected_folder == "loras":
                    raise ValueError("외부 LoRA 관리에서 설치하세요.")
                target = self._target(selected_folder, subfolder, file["name"])
                if target in targets:
                    raise ValueError("같은 이름의 파일이 선택되어 있습니다. 파일을 나누어 설치하거나 하위 폴더를 지정하세요.")
                targets.add(target)
                jobs.append({"id": identity, "name": file["name"], "path": target.relative_to(self.root).as_posix(), "source": file["source"], "state": "queued",
                             "downloaded": 0, "size": file["size"], "speed": 0, "error": "", "warning": "", "file": file, "target": target})
            self._jobs = jobs
            self._keys = dict(keys)
            self._launch()
        return self.status()

    def _launch(self):
        self._state = "running"
        self._cancel = Event()
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def _update(self, job: dict, **values):
        with self._lock:
            job.update(values)

    def _download_metadata(self, file: dict, cancel: Event) -> tuple[dict, str]:
        return download_metadata(urls=file["metadata_urls"], metadata=file["metadata"], client_factory=self.client_factory,
                                 headers_for_url=lambda url: self._headers(url, self._keys), cancel_event=cancel)

    def _download(self, job: dict, cancel: Event):
        file, target = job["file"], job["target"]
        if target.exists() and (not file["sha256"] or sha256_file(target, cancel) != file["sha256"]):
            raise ValueError("같은 이름의 파일이 이미 있습니다. 기존 파일을 확인하거나 하위 폴더를 지정하세요.")
        last_error = None
        for url in file["urls"]:
            try:
                headers = self._headers(url, self._keys)
                if target.exists():
                    size = target.stat().st_size
                else:
                    with self.client_factory() as client:
                        with client.stream("GET", url, headers={**headers, "Range": "bytes=0-0"}) as response:
                            response.raise_for_status()
                            content_range = _parse_content_range(response.headers.get("Content-Range"))
                            size = content_range[2] if response.status_code == 206 and content_range else int(response.headers.get("Content-Length") or 0) if response.status_code == 200 else 0
                    if not size:
                        raise ValueError("다운로드 파일 크기를 확인할 수 없습니다.")
                self._update(job, size=size)
                result = self.downloader.download(url=url, target=target, expected_size=size, expected_sha256=file["sha256"], headers=headers, cancel_event=cancel,
                    progress=lambda value: self._update(job, downloaded=value.get("downloaded", 0), speed=value.get("bytes_per_second", 0)))
                metadata_warning = ""
                warning = download_preview(
                    urls=file["preview_urls"], target=target, root=self.root, client_factory=self.client_factory,
                    headers_for_url=lambda value: self._headers(value, self._keys), cancel_event=cancel,
                )
                try:
                    details, metadata_warning = self._download_metadata(file, cancel)
                    save_model_metadata(target, {
                        **details, "sha256": result.sha256, "download_url": public_download_url(url),
                    }, self.root)
                except (OSError, ValueError, RecursionError):
                    metadata_warning = "모델 파일은 저장했지만 메타데이터를 저장하지 못했습니다. 파일 접근 권한과 기존 메타데이터 형식을 확인하세요."
                self._update(job, state="completed", downloaded=size, speed=0, warning=" · ".join(value for value in (metadata_warning, warning) if value))
                return
            except DownloadCancelled:
                raise
            except Exception as exc:
                last_error = exc
        if isinstance(last_error, ValueError):
            raise last_error
        raise ValueError("다운로드에 실패했습니다. API 키, 다운로드 권한, 디스크 공간과 연결을 확인하세요.") from None

    def _run(self):
        cancel = self._cancel
        for job in self._jobs:
            if job["state"] == "completed":
                continue
            try:
                if cancel.is_set():
                    raise DownloadCancelled()
                self._update(job, state="running", error="")
                self._download(job, cancel)
            except DownloadCancelled:
                self._update(job, state=self._requested, speed=0)
                with self._lock:
                    self._state = self._requested
                return
            except Exception as exc:
                self._update(job, state="failed", error=str(exc) if isinstance(exc, ValueError) else "다운로드에 실패했습니다.", speed=0)
        with self._lock:
            self._state = "failed" if any(job["state"] == "failed" for job in self._jobs) else "completed"

    def clear(self) -> dict:
        with self._lock:
            if self._state not in {"idle", "completed"}:
                raise ValueError("다운로드가 모두 완료된 뒤 목록을 비울 수 있습니다.")
            self._jobs = []
            self._keys = {}
            self._state = "idle"
        return self.status()

    def stop(self, *, pause=False) -> dict:
        with self._lock:
            if self._state == "running":
                self._requested = "paused" if pause else "cancelled"
                self._cancel.set()
        return self.status()

    def resume(self, keys: dict) -> dict:
        self._require_installation()
        with self._lock:
            if self._state not in {"paused", "failed", "cancelled"}:
                raise ValueError("이어받을 다운로드가 없습니다.")
            if any(job["source"] == "civitai" and job["state"] != "completed" for job in self._jobs):
                self._headers("https://civitai.com", keys)
            self._keys = dict(keys)
            self._launch()
        return self.status()
