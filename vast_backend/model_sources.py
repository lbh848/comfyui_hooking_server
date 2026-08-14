"""모델 ↔ 다운로드 소스(HF/Civitai/URL/로컬 업로드) 매핑 관리.

- 로컬 모델 스캔/워크플로우 참조 분석은 modal_backend.workflow_assets를 재사용한다.
- 매핑은 프로젝트 루트의 vast_model_sources.json에 저장하며, 덮어쓰기 전
  배포 환경에도 존재하는 backups/vast_model_sources/에 백업 사본을 남긴다.
"""
from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

SOURCE_TYPES = ("hf", "civitai", "url", "upload")
MAPPING_FILENAME = "vast_model_sources.json"
MAPPING_VERSION = 1


def _mapping_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / MAPPING_FILENAME


def load_mapping(project_root: str | Path) -> dict[str, Any]:
    """저장된 매핑을 읽는다. 없거나 손상되면 빈 구조를 반환한다."""
    path = _mapping_path(project_root)
    if not path.is_file():
        return {"version": MAPPING_VERSION, "sources": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(
            f"[VAST_SOURCES] 매핑 파일 읽기 실패(빈 매핑으로 시작): {path}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return {"version": MAPPING_VERSION, "sources": {}}
    sources = data.get("sources")
    if not isinstance(sources, dict):
        print(f"[VAST_SOURCES] 매핑 파일 형식 이상(sources 누락): {path}")
        return {"version": MAPPING_VERSION, "sources": {}}
    return {"version": MAPPING_VERSION, "sources": sources}


def save_mapping(project_root: str | Path, mapping: Mapping[str, Any]) -> Path:
    """매핑을 저장한다. 기존 파일은 배포 안전 백업 폴더에 보존한다."""
    path = _mapping_path(project_root)
    if path.exists():
        backup_dir = (
            Path(project_root).resolve() / "backups" / "vast_model_sources"
        )
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{MAPPING_FILENAME}.{stamp}.bak"
            shutil.copy2(path, backup_path)
            print(f"[VAST_SOURCES] 매핑 백업: {path} -> {backup_path}")
        except OSError as exc:
            print(
                f"[VAST_SOURCES] 매핑 백업 실패(저장 중단): error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise RuntimeError(
                f"Vast 모델 매핑 백업 실패로 저장을 중단했습니다: {exc}"
            ) from exc
    payload = {
        "version": MAPPING_VERSION,
        "sources": dict(mapping.get("sources") or {}),
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[VAST_SOURCES] 매핑 저장 실패: {path}, error={exc}")
        traceback.print_exc()
        raise RuntimeError(f"Vast 모델 매핑 저장 실패: {exc}") from exc
    return path


def normalize_source_key(kind: str, filename: str) -> str:
    """매핑 키: '<모델폴더>/<파일명>' 형식으로 정규화한다."""
    kind = str(kind or "").strip().strip("/")
    filename = str(filename or "").strip().replace("\\", "/").strip("/")
    if not kind or not filename:
        raise ValueError(f"모델 소스 키가 비어 있습니다: kind={kind!r}, filename={filename!r}")
    if ".." in filename.split("/") or filename.startswith("/"):
        raise ValueError(f"모델 파일명에 상위 경로를 쓸 수 없습니다: {filename}")
    return f"{kind}/{filename}"


def validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """사용자가 입력한 소스 배정을 검증하고 정규화된 dict를 반환한다."""
    source_type = str(source.get("source_type") or "").strip().lower()
    if source_type not in SOURCE_TYPES:
        raise ValueError(
            f"source_type은 {', '.join(SOURCE_TYPES)} 중 하나여야 합니다: {source_type!r}"
        )
    result: dict[str, Any] = {"source_type": source_type}
    if source_type == "hf":
        repo_id = str(source.get("repo_id") or "").strip()
        hf_filename = str(source.get("hf_filename") or "").strip()
        if not repo_id or not hf_filename:
            raise ValueError("HF 소스에는 repo_id와 hf_filename이 모두 필요합니다.")
        result.update(repo_id=repo_id, hf_filename=hf_filename)
    elif source_type == "civitai":
        raw_version = source.get("civitai_version_id")
        try:
            version_id = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"civitai_version_id는 정수여야 합니다: {raw_version!r}"
            ) from exc
        if version_id <= 0:
            raise ValueError(f"civitai_version_id는 양수여야 합니다: {version_id}")
        result.update(civitai_version_id=version_id)
    elif source_type == "url":
        url = str(source.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"url 소스에는 http(s) URL이 필요합니다: {url!r}")
        result.update(url=url)
    # upload는 로컬 파일을 그대로 올리므로 추가 필드 없음
    return result


def civitai_download_url(version_id: int, api_key: str) -> str:
    """Civitai 모델버전 다운로드 URL (API 키 인증)."""
    url = f"https://civitai.com/api/download/models/{int(version_id)}"
    return f"{url}?token={api_key}" if api_key else url


def defaults_from_manifest(project_root: str | Path) -> dict[str, dict[str, Any]]:
    """comfy_installer/resources/install_manifest.json의 models 절에서
    '<kind>/<filename>' → 소스 배정 기본값을 만든다.

    - civitai URL: {version_id}를 파싱해 civitai 소스로
    - huggingface resolve URL: repo/filename을 파싱해 hf 소스로
    - 그 외 http(s): url 소스로
    """
    manifest_path = (
        Path(project_root).resolve()
        / "comfy_installer"
        / "resources"
        / "install_manifest.json"
    )
    if not manifest_path.is_file():
        print(f"[VAST_SOURCES] install_manifest.json 없음: {manifest_path}")
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(
            f"[VAST_SOURCES] install_manifest.json 읽기 실패: {manifest_path}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return {}
    defaults: dict[str, dict[str, Any]] = {}
    for model in manifest.get("models") or []:
        relative = str(model.get("relative_path") or "").strip()
        url = str(model.get("url") or "").strip()
        if not relative.startswith("models/") or not url:
            continue
        parts = relative[len("models/"):].split("/", 1)
        if len(parts) != 2:
            continue
        key = normalize_source_key(parts[0], parts[1])
        if model.get("auth") == "civitai" and "/api/download/models/" in url:
            try:
                version_id = int(
                    url.split("/api/download/models/", 1)[1].split("?")[0].strip("/")
                )
            except ValueError:
                print(f"[VAST_SOURCES] civitai versionId 파싱 실패: {url}")
                continue
            defaults[key] = {
                "source_type": "civitai",
                "civitai_version_id": version_id,
            }
        elif url.startswith("https://huggingface.co/") and "/resolve/" in url:
            rest = url[len("https://huggingface.co/"):]
            repo_id, _, file_path = rest.partition("/resolve/")
            # file_path = '<ref>/<경로>/<파일명>' — 첫 세그먼트가 ref다.
            _, _, filename = file_path.partition("/")
            filename = filename.split("?")[0]
            if repo_id and "/" in repo_id and filename:
                defaults[key] = {
                    "source_type": "hf",
                    "repo_id": repo_id,
                    "hf_filename": filename,
                }
            else:
                print(f"[VAST_SOURCES] HF URL 파싱 실패: {url}")
                defaults[key] = {"source_type": "url", "url": url}
        elif url.startswith(("http://", "https://")):
            defaults[key] = {"source_type": "url", "url": url}
    return defaults


def build_download_plan(
    model_files: list[dict[str, Any]],
    mapping: Mapping[str, Any],
    *,
    manifest_defaults: Mapping[str, Any] | None = None,
    civitai_api_key: str = "",
) -> dict[str, Any]:
    """워크플로우가 참조하는 모델 목록에 매핑을 적용해 준비 계획을 만든다.

    우선순위: 사용자 매핑 → install_manifest 기본값 → 'upload'(이 PC에서 sftp).
    반환: {items: [...], totals: {download_gb, upload_gb}}
    """
    sources = mapping.get("sources") or {}
    defaults = dict(manifest_defaults or {})
    items: list[dict[str, Any]] = []
    download_gb = 0.0
    upload_gb = 0.0
    for model in model_files:
        key = normalize_source_key(model.get("kind", ""), model.get("filename", ""))
        size_gb = float(model.get("size_bytes") or 0) / 1024**3
        source = sources.get(key) or defaults.get(key)
        resolved: dict[str, Any]
        if not source:
            resolved = {"source_type": "upload"}
        elif source.get("source_type") == "civitai":
            resolved = {
                "source_type": "civitai",
                "url": civitai_download_url(
                    int(source["civitai_version_id"]), civitai_api_key
                ),
                "civitai_version_id": source["civitai_version_id"],
            }
        else:
            resolved = dict(source)
        if resolved["source_type"] in {"hf", "civitai", "url"}:
            download_gb += size_gb
        else:
            upload_gb += size_gb
        items.append(
            {
                "key": key,
                "kind": model.get("kind", ""),
                "filename": model.get("filename", ""),
                "size_bytes": int(model.get("size_bytes") or 0),
                "source": resolved,
            }
        )
    return {
        "items": items,
        "totals": {
            "download_gb": round(download_gb, 2),
            "upload_gb": round(upload_gb, 2),
        },
    }
