from __future__ import annotations

import hashlib
import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(RuntimeError):
    """설치 매니페스트 형식 또는 무결성 오류."""


@dataclass(frozen=True)
class InstallManifest:
    source_path: Path
    data: dict[str, Any]
    sha256: str

    @property
    def comfy(self) -> dict[str, Any]:
        return self.data["comfy"]

    @property
    def python(self) -> dict[str, Any]:
        return self.data["python"]

    @property
    def custom_nodes(self) -> list[dict[str, Any]]:
        return self.data["custom_nodes"]

    @property
    def models(self) -> list[dict[str, Any]]:
        return self.data["models"]

    @property
    def workflows(self) -> dict[str, Any]:
        return self.data["workflows"]


def _is_safe_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(normalized) and not path.is_absolute() and ".." not in path.parts


def _require_string(mapping: dict, key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context}.{key} 값이 비어 있거나 문자열이 아닙니다.")
    return value.strip()


def _validate_manifest(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ManifestError("설치 매니페스트 최상위 값은 JSON 객체여야 합니다.")
    if data.get("schema_version") != 1:
        raise ManifestError(
            f"지원하지 않는 설치 매니페스트 버전입니다: {data.get('schema_version')!r}"
        )

    comfy = data.get("comfy")
    if not isinstance(comfy, dict):
        raise ManifestError("comfy 항목이 JSON 객체가 아닙니다.")
    _require_string(comfy, "repository", "comfy")
    _require_string(comfy, "ref", "comfy")
    if comfy.get("version") != "0.20.1":
        raise ManifestError("최초 배포 ComfyUI 버전은 0.20.1이어야 합니다.")

    python = data.get("python")
    if not isinstance(python, dict):
        raise ManifestError("python 항목이 JSON 객체가 아닙니다.")
    if _require_string(python, "version", "python") != "3.12.11":
        raise ManifestError("ComfyUI Python 버전은 3.12.11이어야 합니다.")
    compatibility_packages = python.get("compatibility_packages")
    if not isinstance(compatibility_packages, list) or not all(
        isinstance(item, str) and "==" in item
        for item in compatibility_packages
    ):
        raise ManifestError(
            "python.compatibility_packages가 고정 버전 문자열 배열이 아닙니다."
        )
    gpu_profiles = python.get("gpu_profiles")
    if not isinstance(gpu_profiles, list) or not gpu_profiles:
        raise ManifestError("python.gpu_profiles가 비어 있습니다.")
    profile_ids: set[str] = set()
    for index, profile in enumerate(gpu_profiles):
        context = f"python.gpu_profiles[{index}]"
        if not isinstance(profile, dict):
            raise ManifestError(f"{context}가 JSON 객체가 아닙니다.")
        profile_id = _require_string(profile, "id", context)
        if profile_id in profile_ids:
            raise ManifestError(f"GPU 프로필 ID가 중복됩니다: {profile_id}")
        profile_ids.add(profile_id)
        packages = profile.get("packages")
        if not isinstance(packages, list) or not all(
            isinstance(item, str) and item for item in packages
        ):
            raise ManifestError(f"{context}.packages가 문자열 배열이 아닙니다.")
        sageattention = profile.get("sageattention")
        if sageattention is not None:
            if not isinstance(sageattention, dict):
                raise ManifestError(f"{context}.sageattention이 객체가 아닙니다.")
            _require_string(sageattention, "url", f"{context}.sageattention")
            sage_hash = _require_string(
                sageattention, "sha256", f"{context}.sageattention"
            ).lower()
            if not _SHA256_RE.fullmatch(sage_hash):
                raise ManifestError(
                    f"{context}.sageattention.sha256 형식이 유효하지 않습니다."
                )
            sage_size = sageattention.get("size")
            if not isinstance(sage_size, int) or sage_size <= 0:
                raise ManifestError(
                    f"{context}.sageattention.size가 양의 정수가 아닙니다."
                )

    custom_nodes = data.get("custom_nodes")
    if not isinstance(custom_nodes, list) or not custom_nodes:
        raise ManifestError("custom_nodes가 비어 있습니다.")
    node_names: set[str] = set()
    for index, node in enumerate(custom_nodes):
        context = f"custom_nodes[{index}]"
        if not isinstance(node, dict):
            raise ManifestError(f"{context}가 JSON 객체가 아닙니다.")
        name = _require_string(node, "name", context)
        if name.casefold() in node_names:
            raise ManifestError(f"커스텀 노드 이름이 중복됩니다: {name}")
        node_names.add(name.casefold())
        source_type = node.get("source_type")
        if source_type == "git":
            _require_string(node, "repository", context)
            ref = _require_string(node, "ref", context)
            if not (
                _GIT_SHA_RE.fullmatch(ref)
                or re.fullmatch(r"[A-Za-z0-9._/+:-]+", ref)
            ):
                raise ManifestError(
                    f"{context}.ref 형식이 유효하지 않습니다: {ref!r}"
                )
        elif source_type == "archive":
            _require_string(node, "url", context)
            sha256 = _require_string(node, "sha256", context).lower()
            if not _SHA256_RE.fullmatch(sha256):
                raise ManifestError(f"{context}.sha256 형식이 유효하지 않습니다.")
            size = node.get("size")
            if not isinstance(size, int) or size <= 0:
                raise ManifestError(f"{context}.size가 양의 정수가 아닙니다.")
        else:
            raise ManifestError(
                f"{context}.source_type은 git 또는 archive여야 합니다."
            )

    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ManifestError("models가 비어 있습니다.")
    model_paths: set[str] = set()
    for index, model in enumerate(models):
        context = f"models[{index}]"
        if not isinstance(model, dict):
            raise ManifestError(f"{context}가 JSON 객체가 아닙니다.")
        _require_string(model, "id", context)
        _require_string(model, "url", context)
        relative_path = _require_string(model, "relative_path", context)
        if not _is_safe_relative_path(relative_path):
            raise ManifestError(
                f"{context}.relative_path가 안전한 상대 경로가 아닙니다: "
                f"{relative_path!r}"
            )
        folded_path = relative_path.replace("\\", "/").casefold()
        if folded_path in model_paths:
            raise ManifestError(f"모델 설치 경로가 중복됩니다: {relative_path}")
        model_paths.add(folded_path)
        sha256 = _require_string(model, "sha256", context).lower()
        if not _SHA256_RE.fullmatch(sha256):
            raise ManifestError(f"{context}.sha256 형식이 유효하지 않습니다.")
        size = model.get("size")
        if not isinstance(size, int) or size <= 0:
            raise ManifestError(f"{context}.size가 양의 정수가 아닙니다.")
        auth = model.get("auth", "none")
        if auth not in ("none", "civitai"):
            raise ManifestError(f"{context}.auth 값이 유효하지 않습니다: {auth!r}")

    workflows = data.get("workflows")
    if not isinstance(workflows, dict):
        raise ManifestError("workflows 항목이 JSON 객체가 아닙니다.")
    expected_count = workflows.get("expected_count")
    if expected_count != 17:
        raise ManifestError(
            f"최초 배포 워크플로우 수는 17이어야 합니다: {expected_count!r}"
        )
    excluded = workflows.get("excluded_filenames")
    if not isinstance(excluded, list) or "캐릭터복장추적_v1.json" not in excluded:
        raise ManifestError("제외 워크플로우 목록에 캐릭터복장추적_v1.json이 없습니다.")


def load_install_manifest(
    path: str | os.PathLike[str] | None = None,
) -> InstallManifest:
    manifest_path = (
        Path(path).resolve()
        if path is not None
        else Path(__file__).resolve().parent / "resources" / "install_manifest.json"
    )
    try:
        raw = manifest_path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        _validate_manifest(data)
        return InstallManifest(
            source_path=manifest_path,
            data=data,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    except ManifestError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][MANIFEST] 설치 매니페스트 로드 실패: "
            f"path={manifest_path}, error={exc}"
        )
        traceback.print_exc()
        raise ManifestError(f"설치 매니페스트 로드 실패: {exc}") from exc
