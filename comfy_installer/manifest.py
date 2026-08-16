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
_GIT_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


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

    @property
    def latest_workflow_release(self) -> str:
        releases = self.workflows["release_dependencies"]
        return max(releases, key=lambda value: int(value[1:]))

    @property
    def latest_workflow_count(self) -> int:
        return len(
            self.workflows["release_dependencies"][self.latest_workflow_release]
        )

    @property
    def validation_profiles(self) -> dict[str, Any]:
        value = self.data.get("validation_profiles", {})
        return value if isinstance(value, dict) else {}


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
    if data.get("schema_version") != 2:
        raise ManifestError(
            f"지원하지 않는 설치 매니페스트 버전입니다: {data.get('schema_version')!r}"
        )

    comfy = data.get("comfy")
    if not isinstance(comfy, dict):
        raise ManifestError("comfy 항목이 JSON 객체가 아닙니다.")
    _require_string(comfy, "repository", "comfy")
    _require_string(comfy, "ref", "comfy")
    comfy_version = _require_string(comfy, "version", "comfy")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", comfy_version):
        raise ManifestError(
            f"comfy.version 형식이 유효하지 않습니다: {comfy_version!r}"
        )

    python = data.get("python")
    if not isinstance(python, dict):
        raise ManifestError("python 항목이 JSON 객체가 아닙니다.")
    python_version = _require_string(python, "version", "python")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", python_version):
        raise ManifestError(
            f"python.version 형식이 유효하지 않습니다: {python_version!r}"
        )
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
        kind = _require_string(profile, "kind", context)
        if kind not in {"nvidia", "cpu"}:
            raise ManifestError(f"{context}.kind 값이 유효하지 않습니다: {kind!r}")
        packages = profile.get("packages")
        if not isinstance(packages, list) or not all(
            isinstance(item, str) and item for item in packages
        ):
            raise ManifestError(f"{context}.packages가 문자열 배열이 아닙니다.")
        _require_string(profile, "index_url", context)
        if kind == "nvidia":
            minimum_driver = _require_string(
                profile, "minimum_driver_version", context
            )
            if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", minimum_driver):
                raise ManifestError(
                    f"{context}.minimum_driver_version 형식이 유효하지 않습니다: "
                    f"{minimum_driver!r}"
                )
            minimum_compute = _require_string(
                profile, "minimum_compute_capability", context
            )
            if not re.fullmatch(r"[0-9]+\.[0-9]+", minimum_compute):
                raise ManifestError(
                    f"{context}.minimum_compute_capability 형식이 유효하지 "
                    f"않습니다: {minimum_compute!r}"
                )
            torch_cuda = _require_string(profile, "torch_cuda", context)
            if not re.fullmatch(r"[0-9]+\.[0-9]+", torch_cuda):
                raise ManifestError(
                    f"{context}.torch_cuda 형식이 유효하지 않습니다: "
                    f"{torch_cuda!r}"
                )
            _require_string(profile, "triton_package", context)
        sageattention = profile.get("sageattention")
        if kind == "nvidia" and sageattention is None:
            raise ManifestError(
                f"{context}.sageattention이 NVIDIA 프로필에 없습니다."
            )
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
    if not any(profile.get("kind") == "nvidia" for profile in gpu_profiles):
        raise ManifestError("NVIDIA GPU 프로필이 없습니다.")
    if sum(profile.get("kind") == "cpu" for profile in gpu_profiles) != 1:
        raise ManifestError("CPU 프로필은 정확히 하나여야 합니다.")

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
            ref = node.get("ref")
            tracking_branch = node.get("tracking_branch")
            if (ref is None) == (tracking_branch is None):
                raise ManifestError(
                    f"{context}는 ref 또는 tracking_branch 중 하나만 가져야 합니다."
                )
            if tracking_branch is not None:
                branch = _require_string(node, "tracking_branch", context)
                if (
                    not _GIT_BRANCH_RE.fullmatch(branch)
                    or ".." in branch
                    or "//" in branch
                    or branch.endswith(("/", "."))
                    or branch.casefold().endswith(".lock")
                ):
                    raise ManifestError(
                        f"{context}.tracking_branch 형식이 유효하지 않습니다: "
                        f"{branch!r}"
                    )
            else:
                pinned_ref = _require_string(node, "ref", context)
                if not (
                    _GIT_SHA_RE.fullmatch(pinned_ref)
                    or re.fullmatch(r"[A-Za-z0-9._/+:-]+", pinned_ref)
                ):
                    raise ManifestError(
                        f"{context}.ref 형식이 유효하지 않습니다: {pinned_ref!r}"
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
    model_ids: set[str] = set()
    for index, model in enumerate(models):
        context = f"models[{index}]"
        if not isinstance(model, dict):
            raise ManifestError(f"{context}가 JSON 객체가 아닙니다.")
        model_id = _require_string(model, "id", context)
        if model_id in model_ids:
            raise ManifestError(f"모델 ID가 중복됩니다: {model_id}")
        model_ids.add(model_id)
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
    excluded = workflows.get("excluded_filenames", [])
    if not isinstance(excluded, list) or not all(
        isinstance(name, str)
        and bool(name.strip())
        and name == name.strip()
        and "/" not in name
        and "\\" not in name
        and name not in {".", ".."}
        for name in excluded
    ):
        raise ManifestError(
            "workflows.excluded_filenames가 안전한 파일명 문자열 배열이 아닙니다."
        )
    if len({name.casefold() for name in excluded}) != len(excluded):
        raise ManifestError("workflows.excluded_filenames에 중복 파일명이 있습니다.")

    required_bindings = workflows.get("required_bindings")
    if not isinstance(required_bindings, list) or not required_bindings:
        raise ManifestError("workflows.required_bindings가 비어 있습니다.")
    if not all(
        isinstance(binding, str) and binding.strip()
        for binding in required_bindings
    ):
        raise ManifestError("workflows.required_bindings가 문자열 배열이 아닙니다.")
    if len(set(required_bindings)) != len(required_bindings):
        raise ManifestError("workflows.required_bindings에 중복 값이 있습니다.")

    optional_bindings = workflows.get("optional_bindings", [])
    if not isinstance(optional_bindings, list) or not all(
        isinstance(binding, str) and binding.strip()
        for binding in optional_bindings
    ):
        raise ManifestError("workflows.optional_bindings가 문자열 배열이 아닙니다.")
    if len(set(optional_bindings)) != len(optional_bindings):
        raise ManifestError("workflows.optional_bindings에 중복 값이 있습니다.")
    required_binding_set = set(required_bindings)
    optional_binding_set = set(optional_bindings)
    overlapping_bindings = required_binding_set.intersection(optional_binding_set)
    if overlapping_bindings:
        raise ManifestError(
            "필수/선택 워크플로 바인딩이 중복됩니다: "
            f"{sorted(overlapping_bindings)}"
        )
    known_binding_set = required_binding_set.union(optional_binding_set)

    release_dependencies = workflows.get("release_dependencies")
    if not isinstance(release_dependencies, dict) or not release_dependencies:
        raise ManifestError("workflows.release_dependencies가 비어 있습니다.")
    for release_version in release_dependencies:
        if not isinstance(release_version, str) or not re.fullmatch(
            r"v[1-9][0-9]*", release_version
        ):
            raise ManifestError(
                f"워크플로우 배포 버전 형식이 유효하지 않습니다: {release_version!r}"
            )
    latest_release_version = max(
        release_dependencies,
        key=lambda value: int(value[1:]),
    )
    for release_version, entries in release_dependencies.items():
        context = f"workflows.release_dependencies.{release_version}"
        if not isinstance(entries, list) or not entries:
            raise ManifestError(
                f"{context} 항목이 비어 있거나 배열이 아닙니다."
            )
        entry_ids: set[str] = set()
        covered_bindings: set[str] = set()
        for index, entry in enumerate(entries):
            entry_context = f"{context}[{index}]"
            if not isinstance(entry, dict):
                raise ManifestError(f"{entry_context}가 JSON 객체가 아닙니다.")
            entry_id = _require_string(entry, "id", entry_context)
            if entry_id in entry_ids:
                raise ManifestError(
                    f"{context} 워크플로우 ID가 중복됩니다: {entry_id}"
                )
            entry_ids.add(entry_id)
            bindings = entry.get("bindings")
            if not isinstance(bindings, list) or not bindings or not all(
                isinstance(binding, str) and binding.strip()
                for binding in bindings
            ):
                raise ManifestError(
                    f"{entry_context}.bindings가 비어 있거나 문자열 배열이 아닙니다."
                )
            if len(set(bindings)) != len(bindings):
                raise ManifestError(
                    f"{entry_context}.bindings에 중복 값이 있습니다."
                )
            unknown_bindings = set(bindings) - known_binding_set
            if unknown_bindings:
                raise ManifestError(
                    f"{entry_context}.bindings에 알 수 없는 설정 키가 있습니다: "
                    f"{sorted(unknown_bindings)}"
                )
            duplicate_bindings = covered_bindings.intersection(bindings)
            if duplicate_bindings:
                raise ManifestError(
                    f"{context}에서 설정 키가 여러 워크플로우에 중복됩니다: "
                    f"{sorted(duplicate_bindings)}"
                )
            covered_bindings.update(bindings)
            dependency_ids = entry.get("model_ids")
            if not isinstance(dependency_ids, list) or not all(
                isinstance(model_id, str) and model_id.strip()
                for model_id in dependency_ids
            ):
                raise ManifestError(
                    f"{entry_context}.model_ids가 문자열 배열이 아닙니다."
                )
            if len(set(dependency_ids)) != len(dependency_ids):
                raise ManifestError(
                    f"{entry_context}.model_ids에 중복 값이 있습니다."
                )
            unknown_models = set(dependency_ids) - model_ids
            if unknown_models:
                raise ManifestError(
                    f"{entry_context}.model_ids에 등록되지 않은 모델이 있습니다: "
                    f"{sorted(unknown_models)}"
                )
        if (
            release_version == latest_release_version
            and covered_bindings != known_binding_set
        ):
            raise ManifestError(
                f"최신 배포 {context}가 모든 설정 바인딩을 포함하지 않습니다: "
                f"missing={sorted(known_binding_set - covered_bindings)}"
            )

    validation_profiles = data.get("validation_profiles")
    if not isinstance(validation_profiles, dict):
        raise ManifestError("validation_profiles가 JSON 객체가 아닙니다.")
    h3_profile = validation_profiles.get("minimax_h3")
    if not isinstance(h3_profile, dict):
        raise ManifestError("validation_profiles.minimax_h3가 없습니다.")
    h3_bindings = h3_profile.get("workflow_bindings")
    if not isinstance(h3_bindings, list) or not h3_bindings or not all(
        isinstance(binding, str) and binding.strip()
        for binding in h3_bindings
    ):
        raise ManifestError(
            "validation_profiles.minimax_h3.workflow_bindings가 "
            "비어 있거나 문자열 배열이 아닙니다."
        )
    h3_binding_set = set(h3_bindings)
    if len(h3_binding_set) != len(h3_bindings):
        raise ManifestError(
            "validation_profiles.minimax_h3.workflow_bindings에 중복 값이 있습니다."
        )
    unknown_h3_bindings = h3_binding_set - known_binding_set
    if unknown_h3_bindings:
        raise ManifestError(
            "validation_profiles.minimax_h3.workflow_bindings에 등록되지 않은 "
            f"설정 키가 있습니다: {sorted(unknown_h3_bindings)}"
        )

    fast_h3_bindings = h3_profile.get("fast_workflow_bindings", [])
    if not isinstance(fast_h3_bindings, list) or not all(
        isinstance(binding, str) and binding.strip()
        for binding in fast_h3_bindings
    ):
        raise ManifestError(
            "validation_profiles.minimax_h3.fast_workflow_bindings가 "
            "문자열 배열이 아닙니다."
        )
    fast_h3_binding_set = set(fast_h3_bindings)
    if len(fast_h3_binding_set) != len(fast_h3_bindings):
        raise ManifestError(
            "validation_profiles.minimax_h3.fast_workflow_bindings에 "
            "중복 값이 있습니다."
        )
    if not fast_h3_binding_set.issubset(h3_binding_set):
        raise ManifestError(
            "validation_profiles.minimax_h3.fast_workflow_bindings가 "
            "workflow_bindings의 일부가 아닙니다."
        )

    h3_model_ids = h3_profile.get("model_ids")
    if not isinstance(h3_model_ids, list) or not h3_model_ids or not all(
        isinstance(model_id, str) and model_id.strip()
        for model_id in h3_model_ids
    ):
        raise ManifestError(
            "validation_profiles.minimax_h3.model_ids가 비어 있거나 "
            "문자열 배열이 아닙니다."
        )
    h3_model_id_set = set(h3_model_ids)
    if len(h3_model_id_set) != len(h3_model_ids):
        raise ManifestError(
            "validation_profiles.minimax_h3.model_ids에 중복 값이 있습니다."
        )
    unknown_h3_models = h3_model_id_set - model_ids
    if unknown_h3_models:
        raise ManifestError(
            "validation_profiles.minimax_h3.model_ids에 등록되지 않은 모델이 "
            f"있습니다: {sorted(unknown_h3_models)}"
        )

    defaults_to_validate = ["defaults"]
    if fast_h3_bindings:
        defaults_to_validate.append("fast_defaults")
    for defaults_key in defaults_to_validate:
        defaults = h3_profile.get(defaults_key)
        context = f"validation_profiles.minimax_h3.{defaults_key}"
        if not isinstance(defaults, dict):
            raise ManifestError(
                f"{context}가 JSON 객체가 아닙니다."
            )
        for key in ("width", "height", "steps"):
            value = defaults.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ManifestError(
                    f"{context}.{key}가 양의 정수가 아닙니다: {value!r}"
                )


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
    except ManifestError as exc:
        print(
            "[COMFY_INSTALL][MANIFEST] 설치 매니페스트 검증 실패: "
            f"path={manifest_path}, error={exc}"
        )
        traceback.print_exc()
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][MANIFEST] 설치 매니페스트 로드 실패: "
            f"path={manifest_path}, error={exc}"
        )
        traceback.print_exc()
        raise ManifestError(f"설치 매니페스트 로드 실패: {exc}") from exc
