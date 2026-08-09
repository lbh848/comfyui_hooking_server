from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_manifest(project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root) / "comfy_installer" / "resources" / "install_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def workflow_catalog(project_root: str | Path) -> list[dict[str, Any]]:
    manifest = load_manifest(project_root)
    models = {item["id"]: item for item in manifest.get("models", [])}
    releases = manifest.get("workflows", {}).get("release_dependencies", {}).get("v1", [])
    result: list[dict[str, Any]] = []
    for release in releases:
        model_ids = list(dict.fromkeys(release.get("model_ids", [])))
        missing = [model_id for model_id in model_ids if model_id not in models]
        if missing:
            raise ValueError(
                f"워크플로우 {release.get('id')}의 모델 명세가 없습니다: {', '.join(missing)}"
            )
        size_bytes = sum(int(models[model_id].get("size") or 0) for model_id in model_ids)
        result.append(
            {
                "id": release["id"],
                "bindings": list(release.get("bindings", [])),
                "model_ids": model_ids,
                "model_count": len(model_ids),
                "size_bytes": size_bytes,
                "size_gib": round(size_bytes / 1024**3, 2),
            }
        )
    return result


def selected_install_plan(
    project_root: str | Path,
    selected_ids: list[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = {item["id"]: item for item in workflow_catalog(project_root)}
    unknown = sorted(set(selected_ids) - set(catalog))
    if unknown:
        raise ValueError(f"알 수 없는 워크플로우 선택입니다: {', '.join(unknown)}")
    if not selected_ids:
        raise ValueError("설치할 워크플로우를 하나 이상 선택하세요.")

    workflow_files: list[dict[str, str]] = []
    model_ids: list[str] = []
    for workflow_id in dict.fromkeys(selected_ids):
        entry = catalog[workflow_id]
        source_path = ""
        binding_used = ""
        for binding in entry["bindings"]:
            value: Any = config
            for part in binding.split("."):
                if not isinstance(value, Mapping):
                    value = None
                    break
                value = value.get(part)
            candidate = str(value or "").strip()
            if candidate and Path(candidate).is_file():
                source_path = str(Path(candidate).resolve())
                binding_used = binding
                break
        if not source_path:
            raise FileNotFoundError(
                f"{workflow_id}에 연결된 로컬 워크플로우 파일이 없습니다. "
                "먼저 워크플로우 팩을 풀고 설정 경로를 저장하세요."
            )
        workflow_files.append(
            {
                "id": workflow_id,
                "binding": binding_used,
                "source_path": source_path,
                "remote_name": f"{workflow_id.replace('.', '_')}-{Path(source_path).name}",
            }
        )
        model_ids.extend(entry["model_ids"])

    unique_model_ids = list(dict.fromkeys(model_ids))
    manifest = load_manifest(project_root)
    models = {item["id"]: item for item in manifest.get("models", [])}
    size_bytes = sum(int(models[model_id].get("size") or 0) for model_id in unique_model_ids)
    return {
        "workflow_ids": list(dict.fromkeys(selected_ids)),
        "workflow_files": workflow_files,
        "model_ids": unique_model_ids,
        "model_count": len(unique_model_ids),
        "size_bytes": size_bytes,
        "size_gib": round(size_bytes / 1024**3, 2),
    }
