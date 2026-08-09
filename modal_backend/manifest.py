from __future__ import annotations

import json
from pathlib import Path
import traceback
from typing import Any, Mapping

from comfy_installer.workflow_library import embedded_workflow_base_dir


def load_manifest(project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root) / "comfy_installer" / "resources" / "install_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def workflow_catalog(project_root: str | Path) -> list[dict[str, Any]]:
    manifest = load_manifest(project_root)
    releases = manifest.get("workflows", {}).get("release_dependencies", {}).get("v1", [])
    result: list[dict[str, Any]] = []
    for release in releases:
        result.append(
            {
                "id": release["id"],
                "bindings": list(release.get("bindings", [])),
                "model_count": 0,
                "size_bytes": 0,
                "size_gib": 0.0,
            }
        )
    return result


def _require_user_workflow(
    project_root: str | Path,
    workflow_id: str,
    candidate: str,
) -> Path:
    user_root = embedded_workflow_base_dir(Path(project_root).resolve() / "comfy")
    path = Path(candidate).resolve()
    if not path.is_file():
        print(
            "[MODAL] 사용자 워크플로우 파일 없음: "
            f"workflow_id={workflow_id}, path={path}"
        )
        raise FileNotFoundError(f"{workflow_id}에 연결된 워크플로우 파일이 없습니다: {path}")
    try:
        path.relative_to(user_root)
    except ValueError as exc:
        print(
            "[MODAL] SOYA_USER 밖의 워크플로우 거부: "
            f"workflow_id={workflow_id}, path={path}, user_root={user_root}"
        )
        traceback.print_exc()
        raise ValueError(
            f"{workflow_id}은(는) 설치된 사용자 워크플로우가 아닙니다. "
            f"Modal은 {user_root} 안의 워크플로우만 사용할 수 있습니다."
        ) from exc
    return path


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
            if candidate:
                try:
                    source_path = str(
                        _require_user_workflow(project_root, workflow_id, candidate)
                    )
                    binding_used = binding
                    break
                except FileNotFoundError:
                    continue
        if not source_path:
            print(
                "[MODAL] 설치된 사용자 워크플로우 바인딩 없음: "
                f"workflow_id={workflow_id}, bindings={entry['bindings']}"
            )
            raise FileNotFoundError(
                f"{workflow_id}에 연결된 SOYA_USER 워크플로우 파일이 없습니다. "
                "먼저 로컬 설치기에서 워크플로우를 설치하고 설정 경로를 저장하세요."
            )
        workflow_files.append(
            {
                "id": workflow_id,
                "binding": binding_used,
                "source_path": source_path,
                "remote_name": f"{workflow_id.replace('.', '_')}-{Path(source_path).name}",
            }
        )
    return {
        "workflow_ids": list(dict.fromkeys(selected_ids)),
        "workflow_files": workflow_files,
        "model_count": 0,
        "size_bytes": 0,
        "size_gib": 0.0,
    }
