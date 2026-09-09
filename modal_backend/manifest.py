from __future__ import annotations

import hashlib
import json
import re
import traceback
from pathlib import Path
from typing import Any, Mapping


def load_manifest(project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root) / "comfy_installer" / "resources" / "install_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def workflow_catalog(project_root: str | Path) -> list[dict[str, Any]]:
    manifest = load_manifest(project_root)
    workflows = manifest.get("workflows", {})
    releases = workflows.get("items")
    if not isinstance(releases, list):
        legacy = workflows.get("release_dependencies", {})
        releases = legacy.get("v1", []) if isinstance(legacy, dict) else []
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
    user_root = _soya_user_root(project_root)
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


def _workflow_model_catalog(project_root, workflow_filenames):
    """Resolve installed copies against their own release and embedded manifest."""
    from comfy_installer.model_scope import binding_model_ids
    from modal_backend.workflow_assets import _workflow_string_values

    root = Path(project_root)
    manifest = load_manifest(root)
    base_models = {str(m["id"]): m for m in manifest.get("models", [])}
    packs = []
    distribution = root / "comfy_workflow_library" / "SOYA_DISTRIBUTION"
    for pack_path in sorted(distribution.glob("*/.soya-pack.json")):
        try:
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            packs.append((pack_path.parent.name, pack))
        except Exception as exc:
            print(f"[MODAL] Cannot read workflow pack: path={pack_path}, error={exc}")
            traceback.print_exc()
            raise

    selected = {}
    for filename in dict.fromkeys(workflow_filenames):
        name = Path(str(filename)).name
        user_file = _soya_user_root(root) / name
        digest = ""
        workflow = {}
        if user_file.is_file():
            user_file = _require_user_workflow(root, name, str(user_file))
            payload = user_file.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            try:
                workflow = json.loads(payload)
            except Exception as exc:
                print(f"[MODAL] Cannot parse selected workflow: file={name}, error={exc}")
                traceback.print_exc()
                raise
        matches = []
        for release, pack in packs:
            for item in pack.get("items", []):
                original = str(item.get("filename") or "")
                copy_stem = re.escape(Path(original).stem + "__" + release)
                named_copy = bool(re.fullmatch(copy_stem + r"(?:_[0-9]+)?", Path(name).stem))
                same_hash = bool(digest and digest == str(item.get("sha256") or "").lower())
                rank = (4 if named_copy else 0) + (2 if same_hash else 0) + (1 if name == original else 0)
                if rank:
                    matches.append((rank, pack, item))
        catalog = dict(base_models)
        model_ids = set()
        if matches:
            best = max(rank for rank, _, _ in matches)
            for rank, pack, item in matches:
                if rank != best:
                    continue
                embedded = pack.get("install_manifest") or manifest
                catalog.update({str(m["id"]): m for m in embedded.get("models", [])})
                ids = item.get("model_ids")
                if ids is None:
                    ids = binding_model_ids(embedded.get("workflows", {}), item.get("bindings", []))
                model_ids.update(str(value) for value in ids)
        else:
            print(f"[MODAL] No pack entry for {name}; checking actual model references.")
        # Edited copies may reference additional models. Match actual file references,
        # using the same normalized path vocabulary as the local upload resolver.
        references = {v.replace(chr(92), "/").casefold() for v in _workflow_string_values(workflow)}
        for model_id, model in catalog.items():
            relative = str(model.get("relative_path") or "").replace(chr(92), "/")
            parts = Path(relative).parts
            aliases = {relative.casefold(), Path(relative).name.casefold()}
            if parts and parts[0] == "models":
                aliases.add("/".join(parts[1:]).casefold())
                aliases.add("/".join(parts[2:]).casefold())
            if relative and references.intersection(aliases):
                model_ids.add(model_id)
        for model_id in sorted(model_ids):
            if model_id not in catalog:
                print(f"[MODAL] Missing model download metadata: workflow={name}, id={model_id}")
                raise ValueError(f"Model download metadata is missing: {model_id} ({name})")
            selected[model_id] = catalog[model_id]
    print(f"[MODAL] cloud_direct: workflows={len(workflow_filenames)}, models={len(selected)}")
    return list(selected.values())


def model_entries_for_workflow_files(project_root, workflow_filenames):
    return _workflow_model_catalog(project_root, workflow_filenames)


def model_ids_for_workflow_files(project_root, workflow_filenames):
    return [str(entry["id"]) for entry in _workflow_model_catalog(project_root, workflow_filenames)]


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


def _soya_user_root(project_root: str | Path) -> Path:
    # Modal 원격 런타임은 ``modal_backend``만 마운트하며 로컬 설치기 패키지는
    # 포함하지 않는다. 이 의존성은 로컬 워크플로우를 실제로 탐색할 때만 필요하므로
    # 모듈 import 단계에서는 불러오지 않는다.
    try:
        from comfy_installer.workflow_library import embedded_workflow_base_dir
    except Exception as exc:
        print(
            "[MODAL] 로컬 워크플로우 경로 도우미 import 실패: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    return embedded_workflow_base_dir(Path(project_root).resolve() / "comfy")


def list_soya_user_workflows(project_root: str | Path) -> list[dict[str, Any]]:
    """``SOYA_USER`` 폴더에 실제 존재하는 ``.json`` 워크플로우를 파일명 기준으로 나열한다.

    config.json 바인딩에 의존하지 않고 디스크의 실제 파일만 반환한다. 이것이 Modal
    동기화 카탈로그의 유일한 소스다. 각 항목은 ``{"name": 파일명(.json 포함),
    "source_path": resolve된 절대경로}`` 형태다.

    심볼릭 링크가 SOYA_USER 밖을 가리키는 경우 ``resolve()``가 대상을 따라가므로
    ``user_root.relative_to`` 검증으로 걸러진다. 일반 파일이 아닌 항목(소켓·장치
    등)도 제외한다.
    """
    user_root = _soya_user_root(project_root)
    if not user_root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(user_root.glob("*.json"), key=lambda p: p.name.casefold()):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(user_root)
        except (ValueError, OSError) as exc:
            print(
                "[MODAL] SOYA_USER 내 비정상 경로 제외(외부 탈출 가능성): "
                f"path={path}, error={type(exc).__name__}: {exc}"
            )
            continue
        entries.append({"name": path.name, "source_path": str(resolved)})
    return entries


def _enforce_filename_only(name: str) -> str:
    """동기화 선택 키는 '파일명만' 허용한다. 경로/구분자/절대경로/``..``은 거부한다.

    ``Path(name).name``과 원문이 다르면 구분자나 절대경로가 들어있다는 뜻이므로
    거부한다. 이 강제는 ``_require_user_workflow``의 방어와 별개로 진입 단에서
    걸러내는 이중 방어다.
    """
    raw = str(name or "")
    if not raw or raw in {".", ".."}:
        raise ValueError(f"워크플로우 이름이 비어있거나 잘못되었습니다: {name!r}")
    base = Path(raw).name
    if base != raw:
        raise ValueError(
            f"워크플로우 이름은 파일명만 허용합니다(경로/구분자/절대경로 불가): {name!r}"
        )
    if base in {".", ".."} or not base.rstrip():
        raise ValueError(f"워크플로우 이름으로 사용할 수 없습니다: {name!r}")
    return base


def plan_from_soya_user_names(
    project_root: str | Path,
    selected_names: list[str],
) -> dict[str, Any]:
    """선택된 SOYA_USER 파일명들로 동기화 plan을 만든다.

    ``selected_names``는 파일명(확장자 포함, 예: ``foo.json``)만 받는다. 경로·구분자
    ·``..``·절대경로는 ``_enforce_filename_only``에서 거부한다. 각 파일은
    ``_require_user_workflow``로 (1) 실존 (2) SOYA_USER 하위 (3) ``resolve()`` 후
    심볼릭 링크 외부 탈출 차단 을 모두 검증한다.

    반환 shape는 기존 ``selected_install_plan``과 동일해 ``_run_install``/
    ``_run_saved_workflow``이 그대로 동작한다. ``id``는 항상 파일명(``foo.json``)
    으로 고정한다.
    """
    if not selected_names:
        raise ValueError("동기화할 워크플로우를 하나 이상 선택하세요.")
    user_root = _soya_user_root(project_root)
    workflow_files: list[dict[str, str]] = []
    for raw in dict.fromkeys(selected_names):
        name = _enforce_filename_only(raw)
        candidate = str(user_root / name)
        path = _require_user_workflow(project_root, name, candidate)
        workflow_files.append(
            {
                "id": path.name,
                "binding": "",
                "source_path": str(path),
                "remote_name": path.name,
            }
        )
    return {
        "workflow_ids": [item["id"] for item in workflow_files],
        "workflow_files": workflow_files,
        "model_count": 0,
        "size_bytes": 0,
        "size_gib": 0.0,
    }
