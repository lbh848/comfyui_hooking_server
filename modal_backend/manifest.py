from __future__ import annotations

import hashlib
import json
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


def _match_user_copy_by_hash(
    project_root: Path,
    filename: str,
    by_hash: Mapping[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """SOYA_USER 의 실제 파일 내용을 해시해 팩 원본과 대응시킨다."""

    if not by_hash:
        return None
    try:
        user_root = _soya_user_root(project_root)
        path = user_root / Path(filename).name
        if not path.is_file():
            return None
        digest = hashlib.sha256(path.read_bytes()).hexdigest().lower()
    except Exception as exc:
        print(
            "[MODAL] 사용자 사본 해시 대응 실패(이름 규칙으로 진행): "
            f"file={filename}, error={type(exc).__name__}: {exc}"
        )
        return None
    return by_hash.get(digest)


def _match_user_copy_by_stem(
    filename: str,
    by_stem: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """"{원본stem}__{릴리스}[_n]" 규칙을 되돌려 팩 원본을 찾는다.

    내용을 고친 개조본은 해시가 달라지므로 이름으로만 되돌릴 수 있다.
    대응되는 원본이 여럿이면 (모호하면) 포기한다 — 틀린 모델 목록보다 낫다.
    """

    stem = Path(str(filename)).stem
    if "__" not in stem:
        return None
    base = stem.rsplit("__", 1)[0]
    candidates = by_stem.get(base) or []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"[MODAL] 이름 규칙 대응이 모호해 건너뜁니다: {filename} → {base}")
    return None


def model_ids_for_workflow_files(
    project_root: str | Path,
    workflow_filenames: list[str] | tuple[str, ...],
) -> list[str]:
    """선택한 SOYA_USER 워크플로우 파일명들이 필요로 하는 매니페스트 model_id 목록.

    cloud_direct 모드에서 쓴다. 로컬 파일을 스캔하는 resolve_workflow_models 와 달리
    **매니페스트만 보고** 필요한 모델을 정한다 — 로컬에 파일이 없어도 되기 때문이다.

    경로: 워크플로우 팩(.soya-pack.json)이 파일명↔바인딩을, 설치 매니페스트가
    바인딩↔model_ids 를 갖고 있어 둘을 이어 붙인다.
    """

    root = Path(project_root)
    pack_path = (
        root / "comfy_workflow_library" / "SOYA_DISTRIBUTION" / "v2" / ".soya-pack.json"
    )
    if not pack_path.is_file():
        print(f"[MODAL] 워크플로우 팩 매니페스트가 없습니다: {pack_path}")
        raise FileNotFoundError(f"워크플로우 팩 매니페스트가 없습니다: {pack_path}")
    pack = json.loads(pack_path.read_text(encoding="utf-8"))

    wanted = {str(name).strip() for name in workflow_filenames if str(name).strip()}
    items = [item for item in pack.get("items", []) if isinstance(item, dict)]
    by_filename = {str(item.get("filename") or ""): item for item in items}
    by_hash = {
        str(item.get("sha256") or "").lower(): item
        for item in items
        if item.get("sha256")
    }
    # 설치기는 사용자 사본을 "{원본stem}__{릴리스}[_n].json" 으로 만든다
    # (comfy_installer/workflow_library.py). 그래서 팩의 원본 파일명과 절대
    # 일치하지 않는다 — 이름만 맞춰보면 새로 설치한 모든 환경에서 모델이
    # 0개로 해석되고, cloud_direct 동기화가 **조용히 아무것도 하지 않는다.**
    by_stem: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        stem = Path(str(item.get("filename") or "")).stem
        if stem:
            by_stem.setdefault(stem, []).append(item)

    bindings: set[str] = set()
    matched: set[str] = set()
    unmatched: list[str] = []
    for name in sorted(wanted):
        item = by_filename.get(name)
        source = "filename"
        if item is None:
            # 사본은 원본과 바이트가 같으므로 해시가 가장 확실하다.
            item, source = _match_user_copy_by_hash(root, name, by_hash), "sha256"
        if item is None:
            # 사용자가 내용을 고친 사본은 해시가 달라진다. 이름 규칙으로 되돌린다.
            item, source = _match_user_copy_by_stem(name, by_stem), "stem"
        if item is None:
            unmatched.append(name)
            continue
        matched.add(name)
        if source != "filename":
            print(
                f"[MODAL] 사용자 사본을 팩 원본과 대응시켰습니다({source}): "
                f"{name} → {item.get('filename')}"
            )
        for binding in item.get("bindings", []):
            bindings.add(str(binding))
    if unmatched:
        # 개인 개조본 등 팩에 없는 워크플로우는 매니페스트로 모델을 알 수 없다.
        print(
            "[MODAL] 팩 매니페스트에 없는 워크플로우는 모델 목록을 확정할 수 없습니다: "
            f"{unmatched}"
        )

    manifest = load_manifest(root)
    releases = manifest.get("workflows", {}).get("release_dependencies", {})
    model_ids: list[str] = []
    seen: set[str] = set()
    for entries in releases.values():
        for entry in entries:
            if str(entry.get("id")) not in bindings:
                continue
            for model_id in entry.get("model_ids", []):
                key = str(model_id)
                if key not in seen:
                    seen.add(key)
                    model_ids.append(key)
    print(
        "[MODAL] cloud_direct 모델 해석: "
        f"workflows={len(matched)}, bindings={len(bindings)}, models={len(model_ids)}"
    )
    return model_ids


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
