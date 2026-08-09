from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import traceback
from typing import Any, Iterable, Mapping


_LORA_INPUT_FIELDS = {"lora_name", "lora"}
_IMAGE_INPUT_FIELDS = {"image"}


def build_local_model_index(comfy_root: str | Path) -> dict[str, Any]:
    """로컬 Comfy models 폴더의 실제 파일을 참조명으로 조회할 색인으로 만든다."""

    models_root = (Path(comfy_root).resolve() / "models").resolve()
    if not models_root.is_dir():
        print(f"[MODAL_SYNC] 로컬 Comfy models 폴더가 없습니다: {models_root}")
        raise FileNotFoundError(f"로컬 Comfy models 폴더가 없습니다: {models_root}")

    files: list[dict[str, Any]] = []
    lookup: dict[str, list[dict[str, Any]]] = {}
    try:
        candidates = sorted(models_root.rglob("*"), key=lambda path: str(path).casefold())
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(models_root)
            except ValueError as exc:
                print(
                    "[MODAL_SYNC] models 폴더 밖을 가리키는 모델 파일 거부: "
                    f"candidate={candidate}, resolved={resolved}, root={models_root}"
                )
                raise ValueError(
                    f"로컬 Comfy models 폴더 밖의 파일은 Modal에 전송할 수 없습니다: {candidate}"
                ) from exc

            relative_posix = relative.as_posix()
            is_lora = bool(relative.parts) and relative.parts[0].casefold() == "loras"
            if is_lora:
                if len(relative.parts) < 2:
                    print(f"[MODAL_SYNC] LoRA 상대 경로가 비어 있어 제외: {resolved}")
                    continue
                remote_path = PurePosixPath(*relative.parts[1:]).as_posix()
            else:
                remote_path = relative_posix
            stat = resolved.stat()
            entry = {
                "source_path": str(resolved),
                "remote_path": remote_path,
                "local_relative_path": relative_posix,
                "kind": "lora" if is_lora else "model",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            files.append(entry)

            keys = {
                relative_posix,
                f"models/{relative_posix}",
                relative.name,
                str(resolved).replace("\\", "/"),
            }
            if len(relative.parts) > 1:
                keys.add(PurePosixPath(*relative.parts[1:]).as_posix())
            for key in keys:
                normalized = key.strip().replace("\\", "/").casefold()
                if normalized:
                    lookup.setdefault(normalized, []).append(entry)
    except Exception as exc:
        print(
            "[MODAL_SYNC] 로컬 모델 색인 생성 실패: "
            f"root={models_root}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise

    if not files:
        print(f"[MODAL_SYNC] 로컬 Comfy models 폴더에 파일이 없습니다: {models_root}")
    return {"root": str(models_root), "files": files, "lookup": lookup}


def _embedded_json_values(text: str) -> Iterable[Any]:
    """프롬프트 문자열 등에 포함된 JSON object/array를 손실 없이 찾아낸다."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, (Mapping, list)):
            yield value


def _workflow_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            yield stripped
            for embedded in _embedded_json_values(stripped):
                yield from _workflow_string_values(embedded)
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _workflow_string_values(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _workflow_string_values(child)


def resolve_workflow_model_files(
    workflows: Iterable[Mapping[str, Any]],
    model_index: Mapping[str, Any],
    *,
    hash_cache: dict[str, tuple[int, int, str]] | None = None,
    include_hashes: bool = True,
) -> dict[str, Any]:
    """워크플로우가 실제로 언급하는 로컬 모델만 Modal 동기화 명세로 만든다."""

    lookup = model_index.get("lookup")
    if not isinstance(lookup, Mapping):
        print(f"[MODAL_SYNC] 로컬 모델 색인 형식 오류: {type(lookup).__name__}")
        raise TypeError("로컬 모델 색인의 lookup이 올바르지 않습니다.")

    selected: dict[str, dict[str, Any]] = {}
    for workflow in workflows:
        if not isinstance(workflow, Mapping) or not workflow:
            print(
                "[MODAL_SYNC] 모델 참조를 찾을 워크플로우 형식 오류: "
                f"type={type(workflow).__name__}"
            )
            raise ValueError("Modal 모델 동기화에 사용할 워크플로우가 비어 있습니다.")
        for raw_value in _workflow_string_values(workflow):
            normalized = raw_value.replace("\\", "/").casefold()
            for entry in lookup.get(normalized, []):
                selected[str(entry["source_path"])] = dict(entry)

    next_cache: dict[str, tuple[int, int, str]] = {}
    model_files: list[dict[str, Any]] = []
    lora_files: list[dict[str, Any]] = []
    size_bytes = 0
    for source_path in sorted(selected, key=str.casefold):
        entry = selected[source_path]
        size = int(entry["size"])
        mtime_ns = int(entry["mtime_ns"])
        payload = {
            "source_path": source_path,
            "remote_path": str(entry["remote_path"]),
            "size": size,
        }
        if include_hashes:
            cached = hash_cache.get(source_path) if hash_cache is not None else None
            if cached is not None and cached[:2] == (size, mtime_ns):
                digest = cached[2]
            else:
                digest = _sha256(Path(source_path))
            payload["sha256"] = digest
            next_cache[source_path] = (size, mtime_ns, digest)
        size_bytes += size
        if entry["kind"] == "lora":
            lora_files.append(payload)
        else:
            model_files.append(payload)

    if hash_cache is not None and include_hashes:
        hash_cache.update(next_cache)
    return {
        "model_files": model_files,
        "lora_files": lora_files,
        "model_count": len(model_files) + len(lora_files),
        "size_bytes": size_bytes,
        "size_gib": round(size_bytes / 1024**3, 2),
    }


def _workflow_paths(workflow: Mapping[str, Any], field_names: set[str]) -> list[str]:
    result: list[str] = []
    for node in workflow.values():
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for field, value in inputs.items():
            normalized_field = str(field).lower()
            if normalized_field not in field_names and not any(
                normalized_field.startswith(f"{name}_") for name in field_names
            ):
                continue
            if isinstance(value, str) and value.strip():
                result.append(value.strip().replace("\\", "/"))
    return list(dict.fromkeys(result))


def _safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"안전하지 않은 {label} 상대 경로입니다: {value!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_lora_files(
    workflow: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    names = _workflow_paths(workflow, _LORA_INPUT_FIELDS)
    roots = [
        Path(str(config.get(key) or "")).resolve()
        for key in (
            "lora_load_path",
            "bot_lora_load_path",
            "instance_lora_load_path",
            "style_lora_load_path",
        )
        if str(config.get(key) or "").strip()
    ]
    result: list[dict[str, Any]] = []
    for name in names:
        relative = _safe_relative(name, "LoRA")
        candidates: list[Path] = []
        for root in roots:
            candidates.extend((root / Path(*relative.parts), root.parent / Path(*relative.parts)))
            if relative.parts and relative.parts[0].casefold() == root.name.casefold():
                candidates.append(root.joinpath(*relative.parts[1:]))
        local_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if local_path is None:
            # 설치 매니페스트가 관리하는 기본 LoRA는 모델 Volume에 있으므로 사용자
            # LoRA Volume에 중복 업로드하지 않는다. 실제 누락이면 ComfyUI가 명확한
            # 노드 오류를 반환한다.
            print(f"[MODAL_SYNC] 로컬 사용자 LoRA 파일을 찾지 못해 업로드 생략: {name}")
            continue
        result.append(
            {
                "source_path": str(local_path),
                "remote_path": relative.as_posix(),
                "size": local_path.stat().st_size,
                "sha256": _sha256(local_path),
            }
        )
    return result


def resolve_input_files(
    workflow: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, str]]:
    input_root_raw = str(config.get("comfy_input_dir") or "").strip()
    if not input_root_raw:
        return []
    input_root = Path(input_root_raw).resolve()
    result: list[dict[str, str]] = []
    for name in _workflow_paths(workflow, _IMAGE_INPUT_FIELDS):
        relative = _safe_relative(name, "입력 이미지")
        candidate = input_root.joinpath(*relative.parts).resolve()
        if input_root != candidate and input_root not in candidate.parents:
            raise ValueError(f"ComfyUI input 밖의 이미지는 전송할 수 없습니다: {name!r}")
        if not candidate.is_file():
            print(f"[MODAL_SYNC] 입력 이미지 파일을 찾지 못해 업로드 생략: {candidate}")
            continue
        result.append({"source_path": str(candidate), "remote_name": relative.as_posix()})
    return result


def resolve_explicit_input_files(
    paths: Iterable[str | Path],
    config: Mapping[str, Any],
) -> list[dict[str, str]]:
    """호출자가 명시한 Comfy input 내부 파일·폴더를 원격 입력 목록으로 만든다."""

    input_root_raw = str(config.get("comfy_input_dir") or "").strip()
    if not input_root_raw:
        print("[MODAL_SYNC] 명시 입력 경로 처리 실패: comfy_input_dir 설정이 비어 있습니다.")
        raise ValueError("Modal 입력 동기화에 필요한 Comfy input 폴더가 비어 있습니다.")
    input_root = Path(input_root_raw).resolve()
    result: list[dict[str, str]] = []
    for raw_path in paths:
        candidate = Path(raw_path).resolve()
        if input_root != candidate and input_root not in candidate.parents:
            print(
                "[MODAL_SYNC] 명시 입력 경로 거부: Comfy input 폴더 밖입니다. "
                f"input_root={input_root}, candidate={candidate}"
            )
            raise ValueError(f"ComfyUI input 밖의 경로는 전송할 수 없습니다: {candidate}")
        if not candidate.exists():
            print(f"[MODAL_SYNC] 명시 입력 경로 없음: {candidate}")
            raise FileNotFoundError(f"Modal에 전송할 입력 경로가 없습니다: {candidate}")
        files = [candidate] if candidate.is_file() else sorted(
            path for path in candidate.rglob("*") if path.is_file()
        )
        if not files:
            print(f"[MODAL_SYNC] 명시 입력 폴더가 비어 있습니다: {candidate}")
            raise ValueError(f"Modal에 전송할 입력 폴더가 비어 있습니다: {candidate}")
        for source in files:
            relative = source.relative_to(input_root).as_posix()
            result.append({"source_path": str(source), "remote_name": relative})

    deduplicated: dict[str, dict[str, str]] = {}
    for item in result:
        deduplicated[item["remote_name"]] = item
    return list(deduplicated.values())
