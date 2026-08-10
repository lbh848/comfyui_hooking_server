from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import traceback
from typing import Any, Iterable, Mapping


_LORA_INPUT_FIELDS = {"lora_name", "lora"}
_IMAGE_INPUT_FIELDS = {"image"}
_SOYA_PROMPT_PARSER_CLASS = "SoyaPromptParser_mdsoya"
_SOYA_IPA_PATCH_MAKER_CLASS = "SoyaIPAPatchMaker_mdsoya"
_SOYA_CACHE_INPUT_FIELDS = {
    "embed_cache_data": ("CACHE_PATH", "emb_path"),
    "ipa_cache_data": ("FACE_ID_DIR", "ipa_path"),
}


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


def _workflow_node(
    workflow: Mapping[str, Any],
    node_id: Any,
) -> Mapping[str, Any] | None:
    node = workflow.get(str(node_id))
    if not isinstance(node, Mapping):
        node = workflow.get(node_id)
    return node if isinstance(node, Mapping) else None


def _resolve_linked_strings(
    workflow: Mapping[str, Any],
    value: Any,
    visiting: set[str] | None = None,
) -> list[str]:
    """Comfy 링크를 거슬러 올라가 Primitive 문자열 입력을 찾는다."""

    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return []

    node_id = str(value[0])
    active = set(visiting or ())
    if node_id in active:
        print(f"[MODAL_SYNC] Comfy 문자열 링크 순환을 감지해 중단: node={node_id}")
        return []
    active.add(node_id)

    node = _workflow_node(workflow, value[0])
    if node is None:
        print(f"[MODAL_SYNC] Comfy 문자열 링크 원본 노드를 찾지 못함: node={node_id}")
        return []
    inputs = node.get("inputs")
    if not isinstance(inputs, Mapping):
        print(f"[MODAL_SYNC] Comfy 문자열 링크 원본 입력이 객체가 아님: node={node_id}")
        return []

    resolved: list[str] = []
    preferred_fields = [field for field in ("value", "text") if field in inputs]
    source_values = (
        [inputs[field] for field in preferred_fields]
        if preferred_fields
        else list(inputs.values())
    )
    for source_value in source_values:
        resolved.extend(_resolve_linked_strings(workflow, source_value, active))
    return resolved


def _parse_soya_prompt_sections(text: str) -> dict[str, str]:
    """SoyaPromptParser_mdsoya와 같은 규칙으로 태그 구간을 분리한다."""

    parsed: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    for line in text.split("\n"):
        line = line.rstrip("\r")
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_key is not None:
                parsed[current_key] = "\n".join(current_lines).strip()
            current_key = stripped[1:-1]
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
    if current_key is not None:
        parsed[current_key] = "\n".join(current_lines).strip()
    return parsed


def _parse_json_stream(text: str, label: str) -> list[Any]:
    """Comfy 커스텀 노드와 동일하게 연속된 JSON 객체를 파싱한다."""

    stripped = text.strip()
    if not stripped:
        print(f"[MODAL_SYNC] 캐시 JSON 구간이 비어 있어 전송할 파일이 없음: {label}")
        return []
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    position = 0
    try:
        while position < len(stripped):
            if stripped[position] in " \t\n\r":
                position += 1
                continue
            payload, end = decoder.raw_decode(stripped, idx=position)
            objects.append(payload)
            position = end
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(
            f"[MODAL_SYNC] 캐시 JSON 파싱 실패: label={label}, "
            f"position={position}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise ValueError(f"Modal 입력 캐시 JSON 형식이 올바르지 않습니다: {label}") from exc
    return objects


def _cache_paths_from_json(text: str, path_key: str, label: str) -> list[str]:
    paths: list[str] = []
    for payload in _parse_json_stream(text, label):
        if isinstance(payload, Mapping) and "list" in payload:
            entries = payload["list"]
        elif isinstance(payload, list):
            entries = payload
        else:
            print(
                f"[MODAL_SYNC] 캐시 JSON 루트 형식 오류: label={label}, "
                f"type={type(payload).__name__}"
            )
            raise TypeError(f"Modal 입력 캐시 JSON 루트가 list 형식이 아닙니다: {label}")
        if not isinstance(entries, list):
            print(
                f"[MODAL_SYNC] 캐시 JSON list 필드 형식 오류: label={label}, "
                f"type={type(entries).__name__}"
            )
            raise TypeError(f"Modal 입력 캐시 JSON list 필드가 배열이 아닙니다: {label}")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                print(
                    f"[MODAL_SYNC] 캐시 JSON 항목 형식 오류: label={label}, "
                    f"index={index}, type={type(entry).__name__}"
                )
                raise TypeError(f"Modal 입력 캐시 항목이 객체가 아닙니다: {label}[{index}]")
            raw_path = entry.get(path_key)
            if not isinstance(raw_path, str) or not raw_path.strip():
                print(
                    f"[MODAL_SYNC] 캐시 JSON 경로 누락: label={label}, "
                    f"index={index}, field={path_key}"
                )
                raise ValueError(
                    f"Modal 입력 캐시 경로가 비어 있습니다: {label}[{index}].{path_key}"
                )
            paths.append(raw_path.strip().replace("\\", "/"))
    return paths


def _workflow_cache_paths(workflow: Mapping[str, Any]) -> list[str]:
    """Soya 프롬프트 프로토콜이 참조하는 필수 캐시 경로를 수집한다."""

    result: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, Mapping):
            continue
        if str(node.get("class_type") or "") != _SOYA_IPA_PATCH_MAKER_CLASS:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue

        for field, (section, path_key) in _SOYA_CACHE_INPUT_FIELDS.items():
            value = inputs.get(field)
            if isinstance(value, str) and value.strip():
                result.extend(
                    _cache_paths_from_json(value, path_key, f"node={node_id}.{field}")
                )
                continue
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue

            source_node = _workflow_node(workflow, value[0])
            source_texts = _resolve_linked_strings(workflow, value)
            if not source_texts:
                print(
                    f"[MODAL_SYNC] Soya 캐시 입력 문자열을 찾지 못해 경로 확인을 "
                    f"건너뜀: node={node_id}, field={field}, source={value[0]}"
                )
                continue
            source_is_parser = (
                source_node is not None
                and str(source_node.get("class_type") or "") == _SOYA_PROMPT_PARSER_CLASS
            )
            for source_index, source_text in enumerate(source_texts):
                cache_json = source_text
                if source_is_parser:
                    sections = _parse_soya_prompt_sections(source_text)
                    if section not in sections:
                        continue
                    cache_json = sections[section]
                result.extend(
                    _cache_paths_from_json(
                        cache_json,
                        path_key,
                        f"node={node_id}.{field}[{source_index}]",
                    )
                )
    return list(dict.fromkeys(result))


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
    try:
        cache_names = _workflow_cache_paths(workflow)
    except Exception as exc:
        print(
            f"[MODAL_SYNC] 워크플로우 필수 캐시 경로 해석 실패: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    input_root_raw = str(config.get("comfy_input_dir") or "").strip()
    if not input_root_raw:
        if cache_names:
            print(
                "[MODAL_SYNC] 필수 캐시 입력 경로 처리 실패: "
                "comfy_input_dir 설정이 비어 있습니다."
            )
            raise ValueError("Modal 캐시 입력 동기화에 필요한 Comfy input 폴더가 비어 있습니다.")
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
    for name in cache_names:
        try:
            relative = _safe_relative(name, "필수 캐시 입력")
            candidate = input_root.joinpath(*relative.parts).resolve()
            if input_root != candidate and input_root not in candidate.parents:
                raise ValueError(f"ComfyUI input 밖의 캐시는 전송할 수 없습니다: {name!r}")
            if not candidate.is_file():
                raise FileNotFoundError(f"Modal에 전송할 필수 캐시 파일이 없습니다: {candidate}")
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"[MODAL_SYNC] 필수 캐시 입력 확인 실패: name={name!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
        result.append({"source_path": str(candidate), "remote_name": relative.as_posix()})

    deduplicated: dict[str, dict[str, str]] = {}
    for item in result:
        deduplicated[item["remote_name"]] = item
    return list(deduplicated.values())


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
