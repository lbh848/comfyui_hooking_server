from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


_LORA_INPUT_FIELDS = {"lora_name", "lora"}
_IMAGE_INPUT_FIELDS = {"image"}


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
            # 설치 매니페스트가 관리하는 기본 LoRA는 /models에 있으므로 사용자 Volume에
            # 중복 업로드하지 않는다. 실제 누락이면 ComfyUI가 명확한 노드 오류를 반환한다.
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
