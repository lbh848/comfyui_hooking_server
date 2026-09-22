from __future__ import annotations

import copy
import datetime
import hashlib
import io
import json
import re
import secrets
import shutil
import traceback
import uuid
import zipfile
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

from PIL import Image


ProductionImageComparisonCallback = Callable[[dict[str, Any]], dict[str, Any]]
LogCallback = Callable[[str, str], None]
ProgressCallback = Callable[[dict[str, Any]], None]

_DIAGNOSTIC_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}-[0-9a-f]{8}$")
_SEED_BLOCK_RE = re.compile(
    r"(?m)(^\[SEED\][ \t]*\r?\n)[^\r\n]*"
)
_PROMPT_BLOCK_RE = re.compile(r"(?m)^\[([A-Z][A-Z0-9_]*)\][ \t]*\r?$\n?")
_PROMPT_TITLES = {"긍정프롬프트": "positive", "부정프롬프트": "negative"}
_SEED_INPUT_NAMES = frozenset({"seed", "noise_seed"})


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _normalize_prompt_seed(prompt: str, seed: int) -> str:
    if not isinstance(prompt, str):
        raise TypeError(f"프롬프트는 문자열이어야 합니다: type={type(prompt).__name__}")
    return _SEED_BLOCK_RE.sub(lambda match: f"{match.group(1)}{seed}", prompt)


def _prompt_blocks(prompt: str) -> dict[str, str]:
    matches = list(_PROMPT_BLOCK_RE.finditer(prompt or ""))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        blocks[match.group(1)] = prompt[match.end() : end].strip()
    return blocks


def _inject_prompt_and_seed(
    workflow: Mapping[str, Any],
    *,
    positive: str,
    negative: str,
    seed: int,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(workflow))
    found = {"positive": False, "negative": False}
    for node in result.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        title = str((node.get("_meta") or {}).get("title") or "")
        target = _PROMPT_TITLES.get(title)
        if target:
            if "value" not in inputs:
                raise ValueError(f"{title} 노드에 inputs.value가 없습니다.")
            inputs["value"] = positive if target == "positive" else negative
            found[target] = True
        for input_name in _SEED_INPUT_NAMES:
            raw_value = inputs.get(input_name)
            if isinstance(raw_value, bool):
                continue
            if isinstance(raw_value, (int, float)):
                inputs[input_name] = seed
    missing = [name for name, present in found.items() if not present]
    if missing:
        raise ValueError(
            "변환된 워크플로우에서 프롬프트 노드를 찾지 못했습니다: "
            + ", ".join(missing)
        )
    return result


def _image_evidence(image_bytes: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {
        "byte_size": len(image_bytes),
        "file_sha256": _sha256_bytes(image_bytes),
    }
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgba = image.convert("RGBA")
            result.update(
                {
                    "format": image.format,
                    "width": rgba.width,
                    "height": rgba.height,
                    "pixel_sha256_rgba": _sha256_bytes(rgba.tobytes()),
                }
            )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_COMPARISON] 이미지 증거 해석 실패: "
            f"error={type(exc).__name__}: {exc}, bytes={len(image_bytes)}"
        )
        traceback.print_exc()
        result["decode_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _image_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return ".bin"


def _workflow_diff(left: Any, right: Any) -> dict[str, Any]:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {
            "comparable": False,
            "reason": (
                f"workflow types: left={type(left).__name__}, "
                f"right={type(right).__name__}"
            ),
        }
    left_ids = set(str(key) for key in left)
    right_ids = set(str(key) for key in right)
    input_changes: list[dict[str, Any]] = []
    class_changes: list[dict[str, Any]] = []
    for node_id in sorted(left_ids & right_ids):
        left_node = left.get(node_id, left.get(int(node_id)) if node_id.isdigit() else None)
        right_node = right.get(node_id, right.get(int(node_id)) if node_id.isdigit() else None)
        if not isinstance(left_node, dict) or not isinstance(right_node, dict):
            if left_node != right_node:
                input_changes.append(
                    {"node_id": node_id, "input": "<node>", "left": left_node, "right": right_node}
                )
            continue
        left_class = left_node.get("class_type")
        right_class = right_node.get("class_type")
        if left_class != right_class:
            class_changes.append(
                {"node_id": node_id, "left": left_class, "right": right_class}
            )
        left_inputs = left_node.get("inputs") if isinstance(left_node.get("inputs"), dict) else {}
        right_inputs = right_node.get("inputs") if isinstance(right_node.get("inputs"), dict) else {}
        for input_name in sorted(set(left_inputs) | set(right_inputs)):
            left_value = left_inputs.get(input_name, "<missing>")
            right_value = right_inputs.get(input_name, "<missing>")
            if left_value != right_value:
                input_changes.append(
                    {
                        "node_id": node_id,
                        "class_type": right_class or left_class,
                        "input": input_name,
                        "left": left_value,
                        "right": right_value,
                    }
                )
    return {
        "comparable": True,
        "left_sha256": _canonical_sha256(left),
        "right_sha256": _canonical_sha256(right),
        "identical": left == right,
        "nodes_only_left": sorted(left_ids - right_ids),
        "nodes_only_right": sorted(right_ids - left_ids),
        "class_changes": class_changes,
        "input_change_count": len(input_changes),
        "input_changes": input_changes[:300],
        "input_changes_truncated": len(input_changes) > 300,
    }


def _analyze(
    *,
    snapshot: Mapping[str, Any],
    direct_positive: str,
    character_maker_positive: str,
    cases: list[dict[str, Any]],
    source_modified: bool,
) -> tuple[list[str], dict[str, Any]]:
    findings: list[str] = []
    direct_blocks = _prompt_blocks(direct_positive)
    maker_blocks = _prompt_blocks(character_maker_positive)
    changed_blocks = [
        name
        for name in sorted(set(direct_blocks) | set(maker_blocks))
        if direct_blocks.get(name) != maker_blocks.get(name)
    ]
    if changed_blocks:
        findings.append(
            "Comfy 저장 프롬프트와 캐릭터 메이커 프롬프트의 블록이 다릅니다: "
            + ", ".join(changed_blocks)
        )
    artist_changes = [
        name for name in ("ANIMA_ARTIST", "SDXL_ARTIST") if name in changed_blocks
    ]
    if artist_changes:
        findings.append(
            "Artist 블록이 실제로 다릅니다: " + ", ".join(artist_changes)
        )
    lora_changes = [
        name
        for name in ("LORA_DATA", "STYLE_LORA_DATA", "FACE_LORA_DATA")
        if name in changed_blocks
    ]
    if lora_changes:
        findings.append("LoRA 데이터 블록이 다릅니다: " + ", ".join(lora_changes))

    direct_port = snapshot.get("direct_runtime", {}).get("port")
    program_port = snapshot.get("program_runtime", {}).get("port")
    program_target = snapshot.get("program_runtime", {}).get("execution_target")
    if program_target not in (None, "", "local"):
        findings.append(
            f"프로그램 생성 대상이 로컬 Comfy가 아닙니다: {program_target}"
        )
    elif direct_port != program_port:
        findings.append(
            "Comfy 직접 실행과 프로그램 실행의 포트가 다릅니다: "
            f"direct={direct_port}, program={program_port}"
        )

    case_map = {str(case.get("name")): case for case in cases}
    pair_names = (
        ("direct_comfy_prompt", "program_comfy_prompt", "Comfy 프롬프트"),
        ("direct_character_maker_prompt", "program_character_maker_prompt", "캐릭터 메이커 프롬프트"),
    )
    comparisons: dict[str, Any] = {}
    for direct_name, program_name, label in pair_names:
        direct_case = case_map.get(direct_name) or {}
        program_case = case_map.get(program_name) or {}
        comparison = _workflow_diff(
            direct_case.get("submitted_workflow"),
            program_case.get("submitted_workflow"),
        )
        direct_image = direct_case.get("image") or {}
        program_image = program_case.get("image") or {}
        pixel_equal = bool(
            direct_image.get("pixel_sha256_rgba")
            and direct_image.get("pixel_sha256_rgba")
            == program_image.get("pixel_sha256_rgba")
        )
        comparison["pixel_identical"] = pixel_equal
        comparisons[label] = comparison
        actual_program_target = str(
            program_case.get("execution_target") or "local"
        )
        if actual_program_target != "local":
            findings.append(
                f"{label}의 실제 프로그램 실행 대상이 로컬 Comfy가 아닙니다: "
                f"{actual_program_target}"
            )
        if direct_case.get("status") != "success" or program_case.get("status") != "success":
            findings.append(f"{label} 경로 비교 중 하나 이상이 실패했습니다.")
            continue
        if not comparison.get("identical"):
            findings.append(
                f"{label}에서 Comfy 직접 경로와 프로그램 경로의 최종 제출 워크플로우가 다릅니다"
                f"(입력 차이 {comparison.get('input_change_count', 0)}개)."
            )
        elif pixel_equal:
            findings.append(
                f"{label}에서는 두 경로의 제출 워크플로우와 출력 픽셀이 동일합니다."
            )
        else:
            findings.append(
                f"{label}에서는 제출 워크플로우가 동일하지만 출력 픽셀이 다릅니다. "
                "같은 Comfy 런타임의 비결정성·캐시 또는 실행 시점 상태를 확인해야 합니다."
            )

    dependency_sets = snapshot.get("dependency_sets") or {}
    direct_dependency_port = (
        (dependency_sets.get("direct") or {}).get("port")
        if isinstance(dependency_sets.get("direct"), dict)
        else None
    )
    for runtime_label, dependency_set in dependency_sets.items():
        missing = dependency_set.get("missing_node_classes") if isinstance(dependency_set, dict) else None
        if missing:
            findings.append(
                f"{runtime_label} Comfy에 워크플로우 노드가 등록되지 않았습니다: "
                + ", ".join(str(value) for value in missing)
            )
        if not isinstance(dependency_set, dict):
            continue
        if (
            runtime_label == "program"
            and direct_dependency_port is not None
            and dependency_set.get("port") == direct_dependency_port
        ):
            continue
        declared_mismatches: set[tuple[str, tuple[str, ...], str]] = set()
        for node in dependency_set.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            package = node.get("package")
            if not isinstance(package, dict) or not package.get("version"):
                continue
            installed_version = str(package["version"])
            declared_versions = {
                str((entry.get("declared_package") or {}).get("ver"))
                for entry in (node.get("workflow_declarations") or [])
                if isinstance(entry, dict)
                and (entry.get("declared_package") or {}).get("ver") not in (None, "")
            }
            if declared_versions and declared_versions != {installed_version}:
                declared_mismatches.add(
                    (
                        str(package.get("name") or node.get("python_module") or node.get("class_type")),
                        tuple(sorted(declared_versions)),
                        installed_version,
                    )
                )
        for package_name, declared_versions, installed_version in sorted(
            declared_mismatches
        ):
            findings.append(
                f"{runtime_label}의 {package_name}은 워크플로우 기록 버전"
                f"({', '.join(declared_versions)})과 설치 패키지 버전"
                f"({installed_version})이 다릅니다."
            )

    direct_dependencies = dependency_sets.get("direct")
    program_dependencies = dependency_sets.get("program")
    if isinstance(direct_dependencies, dict) and isinstance(program_dependencies, dict):
        direct_system = direct_dependencies.get("system") or {}
        program_system = program_dependencies.get("system") or {}
        for field, label in (
            ("comfyui_version", "ComfyUI"),
            ("python_version", "Python"),
            ("pytorch_version", "PyTorch"),
        ):
            direct_value = direct_system.get(field) if isinstance(direct_system, dict) else None
            program_value = program_system.get(field) if isinstance(program_system, dict) else None
            if direct_value and program_value and direct_value != program_value:
                findings.append(
                    f"직접/프로그램 런타임의 {label} 버전이 다릅니다: "
                    f"direct={direct_value}, program={program_value}"
                )

        def package_versions(dependency_set: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
            versions: dict[str, tuple[str, str]] = {}
            for node in dependency_set.get("nodes") or []:
                if not isinstance(node, dict):
                    continue
                package = node.get("package")
                if not isinstance(package, dict) or not package.get("name"):
                    continue
                versions[str(package["name"])] = (
                    str(package.get("version") or ""),
                    str(package.get("git_head") or ""),
                )
            return versions

        direct_packages = package_versions(direct_dependencies)
        program_packages = package_versions(program_dependencies)
        for package_name in sorted(set(direct_packages) & set(program_packages)):
            direct_version = direct_packages[package_name]
            program_version = program_packages[package_name]
            if direct_version != program_version:
                findings.append(
                    "직접/프로그램 런타임에서 이 워크플로우가 사용하는 커스텀 노드 "
                    f"{package_name} 버전이 다릅니다: "
                    f"direct={direct_version}, program={program_version}"
                )
    if source_modified:
        findings.append(
            "검사 도중 원본 워크플로우 파일의 SHA-256이 바뀌었습니다. "
            "진단 실행은 시작 시 복사한 스냅샷을 사용했습니다."
        )
    if not findings:
        findings.append("기록된 프롬프트·워크플로우·런타임 차이를 찾지 못했습니다.")
    return findings, {
        "changed_prompt_blocks": changed_blocks,
        "direct_prompt_blocks": direct_blocks,
        "character_maker_prompt_blocks": maker_blocks,
        "path_comparisons": comparisons,
    }


def _report_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Comfy 직접 실행 / 프로그램 실행 이미지 비교 진단",
        "",
        f"- 진단 ID: `{result.get('archive_id', '')}`",
        f"- 생성 방식: `{result.get('generation_workflow', '')}`",
        f"- 진단 seed: `{result.get('seed', '')}`",
        f"- 원본 워크플로우: `{result.get('source_path', '')}`",
        f"- 원본 SHA-256: `{result.get('source_sha256_before', '')}`",
        f"- 원본 변경 감지: `{bool(result.get('source_modified'))}`",
        "",
        "## 판정",
        "",
    ]
    for finding in result.get("conclusions") or []:
        lines.append(f"- {finding}")
    lines.extend(["", "## 실행 결과", ""])
    for case in result.get("cases") or []:
        image = case.get("image") or {}
        lines.extend(
            [
                f"### {case.get('label') or case.get('name')}",
                "",
                f"- 상태: `{case.get('status')}`",
                f"- 경로: `{case.get('route')}`",
                f"- 프롬프트 출처: `{case.get('prompt_source')}`",
                f"- 큐 항목: `{case.get('queue_item_id', '')}`",
                f"- 이미지: `{case.get('image_file', '')}`",
                f"- 픽셀 SHA-256: `{image.get('pixel_sha256_rgba', '')}`",
                f"- 오류: `{case.get('error', '')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 파일 안내",
            "",
            "- `source/`: 검사 시작 시 바이트 그대로 복사한 원본 워크플로우",
            "- `prompts/`: Comfy 저장 프롬프트와 캐릭터 메이커 프롬프트",
            "- `cases/`: 각 실행 이미지, 최종 제출 워크플로우, Comfy 로그와 런타임 상태",
            "- `dependencies.json`: 실제 워크플로우 노드에 한정한 등록·버전 정보",
            "- `analysis.json`: 프롬프트 블록 및 최종 제출 워크플로우 비교",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_archive(work_dir: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_name(f"{archive.name}.part")
    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in sorted(work_dir.rglob("*")):
                if path.is_file():
                    output.write(path, path.relative_to(work_dir).as_posix())
        part.replace(archive)
    except Exception:
        try:
            if part.is_file():
                part.unlink()
        except Exception as cleanup_exc:
            print(
                "[COMFY_INSTALL][IMAGE_COMPARISON] 불완전 ZIP 정리 실패: "
                f"path={part}, error={cleanup_exc}"
            )
            traceback.print_exc()
        raise


def run_image_comparison_diagnostic(
    *,
    project_root: Path,
    cancel_event: Event,
    request: Mapping[str, Any],
    production_call: ProductionImageComparisonCallback,
    log: LogCallback,
    progress: ProgressCallback,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    archive_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if not _DIAGNOSTIC_ID_RE.fullmatch(archive_id):
        raise RuntimeError(f"이미지 비교 진단 ID 생성 실패: {archive_id!r}")
    archive_root = (
        project_root / ".work" / "comfy-installer" / "image-comparison-diagnostics"
    ).resolve()
    work_dir = (archive_root / archive_id).resolve()
    archive = (archive_root / f"{archive_id}.zip").resolve()
    if work_dir.parent != archive_root or archive.parent != archive_root:
        raise RuntimeError("이미지 비교 진단 작업 경로가 안전한 루트를 벗어났습니다.")
    work_dir.mkdir(parents=True, exist_ok=False)

    raw_seed = request.get("seed")
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError):
        seed = secrets.randbelow(2**32)
        print(
            "[COMFY_INSTALL][IMAGE_COMPARISON] CM seed 해석 실패, 진단 seed 생성: "
            f"input={raw_seed!r}, seed={seed}"
        )
    if not 0 <= seed <= 2**32 - 1:
        seed = secrets.randbelow(2**32)
        print(
            "[COMFY_INSTALL][IMAGE_COMPARISON] CM seed 범위 오류, 진단 seed 생성: "
            f"input={raw_seed!r}, seed={seed}"
        )

    result: dict[str, Any] = {
        "operation": "image_comparison_diagnostic",
        "archive_id": archive_id,
        "archive_name": archive.name,
        "started_at": _now_iso(),
        "seed": seed,
        "cases": [],
        "conclusions": [],
        "incomplete": False,
        "cancelled": False,
    }
    snapshot: dict[str, Any] = {}
    source_path: Path | None = None
    source_sha256_before = ""
    diagnostic_character = f"image_comparison_{archive_id.replace('-', '_')}"

    try:
        log("[준비] 현재 캐릭터 메이커 워크플로우와 두 Comfy 실행 경로를 확인합니다.", "info")
        progress({"event": "comparison_prepare", "current": 0, "total": 4})
        snapshot_response = production_call(
            {
                "action": "comparison_snapshot",
                "diagnostic_id": archive_id,
            }
        )
        if not isinstance(snapshot_response, dict):
            raise RuntimeError(
                "비교 진단 스냅샷 응답 형식이 올바르지 않습니다: "
                f"type={type(snapshot_response).__name__}"
            )
        snapshot = snapshot_response
        source_path = Path(str(snapshot.get("source_path") or "")).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"캐릭터 메이커 워크플로우 파일이 없습니다: {source_path}")
        source_bytes = source_path.read_bytes()
        source_sha256_before = _sha256_bytes(source_bytes)
        reported_hash = str(snapshot.get("source_sha256") or "")
        if reported_hash and reported_hash != source_sha256_before:
            raise RuntimeError(
                "워크플로우 스냅샷 중 파일이 변경되었습니다: "
                f"callback={reported_hash}, copied={source_sha256_before}"
            )
        source_copy = work_dir / "source" / source_path.name
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        source_copy.write_bytes(source_bytes)

        direct_base = snapshot.get("converted_workflow")
        if not isinstance(direct_base, dict) or not direct_base:
            raise RuntimeError("Comfy 직접 실행용 변환 워크플로우가 비어 있습니다.")
        direct_positive_raw = str(snapshot.get("direct_positive") or "")
        direct_negative = str(snapshot.get("direct_negative") or "")
        maker_positive_raw = str(request.get("positive") or "")
        maker_negative = str(request.get("negative") or "")
        if not direct_positive_raw.strip():
            raise RuntimeError("원본 워크플로우의 긍정 프롬프트가 비어 있습니다.")
        if not maker_positive_raw.strip():
            raise RuntimeError("캐릭터 메이커 긍정 프롬프트가 비어 있습니다.")

        direct_positive = _normalize_prompt_seed(direct_positive_raw, seed)
        maker_positive = _normalize_prompt_seed(maker_positive_raw, seed)
        _write_text(work_dir / "prompts" / "comfy_positive.txt", direct_positive)
        _write_text(work_dir / "prompts" / "comfy_negative.txt", direct_negative)
        _write_text(work_dir / "prompts" / "character_maker_positive.txt", maker_positive)
        _write_text(work_dir / "prompts" / "character_maker_negative.txt", maker_negative)

        result.update(
            {
                "generation_workflow": snapshot.get("generation_workflow"),
                "workflow_profile": snapshot.get("workflow_profile"),
                "source_path": str(source_path),
                "source_copy": str(source_copy.relative_to(work_dir)),
                "source_sha256_before": source_sha256_before,
                "direct_runtime": snapshot.get("direct_runtime"),
                "program_runtime": snapshot.get("program_runtime"),
            }
        )
        _write_json(work_dir / "snapshot.json", snapshot.get("summary") or {})
        _write_json(
            work_dir / "dependencies.json",
            snapshot.get("dependency_sets") or {},
        )

        cases = [
            {
                "name": "direct_comfy_prompt",
                "label": "Comfy 프롬프트 → Comfy 직접 실행",
                "route": "direct",
                "prompt_source": "comfy",
                "positive": direct_positive,
                "negative": direct_negative,
            },
            {
                "name": "program_comfy_prompt",
                "label": "Comfy 프롬프트 → 프로그램 실행",
                "route": "program",
                "prompt_source": "comfy",
                "positive": direct_positive,
                "negative": direct_negative,
            },
            {
                "name": "direct_character_maker_prompt",
                "label": "캐릭터 메이커 프롬프트 → Comfy 직접 실행",
                "route": "direct",
                "prompt_source": "character_maker",
                "positive": maker_positive,
                "negative": maker_negative,
            },
            {
                "name": "program_character_maker_prompt",
                "label": "캐릭터 메이커 프롬프트 → 프로그램 실행",
                "route": "program",
                "prompt_source": "character_maker",
                "positive": maker_positive,
                "negative": maker_negative,
            },
        ]

        for index, plan in enumerate(cases, start=1):
            if cancel_event.is_set():
                result["cancelled"] = True
                result["incomplete"] = True
                log(
                    "[중단] 현재 실행을 마친 뒤 남은 이미지 비교 케이스를 시작하지 않습니다.",
                    "warning",
                )
                break
            case_result: dict[str, Any] = {
                "name": plan["name"],
                "label": plan["label"],
                "route": plan["route"],
                "prompt_source": plan["prompt_source"],
                "status": "running",
            }
            result["cases"].append(case_result)
            progress(
                {
                    "event": "comparison_generate",
                    "current": index - 1,
                    "total": 4,
                    "item": plan["label"],
                }
            )
            log(f"[실행 {index}/4] {plan['label']}", "info")
            try:
                request_workflow = None
                if plan["route"] == "direct":
                    request_workflow = _inject_prompt_and_seed(
                        direct_base,
                        positive=plan["positive"],
                        negative=plan["negative"],
                        seed=seed,
                    )
                response = production_call(
                    {
                        "action": "comparison_generate",
                        "diagnostic_id": archive_id,
                        "diagnostic_character": diagnostic_character,
                        "case": plan["name"],
                        "route": plan["route"],
                        "positive": plan["positive"],
                        "negative": plan["negative"],
                        "seed": seed,
                        "workflow": request_workflow,
                        "generation_workflow": snapshot.get("generation_workflow"),
                        "workflow_profile": snapshot.get("workflow_profile"),
                        "width": request.get("width"),
                        "height": request.get("height"),
                    }
                )
                if not isinstance(response, dict):
                    raise RuntimeError(
                        "생성 응답 형식이 올바르지 않습니다: "
                        f"type={type(response).__name__}"
                    )
                image_bytes = response.pop("image_bytes", None)
                submitted_workflow = response.pop("submitted_workflow", None)
                comfy_log = str(response.pop("comfy_log", "") or "")
                if not isinstance(image_bytes, bytes) or not image_bytes:
                    raise RuntimeError(
                        str(response.get("error") or "진단 이미지 바이트가 비어 있습니다.")
                    )
                if not isinstance(submitted_workflow, dict) or not submitted_workflow:
                    raise RuntimeError("최종 제출 워크플로우가 수집되지 않았습니다.")

                case_dir = work_dir / "cases" / plan["name"]
                extension = _image_extension(image_bytes)
                image_file = case_dir / f"image{extension}"
                image_file.parent.mkdir(parents=True, exist_ok=True)
                image_file.write_bytes(image_bytes)
                workflow_file = case_dir / "submitted_workflow.json"
                _write_json(workflow_file, submitted_workflow)
                _write_text(case_dir / "comfy.log", comfy_log)
                _write_json(case_dir / "runtime.json", response)
                case_result.update(
                    {
                        "status": "success",
                        "image_file": str(image_file.relative_to(work_dir)),
                        "workflow_file": str(workflow_file.relative_to(work_dir)),
                        "submitted_workflow_sha256": _canonical_sha256(submitted_workflow),
                        "submitted_workflow": submitted_workflow,
                        "image": _image_evidence(image_bytes),
                        **response,
                    }
                )
                log(
                    f"[완료 {index}/4] {plan['label']} · {len(image_bytes):,} bytes",
                    "info",
                )
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][IMAGE_COMPARISON] 케이스 실패: "
                    f"case={plan['name']}, route={plan['route']}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                case_result.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                _write_json(
                    work_dir / "cases" / plan["name"] / "failure.json",
                    case_result,
                )
                result["incomplete"] = True
                log(f"[실패 {index}/4] {plan['label']}: {exc}", "error")
            progress(
                {
                    "event": "comparison_generate",
                    "current": index,
                    "total": 4,
                    "item": plan["label"],
                }
            )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_COMPARISON] 진단 준비 실패: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        result["incomplete"] = True
        result["fatal_error"] = f"{type(exc).__name__}: {exc}"
        result["fatal_traceback"] = traceback.format_exc()
        log(f"[실패] 이미지 비교 진단 준비 실패: {exc}", "error")
    finally:
        try:
            cleanup = production_call(
                {
                    "action": "comparison_cleanup",
                    "diagnostic_id": archive_id,
                    "diagnostic_character": diagnostic_character,
                }
            )
            result["cleanup"] = cleanup
        except Exception as cleanup_exc:
            print(
                "[COMFY_INSTALL][IMAGE_COMPARISON] 임시 생성물 정리 실패: "
                f"character={diagnostic_character}, "
                f"error={type(cleanup_exc).__name__}: {cleanup_exc}"
            )
            traceback.print_exc()
            result["cleanup"] = {
                "success": False,
                "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
            }
            result["incomplete"] = True

    source_sha256_after = ""
    if source_path is not None and source_path.is_file():
        try:
            source_sha256_after = _sha256_bytes(source_path.read_bytes())
        except Exception as exc:
            print(
                "[COMFY_INSTALL][IMAGE_COMPARISON] 원본 워크플로우 사후 해시 실패: "
                f"path={source_path}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            result["source_hash_error"] = f"{type(exc).__name__}: {exc}"
            result["incomplete"] = True
    result["source_sha256_after"] = source_sha256_after
    result["source_modified"] = bool(
        source_sha256_before
        and source_sha256_after
        and source_sha256_before != source_sha256_after
    )

    conclusions, analysis = _analyze(
        snapshot=snapshot,
        direct_positive=str(snapshot.get("direct_positive") or ""),
        character_maker_positive=str(request.get("positive") or ""),
        cases=result["cases"],
        source_modified=bool(result["source_modified"]),
    )
    result["conclusions"] = conclusions
    result["completed_at"] = _now_iso()
    _write_json(work_dir / "analysis.json", analysis)

    serializable_result = copy.deepcopy(result)
    for case in serializable_result.get("cases") or []:
        case.pop("submitted_workflow", None)
    _write_json(work_dir / "summary.json", serializable_result)
    _write_text(work_dir / "report.md", _report_markdown(serializable_result))
    _safe_archive(work_dir, archive)
    result["archive_path"] = str(archive)
    result["archive_name"] = archive.name
    for case in result.get("cases") or []:
        case.pop("submitted_workflow", None)
        case.pop("traceback", None)
    try:
        shutil.rmtree(work_dir)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][IMAGE_COMPARISON] ZIP 생성 후 작업 폴더 정리 실패: "
            f"path={work_dir}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        result["work_dir_cleanup_error"] = f"{type(exc).__name__}: {exc}"
    return result
