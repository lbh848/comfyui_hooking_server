"""SDStudio 프로젝트 프리셋을 현재 에셋 프리셋 구조로 가져오는 도구.

분석 단계에서는 원본 구조만 해석하고 파일을 쓰지 않는다. LLM은 프로그램이
나눈 원문 조각의 카테고리만 선택하며, 최종 문자열은 사용자 편집값 또는 원문을
그대로 사용한다. 커밋은 원본 조각의 누락/중복과 이름 충돌을 다시 검증한 뒤
배포 환경에 존재하는 asset_data/backup 아래에 백업하고 원자적으로 저장한다.
"""

from __future__ import annotations

import base64
import copy
import datetime as _datetime
import hashlib
import itertools
import json
import os
import re
import shutil
import threading
import traceback
import uuid
from typing import Any, Iterable

from . import nai_prompt_parser


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
TAGS_FILE = os.path.join(ASSET_DATA_DIR, "tags.json")
HIDDEN_TAGS_FILE = os.path.join(ASSET_DATA_DIR, "hidden_tags.json")
MANIFEST_FILE = os.path.join(ASSET_DATA_DIR, "preset_import_manifests.json")
BACKUP_DIR = os.path.join(ASSET_DATA_DIR, "backup")

IMPORT_CATEGORIES = (
    "expressions",
    "composition_presets",
    "appearances",
    "outfits",
    "artist_presets",
    "quality_presets",
    "negative_presets",
    "character_negative_presets",
)
SCENE_LLM_CATEGORIES = (
    *IMPORT_CATEGORIES,
    "unassigned",
)
PRESET_LLM_CATEGORIES = (
    *IMPORT_CATEGORIES,
    "unassigned",
)

# SDStudio 임포트의 대상 어댑터는 ANIMA다. 씬별로 생성된 카테고리
# 프리셋을 에셋 일괄생성 슬롯의 대응 필드에 연결할 때 사용하는 최소 매핑이다.
# 태그 문구를 보고 추론하지 않고, 사용자가 확정한 카테고리만 그대로 옮긴다.
SCENE_CHAIN_FIELD_BY_CATEGORY = {
    "appearances": "appearance",
    "outfits": "outfit",
    "expressions": "expression",
    "composition_presets": "composition_preset",
    "artist_presets": "anima_artist_preset",
    "quality_presets": "anima_quality_preset",
    "negative_presets": "anima_negative_preset",
    "character_negative_presets": "character_negative_preset",
}

MAX_VARIANTS_PER_GROUP = 500
MAX_TOTAL_ITEMS = 5000
MAX_CLASSIFY_FRAGMENTS = 30
SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_ANALYSIS_SESSIONS = 12

_PIECE_REF_RE = re.compile(r"<([^<>]+)>")
_ANALYSIS_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSION_LOCK = threading.RLock()


class PresetImportError(ValueError):
    """사용자에게 400 계열 응답으로 보여줄 수 있는 임포트 오류."""


def _now_iso() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).astimezone().isoformat()


def _warning(code: str, message: str, **extra) -> dict:
    return {"code": code, "message": message, **extra}


def _normalized_document_hash(document: dict) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_prompt_fragments(text: str) -> tuple[list[str], list[dict]]:
    """괄호 내부 쉼표와 가중치 문법을 보존하며 최상위 쉼표/개행만 나눈다."""
    if not isinstance(text, str):
        print(f"[PRESET_IMPORT] 프롬프트 분해 실패: 문자열 아님 type={type(text).__name__}")
        raise PresetImportError("프롬프트 값이 문자열이 아닙니다.")

    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    buffer: list[str] = []
    fragments: list[str] = []
    warnings: list[dict] = []

    def flush():
        value = "".join(buffer).strip()
        buffer.clear()
        if value:
            fragments.append(value)

    for index, char in enumerate(text):
        if char in opening:
            stack.append(char)
            buffer.append(char)
            continue
        if char in closing:
            if not stack or stack[-1] != closing[char]:
                warnings.append(_warning(
                    "unbalanced_closing_delimiter",
                    f"닫는 괄호 '{char}'의 짝이 맞지 않습니다.",
                    offset=index,
                ))
            else:
                stack.pop()
            buffer.append(char)
            continue
        if char in {",", "\n", "\r"} and not stack:
            flush()
            continue
        buffer.append(char)
    flush()

    if stack:
        warnings.append(_warning(
            "unclosed_delimiter",
            f"닫히지 않은 괄호가 있습니다: {''.join(stack)}",
        ))
    return fragments, warnings


def _library_piece_map(document: dict) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    libraries = document.get("library", {})
    if not isinstance(libraries, dict):
        print(
            "[PRESET_IMPORT] library 형식 오류: "
            f"type={type(libraries).__name__}; 조각 참조를 확장하지 않음"
        )
        return result
    for library_key, library in libraries.items():
        if not isinstance(library, dict):
            print(
                "[PRESET_IMPORT] library 항목 형식 오류: "
                f"name={library_key!r}, type={type(library).__name__}"
            )
            continue
        library_name = str(library.get("name") or library_key)
        pieces = library.get("pieces", [])
        if not isinstance(pieces, list):
            print(
                "[PRESET_IMPORT] library pieces 형식 오류: "
                f"name={library_name!r}, type={type(pieces).__name__}"
            )
            continue
        for piece in pieces:
            if not isinstance(piece, dict) or not isinstance(piece.get("name"), str):
                print(
                    "[PRESET_IMPORT] 잘못된 프롬프트 조각 건너뜀: "
                    f"library={library_name!r}, piece={piece!r}"
                )
                continue
            result[(library_name, piece["name"])] = piece
    return result


def _prompt_chunk_map(
    document: dict,
    piece_map: dict[tuple[str, str], dict],
) -> dict[str, str]:
    """세션에 포함된 명시적 Prompt Chunk와 고유 라이브러리 조각을 모은다."""
    result: dict[str, str] = {}
    duplicate_piece_names: set[str] = set()
    for (_, piece_name), piece in piece_map.items():
        prompt = piece.get("prompt")
        if not isinstance(prompt, str) or bool(piece.get("multi")):
            continue
        if piece_name in result and result[piece_name] != prompt:
            duplicate_piece_names.add(piece_name)
            result.pop(piece_name, None)
            continue
        if piece_name not in duplicate_piece_names:
            result[piece_name] = prompt

    for field in ("promptChunks", "prompt_chunks", "macros"):
        raw = document.get(field)
        if raw is None:
            continue
        if isinstance(raw, dict):
            values = raw.items()
        elif isinstance(raw, list):
            values = (
                (entry.get("name"), entry.get("prompt", entry.get("text")))
                for entry in raw
                if isinstance(entry, dict)
            )
        else:
            print(
                "[PRESET_IMPORT] NAI Prompt Chunk 저장소 형식 오류: "
                f"field={field}, type={type(raw).__name__}"
            )
            continue
        for name, value in values:
            if not isinstance(name, str) or not isinstance(value, str):
                print(
                    "[PRESET_IMPORT] 잘못된 NAI Prompt Chunk 건너뜀: "
                    f"field={field}, name={name!r}, value_type={type(value).__name__}"
                )
                continue
            result[name.strip()] = value
    return result


def _convert_nai_text(
    text: str,
    prompt_chunks: dict[str, str],
    *,
    item_id: str,
    source_field: str,
) -> dict:
    try:
        converted = nai_prompt_parser.convert_nai_prompt(
            text,
            prompt_chunks=prompt_chunks,
            target="anima",
            max_abs_weight="1.5",
            randomizer_strategy="dynamic_prompt",
        )
    except Exception as exc:
        print(
            "[PRESET_IMPORT] NAI 프롬프트 변환 실패: "
            f"item_id={item_id}, field={source_field}, error={exc}"
        )
        traceback.print_exc()
        raise PresetImportError(
            f"NAI 프롬프트를 변환하지 못했습니다: {source_field}"
        ) from exc
    for warning in converted["warnings"]:
        warning.setdefault("item_id", item_id)
        warning.setdefault("source_field", source_field)
    return converted


def _converted_fragments(
    converted: dict,
    *,
    item_id: str,
    source_field: str,
    start_index: int = 0,
    forced_category: str | None = None,
) -> list[dict]:
    fragments = []
    for offset, source in enumerate(converted["fragments"]):
        category = forced_category or "unassigned"
        fragments.append({
            "id": f"{item_id}-fragment-{start_index + offset}",
            "original_text": source["source_text"],
            "import_text": source["text"],
            "text": source["text"],
            "source_field": source_field,
            "source_region": source.get("region", "base"),
            "category": category,
            "llm_category": None,
            "llm_eligible": forced_category is None,
            "origin": "source",
            "excluded": False,
            "normalization": {
                "kind": source.get("kind", "tag"),
                "changed": bool(source.get("changed")),
                "weight": source.get("weight", "1"),
                "raw_weight": source.get("raw_weight", "1"),
                **copy.deepcopy(source.get("metadata", {})),
            },
        })
    return fragments


def _expand_piece_references(
    text: str,
    piece_map: dict[tuple[str, str], dict],
    *,
    stack: tuple[tuple[str, str], ...] = (),
) -> tuple[str, list[dict]]:
    warnings: list[dict] = []

    def replace(match: re.Match) -> str:
        raw = match.group(1)
        if "." not in raw:
            warnings.append(_warning(
                "invalid_piece_reference",
                f"프롬프트 조각 참조 형식이 올바르지 않습니다: <{raw}>",
            ))
            return match.group(0)
        library_name, piece_name = raw.split(".", 1)
        key = (library_name, piece_name)
        piece = piece_map.get(key)
        if piece is None:
            warnings.append(_warning(
                "missing_piece_reference",
                f"참조한 프롬프트 조각을 찾을 수 없습니다: <{raw}>",
            ))
            return match.group(0)
        if key in stack:
            warnings.append(_warning(
                "cyclic_piece_reference",
                f"순환 프롬프트 조각 참조를 발견했습니다: <{raw}>",
            ))
            return match.group(0)
        if bool(piece.get("multi")):
            warnings.append(_warning(
                "multi_piece_requires_review",
                f"여러 줄 랜덤 조각은 자동 평탄화하지 않습니다: <{raw}>",
            ))
            return match.group(0)
        prompt = piece.get("prompt", "")
        if not isinstance(prompt, str):
            warnings.append(_warning(
                "invalid_piece_prompt",
                f"프롬프트 조각 내용이 문자열이 아닙니다: <{raw}>",
            ))
            return match.group(0)
        expanded, nested_warnings = _expand_piece_references(
            prompt,
            piece_map,
            stack=(*stack, key),
        )
        warnings.extend(nested_warnings)
        return expanded

    previous = text
    # 한 번의 sub 안에서 새로 생긴 참조도 재귀 함수가 처리한다. 원본에 나란히
    # 있던 참조는 정규식이 모두 처리하므로 별도 의미 기반 규칙은 필요 없다.
    expanded = _PIECE_REF_RE.sub(replace, previous)
    return expanded, warnings


def _profile_warning(profile: Any, preset_name: str) -> dict | None:
    if not profile:
        return None
    if not isinstance(profile, str):
        return _warning(
            "unsupported_profile",
            f"'{preset_name}' 프로필 값이 문자열이 아니어서 가져오지 않습니다.",
        )
    encoded = profile.split(",", 1)[1] if "," in profile else profile
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        print(
            "[PRESET_IMPORT] 프로필 base64 판별 실패: "
            f"preset={preset_name!r}, error={exc}"
        )
        traceback.print_exc()
        return _warning(
            "unsupported_profile",
            f"'{preset_name}' 프로필 이미지를 해석할 수 없어 가져오지 않습니다.",
        )
    if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return _warning(
            "unsupported_profile",
            f"'{preset_name}' 프로필 값은 내장 PNG가 아니어서 가져오지 않습니다.",
        )
    return _warning(
        "profile_not_imported",
        f"'{preset_name}' 프로필 PNG는 현재 프리셋 저장 구조에 포함하지 않습니다.",
    )


def _iter_generation_presets(raw_presets: Any) -> Iterable[tuple[str, dict]]:
    if isinstance(raw_presets, list):
        for preset in raw_presets:
            if isinstance(preset, dict):
                yield str(preset.get("type") or "legacy"), preset
            else:
                print(
                    "[PRESET_IMPORT] 잘못된 레거시 preset 항목 건너뜀: "
                    f"type={type(preset).__name__}"
                )
        return
    if not isinstance(raw_presets, dict):
        print(f"[PRESET_IMPORT] presets 형식 오류: type={type(raw_presets).__name__}")
        return
    for workflow_type, values in raw_presets.items():
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            print(
                "[PRESET_IMPORT] preset 묶음 형식 오류: "
                f"workflow={workflow_type!r}, type={type(values).__name__}"
            )
            continue
        for preset in values:
            if isinstance(preset, dict):
                yield str(workflow_type), preset
            else:
                print(
                    "[PRESET_IMPORT] 잘못된 preset 항목 건너뜀: "
                    f"workflow={workflow_type!r}, type={type(preset).__name__}"
                )


def _cleanup_sessions() -> None:
    with _SESSION_LOCK:
        now = _datetime.datetime.now(_datetime.timezone.utc).timestamp()
        expired = [
            key for key, value in _ANALYSIS_SESSIONS.items()
            if now - value["created_timestamp"] > SESSION_TTL_SECONDS
        ]
        for key in expired:
            del _ANALYSIS_SESSIONS[key]
        if len(_ANALYSIS_SESSIONS) <= MAX_ANALYSIS_SESSIONS:
            return
        ordered = sorted(
            _ANALYSIS_SESSIONS.items(),
            key=lambda pair: pair[1]["created_timestamp"],
        )
        for key, _ in ordered[: len(_ANALYSIS_SESSIONS) - MAX_ANALYSIS_SESSIONS]:
            del _ANALYSIS_SESSIONS[key]


def get_analysis_session(import_id: str) -> dict:
    with _SESSION_LOCK:
        _cleanup_sessions()
        session = _ANALYSIS_SESSIONS.get(import_id)
        if session is None:
            print(f"[PRESET_IMPORT] 분석 세션 없음 또는 만료: import_id={import_id!r}")
            raise PresetImportError("분석 세션이 없거나 만료되었습니다. 파일을 다시 분석해주세요.")
        if session.get("committed"):
            print(f"[PRESET_IMPORT] 이미 커밋된 분석 세션 재사용 거부: import_id={import_id!r}")
            raise PresetImportError("이미 가져오기를 완료한 분석입니다. 파일을 다시 분석해주세요.")
        return session


def _normalize_target_name(name: str) -> str:
    """새 SDStudio 임포트 대상 이름에서 경로 구분자로 보일 수 있는 /를 제거한다."""
    return name.replace("/", "-")


def analyze_document(filename: str, document: Any) -> dict:
    """SDStudio 세션 JSON을 쓰기 없이 분석하고 편집 가능한 초안을 반환한다."""
    if not isinstance(document, dict):
        print(f"[PRESET_IMPORT] 분석 거부: 루트 type={type(document).__name__}")
        raise PresetImportError("JSON 최상위 값은 객체여야 합니다.")
    name = document.get("name")
    scenes = document.get("scenes")
    if not isinstance(name, str) or not name.strip():
        print(f"[PRESET_IMPORT] 분석 거부: 프로젝트 이름={name!r}")
        raise PresetImportError("SDStudio 프로젝트 이름이 없거나 올바르지 않습니다.")
    if not isinstance(scenes, dict):
        print(f"[PRESET_IMPORT] 분석 거부: scenes type={type(scenes).__name__}")
        raise PresetImportError("SDStudio scenes 값이 객체가 아닙니다.")
    if not isinstance(document.get("presets", {}), (dict, list)):
        print(
            "[PRESET_IMPORT] 분석 거부: presets type="
            f"{type(document.get('presets')).__name__}"
        )
        raise PresetImportError("SDStudio presets 값이 객체나 배열이 아닙니다.")

    import_id = uuid.uuid4().hex
    project_name = name.strip()
    prefix = _normalize_target_name(f"SDstudio-{project_name}-")
    piece_map = _library_piece_map(document)
    prompt_chunks = _prompt_chunk_map(document, piece_map)
    groups: list[dict] = []
    items: list[dict] = []
    warnings: list[dict] = []
    for field, label in (
        ("inpaints", "인페인트 씬"),
        ("characterPresets", "캐릭터 프리셋"),
    ):
        value = document.get(field)
        if value:
            warning = _warning(
                "unsupported_session_section",
                f"{label} 데이터는 현재 태그 프리셋 가져오기 대상이 아닙니다.",
                source_field=field,
            )
            print(f"[PRESET_IMPORT] {warning['message']} field={field}")
            warnings.append(warning)

    for scene_index, (scene_key, scene) in enumerate(scenes.items()):
        group_id = f"scene-{scene_index}"
        if not isinstance(scene, dict):
            warning = _warning(
                "invalid_scene",
                f"씬 '{scene_key}' 값이 객체가 아니어서 건너뜁니다.",
                group_id=group_id,
            )
            print(f"[PRESET_IMPORT] {warning['message']}")
            warnings.append(warning)
            groups.append({
                "id": group_id,
                "kind": "scene",
                "name": str(scene_key),
                "selected": False,
                "item_ids": [],
                "variant_count": 0,
                "warnings": [warning],
            })
            continue
        scene_name = str(scene.get("name") or scene_key)
        slots = scene.get("slots", [])
        group_warnings: list[dict] = []
        enabled_slots: list[list[dict]] = []
        if not isinstance(slots, list) or not slots:
            group_warnings.append(_warning(
                "empty_scene_slots",
                f"씬 '{scene_name}'에 프롬프트 슬롯이 없습니다.",
                group_id=group_id,
            ))
        else:
            for slot_index, slot in enumerate(slots):
                if not isinstance(slot, list):
                    group_warnings.append(_warning(
                        "invalid_scene_slot",
                        f"씬 '{scene_name}'의 {slot_index + 1}번 슬롯이 배열이 아닙니다.",
                        group_id=group_id,
                    ))
                    enabled_slots = []
                    break
                enabled = [
                    piece for piece in slot
                    if isinstance(piece, dict) and piece.get("enabled", True) is not False
                ]
                bad_count = len(slot) - sum(isinstance(piece, dict) for piece in slot)
                if bad_count:
                    group_warnings.append(_warning(
                        "invalid_prompt_piece",
                        f"씬 '{scene_name}'의 {slot_index + 1}번 슬롯에서 잘못된 항목 {bad_count}개를 제외했습니다.",
                        group_id=group_id,
                    ))
                if not enabled:
                    group_warnings.append(_warning(
                        "empty_enabled_slot",
                        f"씬 '{scene_name}'의 {slot_index + 1}번 슬롯에 활성 프롬프트가 없습니다.",
                        group_id=group_id,
                    ))
                    enabled_slots = []
                    break
                enabled_slots.append(enabled)

        combination_count = 0
        if enabled_slots:
            combination_count = 1
            for slot in enabled_slots:
                combination_count *= len(slot)
        group_item_ids: list[str] = []
        if combination_count > MAX_VARIANTS_PER_GROUP:
            group_warnings.append(_warning(
                "variant_limit_exceeded",
                (
                    f"씬 '{scene_name}'의 조합이 {combination_count}개라 자동 전개 한도 "
                    f"{MAX_VARIANTS_PER_GROUP}개를 초과합니다."
                ),
                group_id=group_id,
            ))
        elif combination_count:
            for variant_index, combination in enumerate(itertools.product(*enabled_slots)):
                item_id = f"{group_id}-item-{variant_index}"
                raw_parts = []
                source_piece_ids = []
                item_warnings: list[dict] = []
                for piece in combination:
                    prompt = piece.get("prompt", "")
                    if not isinstance(prompt, str):
                        item_warnings.append(_warning(
                            "invalid_prompt",
                            f"씬 '{scene_name}'에 문자열이 아닌 프롬프트가 있습니다.",
                            item_id=item_id,
                        ))
                        prompt = str(prompt) if prompt is not None else ""
                    raw_parts.append(prompt)
                    if piece.get("id") is not None:
                        source_piece_ids.append(str(piece.get("id")))
                    if piece.get("characterPrompts"):
                        item_warnings.append(_warning(
                            "character_prompts_not_imported",
                            f"씬 '{scene_name}'의 characterPrompts는 현재 프리셋에 포함하지 않습니다.",
                            item_id=item_id,
                        ))
                source_prompt = ", ".join(part.strip(" ,\r\n") for part in raw_parts if part.strip(" ,\r\n"))
                expanded_prompt, reference_warnings = _expand_piece_references(source_prompt, piece_map)
                for warning in reference_warnings:
                    warning["item_id"] = item_id
                item_warnings.extend(reference_warnings)
                converted_prompt = _convert_nai_text(
                    expanded_prompt,
                    prompt_chunks,
                    item_id=item_id,
                    source_field="scene_prompt",
                )
                item_warnings.extend(converted_prompt["warnings"])
                fragments = _converted_fragments(
                    converted_prompt,
                    item_id=item_id,
                    source_field="scene_prompt",
                )
                canonical_prompts = {"scene_prompt": converted_prompt["canonical"]}
                structured_prompts = {"scene_prompt": converted_prompt["structured"]}
                expanded_sections = [expanded_prompt]
                target_sections = [converted_prompt["prompt"]]
                scene_uc = scene.get("sceneCharacterUC", "")
                if scene_uc is None:
                    scene_uc = ""
                if not isinstance(scene_uc, str):
                    item_warnings.append(_warning(
                        "invalid_scene_character_uc",
                        f"씬 '{scene_name}'의 sceneCharacterUC가 문자열이 아니어서 제외합니다.",
                        item_id=item_id,
                    ))
                elif scene_uc.strip():
                    expanded_uc, uc_reference_warnings = _expand_piece_references(
                        scene_uc,
                        piece_map,
                    )
                    for warning in uc_reference_warnings:
                        warning.update({"item_id": item_id, "source_field": "sceneCharacterUC"})
                    item_warnings.extend(uc_reference_warnings)
                    converted_uc = _convert_nai_text(
                        expanded_uc,
                        prompt_chunks,
                        item_id=item_id,
                        source_field="sceneCharacterUC",
                    )
                    item_warnings.extend(converted_uc["warnings"])
                    fragments.extend(_converted_fragments(
                        converted_uc,
                        item_id=item_id,
                        source_field="sceneCharacterUC",
                        start_index=len(fragments),
                        forced_category="character_negative_presets",
                    ))
                    canonical_prompts["sceneCharacterUC"] = converted_uc["canonical"]
                    structured_prompts["sceneCharacterUC"] = converted_uc["structured"]
                    expanded_sections.append(f"[sceneCharacterUC] {expanded_uc}")
                    target_sections.append(f"[sceneCharacterUC] {converted_uc['prompt']}")
                if scene.get("sceneCharacterPrompts"):
                    item_warnings.append(_warning(
                        "scene_character_prompts_not_imported",
                        f"씬 '{scene_name}'의 캐릭터별 구조화 프롬프트는 현재 프리셋에 포함하지 않습니다.",
                        item_id=item_id,
                    ))
                target_name = _normalize_target_name(prefix + scene_name.strip())
                if variant_index:
                    target_name += f"_v{variant_index}"
                item = {
                    "id": item_id,
                    "group_id": group_id,
                    "source_kind": "scene",
                    "source_name": scene_name,
                    "variant_index": variant_index,
                    "target_name": target_name,
                    "selected": bool(fragments),
                    "source_prompt": (
                        source_prompt
                        + (f"\n[sceneCharacterUC] {scene_uc}" if isinstance(scene_uc, str) and scene_uc.strip() else "")
                    ),
                    "expanded_prompt": "\n".join(expanded_sections),
                    "target_prompt": "\n".join(target_sections),
                    "canonical_prompts": canonical_prompts,
                    "structured_prompts": structured_prompts,
                    "source_piece_ids": source_piece_ids,
                    "allowed_categories": list(SCENE_LLM_CATEGORIES),
                    "fragments": fragments,
                    "warnings": item_warnings,
                }
                items.append(item)
                group_item_ids.append(item_id)
                warnings.extend(item_warnings)
                if len(items) > MAX_TOTAL_ITEMS:
                    print(
                        "[PRESET_IMPORT] 전체 변형 한도 초과: "
                        f"count={len(items)}, limit={MAX_TOTAL_ITEMS}"
                    )
                    raise PresetImportError(
                        f"전체 프롬프트 변형이 {MAX_TOTAL_ITEMS}개를 초과합니다."
                    )

        groups.append({
            "id": group_id,
            "kind": "scene",
            "name": scene_name,
            "selected": bool(group_item_ids),
            "item_ids": group_item_ids,
            "variant_count": combination_count,
            "slot_count": len(enabled_slots),
            "warnings": group_warnings,
        })
        warnings.extend(group_warnings)

    generation_index = 0
    for workflow_type, preset in _iter_generation_presets(document.get("presets", {})):
        preset_name = str(preset.get("name") or f"preset-{generation_index + 1}")
        group_id = f"preset-{generation_index}"
        item_id = f"{group_id}-item-0"
        group_warnings: list[dict] = []
        profile_warning = _profile_warning(preset.get("profile"), preset_name)
        if profile_warning:
            profile_warning.update({"group_id": group_id, "item_id": item_id})
            group_warnings.append(profile_warning)
        for unsupported_field, unsupported_label in (
            ("vibes", "Vibe 이미지"),
            ("characterReferences", "캐릭터 참조 이미지"),
        ):
            if preset.get(unsupported_field):
                warning = _warning(
                    "unsupported_preset_media",
                    f"'{preset_name}'의 {unsupported_label}는 태그 프리셋에 포함하지 않습니다.",
                    group_id=group_id,
                    item_id=item_id,
                    source_field=unsupported_field,
                )
                print(f"[PRESET_IMPORT] {warning['message']}")
                group_warnings.append(warning)
        fragments: list[dict] = []
        source_sections: list[str] = []
        expanded_sections: list[str] = []
        target_sections: list[str] = []
        canonical_prompts: dict[str, dict] = {}
        structured_prompts: dict[str, dict] = {}
        for field in ("frontPrompt", "backPrompt", "uc"):
            value = preset.get(field, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                warning = _warning(
                    "invalid_preset_prompt",
                    f"'{preset_name}'의 {field} 값이 문자열이 아니어서 제외합니다.",
                    group_id=group_id,
                    item_id=item_id,
                )
                print(f"[PRESET_IMPORT] {warning['message']}")
                group_warnings.append(warning)
                continue
            if not value.strip():
                continue
            source_sections.append(f"[{field}] {value}")
            expanded_value, reference_warnings = _expand_piece_references(value, piece_map)
            for warning in reference_warnings:
                warning.update({"group_id": group_id, "item_id": item_id, "source_field": field})
            group_warnings.extend(reference_warnings)
            converted = _convert_nai_text(
                expanded_value,
                prompt_chunks,
                item_id=item_id,
                source_field=field,
            )
            group_warnings.extend(converted["warnings"])
            fragments.extend(_converted_fragments(
                converted,
                item_id=item_id,
                source_field=field,
                start_index=len(fragments),
                forced_category="negative_presets" if field == "uc" else None,
            ))
            expanded_sections.append(f"[{field}] {expanded_value}")
            target_sections.append(f"[{field}] {converted['prompt']}")
            canonical_prompts[field] = converted["canonical"]
            structured_prompts[field] = converted["structured"]
        target_name = _normalize_target_name(prefix + preset_name.strip())
        item = {
            "id": item_id,
            "group_id": group_id,
            "source_kind": "generation_preset",
            "source_name": preset_name,
            "workflow_type": workflow_type,
            "variant_index": 0,
            "target_name": target_name,
            "selected": bool(fragments),
            "source_prompt": "\n".join(source_sections),
            "expanded_prompt": "\n".join(expanded_sections),
            "target_prompt": "\n".join(target_sections),
            "canonical_prompts": canonical_prompts,
            "structured_prompts": structured_prompts,
            "source_piece_ids": [],
            "allowed_categories": list(PRESET_LLM_CATEGORIES),
            "fragments": fragments,
            "warnings": copy.deepcopy(group_warnings),
        }
        items.append(item)
        groups.append({
            "id": group_id,
            "kind": "generation_preset",
            "name": preset_name,
            "workflow_type": workflow_type,
            "selected": bool(fragments),
            "item_ids": [item_id],
            "variant_count": 1 if fragments else 0,
            "slot_count": 1,
            "warnings": group_warnings,
        })
        warnings.extend(group_warnings)
        generation_index += 1
        if len(items) > MAX_TOTAL_ITEMS:
            print(
                "[PRESET_IMPORT] 전체 항목 한도 초과: "
                f"count={len(items)}, limit={MAX_TOTAL_ITEMS}"
            )
            raise PresetImportError(f"전체 프롬프트 항목이 {MAX_TOTAL_ITEMS}개를 초과합니다.")

    shared_group_count = 0
    raw_shareds = document.get("presetShareds", {})
    if raw_shareds is None:
        raw_shareds = {}
    if not isinstance(raw_shareds, dict):
        warning = _warning(
            "invalid_preset_shareds",
            "presetShareds 값이 객체가 아니어서 공유 프롬프트를 가져오지 않습니다.",
        )
        print(
            "[PRESET_IMPORT] presetShareds 형식 오류: "
            f"type={type(raw_shareds).__name__}"
        )
        warnings.append(warning)
    else:
        for shared_index, (workflow_type, shared) in enumerate(raw_shareds.items()):
            if not isinstance(shared, dict):
                warning = _warning(
                    "invalid_shared_preset",
                    f"'{workflow_type}' 공유 설정이 객체가 아니어서 가져오지 않습니다.",
                    source_field="presetShareds",
                )
                print(f"[PRESET_IMPORT] {warning['message']}")
                warnings.append(warning)
                continue
            group_id = f"shared-{shared_index}"
            item_id = f"{group_id}-item-0"
            shared_name = f"공유 설정 · {workflow_type}"
            group_warnings: list[dict] = []
            fragments: list[dict] = []
            source_sections: list[str] = []
            expanded_sections: list[str] = []
            target_sections: list[str] = []
            canonical_prompts: dict[str, dict] = {}
            structured_prompts: dict[str, dict] = {}
            for field in ("characterPrompt", "backgroundPrompt", "uc"):
                value = shared.get(field, "")
                if value is None:
                    value = ""
                if not isinstance(value, str):
                    warning = _warning(
                        "invalid_shared_prompt",
                        f"'{workflow_type}' 공유 설정의 {field} 값이 문자열이 아니어서 제외합니다.",
                        group_id=group_id,
                        item_id=item_id,
                        source_field=field,
                    )
                    print(f"[PRESET_IMPORT] {warning['message']}")
                    group_warnings.append(warning)
                    continue
                if not value.strip():
                    continue
                source_sections.append(f"[{field}] {value}")
                expanded_value, reference_warnings = _expand_piece_references(value, piece_map)
                for warning in reference_warnings:
                    warning.update({
                        "group_id": group_id,
                        "item_id": item_id,
                        "source_field": field,
                    })
                group_warnings.extend(reference_warnings)
                converted = _convert_nai_text(
                    expanded_value,
                    prompt_chunks,
                    item_id=item_id,
                    source_field=field,
                )
                group_warnings.extend(converted["warnings"])
                fragments.extend(_converted_fragments(
                    converted,
                    item_id=item_id,
                    source_field=field,
                    start_index=len(fragments),
                    forced_category="negative_presets" if field == "uc" else None,
                ))
                expanded_sections.append(f"[{field}] {expanded_value}")
                target_sections.append(f"[{field}] {converted['prompt']}")
                canonical_prompts[field] = converted["canonical"]
                structured_prompts[field] = converted["structured"]
            for unsupported_field, unsupported_label in (
                ("characterPrompts", "캐릭터별 구조화 프롬프트"),
                ("vibes", "Vibe 이미지"),
                ("characterReferences", "캐릭터 참조 이미지"),
            ):
                if shared.get(unsupported_field):
                    warning = _warning(
                        "unsupported_shared_data",
                        f"'{workflow_type}' 공유 설정의 {unsupported_label}는 태그 프리셋에 포함하지 않습니다.",
                        group_id=group_id,
                        item_id=item_id,
                        source_field=unsupported_field,
                    )
                    print(f"[PRESET_IMPORT] {warning['message']}")
                    group_warnings.append(warning)
            if not fragments and not group_warnings:
                continue
            target_name = _normalize_target_name(prefix + shared_name)
            item = {
                "id": item_id,
                "group_id": group_id,
                "source_kind": "workflow_shared",
                "source_name": shared_name,
                "workflow_type": str(workflow_type),
                "variant_index": 0,
                "target_name": target_name,
                "selected": bool(fragments),
                "source_prompt": "\n".join(source_sections),
                "expanded_prompt": "\n".join(expanded_sections),
                "target_prompt": "\n".join(target_sections),
                "canonical_prompts": canonical_prompts,
                "structured_prompts": structured_prompts,
                "source_piece_ids": [],
                "allowed_categories": list(PRESET_LLM_CATEGORIES),
                "fragments": fragments,
                "warnings": copy.deepcopy(group_warnings),
            }
            items.append(item)
            groups.append({
                "id": group_id,
                "kind": "workflow_shared",
                "name": shared_name,
                "workflow_type": str(workflow_type),
                "selected": bool(fragments),
                "item_ids": [item_id],
                "variant_count": 1 if fragments else 0,
                "slot_count": 1,
                "warnings": group_warnings,
            })
            warnings.extend(group_warnings)
            shared_group_count += 1
            if len(items) > MAX_TOTAL_ITEMS:
                print(
                    "[PRESET_IMPORT] 전체 항목 한도 초과: "
                    f"count={len(items)}, limit={MAX_TOTAL_ITEMS}"
                )
                raise PresetImportError(
                    f"전체 프롬프트 항목이 {MAX_TOTAL_ITEMS}개를 초과합니다."
                )

    fragment_count = sum(len(item["fragments"]) for item in items)
    llm_fragment_count = sum(
        1 for item in items for fragment in item["fragments"]
        if fragment["llm_eligible"]
    )
    analysis = {
        "success": True,
        "format": "sdstudio_session",
        "import_id": import_id,
        "source": {
            "filename": str(filename or "preset.json"),
            "name": project_name,
            "version": document.get("version"),
            "sha256": _normalized_document_hash(document),
            "prompt_syntax": "nai",
        },
        "target": {
            "adapter": "anima",
            "prompt_syntax": "comfy_explicit_weight",
            "max_abs_weight": 1.5,
            "weight_quantum": 0.1,
            "weight_rounding": "ROUND_HALF_UP",
            "nai_step": 1.05,
            "llm_batch_fragment_limit": MAX_CLASSIFY_FRAGMENTS,
        },
        "categories": list(IMPORT_CATEGORIES),
        "groups": groups,
        "items": items,
        "warnings": warnings,
        "summary": {
            "scene_group_count": sum(group["kind"] == "scene" for group in groups),
            "generation_group_count": sum(group["kind"] == "generation_preset" for group in groups),
            "shared_group_count": shared_group_count,
            "scene_item_count": sum(item["source_kind"] == "scene" for item in items),
            "generation_item_count": sum(item["source_kind"] == "generation_preset" for item in items),
            "shared_item_count": sum(item["source_kind"] == "workflow_shared" for item in items),
            "fragment_count": fragment_count,
            "llm_fragment_count": llm_fragment_count,
            "warning_count": len(warnings),
        },
    }
    created_timestamp = _datetime.datetime.now(_datetime.timezone.utc).timestamp()
    with _SESSION_LOCK:
        _cleanup_sessions()
        _ANALYSIS_SESSIONS[import_id] = {
            "created_timestamp": created_timestamp,
            "analysis": copy.deepcopy(analysis),
            "items_by_id": {item["id"]: copy.deepcopy(item) for item in items},
            "committed": False,
        }
        _cleanup_sessions()
    print(
        "[PRESET_IMPORT] 분석 완료: "
        f"import_id={import_id}, source={project_name!r}, groups={len(groups)}, "
        f"items={len(items)}, fragments={fragment_count}, warnings={len(warnings)}"
    )
    return analysis


def build_classification_payload(import_id: str, targets: Any) -> dict:
    session = get_analysis_session(import_id)
    if not isinstance(targets, list) or not targets:
        print(f"[PRESET_IMPORT] LLM 분류 대상 누락: import_id={import_id}, targets={targets!r}")
        raise PresetImportError("LLM으로 분석할 태그 조각을 선택해주세요.")

    payload_items = []
    seen_items: set[str] = set()
    target_fragment_count = 0
    for target in targets:
        # 이전 내부 호출 형식은 30조각 이하일 때만 안전하게 받아들인다.
        if isinstance(target, str):
            item_id = target
            requested_fragment_ids = None
        elif isinstance(target, dict):
            item_id = target.get("item_id")
            requested_fragment_ids = target.get("fragment_ids")
        else:
            print(f"[PRESET_IMPORT] LLM 분류 대상 형식 오류: target={target!r}")
            raise PresetImportError("LLM 분류 대상 형식이 올바르지 않습니다.")
        if not isinstance(item_id, str) or not item_id or item_id in seen_items:
            print(f"[PRESET_IMPORT] LLM 분류 item_id 오류 또는 중복: item_id={item_id!r}")
            raise PresetImportError("LLM 분석 항목 ID가 없거나 중복되었습니다.")
        seen_items.add(item_id)
        item = session["items_by_id"].get(item_id)
        if item is None:
            print(
                "[PRESET_IMPORT] LLM 분류 알 수 없는 항목: "
                f"import_id={import_id}, item_id={item_id!r}"
            )
            raise PresetImportError(f"분석 세션에 없는 항목입니다: {item_id}")
        eligible_by_id = {
            fragment["id"]: fragment
            for fragment in item["fragments"]
            if fragment["llm_eligible"]
        }
        if requested_fragment_ids is None:
            fragment_ids = list(eligible_by_id)
        else:
            if not isinstance(requested_fragment_ids, list) or not requested_fragment_ids:
                print(
                    "[PRESET_IMPORT] LLM fragment_ids 형식 오류: "
                    f"item_id={item_id}, fragment_ids={requested_fragment_ids!r}"
                )
                raise PresetImportError("LLM 분석 태그 조각 ID 목록이 올바르지 않습니다.")
            if len(set(requested_fragment_ids)) != len(requested_fragment_ids):
                print(
                    "[PRESET_IMPORT] LLM fragment_id 중복: "
                    f"item_id={item_id}, fragment_ids={requested_fragment_ids!r}"
                )
                raise PresetImportError("LLM 분석 태그 조각 ID가 중복되었습니다.")
            unknown = [
                fragment_id for fragment_id in requested_fragment_ids
                if fragment_id not in eligible_by_id
            ]
            if unknown:
                print(
                    "[PRESET_IMPORT] LLM 분석 불가 fragment_id: "
                    f"item_id={item_id}, fragment_ids={unknown!r}"
                )
                raise PresetImportError(
                    f"분석 세션에 없거나 LLM 대상이 아닌 태그 조각입니다: {unknown[0]}"
                )
            fragment_ids = requested_fragment_ids
        eligible = [eligible_by_id[fragment_id] for fragment_id in fragment_ids]
        if not eligible:
            print(f"[PRESET_IMPORT] LLM 분류 스킵 대상 포함: item_id={item_id}")
            continue
        target_fragment_count += len(eligible)
        payload_items.append({
            "item_id": item_id,
            "source_kind": item["source_kind"],
            "group_name": item["source_name"],
            "source_prompt_syntax": "NAI",
            "target_prompt_syntax": "ANIMA / Comfy explicit weight",
            "full_source_prompt_nai": item["source_prompt"],
            "full_target_prompt_anima": item.get("target_prompt", item["expanded_prompt"]),
            "allowed_categories": item["allowed_categories"],
            "context_fragment_count": len(item["fragments"]),
            "fragments": [
                {
                    "fragment_id": fragment["id"],
                    "source_nai": fragment["original_text"],
                    "text": fragment.get("import_text", fragment["text"]),
                    "source_region": fragment.get("source_region", "base"),
                }
                for fragment in eligible
            ],
        })
    if target_fragment_count > MAX_CLASSIFY_FRAGMENTS:
        print(
            "[PRESET_IMPORT] LLM 분류 태그 조각 한도 초과: "
            f"count={target_fragment_count}, limit={MAX_CLASSIFY_FRAGMENTS}"
        )
        raise PresetImportError(
            f"한 번에 분석할 수 있는 태그 조각은 {MAX_CLASSIFY_FRAGMENTS}개입니다."
        )
    if not payload_items:
        print(f"[PRESET_IMPORT] LLM 분류 가능한 조각 없음: targets={targets!r}")
        raise PresetImportError("선택 항목에 LLM이 분류할 조각이 없습니다.")
    return {
        "source_name": session["analysis"]["source"]["name"],
        "source_prompt_syntax": "NAI",
        "target_adapter": "ANIMA",
        "max_abs_weight": 1.5,
        "weight_quantum": 0.1,
        "weight_rounding": "ROUND_HALF_UP",
        "target_fragment_count": target_fragment_count,
        "items": payload_items,
    }


def build_classification_messages(payload: dict) -> list[dict]:
    system = """You classify converted NovelAI/NAI prompt fragments for a preset manager.

The source prompt syntax is NAI. Before this call, a deterministic parser expanded project references and NAI Prompt Chunks, parsed NAI emphasis ({}, [], and numeric ::), converted artist:name to ANIMA @name, and rendered Comfy weights after clamping to an absolute maximum of 1.5 and rounding to one decimal with ROUND_HALF_UP. Treat parentheses ending in a numeric :weight as emphasis; ordinary names such as "muji (uimss)" are plain tag text. Exact pre-rounding weights remain in canonical metadata for provenance.

Read each item's group name, full NAI source prompt, full ANIMA target prompt, and target fragments as natural context. A request contains at most 30 target fragments and may contain only part of a larger item. Decide from meaning and role in context; do not use hard-coded keyword matching.

Category meanings:
- expressions: emotion, facial expression, gaze nuance, or expressive gesture/body language.
- composition_presets: camera, framing, viewpoint, subject count, spatial staging, or general scene composition.
- appearances: persistent physical traits of a depicted character.
- outfits: clothing and wearable items.
- artist_presets: artist/style attribution.
- quality_presets: quality, aesthetic, resolution, or rendering-quality instructions.
- negative_presets: general negative prompt content.
- character_negative_presets: character-specific negative traits.
- unassigned: genuinely ambiguous content that needs human review.

For each item, use only its allowed_categories. Return every provided fragment_id exactly once. Never add, remove, merge, split, translate, normalize, or rewrite fragments. Classify the converted `text`, using `source_nai` only as syntax/provenance context. Output JSON only with this shape:
{"items":[{"item_id":"...","assignments":[{"fragment_id":"...","category":"..."}]}]}"""
    user = (
        "다음 SDStudio 프롬프트 조각을 문맥에 따라 분류하세요. 원문은 바꾸지 말고 "
        "ID와 카테고리만 반환하세요.\n\n입력 데이터:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_classification_response(parsed: Any, payload: dict) -> tuple[bool, str]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        return False, "LLM 응답의 items가 배열이 아닙니다."
    expected_items = {item["item_id"]: item for item in payload["items"]}
    actual_items = parsed["items"]
    if len(actual_items) != len(expected_items):
        return False, "LLM 응답 항목 수가 요청과 다릅니다."
    seen_items: set[str] = set()
    for result in actual_items:
        if not isinstance(result, dict):
            return False, "LLM 응답 항목이 객체가 아닙니다."
        item_id = result.get("item_id")
        if item_id not in expected_items or item_id in seen_items:
            return False, f"알 수 없거나 중복된 item_id입니다: {item_id!r}"
        seen_items.add(item_id)
        assignments = result.get("assignments")
        if not isinstance(assignments, list):
            return False, f"{item_id}의 assignments가 배열이 아닙니다."
        expected_fragments = {
            fragment["fragment_id"] for fragment in expected_items[item_id]["fragments"]
        }
        allowed = set(expected_items[item_id]["allowed_categories"])
        seen_fragments: set[str] = set()
        for assignment in assignments:
            if not isinstance(assignment, dict):
                return False, f"{item_id}의 assignment가 객체가 아닙니다."
            fragment_id = assignment.get("fragment_id")
            category = assignment.get("category")
            if fragment_id not in expected_fragments or fragment_id in seen_fragments:
                return False, f"알 수 없거나 중복된 fragment_id입니다: {fragment_id!r}"
            if category not in allowed:
                return False, f"{fragment_id}의 허용되지 않은 카테고리입니다: {category!r}"
            seen_fragments.add(fragment_id)
        if seen_fragments != expected_fragments:
            missing = sorted(expected_fragments - seen_fragments)
            return False, f"{item_id}에서 누락된 fragment_id가 있습니다: {missing}"
    if seen_items != set(expected_items):
        return False, "LLM 응답에서 누락된 item_id가 있습니다."
    return True, ""


def classification_assignments(parsed: dict) -> dict[str, dict[str, str]]:
    return {
        item["item_id"]: {
            assignment["fragment_id"]: assignment["category"]
            for assignment in item["assignments"]
        }
        for item in parsed["items"]
    }


def _validate_target_name(name: Any, label: str) -> tuple[str | None, str | None]:
    if not isinstance(name, str) or not name.strip():
        return None, f"{label} 이름이 비어 있습니다."
    clean = _normalize_target_name(name.strip())
    if len(clean) > 240:
        return None, f"{label} 이름이 240자를 초과합니다."
    if any(ord(char) < 32 for char in clean):
        return None, f"{label} 이름에 제어 문자가 포함되어 있습니다."
    return clean, None


def validate_draft(
    draft: Any,
    active_tags: Any,
    hidden_tags: Any,
) -> dict:
    """편집 초안을 원본 분석 세션과 대조하고 저장 레코드/충돌을 만든다."""
    errors: list[dict] = []
    warnings: list[dict] = []
    if not isinstance(draft, dict):
        print(f"[PRESET_IMPORT] 초안 형식 오류: type={type(draft).__name__}")
        return {
            "success": False,
            "errors": [_warning("invalid_draft", "가져오기 초안이 객체가 아닙니다.")],
            "warnings": [],
            "records": [],
            "conflicts": [],
            "summary": {},
        }
    import_id = draft.get("import_id", "")
    try:
        session = get_analysis_session(import_id)
    except PresetImportError as exc:
        return {
            "success": False,
            "errors": [_warning("missing_session", str(exc))],
            "warnings": [],
            "records": [],
            "conflicts": [],
            "summary": {},
        }
    raw_items = draft.get("items")
    if not isinstance(raw_items, list):
        print(f"[PRESET_IMPORT] 초안 items 형식 오류: type={type(raw_items).__name__}")
        return {
            "success": False,
            "errors": [_warning("invalid_items", "가져오기 items가 배열이 아닙니다.")],
            "warnings": [],
            "records": [],
            "conflicts": [],
            "summary": {},
        }
    if not isinstance(active_tags, dict) or not isinstance(hidden_tags, dict):
        print(
            "[PRESET_IMPORT] 현재 태그 데이터 형식 오류: "
            f"active={type(active_tags).__name__}, hidden={type(hidden_tags).__name__}"
        )
        return {
            "success": False,
            "errors": [_warning("invalid_current_tags", "현재 프리셋 저장소 구조가 올바르지 않습니다.")],
            "warnings": [],
            "records": [],
            "conflicts": [],
            "summary": {},
        }

    draft_by_id: dict[str, dict] = {}
    for item in raw_items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            errors.append(_warning("invalid_item", "가져오기 항목 형식이 올바르지 않습니다."))
            continue
        if item["id"] in draft_by_id:
            errors.append(_warning("duplicate_item", f"항목 ID가 중복되었습니다: {item['id']}"))
            continue
        if item["id"] not in session["items_by_id"]:
            errors.append(_warning("unknown_item", f"분석 원본에 없는 항목입니다: {item['id']}"))
            continue
        draft_by_id[item["id"]] = item

    missing_items = set(session["items_by_id"]) - set(draft_by_id)
    if missing_items:
        errors.append(_warning(
            "missing_items",
            f"분석 원본 항목 {len(missing_items)}개가 초안에서 누락되었습니다.",
            item_ids=sorted(missing_items),
        ))

    records: list[dict] = []
    record_origins: dict[tuple[str, str], str] = {}
    selected_count = 0
    edited_fragment_count = 0
    excluded_fragment_count = 0
    unassigned_count = 0

    for item_id, original in session["items_by_id"].items():
        draft_item = draft_by_id.get(item_id)
        if draft_item is None or draft_item.get("selected") is not True:
            continue
        selected_count += 1
        target_name, name_error = _validate_target_name(
            draft_item.get("target_name"),
            f"'{original['source_name']}'",
        )
        if name_error:
            errors.append(_warning("invalid_target_name", name_error, item_id=item_id))
            continue
        fragments = draft_item.get("fragments")
        if not isinstance(fragments, list):
            errors.append(_warning(
                "invalid_fragments",
                f"'{target_name}'의 fragments가 배열이 아닙니다.",
                item_id=item_id,
            ))
            continue
        expected = {fragment["id"]: fragment for fragment in original["fragments"]}
        seen: set[str] = set()
        buckets: dict[str, list[str]] = {category: [] for category in IMPORT_CATEGORIES}
        for fragment_index, fragment in enumerate(fragments):
            if not isinstance(fragment, dict) or not isinstance(fragment.get("id"), str):
                errors.append(_warning(
                    "invalid_fragment",
                    f"'{target_name}'의 {fragment_index + 1}번 조각 형식이 올바르지 않습니다.",
                    item_id=item_id,
                ))
                continue
            fragment_id = fragment["id"]
            if fragment_id in seen:
                errors.append(_warning(
                    "duplicate_fragment",
                    f"'{target_name}'에서 조각 ID가 중복되었습니다: {fragment_id}",
                    item_id=item_id,
                ))
                continue
            seen.add(fragment_id)
            is_source = fragment_id in expected
            if not is_source and not fragment_id.startswith("user-"):
                errors.append(_warning(
                    "unknown_fragment",
                    f"'{target_name}'에 출처를 확인할 수 없는 조각이 있습니다: {fragment_id}",
                    item_id=item_id,
                ))
                continue
            origin = fragment.get("origin")
            if is_source and origin not in (None, "source"):
                errors.append(_warning(
                    "invalid_source_origin",
                    f"'{target_name}'의 원본 태그 출처 표시가 바뀌었습니다: {fragment_id}",
                    item_id=item_id,
                    fragment_id=fragment_id,
                ))
                continue
            if not is_source and origin != "user":
                errors.append(_warning(
                    "invalid_user_origin",
                    f"'{target_name}'의 사용자 태그 출처 표시가 올바르지 않습니다: {fragment_id}",
                    item_id=item_id,
                    fragment_id=fragment_id,
                ))
                continue
            llm_category = fragment.get("llm_category")
            if llm_category is not None and llm_category not in (*IMPORT_CATEGORIES, "unassigned"):
                errors.append(_warning(
                    "invalid_llm_category",
                    f"'{target_name}'의 LLM 카테고리 기록이 올바르지 않습니다: {llm_category!r}",
                    item_id=item_id,
                    fragment_id=fragment_id,
                ))
                continue
            excluded = fragment.get("excluded") is True
            if excluded:
                if is_source:
                    excluded_fragment_count += 1
                continue
            text = fragment.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(_warning(
                    "empty_fragment",
                    f"'{target_name}'에 비어 있는 태그 조각이 있습니다.",
                    item_id=item_id,
                    fragment_id=fragment_id,
                ))
                continue
            if len(text) > 10000:
                errors.append(_warning(
                    "fragment_too_long",
                    f"'{target_name}'의 태그 조각이 10,000자를 초과합니다.",
                    item_id=item_id,
                    fragment_id=fragment_id,
                ))
                continue
            category = fragment.get("category")
            if category == "unassigned":
                unassigned_count += 1
                errors.append(_warning(
                    "unassigned_fragment",
                    f"'{target_name}'에 아직 분류하지 않은 태그가 있습니다: {text.strip()}",
                    item_id=item_id,
                    fragment_id=fragment_id,
                ))
                continue
            if category not in IMPORT_CATEGORIES:
                errors.append(_warning(
                    "invalid_category",
                    f"'{target_name}' 태그의 카테고리가 올바르지 않습니다: {category!r}",
                    item_id=item_id,
                    fragment_id=fragment_id,
                ))
                continue
            clean_text = text.strip()
            buckets[category].append(clean_text)
            expected_import_text = expected[fragment_id].get(
                "import_text",
                expected[fragment_id]["original_text"],
            ) if is_source else ""
            if is_source and clean_text != expected_import_text:
                edited_fragment_count += 1

        missing_fragments = set(expected) - seen
        if missing_fragments:
            errors.append(_warning(
                "missing_fragments",
                f"'{target_name}'에서 원본 태그 {len(missing_fragments)}개가 사라졌습니다.",
                item_id=item_id,
                fragment_ids=sorted(missing_fragments),
            ))
        produced = 0
        for category, values in buckets.items():
            if not values:
                continue
            produced += 1
            key = (category, target_name)
            if key in record_origins:
                errors.append(_warning(
                    "duplicate_target",
                    (
                        f"같은 대상 프리셋이 여러 항목에서 생성됩니다: "
                        f"{category}/{target_name}"
                    ),
                    item_id=item_id,
                    other_item_id=record_origins[key],
                ))
                continue
            record_origins[key] = item_id
            record_id = hashlib.sha256(
                f"{category}\0{target_name}".encode("utf-8")
            ).hexdigest()[:20]
            records.append({
                "id": record_id,
                "category": category,
                "name": target_name,
                "values": values,
                "item_id": item_id,
                "group_id": original["group_id"],
                "source_kind": original["source_kind"],
                "source_name": original["source_name"],
            })
        if produced == 0 and not any(error.get("item_id") == item_id for error in errors):
            errors.append(_warning(
                "empty_selected_item",
                f"선택한 항목 '{target_name}'에서 저장할 태그가 없습니다.",
                item_id=item_id,
            ))

    if selected_count == 0:
        errors.append(_warning("nothing_selected", "가져올 항목을 하나 이상 선택해주세요."))

    conflicts: list[dict] = []
    for record in records:
        category = record["category"]
        name = record["name"]
        active_category = active_tags.get(category, {})
        hidden_category = hidden_tags.get(category, {})
        if not isinstance(active_category, dict) or not isinstance(hidden_category, dict):
            errors.append(_warning(
                "invalid_category_store",
                f"현재 '{category}' 저장소가 객체가 아닙니다.",
                category=category,
            ))
            continue
        in_active = name in active_category
        in_hidden = name in hidden_category
        if in_active and in_hidden:
            errors.append(_warning(
                "duplicate_existing_state",
                f"'{category}/{name}'이 활성과 숨김에 동시에 존재합니다.",
                record_id=record["id"],
            ))
            continue
        if not in_active and not in_hidden:
            record["status"] = "new"
            continue
        state = "active" if in_active else "hidden"
        existing = active_category[name] if in_active else hidden_category[name]
        same = existing == record["values"]
        record["status"] = "same" if same else "conflict"
        record["existing_state"] = state
        if same:
            continue
        conflicts.append({
            "record_id": record["id"],
            "category": category,
            "name": name,
            "state": state,
            "existing": existing,
            "incoming": record["values"],
        })

    result = {
        "success": not errors,
        "import_id": import_id,
        "source": copy.deepcopy(session["analysis"]["source"]),
        "errors": errors,
        "warnings": warnings,
        "records": records,
        "conflicts": conflicts,
        "summary": {
            "selected_item_count": selected_count,
            "record_count": len(records),
            "conflict_count": len(conflicts),
            "new_count": sum(record.get("status") == "new" for record in records),
            "same_count": sum(record.get("status") == "same" for record in records),
            "edited_fragment_count": edited_fragment_count,
            "excluded_fragment_count": excluded_fragment_count,
            "unassigned_fragment_count": unassigned_count,
        },
    }
    if errors:
        print(
            "[PRESET_IMPORT] 초안 검증 실패: "
            f"import_id={import_id!r}, selected={selected_count}, errors={errors!r}"
        )
    return result


def _load_manifest() -> dict:
    if not os.path.isfile(MANIFEST_FILE):
        print(f"[PRESET_IMPORT] 기존 매니페스트 없음, 새로 생성: path={MANIFEST_FILE}")
        return {"version": 1, "imports": []}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"[PRESET_IMPORT] 매니페스트 로드 실패: path={MANIFEST_FILE}, error={exc}")
        traceback.print_exc()
        raise PresetImportError("기존 프리셋 임포트 매니페스트를 읽지 못했습니다.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("imports"), list):
        print(f"[PRESET_IMPORT] 매니페스트 구조 오류: path={MANIFEST_FILE}, data={data!r}")
        raise PresetImportError("기존 프리셋 임포트 매니페스트 구조가 올바르지 않습니다.")
    return data


def build_scene_chain_slots(draft: Any, targets: Any) -> dict:
    """커밋 결과의 활성 프리셋을 SDStudio 씬 순서대로 체인 슬롯에 연결한다."""
    if not isinstance(draft, dict) or not isinstance(draft.get("items"), list):
        print(
            "[PRESET_IMPORT_CHAIN] 체인 구성 실패: 초안 items 형식 오류 "
            f"draft_type={type(draft).__name__}"
        )
        raise PresetImportError("체인을 만들 가져오기 초안 형식이 올바르지 않습니다.")
    if not isinstance(targets, list):
        print(
            "[PRESET_IMPORT_CHAIN] 체인 구성 실패: targets 형식 오류 "
            f"type={type(targets).__name__}"
        )
        raise PresetImportError("체인을 만들 프리셋 저장 결과 형식이 올바르지 않습니다.")

    targets_by_item: dict[str, list[dict]] = {}
    for target in targets:
        if not isinstance(target, dict) or not isinstance(target.get("item_id"), str):
            print(f"[PRESET_IMPORT_CHAIN] 잘못된 저장 대상 건너뜀: target={target!r}")
            continue
        targets_by_item.setdefault(target["item_id"], []).append(target)

    chains: list[dict] = []
    omitted_hidden: list[dict] = []
    for draft_item in draft["items"]:
        if not isinstance(draft_item, dict) or draft_item.get("selected") is not True:
            continue
        item_id = draft_item.get("id")
        item_targets = targets_by_item.get(item_id, [])
        if not item_targets:
            print(
                "[PRESET_IMPORT_CHAIN] 선택 항목의 저장 대상 없음, 체인 제외: "
                f"item_id={item_id!r}"
            )
            continue
        source_kind = item_targets[0].get("source_kind")
        if source_kind != "scene":
            continue

        slot = {
            "character": "",
            "appearance": "",
            "outfit": "",
            "expression": "",
            "composition_preset": "",
            "quality_preset": "",
            "character_negative_preset": "",
            "negative_preset": "",
            "artist_preset": "",
            "anima_artist_preset": "",
            "anima_quality_preset": "",
            "anima_negative_preset": "",
            "natural_language_preset": "",
            "natural_language": "",
            "seed": -1,
        }
        for target in item_targets:
            field = SCENE_CHAIN_FIELD_BY_CATEGORY.get(target.get("category"))
            if not field:
                continue
            if target.get("target_state") != "active":
                omitted = {
                    "item_id": item_id,
                    "category": target.get("category", ""),
                    "target_name": target.get("target_name", ""),
                    "target_state": target.get("target_state", ""),
                }
                omitted_hidden.append(omitted)
                print(
                    "[PRESET_IMPORT_CHAIN] 숨김 프리셋 연결 제외: "
                    f"item_id={item_id!r}, category={omitted['category']!r}, "
                    f"name={omitted['target_name']!r}"
                )
                continue
            target_name = target.get("target_name")
            if not isinstance(target_name, str) or not target_name.strip():
                print(
                    "[PRESET_IMPORT_CHAIN] 활성 저장 대상 이름 누락: "
                    f"item_id={item_id!r}, target={target!r}"
                )
                raise PresetImportError("체인에 연결할 프리셋 이름이 비어 있습니다.")
            slot[field] = target_name.strip()
        chains.append(slot)

    print(
        "[PRESET_IMPORT_CHAIN] 씬 체인 구성 완료: "
        f"slots={len(chains)}, hidden_omitted={len(omitted_hidden)}"
    )
    return {
        "chains": chains,
        "slot_count": len(chains),
        "hidden_omitted_count": len(omitted_hidden),
        "hidden_omitted": omitted_hidden,
    }


def _write_json_temp(path: str, data: dict) -> str:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception as exc:
        print(f"[PRESET_IMPORT] UTF-8 임시 저장 실패: path={temp_path}, error={exc}")
        traceback.print_exc()
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as cleanup_exc:
            print(
                "[PRESET_IMPORT] 실패한 임시 파일 정리 실패: "
                f"path={temp_path}, error={cleanup_exc}"
            )
            traceback.print_exc()
        raise


def _backup_existing_files(paths: list[str]) -> dict[str, str | None]:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backups: dict[str, str | None] = {}
    for path in paths:
        if not os.path.isfile(path):
            print(f"[PRESET_IMPORT] 백업할 기존 파일 없음: path={path}")
            backups[path] = None
            continue
        stem, extension = os.path.splitext(os.path.basename(path))
        backup_path = os.path.join(
            BACKUP_DIR,
            f"preset_import_{stem}_{stamp}_{uuid.uuid4().hex[:8]}{extension or '.json'}",
        )
        try:
            shutil.copy2(path, backup_path)
        except Exception as exc:
            print(
                "[PRESET_IMPORT] 저장 전 백업 실패, 커밋 중단: "
                f"source={path}, backup={backup_path}, error={exc}"
            )
            traceback.print_exc()
            raise PresetImportError(f"저장 전 백업에 실패했습니다: {os.path.basename(path)}") from exc
        print(f"[PRESET_IMPORT] 저장 전 백업 완료: {backup_path}")
        backups[path] = backup_path
    return backups


def _persist_transaction(
    active_tags: dict,
    hidden_tags: dict,
    manifest: dict,
) -> list[str]:
    paths = [TAGS_FILE, HIDDEN_TAGS_FILE, MANIFEST_FILE]
    backups = _backup_existing_files(paths)
    temporary: dict[str, str] = {}
    replaced: list[str] = []
    try:
        temporary[TAGS_FILE] = _write_json_temp(TAGS_FILE, active_tags)
        temporary[HIDDEN_TAGS_FILE] = _write_json_temp(HIDDEN_TAGS_FILE, hidden_tags)
        temporary[MANIFEST_FILE] = _write_json_temp(MANIFEST_FILE, manifest)
        for path in paths:
            os.replace(temporary[path], path)
            replaced.append(path)
        print(
            "[PRESET_IMPORT] 원자적 커밋 완료: "
            f"tags={TAGS_FILE}, hidden={HIDDEN_TAGS_FILE}, manifest={MANIFEST_FILE}"
        )
        return [backup for backup in backups.values() if backup]
    except Exception as exc:
        print(f"[PRESET_IMPORT] 원자적 커밋 실패, 롤백 시작: error={exc}")
        traceback.print_exc()
        rollback_errors = []
        for path in reversed(replaced):
            backup = backups.get(path)
            try:
                if backup and os.path.isfile(backup):
                    shutil.copy2(backup, path)
                elif os.path.exists(path):
                    os.remove(path)
            except Exception as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
                print(
                    "[PRESET_IMPORT] 롤백 실패: "
                    f"path={path}, backup={backup}, error={rollback_exc}"
                )
                traceback.print_exc()
        if rollback_errors:
            raise PresetImportError(
                "가져오기 저장과 롤백이 모두 실패했습니다: " + "; ".join(rollback_errors)
            ) from exc
        raise PresetImportError("가져오기 저장에 실패해 기존 파일로 롤백했습니다.") from exc
    finally:
        for temp_path in temporary.values():
            if not os.path.exists(temp_path):
                continue
            try:
                os.remove(temp_path)
            except Exception as cleanup_exc:
                print(
                    "[PRESET_IMPORT] 임시 파일 정리 실패: "
                    f"path={temp_path}, error={cleanup_exc}"
                )
                traceback.print_exc()


def commit_draft(
    draft: dict,
    resolutions: Any,
    active_tags: dict,
    hidden_tags: dict,
) -> dict:
    validation = validate_draft(draft, active_tags, hidden_tags)
    if not validation["success"]:
        print(
            "[PRESET_IMPORT] 검증 실패로 커밋 거부: "
            f"import_id={draft.get('import_id')!r}, errors={validation['errors']}"
        )
        raise PresetImportError("가져오기 초안 검증을 통과하지 못했습니다.")
    if not isinstance(resolutions, list):
        print(f"[PRESET_IMPORT] 충돌 해결값 형식 오류: type={type(resolutions).__name__}")
        raise PresetImportError("충돌 해결값이 배열이 아닙니다.")
    conflict_ids = {conflict["record_id"] for conflict in validation["conflicts"]}
    resolution_map = {}
    for resolution in resolutions:
        if not isinstance(resolution, dict) or not isinstance(resolution.get("record_id"), str):
            print(f"[PRESET_IMPORT] 잘못된 충돌 해결값: {resolution!r}")
            raise PresetImportError("충돌 해결 항목 형식이 올바르지 않습니다.")
        record_id = resolution["record_id"]
        if record_id in resolution_map:
            print(f"[PRESET_IMPORT] 충돌 해결값 ID 중복: record_id={record_id!r}")
            raise PresetImportError("충돌 해결 항목 ID가 중복되었습니다.")
        if record_id not in conflict_ids:
            print(f"[PRESET_IMPORT] 불필요하거나 알 수 없는 충돌 해결값: record_id={record_id!r}")
            raise PresetImportError("현재 충돌 목록에 없는 해결 항목이 포함되었습니다.")
        resolution_map[record_id] = resolution

    new_active = copy.deepcopy(active_tags)
    new_hidden = copy.deepcopy(hidden_tags)
    targets: list[dict] = []
    reserved = {
        category: set((new_active.get(category) or {})) | set((new_hidden.get(category) or {}))
        for category in IMPORT_CATEGORIES
    }

    for record in validation["records"]:
        category = record["category"]
        name = record["name"]
        values = list(record["values"])
        status = record.get("status")
        target_name = name
        action = status
        target_state = record.get("existing_state", "active")

        if status == "new":
            new_active.setdefault(category, {})
            new_active[category][name] = values
            reserved[category].add(name)
            action = "created"
            target_state = "active"
        elif status == "same":
            action = "reused"
        elif record["id"] in conflict_ids:
            resolution = resolution_map.get(record["id"])
            if resolution is None:
                print(
                    "[PRESET_IMPORT] 충돌 해결값 누락: "
                    f"record={category}/{name}, record_id={record['id']}"
                )
                raise PresetImportError(f"충돌 해결값이 없습니다: {category}/{name}")
            choice = resolution.get("choice")
            if choice == "keep":
                action = "kept_existing"
            elif choice == "overwrite":
                # 숨김 충돌도 가져온 값으로 활성화한다. UI에 같은 의미로 표시한다.
                new_hidden.setdefault(category, {}).pop(name, None)
                new_active.setdefault(category, {})
                new_active[category][name] = values
                action = "overwritten_and_activated"
                target_state = "active"
            elif choice == "rename":
                target_name, name_error = _validate_target_name(
                    resolution.get("new_name"),
                    f"'{category}/{name}' 새",
                )
                if name_error:
                    print(f"[PRESET_IMPORT] 충돌 새 이름 오류: {name_error}")
                    raise PresetImportError(name_error)
                if target_name in reserved[category]:
                    print(
                        "[PRESET_IMPORT] 충돌 새 이름 재충돌: "
                        f"category={category}, name={target_name!r}"
                    )
                    raise PresetImportError(
                        f"새 이름도 이미 존재합니다: {category}/{target_name}"
                    )
                new_active.setdefault(category, {})
                new_active[category][target_name] = values
                reserved[category].add(target_name)
                action = "renamed"
                target_state = "active"
            else:
                print(
                    "[PRESET_IMPORT] 알 수 없는 충돌 선택: "
                    f"record_id={record['id']}, choice={choice!r}"
                )
                raise PresetImportError(f"지원하지 않는 충돌 처리 방식입니다: {choice}")
        else:
            print(f"[PRESET_IMPORT] 레코드 상태 오류: record={record!r}")
            raise PresetImportError("가져오기 레코드 상태가 올바르지 않습니다.")

        targets.append({
            "record_id": record["id"],
            "item_id": record["item_id"],
            "group_id": record["group_id"],
            "source_kind": record["source_kind"],
            "category": category,
            "source_name": name,
            "target_name": target_name,
            "target_state": target_state,
            "action": action,
            "value_count": len(values),
        })

    session = get_analysis_session(draft["import_id"])
    manifest = _load_manifest()
    manifest_id = uuid.uuid4().hex
    draft_by_id = {
        item["id"]: item for item in draft.get("items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    selected_items = []
    for item_id, original in session["items_by_id"].items():
        edited = draft_by_id.get(item_id)
        if not edited or edited.get("selected") is not True:
            continue
        original_fragments = {
            fragment["id"]: fragment for fragment in original["fragments"]
        }
        selected_items.append({
            "item_id": item_id,
            "group_id": original["group_id"],
            "source_kind": original["source_kind"],
            "source_name": original["source_name"],
            "source_prompt": original["source_prompt"],
            "expanded_prompt": original.get("expanded_prompt", ""),
            "target_prompt": original.get("target_prompt", ""),
            "canonical_prompts": copy.deepcopy(original.get("canonical_prompts", {})),
            "structured_prompts": copy.deepcopy(original.get("structured_prompts", {})),
            "target_adapter": "anima",
            "max_abs_weight": 1.5,
            "requested_target_name": _normalize_target_name(
                str(edited.get("target_name", "")).strip()
            ),
            "fragments": [
                {
                    "id": fragment.get("id"),
                    "original_text": original_fragments.get(
                        fragment.get("id"), {}
                    ).get("original_text", ""),
                    "import_text": original_fragments.get(
                        fragment.get("id"), {}
                    ).get("import_text", ""),
                    "text": fragment.get("text", ""),
                    "category": fragment.get("category", "unassigned"),
                    "llm_category": fragment.get("llm_category"),
                    "origin": fragment.get("origin", "source"),
                    "excluded": fragment.get("excluded") is True,
                    "normalization": copy.deepcopy(original_fragments.get(
                        fragment.get("id"), {}
                    ).get("normalization", {})),
                }
                for fragment in edited.get("fragments", [])
                if isinstance(fragment, dict)
            ],
        })
    manifest_entry = {
        "id": manifest_id,
        "imported_at": _now_iso(),
        "source": copy.deepcopy(session["analysis"]["source"]),
        "target": copy.deepcopy(session["analysis"].get("target", {})),
        "summary": copy.deepcopy(validation["summary"]),
        "items": selected_items,
        "targets": targets,
    }
    manifest["imports"].append(manifest_entry)
    backup_paths = _persist_transaction(new_active, new_hidden, manifest)
    with _SESSION_LOCK:
        session["committed"] = True
        session["manifest_id"] = manifest_id
    print(
        "[PRESET_IMPORT] 가져오기 완료: "
        f"import_id={draft['import_id']}, manifest_id={manifest_id}, targets={len(targets)}"
    )
    return {
        "success": True,
        "manifest_id": manifest_id,
        "target_count": len(targets),
        "targets": targets,
        "backup_count": len(backup_paths),
        "active_tags": new_active,
        "hidden_tags": new_hidden,
        "validation": validation,
    }
