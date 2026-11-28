"""
Character Maker

캐릭터 확정 전까지의 세계관, 대화, 자유 편집 태그, 참고 이미지와 생성 결과를
단일 영속 세션(default)으로 관리한다. 세션 상태는 temp_root/default/session.json
에 원자적으로 저장되어 서버 재시작에도 유지되며, 배포에서는 제외된다. 이 모듈은
config.json 또는 tags.json을 자동으로 수정하지 않는다. tags.json 반영은
confirm()의 명시적 확정 단계에서만 백업 후 원자적으로 수행한다.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import datetime
import importlib
import io
import json
import math
import os
import shutil
import tempfile
import time
import traceback
import uuid
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import llm_prompt_edit
from . import llm_service
from .danbooru_rag import DanbooruRagError, get_danbooru_rag_service


EDITABLE_FIELDS = ("appearance", "outfit", "expression", "composition")
# 잠금(LLM 수정 보호) 대상: 태그 필드 4종 + 자연어. 자연어는 별도 최상위 문자열 필드.
LOCKABLE_FIELDS = EDITABLE_FIELDS + ("natural_language",)
# 생성 설정에서 사용자가 자유 편집 영역으로 꺼낼 수 있는 프리셋 태그.
# LLM 계약(EDITABLE_FIELDS)에는 포함하지 않아 사용자만 수정할 수 있게 유지한다.
EDITABLE_PRESET_FIELDS = (
    "quality_preset",
    "artist_preset",
    "negative_preset",
    "anima_quality_preset",
    "anima_artist_preset",
    "anima_negative_preset",
    "character_negative_preset",
)
EDITABLE_PRESET_CATEGORIES = {
    "quality_preset": "quality_presets",
    "artist_preset": "artist_presets",
    "negative_preset": "negative_presets",
    "anima_quality_preset": "quality_presets",
    "anima_artist_preset": "artist_presets",
    "anima_negative_preset": "negative_presets",
    "character_negative_preset": "character_negative_presets",
}
EDITABLE_PRESET_LABELS = {
    "quality_preset": "ILXL 품질",
    "artist_preset": "ILXL 아티스트",
    "negative_preset": "ILXL 부정",
    "anima_quality_preset": "ANIMA 품질",
    "anima_artist_preset": "ANIMA 아티스트",
    "anima_negative_preset": "ANIMA 부정",
    "character_negative_preset": "캐릭터 부정",
}
MAX_TAG_ITEM_LENGTH = 6000
MAX_NATURAL_LANGUAGE_LENGTH = 2000
MAKER_MODEL_FAMILIES = ("anima", "moss")
MAKER_REASONING_DEPTHS = ("fast", "deep")
MAX_AI_BBOX_CROPS = 4
RAG_COLD_START_TIMEOUT_SECONDS = 300.0
SINGLE_SESSION_ID = "default"
MAX_REFERENCE_COUNT = 8
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
MAX_CHAT_ITEMS = 40
MAX_CHAT_CONTEXT_ITEMS = 12
MAX_REVISIONS = 12
GENERATION_WORKFLOW_TYPES = ("asset", "illustration")


class CharacterMakerError(ValueError):
    """사용자 입력 또는 현재 세션 상태로 작업을 진행할 수 없을 때 사용."""


def normalize_maker_model_family(
    value: Any, *, default: str = "anima", strict: bool = False
) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    if normalized in MAKER_MODEL_FAMILIES:
        return normalized
    if strict:
        raise CharacterMakerError("메이커 모드는 Anima 또는 MOSS만 선택할 수 있습니다.")
    return default


def normalize_maker_reasoning_depth(
    value: Any, *, default: str = "fast", strict: bool = False
) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    if normalized in MAKER_REASONING_DEPTHS:
        return normalized
    if strict:
        raise CharacterMakerError("추론 레벨은 빠르게 또는 깊게만 선택할 수 있습니다.")
    return default


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _strip_unbalanced_edge_quotes(text: str) -> str:
    """Remove stray unbalanced leading/trailing double quotes from decoded LLM prose."""
    cleaned = text.strip()
    while cleaned and cleaned.count('"') % 2 == 1:
        if cleaned.endswith('"'):
            cleaned = cleaned[:-1].rstrip()
        elif cleaned.startswith('"'):
            cleaned = cleaned[1:].lstrip()
        else:
            break
    return cleaned


def _normalize_tag_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CharacterMakerError(f"{field} 태그는 문자열 배열이어야 합니다.")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise CharacterMakerError(f"{field} 태그에는 문자열만 사용할 수 있습니다.")
        tag = _strip_unbalanced_edge_quotes(raw)
        if not tag:
            continue
        if len(tag) > MAX_TAG_ITEM_LENGTH:
            raise CharacterMakerError(
                f"{field} 태그 한 항목은 {MAX_TAG_ITEM_LENGTH}자를 넘을 수 없습니다."
            )
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(tag)
    if len(normalized) > 160:
        raise CharacterMakerError(f"{field} 태그는 160개를 넘을 수 없습니다.")
    return normalized


def _normalize_fields(value: Any) -> dict[str, list[str]]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise CharacterMakerError("fields는 객체여야 합니다.")
    return {
        field: _normalize_tag_list(value.get(field, []), field=field)
        for field in EDITABLE_FIELDS
    }


def _normalize_locks(value: Any) -> dict[str, bool]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise CharacterMakerError("locks는 객체여야 합니다.")
    return {field: bool(value.get(field, False)) for field in LOCKABLE_FIELDS}


def _normalize_editable_preset_tags(value: Any) -> dict[str, list[str]]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise CharacterMakerError("editable_preset_tags는 객체여야 합니다.")
    return {
        field: _normalize_tag_list(value.get(field, []), field=field)
        for field in EDITABLE_PRESET_FIELDS
    }


def _normalize_editable_preset_enabled(value: Any) -> dict[str, bool]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise CharacterMakerError("editable_preset_enabled는 객체여야 합니다.")
    return {
        field: bool(value.get(field, False)) for field in EDITABLE_PRESET_FIELDS
    }


def _normalize_lora_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CharacterMakerError("캐릭터 메이커 LoRA 목록은 배열이어야 합니다.")
    if len(value) > 12:
        raise CharacterMakerError("캐릭터 메이커 LoRA는 최대 12개까지 사용할 수 있습니다.")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise CharacterMakerError(f"LoRA {index + 1}번 항목은 객체여야 합니다.")
        source = str(raw.get("source") or "asset").strip().lower()
        if source not in {"asset", "bot", "instance", "style"}:
            print(
                "[CHARACTER_MAKER] LoRA source 검증 실패: "
                f"index={index + 1}, source={source!r}, "
                "allowed=('asset', 'bot', 'instance', 'style')"
            )
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 source가 올바르지 않습니다."
            )
        lora_path = str(raw.get("lora_path") or "").strip()
        if not lora_path or len(lora_path) > 1000:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 모델 경로가 비어 있거나 너무 깁니다."
            )
        path_parts = [
            part for part in lora_path.replace("\\", "/").split("/") if part
        ]
        if os.path.isabs(lora_path) or ".." in path_parts:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목은 상대 모델 경로만 사용할 수 있습니다."
            )
        base = str(raw.get("BASE") or "anima").strip().lower()
        if base == "ilxl":
            base = "sdxl"
        if base not in {"anima", "sdxl"}:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 BASE는 anima 또는 sdxl이어야 합니다."
            )
        try:
            strength = float(raw.get("strength", 0.5))
        except (TypeError, ValueError) as exc:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 강도가 숫자가 아닙니다."
            ) from exc
        if not math.isfinite(strength) or not 0.0 <= strength <= 2.0:
            raise CharacterMakerError(
                f"LoRA {index + 1}번 항목의 강도는 0~2 사이여야 합니다."
            )
        identity = (source, lora_path.casefold(), base)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(
            {
                "name": str(raw.get("name") or os.path.basename(lora_path)).strip()[:300],
                "character": str(raw.get("character") or "").strip()[:300],
                "lora_path": lora_path,
                "strength": strength,
                "preview_url": str(raw.get("preview_url") or "").strip()[:2000],
                "trigger": str(raw.get("trigger") or "").strip()[:1000],
                "BASE": base,
                "source": source,
                "lora_id": str(raw.get("lora_id") or "").strip()[:300],
            }
        )
    return normalized


def _parse_llm_payload(
    raw: str, *, require_queries: bool
) -> tuple[dict[str, Any] | None, str]:
    """LLM 응답을 검증·정규화한다.

    Returns:
        (parsed, reason) — 성공 시 (정규화된 딕셔너리, "").
        실패 시 (None, "<구체적 사유>"). 사유는 LIGHBD 자세히 로그와 프론트 에러에
        그대로 노출되므로, 어느 검증 단계에서 왜 막혔는지 사람이 읽을 수 있어야 한다.
        (과거엔 모든 실패를 None 하나로 뭉뚱그려 "형식이 올바르지 않습니다"만 남아
        원인 분석이 불가능했다.)
    """
    try:
        parsed = llm_prompt_edit.parse_llm_json(raw)
    except Exception as exc:
        return None, f"LLM 응답을 JSON으로 파싱하지 못했습니다: {type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "LLM 응답이 JSON 객체가 아닙니다."
    assistant_message = parsed.get("assistant_message")
    if not isinstance(assistant_message, str) or not assistant_message.strip():
        return None, "assistant_message(문자열)가 없거나 비어 있습니다."
    fields = parsed.get("fields")
    if not isinstance(fields, dict):
        return None, "fields(객체)가 없습니다."
    # LLM이 natural_language를 fields 안에 잘못 넣는 사례가 잦아, 최상위 키로 끌어올려
    # 보정한다(이 키 하나 때문에 재시도를 소진하는 낭비를 막기 위함). 최상위에 이미
    # natural_language가 있으면(정상 배치) 그것을 우선하고, fields 내 중복은 무시한다.
    if "natural_language" in fields and "natural_language" not in parsed:
        parsed["natural_language"] = fields["natural_language"]
        print(
            "[CHARACTER_MAKER] natural_language를 fields 안에서 최상위 키로 보정: "
            f"value_type={type(fields['natural_language']).__name__}"
        )
    if "natural_language" in fields:
        fields = {k: v for k, v in fields.items() if k != "natural_language"}
    if set(fields) != set(EDITABLE_FIELDS):
        missing = sorted(set(EDITABLE_FIELDS) - set(fields))
        extra = sorted(set(fields) - set(EDITABLE_FIELDS))
        detail = "; ".join(
            p for p in (
                f"누락={missing}" if missing else "",
                f"잉여={extra}" if extra else "",
            ) if p
        )
        return None, (
            f"fields 키가 {sorted(EDITABLE_FIELDS)}와 정확히 일치해야 합니다. ({detail})"
        )
    coerced_fields: dict[str, Any] = {}
    for field in EDITABLE_FIELDS:
        raw_field_value = fields.get(field)
        if isinstance(raw_field_value, str):
            coerced_fields[field] = [raw_field_value] if raw_field_value.strip() else []
        else:
            coerced_fields[field] = raw_field_value
    try:
        normalized_fields = _normalize_fields(coerced_fields)
    except CharacterMakerError as exc:
        return None, f"fields 값을 정규화하지 못했습니다: {exc}"

    raw_queries = parsed.get("rag_queries", {})
    if require_queries and not isinstance(raw_queries, dict):
        return None, "rag_queries(객체)가 없습니다."
    if require_queries and set(raw_queries) != set(EDITABLE_FIELDS):
        missing = sorted(set(EDITABLE_FIELDS) - set(raw_queries))
        extra = sorted(set(raw_queries) - set(EDITABLE_FIELDS))
        detail = "; ".join(
            p for p in (
                f"누락={missing}" if missing else "",
                f"잉여={extra}" if extra else "",
            ) if p
        )
        return None, (
            f"rag_queries 키가 {sorted(EDITABLE_FIELDS)}와 일치해야 합니다. ({detail})"
        )
    if not isinstance(raw_queries, dict):
        raw_queries = {}
    rag_queries: dict[str, list[str]] = {}
    for field in EDITABLE_FIELDS:
        queries = raw_queries.get(field, [])
        # LLM이 배열 대신 단일 문자열을 반환하는 경우가 잦아 [query]로 정규화한다.
        # 비-리스트/비-문자열(숫자 등)은 여전히 거부(None)하여 의미 왜곡을 막는다.
        if isinstance(queries, str):
            print(
                f"[CHARACTER_MAKER] rag_queries 값을 배열로 정규화: "
                f"field={field}, raw_type=str"
            )
            queries = [queries]
        if not isinstance(queries, list):
            return None, (
                f"rag_queries[{field}]가 배열(또는 단일 문자열)이어야 합니다. "
                f"(실제 타입: {type(queries).__name__})"
            )
        clean_queries: list[str] = []
        for query in queries:
            if not isinstance(query, str):
                return None, (
                    f"rag_queries[{field}]의 원소가 문자열이 아닙니다. "
                    f"(타입: {type(query).__name__})"
                )
            query = query.strip()
            if query and query not in clean_queries:
                clean_queries.append(query[:300])
        rag_queries[field] = clean_queries[:4]

    # natural_language는 선택적 키 — 없거나 비-문자열이면 None(변경 없음 신호).
    natural_language = parsed.get("natural_language")
    if natural_language is not None and not isinstance(natural_language, str):
        natural_language = None
    if isinstance(natural_language, str):
        natural_language = _strip_unbalanced_edge_quotes(natural_language)[:MAX_NATURAL_LANGUAGE_LENGTH]

    return {
        "assistant_message": assistant_message.strip()[:4000],
        "fields": normalized_fields,
        "rag_queries": rag_queries,
        "natural_language": natural_language,
    }, ""


def validate_character_maker_llm_result(
    raw: str, *, require_queries: bool = True
) -> tuple[bool, str]:
    """callLLMTask/callLLMVisionTask result_validator 계약.

    실패 시 구체적 사유를 그대로 넘긴다(LIGHBD per-attempt 로그에 노출).
    """
    parsed, reason = _parse_llm_payload(raw, require_queries=require_queries)
    if parsed is None:
        return False, reason or "LLM 응답 형식이 올바르지 않습니다."
    return True, ""


MOSS_FORENSIC_TEXT_KEYS = (
    "character_overview",
    "hair_and_bangs",
    "headpiece_and_ribbons",
    "eyes_ears_wings_nonhuman",
    "neck_and_shoulders",
    "bodice_and_chest",
    "arms_sleeves_gloves",
    "waist_and_hips",
    "skirt_and_overskirt",
    "legwear",
    "footwear",
    "held_props",
    "expression_and_pose",
)
MOSS_APPEARANCE_SECTION_KEYS = (
    "body_eyes_ears",
    "hair_bangs_curls",
    "headpiece_ribbons",
    "wings_nonhuman",
)
MOSS_OUTFIT_SECTION_KEYS = (
    "neck_shoulders_mantle",
    "bodice_chest",
    "sleeves_gloves",
    "waist_hips",
    "skirt_overskirt",
    "legwear_footwear",
    "legwear",
    "footwear",
    "held_props",
)


def _join_moss_prose_sections(parts: list[str]) -> str:
    """Join non-empty English prose sections cleanly with sentence spacing."""
    result = ""
    for raw in parts:
        cleaned = _strip_unbalanced_edge_quotes(raw)
        if not cleaned:
            continue
        if not result:
            result = cleaned
        elif result.endswith((".", "!", "?")):
            result = f"{result} {cleaned}"
        else:
            result = f"{result}. {cleaned}"
    return result


def _parse_reference_bboxes(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Parse AI-detected normalized [0..1000] detail crop bounding boxes."""
    try:
        parsed = llm_prompt_edit.parse_llm_json(raw)
    except Exception as exc:
        return None, f"영역 탐지(BBox) JSON 파싱 실패: {type(exc).__name__}: {exc}"
    if isinstance(parsed, list):
        raw_regions = parsed
    elif isinstance(parsed, dict):
        raw_regions = (
            parsed.get("regions")
            or parsed.get("boxes")
            or parsed.get("crops")
            or []
        )
    else:
        return None, "영역 탐지(BBox) 응답 형식이 올바르지 않습니다."
    if not isinstance(raw_regions, list):
        return None, "영역 탐지(BBox) regions는 배열이어야 합니다."

    normalized: list[dict[str, Any]] = []
    for item in raw_regions:
        if not isinstance(item, dict):
            continue
        coords = None
        if all(k in item for k in ("x1", "y1", "x2", "y2")):
            coords = (item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2"))
        elif isinstance(item.get("bbox"), (list, tuple)) and len(item["bbox"]) == 4:
            coords = tuple(item["bbox"])
        elif isinstance(item.get("box"), (list, tuple)) and len(item["box"]) == 4:
            coords = tuple(item["box"])
        if coords is None:
            continue
        try:
            vals = [float(v) for v in coords]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in vals):
            continue
        if max(abs(v) for v in vals) <= 1.0:
            vals = [v * 1000.0 for v in vals]
        x1, y1, x2, y2 = [max(0, min(1000, int(round(v)))) for v in vals]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        w = x2 - x1
        h = y2 - y1
        if w < 50 or h < 50:
            continue
        if w * h > 880_000:
            continue
        try:
            image_index = max(1, int(item.get("image_index", 1)))
        except (TypeError, ValueError):
            image_index = 1
        label = _strip_unbalanced_edge_quotes(str(item.get("label") or "")).strip()[:64]
        if not label:
            label = f"Detail region {len(normalized) + 1}"
        normalized.append({
            "image_index": image_index,
            "label": label,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
        if len(normalized) >= MAX_AI_BBOX_CROPS:
            break
    return {"regions": normalized}, ""


def _parse_moss_forensic_audit(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Validate and normalize Stage-1 14-slot visual forensic blueprint."""
    try:
        parsed = llm_prompt_edit.parse_llm_json(raw)
    except Exception as exc:
        return None, f"1단계 시각 분석 JSON 파싱 실패: {type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "1단계 시각 분석 응답이 JSON 객체가 아닙니다."
    normalized: dict[str, Any] = {}
    missing = []
    for key in MOSS_FORENSIC_TEXT_KEYS:
        val = parsed.get(key)
        if isinstance(val, list):
            val = ", ".join(str(x).strip() for x in val if str(x).strip())
        if not isinstance(val, str) or not val.strip():
            missing.append(key)
            normalized[key] = ""
        else:
            normalized[key] = _strip_unbalanced_edge_quotes(val)[:2000]
    if len(missing) > 2:
        return None, f"1단계 시각 분석 필수 구역이 누락되었습니다: {missing}"
    raw_fixes = parsed.get("visual_diff_fixes", [])
    if isinstance(raw_fixes, str):
        raw_fixes = [raw_fixes] if raw_fixes.strip() else []
    if not isinstance(raw_fixes, list):
        raw_fixes = []
    normalized["visual_diff_fixes"] = [
        _strip_unbalanced_edge_quotes(str(item))[:600]
        for item in raw_fixes
        if str(item).strip()
    ][:20]
    scope_note = parsed.get("user_scope_note", "")
    if isinstance(scope_note, str) and scope_note.strip():
        normalized["user_scope_note"] = _strip_unbalanced_edge_quotes(scope_note)[:600]
    return normalized, ""


def _parse_moss_sectioned_synthesis(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Validate Stage-2 sectioned MOSS synthesis and assemble standard maker payload."""
    try:
        parsed = llm_prompt_edit.parse_llm_json(raw)
    except Exception as exc:
        return None, f"2단계 MOSS 프롬프트 JSON 파싱 실패: {type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "2단계 MOSS 프롬프트 응답이 JSON 객체가 아닙니다."

    # Fallback: if model returned standard fields schema directly, delegate safely.
    if (
        isinstance(parsed.get("fields"), dict)
        and "appearance_sections" not in parsed
        and "outfit_sections" not in parsed
    ):
        return _parse_llm_payload(raw, require_queries=False)

    assistant_message = parsed.get("assistant_message")
    if not isinstance(assistant_message, str) or not assistant_message.strip():
        return None, "assistant_message(문자열)가 없거나 비어 있습니다."

    app_sec = parsed.get("appearance_sections")
    outfit_sec = parsed.get("outfit_sections")
    if not isinstance(app_sec, dict):
        return None, "appearance_sections(객체)가 없습니다."
    if not isinstance(outfit_sec, dict):
        return None, "outfit_sections(객체)가 없습니다."

    def _collect_section_parts(
        section_map: dict[str, Any], preferred_keys: tuple[str, ...]
    ) -> list[str]:
        parts: list[str] = []
        seen: set[str] = set()
        for key in (*preferred_keys, *section_map.keys()):
            if key in seen or key not in section_map:
                continue
            seen.add(key)
            val = section_map.get(key, "")
            if isinstance(val, list):
                val = " ".join(str(x).strip() for x in val if str(x).strip())
            if isinstance(val, str) and val.strip():
                parts.append(val)
        return parts

    app_parts = _collect_section_parts(app_sec, MOSS_APPEARANCE_SECTION_KEYS)
    outfit_parts = _collect_section_parts(outfit_sec, MOSS_OUTFIT_SECTION_KEYS)

    appearance_prose = _join_moss_prose_sections(app_parts)
    outfit_prose = _join_moss_prose_sections(outfit_parts)
    official_tag = parsed.get("official_character_tag", "")
    official_tag = _strip_unbalanced_edge_quotes(official_tag) if isinstance(official_tag, str) else ""
    appearance_list = [p for p in (official_tag, appearance_prose) if p]
    outfit_list = [outfit_prose] if outfit_prose else []

    def _coerce_prose_list(value: Any) -> list[str]:
        if isinstance(value, str):
            cleaned = _strip_unbalanced_edge_quotes(value)
            return [cleaned] if cleaned else []
        if isinstance(value, list):
            joined = _join_moss_prose_sections([str(x) for x in value if isinstance(x, str)])
            return [joined] if joined else []
        return []

    expression_list = _coerce_prose_list(parsed.get("expression", ""))
    composition_list = _coerce_prose_list(parsed.get("composition", ""))
    if not appearance_list and not outfit_list:
        return None, "appearance_sections와 outfit_sections가 모두 비어 있습니다."

    try:
        normalized_fields = _normalize_fields({
            "appearance": appearance_list,
            "outfit": outfit_list,
            "expression": expression_list,
            "composition": composition_list,
        })
    except CharacterMakerError as exc:
        return None, f"2단계 fields 정규화 실패: {exc}"

    natural_language = parsed.get("natural_language")
    if natural_language is not None and not isinstance(natural_language, str):
        natural_language = None
    if isinstance(natural_language, str):
        natural_language = _strip_unbalanced_edge_quotes(natural_language)[:MAX_NATURAL_LANGUAGE_LENGTH]

    return {
        "assistant_message": assistant_message.strip()[:4000],
        "fields": normalized_fields,
        "rag_queries": {field: [] for field in EDITABLE_FIELDS},
        "natural_language": natural_language,
    }, ""


def _tag_diff(
    before: dict[str, list[str]], after: dict[str, list[str]]
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for field in EDITABLE_FIELDS:
        old = before.get(field, [])
        new = after.get(field, [])
        old_keys = {item.casefold() for item in old}
        new_keys = {item.casefold() for item in new}
        result[field] = {
            "added": [item for item in new if item.casefold() not in old_keys],
            "removed": [item for item in old if item.casefold() not in new_keys],
        }
    return result


def _assert_within(root: str, target: str) -> None:
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(target)
    try:
        common = os.path.commonpath([root_real, target_real])
    except ValueError as exc:
        raise CharacterMakerError(f"임시 경로 검증 실패: {target_real}") from exc
    if common != root_real or target_real == root_real:
        raise CharacterMakerError(f"임시 경로가 작업 루트를 벗어났습니다: {target_real}")


def _safe_registration_name(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CharacterMakerError(f"{label}은 문자열이어야 합니다.")
    name = value.strip()
    if not name:
        raise CharacterMakerError(f"{label}을 입력하세요.")
    if len(name) > 100:
        raise CharacterMakerError(f"{label}은 100자를 넘을 수 없습니다.")
    if any(ord(ch) < 32 for ch in name) or any(ch in name for ch in '<>:"/\\|?*'):
        raise CharacterMakerError(f"{label}에 파일명 금지 문자를 사용할 수 없습니다.")
    return name


def _existing_registration_name(value: Any, label: str) -> str:
    """Return an existing preset identifier without treating it as a filename."""
    if not isinstance(value, str):
        raise CharacterMakerError(f"{label}은 문자열이어야 합니다.")
    name = value.strip()
    if not name:
        raise CharacterMakerError(f"{label}을 선택하세요.")
    return name


class CharacterMakerService:
    def __init__(
        self,
        asset_manager: Any,
        config_getter: Callable[[], dict[str, Any]],
        *,
        temp_root: str,
    ) -> None:
        self.asset_manager = asset_manager
        self.config_getter = config_getter
        self.temp_root = os.path.realpath(temp_root)
        os.makedirs(self.temp_root, exist_ok=True)
        self.boot_id = uuid.uuid4().hex
        self.sessions: dict[str, dict[str, Any]] = {}
        self._operation_locks: dict[str, asyncio.Lock] = {}
        # 단일 고정 세션을 디스크에서 로드(없으면 새로 생성·저장).
        self._load_or_create_session()

    def _default_settings(self) -> dict[str, Any]:
        config = self.config_getter() or {}
        return {
            "generation_workflow": "asset",
            "asset_workflow_type": str(config.get("asset_workflow_type") or "ilxl"),
            "quality_preset": "",
            "artist_preset": "",
            "negative_preset": "",
            "character_negative_preset": "",
            "anima_quality_preset": "",
            "anima_artist_preset": "",
            "anima_negative_preset": "",
            "img_w": 700,
            "img_h": 1024,
            "seed": -1,
            "rag_enabled": bool(config.get("character_maker_rag_enabled", False)),
            "lora_enabled": False,
            "lora_list": [],
            # 생성 옵션 — 에셋/오토매치와 동일 프롬프트 토큰 세트.
            # pose·ipadapter(참조 이미지)는 CM 미지원 잠금으로 유지한다.
            "style_lora_enabled": False,
            "style_lora_list": [],
            "snapref_enabled": False,
            "snapref_images": [],
            "snapref_dir": "",
            "snapref_mode": "char",
            "snapref_strength": 0.0,
            "snapref_megapixels": 1.0,
            "face_lora_enabled": False,
            "face_lora_list": [],
            "face_lora_upscale_size": "",
            "face_tags": "",
            "eye_tags": "",
            "hrf_sdxl": False,
            "hrf_anima": False,
            "hrf_moss": False,
            "hrf_size": 2.0,
            "hrf_restore_size": True,
            "hrf_control_net": False,
            "sdxl_fd_enabled": False,
            "sdxl_hd_enabled": False,
            "sdxl_ed_enabled": False,
            "anima_fd_enabled": False,
            "anima_hd_enabled": False,
            "anima_ed_enabled": False,
            "face_crop_top": 2.5,
            "face_crop_bottom": 1.0,
        }

    def _session_dir(self) -> str:
        return os.path.join(self.temp_root, SINGLE_SESSION_ID)

    def _session_json_path(self) -> str:
        return os.path.join(self._session_dir(), "session.json")

    def _fresh_session(self) -> dict[str, Any]:
        """새 빈 단일 세션 객체를 만든다(디스크 디렉터리도 보장)."""
        session_dir = self._session_dir()
        os.makedirs(os.path.join(session_dir, "references"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "images"), exist_ok=True)
        now = _now_iso()
        return {
            "id": SINGLE_SESSION_ID,
            "boot_id": self.boot_id,
            "created_at": now,
            "updated_at": now,
            "world_context": "",
            "natural_language": "",
            "llm_natural_language": "",
            "fields": {field: [] for field in EDITABLE_FIELDS},
            "llm_fields": {field: [] for field in EDITABLE_FIELDS},
            "locks": {field: False for field in LOCKABLE_FIELDS},
            "editable_preset_tags": {
                field: [] for field in EDITABLE_PRESET_FIELDS
            },
            "editable_preset_enabled": {
                field: False for field in EDITABLE_PRESET_FIELDS
            },
            "settings": self._default_settings(),
            "chat": [],
            "active_chat_branch_id": "",
            "user_chat_checkpoint_id": "",
            "references": [],
            "revisions": [],
            "active_revision_id": "",
            "llm_active_revision_id": "",
            "finalized": None,
        }

    def _install_single_session(self, session: dict[str, Any]) -> None:
        session["id"] = SINGLE_SESSION_ID
        session["boot_id"] = self.boot_id
        self.sessions = {SINGLE_SESSION_ID: session}
        self._operation_locks = {SINGLE_SESSION_ID: asyncio.Lock()}

    def _load_session_from_disk(self) -> dict[str, Any] | None:
        path = self._session_json_path()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 세션 영속화 파일 로드 실패, 새 세션 시작: "
                f"path={path}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return None
        if not isinstance(data, dict) or data.get("id") != SINGLE_SESSION_ID:
            print(
                f"[CHARACTER_MAKER] 세션 영속화 파일 무효, 새 세션 시작: path={path}"
            )
            return None

        # 누락된 최상위 필드 보완.
        data.setdefault("world_context", "")
        data.setdefault("natural_language", "")
        if not isinstance(data.get("natural_language"), str):
            data["natural_language"] = ""
        data.setdefault("llm_natural_language", "")
        if not isinstance(data.get("llm_natural_language"), str):
            data["llm_natural_language"] = ""
        data.setdefault("fields", {field: [] for field in EDITABLE_FIELDS})
        data.setdefault("llm_fields", {field: [] for field in EDITABLE_FIELDS})
        data.setdefault("locks", {field: False for field in LOCKABLE_FIELDS})
        data.setdefault(
            "editable_preset_tags",
            {field: [] for field in EDITABLE_PRESET_FIELDS},
        )
        data.setdefault(
            "editable_preset_enabled",
            {field: False for field in EDITABLE_PRESET_FIELDS},
        )
        data.setdefault("settings", {})
        data.setdefault("chat", [])
        data.setdefault("active_chat_branch_id", "")
        data.setdefault("user_chat_checkpoint_id", "")
        data.setdefault("references", [])
        data.setdefault("revisions", [])
        data.setdefault("active_revision_id", "")
        data.setdefault("llm_active_revision_id", "")
        data.setdefault("finalized", None)
        data.setdefault("created_at", _now_iso())
        data.setdefault("updated_at", _now_iso())
        for field in EDITABLE_FIELDS:
            data["fields"].setdefault(field, [])
            data["llm_fields"].setdefault(field, [])
            data["locks"].setdefault(field, False)
        data["locks"].setdefault("natural_language", False)
        try:
            data["editable_preset_tags"] = _normalize_editable_preset_tags(
                data.get("editable_preset_tags")
            )
            data["editable_preset_enabled"] = _normalize_editable_preset_enabled(
                data.get("editable_preset_enabled")
            )
        except CharacterMakerError as exc:
            print(
                "[CHARACTER_MAKER] 로드 시 생성 프리셋 편집 상태 오류, 초기화: "
                f"path={path}, error={exc}"
            )
            traceback.print_exc()
            data["editable_preset_tags"] = {
                field: [] for field in EDITABLE_PRESET_FIELDS
            }
            data["editable_preset_enabled"] = {
                field: False for field in EDITABLE_PRESET_FIELDS
            }
        # 과거 채팅은 기준을 추측하지 않는다. 기준 메타데이터가 없는 항목은
        # "unknown"으로 보존하되 새 LLM 요청의 체크포인트 문맥에는 포함하지 않는다.
        migrated_chat_items = 0
        normalized_chat: list[dict[str, Any]] = []
        for item in data["chat"]:
            if not isinstance(item, dict):
                print(
                    "[CHARACTER_MAKER] 로드 시 잘못된 채팅 항목 제외: "
                    f"type={type(item).__name__}"
                )
                continue
            normalized = copy.deepcopy(item)
            item_migrated = False
            if normalized.get("base") not in ("user", "llm", "unknown"):
                normalized["base"] = "unknown"
                item_migrated = True
            branch_id = normalized.get("branch_id")
            if not isinstance(branch_id, str):
                normalized["branch_id"] = ""
                item_migrated = True
            normalized["accepted"] = bool(normalized.get("accepted", False))
            checkpoint_id = normalized.get("checkpoint_id")
            if not isinstance(checkpoint_id, str):
                normalized["checkpoint_id"] = ""
                item_migrated = True
            if item_migrated:
                migrated_chat_items += 1
            normalized_chat.append(normalized)
        data["chat"] = normalized_chat[-MAX_CHAT_ITEMS:]
        if migrated_chat_items:
            print(
                "[CHARACTER_MAKER] 기준 메타데이터 없는 과거 채팅을 "
                f"'unknown'으로 보존: count={migrated_chat_items}"
            )
        if not isinstance(data["active_chat_branch_id"], str):
            print("[CHARACTER_MAKER] 잘못된 활성 채팅 분기 ID를 해제합니다.")
            data["active_chat_branch_id"] = ""
        if not isinstance(data["user_chat_checkpoint_id"], str):
            print("[CHARACTER_MAKER] 잘못된 사용자 채팅 체크포인트 ID를 해제합니다.")
            data["user_chat_checkpoint_id"] = ""
        # 과거 리비전은 사용자 생성(source 없음 → "user")으로 표시한다.
        for item in data["revisions"]:
            if not isinstance(item, dict):
                continue
            item.setdefault("source", "user")
            try:
                item["editable_preset_tags"] = _normalize_editable_preset_tags(
                    item.get("editable_preset_tags")
                )
                item["editable_preset_enabled"] = (
                    _normalize_editable_preset_enabled(
                        item.get("editable_preset_enabled")
                    )
                )
            except CharacterMakerError as exc:
                print(
                    "[CHARACTER_MAKER] 로드 시 리비전 생성 프리셋 편집 상태 오류, "
                    f"초기화: revision={item.get('id')}, error={exc}"
                )
                traceback.print_exc()
                item["editable_preset_tags"] = {
                    field: [] for field in EDITABLE_PRESET_FIELDS
                }
                item["editable_preset_enabled"] = {
                    field: False for field in EDITABLE_PRESET_FIELDS
                }
        # 새 설정 키가 추가되어도 기본값으로 채운다(사용자 값은 보존).
        defaults = self._default_settings()
        for key, value in defaults.items():
            data["settings"].setdefault(key, copy.deepcopy(value))

        # 디스크에 없어진 참고 이미지 정리(조용히 버리지 않고 로그).
        kept_references: list[dict[str, Any]] = []
        for item in data["references"]:
            if not isinstance(item, dict):
                continue
            if not os.path.isfile(item.get("path", "")):
                print(
                    f"[CHARACTER_MAKER] 로드 시 누락된 참고 이미지 제외: "
                    f"reference={item.get('id')}, path={item.get('path')}"
                )
                continue
            kept_references.append(item)
        data["references"] = kept_references

        # 디스크에 없어진 리비전 이미지 정리.
        kept_revisions: list[dict[str, Any]] = []
        for item in data["revisions"]:
            if not isinstance(item, dict):
                continue
            if not os.path.isfile(item.get("image_path", "")):
                print(
                    f"[CHARACTER_MAKER] 로드 시 누락된 리비전 이미지 제외: "
                    f"revision={item.get('id')}, path={item.get('image_path')}"
                )
                continue
            kept_revisions.append(item)
        data["revisions"] = kept_revisions

        # 활성 리비전이 정리 중에 사라졌으면 해제.
        active_id = data.get("active_revision_id", "")
        if active_id and not any(
            item.get("id") == active_id for item in data["revisions"]
        ):
            print(f"[CHARACTER_MAKER] 로드 시 활성 리비전이 없어 해제: id={active_id}")
            data["active_revision_id"] = ""

        # LLM(우측) 활성 리비전이 정리 중에 사라졌으면 해제.
        llm_active_id = data.get("llm_active_revision_id", "")
        if llm_active_id and not any(
            item.get("id") == llm_active_id for item in data["revisions"]
        ):
            print(
                f"[CHARACTER_MAKER] 로드 시 LLM 활성 리비전이 없어 해제: id={llm_active_id}"
            )
            data["llm_active_revision_id"] = ""

        session_dir = self._session_dir()
        os.makedirs(os.path.join(session_dir, "references"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "images"), exist_ok=True)
        return data

    def _load_or_create_session(self) -> None:
        session = self._load_session_from_disk()
        if session is None:
            session = self._fresh_session()
            self._install_single_session(session)
            self._persist_session(session)
            print(
                f"[CHARACTER_MAKER] 단일 세션 새로 생성: session={SINGLE_SESSION_ID}, "
                f"boot={self.boot_id}"
            )
        else:
            self._install_single_session(session)
            print(
                f"[CHARACTER_MAKER] 단일 세션 디스크에서 복원: session={SINGLE_SESSION_ID}, "
                f"boot={self.boot_id}, revisions={len(session['revisions'])}, "
                f"references={len(session['references'])}"
            )

    def _persist_session(self, session: dict[str, Any]) -> None:
        """단일 세션 상태를 session.json에 원자적으로 저장한다."""
        try:
            self._atomic_write_json(self._session_json_path(), session)
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 세션 영속화 실패: "
                f"session={session.get('id')}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    def create_session(self, generation_state: Any = None) -> dict[str, Any]:
        """캐릭터 작업만 비우고 생성 설정은 유지한 단일 세션을 만든다."""
        self._reset_single_session(generation_state)
        session = self.sessions[SINGLE_SESSION_ID]
        self._persist_session(session)
        print(
            f"[CHARACTER_MAKER] 세션 리셋(생성 설정 유지): "
            f"session={SINGLE_SESSION_ID}, boot={self.boot_id}"
        )
        return self.public_session(SINGLE_SESSION_ID)

    def _generation_state_for_reset(self, override: Any = None) -> dict[str, Any]:
        """현재 서버 설정 위에 선택적인 최신 브라우저 설정을 검증·병합한다."""
        current = self.sessions.get(SINGLE_SESSION_ID)
        settings = copy.deepcopy(
            current.get("settings") if isinstance(current, dict) else None
        )
        if not isinstance(settings, dict):
            settings = self._default_settings()
        else:
            for key, value in self._default_settings().items():
                settings.setdefault(key, copy.deepcopy(value))

        try:
            editable_tags = _normalize_editable_preset_tags(
                current.get("editable_preset_tags")
                if isinstance(current, dict)
                else None
            )
            editable_enabled = _normalize_editable_preset_enabled(
                current.get("editable_preset_enabled")
                if isinstance(current, dict)
                else None
            )
            if override is not None:
                if not isinstance(override, dict):
                    raise CharacterMakerError("보존할 생성 설정은 객체여야 합니다.")
                allowed = {
                    "settings",
                    "editable_preset_tags",
                    "editable_preset_enabled",
                }
                unknown = sorted(set(override) - allowed)
                if unknown:
                    raise CharacterMakerError(
                        "새 캐릭터 생성 시 보존할 수 없는 필드입니다: "
                        + ", ".join(unknown)
                    )
                if "settings" in override:
                    holder = {"settings": settings}
                    self._update_settings(holder, override.get("settings"))
                    settings = holder["settings"]
                if "editable_preset_tags" in override:
                    raw_tags = override.get("editable_preset_tags")
                    if not isinstance(raw_tags, dict):
                        raise CharacterMakerError(
                            "editable_preset_tags는 객체여야 합니다."
                        )
                    merged_tags = copy.deepcopy(editable_tags)
                    merged_tags.update(raw_tags)
                    editable_tags = _normalize_editable_preset_tags(merged_tags)
                if "editable_preset_enabled" in override:
                    raw_enabled = override.get("editable_preset_enabled")
                    if not isinstance(raw_enabled, dict):
                        raise CharacterMakerError(
                            "editable_preset_enabled는 객체여야 합니다."
                        )
                    merged_enabled = copy.deepcopy(editable_enabled)
                    merged_enabled.update(raw_enabled)
                    editable_enabled = _normalize_editable_preset_enabled(
                        merged_enabled
                    )
        except CharacterMakerError as exc:
            print(
                "[CHARACTER_MAKER] 새 캐릭터 생성 설정 보존 실패: "
                f"error={exc}, override={override!r}"
            )
            traceback.print_exc()
            raise

        # 에셋 워크플로우는 생성 설정 중에서도 전역 config의 현재값이 권위값이다.
        settings["asset_workflow_type"] = self._live_asset_workflow_type()
        return {
            "settings": settings,
            "editable_preset_tags": editable_tags,
            "editable_preset_enabled": editable_enabled,
        }

    def _reset_single_session(self, generation_state: Any = None) -> None:
        preserved = self._generation_state_for_reset(generation_state)
        session_dir = self._session_dir()
        _assert_within(self.temp_root, session_dir)
        if os.path.isdir(session_dir):
            shutil.rmtree(session_dir)
        session = self._fresh_session()
        session["boot_id"] = self.boot_id
        session.update(preserved)
        self.sessions[SINGLE_SESSION_ID] = session
        # 작업 잠금은 기존 객체를 유지한다(현재 연산이 잡고 있을 수 있음).
        self._operation_locks.setdefault(SINGLE_SESSION_ID, asyncio.Lock())

    def _session(self, session_id: str) -> dict[str, Any]:
        # 단일 세션 모델: 식별자와 무관하게 항상 같은 세션을 반환한다.
        # 영속화 파일이 외부에서 삭제되는 등 예외 상황에서도 새로 복구한다.
        session = self.sessions.get(SINGLE_SESSION_ID)
        if session is None:
            print(
                f"[CHARACTER_MAKER] 단일 세션 누락, 복구: boot={self.boot_id}, "
                f"요청={session_id!r}"
            )
            self._load_or_create_session()
            session = self.sessions[SINGLE_SESSION_ID]
        return session

    def _live_asset_workflow_type(self) -> str:
        """에셋 워크플로우 타입은 전역 config(설정→삽화)를 따른다.

        CM 세션이 과거 스냅샷을 영속화하더라도, 표시/생성에는 항상 현재
        전역값을 쓴다. 설정에서 바꾸고 저장하면 CM에 곧바로 반영되도록.
        """
        config = self.config_getter() or {}
        return str(config.get("asset_workflow_type") or "ilxl")

    def public_session(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        active_revision_id = session.get("active_revision_id", "")
        llm_active_revision_id = session.get("llm_active_revision_id", "")
        references = [
            {
                "id": item["id"],
                "name": item["name"],
                "mime": item["mime"],
                "url": f"/api/character_maker/session/{session_id}/reference/{item['id']}",
            }
            for item in session["references"]
        ]
        revisions = [
            {
                "id": item["id"],
                "created_at": item["created_at"],
                "fields": copy.deepcopy(item["fields"]),
                "natural_language": item.get("natural_language", ""),
                "editable_preset_tags": copy.deepcopy(
                    item.get("editable_preset_tags")
                    or {field: [] for field in EDITABLE_PRESET_FIELDS}
                ),
                "editable_preset_enabled": copy.deepcopy(
                    item.get("editable_preset_enabled")
                    or {field: False for field in EDITABLE_PRESET_FIELDS}
                ),
                "note": item.get("note", ""),
                "source": item.get("source", "user"),
                "stable_model_reuse": copy.deepcopy(
                    item.get("stable_model_reuse") or []
                ),
                "url": f"/api/character_maker/session/{session_id}/image/{item['id']}",
                "active": item["id"] == active_revision_id,
                "llm_active": item["id"] == llm_active_revision_id,
            }
            for item in session["revisions"]
        ]
        result = {
            "id": session["id"],
            "boot_id": session["boot_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "world_context": session["world_context"],
            "natural_language": session.get("natural_language", ""),
            "llm_natural_language": session.get("llm_natural_language", ""),
            "fields": copy.deepcopy(session["fields"]),
            "llm_fields": copy.deepcopy(session["llm_fields"]),
            "locks": copy.deepcopy(session["locks"]),
            "editable_preset_tags": copy.deepcopy(
                session["editable_preset_tags"]
            ),
            "editable_preset_enabled": copy.deepcopy(
                session["editable_preset_enabled"]
            ),
            "settings": copy.deepcopy(session["settings"]),
            "chat": copy.deepcopy(session["chat"]),
            "active_chat_branch_id": session.get("active_chat_branch_id", ""),
            "user_chat_checkpoint_id": session.get("user_chat_checkpoint_id", ""),
            "references": references,
            "revisions": revisions,
            "active_revision_id": active_revision_id,
            "llm_active_revision_id": llm_active_revision_id,
            "finalized": copy.deepcopy(session.get("finalized")),
        }
        # 워크플로우 타입은 CM 세션 스냅샷이 아닌 전역 config의 live 값을 따른다.
        result["settings"]["asset_workflow_type"] = self._live_asset_workflow_type()
        return result

    def operation_lock(self, session_id: str) -> asyncio.Lock:
        session = self._session(session_id)
        lock = self._operation_locks.get(session["id"])
        if lock is None:
            print(
                f"[CHARACTER_MAKER] 작업 잠금 캐시 미스, 재생성: "
                f"session={session['id']}"
            )
            lock = asyncio.Lock()
            self._operation_locks[session["id"]] = lock
        return lock

    def update_session(self, session_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CharacterMakerError("세션 변경값은 객체여야 합니다.")
        session = self._session(session_id)
        if "world_context" in payload:
            world_context = payload.get("world_context")
            if not isinstance(world_context, str):
                raise CharacterMakerError("세계관 정보는 문자열이어야 합니다.")
            session["world_context"] = world_context[:20000]
        # 자연어(사용자 영역) — 자유 텍스트, 빈 값 허용.
        if "natural_language" in payload:
            natural_language = payload.get("natural_language")
            if not isinstance(natural_language, str):
                raise CharacterMakerError("자연어 정보는 문자열이어야 합니다.")
            session["natural_language"] = natural_language[:MAX_NATURAL_LANGUAGE_LENGTH]
        # LLM 영역 자연어 — LLM 수정 결과 반영용. 사용자 영역과 분리.
        if "llm_natural_language" in payload:
            llm_natural_language = payload.get("llm_natural_language")
            if not isinstance(llm_natural_language, str):
                raise CharacterMakerError("LLM 자연어 정보는 문자열이어야 합니다.")
            session["llm_natural_language"] = llm_natural_language[
                :MAX_NATURAL_LANGUAGE_LENGTH
            ]
        if "fields" in payload:
            session["fields"] = _normalize_fields(payload.get("fields"))
        if "llm_fields" in payload:
            session["llm_fields"] = _normalize_fields(payload.get("llm_fields"))
        if "locks" in payload:
            session["locks"] = _normalize_locks(payload.get("locks"))
        if "editable_preset_tags" in payload:
            session["editable_preset_tags"] = _normalize_editable_preset_tags(
                payload.get("editable_preset_tags")
            )
        if "editable_preset_enabled" in payload:
            session["editable_preset_enabled"] = (
                _normalize_editable_preset_enabled(
                    payload.get("editable_preset_enabled")
                )
            )
        if "settings" in payload:
            self._update_settings(session, payload.get("settings"))
        if "active_revision_id" in payload:
            revision_id = str(payload.get("active_revision_id") or "")
            if revision_id and not any(
                item["id"] == revision_id for item in session["revisions"]
            ):
                raise CharacterMakerError("선택한 리비전을 찾을 수 없습니다.")
            session["active_revision_id"] = revision_id
        if "llm_active_revision_id" in payload:
            revision_id = str(payload.get("llm_active_revision_id") or "")
            if revision_id and not any(
                item["id"] == revision_id for item in session["revisions"]
            ):
                raise CharacterMakerError("선택한 LLM 리비전을 찾을 수 없습니다.")
            session["llm_active_revision_id"] = revision_id
        session["updated_at"] = _now_iso()
        self._persist_session(session)
        return self.public_session(session_id)

    def _update_settings(self, session: dict[str, Any], raw: Any) -> None:
        if not isinstance(raw, dict):
            raise CharacterMakerError("제작 설정은 객체여야 합니다.")
        settings = session["settings"]
        if "generation_workflow" in raw:
            generation_workflow = str(raw.get("generation_workflow") or "").strip().lower()
            if generation_workflow not in GENERATION_WORKFLOW_TYPES:
                print(
                    "[CHARACTER_MAKER] 생성 워크플로우 설정 거부: "
                    f"value={raw.get('generation_workflow')!r}, "
                    f"allowed={GENERATION_WORKFLOW_TYPES}"
                )
                raise CharacterMakerError(
                    "generation_workflow는 asset 또는 illustration이어야 합니다."
                )
            settings["generation_workflow"] = generation_workflow
        string_fields = (
            "asset_workflow_type",
            "quality_preset",
            "artist_preset",
            "negative_preset",
            "character_negative_preset",
            "anima_quality_preset",
            "anima_artist_preset",
            "anima_negative_preset",
            "snapref_dir",
            "snapref_mode",
        )
        for key in string_fields:
            if key in raw:
                value = raw.get(key)
                if not isinstance(value, str):
                    raise CharacterMakerError(f"{key} 설정은 문자열이어야 합니다.")
                settings[key] = value.strip()[:200]
        for key, low, high in (("img_w", 256, 4096), ("img_h", 256, 4096)):
            if key in raw:
                value = int(raw.get(key))
                if not low <= value <= high:
                    raise CharacterMakerError(f"{key}는 {low}~{high} 사이여야 합니다.")
                settings[key] = value
        if "seed" in raw:
            seed = int(raw.get("seed"))
            if not -1 <= seed <= 2**32 - 1:
                raise CharacterMakerError("seed는 -1 또는 0~4294967295 사이여야 합니다.")
            settings["seed"] = seed
        for key in ("rag_enabled", "lora_enabled"):
            if key in raw:
                settings[key] = bool(raw.get(key))
        for key in (
            "style_lora_enabled",
            "snapref_enabled",
            "face_lora_enabled",
            "hrf_sdxl",
            "hrf_anima",
            "hrf_moss",
            "hrf_restore_size",
            "hrf_control_net",
            "sdxl_fd_enabled",
            "sdxl_hd_enabled",
            "sdxl_ed_enabled",
            "anima_fd_enabled",
            "anima_hd_enabled",
            "anima_ed_enabled",
        ):
            if key in raw:
                settings[key] = bool(raw.get(key))
        if "lora_list" in raw:
            settings["lora_list"] = _normalize_lora_list(raw.get("lora_list"))
        for key in ("style_lora_list", "face_lora_list"):
            if key in raw:
                settings[key] = _normalize_lora_list(raw.get(key))
        if "snapref_images" in raw:
            images = raw.get("snapref_images")
            if isinstance(images, list):
                settings["snapref_images"] = [
                    copy.deepcopy(image) for image in images if isinstance(image, dict)
                ][:1]
            else:
                print(
                    "[CHARACTER_MAKER] Ref 이미지 설정 유지: "
                    f"목록 형식이 아님, input={images!r}, "
                    f"current={settings.get('snapref_images', [])!r}"
                )
        for key in ("face_tags", "eye_tags", "face_lora_upscale_size"):
            if key in raw:
                value = raw.get(key)
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise CharacterMakerError(f"{key} 설정은 문자열 또는 숫자여야 합니다.")
                settings[key] = str(value).strip()[:500]
        if "hrf_size" in raw:
            try:
                hrf_size = float(raw.get("hrf_size"))
            except (TypeError, ValueError) as exc:
                raise CharacterMakerError("hrf_size는 숫자여야 합니다.") from exc
            if not math.isfinite(hrf_size) or not 1.0 <= hrf_size <= 3.0:
                raise CharacterMakerError("hrf_size는 1.0~3.0 사이여야 합니다.")
            settings["hrf_size"] = hrf_size
        for key, low, high in (
            ("face_crop_top", 0.0, 10.0),
            ("face_crop_bottom", 0.0, 10.0),
            ("snapref_strength", -3.0, 3.0),
            ("snapref_megapixels", 0.25, 2.0),
        ):
            if key in raw:
                try:
                    value = float(raw.get(key))
                except (TypeError, ValueError) as exc:
                    print(f"[CHARACTER_MAKER] 생성 옵션 숫자 파싱 실패: key={key}, input={raw.get(key)!r}")
                    traceback.print_exc()
                    raise CharacterMakerError(f"{key}는 숫자여야 합니다.") from exc
                if not math.isfinite(value) or not low <= value <= high:
                    print(f"[CHARACTER_MAKER] 생성 옵션 범위 오류: key={key}, input={value!r}, range=({low}, {high})")
                    raise CharacterMakerError(f"{key}는 {low}~{high} 사이여야 합니다.")
                settings[key] = value

    def add_reference(
        self, session_id: str, *, filename: str, mime: str, image_bytes: bytes
    ) -> dict[str, Any]:
        session = self._session(session_id)
        if len(session["references"]) >= MAX_REFERENCE_COUNT:
            raise CharacterMakerError(f"참고 이미지는 최대 {MAX_REFERENCE_COUNT}장입니다.")
        if not image_bytes:
            raise CharacterMakerError("참고 이미지 데이터가 비어 있습니다.")
        if len(image_bytes) > MAX_REFERENCE_BYTES:
            raise CharacterMakerError("참고 이미지 한 장은 12MB를 넘을 수 없습니다.")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                fmt = (image.format or "").upper()
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 참고 이미지 검증 실패: "
                f"session={session_id}, filename={filename!r}, error={exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError("올바른 이미지 파일이 아닙니다.") from exc

        ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}
        ext = ext_map.get(fmt, ".img")
        detected_mime = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }.get(fmt, mime or "application/octet-stream")
        reference_id = uuid.uuid4().hex
        path = os.path.join(self.temp_root, session_id, "references", reference_id + ext)
        _assert_within(self.temp_root, path)
        with open(path, "wb") as handle:
            handle.write(image_bytes)
        item = {
            "id": reference_id,
            "name": os.path.basename(filename or f"reference{ext}")[:200],
            "mime": detected_mime,
            "path": path,
        }
        session["references"].append(item)
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] 참고 이미지 추가: session={session_id}, "
            f"reference={reference_id}, bytes={len(image_bytes)}, format={fmt}"
        )
        self._persist_session(session)
        return self.public_session(session_id)

    def remove_reference(self, session_id: str, reference_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        target = next(
            (item for item in session["references"] if item["id"] == reference_id),
            None,
        )
        if target is None:
            print(
                f"[CHARACTER_MAKER] 참고 이미지 삭제 실패: "
                f"session={session_id}, reference={reference_id}"
            )
            raise CharacterMakerError("참고 이미지를 찾을 수 없습니다.")
        path = target["path"]
        _assert_within(self.temp_root, path)
        if os.path.isfile(path):
            os.remove(path)
        session["references"] = [
            item for item in session["references"] if item["id"] != reference_id
        ]
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] 참고 이미지 삭제: "
            f"session={session_id}, reference={reference_id}"
        )
        self._persist_session(session)
        return self.public_session(session_id)

    def reference_path(self, session_id: str, reference_id: str) -> tuple[str, str]:
        session = self._session(session_id)
        item = next(
            (item for item in session["references"] if item["id"] == reference_id),
            None,
        )
        if item is None or not os.path.isfile(item["path"]):
            print(
                f"[CHARACTER_MAKER] 참고 이미지 조회 실패: "
                f"session={session_id}, reference={reference_id}"
            )
            raise CharacterMakerError("참고 이미지를 찾을 수 없습니다.")
        _assert_within(self.temp_root, item["path"])
        return item["path"], item["mime"]

    def _active_revision(self, session: dict[str, Any]) -> dict[str, Any] | None:
        active_id = session.get("active_revision_id", "")
        if not active_id:
            return None
        return next(
            (item for item in session["revisions"] if item["id"] == active_id),
            None,
        )

    def _active_revision_for(
        self, session: dict[str, Any], base: str
    ) -> dict[str, Any] | None:
        """base 가 'llm' 이면 우측(LLM) 활성 리비전, 그 외는 좌측(사용자) 활성 리비전."""
        if base == "llm":
            active_id = session.get("llm_active_revision_id", "")
        else:
            active_id = session.get("active_revision_id", "")
        if not active_id:
            return None
        return next(
            (item for item in session["revisions"] if item["id"] == active_id),
            None,
        )

    def _encode_vision_image(self, path: str) -> tuple[str, str] | None:
        """단일 이미지를 원본 bytes + 감지된 mime 로 (b64, mime) 반환.

        포맷(PNG/WEBP)과 해상도 가공은 하지 않는다 — LLM 전송 포맷은 전역 설정
        (llm_service._normalize_vision_image + llm_vision_compress)에서만 결정한다.
        다중 비전에서 CURRENT/REF 각각이 이 결과 한 장씩으로 전송된다. 실패 시 None.
        """
        _fmt_to_mime = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }
        try:
            with Image.open(path) as source:
                fmt = (source.format or "").upper()
            mime = _fmt_to_mime.get(fmt, "application/octet-stream")
            with open(path, "rb") as handle:
                raw = handle.read()
            if not raw:
                print(f"[CHARACTER_MAKER] 비전 이미지 파일이 비어 있음: path={path}")
                return None
            return base64.b64encode(raw).decode("ascii"), mime
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 비전 이미지 인코딩 실패: "
                f"path={path}, error={exc}"
            )
            traceback.print_exc()
            return None

    @staticmethod
    def _banner_font(size: int):
        for font_name in (
            "arialbd.ttf",
            "segoeuib.ttf",
            "DejaVuSans-Bold.ttf",
            "LiberationSans-Bold.ttf",
        ):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default(size=size)

    def _encode_vision_image_with_banner(
        self,
        path: str,
        *,
        banner_text: str,
        short_banner: str,
        banner_color: str,
    ) -> tuple[str, str] | None:
        """Return a base64 PNG copy with a padded color header banner above the image.

        Never modifies the original file on disk. Falls back to _encode_vision_image
        when path is not on disk (e.g. unit-test stubs) or if rendering fails.
        """
        if not path or not os.path.isfile(path):
            return self._encode_vision_image(path)
        try:
            with Image.open(path) as source:
                oriented = ImageOps.exif_transpose(source)
                if oriented.mode in ("RGBA", "LA") or (
                    oriented.mode == "P" and "transparency" in oriented.info
                ):
                    rgba = oriented.convert("RGBA")
                    rgb = Image.new("RGB", rgba.size, "#ffffff")
                    rgb.paste(rgba, mask=rgba.split()[-1])
                else:
                    rgb = oriented.convert("RGB")
            width, height = rgb.size
            banner_h = max(44, min(140, int(round(width * 0.065))))
            canvas = Image.new("RGB", (width, height + banner_h), banner_color)
            canvas.paste(rgb, (0, banner_h))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, banner_h - 2, width, banner_h - 1), fill="#ffffff")
            max_text_w = max(40, width - 24)
            chosen_text = short_banner
            chosen_font = self._banner_font(14)
            matched = False
            for candidate_text in (banner_text, short_banner):
                size = max(16, int(round(banner_h * 0.48)))
                while size >= 14:
                    font = self._banner_font(size)
                    bbox = draw.textbbox((0, 0), candidate_text, font=font)
                    if (bbox[2] - bbox[0]) <= max_text_w:
                        chosen_text = candidate_text
                        chosen_font = font
                        matched = True
                        break
                    size -= 2
                if matched:
                    break
            bbox = draw.textbbox((0, 0), chosen_text, font=chosen_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = max(12, (width - text_w) // 2)
            text_y = max(2, (banner_h - 2 - text_h) // 2 - bbox[1])
            draw.text((text_x, text_y), chosen_text, fill="#ffffff", font=chosen_font)
            out = io.BytesIO()
            canvas.save(out, format="PNG")
            raw = out.getvalue()
            if not raw:
                return None
            return base64.b64encode(raw).decode("ascii"), "image/png"
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 배너 비전 이미지 생성 실패, 원본 인코딩으로 폴백: "
                f"path={path}, error={exc}"
            )
            traceback.print_exc()
            return self._encode_vision_image(path)

    def _encode_reference_bbox_crops(
        self,
        reference_entries: list[dict[str, Any]],
        regions: list[dict[str, Any]],
        *,
        banner_color: str = "#be123c",
    ) -> list[dict[str, Any]]:
        """Crop AI-selected normalized [0..1000] bounding boxes from reference images.

        Works for any aspect ratio or layout (portrait, landscape, turnaround sheets,
        seated poses, bust shots, or cards with margins). Adds a small safety padding
        margin around each region and upscales cleanly so tiny details are legible.
        """
        if not reference_entries or not regions:
            return []
        loaded_images: dict[int, Image.Image] = {}
        crops: list[dict[str, Any]] = []
        try:
            for crop_idx, region in enumerate(regions[:MAX_AI_BBOX_CROPS], start=1):
                raw_idx = int(region.get("image_index", 1) or 1)
                ref_idx = raw_idx if 1 <= raw_idx <= len(reference_entries) else 1
                entry = reference_entries[ref_idx - 1]
                path = str(entry.get("path") or "")
                image_id = str(entry.get("image_id") or f"USER_UPLOADED_{ref_idx}")
                ref_name = str(entry.get("name") or "")
                if not path or not os.path.isfile(path):
                    continue
                rgb = loaded_images.get(ref_idx)
                if rgb is None:
                    with Image.open(path) as source:
                        oriented = ImageOps.exif_transpose(source)
                        if oriented.mode in ("RGBA", "LA") or (
                            oriented.mode == "P" and "transparency" in oriented.info
                        ):
                            rgba = oriented.convert("RGBA")
                            rgb = Image.new("RGB", rgba.size, "#ffffff")
                            rgb.paste(rgba, mask=rgba.split()[-1])
                        else:
                            rgb = oriented.convert("RGB")
                    loaded_images[ref_idx] = rgb
                width, height = rgb.size
                if width < 64 or height < 64:
                    continue
                x1 = int(region["x1"])
                y1 = int(region["y1"])
                x2 = int(region["x2"])
                y2 = int(region["y2"])
                # Expand slightly (~4% of box size) so ornament tips or borders aren't clipped.
                pad_x = max(10, int(round((x2 - x1) * 0.04)))
                pad_y = max(10, int(round((y2 - y1) * 0.04)))
                px0 = max(0, int(round(width * max(0, x1 - pad_x) / 1000.0)))
                py0 = max(0, int(round(height * max(0, y1 - pad_y) / 1000.0)))
                px1 = min(width, int(round(width * min(1000, x2 + pad_x) / 1000.0)))
                py1 = min(height, int(round(height * min(1000, y2 + pad_y) / 1000.0)))
                if (px1 - px0) < 24 or (py1 - py0) < 24:
                    continue
                crop = rgb.crop((px0, py0, px1, py1))
                max_side = max(crop.width, crop.height)
                target_max = max(512, min(768, int(round(max_side * 1.6))))
                scale = target_max / max(1, max_side)
                new_w = max(240, int(round(crop.width * scale)))
                new_h = max(64, int(round(crop.height * scale)))
                if new_w > 768:
                    ratio = 768.0 / new_w
                    new_w = 768
                    new_h = max(64, int(round(new_h * ratio)))
                resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
                banner_h = 38
                canvas = Image.new("RGB", (new_w, banner_h + new_h), banner_color)
                canvas.paste(resized, (0, banner_h))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, banner_h - 2, new_w, banner_h - 1), fill="#ffffff")
                label = str(region.get("label") or f"Detail {crop_idx}").strip()[:64]
                crop_id = f"{image_id}_CROP_{crop_idx}"
                banner_text = f"{crop_id} | {label}"
                chosen_text = crop_id
                chosen_font = self._banner_font(14)
                for candidate in (banner_text, crop_id):
                    size = 17
                    while size >= 12:
                        font = self._banner_font(size)
                        bbox = draw.textbbox((0, 0), candidate, font=font)
                        if (bbox[2] - bbox[0]) <= new_w - 16:
                            chosen_text = candidate
                            chosen_font = font
                            break
                        size -= 1
                    else:
                        continue
                    break
                tb = draw.textbbox((0, 0), chosen_text, font=chosen_font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
                draw.text(
                    (max(8, (new_w - tw) // 2), max(2, (banner_h - 2 - th) // 2 - tb[1])),
                    chosen_text,
                    fill="#ffffff",
                    font=chosen_font,
                )
                out = io.BytesIO()
                canvas.save(out, format="PNG")
                raw = out.getvalue()
                if raw:
                    crops.append({
                        "image_id": crop_id,
                        "parent_image_id": image_id,
                        "reference_index": ref_idx,
                        "name": ref_name,
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "b64": base64.b64encode(raw).decode("ascii"),
                        "mime": "image/png",
                        "description": (
                            f"AI-selected close-up crop #{crop_idx} of {image_id} "
                            f"(bbox [{x1},{y1},{x2},{y2}]): {label}."
                        ),
                    })
            return crops
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] BBox 확대 크롭 생성 실패, 스킵: error={exc}"
            )
            return []

    def _revision_vision_inputs(
        self, session: dict[str, Any], *, base: str = "user"
    ) -> list[dict[str, Any]]:
        """revise 비전 이미지와 역할 manifest를 함께 준비한다.

        CURRENT가 없거나 인코딩에 실패해도 남은 REF의 역할이 CURRENT로 밀리지 않도록
        각 이미지에 결정론적인 role 메타데이터를 유지한다. 실제 provider 전송은
        ``(b64, mime)`` 배열이지만, 같은 순서의 manifest가 텍스트 입력에 포함된다.
        """
        candidates: list[dict[str, Any]] = []
        active = self._active_revision_for(session, base)
        if active:
            active_path = active.get("image_path", "")
            if active_path and os.path.isfile(active_path):
                candidates.append(
                    {
                        "role": "CURRENT",
                        "path": active_path,
                        "revision_id": str(active.get("id") or ""),
                        "source": base,
                    }
                )
            else:
                print(
                    f"[CHARACTER_MAKER] CURRENT 이미지 파일 없음: "
                    f"session={session['id']}, base={base}, path={active_path!r}"
                )
        for index, item in enumerate(
            session["references"][:MAX_REFERENCE_COUNT], start=1
        ):
            reference_path = item.get("path", "")
            if reference_path and os.path.isfile(reference_path):
                candidates.append(
                    {
                        "role": "REF",
                        "path": reference_path,
                        "reference_index": index,
                        "reference_id": str(item.get("id") or ""),
                    }
                )
            else:
                print(
                    f"[CHARACTER_MAKER] REF 이미지 파일 없음: "
                    f"session={session['id']}, reference={item.get('id')}, "
                    f"path={reference_path!r}"
                )
        if not candidates:
            print(
                f"[CHARACTER_MAKER] 비전 입력 없음: "
                f"session={session['id']}, active_revision 없음, reference 없음"
            )
            return []

        prepared: list[dict[str, Any]] = []
        for candidate in candidates:
            encoded = self._encode_vision_image(candidate["path"])
            if not encoded:
                print(
                    f"[CHARACTER_MAKER] 비전 이미지 스킵: "
                    f"session={session['id']}, role={candidate['role']}, "
                    f"path={candidate['path']}"
                )
                continue
            image_b64, image_mime = encoded
            manifest = {
                "position": len(prepared) + 1,
                "role": candidate["role"],
            }
            if candidate["role"] == "CURRENT":
                manifest.update(
                    {
                        "source": candidate["source"],
                        "revision_id": candidate["revision_id"],
                        "description": "Latest selected generated image to modify.",
                    }
                )
            else:
                manifest.update(
                    {
                        "reference_index": candidate["reference_index"],
                        "reference_id": candidate["reference_id"],
                        "description": "User-provided visual reference; do not treat as the edit target.",
                    }
                )
            prepared.append(
                {
                    "b64": image_b64,
                    "mime": image_mime,
                    "manifest": manifest,
                }
            )
        if not prepared:
            raise CharacterMakerError("비전 입력 이미지를 준비하지 못했습니다.")
        return prepared

    def _revision_vision_images(
        self, session: dict[str, Any], *, base: str = "user"
    ) -> list[tuple[str, str]]:
        """하위 호환용 이미지 배열. 새 호출 경로는 역할 manifest도 함께 사용한다."""
        return [
            (item["b64"], item["mime"])
            for item in self._revision_vision_inputs(session, base=base)
        ]

    @staticmethod
    def _image_legend(image_manifest: list[dict[str, Any]]) -> str:
        if not image_manifest:
            return "No images are attached. Reason only from text and current fields."
        has_banners = any(item.get("banner_color") for item in image_manifest)
        has_current = any(item.get("role") == "CURRENT" for item in image_manifest)
        has_zoom = any(item.get("role") == "REF_DETAIL_ZOOM" for item in image_manifest)
        zoom_note = (
            " Images marked *_CROP_* are magnified close-up regions cropped directly from the "
            "corresponding USER_UPLOADED_* reference image — inspect these close-ups carefully "
            "for small ornaments, bangs coverage, collar/trim edges, bodice panels, legwear, "
            "footwear, and held props, and never treat them as separate characters or panels."
            if has_zoom
            else ""
        )
        if has_banners and has_current:
            return (
                "Images are attached separately in image_manifest order. "
                "Identify each image by the colored header banner added above its top edge: "
                "the blue GENERATED_CURRENT banner marks the existing generated character image "
                "to inspect/edit, and red USER_UPLOADED_1, USER_UPLOADED_2, ... banners mark "
                "reference images uploaded by the user in this turn. "
                "Never swap GENERATED_CURRENT and USER_UPLOADED_*, and ignore the added "
                "header banners when writing tags or descriptions."
                + zoom_note
            )
        if has_banners:
            return (
                "No GENERATED_CURRENT image is attached. Every attached image is a user-uploaded "
                "reference image marked with a red USER_UPLOADED_1, USER_UPLOADED_2, ... banner "
                "above its top edge. Do not treat a USER_UPLOADED_* image as a prior generated "
                "output, and ignore the added header banners when writing tags or descriptions."
                + zoom_note
            )
        if has_current:
            return (
                "Images are attached separately in image_manifest order. "
                "Exactly one CURRENT image is the edit target. Every REF image is "
                "user-provided reference evidence and is not the edit target."
                + zoom_note
            )
        return (
            "No CURRENT image is attached. Every attached image is a REF supplied "
            "only as reference evidence; do not treat the first REF as a generated "
            "image to modify."
            + zoom_note
        )

    @staticmethod
    def _chat_context_for_request(
        session: dict[str, Any],
        *,
        base: str,
        branch_id: str,
        current_message_id: str,
    ) -> list[dict[str, Any]]:
        """기준별 LLM 문맥을 사용자 체크포인트와 활성 분기로 제한한다."""
        selected: list[dict[str, Any]] = []
        for item in session.get("chat", []):
            if item.get("id") == current_message_id:
                continue
            accepted = bool(item.get("accepted", False))
            active_branch_item = (
                base == "llm"
                and bool(branch_id)
                and item.get("branch_id") == branch_id
            )
            if not accepted and not active_branch_item:
                continue
            selected.append(
                {
                    "role": "assistant"
                    if item.get("role") == "assistant"
                    else "user",
                    "content": str(item.get("content") or ""),
                    "base": item.get("base")
                    if item.get("base") in ("user", "llm")
                    else "unknown",
                    "accepted": accepted,
                }
            )
        return selected[-MAX_CHAT_CONTEXT_ITEMS:]

    def _revision_messages(
        self,
        session: dict[str, Any],
        feedback: str,
        *,
        rag_enabled: bool,
        base: str = "user",
        branch_id: str = "",
        current_message_id: str = "",
        image_manifest: list[dict[str, Any]] | None = None,
        model_family: str = "anima",
    ) -> list[dict[str, str]]:
        resolved_family = normalize_maker_model_family(model_family)
        vision_banner_rules = (
            "When attached images carry a colored header banner above the top edge, read that "
            "banner to identify each image reliably: blue GENERATED_CURRENT is the existing "
            "generated character image to inspect/fix, red USER_UPLOADED_1, USER_UPLOADED_2, "
            "... are reference images uploaded by the user in the current turn, and red "
            "USER_UPLOADED_*_DETAIL_ZOOM images are 2x2 magnified close-up crop sheets of the "
            "same reference image (1:Head/Headpiece, 2:Torso/Sleeves/Props, 3:Waist/Hips/Skirt, "
            "4:Legs/Footwear). Always inspect both the full reference image and its *_DETAIL_ZOOM "
            "close-ups carefully to verify fine details; never confuse or swap GENERATED_CURRENT "
            "with USER_UPLOADED_*, never treat a 2x2 zoom sheet as a multi-panel layout or "
            "multiple characters, and never describe the added header banners or zoom labels. "
        )
        prior_turn_rules = (
            "Prior turns in recent_conversation may refer to earlier images that are no longer "
            "attached; base visual observations on the images attached to the current turn "
            "rather than carrying over position cues from earlier turns (such as 'leftmost "
            "character') onto a different image. "
            "Keep character fields focused on the character rather than background scenery or "
            "weather effects unless the user explicitly asks for them. "
        )
        natural_language_restrictions = (
            "STRICT RULE FOR top-level natural_language (MANDATORY): "
            "natural_language is a reusable character-level field shared across different "
            "expression presets, so NEVER include facial expressions, emotions, personality or "
            "affect adjectives, gaze cues, pose descriptions, or atmosphere/mood/vibe/lighting-mood "
            "descriptions in natural_language. Forbidden in natural_language: smiles, frowns, "
            "blushes, winks, tears, eye-contact/gaze (e.g. 'smiling warmly', 'smiling cheerfully', "
            "'looking at the viewer', 'cheerful', 'confident', 'shy', 'angry', 'sad', 'serene', "
            "'playful', 'emotional intent', 'posing confidently'), and atmosphere/mood/vibe/"
            "lighting-mood wording (e.g. 'warm atmosphere', 'cozy mood', 'ethereal vibe', "
            "'dramatic mood', 'melancholic ambiance', 'mystical aura', 'moody lighting'). "
            "Put ALL facial expressions, emotions, and gaze exclusively in fields.expression, "
            "and ALL pose/framing exclusively in fields.composition — never in natural_language "
            "(and never in fields.appearance or fields.outfit). "
            "If current_natural_language already contains any expression, emotion, pose, mood, "
            "atmosphere, or redundant full-character summary and natural_language is unlocked, "
            "actively delete those parts. "
        )
        has_attached_images = bool(image_manifest)
        if resolved_family == "moss":
            visual_audit_rules = (
                "PRE-WRITE VISUAL AUDIT (when images are attached): "
                "Emit a compact 'visual_audit' object as the FIRST JSON key, written in concise, "
                "concrete English (NEVER Korean) with 5 short literal entries inspected from BOTH "
                "the full image and any *_CROP_* close-ups: "
                "head_hair_wings (hair color/ponytail sweep side, full blunt bangs vs parted forehead, "
                "spiral curls/ringlets, exact headpiece silhouette/gems/ribbons, eyes, ear tilt, exact "
                "count & structure of wings/tails/horns), "
                "shoulders_bodice_sleeves (explicitly separate character's RIGHT shoulder vs LEFT "
                "shoulder — which is bare vs covered by a mantle/cape — exact vertical ending length "
                "of any cape/mantle, and which exact edge has fur/trim; choker vs high collar; bodice "
                "neckline cut & CENTER panel color/bars vs SIDE panels color/clasps; chest bow/brooch; "
                "RIGHT sleeve/glove vs LEFT sleeve/glove), "
                "waist_hips_skirt_overskirt (sash solid vs striped; explicitly separate RIGHT hip vs "
                "LEFT hip bows/crests/ruffles; inner skirt shape & hem emblem; exact ending length of "
                "overskirt/peplum flaps relative to mid-thigh/knees), "
                "legwear_footwear_props (exact legwear: seamless full pantyhose/tights vs thigh-highs "
                "vs socks vs bare legs + pattern; exact boot shaft height/stripes/cuff gems/heels; "
                "prop relative length, shaft color/thickness, grip point relative to finial, finial "
                "shape, angle), "
                "and mismatches_to_fix (list of mismatches between GENERATED_CURRENT and USER_UPLOADED_* "
                "naming exact old phrases in current_fields to delete, or [] on first turn). "
                "MANDATORY: Every color, asymmetry, bare-skin zone, and ornament recorded in "
                "visual_audit MUST be explicitly transferred into fields.appearance or fields.outfit "
                "— never drop a color or collapse an asymmetric left/right garment into a symmetric one. "
                if has_attached_images
                else ""
            )
            system = (
                "You are a collaborative character-design editor for an image-generation UI "
                "operating in MOSS mode (Qwen3.5 natural-language text encoder). Reason from world "
                "context, conversation, visual evidence, references, and user feedback. Do not use "
                "hard-coded keyword matching. Modify four fields (appearance, outfit, expression, "
                "composition) and top-level natural_language. Preserve locked fields exactly. "
                + vision_banner_rules
                + visual_audit_rules
                + "When USER_UPLOADED_* images are attached: if the user asks to transfer only "
                "specific attributes, extract those attributes while preserving unrelated fields; "
                "when recreating a reference character or fixing GENERATED_CURRENT against "
                "USER_UPLOADED_*, inspect every zone from head to footwear using both full and "
                "*_CROP_* close-up images, and actively DELETE/REPLACE every wrong, generic, or "
                "contradictory phrase in current_fields rather than appending onto broken text. "
                + prior_turn_rules
                + natural_language_restrictions
                + "MOSS natural-language rules (MANDATORY): "
                "1. Concrete, dense English prose across ALL image fields (no Danbooru tag lists). "
                "Cover every visible zone thoroughly without filler or repetition across fields. "
                "2. Positive-only phrasing (CRITICAL): NEVER write negative phrasing inside fields "
                "or natural_language ('no X', 'without X', 'not pleated', 'no bare thighs', 'instead "
                "of thigh-highs') — naming forbidden items causes MOSS to draw them. State ONLY the "
                "positive visible form (e.g., 'smooth sleek teal sleeve', 'smooth gently scalloped "
                "flared white mini-skirt', 'seamless full-length sheer milky-white pantyhose tights "
                "covering the entire legs from under the skirt down into the boots'). "
                "3. Avoid strong-prior generic stereotypes that distort actual geometry: do not call "
                "a narrow pointed spire/circlet a generic 'crown tiara'; do not call a slender short "
                "baton a 'magic staff' or 'scepter'; do not call short mid-thigh overskirt flaps "
                "'coat-tails' (write 'short mid-thigh-length peplum overskirt panels ending well "
                "above the knees'); do not call a smooth flared skirt 'pleated' unless crisp pleats "
                "are visible; for an asymmetric one-shoulder mantle, explicitly name which shoulder "
                "is covered vs bare and which exact edge has fur/trim (top/inner collar edge vs "
                "bottom hem); when bangs cover the forehead, explicitly write 'full blunt bangs "
                "completely covering her entire forehead down to her eyebrows' and place forehead "
                "ornaments 'resting directly across the top of her bangs' (never 'onto her forehead'). "
                "4. Explicit L/R asymmetry, color-zone ownership, exact counts, and prop anatomy: "
                "assign each color/pattern (solid vs striped vs cross-grid) to its exact garment "
                "zone (center corset panel vs side panels vs chest bow vs sash vs underlayer); state "
                "explicit left-vs-right placement for shoulders, sleeves, gloves, hip bows/crests/"
                "ruffles, ponytail sweep side, and props; state exact counts ('exactly four wings — "
                "one upper pair and one lower pair'); for held props state relative length, shaft "
                "thickness/color, exact grip position right beneath or along the shaft relative to "
                "the finial, finial motif/size, and shaft angle/tip. "
                "5. Official Danbooru character tag exception: ONLY for an established character "
                "with a confirmed official Danbooru tag (or explicitly supplied), place that exact "
                "tag as the FIRST element of fields.appearance (lowercase, spaces, unescaped "
                "parentheses e.g. 'asuna (blue archive)'). Omit it for original/unidentified characters. "
                "6. Field scope: "
                "fields.appearance = build, gender/age presentation ('a slender young woman'), hair "
                "color/style/sweep/bangs/curls, headpiece/ribbons, eyes, facial marks, ears/wings/"
                "tails/horns. "
                "fields.outfit = top-to-bottom clothing: bare skin zones, choker/collar, asymmetric "
                "mantle & exact trim edges, bodice cut & color zones, L/R sleeves & gloves, sash & "
                "L/R hip ornaments, inner skirt shape/emblem, overskirt/peplum length/borders, exact "
                "legwear (pantyhose vs thigh-highs vs socks vs bare), footwear, and held prop. "
                "fields.expression = ONLY facial expression, brows, eyes, mouth, blush, gaze. "
                "fields.composition = ONLY framing/shot scale, angle, posture, hand placement. When "
                "USER_UPLOADED_* shows a full-body outfit and no tight crop is requested, frame as a "
                "full-body standing portrait showing head to footwear so all regions can be verified. "
                "NEVER copy background colors, patterns, UI cards, text, speech bubbles, floating "
                "sparkles, or decorative background motifs from USER_UPLOADED_* into fields.composition "
                "or any other field — background is controlled separately by the style preset. "
                "7. Top-level natural_language: NEVER write a full-character recap summary or any "
                "expression/emotion/gaze/mood/pose wording. Default to '' (empty string); for complex "
                "reference characters, optionally write ONE short strictly expression-free structural "
                "disambiguation sentence reinforcing ONLY 3-5 drift-prone physical constraints (e.g., "
                "full blunt bangs covering forehead, exact wing count, short mid-thigh overskirt "
                "length, asymmetric left-only mantle / bare right shoulder, prop grip point) without "
                "repeating a general character summary. "
                "8. Forbidden tokens: rewrite legacy Danbooru tags on unlocked fields into natural "
                "English. Never emit subject-count tags (1girl, solo), underscores, emphasis weights "
                "((tag:1.2)), quality/score/rating/artist/LoRA triggers, layout JSON/bboxes, abstract "
                "vibe/aura words, or figurative animal/object similes. "
                "9. Single-character scope: if the user requests a minimal off-screen limb touching "
                "the character, keep the character as the sole full subject and describe only the "
                "anonymous partial limb and contact point. "
                "10. Output JSON: return one JSON object with "
                + ("visual_audit (first, in English), " if has_attached_images else "")
                + "fields, natural_language, rag_queries, and assistant_message. "
                "Write assistant_message in the user's language (Korean when the user writes in "
                "Korean); write visual_audit, fields, and natural_language in English only. "
                "fields has exactly appearance/outfit/expression/composition string arrays (one "
                "coherent prose string per field, or [official_danbooru_tag, prose] for appearance; "
                "[] allowed). natural_language is a TOP-LEVEL key, never inside fields. rag_queries "
                "has the same four keys mapping to string arrays ([] allowed when RAG is disabled). "
                f"Danbooru RAG is {'enabled' if rag_enabled else 'disabled'}."
            )
        else:
            model_guidance = (
                "Natural-language guidance for the Anima model (which uses a Qwen LLM text encoder "
                "and understands descriptive sentences well): when natural_language is used, write "
                "concrete, neutral English sentences describing physical appearance details, outfit "
                "construction, accessory/prop placement, or spatial structure that pure tags cannot "
                "convey precisely (e.g. 'a large blue peony flower tucked behind her left ear'); "
                "never include facial expressions, emotions, mood, or atmosphere in natural_language "
                "(keep all expression/emotion details in fields.expression); natural_language may be "
                "empty string '' when tags already cover the visual details. "
            )
            system = (
                "You are a collaborative character-design editor for an image-generation UI. "
                "Reason from the world context, provided checkpoint/branch conversation, current "
                "visual evidence, references, and user feedback. Do not use hard-coded keyword "
                "matching. "
                "You may modify four tag fields (appearance, outfit, expression, composition) "
                "and a SEPARATE top-level field natural_language. "
                "All other generation settings are read-only presets. Preserve locked fields exactly. "
                + vision_banner_rules
                + "When USER_UPLOADED_* (REF) images are attached: if the user asks to transfer only "
                "specific attributes (for example, outfit only), extract all visible details of "
                "those requested attributes into concrete tags and natural_language while preserving "
                "unrelated fields of the current character; if the user asks to recreate the "
                "reference character or compare GENERATED_CURRENT against USER_UPLOADED_* to fix "
                "wrong or missing tags, systematically inspect every visible region — hairstyle "
                "(ponytail, twin tails, braids, half-up, loose hair, bangs, side curls, hair "
                "highlights/inner color), hair ornaments, eye color, ears, wings/tails/horns, "
                "collar or shoulder mantle/cape, dress/top/sleeves/waist/ribbons and their exact "
                "placement, skirt layers, legwear/tights/socks, footwear, and held props — actively "
                "remove tags in current_fields that contradict USER_UPLOADED_* or caused wrong "
                "features in GENERATED_CURRENT, and add every prominent feature visible in "
                "USER_UPLOADED_* that is missing from GENERATED_CURRENT. "
                + prior_turn_rules
                + natural_language_restrictions
                + "natural_language is optional neutral descriptive text inserted into the image "
                "prompt; it may be empty. "
                + model_guidance
                + "Return one JSON object with assistant_message, fields, rag_queries, and optionally "
                "natural_language. "
                "fields must contain exactly appearance/outfit/expression/composition string arrays. "
                "natural_language is a TOP-LEVEL key (sibling of fields and rag_queries), NEVER a key "
                "inside fields. "
                "rag_queries must contain the same four keys, each mapping to a non-empty ARRAY of "
                "short Korean or English semantic search unit strings (never a bare string). "
                "Tags should describe visible, image-generatable details. "
                f"Danbooru RAG is {'enabled' if rag_enabled else 'disabled'}."
            )
        manifest = copy.deepcopy(image_manifest or [])
        history = self._chat_context_for_request(
            session,
            base=base,
            branch_id=branch_id,
            current_message_id=current_message_id,
        )
        base_fields = (
            session["llm_fields"] if base == "llm" else session["fields"]
        )
        base_natural_language = (
            session["llm_natural_language"]
            if base == "llm"
            else session["natural_language"]
        )
        user_payload = {
            "model_family": resolved_family,
            "world_context": session["world_context"],
            "feedback": feedback,
            "current_fields": base_fields,
            "current_natural_language": base_natural_language,
            "locks": session["locks"],
            "read_only_settings": session["settings"],
            "conversation_scope": (
                "accepted user checkpoint plus the active LLM branch"
                if base == "llm"
                else "accepted user checkpoint only; superseded unaccepted branches are excluded"
            ),
            "branch_id": branch_id,
            "user_chat_checkpoint_id": session.get("user_chat_checkpoint_id", ""),
            "recent_conversation": history,
            "image_legend": self._image_legend(manifest),
            "image_manifest": manifest,
        }
        user_instruction = (
            "Update the character description in MOSS natural-language mode and return JSON only.\n"
            if resolved_family == "moss"
            else "캐릭터 상태를 수정하고 JSON만 반환하세요.\n"
        )
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user_instruction
                + json.dumps(user_payload, ensure_ascii=False, indent=2),
            },
        ]

    async def _call_structured_llm(
        self,
        task_key: str,
        messages: list[dict[str, str]],
        images: list[tuple[str, ...]] | None,
        *,
        parser: Callable[[str], tuple[dict[str, Any] | None, str]],
        call_label: str = "캐릭터 메이커",
    ) -> dict[str, Any]:
        from modes.lighbd_service import (
            call_logged_llm_task,
            call_logged_llm_vision_task,
        )

        def validator(raw: str) -> tuple[bool, str]:
            parsed_val, reason_val = parser(raw)
            if parsed_val is None:
                return False, reason_val or "LLM 응답 형식이 올바르지 않습니다."
            return True, ""

        metadata: dict[str, Any] = {"prompt_id": task_key}
        if images:
            metadata["vision_image_count"] = len(images)
        context = llm_service.create_llm_execution_context(
            task_key,
            call_name=call_label,
            json_mode=True,
            metadata=metadata,
        )
        if images:
            execution_result = await call_logged_llm_vision_task(
                context,
                messages,
                images=images,
                result_validator=validator,
            )
        else:
            execution_result = await call_logged_llm_task(
                context,
                messages,
                result_validator=validator,
                retry_messages=messages,
            )
        raw = execution_result.to_legacy()
        if not execution_result.accepted or (
            isinstance(raw, str) and raw.strip().startswith("[LLM 실패]")
        ):
            print(
                f"[CHARACTER_MAKER] LLM 호출 최종 실패: task={task_key}, "
                f"slot={execution_result.final_slot}, result={raw}"
            )
            raise CharacterMakerError(raw)
        parsed, reason = parser(raw)
        if parsed is None:
            print(
                f"[CHARACTER_MAKER] LLM JSON 검증 실패: task={task_key}, "
                f"reason={reason}, raw={str(raw)[:1000]!r}"
            )
            raise CharacterMakerError(reason)
        return parsed

    async def _call_revision_llm(
        self,
        task_key: str,
        messages: list[dict[str, str]],
        images: list[tuple[str, ...]] | None,
        *,
        require_queries: bool,
        call_label: str = "캐릭터 메이커",
    ) -> dict[str, Any]:
        return await self._call_structured_llm(
            task_key,
            messages,
            images,
            parser=lambda raw: _parse_llm_payload(raw, require_queries=require_queries),
            call_label=call_label,
        )

    def _reference_bbox_messages(
        self,
        feedback: str,
        ref_manifest: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Lightweight vision prompt asking AI to select tight detail-crop bboxes."""
        system = (
            "You select tight close-up crop bounding boxes for a character design reference image. "
            "Inspect the attached USER_UPLOADED_* reference image(s) (portrait, landscape, turnaround "
            "sheet, seated pose, bust shot, or card with text/margins). "
            "Return 2 to 4 tight, non-redundant bounding boxes covering the most visually intricate "
            "character zones relevant to user_feedback (for full character recreation, typically: "
            "1) head/bangs/headpiece/ears, 2) shoulders/collar/upper torso/sleeves/held props, "
            "3) waist/hips/lower garment/skirt/emblems, 4) legs/legwear/footwear; for partial "
            "requests like outfit-only or face-only, focus boxes tightly on the requested zones). "
            "Ignore decorative card frames, text banners, speech bubbles, and empty background margins "
            "— wrap ONLY the character regions tightly. "
            "Coordinate system: normalized integers 0..1000 where x1 is left edge, y1 is top edge, "
            "x2 is right edge, y2 is bottom edge (0 <= x1 < x2 <= 1000, 0 <= y1 < y2 <= 1000). "
            "Return ONE JSON object only with key 'regions': an array of up to 4 objects, each with "
            "keys image_index (1-based integer), label (short English label, 2-6 words), x1, y1, x2, y2."
        )
        payload = {
            "user_feedback": feedback,
            "reference_images": ref_manifest,
        }
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Identify 2 to 4 tight detail-crop bounding boxes on the reference image(s). "
                    "Return JSON only.\n"
                    + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                ),
            },
        ]

    async def _detect_reference_bboxes(
        self,
        feedback: str,
        ref_vision_images: list[tuple[str, ...]],
        ref_manifest: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run AI bbox localization on reference image(s); never fails the main request."""
        if not ref_vision_images:
            return []
        try:
            messages = self._reference_bbox_messages(feedback, ref_manifest)
            parsed = await self._call_structured_llm(
                "character_maker_feedback",
                messages,
                ref_vision_images,
                parser=_parse_reference_bboxes,
                call_label="에셋 캐릭터 메이커 · 깊게 영역 탐지(BBox)",
            )
            regions = list(parsed.get("regions") or [])
            print(
                f"[CHARACTER_MAKER] 깊게 BBox 영역 탐지 완료: count={len(regions)}, "
                f"regions={regions}"
            )
            return regions
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 깊게 BBox 영역 탐지 실패, 원본 전체 이미지로 계속 진행: "
                f"error={type(exc).__name__}: {exc}"
            )
            return []

    def _moss_forensic_audit_messages(
        self,
        session: dict[str, Any],
        feedback: str,
        *,
        image_manifest: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Stage-1 vision forensic inspector for MOSS reference recreation & visual diff."""
        has_current = any(item.get("role") == "CURRENT" for item in image_manifest)
        has_refs = any(item.get("role") == "REF" for item in image_manifest)
        has_crops = any(item.get("role") == "REF_DETAIL_ZOOM" for item in image_manifest)
        crop_phrase = (
            " and its AI-cropped close-up images (USER_UPLOADED_*_CROP_*)"
            if has_crops
            else ""
        )
        if has_current and has_refs:
            diff_rule = (
                "GENERATED_CURRENT (blue banner) IS attached alongside USER_UPLOADED_* (red banner)"
                + crop_phrase
                + ". Compare GENERATED_CURRENT against USER_UPLOADED_* region by region and list "
                "EVERY concrete discrepancy in visual_diff_fixes (wrong hairstyle count, swapped "
                "left/right shoulder/arm/hip/hand, wrong bodice color, wrong mantle length or fur "
                "edge, wrong overskirt length, missing tights/pantyhose, wrong boot height, wrong "
                "prop length/color/hand/finial, leaked background, etc.). "
            )
        elif has_current:
            diff_rule = (
                "GENERATED_CURRENT is attached with no separate reference image. Compare "
                "GENERATED_CURRENT against user_feedback and current_fields, and list EVERY "
                "missing, drifted, or contradictory visual detail in visual_diff_fixes. "
            )
        else:
            diff_rule = (
                "No GENERATED_CURRENT image is attached; inspect USER_UPLOADED_*"
                + crop_phrase
                + " thoroughly and set visual_diff_fixes to []. "
            )
        system = (
            "You are Stage 1 (Forensic Visual Inspector) of a deep character-design pipeline "
            "for the MOSS image model. Your ONLY job is exhaustive, pixel-accurate visual "
            "observation of the attached reference character image(s)"
            + crop_phrase
            + ". Write EVERY JSON string value in concise, concrete English (1-2 dense sentences "
            "per slot; NEVER write Korean). Use exact English garment and visual terminology "
            "directly. Do NOT guess generic tropes and do NOT omit small ornaments, colors, "
            "garment lengths, or asymmetries. "
            + diff_rule
            + "CRITICAL VERIFICATION CHECKLIST (inspect matching close-up crops when present): "
            "1. Hair count & bangs: verify exact hairstyle count — single high/side ponytail vs "
            "twin tails vs half-up vs loose hair (never mistake hair trailing behind one shoulder "
            "for a second ponytail). State which side any ponytail sweeps toward (viewer's left vs "
            "viewer's right). Verify whether blunt bangs completely cover the forehead down to the "
            "eyebrows or leave forehead skin parted/exposed. Note cheek sidelocks and spiral curls. "
            "2. Headpiece & hair accessories: exact vertical silhouette and height (tall narrow "
            "pointed spire/diadem vs low flat circlet vs hat/hairpin), metal color (silver vs "
            "gold), forehead arch/wire and hanging gem resting over bangs, AND distinguish fabric "
            "cloth ribbon bows tied behind/around the headpiece (exact colors & tails) from actual "
            "feathered/wing ornaments (never mistake cloth bow loops for angel wings). "
            "3. Eyes, ears, wings/nonhuman: exact eye color/shape, ear shape and tilt, body build, "
            "and EXACT count & geometry of any wings/horns/tail (e.g., exact wing count such as "
            "4 = 2 upper + 2 lower, angular geometric stained-glass crystal vs rounded butterfly/"
            "feathered, frame color, translucent membrane tint colors — or 'none'). "
            "4. Neck, shoulders & outer mantle/cape length: "
            "- Distinguish a low fitted choker (leaving upper neck and collarbones bare) vs a high "
            "standing collar / turtleneck / scarf, noting exact choker colors and center gem. "
            "- CRITICAL ANATOMICAL RIGHT vs LEFT (when facing the viewer, character's RIGHT side is "
            "on the VIEWER'S LEFT, and character's LEFT side is on the VIEWER'S RIGHT): explicitly "
            "separate [character's RIGHT shoulder (viewer's left)] vs [character's LEFT shoulder "
            "(viewer's right)] (bare skin vs covered by an asymmetrical shoulder mantle/cape). "
            "- MANDATORY VERTICAL ENDING LENGTH: for ANY cape, mantle, cloak, shawl, or coat, "
            "state explicitly WHERE its bottom edge stops vertically on the body (upper-arm / "
            "elbow-length short mantle, waist-length, hip-length, mid-thigh, knee-length, or "
            "calf/ankle/floor-length). "
            "- If fur/lace trim is on a shoulder garment, verify WHICH exact edge carries it (top "
            "shoulder ridge & inner front edge ONLY vs bottom hem vs all edges), plus studs, "
            "embroidery motifs, brooches, and hanging pom-poms. "
            "5. Upper garment / bodice color zones: specify neckline cut (strapless/sweetheart/"
            "collared), center chest bow & brooch exact color/shape, CENTER front panel exact color "
            "and horizontal bars/chevrons/fasteners/buttons, and SIDE panels exact color, trim "
            "borders, and clasp tabs. "
            "6. Arms, sleeves & gloves — SEPARATE RIGHT vs LEFT: explicitly record BOTH "
            "[character's RIGHT arm (viewer's left)] (upper-arm sleeve color/cut/off-shoulder fit, "
            "forearm/cuff shape & color, glove length/color) AND [character's LEFT arm (viewer's "
            "right)] (upper-arm sleeve/inner flared sleeve color, forearm, glove length/color). "
            "WRIST-vs-HIP CHECK: if a hand rests on a hip, verify whether a nearby ribbon bow or "
            "ornament belongs to the WAIST/HIP BELT (record under waist_and_hips) vs the wrist/arm. "
            "7. Waist, hips & lower garment: verify whether the waist belt/sash is solid or striped "
            "(exact stripe colors), explicitly separate [character's RIGHT hip (viewer's left)] vs "
            "[character's LEFT hip (viewer's right)] ornaments (hip bow with buckle & tails vs hip "
            "crest/armor/ruffle), inner skirt or trouser cut/texture (smooth gently scalloped "
            "flared mini-skirt vs knife-pleated skirt vs shorts/pants) and exact front hem emblem, "
            "AND any outer overskirt/peplum flaps (exact colors, surface grid/pattern, border color "
            "& thickness, tip cutout inlays, ruffled underlayer if visible, and EXACT ending height "
            "relative to the legs: upper-thigh, mid-thigh well above knees, knee-length, or calf). "
            "8. Legwear & footwear: distinguish seamless full-length pantyhose/tights (covering "
            "entire thighs with zero bare skin above stockings) vs thigh-high stockings vs socks vs "
            "bare legs, plus opacity and faint lace motifs; record EXACT footwear shaft height "
            "(shoes, ankle boots, mid-calf boots below the knees, knee-high boots, or thigh-high "
            "boots), main color, vertical stripe/panel color, strap clasps, pointed/winged cuff "
            "trim, exact gem position (top front of cuff vs ankle), and heel color. "
            "9. Held props — ALL 5 ATTRIBUTES: inspect pixels directly and record (a) exact hand "
            "holding it (character's RIGHT [viewer's left] vs LEFT [viewer's right]), (b) EXACT "
            "length relative to the body (palm-sized, short forearm-length ~35-45cm baton/wand, "
            "waist-length rod, or tall full-height floor-length staff), (c) EXACT shaft/body color "
            "& thickness (verify from pixels — e.g. dark midnight-blue vs black vs silver vs gold), "
            "(d) EXACT grip position along the shaft (e.g. gripped right at the top collar directly "
            "beneath the finial vs mid-shaft vs bottom handle) and diagonal angle, and (e) exact "
            "finial motif/colors/size and opposite ferrule tip color (or 'none'). "
            "10. Ignore reference UI frames, speech bubbles, text banners, background wallpaper, "
            "and added red/blue inspection headers. "
            "Return ONE JSON object with these exact keys: "
            "character_overview, hair_and_bangs, headpiece_and_ribbons, eyes_ears_wings_nonhuman, "
            "neck_and_shoulders, bodice_and_chest, arms_sleeves_gloves, waist_and_hips, "
            "skirt_and_overskirt, legwear, footwear, held_props, expression_and_pose, "
            "user_scope_note, visual_diff_fixes."
        )
        payload = {
            "world_context": session.get("world_context", ""),
            "user_feedback": feedback,
            "current_fields": session.get("fields", {}),
            "image_legend": self._image_legend(image_manifest),
            "image_manifest": image_manifest,
        }
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Inspect all attached images and close-up crops and return the complete "
                    "14-zone English forensic audit JSON only.\n"
                    + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                ),
            },
        ]

    def _moss_prompt_synthesis_messages(
        self,
        session: dict[str, Any],
        feedback: str,
        forensic_audit: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Stage-2 structured MOSS natural-language prompt synthesizer."""
        system = (
            "Stage 2 (MOSS Prompt Synthesizer for Qwen3.5 text encoder). Cross-check forensic_audit "
            "against the attached reference image(s) and faithfully translate EVERY detail into "
            "concrete, literal English MOSS prose split across the JSON sections below. NEVER "
            "invent, alter, drop, or swap any color, metal, motif, material, left/right side, "
            "garment length, boot height, or prop shape!\n"
            "RULES:\n"
            "1. Scope & Diff: For full recreation or reference-fix requests, rewrite all applicable "
            "sections completely from forensic_audit. For partial-attribute requests (e.g. hair "
            "only), keep unrelated current_fields and rewrite only requested sections. If "
            "forensic_audit.visual_diff_fixes is non-empty, fix EVERY listed discrepancy across "
            "all fields.\n"
            "2. Literal Fidelity (1-3 dense English sentences per non-empty section; '' if absent): "
            "Preserve exact metal/colors (silver vs gold, dark navy vs teal vs white), exact hair "
            "count & bangs, exact wing count & stained-glass/crystal geometry (never add feathers "
            "if crystal), exact character RIGHT vs LEFT shoulders/arms/hips/hands (for ANY "
            "asymmetrical garment, side ponytail, hip bow, or held prop, ALWAYS pair anatomical "
            "side with viewer side in parentheses — e.g. 'her left shoulder (viewer's right)', "
            "'her right hip (viewer's left)', 'her left hand (viewer's right)' — and keep every "
            "section 100% consistent on which side wears which garment!), exact fur edge, exact "
            "bodice color zones, exact skirt pattern, exact legwear (full seamless tights vs "
            "stockings), exact footwear height (preserve mid-calf below the knees literally — NEVER "
            "turn mid-calf boots into 'knee-high'), and exact prop hand, shaft color, finial motif "
            "butterfly crystal finial), grip point, and tip color.\n"
            "3. Positive-Only Phrasing: NEVER use negative words ('no X', 'without X', 'not "
            "pleated', 'no bare thighs') in English descriptions; describe only visible form.\n"
            "4. Anti-Drift Length & Noun Locks:\n"
            "- Never write bare 'cape', 'half-cape', or 'mantle' without vertical ending length "
            "(e.g. 'short elbow-length asymmetrical left-shoulder mantle ending at the elbow well "
            "above her waist').\n"
            "- Never write 'staff', 'baton-staff', 'magic staff', or 'scepter' in ANY field "
            "(held_props, composition, or natural_language) for a short forearm-length prop; use "
            "'short, slender forearm-length baton/wand' with its exact shaft color and grip right "
            "below the finial.\n"
            "- Never call short mid-thigh overskirt flaps 'coat-tails'; write 'short, stiff, "
            "pointed mid-thigh-length peplum overskirt panels ending at mid-thigh well above her "
            "knees'.\n"
            "- Never invent 'high standing collar' for a low fitted choker, never invent 'wing "
            "motifs' for a fabric ribbon bow, and place forehead ornaments 'resting across the top "
            "of her bangs' (never 'onto her forehead').\n"
            "5. Strict Separation:\n"
            "- official_character_tag: confirmed official Danbooru tag only, else ''.\n"
            "- appearance_sections: body_eyes_ears, hair_bangs_curls, headpiece_ribbons, "
            "wings_nonhuman.\n"
            "- outfit_sections: neck_shoulders_mantle, bodice_chest, sleeves_gloves, waist_hips, "
            "skirt_overskirt, legwear, footwear, held_props.\n"
            "- expression: ONLY face, brows, eyes, mouth, blush, gaze.\n"
            "- composition: ONLY framing (full-body head-to-footwear standing portrait when lower "
            "outfit/footwear exists), angle, posture, and exact right/left hand placement (matching "
            "held_props noun). ZERO background/UI/text/frames.\n"
            "- natural_language: ONE concise expression-free structural anchor sentence reinforcing "
            "the 5-6 highest-drift constraints (bangs, hair count, wing count, short mantle length "
            "& bare shoulder, mid-thigh overskirt, boot height, short baton hand & shaft color). "
            "ZERO emotion/pose/smile.\n"
            "6. English-only in all prompt fields (zero Korean outside assistant_message). No "
            "Danbooru tags (1girl, solo), underscores, weights, or quality/artist triggers.\n"
            "Return ONE JSON object with exact keys: assistant_message (Korean if user_feedback is "
            "Korean), official_character_tag, appearance_sections, outfit_sections, expression, "
            "composition, natural_language, rag_queries."
        )
        payload: dict[str, Any] = {
            "user_feedback": feedback,
            "forensic_audit": forensic_audit,
        }
        world_ctx = str(session.get("world_context") or "").strip()
        if world_ctx:
            payload["world_context"] = world_ctx
        current_fields = session.get("fields") or {}
        if any(current_fields.get(k) for k in EDITABLE_FIELDS):
            payload["current_fields"] = current_fields
        current_nl = str(session.get("natural_language") or "").strip()
        if current_nl:
            payload["current_natural_language"] = current_nl
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Synthesize MOSS English prompt sections faithfully from forensic_audit. "
                    "Return JSON only.\n"
                    + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                ),
            },
        ]

    def _log_cm_history(
        self,
        task_key: str,
        messages: list[dict[str, str]],
        output: Any,
        t0: float,
        *,
        status: str,
        error: str = "",
        sink: dict | None = None,
        call_label: str = "캐릭터 메이커",
        llm_slot: str = "",
    ) -> None:
        """캐릭터 메이커 LLM 호출을 LIGHBD 자세히(lighbd_history.jsonl)에 기록."""
        try:
            from modes.lighbd_service import _log_lighbd_history

            elapsed = round(time.perf_counter() - t0, 3)
            output_text = output if isinstance(output, str) else str(output)
            sk = sink or {}
            resolved_slot = str(llm_slot or sk.get("llm_slot") or "").strip().lower()
            if not resolved_slot and task_key:
                try:
                    resolved_slot, _ = llm_service._routing_for(task_key)
                except Exception:
                    resolved_slot = ""
            service_name = ""
            model_name = ""
            if resolved_slot:
                try:
                    suffix = llm_service._slot_suffix(resolved_slot)
                    service_name = str(
                        llm_service._base_config_get(f"llm_service{suffix}", "openai") or ""
                    )
                    model_name = str(
                        llm_service._base_config_get(f"llm_model{suffix}", "") or ""
                    )
                except Exception:
                    pass
            record = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "prompt_id": task_key,
                "task_key": task_key,
                "call_name": call_label,
                "llm_slot": resolved_slot,
                "service": service_name,
                "model": model_name,
                "input": messages,
                "output": output_text,
                "completion_tokens": int(sk.get("completion_tokens") or 0),
                "prompt_tokens": int(sk.get("prompt_tokens") or 0),
                "elapsed": elapsed,
                "tps": float(sk.get("tps") or 0.0),
                "status": status,
            }
            if error:
                record["error"] = error
            _log_lighbd_history(record)
        except Exception as exc:
            print(f"[CHARACTER_MAKER] LIGHBD 히스토리 기록 실패: {exc}")
            traceback.print_exc()

    async def _ensure_rag_ready(self, service: Any) -> dict[str, Any]:
        """Allow model/index cold start without weakening search timeouts."""
        status = await asyncio.to_thread(service.status)
        if status.get("loaded"):
            return {
                "success": True,
                "loaded": True,
                "row_count": int(status.get("row_count") or 0),
                "variant": str(status.get("variant") or "b"),
                "mode": "embedded",
            }
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(service.warmup),
                timeout=RAG_COLD_START_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 서비스 최초 준비 시간 초과: "
                f"timeout={RAG_COLD_START_TIMEOUT_SECONDS}"
            )
            traceback.print_exc()
            raise CharacterMakerError(
                "내장 RAG 최초 준비가 "
                f"{RAG_COLD_START_TIMEOUT_SECONDS:g}초 안에 끝나지 않았습니다."
            ) from exc

    async def _rag_search(
        self, query: str, *, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        top_k = max(1, min(20, int(config.get("character_maker_rag_top_k", 5))))
        threshold = float(config.get("character_maker_rag_threshold", 0.0))
        timeout_seconds = max(
            2.0, min(120.0, float(config.get("character_maker_rag_timeout_sec", 20.0)))
        )
        service = get_danbooru_rag_service()
        await self._ensure_rag_ready(service)
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    service.search,
                    query,
                    top_k=top_k,
                    threshold=threshold,
                    categories={0},
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 검색 시간 초과: "
                f"query={query!r}, timeout={timeout_seconds}"
            )
            traceback.print_exc()
            raise CharacterMakerError(
                f"내장 RAG 검색이 {timeout_seconds:g}초 안에 끝나지 않았습니다."
            ) from exc
        except DanbooruRagError as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 검색 실패: "
                f"query={query!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(str(exc)) from exc
        except Exception as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 검색 예외: "
                f"query={query!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(f"내장 RAG 검색 실패: {exc}") from exc
        if not isinstance(results, list):
            print(
                f"[CHARACTER_MAKER_RAG] 검색 결과 누락: query={query!r}, "
                f"result_type={type(results).__name__}"
            )
            raise CharacterMakerError("RAG 검색 결과 배열이 없습니다.")
        return [item for item in results if isinstance(item, dict) and item.get("tag")]

    async def _rag_refine(
        self,
        *,
        task_key: str,
        session: dict[str, Any],
        draft: dict[str, Any],
        config: dict[str, Any],
        base: str = "user",
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        base_fields = (
            session["llm_fields"] if base == "llm" else session["fields"]
        )
        jobs: list[tuple[str, str]] = []
        for field in EDITABLE_FIELDS:
            if session["locks"][field]:
                continue
            queries = list(draft["rag_queries"].get(field, []))
            if not queries:
                queries = list(draft["fields"].get(field, []))
            for query in queries[:4]:
                jobs.append((field, query))
        if not jobs:
            print(
                f"[CHARACTER_MAKER_RAG] 검색 단위 없음: session={session['id']}, "
                "현재 필드를 유지합니다."
            )
            return copy.deepcopy(base_fields), {"queries": [], "candidates": {}}

        unique_queries = list(dict.fromkeys(query for _, query in jobs))
        semaphore = asyncio.Semaphore(4)

        async def _bounded_search(query: str) -> tuple[str, list[dict[str, Any]]]:
            async with semaphore:
                return query, await self._rag_search(query, config=config)

        unique_results = await asyncio.gather(
            *(_bounded_search(query) for query in unique_queries)
        )
        hits_by_query = dict(unique_results)
        candidates: dict[str, list[dict[str, Any]]] = {
            field: [] for field in EDITABLE_FIELDS
        }
        for field in EDITABLE_FIELDS:
            for tag in base_fields[field]:
                candidates[field].append(
                    {
                        "query": "(current base value)",
                        "tag": tag,
                        "score": None,
                        "definition": "Base tag currently in context; trusted for preservation.",
                        "aliases": [],
                    }
                )
        for field, query in jobs:
            hits = hits_by_query.get(query, [])
            for hit in hits:
                candidate = {
                    "query": query,
                    "tag": str(hit.get("tag") or ""),
                    "score": hit.get("score"),
                    "definition": str(hit.get("definition") or "")[:500],
                    "aliases": hit.get("aliases"),
                }
                if candidate["tag"] and not any(
                    item["tag"].casefold() == candidate["tag"].casefold()
                    for item in candidates[field]
                ):
                    candidates[field].append(candidate)

        selection_messages = [
            {
                "role": "system",
                "content": (
                    "Select Danbooru tags for the character edit. Do not use hard-coded keyword "
                    "matching. For every unlocked field, fields may contain only exact tag strings "
                    "present in that field's candidate_pool. Preserve locked fields exactly. "
                    "Return JSON with assistant_message, fields, and rag_queries (arrays only, empty arrays allowed; never a bare string)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "world_context": session["world_context"],
                        "current_fields": base_fields,
                        "locks": session["locks"],
                        "draft_intent": draft,
                        "candidate_pool": candidates,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        selected = await self._call_revision_llm(
            task_key, selection_messages, None, require_queries=False,
            call_label="캐릭터 메이커 · 2단계(RAG 태그선택)",
        )

        final_fields: dict[str, list[str]] = {}
        dropped: dict[str, list[str]] = {}
        for field in EDITABLE_FIELDS:
            if session["locks"][field]:
                final_fields[field] = list(base_fields[field])
                dropped[field] = []
                continue
            allowed = {
                str(item.get("tag") or "").casefold(): str(item.get("tag") or "")
                for item in candidates[field]
                if str(item.get("tag") or "").strip()
            }
            if not allowed:
                final_fields[field] = list(base_fields[field])
                dropped[field] = list(selected["fields"][field])
                print(
                    f"[CHARACTER_MAKER_RAG] 후보 없음으로 현재값 유지: "
                    f"session={session['id']}, field={field}"
                )
                continue
            accepted: list[str] = []
            rejected: list[str] = []
            for tag in selected["fields"][field]:
                canonical = allowed.get(tag.casefold())
                if canonical and canonical not in accepted:
                    accepted.append(canonical)
                else:
                    rejected.append(tag)
            final_fields[field] = accepted
            dropped[field] = rejected
            if rejected:
                print(
                    f"[CHARACTER_MAKER_RAG] 후보 외 태그 제거: "
                    f"session={session['id']}, field={field}, tags={rejected}"
                )
        return final_fields, {
            "queries": [{"field": field, "query": query} for field, query in jobs],
            "candidates": candidates,
            "dropped": dropped,
        }

    async def suggest_asset_values(
        self,
        character: str,
        values: dict,
        message: str,
        history: list,
        *,
        image_path: str | None = None,
        reference_images: list[dict[str, Any]] | None = None,
        model_family: str = "anima",
        reasoning_depth: str = "fast",
        deep_pass: int = 1,
        precomputed_bbox_regions: list[dict[str, Any]] | None = None,
        bbox_regions_out: list[dict[str, Any]] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """Use the configured editor and logs without mutating the legacy singleton session."""
        def _notify_progress(label: str) -> None:
            if callable(progress_callback) and label:
                try:
                    progress_callback(label)
                except Exception:
                    pass

        fields = _normalize_fields({field: values.get(f"raw_{field}_tags", []) for field in EDITABLE_FIELDS})
        session = {
            "fields": fields, "natural_language": values.get("natural_language", ""),
            "world_context": f"Character: {character}", "settings": {},
            "locks": dict.fromkeys(LOCKABLE_FIELDS, False), "chat": history,
        }
        resolved_family = normalize_maker_model_family(model_family, strict=True)
        resolved_depth = normalize_maker_reasoning_depth(reasoning_depth, strict=True)
        vision_images: list[tuple[str, ...]] = []
        manifest: list[dict[str, Any]] = []
        has_references = bool(reference_images)
        if image_path:
            if has_references or resolved_depth == "deep":
                encoded = self._encode_vision_image_with_banner(
                    image_path,
                    banner_text="GENERATED_CURRENT | Current generated image (edit target)",
                    short_banner="GENERATED_CURRENT",
                    banner_color="#1d4ed8",
                )
            else:
                encoded = self._encode_vision_image(image_path)
            if encoded is None:
                raise CharacterMakerError("피드백할 이미지를 읽지 못했습니다. 다시 생성하세요.")
            if has_references or resolved_depth == "deep":
                vision_images.append((
                    encoded[0],
                    encoded[1],
                    "Image 1 [GENERATED_CURRENT] (blue top banner): Existing generated character "
                    "image to inspect and modify.",
                ))
                manifest.append({
                    "position": 1,
                    "role": "CURRENT",
                    "image_id": "GENERATED_CURRENT",
                    "banner_color": "blue",
                    "description": (
                        "Existing generated image of the target character to inspect and modify "
                        "(marked with a blue GENERATED_CURRENT banner above the top edge)."
                    ),
                })
            else:
                vision_images.append(encoded)
                manifest.append({
                    "position": 1,
                    "role": "CURRENT",
                    "description": "Latest selected generated image of the target character to modify.",
                })
        ref_entries: list[dict[str, Any]] = []
        ref_only_vision: list[tuple[str, ...]] = []
        ref_only_manifest: list[dict[str, Any]] = []
        for ref_index, ref in enumerate(reference_images or [], start=1):
            ref_path = ref.get("path", "") if isinstance(ref, dict) else str(ref or "")
            ref_name = str(ref.get("name") or "").strip() if isinstance(ref, dict) else os.path.basename(ref_path)
            image_id = f"USER_UPLOADED_{ref_index}"
            encoded = self._encode_vision_image_with_banner(
                ref_path,
                banner_text=f"{image_id} | User-uploaded reference image",
                short_banner=image_id,
                banner_color="#be123c",
            )
            if encoded is None:
                raise CharacterMakerError("첨부한 참고 이미지를 읽지 못했습니다. 다시 업로드해 주세요.")
            position = len(vision_images) + 1
            label_name = f" ({ref_name})" if ref_name else ""
            vision_entry = (
                encoded[0],
                encoded[1],
                f"Image {position} [{image_id}] (red top banner): Reference image #{ref_index} "
                f"uploaded by the user{label_name}. Not the generated output.",
            )
            manifest_entry = {
                "position": position,
                "role": "REF",
                "image_id": image_id,
                "reference_index": ref_index,
                "banner_color": "red",
                "name": ref_name,
                "description": (
                    f"User-uploaded reference image #{ref_index} attached to the current message "
                    f"(marked with a red {image_id} banner above the top edge). "
                    "Not the generated output; use it as the visual reference requested by the user."
                ),
            }
            vision_images.append(vision_entry)
            manifest.append(manifest_entry)
            ref_entries.append({
                "path": ref_path,
                "name": ref_name,
                "image_id": image_id,
                "reference_index": ref_index,
            })
            ref_only_vision.append((
                encoded[0],
                encoded[1],
                f"Image {ref_index} [{image_id}] (red top banner): Reference image #{ref_index}{label_name}.",
            ))
            ref_only_manifest.append({
                "image_index": ref_index,
                "image_id": image_id,
                "name": ref_name,
            })

        if resolved_depth == "deep" and ref_entries:
            if precomputed_bbox_regions is not None:
                bbox_regions = list(precomputed_bbox_regions)
            else:
                _notify_progress("영역 탐지 중 (AI BBox)…")
                bbox_regions = await self._detect_reference_bboxes(
                    message, ref_only_vision, ref_only_manifest,
                )
            if bbox_regions_out is not None:
                bbox_regions_out.clear()
                bbox_regions_out.extend(bbox_regions)
            if bbox_regions:
                bbox_crops = self._encode_reference_bbox_crops(
                    ref_entries,
                    bbox_regions,
                    banner_color="#be123c",
                )
                for crop_item in bbox_crops:
                    crop_pos = len(vision_images) + 1
                    vision_images.append((
                        crop_item["b64"],
                        crop_item["mime"],
                        f"Image {crop_pos} [{crop_item['image_id']}] (red top banner): "
                        f"{crop_item['description']}",
                    ))
                    manifest.append({
                        "position": crop_pos,
                        "role": "REF_DETAIL_ZOOM",
                        "image_id": crop_item["image_id"],
                        "parent_image_id": crop_item["parent_image_id"],
                        "reference_index": crop_item["reference_index"],
                        "banner_color": "red",
                        "name": crop_item["name"],
                        "label": crop_item["label"],
                        "bbox": crop_item["bbox"],
                        "description": crop_item["description"],
                    })

        images = vision_images or None
        pass_tag = f"{deep_pass}차" if resolved_depth == "deep" else ""
        if resolved_depth == "deep" and resolved_family == "moss" and images:
            _notify_progress(
                "1차 정밀 시각 분석 중…" if deep_pass == 1 else "원본 대조·2차 디테일 분석 중…"
            )
            audit_messages = self._moss_forensic_audit_messages(
                session, message, image_manifest=manifest,
            )
            forensic_audit = await self._call_structured_llm(
                "character_maker_feedback",
                audit_messages,
                images,
                parser=_parse_moss_forensic_audit,
                call_label=f"에셋 캐릭터 메이커 · MOSS 깊게 {pass_tag}(영문 시각 분석)",
            )
            _notify_progress(
                "1차 프롬프트 합성 중…" if deep_pass == 1 else "2차 보정 프롬프트 합성 중…"
            )
            synth_images = [
                img
                for img, item in zip(vision_images, manifest)
                if item.get("role") != "REF_DETAIL_ZOOM"
            ] or None
            synth_messages = self._moss_prompt_synthesis_messages(
                session, message, forensic_audit,
            )
            return await self._call_structured_llm(
                "character_maker_feedback" if synth_images else "character_maker_draft",
                synth_messages,
                synth_images,
                parser=_parse_moss_sectioned_synthesis,
                call_label=f"에셋 캐릭터 메이커 · MOSS 깊게 {pass_tag}(프롬프트 합성)",
            )

        if resolved_depth == "deep":
            _notify_progress(
                "1차 상세 프롬프트 작성 중…" if deep_pass == 1 else "원본 대조·2차 디테일 보정 중…"
            )
        messages = self._revision_messages(
            session, message, rag_enabled=False, image_manifest=manifest,
            model_family=resolved_family,
        )
        mode_tag = "MOSS" if resolved_family == "moss" else "Anima"
        depth_tag = f" 깊게 {pass_tag}" if resolved_depth == "deep" else ""
        return await self._call_revision_llm(
            "character_maker_feedback" if images else "character_maker_draft",
            messages, images, require_queries=False,
            call_label=f"에셋 캐릭터 메이커 · {mode_tag}{depth_tag}",
        )

    async def revise(self, session_id: str, payload: Any) -> dict[str, Any]:
        async with self.operation_lock(session_id):
            return await self._revise_locked(session_id, payload)

    async def _revise_locked(self, session_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CharacterMakerError("수정 요청은 객체여야 합니다.")
        session = self._session(session_id)
        base = str(payload.get("base") or "user")
        if base not in ("user", "llm"):
            raise CharacterMakerError("base 는 user 또는 llm 이어야 합니다.")
        # base=="user" 일 때만 사용자 fields 를 동기화한다.
        # base=="llm" 은 사용자 fields 를 건드리지 않고 LLM 작업 영역(llm_fields)에서 출발한다.
        sync_keys = (
            "world_context",
            "locks",
            "settings",
            "editable_preset_tags",
            "editable_preset_enabled",
        )
        if base == "user":
            sync_keys = sync_keys + ("fields", "natural_language")
        self.update_session(
            session_id,
            {key: payload[key] for key in sync_keys if key in payload},
        )
        feedback = payload.get("message")
        if not isinstance(feedback, str) or not feedback.strip():
            raise CharacterMakerError("LLM에게 전달할 요청이나 피드백을 입력하세요.")
        feedback = feedback.strip()[:8000]
        base_fields = (
            session["llm_fields"] if base == "llm" else session["fields"]
        )
        before = copy.deepcopy(base_fields)
        before_nl = (
            session["llm_natural_language"]
            if base == "llm"
            else session["natural_language"]
        )
        draft_nl = before_nl
        previous_branch_id = str(session.get("active_chat_branch_id") or "")
        if base == "user" or not previous_branch_id:
            branch_id = uuid.uuid4().hex
            session["active_chat_branch_id"] = branch_id
        else:
            branch_id = previous_branch_id
        user_message_id = uuid.uuid4().hex
        session["chat"].append(
            {
                "id": user_message_id,
                "role": "user",
                "content": feedback,
                "at": _now_iso(),
                "base": base,
                "branch_id": branch_id,
                "accepted": False,
                "checkpoint_id": "",
            }
        )
        session["chat"] = session["chat"][-MAX_CHAT_ITEMS:]

        config = self.config_getter() or {}
        rag_enabled = bool(session["settings"].get("rag_enabled", False))
        active = self._active_revision_for(session, base)
        task_key = "character_maker_feedback" if active else "character_maker_draft"
        try:
            vision_inputs = self._revision_vision_inputs(session, base=base)
            images = [
                (item["b64"], item["mime"]) for item in vision_inputs
            ]
            image_manifest = [
                copy.deepcopy(item["manifest"]) for item in vision_inputs
            ]
            model_family = normalize_maker_model_family(
                payload.get("model_family"), strict=True
            )
            messages = self._revision_messages(
                session,
                feedback,
                rag_enabled=rag_enabled,
                base=base,
                branch_id=branch_id,
                current_message_id=user_message_id,
                image_manifest=image_manifest,
                model_family=model_family,
            )
            draft = await self._call_revision_llm(
                task_key, messages, images or None, require_queries=True,
                call_label="캐릭터 메이커 · 1단계(수정)",
            )

            for field in EDITABLE_FIELDS:
                if session["locks"][field]:
                    draft["fields"][field] = list(before[field])
            # 자연어: 잠금 상태이거나 LLM이 반환하지 않았으면 이전값 유지.
            if (
                not session["locks"].get("natural_language")
                and draft.get("natural_language") is not None
            ):
                draft_nl = draft.get("natural_language") or ""

            rag_meta: dict[str, Any] = {"enabled": False}
            if rag_enabled:
                refined_fields, rag_details = await self._rag_refine(
                    task_key=task_key,
                    session=session,
                    draft=draft,
                    config=config,
                    base=base,
                )
                final_fields = refined_fields
                rag_meta = {"enabled": True, **rag_details}
            else:
                final_fields = draft["fields"]
        except Exception:
            session["chat"] = [
                item
                for item in session["chat"]
                if item.get("id") != user_message_id
            ]
            session["active_chat_branch_id"] = previous_branch_id
            print(
                f"[CHARACTER_MAKER] 실패한 수정 대화 롤백: "
                f"session={session_id}, base={base}, branch={branch_id}"
            )
            raise

        for field in EDITABLE_FIELDS:
            if session["locks"][field]:
                final_fields[field] = list(before[field])
        # LLM 수정 결과는 항상 LLM 작업 영역(llm_fields)에 기록한다.
        # 사용자 영역(fields)은 accept 로만 갱신된다.
        session["llm_fields"] = _normalize_fields(final_fields)
        session["llm_natural_language"] = (draft_nl or "")[:MAX_NATURAL_LANGUAGE_LENGTH]
        assistant_message = draft["assistant_message"]
        session["chat"].append(
            {
                "id": uuid.uuid4().hex,
                "role": "assistant",
                "content": assistant_message,
                "at": _now_iso(),
                "base": base,
                "branch_id": branch_id,
                "accepted": False,
                "checkpoint_id": "",
            }
        )
        session["chat"] = session["chat"][-MAX_CHAT_ITEMS:]
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] LLM 수정 완료: session={session_id}, "
            f"task={task_key}, base={base}, branch={branch_id}, "
            f"vision_images={len(images)}, rag={rag_enabled}"
        )
        self._persist_session(session)
        return {
            "success": True,
            "session": self.public_session(session_id),
            "diff": _tag_diff(before, session["llm_fields"]),
            "rag": rag_meta,
        }

    def save_generation_artifacts(
        self,
        session_id: str,
        *,
        image_bytes: bytes,
        positive: str,
        negative: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """생성 결과를 캐릭터 메이커 임시 루트에 안전하게 저장한다.

        삽화 워크플로우는 에셋 모드의 전용 저장기를 통과하지 않으므로 동일한
        리비전 계약(image_path + prompt_path)을 여기서 만든다. 파일명은 매번 새로
        생성하여 기존 데이터 파일을 덮어쓰지 않는다.
        """
        session = self._session(session_id)
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            print(
                f"[CHARACTER_MAKER] 생성 이미지 저장 거부: "
                f"session={session_id}, bytes={len(image_bytes) if image_bytes else 0}"
            )
            raise CharacterMakerError("생성된 이미지 데이터가 비어 있습니다.")
        if not isinstance(positive, str) or not positive.strip():
            print(
                f"[CHARACTER_MAKER] 생성 이미지 저장 거부: "
                f"session={session_id}, positive 비어 있음"
            )
            raise CharacterMakerError("생성 프롬프트가 비어 있습니다.")
        if not isinstance(negative, str):
            raise CharacterMakerError("부정 프롬프트는 문자열이어야 합니다.")

        image_dir = os.path.join(self._session_dir(), "images")
        _assert_within(self.temp_root, image_dir)
        os.makedirs(image_dir, exist_ok=True)
        stem = f"{int(time.time())}_{uuid.uuid4().hex[:10]}"
        image_path = os.path.join(image_dir, stem + ".webp")
        prompt_path = os.path.join(image_dir, stem + "_prompt.json")
        _assert_within(self.temp_root, image_path)
        _assert_within(self.temp_root, prompt_path)

        try:
            with Image.open(io.BytesIO(bytes(image_bytes))) as image:
                image.load()
                save_image = image if image.mode == "RGBA" else image.convert("RGB")
                save_image.save(image_path, format="WEBP", quality=90, method=4)
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 생성 이미지 WEBP 저장 실패: "
                f"session={session_id}, path={image_path}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError("생성 이미지를 임시 파일로 저장하지 못했습니다.") from exc

        prompt_record = {
            "positive": positive,
            "negative": negative,
            "storage_group": "character_maker",
            "storage_session": session["id"],
        }
        if isinstance(metadata, dict):
            prompt_record.update(copy.deepcopy(metadata))
        try:
            self._atomic_write_json(prompt_path, prompt_record)
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 생성 프롬프트 기록 저장 실패: "
                f"session={session_id}, path={prompt_path}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            try:
                if os.path.isfile(image_path):
                    os.remove(image_path)
            except Exception as cleanup_exc:
                print(
                    f"[CHARACTER_MAKER] 프롬프트 저장 실패 이미지 정리 실패: "
                    f"path={image_path}, error={cleanup_exc}"
                )
                traceback.print_exc()
            raise CharacterMakerError("생성 프롬프트 기록을 저장하지 못했습니다.") from exc

        print(
            f"[CHARACTER_MAKER] 생성 결과 임시 저장 완료: "
            f"session={session_id}, image={image_path}, prompt={prompt_path}"
        )
        return image_path, prompt_path

    def add_revision(
        self,
        session_id: str,
        *,
        image_path: str,
        prompt_path: str,
        positive: str,
        negative: str,
        note: str = "",
        source: str = "user",
        stable_model_reuse: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if source not in ("user", "llm"):
            raise CharacterMakerError("source 는 user 또는 llm 이어야 합니다.")
        session = self._session(session_id)
        _assert_within(self.temp_root, image_path)
        _assert_within(self.temp_root, prompt_path)
        if not os.path.isfile(image_path):
            raise CharacterMakerError("생성된 임시 이미지 파일을 찾을 수 없습니다.")
        revision_id = uuid.uuid4().hex
        fields_snapshot = (
            copy.deepcopy(session["llm_fields"])
            if source == "llm"
            else copy.deepcopy(session["fields"])
        )
        natural_language_snapshot = (
            session["llm_natural_language"]
            if source == "llm"
            else session["natural_language"]
        )
        item = {
            "id": revision_id,
            "created_at": _now_iso(),
            "source": source,
            "fields": fields_snapshot,
            "natural_language": natural_language_snapshot,
            # LLM 편집 필드와 별개인 사용자 전용 생성 프리셋 태그도
            # 이미지 생성 시점 그대로 확정 단계까지 운반한다.
            "editable_preset_tags": copy.deepcopy(
                session["editable_preset_tags"]
            ),
            "editable_preset_enabled": copy.deepcopy(
                session["editable_preset_enabled"]
            ),
            "settings": copy.deepcopy(session["settings"]),
            "image_path": image_path,
            "prompt_path": prompt_path,
            "positive": positive,
            "negative": negative,
            "note": str(note or "")[:1000],
            "stable_model_reuse": copy.deepcopy(stable_model_reuse or []),
        }
        session["revisions"].append(item)
        if source == "llm":
            session["llm_active_revision_id"] = revision_id
        else:
            session["active_revision_id"] = revision_id
        while len(session["revisions"]) > MAX_REVISIONS:
            removed = session["revisions"].pop(0)
            for key in ("image_path", "prompt_path"):
                path = removed.get(key)
                if path and os.path.isfile(path):
                    try:
                        _assert_within(self.temp_root, path)
                        os.remove(path)
                    except Exception as exc:
                        print(
                            f"[CHARACTER_MAKER] 오래된 리비전 파일 정리 실패: "
                            f"session={session_id}, path={path}, error={exc}"
                        )
                        traceback.print_exc()
        # 정리로 활성/LLM 활성 리비전이 사라졌으면 해제한다.
        if session["active_revision_id"] and not any(
            item.get("id") == session["active_revision_id"]
            for item in session["revisions"]
        ):
            print(
                f"[CHARACTER_MAKER] 활성 리비전 정리로 해제: "
                f"session={session_id}"
            )
            session["active_revision_id"] = ""
        if session["llm_active_revision_id"] and not any(
            item.get("id") == session["llm_active_revision_id"]
            for item in session["revisions"]
        ):
            print(
                f"[CHARACTER_MAKER] LLM 활성 리비전 정리로 해제: "
                f"session={session_id}"
            )
            session["llm_active_revision_id"] = ""
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] 리비전 추가: session={session_id}, "
            f"revision={revision_id}, source={source}, count={len(session['revisions'])}"
        )
        self._persist_session(session)
        return self.public_session(session_id)

    def revision_path(self, session_id: str, revision_id: str) -> tuple[str, str]:
        session = self._session(session_id)
        item = next(
            (item for item in session["revisions"] if item["id"] == revision_id),
            None,
        )
        if item is None or not os.path.isfile(item["image_path"]):
            print(
                f"[CHARACTER_MAKER] 리비전 이미지 조회 실패: "
                f"session={session_id}, revision={revision_id}"
            )
            raise CharacterMakerError("리비전 이미지를 찾을 수 없습니다.")
        _assert_within(self.temp_root, item["image_path"])
        ext = os.path.splitext(item["image_path"])[1].lower()
        mime = {
            ".webp": "image/webp",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(ext, "application/octet-stream")
        return item["image_path"], mime

    def accept(self, session_id: str) -> dict[str, Any]:
        """LLM(좌측) 작업 결과를 사용자(우측) 영역과 대화 체크포인트로 병합한다.

        - llm_fields → fields (태그 복사)
        - llm_active_revision_id → active_revision_id (이미지 복사, 리비전은 복제하지 않고 id 공유)
        - 현재 활성 채팅 분기의 미승인 메시지 → 사용자 체크포인트로 승인
        - llm_fields / llm_active_revision_id 는 유지하여 이어 편집 가능
        """
        session = self._session(session_id)
        llm_revision_id = session.get("llm_active_revision_id", "")
        llm_fields = session.get("llm_fields")
        if not llm_revision_id or not any(
            item.get("id") == llm_revision_id for item in session["revisions"]
        ):
            raise CharacterMakerError("accept 할 LLM 결과 이미지가 없습니다.")
        has_tags = bool(llm_fields) and any(
            (llm_fields.get(field) or []) for field in EDITABLE_FIELDS
        )
        if not has_tags:
            raise CharacterMakerError("LLM 결과에 복사할 태그가 없습니다.")
        session["fields"] = copy.deepcopy(llm_fields)
        session["natural_language"] = session.get("llm_natural_language", "")
        session["active_revision_id"] = llm_revision_id
        branch_id = str(session.get("active_chat_branch_id") or "")
        accepted_count = 0
        checkpoint_id = str(session.get("user_chat_checkpoint_id") or "")
        if branch_id:
            pending_items = [
                item
                for item in session.get("chat", [])
                if item.get("branch_id") == branch_id
                and not bool(item.get("accepted", False))
            ]
            if pending_items:
                checkpoint_id = uuid.uuid4().hex
                for item in pending_items:
                    item["accepted"] = True
                    item["checkpoint_id"] = checkpoint_id
                    accepted_count += 1
                session["user_chat_checkpoint_id"] = checkpoint_id
            else:
                print(
                    f"[CHARACTER_MAKER] accept 채팅 병합 스킵: "
                    f"session={session_id}, branch={branch_id}, 미승인 메시지 없음"
                )
        else:
            print(
                f"[CHARACTER_MAKER] accept 채팅 병합 스킵: "
                f"session={session_id}, 활성 채팅 분기 없음"
            )
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] accept: LLM 결과를 사용자 영역으로 복사: "
            f"session={session_id}, revision={llm_revision_id}, "
            f"branch={branch_id or 'none'}, chat_accepted={accepted_count}, "
            f"checkpoint={checkpoint_id or 'none'}"
        )
        self._persist_session(session)
        return self.public_session(session_id)

    async def test_rag(
        self, query: str = "", config_override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        config = copy.deepcopy(self.config_getter() or {})
        if config_override:
            config.update(config_override)
        started = time.perf_counter()
        service = get_danbooru_rag_service()
        try:
            health = await self._ensure_rag_ready(service)
            results = []
            if query.strip():
                results = await self._rag_search(query.strip()[:300], config=config)
            else:
                print("[CHARACTER_MAKER_RAG] 검색어 없이 내장 상태 확인만 완료")
        except CharacterMakerError:
            raise
        except DanbooruRagError as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 테스트 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(str(exc)) from exc
        except Exception as exc:
            print(
                "[CHARACTER_MAKER_RAG] 내장 테스트 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(f"내장 RAG 테스트 실패: {exc}") from exc
        return {
            "success": True,
            "health": health,
            "results": results,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    @staticmethod
    def _atomic_write_json(path: str, value: dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".character_maker_", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
            raise

    def confirm(self, session_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CharacterMakerError("확정 요청은 객체여야 합니다.")
        session = self._session(session_id)
        registration_mode = str(payload.get("registration_mode") or "new").strip()
        if registration_mode not in ("new", "existing"):
            print(
                f"[CHARACTER_MAKER] 확정 등록 방식 오류: session={session_id}, "
                f"registration_mode={registration_mode!r}"
            )
            raise CharacterMakerError("등록 방식은 new 또는 existing 이어야 합니다.")
        appearance_mode = str(payload.get("appearance_mode") or "new").strip()
        if appearance_mode not in ("new", "existing"):
            print(
                f"[CHARACTER_MAKER] 확정 외모 방식 오류: session={session_id}, "
                f"appearance_mode={appearance_mode!r}"
            )
            raise CharacterMakerError("외모 등록 방식은 new 또는 existing 이어야 합니다.")
        outfit_mode = str(payload.get("outfit_mode") or "new").strip()
        if outfit_mode not in ("new", "existing"):
            print(
                f"[CHARACTER_MAKER] 확정 복장 방식 오류: session={session_id}, "
                f"outfit_mode={outfit_mode!r}"
            )
            raise CharacterMakerError("복장 등록 방식은 new 또는 existing 이어야 합니다.")
        set_representative = payload.get("set_representative", False)
        if not isinstance(set_representative, bool):
            print(
                f"[CHARACTER_MAKER] 대표 이미지 설정값 오류: session={session_id}, "
                f"value={set_representative!r}"
            )
            raise CharacterMakerError("대표 이미지 설정값은 true 또는 false 여야 합니다.")

        def registration_name(raw_value: Any, label: str, mode: str) -> str:
            try:
                if mode == "existing":
                    return _existing_registration_name(raw_value, label)
                return _safe_registration_name(raw_value, label)
            except CharacterMakerError as exc:
                print(
                    f"[CHARACTER_MAKER] 확정 이름 검증 실패: session={session_id}, "
                    f"label={label}, mode={mode}, value={raw_value!r}, error={exc}"
                )
                traceback.print_exc()
                raise

        character_name = registration_name(
            payload.get("character_name"), "캐릭터명", registration_mode
        )
        appearance_name = registration_name(
            payload.get("appearance_name"), "외모 프리셋명", appearance_mode
        )
        outfit_name = registration_name(
            payload.get("outfit_name"), "복장 프리셋명", outfit_mode
        )

        revision_id = str(session.get("active_revision_id") or "")
        requested_revision_id = str(payload.get("revision_id") or "")
        if not revision_id:
            print(
                f"[CHARACTER_MAKER] 확정 거부: session={session_id}, "
                "현재 사용자 이미지 리비전 없음"
            )
            raise CharacterMakerError(
                "확정할 현재 사용자 이미지가 없습니다. 사용자 이미지를 먼저 생성하세요."
            )
        if requested_revision_id and requested_revision_id != revision_id:
            print(
                f"[CHARACTER_MAKER] 확정 리비전 불일치: session={session_id}, "
                f"requested={requested_revision_id}, active_user={revision_id}"
            )
            raise CharacterMakerError(
                "확정창을 연 뒤 사용자 이미지가 변경되었습니다. 확정창을 다시 여세요."
            )
        promote_revision = next(
            (item for item in session["revisions"] if item.get("id") == revision_id),
            None,
        )
        if promote_revision is None:
            print(
                f"[CHARACTER_MAKER] 확정 거부: session={session_id}, "
                f"활성 사용자 리비전 누락={revision_id}"
            )
            raise CharacterMakerError("확정할 현재 사용자 이미지 리비전을 찾을 수 없습니다.")
        source_image = str(promote_revision.get("image_path") or "")
        if not source_image or not os.path.isfile(source_image):
            print(
                f"[CHARACTER_MAKER] 확정 거부: session={session_id}, "
                f"사용자 이미지 파일 누락, revision={revision_id}, path={source_image!r}"
            )
            raise CharacterMakerError("확정할 현재 사용자 이미지 파일을 찾을 수 없습니다.")
        _assert_within(self.temp_root, source_image)
        try:
            revision_fields = _normalize_fields(promote_revision.get("fields"))
        except CharacterMakerError as exc:
            print(
                f"[CHARACTER_MAKER] 리비전 필드 스냅샷 오류: session={session_id}, "
                f"revision={revision_id}, error={exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(
                f"사용자 이미지의 태그 스냅샷이 올바르지 않습니다: {exc}"
            ) from exc
        try:
            revision_editable_preset_tags = _normalize_editable_preset_tags(
                promote_revision.get("editable_preset_tags")
            )
            revision_editable_preset_enabled = (
                _normalize_editable_preset_enabled(
                    promote_revision.get("editable_preset_enabled")
                )
            )
        except CharacterMakerError as exc:
            print(
                f"[CHARACTER_MAKER] 리비전 생성 프리셋 편집 스냅샷 오류: "
                f"session={session_id}, revision={revision_id}, error={exc}"
            )
            traceback.print_exc()
            raise CharacterMakerError(
                f"사용자 이미지의 생성 프리셋 편집 스냅샷이 올바르지 않습니다: {exc}"
            ) from exc
        if appearance_mode == "new" and not revision_fields["appearance"]:
            print(
                f"[CHARACTER_MAKER] 확정 거부: session={session_id}, "
                f"revision={revision_id}, 외모 태그 비어 있음"
            )
            raise CharacterMakerError(
                "새 외모 프리셋을 등록하려면 사용자 이미지의 외모 태그가 필요합니다."
            )
        if outfit_mode == "new" and not revision_fields["outfit"]:
            print(
                f"[CHARACTER_MAKER] 확정 거부: session={session_id}, "
                f"revision={revision_id}, 복장 태그 비어 있음"
            )
            raise CharacterMakerError(
                "새 복장 프리셋을 등록하려면 사용자 이미지의 복장 태그가 필요합니다."
            )

        expression_mode = str(payload.get("expression_mode") or "")
        if expression_mode not in ("existing", "new"):
            print(
                f"[CHARACTER_MAKER] 확정 표정 방식 오류: session={session_id}, "
                f"expression_mode={expression_mode!r}"
            )
            raise CharacterMakerError(
                "캐릭터 카드를 등록하려면 기존 또는 새 표정 프리셋을 선택하세요."
            )
        expression_name = registration_name(
            payload.get("expression_name"), "표정 프리셋명", expression_mode
        )
        if expression_mode == "new" and not revision_fields["expression"]:
            print(
                f"[CHARACTER_MAKER] 새 표정 등록 거부: session={session_id}, "
                f"revision={revision_id}, 표정 태그 비어 있음"
            )
            raise CharacterMakerError(
                "새 표정 프리셋을 등록하려면 사용자 이미지의 표정 태그가 필요합니다."
            )

        composition_mode = str(payload.get("composition_mode") or "none")
        if composition_mode not in ("none", "new"):
            print(
                f"[CHARACTER_MAKER] 확정 구도 방식 오류: session={session_id}, "
                f"composition_mode={composition_mode!r}"
            )
            raise CharacterMakerError("구도 등록 방식이 올바르지 않습니다.")
        composition_name = ""
        if composition_mode == "new":
            composition_name = registration_name(
                payload.get("composition_name"), "구도 프리셋명", composition_mode
            )
            if not revision_fields["composition"]:
                print(
                    f"[CHARACTER_MAKER] 새 구도 등록 거부: session={session_id}, "
                    f"revision={revision_id}, 구도 태그 비어 있음"
                )
                raise CharacterMakerError(
                    "새 구도 프리셋을 등록하려면 사용자 이미지의 구도/기타 태그가 필요합니다."
                )

        natural_language_mode = str(
            payload.get("natural_language_mode") or "none"
        ).strip()
        if natural_language_mode not in ("none", "new"):
            print(
                f"[CHARACTER_MAKER] 확정 자연어 방식 오류: session={session_id}, "
                f"natural_language_mode={natural_language_mode!r}"
            )
            raise CharacterMakerError("자연어 등록 방식이 올바르지 않습니다.")
        revision_natural_language = promote_revision.get("natural_language", "")
        if not isinstance(revision_natural_language, str):
            print(
                f"[CHARACTER_MAKER] 리비전 자연어 스냅샷 형식 오류: "
                f"session={session_id}, revision={revision_id}, "
                f"type={type(revision_natural_language).__name__}"
            )
            raise CharacterMakerError(
                "사용자 이미지의 자연어 스냅샷이 올바르지 않습니다."
            )
        natural_language_text = revision_natural_language.strip()
        natural_language_name = ""
        if natural_language_mode == "new":
            natural_language_name = registration_name(
                payload.get("natural_language_name"),
                "자연어 프리셋명",
                natural_language_mode,
            )
            if not natural_language_text:
                print(
                    f"[CHARACTER_MAKER] 새 자연어 프리셋 등록 거부: "
                    f"session={session_id}, revision={revision_id}, 자연어 비어 있음"
                )
                raise CharacterMakerError(
                    "자연어 프리셋을 등록하려면 사용자 이미지 생성 당시 자연어가 필요합니다."
                )

        raw_editable_registrations = payload.get(
            "editable_preset_registrations", {}
        )
        if not isinstance(raw_editable_registrations, dict):
            print(
                f"[CHARACTER_MAKER] 생성 프리셋 등록 요청 형식 오류: "
                f"session={session_id}, "
                f"type={type(raw_editable_registrations).__name__}"
            )
            raise CharacterMakerError("생성 프리셋 등록 요청은 객체여야 합니다.")
        unknown_registration_fields = sorted(
            set(raw_editable_registrations) - set(EDITABLE_PRESET_FIELDS)
        )
        if unknown_registration_fields:
            print(
                f"[CHARACTER_MAKER] 알 수 없는 생성 프리셋 등록 필드: "
                f"session={session_id}, fields={unknown_registration_fields}"
            )
            raise CharacterMakerError(
                "알 수 없는 생성 프리셋 등록 항목이 있습니다: "
                + ", ".join(unknown_registration_fields)
            )

        editable_preset_registrations: dict[str, dict[str, str]] = {}
        planned_names: set[tuple[str, str]] = set()
        for field in EDITABLE_PRESET_FIELDS:
            raw_registration = raw_editable_registrations.get(field, {})
            if raw_registration is None:
                raw_registration = {}
            if not isinstance(raw_registration, dict):
                print(
                    f"[CHARACTER_MAKER] 생성 프리셋 등록 항목 형식 오류: "
                    f"session={session_id}, field={field}, "
                    f"type={type(raw_registration).__name__}"
                )
                raise CharacterMakerError(
                    f"{EDITABLE_PRESET_LABELS[field]} 등록 설정은 객체여야 합니다."
                )
            mode = str(raw_registration.get("mode") or "none").strip()
            if mode not in ("none", "new"):
                print(
                    f"[CHARACTER_MAKER] 생성 프리셋 등록 방식 오류: "
                    f"session={session_id}, field={field}, mode={mode!r}"
                )
                raise CharacterMakerError(
                    f"{EDITABLE_PRESET_LABELS[field]} 등록 방식이 올바르지 않습니다."
                )
            if mode == "none":
                continue
            if not revision_editable_preset_enabled[field]:
                print(
                    f"[CHARACTER_MAKER] 잠긴 생성 프리셋 신규 등록 거부: "
                    f"session={session_id}, revision={revision_id}, field={field}"
                )
                raise CharacterMakerError(
                    f"{EDITABLE_PRESET_LABELS[field]}은(는) 선택 이미지 생성 당시 "
                    "자유 편집 상태가 아니었습니다."
                )
            if not revision_editable_preset_tags[field]:
                print(
                    f"[CHARACTER_MAKER] 빈 생성 프리셋 신규 등록 거부: "
                    f"session={session_id}, revision={revision_id}, field={field}"
                )
                raise CharacterMakerError(
                    f"{EDITABLE_PRESET_LABELS[field]} 프리셋으로 저장할 태그가 없습니다."
                )
            preset_name = registration_name(
                raw_registration.get("name"),
                f"{EDITABLE_PRESET_LABELS[field]} 프리셋명",
                "new",
            )
            category = EDITABLE_PRESET_CATEGORIES[field]
            planned_key = (category, preset_name.casefold())
            if planned_key in planned_names:
                print(
                    f"[CHARACTER_MAKER] 생성 프리셋 신규 이름 중복: "
                    f"session={session_id}, category={category}, name={preset_name!r}"
                )
                raise CharacterMakerError(
                    f"같은 종류에 동일한 새 프리셋명이 중복되었습니다: {preset_name}"
                )
            planned_names.add(planned_key)
            editable_preset_registrations[field] = {
                "mode": "new",
                "name": preset_name,
                "category": category,
            }

        asset_mode_module = importlib.import_module("modes.asset_mode")
        tags_file = asset_mode_module.TAGS_FILE
        asset_dir = asset_mode_module.ASSET_DIR
        requirements_dir = asset_mode_module.NAME_MAPPING_BACKUP_DIR
        old_tags = self.asset_manager.get_tags()
        new_tags = copy.deepcopy(old_tags)

        collisions: list[str] = []
        characters = new_tags.get("characters", {})
        if not isinstance(characters, dict):
            print(
                f"[CHARACTER_MAKER] 캐릭터 태그 구조 오류: session={session_id}, "
                f"type={type(characters).__name__}"
            )
            raise CharacterMakerError("캐릭터 태그 데이터 구조가 올바르지 않습니다.")
        appearances = new_tags.setdefault("appearances", {})
        if not isinstance(appearances, dict):
            print(
                f"[CHARACTER_MAKER] 외모 태그 구조 오류: session={session_id}, "
                f"type={type(appearances).__name__}"
            )
            raise CharacterMakerError("외모 태그 데이터 구조가 올바르지 않습니다.")
        outfits = new_tags.setdefault("outfits", {})
        if not isinstance(outfits, dict):
            print(
                f"[CHARACTER_MAKER] 복장 태그 구조 오류: session={session_id}, "
                f"type={type(outfits).__name__}"
            )
            raise CharacterMakerError("복장 태그 데이터 구조가 올바르지 않습니다.")
        expressions = new_tags.setdefault("expressions", {})
        if not isinstance(expressions, dict):
            print(
                f"[CHARACTER_MAKER] 표정 태그 구조 오류: session={session_id}, "
                f"type={type(expressions).__name__}"
            )
            raise CharacterMakerError("표정 태그 데이터 구조가 올바르지 않습니다.")
        composition_presets = new_tags.setdefault("composition_presets", {})
        if not isinstance(composition_presets, dict):
            print(
                f"[CHARACTER_MAKER] 구도 태그 구조 오류: session={session_id}, "
                f"type={type(composition_presets).__name__}"
            )
            raise CharacterMakerError("구도 태그 데이터 구조가 올바르지 않습니다.")
        natural_language_presets = new_tags.setdefault(
            "natural_language_presets", {}
        )
        if not isinstance(natural_language_presets, dict):
            print(
                f"[CHARACTER_MAKER] 자연어 프리셋 구조 오류: session={session_id}, "
                f"type={type(natural_language_presets).__name__}"
            )
            raise CharacterMakerError("자연어 프리셋 데이터 구조가 올바르지 않습니다.")
        editable_category_targets: dict[str, dict[str, Any]] = {}
        for category, label in (
            ("quality_presets", "품질"),
            ("artist_presets", "아티스트"),
            ("negative_presets", "부정"),
            ("character_negative_presets", "캐릭터 부정"),
        ):
            target = new_tags.setdefault(category, {})
            if not isinstance(target, dict):
                print(
                    f"[CHARACTER_MAKER] {label} 프리셋 구조 오류: "
                    f"session={session_id}, category={category}, "
                    f"type={type(target).__name__}"
                )
                raise CharacterMakerError(
                    f"{label} 프리셋 데이터 구조가 올바르지 않습니다."
                )
            editable_category_targets[category] = target
        if registration_mode == "new" and character_name in characters:
            collisions.append(f"캐릭터 '{character_name}'")
        if registration_mode == "existing" and character_name not in characters:
            print(
                f"[CHARACTER_MAKER] 기존 캐릭터 추가 실패: session={session_id}, "
                f"character={character_name!r}, reason=not_found"
            )
            raise CharacterMakerError(f"기존 캐릭터 '{character_name}'을 찾을 수 없습니다.")
        if appearance_mode == "new" and appearance_name in appearances:
            collisions.append(f"외모 '{appearance_name}'")
        if appearance_mode == "existing" and appearance_name not in appearances:
            print(
                f"[CHARACTER_MAKER] 기존 외모 조회 실패: session={session_id}, "
                f"appearance={appearance_name!r}"
            )
            raise CharacterMakerError(
                f"기존 외모 프리셋 '{appearance_name}'을 찾을 수 없습니다."
            )
        if outfit_mode == "new" and outfit_name in outfits:
            collisions.append(f"복장 '{outfit_name}'")
        if outfit_mode == "existing" and outfit_name not in outfits:
            print(
                f"[CHARACTER_MAKER] 기존 복장 조회 실패: session={session_id}, "
                f"outfit={outfit_name!r}"
            )
            raise CharacterMakerError(
                f"기존 복장 프리셋 '{outfit_name}'을 찾을 수 없습니다."
            )
        if expression_mode == "new" and expression_name in expressions:
            collisions.append(f"표정 '{expression_name}'")
        if expression_mode == "existing" and expression_name not in expressions:
            print(
                f"[CHARACTER_MAKER] 기존 표정 조회 실패: session={session_id}, "
                f"expression={expression_name!r}"
            )
            raise CharacterMakerError(f"기존 표정 프리셋 '{expression_name}'을 찾을 수 없습니다.")
        if (
            composition_mode == "new"
            and composition_name in composition_presets
        ):
            collisions.append(f"구도 '{composition_name}'")
        if (
            natural_language_mode == "new"
            and natural_language_name in natural_language_presets
        ):
            collisions.append(f"자연어 '{natural_language_name}'")
        for field, registration in editable_preset_registrations.items():
            target = editable_category_targets[registration["category"]]
            if registration["name"] in target:
                collisions.append(
                    f"{EDITABLE_PRESET_LABELS[field]} '{registration['name']}'"
                )
        if collisions:
            print(
                f"[CHARACTER_MAKER] 확정 충돌: session={session_id}, "
                f"collisions={collisions}"
            )
            raise CharacterMakerError("이미 존재하는 이름입니다: " + ", ".join(collisions))

        if appearance_mode == "new":
            appearances[appearance_name] = list(revision_fields["appearance"])
        if outfit_mode == "new":
            outfits[outfit_name] = list(revision_fields["outfit"])
        if expression_mode == "new":
            expressions[expression_name] = list(revision_fields["expression"])
        if composition_mode == "new":
            composition_presets[composition_name] = list(revision_fields["composition"])
        if natural_language_mode == "new":
            natural_language_presets[natural_language_name] = natural_language_text
        for field, registration in editable_preset_registrations.items():
            editable_category_targets[registration["category"]][
                registration["name"]
            ] = list(revision_editable_preset_tags[field])
        if registration_mode == "new":
            new_tags.setdefault("characters", {})[character_name] = {
                "appearance": appearance_name,
                "outfit": outfit_name,
                "expression": expression_name,
            }

        char_dir = os.path.join(asset_dir, self.asset_manager._safe_dirname(character_name))
        char_dir_existed = os.path.exists(char_dir)
        if registration_mode == "new" and char_dir_existed:
            print(
                f"[CHARACTER_MAKER] 신규 캐릭터 폴더 충돌: session={session_id}, "
                f"character={character_name!r}, path={char_dir}"
            )
            raise CharacterMakerError(
                "동일한 저장 폴더가 이미 존재합니다. 다른 캐릭터명을 사용하세요."
            )
        promotion_outfit_dir = os.path.join(
            char_dir, self.asset_manager._safe_dirname(outfit_name)
        )
        promotion_outfit_dir_existed = os.path.exists(promotion_outfit_dir)

        destination = os.path.join(
            promotion_outfit_dir,
            self.asset_manager._safe_dirname(expression_name),
        )
        destination_existed = os.path.isdir(destination)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tags_changed = new_tags != old_tags
        backup_path = ""
        representative_backup_path = ""
        promoted_image = ""
        image_target = ""
        prompt_target = ""
        image_created = False
        prompt_created = False
        representative_path = os.path.join(destination, "_representative.json")
        representative_file_existed = False
        representative_changed = False
        representative_update_reason = "preserved"
        tags_written = False
        try:
            if tags_changed:
                os.makedirs(requirements_dir, exist_ok=True)
                backup_path = os.path.join(
                    requirements_dir, f"tags_before_character_maker_{stamp}.json"
                )
                if os.path.isfile(tags_file):
                    shutil.copy2(tags_file, backup_path)
                    print(
                        f"[CHARACTER_MAKER] tags.json 백업 완료: "
                        f"session={session_id}, backup={backup_path}"
                    )
                else:
                    with open(backup_path, "w", encoding="utf-8") as handle:
                        json.dump(old_tags, handle, ensure_ascii=False, indent=2)
                    print(
                        f"[CHARACTER_MAKER] 메모리 태그 백업 완료: "
                        f"session={session_id}, backup={backup_path}"
                    )
                self._atomic_write_json(tags_file, new_tags)
                tags_written = True
                self.asset_manager._tags = copy.deepcopy(new_tags)
                self.asset_manager._tags_loaded = True
            else:
                print(
                    f"[CHARACTER_MAKER] 기존 프리셋만 사용하여 태그 저장 생략: "
                    f"session={session_id}, character={character_name!r}"
                )

            os.makedirs(destination, exist_ok=True)
            source_prompt = str(promote_revision.get("prompt_path") or "")
            original_image_name = os.path.basename(source_image)
            original_image_base, image_extension = os.path.splitext(original_image_name)
            image_name = original_image_name
            image_base = original_image_base
            suffix = 2
            while True:
                prompt_name = f"{image_base}_prompt.json"
                image_target = os.path.join(destination, image_name)
                prompt_target = os.path.join(destination, prompt_name)
                if not os.path.exists(image_target) and not os.path.exists(prompt_target):
                    break
                image_base = f"{original_image_base}_{suffix}"
                image_name = f"{image_base}{image_extension}"
                suffix += 1
            if image_name != original_image_name:
                print(
                    f"[CHARACTER_MAKER] 카드 파일명 충돌 회피: session={session_id}, "
                    f"original={original_image_name!r}, selected={image_name!r}"
                )
            with open(source_image, "rb") as source_handle, open(
                image_target, "xb"
            ) as target_handle:
                image_created = True
                shutil.copyfileobj(source_handle, target_handle)

            prompt_payload: dict[str, Any] = {}
            if source_prompt and os.path.isfile(source_prompt):
                try:
                    _assert_within(self.temp_root, source_prompt)
                    with open(source_prompt, "r", encoding="utf-8") as handle:
                        loaded_prompt = json.load(handle)
                    if isinstance(loaded_prompt, dict):
                        prompt_payload = loaded_prompt
                    else:
                        print(
                            f"[CHARACTER_MAKER] 원본 프롬프트 기록 형식 오류, "
                            f"리비전 스냅샷으로 대체: session={session_id}, "
                            f"revision={revision_id}, type={type(loaded_prompt).__name__}"
                        )
                except Exception as prompt_exc:
                    print(
                        f"[CHARACTER_MAKER] 원본 프롬프트 기록 로드 실패, "
                        f"리비전 스냅샷으로 대체: session={session_id}, "
                        f"revision={revision_id}, path={source_prompt!r}, "
                        f"error={type(prompt_exc).__name__}: {prompt_exc}"
                    )
                    traceback.print_exc()
            else:
                print(
                    f"[CHARACTER_MAKER] 원본 프롬프트 기록 없음, "
                    f"리비전 스냅샷으로 생성: session={session_id}, "
                    f"revision={revision_id}, path={source_prompt!r}"
                )
            used_positive = prompt_payload.get("positive")
            if not isinstance(used_positive, str) or not used_positive.strip():
                print(
                    f"[CHARACTER_MAKER] 원본 긍정 프롬프트 없음, "
                    f"리비전 값으로 대체: session={session_id}, revision={revision_id}"
                )
                used_positive = str(promote_revision.get("positive") or "")
            used_negative = prompt_payload.get("negative")
            if not isinstance(used_negative, str):
                print(
                    f"[CHARACTER_MAKER] 원본 부정 프롬프트 형식 오류, "
                    f"리비전 값으로 대체: session={session_id}, revision={revision_id}, "
                    f"type={type(used_negative).__name__}"
                )
                used_negative = str(promote_revision.get("negative") or "")
            prompt_payload.update(
                {
                    "positive": used_positive,
                    "negative": used_negative,
                    "character": character_name,
                    "appearance": appearance_name,
                    "outfit": outfit_name,
                    "expression": expression_name,
                    "storage_group": "",
                    "storage_outfit": outfit_name,
                    "character_maker_fields": copy.deepcopy(revision_fields),
                    "character_maker_natural_language": str(
                        promote_revision.get("natural_language") or ""
                    ),
                    "character_maker_editable_preset_tags": copy.deepcopy(
                        revision_editable_preset_tags
                    ),
                    "character_maker_editable_preset_enabled": copy.deepcopy(
                        revision_editable_preset_enabled
                    ),
                    "character_maker_settings": copy.deepcopy(
                        promote_revision.get("settings") or {}
                    ),
                    "composition_preset": composition_name,
                    "source_revision": revision_id,
                    "source_revision_source": str(
                        promote_revision.get("source") or "user"
                    ),
                }
            )
            with open(prompt_target, "x", encoding="utf-8") as prompt_handle:
                prompt_created = True
                json.dump(
                    prompt_payload,
                    prompt_handle,
                    ensure_ascii=False,
                    indent=2,
                )
                prompt_handle.flush()
                os.fsync(prompt_handle.fileno())

            representative_file_existed = os.path.isfile(representative_path)
            valid_representative = False
            if representative_file_existed:
                try:
                    with open(representative_path, "r", encoding="utf-8") as handle:
                        representative_payload = json.load(handle)
                    representative_filename = (
                        representative_payload.get("filename", "")
                        if isinstance(representative_payload, dict)
                        else ""
                    )
                    valid_representative = (
                        isinstance(representative_filename, str)
                        and representative_filename == os.path.basename(
                            representative_filename
                        )
                        and bool(representative_filename)
                        and os.path.isfile(
                            os.path.join(destination, representative_filename)
                        )
                    )
                    if not valid_representative:
                        print(
                            f"[CHARACTER_MAKER] 기존 대표 이미지 기록이 유효하지 않아 "
                            f"새 카드로 자동 복구: session={session_id}, "
                            f"path={representative_path}"
                        )
                except Exception as representative_exc:
                    print(
                        f"[CHARACTER_MAKER] 기존 대표 이미지 기록 확인 실패, "
                        f"새 카드로 자동 복구: session={session_id}, "
                        f"path={representative_path}, "
                        f"error={type(representative_exc).__name__}: "
                        f"{representative_exc}"
                    )
                    traceback.print_exc()

            if set_representative or not valid_representative:
                if representative_file_existed:
                    os.makedirs(requirements_dir, exist_ok=True)
                    representative_backup_path = os.path.join(
                        requirements_dir,
                        "representative_before_character_maker_"
                        f"{stamp}_{uuid.uuid4().hex[:8]}.json",
                    )
                    shutil.copy2(representative_path, representative_backup_path)
                    print(
                        f"[CHARACTER_MAKER] 대표 이미지 기록 백업 완료: "
                        f"session={session_id}, backup={representative_backup_path}"
                    )
                self._atomic_write_json(
                    representative_path,
                    {"filename": image_name},
                )
                representative_changed = True
                representative_update_reason = (
                    "requested" if set_representative else "missing"
                )
            else:
                print(
                    f"[CHARACTER_MAKER] 기존 대표 이미지 유지: "
                    f"session={session_id}, path={representative_path}"
                )
            promoted_image = image_target
        except Exception as exc:
            print(
                f"[CHARACTER_MAKER] 확정 저장 실패, 롤백 시작: "
                f"session={session_id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            if tags_written:
                try:
                    self._atomic_write_json(tags_file, old_tags)
                    self.asset_manager._tags = copy.deepcopy(old_tags)
                    self.asset_manager._tags_loaded = True
                except Exception as rollback_exc:
                    print(
                        f"[CHARACTER_MAKER] tags.json 롤백 실패: "
                        f"session={session_id}, error={rollback_exc}, "
                        f"backup={backup_path}"
                    )
                    traceback.print_exc()
            if representative_changed:
                try:
                    if representative_backup_path and os.path.isfile(
                        representative_backup_path
                    ):
                        shutil.copy2(representative_backup_path, representative_path)
                    elif not representative_file_existed and os.path.isfile(
                        representative_path
                    ):
                        os.remove(representative_path)
                except Exception as rollback_exc:
                    print(
                        f"[CHARACTER_MAKER] 대표 이미지 기록 롤백 실패: "
                        f"session={session_id}, path={representative_path}, "
                        f"error={rollback_exc}"
                    )
                    traceback.print_exc()
            for created_path, was_created in (
                (prompt_target, prompt_created),
                (image_target, image_created),
            ):
                if was_created and created_path and os.path.isfile(created_path):
                    try:
                        os.remove(created_path)
                    except Exception as cleanup_exc:
                        print(
                            f"[CHARACTER_MAKER] 실패한 카드 파일 정리 실패: "
                            f"path={created_path}, error={cleanup_exc}"
                        )
                        traceback.print_exc()

            cleanup_dir = ""
            if not char_dir_existed and os.path.isdir(char_dir):
                cleanup_dir = char_dir
            elif (
                not promotion_outfit_dir_existed
                and os.path.isdir(promotion_outfit_dir)
            ):
                cleanup_dir = promotion_outfit_dir
            elif not destination_existed and os.path.isdir(destination):
                cleanup_dir = destination
            if cleanup_dir:
                try:
                    asset_root_real = os.path.realpath(asset_dir)
                    cleanup_real = os.path.realpath(cleanup_dir)
                    if (
                        os.path.commonpath([asset_root_real, cleanup_real])
                        == asset_root_real
                        and cleanup_real != asset_root_real
                    ):
                        shutil.rmtree(cleanup_real)
                    else:
                        print(
                            f"[CHARACTER_MAKER] 실패한 폴더 정리 거부: "
                            f"path={cleanup_dir}, asset_root={asset_dir}"
                        )
                except Exception as cleanup_exc:
                    print(
                        f"[CHARACTER_MAKER] 실패한 카드 폴더 정리 실패: "
                        f"path={cleanup_dir}, error={cleanup_exc}"
                    )
                    traceback.print_exc()
            raise CharacterMakerError(f"캐릭터 확정 저장 실패: {exc}") from exc

        finalized = {
            "at": _now_iso(),
            "registration_mode": registration_mode,
            "character_name": character_name,
            "appearance_mode": appearance_mode,
            "appearance_name": appearance_name,
            "outfit_mode": outfit_mode,
            "outfit_name": outfit_name,
            "expression_name": expression_name,
            "composition_name": composition_name,
            "natural_language_mode": natural_language_mode,
            "natural_language_name": natural_language_name,
            "editable_preset_registrations": copy.deepcopy(
                editable_preset_registrations
            ),
            "revision_id": revision_id,
            "promoted_image": bool(promoted_image),
            "promoted_filename": os.path.basename(promoted_image),
            "representative_updated": representative_changed,
            "representative_update_reason": representative_update_reason,
            "backup_path": backup_path,
            "representative_backup_path": representative_backup_path,
        }
        session["finalized"] = finalized
        session["updated_at"] = _now_iso()
        print(
            f"[CHARACTER_MAKER] 캐릭터 확정 완료: session={session_id}, "
            f"registration_mode={registration_mode}, character={character_name!r}, "
            f"appearance_mode={appearance_mode}, outfit_mode={outfit_mode}, "
            f"natural_language_mode={natural_language_mode}, "
            f"image={bool(promoted_image)}, "
            f"representative={representative_update_reason}"
        )
        self._persist_session(session)
        return {
            "success": True,
            "finalized": finalized,
            "session": self.public_session(session_id),
        }

    def delete_session(self, session_id: str) -> None:
        # 단일 고정 세션: 삭제 요청은 빈 세션 리셋과 같다.
        # _session을 먼저 호출해 세션이 없으면 복구(예외 방지).
        self._session(session_id)
        self._reset_single_session()
        self._persist_session(self.sessions[SINGLE_SESSION_ID])
        print(
            f"[CHARACTER_MAKER] 세션 삭제(리셋) 완료: session={SINGLE_SESSION_ID}"
        )
