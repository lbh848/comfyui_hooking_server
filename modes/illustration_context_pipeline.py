"""CHAT -> CHARACTER-RESOLVE -> optional PROFILE-RESOLVE -> downstream pipeline.

RisuAI는 Comfy history에 이미지가 여러 장 있어도 첫 장만 소비한다. 이 모듈은
최초 CHAT 요청과 후속 결과 회수 요청을 구분하고, 한 세션의 모든 장면 프롬프트와
이미지를 서버 메모리에 보관할 수 있는 공통 형식을 제공한다.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_left
import illustration_flow
import contextvars
import hashlib
import json
import math
import os
import re
import time
import traceback
import datetime
import uuid
from copy import deepcopy
from urllib.parse import quote

import yaml

from modes import (
    illustration_original_assets,
    lighbd_service,
    llm_service,
    multi_char_mask,
    postprocess,
    visual_profile_translation_cache,
)
from modes.visual_profiles import (
    build_natural_profile_catalog,
    profile_by_id,
    profile_by_name,
    resolve_visual_base,
    tag_values as visual_tag_values,
    visual_profile_names,
)
from modes.illustration_prompt_assembly import PromptText, input_variables, is_symbolic, prompt_join, prompt_sub, symbolic_call2_template, variable
from modes import illustration_prompt_builders as prompt_builders


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts", "lighbd")
PROMPT_BACKUP_DIR = os.path.join(BASE_DIR, "backups", "prompt_files")
SESSION_DIR = os.path.join(BASE_DIR, "logs", "illustration_context_sessions")
WORKFLOW_BACKUP_DIR = os.path.join(BASE_DIR, "workflow_backup")

_BACKUP_IMAGE_EXTENSIONS = (".webp", ".avif", ".png", ".gif")
_BACKUP_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,200}")

CONTEXT_PREFIX = "__LB_ILLUST_CONTEXT_V1__"
RESULT_PREFIX = "__LB_ILLUST_RESULT_V1__"
REGENERATE_PREFIX = "__LB_ILLUST_REGENERATE_V1__"
PROMPT_BATCH_PREFIX = "__LB_ILLUST_PROMPT_BATCH_V1__"
EASY_EDIT_PREFIX = "__LB_ILLUST_EASY_EDIT_V1__"
MAX_ILLUSTRATION_SLOT_COUNT = 65
MAX_EASY_EDIT_DIRECTION_LENGTH = 4000
PROFILE_CONTEXT_TRANSLATE_TASK_KEY = "illustration_profile_context_translate"
PROFILE_CONTEXT_TRANSLATE_CALL_NAME = "PROFILE-CONTEXT-TRANSLATE"
_CHARACTER_IDENTITY_LORA_FIELDS = (
    "loras",
    "loras_solo",
    "loras_group",
    "face_loras",
)
_profile_context_translation_lock = asyncio.Lock()

# 삽화 1회 생성(build_from_context) 동안 발생한 모든 LLM 호출의 history_id를
# 수집하는 컨텍스트 변수. _call_pipeline_llm 가 성공/실패/취소/실패시도 레코드의
# id를 여기에 append 하고, build_from_context 가 종료된 뒤 이 목록을 반환한다.
# contextvars 로 asyncio gather/병렬 실행에도 안전하게 per-run 수집된다.
# 이 id 목록은 백업 _info.json 의 llm_trace 로 저장되어 "LLM 흐름 추적" 버튼이
# 해당 백업을 만든 MULTI-CHAR-MASK~CALL3 전체 흐름을 정확히 매칭하는 데 쓰인다.
_llm_trace_ctx: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "illustration_llm_trace", default=None
)


def _trace_append(history_id: str) -> None:
    """현재 실행 중인 build_from_context 의 trace 목록에 history_id를 추가한다.

    활성 trace가 없으면(독립 호출·재생성 등) 아무 것도 하지 않는다.
    """
    trace = _llm_trace_ctx.get()
    if trace is None:
        return
    hid = str(history_id or "").strip()
    if not hid:
        return
    if hid not in trace:
        trace.append(hid)

PROMPT_FILES = {
    "call1_enhance": "enhance.txt",
    "time_place": "time_place.txt",
    "character_resolve": "character.txt",
    "profile_resolve": "profile.txt",
    "call2_jailbreak": "jailbreak.txt",
    "call2_job": "job.txt",
    "call2_prefill": "prefill.txt",
    "call2_common": "system.txt",
    "call2_explicit": "explicit.txt",
    "call2_select": "scene_select.txt",
    "call2_plan": "plan.txt",
    "call2_detail": "detail.txt",
    "call2_keyvis": "keyvisual.txt",
    "call2_fallback": "fallback.txt",
    # Combined TOON response format used by the ENQUEUE endpoint.
    "call2_format": "format.txt",
    "call2_preset": "preset.txt",
    "call3_speak": "speak.txt",
    "call3_manga": "manga.txt",
    "call3_subtitle": "subtitle.txt",
    # NSFW(SOFT/HARD) 버블 타입 보강. manga 모드 + nsfw 토글 ON일 때만 manga 프롬프트
    # 끝에 추가로 주입한다(일반 작업에 노출 안 됨).
    "call3_manga_nsfw": "manga_nsfw.txt",
    "multi_char_mask": "multi_char_mask.txt",
}

# 삽화 출력 모드. UI의 단일 드롭박스 원천이며 merged_toggles가 아래 두 불린으로
# 전개해 server.py 기존 분기 로직이 그대로 동작하게 한다.
#   illustration    -> 일반 삽화(OUTFIT/CALL2/CALL3 + Comfy)만
#   original_asset  -> 업로드 원본 에셋 선택만
#   both            -> 일반 삽화 + 원본 에셋 병행
ILLUSTRATION_OUTPUT_MODES = ("illustration", "original_asset", "both")
_ILLUSTRATION_OUTPUT_MODE_SET = frozenset(ILLUSTRATION_OUTPUT_MODES)
DEFAULT_ILLUSTRATION_OUTPUT_MODE = "illustration"


def _normalize_illustration_output_mode(
    value: object,
    *,
    legacy_booleans: tuple[bool, bool],
) -> str:
    """정규화된 출력 모드를 반환. 잘못된 값/과거 불륀 저장값은 안전하게 추론한다."""
    raw = str(value or "").strip().lower()
    if raw in _ILLUSTRATION_OUTPUT_MODE_SET:
        return raw
    if value not in (None, ""):
        print(
            f"[ILLUST_CONTEXT] 지원하지 않는 출력 모드 {raw!r}, "
            f"기존 토글 상태에서 추론"
        )
    illustration_enabled, original_asset_enabled = legacy_booleans
    if illustration_enabled and original_asset_enabled:
        return "both"
    if original_asset_enabled:
        return "original_asset"
    return "illustration"


DEFAULT_TOGGLES = {
    "illustration_output_mode": DEFAULT_ILLUSTRATION_OUTPUT_MODE,
    "original_asset_count": 1,
    "original_asset_instruction": "",
    "outfit_enabled": True,
    "time_place_enabled": True,
    "call2_plan_enabled": True,
    "call1_parallel_enabled": True,
    "call1_parallel_max_concurrency": 3,
    "call1_parallel_slow_retry_enabled": False,
    "call1_parallel_slow_retry_remaining": 1,
    "call1_parallel_slow_retry_progress_enabled": True,
    "call1_parallel_slow_retry_progress_threshold": 50,
    "call1_parallel_slow_retry_tps_enabled": False,
    "call1_parallel_slow_retry_tps_threshold": 5.0,
    "call1_parallel_slow_retry_condition_operator": "and",
    "call1_context_turns": 5,
    "call2_context_turns": 5,
    "call2_parallel_enabled": True,
    "call2_parallel_max_concurrency": 3,
    "call2_parallel_slow_retry_enabled": False,
    "call2_parallel_slow_retry_remaining": 1,
    "call2_parallel_slow_retry_progress_enabled": True,
    "call2_parallel_slow_retry_progress_threshold": 50,
    "call2_parallel_slow_retry_tps_enabled": False,
    "call2_parallel_slow_retry_tps_threshold": 5.0,
    "call2_parallel_slow_retry_condition_operator": "and",
    "call2_detail_parallel_enabled": True,
    "call2_detail_parallel_max_concurrency": 3,
    "call2_detail_parallel_slow_retry_enabled": False,
    "call2_detail_parallel_slow_retry_remaining": 1,
    "call2_detail_parallel_slow_retry_progress_enabled": True,
    "call2_detail_parallel_slow_retry_progress_threshold": 50,
    "call2_detail_parallel_slow_retry_tps_enabled": False,
    "call2_detail_parallel_slow_retry_tps_threshold": 5.0,
    "call2_detail_parallel_slow_retry_condition_operator": "and",
    "call3_context_turns": 5,
    "multi_char_mask_enabled": True,
    "multi_char_mask_inline_enabled": False,
    "single_char_mask_enabled": True,
    "single_char_mask_skip_enabled": True,
    "single_char_mask_skip_area_percent": 30.0,
    "speak_enabled": True,
    "call3_prompt_mode": "speak",
    "speak_language": "한국어",
    "speak_emotion_enabled": False,
    "speak_emotions": "",
    "nsfw": False,
    "supplement": True,
    "key_visual": True,
    "keyvis_in_detail_enabled": False,
    "minimal_background_description": True,
    "character_limit": 3,
    # scene_mode: 두 모드 모두 output_count_min/max를 강제한다. "auto"는 그 범위
    # 안에서 문맥에 맞는 수량을 고르고, "manual"은 지정 범위 자체만 전달한다.
    "scene_mode": "manual",
    # output_count_min/max: 삽화 총 장면 수의 유일한 소스. PLAN은 이 총량을,
    # 병렬 Call2-detail worker는 (총량÷worker수)를 받아 3배 과잉 생성을 방지한다.
    "output_count_min": 15,
    "output_count_max": 17,
    "context_history": True,
    "focus": "",
    "direction": "",
    "prompt_format": "v3",
    "positive_note": "",
    "negative_note": "",
    "compat_comfy": True,
    "compat_character_prompt": "separate",
}


# 삽화 장면 수의 유일한 카운트 규칙. presets.json(봇 시스템 프롬프트)에 하드코딩되던
# 블록을 분리해 편집 불가능한 고정 프롬프트로 중앙화했다. {min}/{max}는 서버가
# output_count_min/max 토글(PLAN은 총량, 병렬 Call2-detail worker는 총량÷worker수)로
# 치환해 주입한다. 이 템플릿 자체는 사용자가 직접 편집하지 않는다.
OUTPUT_COUNT_RULE_TEMPLATE = """## Output Count Rule
Each response MUST contain a minimum of {min} and a maximum of {max} image tags.
This is a hard constraint, not a suggestion.
- Use materially different, directly supported visible moments. When one sustained event must supply several images, vary its supported action, pose, reaction, contact, or spatial relationship instead of using invisible states or unrelated filler.
- If the scene contains more than {max} potential visual moments, select the {max} most impactful ones.
- Never output fewer than {min} images or more than {max} images."""


def render_output_count_rule(min_value: int, max_value: int) -> str:
    """output_count_min/max 값을 카운트 규칙 템플릿에 치환해 반환한다."""
    return OUTPUT_COUNT_RULE_TEMPLATE.replace("{min}", str(int(min_value))).replace(
        "{max}", str(int(max_value))
    )


def render_plan_output_count_rule(toggles: dict, available_slot_count: int) -> str:
    """Render the PLAN count contract; auto chooses only inside the hard range."""
    available = max(0, int(available_slot_count))
    minimum = min(int(toggles["output_count_min"]), available)
    maximum = min(int(toggles["output_count_max"]), available)
    rule = render_output_count_rule(minimum, maximum)
    if str(toggles.get("scene_mode")) == "auto":
        rule += (
            "\nWithin this hard range, choose the count that best represents the "
            f"narrative across the {available} available slots."
        )
    return rule

_SESSIONS: dict[str, dict] = {}
_LOOKUP_KEYS: dict[str, str] = {}
_LOOKUP_KEY_RE = re.compile(r"[0-9a-f]{24}")
_CANONICAL_SESSION_RE = re.compile(r"risu_([0-9a-f]{64})")


def session_lookup_key(session_id: str) -> str:
    """Return the 24-hex URL lookup key for canonical Risu illustration sessions."""
    match = _CANONICAL_SESSION_RE.fullmatch(str(session_id or "").lower())
    return match.group(1)[:24] if match else ""


def _persisted_lookup_matches(lookup_key: str) -> list[str]:
    if not _LOOKUP_KEY_RE.fullmatch(str(lookup_key or "")) or not os.path.isdir(SESSION_DIR):
        return []
    prefix = f"risu_{lookup_key}"
    suffix = ".json"
    matches = []
    try:
        for name in os.listdir(SESSION_DIR):
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            session_id = name[:-len(suffix)]
            if session_lookup_key(session_id) == lookup_key:
                matches.append(session_id)
    except Exception as e:
        print(f"[ILLUST_CONTEXT] lookup metadata scan failed: key={lookup_key}, error={e}")
        traceback.print_exc()
        raise
    return matches


def _register_lookup_key(session_id: str, lookup_key: str = "") -> str:
    key = str(lookup_key or session_lookup_key(session_id)).lower()
    if not key:
        return ""
    if not _LOOKUP_KEY_RE.fullmatch(key):
        raise ValueError(f"invalid illustration lookup key: {key!r}")

    existing = _LOOKUP_KEYS.get(key)
    if existing and existing != session_id:
        raise ValueError(
            f"illustration lookup key collision: key={key}, "
            f"existing={existing}, incoming={session_id}"
        )

    persisted = [value for value in _persisted_lookup_matches(key) if value != session_id]
    if persisted:
        raise ValueError(
            f"persisted illustration lookup key collision: key={key}, "
            f"existing={persisted}, incoming={session_id}"
        )

    _LOOKUP_KEYS[key] = session_id
    return key


def _session_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, f"{session_id}.json")


def _persist_session_metadata(session: dict) -> None:
    """재생성에 필요한 slot/RAW descriptor만 저장한다. 이미지 bytes는 저장하지 않는다."""
    try:
        os.makedirs(SESSION_DIR, exist_ok=True)
        data = {
            "session_id": session.get("session_id", ""),
            "lookup_key": session.get("lookup_key", ""),
            "status": session.get("status", "ready"),
            "context": session.get("context", ""),
            "items": session.get("items") or [],
            "requested_count": session.get("requested_count", len(session.get("items") or [])),
            "success_count": session.get("success_count", len(session.get("items") or [])),
            "failure_count": session.get("failure_count", 0),
            "failures": session.get("failures") or [],
            "progress": session.get("progress") or {},
            "error": session.get("error", ""),
            "created_at": session.get("created_at", time.time()),
            "updated_at": session.get("updated_at", time.time()),
        }
        with open(_session_path(str(data["session_id"])), "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 세션 metadata 저장 실패: {e}")
        traceback.print_exc()


def _load_session_metadata(session_id: str) -> dict | None:
    path = _session_path(session_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("session_id") != session_id:
            print(f"[ILLUST_CONTEXT] 저장 세션 형식 불일치: {path}")
            return None
        items = data.get("items") or []
        data["items"] = items
        data["images"] = [None] * len(items)
        data["status"] = "ready"
        data["requested_count"] = max(len(items), int(data.get("requested_count") or len(items)))
        data["success_count"] = len(items)
        data["failures"] = data.get("failures") or []
        data["failure_count"] = max(
            len(data["failures"]),
            int(data.get("failure_count") or 0),
            data["requested_count"] - data["success_count"],
        )
        data["lookup_key"] = _register_lookup_key(
            session_id,
            str(data.get("lookup_key") or ""),
        )
        _SESSIONS[session_id] = data
        print(f"[ILLUST_CONTEXT] 재생성 세션 metadata 복원: session={session_id}, items={len(items)}")
        return data
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 세션 metadata 로드 실패: {path}: {e}")
        traceback.print_exc()
        return None


def merged_toggles(value: dict | None) -> dict:
    out = deepcopy(DEFAULT_TOGGLES)
    if isinstance(value, dict):
        for key in out:
            if key in value:
                out[key] = value[key]
        # Existing shared settings keep their effective DETAIL batch count until
        # an independent DETAIL value is saved. Never overwrite an explicit value.
        for suffix in ("enabled", "max_concurrency"):
            shared_key = f"call2_parallel_{suffix}"
            detail_key = f"call2_detail_parallel_{suffix}"
            if detail_key not in value and shared_key in value:
                out[detail_key] = value[shared_key]
                print(f"[ILLUST_CONTEXT:CALL2] 기존 공통 설정을 DETAIL 초기값으로 유지: {detail_key}={out[detail_key]!r}")
        # Older retired switches do not override the independent stage options.
        for key in ("call1_enabled", "profile_resolve_enabled"):
            if key in value:
                print(f"[ILLUST_CONTEXT] 폐기된 필수 단계 토글 무시: {key}={value[key]!r}")
        for key in value:
            if key.startswith("call1_backtranslate_"):
                print(f"[ILLUST_CONTEXT] 제거된 역번역 설정 무시: {key}={value[key]!r}")
        if "compat_character_divider" in value:
            print(
                f"[ILLUST_CONTEXT] 폐기된 캐릭터 구분 설정 무시: "
                f"compat_character_divider={value['compat_character_divider']!r}, "
                "RAW 캐릭터 블록은 파이프로 구분"
            )
        # Fold the old two-switch dialogue gate into the existing single key.
        # Do not emit call3_enabled, so normalization is idempotent and the
        # remaining switch can be turned back on without a hidden veto.
        if "call3_enabled" in value:
            out["speak_enabled"] = bool(out["speak_enabled"]) and bool(value["call3_enabled"])
            print(
                f"[ILLUST_CONTEXT:CALL3] 기존 대사 토글 통합: "
                f"call3_enabled={value['call3_enabled']!r}, speak_enabled={out['speak_enabled']}"
            )
        # 구버전의 자유 입력 preset 값은 V1/V3 선택값으로 한 번만 해석한다.
        # tutorial/default/빈 값은 모두 기존 preset.txt(V3)를 뜻했다.
        if "prompt_format" not in value and "preset" in value:
            legacy_preset = str(value.get("preset") or "").strip().lower()
            if legacy_preset == "v1":
                out["prompt_format"] = "v1"
            elif legacy_preset in ("", "default", "tutorial", "v3"):
                out["prompt_format"] = "v3"
            else:
                print(
                    f"[ILLUST_CONTEXT] 알 수 없는 기존 RAW 프롬프트 프리셋 "
                    f"{legacy_preset!r}, V3로 전환"
                )
                out["prompt_format"] = "v3"
    for key in ("outfit_enabled", "time_place_enabled", "call2_plan_enabled", "call2_parallel_enabled",
                "call2_detail_parallel_enabled", "multi_char_mask_inline_enabled", "keyvis_in_detail_enabled",
                "single_char_mask_skip_enabled"):
        out[key] = _bool(out[key])
    try:
        area_percent = float(out["single_char_mask_skip_area_percent"])
        if not math.isfinite(area_percent):
            raise ValueError(f"유한하지 않은 면적 비율: {area_percent!r}")
        out["single_char_mask_skip_area_percent"] = max(0.0, min(100.0, area_percent))
        if out["single_char_mask_skip_area_percent"] != area_percent:
            print(f"[ILLUST_CONTEXT] 마스크 생략 면적을 0~100%로 보정: {area_percent!r}")
    except (TypeError, ValueError, OverflowError) as exc:
        print(f"[ILLUST_CONTEXT] 마스크 생략 면적 값 오류, 기본값 사용: "
              f"value={out['single_char_mask_skip_area_percent']!r}, error={exc}")
        traceback.print_exc()
        out["single_char_mask_skip_area_percent"] = DEFAULT_TOGGLES["single_char_mask_skip_area_percent"]
    prompt_format = str(out.get("prompt_format") or "").strip().lower()
    if prompt_format not in ("v1", "v3", "chansub"):
        print(f"[ILLUST_CONTEXT] 지원하지 않는 프롬프트 입력 형식 {prompt_format!r}, V3 사용")
        prompt_format = "v3"
    out["prompt_format"] = prompt_format
    call3_prompt_mode = str(out.get("call3_prompt_mode") or "").strip().lower()
    if call3_prompt_mode not in ("speak", "manga", "subtitle"):
        print(
            f"[ILLUST_CONTEXT] 지원하지 않는 CALL3 대사 프롬프트 "
            f"{call3_prompt_mode!r}, speak 사용"
        )
        call3_prompt_mode = "speak"
    out["call3_prompt_mode"] = call3_prompt_mode
    for prefix, label in (
        ("call1_parallel", "OUTFIT 병렬"),
        ("call2_parallel", "CALL2 병렬"),
        ("call2_detail_parallel", "DETAIL 병렬"),
    ):
        operator_key = f"{prefix}_slow_retry_condition_operator"
        condition_operator = str(out.get(operator_key) or "").strip().lower()
        if condition_operator not in ("and", "or"):
            print(
                f"[ILLUST_CONTEXT:{label}] 지원하지 않는 느린 요청 조건 결합 방식 "
                f"{condition_operator!r}, and 사용"
            )
            condition_operator = "and"
        out[operator_key] = condition_operator
    try:
        for prefix in ("call1_parallel", "call2_parallel"):
            out[f"{prefix}_max_concurrency"] = max(
                1,
                min(16, int(out[f"{prefix}_max_concurrency"])),
            )
            out[f"{prefix}_slow_retry_remaining"] = max(
                1,
                min(16, int(out[f"{prefix}_slow_retry_remaining"])),
            )
            out[f"{prefix}_slow_retry_progress_threshold"] = max(
                1,
                min(99, int(out[f"{prefix}_slow_retry_progress_threshold"])),
            )
            out[f"{prefix}_slow_retry_tps_threshold"] = max(
                0.1,
                min(1000.0, float(out[f"{prefix}_slow_retry_tps_threshold"])),
            )
        out["call1_context_turns"] = max(0, min(30, int(out["call1_context_turns"])))
        # call2/call3 전용 키가 없거나 무효하면 call1 값으로 폴백(하위호환).
        for _ck in ("call2_context_turns", "call3_context_turns"):
            _raw = value.get(_ck) if isinstance(value, dict) else None
            try:
                out[_ck] = max(0, min(30, int(_raw)))
            except (TypeError, ValueError):
                print(f"[ILLUST_CONTEXT] {_ck} 값 무효({_raw!r}), call1_context_turns로 폴백")
                out[_ck] = out["call1_context_turns"]
        out["character_limit"] = max(1, min(3, int(out["character_limit"])))
        out["scene_mode"] = "auto" if str(out.get("scene_mode")) == "auto" else "manual"
        out["output_count_min"] = max(1, min(30, int(out["output_count_min"])))
        out["output_count_max"] = max(1, min(30, int(out["output_count_max"])))
        out["original_asset_count"] = max(
            1,
            min(30, int(out["original_asset_count"])),
        )
        if out["output_count_min"] > out["output_count_max"]:
            print(
                f"[ILLUST_CONTEXT] output_count_min({out['output_count_min']}) > "
                f"output_count_max({out['output_count_max']}), min을 max로 보정"
            )
            out["output_count_min"] = out["output_count_max"]
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 토글 숫자 보정 실패: {e}")
        traceback.print_exc()
        out.update({
            "call1_parallel_max_concurrency": DEFAULT_TOGGLES[
                "call1_parallel_max_concurrency"
            ],
            "call1_parallel_slow_retry_remaining": DEFAULT_TOGGLES[
                "call1_parallel_slow_retry_remaining"
            ],
            "call1_parallel_slow_retry_progress_threshold": DEFAULT_TOGGLES[
                "call1_parallel_slow_retry_progress_threshold"
            ],
            "call1_parallel_slow_retry_tps_threshold": DEFAULT_TOGGLES[
                "call1_parallel_slow_retry_tps_threshold"
            ],
            "call2_parallel_max_concurrency": DEFAULT_TOGGLES[
                "call2_parallel_max_concurrency"
            ],
            "call2_parallel_slow_retry_remaining": DEFAULT_TOGGLES[
                "call2_parallel_slow_retry_remaining"
            ],
            "call2_parallel_slow_retry_progress_threshold": DEFAULT_TOGGLES[
                "call2_parallel_slow_retry_progress_threshold"
            ],
            "call2_parallel_slow_retry_tps_threshold": DEFAULT_TOGGLES[
                "call2_parallel_slow_retry_tps_threshold"
            ],
            "call1_context_turns": DEFAULT_TOGGLES["call1_context_turns"],
            "call2_context_turns": DEFAULT_TOGGLES["call2_context_turns"],
            "call3_context_turns": DEFAULT_TOGGLES["call3_context_turns"],
            "character_limit": DEFAULT_TOGGLES["character_limit"],
            "scene_mode": DEFAULT_TOGGLES["scene_mode"],
            "output_count_min": DEFAULT_TOGGLES["output_count_min"],
            "output_count_max": DEFAULT_TOGGLES["output_count_max"],
            "original_asset_count": DEFAULT_TOGGLES["original_asset_count"],
        })
    try:
        out["call2_detail_parallel_max_concurrency"] = max(
            1, min(16, int(out["call2_detail_parallel_max_concurrency"])))
    except (TypeError, ValueError, OverflowError) as e:
        print(f"[ILLUST_CONTEXT:DETAIL] 병렬 개수 보정 실패: value={out['call2_detail_parallel_max_concurrency']!r}, error={e}")
        traceback.print_exc()
        out["call2_detail_parallel_max_concurrency"] = DEFAULT_TOGGLES["call2_detail_parallel_max_concurrency"]
    for suffix, cast, minimum, maximum in (
        ("remaining", int, 1, 16),
        ("progress_threshold", int, 1, 99),
        ("tps_threshold", float, 0.1, 1000.0),
    ):
        key = f"call2_detail_parallel_slow_retry_{suffix}"
        try:
            out[key] = max(minimum, min(maximum, cast(out[key])))
        except (TypeError, ValueError, OverflowError) as e:
            print(f"[ILLUST_CONTEXT:DETAIL] 느린 요청 조건 보정 실패: key={key}, value={out[key]!r}, error={e}")
            traceback.print_exc()
            out[key] = DEFAULT_TOGGLES[key]
    # 출력 모드 단일 드롭박스 -> illustration_enabled/original_asset_enabled 전개.
    # 입력에 모드가 없으면(구버전 저장값) 두 불린 조합에서 모드를 추론한다.
    raw_output_mode = (
        value.get("illustration_output_mode")
        if isinstance(value, dict) else None
    )
    legacy_illust = bool(
        value.get("illustration_enabled", True)
    ) if isinstance(value, dict) else True
    legacy_asset = bool(
        value.get("original_asset_enabled", False)
    ) if isinstance(value, dict) else False
    output_mode = _normalize_illustration_output_mode(
        raw_output_mode,
        legacy_booleans=(legacy_illust, legacy_asset),
    )
    out["illustration_output_mode"] = output_mode
    out["illustration_enabled"] = output_mode in ("illustration", "both")
    out["original_asset_enabled"] = output_mode in ("original_asset", "both")
    instruction = str(out.get("original_asset_instruction") or "")
    if len(instruction) > illustration_original_assets.MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS:
        print(
            f"[ILLUST_ORIGINAL_ASSET] 설정 지시문 길이 제한 적용: "
            f"chars={len(instruction)}, "
            f"max={illustration_original_assets.MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS}"
        )
        instruction = instruction[
            :illustration_original_assets.MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS
        ]
    out["original_asset_instruction"] = instruction
    # 예전 UI에서 저장한 고정 배치 크기는 더 이상 사용하지 않는다. CALL2-PLAN이
    # 선택한 전체 장면 수를 최대 동시 요청 수에 맞춰 자동 분배한다.
    out.pop("call2_parallel_batch_size", None)
    # OUTFIT은 하나의 누적 상태 타임라인을 위한 단일 호출이다.
    # 예전 segment 청크 크기 설정은 사용하지 않는다.
    out.pop("call1_parallel_chunk_size", None)
    return out


def should_enable_multi_char_layout(toggles: dict | None, provider: str) -> bool:
    """현재 생성 설정에서 Regional 다중 캐릭터 마스크를 사용할지 반환한다."""
    normalized = merged_toggles(toggles)
    return (
        bool(normalized.get("multi_char_mask_enabled", True))
        and str(provider or "").strip().lower() == "comfy"
        and str(normalized.get("prompt_format") or "v3").strip().lower() == "v3"
    )


def load_prompt_files() -> dict:
    result = {}
    for key, filename in PROMPT_FILES.items():
        path = os.path.join(PROMPTS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                result[key] = f.read()
        except Exception as e:
            print(f"[ILLUST_CONTEXT] 프롬프트 로드 실패: {path}: {e}")
            traceback.print_exc()
            result[key] = ""
    return result


def save_prompt_files(values: dict) -> list[str]:
    """UI 편집본 저장. 기존 텍스트는 배포 가능한 backups/에 먼저 백업한다."""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(PROMPT_BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved = []
    for key, filename in PROMPT_FILES.items():
        if key not in values:
            continue
        path = os.path.join(PROMPTS_DIR, filename)
        try:
            if os.path.exists(path):
                backup = os.path.join(
                    PROMPT_BACKUP_DIR,
                    f"lighbd_{filename}.{stamp}.bak",
                )
                with open(path, "r", encoding="utf-8") as src:
                    old = src.read()
                with open(backup, "w", encoding="utf-8") as dst:
                    dst.write(old)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(str(values[key]))
            saved.append(key)
        except Exception as e:
            print(f"[ILLUST_CONTEXT] 프롬프트 저장 실패: {path}: {e}")
            traceback.print_exc()
            raise
    return saved


def _json_after_prefix(positive: str, prefix: str) -> dict | None:
    if not isinstance(positive, str) or not positive.lstrip().startswith(prefix):
        return None
    raw = positive.lstrip()[len(prefix):].lstrip("\r\n ")
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            print(f"[ILLUST_CONTEXT] {prefix} payload가 object가 아님")
            return None
        return value
    except Exception as e:
        print(f"[ILLUST_CONTEXT] {prefix} JSON 파싱 실패: {e}; raw={raw!r}")
        traceback.print_exc()
        return None


def parse_context_request(positive: str) -> dict | None:
    payload = _json_after_prefix(positive, CONTEXT_PREFIX)
    if payload is None:
        return None
    session_id = str(payload.get("session_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", session_id):
        print(f"[ILLUST_CONTEXT] 잘못된 session_id: {session_id!r}")
        return None
    chats = []
    for item in payload.get("chats") or []:
        if not isinstance(item, dict):
            continue
        data = str(item.get("data") or item.get("content") or "").strip()
        if not data:
            continue
        role = str(item.get("role") or "user").lower()
        role = "char" if role in ("char", "assistant") else "user"
        chats.append({"role": role, "data": data})
    if not chats:
        print(f"[ILLUST_CONTEXT] CHAT 데이터가 비어 있음: session={session_id}")
        return None
    action = str(payload.get("action") or "regenerate").strip().lower()
    if action not in ("regenerate", "generate", "result", "asset_reroll"):
        print(f"[ILLUST_CONTEXT] 지원하지 않는 CONTEXT action: session={session_id}, action={action!r}")
        return None
    slot = None
    if action in ("generate", "result"):
        try:
            slot = int(payload.get("slot"))
        except Exception as e:
            print(
                f"[ILLUST_CONTEXT] CONTEXT {action} slot 파싱 실패: "
                f"session={session_id}, error={e}, payload={payload}"
            )
            return None
    target_slotted = str(payload.get("target_slotted") or "").strip()
    if target_slotted and not re.search(r"\[Slot\s+\d+\]", target_slotted):
        print(f"[ILLUST_CONTEXT] target_slotted에 슬롯 마커가 없어 폴백 사용: session={session_id}")
        target_slotted = ""
    payload["session_id"] = session_id
    payload["chats"] = chats
    payload["target_slotted"] = target_slotted
    payload["action"] = action
    payload["slot"] = slot
    return payload


def parse_result_request(positive: str) -> dict | None:
    payload = _json_after_prefix(positive, RESULT_PREFIX)
    if payload is None:
        return None
    session_id = str(payload.get("session_id") or "")
    try:
        index = int(payload.get("index")) if payload.get("index") is not None else None
        slot = int(payload.get("slot")) if payload.get("slot") is not None else None
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 결과 index 파싱 실패: {e}; payload={payload}")
        traceback.print_exc()
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", session_id) or (index is None and slot is None) or (index is not None and index < 1):
        print(f"[ILLUST_CONTEXT] 잘못된 결과 요청: session={session_id!r}, index={index}, slot={slot}")
        return None
    return {"session_id": session_id, "index": index, "slot": slot}


def parse_prompt_batch_request(positive: str) -> dict | None:
    """원본 LightBoard 콜백이 확정한 슬롯/프롬프트 배치를 검증한다.

    이 경로는 CHAT, 캐릭터 정보, 이미지 bytes를 플러그인과 주고받지 않는다.
    Risu Lua가 이미 만든 최종 이미지 프롬프트만 generation 채널로 받는다.
    """
    payload = _json_after_prefix(positive, PROMPT_BATCH_PREFIX)
    if payload is None:
        return None

    session_id = str(payload.get("session_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", session_id):
        print(f"[ILLUST_PROMPT_BATCH] 잘못된 session_id: {session_id!r}")
        return None

    raw_items = payload.get("items")
    if (
        not isinstance(raw_items, list)
        or not 1 <= len(raw_items) <= MAX_ILLUSTRATION_SLOT_COUNT
    ):
        count = len(raw_items) if isinstance(raw_items, list) else -1
        print(
            f"[ILLUST_PROMPT_BATCH] 잘못된 items 개수: "
            f"session={session_id}, count={count}, "
            f"max={MAX_ILLUSTRATION_SLOT_COUNT}"
        )
        return None

    items = []
    seen_slots: set[int] = set()
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            print(
                f"[ILLUST_PROMPT_BATCH] item 형식 오류: "
                f"session={session_id}, index={index}, type={type(raw_item).__name__}"
            )
            return None
        try:
            slot = int(raw_item.get("slot"))
        except Exception as e:
            print(
                f"[ILLUST_PROMPT_BATCH] slot 파싱 실패: "
                f"session={session_id}, index={index}, error={e}, item={raw_item}"
            )
            traceback.print_exc()
            return None
        if slot < -1 or slot in seen_slots:
            print(
                f"[ILLUST_PROMPT_BATCH] slot 범위/중복 오류: "
                f"session={session_id}, index={index}, slot={slot}"
            )
            return None

        raw_positive = str(raw_item.get("positive") or "").strip()
        raw_negative = str(raw_item.get("negative") or "").strip()
        if not raw_positive:
            print(
                f"[ILLUST_PROMPT_BATCH] positive 비어 있음: "
                f"session={session_id}, index={index}, slot={slot}"
            )
            return None
        if len(raw_positive) > 200_000 or len(raw_negative) > 200_000:
            print(
                f"[ILLUST_PROMPT_BATCH] 프롬프트 길이 초과: "
                f"session={session_id}, index={index}, slot={slot}, "
                f"positive={len(raw_positive)}, negative={len(raw_negative)}"
            )
            return None

        seen_slots.add(slot)
        items.append({
            "kind": "keyvis" if slot == -1 else "scene",
            "slot": slot,
            "raw_positive": raw_positive,
            "raw_negative": raw_negative,
            "source": "risu_module_prompt_batch_v1",
        })

    return {
        "protocol": "prompt_batch_v1",
        "session_id": session_id,
        "items": items,
    }


def parse_regenerate_request(positive: str) -> dict | None:
    payload = _json_after_prefix(positive, REGENERATE_PREFIX)
    if payload is None:
        return None
    session_id = str(payload.get("session_id") or "")
    try:
        slot = int(payload.get("slot"))
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 재생성 slot 파싱 실패: {e}; payload={payload}")
        traceback.print_exc()
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", session_id):
        print(f"[ILLUST_CONTEXT] 잘못된 재생성 session_id: {session_id!r}")
        return None
    return {"session_id": session_id, "slot": slot}


def parse_easy_edit_request(positive: str) -> dict | None:
    payload = _json_after_prefix(positive, EASY_EDIT_PREFIX)
    if payload is None:
        return None
    session_id = str(payload.get("session_id") or "")
    try:
        slot = int(payload.get("slot"))
    except Exception as e:
        print(f"[ILLUST_CONTEXT:EDIT] slot 파싱 실패: {e}; payload={payload}")
        traceback.print_exc()
        return None
    direction = str(payload.get("direction") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", session_id):
        print(f"[ILLUST_CONTEXT:EDIT] 잘못된 session_id: {session_id!r}")
        return None
    if slot < -1:
        print(f"[ILLUST_CONTEXT:EDIT] 잘못된 slot: {slot}")
        return None
    if not direction or len(direction) > MAX_EASY_EDIT_DIRECTION_LENGTH:
        print(
            f"[ILLUST_CONTEXT:EDIT] direction 길이 오류: "
            f"session={session_id}, slot={slot}, length={len(direction)}"
        )
        return None
    return {"session_id": session_id, "slot": slot, "direction": direction}


def create_session(session_id: str, context: str) -> dict:
    lookup_key = _register_lookup_key(session_id)
    session = {
        "session_id": session_id,
        "lookup_key": lookup_key,
        "status": "building",
        "context": context,
        "items": [],
        "images": [],
        "requested_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "failures": [],
        "progress": {
            "phase": "queued",
            "label": "서버 작업 대기",
            "value": 0,
            "done": 0,
            "total": 0,
        },
        "error": "",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _SESSIONS[session_id] = session
    return session


def set_session_progress(
    session_id: str,
    phase: str,
    label: str,
    value: float = 0,
    done: int = 0,
    total: int = 0,
) -> None:
    """브리지에 노출할 비민감 진행 상태를 현재 세션에 기록한다."""
    session = _SESSIONS.get(session_id)
    if session is None:
        print(
            f"[ILLUST_CONTEXT] 진행 상태 저장 실패 - 세션 없음: "
            f"session={session_id}, phase={phase!r}, label={label!r}"
        )
        return
    try:
        numeric_value = max(0.0, min(100.0, float(value)))
        numeric_done = max(0, int(done))
        numeric_total = max(0, int(total))
        if numeric_total and numeric_done > numeric_total:
            numeric_done = numeric_total
        session["progress"] = {
            "phase": re.sub(r"[^a-z0-9_-]", "_", str(phase or "building").lower())[:32],
            "label": str(label or "처리 중").replace("\r", " ").replace("\n", " ")[:160],
            "value": round(numeric_value, 1),
            "done": numeric_done,
            "total": numeric_total,
        }
        session["updated_at"] = time.time()
    except Exception as e:
        print(
            f"[ILLUST_CONTEXT] 진행 상태 보정 실패: session={session_id}, "
            f"phase={phase!r}, label={label!r}, error={e}"
        )
        traceback.print_exc()


def set_session_regenerate_started(
    session_id: str,
    slot: int,
    operation_label: str = "재생성",
    *,
    whole_session: bool = False,
) -> None:
    """기존 결과를 유지한 채 특정 슬롯의 서버 생성 진행상황을 노출한다."""
    session = get_session(session_id)
    if session is None:
        print(
            f"[ILLUST_CONTEXT] 재생성 진행 시작 실패 - 세션 없음: "
            f"session={session_id}, slot={slot}"
        )
        return
    safe_operation_label = (
        str(operation_label or "재생성")
        .replace("\r", " ")
        .replace("\n", " ")[:40]
    )
    slot_label = "키비주얼" if slot == -1 else f"슬롯 {slot}"
    label = f"{slot_label} 서버 {safe_operation_label} 중"
    value = 0
    done = 0
    total = 1
    session.pop("_active_slot_progress", None)

    if whole_session:
        items = session.get("items") or []
        current = next((
            index
            for index, item in enumerate(items, start=1)
            if isinstance(item, dict) and item.get("slot") == slot
        ), 0)
        if current > 0:
            total = len(items)
            done = current - 1
            value = (done / total) * 100
            label = (
                f"전체 {total}장 중 {current}장째 · "
                f"{slot_label} 서버 {safe_operation_label} 중"
            )
            session["_active_slot_progress"] = {
                "slot": slot,
                "current": current,
                "total": total,
                "operation_label": safe_operation_label,
            }
        else:
            print(
                f"[ILLUST_CONTEXT] 전체 생성 슬롯 순번 계산 실패: "
                f"session={session_id}, slot={slot}, items={len(items)}"
            )
    session["error"] = ""
    set_session_progress(
        session_id,
        "regenerating",
        label,
        value,
        done,
        total,
    )


def set_session_regenerate_error(session_id: str, slot: int, error: str) -> None:
    """재생성 실패를 표시하되 기존 준비된 이미지와 ready 상태는 보존한다."""
    session = get_session(session_id)
    if session is None:
        print(
            f"[ILLUST_CONTEXT] 재생성 오류 저장 실패 - 세션 없음: "
            f"session={session_id}, slot={slot}, error={error}"
        )
        return
    try:
        session.pop("_active_slot_progress", None)
        safe_error = str(error or "재생성 실패").replace("\r", " ").replace("\n", " ")[:300]
        session["error"] = safe_error
        session["progress"] = {
            "phase": "error",
            "label": f"슬롯 {slot} 재생성 실패: {safe_error}"[:160],
            "value": 0,
            "done": 0,
            "total": 1,
        }
        session["updated_at"] = time.time()
    except Exception as e:
        print(
            f"[ILLUST_CONTEXT] 재생성 오류 보정 실패: "
            f"session={session_id}, slot={slot}, error={e}"
        )
        traceback.print_exc()


def get_session(session_id: str) -> dict | None:
    session = _SESSIONS.get(session_id)
    if session is None:
        session = _load_session_metadata(session_id)
    if session is None:
        print(f"[ILLUST_CONTEXT] 세션 캐시 미스: {session_id}")
    return session


def recent_session_summaries(limit: int = 20) -> list[dict]:
    """플러그인 상태 화면용 비민감 세션 요약을 반환한다."""
    try:
        safe_limit = max(1, min(50, int(limit)))
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 세션 요약 limit 보정 실패: limit={limit!r}, error={e}")
        traceback.print_exc()
        safe_limit = 20

    summaries = []
    sessions = sorted(
        _SESSIONS.values(),
        key=lambda session: float(session.get("updated_at") or 0),
        reverse=True,
    )
    for session in sessions[:safe_limit]:
        raw_progress = session.get("progress") or {}
        items = session.get("items") or []
        summaries.append({
            "session_id": str(session.get("session_id") or ""),
            "status": str(session.get("status") or "missing"),
            "error": str(session.get("error") or "")[:300],
            "progress": {
                "phase": str(raw_progress.get("phase") or "building")[:32],
                "label": str(raw_progress.get("label") or "처리 중")[:160],
                "value": raw_progress.get("value", 0),
                "done": raw_progress.get("done", 0),
                "total": raw_progress.get("total", 0),
            },
            "item_count": len(items),
            "requested_count": int(session.get("requested_count") or len(items)),
            "success_count": int(session.get("success_count") or len(items)),
            "failure_count": int(session.get("failure_count") or 0),
            "created_at": session.get("created_at", 0),
            "updated_at": session.get("updated_at", 0),
        })
    return summaries


def set_session_result(
    session_id: str,
    items: list,
    images: list[bytes],
    *,
    requested_count: int | None = None,
    failures: list[dict] | None = None,
) -> None:
    session = _SESSIONS.get(session_id)
    if session is None:
        print(f"[ILLUST_CONTEXT] 결과 저장 실패 - 세션 없음: {session_id}")
        return
    session["items"] = deepcopy(items)
    session["images"] = list(images)
    success_count = len(images)
    try:
        requested_count = max(success_count, int(requested_count or success_count))
    except Exception:
        requested_count = success_count
    safe_failures = []
    for failure in failures or []:
        if not isinstance(failure, dict):
            continue
        try:
            slot = int(failure.get("slot"))
        except Exception:
            slot = None
        safe_failures.append({
            "slot": slot,
            "error": str(failure.get("error") or "생성 실패")
            .replace("\r", " ")
            .replace("\n", " ")[:300],
        })
    failure_count = max(len(safe_failures), requested_count - success_count)
    partial = failure_count > 0
    session["requested_count"] = requested_count
    session["success_count"] = success_count
    session["failure_count"] = failure_count
    session["failures"] = safe_failures
    session["status"] = "ready"
    session["progress"] = {
        "phase": "ready_partial" if partial else "ready",
        "label": (
            f"성공 {success_count}/{requested_count}장 반환 준비 완료 · "
            f"최종 실패 {failure_count}장 제외"
            if partial
            else f"전체 {success_count}장 반환 준비 완료"
        ),
        "value": 100,
        "done": success_count,
        "total": requested_count,
    }
    session["updated_at"] = time.time()
    _persist_session_metadata(session)


def set_session_error(session_id: str, error: str) -> None:
    session = _SESSIONS.get(session_id)
    if session is None:
        print(f"[ILLUST_CONTEXT] 에러 저장 실패 - 세션 없음: {session_id}; error={error}")
        return
    session["status"] = "error"
    session["error"] = str(error)
    previous = session.get("progress") or {}
    session["progress"] = {
        "phase": "error",
        "label": str(error).replace("\r", " ").replace("\n", " ")[:160] or "처리 실패",
        "value": previous.get("value", 0),
        "done": previous.get("done", 0),
        "total": previous.get("total", 0),
    }
    session["updated_at"] = time.time()


def session_image(session_id: str, index: int) -> bytes | None:
    session = get_session(session_id)
    if not session or session.get("status") != "ready":
        print(f"[ILLUST_CONTEXT] 이미지 회수 실패 - 준비 안 됨: session={session_id}, index={index}")
        return None
    images = session.get("images") or []
    if index < 1 or index > len(images):
        print(f"[ILLUST_CONTEXT] 이미지 회수 범위 초과: session={session_id}, index={index}, count={len(images)}")
        return None
    image = images[index - 1]
    if image is None and index - 1 < len(session.get("items") or []):
        descriptor = session["items"][index - 1]
        if str(descriptor.get("kind") or "") == "original_asset":
            image = illustration_original_assets.load_original_asset_bytes(
                os.path.join(BASE_DIR, "bot"),
                descriptor,
            )
            if image:
                images[index - 1] = image
            else:
                print(
                    f"[ILLUST_CONTEXT] 원본 에셋 지연 회수 실패: "
                    f"session={session_id}, index={index}"
                )
        if image is None:
            try:
                slot = int(descriptor.get("slot"))
            except Exception as e:
                print(
                    f"[ILLUST_CONTEXT] 백업 이미지 인덱스 복원 실패 - 슬롯 파싱 오류: "
                    f"session={session_id}, index={index}, error={e}"
                )
                traceback.print_exc()
            else:
                image = _load_session_backup_image(session_id, slot, descriptor)
                if image:
                    images[index - 1] = image
    return image


def _load_session_backup_image(
    session_id: str,
    slot: int,
    item: dict,
) -> bytes | None:
    """지속 metadata의 backup_name으로 생성 이미지를 지연 복원한다."""
    backup_name = str(item.get("backup_name") or "").strip()
    if not backup_name:
        print(
            f"[ILLUST_CONTEXT] 백업 이미지 지연 복원 불가 - backup_name 없음: "
            f"session={session_id}, slot={slot}"
        )
        return None
    if _BACKUP_NAME_RE.fullmatch(backup_name) is None:
        print(
            f"[ILLUST_CONTEXT] 백업 이미지 지연 복원 거부 - 잘못된 backup_name: "
            f"session={session_id}, slot={slot}, backup={backup_name!r}"
        )
        return None

    candidates = [
        os.path.join(WORKFLOW_BACKUP_DIR, backup_name + extension)
        for extension in _BACKUP_IMAGE_EXTENSIONS
    ]
    image_path = next((path for path in candidates if os.path.isfile(path)), "")
    if not image_path:
        print(
            f"[ILLUST_CONTEXT] 백업 이미지 지연 복원 실패 - 파일 없음: "
            f"session={session_id}, slot={slot}, backup={backup_name!r}, "
            f"dir={WORKFLOW_BACKUP_DIR!r}"
        )
        return None

    try:
        with open(image_path, "rb") as f:
            image = f.read()
    except Exception as e:
        print(
            f"[ILLUST_CONTEXT] 백업 이미지 지연 복원 실패 - 읽기 오류: "
            f"session={session_id}, slot={slot}, path={image_path!r}, error={e}"
        )
        traceback.print_exc()
        return None
    if not image:
        print(
            f"[ILLUST_CONTEXT] 백업 이미지 지연 복원 실패 - 빈 파일: "
            f"session={session_id}, slot={slot}, path={image_path!r}"
        )
        return None

    print(
        f"[ILLUST_CONTEXT] 백업 이미지 지연 복원: "
        f"session={session_id}, slot={slot}, backup={backup_name!r}, "
        f"bytes={len(image)}"
    )
    return image


def session_image_by_slot(session_id: str, slot: int) -> bytes | None:
    session = get_session(session_id)
    if not session or session.get("status") != "ready":
        print(f"[ILLUST_CONTEXT] 슬롯 이미지 회수 실패 - 준비 안 됨: session={session_id}, slot={slot}")
        return None
    items = session.get("items") or []
    images = session.get("images") or []
    for index, item in enumerate(items):
        try:
            item_slot = int(item.get("slot"))
        except Exception:
            continue
        if item_slot == int(slot) and index < len(images):
            image = images[index]
            if image is None and str(item.get("kind") or "") == "original_asset":
                image = illustration_original_assets.load_original_asset_bytes(
                    os.path.join(BASE_DIR, "bot"),
                    item,
                )
                if image:
                    images[index] = image
                else:
                    print(
                        f"[ILLUST_CONTEXT] 원본 에셋 슬롯 지연 회수 실패: "
                        f"session={session_id}, slot={slot}"
                    )
            if image is None:
                image = _load_session_backup_image(session_id, slot, item)
                if image:
                    images[index] = image
            return image
    print(f"[ILLUST_CONTEXT] 슬롯 이미지 없음: session={session_id}, slot={slot}")
    return None


def session_item_by_slot(session_id: str, slot: int) -> dict | None:
    session = get_session(session_id)
    if not session or session.get("status") != "ready":
        print(f"[ILLUST_CONTEXT] 재생성 descriptor 회수 실패: session={session_id}, slot={slot}")
        return None
    for item in session.get("items") or []:
        try:
            if int(item.get("slot")) == int(slot):
                return deepcopy(item)
        except Exception:
            continue
    print(f"[ILLUST_CONTEXT] 재생성 descriptor 없음: session={session_id}, slot={slot}")
    return None


def update_session_image_by_slot(
    session_id: str,
    slot: int,
    image: bytes,
    *,
    item_updates: dict | None = None,
) -> bool:
    session = get_session(session_id)
    if not session or session.get("status") != "ready" or not image:
        print(f"[ILLUST_CONTEXT] 재생성 캐시 갱신 실패: session={session_id}, slot={slot}")
        return False
    items = session.get("items") or []
    images = session.get("images") or []
    for index, item in enumerate(items):
        try:
            if int(item.get("slot")) == int(slot) and index < len(images):
                images[index] = image
                if isinstance(item_updates, dict):
                    for key, value in item_updates.items():
                        if value is None:
                            item.pop(str(key), None)
                        else:
                            item[str(key)] = deepcopy(value)
                session["status"] = "ready"
                session["error"] = ""
                active_progress = session.pop("_active_slot_progress", None)
                if (
                    isinstance(active_progress, dict)
                    and active_progress.get("slot") == slot
                    and active_progress.get("current", 0) > 0
                    and active_progress.get("total", 0) > 0
                ):
                    current = int(active_progress["current"])
                    total = int(active_progress["total"])
                    operation_label = str(
                        active_progress.get("operation_label") or "전체 생성"
                    )
                    slot_label = "키비주얼" if slot == -1 else f"슬롯 {slot}"
                    session["progress"] = {
                        "phase": "ready",
                        "label": (
                            f"전체 {total}장 중 {current}장 완료 · "
                            f"{slot_label} 서버 {operation_label} 완료"
                        )[:160],
                        "value": round((current / total) * 100, 1),
                        "done": current,
                        "total": total,
                    }
                else:
                    session["progress"] = {
                        "phase": "ready",
                        "label": f"슬롯 {slot} 서버 재생성 완료",
                        "value": 100,
                        "done": 1,
                        "total": 1,
                    }
                session["updated_at"] = time.time()
                _persist_session_metadata(session)
                print(f"[ILLUST_CONTEXT] 재생성 캐시 갱신: session={session_id}, slot={slot}")
                return True
        except Exception:
            continue
    print(f"[ILLUST_CONTEXT] 재생성 캐시 slot 없음: session={session_id}, slot={slot}")
    return False


def _pct(value) -> str:
    return quote(str(value or ""), safe="")


def session_manifest(session_id: str) -> str:
    session = get_session(session_id)
    if not session:
        return "STATUS|missing\nCOUNT|0\nERROR|session_not_found"
    items = session.get("items") or []
    lines = [f"STATUS|{session.get('status', 'missing')}", f"COUNT|{len(items)}"]
    if session.get("error"):
        lines.append(f"ERROR|{_pct(session['error'])}")
    for idx, item in enumerate(items, start=1):
        lines.append("|".join([
            "ITEM", str(idx), _pct(item.get("kind", "scene")), str(item.get("slot", "")),
            _pct(item.get("camera")), _pct(item.get("scene")),
            _pct(item.get("supplement")), _pct(item.get("speak")),
        ]))
        for ch in item.get("characters") or []:
            if not isinstance(ch, dict):
                continue
            lines.append("|".join([
                "CHAR", str(idx), _pct(ch.get("name")), _pct(ch.get("positive")),
                _pct(ch.get("negative")), _pct(ch.get("position")),
            ]))
    return "\n".join(lines)


def ready_session_id_by_lookup_key(lookup_key: str) -> str:
    """Resolve a compact lookup key to one ready canonical session ID."""
    key = str(lookup_key or "").strip().lower()
    if not _LOOKUP_KEY_RE.fullmatch(key):
        raise ValueError(f"invalid illustration lookup key: {key!r}")

    session_id = _LOOKUP_KEYS.get(key)
    if not session_id:
        matches = _persisted_lookup_matches(key)
        if len(matches) > 1:
            raise LookupError(f"ambiguous illustration lookup key: key={key}, sessions={matches}")
        if len(matches) == 1:
            session_id = matches[0]
            _LOOKUP_KEYS[key] = session_id
    if not session_id:
        raise KeyError(f"illustration lookup key not found: {key}")

    session = get_session(session_id)
    if not session:
        raise KeyError(f"illustration session not found: key={key}")
    if str(session.get("status") or "missing") != "ready":
        raise RuntimeError(
            f"illustration session not ready: key={key}, status={session.get('status')}"
        )
    return session_id


def session_slots_by_lookup_key(lookup_key: str) -> list[int]:
    """Return only the ready slot numbers for the short HTTPS manifest route."""
    key = str(lookup_key or "").strip().lower()
    session_id = ready_session_id_by_lookup_key(key)
    session = get_session(session_id)
    if not session:
        raise KeyError(f"illustration session not found: key={key}")

    items = session.get("items") or []
    slots: list[int] = []
    seen: set[int] = set()
    for item in items:
        try:
            slot = int(item.get("slot"))
        except Exception as e:
            print(
                f"[ILLUST_CONTEXT] short manifest slot parse failed: "
                f"key={key}, item={item}, error={e}"
            )
            traceback.print_exc()
            raise ValueError("invalid slot in illustration session") from e
        if slot < -1 or slot in seen:
            raise ValueError(f"invalid or duplicate illustration slot: key={key}, slot={slot}")
        seen.add(slot)
        slots.append(slot)

    if not 1 <= len(slots) <= MAX_ILLUSTRATION_SLOT_COUNT:
        raise ValueError(
            f"invalid illustration slot count: key={key}, count={len(slots)}, "
            f"max={MAX_ILLUSTRATION_SLOT_COUNT}"
        )
    return slots


def context_text(chats: list[dict]) -> str:
    return "\n\n".join(
        ("[CHAR]" if item.get("role") == "char" else "[USER]") + "\n" + str(item.get("data") or "")
        for item in chats
    )


def _latest_narrative(chats: list[dict]) -> tuple[int, str]:
    for index in range(len(chats) - 1, -1, -1):
        if chats[index].get("role") == "char" and str(chats[index].get("data") or "").strip():
            return index, str(chats[index]["data"]).strip()
    print("[ILLUST_CONTEXT] 최신 CHAR 서사를 찾지 못함")
    return -1, ""


def _normalize_messages(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        role = "assistant" if role in ("char", "assistant") else role
        value = msg.get("content") or ""
        content = value.strip() if isinstance(value, PromptText) else str(value).strip()
        if not content:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    if out and out[0]["role"] == "assistant":
        out.insert(0, {"role": "user", "content": "Below is the preceding conversation for context."})
    return out


def _history_messages_text(messages: list[dict]) -> str:
    return "\n\n".join(
        ("[CHAR]" if item.get("role") == "char" else "[USER]")
        + "\n"
        + str(item.get("data") or item.get("content") or "").strip()
        for item in messages
        if str(item.get("data") or item.get("content") or "").strip()
    ).strip()


_HISTORY_OPERATION_RELATIONS = {
    "new": "NO_PRIOR_REFERENCE",
    "append": "PRIOR_COMMITTED_TURN",
    "duplicate": "SAME_ACTIVE_TURN_EXACT",
    "reroll": "SAME_ACTIVE_TURN_REPLACED",
}


def _turn_target_id(history_id: str, branch_id: str, base_context_hash: str) -> str:
    """Build a stable turn identity from server-owned history provenance."""
    history = str(history_id or "").strip()
    branch = str(branch_id or "").strip()
    base_hash = str(base_context_hash or "").strip()
    if not history or not branch or not base_hash:
        return ""
    seed = "\x00".join(("illustration-turn-v1", history, branch, base_hash))
    return "turn_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def build_reference_provenance(history_plan: dict | None) -> dict:
    """Normalize pre-OUTFIT history facts without asking an LLM to classify them."""
    if not isinstance(history_plan, dict):
        return {
            "history_operation": "untracked",
            "turn_relation": "NO_HISTORY",
            "history_id": "",
            "branch_id": "",
            "current_turn_target_id": "",
            "previous_turn_target_id": "",
            "target_comparison": "unavailable",
            "classification_source": "none",
        }

    operation = str(history_plan.get("operation") or "").strip().lower()
    relation = _HISTORY_OPERATION_RELATIONS.get(operation)
    if relation is None:
        relation = "UNKNOWN_HISTORY_RELATION"
        print(
            f"[ILLUST_CONTEXT:REFERENCE] 알 수 없는 히스토리 연산: "
            f"operation={operation!r}, history={history_plan.get('history_id')!r}"
        )

    record_before = history_plan.get("record_before") or {}
    if not isinstance(record_before, dict):
        print(
            f"[ILLUST_CONTEXT:REFERENCE] 이전 히스토리 레코드 형식 오류: "
            f"type={type(record_before).__name__}, operation={operation!r}"
        )
        record_before = {}
    source = record_before.get("source") or {}
    if not isinstance(source, dict):
        print(
            f"[ILLUST_CONTEXT:REFERENCE] 히스토리 source 형식 오류: "
            f"type={type(source).__name__}, operation={operation!r}"
        )
        source = {}
    active_turn = record_before.get("active_turn") or {}
    if not isinstance(active_turn, dict):
        print(
            f"[ILLUST_CONTEXT:REFERENCE] 이전 active_turn 형식 오류: "
            f"type={type(active_turn).__name__}, operation={operation!r}"
        )
        active_turn = {}

    history_id = str(history_plan.get("history_id") or "").strip()
    branch_id = str(source.get("branch_id") or "main").strip() or "main"
    current_target_id = _turn_target_id(
        history_id,
        branch_id,
        str(history_plan.get("base_context_hash") or ""),
    )
    previous_target_id = _turn_target_id(
        history_id,
        branch_id,
        str(active_turn.get("base_context_hash") or ""),
    )
    if current_target_id and previous_target_id:
        target_comparison = (
            "same" if current_target_id == previous_target_id else "different"
        )
    else:
        target_comparison = "unavailable"

    if (
        operation in ("append", "reroll", "duplicate")
        and target_comparison == "unavailable"
    ):
        print(
            f"[ILLUST_CONTEXT:REFERENCE] 턴 target 비교 metadata 부족: "
            f"operation={operation}, history={history_id}, branch={branch_id}, "
            f"current_target={bool(current_target_id)}, "
            f"previous_target={bool(previous_target_id)}"
        )
    elif operation in ("reroll", "duplicate") and target_comparison != "same":
        print(
            f"[ILLUST_CONTEXT:REFERENCE] 동일 활성 턴 연산의 target 비교 불일치: "
            f"operation={operation}, comparison={target_comparison}, "
            f"history={history_id}, branch={branch_id}"
        )
    elif operation == "append" and target_comparison == "same":
        print(
            f"[ILLUST_CONTEXT:REFERENCE] 과거 턴 추가인데 target id가 동일함: "
            f"history={history_id}, branch={branch_id}"
        )

    provenance = {
        "history_operation": operation or "unknown",
        "turn_relation": relation,
        "history_id": history_id,
        "branch_id": branch_id,
        "current_turn_target_id": current_target_id,
        "previous_turn_target_id": previous_target_id,
        "target_comparison": target_comparison,
        "classification_source": "history_alignment",
    }
    print(
        f"[ILLUST_CONTEXT:REFERENCE] 턴 관계 확정: "
        f"operation={provenance['history_operation']}, "
        f"relation={provenance['turn_relation']}, "
        f"target={provenance['target_comparison']}, "
        f"history={history_id}, branch={branch_id}"
    )
    return provenance


def _split_top_level_authority_tags(value: str) -> list[str]:
    """Split schema-owned comma tags without breaking weighted/grouped tags."""
    source = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    tags: list[str] = []
    current: list[str] = []
    stack: list[str] = []
    closing_for = {"(": ")", "[": "]", "{": "}"}
    escaped = False
    for char in source:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char in closing_for:
            stack.append(closing_for[char])
            current.append(char)
            continue
        if stack and char == stack[-1]:
            stack.pop()
            current.append(char)
            continue
        if char in (",", "\n") and not stack:
            tag = "".join(current).strip()
            if tag:
                tags.append(tag)
            current = []
            continue
        current.append(char)
    tag = "".join(current).strip()
    if tag:
        tags.append(tag)

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        folded = tag.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        deduped.append(tag)
    return deduped


def extract_authoritative_fixed_appearance(character_reference: str) -> dict[str, str]:
    """Extract only schema-declared ``-Appearance`` sections from lb.extra.

    This is structural parsing of the character dictionary, not semantic tag or
    narrative keyword matching. The source text is never modified.
    """
    source = str(character_reference or "")
    character_headers = list(re.finditer(r"(?m)^###\s+([^\r\n]+)\s*$", source))
    extracted: dict[str, str] = {}
    missing = []
    for index, header in enumerate(character_headers):
        name = header.group(1).strip()
        block_end = (
            character_headers[index + 1].start()
            if index + 1 < len(character_headers)
            else len(source)
        )
        block = source[header.end():block_end]
        marker = re.search(
            r"(?mi)^\s*-Appearance(?:\s*:\s*(.*))?\s*$",
            block,
        )
        if marker is None:
            missing.append(name)
            continue
        value_start = marker.end()
        next_section = re.search(r"(?m)^\s*-[^\r\n]+\s*$", block[value_start:])
        value_end = (
            value_start + next_section.start()
            if next_section is not None
            else len(block)
        )
        inline_value = str(marker.group(1) or "").strip()
        continuation = block[value_start:value_end].strip()
        value = "\n".join(
            part for part in (inline_value, continuation) if part
        )
        if not value:
            missing.append(name)
            continue
        extracted[name] = value
    if missing:
        print(
            "[ILLUST_CONTEXT:CALL2_DETAIL] lb.extra Appearance 섹션 누락/비어 있음: "
            f"characters={missing}"
        )
    print(
        "[ILLUST_CONTEXT:CALL2_DETAIL] 고정 Appearance 구조 추출: "
        f"characters={list(extracted)}, "
        f"chars={sum(len(value) for value in extracted.values())}"
    )
    return extracted


def extract_authoritative_default_outfits(
    character_reference: str,
) -> dict[str, list[str]]:
    """Extract complete ``-default_outfit`` tag sets from the character schema.

    This parses declared sections and comma boundaries only. It does not infer
    garment meaning from words or narrative text.
    """
    source = str(character_reference or "")
    character_headers = list(re.finditer(r"(?m)^###\s+([^\r\n]+)\s*$", source))
    extracted: dict[str, list[str]] = {}
    missing: list[str] = []
    for index, header in enumerate(character_headers):
        name = header.group(1).strip()
        block_end = (
            character_headers[index + 1].start()
            if index + 1 < len(character_headers)
            else len(source)
        )
        block = source[header.end():block_end]
        marker = re.search(
            r"(?mi)^\s*-default_outfit(?:\s*:\s*(.*))?\s*$",
            block,
        )
        if marker is None:
            missing.append(name)
            continue
        value_start = marker.end()
        next_section = re.search(r"(?m)^\s*-[^\r\n]+\s*$", block[value_start:])
        value_end = (
            value_start + next_section.start()
            if next_section is not None
            else len(block)
        )
        inline_value = str(marker.group(1) or "").strip()
        continuation = block[value_start:value_end].strip()
        value = "\n".join(part for part in (inline_value, continuation) if part)
        tags = _split_top_level_authority_tags(value)
        if not tags:
            missing.append(name)
            continue
        extracted[name] = tags
    if missing:
        print(
            "[ILLUST_CONTEXT:WARDROBE_BASE] lb.extra default_outfit 섹션 "
            f"누락/비어 있음: characters={missing}"
        )
    print(
        "[ILLUST_CONTEXT:WARDROBE_BASE] 기본 복장 구조 추출: "
        f"characters={list(extracted)}, tags={sum(len(tags) for tags in extracted.values())}"
    )
    return extracted


def classify_last_visual_reference(
    reference_provenance: dict | None,
    previous_visual: dict | None,
) -> dict:
    """Classify generated visual history from server-owned turn metadata.

    This deliberately does not inspect prompt words.  A missing or contradictory
    turn relationship is downgraded to SOFT_REFERENCE instead of asking CALL2 to
    infer chronology from generated tags.
    """
    visual = previous_visual if isinstance(previous_visual, dict) else {}
    usable_entries = {
        str(name): value
        for name, value in visual.items()
        if str(name).strip() and isinstance(value, dict) and value
    }
    provenance = (
        reference_provenance
        if isinstance(reference_provenance, dict)
        else {}
    )
    operation = str(provenance.get("history_operation") or "unknown").strip().lower()
    comparison = str(provenance.get("target_comparison") or "unavailable").strip().lower()

    if not usable_entries:
        reference_type = "IGNORE"
        reason = "no usable generated visual payload"
    elif operation == "new":
        reference_type = "IGNORE"
        reason = "new history operation has no prior visual authority"
    elif operation in ("duplicate", "reroll") and comparison == "same":
        reference_type = "REROLL"
        reason = (
            "same active turn exact-content rendering attempt"
            if operation == "duplicate"
            else "same active turn replacement rendering attempt"
        )
    elif operation == "append" and comparison == "different":
        reference_type = "CONTINUITY"
        reason = "different target from an earlier committed turn"
    else:
        reference_type = "SOFT_REFERENCE"
        reason = (
            "turn metadata is missing or contradicts the declared history operation"
        )

    result = {
        "reference_type": reference_type,
        "reference_reason": reason,
        "history_operation": operation,
        "turn_relation": str(provenance.get("turn_relation") or "UNKNOWN_HISTORY_RELATION"),
        "target_comparison": comparison,
        "branch_id": str(provenance.get("branch_id") or ""),
        "current_turn_target_id": str(
            provenance.get("current_turn_target_id") or ""
        ),
        "previous_turn_target_id": str(
            provenance.get("previous_turn_target_id") or ""
        ),
        "character_count": len(usable_entries),
    }
    print(
        "[ILLUST_CONTEXT:CALL2_DETAIL] LAST VISUAL 서버 분류: "
        f"type={reference_type}, reason={reason}, operation={operation}, "
        f"target={comparison}, characters={list(usable_entries)}"
    )
    return result


def _classified_visual_reference_content(
    classification: dict,
    previous_visual: dict | None,
) -> str:
    """Render the DETAIL-only generated reference with type-scoped authority."""
    reference_type = str(
        (classification or {}).get("reference_type") or "IGNORE"
    ).strip().upper()
    if reference_type == "IGNORE":
        print(
            "[ILLUST_CONTEXT:CALL2_DETAIL] LAST VISUAL payload 미전달: "
            f"reason={(classification or {}).get('reference_reason') or 'IGNORE'}"
        )
        return ""
    if not isinstance(previous_visual, dict) or not previous_visual:
        print(
            "[ILLUST_CONTEXT:CALL2_DETAIL] LAST VISUAL 분류는 있으나 payload가 비어 있음: "
            f"type={reference_type}"
        )
        return ""

    if reference_type == "CONTINUITY":
        type_rule = (
            "This is an earlier chronological scene. It may support only temporary visual "
            "states that logically continue into the assigned current scene. Never copy its "
            "pose, action, expression, gaze, camera, framing, or composition automatically."
        )
    elif reference_type == "REROLL":
        type_rule = (
            "This is another rendering attempt of the same active turn, not a previous story "
            "event. Use it only as optional visual guidance. Re-evaluate pose, action, expression, "
            "gaze, camera, framing, composition, and every temporary choice from the current scene. "
            "No generated detail may be inherited merely because it appears below."
        )
    metadata = {
        key: value
        for key, value in (classification or {}).items()
        if key != "character_count"
    }
    if reference_type == "SOFT_REFERENCE":
        print(
            "[ILLUST_CONTEXT:CALL2_DETAIL] SOFT_REFERENCE 본문 차단: "
            f"reason={metadata.get('reference_reason') or 'unverified chronology'}"
        )
        return (
            "# CLASSIFIED LAST VISUAL REFERENCE\n"
            "The server classified an available generated image as SOFT_REFERENCE, but its "
            "chronology is unverified. Its generated prompt payload is intentionally withheld. "
            "Do not infer or inherit identity, wardrobe, pose, expression, action, composition, "
            "or temporary state from it. Use the assigned story passage and authoritative "
            "continuity sources instead.\n\n"
            "# SERVER REFERENCE CLASSIFICATION\n"
            + json.dumps(metadata, ensure_ascii=False, indent=2)
        )

    return (
        "# CLASSIFIED LAST VISUAL REFERENCE\n"
        "The server, not the model, has classified this generated reference.\n"
        f"{type_rule}\n\n"
        "AUTHORITATIVE FIXED APPEARANCE and story-grounded continuity_note always override "
        "this payload. The generated positive_tags below are never a persistent identity source "
        "and never evidence that clothing was changed. For a verified CONTINUITY reference only, "
        "the planner may retain its garment design details when the story establishes that the "
        "same outfit continues; story-grounded worn/carried/removed state always wins. "
        "Do not promote, complete, or repeat an appearance tag "
        "unless it is independently supported by AUTHORITATIVE FIXED APPEARANCE.\n\n"
        "# SERVER REFERENCE CLASSIFICATION\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n\n# NON-AUTHORITATIVE GENERATED REFERENCE PAYLOAD\n"
        + json.dumps(previous_visual, ensure_ascii=False, indent=2)
    )


def _call1_state_for_prompt(
    character_state: dict | None,
    costume_reference: str = "",
    visual_profiles: dict[str, dict] | None = None,
) -> dict:
    """Remove generated visual data that OUTFIT does not consume.

    Stored state is left untouched. OUTFIT still receives current wardrobe and its
    evidence timeline. A duplicated default outfit is removed only when that
    character is present in the dedicated lb.extra costume block.
    """
    if not isinstance(character_state, dict):
        if character_state is not None:
            print(
                f"[ILLUST_CONTEXT:OUTFIT] 추적 캐릭터 상태 형식 오류로 빈 상태 사용: "
                f"type={type(character_state).__name__}"
            )
        return {}

    result = deepcopy(character_state)
    costume_names = {
        match.group(1).strip().casefold()
        for match in re.finditer(
            r"(?m)^###\s+([^\r\n]+)\s*$",
            str(costume_reference or ""),
        )
        if match.group(1).strip()
    }
    removed_visual = 0
    removed_default = 0
    preserved_default = 0
    profile_ids_replaced = 0
    profile_timelines_removed = 0
    malformed = []
    for name, tracked in result.items():
        if not isinstance(tracked, dict):
            malformed.append(str(name))
            continue
        if "last_visual_reference" in tracked:
            tracked.pop("last_visual_reference", None)
            removed_visual += 1
        if "visual_base_timeline" in tracked:
            tracked.pop("visual_base_timeline", None)
            profile_timelines_removed += 1
        tracked_profile_id = str(
            tracked.pop("active_visual_profile_id", "") or ""
        ).strip()
        if tracked_profile_id:
            canonical_name = str(tracked.get("canonical_name") or name).strip()
            character_profiles = _authority_values_for_name(
                visual_profiles or {},
                canonical_name,
            )
            profile = (
                profile_by_id(character_profiles, tracked_profile_id)
                if isinstance(character_profiles, dict)
                else None
            )
            semantic_names = visual_profile_names(profile or {})
            if semantic_names:
                tracked["active_visual_profile"] = semantic_names[0]
                profile_ids_replaced += 1
            else:
                print(
                    f"[ILLUST_CONTEXT:OUTFIT] 추적 프로필 ID의 의미 이름을 찾지 못해 "
                    f"OUTFIT 입력에서 제거: character={canonical_name!r}, "
                    f"profile_id={tracked_profile_id!r}"
                )
        if "default_outfit_reference" in tracked:
            canonical_name = str(tracked.get("canonical_name") or name).strip()
            if canonical_name.casefold() in costume_names:
                tracked.pop("default_outfit_reference", None)
                removed_default += 1
            else:
                preserved_default += 1
    if malformed:
        print(
            f"[ILLUST_CONTEXT:OUTFIT] 비정상 추적 캐릭터 상태는 원형 보존: "
            f"characters={malformed}"
        )

    try:
        before_chars = len(json.dumps(character_state, ensure_ascii=False, indent=2))
        after_chars = len(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            f"[ILLUST_CONTEXT:OUTFIT] 입력 상태 정리: characters={len(result)}, "
            f"visual_removed={removed_visual}, default_duplicate_removed={removed_default}, "
            f"default_preserved_without_extra={preserved_default}, "
            f"profile_ids_replaced={profile_ids_replaced}, "
            f"profile_timelines_removed={profile_timelines_removed}, "
            f"chars={before_chars}->{after_chars} (-{before_chars - after_chars})"
        )
    except Exception as e:
        print(f"[ILLUST_CONTEXT:OUTFIT] 입력 상태 정리 크기 계산 실패: error={e}")
        traceback.print_exc()
    return result


def _call1_wardrobe_state_for_prompt(
    character_state: dict | None,
    costume_reference: str = "",
    visual_profiles: dict[str, dict] | None = None,
) -> dict:
    """Project OUTFIT wardrobe state without duplicating profile authority fields."""
    result = _call1_state_for_prompt(
        character_state,
        costume_reference,
        visual_profiles,
    )
    for tracked in result.values():
        if not isinstance(tracked, dict):
            continue
        tracked.pop("active_visual_profile", None)
        tracked.pop("active_visual_profile_state", None)
    return result


def _selected_visual_profiles(
    visual_profiles: dict[str, dict] | None,
    character_names: list[str],
    *,
    multiple_only: bool = False,
) -> dict[str, dict]:
    """Return exact registered profile entries for resolved CURRENT characters."""
    selected: dict[str, dict] = {}
    for name in character_names or []:
        canonical_name = str(name or "").strip()
        if not canonical_name:
            continue
        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            canonical_name,
        )
        if not isinstance(character_profiles, dict):
            print(
                f"[ILLUST_CONTEXT:PROFILE_RESOLVE] 등록 프로필 없음: "
                f"character={canonical_name!r}"
            )
            continue
        if multiple_only and len(character_profiles.get("profiles") or []) <= 1:
            continue
        selected[canonical_name] = deepcopy(character_profiles)
    return selected


def _persona_identity_context(
    visual_profiles: dict[str, dict] | None,
    character_names: list[str] | None = None,
) -> str:
    """Render one semantic identity note without parsing first-person wording in code."""
    persona_names = [
        str(name or "").strip()
        for name, value in (visual_profiles or {}).items()
        if str(name or "").strip()
        and isinstance(value, dict)
        and value.get("is_persona") is True
    ]
    persona_names = list(dict.fromkeys(persona_names))
    if len(persona_names) > 1:
        print(
            f"[ILLUST_CONTEXT:PERSONA] 페르소나가 둘 이상이라 정체성 안내 생략: "
            f"characters={persona_names}"
        )
        return ""
    if not persona_names:
        return ""

    persona_name = persona_names[0]
    allowed_names = {
        str(name or "").strip().casefold()
        for name in (character_names or [])
        if str(name or "").strip()
    }
    if allowed_names and persona_name.casefold() not in allowed_names:
        return ""
    return (
        "# REGISTERED USER PERSONA IDENTITY\n"
        f"The exact canonical character `{persona_name}` is the configured user persona. "
        "When the narrative is genuinely written from the user/player's first-person self "
        "perspective, interpret that self-reference as this canonical character. Resolve "
        "perspective, quotation, embedded speech or thought, and narrator changes from the "
        "full discourse; first-person language belonging to another established viewpoint "
        "does not refer to the user persona. This identity mapping does not by itself invent "
        "participation, an action, a visible state, or a profile transition."
    )


def _has_configured_identity_lora(value) -> bool:
    """Return whether one card-local identity-LoRA field has usable content."""
    if isinstance(value, (list, tuple, set)):
        return any(bool(item) for item in value)
    if isinstance(value, dict):
        return bool(value)
    return bool(str(value or "").strip())


def single_preset_subject_candidates(
    visual_profiles: dict[str, dict] | None,
    character_names: list[str] | None,
) -> tuple[list[str], str]:
    """Derive Single V5 named-subject authority from registered render identity.

    Character-LoRA ownership is authoritative when present. Tag/reference-only bots
    remain usable through a non-persona roster fallback, then the complete CURRENT
    roster. This is card metadata reasoning, never narrative keyword matching.
    """
    roster = list(dict.fromkeys(
        str(name or "").strip()
        for name in (character_names or [])
        if str(name or "").strip()
    ))
    lora_owners: list[str] = []
    non_persona: list[str] = []
    for name in roster:
        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            name,
        )
        if not isinstance(character_profiles, dict):
            continue
        if character_profiles.get("is_persona") is not True:
            non_persona.append(name)
        owns_identity_lora = False
        for profile_index, profile in enumerate(
            character_profiles.get("profiles") or [],
            start=1,
        ):
            if not isinstance(profile, dict):
                print(
                    "[ILLUST_CONTEXT:CALL2_PLAN] 캐릭터 카드 형식 오류로 LoRA 소유권 "
                    f"판정에서 스킵: character={name!r}, profile_index={profile_index}, "
                    f"type={type(profile).__name__}"
                )
                continue
            overrides = profile.get("render_overrides") or {}
            if not isinstance(overrides, dict):
                print(
                    "[ILLUST_CONTEXT:CALL2_PLAN] render_overrides 형식 오류로 LoRA 소유권 "
                    f"판정에서 스킵: character={name!r}, profile_index={profile_index}, "
                    f"type={type(overrides).__name__}"
                )
                continue
            if any(
                _has_configured_identity_lora(overrides.get(field))
                for field in _CHARACTER_IDENTITY_LORA_FIELDS
            ):
                owns_identity_lora = True
                break
        if owns_identity_lora:
            lora_owners.append(name)

    if lora_owners:
        return lora_owners, "configured_character_identity_lora"
    if non_persona:
        return non_persona, "registered_non_persona_fallback"
    if roster:
        return roster, "current_roster_fallback"
    print(
        "[ILLUST_CONTEXT:CALL2_PLAN] Single V5 주체 권위 생성 실패: "
        "CURRENT canonical roster가 비어 있음"
    )
    return [], "unavailable"


def _character_card_identity_context(
    visual_profiles: dict[str, dict] | None,
    character_names: list[str] | None = None,
) -> str:
    """Render card-authored identity hints without turning cards into story state."""
    sections: list[str] = []
    for name in character_names or []:
        canonical_name = str(name or "").strip()
        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            canonical_name,
        )
        if not canonical_name or not isinstance(character_profiles, dict):
            continue

        aliases: list[str] = []
        descriptions: list[str] = []
        seen_aliases: set[str] = set()
        seen_descriptions: set[str] = set()
        for profile in character_profiles.get("profiles") or []:
            if not isinstance(profile, dict):
                continue
            for raw_alias in profile.get("aliases") or []:
                alias = str(raw_alias or "").strip()
                folded = alias.casefold()
                if alias and folded not in seen_aliases:
                    seen_aliases.add(folded)
                    aliases.append(alias)
            description = str(
                profile.get("visual_context_english")
                or profile.get("visual_context")
                or ""
            ).strip()
            folded_description = description.casefold()
            if description and folded_description not in seen_descriptions:
                seen_descriptions.add(folded_description)
                descriptions.append(description)

        if not aliases and not descriptions:
            continue
        lines = [f"### {canonical_name}"]
        if aliases:
            lines.append(
                "Card-authored aliases or semantic names: "
                + json.dumps(aliases, ensure_ascii=False)
            )
        if descriptions:
            lines.append("Card-authored natural-language identity references:")
            lines.extend(f"- {description}" for description in descriptions)
        sections.append("\n".join(lines))

    if not sections:
        return ""
    return (
        "# REGISTERED CHARACTER CARD IDENTITY REFERENCES\n"
        "These existing card-authored names and natural-language descriptions may help "
        "connect a narrative alias, localized name, title, or description to one exact "
        "canonical roster name. They are identity reference only. They do not establish "
        "CURRENT participation, story chronology, wardrobe, action, or active profile, "
        "and nothing from the story is stored back into a card.\n\n"
        + "\n\n".join(sections)
    )


def _profile_state_for_prompt(
    character_state: dict | None,
    character_names: list[str],
    visual_profiles: dict[str, dict] | None,
) -> dict[str, dict]:
    """Render only the prior semantic profile state needed by PROFILE-RESOLVE."""
    result: dict[str, dict] = {}
    states = character_state if isinstance(character_state, dict) else {}
    for name in character_names or []:
        canonical_name = str(name or "").strip()
        if not canonical_name:
            continue
        tracked = next((
            value
            for key, value in states.items()
            if isinstance(value, dict)
            and str(value.get("canonical_name") or key).strip().casefold()
            == canonical_name.casefold()
        ), None)
        if not isinstance(tracked, dict):
            continue
        profile_id = str(tracked.get("active_visual_profile_id") or "").strip()
        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            canonical_name,
        )
        profile = (
            profile_by_id(character_profiles, profile_id)
            if isinstance(character_profiles, dict) and profile_id
            else None
        )
        registered_profiles = (
            list(character_profiles.get("profiles") or [])
            if isinstance(character_profiles, dict)
            else []
        )
        profile_number = next(
            (
                index
                for index, candidate in enumerate(registered_profiles, start=1)
                if candidate is profile
            ),
            0,
        )
        if not profile_number:
            print(
                f"[ILLUST_CONTEXT:PROFILE_RESOLVE] 이전 프로필 참조 번호 없음: "
                f"character={canonical_name!r}, profile_id={profile_id!r}"
            )
            continue
        item = {
            "active_profile_ref": f"[{profile_number}]",
        }
        state_description = str(
            tracked.get("active_visual_profile_state") or ""
        ).strip()
        if state_description:
            item["active_visual_profile_state"] = state_description
        result[canonical_name] = item
    return result


def _state_without_generated_visual_references(
    character_state: dict | None,
    *,
    source: str,
) -> dict:
    """Copy tracked state while removing generated visual payloads.

    The current wardrobe and its evidence remain available.  This projection is
    used by stages that need authoritative state but must not inherit an earlier
    generated prompt.  Stored history is never mutated.
    """
    if not isinstance(character_state, dict):
        if character_state is not None:
            print(
                f"[ILLUST_CONTEXT:{source}] 추적 캐릭터 상태 형식 오류로 빈 상태 사용: "
                f"type={type(character_state).__name__}"
            )
        return {}

    removed = 0

    def scrub(value):
        nonlocal removed
        if isinstance(value, dict):
            cleaned = {}
            for key, child in value.items():
                if str(key) == "last_visual_reference":
                    removed += 1
                    continue
                cleaned[key] = scrub(child)
            return cleaned
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return deepcopy(value)

    result = scrub(character_state)
    try:
        before_chars = len(json.dumps(character_state, ensure_ascii=False, indent=2))
        after_chars = len(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            f"[ILLUST_CONTEXT:{source}] generated visual 상태 제거: "
            f"references={removed}, chars={before_chars}->{after_chars} "
            f"(-{before_chars - after_chars})"
        )
    except Exception as e:
        print(
            f"[ILLUST_CONTEXT:{source}] generated visual 상태 크기 계산 실패: "
            f"error={e}"
        )
        traceback.print_exc()
    return result


_PROTECTED_BOUNDARY_PAIRS = {
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "«": "»",
    "‹": "›",
    "(": ")",
    "（": "）",
    "[": "]",
    "{": "}",
    "【": "】",
}
_SYMMETRIC_BOUNDARY_QUOTES = {'"', "'"}


def _is_ascii_word_character(value: str) -> bool:
    return bool(value) and value.isascii() and (value.isalnum() or value == "_")


def _is_escaped_delimiter(source: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and source[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _advance_protected_boundary_state(
    source: str,
    index: int,
    stack: list[str],
) -> int:
    token = source[index]
    if token in _SYMMETRIC_BOUNDARY_QUOTES:
        if _is_escaped_delimiter(source, index):
            return index + 1
        previous = source[index - 1] if index > 0 else ""
        following = source[index + 1] if index + 1 < len(source) else ""
        if token == "'" and _is_ascii_word_character(previous) and _is_ascii_word_character(following):
            return index + 1
        if stack and stack[-1] == token:
            stack.pop()
        elif token == "'" and _is_ascii_word_character(previous):
            # A possessive apostrophe without an opening quote is ordinary prose.
            return index + 1
        else:
            stack.append(token)
        return index + 1
    if stack and token == stack[-1]:
        stack.pop()
        return index + 1
    closing = _PROTECTED_BOUNDARY_PAIRS.get(token)
    if closing:
        stack.append(closing)
    return index + 1


def _boundaries_outside_protected_spans(
    source: str,
    boundaries: list[tuple[int, int]],
    *,
    boundary_kind: str,
) -> list[tuple[int, int]]:
    """Exclude separators that would split an unfinished quotation or aside."""
    eligible: list[tuple[int, int]] = []
    stack: list[str] = []
    cursor = 0
    skipped = 0
    for boundary_start, boundary_end in boundaries:
        while cursor < boundary_start:
            cursor = _advance_protected_boundary_state(source, cursor, stack)
        if stack:
            skipped += 1
            continue
        eligible.append((boundary_start, boundary_end))
    if skipped:
        print(
            "[ILLUST_CONTEXT:SEGMENT] 열린 대사/괄호 내부 경계 생략: "
            f"kind={boundary_kind}, skipped={skipped}, eligible={len(eligible)}, "
            f"source_len={len(source)}"
        )
    return eligible


def _context_segment_boundaries(source: str) -> tuple[list[tuple[int, int]], bool]:
    """Return structural prose separators without splitting unfinished speech."""
    raw_paragraph_boundaries = [
        (match.start(), match.end())
        for match in re.finditer(r"\n\n+", source)
    ]
    paragraph_boundaries = _boundaries_outside_protected_spans(
        source,
        raw_paragraph_boundaries,
        boundary_kind="paragraph",
    )
    if not source:
        return paragraph_boundaries, False

    text_bytes = len(source.encode("utf-8"))
    maximum_gap = max(512, math.ceil(text_bytes / 64) * 3)
    previous_end = 0
    needs_coverage = False
    for boundary_start, boundary_end in paragraph_boundaries:
        if len(source[previous_end:boundary_start].encode("utf-8")) > maximum_gap:
            needs_coverage = True
            break
        previous_end = boundary_end
    if (
        not needs_coverage
        and len(source[previous_end:].encode("utf-8")) > maximum_gap
    ):
        needs_coverage = True
    if not needs_coverage:
        return paragraph_boundaries, False

    raw_sentence_boundaries = [
        match.span("space")
        for match in re.finditer(
            r"[.!?。！？](?:[\"'”’」』»›)）\]】}]*)(?P<space>\s+)",
            source,
        )
    ]
    sentence_boundaries = _boundaries_outside_protected_spans(
        source,
        raw_sentence_boundaries,
        boundary_kind="sentence",
    )
    combined = sorted(paragraph_boundaries + sentence_boundaries)
    merged: list[list[int]] = []
    for boundary_start, boundary_end in combined:
        if merged and boundary_start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], boundary_end)
        else:
            merged.append([boundary_start, boundary_end])
    boundaries = [(start, end) for start, end in merged]
    print(
        "[ILLUST_CONTEXT:SEGMENT] 긴 문단 안전 경계 보강: "
        f"bytes={text_bytes}, paragraphs={len(paragraph_boundaries)}, "
        f"candidates={len(boundaries)}, maximum_gap={maximum_gap}"
    )
    return boundaries, True


def _segment_current_context(
    text: str, *, slotted_context: str | None = None,
) -> tuple[str, dict[str, dict]]:
    """Give OUTFIT stable structural segment ids without interpreting keywords."""
    if slotted_context is not None:
        source = str(slotted_context or "").strip()
        markers = []
        seen_slots = set()
        for marker in _SLOT_MARKER_RE.finditer(source):
            slot = int(marker.group(1))
            if slot in seen_slots:
                print(f"[ILLUST_CONTEXT:SEGMENT] 반복된 삽입 위치 마커 생략: slot={slot}")
                continue
            seen_slots.add(slot)
            markers.append(marker)
        if markers:
            projected, source_indexes = _slotless_projection_with_source_indexes(source)
            segments = {}
            rendered = []
            start = 0
            for index, marker in enumerate(markers):
                # One source passage per actual insertion position. The final
                # passage includes the tail, rather than creating another C ID
                # and assigning it to an already occupied slot afterward.
                end = len(source) if index == len(markers) - 1 else marker.start()
                projected_start = bisect_left(source_indexes, start)
                projected_end = bisect_left(source_indexes, end)
                passage = projected[projected_start:projected_end]
                content = passage.strip()
                if not content:
                    print(f"[ILLUST_CONTEXT:SEGMENT] 본문 없는 삽입 후보 생략: slot={marker.group(1)}")
                    start = marker.end()
                    continue
                segment_id = f"C{len(segments) + 1:03d}"
                content_start = projected_start + len(passage) - len(passage.lstrip())
                segments[segment_id] = {
                    "id": segment_id, "text": content,
                    "start": content_start, "end": content_start + len(content),
                    "slot": int(marker.group(1)),
                }
                rendered.append(f"[{segment_id}]\n{content}")
                start = marker.end()
            return "\n\n".join(rendered), segments
        print("[ILLUST_CONTEXT:SEGMENT] 삽입 후보 마커 없음; 원문 구간만 생성")
    source = str(text or "")
    segments: dict[str, dict] = {}
    rendered = []
    boundaries, coverage_mode = _context_segment_boundaries(source)
    if coverage_mode:
        spans = []
        cursor = 0
        for boundary_start, boundary_end in boundaries:
            spans.append((cursor, boundary_start))
            cursor = boundary_end
        spans.append((cursor, len(source)))
    else:
        spans = [
            (match.start(), match.end())
            for match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", source)
        ]

    for start, end in spans:
        segment_text = source[start:end]
        if not segment_text.strip():
            continue
        segment_id = f"C{len(segments) + 1:03d}"
        segments[segment_id] = {
            "id": segment_id,
            "text": segment_text,
            "start": start,
            "end": end,
        }
        rendered.append(f"[{segment_id}]\n{segment_text}")
    if not segments and source.strip():
        segments["C001"] = {"id": "C001", "text": source, "start": 0, "end": len(source)}
        rendered.append(f"[C001]\n{source}")
    return "\n\n".join(rendered), segments


def _json_object_from_text(text: str, *, array_key: str | None = None) -> dict | None:
    source = str(text or "").lstrip().rstrip(" \t")
    candidates = [match.group(1).lstrip().rstrip(" \t") for match in re.finditer(
        r"```(?:[A-Za-z0-9_-]+)?\s*\n([\s\S]*?)```", source, re.I)]
    candidates.append(source)
    for candidate in candidates:
        # Remove enclosing syntax only; never edit tags or punctuation inside
        # a string value. A missing closing wrapper does not damage its body.
        while opening := re.match(
            r"^(?:```[^\r\n]*\r?\n|<lb[-_]xnai[^>]*>\s*|<scenes>\s*|\[TOON\]\s*)",
            candidate, re.I,
        ):
            candidate = candidate[opening.end():].lstrip().rstrip(" \t")
        while closing := re.search(
            r"(?:```|</lb[-_]xnai>|</scenes>|\[/TOON\])\s*$", candidate, re.I,
        ):
            candidate = candidate[:closing.start()].lstrip().rstrip(" \t")
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = None
            # Decode a complete object surrounded by prose without joining it
            # to unrelated braces in that prose or repairing truncated JSON.
            offsets = [0] if candidate.startswith(("{", "[")) else []
            for offset in offsets:
                try:
                    found, _end = json.JSONDecoder().raw_decode(candidate, offset)
                except json.JSONDecodeError:
                    continue
                if isinstance(found, dict):
                    value = found
                    break
                if array_key and isinstance(found, list):
                    value = found
                    break
            if value is None:
                try:
                    value = yaml.safe_load(candidate if candidate.startswith(("{", "["))
                                           else _normalize_toon(candidate))
                except yaml.YAMLError as exc:
                    print(f"[ILLUST_CONTEXT:PARSE] YAML 복구 불가: error={exc}, input={candidate[:500]!r}")
                    traceback.print_exc()
            # A YAML document may contain inline JSON arrays such as
            # characters: []. Read its root before looking inside prose for
            # a JSON document; otherwise that nested array becomes the batch.
            if value is None or not isinstance(value, (dict, list)):
                if not candidate.startswith(("{", "[")):
                    for opening in re.finditer(r"[\[{]", candidate):
                        try:
                            found, _end = json.JSONDecoder().raw_decode(candidate, opening.start())
                        except json.JSONDecodeError:
                            continue
                        if isinstance(found, dict) and (not array_key or array_key in found):
                            value = found
                            break
        if isinstance(value, list) and array_key:
            print(f"[ILLUST_CONTEXT:PARSE] 배열 루트 복구: field={array_key}, count={len(value)}")
            value = {array_key: value}
        if isinstance(value, dict):
            if candidate != source or not source.startswith("{"):
                print(f"[ILLUST_CONTEXT:PARSE] 알려진 구조화 형식 복구: keys={list(value)}, input={source[:100]!r}")
            return value
    print(f"[ILLUST_CONTEXT:PARSE] 복구 가능한 구조화 object 없음: input={source[:500]!r}")
    return None


def _canonical_name_map(character_names: str) -> dict[str, str]:
    names = [name.strip() for name in str(character_names or "").split(",") if name.strip()]
    return {name.casefold(): name for name in names}


def _normalize_analysis_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _analysis_evidence_matches_segment(evidence: str, segment_text: str) -> bool:
    """Compare literal evidence while tolerating transport-only whitespace changes."""
    normalized_evidence = re.sub(
        r"\s+", " ", _normalize_analysis_text(evidence)
    ).strip()
    normalized_segment = re.sub(
        r"\s+", " ", _normalize_analysis_text(segment_text)
    ).strip()
    return bool(
        normalized_evidence
        and normalized_segment
        and normalized_evidence in normalized_segment
    )


def _analysis_evidence_matches_context(
    evidence: str,
    current_context: str,
    start_offset: int = 0,
) -> bool:
    """Accept exact evidence split across consecutive narrative paragraphs.

    OUTFIT may quote one semantic change across several numbered paragraphs while
    omitting server-added segment labels or timestamp separators.  Preserve that
    natural-language judgment only when every quoted line is found verbatim and
    in order in CURRENT after the claimed anchor.
    """
    normalized_evidence = _normalize_analysis_text(evidence)
    source = str(current_context or "")
    try:
        offset = max(0, min(len(source), int(start_offset or 0)))
    except Exception as e:
        print(
            "[ILLUST_CONTEXT:OUTFIT] 근거 검색 시작 위치 보정 실패: "
            f"start_offset={start_offset!r}, error={e}"
        )
        traceback.print_exc()
        offset = 0
    normalized_context = re.sub(
        r"\s+", " ", _normalize_analysis_text(source[offset:])
    ).strip()
    normalized_whole = re.sub(r"\s+", " ", normalized_evidence).strip()
    if not normalized_whole or not normalized_context:
        return False
    if normalized_whole in normalized_context:
        return True

    evidence_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in normalized_evidence.split("\n")
        if re.sub(r"\s+", " ", line).strip()
    ]
    if len(evidence_lines) < 2:
        return False
    cursor = 0
    for line in evidence_lines:
        position = normalized_context.find(line, cursor)
        if position < 0:
            return False
        cursor = position + len(line)
    return True


def _contains_canonical_name(text: str, name: str) -> bool:
    """Check a supplied canonical name as a complete token, not as inference."""
    source = _normalize_analysis_text(text)
    target = str(name or "").strip()
    if not source or not target:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(target)}(?!\w)", source, re.I))


def parse_call1_analysis(
    text: str,
    current_context: str,
    segments: dict[str, dict],
    character_names: str,
    history_context: str = "",
    visual_profiles: dict[str, dict] | None = None,
    resolved_characters: dict | None = None,
) -> dict | None:
    """Validate OUTFIT wardrobe/hairstyle output with optional pre-resolved entities."""
    raw = _json_object_from_text(text)
    if raw is None:
        return None
    if isinstance(resolved_characters, dict):
        raw = deepcopy(raw)
        raw["reference_assignments"] = []
        raw["history_characters"] = list(
            resolved_characters.get("history_characters") or []
        )
        raw["current_characters"] = list(
            resolved_characters.get("current_characters") or []
        )
        # Resolver commentary includes deliberate exclusions (unregistered people,
        # hypothetical mentions). It is not a failed identity resolution.
        raw["unresolved_references"] = []
        if resolved_characters.get("uncertainties"):
            print(
                "[ILLUST_CONTEXT:OUTFIT] 선행 캐릭터 선택의 설명을 실패로 승격하지 않음: "
                f"notes={resolved_characters['uncertainties']}"
            )
    canonical = _canonical_name_map(character_names)
    warnings = []
    fallback_errors = []

    def normalize_name(value) -> str:
        name = str(value or "").strip()
        return canonical.get(name.casefold(), name)

    history_characters = []
    for item in raw.get("history_characters") or []:
        name = normalize_name(item.get("name") if isinstance(item, dict) else item)
        if name and name not in history_characters:
            history_characters.append(name)

    current_characters = []
    current_names = set()
    for item in raw.get("current_characters") or []:
        if isinstance(item, dict):
            name = normalize_name(item.get("name"))
            confidence = item.get("confidence", 1.0)
        else:
            name = normalize_name(item)
            confidence = 1.0
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except Exception:
            confidence = 0.0
        if not name or name.casefold() in current_names:
            continue
        if confidence < 0.70:
            warnings.append(f"현재 캐릭터 신뢰도 낮아 폐기: {name}={confidence:.2f}")
            continue
        current_names.add(name.casefold())
        current_characters.append({"name": name, "confidence": confidence})

    assignments = []
    for index, item in enumerate(raw.get("reference_assignments") or [], start=1):
        if not isinstance(item, dict):
            warnings.append(f"지칭 할당 형식 오류로 폐기: index={index}")
            continue
        segment_id = str(item.get("segment_id") or "").strip()
        surface = str(item.get("surface") or "")
        name = normalize_name(item.get("canonical_name") or item.get("name"))
        replacement = str(item.get("replacement") or name).strip()
        try:
            occurrence = max(1, int(item.get("occurrence") or 1))
            confidence = max(0.0, min(1.0, float(item.get("confidence", 1.0))))
        except Exception:
            occurrence = 1
            confidence = 0.0
        if segment_id not in segments or not surface or not name:
            warnings.append(
                f"지칭 할당 필수값 오류로 폐기: index={index}, segment={segment_id!r}, "
                f"surface={surface!r}, name={name!r}"
            )
            continue
        if surface not in str(segments[segment_id].get("text") or ""):
            warnings.append(
                f"지칭 원문 불일치로 폐기: segment={segment_id}, surface={surface!r}"
            )
            continue
        if canonical and name.casefold() not in canonical:
            warnings.append(f"정식 이름 목록 밖 지칭 할당으로 폐기: {name}")
            continue
        if name.casefold() not in replacement.casefold():
            replacement = name
        if confidence < 0.70:
            warnings.append(
                f"지칭 할당 신뢰도 낮아 폐기: {segment_id}/{surface}={confidence:.2f}"
            )
            continue
        assignments.append({
            "ref_id": f"REF_{len(assignments) + 1:03d}",
            "segment_id": segment_id,
            "surface": surface,
            "occurrence": occurrence,
            "canonical_name": name,
            "replacement": replacement,
            "confidence": confidence,
        })
        if name.casefold() not in current_names:
            current_names.add(name.casefold())
            current_characters.append({"name": name, "confidence": confidence})

    if not isinstance(resolved_characters, dict):
        for folded, name in canonical.items():
            if _contains_canonical_name(current_context, name) and folded not in current_names:
                current_names.add(folded)
                current_characters.append({"name": name, "confidence": 1.0})
                warnings.append(f"OUTFIT이 원문 정식 이름을 누락해 서버가 보완: {name}")

        history_names = {name.casefold() for name in history_characters}
        for folded, name in canonical.items():
            if _contains_canonical_name(history_context, name) and folded not in history_names:
                history_names.add(folded)
                history_characters.append(name)
                warnings.append(f"OUTFIT이 과거 히스토리 정식 이름을 누락해 서버가 보완: {name}")

    raw_wardrobe_at_start = raw.get("wardrobe_at_start", [])
    if not isinstance(raw_wardrobe_at_start, list):
        print(
            "[ILLUST_CONTEXT:OUTFIT] wardrobe_at_start가 list가 아님: "
            f"type={type(raw_wardrobe_at_start).__name__}"
        )
        return None
    wardrobe_at_start = []
    baseline_names = set()
    for index, item in enumerate(raw_wardrobe_at_start, start=1):
        if not isinstance(item, dict):
            warnings.append(f"CURRENT 시작 복장 형식 오류로 폐기: index={index}")
            continue
        name = normalize_name(item.get("character") or item.get("name"))
        state = str(item.get("state") or item.get("wardrobe_state") or "").strip()
        if not name or not state:
            warnings.append(
                f"CURRENT 시작 복장 필수값 오류로 폐기: "
                f"index={index}, character={name!r}"
            )
            continue
        if name.casefold() not in current_names:
            warnings.append(
                f"CURRENT 캐릭터 밖 시작 복장으로 폐기: "
                f"index={index}, character={name!r}"
            )
            continue
        if name.casefold() in baseline_names:
            warnings.append(f"CURRENT 시작 복장 중복으로 후속 항목 폐기: {name}")
            continue
        baseline_names.add(name.casefold())
        wardrobe_at_start.append({
            "character": name,
            "state": state,
        })
    for current_character in current_characters:
        name = str(current_character.get("name") or "").strip()
        if name and name.casefold() not in baseline_names:
            warning = f"CURRENT 시작 복장 누락으로 추적/default 폴백 사용: character={name}"
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT] {warning}")

    wardrobe_events = []
    changing_operations = {
        "wear", "add", "remove", "replace", "set", "open", "close",
        "adjust", "nude", "topless", "bottomless", "reset_default",
        "contextual_reset",
    }
    for index, item in enumerate(raw.get("wardrobe_events") or [], start=1):
        if not isinstance(item, dict):
            warnings.append(f"복장 사건 형식 오류로 폐기: index={index}")
            continue
        segment_id = str(item.get("segment_id") or "").strip()
        name = normalize_name(item.get("character") or item.get("name"))
        operation = str(item.get("operation") or "keep").strip().lower()
        evidence = str(item.get("evidence") or "").strip()
        wardrobe_change = str(item.get("wardrobe_change") or "").strip()
        # legacy 출력 호환: 신규 OUTFIT은 wardrobe_change만 내고 items를 비운다.
        # 하위 호환을 위해 items는 계속 파싱해 둔다(과거 기록/구 출력).
        items = item.get("items") or []
        if not isinstance(items, list):
            items = [items]
        items = [str(value).strip() for value in items if str(value).strip()]
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 1.0))))
        except Exception:
            confidence = 0.0
        if not name:
            warnings.append(f"복장 사건 캐릭터 없어 폐기: index={index}")
            continue
        if name.casefold() not in current_names:
            warnings.append(
                f"CURRENT 캐릭터 밖 복장 사건으로 폐기: "
                f"index={index}, character={name!r}"
            )
            continue
        if operation in changing_operations:
            segment = segments.get(segment_id) or {}
            segment_text = str(segment.get("text") or "")
            segment_match = _analysis_evidence_matches_segment(
                evidence,
                segment_text,
            )
            context_match = bool(
                segment_id in segments
                and evidence
                and _analysis_evidence_matches_context(
                    evidence,
                    current_context,
                    int(segment.get("start") or 0),
                )
            )
            if not segment_id or not evidence or not (segment_match or context_match):
                warnings.append(
                    f"복장 변경 근거 불일치로 폐기: character={name}, operation={operation}, "
                    f"segment={segment_id!r}"
                )
                continue
            if not segment_match and context_match:
                warnings.append(
                    f"복장 변경 근거가 여러 CURRENT segment에 걸쳐 있어 보존: "
                    f"character={name}, operation={operation}, segment={segment_id!r}"
                )
        if confidence < 0.70:
            warnings.append(
                f"복장 사건 신뢰도 낮아 폐기: {name}/{operation}={confidence:.2f}"
            )
            continue
        wardrobe_events.append({
            "segment_id": segment_id,
            "character": name,
            "operation": operation,
            "wardrobe_change": wardrobe_change,
            "items": items,
            "state_after": deepcopy(item.get("state_after")),
            "evidence": evidence,
            "confidence": confidence,
        })

    initial_visual_bases = []
    for index, item in enumerate(raw.get("initial_visual_bases") or [], start=1):
        if not isinstance(item, dict):
            warnings.append(f"초기 외형 기반 형식 오류로 폐기: index={index}")
            continue
        anchor_segment = str(item.get("anchor_segment") or "").strip()
        name = normalize_name(item.get("character") or item.get("name"))
        profile_id = str(
            item.get("target_visual_profile_id")
            or item.get("visual_profile_id")
            or ""
        ).strip()
        initial_state = str(
            item.get("initial_state") or item.get("visual_state") or ""
        ).strip()
        raw_evidence = item.get("evidence") or []
        if isinstance(raw_evidence, dict):
            raw_evidence = [raw_evidence]
        try:
            confidence = max(
                0.0, min(1.0, float(item.get("confidence", 1.0)))
            )
        except Exception:
            confidence = 0.0
        character_profiles = _authority_values_for_name(visual_profiles or {}, name)
        if not name or not isinstance(character_profiles, dict):
            warnings.append(
                f"초기 외형 기반 캐릭터/카탈로그 없음으로 폐기: "
                f"index={index}, character={name!r}"
            )
            continue
        profile = profile_by_id(character_profiles, profile_id)
        if profile is None:
            warnings.append(
                f"등록되지 않은 초기 외형 프로필 ID로 폐기: "
                f"character={name}, profile={profile_id!r}"
            )
            continue
        default_profile_id = str(
            character_profiles.get("default_visual_profile_id") or ""
        ).strip()
        if profile_id == default_profile_id:
            warnings.append(
                f"기본 프로필과 같은 초기 외형 기반은 불필요해 폐기: "
                f"character={name}, profile={profile_id!r}"
            )
            continue
        evidence_items = []
        evidence_valid = isinstance(raw_evidence, list) and bool(raw_evidence)
        has_anchor_evidence = False
        for evidence_index, evidence_item in enumerate(raw_evidence, start=1):
            if not isinstance(evidence_item, dict):
                warnings.append(
                    f"초기 외형 근거 형식 오류: character={name}, "
                    f"evidence_index={evidence_index}"
                )
                evidence_valid = False
                break
            evidence_segment = str(evidence_item.get("segment_id") or "").strip()
            excerpt = str(
                evidence_item.get("excerpt") or evidence_item.get("evidence") or ""
            ).strip()
            segment_text = str(
                (segments.get(evidence_segment) or {}).get("text") or ""
            )
            if (
                not evidence_segment
                or not excerpt
                or not _analysis_evidence_matches_segment(excerpt, segment_text)
            ):
                warnings.append(
                    f"초기 외형 근거 불일치: character={name}, "
                    f"segment={evidence_segment!r}"
                )
                evidence_valid = False
                break
            if evidence_segment == anchor_segment:
                has_anchor_evidence = True
            evidence_items.append({
                "segment_id": evidence_segment,
                "excerpt": excerpt,
            })
        if (
            anchor_segment not in segments
            or not initial_state
            or not evidence_valid
            or not has_anchor_evidence
        ):
            warnings.append(
                f"초기 외형 기반 필수값/앵커 근거 오류로 폐기: "
                f"character={name}, profile={profile_id!r}, "
                f"anchor={anchor_segment!r}"
            )
            continue
        if confidence < 0.70:
            warnings.append(
                f"초기 외형 기반 신뢰도 낮아 폐기: "
                f"{name}/{profile_id}={confidence:.2f}"
            )
            continue
        initial_visual_bases.append({
            "character": name,
            "target_visual_profile_id": profile_id,
            "initial_state": initial_state,
            "anchor_segment": anchor_segment,
            "evidence": evidence_items,
            "confidence": confidence,
        })

    profile_events = []
    visual_base_events = []
    for index, item in enumerate(raw.get("profile_events") or [], start=1):
        if not isinstance(item, dict):
            warning = f"프로필 사건 형식 오류로 폐기: index={index}"
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT] {warning}; item={item!r}")
            continue
        segment_id = str(item.get("segment_id") or "").strip().upper()
        name = normalize_name(item.get("character") or item.get("name"))
        profile_name = str(item.get("profile") or "").strip()
        state_description = str(item.get("state") or "").strip()
        try:
            confidence = max(
                0.0, min(1.0, float(item.get("confidence", 1.0)))
            )
        except Exception:
            confidence = 0.0
        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            name,
        )
        if not name or not isinstance(character_profiles, dict):
            warning = (
                f"프로필 사건 캐릭터/카탈로그 없음으로 폐기: "
                f"index={index}, character={name!r}"
            )
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT] {warning}")
            continue
        try:
            profile = profile_by_name(character_profiles, profile_name)
        except Exception as exc:
            warnings.append(
                f"프로필 의미 이름 해석 실패로 폐기: "
                f"character={name}, profile={profile_name!r}, error={exc}"
            )
            print(
                f"[ILLUST_CONTEXT:OUTFIT] 프로필 의미 이름 해석 실패: "
                f"character={name}, profile={profile_name!r}, error={exc}"
            )
            traceback.print_exc()
            continue
        if profile is None:
            warning = (
                f"등록되지 않은 프로필 의미 이름으로 폐기: "
                f"character={name}, profile={profile_name!r}"
            )
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT] {warning}")
            continue
        if not state_description:
            warning = (
                f"프로필 자연어 상태 없음으로 폐기: "
                f"character={name}, profile={profile_name!r}, segment={segment_id!r}"
            )
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT] {warning}")
            continue
        if confidence < 0.70:
            warning = (
                f"프로필 사건 신뢰도 낮아 폐기: "
                f"{name}/{profile_name}={confidence:.2f}"
            )
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT] {warning}")
            continue
        normalized_profile_event = {
            "segment_id": segment_id,
            "character": name,
            "profile": profile_name,
            "state": state_description,
            "confidence": confidence,
        }
        profile_events.append(normalized_profile_event)
        profile_id = str(profile.get("id") or "").strip()
        if segment_id == "START":
            duplicate_start = next((
                existing
                for existing in initial_visual_bases
                if str(existing.get("character") or "").casefold()
                == name.casefold()
            ), None)
            if duplicate_start is not None:
                warning = (
                    f"캐릭터별 START 프로필 중복으로 후속 항목 폐기: "
                    f"character={name}, profile={profile_name!r}"
                )
                warnings.append(warning)
                print(f"[ILLUST_CONTEXT:OUTFIT] {warning}")
                profile_events.pop()
                continue
            initial_visual_bases.append({
                "character": name,
                "target_visual_profile_id": profile_id,
                "visual_profile_name": profile_name,
                "initial_state": state_description,
                "anchor_segment": "START",
                "evidence": [],
                "confidence": confidence,
            })
            continue
        if segment_id not in segments:
            warning = (
                f"프로필 변경 segment 오류로 폐기: character={name}, "
                f"profile={profile_name!r}, segment={segment_id!r}"
            )
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT] {warning}")
            profile_events.pop()
            continue
        visual_base_events.append({
            "segment_id": segment_id,
            "character": name,
            "target_visual_profile_id": profile_id,
            "visual_profile_name": profile_name,
            "visual_change": state_description,
            "evidence": "",
            "confidence": confidence,
        })

    # 구형 OUTFIT 출력과 저장된 추적 결과를 위한 내부 호환 경로다. 신규 프롬프트는
    # profile_events만 출력하며 아래 visual_base_events에는 내부 ID가 노출되지 않는다.
    for index, item in enumerate(raw.get("visual_base_events") or [], start=1):
        if not isinstance(item, dict):
            warnings.append(f"외형 기반 사건 형식 오류로 폐기: index={index}")
            continue
        segment_id = str(item.get("segment_id") or "").strip()
        name = normalize_name(item.get("character") or item.get("name"))
        profile_id = str(
            item.get("target_visual_profile_id")
            or item.get("visual_profile_id")
            or ""
        ).strip()
        visual_change = str(item.get("visual_change") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        try:
            confidence = max(
                0.0, min(1.0, float(item.get("confidence", 1.0)))
            )
        except Exception:
            confidence = 0.0
        character_profiles = _authority_values_for_name(visual_profiles or {}, name)
        if not name or not isinstance(character_profiles, dict):
            warnings.append(
                f"외형 기반 사건 캐릭터/카탈로그 없음으로 폐기: "
                f"index={index}, character={name!r}"
            )
            continue
        profile = profile_by_id(character_profiles, profile_id)
        if profile is None:
            warnings.append(
                f"등록되지 않은 외형 프로필 ID로 폐기: "
                f"character={name}, profile={profile_id!r}"
            )
            continue
        segment_text = str((segments.get(segment_id) or {}).get("text") or "")
        if (
            not segment_id
            or not visual_change
            or not evidence
            or not _analysis_evidence_matches_segment(evidence, segment_text)
        ):
            warnings.append(
                f"외형 기반 변경 근거 불일치로 폐기: character={name}, "
                f"profile={profile_id!r}, segment={segment_id!r}"
            )
            continue
        if confidence < 0.70:
            warnings.append(
                f"외형 기반 사건 신뢰도 낮아 폐기: "
                f"{name}/{profile_id}={confidence:.2f}"
            )
            continue
        visual_base_events.append({
            "segment_id": segment_id,
            "character": name,
            "target_visual_profile_id": profile_id,
            "visual_change": visual_change,
            "evidence": evidence,
            "confidence": confidence,
        })

    # hairstyle_events: hairstyle "arrangement" 전환만 추적(ponytail/twintails/braid/
    # hair bun/hair down/two side up/side ponytail 등). 색/길이/앞머리/눈/신체/종족/스킨/
    # 흉터/포즈/의상/헤어 액세서리는 fixed appearance 또는 wardrobe 영역이므로 여기서 다루지
    # 않는다. 서버는 의미를 해석하지 않고 이벤트만 보존한다.
    hairstyle_events = []
    hairstyle_operations = {"replace", "add", "remove", "reset_default"}
    for index, item in enumerate(raw.get("hairstyle_events") or [], start=1):
        if not isinstance(item, dict):
            warnings.append(f"헤어스타일 사건 형식 오류로 폐기: index={index}")
            continue
        segment_id = str(item.get("segment_id") or "").strip()
        name = normalize_name(item.get("character") or item.get("name"))
        operation = str(item.get("operation") or "").strip().lower()
        evidence = str(item.get("evidence") or "").strip()
        hairstyle_change = str(item.get("hairstyle_change") or "").strip()
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 1.0))))
        except Exception:
            confidence = 0.0
        if not name:
            warnings.append(f"헤어스타일 사건 캐릭터 없어 폐기: index={index}")
            continue
        if name.casefold() not in current_names:
            warnings.append(
                f"CURRENT 캐릭터 밖 헤어스타일 사건으로 폐기: "
                f"index={index}, character={name!r}"
            )
            continue
        if operation not in hairstyle_operations:
            warnings.append(
                f"헤어스타일 사건 operation 미지정/범위외로 폐기: character={name}, "
                f"operation={operation!r}"
            )
            continue
        segment_text = str((segments.get(segment_id) or {}).get("text") or "")
        if (
            not segment_id
            or not hairstyle_change
            or not evidence
            or not _analysis_evidence_matches_segment(evidence, segment_text)
        ):
            warnings.append(
                f"헤어스타일 변경 근거 불일치로 폐기: character={name}, operation={operation}, "
                f"segment={segment_id!r}"
            )
            continue
        if confidence < 0.70:
            warnings.append(
                f"헤어스타일 사건 신뢰도 낮아 폐기: {name}/{operation}={confidence:.2f}"
            )
            continue
        hairstyle_events.append({
            "segment_id": segment_id,
            "character": name,
            "operation": operation,
            "hairstyle_change": hairstyle_change,
            "evidence": evidence,
            "confidence": confidence,
        })

    unresolved = raw.get("unresolved_references") or []
    if not isinstance(unresolved, list):
        unresolved = [unresolved]
    unresolved = [deepcopy(item) for item in unresolved if item not in (None, "", {})]
    if unresolved:
        fallback_errors.append(f"미해결 지칭 {len(unresolved)}건")
    if character_names.strip() and not current_characters:
        fallback_errors.append("현재 캐릭터 목록이 비어 있음")

    # 구형 OUTFIT/저장 응답에 profile_events가 실제로 포함된 경우에만 완전성을
    # 검사한다. 신규 OUTFIT은 프로필 결정을 전용 직렬 단계에 넘기므로 이 키를
    # 출력하지 않는다.
    if "profile_events" in raw:
        start_profile_characters = {
            str(item.get("character") or "").strip().casefold()
            for item in profile_events
            if str(item.get("segment_id") or "").strip().upper() == "START"
        }
        for current_character in current_characters:
            name = str(current_character.get("name") or "").strip()
            character_profiles = _visual_profiles_for_name(visual_profiles, name)
            if (
                character_profiles is None
                or len(character_profiles.get("profiles") or []) <= 1
                or name.casefold() in start_profile_characters
            ):
                continue
            warning = (
                f"다중 프로필 CURRENT 캐릭터의 START 판정 누락: character={name}; "
                "이전 추적 프로필 또는 등록 기본 프로필 폴백 사용"
            )
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT] {warning}")

    return {
        "reference_assignments": assignments,
        "history_characters": history_characters,
        "current_characters": current_characters,
        "wardrobe_at_start": wardrobe_at_start,
        "wardrobe_events": wardrobe_events,
        "profile_events": profile_events,
        "initial_visual_bases": initial_visual_bases,
        "visual_base_events": visual_base_events,
        "hairstyle_events": hairstyle_events,
        "unresolved_references": unresolved,
        "validation_warnings": warnings,
        "fallback_errors": fallback_errors,
        # 기존 소비자를 위해 한 목록도 유지하되, 폴백 여부는 치명 오류만 결정한다.
        "validation_errors": warnings + fallback_errors,
        "fallback_required": bool(fallback_errors),
    }


def _parse_legacy_profile_resolution(
    text: str,
    current_context: str,
    segments: dict[str, dict],
    current_characters: list[dict],
    visual_profiles: dict[str, dict] | None,
) -> dict | None:
    """Validate the dedicated PROFILE-RESOLVE four-field event schema."""
    raw = _json_object_from_text(text)
    if raw is None:
        print("[ILLUST_CONTEXT:PROFILE_RESOLVE] JSON object 파싱 실패")
        return None
    raw_events = raw.get("profile_events")
    if not isinstance(raw_events, list):
        print(
            "[ILLUST_CONTEXT:PROFILE_RESOLVE] profile_events가 list가 아님: "
            f"type={type(raw_events).__name__}"
        )
        return None

    names = []
    normalized_current = []
    for item in current_characters or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            confidence = item.get("confidence", 1.0)
        else:
            name = str(item or "").strip()
            confidence = 1.0
        if not name or name.casefold() in {value.casefold() for value in names}:
            continue
        names.append(name)
        normalized_current.append({"name": name, "confidence": confidence})

    selected_profiles = _selected_visual_profiles(
        visual_profiles,
        names,
        multiple_only=True,
    )
    composite = {
        "reference_assignments": [],
        "history_characters": [],
        "current_characters": normalized_current,
        "profile_events": raw_events,
        "wardrobe_events": [],
        "hairstyle_events": [],
        "unresolved_references": [],
    }
    parsed = parse_call1_analysis(
        json.dumps(composite, ensure_ascii=False),
        current_context,
        segments,
        ", ".join(names),
        visual_profiles=selected_profiles,
    )
    if parsed is None:
        print("[ILLUST_CONTEXT:PROFILE_RESOLVE] 프로필 사건 검증 실패")
        return None
    parsed_events = list(parsed.get("profile_events") or [])
    start_counts = {
        name.casefold(): sum(
            1
            for event in parsed_events
            if str(event.get("character") or "").strip().casefold() == name.casefold()
            and str(event.get("segment_id") or "").strip().upper() == "START"
        )
        for name in selected_profiles
    }
    incomplete = [
        name
        for name in selected_profiles
        if start_counts.get(name.casefold(), 0) != 1
    ]
    if incomplete:
        print(
            "[ILLUST_CONTEXT:PROFILE_RESOLVE] 다중 프로필 캐릭터 START 완전성 검증 실패: "
            f"characters={incomplete}, counts={start_counts}"
        )
        return None
    return {
        "profile_events": parsed_events,
        "initial_visual_bases": list(parsed.get("initial_visual_bases") or []),
        "visual_base_events": list(parsed.get("visual_base_events") or []),
        "validation_warnings": list(parsed.get("validation_warnings") or []),
        "validation_errors": list(parsed.get("validation_errors") or []),
    }


def _materialize_profile_character_records(
    characters: list[dict],
    visual_profiles: dict[str, dict] | None,
    *,
    uncertainties: list[str] | None = None,
    warnings: list[str] | None = None,
    repair_requests: list[dict] | None = None,
) -> dict:
    """Convert the compact LLM contract into the existing internal profile timeline."""
    profile_events = []
    initial_visual_bases = []
    visual_base_events = []
    current_characters = []
    history_characters = []

    for character in characters or []:
        if not isinstance(character, dict):
            continue
        name = str(character.get("name") or "").strip()
        if not name:
            continue
        current_characters.append({"name": name, "confidence": 1.0})
        if bool(character.get("in_history")):
            history_characters.append(name)
        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            name,
        )
        if not isinstance(character_profiles, dict):
            print(
                "[ILLUST_CONTEXT:PROFILE_RESOLVE] 내부 변환용 캐릭터 프로필 없음: "
                f"character={name!r}"
            )
            continue
        for event in character.get("profile_timeline") or []:
            if not isinstance(event, dict):
                continue
            segment_id = str(event.get("at") or "").strip().upper()
            profile_id = str(event.get("profile_id") or "").strip()
            reason = str(event.get("reason") or "").strip()
            profile = profile_by_id(character_profiles, profile_id)
            if not segment_id or not isinstance(profile, dict):
                print(
                    "[ILLUST_CONTEXT:PROFILE_RESOLVE] 내부 변환용 프로필 사건 오류: "
                    f"character={name!r}, at={segment_id!r}, profile_id={profile_id!r}"
                )
                continue
            profile_names = visual_profile_names(profile)
            profile_name = (
                profile_names[0]
                if profile_names
                else str(profile.get("label") or profile_id).strip()
            )
            profile_events.append({
                "segment_id": segment_id,
                "character": name,
                "profile": profile_name,
                "profile_id": profile_id,
                "state": reason,
                "confidence": 1.0,
            })
            if segment_id == "START":
                initial_visual_bases.append({
                    "character": name,
                    "target_visual_profile_id": profile_id,
                    "visual_profile_name": profile_name,
                    "initial_state": reason,
                    "anchor_segment": "START",
                    "evidence": [reason] if reason else [],
                    "confidence": 1.0,
                })
            else:
                visual_base_events.append({
                    "segment_id": segment_id,
                    "character": name,
                    "target_visual_profile_id": profile_id,
                    "visual_profile_name": profile_name,
                    "visual_change": reason,
                    "evidence": reason,
                    "confidence": 1.0,
                })

    normalized_warnings = list(warnings or [])
    normalized_uncertainties = [
        str(item).strip()
        for item in uncertainties or []
        if str(item).strip()
    ]
    return {
        "characters": deepcopy(characters or []),
        "history_characters": history_characters,
        "current_characters": current_characters,
        "uncertainties": normalized_uncertainties,
        "profile_events": profile_events,
        "initial_visual_bases": initial_visual_bases,
        "visual_base_events": visual_base_events,
        "repair_requests": deepcopy(repair_requests or []),
        "validation_warnings": normalized_warnings,
        "validation_errors": normalized_warnings,
    }


def _profile_resolution_name_contract_issues(
    text: str,
    canonical_names: list[str],
) -> list[dict]:
    """Return rejected character items without guessing identity from name text."""
    raw = _json_object_from_text(text)
    if raw is None:
        return [{"source_index": 0, "reason": "JSON object 없음", "item": None}]
    raw_characters = raw.get("characters")
    if not isinstance(raw_characters, list):
        return [{
            "source_index": 0,
            "reason": "characters가 list가 아님",
            "item": deepcopy(raw_characters),
        }]

    allowed = {
        str(name or "").strip()
        for name in canonical_names or []
        if str(name or "").strip()
    }
    issues = []
    seen = set()
    for index, item in enumerate(raw_characters, start=1):
        if not isinstance(item, dict):
            issues.append({
                "source_index": index,
                "reason": f"characters[{index}]가 object가 아님",
                "item": deepcopy(item),
            })
            continue
        name = str(item.get("name") or "").strip()
        if name not in allowed:
            issues.append({
                "source_index": index,
                "reason": (
                    f"characters[{index}].name={name!r}가 "
                    "정확한 등록명 목록에 없음"
                ),
                "item": deepcopy(item),
            })
            continue
        if name in seen:
            issues.append({
                "source_index": index,
                "reason": f"characters[{index}].name={name!r} 중복",
                "item": deepcopy(item),
            })
            continue
        seen.add(name)
    return issues


def _profile_resolution_name_contract_errors(
    text: str,
    canonical_names: list[str],
) -> list[str]:
    return [
        str(issue.get("reason") or "등록명 계약 위반")
        for issue in _profile_resolution_name_contract_issues(
            text,
            canonical_names,
        )
    ]


def parse_profile_resolution(
    text: str,
    current_context: str,
    segments: dict[str, dict],
    current_characters: list[dict] | None,
    visual_profiles: dict[str, dict] | None,
    *,
    profile_inference_enabled: bool = True,
) -> dict | None:
    """Validate the compact CURRENT-character/profile contract.

    Legacy four-field ``profile_events`` output remains readable for stored records
    and older callers, while new calls use ``characters[].profile_timeline``.
    """
    raw = _json_object_from_text(text)
    if raw is None:
        print("[ILLUST_CONTEXT:PROFILE_RESOLVE] JSON object 파싱 실패")
        return None
    if "characters" not in raw and "profile_events" in raw:
        return _parse_legacy_profile_resolution(
            text,
            current_context,
            segments,
            current_characters or [],
            visual_profiles,
        )

    raw_characters = raw.get("characters")
    raw_uncertainties = raw.get("uncertainties")
    if not isinstance(raw_characters, list) or not isinstance(raw_uncertainties, list):
        print(
            "[ILLUST_CONTEXT:PROFILE_RESOLVE] 최상위 스키마 오류: "
            f"characters={type(raw_characters).__name__}, "
            f"uncertainties={type(raw_uncertainties).__name__}"
        )
        return None

    registered_names = {
        str(name or "").strip().casefold(): str(name or "").strip()
        for name, value in (visual_profiles or {}).items()
        if str(name or "").strip() and isinstance(value, dict)
    }
    warnings = []
    name_contract_errors = []
    repair_requests = []
    normalized_characters = []
    seen_names = set()
    segment_rank = {
        segment_id: index
        for index, segment_id in enumerate(segments or {}, start=1)
    }

    for index, item in enumerate(raw_characters, start=1):
        if not isinstance(item, dict):
            warning = f"등장인물 항목 형식 오류로 폐기: index={index}, item={item!r}"
            warnings.append(warning)
            name_contract_errors.append(warning)
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
            continue
        raw_name = str(item.get("name") or "").strip()
        name = registered_names.get(raw_name.casefold(), "")
        if not name:
            warning = f"등록되지 않은 CURRENT 캐릭터로 폐기: index={index}, name={raw_name!r}"
            warnings.append(warning)
            name_contract_errors.append(warning)
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
            continue
        folded = name.casefold()
        if folded in seen_names:
            warning = f"중복 CURRENT 캐릭터로 후속 항목 폐기: character={name}"
            warnings.append(warning)
            name_contract_errors.append(warning)
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
            continue
        seen_names.add(folded)

        if profile_inference_enabled:
            in_history = item.get("in_history")
            if not isinstance(in_history, bool):
                warning = (
                    f"in_history가 bool이 아니어서 false 사용: "
                    f"character={name}, value={in_history!r}"
                )
                warnings.append(warning)
                print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
                in_history = False
        else:
            # CHARACTER-RESOLVE sees CURRENT only. Prior participation is derived
            # deterministically from the server's tracked state after this parse.
            in_history = False

        raw_timeline = item.get("profile_timeline")
        timeline_format_error = not isinstance(raw_timeline, list)
        if timeline_format_error:
            warning = (
                f"profile_timeline이 list가 아니어서 빈 목록 사용: "
                f"character={name}, type={type(raw_timeline).__name__}"
            )
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
            raw_timeline = []

        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            name,
        )
        registered_profiles = (
            list(character_profiles.get("profiles") or [])
            if isinstance(character_profiles, dict)
            else []
        )
        is_multi_profile = len(registered_profiles) > 1
        normalized_timeline = []
        invalid_reasons = []
        seen_points: dict[str, str] = {}

        if not profile_inference_enabled:
            if raw_timeline:
                warning = (
                    f"프로필 추론 OFF인데 타임라인이 있어 무시: "
                    f"character={name}, events={len(raw_timeline)}"
                )
                warnings.append(warning)
                print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
        elif not is_multi_profile:
            if raw_timeline:
                warning = (
                    f"단일 프로필 캐릭터 타임라인은 불필요해 무시: "
                    f"character={name}, events={len(raw_timeline)}"
                )
                warnings.append(warning)
                print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
        else:
            if timeline_format_error:
                invalid_reasons.append("profile_timeline 형식 오류")
            for event_index, event in enumerate(raw_timeline, start=1):
                if not isinstance(event, dict):
                    invalid_reasons.append(f"event[{event_index}] 형식 오류")
                    continue
                at = str(event.get("at") or "").strip().upper()
                profile_ref = str(event.get("profile_ref") or "").strip()
                profile_id = ""
                if at != "START" and at not in (segments or {}):
                    invalid_reasons.append(
                        f"event[{event_index}] 존재하지 않는 구간 {at!r}"
                    )
                    continue
                profile = None
                if profile_ref:
                    match = re.fullmatch(r"\[(\d+)\]", profile_ref)
                    profile_number = int(match.group(1)) if match else 0
                    if 1 <= profile_number <= len(registered_profiles):
                        profile = registered_profiles[profile_number - 1]
                        profile_id = str(profile.get("id") or "").strip()
                    else:
                        invalid_reasons.append(
                            f"event[{event_index}] 등록되지 않은 profile_ref "
                            f"{profile_ref!r}"
                        )
                        continue
                else:
                    # Stored records and pre-migration callers may still contain the
                    # former internal-ID contract. New resolver prompts never expose it.
                    profile_id = str(event.get("profile_id") or "").strip()
                    profile = profile_by_id(character_profiles, profile_id)
                if not isinstance(profile, dict):
                    invalid_reasons.append(
                        f"event[{event_index}] 등록되지 않은 profile_ref/profile_id "
                        f"{profile_ref or profile_id!r}"
                    )
                    continue
                previous_id = seen_points.get(at)
                if previous_id is not None:
                    if previous_id != profile_id:
                        invalid_reasons.append(
                            f"동일 시점 프로필 충돌 at={at}, ids={[previous_id, profile_id]}"
                        )
                    else:
                        warnings.append(
                            f"동일 프로필 사건 중복 제거: character={name}, at={at}, "
                            f"profile_id={profile_id}"
                        )
                    continue
                seen_points[at] = profile_id
                reason = str(event.get("reason") or "").strip()
                if not reason:
                    warning = (
                        f"프로필 판단 reason 누락: character={name}, "
                        f"at={at}, profile_ref={profile_ref or profile_id!r}"
                    )
                    warnings.append(warning)
                    print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
                normalized_timeline.append({
                    "at": at,
                    "profile_id": profile_id,
                    "reason": reason,
                })

            normalized_timeline.sort(
                key=lambda event: (
                    -1 if event["at"] == "START" else segment_rank.get(event["at"], 10**9)
                )
            )
            start_count = sum(
                1 for event in normalized_timeline if event.get("at") == "START"
            )
            if start_count != 1:
                invalid_reasons.append(f"START 개수 오류: {start_count}")

        normalized_characters.append({
            "name": name,
            "in_history": in_history,
            "profile_timeline": normalized_timeline,
        })
        if invalid_reasons:
            request = {
                "character": name,
                "reasons": invalid_reasons,
                "raw_profile_timeline": deepcopy(raw_timeline),
            }
            repair_requests.append(request)
            warning = f"프로필 타임라인 부분 교정 필요: {request}"
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")

    if name_contract_errors:
        print(
            "[ILLUST_CONTEXT:PROFILE_RESOLVE] 등록명 계약 위반으로 응답 전체 거부: "
            f"errors={name_contract_errors}"
        )
        return None

    uncertainties = []
    for index, item in enumerate(raw_uncertainties, start=1):
        if not isinstance(item, str) or not item.strip():
            warning = f"불확실성 항목 형식 오류로 폐기: index={index}, item={item!r}"
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
            continue
        uncertainties.append(item.strip())

    if registered_names and not normalized_characters:
        warning = "CURRENT 등장인물 결과가 비어 있음"
        warnings.append(warning)
        print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")

    return _materialize_profile_character_records(
        normalized_characters,
        visual_profiles,
        uncertainties=uncertainties,
        warnings=warnings,
        repair_requests=repair_requests,
    )


def _empty_profile_result() -> dict:
    return {
        "characters": [],
        "history_characters": [],
        "current_characters": [],
        "uncertainties": [],
        "profile_events": [],
        "initial_visual_bases": [],
        "visual_base_events": [],
        "repair_requests": [],
        "validation_warnings": [],
        "validation_errors": [],
    }


def _mark_current_characters_with_tracked_history(
    character_result: dict,
    previous_state: dict | None,
    visual_profiles: dict[str, dict] | None,
) -> dict:
    """Derive prior participation without exposing PAST narrative to the LLM."""
    tracked_names = {
        str(value.get("canonical_name") or key).strip().casefold()
        for key, value in (
            previous_state.items() if isinstance(previous_state, dict) else []
        )
        if isinstance(value, dict)
        and str(value.get("canonical_name") or key).strip()
    }
    characters = []
    for item in (character_result or {}).get("characters") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        character = deepcopy(item)
        character["in_history"] = name.casefold() in tracked_names
        characters.append(character)

    history_names = [
        item["name"] for item in characters if bool(item.get("in_history"))
    ]
    print(
        "[ILLUST_CONTEXT:CHARACTER_RESOLVE] 서버 추적 상태로 과거 참여 표시: "
        f"current={[item['name'] for item in characters]}, history={history_names}"
    )
    return _materialize_profile_character_records(
        characters,
        visual_profiles,
        uncertainties=list((character_result or {}).get("uncertainties") or []),
        warnings=list((character_result or {}).get("validation_warnings") or []),
        repair_requests=list((character_result or {}).get("repair_requests") or []),
    )


def _default_profile_result(
    visual_profiles: dict[str, dict] | None,
    characters: list[dict] | None = None,
) -> dict:
    """Lock resolved CURRENT multi-profile characters to registered defaults."""
    if characters is None:
        source_characters = [
            {
                "name": str(name or "").strip(),
                "in_history": False,
                "profile_timeline": [],
            }
            for name, value in (visual_profiles or {}).items()
            if str(name or "").strip()
            and isinstance(value, dict)
            and len(value.get("profiles") or []) > 1
        ]
    else:
        source_characters = [
            {
                "name": str(item.get("name") or "").strip(),
                "in_history": bool(item.get("in_history")),
                "profile_timeline": [],
            }
            for item in characters
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]

    normalized = []
    warnings = []
    for item in source_characters:
        name = item["name"]
        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            name,
        )
        if not isinstance(character_profiles, dict):
            warning = f"기본 프로필 고정용 캐릭터 없음: character={name!r}"
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
            continue
        timeline = []
        if len(character_profiles.get("profiles") or []) > 1:
            try:
                base = resolve_visual_base(character_profiles)
                profile_id = str(base.get("visual_profile_id") or "").strip()
                profile = profile_by_id(character_profiles, profile_id)
            except Exception as e:
                print(
                    "[ILLUST_CONTEXT:PROFILE_RESOLVE] 기본 프로필 고정 실패: "
                    f"character={name!r}, error={e}"
                )
                traceback.print_exc()
                continue
            if not profile_id or not isinstance(profile, dict):
                warning = (
                    f"기본 프로필 고정값 없음: character={name!r}, "
                    f"profile_id={profile_id!r}"
                )
                warnings.append(warning)
                print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
                continue
            timeline = [{"at": "START", "profile_id": profile_id}]
        normalized.append({
            "name": name,
            "in_history": bool(item.get("in_history")),
            "profile_timeline": timeline,
        })

    return _materialize_profile_character_records(
        normalized,
        visual_profiles,
        warnings=warnings,
    )


def _profile_previous_id(
    previous_state: dict | None,
    character_name: str,
    character_profiles: dict,
) -> str:
    states = previous_state if isinstance(previous_state, dict) else {}
    tracked = next((
        value
        for key, value in states.items()
        if isinstance(value, dict)
        and str(value.get("canonical_name") or key).strip().casefold()
        == str(character_name or "").strip().casefold()
    ), None)
    profile_id = str((tracked or {}).get("active_visual_profile_id") or "").strip()
    return profile_id if profile_by_id(character_profiles, profile_id) is not None else ""


def _apply_profile_resolution_fallbacks(
    profile_result: dict,
    visual_profiles: dict[str, dict] | None,
    previous_state: dict | None,
    *,
    default_only: bool,
) -> dict:
    """Fill only missing START decisions; valid LLM events remain untouched."""
    characters = deepcopy(profile_result.get("characters") or [])
    warnings = list(profile_result.get("validation_warnings") or [])
    for character in characters:
        if not isinstance(character, dict):
            continue
        name = str(character.get("name") or "").strip()
        character_profiles = _authority_values_for_name(
            visual_profiles or {},
            name,
        )
        if (
            not name
            or not isinstance(character_profiles, dict)
            or len(character_profiles.get("profiles") or []) <= 1
        ):
            character["profile_timeline"] = []
            continue
        timeline = list(character.get("profile_timeline") or [])
        starts = [event for event in timeline if event.get("at") == "START"]
        if len(starts) == 1 and not default_only:
            continue
        if default_only:
            profile_id = str(
                character_profiles.get("default_visual_profile_id") or ""
            ).strip()
        else:
            profile_id = _profile_previous_id(
                previous_state,
                name,
                character_profiles,
            ) or str(character_profiles.get("default_visual_profile_id") or "").strip()
        if profile_by_id(character_profiles, profile_id) is None:
            warning = (
                f"START 폴백 프로필 없음: character={name!r}, profile_id={profile_id!r}"
            )
            warnings.append(warning)
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")
            continue
        timeline = [event for event in timeline if event.get("at") != "START"]
        character["profile_timeline"] = [
            {
                "at": "START",
                "profile_id": profile_id,
                "reason": (
                    "프로필 응답이 유효한 START를 제공하지 않아 이전 추적 상태 또는 "
                    "등록 기본 프로필을 안전 폴백으로 적용했다."
                ),
            },
            *timeline,
        ]
        reason = "추론 OFF 기본 프로필" if default_only else "교정 실패 안전 폴백"
        warning = f"{reason} 적용: character={name}, profile_id={profile_id}"
        warnings.append(warning)
        print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {warning}")

    return _materialize_profile_character_records(
        characters,
        visual_profiles,
        uncertainties=list(profile_result.get("uncertainties") or []),
        warnings=warnings,
        repair_requests=list(profile_result.get("repair_requests") or []),
    )


def _filter_profile_result_for_characters(
    profile_result: dict | None,
    character_names: list[str],
) -> dict:
    """Keep profile resolution data only for the selected CURRENT characters."""
    selected = {
        str(name or "").strip().casefold()
        for name in character_names or []
        if str(name or "").strip()
    }
    source = profile_result if isinstance(profile_result, dict) else {}
    if not selected:
        return {
            **_empty_profile_result(),
            "validation_warnings": list(source.get("validation_warnings") or []),
            "validation_errors": list(source.get("validation_errors") or []),
        }

    def belongs(item: dict) -> bool:
        return (
            isinstance(item, dict)
            and str(item.get("character") or "").strip().casefold() in selected
        )

    filtered_characters = [
        deepcopy(item)
        for item in source.get("characters") or []
        if isinstance(item, dict)
        and str(item.get("name") or "").strip().casefold() in selected
    ]
    filtered_history = [
        deepcopy(item)
        for item in source.get("history_characters") or []
        if (
            str(item.get("name") if isinstance(item, dict) else item or "")
            .strip()
            .casefold()
            in selected
        )
    ]
    filtered_current = [
        deepcopy(item)
        for item in source.get("current_characters") or []
        if (
            str(item.get("name") if isinstance(item, dict) else item or "")
            .strip()
            .casefold()
            in selected
        )
    ]
    return {
        "characters": filtered_characters,
        "history_characters": filtered_history,
        "current_characters": filtered_current,
        "uncertainties": list(source.get("uncertainties") or []),
        "profile_events": [
            deepcopy(item) for item in source.get("profile_events") or [] if belongs(item)
        ],
        "initial_visual_bases": [
            deepcopy(item)
            for item in source.get("initial_visual_bases") or []
            if belongs(item)
        ],
        "visual_base_events": [
            deepcopy(item)
            for item in source.get("visual_base_events") or []
            if belongs(item)
        ],
        "validation_warnings": list(source.get("validation_warnings") or []),
        "validation_errors": list(source.get("validation_errors") or []),
        "repair_requests": [
            deepcopy(item)
            for item in source.get("repair_requests") or []
            if belongs(item)
        ],
    }


def profile_authority_text(
    profile_result: dict | None,
    visual_profiles: dict[str, dict] | None,
) -> str:
    """Render preselected profile-specific appearance/outfit authority as prose."""
    lines: list[str] = []
    for event in (profile_result or {}).get("profile_events") or []:
        if not isinstance(event, dict):
            continue
        name = str(event.get("character") or "").strip()
        profile_name = str(event.get("profile") or "").strip()
        segment_id = str(event.get("segment_id") or "").strip().upper()
        state = str(event.get("state") or "").strip()
        character_profiles = _authority_values_for_name(visual_profiles or {}, name)
        if not name or not profile_name or not isinstance(character_profiles, dict):
            print(
                "[ILLUST_CONTEXT:PROFILE_RESOLVE] 권위 문맥 구성용 캐릭터/프로필 없음: "
                f"character={name!r}, profile={profile_name!r}"
            )
            continue
        try:
            profile = profile_by_name(character_profiles, profile_name)
            base = (
                resolve_visual_base(character_profiles, str(profile.get("id") or ""))
                if isinstance(profile, dict)
                else None
            )
        except Exception as e:
            print(
                "[ILLUST_CONTEXT:PROFILE_RESOLVE] 권위 문맥 프로필 해석 실패: "
                f"character={name!r}, profile={profile_name!r}, error={e}"
            )
            traceback.print_exc()
            continue
        if not isinstance(profile, dict):
            print(
                "[ILLUST_CONTEXT:PROFILE_RESOLVE] 권위 문맥 등록 프로필 없음: "
                f"character={name!r}, profile={profile_name!r}"
            )
            continue
        if not isinstance(base, dict):
            print(
                "[ILLUST_CONTEXT:PROFILE_RESOLVE] 권위 문맥 프로필 기반값 없음: "
                f"character={name!r}, profile={profile_name!r}"
            )
            continue
        appearance_values = visual_tag_values(base.get("appearance") or [])
        gender_tag = str(character_profiles.get("gender_tag") or "").strip()
        if gender_tag and gender_tag.casefold() not in {
            value.casefold() for value in appearance_values
        }:
            appearance_values.insert(0, gender_tag)
        appearance = ", ".join(appearance_values) or "(none)"
        outfit = ", ".join(visual_tag_values(base.get("outfit") or [])) or "(none)"
        lines.append(f"### {segment_id} · {name} · {profile_name}")
        if state:
            lines.append(f"- profile state: {state}")
        lines.extend([
            f"- authoritative appearance: {appearance}",
            f"- profile default outfit: {outfit}",
        ])
    if not lines:
        return ""
    return (
        "Profile selection is already complete. The chronological profile-specific "
        "appearance and default outfit below override any generic/default character "
        "reference. Do not choose another profile. A later explicit wardrobe or "
        "hairstyle-arrangement change may overlay this base.\n\n"
        + "\n".join(lines)
    )


def profile_start_reference(
    profile_result: dict | None,
    visual_profiles: dict[str, dict] | None,
) -> str:
    """Build lb.extra-shaped START references from the already selected profiles."""
    blocks: list[str] = []
    for initial in (profile_result or {}).get("initial_visual_bases") or []:
        if not isinstance(initial, dict):
            continue
        name = str(initial.get("character") or "").strip()
        profile_id = str(initial.get("target_visual_profile_id") or "").strip()
        character_profiles = _authority_values_for_name(visual_profiles or {}, name)
        if not name or not profile_id or not isinstance(character_profiles, dict):
            print(
                "[ILLUST_CONTEXT:PROFILE_RESOLVE] START reference 구성값 없음: "
                f"character={name!r}, profile_id={profile_id!r}"
            )
            continue
        try:
            base = resolve_visual_base(character_profiles, profile_id)
        except Exception as e:
            print(
                "[ILLUST_CONTEXT:PROFILE_RESOLVE] START reference 프로필 해석 실패: "
                f"character={name!r}, profile_id={profile_id!r}, error={e}"
            )
            traceback.print_exc()
            continue
        appearance_values = visual_tag_values(base.get("appearance") or [])
        gender_tag = str(character_profiles.get("gender_tag") or "").strip()
        if gender_tag and gender_tag.casefold() not in {
            value.casefold() for value in appearance_values
        }:
            appearance_values.insert(0, gender_tag)
        appearance = ", ".join(appearance_values)
        outfit = ", ".join(visual_tag_values(base.get("outfit") or []))
        blocks.append(
            f"### {name}\n-Name\n{name}\n-Appearance\n{appearance}"
            f"\n-default_outfit\n{outfit}"
        )
    return "\n\n".join(blocks)


def _preselected_profile_inputs(
    profile_result: dict | None,
    visual_profiles: dict[str, dict] | None,
    extra_reference: str,
    extra_costume: str,
) -> tuple[str, str, str]:
    """Compose the profile authority and replacement dictionaries for consumers."""
    authority = profile_authority_text(profile_result, visual_profiles)
    start_reference = profile_start_reference(profile_result, visual_profiles)
    preselected_names = [
        str(item.get("character") or "").strip()
        for item in (profile_result or {}).get("initial_visual_bases") or []
        if isinstance(item, dict) and str(item.get("character") or "").strip()
    ]

    def replace_generic(source: str) -> str:
        return "\n\n".join(
            item
            for item in (
                _exclude_character_reference(source, preselected_names),
                start_reference,
            )
            if str(item or "").strip()
        )

    return authority, replace_generic(extra_reference), replace_generic(extra_costume)


def apply_reference_assignments(
    current_context: str,
    segments: dict[str, dict],
    assignments: list[dict],
) -> tuple[str, list[str], dict[str, str]]:
    """Apply validated OUTFIT assignments by exact segment span, never by keyword list."""
    source = str(current_context or "")
    replacements = []
    errors = []
    ref_values = {}
    for item in assignments or []:
        segment = segments.get(str(item.get("segment_id") or ""))
        if not segment:
            errors.append(f"지칭 치환 segment 없음: {item!r}")
            continue
        surface = str(item.get("surface") or "")
        occurrence = int(item.get("occurrence") or 1)
        segment_text = str(segment.get("text") or "")
        cursor = 0
        found = -1
        for _ in range(occurrence):
            found = segment_text.find(surface, cursor)
            if found < 0:
                break
            cursor = found + len(surface)
        if found < 0:
            errors.append(
                f"지칭 치환 위치 없음: segment={item.get('segment_id')}, "
                f"surface={surface!r}, occurrence={occurrence}"
            )
            continue
        start = int(segment["start"]) + found
        end = start + len(surface)
        replacement = str(item.get("replacement") or item.get("canonical_name") or "")
        ref_id = str(item.get("ref_id") or f"REF_{len(ref_values) + 1:03d}")
        ref_values[f"__{ref_id}__"] = replacement
        replacements.append((start, end, replacement, ref_id))
    replacements.sort(key=lambda value: value[0], reverse=True)
    last_start = len(source) + 1
    for start, end, replacement, ref_id in replacements:
        if end > last_start:
            errors.append(f"겹치는 지칭 치환 무시: ref={ref_id}, start={start}, end={end}")
            continue
        source = source[:start] + replacement + source[end:]
        last_start = start
    return source, errors, ref_values


def apply_reference_assignments_to_slotted(
    slotted_context: str,
    segments: dict[str, dict],
    assignments: list[dict],
) -> tuple[str, list[str]]:
    """Apply the same assignments while preserving every original Slot marker."""
    source = str(slotted_context or "")
    projected, source_indexes = _slotless_projection_with_source_indexes(source)
    operations = []
    errors = []
    for item in assignments or []:
        segment = segments.get(str(item.get("segment_id") or ""))
        if not segment:
            errors.append(f"슬롯 지칭 치환 segment 없음: {item!r}")
            continue
        segment_text = str(segment.get("text") or "")
        segment_span = _find_position_span(projected, segment_text, 0)
        if segment_span is None:
            errors.append(
                f"슬롯 본문 segment 위치 없음: segment={item.get('segment_id')}"
            )
            continue
        segment_start, segment_end = segment_span
        surface = str(item.get("surface") or "")
        occurrence = int(item.get("occurrence") or 1)
        projected_segment = projected[segment_start:segment_end]
        cursor = 0
        found = -1
        for _ in range(occurrence):
            found = projected_segment.find(surface, cursor)
            if found < 0:
                break
            cursor = found + len(surface)
        if found < 0:
            errors.append(
                f"슬롯 본문 지칭 위치 없음: segment={item.get('segment_id')}, "
                f"surface={surface!r}, occurrence={occurrence}"
            )
            continue
        projected_start = segment_start + found
        projected_end = projected_start + len(surface)
        if projected_start >= len(source_indexes) or projected_end <= 0:
            errors.append(f"슬롯 본문 투영 범위 오류: item={item!r}")
            continue
        source_start = source_indexes[projected_start]
        source_end = source_indexes[projected_end - 1] + 1
        replacement = str(item.get("replacement") or item.get("canonical_name") or "")
        operations.append((source_start, source_end, replacement))
    for start, end, replacement in sorted(operations, key=lambda value: value[0], reverse=True):
        source = source[:start] + replacement + source[end:]
    return source, errors


def _filter_character_reference(extra_reference: str, selected_names: list[str]) -> str:
    selected = {str(name or "").strip().casefold() for name in selected_names if str(name or "").strip()}
    if not selected:
        return ""
    source = str(extra_reference or "")
    matches = list(re.finditer(r"(?m)^###\s+([^\r\n]+)\s*$", source))
    blocks = []
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        if name.casefold() not in selected:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        blocks.append(source[match.start():end].strip())
    missing = sorted(selected - {
        re.match(r"(?m)^###\s+([^\r\n]+)", block).group(1).strip().casefold()
        for block in blocks
        if re.match(r"(?m)^###\s+([^\r\n]+)", block)
    })
    if missing:
        print(f"[ILLUST_CONTEXT:CALL2] 선택 캐릭터 lb.extra 누락: names={missing}")
    return "\n\n".join(blocks)


def _exclude_character_reference(extra_reference: str, excluded_names: list[str]) -> str:
    """Remove exact character-schema blocks replaced by preselected profile authority."""
    excluded = {
        str(name or "").strip().casefold()
        for name in excluded_names or []
        if str(name or "").strip()
    }
    source = str(extra_reference or "")
    if not excluded or not source.strip():
        return source.strip()
    matches = list(re.finditer(r"(?m)^###\s+([^\r\n]+)\s*$", source))
    if not matches:
        print(
            "[ILLUST_CONTEXT:PROFILE_RESOLVE] 교체할 캐릭터 reference 블록 헤더 없음: "
            f"characters={sorted(excluded)}"
        )
        return source.strip()
    kept = []
    prefix = source[:matches[0].start()].strip()
    if prefix:
        kept.append(prefix)
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        if name.casefold() in excluded:
            continue
        kept.append(source[match.start():end].strip())
    return "\n\n".join(item for item in kept if item)


def _selected_character_states(states: dict, selected_names: list[str]) -> dict:
    selected = {str(name or "").strip().casefold() for name in selected_names if str(name or "").strip()}
    result = {}
    for key, value in (states or {}).items():
        if not isinstance(value, dict):
            continue
        name = str(value.get("canonical_name") or key).strip()
        if not selected or name.casefold() in selected:
            result[str(key)] = deepcopy(value)
    return result


def apply_wardrobe_events(
    state_before: dict,
    current_characters: list[dict],
    wardrobe_events: list[dict],
    current_message_id: str,
    selected_reference: str = "",
    default_outfits: dict[str, list[str]] | None = None,
    hairstyle_events: list[dict] | None = None,
) -> dict:
    """Apply OUTFIT events as sparse deltas over the complete default outfit.

    OUTFIT intentionally emits compact changed items, not a complete visual tag
    snapshot. Therefore omitted default items are never removed implicitly by a
    short ``set``/``replace`` payload.

    ``hairstyle_events`` are stored verbatim into each character's
    ``hairstyle_timeline`` (append-only, capped). The server never interprets
    hairstyle semantics; CALL2 may replace a conflicting fixed arrangement only
    when the stored event carries direct literal narrative evidence.
    """
    states = deepcopy(state_before or {})
    parsed_defaults = (
        deepcopy(default_outfits)
        if isinstance(default_outfits, dict)
        else extract_authoritative_default_outfits(selected_reference)
    )
    defaults_by_name = {
        str(name).strip().casefold(): [
            str(item).strip()
            for item in items or []
            if str(item).strip()
        ]
        for name, items in parsed_defaults.items()
        if str(name).strip()
    }

    def state_key(name: str) -> str:
        for key, value in states.items():
            if isinstance(value, dict) and str(value.get("canonical_name") or key).casefold() == name.casefold():
                return str(key)
        base = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or uuid.uuid4().hex[:12]
        candidate = base
        suffix = 2
        while candidate in states:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def default_items_for(name: str) -> list[str]:
        return list(defaults_by_name.get(str(name).strip().casefold(), []))

    def ensure_state(name: str) -> str:
        key = state_key(name)
        default_items = default_items_for(name)
        default_reference = (
            _filter_character_reference(selected_reference, [name])
            or str(selected_reference or "")
        )
        if key not in states or not isinstance(states.get(key), dict):
            if key in states:
                print(
                    f"[ILLUST_CONTEXT:WARDROBE_BASE] 비정상 캐릭터 상태를 기본값으로 복구: "
                    f"character={name}, type={type(states.get(key)).__name__}"
                )
            states[key] = {
                "canonical_name": name,
                "default_outfit_reference": default_reference,
                "current_wardrobe": {
                    "body_state": "clothed" if default_items else "unknown",
                    "worn": list(default_items),
                    "removed": [],
                },
                "wardrobe_timeline": [],
                "hairstyle_timeline": [],
            }
            if default_items:
                print(
                    f"[ILLUST_CONTEXT:WARDROBE_BASE] 신규 캐릭터 기본 복장 초기화: "
                    f"character={name}, tags={default_items}"
                )
            else:
                print(
                    f"[ILLUST_CONTEXT:WARDROBE_BASE] 신규 캐릭터 기본 복장 없음: "
                    f"character={name}"
                )
            return key

        tracked = states[key]
        tracked.setdefault("canonical_name", name)
        if default_reference and not tracked.get("default_outfit_reference"):
            tracked["default_outfit_reference"] = default_reference
        raw_wardrobe = (
            deepcopy(tracked.get("current_wardrobe"))
            if isinstance(tracked.get("current_wardrobe"), dict)
            else {}
        )
        wardrobe_metadata = {
            field: deepcopy(value)
            for field, value in raw_wardrobe.items()
            if field not in {"body_state", "worn", "removed", "source"}
        }
        if str(raw_wardrobe.get("source") or "").strip() == "call2_visual_candidate":
            wardrobe = {
                "body_state": "clothed" if default_items else "unknown",
                "worn": list(default_items),
                "removed": [],
            }
            print(
                f"[ILLUST_CONTEXT:WARDROBE_BASE] 구 generated visual 후보 상태를 "
                f"기본 복장으로 재초기화: character={name}, tags={default_items}"
            )
        else:
            wardrobe = _normalize_outfit_state(raw_wardrobe)
        if default_items and wardrobe["body_state"] not in {"nude", "underwear_only"}:
            removed_folded = {item.casefold() for item in wardrobe["removed"]}
            existing_folded = {item.casefold() for item in wardrobe["worn"]}
            restored = [
                item for item in default_items
                if item.casefold() not in removed_folded
                and item.casefold() not in existing_folded
            ]
            if restored:
                wardrobe["worn"] = [
                    item for item in default_items
                    if item.casefold() not in removed_folded
                ] + list(wardrobe["worn"])
                # De-duplicate again because existing state can contain a differently
                # cased copy of a default item.
                wardrobe = _normalize_outfit_state(wardrobe)
                print(
                    f"[ILLUST_CONTEXT:WARDROBE_BASE] 추적 상태에 누락된 기본 복장 복구: "
                    f"character={name}, added={restored}"
                )
            if wardrobe["body_state"] == "unknown":
                wardrobe["body_state"] = "clothed"
        wardrobe.update(wardrobe_metadata)
        tracked["current_wardrobe"] = wardrobe
        tracked.setdefault("wardrobe_timeline", [])
        tracked.setdefault("hairstyle_timeline", [])
        return key

    for item in current_characters or []:
        name = str(item.get("name") if isinstance(item, dict) else item).strip()
        if not name:
            continue
        key = ensure_state(name)
        states[key]["last_seen_message_id"] = str(current_message_id or "")

    for event in wardrobe_events or []:
        name = str(event.get("character") or "").strip()
        if not name:
            print(f"[ILLUST_CONTEXT:WARDROBE_DELTA] 캐릭터 없는 이벤트 스킵: event={event!r}")
            continue
        key = ensure_state(name)
        wardrobe = deepcopy(states[key].get("current_wardrobe") or {})
        worn = [str(value) for value in wardrobe.get("worn") or [] if str(value).strip()]
        removed = [str(value) for value in wardrobe.get("removed") or [] if str(value).strip()]
        operation = str(event.get("operation") or "keep").lower()
        items = [str(value) for value in event.get("items") or [] if str(value).strip()]
        wardrobe_change = str(event.get("wardrobe_change") or "").strip()
        state_after = event.get("state_after")
        # 신규 OUTFIT은 items 대신 자연어 wardrobe_change만 낸다. CALL2가 이 의미 변화를
        # 권위 복장 태그로 번역하기 전까지는 worn/removed를 보존하고 state_after(body_state)
        # 만 반영한다(레거시 items 가 있으면 기존대로 희소 태그 병합을 수행한다).
        semantic_only = bool(wardrobe_change) and not items
        state_label = (
            str(state_after.get("body_state") or "").strip().lower()
            if isinstance(state_after, dict)
            else str(state_after or "").strip().lower()
        )
        if operation in ("nude",) or state_label == "nude":
            removed = list(dict.fromkeys(removed + worn + items))
            worn = []
            wardrobe["body_state"] = "nude"
        elif operation in ("remove", "topless", "bottomless"):
            lowered = {value.casefold() for value in items}
            still_worn = [value for value in worn if value.casefold() not in lowered]
            removed = list(dict.fromkeys(removed + [value for value in worn if value.casefold() in lowered] + items))
            worn = still_worn
            if state_label:
                wardrobe["body_state"] = state_label
            elif operation in ("topless", "bottomless"):
                wardrobe["body_state"] = operation
        elif operation in ("wear", "add"):
            worn = list(dict.fromkeys(worn + items))
            lowered = {value.casefold() for value in items}
            removed = [value for value in removed if value.casefold() not in lowered]
            wardrobe["body_state"] = state_label or "clothed"
        elif operation in ("replace", "set"):
            # OUTFIT's compact list is a sparse change description. Treat it as an
            # overlay; absence from that list is never evidence that a default
            # garment disappeared.
            worn = list(dict.fromkeys(worn + items))
            lowered = {value.casefold() for value in items}
            removed = [value for value in removed if value.casefold() not in lowered]
            wardrobe["body_state"] = state_label or ("clothed" if worn else "unknown")
            if not semantic_only:
                print(
                    f"[ILLUST_CONTEXT:WARDROBE_DELTA] {operation} 희소 병합: "
                    f"character={name}, items={items}, retained_base={len(worn) - len(items)}"
                )
        elif operation in ("reset_default", "contextual_reset"):
            default_items = default_items_for(name)
            worn = list(dict.fromkeys(default_items + items))
            removed = []
            wardrobe["body_state"] = state_label or ("clothed" if worn else "unknown")
        elif operation in ("open", "close", "adjust"):
            wardrobe["body_state"] = state_label or str(wardrobe.get("body_state") or "partial")
        wardrobe["worn"] = worn
        wardrobe["removed"] = removed
        wardrobe["last_event"] = deepcopy(event)
        if semantic_only:
            # 레거시 태그(items) 없이 자연어 의미만 온 이벤트. 옷 태그 해석은 CALL2가
            # wardrobe_change를 번역할 때까지 보류되므로, 현재는 body_state만 반영한다.
            print(
                f"[ILLUST_CONTEXT:WARDROBE_DELTA] semantic event 보류(CALL2 해석 대기): "
                f"character={name}, operation={operation}, "
                f"wardrobe_change={wardrobe_change!r}, "
                f"body_state={wardrobe.get('body_state')}, worn={worn}"
            )
        states[key]["current_wardrobe"] = wardrobe
        timeline = list(states[key].get("wardrobe_timeline") or [])
        timeline.append(deepcopy(event))
        states[key]["wardrobe_timeline"] = timeline[-50:]
        states[key]["last_seen_message_id"] = str(current_message_id or "")

    # hairstyle timeline: 서버는 의미를 해석하지 않고 이벤트를 append 한다.
    # CALL2/AUDIT는 literal evidence가 명시한 직접 변경만 fixed appearance 예외로 해석한다.
    for event in (hairstyle_events or []):
        name = str(event.get("character") or "").strip()
        if not name:
            print(f"[ILLUST_CONTEXT:HAIRSTYLE] 캐릭터 없는 이벤트 스킵: event={event!r}")
            continue
        key = ensure_state(name)
        timeline = list(states[key].get("hairstyle_timeline") or [])
        timeline.append(deepcopy(event))
        states[key]["hairstyle_timeline"] = timeline[-50:]
        states[key]["last_seen_message_id"] = str(current_message_id or "")
        print(
            f"[ILLUST_CONTEXT:HAIRSTYLE] timeline 갱신: character={name}, "
            f"operation={event.get('operation')}, "
            f"hairstyle_change={str(event.get('hairstyle_change') or '')!r}"
        )
    return states


def record_wardrobe_start_resolution(
    states: dict,
    wardrobe_at_start: list[dict],
    current_message_id: str,
) -> dict:
    """Persist OUTFIT's semantic start state beside the existing event timeline."""
    result = deepcopy(states or {})
    for item in wardrobe_at_start or []:
        if not isinstance(item, dict):
            print(
                "[ILLUST_CONTEXT:WARDROBE_BASELINE] 저장할 시작 복장 형식 오류로 스킵: "
                f"item={item!r}"
            )
            continue
        name = str(item.get("character") or "").strip()
        state = str(item.get("state") or "").strip()
        if not name or not state:
            print(
                "[ILLUST_CONTEXT:WARDROBE_BASELINE] 저장할 시작 복장 필수값 없어 스킵: "
                f"character={name!r}, state={state!r}"
            )
            continue
        matched_key = next((
            key
            for key, value in result.items()
            if isinstance(value, dict)
            and str(value.get("canonical_name") or key).strip().casefold()
            == name.casefold()
        ), None)
        if matched_key is None:
            print(
                "[ILLUST_CONTEXT:WARDROBE_BASELINE] 대응 추적 캐릭터가 없어 시작 복장 저장 스킵: "
                f"character={name!r}, message_id={current_message_id!r}"
            )
            continue
        result[matched_key]["last_resolved_wardrobe_start"] = {
            "message_id": str(current_message_id or ""),
            "state": state,
        }
    return result


_OUTFIT_BODY_STATES = {
    "clothed", "partial", "nude", "topless", "bottomless",
    "underwear_only", "unknown",
}


def _normalize_outfit_state(value) -> dict:
    """Normalize a logical wardrobe snapshot without interpreting prose tags."""
    raw = value if isinstance(value, dict) else {}
    body_state = str(raw.get("body_state") or "unknown").strip().lower()
    if body_state not in _OUTFIT_BODY_STATES:
        body_state = "unknown"

    def normalized_items(field: str) -> list[str]:
        items = raw.get(field) or []
        if not isinstance(items, list):
            items = [items]
        result = []
        seen = set()
        for item in items:
            text = str(item or "").strip()
            folded = text.casefold()
            if text and folded not in seen:
                seen.add(folded)
                result.append(text)
        return result

    return {
        "body_state": body_state,
        "worn": normalized_items("worn"),
        "removed": normalized_items("removed"),
    }


def _outfit_state_is_known(value) -> bool:
    state = _normalize_outfit_state(value)
    return bool(
        state["body_state"] != "unknown"
        or state["worn"]
        or state["removed"]
    )


def _resolved_wardrobe_continuity_note(wardrobe_snapshot: dict) -> str:
    """Render only each assigned character's resolved scene outfit for DETAIL."""
    statements = []
    for raw_name, raw_outfit in (wardrobe_snapshot or {}).items():
        name = str(raw_name or "").strip()
        outfit = _normalize_outfit_state(raw_outfit)
        if not name or not _outfit_state_is_known(outfit):
            continue
        facts = []
        if outfit["body_state"] != "unknown":
            facts.append(f"body_state is {outfit['body_state']}")
        if outfit["worn"]:
            facts.append("worn items are " + ", ".join(outfit["worn"]))
        if outfit["removed"]:
            facts.append("removed items are " + ", ".join(outfit["removed"]))
        if facts:
            statements.append(f"- {name}: " + "; ".join(facts) + ".")
    if not statements:
        return ""
    return (
        "Server-resolved wardrobe for this story instant (authoritative for DETAIL):\n"
        + "\n".join(statements)
        + "\nTranslate these resolved items into outfit_state without inventing a different outfit."
    )


def _outfit_states_equal(left, right) -> bool:
    left_state = _normalize_outfit_state(left)
    right_state = _normalize_outfit_state(right)
    return (
        left_state["body_state"] == right_state["body_state"]
        and {item.casefold() for item in left_state["worn"]}
        == {item.casefold() for item in right_state["worn"]}
        and {item.casefold() for item in left_state["removed"]}
        == {item.casefold() for item in right_state["removed"]}
    )


def _outfit_contract_conflict(expected, actual) -> str:
    """Return a structural PLAN/DETAIL wardrobe conflict without tag heuristics."""
    expected_state = _normalize_outfit_state(expected)
    actual_state = _normalize_outfit_state(actual)
    if (
        expected_state["body_state"] != "unknown"
        and actual_state["body_state"] != expected_state["body_state"]
    ):
        return (
            f"body_state expected={expected_state['body_state']!r}, "
            f"actual={actual_state['body_state']!r}"
        )

    actual_worn = {item.casefold() for item in actual_state["worn"]}
    missing_worn = [
        item for item in expected_state["worn"]
        if item.casefold() not in actual_worn
    ]
    if missing_worn:
        return f"PLAN worn 누락={missing_worn}"

    actual_removed = {item.casefold() for item in actual_state["removed"]}
    missing_removed = [
        item for item in expected_state["removed"]
        if item.casefold() not in actual_removed
    ]
    if missing_removed:
        return f"PLAN removed 누락={missing_removed}"
    return ""


def _character_state(states: dict, name: str) -> dict:
    folded = str(name or "").strip().casefold()
    for key, value in (states or {}).items():
        if not isinstance(value, dict):
            continue
        canonical = str(value.get("canonical_name") or key).strip()
        if canonical.casefold() == folded:
            return value
    return {}


def _scene_wardrobe_continuity_note(
    wardrobe_at_start: list[dict],
    applicable_events: list[dict],
    plan_character_names: list[str],
) -> tuple[str, list[str]]:
    """Bind OUTFIT's start state and chronological CURRENT deltas without tag parsing."""
    canonical_by_name = {
        str(name or "").strip().casefold(): str(name or "").strip()
        for name in plan_character_names or []
        if str(name or "").strip()
    }
    statements: list[str] = []
    affected: list[str] = []
    for item in wardrobe_at_start or []:
        if not isinstance(item, dict):
            continue
        item_name = str(item.get("character") or "").strip()
        canonical_name = canonical_by_name.get(item_name.casefold())
        state = str(item.get("state") or "").strip()
        if not canonical_name or not state:
            continue
        statements.append(
            f"{canonical_name} at the start of CURRENT: {state}"
        )
        if canonical_name.casefold() not in {name.casefold() for name in affected}:
            affected.append(canonical_name)
    for event in applicable_events or []:
        event_name = str(event.get("character") or "").strip()
        canonical_name = canonical_by_name.get(event_name.casefold())
        if not canonical_name:
            continue
        change = str(event.get("wardrobe_change") or "").strip()
        evidence = str(event.get("evidence") or "").strip()
        if change and evidence and (
            re.sub(r"\s+", " ", _normalize_analysis_text(change)).casefold()
            != re.sub(r"\s+", " ", _normalize_analysis_text(evidence)).casefold()
        ):
            statement = (
                f"{change} The original passage states: {evidence}"
            )
        else:
            statement = change or evidence
        if not statement:
            legacy_items = [
                str(item or "").strip()
                for item in event.get("items") or []
                if str(item or "").strip()
            ]
            if legacy_items:
                statement = (
                    "The tracked passage establishes a wardrobe change involving "
                    + ", ".join(legacy_items)
                    + "."
                )
        if not statement:
            print(
                "[ILLUST_CONTEXT:CALL2_PLAN] 자연어 연속성으로 전달할 복장 사건 내용 없음: "
                f"character={canonical_name}, event={event!r}"
            )
            continue
        segment_id = str(event.get("segment_id") or "").strip()
        event_prefix = f"then at {segment_id}" if segment_id else "then in CURRENT"
        statements.append(f"{canonical_name} {event_prefix}: {statement}")
        if canonical_name.casefold() not in {name.casefold() for name in affected}:
            affected.append(canonical_name)
    if not statements:
        return "", []
    return (
        "OUTFIT-resolved wardrobe timeline for this story instant. Treat it as the natural-language "
        "authority for wardrobe, coverage, exposure, and continuity-relevant carried garments. "
        "Read the statements chronologically; a later statement supersedes a conflicting earlier "
        "state, while unmentioned parts of the earlier state continue.\n"
        + "\n".join(statements),
        affected,
    )


def _visual_profiles_for_name(
    visual_profiles: dict[str, dict] | None,
    name: str,
) -> dict | None:
    value = _authority_values_for_name(visual_profiles or {}, name)
    return value if isinstance(value, dict) else None


def _visual_state_key(states: dict, name: str) -> str:
    folded = str(name or "").strip().casefold()
    existing = next((
        key for key, value in (states or {}).items()
        if isinstance(value, dict)
        and str(value.get("canonical_name") or key).strip().casefold() == folded
    ), None)
    if existing is not None:
        return existing
    return re.sub(r"[^a-z0-9]+", "_", folded).strip("_") or uuid.uuid4().hex[:12]


def apply_initial_visual_bases(
    states: dict,
    initial_visual_bases: list[dict],
    current_message_id: str,
    visual_profiles: dict[str, dict] | None,
) -> dict:
    """Apply the preselected profile at the start of CURRENT chronology."""
    result = deepcopy(states or {})
    for initial in initial_visual_bases or []:
        name = str(initial.get("character") or "").strip()
        character_profiles = _visual_profiles_for_name(visual_profiles, name)
        if not name or character_profiles is None:
            print(
                f"[ILLUST_CONTEXT:VISUAL_BASE] 초기 프로필 캐릭터/카탈로그 없음: "
                f"initial={initial!r}"
            )
            continue
        key = _visual_state_key(result, name)
        state = result.setdefault(key, {
            "canonical_name": name,
            "visual_base_timeline": [],
        })
        tracked_profile_id = str(
            state.get("active_visual_profile_id") or ""
        ).strip()
        try:
            base = resolve_visual_base(
                character_profiles,
                str(initial.get("target_visual_profile_id") or ""),
            )
        except Exception as exc:
            print(
                f"[ILLUST_CONTEXT:VISUAL_BASE] 초기 프로필 해석 실패, 시드 스킵: "
                f"initial={initial!r}, error={exc}"
            )
            traceback.print_exc()
            continue
        changed = tracked_profile_id != base["visual_profile_id"]
        state["canonical_name"] = name
        state["active_visual_profile_id"] = base["visual_profile_id"]
        state["active_visual_profile_state"] = str(
            initial.get("initial_state") or ""
        ).strip()
        state.pop("active_outfit_id", None)
        current_wardrobe = _normalize_outfit_state(state.get("current_wardrobe"))
        # A corrected profile cannot carry the fallback wardrobe of the rejected
        # profile. If the profile stayed the same, preserve stronger tracked
        # wardrobe continuity and fill only an unknown state from the card base.
        if (
            (changed and bool(tracked_profile_id))
            or not _outfit_state_is_known(current_wardrobe)
        ):
            worn = visual_tag_values(base.get("outfit") or [])
            state["current_wardrobe"] = {
                "body_state": "clothed" if worn else "unknown",
                "worn": worn,
                "removed": [],
            }
        if changed:
            timeline = state.setdefault("visual_base_timeline", [])
            timeline.append({
                **deepcopy(initial),
                "message_id": str(current_message_id or ""),
                "initial_state": True,
                "applied_change": True,
                "previous_visual_profile_id": tracked_profile_id,
            })
        state["last_seen_message_id"] = str(current_message_id or "")
        print(
            f"[ILLUST_CONTEXT:VISUAL_BASE] 최전단 START 프로필 적용: "
            f"character={name}, tracked={tracked_profile_id!r}, "
            f"selected={base['visual_profile_name']!r}, changed={changed}"
        )
    return result


def apply_visual_base_events(
    states: dict,
    current_characters: list[dict],
    visual_base_events: list[dict],
    current_message_id: str,
    visual_profiles: dict[str, dict] | None,
) -> dict:
    """Apply exact server-validated flat profile routes without semantic matching."""
    result = deepcopy(states or {})
    names: list[str] = []
    for item in current_characters or []:
        name = str(item.get("name") if isinstance(item, dict) else item or "").strip()
        if name and name.casefold() not in {value.casefold() for value in names}:
            names.append(name)
    for event in visual_base_events or []:
        name = str(event.get("character") or "").strip()
        if name and name.casefold() not in {value.casefold() for value in names}:
            names.append(name)

    for name in names:
        character_profiles = _visual_profiles_for_name(visual_profiles, name)
        if character_profiles is None:
            print(
                f"[ILLUST_CONTEXT:VISUAL_BASE] 캐릭터 프로필 카탈로그 없음: "
                f"character={name!r}"
            )
            continue
        key = _visual_state_key(result, name)
        state = result.setdefault(key, {})
        state.setdefault("canonical_name", name)
        state.setdefault("visual_base_timeline", [])
        requested_profile = str(state.get("active_visual_profile_id") or "").strip()
        base = resolve_visual_base(character_profiles, requested_profile)
        if not requested_profile:
            print(
                f"[ILLUST_CONTEXT:VISUAL_BASE] 추적·서사 초기 프로필이 없어 "
                f"등록 기본 프로필로 최종 폴백: "
                f"character={name}, profile={base['visual_profile_id']}"
            )
        state["active_visual_profile_id"] = base["visual_profile_id"]
        state.pop("active_outfit_id", None)
        current_wardrobe = _normalize_outfit_state(state.get("current_wardrobe"))
        if not _outfit_state_is_known(current_wardrobe):
            worn = visual_tag_values(base.get("outfit") or [])
            state["current_wardrobe"] = {
                "body_state": "clothed" if worn else "unknown",
                "worn": worn,
                "removed": [],
            }

    for event in visual_base_events or []:
        name = str(event.get("character") or "").strip()
        character_profiles = _visual_profiles_for_name(visual_profiles, name)
        if not name or character_profiles is None:
            print(
                f"[ILLUST_CONTEXT:VISUAL_BASE] 적용할 캐릭터 프로필 없음: event={event!r}"
            )
            continue
        try:
            base = resolve_visual_base(
                character_profiles,
                str(event.get("target_visual_profile_id") or ""),
            )
        except Exception as exc:
            print(
                f"[ILLUST_CONTEXT:VISUAL_BASE] 검증 후 외형 기반 해석 실패, 사건 스킵: "
                f"event={event!r}, error={exc}"
            )
            traceback.print_exc()
            continue
        key = _visual_state_key(result, name)
        state = result.setdefault(key, {
            "canonical_name": name,
            "visual_base_timeline": [],
        })
        changed = str(state.get("active_visual_profile_id") or "") != base["visual_profile_id"]
        state["active_visual_profile_id"] = base["visual_profile_id"]
        state["active_visual_profile_state"] = str(
            event.get("visual_change") or ""
        ).strip()
        state.pop("active_outfit_id", None)
        worn = visual_tag_values(base.get("outfit") or [])
        state["current_wardrobe"] = {
            "body_state": "clothed" if worn else "unknown",
            "worn": worn,
            "removed": [],
        }
        timeline = state.setdefault("visual_base_timeline", [])
        timeline.append({
            **deepcopy(event),
            "message_id": str(current_message_id or ""),
            "applied_change": changed,
        })
        state["last_seen_message_id"] = str(current_message_id or "")
        print(
            f"[ILLUST_CONTEXT:VISUAL_BASE] 외형 기반 사건 적용: "
            f"character={name}, profile={base['visual_profile_id']}, changed={changed}"
        )
    return result


def visual_base_snapshot(
    states: dict,
    character_names: list[str],
    visual_profiles: dict[str, dict] | None,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for name in character_names or []:
        character_profiles = _visual_profiles_for_name(visual_profiles, name)
        if character_profiles is None:
            print(
                f"[ILLUST_CONTEXT:VISUAL_BASE] 스냅샷 프로필 없음: character={name!r}"
            )
            continue
        tracked = _character_state(states, name)
        base = resolve_visual_base(
            character_profiles,
            str((tracked or {}).get("active_visual_profile_id") or ""),
        )
        profile_state = str(
            (tracked or {}).get("active_visual_profile_state") or ""
        ).strip()
        if profile_state:
            base["visual_profile_state"] = profile_state
        result[name] = base
    return result


def bind_scene_plan_anchor_passages(
    scene_plan: list[dict],
    segments: dict[str, dict],
) -> list[dict]:
    """Carry each selected segment's original prose into the downstream LLM handoff."""
    bound = []
    for plan_index, raw_plan in enumerate(scene_plan, start=1):
        plan = deepcopy(raw_plan)
        anchor_segment = str(plan.get("anchor_segment") or "").strip()
        segment = segments.get(anchor_segment)
        anchor_passage = str(
            segment.get("text") if isinstance(segment, dict) else ""
        ).strip()
        if not anchor_passage:
            print(
                "[ILLUST_CONTEXT:CALL2_PLAN] 장면 앵커 원문 연결 실패: "
                f"plan={plan_index}, anchor={anchor_segment!r}, "
                f"segments={list(segments)}"
            )
            raise ValueError(
                f"scene_plan[{plan_index}] 앵커 원문이 없습니다: {anchor_segment!r}"
            )
        plan["anchor_passage"] = anchor_passage
        bound.append(plan)
    print(
        "[ILLUST_CONTEXT:CALL2_PLAN] 장면 앵커 원문 연결 완료: "
        f"plans={[(item.get('plan_id'), item.get('anchor_segment'), len(str(item.get('anchor_passage') or ''))) for item in bound]}"
    )
    return bound


def _last_visual_by_character(descriptors: list[dict]) -> dict:
    result = {}
    ordered = sorted(
        [item for item in descriptors if str(item.get("kind") or "") == "scene"],
        key=lambda item: int(item.get("slot") or 0),
    )
    for descriptor in ordered:
        for character in descriptor.get("characters") or []:
            name = str(character.get("name") or "").strip()
            if not name:
                continue
            outfit_state = deepcopy(character.get("outfit_state") or {})
            if not outfit_state:
                print(
                    f"[ILLUST_CONTEXT:CALL2] 캐릭터 outfit_state 누락: "
                    f"name={name}, slot={descriptor.get('slot')}"
                )
            result[name] = {
                "source_slot": descriptor.get("slot"),
                "positive_tags": str(character.get("positive") or ""),
                "outfit_state": outfit_state,
                "visual_profile_id": str(
                    (
                        _authority_values_for_name(
                            descriptor.get("visual_base_snapshot") or {},
                            name,
                        )
                        or {}
                    ).get("visual_profile_id")
                    or ""
                ),
            }
    return result


def merge_last_visual_into_states(
    states: dict,
    last_visual: dict,
    current_message_id: str,
    *,
    allow_visual_initialization: bool,
) -> dict:
    result = deepcopy(states or {})
    for name, visual in (last_visual or {}).items():
        key = next((
            existing_key
            for existing_key, value in result.items()
            if isinstance(value, dict)
            and str(value.get("canonical_name") or existing_key).casefold() == str(name).casefold()
        ), None)
        if key is None:
            key = re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_") or uuid.uuid4().hex[:12]
            result[key] = {
                "canonical_name": str(name),
                "current_wardrobe": {"body_state": "unknown", "worn": [], "removed": []},
                "wardrobe_timeline": [],
            }
        result[key]["last_visual_reference"] = deepcopy(visual)
        result[key]["last_seen_message_id"] = str(current_message_id or "")
        outfit_state = visual.get("outfit_state") if isinstance(visual, dict) else None
        if isinstance(outfit_state, dict) and outfit_state and allow_visual_initialization:
            print(
                f"[ILLUST_CONTEXT:WARDROBE_BASE] generated visual outfit은 참고로만 저장: "
                f"character={name}, current_message_id={current_message_id!r}"
            )
    return result


def _splice_enhancements(body: str, output: str) -> str:
    result = body
    base_tags = []
    for block in re.findall(r"\[CharacterBaseTags\]([\s\S]*?)\[/CharacterBaseTags\]", output or "", re.I):
        value = block.strip()
        if value:
            base_tags.append(value)
    pattern = re.compile(
        r"\[Position\]([\s\S]*?)\[/Position\]\s*((?:(?!\[Position\]|\[CharacterBaseTags\])[\s\S])*)",
        re.I,
    )
    offset = 0
    for match in pattern.finditer(output or ""):
        anchor = match.group(1).strip()
        insertion = match.group(2).strip()
        if not anchor or not insertion:
            continue
        pos = result.find(anchor)
        if pos < 0:
            compact_anchor = re.sub(r"\s+", " ", anchor).strip()
            for candidate in re.finditer(r"[^\n]+", result):
                if compact_anchor in re.sub(r"\s+", " ", candidate.group(0)):
                    pos = candidate.end()
                    anchor = ""
                    break
        else:
            pos += len(anchor)
        if pos < 0:
            print(f"[ILLUST_CONTEXT:OUTFIT] 삽입 위치를 찾지 못함: {anchor!r}")
            continue
        result = result[:pos] + "\n\n" + insertion + result[pos:]
        offset += 1
    if base_tags:
        result = result.rstrip() + "\n\n" + "\n\n".join(base_tags)
    return result


def insert_slots(text: str) -> str:
    # target_slotted가 없는 구형/비상 요청도 현재 모듈과 같은 대사 경계를
    # 보존한다. 이 폴백은 기존 계약대로 문장 보조 후보를 새로 만들지 않는다.
    source = str(text or "").strip()
    paragraph_boundaries = _boundaries_outside_protected_spans(
        source,
        [match.span() for match in re.finditer(r"\n\n+", source)],
        boundary_kind="fallback-paragraph",
    )
    result = source
    for slot_index, (start, end) in reversed(list(enumerate(paragraph_boundaries))):
        result = result[:start] + f"\n\n[Slot {slot_index}]\n\n" + result[end:]
    return result


_SLOT_MARKER_RE = re.compile(r"\[Slot\s+(\d+)\]")
_PROTECTED_SLOT_TOKEN_RE = re.compile(r"__SLOT_[0-9]+__|\[C[0-9]+\]")


def _protect_slot_markers(text: str, segment_slots=None) -> tuple[str, list[tuple[str, str]]]:
    """LLM이 슬롯을 구조 태그로 해석하지 않도록 불투명 토큰으로 치환한다."""
    source = str(text or "")
    protected_markers: list[tuple[str, str]] = []

    anchors = {slot: anchor for anchor, slot in (segment_slots or {}).items()}
    def replace(match: re.Match) -> str:
        anchor = anchors.get(int(match.group(1)))
        token = f"[{anchor}]" if anchor else f"__SLOT_{len(protected_markers)}__"
        protected_markers.append((token, match.group(0)))
        return token

    protected = _SLOT_MARKER_RE.sub(replace, source)
    return protected, protected_markers


def _restore_slot_markers(
    text: str,
    protected_markers: list[tuple[str, str]],
) -> tuple[str, bool, str]:
    """보호 토큰의 개수와 순서를 확인하고 원래 슬롯 마커로 복원한다."""
    value = str(text or "")
    if not protected_markers:
        return value, True, ""

    expected = [token for token, _marker in protected_markers]
    actual = _PROTECTED_SLOT_TOKEN_RE.findall(value)
    if actual != expected:
        return (
            value,
            False,
            f"보호 슬롯 토큰 불일치(expected={expected}, actual={actual})",
        )

    restored = value
    for token, marker in protected_markers:
        restored = restored.replace(token, marker, 1)
    return restored, True, ""


def _slotless_projection_with_source_indexes(text: str) -> tuple[str, list[int]]:
    """슬롯 마커만 숨긴 문자열과 각 문자의 원문 인덱스를 반환한다."""
    source = str(text or "")
    projected_parts = []
    source_indexes = []
    cursor = 0
    for match in _SLOT_MARKER_RE.finditer(source):
        projected_parts.append(source[cursor:match.start()])
        source_indexes.extend(range(cursor, match.start()))
        cursor = match.end()
    projected_parts.append(source[cursor:])
    source_indexes.extend(range(cursor, len(source)))
    return "".join(projected_parts), source_indexes


def _find_position_span(
    projected: str,
    anchor: str,
    start_offset: int,
) -> tuple[int, int] | None:
    """OUTFIT Position을 슬롯 없는 투영 본문에서 찾는다."""
    exact_start = projected.find(anchor, start_offset)
    if exact_start >= 0:
        return exact_start, exact_start + len(anchor)

    # 슬롯 마커 주위의 빈 줄 개수만 달라진 경우도 같은 위치로 취급한다.
    pieces = re.split(r"(\s+)", anchor)
    flexible_pattern = "".join(
        r"\s+" if piece.isspace() else re.escape(piece)
        for piece in pieces
        if piece
    )
    if not flexible_pattern:
        return None
    match = re.search(flexible_pattern, projected[start_offset:])
    if not match:
        return None
    return start_offset + match.start(), start_offset + match.end()


def build_segment_slot_map(
    slotted_context: str,
    segments: dict[str, dict],
) -> tuple[dict[str, int], str, str]:
    """Read the C aliases assigned when actual insertion candidates were built."""
    mapping = {}
    rendered = []
    missing = []
    for segment_id, segment in segments.items():
        slot = segment.get("slot")
        if type(slot) is not int:
            missing.append(segment_id)
            print(f"[ILLUST_CONTEXT:CALL2_PLAN] C alias has no insertion position: segment={segment_id}, segment_data={segment!r}")
            continue
        mapping[segment_id] = slot
        rendered.append(f"[{segment_id}]\n{segment.get('text') or ''}")
    if missing:
        reason = f"C alias has no insertion position: segments={missing}"
        return mapping, "\n\n".join(rendered), reason
    if not mapping:
        reason = "No selectable C/Slot candidates"
        print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}: source_len={len(slotted_context)}, segments={len(segments)}")
        return {}, "", reason
    print(f"[ILLUST_CONTEXT:CALL2_PLAN] C-Slot one-to-one aliases: mapping={mapping}")
    return mapping, "\n\n".join(rendered), ""


def _merge_call1_output_into_slotted(
    slotted_body: str,
    call1_output: str,
) -> str:
    """OUTFIT Position 블록을 슬롯을 보존한 본문의 실제 위치에 합친다."""
    source = str(slotted_body or "")
    output = str(call1_output or "")
    if not output.strip():
        print("[ILLUST_CONTEXT:OUTFIT] 슬롯 본문에 합칠 OUTFIT 응답이 비어 있음")
        return source

    expected_slots = _SLOT_MARKER_RE.findall(source)
    unexpected_slots = _SLOT_MARKER_RE.findall(output)
    if unexpected_slots:
        print(
            f"[ILLUST_CONTEXT:OUTFIT] 슬롯을 숨긴 OUTFIT 응답에 예상하지 못한 "
            f"슬롯 마커가 포함됨: slots={unexpected_slots}; 제거 후 병합"
        )
        output = _SLOT_MARKER_RE.sub("", output)

    position_pattern = re.compile(
        r"\[Position\]([\s\S]*?)\[/Position\]\s*"
        r"((?:(?!\[Position\]|\[CharacterBaseTags\])[\s\S])*)",
        re.I,
    )
    projected, source_indexes = _slotless_projection_with_source_indexes(source)
    operations = []
    fallback_blocks = []
    projection_cursor = 0

    for block_index, match in enumerate(position_pattern.finditer(output), start=1):
        anchor = match.group(1).strip()
        insertion = match.group(2).strip()
        raw_block = match.group(0).strip()
        if not anchor or not insertion:
            print(
                f"[ILLUST_CONTEXT:OUTFIT] Position 병합 블록이 비어 있음: "
                f"block={block_index}, anchor_len={len(anchor)}, "
                f"insertion_len={len(insertion)}"
            )
            if raw_block:
                fallback_blocks.append(raw_block)
            continue

        projected_span = _find_position_span(
            projected,
            anchor,
            projection_cursor,
        )
        if projected_span is None and projection_cursor > 0:
            print(
                f"[ILLUST_CONTEXT:OUTFIT] Position 순차 검색 실패, 전체 본문 재검색: "
                f"block={block_index}, anchor={anchor[:120]!r}"
            )
            projected_span = _find_position_span(projected, anchor, 0)
        if projected_span is None:
            print(
                f"[ILLUST_CONTEXT:OUTFIT] 슬롯 본문에서 Position을 찾지 못함: "
                f"block={block_index}, anchor={anchor[:120]!r}"
            )
            fallback_blocks.append(raw_block)
            continue

        projected_start, projected_end = projected_span
        if (
            projected_start >= len(source_indexes)
            or projected_end <= projected_start
            or projected_end > len(source_indexes)
        ):
            print(
                f"[ILLUST_CONTEXT:OUTFIT] Position 원문 인덱스 변환 실패: "
                f"block={block_index}, projected=({projected_start}, {projected_end}), "
                f"index_count={len(source_indexes)}"
            )
            fallback_blocks.append(raw_block)
            continue

        source_start = source_indexes[projected_start]
        source_end = source_indexes[projected_end - 1] + 1
        if any(
            source_start < existing_end and source_end > existing_start
            for existing_start, existing_end, _replacement in operations
        ):
            print(
                f"[ILLUST_CONTEXT:OUTFIT] Position 병합 범위가 기존 블록과 겹침: "
                f"block={block_index}, source=({source_start}, {source_end})"
            )
            fallback_blocks.append(raw_block)
            continue

        matched_source = source[source_start:source_end]
        replacement = (
            f"[Position]{matched_source}[/Position]\n"
            f"{insertion}"
        )
        operations.append((source_start, source_end, replacement))
        projection_cursor = projected_end

    result = source
    for source_start, source_end, replacement in sorted(
        operations,
        key=lambda operation: operation[0],
        reverse=True,
    ):
        result = result[:source_start] + replacement + result[source_end:]

    character_base_blocks = [
        match.group(0).strip()
        for match in re.finditer(
            r"\[CharacterBaseTags\][\s\S]*?\[/CharacterBaseTags\]",
            output,
            re.I,
        )
        if match.group(0).strip()
    ]
    append_blocks = fallback_blocks + character_base_blocks
    if append_blocks:
        result = result.rstrip() + "\n\n" + "\n\n".join(append_blocks)
    actual_slots = _SLOT_MARKER_RE.findall(result)
    if actual_slots != expected_slots:
        print(
            f"[ILLUST_CONTEXT:OUTFIT] Position 병합 후 슬롯 검증 실패: "
            f"expected={expected_slots}, actual={actual_slots}; 원본 슬롯 본문 사용"
        )
        return source
    return result


def remove_slot_markers(text: str) -> str:
    """원문 슬롯 본문에서 배치용 마커만 제거한다."""
    return _SLOT_MARKER_RE.sub("", str(text or "")).strip()


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() not in ("", "0", "false", "off", "끄기", "null")
    return bool(value)


def _render_conditionals(text: str, values: dict) -> str:
    # custom closer를 Risu의 일반 closer와 같게 취급한다.
    text = prompt_sub(r"\{\{/(?:compat-comfy|nsfw|supplement|context|history-length|history-null)\}\}", "{{/when}}", text)

    def cond(expr: str) -> bool:
        bits = [b.strip() for b in expr.strip(": ").split("::") if b.strip()]
        if not bits:
            return False
        if bits[0] == "keep":
            bits = bits[1:]
        if bits and bits[0] == "toggle":
            bits = bits[1:]
        if not bits:
            return False
        left = values.get(bits[0], bits[0])
        if len(bits) >= 3 and bits[1] in ("tis", "tisnot", "<", ">"):
            op, right = bits[1], bits[2]
            if op == "tis":
                return str(left) == str(right)
            if op == "tisnot":
                return str(left) != str(right)
            try:
                return float(left) < float(right) if op == "<" else float(left) > float(right)
            except Exception:
                return False
        return _bool(left)

    # innermost block부터 반복 처리한다.
    block = re.compile(r"\{\{#when(?:::|\s+)([^{}]*?)\}\}((?:(?!\{\{#when)[\s\S])*?)\{\{/when\}\}")
    for _ in range(80):
        match = block.search(text)
        if not match:
            break
        body = text[match.start(2):match.end(2)]
        divider = body.find("{{:else}}")
        yes = body if divider < 0 else body[:divider]
        no = "" if divider < 0 else body[divider + len("{{:else}}"):]
        selected = yes if cond(match.group(1)) else no
        text = text[:match.start()] + variable(selected, "조건 설정에 따라 포함된 지침") + text[match.end():]
    return text


def render_call2_prompt(
    text: str,
    toggles: dict,
    history: str = "",
    *,
    include_scene_count_limit: bool = True,
    include_server_limits: bool = True,
) -> str:
    """Risu 토글 매크로를 서버 설정으로 렌더링한다.

    include_server_limits=False면 "# Server limits" 섹션 전체를 생략한다.
    include_scene_count_limit=False면 그 섹션의 장면 수(min/max) 라인만
    생략한다. PLAN과 병렬 DETAIL은 각 태스크 메시지에 자신의 정확한
    카운트 계약을 별도로 받으므로 role system에 공통 제한을 중복하지 않는다.
    """
    if is_symbolic():
        return symbolic_call2_template(str(text or ""))
    text = PromptText(str(text or ""))
    # 복잡한 history/client-comment 블록은 서버 값으로 명시적으로 재구성한다.
    prefix = text
    suffix = ""
    if "## Character Tag History" in prefix and "# Example" in prefix:
        history_pos = prefix.index("## Character Tag History")
        # 원본은 히스토리 제목 직전에 3중 Risu 조건문을 연다. 제목부터만
        # 잘라내면 여는 매크로가 남으므로 첫 조건문부터 섹션 전체를 교체한다.
        section_pos = prefix.rfind("{{#when", 0, history_pos)
        if section_pos < 0:
            section_pos = history_pos
        tail = prefix[history_pos + len("## Character Tag History"):]
        prefix = prefix[:section_pos]
        _, suffix = tail.split("# Example", 1)
        suffix = "# Example" + suffix
        if toggles.get("context_history") and history.strip():
            prefix += "## Character Tag History\n\n" + variable(history.strip(), "캐릭터 태그 히스토리") + "\n\n"
        comments = []
        if str(toggles.get("focus") or "").strip():
            comments.append(
                'I want to focus on the character(s): "' + variable(str(toggles["focus"]).strip(), "사용자 집중 대상") + '". Do not make scenes for others.'
            )
        if str(toggles.get("direction") or "").strip():
            comments.append(variable(str(toggles["direction"]).strip(), "사용자 연출 지시"))
        prefix += "## Client Comments\n\n<instruction>\n" + (prompt_join("\n\n", comments) if comments else "(None specified)") + "\n</instruction>\n\n"
        text = prefix + suffix

    risu_values = {
        "lb-xnai.nsfw": "1" if toggles.get("nsfw") else "0",
        "lb-xnai.supplement": "1" if toggles.get("supplement") else "0",
        "lb-xnai.kv.off": "0" if toggles.get("key_visual") else "1",
        "lb-xnai.background.minimal": (
            "1" if toggles.get("minimal_background_description", True) else "0"
        ),
        "lb-xnai.compat.comfy": "1" if toggles.get("compat_comfy") else "0",
        "lb-xnai.compat.charPrompt": "1" if toggles.get("compat_character_prompt") == "separate" else "0",
        "lb-xnai.context": "1" if toggles.get("context_history") else "0",
        "lb-xnai.characters": str(max(0, 3 - int(toggles.get("character_limit", 3)))),
        "lb-xnai-history": history or "null",
    }
    for key, value in risu_values.items():
        text = text.replace("{{getglobalvar::toggle_" + key + "}}", variable(value, "생성 설정: " + key))
        text = text.replace("{{getvar::" + key + "}}", variable(value, "생성 설정: " + key))

    def dict_element(match):
        try:
            table = json.loads(match.group(1))
            return variable(str(table.get(str(match.group(2)).strip(), "")), "설정 매크로 선택값")
        except Exception as e:
            print(f"[ILLUST_CONTEXT] dictelement 렌더 실패: {e}; expr={match.group(0)!r}")
            traceback.print_exc()
            return ""

    text = prompt_sub(r"\{\{dictelement::(\{[^{}]*\})::([^{}]*)\}\}", dict_element, text)
    text = _render_conditionals(text, risu_values)
    # Risu에만 존재하는 잔여 매크로는 LLM으로 보내지 않고 로그에 남긴다.
    leftovers = re.findall(r"\{\{[^\n]{0,240}?\}\}", text)
    if leftovers:
        print(f"[ILLUST_CONTEXT] 렌더 후 잔여 Risu 매크로 {len(leftovers)}개 제거: {leftovers}")
        text = prompt_sub(r"\{\{[^\n]*?\}\}", "", text)
    # output_count_min/max는 scene_mode와 무관한 유일한 장면 수 권위다.
    if include_server_limits:
        limits = []
        if include_scene_count_limit:
            limits.append(
                "Generate between " + variable(int(toggles['output_count_min']), "최소 장면 수") + " and "
                + variable(int(toggles['output_count_max']), "최대 장면 수") + " scenes."
            )
        limits.append(
            "Maximum fully visible characters per image: " + variable(int(toggles['character_limit']), "인물 수 제한") + "."
        )
        limits.append(
            "Key visual: " + variable('required' if toggles.get('key_visual') else 'disabled; omit keyvis', "대표 이미지 설정") + "."
        )
        text += "\n\n# Server limits\n- " + prompt_join("\n- ", limits)
    return text.strip()


def _extract_lb_block(text: str) -> str:
    match = re.search(r"<lb[-_]xnai[^>]*>([\s\S]*?)</lb[-_]xnai>", text or "", re.I)
    return match.group(1).strip() if match else ""


def _normalize_toon(text: str) -> str:
    text = re.sub(r"\b(scenes|characters|regions)\[\d+\]:", r"\1:", text)
    text = text.replace("\t", "  ")

    # CALL2/CALL3 commonly emit TOON string fields as unquoted YAML scalars.
    # Natural-language values can legally contain YAML syntax such as ``: ``
    # or `` #`` (for example, ``supplement: split-screen: left ...``), which
    # makes PyYAML reject or truncate an otherwise valid plan.  These schema
    # fields are always strings, so encode each one as a JSON string; JSON
    # strings are valid YAML scalars and preserve punctuation verbatim.
    text_fields = {
        "camera", "positive", "negative", "name", "position",
        "scene", "supplement", "speak", "scene_brief",
        "outfit", "hair", "time", "place",
        "background_prompt", "composition_prompt", "character_prompt",
    }
    scalar_line = re.compile(
        r"^(\s*(?:-\s+)?)([A-Za-z_][\w-]*):(\s*)(.*)$"
    )
    normalized_lines = []
    block_indent = None
    table_columns = None
    table_indent = 0
    for line in text.splitlines():
        if block_indent is not None:
            indent = len(line) - len(line.lstrip())
            if not line.strip() or indent > block_indent:
                normalized_lines.append(line)
                continue
            block_indent = None
        table_header = re.fullmatch(r'([ ]*)([A-Za-z_][\w-]*)\[\d+\]\{([^{}]+)\}:[ ]*', line)
        if table_header:
            table_indent = len(table_header.group(1))
            table_columns = [column.strip() for column in table_header.group(3).split(',')]
            normalized_lines.append(f"{table_header.group(1)}{table_header.group(2)}:")
            continue
        if table_columns is not None and line.strip():
            indent = len(line) - len(line.lstrip())
            if indent > table_indent:
                # TOON rows are comma-separated scalar cells. Preserve quoted
                # commas and JSON escapes, and bind each cell to its own key.
                cells = re.findall(r'(?:^|,)[ \t]*("(?:[^"\\]|\\.)*"|[^,\r\n]*)', line.lstrip())
                if len(cells) == len(table_columns):
                    for index, (column, cell) in enumerate(zip(table_columns, cells)):
                        cell = cell.strip()
                        value = (json.loads(cell) if cell.startswith('"')
                                 or cell in {"true", "false", "null"}
                                 or re.fullmatch(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?', cell)
                                 else cell)
                        prefix = " " * indent + ("- " if index == 0 else "  ")
                        normalized_lines.append(f"{prefix}{column}: {json.dumps(value, ensure_ascii=False)}")
                    continue
                print(f"[ILLUST_CONTEXT:PARSE] TOON 행 필드 매핑 불가, 원문 보존: columns={table_columns!r}, input={line!r}")
            else:
                table_columns = None
        match = scalar_line.match(line)
        if not match or match.group(2) not in text_fields:
            normalized_lines.append(line)
            continue

        value = match.group(4).strip()
        if not value:
            normalized_lines.append(line)
            continue
        if re.fullmatch(r"[|>](?:[1-9][+-]?|[+-][1-9]?)?", value):
            # YAML's block scalar syntax owns the following indented lines.
            # Quoting this marker would turn valid multiline text into damage.
            normalized_lines.append(line)
            block_indent = len(line) - len(line.lstrip())
            continue

        # Avoid adding literal quote characters when the model already used a
        # valid quoted YAML scalar.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            try:
                loaded = yaml.safe_load(value)
                if isinstance(loaded, str):
                    value = loaded
            except Exception:
                pass
        normalized_lines.append(
            f"{match.group(1)}{match.group(2)}: {json.dumps(value, ensure_ascii=False)}"
        )
    return "\n".join(normalized_lines) + ("\n" if text.endswith("\n") else "")


def _descriptor(raw: dict, kind: str, fallback_slot: int) -> dict:
    chars = []
    for ch in raw.get("characters") or []:
        if not isinstance(ch, dict):
            continue
        if ch.get("negative"):
            print(f"[ILLUST_CONTEXT:CALL2] LLM negative ignored (disabled): "
                  f"kind={kind}, name={ch.get('name')!r}, value={ch['negative']!r}")
        raw_authority_exceptions = ch.get("authority_exceptions") or []
        if not isinstance(raw_authority_exceptions, list):
            print(
                "[ILLUST_CONTEXT:CALL2_AUTHORITY] authority_exceptions가 list가 "
                f"아니어서 단일 항목으로 정규화: type={type(raw_authority_exceptions).__name__}, "
                f"value={raw_authority_exceptions!r}"
            )
            raw_authority_exceptions = [raw_authority_exceptions]
        chars.append({
            "positive": str(ch.get("positive") or "").strip(),
            "negative": "",  # LLM negative control disabled; keep descriptor compatibility.
            "name": str(ch.get("name") or "").strip(),
            "position": str(ch.get("position") or "").strip(),
            "authority_exceptions": [
                str(value).strip()
                for value in raw_authority_exceptions
                if str(value).strip()
            ],
            "outfit_state": deepcopy(ch.get("outfit_state") or {}),
        })
    slot_value = -1 if kind == "keyvis" else raw.get("slot", fallback_slot)
    try:
        slot_value = int(slot_value)
    except Exception:
        slot_value = fallback_slot
    return {
        "kind": kind,
        "plan_id": str(raw.get("plan_id") or "").strip(),
        "slot": slot_value,
        "camera": str(raw.get("camera") or "").strip(),
        "scene": str(raw.get("scene") or "").strip(),
        "supplement": str(raw.get("supplement") or "").strip(),
        "speak": "",
        "characters": chars,
    }


def parse_toon_plan(text: str, toggles: dict, source: str = "CALL2") -> list[dict]:
    source = re.sub(r"[^A-Za-z0-9_-]", "", str(source or "TOON").upper()) or "TOON"
    data = _json_object_from_text(text)
    if not isinstance(data, dict):
        print(f"[ILLUST_CONTEXT:{source}] 복구 가능한 keyvis/scene 문서 없음: input={str(text)[:500]!r}")
        return []
    out = []
    keyvis = data.get("keyvis")
    if toggles.get("key_visual") and isinstance(keyvis, dict):
        out.append(_descriptor(keyvis, "keyvis", -1))
    scenes = data.get("scenes") or []
    if not isinstance(scenes, list):
        print(f"[ILLUST_CONTEXT:{source}] scenes가 list가 아님: {type(scenes).__name__}")
        scenes = []
    # output_count_max는 scene_mode와 무관한 유일한 장면 수 상한이다.
    # auto는 범위 안의 수량 선택만 LLM에 맡기며 범위 자체를 해제하지 않는다.
    scene_cap = int(toggles["output_count_max"])
    capped = scenes[:scene_cap]
    for index, raw in enumerate(capped, start=1):
        if isinstance(raw, dict):
            out.append(_descriptor(raw, "scene", index))
    if not out:
        print(f"[ILLUST_CONTEXT:{source}] 유효한 keyvis/scene 결과가 없음")
    return out


def _parse_call2_keyvis_output(
    text: str,
    toggles: dict,
    allowed_character_names: list[str],
    source: str,
    *,
    mask_inline: bool = False,
    allow_scenes: bool = False,
) -> tuple[dict | None, str]:
    """Read one KEYVIS; candidate names guide generation without rejecting output."""
    local_toggles = deepcopy(toggles)
    local_toggles.update({
        "key_visual": True,
        # Keep accidental scene objects in the parsed result so this validator
        # can reject the KEYVIS response whenever any scene was emitted.
        "scene_mode": "auto",
    })
    raw = None
    if mask_inline or allow_scenes:
        documents = (re.findall(r"<lb[-_]xnai[^>]*>([\s\S]*?)</lb[-_]xnai>", text or "", re.I)
                     if allow_scenes else []) or [text]
        for document in documents:
            candidate = _json_object_from_text(document)
            if isinstance(candidate, dict) and isinstance(candidate.get("keyvis"), dict):
                raw = candidate
                if len(documents) > 1:
                    print(f"[ILLUST_CONTEXT:{source}] 분리된 응답 문서에서 KEYVIS 복구: documents={len(documents)}")
                break
    if mask_inline:
        from modes import illustration_scene_pipeline
        if not isinstance(raw, dict) or not isinstance(raw.get("keyvis"), dict):
            reason = "자체 마스크 KEYVIS 응답에 keyvis object가 없습니다"
            print(f"[ILLUST_CONTEXT:{source}] {reason}: input={text!r}")
            return None, reason
        if raw.get("scenes") and not allow_scenes:
            reason = "KEYVIS 전용 응답에 scenes가 포함됨"
            print(f"[ILLUST_CONTEXT:{source}] {reason}: input={raw!r}")
            return None, reason
        try:
            descriptor = illustration_scene_pipeline._parse_detail_scene(
                {**raw["keyvis"], "slot": -1}, {"slot": -1},
                mask_inline=True,
                single_char_mask_enabled=bool(toggles.get("single_char_mask_enabled", True)),
            )
            descriptor["kind"] = "keyvis"
            descriptors = [descriptor]
        except (ValueError, TypeError) as exc:
            print(f"[ILLUST_CONTEXT:{source}] 자체 마스크 KEYVIS 파싱 실패: input={raw!r}, error={exc}")
            traceback.print_exc()
            return None, str(exc)
    elif allow_scenes:
        # Preserve a usable Key Visual even when an ordinary scene is malformed.
        descriptors = ([_descriptor(raw["keyvis"], "keyvis", -1)]
                       if isinstance(raw, dict) and isinstance(raw.get("keyvis"), dict) else [])
    else:
        descriptors = parse_toon_plan(text, local_toggles, source)
    keyvis = [
        item for item in descriptors
        if str(item.get("kind") or "") == "keyvis"
    ]
    scenes = [
        item for item in descriptors
        if str(item.get("kind") or "") == "scene"
    ]

    def fail(reason: str) -> tuple[None, str]:
        print(f"[ILLUST_CONTEXT:{source}] 독립 KEYVIS 검증 실패: {reason}")
        return None, reason

    if len(keyvis) != 1:
        return fail(f"keyvis 수 불일치: expected=1, actual={len(keyvis)}")
    if scenes and not allow_scenes:
        return fail(
            "KEYVIS 전용 응답에 scene이 포함됨: "
            f"slots={[item.get('slot') for item in scenes]}"
        )

    descriptor = keyvis[0]
    characters = [
        item for item in descriptor.get("characters") or []
        if isinstance(item, dict)
    ]
    if not str(descriptor.get("camera") or "").strip():
        return fail("background_prompt가 비어 있음" if mask_inline else "camera가 비어 있음")
    if not str(descriptor.get("scene") or "").strip():
        return fail("composition_prompt가 비어 있음" if mask_inline else "scene이 비어 있음")
    if not characters:
        return fail("regions가 비어 있음" if mask_inline else "characters가 비어 있음")
    character_limit = max(1, min(3, int(toggles.get("character_limit", 3))))
    if len(characters) > character_limit:
        return fail(
            f"완전 가시 캐릭터가 설정 상한을 초과함: "
            f"count={len(characters)}, limit={character_limit}"
        )

    actual_names = []
    for index, character in enumerate(characters, start=1):
        name = str(character.get("name") or "").strip()
        positive = str(character.get("positive") or "").strip()
        if not name:
            return fail(f"{'regions' if mask_inline else 'characters'}[{index}].name이 비어 있음")
        if not positive:
            return fail(f"regions[{index}].character_prompt가 비어 있음" if mask_inline
                        else f"characters[{index}].positive가 비어 있음")
        if name.casefold() in {value.casefold() for value in actual_names}:
            return fail(f"캐릭터 이름이 중복됨: name={name!r}")
        actual_names.append(name)

    allowed = {
        str(name or "").strip().casefold(): str(name or "").strip()
        for name in allowed_character_names
        if str(name or "").strip()
    }
    if allowed:
        unexpected = [name for name in actual_names if name.casefold() not in allowed]
        if unexpected:
            print(
                f"[ILLUST_CONTEXT:{source}] canonical 목록 밖 이름도 반환값 유지: "
                f"unexpected={unexpected}, "
                f"allowed={list(allowed.values())}"
            )
    return descriptor, ""


def descriptors_to_toon(descriptors: list[dict]) -> str:
    """Serialize merged PLAN/DETAIL descriptors into one diagnostic CALL2 block."""
    data: dict[str, object] = {"scenes": []}
    for descriptor in descriptors:
        raw = {
            "camera": str(descriptor.get("camera") or ""),
            "characters": deepcopy(descriptor.get("characters") or []),
            "scene": str(descriptor.get("scene") or ""),
            "supplement": str(descriptor.get("supplement") or ""),
        }
        if str(descriptor.get("kind") or "") == "keyvis":
            data["keyvis"] = raw
        else:
            if str(descriptor.get("plan_id") or "").strip():
                raw["plan_id"] = str(descriptor["plan_id"]).strip()
            raw["slot"] = int(descriptor.get("slot") or 0)
            raw["anonymous_partner_fragment"] = bool(
                descriptor.get("anonymous_partner_fragment", False)
            )
            data["scenes"].append(raw)
    body = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    return f"<lb-xnai>\n{body}\n</lb-xnai>"


def _authority_values_for_name(values: dict, name: str):
    folded = str(name or "").strip().casefold()
    for candidate_name, value in (values or {}).items():
        if str(candidate_name or "").strip().casefold() == folded:
            return value
    return None


def _balanced_call2_scene_plan_batches(
    scene_plan: list[dict],
    max_concurrency: int,
) -> list[list[dict]]:
    """선택/계획 장면을 PLAN 또는 DETAIL 작업 수에 서사 순서대로 균등 분배한다."""
    if not scene_plan:
        return []
    worker_count = min(max(1, int(max_concurrency)), len(scene_plan))
    base_size, larger_worker_count = divmod(len(scene_plan), worker_count)
    batch_sizes = [
        base_size + (1 if index < larger_worker_count else 0)
        for index in range(worker_count)
    ]
    batches = []
    cursor = 0
    for batch_size in batch_sizes:
        batches.append(scene_plan[cursor:cursor + batch_size])
        cursor += batch_size
    return batches


async def _run_call2_keyvis(
    *,
    call2_context_messages: list[dict],
    allowed_character_names: list[str],
    toggles: dict,
    stream_notify,
    mask_inline: bool = False,
) -> tuple[dict, str]:
    """Generate a promotional composition using supplied shared wardrobe continuity."""
    messages = deepcopy(call2_context_messages)
    allowed = [
        str(name or "").strip()
        for name in allowed_character_names
        if str(name or "").strip()
    ]
    if not allowed:
        print(
            "[ILLUST_CONTEXT:CALL2_KEYVIS] 현재 canonical roster가 비어 있음: "
            "선택된 프로필과 현재 문맥에 명시된 canonical 이름만 사용하도록 요청"
        )
    roster_text = json.dumps(allowed, ensure_ascii=False)
    messages.append(prompt_builders.keyvis_task_message(
        max(1, min(3, int(toggles.get("character_limit", 3)))), roster_text, allowed,
        mask_inline=mask_inline))
    try:
        print(
            "[ILLUST_CONTEXT:CALL2_KEYVIS] 독립 입력 준비: "
            f"messages={len(messages)}, "
            f"chars={sum(len(str(item.get('content') or '')) for item in messages)}, "
            f"allowed_characters={allowed}"
        )
    except Exception as e:
        print(f"[ILLUST_CONTEXT:CALL2_KEYVIS] 입력 크기 계산 실패: error={e}")
        traceback.print_exc()

    def validate(result):
        descriptor, reason = _parse_call2_keyvis_output(
            result,
            toggles,
            allowed,
            "CALL2-KEYVIS-RETRY-CHECK",
            mask_inline=mask_inline,
        )
        return bool(descriptor), reason or "CALL2-KEYVIS 검증 실패"

    raw_output = await _call_pipeline_llm(
        "CALL2-KEYVIS",
        _normalize_messages(messages),
        stream_notify,
        result_validator=validate,
    )
    descriptor, reason = _parse_call2_keyvis_output(
        raw_output,
        toggles,
        allowed,
        "CALL2-KEYVIS",
        mask_inline=mask_inline,
    )
    if descriptor is None:
        print(
            f"[ILLUST_CONTEXT:CALL2_KEYVIS] 최종 응답 검증 실패: reason={reason}"
        )
        raise ValueError(reason or "CALL2-KEYVIS 최종 검증 실패")
    return descriptor, raw_output


def candidate_slots(target_slotted: str) -> list[int]:
    """Return the module's ordered paragraph slots without trusting CALL output."""
    slots: list[int] = []
    seen: set[int] = set()
    for raw in re.findall(r"\[Slot\s+(\d+)\]", str(target_slotted or "")):
        slot = int(raw)
        if slot not in seen:
            seen.add(slot)
            slots.append(slot)
    return slots


def _pick_slot(raw: int, candidates: list[int], used: set[int]) -> int | None:
    """CALL2가 고른 raw 슬롯을 신뢰해 후보 중 하나를 고른다.

    raw가 valid(후보 안) & unique(미사용)면 그대로 쓴다. invalid(후보 밖)이거나
    duplicate(이미 사용됨)면 raw에 가장 가까운 미사용 후보로 보정한다.
    사용 가능한 후보가 없으면 None을 돌려 scene 드롭을 유도한다.
    """
    if not candidates:
        return None
    if raw in candidates and raw not in used:
        used.add(raw)
        return raw
    free = [c for c in candidates if c not in used]
    if not free:
        return None
    # raw와 거리가 가장 가까운 미사용 후보(동점이면 작은 값)
    chosen = min(free, key=lambda c: (abs(c - raw), c))
    used.add(chosen)
    return chosen


def _anchor_text(value: str, *, tail: bool = False, limit: int = 180) -> str:
    """Return a compact nearby-text anchor suitable for transport to Risu.

    CALL2 sees ``target_slotted`` before the server's even-slot redistribution.
    Keeping a bounded text fragment from that original boundary preserves the
    model's intended location while leaving the legacy numeric redistribution
    and image cache protocol unchanged.
    """
    compact = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[-limit:] if tail else compact[:limit]


def slot_context_anchors(target_slotted: str) -> dict[int, dict[str, str]]:
    """Build ordered before/after text anchors for every original Slot marker."""
    source = str(target_slotted or "")
    matches = list(re.finditer(r"\[Slot\s+(\d+)\]", source))
    anchors: dict[int, dict[str, str]] = {}
    for index, match in enumerate(matches):
        slot = int(match.group(1))
        previous_end = matches[index - 1].end() if index > 0 else 0
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        anchors[slot] = {
            "anchor_before": _anchor_text(source[previous_end:match.start()], tail=True),
            "anchor_after": _anchor_text(source[match.end():next_start], tail=False),
        }
    return anchors


def attach_descriptor_anchors(descriptors: list[dict], target_slotted: str) -> list[dict]:
    """Attach CALL2-boundary context before numeric slots are redistributed."""
    anchors = slot_context_anchors(target_slotted)
    attached = 0
    for item in descriptors:
        if str(item.get("kind")) != "scene":
            continue
        try:
            call2_slot = int(item.get("slot"))
        except Exception as e:
            print(f"[ILLUST_CONTEXT] CALL2 앵커 slot 파싱 실패: item={item!r}, error={e}")
            traceback.print_exc()
            item["anchor_before"] = ""
            item["anchor_after"] = ""
            continue
        pair = anchors.get(call2_slot) or {}
        item["anchor_before"] = str(pair.get("anchor_before") or "")
        item["anchor_after"] = str(pair.get("anchor_after") or "")
        item["anchor_version"] = 1
        if item["anchor_before"] or item["anchor_after"]:
            attached += 1
        else:
            print(
                f"[ILLUST_CONTEXT] CALL2 슬롯 주변 문구 없음: "
                f"slot={call2_slot}, candidates={list(anchors)}"
            )
    print(
        f"[ILLUST_CONTEXT] CALL2 문구 앵커 저장: "
        f"scenes={sum(1 for item in descriptors if str(item.get('kind')) == 'scene')}, "
        f"attached={attached}"
    )
    return descriptors


def sanitize_descriptor_slots(descriptors: list[dict], target_slotted: str) -> list[dict]:
    """CALL2가 선택한 slot을 신뢰하고, 범위 밖/중복만 보정한다.

    재배치(even redistribution)를 하지 않는다. CALL2는 target_slotted의
    [Slot N] 마커를 보고 슬롯을 고르므로, valid & unique 선택은 그대로 존중하고
    invalid(후보 밖) 또는 duplicate(이미 사용된 슬롯)만 가장 가까운 미사용 후보로 보정한다.
    후보 수보다 scene이 많으면 초과분은 드롭한다(리스 회수 프로토콜이 중복 슬롯을
    처리하지 못하므로, 빈 슬롯이 없을 때는 드롭이 안전하다).
    """
    candidates = candidate_slots(target_slotted)
    used: set[int] = set()
    normalized: list[dict] = []
    dropped = 0
    for item in descriptors:
        if str(item.get("kind")) != "scene":
            normalized.append(item)
            continue
        try:
            raw = int(item.get("slot"))
        except Exception:
            raw = -1
        chosen = _pick_slot(raw, candidates, used)
        if chosen is None:
            dropped += 1
            continue
        item["slot"] = chosen
        normalized.append(item)

    scene_slots = [it["slot"] for it in normalized if str(it.get("kind")) == "scene"]
    if dropped:
        print(
            f"[ILLUST_CONTEXT] 슬롯 후보 초과로 {dropped}개 scene 드롭: "
            f"candidates={len(candidates)}"
        )
    print(
        f"[ILLUST_CONTEXT] 장면 슬롯 보정(CALL2 신뢰): "
        f"candidates={candidates}, slots={scene_slots}"
    )
    return normalized


def normalize_call3_dialogue_output(text: str, segment_slots=None) -> str:
    """Normalize decoded dialogue JSON or legacy Scene text for the renderer."""
    source = str(text or "")
    if source.lstrip().startswith("{") or re.match(r"```(?:json)?\s*\{", source.lstrip(), re.I):
        document = _json_object_from_text(source)
        if document is None or not isinstance(document.get("scenes"), list):
            print(f"[ILLUST_CONTEXT:CALL3] 대사 JSON 파싱/필수 scenes 실패: raw={source!r}")
            return ""
        lines = []
        for scene in document["scenes"]:
            try:
                if not isinstance(scene, dict) or not isinstance(scene.get("entries"), list):
                    raise ValueError("장면 object와 entries 배열이 필요합니다")
                anchor = scene.get("slot")
                slot = (segment_slots or {}).get(str(anchor))
                if slot is None:
                    slot = int(anchor)
            except Exception as e:
                print(f"[ILLUST_CONTEXT:CALL3] 대사 JSON 장면을 읽지 못함: scene={scene!r}, error={e}")
                traceback.print_exc()
                continue
            lines.append(f"[Scene slot={slot}]")
            for entry in scene["entries"]:
                if not isinstance(entry, dict):
                    print(f"[ILLUST_CONTEXT:CALL3] 대사 JSON 엔트리 object 누락: slot={slot}, entry={entry!r}")
                    continue
                speaker = entry.get("speaker")
                body = entry.get("text")
                kind = entry.get("type", "speech")
                if (not isinstance(speaker, str) or not speaker.strip()
                        or not isinstance(body, str) or not body.strip()
                        or kind not in ("speech", "thought")):
                    print(f"[ILLUST_CONTEXT:CALL3] 대사 JSON 발화자/본문/유형 누락: slot={slot}, entry={entry!r}")
                    continue
                body = body.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
                content = f"({body})" if kind == "thought" else f'"{body}"'
                tag = str(entry.get("tag") or "").strip().lstrip("#")
                lines.append(f"{speaker.strip()}: {content}" + (f" #{tag}" if tag else ""))
        text = "\n".join(lines)
    if segment_slots:
        text = re.sub(r"^([ \t]*)\[(?:Scene\s+)?(C\d+)\](.*)$",
                      lambda match: match.group(1) + f"[Scene slot={segment_slots[match.group(2)]}]" + match.group(3)
                      if match.group(2) in segment_slots else match.group(0), str(text or ""), flags=re.M)
    output_lines: list[str] = []
    current_slot: int | None = None
    removed: list[tuple[int | None, str]] = []
    for raw_line in str(text or "").splitlines():
        header = _CALL3_SCENE_ENTRY_RE.match(raw_line)
        if header:
            current_slot = int(header.group("slot"))
            entry_text = header.group("tail").strip()
        else:
            entry_text = raw_line.strip()
        if not entry_text:
            output_lines.append(raw_line)
            continue

        segments = postprocess.parse_speak(entry_text, strip_emotion=True)
        if current_slot is not None and segments and all(
            str(segment.get("speaker") or "").strip()
            and str(segment.get("text") or "").strip()
            for segment in segments
        ):
            output_lines.append(raw_line)
            continue

        removed.append((current_slot, entry_text))
        if header:
            output_lines.append(header.group("header"))
    if removed:
        print(
            "[ILLUST_CONTEXT:CALL3] 이름: 대사/생각 형식이 아닌 줄 무시 "
            "(Scene 또는 발화자/본문 누락): "
            f"removed={removed}"
        )
    normalized = "\n".join(output_lines)
    if str(text or "").endswith("\n"):
        normalized += "\n"
    return normalized


def parse_speak_output(text: str, max_entries_per_scene: int | None = None) -> dict[int, str]:
    """Parse CALL3 Scene blocks and optionally enforce a structural entry limit."""
    result: dict[int, list[str]] = {}
    current = None
    dropped: dict[int, int] = {}
    for line in normalize_call3_dialogue_output(text).splitlines():
        match = re.match(r"\s*\[Scene\s+slot\s*=\s*(-?\d+)\]\s*(.*)", line, re.I)
        if match:
            current = int(match.group(1))
            result.setdefault(current, [])
            tail = match.group(2).strip()
            if tail:
                if max_entries_per_scene is None or len(result[current]) < max_entries_per_scene:
                    result[current].append(tail)
                else:
                    dropped[current] = dropped.get(current, 0) + 1
        elif current is not None and line.strip() and not line.lstrip().startswith("["):
            if max_entries_per_scene is None or len(result[current]) < max_entries_per_scene:
                result[current].append(line.strip())
            else:
                dropped[current] = dropped.get(current, 0) + 1
    for slot, count in dropped.items():
        print(
            f"[ILLUST_CONTEXT:CALL3] Speak 장면 발화 상한 적용: "
            f"slot={slot}, limit={max_entries_per_scene}, dropped={count}"
        )
    return {slot: "\n".join(lines).strip() for slot, lines in result.items() if lines}


def build_call3_scene_selection(
    descriptors: list[dict],
    slotted_context: str = "",
    *, segment_slots=None,
) -> tuple[list[int], str]:
    """Serialize selected scenes and bounded dialogue windows for CALL3."""
    source = str(slotted_context or "")
    marker_matches = list(re.finditer(r"\[Slot\s+(\d+)\]", source))
    marker_by_slot = {int(match.group(1)): match for match in marker_matches}
    marker_index_by_slot = {
        int(match.group(1)): index
        for index, match in enumerate(marker_matches)
    }

    def dialogue_windows(slot: int) -> tuple[str, str]:
        marker = marker_by_slot.get(slot)
        index = marker_index_by_slot.get(slot)
        if marker is None or index is None:
            return "", ""
        previous_end = marker_matches[index - 1].end() if index > 0 else 0
        next_start = (
            marker_matches[index + 1].start()
            if index + 1 < len(marker_matches)
            else len(source)
        )
        upper = re.sub(r"\[Slot\s+\d+\]", "", source[previous_end:marker.start()]).strip()
        lower = re.sub(r"\[Slot\s+\d+\]", "", source[marker.end():next_start]).strip()
        return upper[-2_000:], lower[:2_000]

    anchors = {slot: anchor for anchor, slot in (segment_slots or {}).items()}
    selected_scenes = []
    selected_slots = []
    for descriptor in descriptors:
        if str(descriptor.get("kind") or "") != "scene":
            continue
        try:
            slot = int(descriptor.get("slot"))
        except Exception as e:
            print(
                f"[ILLUST_CONTEXT:CALL3] 선택 장면 slot 직렬화 실패: "
                f"descriptor={descriptor!r}, error={e}"
            )
            traceback.print_exc()
            raise RuntimeError("CALL3 선택 장면 slot을 직렬화할 수 없습니다") from e

        characters = []
        for character in descriptor.get("characters") or []:
            if not isinstance(character, dict):
                print(
                    f"[ILLUST_CONTEXT:CALL3] 선택 장면의 캐릭터 항목 무시: "
                    f"slot={slot}, value={character!r}"
                )
                continue
            characters.append({
                "name": str(character.get("name") or "").strip(),
                "position": str(character.get("position") or "").strip(),
            })

        selected_slots.append(slot)
        upper_window, lower_window = dialogue_windows(slot)
        selected_scenes.append({
            "slot": anchors.get(slot, slot),
            "scene": str(descriptor.get("scene") or "").strip(),
            "camera": str(descriptor.get("camera") or "").strip(),
            "supplement": str(descriptor.get("supplement") or "").strip(),
            "characters": characters,
            "upper_window": upper_window,
            "lower_window": lower_window,
            "dialogue_priority": ["upper_window", "lower_window"],
        })

    return selected_slots, json.dumps(
        {"selected_scenes": selected_scenes},
        ensure_ascii=False,
        indent=2,
    )


def validate_call3_slot_coverage(
    text: str,
    expected_slots: list[int],
) -> tuple[bool, str]:
    """CALL3가 선택된 모든 slot header를 순서대로 작성했는지 검증한다.

    Scene header만 있고 본문이 비어 있는 블록은 의도적인 무대사 장면이다. 대사가
    있는 slot 목록이 아니라 header 목록으로 구조적 완전성을 판단해야 한다.
    """
    expected = list(dict.fromkeys(int(slot) for slot in expected_slots))
    parsed = parse_speak_output(text)
    populated = list(parsed)
    emitted_headers = [
        int(match.group(1))
        for match in re.finditer(
            r"(?im)^\s*\[Scene\s+slot\s*=\s*(-?\d+)\]",
            str(text or ""),
        )
    ]
    missing = [slot for slot in expected if slot not in emitted_headers]
    unexpected = [slot for slot in emitted_headers if slot not in expected]
    seen_headers = set()
    duplicates = []
    for slot in emitted_headers:
        if slot in seen_headers and slot not in duplicates:
            duplicates.append(slot)
        seen_headers.add(slot)
    silent = [slot for slot in expected if slot not in parsed]
    if emitted_headers != expected:
        reason = (
            f"CALL3 선택 slot 불일치: expected={expected}, populated={populated}, "
            f"headers={emitted_headers}, missing={missing}, unexpected={unexpected}, "
            f"duplicates={duplicates}, silent={silent}"
        )
        print(f"[ILLUST_CONTEXT:CALL3] {reason}")
        return False, reason
    return True, ""


def _call3_roster_names(character_names: str) -> list[str]:
    """CALL3의 쉼표 구분 내부 발화자 ID 목록을 입력 순서대로 반환한다."""
    names = []
    seen = set()
    for value in str(character_names or "").split(","):
        name = value.strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


_CALL3_SCENE_ENTRY_RE = re.compile(
    r"^(?P<header>\s*\[Scene\s+slot\s*=\s*(?P<slot>-?\d+)\])"
    r"(?P<spacing>\s*)(?P<tail>.*)$",
    re.I,
)


def _call3_dialogue_entries(text: str) -> list[dict]:
    """CALL3 Scene 블록에서 실제 대사/생각 엔트리와 원본 줄 위치를 반환한다."""
    entries = []
    current_slot = None
    entry_index = 0
    for line_index, raw_line in enumerate(str(text or "").splitlines()):
        header_match = _CALL3_SCENE_ENTRY_RE.match(raw_line)
        if header_match:
            current_slot = int(header_match.group("slot"))
            entry_text = header_match.group("tail").strip()
            header_tail = bool(entry_text)
        elif (
            current_slot is not None
            and raw_line.strip()
            and not raw_line.lstrip().startswith("[")
        ):
            entry_text = raw_line.strip()
            header_tail = False
        else:
            continue

        if not entry_text:
            continue
        for segment in postprocess.parse_speak(entry_text, strip_emotion=True):
            entry_index += 1
            entries.append({
                "entry": entry_index,
                "line_index": line_index,
                "slot": current_slot,
                "header_tail": header_tail,
                "speaker": str(segment.get("speaker") or "").strip(),
                "text": str(segment.get("text") or ""),
            })
    return entries


def _call3_leaked_roster_names(body: str, roster: list[str]) -> list[str]:
    """대사 본문 하나에 메타데이터용 roster ID가 들어갔는지 반환한다."""
    leaked_names = []
    for name in roster:
        # 뒤에 한국어 호칭이 공백 없이 붙은 ``Masachika군``도 잡되,
        # 더 긴 영문 식별자의 일부만 일치시키지는 않는다.
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
        if re.search(pattern, body, re.I):
            leaked_names.append(name)
    return leaked_names


def _call3_dialogue_roster_leaks(text: str, character_names: str) -> list[dict]:
    """따옴표/괄호 안 대사 본문으로 유출된 내부 발화자 ID를 찾는다.

    장소나 상황을 추론하는 키워드 매칭이 아니라, 서버가 직접 제공한 구조화 roster
    ID가 메타데이터 경계를 넘어갔는지만 검증한다. 콜론 왼쪽 speaker와 #태그는
    검사 대상이 아니다.
    """
    roster = _call3_roster_names(character_names)
    if not roster:
        return []

    leaks = []
    for entry in _call3_dialogue_entries(text):
        leaked_names = _call3_leaked_roster_names(entry["text"], roster)
        if leaked_names:
            leaks.append({
                "entry": entry["entry"],
                "speaker": entry["speaker"],
                "names": leaked_names,
            })
    return leaks


def _remove_call3_roster_leaking_dialogue_entries(
    text: str,
    character_names: str,
) -> tuple[str, list[dict]]:
    """내부 roster ID가 본문에 유출된 CALL3 엔트리만 원문에서 제거한다.

    한 줄짜리 Scene header 뒤에 대사가 붙은 형식은 대사 tail만 제거하고 header는
    보존한다. 그 결과 어떤 Scene에 대사가 하나도 남지 않으면 후속 speak_map에서
    해당 slot은 빈 문자열이 되어 말풍선/VN 대사창 후처리를 건너뛴다.
    """
    source = str(text or "")
    roster = _call3_roster_names(character_names)
    if not roster:
        print(
            "[ILLUST_CONTEXT:CALL3-RECOVERY] roster가 비어 있어 유출 대사 제거 불가: "
            f"character_names={character_names!r}"
        )
        return source, []

    removed = []
    leaking_lines = set()
    header_tail_lines = set()
    for entry in _call3_dialogue_entries(source):
        leaked_names = _call3_leaked_roster_names(entry["text"], roster)
        if not leaked_names:
            continue
        leaking_lines.add(entry["line_index"])
        if entry["header_tail"]:
            header_tail_lines.add(entry["line_index"])
        removed.append({
            "entry": entry["entry"],
            "slot": entry["slot"],
            "speaker": entry["speaker"],
            "names": leaked_names,
        })

    if not removed:
        print(
            "[ILLUST_CONTEXT:CALL3-RECOVERY] 제거할 roster ID 유출 대사를 찾지 못함: "
            f"character_names={character_names!r}"
        )
        return source, []

    output_lines = []
    for line_index, raw_line in enumerate(source.splitlines()):
        if line_index not in leaking_lines:
            output_lines.append(raw_line)
            continue
        if line_index in header_tail_lines:
            header_match = _CALL3_SCENE_ENTRY_RE.match(raw_line)
            if header_match:
                output_lines.append(header_match.group("header"))
                continue
            print(
                "[ILLUST_CONTEXT:CALL3-RECOVERY] Scene header tail 제거 중 header 재파싱 실패: "
                f"line_index={line_index}, line={raw_line!r}"
            )
        # 일반 대사 줄이거나 header 재파싱에 실패한 유출 줄이면 줄 전체를 제거한다.

    sanitized = "\n".join(output_lines)
    if source.endswith("\n"):
        sanitized += "\n"
    return sanitized, removed


def _call3_scene_blocks(text: str) -> list[dict]:
    """CALL3 텍스트를 header와 본문 줄로 분리한다."""
    blocks = []
    current = None
    for raw_line in str(text or "").splitlines():
        header_match = _CALL3_SCENE_ENTRY_RE.match(raw_line)
        if header_match:
            if current is not None:
                blocks.append(current)
            current = {
                "slot": int(header_match.group("slot")),
                "lines": [],
            }
            tail = header_match.group("tail").strip()
            if tail:
                current["lines"].append(tail)
            continue
        if current is not None:
            current["lines"].append(raw_line)
    if current is not None:
        blocks.append(current)
    return blocks


def recover_call3_partial_output(
    text: str,
    expected_slots: list[int],
    character_names: str,
    output_language: str,
) -> tuple[str, dict]:
    """안전한 CALL3 블록만 보존하고 나머지 slot은 명시적 무대사로 복구한다.

    슬롯 판단은 Scene header라는 구조화 데이터만 사용한다. 예상 밖 블록과 중복
    블록은 버리고, 빠진 블록은 빈 header로 채운다. 현지화 대사에 내부 roster ID가
    유출된 경우에는 기존 엔트리 단위 제거기를 적용한다.
    """
    expected = list(dict.fromkeys(int(slot) for slot in expected_slots))
    source = normalize_call3_dialogue_output(text)
    removed_entries = []
    if (
        source
        and _call3_dialogue_requires_localized_names(output_language)
        and _call3_dialogue_roster_leaks(source, character_names)
    ):
        source, removed_entries = _remove_call3_roster_leaking_dialogue_entries(
            source,
            character_names,
        )

    blocks_by_slot = {}
    unexpected_headers = []
    duplicate_headers = []
    emitted_headers = []
    expected_set = set(expected)
    for block in _call3_scene_blocks(source):
        slot = int(block["slot"])
        emitted_headers.append(slot)
        if slot not in expected_set:
            unexpected_headers.append(slot)
            continue
        if slot in blocks_by_slot:
            duplicate_headers.append(slot)
            continue
        blocks_by_slot[slot] = block

    missing_headers = [slot for slot in expected if slot not in blocks_by_slot]
    normalized_blocks = []
    for slot in expected:
        block = blocks_by_slot.get(slot) or {"lines": []}
        body_lines = list(block.get("lines") or [])
        while body_lines and not str(body_lines[0]).strip():
            body_lines.pop(0)
        while body_lines and not str(body_lines[-1]).strip():
            body_lines.pop()
        normalized = f"[Scene slot={slot}]"
        if body_lines:
            normalized += "\n" + "\n".join(body_lines)
        normalized_blocks.append(normalized)
    normalized_output = "\n\n".join(normalized_blocks)

    residual_leaks = _call3_dialogue_roster_leaks(
        normalized_output,
        character_names,
    )
    if residual_leaks and _call3_dialogue_requires_localized_names(output_language):
        print(
            "[ILLUST_CONTEXT:CALL3-RECOVERY] 부분 복구 후 roster ID 유출이 남아 "
            "모든 CALL3 대사를 포기하고 무대사 header만 유지: "
            f"leaks={residual_leaks}, slots={expected}"
        )
        normalized_output = "\n\n".join(
            f"[Scene slot={slot}]" for slot in expected
        )

    populated_slots = list(parse_speak_output(normalized_output))
    silent_slots = [slot for slot in expected if slot not in populated_slots]
    metadata = {
        "emitted_headers": emitted_headers,
        "missing_headers": missing_headers,
        "unexpected_headers": unexpected_headers,
        "duplicate_headers": duplicate_headers,
        "populated_slots": populated_slots,
        "silent_slots": silent_slots,
        "removed_entries": removed_entries,
        "residual_leaks": residual_leaks,
    }
    print(
        "[ILLUST_CONTEXT:CALL3-RECOVERY] 슬롯별 부분 복구 성공: "
        f"populated={populated_slots}, silent={silent_slots}, "
        f"missing_headers={missing_headers}, unexpected={unexpected_headers}, "
        f"duplicates={duplicate_headers}, removed={removed_entries}"
    )
    return normalized_output, metadata


def _merge_call3_slot_repairs(
    baseline: str,
    repaired: str,
    expected_slots: list[int],
    repair_slots: list[int],
) -> str:
    """Replace only explicitly repaired Scene blocks and preserve every other block."""
    baseline_by_slot = {
        int(block["slot"]): block for block in _call3_scene_blocks(baseline)
    }
    repaired_by_slot = {
        int(block["slot"]): block for block in _call3_scene_blocks(repaired)
    }
    repair_set = {int(slot) for slot in repair_slots}
    normalized_blocks = []
    for slot in expected_slots:
        block = (
            repaired_by_slot.get(int(slot))
            if int(slot) in repair_set
            else baseline_by_slot.get(int(slot))
        ) or {"slot": int(slot), "lines": []}
        body_lines = list(block.get("lines") or [])
        while body_lines and not str(body_lines[0]).strip():
            body_lines.pop(0)
        while body_lines and not str(body_lines[-1]).strip():
            body_lines.pop()
        normalized = f"[Scene slot={int(slot)}]"
        if body_lines:
            normalized += "\n" + "\n".join(body_lines)
        normalized_blocks.append(normalized)
    return "\n\n".join(normalized_blocks)


def _call3_dialogue_requires_localized_names(output_language: str) -> bool:
    """영어 출력에서는 roster ID 자체가 자연스러운 고유명일 수 있다."""
    language = str(output_language or "").strip().casefold()
    return language not in {"영어", "english", "en"}


def validate_call3_output_contract(
    text: str,
    expected_slots: list[int],
    character_names: str,
    output_language: str,
) -> tuple[bool, str]:
    """CALL3의 slot 완전성과 대사 본문/내부 ID 경계를 함께 검증한다."""
    valid, reason = validate_call3_slot_coverage(text, expected_slots)
    if not valid:
        return False, reason

    if not _call3_dialogue_requires_localized_names(output_language):
        print(
            "[ILLUST_CONTEXT:CALL3] 영어 대사 출력이므로 roster ID 본문 유출 검사를 "
            f"적용하지 않음: language={output_language!r}"
        )
        return True, ""

    roster = _call3_roster_names(character_names)
    if not roster:
        print(
            "[ILLUST_CONTEXT:CALL3] roster가 비어 있어 대사 본문 내부 ID 검사를 "
            f"건너뜀: language={output_language!r}"
        )
        return True, ""

    leaks = _call3_dialogue_roster_leaks(text, character_names)
    if leaks:
        reason = (
            "CALL3 대사 본문에 내부 발화자 ID 유출: "
            f"language={output_language!r}, leaks={leaks}"
        )
        print(f"[ILLUST_CONTEXT:CALL3] {reason}")
        return False, reason
    return True, ""


def build_call3_dialogue_system_prompt(
    prompts: dict,
    toggles: dict,
    extra_names: str,
) -> tuple[str, str]:
    """Select the Speak/Manga/Subtitle prompt and append compatible instructions."""
    prompt_mode = str(toggles.get("call3_prompt_mode") or "speak").strip().lower()
    prompt_key = {
        "manga": "call3_manga",
        "subtitle": "call3_subtitle",
    }.get(prompt_mode, "call3_speak")
    selected_prompt = str(prompts.get(prompt_key) or "").strip()
    if not selected_prompt:
        print(
            f"[ILLUST_CONTEXT:CALL3] 선택한 대사 프롬프트가 비어 있음: "
            f"mode={prompt_mode}, key={prompt_key}"
        )
        raise RuntimeError(f"CALL3 {prompt_mode} 프롬프트가 비어 있습니다")

    emotion_instruction = ""
    if prompt_mode == "speak" and toggles.get("speak_emotion_enabled"):
        emotion_instruction = "\nSet each emitted entry's tag field to one emotion label, without the # prefix."
        emotions = str(toggles.get("speak_emotions") or "").strip()
        if emotions:
            emotion_instruction += " Allowed labels: " + variable(emotions, "허용 감정 태그")
    elif prompt_mode == "manga" and toggles.get("speak_emotion_enabled"):
        print("[ILLUST_CONTEXT:CALL3] Manga 모드에서는 감정 태그 설정을 사용하지 않음")
    elif prompt_mode == "subtitle" and toggles.get("speak_emotion_enabled"):
        print("[ILLUST_CONTEXT:CALL3] Subtitle 모드에서는 감정 태그 설정을 사용하지 않음")

    # NSFW 버블 타입(#nsfw_soft/#nsfw_hard) 보강 블록. manga 모드이고 nsfw 토글이
    # 켜져 있을 때만 manga 프롬프트 끝에 붙인다. 일반 장면엔 노출되지 않는다.
    nsfw_instruction = ""
    if prompt_mode == "manga" and toggles.get("nsfw"):
        nsfw_block = str(prompts.get("call3_manga_nsfw") or "").strip()
        if nsfw_block:
            nsfw_instruction = (
                "\n[조건: " + variable("", "만화 NSFW 설정") + " 활성화]\n" + nsfw_block
                if is_symbolic() else "\n" + variable(nsfw_block, "NSFW 설정에 따라 추가된 만화 지침")
            )
        else:
            print("[ILLUST_CONTEXT:CALL3] nsfw 토글 ON이나 manga_nsfw 프롬프트가 비어 있어 SOFT/HARD 타입 미주입")

    # CALL3에는 lb.extra 중 캐릭터 영문 이름 리스트만 넘긴다(시스템 프롬프트/복장 제외).
    # speak/manga 프롬프트의 {character_names} 자리표시자를 치환한다.
    names = str(extra_names or "").strip()
    if "{character_names}" in selected_prompt:
        system_prompt = PromptText(selected_prompt).replace("{character_names}", variable(names, "캐릭터 이름 목록"))
    elif names:
        system_prompt = selected_prompt + "\n\nCharacter names: " + variable(names, "캐릭터 이름 목록")
    else:
        system_prompt = selected_prompt
    output_language = str(toggles.get("speak_language") or "한국어").strip() or "한국어"
    roster_boundary_instruction = ""
    if _call3_dialogue_requires_localized_names(output_language):
        roster_boundary_instruction = (
            "\nAn exact roster identifier may appear as machine-readable metadata only in the "
            "speaker field. Do not repeat it in the text field. Inside the text body, use a natural " + variable(output_language, "출력 언어") + " form of address "
            "supported by the original narrative and bounded conversation, or omit direct "
            "address when uncertain."
        )
    language_instruction = (
        "# OUTPUT LANGUAGE — HARD REQUIREMENT\n"
        + "Write every dialogue, thought, inner monologue, and newly created reaction in " + variable(output_language, "출력 언어") + ".\n"
        "Character names, scene slot anchors, JSON keys, type values, and style tags may remain in their "
        "prescribed form. Do not switch the spoken text to another language even when the source "
        "narrative or examples use another language. Before answering, silently verify that every "
        "spoken or thought line follows the required output language: " + variable(output_language, "출력 언어") + "."
        + roster_boundary_instruction
    )
    system_prompt = language_instruction + "\n\n" + system_prompt
    system_prompt += emotion_instruction
    system_prompt += nsfw_instruction
    return prompt_mode, system_prompt


def build_call3_scene_request(
    original_narrative: str,
    selected_scene_payload: str,
    output_language: str,
    prompt_mode: str,
) -> str:
    """Build the scene request without exposing internal stage names in subtitle mode."""
    scene_heading = "Selected illustrated scenes"
    return (
        "[Original narrative]\n" + variable(original_narrative, "현재 원문")
        + f"\n\n[{scene_heading}]\n" + variable(selected_scene_payload, "선택된 삽화 장면")
        + "\n\nLanguage: " + variable(output_language, "출력 언어")
    )


def _build_character_history(extra_reference: str) -> str:
    # 서버가 보유한 lb.extra 자체가 가장 안정적인 외형 이력/영문 이름 사전이다.
    return str(extra_reference or "").strip()


# 삽화 CALL 이름 → 외부 LLM 분기 task_key. 기존 illustration_call2는 DETAIL과
# ENQUEUE 경로에 유지하고 SELECT/PLAN/KEYVIS는 독립 분리한다.
_CALL_TASK_KEYS = {
    "ORIGINAL-ASSET": "illustration_original_asset",
    "ORIGINAL-ASSET-RECOVERY": "illustration_original_asset_recovery",
    "OUTFIT": "illustration_call1",
    "CALL1": "illustration_call1",
    "TIME & PLACE": "illustration_time_place",
    "CHARACTER-RESOLVE": "illustration_character_resolve",
    "CHARACTER-RESOLVE-REPAIR": "illustration_character_resolve",
    "PROFILE-RESOLVE": "illustration_profile_resolve",
    "PROFILE-RESOLVE-REPAIR": "illustration_profile_resolve",
    PROFILE_CONTEXT_TRANSLATE_CALL_NAME: PROFILE_CONTEXT_TRANSLATE_TASK_KEY,
    "CALL2": "illustration_call2",
    "CALL2-SELECT": "illustration_call2_select",
    "CALL2-PLAN": "illustration_call2_plan",
    "CALL2-KEYVIS": "illustration_call2_keyvis",
    "CALL2-FALLBACK": "illustration_call2",
    "CALL3": "illustration_call3",
    "CALL3-CORRECTION": "illustration_call3",
    "MULTI-CHAR-MASK": "illustration_multi_char_mask",
}

# 상태 추적·프로필·CALL2·CALL3 호출을 큐 서브태스크로 표시하기 위한 그룹 정의.
# 다중캐릭터마스크(MULTI-CHAR-MASK)는 병렬 청크용 wrapper가
# index/total을 직접 주입하므로 여기서 제외한다.
_CALL_LAYOUT_ORDER = {
    "call3": -100,
    "call3_correction": -90,
}


_CALL_QUEUE_SUBTASK_GROUPS = {
    "ORIGINAL-ASSET": ("original_asset", "원본 에셋 선택"),
    "ORIGINAL-ASSET-RECOVERY": (
        "original_asset_recovery",
        "원본 에셋 실패 항목 복구",
    ),
    "OUTFIT": ("call1", "OUTFIT 복장·헤어 분석"),
    "CALL1": ("call1", "OUTFIT 복장·헤어 분석"),
    "TIME & PLACE": ("time_place", "TIME & PLACE 시간·장소 추적"),
    "CHARACTER-RESOLVE": ("character_resolve", "CURRENT 등장인물 판별"),
    "CHARACTER-RESOLVE-REPAIR": (
        "character_resolve",
        "CURRENT 등장인물 이름 교정",
    ),
    "PROFILE-RESOLVE": ("profile_resolve", "CURRENT 프로필 캐릭터별 병렬 결정"),
    "PROFILE-RESOLVE-REPAIR": ("profile_resolve", "프로필 오류 항목 교정"),
    PROFILE_CONTEXT_TRANSLATE_CALL_NAME: (
        "profile_context_translation",
        "프로필 영어 컨텍스트 번역",
    ),
    "CALL2": ("call2", "CALL2 장면/태그 빌드"),
    "CALL2-SELECT": ("call2_select", "SLOT-SELECTER 장면 선택"),
    "CALL2-PLAN": ("call2_plan", "CALL2 장면 PLAN"),
    "CALL2-DETAIL": ("call2_detail", "DETAIL 장면 묘사"),
    "CALL2-KEYVIS": ("call2_keyvis", "KEYVIS 대표 이미지"),
    "CALL2-FALLBACK": ("call2", "CALL2 폴백"),
    "CALL3": ("call3", "POST PROCESS 대사 빌드"),
    "CALL3-CORRECTION": ("call3_correction", "POST PROCESS 슬롯/언어 교정"),
}


@illustration_flow.llm_call
async def _call_pipeline_llm(
    call_name: str,
    messages: list[dict],
    stream_notify=None,
    result_validator=None,
    json_mode: bool = False,
    stream_observer=None,
    history_id: str = "",
    parent_execution_id: str = "",
    history_ids_sink: list[str] | None = None,
    force_slot: str | None = None,
) -> str:
    """삽화 파이프라인 LLM 호출. 단계별 외부 LLM 분기 task_key를 경유한다.

    외부 LLM 분기 탭에서 CALL별로 LLM1/LLM2/LLM3 을 선택하거나 폴백을 켤 수 있다.
    실패 시 callLLMTask 가 지정된 폴백 LLM 으로 재시도한다.
    """
    started = time.time()
    task_key = _CALL_TASK_KEYS.get(call_name)
    if task_key is None and call_name.startswith(("OUTFIT ", "CALL1 ")):
        task_key = _CALL_TASK_KEYS["OUTFIT"]
    if task_key is None and call_name.startswith("CALL2-PLAN"):
        task_key = _CALL_TASK_KEYS["CALL2-PLAN"]
    if task_key is None and call_name.startswith("CALL2-DETAIL"):
        task_key = _CALL_TASK_KEYS["CALL2"]
    if task_key is None and call_name.startswith("CALL2-KEYVIS"):
        task_key = _CALL_TASK_KEYS["CALL2-KEYVIS"]
    if task_key is None and call_name.startswith("MULTI-CHAR-MASK"):
        task_key = _CALL_TASK_KEYS["MULTI-CHAR-MASK"]
    if task_key is None:
        print(
            f"[ILLUST_CONTEXT:{call_name}] 등록되지 않은 CALL 이름, "
            "illustration_call2 라우팅 사용"
        )
        task_key = "illustration_call2"
    execution_id = str(history_id or uuid.uuid4().hex)
    parent_execution_id = str(parent_execution_id or "")
    # 이 호출의 메인 레코드(성공/취소/예외 모두 동일 history_id)를 trace에 등록.
    # MULTI-CHAR-MASK는 slot(장면)마다 별도 호출되므로 전역 trace 대신 호출자가 넘긴
    # sink(history_ids_sink)에만 담아 백업별로 자기 slot 것만 주입되게 한다.
    if history_ids_sink is not None:
        if execution_id not in history_ids_sink:
            history_ids_sink.append(execution_id)
    else:
        _trace_append(execution_id)
    model = llm_service.routing_primary_model(task_key)
    service = llm_service.routing_primary_service(task_key) or ""
    base_call_name = call_name.split()[0] if call_name else ""
    queue_subtask_group = (_CALL_QUEUE_SUBTASK_GROUPS.get(call_name)
                           or _CALL_QUEUE_SUBTASK_GROUPS.get(base_call_name))
    illustration_flow.llm_metadata(
        execution_id=execution_id,
        task_key=task_key,
        model=model,
        service=service,
        layout_group=(queue_subtask_group[0] if queue_subtask_group else task_key),
        layout_order=(
            _CALL_LAYOUT_ORDER.get(queue_subtask_group[0], 0)
            if queue_subtask_group
            else 0
        ),
    )
    history_record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "prompt_id": f"illustration_context:{call_name}",
        "call_name": call_name,
        "task_key": task_key,
        "model": model,
        "service": service,
        "input": deepcopy(messages),
        "input_variables": input_variables(messages),
        "output": "",
        "completion_tokens": 0,
        "elapsed": 0.0,
        "tps": 0.0,
        "history_id": execution_id,
        "execution_id": execution_id,
        "parent_execution_id": parent_execution_id,
        # 각 시도의 실제 슬롯(폴백 포함)을 성공·실패·취소 기록에 보존한다.
        "phase": "",
        "llm_slot": llm_service.routing_primary_slot(task_key),
        "attempt": 0,
        "total_attempts": 0,
        "attempt_id": "",
        "finish_reason": "",
        "finish_message": "",
        "max_output_tokens": llm_service.slot_max_output_tokens(
            llm_service.routing_primary_slot(task_key)
        ),
    }
    history_logged = False
    terminal_notified = False
    attempt_meta: dict = {}
    provider_metadata: dict = {}
    provider_detail_keys = (
        "completion_tokens",
        "prompt_tokens",
        "elapsed",
        "tps",
        "ttft",
        "finish_reason",
        "finish_message",
        "max_output_tokens",
        "pdf_tokens_before",
        "pdf_pages_after",
        "reasoning_tokens",
        "reasoning_chars",
        "http_status",
        "http_headers_elapsed",
        "http_request_sent_elapsed",
        "http_request_bytes",
        "http_request_sha256",
        "first_byte_elapsed",
        "last_raw_elapsed",
        "raw_bytes",
        "transport_raw_response",
        *llm_service.LLM_STREAM_TIMING_FIELDS,
    )

    def _provider_details() -> dict:
        return {
            key: provider_metadata[key]
            for key in provider_detail_keys
            if key in provider_metadata and provider_metadata[key] is not None
        }

    async def _notify(event: dict):
        # stream_notify 이벤트에 큐 서브태스크 그룹을 주입한다.
        # 역번역/다중캐릭터마스크 wrapper가 이미 queue_subtask를 넣은 경우 유지한다.
        if not stream_notify:
            return
        event.setdefault("execution_id", execution_id)
        event.setdefault("parent_execution_id", parent_execution_id)
        if "queue_subtask" not in event:
            base = call_name.split()[0] if call_name else ""
            grp = (_CALL_QUEUE_SUBTASK_GROUPS.get(call_name)
                   or _CALL_QUEUE_SUBTASK_GROUPS.get(base))
            if grp:
                event["queue_subtask"] = {
                    "group_id": grp[0],
                    "group_label": grp[1],
                    "index": 1,
                    "total": 1,
                }
        await stream_notify(event)

    async def _record_execution_event(event: dict) -> None:
        """라우팅 재시도 이벤트를 LB 자세히 이력에 반영한다.

        attempt_success: 최종 채택된 시도의 phase/slot/attempt 등을 잡아
        성공 레코드의 'LLM 실행 연결 정보'를 채운다(폴백 슬롯 포함).
        attempt_failure: 버려지는 실패 응답을 별도 error 레코드로 남긴다.
        """
        etype = str(event.get("type") or "")
        if etype == "attempt_start":
            provider_metadata.clear()
            history_record["input"] = deepcopy(messages)
            history_record["input_variables"] = input_variables(messages)
        recorded_event = dict(event)
        if etype in {"attempt_success", "attempt_failure", "execution_complete"}:
            recorded_event.update(_provider_details())
        slot = recorded_event.get("slot") or recorded_event.get("llm_slot") or llm_service.routing_primary_slot(task_key)
        suffix = llm_service._slot_suffix(slot)
        history_record.update({
            "llm_slot": slot,
            "model": llm_service._base_config_get(f"llm_model{suffix}", "") or "",
            "service": llm_service._base_config_get(f"llm_service{suffix}", "") or "openai",
        })
        illustration_flow.llm_attempt(
            recorded_event,
            llm_service._base_config_get(f"llm_model{suffix}", "") or "",
            llm_service._base_config_get(f"llm_service{suffix}", "") or "openai",
        )
        event = recorded_event
        if etype in {"attempt_start", "attempt_success", "attempt_failure"}:
            attempt_meta.clear()
            attempt_meta.update({
                "phase": str(event.get("phase") or ""),
                "llm_slot": str(event.get("llm_slot") or event.get("slot") or ""),
                "attempt": int(event.get("attempt") or 0),
                "total_attempts": int(event.get("total_attempts") or 0),
                "attempt_id": str(event.get("attempt_id") or ""),
            })
            if etype != "attempt_failure":
                return
        if etype != "attempt_failure":
            return
        raw_result = event.get("raw_response", event.get("result"))
        attempt_exception = event.get("error") or event.get("exception")
        raw_output = "" if raw_result is None else str(raw_result)
        if not raw_output and attempt_exception is not None:
            raw_output = str(attempt_exception)
        history_record["output"] = raw_output
        reason = str(event.get("reason") or raw_output or "LLM 시도 실패")
        failure_record = dict(history_record)
        failure_record.update({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "history_id": uuid.uuid4().hex,
            "parent_history_id": execution_id,
            "execution_id": execution_id,
            "parent_execution_id": parent_execution_id,
            "attempt_id": str(event.get("attempt_id") or ""),
            "phase": str(event.get("phase") or ""),
            "llm_slot": str(event.get("slot") or ""),
            "attempt": int(event.get("attempt") or 0),
            "total_attempts": int(event.get("total_attempts") or 0),
            "output": raw_output,
            "elapsed": round(time.time() - started, 3),
            "status": "error",
            "error": reason,
        })
        failure_record.update(_provider_details())
        print(
            f"[ILLUST_CONTEXT:{call_name}] LLM 시도 실패 기록: "
            f"phase={failure_record['phase']}, slot={failure_record['llm_slot']}, "
            f"attempt={failure_record['attempt']}/{failure_record['total_attempts']}, "
            f"reason={reason}, raw={raw_output[:300]!r}"
        )
        # 버려지는 실패 시도도 별도 error 레코드로 남으므로 trace에 포함.
        # MULTI-CHAR-MASK 경로(history_ids_sink 사용)는 sink에만 담는다.
        _failure_hid = failure_record.get("history_id")
        if history_ids_sink is not None:
            if _failure_hid and _failure_hid not in history_ids_sink:
                history_ids_sink.append(_failure_hid)
        else:
            _trace_append(_failure_hid)
        lighbd_service._log_lighbd_history(failure_record)

    try:
        if stream_notify:
            await _notify({
                "type": "start", "call_name": call_name, "model": model, "text": "",
            })
        call_kwargs = {}
        call_kwargs["execution_id"] = execution_id
        call_kwargs["parent_execution_id"] = parent_execution_id
        call_kwargs["execution_observer"] = _record_execution_event
        call_kwargs["metadata_sink"] = provider_metadata
        if result_validator is not None:
            call_kwargs["result_validator"] = result_validator
        if json_mode:
            call_kwargs["json_mode"] = True
        if stream_observer is not None:
            call_kwargs["stream_observer"] = stream_observer
        if force_slot is not None:
            call_kwargs["force_slot"] = force_slot
        stream_metadata_token = llm_service._stream_metadata_ctx.set({
            "task_key": task_key,
            "call_name": call_name,
            "execution_id": execution_id,
            "parent_execution_id": parent_execution_id,
        })
        base64_response_mode_token = None
        if call_name.startswith("CALL2-DETAIL"):
            # The validator retains healthy scenes and narrows this list before
            # the next routed attempt. Other callers keep immutable requests.
            call_kwargs["retry_messages"] = messages
            base64_response_mode_token = (
                llm_service._gemini_base64_response_mode_ctx.set(
                    llm_service.GEMINI_BASE64_RESPONSE_ILLUSTRATION_SCENES
                )
            )
        try:
            result = await llm_service.callLLMTask(task_key, messages, **call_kwargs)
        finally:
            if base64_response_mode_token is not None:
                llm_service._gemini_base64_response_mode_ctx.reset(
                    base64_response_mode_token
                )
            llm_service._stream_metadata_ctx.reset(stream_metadata_token)
        if not result or str(result).startswith("[LLM 실패]"):
            print(f"[ILLUST_CONTEXT:{call_name}] LLM 호출 실패: {result}")
            if stream_notify:
                await _notify({"type": "error", "call_name": call_name, "error": str(result)})
                terminal_notified = True
            raise RuntimeError(str(result or f"빈 {call_name} 응답"))
        elapsed = time.time() - started
        tokens = int(
            provider_metadata.get("completion_tokens")
            if provider_metadata.get("completion_tokens") is not None
            else max(1, len(str(result)) // 3)
        )
        prompt_tokens = int(
            provider_metadata.get("prompt_tokens")
            if provider_metadata.get("prompt_tokens") is not None
            else llm_service._approx_input_tokens(messages)
        )
        tps = float(
            provider_metadata.get("tps")
            if provider_metadata.get("tps") is not None
            else (tokens / elapsed if elapsed > 0 else 0.0)
        )
        illustration_flow.llm_metadata(tps=tps)
        ttft = float(
            provider_metadata.get("ttft")
            if provider_metadata.get("ttft") is not None
            else elapsed
        )
        if stream_notify:
            await _notify({
                "type": "done",
                "call_name": call_name,
                "model": model,
                "text": str(result),
                "completion_tokens": tokens,
                "prompt_tokens": prompt_tokens,
                "elapsed": elapsed,
                "tps": tps,
                "ttft": ttft,
                "finish_reason": provider_metadata.get("finish_reason", ""),
                "finish_message": provider_metadata.get("finish_message", ""),
                "max_output_tokens": provider_metadata.get(
                    "max_output_tokens",
                    history_record["max_output_tokens"],
                ),
                **{key: provider_metadata.get(key) for key in llm_service.LLM_STREAM_TIMING_FIELDS},
            })
            terminal_notified = True
        history_record.update({
            "output": str(result),
            "completion_tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": round(elapsed, 3),
            "tps": round(tps, 1),
            "ttft": round(ttft, 3),
            "status": "ok",
            "phase": attempt_meta.get("phase", ""),
            "llm_slot": attempt_meta.get("llm_slot", ""),
            "attempt": attempt_meta.get("attempt", 0),
            "total_attempts": attempt_meta.get("total_attempts", 0),
            "attempt_id": attempt_meta.get("attempt_id", ""),
            "finish_reason": provider_metadata.get("finish_reason", ""),
            "finish_message": provider_metadata.get("finish_message", ""),
            "max_output_tokens": provider_metadata.get(
                "max_output_tokens",
                history_record["max_output_tokens"],
            ),
        })
        history_record.update(_provider_details())
        lighbd_service._log_lighbd_history(history_record)
        history_logged = True
        return str(result)
    except asyncio.CancelledError:
        if stream_notify and not terminal_notified:
            try:
                await _notify({
                    "type": "cancelled",
                    "call_name": call_name,
                    "reason": "parent_cancelled",
                })
                terminal_notified = True
            except Exception as notify_error:
                print(
                    f"[ILLUST_CONTEXT:{call_name}] 취소 스트림 알림 실패: "
                    f"{notify_error}"
                )
                traceback.print_exc()
        if execution_id and not history_logged:
            elapsed = time.time() - started
            history_record.update({
                "elapsed": round(elapsed, 3),
                "status": "cancelled",
                "error": "선착순 경주에서 패배해 취소됨",
            })
            history_record.update(attempt_meta)
            history_record.update(_provider_details())
            lighbd_service._log_lighbd_history(history_record)
            history_logged = True
        print(
            f"[ILLUST_CONTEXT:{call_name}] LLM 호출 취소: "
            f"history_id={execution_id}"
        )
        raise
    except Exception as e:
        if stream_notify and not terminal_notified:
            try:
                await _notify({
                    "type": "error",
                    "call_name": call_name,
                    "error": str(e),
                })
                terminal_notified = True
            except Exception as notify_error:
                print(
                    f"[ILLUST_CONTEXT:{call_name}] 오류 스트림 알림 실패: "
                    f"{notify_error}"
                )
                traceback.print_exc()
        if not history_logged:
            elapsed = time.time() - started
            history_record.update({
                "elapsed": round(elapsed, 3),
                "status": "error",
                "error": str(e),
            })
            history_record.update(attempt_meta)
            history_record.update(_provider_details())
            lighbd_service._log_lighbd_history(history_record)
        print(f"[ILLUST_CONTEXT:{call_name}] 호출 예외: {e}")
        traceback.print_exc()
        raise


async def _build_call3_dialogue_with_recovery(
    speak_messages: list[dict],
    selected_slots: list[int],
    character_names: str,
    speak_language: str,
    stream_notify=None,
    call_name: str = "CALL3",
    correction_call_name: str = "CALL3-CORRECTION",
    *, segment_slots=None,
) -> dict:
    """CALL3를 실행하고 실패 범위를 대사 슬롯 안으로 격리한다."""
    state = {
        "output": "",
        "initial_output": "",
        "correction_used": False,
        "partial_recovery_used": False,
        "dialogue_drop_recovery_used": False,
        "dropped_dialogue_entries": [],
        "silent_slots": [],
        "failure_reason": "",
    }
    normalized_messages = _normalize_messages(speak_messages)
    try:
        initial_output = await _call_pipeline_llm(
            call_name,
            normalized_messages,
            stream_notify,
            json_mode=True,
        )
    except Exception as initial_error:
        state["failure_reason"] = str(initial_error)
        print(
            "[ILLUST_CONTEXT:CALL3-RECOVERY] 최초 CALL3 호출/라우팅이 소진되어 "
            "모든 선택 슬롯을 무대사로 두고 이미지 파이프라인 계속: "
            f"slots={selected_slots}, error={type(initial_error).__name__}: "
            f"{initial_error}"
        )
        traceback.print_exc()
        recovered, recovery = recover_call3_partial_output(
            "",
            selected_slots,
            character_names,
            speak_language,
        )
        state.update({
            "output": recovered,
            "partial_recovery_used": True,
            "silent_slots": recovery["silent_slots"],
        })
        return state

    state["initial_output"] = initial_output
    normalized_initial_output = normalize_call3_dialogue_output(initial_output, segment_slots)
    call3_valid, call3_failure_reason = validate_call3_output_contract(
        normalized_initial_output,
        selected_slots,
        character_names,
        speak_language,
    )
    if call3_valid:
        state["output"] = normalized_initial_output
        state["silent_slots"] = [
            slot for slot in selected_slots
            if slot not in parse_speak_output(normalized_initial_output)
        ]
        return state

    state["failure_reason"] = call3_failure_reason
    baseline_output, baseline_recovery = recover_call3_partial_output(
        normalized_initial_output,
        selected_slots,
        character_names,
        speak_language,
    )
    removed_entries = list(baseline_recovery["removed_entries"])
    repair_slots = list(dict.fromkeys([
        *baseline_recovery["missing_headers"],
        *[
            int(entry["slot"])
            for entry in removed_entries
            if entry.get("slot") is not None
        ],
    ]))
    # Header order, unexpected slots, and duplicates are structural defects that
    # the server can repair deterministically. Only genuinely missing dialogue
    # slots or slots containing an unsafe localized roster leak need another LLM.
    if not repair_slots:
        print(
            "[ILLUST_CONTEXT:CALL3-RECOVERY] LLM 없이 구조만 슬롯별 정규화: "
            f"slots={selected_slots}, reason={call3_failure_reason}"
        )
        state.update({
            "output": baseline_output,
            "partial_recovery_used": True,
            "dialogue_drop_recovery_used": bool(removed_entries),
            "dropped_dialogue_entries": removed_entries,
            "silent_slots": baseline_recovery["silent_slots"],
        })
        return state

    state["correction_used"] = True
    state["partial_recovery_used"] = True
    state["dialogue_drop_recovery_used"] = bool(removed_entries)
    state["dropped_dialogue_entries"] = removed_entries
    # 검증/로그에는 내부 호출명을 유지하되 교정 LLM에는 실패 내용만 전달한다.
    # CALL 번호는 서버 라우팅 명칭일 뿐 모델의 역할 설명이 아니다.
    failure_role = "dialogue output"
    model_failure_reason = re.sub(
        r"\bCALL[1235](?:-[A-Z-]+)?\b",
        failure_role,
        call3_failure_reason,
        flags=re.IGNORECASE,
    )
    roster_correction_instruction = ""
    if _call3_dialogue_requires_localized_names(speak_language):
        roster_correction_instruction = (
            "Keep each exact roster identifier as machine-readable speaker metadata only "
            "in the speaker field. Never repeat a roster identifier in the text field. "
            "Infer a natural in-story form of address "
            "from the original narrative and bounded scene windows; if uncertain, omit "
            "the direct address. Preserve the speaker field and required style tag. "
        )
    print(
        f"[ILLUST_CONTEXT:CALL3-CORRECTION] 최초 CALL3 결과가 출력 계약을 위반해 "
        f"실패 슬롯만 교정 호출 1회 실행: repair_slots={repair_slots}, "
        f"preserved_slots={[slot for slot in selected_slots if slot not in set(repair_slots)]}, "
        f"reason={call3_failure_reason}"
    )
    anchor_by_slot = {slot: anchor for anchor, slot in (segment_slots or {}).items()}
    model_repair_slots = [anchor_by_slot.get(slot, slot) for slot in repair_slots]
    if segment_slots:
        model_failure_reason = "Missing or invalid required scene objects or dialogue entries for the listed C anchors."
    retry_messages = deepcopy(speak_messages)
    retry_messages.extend([{
        "role": "assistant",
        "content": variable(initial_output if segment_slots else normalized_initial_output, "교정할 이전 LLM 응답"),
    }, {
        "role": "user",
        "content": (
            "Repair only the failed scene slots listed below. Every other scene object has "
            "already passed validation and is preserved by the server; do not repeat, rewrite, "
            "summarize, or mention it. "
            + "Repair anchors, in order: " + variable(model_repair_slots, "교정할 원문 앵커 목록") + ". "
            + 'Return one JSON object with a scenes array. Emit exactly one scene object '
            + 'for every repair anchor, using its exact value in slot, and no other scenes. '
            + 'Each scene has an entries array; each entry has speaker, type (speech or thought), '
            + 'text, and an optional tag. '
            "A scene that should intentionally remain silent, or has no suitable "
            "visible or narratively present speaker, must have an empty entries array. "
            "Do not invent narration, action description, dialogue, or "
            "thought merely to fill such a scene. Every other scene must contain at "
            "least one dialogue, thought, or inner monologue entry. "
            "Write every dialogue and thought in " + variable(speak_language, "출력 언어") + "; this language rule is mandatory. "
            + roster_correction_instruction
            + "Character names in speaker fields, slot anchors, JSON keys, type values, and required tags are "
            "the only language exceptions. "
            "Validation failure to fix: " + variable(model_failure_reason, "교정할 오류 내용") + ". "
            "An intentionally silent scene uses entries: [], never a placeholder entry. "
            "Output only the repaired JSON object. Apply any active Base64 transport to "
            "JSON string values only; never encode keys or the complete document."
        ),
    }])
    try:
        corrected_output = await _call_pipeline_llm(
            correction_call_name,
            _normalize_messages(retry_messages),
            stream_notify,
            json_mode=True,
            result_validator=lambda result: validate_call3_output_contract(
                normalize_call3_dialogue_output(result, segment_slots),
                repair_slots,
                character_names,
                speak_language,
            ),
        )
        corrected_output = normalize_call3_dialogue_output(corrected_output, segment_slots)
        merged_output = _merge_call3_slot_repairs(
            baseline_output,
            corrected_output,
            selected_slots,
            repair_slots,
        )
        merged_valid, merged_reason = validate_call3_output_contract(
            merged_output,
            selected_slots,
            character_names,
            speak_language,
        )
        if not merged_valid:
            print(
                "[ILLUST_CONTEXT:CALL3-RECOVERY] 슬롯별 교정 병합 후 검증 실패: "
                f"repair_slots={repair_slots}, reason={merged_reason}"
            )
            raise ValueError(merged_reason or "CALL3 슬롯별 교정 병합 실패")
        state["output"] = merged_output
        state["silent_slots"] = [
            slot for slot in selected_slots
            if slot not in parse_speak_output(merged_output)
        ]
        return state
    except Exception as correction_error:
        state["failure_reason"] = str(correction_error)
        print(
            "[ILLUST_CONTEXT:CALL3-RECOVERY] CALL3 교정/라우팅 폴백 소진, "
            "최초 결과에서 안전한 슬롯별 대사만 복구하고 이미지 파이프라인 계속: "
            f"repair_slots={repair_slots}, error={type(correction_error).__name__}: "
            f"{correction_error}"
        )
        traceback.print_exc()
        state.update({
            "output": baseline_output,
            "silent_slots": baseline_recovery["silent_slots"],
        })
        return state


class ParallelPipelineJobsError(RuntimeError):
    """Parallel group failure that retains every independently valid result."""

    def __init__(self, message: str, resolved_values: dict[int, dict], failures: dict[int, str]):
        super().__init__(message)
        self.resolved_values = dict(resolved_values)
        self.failures = dict(failures)


def _detail_batches_for_plan_batches(
    plan_batches: list[list[dict]], max_concurrency: int | None = None,
) -> list[list[dict]]:
    """Balance DETAIL scenes independently, preserving 1:1 when counts match."""
    if not plan_batches:
        print("[ILLUST_CONTEXT:CALL2-DETAIL] 분배할 PLAN 배치 없음")
        return []
    if max_concurrency is None or max_concurrency == len(plan_batches):
        return [list(batch) for batch in plan_batches]
    return _balanced_call2_scene_plan_batches(
        [scene for batch in plan_batches for scene in batch], max_concurrency)


@illustration_flow.collect_branches
async def _run_parallel_pipeline_jobs(
    jobs: list[dict],
    *,
    group_id: str,
    group_label: str,
    max_concurrency: int,
    slow_retry_enabled: bool,
    slow_retry_remaining: int,
    slow_retry_progress_enabled: bool,
    slow_retry_progress_threshold: int,
    slow_retry_tps_enabled: bool,
    slow_retry_tps_threshold: float,
    slow_retry_condition_operator: str,
    stream_notify,
    invoke,
    on_resolved=None,
    fail_fast: bool = False,
    request_semaphore=None,
) -> list[dict]:
    """Run capped LLM races and optionally finalize each accepted winner once."""
    if not jobs:
        print(f"[ILLUST_CONTEXT:{group_id}] 실행할 병렬 작업이 비어 있음")
        return []

    concurrency = max(1, min(16, int(max_concurrency)))
    remaining_limit = max(1, min(16, int(slow_retry_remaining)))
    progress_threshold = max(1, min(99, int(slow_retry_progress_threshold)))
    tps_threshold = max(0.1, min(1000.0, float(slow_retry_tps_threshold)))
    operator = str(slow_retry_condition_operator or "and").strip().lower()
    if operator not in ("and", "or"):
        print(
            f"[ILLUST_CONTEXT:{group_id}_HEDGE] 조건 결합 방식이 무효함: "
            f"value={slow_retry_condition_operator!r}; and 사용"
        )
        operator = "and"
    hedge_active = bool(slow_retry_enabled) and concurrency >= 2 and len(jobs) >= 2
    if slow_retry_enabled and not hedge_active:
        print(
            f"[ILLUST_CONTEXT:{group_id}_HEDGE] 느린 요청 재시도 비활성: "
            f"jobs={len(jobs)}, max_concurrency={concurrency}"
        )

    semaphore = request_semaphore if request_semaphore is not None else asyncio.Semaphore(concurrency)
    states: dict[int, dict] = {}
    for index, job in enumerate(jobs, start=1):
        states[index] = {
            "job": job,
            "tasks": set(),
            "duplicate_started": False,
            "hedge_evaluated": False,
            "failure_reasons": [],
            "attempt_outcomes": {},
            "race_result": None,
            "execution_group_id": uuid.uuid4().hex if hedge_active else "",
            "history_ids": {
                "primary": uuid.uuid4().hex if hedge_active else "",
                "duplicate": "",
            },
            "progress": {
                kind: {
                    "streaming": False,
                    "stream_id": "",
                    "partial_length": 0,
                    "started_at": 0.0,
                }
                for kind in ("primary", "duplicate")
            },
        }

    async def run_attempt(index: int, attempt_kind: str) -> dict:
        state = states[index]
        job = state["job"]
        attempt_progress = state["progress"][attempt_kind]

        def observe_stream(event: dict) -> None:
            event_type = str(event.get("type") or "")
            if event_type == "request_mode":
                attempt_progress["streaming"] = bool(event.get("streaming"))
                attempt_progress["stream_id"] = ""
                attempt_progress["partial_length"] = 0
                return
            stream_id = str(event.get("stream_id") or "")
            if event_type == "stream_open" or (
                stream_id and stream_id != attempt_progress["stream_id"]
            ):
                attempt_progress["stream_id"] = stream_id
                attempt_progress["partial_length"] = 0
            attempt_progress["streaming"] = True
            try:
                if event.get("partial_length") is not None:
                    attempt_progress["partial_length"] = max(
                        0,
                        int(event["partial_length"]),
                    )
                elif event.get("partial_text") is not None:
                    attempt_progress["partial_length"] = len(
                        str(event.get("partial_text") or "")
                    )
            except (TypeError, ValueError) as e:
                print(
                    f"[ILLUST_CONTEXT:{group_id}_HEDGE] 스트림 길이 파싱 실패: "
                    f"job={index}/{len(jobs)}, attempt={attempt_kind}, error={e}"
                )
                traceback.print_exc()

        job_stream_notify = None
        if stream_notify:
            async def job_stream_notify(event: dict):
                payload = dict(event)
                payload["queue_subtask"] = {
                    "group_id": group_id.lower(),
                    "group_label": group_label,
                    "index": index,
                    "total": len(jobs),
                }
                await stream_notify(payload)

        try:
            ready_context = None
            if job.get("ready") is not None:
                # PLAN readiness must not occupy a DETAIL request slot. Keep
                # the accepted PLAN winner's flow context for the actual call.
                job, ready_context = await asyncio.shield(job["ready"])
            async with semaphore:
                attempt_progress["started_at"] = time.monotonic()
                invocation = invoke(
                    job,
                    index,
                    len(jobs),
                    attempt_kind,
                    observe_stream if hedge_active else None,
                    state["history_ids"][attempt_kind],
                    job_stream_notify,
                )
                if ready_context is None:
                    value = await invocation
                else:
                    value = await illustration_flow.join(ready_context.run(
                        illustration_flow.create_task, invocation,
                        flow_label=f"{group_label} {index} · {attempt_kind}",
                    ))
            raw = str(value.get("raw") or "") if isinstance(value, dict) else str(value or "")
            return {
                "ok": True,
                "value": value,
                "raw": raw,
                "output_length": len(raw),
                "attempt_kind": attempt_kind,
                "completed_at": time.monotonic(),
                "resolved_context": contextvars.copy_context() if on_resolved else None,
            }
        except asyncio.CancelledError:
            print(
                f"[ILLUST_CONTEXT:{group_id}_HEDGE] 선착순에서 밀린 요청 취소: "
                f"job={index}/{len(jobs)}, attempt={attempt_kind}"
            )
            raise
        except Exception as e:
            print(
                f"[ILLUST_CONTEXT:{group_id}] 병렬 작업 호출 실패: "
                f"job={index}/{len(jobs)}, attempt={attempt_kind}, error={e}"
            )
            traceback.print_exc()
            return {
                "ok": False,
                "value": None,
                "raw": "",
                "output_length": 0,
                "reason": str(e),
                "error": e,
                "attempt_kind": attempt_kind,
                "completed_at": time.monotonic(),
            }

    pending: set[asyncio.Task] = set()
    finalizing: set[asyncio.Task] = set()
    task_metadata: dict[asyncio.Task, tuple[int, str]] = {}
    resolved: dict[int, dict] = {}
    failed: dict[int, str] = {}

    def start_attempt(index: int, attempt_kind: str) -> None:
        task = illustration_flow.create_task(run_attempt(index, attempt_kind),
            flow_label=None if states[index]["job"].get("ready") is not None
            else f"{group_label} {index} · {attempt_kind}")
        pending.add(task)
        task_metadata[task] = (index, attempt_kind)
        states[index]["tasks"].add(task)

    for job_index in range(1, len(jobs) + 1):
        start_attempt(job_index, "primary")

    try:
        while pending or finalizing:
            done, _waiting = await asyncio.wait(
                pending | finalizing,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Winner callbacks may prepare masks and enqueue images independently
            # while the coordinator keeps observing the remaining LLM requests.
            for task in done & finalizing:
                task.result()
            finalizing.difference_update(done)
            done.intersection_update(pending)
            pending.difference_update(done)
            completed = []
            for task in done:
                index, attempt_kind = task_metadata.pop(task)
                states[index]["tasks"].discard(task)
                try:
                    outcome = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    print(
                        f"[ILLUST_CONTEXT:{group_id}] 병렬 작업 예외: "
                        f"job={index}/{len(jobs)}, attempt={attempt_kind}, error={e}"
                    )
                    traceback.print_exception(type(e), e, e.__traceback__)
                    outcome = {
                        "ok": False,
                        "reason": f"unexpected_error: {e}",
                        "error": e,
                        "attempt_kind": attempt_kind,
                        "completed_at": time.monotonic(),
                    }
                completed.append((float(outcome["completed_at"]), index, outcome))

            for _completed_at, index, outcome in sorted(completed, key=lambda item: (item[0], item[1])):
                state = states[index]
                state["attempt_outcomes"][outcome["attempt_kind"]] = outcome
                if index in resolved:
                    print(
                        f"[ILLUST_CONTEXT:{group_id}_HEDGE] 중복 완료 결과 폐기: "
                        f"job={index}/{len(jobs)}, attempt={outcome['attempt_kind']}"
                    )
                    continue
                if outcome.get("ok"):
                    resolved[index] = outcome
                    if state["duplicate_started"]:
                        winner = outcome["attempt_kind"]
                        loser = "duplicate" if winner == "primary" else "primary"
                        loser_progress = state["progress"][loser]
                        elapsed = max(
                            0.001,
                            time.monotonic() - float(loser_progress.get("started_at") or 0.0),
                        ) if loser_progress.get("started_at") else 0.0
                        state["race_result"] = {
                            "winner": winner,
                            "loser": loser,
                            "loser_progress": round(min(
                                99.0,
                                int(loser_progress.get("partial_length") or 0)
                                / max(1, int(outcome.get("output_length") or 1))
                                * 100.0,
                            ), 1),
                            "loser_streaming": bool(loser_progress.get("streaming")),
                            "loser_elapsed": elapsed,
                        }
                    for sibling in list(state["tasks"]):
                        if not sibling.done():
                            sibling.cancel()
                    if on_resolved:
                        # Keep the winning LLM's flow frontier so masks and images
                        # depend on that response, including when a duplicate wins.
                        callback = on_resolved(index, outcome["value"])
                        finalizing.add(outcome["resolved_context"].run(
                            illustration_flow.create_task, callback,
                        ))
                    continue
                state["failure_reasons"].append(str(outcome.get("reason") or "unknown_failure"))

            for index, state in states.items():
                if index in resolved or index in failed or state["tasks"]:
                    continue
                failed[index] = state["failure_reasons"][-1] if state["failure_reasons"] else "unknown_failure"
                if fail_fast:
                    print(
                        f"[ILLUST_CONTEXT:{group_id}] 병렬 작업 최종 실패로 나머지 취소: "
                        f"job={index}/{len(jobs)}, input={state['job']!r}, reason={failed[index]}"
                    )
                    last_outcome = next(reversed(state["attempt_outcomes"].values()), {})
                    error = last_outcome.get("error")
                    if isinstance(error, Exception):
                        raise error
                    raise RuntimeError(failed[index])

            unresolved = [
                index
                for index in range(1, len(jobs) + 1)
                if index not in resolved and index not in failed
            ]
            if hedge_active and resolved and 0 < len(unresolved) <= remaining_limit:
                resolved_units = [
                    max(1, int(states[index]["job"].get("weight") or 1))
                    for index in resolved
                ]
                resolved_lengths = [
                    max(1, int(resolved[index].get("output_length") or 1))
                    for index in resolved
                ]
                chars_per_unit = sum(resolved_lengths) / max(1, sum(resolved_units))
                for index in unresolved:
                    state = states[index]
                    if state["hedge_evaluated"]:
                        continue
                    primary_progress = state["progress"]["primary"]
                    if not primary_progress.get("started_at"):
                        continue
                    state["hedge_evaluated"] = True
                    expected_length = max(
                        1.0,
                        chars_per_unit * max(1, int(state["job"].get("weight") or 1)),
                    )
                    partial_length = int(primary_progress.get("partial_length") or 0)
                    streaming = bool(primary_progress.get("streaming"))
                    estimated_progress = min(
                        99.0,
                        partial_length / expected_length * 100.0,
                    ) if streaming else 0.0
                    elapsed = max(
                        0.001,
                        time.monotonic() - float(primary_progress["started_at"]),
                    )
                    estimated_tps = (partial_length / 3.0) / elapsed if streaming else 0.0
                    conditions = []
                    if slow_retry_progress_enabled:
                        conditions.append(("progress", estimated_progress < progress_threshold))
                    if slow_retry_tps_enabled:
                        conditions.append(("tps", estimated_tps < tps_threshold))
                    if not conditions:
                        should_duplicate = False
                    elif operator == "or":
                        should_duplicate = any(result for _name, result in conditions)
                    else:
                        should_duplicate = all(result for _name, result in conditions)
                    condition_text = ", ".join(
                        f"{name}={'met' if result else 'not_met'}"
                        for name, result in conditions
                    ) or "none_enabled"
                    if should_duplicate:
                        state["duplicate_started"] = True
                        state["history_ids"]["duplicate"] = uuid.uuid4().hex
                        print(
                            f"[ILLUST_CONTEXT:{group_id}_HEDGE] 느리다고? 다시해! "
                            f"job={index}/{len(jobs)}, remaining={len(unresolved)}, "
                            f"progress={estimated_progress:.1f}%, tps={estimated_tps:.1f}, "
                            f"operator={operator.upper()}, conditions={condition_text}"
                        )
                        start_attempt(index, "duplicate")
                    else:
                        print(
                            f"[ILLUST_CONTEXT:{group_id}_HEDGE] 느린 요청 조건 불충족: "
                            f"job={index}/{len(jobs)}, progress={estimated_progress:.1f}%, "
                            f"tps={estimated_tps:.1f}, conditions={condition_text}"
                        )
    except asyncio.CancelledError:
        print(f"[ILLUST_CONTEXT:{group_id}] 병렬 조정 상위 작업 취소: pending={len(pending)}, finalizing={len(finalizing)}")
        for task in pending | finalizing:
            task.cancel()
        await illustration_flow.gather(*(pending | finalizing), return_exceptions=True)
        raise
    except Exception as e:
        print(f"[ILLUST_CONTEXT:{group_id}] 병렬 조정 예외: {e}")
        traceback.print_exc()
        for task in pending | finalizing:
            task.cancel()
        await illustration_flow.gather(*(pending | finalizing), return_exceptions=True)
        raise

    history_updates = {}
    for index, state in states.items():
        if not state["duplicate_started"]:
            continue
        race = state.get("race_result")
        for attempt_kind, role_label in (("primary", "원본"), ("duplicate", "느리다고? 다시해!")):
            history_id = state["history_ids"].get(attempt_kind, "")
            if not history_id or not state["progress"][attempt_kind].get("started_at"):
                continue
            if race and attempt_kind == race["winner"]:
                history_updates[history_id] = {
                    "call_name": f"{group_label} {index}/{len(jobs)} [{role_label} · 승리]",
                    "status": "race_won",
                    "race_outcome": "winner",
                    "parent_execution_id": state["execution_group_id"],
                }
            elif race and attempt_kind == race["loser"]:
                history_updates[history_id] = {
                    "call_name": f"{group_label} {index}/{len(jobs)} [{role_label} · 패배]",
                    "status": "race_lost",
                    "race_outcome": "loser",
                    "race_progress": float(race["loser_progress"]),
                    "race_streaming": bool(race["loser_streaming"]),
                    "race_elapsed": round(float(race["loser_elapsed"]), 3),
                    "parent_execution_id": state["execution_group_id"],
                }
            else:
                history_updates[history_id] = {
                    "call_name": f"{group_label} {index}/{len(jobs)} [{role_label} · 경주 실패]",
                    "race_outcome": "failed",
                    "parent_execution_id": state["execution_group_id"],
                }
    if history_updates:
        lighbd_service._update_lighbd_history_records(history_updates)

    if failed:
        for index, reason in sorted(failed.items()):
            print(
                f"[ILLUST_CONTEXT:{group_id}] 병렬 작업 최종 실패: "
                f"job={index}/{len(jobs)}, reason={reason}"
            )
        message = (
            f"{group_label} 병렬 작업 {len(failed)}/{len(jobs)}개 실패: "
            + "; ".join(f"{index}={reason}" for index, reason in sorted(failed.items()))
        )
        raise ParallelPipelineJobsError(
            message,
            {
                index: outcome["value"]
                for index, outcome in resolved.items()
            },
            failed,
        )

    print(
        f"[ILLUST_CONTEXT:{group_id}] 병렬 작업 완료: "
        f"jobs={len(jobs)}, max_concurrency={concurrency}"
    )
    return [resolved[index]["value"] for index in range(1, len(jobs) + 1)]


def _merge_call1_shard_values(
    shard_values: list[dict],
    segment_order: list[str],
) -> tuple[dict, list[str], list[str]]:
    """Merge disjoint OUTFIT shard JSON without semantic keyword inference."""
    merged = {
        "wardrobe_at_start": [],
        "wardrobe_events": [],
        "hairstyle_events": [],
    }
    warnings = []
    fallback_errors = []
    wardrobe_seen = set()
    hairstyle_seen = set()
    segment_rank = {segment_id: index for index, segment_id in enumerate(segment_order)}
    selected_baseline = None

    for shard_index, value in enumerate(shard_values, start=1):
        assigned_ids = set(value.get("assigned_segment_ids") or [])
        raw = value.get("value") if isinstance(value.get("value"), dict) else {}

        shard_baseline = raw.get("wardrobe_at_start") or []
        if not isinstance(shard_baseline, list):
            warnings.append(f"OUTFIT shard {shard_index} 시작 복장 형식 오류로 폐기")
        elif shard_baseline:
            normalized_baseline = json.dumps(
                shard_baseline,
                ensure_ascii=False,
                sort_keys=True,
            )
            if selected_baseline is None:
                selected_baseline = normalized_baseline
                merged["wardrobe_at_start"] = deepcopy(shard_baseline)
            elif normalized_baseline != selected_baseline:
                warnings.append(
                    f"OUTFIT shard {shard_index} 시작 복장 판정이 선행 shard와 달라 "
                    "선행 판정을 유지"
                )

        for item in raw.get("wardrobe_events") or []:
            if not isinstance(item, dict):
                warnings.append(f"OUTFIT shard {shard_index} 복장 이벤트 형식 오류로 폐기")
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id and segment_id not in assigned_ids:
                warnings.append(
                    f"OUTFIT shard {shard_index} 담당 밖 복장 이벤트 폐기: segment={segment_id!r}"
                )
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in wardrobe_seen:
                wardrobe_seen.add(key)
                merged["wardrobe_events"].append(deepcopy(item))

        for item in raw.get("hairstyle_events") or []:
            if not isinstance(item, dict):
                warnings.append(f"OUTFIT shard {shard_index} 헤어스타일 이벤트 형식 오류로 폐기")
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id and segment_id not in assigned_ids:
                warnings.append(
                    f"OUTFIT shard {shard_index} 담당 밖 헤어스타일 이벤트 폐기: segment={segment_id!r}"
                )
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in hairstyle_seen:
                hairstyle_seen.add(key)
                merged["hairstyle_events"].append(deepcopy(item))

    merged["wardrobe_events"].sort(
        key=lambda item: segment_rank.get(
            str(item.get("segment_id") or ""),
            len(segment_rank),
        )
    )
    merged["hairstyle_events"].sort(
        key=lambda item: segment_rank.get(
            str(item.get("segment_id") or ""),
            len(segment_rank),
        )
    )
    return merged, warnings, fallback_errors


async def _run_parallel_call1_analysis(
    *,
    call1_system: str,
    segmented_current: str,
    current_segments: dict[str, dict],
    history_text: str,
    toggles: dict,
    stream_notify,
) -> tuple[str, list[str], list[str]]:
    segment_ids = list(current_segments)
    max_concurrency = int(toggles["call1_parallel_max_concurrency"])
    # 작업 수 = 동시 호출 LLM 수. segment를 max_concurrency개 작업에 서사 순서대로
    # 균등 분할한다. segment가 동시 호출 수보다 적으면 그 수만큼만 만든다.
    worker_count = min(max(1, max_concurrency), len(segment_ids))
    base_size, larger_worker_count = divmod(len(segment_ids), worker_count)
    chunk_sizes = [
        base_size + (1 if index < larger_worker_count else 0)
        for index in range(worker_count)
    ]
    chunks = []
    cursor = 0
    for chunk_size in chunk_sizes:
        chunks.append(segment_ids[cursor:cursor + chunk_size])
        cursor += chunk_size
    jobs = [
        {
            "assigned_segment_ids": chunk,
            "weight": len(chunk),
        }
        for chunk in chunks
    ]
    print(
        f"[ILLUST_CONTEXT:OUTFIT_PARALLEL] 분석 작업 준비: "
        f"segments={len(segment_ids)}, max_concurrency={max_concurrency}, "
        f"jobs={len(jobs)}, distribution={chunk_sizes}"
    )

    async def invoke(
        job,
        index,
        total,
        attempt_kind,
        stream_observer,
        history_id,
        job_stream_notify,
    ):
        assigned = list(job["assigned_segment_ids"])
        shard_instruction = (
            "\n\n# Parallel shard contract\n"
            f"This is shard {index}/{total}. Read the full context for discourse understanding, "
            "emit the complete wardrobe_at_start for every resolved CURRENT character, but emit "
            "wardrobe_events and hairstyle_events only "
            f"for these assigned segment IDs: {json.dumps(assigned, ensure_ascii=False)}.\n"
            "Use only the resolved CURRENT canonical names supplied by the system prompt. "
            "Return one JSON object with exactly those three arrays."
        )
        messages = _normalize_messages([
            {"role": "system", "content": call1_system + shard_instruction},
            {
                "role": "user",
                "content": (
                    "# PAST HISTORY\n"
                    + (history_text or "(empty)")
                    + "\n\n# FULL CURRENT CONTEXT SEGMENTS\n"
                    + segmented_current
                    + "\n\n# ASSIGNED SEGMENT IDS\n"
                    + json.dumps(assigned, ensure_ascii=False)
                ),
            },
        ])

        def validate(result):
            raw = _json_object_from_text(result)
            if raw is None:
                return False, "OUTFIT shard JSON object 없음"
            for key in ("wardrobe_at_start", "wardrobe_events", "hairstyle_events"):
                if key not in raw or not isinstance(raw.get(key), list):
                    return False, f"OUTFIT shard {key}가 list가 아님"
            return True, ""

        call_name = f"OUTFIT {index}/{total}"
        if attempt_kind == "duplicate":
            call_name += " [느리다고? 다시해!]"
        raw_output = await _call_pipeline_llm(
            call_name,
            messages,
            job_stream_notify,
            result_validator=validate,
            json_mode=True,
            stream_observer=stream_observer,
            history_id=history_id,
        )
        raw_value = _json_object_from_text(raw_output)
        if raw_value is None:
            raise ValueError(f"OUTFIT shard {index}/{total} JSON 파싱 실패")
        return {
            "raw": raw_output,
            "value": raw_value,
            "assigned_segment_ids": assigned,
        }

    failed_shard_warnings: list[str] = []
    try:
        shard_values = await _run_parallel_pipeline_jobs(
            jobs,
            group_id="OUTFIT_PARALLEL",
            group_label="OUTFIT 병렬 분석",
            max_concurrency=int(toggles["call1_parallel_max_concurrency"]),
            slow_retry_enabled=bool(toggles["call1_parallel_slow_retry_enabled"]),
            slow_retry_remaining=int(toggles["call1_parallel_slow_retry_remaining"]),
            slow_retry_progress_enabled=bool(toggles["call1_parallel_slow_retry_progress_enabled"]),
            slow_retry_progress_threshold=int(toggles["call1_parallel_slow_retry_progress_threshold"]),
            slow_retry_tps_enabled=bool(toggles["call1_parallel_slow_retry_tps_enabled"]),
            slow_retry_tps_threshold=float(toggles["call1_parallel_slow_retry_tps_threshold"]),
            slow_retry_condition_operator=str(toggles["call1_parallel_slow_retry_condition_operator"]),
            stream_notify=stream_notify,
            invoke=invoke,
        )
    except ParallelPipelineJobsError as group_error:
        # 각 shard는 공통 전체 문맥을 읽지만 담당 segment의 사건만 출력한다.
        # 라우팅 재시도까지 소진된 shard 때문에 이미 검증된 다른 shard를 버리고
        # 전체 OUTFIT을 다시 호출하지 않는다. 실패 shard의 segment 사건만 누락시키고
        # 성공 shard를 시간순으로 병합해 후속 CALL2에 전달한다.
        shard_values = [
            group_error.resolved_values[index]
            for index in sorted(group_error.resolved_values)
        ]
        for failed_index, reason in sorted(group_error.failures.items()):
            assigned = list(jobs[failed_index - 1].get("assigned_segment_ids") or [])
            warning = (
                f"OUTFIT shard {failed_index}/{len(jobs)} 최종 실패로 담당 사건만 누락: "
                f"segments={assigned}, reason={reason}"
            )
            failed_shard_warnings.append(warning)
            print(f"[ILLUST_CONTEXT:OUTFIT_PARALLEL] {warning}")
        if not shard_values:
            print(
                "[ILLUST_CONTEXT:OUTFIT_PARALLEL] 모든 shard가 실패해 빈 사건 목록으로 "
                f"계속 진행: failures={group_error.failures}"
            )
    merged, merge_warnings, merge_fallback_errors = _merge_call1_shard_values(
        shard_values,
        segment_ids,
    )
    merge_warnings.extend(failed_shard_warnings)
    compact = {
        "wardrobe_at_start": list(merged.get("wardrobe_at_start") or []),
        "wardrobe_events": list(merged.get("wardrobe_events") or []),
        "hairstyle_events": list(merged.get("hairstyle_events") or []),
    }
    return json.dumps(compact, ensure_ascii=False), merge_warnings, merge_fallback_errors


async def _run_resolution_stage(
    *,
    profile_system: str,
    segmented_current: str,
    current_context: str,
    current_segments: dict[str, dict],
    candidate_names: list[str],
    previous_state: dict,
    visual_profiles: dict[str, dict] | None,
    profile_inference_enabled: bool,
    stream_notify,
    history_ids_sink: list[str] | None = None,
) -> tuple[str, dict]:
    """Run one separated CHARACTER or PROFILE resolution stage.

    ``profile_inference_enabled=False`` is the always-on CURRENT-character stage.
    ``True`` is the optional profile-only stage and receives only characters already
    confirmed by the first stage.
    """
    stage_call_name = (
        "PROFILE-RESOLVE" if profile_inference_enabled else "CHARACTER-RESOLVE"
    )
    current_names = [str(name or "").strip() for name in candidate_names or []]
    current_names = list(dict.fromkeys(name for name in current_names if name))
    selected_profiles = _selected_visual_profiles(
        visual_profiles,
        current_names,
        multiple_only=False,
    )
    if not selected_profiles:
        print(
            f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] "
            "등록 캐릭터가 없어 단계 호출 건너뜀: "
            f"candidates={current_names}"
        )
        return "", _empty_profile_result()

    selected_names = list(selected_profiles)
    persona_context = _persona_identity_context(
        selected_profiles,
        selected_names,
    )
    card_identity_context = _character_card_identity_context(
        selected_profiles,
        selected_names,
    )
    repair_call_name = (
        "PROFILE-RESOLVE-REPAIR"
        if profile_inference_enabled
        else "CHARACTER-RESOLVE-REPAIR"
    )
    catalog = (
        build_natural_profile_catalog(selected_profiles)
        if profile_inference_enabled
        else ""
    )
    tracked_state = (
        _profile_state_for_prompt(
            previous_state,
            selected_names,
            selected_profiles,
        )
        if profile_inference_enabled
        else {}
    )
    messages = _normalize_messages(prompt_builders.resolution_messages(
        profile_inference_enabled, selected_names, tracked_state, catalog,
        segmented_current, card_identity_context, persona_context, profile_system))

    def validate_shape(result):
        raw = _json_object_from_text(result)
        if raw is None:
            return False, f"{stage_call_name} JSON object 없음"
        if not isinstance(raw.get("characters"), list):
            return False, f"{stage_call_name} characters가 list가 아님"
        if not isinstance(raw.get("uncertainties"), list):
            return False, f"{stage_call_name} uncertainties가 list가 아님"
        return True, ""

    def validate_canonical_names(names):
        def validate(result):
            valid, reason = validate_shape(result)
            if not valid:
                return valid, reason
            errors = _profile_resolution_name_contract_errors(result, names)
            if errors:
                return False, f"{stage_call_name} 등록명 계약 위반: " + "; ".join(errors)
            return True, ""

        return validate

    raw_output = await _call_pipeline_llm(
        stage_call_name,
        messages,
        stream_notify,
        result_validator=validate_shape,
        json_mode=True,
        history_ids_sink=history_ids_sink,
    )
    name_contract_issues = _profile_resolution_name_contract_issues(
        raw_output,
        selected_names,
    )
    name_repair_warnings: list[str] = []
    if name_contract_issues:
        print(
            f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] "
            "등록명 계약 위반 감지, "
            f"오류 항목만 교정 호출: issues={name_contract_issues}"
        )
        primary_raw = _json_object_from_text(raw_output) or {}
        rejected_indices = {
            int(issue.get("source_index") or 0)
            for issue in name_contract_issues
            if int(issue.get("source_index") or 0) > 0
        }
        accepted_names = [
            str(item.get("name") or "").strip()
            for index, item in enumerate(
                primary_raw.get("characters") or [],
                start=1,
            )
            if index not in rejected_indices
            and isinstance(item, dict)
            and str(item.get("name") or "").strip()
        ]
        accepted_name_set = set(accepted_names)
        available_names = [
            name for name in selected_names
            if name not in accepted_name_set
        ]
        available_name_set = set(available_names)
        primary_characters = list(primary_raw.get("characters") or [])
        repairable_name_issues = [
            issue for issue in name_contract_issues
            if int(issue.get("source_index") or 0) in rejected_indices
            and isinstance(issue.get("item"), dict)
        ]
        repairable_indices = {
            int(issue.get("source_index") or 0)
            for issue in repairable_name_issues
        }

        def inspect_name_item_repairs(result):
            raw = _json_object_from_text(result)
            if raw is None:
                return (
                    False,
                    f"{repair_call_name} 이름 교정 JSON object 없음",
                    {},
                    {},
                    [],
                )
            repairs = raw.get("repairs")
            if not isinstance(repairs, list):
                return (
                    False,
                    f"{repair_call_name} 이름 교정 repairs가 list가 아님",
                    {},
                    raw,
                    [],
                )
            if not isinstance(raw.get("uncertainties"), list):
                return (
                    False,
                    f"{repair_call_name} 이름 교정 uncertainties가 list가 아님",
                    {},
                    raw,
                    [],
                )
            replacements = {}
            used_names = set(accepted_names)
            invalid_items = []
            for repair_index, repair in enumerate(repairs, start=1):
                source_index = None
                name = ""
                reason = ""
                if not isinstance(repair, dict):
                    reason = f"repairs[{repair_index}]가 object가 아님"
                else:
                    source_index = repair.get("source_index")
                    name = str(repair.get("name") or "").strip()
                    if not isinstance(source_index, int) or isinstance(source_index, bool):
                        reason = f"source_index가 정수가 아님: {source_index!r}"
                    elif source_index not in repairable_indices:
                        reason = f"교정 가능한 거부 항목이 아님: {source_index!r}"
                    elif source_index in replacements:
                        reason = f"source_index 중복: {source_index}"
                    elif name in accepted_name_set:
                        reason = f"이미 승인된 캐릭터 중복: {name!r}"
                    elif name not in available_name_set:
                        reason = f"사용 가능한 정확한 등록명이 아님: {name!r}"
                    elif name in used_names:
                        reason = f"교정 결과 캐릭터 중복: {name!r}"
                    else:
                        original = primary_characters[source_index - 1]
                        replacement = deepcopy(original)
                        replacement["name"] = name
                        replacements[source_index] = replacement
                        used_names.add(name)
                if reason:
                    invalid_items.append(
                        f"repair_index={repair_index}, source_index={source_index!r}, "
                        f"name={name!r}, reason={reason}"
                    )
            if repairs and not replacements:
                return (
                    False,
                    f"{repair_call_name} 이름 교정 유효 항목 없음: "
                    + "; ".join(invalid_items),
                    {},
                    raw,
                    invalid_items,
                )
            return True, "", replacements, raw, invalid_items

        def validate_name_item_repairs(result):
            valid, reason, _replacements, _raw, _invalid = (
                inspect_name_item_repairs(result)
            )
            return valid, reason

        repair_context = "\n\n".join(x for x in (
            persona_context,
            card_identity_context if not profile_inference_enabled else "",
            "# FULL CURRENT CONTEXT SEGMENTS\n" + segmented_current,
        ) if x)
        repair_targets = [
            {
                "source_index": int(issue.get("source_index") or 0),
                "invalid_name": str(
                    (issue.get("item") or {}).get("name") or ""
                ).strip(),
            }
            for issue in repairable_name_issues
        ]
        repair_messages = _normalize_messages([
            {
                "role": "system",
                "content": (
                    "# REJECTED-ITEM CANONICAL NAME REPAIR\n"
                    "Resolve only the canonical name of each supplied rejected character "
                    "item from the natural-language discourse. Accepted names are "
                    "authoritative and unavailable. A repair name must be copied verbatim "
                    "from REMAINING AVAILABLE REGISTERED NAMES, and each available name may "
                    "be used at most once. Return only this JSON shape:\n"
                    '{"repairs":[{"source_index":1,"name":"exact canonical name"}],'
                    '"uncertainties":[]}\n'
                    "Do not return character objects, in_history, profile data, appearance, "
                    "or outfit decisions; the server preserves every non-name field from "
                    "the rejected source item. Keep source_index unchanged. Repair an item "
                    "only when the full discourse establishes its exact registered identity. "
                    "The rejected item's existing name is known-invalid evidence, not a "
                    "permitted output value; never copy or preserve it unless that exact "
                    "spelling independently appears in REMAINING AVAILABLE REGISTERED NAMES. "
                    "Never choose a roster character merely from role, appearance, outfit, "
                    "profile, or name resemblance. If the item is an unregistered distinct "
                    "person, refers to an already accepted identity, or remains uncertain, "
                    "omit that source_index from repairs and explain the omission briefly in "
                    "uncertainties."
                ),
            },
            {
                "role": "user",
                "content": (
                    "# REJECTED CHARACTER ITEMS ONLY\n"
                    + variable(json.dumps(repairable_name_issues, ensure_ascii=False, indent=2), "교정할 캐릭터 항목")
                    + "\n\n"
                    + variable(repair_context, "이름 교정용 문맥")
                    + "\n\n# FINAL CANONICAL NAME REPAIR AUTHORITY\n"
                    "The invalid_name values below came from the rejected response. They "
                    "identify what must be corrected and must not be copied into repairs.\n"
                    "# REPAIR TARGETS\n"
                    + variable(json.dumps(repair_targets, ensure_ascii=False, indent=2), "교정 대상 이름")
                    + "\n\n# ALREADY ACCEPTED CHARACTER NAMES (UNAVAILABLE)\n"
                    + variable(json.dumps(accepted_names, ensure_ascii=False), "이미 확정된 이름")
                    + "\n\n# REMAINING AVAILABLE REGISTERED NAMES (ONLY VALID name VALUES)\n"
                    + variable(json.dumps(available_names, ensure_ascii=False), "남은 등록 이름")
                    + "\nCopy every repairs[].name character-for-character from the "
                    "REMAINING AVAILABLE REGISTERED NAMES list immediately above. Never "
                    "translate, transliterate, localize, or reuse an invalid_name. Omit a "
                    "target when the discourse does not establish which exact remaining "
                    "registered name it is. Return only:\n"
                    '{"repairs":[{"source_index":1,"name":"exact remaining registered name"}],'
                    '"uncertainties":[]}'
                ),
            },
        ])
        replacements = {}
        repair_uncertainties = []
        if not repairable_name_issues:
            warning = (
                "이름 교정 가능한 object 항목이 없어 거부 항목 폐기: "
                f"source_indices={sorted(rejected_indices)}"
            )
            print(f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] {warning}")
            name_repair_warnings.append(warning)
        elif not available_names:
            warning = (
                "이름 교정에 사용할 미배정 등록명이 없어 거부 항목 폐기: "
                f"source_indices={sorted(rejected_indices)}, "
                f"accepted_names={accepted_names}"
            )
            print(f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] {warning}")
            name_repair_warnings.append(warning)
        else:
            try:
                repair_output = await _call_pipeline_llm(
                    repair_call_name,
                    repair_messages,
                    stream_notify,
                    result_validator=validate_name_item_repairs,
                    json_mode=True,
                    history_ids_sink=history_ids_sink,
                )
                (
                    repair_valid,
                    repair_reason,
                    replacements,
                    repaired_raw,
                    invalid_repair_items,
                ) = inspect_name_item_repairs(repair_output)
                if not repair_valid:
                    raise ValueError(repair_reason)
                repair_uncertainties = list(
                    repaired_raw.get("uncertainties") or []
                )
                for invalid_item in invalid_repair_items:
                    warning = f"이름 교정 응답 항목 폐기: {invalid_item}"
                    print(f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] {warning}")
                    name_repair_warnings.append(warning)
            except Exception as e:
                warning = (
                    "오류 항목 이름 교정 호출 실패로 거부 항목만 폐기하고 계속 진행: "
                    f"source_indices={sorted(rejected_indices)}, error={e}"
                )
                print(f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] {warning}")
                traceback.print_exc()
                name_repair_warnings.append(warning)
                replacements = {}
                repair_uncertainties = []

        discarded_indices = rejected_indices - set(replacements)
        merged_characters = []
        for index, item in enumerate(primary_characters, start=1):
            if index in rejected_indices:
                replacement = replacements.get(index)
                if replacement is not None:
                    merged_characters.append(replacement)
                continue
            merged_characters.append(deepcopy(item))
        raw_output = json.dumps({
            "characters": merged_characters,
            "uncertainties": [
                *list(primary_raw.get("uncertainties") or []),
                *repair_uncertainties,
            ],
        }, ensure_ascii=False)
        remaining_name_errors = _profile_resolution_name_contract_errors(
            raw_output,
            selected_names,
        )
        if remaining_name_errors:
            print(
                f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] "
                "안전 병합 후에도 등록명 계약 위반: "
                f"errors={remaining_name_errors}, output={raw_output[:1000]!r}"
            )
            raise ValueError(
                f"{stage_call_name} 안전 병합 내부 오류: "
                + "; ".join(remaining_name_errors)
            )
        if discarded_indices:
            warning = (
                "미해결 이름 항목 폐기: "
                f"source_indices={sorted(discarded_indices)}"
            )
            print(f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] {warning}")
            name_repair_warnings.append(warning)
        print(
            f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] "
            "오류 항목 이름 교정 병합 완료: "
            f"repaired_indices={sorted(replacements)}, "
            f"discarded_indices={sorted(discarded_indices)}"
        )

    normalized_raw = _json_object_from_text(raw_output) or {}
    normalized_uncertainties = list(normalized_raw.get("uncertainties") or [])
    if profile_inference_enabled:
        selected_name_set = set(selected_names)
        returned_by_name = {
            str(item.get("name") or "").strip(): item
            for item in normalized_raw.get("characters") or []
            if isinstance(item, dict)
            and str(item.get("name") or "").strip() in selected_name_set
        }
        normalized_characters = []
        for name in selected_names:
            returned = returned_by_name.get(name) or {}
            normalized_characters.append({
                "name": name,
                "in_history": False,
                "profile_timeline": deepcopy(
                    returned.get("profile_timeline")
                    if "profile_timeline" in returned
                    else []
                ),
            })
    else:
        normalized_characters = [
            {
                "name": str(item.get("name") or "").strip(),
                "profile_timeline": [],
            }
            for item in normalized_raw.get("characters") or []
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
    parse_output = json.dumps({
        "characters": normalized_characters,
        "uncertainties": normalized_uncertainties,
    }, ensure_ascii=False)

    parsed = parse_profile_resolution(
        parse_output,
        current_context,
        current_segments,
        [{"name": name, "confidence": 1.0} for name in selected_names],
        selected_profiles,
        profile_inference_enabled=profile_inference_enabled,
    )
    if parsed is None:
        raise ValueError(f"{stage_call_name} 최종 응답 파싱 실패")
    if name_repair_warnings:
        parsed["validation_warnings"] = [
            *list(parsed.get("validation_warnings") or []),
            *name_repair_warnings,
        ]
        parsed["validation_errors"] = list(parsed["validation_warnings"])
    repair_requests = list(parsed.get("repair_requests") or [])
    if profile_inference_enabled and repair_requests:
        repair_names = [
            str(item.get("character") or "").strip()
            for item in repair_requests
            if isinstance(item, dict) and str(item.get("character") or "").strip()
        ]
        repair_names = list(dict.fromkeys(repair_names))
        repair_profiles = _selected_visual_profiles(
            selected_profiles,
            repair_names,
            multiple_only=True,
        )
        if repair_profiles:
            repair_messages = _normalize_messages([
                {
                    "role": "system",
                    "content": (
                        str(profile_system or "").strip()
                        + "\n\n# PARTIAL REPAIR CONTRACT\n"
                        "Correct only the supplied rejected characters. Return the same "
                        "characters/uncertainties schema and include exactly those characters. "
                           "Preserve every valid narrative decision, but use only bracketed "
                           "profile_ref values supplied for that character. Do not output "
                           "unrelated characters."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "# REJECTED CHARACTER PROFILE ITEMS\n"
                        + variable(json.dumps(repair_requests, ensure_ascii=False, indent=2), "교정할 프로필 항목")
                        + "\n\n# PREVIOUSLY TRACKED PROFILE STATE\n"
                        + variable(json.dumps(tracked_state, ensure_ascii=False, indent=2), "이전 프로필 추적 상태")
                        + ("\n\n" + variable(persona_context, "등록 사용자 페르소나") if persona_context else "")
                        + "\n\n# REGISTERED PROFILE CATALOG FOR REJECTED CHARACTERS\n"
                        + variable(build_natural_profile_catalog(repair_profiles), "교정 대상 등록 프로필")
                        + "\n\n# FULL CURRENT CONTEXT SEGMENTS\n"
                        + variable(segmented_current, "현재 원문과 구간")
                    ),
                },
            ])
            try:
                repair_output = await _call_pipeline_llm(
                    "PROFILE-RESOLVE-REPAIR",
                    repair_messages,
                    stream_notify,
                    result_validator=validate_canonical_names(repair_names),
                    json_mode=True,
                    history_ids_sink=history_ids_sink,
                )
                remaining_name_errors = _profile_resolution_name_contract_errors(
                    repair_output,
                    repair_names,
                )
                if remaining_name_errors:
                    raise ValueError(
                        "PROFILE-RESOLVE 부분 교정 등록명 위반: "
                        + "; ".join(remaining_name_errors)
                    )
                repaired = parse_profile_resolution(
                    repair_output,
                    current_context,
                    current_segments,
                    [{"name": name, "confidence": 1.0} for name in repair_names],
                    repair_profiles,
                    profile_inference_enabled=True,
                )
                if repaired is None:
                    print(
                        "[ILLUST_CONTEXT:PROFILE_RESOLVE] 부분 교정 응답 파싱 실패: "
                        f"characters={repair_names}"
                    )
                else:
                    still_invalid = {
                        str(item.get("character") or "").strip().casefold()
                        for item in repaired.get("repair_requests") or []
                        if isinstance(item, dict)
                    }
                    repaired_by_name = {
                        str(item.get("name") or "").strip().casefold(): item
                        for item in repaired.get("characters") or []
                        if isinstance(item, dict) and str(item.get("name") or "").strip()
                    }
                    merged_characters = deepcopy(parsed.get("characters") or [])
                    repaired_names = []
                    for item in merged_characters:
                        name = str(item.get("name") or "").strip()
                        folded = name.casefold()
                        replacement = repaired_by_name.get(folded)
                        if replacement is None or folded in still_invalid:
                            continue
                        item["profile_timeline"] = deepcopy(
                            replacement.get("profile_timeline") or []
                        )
                        repaired_names.append(name)
                    remaining_requests = [
                        item for item in repair_requests
                        if str(item.get("character") or "").strip().casefold()
                        not in {name.casefold() for name in repaired_names}
                    ]
                    repair_warnings = list(parsed.get("validation_warnings") or [])
                    repair_warnings.extend(repaired.get("validation_warnings") or [])
                    if repaired_names:
                        message = f"부분 교정 반영 완료: characters={repaired_names}"
                        repair_warnings.append(message)
                        print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE] {message}")
                    parsed = _materialize_profile_character_records(
                        merged_characters,
                        selected_profiles,
                        uncertainties=[
                            *list(parsed.get("uncertainties") or []),
                            *list(repaired.get("uncertainties") or []),
                        ],
                        warnings=repair_warnings,
                        repair_requests=remaining_requests,
                    )
            except Exception as e:
                print(
                    "[ILLUST_CONTEXT:PROFILE_RESOLVE] 부분 교정 호출 실패, "
                    f"안전 폴백 진행: characters={repair_names}, error={e}"
                )
                traceback.print_exc()

    if profile_inference_enabled:
        parsed = _apply_profile_resolution_fallbacks(
            parsed,
            selected_profiles,
            previous_state,
            default_only=False,
        )
    else:
        raw_output = json.dumps({
            "characters": [
                {"name": str(item.get("name") or "").strip()}
                for item in parsed.get("characters") or []
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ],
            "uncertainties": list(parsed.get("uncertainties") or []),
        }, ensure_ascii=False)
    current_result_names = [
        str(item.get("name") or "").strip()
        for item in parsed.get("characters") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    print(
        f"[ILLUST_CONTEXT:{stage_call_name.replace('-', '_')}] 단계 완료: "
        f"characters={current_result_names}, events={len(parsed['profile_events'])}, "
        f"profile_stage={profile_inference_enabled}, "
        f"unresolved_repairs={len(parsed.get('repair_requests') or [])}"
    )
    return raw_output, parsed


async def _run_character_resolution(
    *,
    character_system: str,
    segmented_current: str,
    current_context: str,
    current_segments: dict[str, dict],
    candidate_names: list[str],
    visual_profiles: dict[str, dict] | None,
    stream_notify,
    history_ids_sink: list[str] | None = None,
) -> tuple[str, dict]:
    """Always-on CURRENT participation observation without PAST narrative."""
    return await _run_resolution_stage(
        profile_system=character_system,
        segmented_current=segmented_current,
        current_context=current_context,
        current_segments=current_segments,
        candidate_names=candidate_names,
        previous_state={},
        visual_profiles=visual_profiles,
        profile_inference_enabled=False,
        stream_notify=stream_notify,
        history_ids_sink=history_ids_sink,
    )


async def _run_profile_resolution(
    *,
    profile_system: str,
    segmented_current: str,
    current_context: str,
    current_segments: dict[str, dict],
    candidate_names: list[str],
    previous_state: dict,
    visual_profiles: dict[str, dict] | None,
    stream_notify,
    history_ids_sink: list[str] | None = None,
) -> tuple[str, dict]:
    """Resolve each CURRENT multi-profile character in an independent parallel job."""
    selected_profiles = _selected_visual_profiles(
        visual_profiles,
        candidate_names,
        multiple_only=True,
    )
    if not selected_profiles:
        print(
            "[ILLUST_CONTEXT:PROFILE_RESOLVE_PARALLEL] "
            "판정할 CURRENT 다중 프로필 캐릭터가 없어 호출 건너뜀"
        )
        return "", _empty_profile_result()

    jobs = [
        {
            "name": name,
            "profiles": {name: deepcopy(character_profiles)},
            "weight": max(1, len(current_segments or {})),
        }
        for name, character_profiles in selected_profiles.items()
    ]

    async def invoke(
        job,
        _index,
        _total,
        _attempt_kind,
        _stream_observer,
        _history_id,
        job_stream_notify,
    ):
        name = str(job.get("name") or "").strip()
        raw, parsed = await _run_resolution_stage(
            profile_system=profile_system,
            segmented_current=segmented_current,
            current_context=current_context,
            current_segments=current_segments,
            candidate_names=[name],
            previous_state=previous_state,
            visual_profiles=job.get("profiles") or {},
            profile_inference_enabled=True,
            stream_notify=job_stream_notify,
            history_ids_sink=history_ids_sink,
        )
        return {
            "name": name,
            "raw": raw,
            "parsed": parsed,
        }

    values_by_index: dict[int, dict] = {}
    failed_by_index: dict[int, str] = {}
    try:
        values = await _run_parallel_pipeline_jobs(
            jobs,
            group_id="PROFILE_RESOLVE_PARALLEL",
            group_label="PROFILE-RESOLVE 캐릭터별 병렬 판정",
            # 실제 API 동시성은 통합 LLM 큐의 슬롯별 게이트가 최종 제한한다.
            # 여기서는 모든 캐릭터 작업을 즉시 큐에 올려 불필요한 직렬화를 만들지 않는다.
            max_concurrency=len(jobs),
            slow_retry_enabled=False,
            slow_retry_remaining=1,
            slow_retry_progress_enabled=False,
            slow_retry_progress_threshold=50,
            slow_retry_tps_enabled=False,
            slow_retry_tps_threshold=1.0,
            slow_retry_condition_operator="and",
            stream_notify=stream_notify,
            invoke=invoke,
        )
        values_by_index = {
            index: value for index, value in enumerate(values, start=1)
        }
    except ParallelPipelineJobsError as group_error:
        values_by_index = dict(group_error.resolved_values)
        failed_by_index = dict(group_error.failures)

    for index, reason in sorted(failed_by_index.items()):
        job = jobs[index - 1]
        name = str(job.get("name") or "").strip()
        warning = (
            f"캐릭터별 프로필 판정 최종 실패로 이전 추적 상태 또는 등록 기본 "
            f"프로필 폴백 적용: character={name}, reason={reason}"
        )
        print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE_PARALLEL] {warning}")
        empty_character_result = _materialize_profile_character_records(
            [{
                "name": name,
                "in_history": False,
                "profile_timeline": [],
            }],
            job.get("profiles") or {},
            warnings=[warning],
        )
        fallback = _apply_profile_resolution_fallbacks(
            empty_character_result,
            job.get("profiles") or {},
            previous_state,
            default_only=False,
        )
        values_by_index[index] = {
            "name": name,
            "raw": "",
            "parsed": fallback,
        }

    merged_characters: list[dict] = []
    merged_uncertainties: list[str] = []
    merged_warnings: list[str] = []
    merged_repair_requests: list[dict] = []
    for index, job in enumerate(jobs, start=1):
        name = str(job.get("name") or "").strip()
        value = values_by_index.get(index) or {}
        parsed = value.get("parsed") or _empty_profile_result()
        character = next((
            deepcopy(item)
            for item in parsed.get("characters") or []
            if isinstance(item, dict)
            and str(item.get("name") or "").strip().casefold() == name.casefold()
        ), None)
        if character is None:
            warning = (
                f"캐릭터별 프로필 결과에 대상 항목이 없어 빈 타임라인 병합: "
                f"character={name}"
            )
            print(f"[ILLUST_CONTEXT:PROFILE_RESOLVE_PARALLEL] {warning}")
            character = {
                "name": name,
                "in_history": False,
                "profile_timeline": [],
            }
            merged_warnings.append(warning)
        merged_characters.append(character)
        merged_uncertainties.extend(parsed.get("uncertainties") or [])
        merged_warnings.extend(parsed.get("validation_warnings") or [])
        merged_repair_requests.extend(parsed.get("repair_requests") or [])

    merged = _materialize_profile_character_records(
        merged_characters,
        selected_profiles,
        uncertainties=merged_uncertainties,
        warnings=merged_warnings,
        repair_requests=merged_repair_requests,
    )

    raw_characters = []
    for character in merged.get("characters") or []:
        name = str(character.get("name") or "").strip()
        character_profiles = _authority_values_for_name(selected_profiles, name)
        registered_profiles = (
            list(character_profiles.get("profiles") or [])
            if isinstance(character_profiles, dict)
            else []
        )
        raw_timeline = []
        for event in character.get("profile_timeline") or []:
            profile_id = str(event.get("profile_id") or "").strip()
            profile_number = next((
                profile_index
                for profile_index, profile in enumerate(registered_profiles, start=1)
                if str(profile.get("id") or "").strip() == profile_id
            ), 0)
            if not profile_number:
                print(
                    "[ILLUST_CONTEXT:PROFILE_RESOLVE_PARALLEL] "
                    f"병합 출력용 profile_ref 역매핑 실패: "
                    f"character={name}, profile_id={profile_id!r}"
                )
                continue
            raw_event = {
                "at": str(event.get("at") or "").strip().upper(),
                "profile_ref": f"[{profile_number}]",
                "reason": str(event.get("reason") or "").strip(),
            }
            raw_timeline.append(raw_event)
        raw_characters.append({
            "name": name,
            "profile_timeline": raw_timeline,
        })
    raw_output = json.dumps({
        "characters": raw_characters,
        "uncertainties": list(merged.get("uncertainties") or []),
    }, ensure_ascii=False)
    print(
        "[ILLUST_CONTEXT:PROFILE_RESOLVE_PARALLEL] 캐릭터별 병렬 판정 병합 완료: "
        f"characters={[item.get('name') for item in raw_characters]}, "
        f"failed={sorted(failed_by_index)}"
    )
    return raw_output, merged


def _merge_character_and_profile_stage_results(
    character_result: dict,
    profile_result: dict,
    visual_profiles: dict[str, dict] | None,
) -> dict:
    """Overlay profile-only timelines onto authoritative CURRENT characters."""
    profile_by_name = {
        str(item.get("name") or "").strip().casefold(): item
        for item in profile_result.get("characters") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    merged_characters = []
    for character in character_result.get("characters") or []:
        if not isinstance(character, dict):
            continue
        name = str(character.get("name") or "").strip()
        if not name:
            continue
        profile_character = profile_by_name.get(name.casefold()) or {}
        merged_characters.append({
            "name": name,
            "in_history": bool(character.get("in_history")),
            "profile_timeline": deepcopy(
                profile_character.get("profile_timeline") or []
            ),
        })
    return _materialize_profile_character_records(
        merged_characters,
        visual_profiles,
        uncertainties=[
            *list(character_result.get("uncertainties") or []),
            *list(profile_result.get("uncertainties") or []),
        ],
        warnings=[
            *list(character_result.get("validation_warnings") or []),
            *list(profile_result.get("validation_warnings") or []),
        ],
        repair_requests=list(profile_result.get("repair_requests") or []),
    )


def _parse_profile_context_translations(
    result,
    expected_refs: list[str],
) -> dict[str, str]:
    raw = _json_object_from_text(result)
    if raw is None:
        raise ValueError("프로필 영어 번역 응답에 JSON object가 없습니다.")
    translations = raw.get("translations")
    if not isinstance(translations, list):
        raise ValueError("프로필 영어 번역 응답의 translations가 list가 아닙니다.")
    parsed: dict[str, str] = {}
    for index, item in enumerate(translations):
        if not isinstance(item, dict):
            raise ValueError(
                "프로필 영어 번역 항목이 object가 아닙니다: "
                f"index={index}, value={item!r}"
            )
        ref = str(item.get("ref") or "").strip()
        english = str(item.get("english") or "").strip()
        if not ref or not english:
            raise ValueError(
                "프로필 영어 번역 항목의 ref 또는 english가 비어 있습니다: "
                f"index={index}, value={item!r}"
            )
        if ref in parsed:
            raise ValueError(f"프로필 영어 번역 ref가 중복되었습니다: {ref!r}")
        parsed[ref] = english
    expected = set(expected_refs)
    actual = set(parsed)
    if actual != expected:
        raise ValueError(
            "프로필 영어 번역 ref 집합이 요청과 다릅니다: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return parsed


def plan_profile_context_translations(
    visual_profiles: dict[str, dict],
    *,
    cache_namespace: str,
    cache: dict,
) -> tuple[dict, dict, list, dict]:
    """Read-only cache preparation shared by generation and flow preview."""
    translated_profiles = deepcopy(visual_profiles or {})
    namespace = str(cache_namespace or "").strip()
    summary = {
        "cache_namespace": namespace,
        "profile_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "empty_fields": 0,
        "translated_fields": 0,
        "fallback_fields": 0,
        "cache_saved": False,
    }
    if not translated_profiles:
        print(
            "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 번역할 프로필이 없어 생략: "
            f"namespace={namespace!r}, profiles={translated_profiles!r}"
        )
        return translated_profiles, summary, [], {}
    if not namespace:
        print(
            "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 캐시 namespace가 없어 번역 생략: "
            f"characters={list(translated_profiles)}, state={summary!r}"
        )
        return translated_profiles, summary, [], {}

    misses: list[dict] = []
    targets_by_ref: dict[str, tuple[dict, str, str, str, str]] = {}
    for character_name, character in translated_profiles.items():
        profiles = character.get("profiles") or []
        for profile in profiles:
            if not isinstance(profile, dict):
                print(
                    "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] object가 아닌 프로필 생략: "
                    f"namespace={namespace!r}, character={character_name!r}, "
                    f"profile={profile!r}"
                )
                continue
            profile_id = str(profile.get("id") or "").strip()
            if not profile_id:
                print(
                    "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] ID 없는 프로필 생략: "
                    f"namespace={namespace!r}, character={character_name!r}, "
                    f"profile={profile!r}"
                )
                continue
            summary["profile_count"] += 1
            for field in visual_profile_translation_cache.TRANSLATABLE_FIELDS:
                source = str(profile.get(field) or "").strip()
                cached = visual_profile_translation_cache.cached_translation(
                    cache,
                    bot_name=namespace,
                    character_name=character_name,
                    profile_id=profile_id,
                    field=field,
                    source_text=source,
                )
                if cached == "":
                    summary["empty_fields"] += 1
                    continue
                english_field = f"{field}_english"
                if cached is not None:
                    profile[english_field] = cached
                    summary["cache_hits"] += 1
                    continue
                ref = f"text_{len(misses) + 1}"
                purpose = (
                    "profile selection conditions"
                    if field == "selection_guide"
                    else "representative-image appearance reference"
                )
                misses.append({
                    "ref": ref,
                    "purpose": purpose,
                    "source": source,
                })
                targets_by_ref[ref] = (
                    profile,
                    str(character_name),
                    profile_id,
                    field,
                    source,
                )
                summary["cache_misses"] += 1

    return translated_profiles, summary, misses, targets_by_ref


async def prepare_profile_context_translations(
    visual_profiles: dict[str, dict],
    *,
    cache_namespace: str,
    stream_notify=None,
    history_ids_sink: list[str] | None = None,
) -> tuple[dict[str, dict], dict]:
    """Attach cached/lazily translated English prose to selected profile copies."""
    namespace = str(cache_namespace or "").strip()
    cache = visual_profile_translation_cache.load_cache() if visual_profiles and namespace else {}
    translated_profiles, summary, misses, targets_by_ref = plan_profile_context_translations(
        visual_profiles, cache_namespace=namespace, cache=cache,
    )
    if not translated_profiles or not namespace:
        return translated_profiles, summary
    if not misses:
        print(
            "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 모든 원문이 캐시에 있어 "
            "락 없는 빠른 경로로 LLM 호출 생략: "
            f"namespace={namespace!r}, state={summary!r}"
        )
        return translated_profiles, summary

    async with _profile_context_translation_lock:
        # Another request may have filled these misses while this request waited.
        # Reload atomically replaced cache data and only translate what is still absent.
        cache = visual_profile_translation_cache.load_cache()
        remaining_misses: list[dict] = []
        remaining_targets: dict[str, tuple[dict, str, str, str, str]] = {}
        for item in misses:
            ref = item["ref"]
            profile, character_name, profile_id, field, source = targets_by_ref[ref]
            cached = visual_profile_translation_cache.cached_translation(
                cache,
                bot_name=namespace,
                character_name=character_name,
                profile_id=profile_id,
                field=field,
                source_text=source,
            )
            if cached == "":
                summary["empty_fields"] += 1
                summary["cache_misses"] -= 1
                continue
            if cached is not None:
                profile[f"{field}_english"] = cached
                summary["cache_hits"] += 1
                summary["cache_misses"] -= 1
                continue
            remaining_misses.append(item)
            remaining_targets[ref] = targets_by_ref[ref]
        misses = remaining_misses
        targets_by_ref = remaining_targets

        if not misses:
            print(
                "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 락 대기 중 캐시가 채워져 "
                "LLM 호출 생략: "
                f"namespace={namespace!r}, state={summary!r}"
            )
            return translated_profiles, summary

        system_prompt = (
            "Translate the supplied character-profile context into concise, faithful English "
            "for a downstream illustration scene-selection model. Preserve the full meaning, "
            "conditions, chronology, uncertainty, names, bracketed references, and Danbooru-style "
            "tags. Do not add facts, infer age, merge entries, or turn an appearance reference "
            "into a selection condition. Return only one JSON object with this shape: "
            '{"translations":[{"ref":"text_1","english":"..."}]}. '
            "Include every supplied ref exactly once and no additional refs."
        )
        messages = _normalize_messages([
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Translate only these missing cached fields. Each entry remains independent.\n\n"
                    + variable(json.dumps(misses, ensure_ascii=False, indent=2), "번역할 프로필 자료")
                ),
            },
        ])
        expected_refs = [item["ref"] for item in misses]

        def validate_translation_result(result):
            try:
                _parse_profile_context_translations(result, expected_refs)
                return True, ""
            except Exception as exc:
                return False, str(exc)

        try:
            raw = await _call_pipeline_llm(
                PROFILE_CONTEXT_TRANSLATE_CALL_NAME,
                messages,
                stream_notify,
                result_validator=validate_translation_result,
                json_mode=True,
                history_ids_sink=history_ids_sink,
            )
            parsed = _parse_profile_context_translations(raw, expected_refs)
        except asyncio.CancelledError:
            print(
                "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 번역 취소: "
                f"namespace={namespace!r}, misses={misses!r}, state={summary!r}"
            )
            raise
        except Exception as exc:
            summary["fallback_fields"] = len(misses)
            print(
                "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 번역 실패, 이번 실행은 원문 사용: "
                f"namespace={namespace!r}, misses={misses!r}, state={summary!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return translated_profiles, summary

        for ref, english in parsed.items():
            profile, character_name, profile_id, field, source = targets_by_ref[ref]
            profile[f"{field}_english"] = english
            visual_profile_translation_cache.update_cached_translation(
                cache,
                bot_name=namespace,
                character_name=character_name,
                profile_id=profile_id,
                field=field,
                source_text=source,
                english=english,
            )
            summary["translated_fields"] += 1

        try:
            visual_profile_translation_cache.save_cache(cache)
            summary["cache_saved"] = True
        except Exception as exc:
            print(
                "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 번역은 사용하지만 캐시 저장 실패: "
                f"namespace={namespace!r}, state={summary!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

        print(
            "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 지연 번역 준비 완료: "
            f"namespace={namespace!r}, state={summary!r}"
        )
        return translated_profiles, summary


@illustration_flow.stage(
    "PROFILE-CONTEXT-CACHE",
    executor="process",
    layout_group="profile_context_translation",
)
async def _record_profile_context_cache_summary(summary: dict) -> dict:
    normalized = dict(summary or {})
    print(
        "[ILLUST_CONTEXT:PROFILE_CONTEXT_TRANSLATE] 작업 흐름 캐시 상태 기록: "
        f"state={normalized!r}"
    )
    return normalized


async def resolve_profiles_before_generation(
    *,
    payload: dict,
    toggles: dict | None,
    history_plan: dict | None,
    visual_profiles: dict[str, dict] | None,
    profile_translation_namespace: str = "",
    stream_notify=None,
    progress=None,
    history_ids_sink: list[str] | None = None,
    stage_runner=None,
) -> tuple[str, dict]:
    """Resolve CURRENT characters and every applicable multi-profile timeline."""
    # The preview supplies side-effect-free stage implementations. The actual
    # ordering and CURRENT-character branch selection remain in this function.
    async def run_stage(operation, *args, **kwargs):
        if stage_runner is not None:
            return await stage_runner(operation, *args, **kwargs)
        return await operation(*args, **kwargs)
    candidate_names = [
        str(name or "").strip()
        for name, value in (visual_profiles or {}).items()
        if str(name or "").strip()
        and isinstance(value, dict)
    ]
    if not candidate_names:
        print(
            "[ILLUST_CONTEXT:CHARACTER_RESOLVE] 등록된 캐릭터가 없어 "
            "최전단 호출 건너뜀"
        )
        return "", _empty_profile_result()

    chats = payload.get("chats") or []
    target_index, narrative = _latest_narrative(chats)
    if target_index < 0 or not str(narrative or "").strip():
        print(
            "[ILLUST_CONTEXT:CHARACTER_RESOLVE] CURRENT 등장인물 판별용 최신 CHAR 서사 없음: "
            f"chats={len(chats)}"
        )
        raise RuntimeError("CURRENT 등장인물 판별용 최신 CHAR 서사를 찾지 못했습니다")
    # The Risu module applies the selected context-filter policy before transport.
    # Keep that text authoritative so XML "off" and "OUTPUT only" remain distinct.
    current_context = str(narrative or "").strip()
    current_slotted = str(payload.get("target_slotted") or "").strip() or insert_slots(current_context)
    current_context = remove_slot_markers(current_slotted)
    segmented_current, current_segments = _segment_current_context(
        current_context, slotted_context=current_slotted)
    if progress:
        await progress(1, "character_resolve", "CURRENT 등장인물 판별")
    prompts = load_prompt_files()
    character_output, character_result = await run_stage(_run_character_resolution,
        character_system=prompts.get("character_resolve", ""),
        segmented_current=segmented_current,
        current_context=current_context,
        current_segments=current_segments,
        candidate_names=candidate_names,
        visual_profiles=visual_profiles,
        stream_notify=stream_notify,
        history_ids_sink=history_ids_sink,
    )
    previous_state = (history_plan or {}).get("state_before") or {}
    character_result = _mark_current_characters_with_tracked_history(
        character_result,
        previous_state,
        visual_profiles,
    )
    current_names = [
        str(item.get("name") or "").strip()
        for item in character_result.get("characters") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    current_profiles = _selected_visual_profiles(
        visual_profiles,
        current_names,
        multiple_only=False,
    )
    multi_profile_names = [
        name
        for name in current_names
        if len((current_profiles.get(name) or {}).get("profiles") or []) > 1
    ]
    if not multi_profile_names:
        print(
            "[ILLUST_CONTEXT:PROFILE_RESOLVE] CURRENT 다중 프로필 캐릭터가 없어 "
            f"프로필 LLM 호출 건너뜀: characters={current_names}"
        )
        return character_output, character_result

    if progress:
        await progress(2, "profile_context_translation", "프로필 영어 컨텍스트 준비")
    profile_resolution_inputs = _selected_visual_profiles(
        current_profiles,
        multi_profile_names,
        multiple_only=True,
    )
    translated_profiles, _translation_summary = (
        await run_stage(prepare_profile_context_translations,
            profile_resolution_inputs,
            cache_namespace=profile_translation_namespace,
            stream_notify=stream_notify,
            history_ids_sink=history_ids_sink,
        )
    )
    await run_stage(_record_profile_context_cache_summary, _translation_summary)
    if progress:
        await progress(3, "profile_resolve", "CURRENT 다중 프로필 결정")
    profile_output, profile_result = await run_stage(_run_profile_resolution,
        profile_system=prompts.get("profile_resolve", ""),
        segmented_current=segmented_current,
        current_context=current_context,
        current_segments=current_segments,
        candidate_names=multi_profile_names,
        previous_state=previous_state,
        visual_profiles=translated_profiles,
        stream_notify=stream_notify,
        history_ids_sink=history_ids_sink,
    )
    return profile_output, _merge_character_and_profile_stage_results(
        character_result,
        profile_result,
        current_profiles,
    )


def _parse_multi_char_layout_response(
    text: str,
    expected_names: list[str],
    source_characters: list[dict] | None = None,
) -> dict:
    source = str(text or "").strip()
    if not source:
        raise ValueError("마스크 레이아웃 응답이 비어 있습니다")
    if source.startswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source, flags=re.I)
        source = re.sub(r"\s*```$", "", source)
    object_start = source.find("{")
    if object_start < 0:
        raise ValueError("마스크 레이아웃 응답에 JSON object가 없습니다")
    try:
        value, _end = json.JSONDecoder().raw_decode(source[object_start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"마스크 레이아웃 JSON 파싱 실패: {exc}") from exc
    if isinstance(value, dict) and isinstance(value.get("regions"), list):
        for raw_region in value["regions"]:
            if isinstance(raw_region, dict) and "outfit_state" in raw_region:
                print(f"[ILLUST_CONTEXT:MULTI_CHAR] 모델의 outfit_state 무시: character={raw_region.get('name')!r}")
                raw_region.pop("outfit_state")
    if source_characters is not None:
        if not isinstance(value, dict) or not isinstance(value.get("regions"), list):
            raise ValueError("마스크 레이아웃 regions가 list가 아닙니다")
        source_by_name = {
            str(character.get("name") or "").strip().casefold(): character
            for character in source_characters
            if isinstance(character, dict)
            and str(character.get("name") or "").strip()
        }
        ignored_rewrites = []
        for raw_region in value["regions"]:
            if not isinstance(raw_region, dict):
                continue
            region_name = str(raw_region.get("name") or "").strip()
            character = source_by_name.get(region_name.casefold())
            if character is None:
                continue
            authoritative_positive = str(character.get("positive") or "").strip()
            authoritative_negative = str(character.get("negative") or "").strip()
            model_prompt = str(raw_region.get("character_prompt") or "").strip()
            if model_prompt and model_prompt != authoritative_positive:
                ignored_rewrites.append(region_name)
            # Call5 owns only spatial decomposition. Character text is injected
            # from the already parsed/validated Call2 descriptor on the server.
            raw_region["character_prompt"] = authoritative_positive
            raw_region["positive"] = authoritative_positive
            raw_region["negative"] = authoritative_negative
            outfit_state = character.get("outfit_state")
            if isinstance(outfit_state, dict) and outfit_state:
                raw_region["outfit_state"] = deepcopy(outfit_state)
            else:
                raw_region.pop("outfit_state", None)
        if ignored_rewrites:
            print(
                "[ILLUST_CONTEXT:MULTI_CHAR] 모델의 캐릭터 재작성 결과 무시: "
                f"characters={ignored_rewrites}"
            )
    if isinstance(value, dict):
        value["mask_render_version"] = multi_char_mask.MASK_RENDER_VERSION
    return multi_char_mask.validate_multi_char_layout(
        value,
        expected_names,
        require_prompt_separation=True,
        require_character_prompt=True,
    )


@illustration_flow.collect_branches
async def calculate_multi_char_layouts(
    descriptors: list[dict],
    prompt_template: str,
    stream_notify=None,
    positive_note: str = "",
    canvas_sizes: dict | None = None,
    single_char_mask_enabled: bool = True,
    mask_inline: bool = False,
) -> None:
    """1~3인 descriptor의 영역과 배경/캐릭터 프롬프트를 병렬 계산한다."""
    targets = []
    for descriptor in descriptors:
        characters = [
            character
            for character in (descriptor.get("characters") or [])
            if isinstance(character, dict) and str(character.get("name") or "").strip()
        ]
        if len(characters) > len(multi_char_mask.MASK_CHANNELS):
            descriptor["multi_char_layout_error"] = (
                f"Regional RGB 마스크는 최대 {len(multi_char_mask.MASK_CHANNELS)}명까지 지원합니다: "
                f"actual={len(characters)}"
            )
            print(
                f"[ILLUST_CONTEXT:MULTI_CHAR] 캐릭터 수 초과로 해당 슬롯 제외: "
                f"slot={descriptor.get('slot')}, characters={len(characters)}"
            )
        elif len(characters) == 1 and not single_char_mask_enabled:
            print(f"[ILLUST_CONTEXT:MULTI_CHAR] 1인도 마스크 OFF로 영역 계산 생략: slot={descriptor.get('slot')}")
        elif characters:
            targets.append((descriptor, characters))
        else:
            print(f"[ILLUST_CONTEXT:MULTI_CHAR] 명명된 인물이 없어 영역 계산 생략: slot={descriptor.get('slot')}")
    if not targets:
        print("[ILLUST_CONTEXT:MULTI_CHAR] 영역 계산 대상 없음")
        return

    if mask_inline:
        # Keep the generated mask at its normal graph position without
        # inventing another LLM call or a separate history record.
        run = illustration_flow._run.get()
        dependencies = illustration_flow._frontier.get()
        nodes = []
        for descriptor, characters in targets:
            slot = descriptor.get("slot")
            source_call = "CALL2-KEYVIS" if descriptor.get("kind") == "keyvis" else "CALL2-DETAIL"
            node_id = None
            if run is not None:
                source_node = next((run["nodes"][key] for key in dependencies
                                    if key in run["nodes"]), {})
                if str(source_node.get("label") or "").startswith("CALL2-DETAIL"):
                    source_call = "CALL2-DETAIL"
                node_id = illustration_flow.add_node(
                    run, f"MULTI-CHAR-MASK slot={slot}", dependencies=dependencies,
                    kind="stage", executor="process", status="processing", started_at=time.time(),
                    task_key="illustration_multi_char_mask", layout_group="illustration_multi_char_mask",
                    illustration_slot=slot, mask_generation_mode="inline", source_call_name=source_call,
                    source_node_id=source_node.get("id"), model=source_node.get("model"),
                    service=source_node.get("service"),
                    summary=f"현재 모델에서 자체 생성 · {source_call} · 별도 LLM 호출 없음",
                    input=deepcopy(descriptor.get("multi_char_layout_request") or {
                        "source_call": source_call, "slot": slot,
                    }),
                )
                nodes.append(node_id)
            try:
                layout = multi_char_mask.validate_multi_char_layout(
                    descriptor.get("multi_char_layout"),
                    [character["name"] for character in characters],
                    require_prompt_separation=True,
                )
                descriptor["multi_char_layout"] = layout
                descriptor.pop("multi_char_layout_error", None)
                if node_id is not None:
                    illustration_flow.update(run, node_id, status="completed", output=deepcopy(layout))
                print(f"[ILLUST_CONTEXT:MULTI_CHAR] 현재 모델 자체 생성 마스크 사용: slot={slot}, source={source_call}")
            except Exception as exc:
                descriptor.pop("multi_char_layout", None)
                descriptor["multi_char_layout_error"] = str(exc)
                print(f"[ILLUST_CONTEXT:MULTI_CHAR] 자체 생성 마스크 실패: slot={slot}, input={descriptor!r}, error={exc}")
                traceback.print_exc()
                if node_id is not None:
                    illustration_flow.update(run, node_id, status="failed", error=str(exc),
                                             output={"error": str(exc), "raw_response": descriptor.get("multi_char_layout_raw_response", "")})
        if nodes:
            illustration_flow._frontier.set(tuple(nodes))
        return

    system_prompt = str(prompt_template or "").strip()
    if not system_prompt:
        error = "multi_char_mask.txt 프롬프트가 비어 있습니다"
        print(f"[ILLUST_CONTEXT:MULTI_CHAR] 레이아웃 계산 불가: {error}")
        for descriptor, _characters in targets:
            descriptor["multi_char_layout_error"] = error
        return

    async def calculate_one(index: int, descriptor: dict, characters: list[dict]) -> None:
        expected_names = [str(character.get("name") or "").strip() for character in characters]
        slot = descriptor.get("slot")
        scene_payload = {
            "slot": descriptor.get("anchor_segment") or ("KEYVIS" if slot == -1 else slot),
            "camera": str(descriptor.get("camera") or ""),
            "scene": str(descriptor.get("scene") or ""),
            "supplement": str(descriptor.get("supplement") or ""),
            "speak": str(descriptor.get("speak") or ""),
            "characters": [{
                "name": str(character.get("name") or ""),
                "position_hint": str(character.get("position") or ""),
                "validated_positive": str(character.get("positive") or ""),
                "validated_negative": str(character.get("negative") or ""),
            } for character in characters],
        }
        clean_positive_note = str(positive_note or "").strip()
        canvas = (canvas_sizes or {}).get("group" if len(characters) >= 2 else "solo")
        if canvas:
            scene_payload["canvas"] = (
                f"The generation canvas is {canvas[0]} pixels wide and {canvas[1]} pixels tall. "
                "Project the body on this canvas; normalize horizontal coordinates by its width "
                "and vertical coordinates by its height."
            )
        else:
            print(f"[ILLUST_CONTEXT:MULTI_CHAR] 캔버스 크기 미제공: slot={slot}; 장면의 정규화 배치만 계산")
        if clean_positive_note:
            scene_payload["positive_note"] = clean_positive_note
        descriptor["multi_char_layout_request"] = deepcopy(scene_payload)
        descriptor.pop("multi_char_layout_raw_response", None)
        messages = prompt_builders.mask_messages(system_prompt, scene_payload)

        def validate_result(result: str):
            try:
                _parse_multi_char_layout_response(
                    result,
                    expected_names,
                    characters,
                )
                return True, ""
            except Exception as exc:
                print(
                    f"[ILLUST_CONTEXT:MULTI_CHAR] 레이아웃 응답 검증 실패: "
                    f"slot={slot}, names={expected_names}, error={exc}"
                )
                traceback.print_exc()
                return False, str(exc)

        layout_stream_notify = None
        if stream_notify:
            async def layout_stream_notify(event: dict):
                payload = dict(event)
                payload["queue_subtask"] = {
                    "group_id": "multi_char_mask",
                    "group_label": "캐릭터 영역 마스크",
                    "index": index,
                    "total": len(targets),
                }
                await stream_notify(payload)

        # MULTI-CHAR-MASK는 slot(장면)마다 별도 호출되므로 이 호출의 history_id(메인 +
        # 버려지는 실패 시도)를 전역 trace 대신 sink에 모아 descriptor에 저장한다.
        # server.py 가 백업별로 (공통 trace + 자기 slot의 mask id) 만 주입하도록 쓴다.
        mask_history_ids: list[str] = []
        try:
            result = await _call_pipeline_llm(
                f"MULTI-CHAR-MASK slot={slot}",
                messages,
                layout_stream_notify,
                result_validator=validate_result,
                json_mode=True,
                history_ids_sink=mask_history_ids,
            )
            descriptor["multi_char_history_ids"] = [
                hid for hid in mask_history_ids if hid
            ]
            descriptor["multi_char_layout_raw_response"] = str(result or "")
            layout = _parse_multi_char_layout_response(
                result,
                expected_names,
                characters,
            )
            by_name = {
                str(character.get("name") or "").strip().casefold(): character
                for character in characters
            }
            descriptor["characters"] = [
                by_name[name.casefold()]
                for name in layout["character_order"]
            ]
            descriptor["multi_char_layout"] = layout
            descriptor.pop("multi_char_layout_error", None)
            print(
                f"[ILLUST_CONTEXT:MULTI_CHAR] 레이아웃 계산 완료: "
                f"slot={slot}, order={layout['character_order']}"
            )
        except Exception as exc:
            descriptor["multi_char_history_ids"] = [
                hid for hid in mask_history_ids if hid
            ]
            descriptor.pop("multi_char_layout", None)
            descriptor["multi_char_layout_error"] = str(exc)
            print(
                f"[ILLUST_CONTEXT:MULTI_CHAR] 레이아웃 계산 실패(해당 슬롯만 제외): "
                f"slot={slot}, names={expected_names}, error={exc}"
            )
            traceback.print_exc()

    await illustration_flow.gather(*(
        calculate_one(index, descriptor, characters)
        for index, (descriptor, characters) in enumerate(targets, start=1)
    ))


def build_raw_prompt(descriptor: dict, narrative: str, prompts: dict, toggles: dict) -> tuple[str, str]:
    template = prompts.get("call2_preset") or "[Positive]\n[SETUP]\n{setup}\n[CHAR]\n{char}\n[SUPPLEMENT]\n{supplement}\n\n[Negative]\n"
    positive_part, marker, negative_part = template.partition("[Negative]")
    chars = descriptor.get("characters") or []
    char_positive = " | ".join(str(ch.get("positive") or "") for ch in chars if str(ch.get("positive") or "").strip())
    char_negative = ", ".join(str(ch.get("negative") or "") for ch in chars if str(ch.get("negative") or "").strip())
    names = ", ".join(str(ch.get("name") or "") for ch in chars if str(ch.get("name") or "").strip())
    setup = ", ".join(x for x in (descriptor.get("camera", ""), descriptor.get("scene", "")) if x)
    supplement = str(descriptor.get("supplement") or "").strip()
    positive_note = str(toggles.get("positive_note") or "").strip()
    prompt_format = str(toggles.get("prompt_format") or "v3").strip().lower()

    # 모든 포맷(v1/v3/chansub)이 동일하게 V3 마커([SETUP]/[CHAR]/[SUPPLEMENT])를 내보낸다.
    # 포맷별 최종 조립(LoRA/품질 프리셋/챈섭 평탄화)은 후속 처리기(process_prompt)가 수행.
    if prompt_format not in ("v1", "v3", "chansub"):
        print(f"[ILLUST_CONTEXT] RAW 생성 중 알 수 없는 입력 형식 {prompt_format!r}, V3 사용")
    if positive_part.startswith("[Positive]"):
        positive_part = positive_part[len("[Positive]"):]
    replacements = {
        "{chat}": narrative,
        "{slot}": str(descriptor.get("slot", "")),
        "{speak}": descriptor.get("speak") or "",
        "{name}": names,
        "{setup}": setup,
        "{prompt}": setup,
        "{char}": char_positive,
        "{supplement}": supplement,
    }
    positive = PromptText(positive_part)
    for key, value in replacements.items():
        label = {
            "{chat}": "현재 원문", "{slot}": "장면 슬롯 번호", "{speak}": "POST PROCESS 대사 결과",
            "{name}": "캐릭터 이름 목록", "{setup}": "카메라·장면 태그", "{prompt}": "카메라·장면 태그",
            "{char}": "캐릭터별 긍정 태그", "{supplement}": "보충 묘사 태그",
        }[key]
        positive = positive.replace(key, variable(value, label))
    if positive_note:
        positive = positive.rstrip() + "\n" + variable(positive_note, "추가 긍정 프롬프트")
    negative = PromptText(negative_part.strip() if marker else "")
    if char_negative:
        print(f"[ILLUST_CONTEXT] LLM negative ignored (disabled): "
              f"slot={descriptor.get('slot')}, value={char_negative!r}")
        # LLM-generated negatives disabled by user request. Presets and the
        # direct negative_note below remain authoritative. Kept for reversal:
        # negative = prompt_join(", ", (x for x in (negative, variable(char_negative, "캐릭터별 부정 태그")) if x))
    if str(toggles.get("negative_note") or "").strip():
        negative = prompt_join(", ", (x for x in (negative, variable(str(toggles["negative_note"]).strip(), "추가 부정 프롬프트")) if x))
    return positive.strip(), negative.strip()


async def build_from_context(
    payload: dict,
    toggles: dict | None,
    extra_reference: str,
    progress=None,
    stream_notify=None,
    on_call2_plan_ready=None,
    on_call2_ready=None,
    on_keyvis_ready=None,
    extra_instruction: str = "",
    extra_costume: str = "",
    extra_names: str = "",
    character_names: str = "",
    enable_multi_char_layout: bool = False,
    history_plan: dict | None = None,
    visual_profile_catalog: str = "",
    visual_profiles: dict[str, dict] | None = None,
    profile_translation_namespace: str = "",
    pre_resolved_profile_output: str = "",
    pre_resolved_profile_result: dict | None = None,
    first_pass_single_v5: bool = False,
) -> dict:
    # Ordinary scenes use the streaming PLAN -> DETAIL implementation exclusively.
    # TOON decoding remains shared with Key Visual and the ENQUEUE endpoint.
    from modes import illustration_scene_pipeline

    return await illustration_scene_pipeline.build_from_context(
        payload, toggles, extra_reference,
        progress=progress, stream_notify=stream_notify,
        on_call2_plan_ready=on_call2_plan_ready, on_call2_ready=on_call2_ready,
        on_keyvis_ready=on_keyvis_ready, extra_instruction=extra_instruction,
        extra_costume=extra_costume, extra_names=extra_names,
        character_names=character_names,
        enable_multi_char_layout=enable_multi_char_layout, history_plan=history_plan,
        visual_profiles=visual_profiles,
        profile_translation_namespace=profile_translation_namespace,
        pre_resolved_profile_output=pre_resolved_profile_output,
        pre_resolved_profile_result=pre_resolved_profile_result,
        first_pass_single_v5=first_pass_single_v5,
    )
