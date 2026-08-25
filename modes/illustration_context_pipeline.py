"""CHAT -> CALL1/CALL2/CALL3 -> 기존 RAW 삽화 프롬프트 전단계.

RisuAI는 Comfy history에 이미지가 여러 장 있어도 첫 장만 소비한다. 이 모듈은
최초 CHAT 요청과 후속 결과 회수 요청을 구분하고, 한 세션의 모든 장면 프롬프트와
이미지를 서버 메모리에 보관할 수 있는 공통 형식을 제공한다.
"""

from __future__ import annotations

import asyncio
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

from modes import lighbd_service, llm_service, multi_char_mask, postprocess
from modes.visual_profiles import (
    profile_by_id,
    resolve_visual_base,
    tag_values as visual_tag_values,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts", "lighbd")
REQUIREMENTS_DIR = os.path.join(BASE_DIR, "요구사항")
SESSION_DIR = os.path.join(BASE_DIR, "logs", "illustration_context_sessions")

CONTEXT_PREFIX = "__LB_ILLUST_CONTEXT_V1__"
RESULT_PREFIX = "__LB_ILLUST_RESULT_V1__"
REGENERATE_PREFIX = "__LB_ILLUST_REGENERATE_V1__"
PROMPT_BATCH_PREFIX = "__LB_ILLUST_PROMPT_BATCH_V1__"
EASY_EDIT_PREFIX = "__LB_ILLUST_EASY_EDIT_V1__"
MAX_ILLUSTRATION_SLOT_COUNT = 65
MAX_EASY_EDIT_DIRECTION_LENGTH = 4000
CALL5_MAX_PAIRWISE_OVERLAP_RATIO = 0.60

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
    "call1_backtranslate": "backtranslate.txt",
    "call1_enhance": "enhance.txt",
    "call2_jailbreak": "jailbreak.txt",
    "call2_job": "job.txt",
    "call2_prefill": "prefill.txt",
    "call2_thoughts": "thoughts.txt",
    "call2_system": "system.txt",
    "call2_format": "format.txt",
    "call2_preset": "preset.txt",
    "call3_speak": "speak.txt",
    "call3_manga": "manga.txt",
    # NSFW(SOFT/HARD) 버블 타입 보강. manga 모드 + nsfw 토글 ON일 때만 manga 프롬프트
    # 끝에 추가로 주입한다(일반 작업에 노출 안 됨).
    "call3_manga_nsfw": "manga_nsfw.txt",
    "call2_fix": "repair.txt",
    "multi_char_mask": "multi_char_mask.txt",
}

DEFAULT_TOGGLES = {
    "call1_backtranslate_enabled": False,
    "call1_backtranslate_max_concurrency": 4,
    "call1_backtranslate_slow_retry_enabled": False,
    "call1_backtranslate_slow_retry_remaining": 1,
    "call1_backtranslate_slow_retry_progress_enabled": True,
    "call1_backtranslate_slow_retry_progress_threshold": 50,
    "call1_backtranslate_slow_retry_tps_enabled": False,
    "call1_backtranslate_slow_retry_tps_threshold": 5.0,
    "call1_backtranslate_slow_retry_condition_operator": "and",
    "call1_backtranslate_failure_strategy": "fallback",
    "call1_enabled": True,
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
    "call3_context_turns": 5,
    "call3_enabled": True,
    "multi_char_mask_enabled": True,
    "speak_enabled": True,
    "call3_prompt_mode": "speak",
    "speak_language": "한국어",
    "speak_emotion_enabled": False,
    "speak_emotions": "",
    "nsfw": False,
    "supplement": True,
    "key_visual": True,
    "minimal_background_description": True,
    "character_limit": 3,
    # scene_mode: "manual" = 서버가 최소/최대 강제, "auto" = lb-xnai(call2)에 완전 방임
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
    "compat_character_divider": "newline",
    "compat_character_prompt": "separate",
}


# 삽화 장면 수의 유일한 카운트 규칙. presets.json(봇 시스템 프롬프트)에 하드코딩되던
# 블록을 분리해 편집 불가능한 고정 프롬프트로 중앙화했다. {min}/{max}는 서버가
# output_count_min/max 토글(PLAN은 총량, 병렬 Call2-detail worker는 총량÷worker수)로
# 치환해 주입한다. 이 템플릿 자체는 사용자가 직접 편집하지 않는다.
OUTPUT_COUNT_RULE_TEMPLATE = """## Output Count Rule
Each response MUST contain a minimum of {min} and a maximum of {max} image tags.
This is a hard constraint, not a suggestion.
- If the scene naturally calls for fewer than {min} distinct visual moments, find additional meaningful moments to illustrate (a gesture, an environment shot, a character's expression, an object of focus).
- Character Count & Focus: Tailor the character count to the specific focus of the image. If character interaction is emphasized, include a maximum of 2 characters. If a character's emotion or expression is the focal point, restrict the image to a maximum of 1 character.
- Distribution Ratio: Across your total image output, maintain a recommended ratio of 70-80% single-character images and 20-30% two-character interaction images.
- If the scene contains more than {max} potential visual moments, select the {max} most impactful ones.
- Never output under {min} images. Never output {max} or more."""


def render_output_count_rule(min_value: int, max_value: int) -> str:
    """output_count_min/max 값을 카운트 규칙 템플릿에 치환해 반환한다."""
    return OUTPUT_COUNT_RULE_TEMPLATE.replace("{min}", str(int(min_value))).replace(
        "{max}", str(int(max_value))
    )

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
    prompt_format = str(out.get("prompt_format") or "").strip().lower()
    if prompt_format not in ("v1", "v3", "chansub"):
        print(f"[ILLUST_CONTEXT] 지원하지 않는 프롬프트 입력 형식 {prompt_format!r}, V3 사용")
        prompt_format = "v3"
    out["prompt_format"] = prompt_format
    call3_prompt_mode = str(out.get("call3_prompt_mode") or "").strip().lower()
    if call3_prompt_mode not in ("speak", "manga"):
        print(
            f"[ILLUST_CONTEXT] 지원하지 않는 CALL3 대사 프롬프트 "
            f"{call3_prompt_mode!r}, speak 사용"
        )
        call3_prompt_mode = "speak"
    out["call3_prompt_mode"] = call3_prompt_mode
    failure_strategy = str(
        out.get("call1_backtranslate_failure_strategy") or ""
    ).strip().lower()
    if failure_strategy not in ("fallback", "retry_abort"):
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE] 지원하지 않는 실패 전략 "
            f"{failure_strategy!r}, fallback 사용"
        )
        failure_strategy = "fallback"
    out["call1_backtranslate_failure_strategy"] = failure_strategy
    condition_operator = str(
        out.get("call1_backtranslate_slow_retry_condition_operator") or ""
    ).strip().lower()
    if condition_operator not in ("and", "or"):
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE] 지원하지 않는 느린 요청 조건 결합 방식 "
            f"{condition_operator!r}, and 사용"
        )
        condition_operator = "and"
    out["call1_backtranslate_slow_retry_condition_operator"] = condition_operator
    for prefix, label in (
        ("call1_parallel", "CALL1 병렬"),
        ("call2_parallel", "CALL2 병렬"),
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
        out["call1_backtranslate_max_concurrency"] = max(
            1,
            min(16, int(out["call1_backtranslate_max_concurrency"])),
        )
        out["call1_backtranslate_slow_retry_remaining"] = max(
            1,
            min(16, int(out["call1_backtranslate_slow_retry_remaining"])),
        )
        out["call1_backtranslate_slow_retry_progress_threshold"] = max(
            1,
            min(99, int(out["call1_backtranslate_slow_retry_progress_threshold"])),
        )
        out["call1_backtranslate_slow_retry_tps_threshold"] = max(
            0.1,
            min(1000.0, float(out["call1_backtranslate_slow_retry_tps_threshold"])),
        )
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
            "call1_backtranslate_max_concurrency": DEFAULT_TOGGLES[
                "call1_backtranslate_max_concurrency"
            ],
            "call1_backtranslate_slow_retry_remaining": DEFAULT_TOGGLES[
                "call1_backtranslate_slow_retry_remaining"
            ],
            "call1_backtranslate_slow_retry_progress_threshold": DEFAULT_TOGGLES[
                "call1_backtranslate_slow_retry_progress_threshold"
            ],
            "call1_backtranslate_slow_retry_tps_threshold": DEFAULT_TOGGLES[
                "call1_backtranslate_slow_retry_tps_threshold"
            ],
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
        })
    # 예전 UI에서 저장한 고정 배치 크기는 더 이상 사용하지 않는다. CALL2-PLAN이
    # 선택한 전체 장면 수를 최대 동시 요청 수에 맞춰 자동 분배한다.
    out.pop("call2_parallel_batch_size", None)
    # CALL1 병렬도 segment를 최대 동시 요청 수만큼 균등 분할하므로 청크당 segment 수
    # 설정은 더 이상 사용하지 않는다. 과거 저장값이 남아 있으면 무시.
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
    """UI 편집본 저장. 기존 텍스트는 요구사항/에 먼저 백업한다."""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(REQUIREMENTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved = []
    for key, filename in PROMPT_FILES.items():
        if key not in values:
            continue
        path = os.path.join(PROMPTS_DIR, filename)
        try:
            if os.path.exists(path):
                backup = os.path.join(REQUIREMENTS_DIR, f"lighbd_{filename}.{stamp}.bak")
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
    if action not in ("regenerate", "generate", "result"):
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
    return images[index - 1]


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
            return images[index]
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
        content = str(msg.get("content") or "").strip()
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
        + _strip_nodes(str(item.get("data") or item.get("content") or ""))
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
    """Normalize pre-CALL1 history facts without asking an LLM to classify them."""
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


def _fixed_appearance_authority_content(fixed_appearance: dict[str, str]) -> str:
    if not fixed_appearance:
        return ""
    return (
        "# AUTHORITATIVE FIXED APPEARANCE\n"
        "For each named character, this server-extracted map is the only authority for "
        "persistent identity and is a complete base, not a menu. Copy every supplied tag. "
        "Narrative and the assigned PLAN may control scene-specific pose, action, expression, "
        "and a directly conflicting temporary appearance change; a separate server audit "
        "validates the exact replaced base tag. Authoritative wardrobe state controls "
        "temporary attire. Generated visual references may inform a scene only as permitted "
        "by SERVER REFERENCE CLASSIFICATION. None of those sources may silently shorten, "
        "extend, or replace persistent traits.\n\n"
        + json.dumps(fixed_appearance, ensure_ascii=False, indent=2)
    )


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
    else:
        type_rule = (
            "Chronology is unverified. Use this only as optional composition or visual guidance. "
            "Do not inherit any identity, wardrobe, pose, expression, action, or temporary state."
        )

    metadata = {
        key: value
        for key, value in (classification or {}).items()
        if key != "character_count"
    }
    return (
        "# CLASSIFIED LAST VISUAL REFERENCE\n"
        "The server, not the model, has classified this generated reference.\n"
        f"{type_rule}\n\n"
        "AUTHORITATIVE FIXED APPEARANCE and each assigned wardrobe_snapshot always override "
        "this payload. The generated positive_tags below are never a persistent identity source "
        "and never a wardrobe authority. Do not promote, complete, or repeat an appearance tag "
        "unless it is independently supported by AUTHORITATIVE FIXED APPEARANCE.\n\n"
        "# SERVER REFERENCE CLASSIFICATION\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n\n# NON-AUTHORITATIVE GENERATED REFERENCE PAYLOAD\n"
        + json.dumps(previous_visual, ensure_ascii=False, indent=2)
    )


def _call1_state_for_prompt(
    character_state: dict | None,
    costume_reference: str = "",
) -> dict:
    """Remove generated visual data that CALL1 does not consume.

    Stored state is left untouched. CALL1 still receives current wardrobe and its
    evidence timeline. A duplicated default outfit is removed only when that
    character is present in the dedicated lb.extra costume block.
    """
    if not isinstance(character_state, dict):
        if character_state is not None:
            print(
                f"[ILLUST_CONTEXT:CALL1] 추적 캐릭터 상태 형식 오류로 빈 상태 사용: "
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
    malformed = []
    for name, tracked in result.items():
        if not isinstance(tracked, dict):
            malformed.append(str(name))
            continue
        if "last_visual_reference" in tracked:
            tracked.pop("last_visual_reference", None)
            removed_visual += 1
        if "default_outfit_reference" in tracked:
            canonical_name = str(tracked.get("canonical_name") or name).strip()
            if canonical_name.casefold() in costume_names:
                tracked.pop("default_outfit_reference", None)
                removed_default += 1
            else:
                preserved_default += 1
    if malformed:
        print(
            f"[ILLUST_CONTEXT:CALL1] 비정상 추적 캐릭터 상태는 원형 보존: "
            f"characters={malformed}"
        )

    try:
        before_chars = len(json.dumps(character_state, ensure_ascii=False, indent=2))
        after_chars = len(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            f"[ILLUST_CONTEXT:CALL1] 입력 상태 정리: characters={len(result)}, "
            f"visual_removed={removed_visual}, default_duplicate_removed={removed_default}, "
            f"default_preserved_without_extra={preserved_default}, "
            f"chars={before_chars}->{after_chars} (-{before_chars - after_chars})"
        )
    except Exception as e:
        print(f"[ILLUST_CONTEXT:CALL1] 입력 상태 정리 크기 계산 실패: error={e}")
        traceback.print_exc()
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


def _segment_current_context(text: str) -> tuple[str, dict[str, dict]]:
    """Give CALL1 stable segment ids without hardcoding a pronoun vocabulary."""
    source = str(text or "")
    segments: dict[str, dict] = {}
    rendered = []
    cursor = 0
    index = 1
    for match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", source):
        content = match.group(0).strip()
        if not content:
            continue
        segment_id = f"C{index:03d}"
        segments[segment_id] = {
            "id": segment_id,
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        rendered.append(f"[{segment_id}]\n{match.group(0)}")
        cursor = match.end()
        index += 1
    if not segments and source.strip():
        segments["C001"] = {"id": "C001", "text": source, "start": 0, "end": len(source)}
        rendered.append(f"[C001]\n{source}")
    return "\n\n".join(rendered), segments


def _json_object_from_text(text: str) -> dict | None:
    source = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", source, re.I)
    if fenced:
        source = fenced.group(1).strip()
    start = source.find("{")
    end = source.rfind("}")
    if start < 0 or end <= start:
        print("[ILLUST_CONTEXT:CALL1] 구조화 JSON object를 찾지 못함")
        return None
    try:
        value = json.loads(source[start:end + 1])
        if not isinstance(value, dict):
            print(f"[ILLUST_CONTEXT:CALL1] 구조화 결과 루트가 object가 아님: {type(value).__name__}")
            return None
        return value
    except Exception as e:
        print(f"[ILLUST_CONTEXT:CALL1] 구조화 JSON 파싱 실패: {e}; raw={source!r}")
        traceback.print_exc()
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
) -> dict | None:
    """Validate CALL1's compact entity/coreference/wardrobe analysis."""
    raw = _json_object_from_text(text)
    if raw is None:
        return None
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

    for folded, name in canonical.items():
        if _contains_canonical_name(current_context, name) and folded not in current_names:
            current_names.add(folded)
            current_characters.append({"name": name, "confidence": 1.0})
            warnings.append(f"CALL1이 원문 정식 이름을 누락해 서버가 보완: {name}")

    history_names = {name.casefold() for name in history_characters}
    for folded, name in canonical.items():
        if _contains_canonical_name(history_context, name) and folded not in history_names:
            history_names.add(folded)
            history_characters.append(name)
            warnings.append(f"CALL1이 과거 히스토리 정식 이름을 누락해 서버가 보완: {name}")

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
        # legacy 출력 호환: 신규 CALL1은 wardrobe_change만 내고 items를 비운다.
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
        if operation in changing_operations:
            segment_text = str((segments.get(segment_id) or {}).get("text") or "")
            if (
                not segment_id
                or not evidence
                or not _analysis_evidence_matches_segment(evidence, segment_text)
            ):
                warnings.append(
                    f"복장 변경 근거 불일치로 폐기: character={name}, operation={operation}, "
                    f"segment={segment_id!r}"
                )
                continue
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

    visual_base_events = []
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

    return {
        "reference_assignments": assignments,
        "history_characters": history_characters,
        "current_characters": current_characters,
        "wardrobe_events": wardrobe_events,
        "visual_base_events": visual_base_events,
        "hairstyle_events": hairstyle_events,
        "unresolved_references": unresolved,
        "validation_warnings": warnings,
        "fallback_errors": fallback_errors,
        # 기존 소비자를 위해 한 목록도 유지하되, 폴백 여부는 치명 오류만 결정한다.
        "validation_errors": warnings + fallback_errors,
        "fallback_required": bool(fallback_errors),
    }


def apply_reference_assignments(
    current_context: str,
    segments: dict[str, dict],
    assignments: list[dict],
) -> tuple[str, list[str], dict[str, str]]:
    """Apply validated CALL1 assignments by exact segment span, never by keyword list."""
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
    """Apply CALL1 events as sparse deltas over the complete default outfit.

    CALL1 intentionally emits compact changed items, not a complete visual tag
    snapshot. Therefore omitted default items are never removed implicitly by a
    short ``set``/``replace`` payload.

    ``hairstyle_events`` are stored verbatim into each character's
    ``hairstyle_timeline`` (append-only, capped). The server never interprets
    hairstyle semantics; CALL2/AUDIT resolve them against fixed appearance.
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
        # 신규 CALL1은 items 대신 자연어 wardrobe_change만 낸다. CALL2가 이 의미 변화를
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
            # CALL1's compact list is a sparse change description. Treat it as an
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
    # CALL2/AUDIT가 이 timeline을 fixed appearance에 대해 해석한다.
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
    applicable_events: list[dict],
    plan_character_names: list[str],
) -> tuple[str, list[str]]:
    """Keep CALL1's story meaning intact for one scene without interpreting tags."""
    canonical_by_name = {
        str(name or "").strip().casefold(): str(name or "").strip()
        for name in plan_character_names or []
        if str(name or "").strip()
    }
    statements: list[str] = []
    affected: list[str] = []
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
        statements.append(f"{canonical_name}: {statement}")
        if canonical_name.casefold() not in {name.casefold() for name in affected}:
            affected.append(canonical_name)
    if not statements:
        return "", []
    return (
        "By this point in the story, keep these chronological wardrobe, coverage, and exposure "
        "changes in force as natural-language visual authority. Read the statements by meaning; "
        "when they conflict, a later statement supersedes an earlier one.\n"
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
        result[name] = base
    return result


def _visual_base_authority_note(snapshot: dict[str, dict]) -> str:
    statements: list[str] = []
    for name, base in (snapshot or {}).items():
        appearance = ", ".join(visual_tag_values(base.get("appearance") or [])) or "(none)"
        outfit = ", ".join(visual_tag_values(base.get("outfit") or [])) or "(none)"
        statements.append(
            f"{name} is in visual profile `{base.get('visual_profile_id')}` "
            f"({base.get('visual_profile_label')}). "
            f"Its complete fixed appearance is: {appearance}. "
            f"Its default-outfit reference is: {outfit}. This outfit is a fallback "
            "reference, not fixed identity: preserve it when it fits, but design a "
            "different coherent outfit when the full scene context calls for one."
        )
    if not statements:
        return ""
    return (
        "For this exact scene, these server-selected visual profiles override the default "
        "appearance profile for the same logical character. Do not choose another profile. "
        "Only fixed appearance is mandatory; each default outfit below is a reference fallback. "
        "Apply later natural-language wardrobe continuity first, then use scene-appropriate "
        "attire when the full context calls for it.\n"
        + "\n".join(statements)
    )


def bind_scene_plan_wardrobes(
    scene_plan: list[dict],
    segment_order: list[str],
    state_before: dict,
    current_characters: list[dict],
    wardrobe_events: list[dict],
    current_message_id: str,
    selected_reference: str = "",
    default_outfits: dict[str, list[str]] | None = None,
    visual_profiles: dict[str, dict] | None = None,
    visual_base_events: list[dict] | None = None,
) -> list[dict]:
    """Bind a base snapshot and literal story continuity to each planned scene."""
    rank = {str(segment_id): index for index, segment_id in enumerate(segment_order)}
    normalized_plan = []
    resolved_outfits: dict[str, dict] = {}
    for plan_index, raw_plan in enumerate(scene_plan, start=1):
        plan = deepcopy(raw_plan)
        anchor_segment = str(plan.get("anchor_segment") or "").strip()
        if anchor_segment not in rank:
            print(
                f"[ILLUST_CONTEXT:CALL2_PLAN] 복장 스냅샷 기준 segment 없음: "
                f"plan={plan_index}, anchor={anchor_segment!r}, "
                f"segments={list(rank)}"
            )
            raise ValueError(
                f"scene_plan[{plan_index}] 복장 기준 segment 없음: {anchor_segment!r}"
            )

        applicable_events = []
        for event in wardrobe_events or []:
            event_segment = str(event.get("segment_id") or "").strip()
            if event_segment not in rank:
                print(
                    f"[ILLUST_CONTEXT:CALL2_PLAN] 복장 이벤트 segment를 찾지 못해 "
                    f"장면 스냅샷에서 제외: anchor={anchor_segment}, "
                    f"event_segment={event_segment!r}, event={event!r}"
                )
                continue
            if rank[event_segment] <= rank[anchor_segment]:
                applicable_events.append(event)

        applicable_visual_events = []
        for event in visual_base_events or []:
            event_segment = str(event.get("segment_id") or "").strip()
            if event_segment not in rank:
                print(
                    f"[ILLUST_CONTEXT:VISUAL_BASE] 외형 기반 이벤트 segment를 찾지 못해 "
                    f"장면 스냅샷에서 제외: anchor={anchor_segment}, "
                    f"event_segment={event_segment!r}, event={event!r}"
                )
                continue
            if rank[event_segment] <= rank[anchor_segment]:
                applicable_visual_events.append(event)

        plan_names = [str(name or "").strip() for name in plan.get("characters") or []]
        plan_names = [name for name in plan_names if name]
        continuity_note, continuity_characters = _scene_wardrobe_continuity_note(
            applicable_events,
            plan_names,
        )
        all_current = list(current_characters or []) + [
            {"name": name, "confidence": 1.0}
            for name in plan_names
        ]
        visual_states_at_scene = apply_visual_base_events(
            state_before,
            all_current,
            applicable_visual_events,
            current_message_id,
            visual_profiles,
        )
        scene_visual_bases = visual_base_snapshot(
            visual_states_at_scene,
            plan_names,
            visual_profiles,
        )
        scene_default_outfits = deepcopy(default_outfits or {})
        for name, base in scene_visual_bases.items():
            scene_default_outfits[name] = visual_tag_values(base.get("outfit") or [])
        states_at_scene = apply_wardrobe_events(
            visual_states_at_scene,
            all_current,
            applicable_events,
            current_message_id,
            selected_reference=selected_reference,
            default_outfits=scene_default_outfits,
        )
        planned_outfits = plan.get("planned_outfits") or {}
        wardrobe_snapshot = {}
        wardrobe_sources = {}
        for name in plan_names:
            folded_name = name.casefold()
            proposal = next((
                value
                for proposal_name, value in planned_outfits.items()
                if str(proposal_name).casefold() == folded_name
            ), {})
            planned_outfit = _normalize_outfit_state(proposal)
            tracked = _character_state(states_at_scene, name)
            tracked_outfit = _normalize_outfit_state(
                tracked.get("current_wardrobe") if isinstance(tracked, dict) else {}
            )
            if _outfit_state_is_known(planned_outfit):
                outfit = planned_outfit
                source = "call2_plan"
            elif _outfit_state_is_known(tracked_outfit):
                outfit = tracked_outfit
                source = "default_base_plus_sparse_history"
            elif folded_name in resolved_outfits:
                outfit = deepcopy(resolved_outfits[folded_name])
                source = "call2_plan_carried"
            else:
                outfit = planned_outfit
                source = "unknown"
                print(
                    f"[ILLUST_CONTEXT:CALL2_PLAN] 기본·추적 복장 상태가 모두 unknown, "
                    f"generated visual은 권위로 승격하지 않음: "
                    f"plan={plan_index}, anchor={anchor_segment}, character={name}"
                )
            wardrobe_snapshot[name] = outfit
            wardrobe_sources[name] = source
            resolved_outfits[folded_name] = deepcopy(outfit)

        plan["wardrobe_snapshot"] = wardrobe_snapshot
        plan["wardrobe_sources"] = wardrobe_sources
        plan["visual_base_snapshot"] = scene_visual_bases
        authority_note = _visual_base_authority_note(scene_visual_bases)
        if authority_note:
            plan["visual_base_authority"] = authority_note
        if continuity_note:
            plan["continuity_note"] = continuity_note
            # Internal parser guidance only. It is deliberately omitted from the
            # LLM payload so the actual inter-LLM handoff stays natural-language.
            plan["_continuity_characters"] = continuity_characters
        normalized_plan.append(plan)

    print(
        f"[ILLUST_CONTEXT:CALL2_PLAN] 장면별 복장 스냅샷 확정: "
        f"plans={[(item.get('plan_id'), item.get('anchor_segment'), item.get('slot'), item.get('wardrobe_sources')) for item in normalized_plan]}"
    )
    return normalized_plan


def _public_call2_scene_plan(plan: dict) -> dict:
    """Expose only routing plus natural meaning to a downstream CALL2 worker."""
    public = {
        "slot": int(plan.get("slot") or 0),
        "characters": [
            str(name or "").strip()
            for name in plan.get("characters") or []
            if str(name or "").strip()
        ],
        "scene_brief": str(plan.get("scene_brief") or "").strip(),
    }
    continuity_note = str(plan.get("continuity_note") or "").strip()
    if continuity_note:
        public["continuity_note"] = continuity_note
    visual_base_authority = str(plan.get("visual_base_authority") or "").strip()
    if visual_base_authority:
        public["visual_base_authority"] = visual_base_authority
    return public


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


def _strip_nodes(text: str) -> str:
    return re.sub(r"<[^>]+>[\s\S]*?</[^>]+>|<[^>]+/?>", "", str(text or ""), flags=re.I).strip()


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
            print(f"[ILLUST_CONTEXT:CALL1] 삽입 위치를 찾지 못함: {anchor!r}")
            continue
        result = result[:pos] + "\n\n" + insertion + result[pos:]
        offset += 1
    if base_tags:
        result = result.rstrip() + "\n\n" + "\n\n".join(base_tags)
    return result


def insert_slots(text: str) -> str:
    # v13 lb-xnai.gen.insertSlots와 바이트 단위로 같은 규칙을 쓴다.
    # (연속된 빈 줄마다 0부터 슬롯을 넣고, 줄 사이 공백은 별도 해석하지 않음)
    slot_index = 0

    def replace(_match):
        nonlocal slot_index
        value = f"\n\n[Slot {slot_index}]\n\n"
        slot_index += 1
        return value

    return re.sub(r"\n\n+", replace, str(text or "").strip())


_SLOT_MARKER_RE = re.compile(r"\[Slot\s+(\d+)\]")
_PROTECTED_SLOT_TOKEN_RE = re.compile(r"__SLOT_[0-9]+__")


def _protect_slot_markers(text: str) -> tuple[str, list[tuple[str, str]]]:
    """LLM이 슬롯을 구조 태그로 해석하지 않도록 불투명 토큰으로 치환한다."""
    source = str(text or "")
    protected_markers: list[tuple[str, str]] = []

    def replace(match: re.Match) -> str:
        token = f"__SLOT_{len(protected_markers)}__"
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
    """CALL1 Position을 슬롯 없는 투영 본문에서 찾는다."""
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
    """Bind every Cxxx segment to its server-owned insertion slot.

    A slot marker is the insertion boundary immediately after the preceding
    prose.  Therefore each segment uses the first following marker; only text
    after the final marker falls back to that last marker.  CALL2 never derives
    a slot number from the numeric part of a segment ID.
    """
    source = str(slotted_context or "")
    markers = list(_SLOT_MARKER_RE.finditer(source))
    if not markers:
        reason = "segment-slot 매핑 대상에 Slot 마커가 없음"
        print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
        return {}, "", reason
    if not segments:
        reason = "segment-slot 매핑 대상 segment가 없음"
        print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
        return {}, "", reason

    projected, source_indexes = _slotless_projection_with_source_indexes(source)
    _rendered_projection, projected_segments = _segment_current_context(projected)
    requested_items = list(segments.items())
    projected_items = list(projected_segments.items())
    spans: list[tuple[str, int, int, str]] = []
    excluded_segments: list[str] = []
    mapping_errors: list[str] = []

    if len(requested_items) == len(projected_items):
        for (segment_id, segment), (_projected_id, projected_segment) in zip(
            requested_items,
            projected_items,
        ):
            spans.append((
                str(segment_id),
                int(projected_segment["start"]),
                int(projected_segment["end"]),
                str(segment.get("text") or ""),
            ))
    else:
        print(
            f"[ILLUST_CONTEXT:CALL2_PLAN] 슬롯 투영 segment 수가 달라 "
            f"순차 본문 앵커로 매핑: requested={len(requested_items)}, "
            f"projected={len(projected_items)}"
        )
        projection_cursor = 0
        for segment_id, segment in requested_items:
            text = str(segment.get("text") or "")
            span = _find_position_span(projected, text, projection_cursor)
            if span is None:
                reason = (
                    f"segment-slot 본문 위치를 찾지 못함: segment={segment_id}, "
                    f"cursor={projection_cursor}, text={text!r}"
                )
                print(
                    f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}; "
                    "해당 segment만 PLAN 후보에서 제외하고 후속 매핑 계속"
                )
                excluded_segments.append(str(segment_id))
                mapping_errors.append(reason)
                continue
            start, end = span
            spans.append((str(segment_id), start, end, text))
            projection_cursor = end

    mapping: dict[str, int] = {}
    rendered = []
    for segment_id, projected_start, projected_end, text in spans:
        if (
            projected_start < 0
            or projected_end <= projected_start
            or projected_end > len(source_indexes)
        ):
            reason = (
                f"segment-slot 투영 범위 오류: segment={segment_id}, "
                f"span=({projected_start},{projected_end}), "
                f"projection_length={len(source_indexes)}"
            )
            print(
                f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}; "
                "해당 segment만 PLAN 후보에서 제외"
            )
            excluded_segments.append(segment_id)
            mapping_errors.append(reason)
            continue
        source_start = source_indexes[projected_start]
        source_end = source_indexes[projected_end - 1] + 1
        following = next(
            (marker for marker in markers if marker.start() >= source_end),
            None,
        )
        preceding = next(
            (marker for marker in reversed(markers) if marker.end() <= source_start),
            None,
        )
        marker = following or preceding
        if marker is None:
            reason = (
                f"segment 주변 Slot 마커를 찾지 못함: segment={segment_id}, "
                f"source_span=({source_start},{source_end})"
            )
            print(
                f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}; "
                "해당 segment만 PLAN 후보에서 제외"
            )
            excluded_segments.append(segment_id)
            mapping_errors.append(reason)
            continue
        slot = int(marker.group(1))
        mapping[segment_id] = slot
        rendered.append(f"[{segment_id} slot={slot}]\n{text}")

    print(
        f"[ILLUST_CONTEXT:CALL2_PLAN] segment-slot 권위 매핑 생성: "
        f"segments={len(mapping)}, slots={sorted(set(mapping.values()))}"
    )
    if mapping_errors:
        reason = (
            f"segment-slot 부분 매핑: mapped={len(mapping)}/{len(requested_items)}, "
            f"excluded={excluded_segments}; first_error={mapping_errors[0]}"
        )
        print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
        return mapping, "\n\n".join(rendered), reason
    return mapping, "\n\n".join(rendered), ""


def _merge_call1_output_into_slotted(
    slotted_body: str,
    call1_output: str,
) -> str:
    """CALL1 Position 블록을 슬롯을 보존한 본문의 실제 위치에 합친다."""
    source = str(slotted_body or "")
    output = str(call1_output or "")
    if not output.strip():
        print("[ILLUST_CONTEXT:CALL1] 슬롯 본문에 합칠 CALL1 응답이 비어 있음")
        return source

    expected_slots = _SLOT_MARKER_RE.findall(source)
    unexpected_slots = _SLOT_MARKER_RE.findall(output)
    if unexpected_slots:
        print(
            f"[ILLUST_CONTEXT:CALL1] 슬롯을 숨긴 CALL1 응답에 예상하지 못한 "
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
                f"[ILLUST_CONTEXT:CALL1] Position 병합 블록이 비어 있음: "
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
                f"[ILLUST_CONTEXT:CALL1] Position 순차 검색 실패, 전체 본문 재검색: "
                f"block={block_index}, anchor={anchor[:120]!r}"
            )
            projected_span = _find_position_span(projected, anchor, 0)
        if projected_span is None:
            print(
                f"[ILLUST_CONTEXT:CALL1] 슬롯 본문에서 Position을 찾지 못함: "
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
                f"[ILLUST_CONTEXT:CALL1] Position 원문 인덱스 변환 실패: "
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
                f"[ILLUST_CONTEXT:CALL1] Position 병합 범위가 기존 블록과 겹침: "
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
            f"[ILLUST_CONTEXT:CALL1] Position 병합 후 슬롯 검증 실패: "
            f"expected={expected_slots}, actual={actual_slots}; 원본 슬롯 본문 사용"
        )
        return source
    return result


def split_backtranslation_chunks(text: str, max_concurrency: int) -> list[str]:
    """현재 응답을 연속 슬롯 묶음으로 균등 분할한다.

    슬롯 하나마다 요청하지 않고 ``max_concurrency`` 개 이하의 연속 묶음을 만든다.
    마지막 슬롯 뒤의 꼬리 본문은 마지막 슬롯 단위에 포함한다. 슬롯 마커가 없는
    짧은 응답은 한 묶음으로 처리한다.
    """
    source = str(text or "")
    if not source:
        print("[ILLUST_CONTEXT:BACKTRANSLATE] 분할할 current context가 비어 있음")
        return []
    try:
        concurrency = max(1, min(16, int(max_concurrency)))
    except (TypeError, ValueError) as e:
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE] 최대 병렬 개수 파싱 실패: "
            f"value={max_concurrency!r}, error={e}; 1 사용"
        )
        concurrency = 1

    matches = list(_SLOT_MARKER_RE.finditer(source))
    if not matches:
        return [source]

    units = []
    cursor = 0
    for match in matches:
        units.append(source[cursor:match.end()])
        cursor = match.end()
    if cursor < len(source):
        units[-1] += source[cursor:]

    group_count = min(concurrency, len(units))
    base_size, remainder = divmod(len(units), group_count)
    chunks = []
    offset = 0
    for group_index in range(group_count):
        size = base_size + (1 if group_index < remainder else 0)
        chunks.append("".join(units[offset:offset + size]))
        offset += size
    if "".join(chunks) != source:
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE] 슬롯 묶음 재조립 검증 실패: "
            f"source_len={len(source)}, chunks={len(chunks)}"
        )
        raise RuntimeError("역번역 슬롯 묶음 분할 결과가 원문과 일치하지 않습니다")
    return chunks


def remove_slot_markers(text: str) -> str:
    """역번역된 슬롯 본문에서 배치용 마커만 제거한다."""
    return _SLOT_MARKER_RE.sub("", str(text or "")).strip()


def _valid_backtranslation(chunk: str, translated: str) -> tuple[bool, str]:
    value = str(translated or "")
    if len(value.strip()) == 0:
        return False, "응답 길이가 0임"
    expected = _SLOT_MARKER_RE.findall(str(chunk or ""))
    actual = _SLOT_MARKER_RE.findall(value)
    if actual != expected:
        return False, f"슬롯 마커 불일치(expected={expected}, actual={actual})"
    source_body = remove_slot_markers(chunk)
    translated_body = remove_slot_markers(value)
    if source_body and len(translated_body.strip()) == 0:
        return False, "번역 본문 길이가 0임"
    return True, ""


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() not in ("", "0", "false", "off", "끄기", "null")
    return bool(value)


def _render_conditionals(text: str, values: dict) -> str:
    # custom closer를 Risu의 일반 closer와 같게 취급한다.
    text = re.sub(r"\{\{/(?:compat-comfy|nsfw|supplement|context|history-length|history-null)\}\}", "{{/when}}", text)

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
        body = match.group(2)
        yes, sep, no = body.partition("{{:else}}")
        text = text[:match.start()] + (yes if cond(match.group(1)) else (no if sep else "")) + text[match.end():]
    return text


def render_call2_prompt(
    text: str,
    toggles: dict,
    history: str = "",
    *,
    include_scene_count_limit: bool = True,
) -> str:
    """Risu 토글 매크로를 서버 설정으로 렌더링한다.

    include_scene_count_limit=False 면 "# Server limits"에서 장면 수(min/max) 라인을
    생략한다. 병렬 Call2-detail worker는 PLAN이 정한 총량을 worker 수로 나눈
    per-worker 카운트 규칙을 별도로 주입받으므로, detail system에는 전체 카운트 라인을
    붙이지 않는다(3배 과잉 생성 원천 차단).
    """
    text = str(text or "")
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
            prefix += "## Character Tag History\n\n" + history.strip() + "\n\n"
        comments = []
        if str(toggles.get("focus") or "").strip():
            comments.append(
                f'I want to focus on the character(s): "{str(toggles["focus"]).strip()}". Do not make scenes for others.'
            )
        if str(toggles.get("direction") or "").strip():
            comments.append(str(toggles["direction"]).strip())
        prefix += "## Client Comments\n\n<instruction>\n" + ("\n\n".join(comments) if comments else "(None specified)") + "\n</instruction>\n\n"
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
        text = text.replace("{{getglobalvar::toggle_" + key + "}}", value)
        text = text.replace("{{getvar::" + key + "}}", value)

    def dict_element(match):
        try:
            table = json.loads(match.group(1))
            return str(table.get(str(match.group(2)).strip(), ""))
        except Exception as e:
            print(f"[ILLUST_CONTEXT] dictelement 렌더 실패: {e}; expr={match.group(0)!r}")
            traceback.print_exc()
            return ""

    text = re.sub(r"\{\{dictelement::(\{[^{}]*\})::([^{}]*)\}\}", dict_element, text)
    text = _render_conditionals(text, risu_values)
    # Risu에만 존재하는 잔여 매크로는 LLM으로 보내지 않고 로그에 남긴다.
    leftovers = re.findall(r"\{\{[^\n]{0,240}?\}\}", text)
    if leftovers:
        print(f"[ILLUST_CONTEXT] 렌더 후 잔여 Risu 매크로 {len(leftovers)}개 제거: {leftovers}")
        text = re.sub(r"\{\{[^\n]*?\}\}", "", text)
    # scene_mode == "auto" 면 장면 수에 대한 서버 제한을 일절 붙이지 않고
    # lb-xnai(call2)에 완전히 맡긴다(템플릿의 scene.quantity 도 3으로 무력화됨).
    limits = []
    if include_scene_count_limit and str(toggles.get("scene_mode")) != "auto":
        limits.append(
            f"Generate between {int(toggles['output_count_min'])} and "
            f"{int(toggles['output_count_max'])} scenes."
        )
    limits.append(f"Maximum fully visible characters per image: {int(toggles['character_limit'])}.")
    limits.append(
        f"Key visual: {'required' if toggles.get('key_visual') else 'disabled; omit keyvis'} ."
    )
    text += "\n\n# Server limits\n- " + "\n- ".join(limits)
    return text.strip()


def _keyvis_only_call2_system(rendered_system: str) -> str:
    """Remove Scene-only selection/output sections from the KEYVIS worker prompt."""
    source = str(rendered_system or "").strip()
    without_scene, scene_count = re.subn(
        r"(?ms)^### Scene\s*$.*?(?=^## Client Comments\s*$)",
        "",
        source,
        count=1,
    )
    without_examples, example_count = re.subn(
        r"(?ms)^# Example\s*$.*\Z",
        "",
        without_scene,
        count=1,
    )
    if scene_count != 1:
        print(
            "[ILLUST_CONTEXT:CALL2_KEYVIS] 공유 프롬프트에서 Scene 전용 섹션을 "
            f"찾지 못함: matches={scene_count}"
        )
    if example_count != 1:
        print(
            "[ILLUST_CONTEXT:CALL2_KEYVIS] 공유 프롬프트에서 공통 출력 예시를 "
            f"찾지 못함: matches={example_count}"
        )
    result = without_examples.strip()
    if not result:
        print(
            "[ILLUST_CONTEXT:CALL2_KEYVIS] KEYVIS 전용 system 축약 결과가 비어 "
            "공유 system을 그대로 사용"
        )
        return source
    print(
        "[ILLUST_CONTEXT:CALL2_KEYVIS] Scene 전용 지시 제거: "
        f"chars={len(source)}->{len(result)} (-{len(source) - len(result)})"
    )
    return result


def _extract_lb_block(text: str) -> str:
    match = re.search(r"<lb[-_]xnai[^>]*>([\s\S]*?)</lb[-_]xnai>", text or "", re.I)
    return match.group(1).strip() if match else ""


def _normalize_toon(text: str) -> str:
    text = re.sub(r"\b(scenes|characters)\[\d+\]:", r"\1:", text)
    text = text.replace("\t", "  ")

    # CALL2/CALL3 commonly emit TOON string fields as unquoted YAML scalars.
    # Natural-language values can legally contain YAML syntax such as ``: ``
    # or `` #`` (for example, ``supplement: split-screen: left ...``), which
    # makes PyYAML reject or truncate an otherwise valid plan.  These schema
    # fields are always strings, so encode each one as a JSON string; JSON
    # strings are valid YAML scalars and preserve punctuation verbatim.
    text_fields = {
        "camera", "positive", "negative", "name", "position",
        "scene", "supplement", "speak",
    }
    scalar_line = re.compile(
        r"^(\s*(?:-\s+)?)([A-Za-z_][\w-]*):(\s*)(.*)$"
    )
    normalized_lines = []
    for line in text.splitlines():
        match = scalar_line.match(line)
        if not match or match.group(2) not in text_fields:
            normalized_lines.append(line)
            continue

        value = match.group(4).strip()
        if not value:
            normalized_lines.append(line)
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
    return "\n".join(normalized_lines)


def _descriptor(raw: dict, kind: str, fallback_slot: int) -> dict:
    chars = []
    for ch in raw.get("characters") or []:
        if not isinstance(ch, dict):
            continue
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
            "negative": str(ch.get("negative") or "").strip(),
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
    inner = _extract_lb_block(text)
    if not inner:
        toon_match = re.search(r"\[TOON\]([\s\S]*?)\[/TOON\]", text or "", re.I)
        inner = toon_match.group(1).strip() if toon_match else ""
    if not inner:
        print(f"[ILLUST_CONTEXT:{source}] <lb-xnai> 또는 [TOON] 블록이 없음")
        return []
    try:
        data = yaml.safe_load(_normalize_toon(inner))
    except Exception as e:
        print(f"[ILLUST_CONTEXT:{source}] TOON/YAML 파싱 실패: {e}\n{inner}")
        traceback.print_exc()
        return []
    if not isinstance(data, dict):
        print(f"[ILLUST_CONTEXT:{source}] TOON 루트가 object가 아님: {type(data).__name__}")
        return []
    out = []
    keyvis = data.get("keyvis")
    if toggles.get("key_visual") and isinstance(keyvis, dict):
        out.append(_descriptor(keyvis, "keyvis", -1))
    scenes = data.get("scenes") or []
    if not isinstance(scenes, list):
        print(f"[ILLUST_CONTEXT:{source}] scenes가 list가 아님: {type(scenes).__name__}")
        scenes = []
    # auto 모드는 파싱 단계에서도 장면 수를 컷하지 않고 lb-xnai(call2)의 결정을
    # 그대로 수용한다. manual 모드일 때만 output_count_max 상한으로 잘라낸다.
    scene_cap = None if str(toggles.get("scene_mode")) == "auto" else int(toggles["output_count_max"])
    capped = scenes if scene_cap is None else scenes[:scene_cap]
    for index, raw in enumerate(capped, start=1):
        if isinstance(raw, dict):
            out.append(_descriptor(raw, "scene", index))
    if not out:
        print(f"[ILLUST_CONTEXT:{source}] 유효한 keyvis/scene 결과가 없음")
    return out


def validate_complete_call2_output(
    text: str,
    toggles: dict,
    target_slotted: str,
    source: str,
    expected_slots: list[int] | None = None,
) -> tuple[list[dict], str]:
    """Reject partial global CALL2 output instead of accepting one valid shard."""
    def fail(reason: str) -> tuple[list[dict], str]:
        print(f"[ILLUST_CONTEXT:{source}] 전체 CALL2 검증 실패: {reason}")
        return [], reason

    descriptors = parse_toon_plan(text, toggles, source)
    if not descriptors:
        return fail(f"{source} TOON 파싱 실패")

    keyvis = [item for item in descriptors if str(item.get("kind") or "") == "keyvis"]
    scenes = [item for item in descriptors if str(item.get("kind") or "") == "scene"]
    if toggles.get("key_visual") and len(keyvis) != 1:
        return fail(f"{source} keyvis 수 불일치: expected=1, actual={len(keyvis)}")
    if keyvis and (
        not str(keyvis[0].get("camera") or "").strip()
        or not str(keyvis[0].get("scene") or "").strip()
        or not (keyvis[0].get("characters") or [])
    ):
        return fail(f"{source} keyvis 필수 camera/scene/characters가 비어 있음")

    candidates = candidate_slots(target_slotted)
    actual_slots = [int(item.get("slot")) for item in scenes]
    if len(set(actual_slots)) != len(actual_slots):
        return fail(f"{source} scene slot 중복: slots={actual_slots}")
    unknown_slots = [slot for slot in actual_slots if slot not in set(candidates)]
    if unknown_slots:
        return fail(
            f"{source} 후보 밖 scene slot: slots={unknown_slots}, candidates={candidates}"
        )

    if expected_slots is not None:
        normalized_expected = [int(slot) for slot in expected_slots]
        if actual_slots != normalized_expected:
            return fail(
                f"{source} PLAN scene slot 불일치: "
                f"expected={normalized_expected}, actual={actual_slots}"
            )
    elif str(toggles.get("scene_mode")) != "auto":
        minimum = min(int(toggles["output_count_min"]), len(candidates))
        maximum = min(int(toggles["output_count_max"]), len(candidates))
        if not minimum <= len(scenes) <= maximum:
            return fail(
                f"{source} 장면 수 범위 위반: count={len(scenes)}, "
                f"required={minimum}..{maximum}"
            )
    elif not scenes:
        return fail(f"{source} auto 모드에서 장면을 하나도 반환하지 않음")

    for item in scenes:
        if (
            not str(item.get("camera") or "").strip()
            or not str(item.get("scene") or "").strip()
        ):
            return fail(
                f"{source} scene 필수 camera/scene이 비어 있음: item={item!r}"
            )
    return descriptors, ""


def _parse_call2_keyvis_output(
    text: str,
    toggles: dict,
    allowed_character_names: list[str],
    source: str,
) -> tuple[dict | None, str]:
    """Validate the independent KEYVIS worker's one-object contract."""
    local_toggles = deepcopy(toggles)
    local_toggles.update({
        "key_visual": True,
        # Keep any accidental scene objects visible so this validator can
        # reject them instead of having a manual scene cap silently drop them.
        "scene_mode": "auto",
    })
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
    if scenes:
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
        return fail("camera가 비어 있음")
    if not str(descriptor.get("scene") or "").strip():
        return fail("scene이 비어 있음")
    if not characters:
        return fail("characters가 비어 있음")
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
            return fail(f"characters[{index}].name이 비어 있음")
        if not positive:
            return fail(f"characters[{index}].positive가 비어 있음")
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
            return fail(
                f"허용 canonical 캐릭터 밖 이름: unexpected={unexpected}, "
                f"allowed={list(allowed.values())}"
            )
    return descriptor, ""


def _repair_call2_plan_slot_collisions(
    scene_plan: list[dict],
    candidates: list[int],
    *,
    segment_slot_map: dict[str, int] | None = None,
    log_errors: bool = True,
) -> tuple[list[dict], int]:
    """Resolve server-derived PLAN slot collisions without another LLM call.

    Several consecutive Cxxx segments can legitimately map to the same following
    insertion slot.  Keep the latest segment at that authoritative slot and move
    earlier plans only into unused candidate slots between the previous selected
    authoritative slot and the collision.  This preserves narrative order and all
    non-conflicting authoritative assignments.  Earlier overflow plans are dropped
    only when that interval has no room.
    """

    if len({int(item["slot"]) for item in scene_plan}) == len(scene_plan):
        return scene_plan, 0

    candidate_positions = {slot: index for index, slot in enumerate(candidates)}
    segment_positions = {
        segment: index
        for index, segment in enumerate((segment_slot_map or {}).keys())
    }
    indexed_plan = list(enumerate(scene_plan))
    indexed_plan.sort(
        key=lambda pair: (
            candidate_positions[int(pair[1]["slot"])],
            segment_positions.get(
                str(pair[1].get("anchor_segment") or ""),
                pair[0],
            ),
            pair[0],
        )
    )

    groups: list[tuple[int, list[dict]]] = []
    for _original_index, item in indexed_plan:
        authoritative_slot = int(item["slot"])
        if groups and groups[-1][0] == authoritative_slot:
            groups[-1][1].append(item)
        else:
            groups.append((authoritative_slot, [item]))

    repaired: list[dict] = []
    dropped_count = 0
    previous_authoritative_position = -1
    for authoritative_slot, group in groups:
        authoritative_position = candidate_positions[authoritative_slot]
        if len(group) == 1:
            repaired.append(group[0])
            previous_authoritative_position = authoritative_position
            continue

        # The server slot follows every segment in this group, so the latest
        # segment is the best structurally grounded owner of the exact slot.
        keeper = group[-1]
        movable = group[:-1]
        free_positions = list(
            range(previous_authoritative_position + 1, authoritative_position)
        )
        movable_count = min(len(movable), len(free_positions))
        dropped = movable[: len(movable) - movable_count]
        moved = movable[len(movable) - movable_count :]
        assigned_positions = (
            free_positions[-movable_count:] if movable_count else []
        )

        for item in dropped:
            dropped_count += 1
            if log_errors:
                print(
                    "[ILLUST_CONTEXT:CALL2_PLAN] 중복 slot 장면 제외: "
                    f"anchor={item.get('anchor_segment')!r}, "
                    f"server_slot={authoritative_slot}, reason=이전 미사용 후보 없음"
                )
        for item, assigned_position in zip(moved, assigned_positions):
            assigned_slot = candidates[assigned_position]
            item["slot"] = assigned_slot
            repaired.append(item)
            if log_errors:
                print(
                    "[ILLUST_CONTEXT:CALL2_PLAN] 중복 slot 로컬 보정: "
                    f"anchor={item.get('anchor_segment')!r}, "
                    f"server_slot={authoritative_slot}, assigned_slot={assigned_slot}"
                )

        repaired.append(keeper)
        if log_errors:
            print(
                "[ILLUST_CONTEXT:CALL2_PLAN] 중복 slot 권위 위치 유지: "
                f"anchor={keeper.get('anchor_segment')!r}, "
                f"server_slot={authoritative_slot}"
            )
        previous_authoritative_position = authoritative_position

    return repaired, dropped_count


def _scene_brief_word_set(item: dict) -> set[str]:
    """scene_brief를 비교 가능한 단어 집합으로 정규화(소문자, 3자 이상 토큰)."""
    text = str(item.get("scene_brief") or "").lower()
    return {tok for tok in re.split(r"[^a-z0-9]+", text) if len(tok) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _select_diverse_scenes(
    scene_plan: list[dict],
    maximum: int,
    candidates: list[int],
    *,
    log_errors: bool = True,
) -> list[dict]:
    """PLAN이 max 초과 반환했을 때만 호출하는 다양성 자르기(maximin).

    scene_brief 단어 Jaccard + characters 이름 Jaccard + 원문 인접도(slot 순서)로
    NxN 유사도 행렬을 만들고, 거리 최댓값 조합(farthest-point sampling)으로
    maximum개를 고른다. 비슷한 연속 beat 장면은 유사도가 높아 한 개만 생존한다.
    선택 뒤 candidates 순(원문 순)으로 재정렬한다. 임베딩·외부 호출 없이 순수 파이썬.
    """
    n = len(scene_plan)
    if maximum >= n:
        return scene_plan

    candidate_positions = {slot: index for index, slot in enumerate(candidates)}

    def order_of(item: dict, fallback: int) -> int:
        slot = item.get("slot")
        if slot in candidate_positions:
            return candidate_positions[slot]
        return fallback

    orders = [order_of(item, idx) for idx, item in enumerate(scene_plan)]
    briefs = [_scene_brief_word_set(item) for item in scene_plan]
    chars = [
        {str(name).strip().lower() for name in (item.get("characters") or [])}
        for item in scene_plan
    ]

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            tag_sim = 0.5 * _jaccard(briefs[i], briefs[j]) + 0.5 * _jaccard(chars[i], chars[j])
            prox = 1.0 / (1.0 + abs(orders[i] - orders[j]))
            value = 0.7 * tag_sim + 0.3 * prox
            sim[i][j] = value
            sim[j][i] = value

    # 시드: 원문 순 가장 첫 장면
    seed = min(range(n), key=lambda idx: (orders[idx], idx))
    picked = [seed]
    remaining = set(range(n))
    remaining.discard(seed)
    while len(picked) < maximum:
        best_idx = None
        best_score = None
        for r in remaining:
            closest = max(sim[r][p] for p in picked)
            if best_score is None or closest < best_score:
                best_score = closest
                best_idx = r
        picked.append(best_idx)
        remaining.discard(best_idx)

    picked.sort(key=lambda idx: (orders[idx], idx))
    result = [scene_plan[idx] for idx in picked]
    if log_errors:
        kept_anchors = [
            str(scene_plan[idx].get("anchor_segment") or "") for idx in picked
        ]
        print(
            "[ILLUST_CONTEXT:CALL2_PLAN] PLAN 과다 반환 → maximin 다양성 자르기: "
            f"plan={n}, maximum={maximum}, dropped={n - maximum}, "
            f"kept_anchors={kept_anchors}"
        )
    return result


def parse_call2_plan(
    text: str,
    toggles: dict,
    target_slotted: str,
    *,
    segment_slot_map: dict[str, int] | None = None,
    log_errors: bool = True,
) -> tuple[dict | None, str]:
    """Parse a global CALL2 plan or recognize a legacy complete TOON response."""
    source = str(text or "").strip()
    if not source:
        reason = "CALL2-PLAN 응답이 비어 있음"
        if log_errors:
            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
        return None, reason

    if re.search(r"<lb[-_]xnai|\[TOON\]", source, re.I):
        descriptors = parse_toon_plan(source, toggles, "CALL2-PLAN-LEGACY")
        if descriptors:
            return {
                "mode": "legacy",
                "descriptors": descriptors,
                "scene_plan": [],
                "keyvis_descriptor": None,
            }, ""

    object_start = source.find("{")
    object_end = source.rfind("}")
    if object_start < 0 or object_end <= object_start:
        reason = "CALL2-PLAN JSON object를 찾지 못함"
        if log_errors:
            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}: raw={source!r}")
        return None, reason
    try:
        raw = json.loads(source[object_start:object_end + 1])
    except Exception as e:
        reason = f"CALL2-PLAN JSON 파싱 실패: {e}"
        if log_errors:
            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}: raw={source!r}")
            traceback.print_exc()
        return None, reason
    if not isinstance(raw, dict):
        reason = f"CALL2-PLAN 루트가 object가 아님: {type(raw).__name__}"
        if log_errors:
            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
        return None, reason

    candidates = candidate_slots(target_slotted)
    candidate_set = set(candidates)
    scene_plan = []
    for index, item in enumerate(raw.get("scene_plan") or [], start=1):
        if not isinstance(item, dict):
            reason = f"scene_plan[{index}]가 object가 아님"
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}: item={item!r}")
            return None, reason
        source_segments = item.get("source_segments") or []
        if not isinstance(source_segments, list):
            source_segments = [source_segments]
        source_segments = [
            str(value).strip() for value in source_segments if str(value).strip()
        ]
        anchor_segment = str(item.get("anchor_segment") or "").strip()
        if segment_slot_map is not None:
            if not anchor_segment:
                reason = f"scene_plan[{index}] anchor_segment가 비어 있음"
                if log_errors:
                    print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}: item={item!r}")
                return None, reason
            if anchor_segment not in segment_slot_map:
                reason = (
                    f"scene_plan[{index}] 매핑 밖 anchor_segment: "
                    f"anchor={anchor_segment!r}"
                )
                if log_errors:
                    print(
                        f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}, "
                        f"mapped_segments={list(segment_slot_map)}"
                    )
                return None, reason
            # Compact PLAN output only needs one authoritative anchor.  Older
            # responses may still provide a wider source_segments list.
            if not source_segments:
                source_segments = [anchor_segment]
            unknown_segments = [
                value for value in source_segments if value not in segment_slot_map
            ]
            if unknown_segments:
                reason = (
                    f"scene_plan[{index}] 매핑 밖 source_segments: "
                    f"segments={unknown_segments}"
                )
                if log_errors:
                    print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
                return None, reason
            if anchor_segment not in source_segments:
                reason = (
                    f"scene_plan[{index}] anchor_segment가 source_segments에 없음: "
                    f"anchor={anchor_segment!r}, source_segments={source_segments}"
                )
                if log_errors:
                    print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
                return None, reason
            slot = int(segment_slot_map[anchor_segment])
            if item.get("slot") not in (None, ""):
                try:
                    model_slot = int(item.get("slot"))
                except Exception:
                    model_slot = None
                if model_slot != slot and log_errors:
                    print(
                        f"[ILLUST_CONTEXT:CALL2_PLAN] 모델 slot을 무시하고 "
                        f"anchor 권위 매핑 사용: plan={index}, model_slot={item.get('slot')!r}, "
                        f"anchor={anchor_segment}, server_slot={slot}"
                    )
        else:
            try:
                slot = int(item.get("slot"))
            except Exception as e:
                reason = f"scene_plan[{index}] slot 파싱 실패: {e}"
                if log_errors:
                    print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}: item={item!r}")
                    traceback.print_exc()
                return None, reason
        if slot not in candidate_set:
            reason = f"scene_plan[{index}] 후보 밖 slot: slot={slot}, candidates={candidates}"
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason
        characters = item.get("characters") or []
        if not isinstance(characters, list):
            characters = [characters]
        normalized_characters = []
        planned_outfits = {}
        for character_index, value in enumerate(characters, start=1):
            if isinstance(value, dict):
                name = str(value.get("name") or "").strip()
                raw_outfit = value.get("outfit_state")
                if raw_outfit is not None:
                    if not isinstance(raw_outfit, dict):
                        reason = (
                            f"scene_plan[{index}].characters[{character_index}] "
                            f"outfit_state가 object가 아님"
                        )
                        if log_errors:
                            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}: item={value!r}")
                        return None, reason
                    body_state = str(raw_outfit.get("body_state") or "").strip().lower()
                    if body_state not in _OUTFIT_BODY_STATES:
                        reason = (
                            f"scene_plan[{index}].characters[{character_index}] "
                            f"body_state 오류: {body_state!r}"
                        )
                        if log_errors:
                            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
                        return None, reason
                    if not isinstance(raw_outfit.get("worn", []), list) or not isinstance(
                        raw_outfit.get("removed", []),
                        list,
                    ):
                        reason = (
                            f"scene_plan[{index}].characters[{character_index}] "
                            "worn/removed가 list가 아님"
                        )
                        if log_errors:
                            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
                        return None, reason
                if name and raw_outfit is not None:
                    planned_outfits[name] = _normalize_outfit_state(raw_outfit)
            else:
                name = str(value or "").strip()
            if name and name.casefold() not in {
                existing.casefold() for existing in normalized_characters
            }:
                normalized_characters.append(name)
        scene_brief = str(item.get("scene_brief") or "").strip()
        if not scene_brief:
            reason = (
                f"scene_plan[{index}] scene_brief가 비어 있음: "
                f"brief={scene_brief!r}"
            )
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason
        if not normalized_characters and log_errors:
            print(
                f"[ILLUST_CONTEXT:CALL2_PLAN] 이름 있는 추적 캐릭터가 없는 장면 수용: "
                f"plan={index}, anchor={anchor_segment!r}, brief={scene_brief!r}"
            )
        scene_plan.append({
            "plan_id": str(item.get("plan_id") or f"S{index:03d}").strip() or f"S{index:03d}",
            "slot": slot,
            "anchor_segment": anchor_segment,
            "source_segments": source_segments,
            "characters": normalized_characters,
            "planned_outfits": planned_outfits,
            "scene_brief": scene_brief,
        })

    if not scene_plan:
        reason = "CALL2-PLAN이 장면을 선택하지 않음"
        if log_errors:
            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
        return None, reason
    scene_plan, dropped_collision_count = _repair_call2_plan_slot_collisions(
        scene_plan,
        candidates,
        segment_slot_map=segment_slot_map,
        log_errors=log_errors,
    )
    scene_plan.sort(key=lambda item: candidates.index(item["slot"]))
    for index, item in enumerate(scene_plan, start=1):
        item["plan_id"] = f"S{index:03d}"

    scene_mode = str(toggles.get("scene_mode"))
    maximum = min(int(toggles["output_count_max"]), len(candidates))
    if len(scene_plan) > maximum:
        # 모델이 max 초과 반환 → 유사도 maximin으로 maximum개만 남기고 나머지는 자름.
        # 정상(max 이하)이면 PLAN 결과를 그대로 사용한다.
        scene_plan = _select_diverse_scenes(
            scene_plan,
            maximum,
            candidates,
            log_errors=log_errors,
        )
        scene_plan.sort(key=lambda item: candidates.index(item["slot"]))
        for index, item in enumerate(scene_plan, start=1):
            item["plan_id"] = f"S{index:03d}"

    if scene_mode != "auto":
        minimum = min(int(toggles["output_count_min"]), len(candidates))
        repaired_minimum = max(0, minimum - dropped_collision_count)
        if len(scene_plan) < repaired_minimum:
            reason = (
                f"CALL2-PLAN 장면 수 범위 위반(과소): count={len(scene_plan)}, "
                f"required={repaired_minimum}..{maximum}"
            )
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason

    keyvis_descriptor = None
    keyvis_plan = None
    raw_keyvis = raw.get("keyvis")
    raw_keyvis_plan = raw.get("keyvis_plan")
    if toggles.get("key_visual"):
        if isinstance(raw_keyvis, dict):
            complete_keyvis = _descriptor(raw_keyvis, "keyvis", -1)
            if (
                complete_keyvis.get("camera")
                and complete_keyvis.get("scene")
                and complete_keyvis.get("characters")
            ):
                # Backward compatibility for saved/custom planner prompts that
                # still emit the former fully-expanded key visual.
                keyvis_descriptor = complete_keyvis
            elif raw_keyvis_plan is None:
                raw_keyvis_plan = raw_keyvis
        if keyvis_descriptor is None and not isinstance(raw_keyvis_plan, dict):
            reason = "CALL2-PLAN keyvis_plan이 없거나 object가 아님"
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason
        if keyvis_descriptor is None:
            raw_characters = raw_keyvis_plan.get("characters") or []
            if not isinstance(raw_characters, list):
                raw_characters = [raw_characters]
            keyvis_characters = []
            for value in raw_characters:
                name = str(value.get("name") if isinstance(value, dict) else value or "").strip()
                if name and name.casefold() not in {
                    existing.casefold() for existing in keyvis_characters
                }:
                    keyvis_characters.append(name)
            keyvis_brief = str(
                raw_keyvis_plan.get("scene_brief")
                or raw_keyvis_plan.get("brief")
                or ""
            ).strip()
            if not keyvis_characters or not keyvis_brief:
                reason = (
                    "CALL2-PLAN keyvis_plan characters 또는 scene_brief가 비어 있음: "
                    f"characters={keyvis_characters}, brief={keyvis_brief!r}"
                )
                if log_errors:
                    print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
                return None, reason
            keyvis_plan = {
                "characters": keyvis_characters,
                "scene_brief": keyvis_brief,
            }
    elif isinstance(raw_keyvis, dict) or isinstance(raw_keyvis_plan, dict):
        print("[ILLUST_CONTEXT:CALL2_PLAN] Key Visual 비활성인데 keyvis 계획이 반환되어 폐기")

    return {
        "mode": "plan",
        "scene_plan": scene_plan,
        "keyvis_descriptor": keyvis_descriptor,
        "keyvis_plan": keyvis_plan,
        "descriptors": [],
    }, ""


def _match_call2_detail_characters(
    item: dict,
    expected_names: list[str],
    location: str,
) -> tuple[dict[str, dict], str]:
    """Validate PLAN character identity and repair one unambiguous missing name."""
    expected_by_name = {
        str(name).strip().casefold(): str(name).strip()
        for name in expected_names
        if str(name).strip()
    }
    characters = [
        character
        for character in item.get("characters") or []
        if isinstance(character, dict)
    ]
    actual_by_name = {
        str(character.get("name") or "").strip().casefold(): character
        for character in characters
        if str(character.get("name") or "").strip()
    }
    if set(actual_by_name) == set(expected_by_name):
        return actual_by_name, ""

    if (
        len(expected_by_name) == 1
        and len(characters) == 1
        and not str(characters[0].get("name") or "").strip()
    ):
        expected_name = next(iter(expected_by_name.values()))
        characters[0]["name"] = expected_name
        print(
            f"[ILLUST_CONTEXT:CALL2_DETAIL] PLAN으로 단일 누락 캐릭터 이름 복구: "
            f"{location}, name={expected_name}"
        )
        return {expected_name.casefold(): characters[0]}, ""

    return {}, (
        f"CALL2-DETAIL PLAN 캐릭터 불일치: {location}, "
        f"expected={list(expected_by_name.values())}, "
        f"actual={[character.get('name') for character in actual_by_name.values()]}"
    )


def _parse_call2_detail_output(
    text: str,
    toggles: dict,
    assigned_slots: list[int],
    assigned_plan_ids: list[str],
    source: str,
    assigned_wardrobes_by_slot: dict[int, dict[str, dict]] | None = None,
    assigned_keyvis_plan: dict | None = None,
    assigned_characters_by_slot: dict[int, list[str]] | None = None,
    assigned_scene_context_by_slot: dict[int, dict] | None = None,
) -> tuple[list[dict], str]:
    local_toggles = deepcopy(toggles)
    local_toggles.update({
        "key_visual": bool(assigned_keyvis_plan),
        "scene_mode": "manual",
        "output_count_min": len(assigned_slots),
        "output_count_max": len(assigned_slots),
    })
    parsed_descriptors = parse_toon_plan(text, local_toggles, source)
    descriptors = [
        item
        for item in parsed_descriptors
        if str(item.get("kind") or "") == "scene"
    ]
    keyvis_descriptors = [
        item
        for item in parsed_descriptors
        if str(item.get("kind") or "") == "keyvis"
    ]
    if assigned_keyvis_plan:
        if len(keyvis_descriptors) != 1:
            return [], (
                f"CALL2-DETAIL keyvis 수 불일치: expected=1, actual={len(keyvis_descriptors)}"
            )
        keyvis = keyvis_descriptors[0]
        if (
            not str(keyvis.get("camera") or "").strip()
            or not str(keyvis.get("scene") or "").strip()
            or not (keyvis.get("characters") or [])
        ):
            return [], f"CALL2-DETAIL keyvis 필수 camera/scene/characters가 비어 있음: item={keyvis!r}"
        _actual_keyvis, keyvis_reason = _match_call2_detail_characters(
            keyvis,
            list(assigned_keyvis_plan.get("characters") or []),
            "keyvis",
        )
        if keyvis_reason:
            return [], keyvis_reason
    else:
        keyvis_descriptors = []
    actual_slots = []
    for item in descriptors:
        if (
            not str(item.get("camera") or "").strip()
            or not str(item.get("scene") or "").strip()
        ):
            return [], f"CALL2-DETAIL 필수 camera/scene이 비어 있음: item={item!r}"
        try:
            slot = int(item.get("slot"))
        except Exception:
            return [], f"CALL2-DETAIL slot 파싱 실패: item={item!r}"
        expected_characters = None
        if assigned_characters_by_slot is not None:
            expected_characters = [
                str(name or "").strip()
                for name in assigned_characters_by_slot.get(slot, [])
                if str(name or "").strip()
            ]
        if expected_characters and not (item.get("characters") or []):
            reason = (
                f"CALL2-DETAIL 이름 있는 PLAN 캐릭터가 누락됨: "
                f"slot={slot}, expected={expected_characters}"
            )
            print(
                f"[ILLUST_CONTEXT:CALL2_DETAIL] {reason}: "
                f"source={source}, item={item!r}"
            )
            return [], reason
        actual_slots.append(slot)
    if len(actual_slots) != len(assigned_slots):
        return [], (
            f"CALL2-DETAIL 장면 수 불일치: assigned={assigned_slots}, actual={actual_slots}"
        )
    if len(set(actual_slots)) != len(actual_slots) or set(actual_slots) != set(assigned_slots):
        return [], (
            f"CALL2-DETAIL slot 불일치: assigned={assigned_slots}, actual={actual_slots}"
        )
    if len(assigned_plan_ids) != len(assigned_slots):
        return [], (
            f"CALL2-DETAIL 서버 PLAN 매핑 길이 불일치: "
            f"slots={assigned_slots}, plan_ids={assigned_plan_ids}"
        )
    plan_id_by_slot = dict(zip(assigned_slots, assigned_plan_ids))
    by_slot = {int(item["slot"]): item for item in descriptors}
    discarded_slots: set[int] = set()
    for slot, item in by_slot.items():
        # plan_id는 모델 출력 계약이 아니라 서버 내부 식별자다. 검증을 통과한
        # 고유 slot을 신뢰하고 전역 PLAN에서 확정한 값을 항상 주입한다.
        item["plan_id"] = plan_id_by_slot[slot]
        assigned_scene_context = (
            (assigned_scene_context_by_slot or {}).get(slot) or {}
        )
        item["scene_brief"] = str(
            assigned_scene_context.get("scene_brief") or ""
        ).strip()
        item["continuity_note"] = str(
            assigned_scene_context.get("continuity_note") or ""
        ).strip()
        item["visual_base_snapshot"] = deepcopy(
            assigned_scene_context.get("visual_base_snapshot") or {}
        )
        expected_wardrobes = (assigned_wardrobes_by_slot or {}).get(slot) or {}
        if expected_wardrobes:
            expected_by_name = {
                str(name).strip().casefold(): (str(name).strip(), _normalize_outfit_state(outfit))
                for name, outfit in expected_wardrobes.items()
                if str(name).strip()
            }
            actual_by_name, character_reason = _match_call2_detail_characters(
                item,
                [value[0] for value in expected_by_name.values()],
                f"slot={slot}",
            )
            if character_reason:
                discarded_slots.add(slot)
                print(
                    f"[ILLUST_CONTEXT:CALL2_DETAIL] PLAN 캐릭터 불일치로 "
                    f"해당 슬롯 폐기: source={source}, slot={slot}, "
                    f"reason={character_reason}"
                )
                continue
            for folded, (expected_name, expected_outfit) in expected_by_name.items():
                character = actual_by_name[folded]
                actual_outfit = character.get("outfit_state")
                conflict = _outfit_contract_conflict(expected_outfit, actual_outfit)
                if conflict:
                    print(
                        f"[ILLUST_CONTEXT:CALL2_DETAIL] 참고 복장과 다른 contextual 후보를 "
                        f"권위 감사로 전달: slot={slot}, character={expected_name}, "
                        f"reference={expected_outfit}, actual={_normalize_outfit_state(actual_outfit)}, "
                        f"difference={conflict}"
                    )
                normalized_actual = _normalize_outfit_state(actual_outfit)
                if not _outfit_states_equal(normalized_actual, expected_outfit):
                    print(
                        f"[ILLUST_CONTEXT:CALL2_DETAIL] DETAIL contextual 복장 보존: "
                        f"slot={slot}, character={expected_name}, "
                        f"reference={expected_outfit}, actual={normalized_actual}"
                    )
                # default_outfit/추적 스냅샷은 참고·연속성 기준이다. DETAIL이 문맥에
                # 맞는 다른 복장을 구성했다면 권위 감사가 의미를 검증할 수 있도록
                # 알려진 실제 출력을 보존한다. 여기서 스냅샷으로 덮으면 감사 전에
                # 새 복장 정보가 유실된다.
                character["outfit_state"] = (
                    deepcopy(normalized_actual)
                    if _outfit_state_is_known(normalized_actual) or (
                        expected_outfit["body_state"] == "unknown"
                        and not _outfit_state_is_known(expected_outfit)
                    )
                    else deepcopy(expected_outfit)
                )
    return keyvis_descriptors + [
        by_slot[slot]
        for slot in assigned_slots
        if slot not in discarded_slots
    ], ""


def _parse_call2_detail_partial(
    text: str,
    toggles: dict,
    assigned_slots: list[int],
    assigned_plan_ids: list[str],
    source: str,
    assigned_wardrobes_by_slot: dict[int, dict[str, dict]] | None = None,
    assigned_characters_by_slot: dict[int, list[str]] | None = None,
    assigned_scene_context_by_slot: dict[int, dict] | None = None,
) -> tuple[dict[int, dict], list[int], list[int], str]:
    """CALL2-DETAIL 부분 허용 파서.

    _parse_call2_detail_output 의 strict 검증과 달리 slot 불일치/장면 수 불일치가
    나도 전체를 폐기하지 않는다. 할당 슬롯 집합에 정확히 들어오며 per-scene 필수
    검증(camera/scene/slot/이름있는 PLAN 캐릭터)을 통과한 장면만 보존하고, 그
    슬롯에 plan_id 주입·복장 확정까지 마친다. CALL2-DETAIL 의 ①전부/②실패분만
    교대 루프가 매 단계에서 "이미 확보한 좋은 슬롯"을 그대로 가져가도록 쓴다.

    반환: (kept_by_slot, missing_slots, char_discarded_slots, hard_reason)
      - kept_by_slot: {slot: descriptor} (plan_id/복장 확정 완료)
      - missing_slots: assigned 중 kept/char_discarded 모두 아닌, 아직 채워야 할 슬롯
      - char_discarded_slots: 이번 호출에서 PLAN 캐릭터 불일치로 폐기된 슬롯(재시도 무의미)
      - hard_reason: 장면이 아예 파싱되지 않은 등 부분 보존이 불가능한 사유(빈 문자열=정상)
    """
    local_toggles = deepcopy(toggles)
    local_toggles.update({
        "key_visual": False,
        "scene_mode": "manual",
        "output_count_min": len(assigned_slots),
        "output_count_max": len(assigned_slots),
    })
    parsed_descriptors = parse_toon_plan(text, local_toggles, source)
    scene_descriptors = [
        item
        for item in parsed_descriptors
        if str(item.get("kind") or "") == "scene"
    ]
    if not scene_descriptors:
        return {}, list(assigned_slots), [], f"CALL2-DETAIL 파싱된 장면 없음: source={source}"

    assigned_set = set(assigned_slots)
    plan_id_by_slot = dict(zip(assigned_slots, assigned_plan_ids))
    kept_by_slot: dict[int, dict] = {}
    for item in scene_descriptors:
        if (
            not str(item.get("camera") or "").strip()
            or not str(item.get("scene") or "").strip()
        ):
            print(
                f"[ILLUST_CONTEXT:CALL2_DETAIL_PARTIAL] 필수 camera/scene 비어 스킵: "
                f"source={source}, item={item!r}"
            )
            continue
        try:
            slot = int(item.get("slot"))
        except Exception:
            print(
                f"[ILLUST_CONTEXT:CALL2_DETAIL_PARTIAL] slot 파싱 실패 스킵: "
                f"source={source}, item={item!r}"
            )
            continue
        if slot not in assigned_set:
            print(
                f"[ILLUST_CONTEXT:CALL2_DETAIL_PARTIAL] 할당 밖 slot 스킵: "
                f"source={source}, slot={slot}, assigned={sorted(assigned_set)}"
            )
            continue
        if slot in kept_by_slot:
            # 중복 slot: 이미 확보한 첫 장면을 유지하고 나머지는 무시한다.
            continue
        expected_characters = None
        if assigned_characters_by_slot is not None:
            expected_characters = [
                str(name or "").strip()
                for name in assigned_characters_by_slot.get(slot, [])
                if str(name or "").strip()
            ]
        if expected_characters and not (item.get("characters") or []):
            print(
                f"[ILLUST_CONTEXT:CALL2_DETAIL_PARTIAL] 이름 있는 PLAN 캐릭터 누락 스킵: "
                f"source={source}, slot={slot}, expected={expected_characters}"
            )
            continue
        kept_by_slot[slot] = item

    # 보존된 슬롯에 대해 plan_id 주입 + 복장 확정(strict 경로와 동일 규칙).
    discarded_slots: set[int] = set()
    for slot, item in kept_by_slot.items():
        item["plan_id"] = plan_id_by_slot.get(slot, "")
        assigned_scene_context = (
            (assigned_scene_context_by_slot or {}).get(slot) or {}
        )
        item["scene_brief"] = str(
            assigned_scene_context.get("scene_brief") or ""
        ).strip()
        item["continuity_note"] = str(
            assigned_scene_context.get("continuity_note") or ""
        ).strip()
        item["visual_base_snapshot"] = deepcopy(
            assigned_scene_context.get("visual_base_snapshot") or {}
        )
        expected_wardrobes = (assigned_wardrobes_by_slot or {}).get(slot) or {}
        if not expected_wardrobes:
            continue
        expected_by_name = {
            str(name).strip().casefold(): (str(name).strip(), _normalize_outfit_state(outfit))
            for name, outfit in expected_wardrobes.items()
            if str(name).strip()
        }
        actual_by_name, character_reason = _match_call2_detail_characters(
            item,
            [value[0] for value in expected_by_name.values()],
            f"slot={slot}",
        )
        if character_reason:
            discarded_slots.add(slot)
            print(
                f"[ILLUST_CONTEXT:CALL2_DETAIL_PARTIAL] PLAN 캐릭터 불일치로 스킵: "
                f"source={source}, slot={slot}, reason={character_reason}"
            )
            continue
        for folded, (expected_name, expected_outfit) in expected_by_name.items():
            character = actual_by_name[folded]
            actual_outfit = character.get("outfit_state")
            conflict = _outfit_contract_conflict(expected_outfit, actual_outfit)
            if conflict:
                print(
                    f"[ILLUST_CONTEXT:CALL2_DETAIL_PARTIAL] 참고 복장과 다른 contextual "
                    f"후보를 권위 감사로 전달: slot={slot}, character={expected_name}, "
                    f"reference={expected_outfit}, actual={_normalize_outfit_state(actual_outfit)}, "
                    f"difference={conflict}"
                )
            normalized_actual = _normalize_outfit_state(actual_outfit)
            if not _outfit_states_equal(normalized_actual, expected_outfit):
                print(
                    f"[ILLUST_CONTEXT:CALL2_DETAIL_PARTIAL] DETAIL contextual 복장 보존: "
                    f"slot={slot}, character={expected_name}, "
                    f"reference={expected_outfit}, actual={normalized_actual}"
                )
            # strict 경로와 동일하게 알려진 contextual outfit resolution을 audit
            # 전까지 보존한다. 기본 복장은 참고값이므로 여기서 강제 복원하지 않는다.
            character["outfit_state"] = (
                deepcopy(normalized_actual)
                if _outfit_state_is_known(normalized_actual) or (
                    expected_outfit["body_state"] == "unknown"
                    and not _outfit_state_is_known(expected_outfit)
                )
                else deepcopy(expected_outfit)
            )

    for slot in discarded_slots:
        kept_by_slot.pop(slot, None)
    missing_slots = [
        slot
        for slot in assigned_slots
        if slot not in kept_by_slot and slot not in discarded_slots
    ]
    return kept_by_slot, missing_slots, sorted(discarded_slots), ""


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
            data["scenes"].append(raw)
    body = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    return f"<lb-xnai>\n{body}\n</lb-xnai>"


def _authority_output_tag(tag: str) -> str:
    """Normalize only the schema-required named-character count label."""
    value = str(tag or "").strip()
    folded = value.casefold()
    if folded == "1girl":
        return "girl"
    if folded == "1boy":
        return "boy"
    return value


def _authority_tag_identity(tag: str) -> str:
    """Return a structural comparison key without semantic keyword matching."""
    value = _authority_output_tag(tag).strip()
    weighted = re.fullmatch(
        r"\(\s*(?P<tag>.+?)\s*:\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*\)",
        value,
    )
    if weighted:
        value = weighted.group("tag").strip()
    explicit_weight = re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)::(?P<tag>.+?)::",
        value,
    )
    if explicit_weight:
        value = explicit_weight.group("tag").strip()
    return value.casefold()


def _authority_values_for_name(values: dict, name: str):
    folded = str(name or "").strip().casefold()
    for candidate_name, value in (values or {}).items():
        if str(candidate_name or "").strip().casefold() == folded:
            return value
    return None


def _descriptor_authority_tags(
    descriptor: dict,
    name: str,
    fixed_appearance: dict[str, str],
    default_outfits: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    base = _authority_values_for_name(
        descriptor.get("visual_base_snapshot") or {},
        name,
    )
    if isinstance(base, dict):
        fixed_tags = [
            _authority_output_tag(tag)
            for tag in visual_tag_values(base.get("appearance") or [])
            if _authority_output_tag(tag)
        ]
        default_tags = [
            str(tag).strip()
            for tag in visual_tag_values(base.get("outfit") or [])
            if str(tag).strip()
        ]
        return fixed_tags, default_tags
    fixed_raw = _authority_values_for_name(fixed_appearance, name)
    default_raw = _authority_values_for_name(default_outfits, name)
    fixed_tags = [
        _authority_output_tag(tag)
        for tag in _split_top_level_authority_tags(str(fixed_raw or ""))
        if _authority_output_tag(tag)
    ]
    default_tags = [
        str(tag).strip() for tag in (default_raw or []) if str(tag).strip()
    ]
    return fixed_tags, default_tags


def _call2_authority_audit_entries(
    descriptors: list[dict],
    fixed_appearance: dict[str, str],
    default_outfits: dict[str, list[str]],
    hairstyle_history: dict[str, list[dict]] | None = None,
) -> tuple[list[dict], dict[int, tuple[str, int, str]]]:
    entries: list[dict] = []
    entry_keys: dict[int, tuple[str, int, str]] = {}
    next_id = 1
    for descriptor in descriptors or []:
        kind = str(descriptor.get("kind") or "scene")
        slot = int(descriptor.get("slot") or 0)
        scene_context = {
            "anchor_before": str(descriptor.get("anchor_before") or ""),
            "anchor_after": str(descriptor.get("anchor_after") or ""),
            "scene_brief": str(descriptor.get("scene_brief") or ""),
            "continuity_note": str(descriptor.get("continuity_note") or ""),
            "camera": str(descriptor.get("camera") or ""),
            "scene": str(descriptor.get("scene") or ""),
            "supplement": str(descriptor.get("supplement") or ""),
        }
        for character in descriptor.get("characters") or []:
            name = str(character.get("name") or "").strip()
            if not name:
                continue
            fixed_tags, default_tags = _descriptor_authority_tags(
                descriptor,
                name,
                fixed_appearance,
                default_outfits,
            )
            generated_outfit_state = _normalize_outfit_state(
                character.get("outfit_state")
            )
            generated_tags = _split_top_level_authority_tags(
                str(character.get("positive") or "")
            )
            authority_ids = {
                _authority_tag_identity(tag) for tag in fixed_tags + default_tags
            }
            if not authority_ids:
                print(
                    f"[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] 권위 태그는 없지만 시각 완성도 "
                    f"감사는 유지: kind={kind}, slot={slot}, character={name}"
                )
            entry_id = next_id
            next_id += 1
            entry_keys[entry_id] = (kind, slot, name.casefold())
            entries.append({
                "id": entry_id,
                "kind": kind,
                "slot": slot,
                "character": name,
                "scene_context": scene_context,
                "fixed_appearance": fixed_tags,
                "default_outfit": default_tags,
                "generated_positive": generated_tags,
                "generated_outfit_state": generated_outfit_state,
                "hairstyle_history": (hairstyle_history or {}).get(
                    name.casefold(), []
                ),
            })
    return entries, entry_keys


def _parse_call2_authority_audit_output(
    text: str,
    entries: list[dict],
    entry_keys: dict[int, tuple[str, int, str]],
) -> tuple[dict[tuple[str, int, str], dict], str]:
    def reject(reason: str):
        print(f"[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] 응답 거부: reason={reason}")
        return {}, reason

    raw = _json_object_from_text(text)
    if raw is None:
        return reject("CALL2-AUTHORITY-AUDIT JSON object 파싱 실패")
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list):
        return reject("CALL2-AUTHORITY-AUDIT entries가 list가 아님")
    candidates = {int(item["id"]): item for item in entries}
    observed_ids: set[int] = set()
    decisions: dict[tuple[str, int, str], dict] = {}
    for index, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            return reject(
                f"CALL2-AUTHORITY-AUDIT entries[{index}]가 object가 아님"
            )
        try:
            entry_id = int(raw_entry.get("id"))
        except Exception as e:
            print(
                f"[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] id 파싱 실패: "
                f"index={index}, value={raw_entry.get('id')!r}, error={e}"
            )
            traceback.print_exc()
            return reject(
                f"CALL2-AUTHORITY-AUDIT entries[{index}].id 파싱 실패"
            )
        if entry_id not in candidates or entry_id in observed_ids:
            return reject(
                f"CALL2-AUTHORITY-AUDIT id 불일치/중복: id={entry_id}, "
                f"expected={sorted(candidates)}"
            )
        observed_ids.add(entry_id)
        candidate = candidates[entry_id]
        authority_by_id = {
            _authority_tag_identity(tag): tag
            for tag in candidate["fixed_appearance"] + candidate["default_outfit"]
            if _authority_tag_identity(tag)
        }
        generated_by_id = {
            _authority_tag_identity(tag): tag
            for tag in candidate["generated_positive"]
            if _authority_tag_identity(tag)
        }

        normalized_fields: dict[str, object] = {}
        for field, allowed in (
            ("authority_exceptions", authority_by_id),
            ("forbidden_additions", generated_by_id),
            ("conflicts", generated_by_id),
        ):
            values = raw_entry.get(field) or []
            if not isinstance(values, list):
                return reject(
                    f"CALL2-AUTHORITY-AUDIT entries[{index}].{field}가 list가 아님"
                )
            normalized: list[str] = []
            seen: set[str] = set()
            dropped: list[str] = []
            for value in values:
                identity = _authority_tag_identity(value)
                if not identity or identity not in allowed:
                    # apply_call2_authority_base 도 후보 밖 값은 그 값만 스킵하므로
                    # 파싱 단계에서 응답 전체를 거부하면 한 값의 실수가 모든 엔트리의
                    # audit 결정을 날리게 된다. 그 값만 로그 남기고 버린다.
                    dropped.append(str(value))
                    continue
                if identity in seen:
                    continue
                seen.add(identity)
                normalized.append(allowed[identity])
            if dropped:
                print(
                    f"[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] 후보 밖 {field} "
                    f"스킵(그 값만, 응답 전체는 유지): id={entry_id}, "
                    f"dropped={dropped}"
                )
            normalized_fields[field] = normalized
        for field in ("required_additions", "scene_additions"):
            values = raw_entry.get(field) or []
            if not isinstance(values, list):
                return reject(
                    f"CALL2-AUTHORITY-AUDIT entries[{index}].{field}가 list가 아님"
                )
            normalized: list[str] = []
            seen: set[str] = set()
            for value in values:
                addition = re.sub(r"\s+", " ", str(value or "")).strip(" ,")
                identity = _authority_tag_identity(addition)
                if not identity or identity in seen:
                    continue
                seen.add(identity)
                normalized.append(addition)
            if len(normalized) > 16:
                print(
                    f"[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] {field} 16개 초과분 스킵: "
                    f"id={entry_id}, dropped={normalized[16:]}"
                )
                normalized = normalized[:16]
            normalized_fields[field] = normalized
        camera_replacement = re.sub(
            r"\s+", " ", str(raw_entry.get("camera_replacement") or "")
        ).strip(" ,")
        if len(camera_replacement) > 300:
            print(
                "[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] camera_replacement 길이 초과로 "
                f"300자까지 보존: id={entry_id}, length={len(camera_replacement)}"
            )
            camera_replacement = camera_replacement[:300].rstrip(" ,")
        normalized_fields["camera_replacement"] = camera_replacement
        decisions[entry_keys[entry_id]] = normalized_fields
    if observed_ids != set(candidates):
        return reject(
            f"CALL2-AUTHORITY-AUDIT 응답 id 누락: expected={sorted(candidates)}, "
            f"actual={sorted(observed_ids)}"
        )
    return decisions, ""


async def _run_call2_authority_audit(
    descriptors: list[dict],
    fixed_appearance: dict[str, str],
    default_outfits: dict[str, list[str]],
    current_context: str,
    stream_notify,
    hairstyle_history: dict[str, list[dict]] | None = None,
    toggles: dict | None = None,
) -> tuple[dict[tuple[str, int, str], dict], str, str]:
    entries, entry_keys = _call2_authority_audit_entries(
        descriptors,
        fixed_appearance,
        default_outfits,
        hairstyle_history,
    )
    if not entries:
        print(
            "[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] 감사 가능한 권위 태그가 없어 "
            "semantic audit LLM 호출 생략"
        )
        return {}, "", "not_needed"

    if (toggles or {}).get("minimal_background_description", True):
        audit_background_instruction = (
            "Environment is a last-priority completeness concern: keep an existing minimal "
            "location cue or `simple background` as-is, never add a second background "
            "description, and when no clear or story-important background exists add only "
            "`simple background`. Never request decorative props, weather, time, or elaborate "
            "lighting merely for background detail. "
        )
    else:
        audit_background_instruction = (
            "Environment is a normal visual-completeness concern: preserve and, when missing, "
            "request enough concrete story-supported location, time, weather, lighting, scenery, "
            "furniture, and prominent prop detail to make the setting readable. Multiple "
            "complementary environment additions are allowed when they express distinct visible "
            "facts. Never invent unsupported decoration, and request `simple background` only "
            "when the narrative provides no meaningful environment. "
        )

    system_prompt = (
        "You are CALL2-AUTHORITY-AUDIT. Perform the existing authority audit and a final visual-"
        "completeness repair in this same call. Read CURRENT CONTEXT and each entry's scene_context by "
        "meaning and chronology; never use keyword matching. The complete fixed_appearance is a "
        "mandatory identity base. default_outfit is only a fallback wardrobe reference, not fixed "
        "identity and not a mandatory outfit. A short historical description is not automatically a "
        "complete replacement outfit, but the full scene may make a different coherent outfit "
        "appropriate even without an explicit sentence listing garments. Judge that from the whole "
        "narrative, role, activity, occasion, setting, and continuity by meaning and common sense, never "
        "by matching individual words. For each id, return authority_exceptions for exact supplied "
        "fixed-appearance tags only when the assigned scene explicitly and temporarily replaces them. "
        "For default-outfit tags, also return authority_exceptions when the tracked wardrobe or the "
        "scene's coherent contextual outfit replaces the fallback as a set; explicit removal wording is "
        "not required for such a contextual wardrobe replacement. When replacement is warranted, except "
        "every default garment or accessory that should not carry into the new outfit, while preserving "
        "an item only when it logically remains. Do not grant an authority exception for an accessory merely because it is "
        "physically associated with one explicitly removed garment; unless the whole outfit is contextually "
        "replaced, the scene must establish removal of that accessory itself. generated_outfit_state is an untrusted proposal, but it "
        "is evidence to judge together with generated_positive and the full context rather than being "
        "overwritten merely for differing from default_outfit. Return forbidden_additions for exact "
        "generated_positive tags that invent a persistent identity, body, hair, face, eye, skin, or "
        "species trait, or a wardrobe detail that is incoherent with the assigned scene. Do not flag a "
        "coherent scene-appropriate garment merely because it is absent from default_outfit. An entry may include `hairstyle_history`: a "
        "chronological list of semantic hairstyle-arrangement transitions. An active transition in "
        "that history may temporarily authorize, for this scene only, replacement of the directly "
        "conflicting fixed hairstyle-arrangement tag — list the suppressed fixed arrangement tag "
        "(e.g. `ponytail`) in `authority_exceptions` and do not flag the active arrangement tag "
        "(e.g. `twintails`) as a forbidden addition, because it is a temporary hairstyle state, not "
        "a new persistent trait. Authorize such overrides ONLY for hairstyle-arrangement tags "
        "(ponytail, twintails, braid, hair bun, two side up, side ponytail, hair down); hair color, "
        "hair length, texture, bangs, sidelocks, ahoge, eyes, body, species, and any other fixed "
        "trait remain mandatory and must never be excepted because of hairstyle history. The "
        "override ends at `reset_default` or a later conflicting hairstyle event. Do not classify pose, action, expression, gaze, "
        "or temporary scene state as a forbidden addition. Return conflicts even when every base "
        "tag is already present, but only for exact generated_positive tags that directly "
        "contradict fixed appearance and are not supported by that assigned scene. When a contextual "
        "outfit replaces the fallback, also return as conflicts any exact generated default-outfit "
        "tags that were mistakenly retained and do not belong with the resolved new outfit. Do not report camera "
        "invisibility, brevity, or ordinary scene/action/expression tags as authority conflicts. "
        "Then silently cross-check whether the generated camera, scene, supplement, character tags, "
        "scene_brief, and natural-language continuity_note form one physically possible image. For "
        "explicit content, judge the whole scene-specific visual bundle: participant roles and relative "
        "positions, exact action/contact and touched anatomy, visible exposure, displaced or removed "
        "clothing, pose, expression, and whether the camera actually includes story-essential evidence. "
        "Do not invent a sexual act, body detail, intensity, or exposure unsupported by the story. "
        "Contextual wardrobe design is allowed as described above, but it must remain coherent with the "
        "scene and must not rewrite identity. Do not euphemize or omit an explicit fact that the assigned image is meant to show. "
        "Put only missing character-level visible facts in required_additions, and only missing scene-level "
        "facts such as overall interaction or `nsfw` in scene_additions. "
        + audit_background_instruction
        + "Each list item is one "
        "concise tag or natural visual phrase; it need not be validated against an external tag dictionary. "
        "Use camera_replacement only when the present framing or perspective cannot show an essential fact; "
        "then return one complete coherent replacement camera string, otherwise return an empty string. "
        "Keep all repairs minimal and mutually compatible. Copy authority/conflict candidate strings exactly. "
        "Return compact JSON only: "
        '{"entries":[{"id":1,"authority_exceptions":[],"forbidden_additions":[],"conflicts":[],"required_additions":[],"scene_additions":[],"camera_replacement":""}]}.'
    )
    messages = [{"role": "system", "content": system_prompt}, {
        "role": "user",
        "content": (
            "# CURRENT CONTEXT\n"
            + str(current_context or "")
            + "\n\n# AUDIT ENTRIES\n"
            + json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
        ),
    }]

    def validate(result):
        _decisions, reason = _parse_call2_authority_audit_output(
            result,
            entries,
            entry_keys,
        )
        return bool(_decisions), reason or "CALL2-AUTHORITY-AUDIT 검증 실패"

    try:
        raw_output = await _call_pipeline_llm(
            "CALL2-AUTHORITY-AUDIT",
            _normalize_messages(messages),
            stream_notify,
            result_validator=validate,
            json_mode=True,
        )
        decisions, reason = _parse_call2_authority_audit_output(
            raw_output,
            entries,
            entry_keys,
        )
        if reason:
            print(
                f"[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] 최종 응답 검증 실패, "
                f"기본 세트 전부 복원하는 degraded 모드 사용: reason={reason}, "
                f"raw={raw_output[:1000]!r}"
            )
            return {}, raw_output, "degraded"
        print(
            f"[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] semantic audit 완료: "
            f"entries={len(entries)}, decisions={len(decisions)}"
        )
        return decisions, raw_output, "ok"
    except asyncio.CancelledError:
        print("[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] 상위 작업 취소")
        raise
    except Exception as e:
        print(
            f"[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] LLM 실패, 기본 세트 전부 "
            f"복원하는 degraded 모드 사용: error={e}"
        )
        traceback.print_exc()
        return {}, "", "degraded"


def apply_call2_authority_base(
    descriptors: list[dict],
    fixed_appearance: dict[str, str],
    default_outfits: dict[str, list[str]],
    semantic_decisions: dict[tuple[str, int, str], dict] | None = None,
    semantic_status: str = "not_run",
) -> list[dict]:
    """Restore fixed appearance and validate the default-outfit fallback.

    Only the separate semantic audit may approve exact authority exceptions.
    DETAIL/PLAN fields remain untrusted proposals, but an audited contextual
    outfit may replace the default reference as a set. Other omissions are
    deterministic server repairs. This function compares only server-provided
    tag-set membership; it does not classify narrative words.
    """
    audits: list[dict] = []
    for descriptor in descriptors or []:
        kind = str(descriptor.get("kind") or "scene")
        slot = int(descriptor.get("slot") or 0)
        descriptor_decisions = [
            (semantic_decisions or {}).get(
                (
                    kind,
                    slot,
                    str(character.get("name") or "").strip().casefold(),
                ),
                {},
            )
            for character in descriptor.get("characters") or []
            if str(character.get("name") or "").strip()
        ]
        camera_replacements: list[str] = []
        for decision in descriptor_decisions:
            replacement = str(decision.get("camera_replacement") or "").strip()
            if replacement and replacement.casefold() not in {
                value.casefold() for value in camera_replacements
            }:
                camera_replacements.append(replacement)
        applied_camera_replacement = ""
        if camera_replacements:
            applied_camera_replacement = camera_replacements[0]
            if len(camera_replacements) > 1:
                print(
                    "[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] 캐릭터별 camera_replacement "
                    f"불일치, 첫 결정 사용: kind={kind}, slot={slot}, "
                    f"candidates={camera_replacements}"
                )
            descriptor["camera"] = applied_camera_replacement

        scene_additions: list[str] = []
        scene_addition_ids: set[str] = set()
        for decision in descriptor_decisions:
            for raw_addition in decision.get("scene_additions") or []:
                addition = str(raw_addition or "").strip()
                identity = _authority_tag_identity(addition)
                if not identity or identity in scene_addition_ids:
                    continue
                scene_addition_ids.add(identity)
                scene_additions.append(addition)
        existing_scene_tags = _split_top_level_authority_tags(
            str(descriptor.get("scene") or "")
        )
        existing_scene_ids = {
            _authority_tag_identity(tag) for tag in existing_scene_tags
        }
        applied_scene_additions = [
            addition for addition in scene_additions
            if _authority_tag_identity(addition) not in existing_scene_ids
        ]
        if applied_scene_additions:
            descriptor["scene"] = ", ".join(
                existing_scene_tags + applied_scene_additions
            )
        for character in descriptor.get("characters") or []:
            name = str(character.get("name") or "").strip()
            if not name:
                print(
                    f"[ILLUST_CONTEXT:CALL2_AUTHORITY] 이름 없는 캐릭터 audit 스킵: "
                    f"kind={kind}, slot={slot}, character={character!r}"
                )
                continue

            fixed_tags, default_tags = _descriptor_authority_tags(
                descriptor,
                name,
                fixed_appearance,
                default_outfits,
            )
            if not fixed_tags and not default_tags:
                print(
                    f"[ILLUST_CONTEXT:CALL2_AUTHORITY] 캐릭터 권위 기준 없음: "
                    f"kind={kind}, slot={slot}, character={name}"
                )

            outfit_state = _normalize_outfit_state(character.get("outfit_state"))
            body_state = outfit_state["body_state"]
            wardrobe_authority = default_tags

            allowed_authority = {
                _authority_tag_identity(tag): tag
                for tag in fixed_tags + wardrobe_authority
                if _authority_tag_identity(tag)
            }
            valid_exceptions: list[str] = []
            rejected_exceptions: list[str] = []
            seen_exceptions: set[str] = set()
            semantic_decision = (semantic_decisions or {}).get(
                (kind, slot, name.casefold()),
                {},
            )
            untrusted_exceptions = list(character.get("authority_exceptions") or [])
            if untrusted_exceptions:
                rejected_exceptions.extend(
                    str(value or "").strip()
                    for value in untrusted_exceptions
                    if str(value or "").strip()
                )
                print(
                    f"[ILLUST_CONTEXT:CALL2_AUTHORITY] DETAIL/PLAN 자체 예외는 무시: "
                    f"kind={kind}, slot={slot}, character={name}, "
                    f"exceptions={untrusted_exceptions}"
                )
            requested_exceptions = list(
                semantic_decision.get("authority_exceptions") or []
            )
            for raw_exception in requested_exceptions:
                exception = str(raw_exception or "").strip()
                identity = _authority_tag_identity(exception)
                if not identity or identity not in allowed_authority:
                    rejected_exceptions.append(exception)
                    continue
                if identity in seen_exceptions:
                    continue
                seen_exceptions.add(identity)
                valid_exceptions.append(allowed_authority[identity])
            exception_ids = {
                _authority_tag_identity(tag) for tag in valid_exceptions
            }
            if rejected_exceptions:
                print(
                    f"[ILLUST_CONTEXT:CALL2_AUTHORITY] 기준 밖 authority_exceptions 거부: "
                    f"kind={kind}, slot={slot}, character={name}, "
                    f"rejected={rejected_exceptions}"
                )
            # Keep semantic decisions in the top-level audit only. Do not expand
            # DETAIL/Call3/generated RAW character schemas with audit metadata.
            character.pop("authority_exceptions", None)

            generated_tags = _split_top_level_authority_tags(
                str(character.get("positive") or "")
            )
            generated_by_id = {
                _authority_tag_identity(tag): tag for tag in generated_tags
            }
            semantic_forbidden: list[str] = []
            for raw_forbidden in semantic_decision.get("forbidden_additions") or []:
                identity = _authority_tag_identity(raw_forbidden)
                if identity in generated_by_id and identity not in {
                    _authority_tag_identity(tag) for tag in semantic_forbidden
                }:
                    semantic_forbidden.append(generated_by_id[identity])
            semantic_forbidden_ids = {
                _authority_tag_identity(tag) for tag in semantic_forbidden
            }
            semantic_conflicts: list[str] = []
            for raw_conflict in semantic_decision.get("conflicts") or []:
                identity = _authority_tag_identity(raw_conflict)
                if (
                    identity in generated_by_id
                    and identity not in semantic_forbidden_ids
                    and identity not in {
                        _authority_tag_identity(tag) for tag in semantic_conflicts
                    }
                ):
                    semantic_conflicts.append(generated_by_id[identity])
            semantic_conflict_ids = {
                _authority_tag_identity(tag) for tag in semantic_conflicts
            }
            generated_ids = {
                _authority_tag_identity(tag) for tag in generated_tags
            }
            semantic_required: list[str] = []
            semantic_required_ids: set[str] = set()
            for raw_required in semantic_decision.get("required_additions") or []:
                required = str(raw_required or "").strip()
                identity = _authority_tag_identity(required)
                if (
                    not identity
                    or identity in semantic_required_ids
                    or identity in generated_ids
                    or identity in exception_ids
                    or identity in semantic_forbidden_ids
                    or identity in semantic_conflict_ids
                ):
                    continue
                semantic_required_ids.add(identity)
                semantic_required.append(required)
            mandatory_fixed = [
                tag for tag in fixed_tags
                if _authority_tag_identity(tag) not in exception_ids
            ]
            mandatory_wardrobe = [
                tag for tag in wardrobe_authority
                if _authority_tag_identity(tag) not in exception_ids
            ]
            missing_fixed = [
                tag for tag in mandatory_fixed
                if _authority_tag_identity(tag) not in generated_ids
            ]
            missing_wardrobe = [
                tag for tag in mandatory_wardrobe
                if _authority_tag_identity(tag) not in generated_ids
            ]

            excluded_ids = (
                exception_ids | semantic_forbidden_ids | semantic_conflict_ids
            )
            forbidden_added_removed = [
                tag for tag in generated_tags
                if _authority_tag_identity(tag) in semantic_forbidden_ids
            ]
            conflicts_removed = [
                tag for tag in generated_tags
                if _authority_tag_identity(tag) in semantic_conflict_ids
            ]
            remaining_generated = [
                tag for tag in generated_tags
                if _authority_tag_identity(tag) not in excluded_ids
            ]
            combined: list[str] = []
            combined_ids: set[str] = set()
            for tag in (
                mandatory_fixed
                + mandatory_wardrobe
                + remaining_generated
                + semantic_required
            ):
                identity = _authority_tag_identity(tag)
                if not identity or identity in combined_ids:
                    continue
                combined_ids.add(identity)
                combined.append(tag)
            character["positive"] = ", ".join(combined)

            existing_worn = [
                tag for tag in outfit_state["worn"]
                if _authority_tag_identity(tag) not in excluded_ids
            ]
            wardrobe_authority_ids = {
                _authority_tag_identity(tag) for tag in wardrobe_authority
            }
            normalized_worn: list[str] = []
            normalized_worn_ids: set[str] = set()
            for tag in mandatory_wardrobe + existing_worn:
                identity = _authority_tag_identity(tag)
                if not identity or identity in normalized_worn_ids:
                    continue
                normalized_worn_ids.add(identity)
                normalized_worn.append(tag)
            outfit_state["worn"] = normalized_worn
            outfit_state["removed"] = [
                tag for tag in outfit_state["removed"]
                if (
                    _authority_tag_identity(tag) not in wardrobe_authority_ids
                    or _authority_tag_identity(tag) in exception_ids
                )
            ]
            if mandatory_wardrobe and body_state in {"unknown", "nude", "underwear_only"}:
                outfit_state["body_state"] = (
                    "partial" if any(
                        _authority_tag_identity(tag) in exception_ids
                        for tag in wardrobe_authority
                    ) else "clothed"
                )
            character["outfit_state"] = outfit_state

            audit = {
                "kind": kind,
                "slot": slot,
                "character": name,
                "missing_fixed_added": missing_fixed,
                "missing_wardrobe_added": missing_wardrobe,
                "authority_exceptions": valid_exceptions,
                "forbidden_added_removed": forbidden_added_removed,
                "conflicts_removed": conflicts_removed,
                "rejected_exceptions": rejected_exceptions,
                "semantic_status": semantic_status,
            }
            if semantic_required:
                audit["required_additions"] = semantic_required
            if applied_scene_additions:
                audit["scene_additions"] = applied_scene_additions
            if applied_camera_replacement:
                audit["camera_replacement"] = applied_camera_replacement
            audits.append(audit)
            print(
                "[ILLUST_CONTEXT:CALL2_AUTHORITY_AUDIT] "
                + json.dumps(audit, ensure_ascii=False, separators=(",", ":"))
            )
    return audits


def _balanced_call2_scene_plan_batches(
    scene_plan: list[dict],
    max_concurrency: int,
) -> list[list[dict]]:
    """PLAN 결과를 가능한 DETAIL 작업 수에 서사 순서대로 균등 분배한다."""
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
) -> tuple[dict, str]:
    """Generate one independent Key Visual descriptor without a PLAN dependency."""
    messages = deepcopy(call2_context_messages)
    allowed = [
        str(name or "").strip()
        for name in allowed_character_names
        if str(name or "").strip()
    ]
    if not allowed:
        print(
            "[ILLUST_CONTEXT:CALL2_KEYVIS] 현재 canonical roster가 비어 있음: "
            "CHARACTER DICTIONARY와 현재 문맥에 명시된 canonical 이름만 사용하도록 요청"
        )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = str(messages[0].get("content") or "") + (
            "\n\n# Independent CALL2-KEYVIS override\n"
            "Create exactly one standalone promotional Key Visual from the supplied current context. "
            "This worker runs independently from CALL2-PLAN: do not select narrative slots, do not "
            "output a scene plan, and do not wait for or refer to another worker. Synthesize the central "
            "relationship, contrast, or theme into one magazine-cover-level composition instead of "
            "copying one presumed planned scene. Output exactly one keyvis object and no scene objects. "
            "Use only canonical character names supported by the supplied dictionary and current context. "
            "For named characters, persistent appearance comes only from the server-supplied "
            "AUTHORITATIVE FIXED APPEARANCE block; other CHARACTER DICTIONARY sections are not "
            "identity sources. Do not "
            "creatively fill missing identity traits from narrative prose. Narrative may control pose, "
            "action, expression, composition, and temporary visual state. "
            "Rebuild each named character from the complete fixed appearance. Treat the supplied current "
            "wardrobe as continuity and default_outfit as a fallback reference, not fixed identity. If the full "
            "Key Visual concept calls for different attire, design one coherent context-appropriate outfit by "
            "meaning and replace the fallback as a set; do not keyword-match or mix incompatible default garments "
            "into it. A separate server audit validates the contextual replacement while keeping every fixed "
            "appearance tag mandatory. Generated visual references "
            "are intentionally absent and must not be reconstructed as identity facts. "
            "Before returning, silently verify that the composition is one physically possible image and "
            "that its camera can actually show every story-essential action, contact, exposure, displaced "
            "garment, and visible anatomy. Treat explicit content as a coherent scene-specific detail bundle, "
            "not as one isolated tag, while adding nothing the story does not support. "
            "This override "
            "supersedes every global scene-count, slot-selection, and combined keyvis/scene requirement. "
            "Every characters[] entry must include its exact canonical name even when cropped or partially "
            "visible. characters[].negative is optional: include it only when the Client explicitly "
            "supplied that negative; otherwise omit the field."
        )

    roster_text = json.dumps(allowed, ensure_ascii=False)
    messages.append({
        "role": "user",
        "content": (
            "# INDEPENDENT KEY VISUAL\n"
            "Return one complete Key Visual descriptor only. Do not output scenes, slots, plan_id, "
            "analysis, JSON, or prose outside the <lb-xnai> block. Choose at least one and at most "
            f"{max(1, min(3, int(toggles.get('character_limit', 3))))} named characters when supported "
            "by the current context.\n\n"
            "# CURRENT CANONICAL CHARACTER ROSTER\n"
            + (roster_text if allowed else "(unavailable; use only names in CHARACTER DICTIONARY)")
            + "\n\n# OUTPUT FORMAT\n"
            "<lb-xnai>\n"
            "keyvis:\n"
            "  camera: ...\n"
            "  characters:\n"
            "    - positive: ...\n"
            "      name: canonical name\n"
            "      position: ...\n"
            "      outfit_state:\n"
            "        body_state: clothed|partial|nude|topless|bottomless|underwear_only|unknown\n"
            "        worn: [...]\n"
            "        removed: [...]\n"
            "  scene: ...\n"
            "  supplement: ...\n"
            "scenes: []\n"
            "</lb-xnai>"
        ),
    })
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
    )
    if descriptor is None:
        print(
            f"[ILLUST_CONTEXT:CALL2_KEYVIS] 최종 응답 검증 실패: reason={reason}"
        )
        raise ValueError(reason or "CALL2-KEYVIS 최종 검증 실패")
    return descriptor, raw_output


async def _run_parallel_call2_details(
    *,
    scene_plan: list[dict],
    call2_context_messages: list[dict],
    call2_format: str,
    toggles: dict,
    stream_notify,
    call2_thoughts: str = "",
) -> tuple[list[dict], list[str]]:
    source_format = str(call2_format or "").strip()
    keyvis_marker = re.search(r"(?m)^keyvis:\s*$", source_format)
    if keyvis_marker is not None:
        detail_output_format = (
            source_format[:keyvis_marker.start()].rstrip()
            + "\n</lb-xnai>"
        )
    else:
        detail_output_format = source_format
        print(
            "[ILLUST_CONTEXT:CALL2_DETAIL] format.txt에 keyvis 구조가 없어 "
            "입력 형식을 그대로 사용"
        )
    if not detail_output_format:
        print(
            "[ILLUST_CONTEXT:CALL2_DETAIL] scene-only 출력 형식 생성 실패: "
            "call2_format이 비어 있음"
        )
        raise ValueError("CALL2-DETAIL 출력 형식이 비어 있습니다")
    detail_output_format = re.sub(
        r"(?m)^\s+negative:\s*\.\.\.\s*\r?\n?",
        "",
        detail_output_format,
    )
    detail_checklist = str(call2_thoughts or "").strip()
    if toggles.get("minimal_background_description", True):
        detail_background_instruction = (
            "Prioritize each character's current clothing or exposure state, pose, action, "
            "expression, gaze, and interaction before environment detail. Keep the environment "
            "to the smallest story-supported cue, or use only `simple background` when no clear "
            "or important background exists. Do not invent decorative props, weather, time, or "
            "elaborate lighting. Make the camera, relative positions, actions, contact, visible "
            "anatomy, garment displacement, expressions, and minimal background agree as one "
            "physically possible image. "
        )
    else:
        detail_background_instruction = (
            "Prioritize each character's current clothing or exposure state, pose, action, "
            "expression, gaze, and interaction, while also describing the story-supported "
            "environment at a useful visual density. Include concrete location, time, weather, "
            "lighting, scenery, furniture, and prominent props when established. Use multiple "
            "complementary environment details when they express distinct visible facts; do not "
            "invent unsupported decoration or collapse a specific setting into `simple background`. "
            "Make the camera, relative positions, actions, contact, visible anatomy, garment "
            "displacement, expressions, and environment agree as one physically possible image. "
        )
    max_concurrency = int(toggles["call2_parallel_max_concurrency"])
    batches = _balanced_call2_scene_plan_batches(scene_plan, max_concurrency)
    jobs = [{"plans": batch, "weight": len(batch)} for batch in batches]
    distribution = [len(batch) for batch in batches]
    print(
        f"[ILLUST_CONTEXT:CALL2_DETAIL] 상세 장면 배치 준비: "
        f"selected_scenes={len(scene_plan)}, workers={len(jobs)}, "
        f"distribution={distribution}"
    )
    # 카운트 규칙의 총량(output_count_min/max)을 실제 worker 수로 나눈다.
    # 각 worker는 자기 몫(per_worker)만 생성해야 3배 과잉 생성이 생기지 않는다.
    worker_count = max(1, len(jobs))
    total_min = int(toggles["output_count_min"])
    total_max = int(toggles["output_count_max"])
    per_worker_min = max(1, total_min // worker_count)
    per_worker_max = max(per_worker_min, math.ceil(total_max / worker_count))
    print(
        f"[ILLUST_CONTEXT:CALL2_DETAIL] per-worker 카운트 기준: "
        f"workers={worker_count}, total={total_min}..{total_max}, "
        f"per_worker={per_worker_min}..{per_worker_max}"
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
        plans = list(job["plans"])
        assigned_slots = [int(item["slot"]) for item in plans]
        assigned_plan_ids = [str(item["plan_id"]) for item in plans]
        assigned_wardrobes_by_slot = {
            int(item["slot"]): deepcopy(item.get("wardrobe_snapshot") or {})
            for item in plans
        }
        assigned_characters_by_slot = {
            int(item["slot"]): list(item.get("characters") or [])
            for item in plans
        }
        assigned_scene_context_by_slot = {
            int(item["slot"]): {
                "scene_brief": str(item.get("scene_brief") or "").strip(),
                "continuity_note": str(item.get("continuity_note") or "").strip(),
                "continuity_characters": list(item.get("_continuity_characters") or []),
                "visual_base_snapshot": deepcopy(
                    item.get("visual_base_snapshot") or {}
                ),
            }
            for item in plans
        }
        # per-worker 카운트 기준(총량÷worker수)을 이 worker에 실제 할당된 batch 크기로
        # clamp한다. 규칙이 할당량보다 커지면 worker가 빈 장면을 채우려다 검증에 걸려
        # 재시도하게 된다. 할당량(len(plans))이 곧 이 shard의 진짜 장면 수다.
        def build_messages(request_plans: list[dict], *, partial: bool) -> list[dict]:
            # per-worker 카운트 기준(총량÷worker수)을 이번 요청에 실제 담긴 batch 크기로
            # clamp한다. 전부 예측(①)은 shard 전체(len(plans)), 실패분만(②)은 missing 수.
            # 규칙이 할당량보다 커지면 worker가 빈 장면을 채우려다 검증에 걸려 재시도한다.
            request_count = len(request_plans)
            req_min = max(1, min(per_worker_min, request_count))
            req_max = max(req_min, min(per_worker_max, request_count))
            req_rule = render_output_count_rule(req_min, req_max)
            base = deepcopy(call2_context_messages)
            if base and base[0].get("role") == "system":
                base[0]["content"] = str(base[0].get("content") or "") + (
                    "\n\n# Parallel CALL2-DETAIL instructions\n"
                    "The global planner already selected the visual beats. Do not select, add, remove, or move a scene. "
                    "Copy every assigned slot exactly; the server will attach plan_id after slot validation. "
                    "Omit keyvis completely. "
                    + "An assigned plan may have characters: []. That means no named tracked character is "
                    "present: output characters: [] for that scene, keep anonymous background people only "
                    "in scene/supplement, and do not invent a canonical character. This shard-specific rule "
                    "overrides any global requirement that every scene contain a key character. Every characters[] "
                    "entry for a named character must include its exact canonical name even when cropped "
                    "or partially visible. characters[].negative is optional: include it only when the "
                    "Client explicitly supplied that negative; otherwise omit the field."
                    # 이 요청의 카운트 규칙은 전체 총량이 아니라 (총량÷worker수) per-worker 값을
                    # 이번 batch 크기로 맞춘 것이다. 다른 전역 카운트 지시보다 이 값이 우선한다.
                    + "\n\n# SHARD OUTPUT COUNT RULE (per worker)\n"
                    + req_rule
                )
            public_request_plans = [
                _public_call2_scene_plan(plan) for plan in request_plans
            ]
            assigned_plan_payload = (
                "# ASSIGNED GLOBAL SCENE PLAN\n"
                + json.dumps(public_request_plans, ensure_ascii=False, indent=2)
            )
            expand_instruction = (
                "Expand each plan into a complete, coherent visual tag bundle with camera, scene, "
                "character positives, outfit_state, and supplement. The structured wardrobe snapshot is "
                "kept server-side for validation rather than used as an inter-LLM semantic handoff. Start "
                "from fixed appearance and resolve wardrobe from tracked continuity, the default-outfit "
                "reference, and the full assigned scene. "
                "When continuity_note is present, read that natural-language chronology by meaning and "
                "treat it as authority for the affected character's current wardrobe, coverage, and "
                "exposure; coarse operation/body-state hints or a stale snapshot must never simplify, "
                "euphemize, or contradict it. Include every fixed-appearance tag. Use the complete default outfit "
                "only as fallback when the tracked wardrobe and full scene do not call for something different. "
                "When different attire is contextually appropriate, design one coherent outfit by meaning and "
                "replace the default as a set even if no sentence lists every garment; never keyword-match or "
                "carry incompatible default pieces into it. A separate server audit validates the contextual "
                "outfit and keeps fixed identity mandatory. Never advance state beyond the assigned scene. "
                + detail_background_instruction
                + "Never repeat scene-wide environment, "
                "lighting, weather, time, character-count, or shared background-prop tags in characters[].positive. "
                "Do not reduce a "
                "story-essential explicit state to an ambiguous isolated tag or crop it out. "
                "When a plan has characters: [], preserve characters: [] and express anonymous people "
                "only through scene tags and supplement. "
                "Copy slot exactly into every scene object and preserve plan order. The server assigns "
                "plan_id from the validated slot."
            )
            if partial:
                expand_instruction = (
                    "These are the ONLY scenes still missing from this shard. Produce exactly these slots "
                    "and no others — do not repeat scenes already delivered earlier in this shard. "
                    + expand_instruction
                )
            base.extend([{
                "role": "user",
                "content": (
                    assigned_plan_payload
                    + "\n\n"
                    + expand_instruction
                    + (
                        "\n\n# DETAIL CHECKLIST\n"
                        + detail_checklist
                        if detail_checklist
                        else ""
                    )
                    + "\n\n"
                    "# OUTPUT FORMAT\n"
                    + detail_output_format
                    + "\n\nReturn one <lb-xnai> block containing scenes only. Omit keyvis."
                ),
            }])
            return base

        # CALL2-DETAIL 부분 재시도 루프: ①전부예측(primary) → ②실패분만(fallback) →
        # 한 사이클(①+②)이 통째로 실패하면 retry+1 하여 max_cycles 까지 ①으로 되돌아간다.
        # 이미 검증 통과한 좋은 슬롯은 kept_by_slot 에 보존해 두고, 매 단계는 지정 슬롯을
        # 1회씩만(force_slot) 부른다. 상한·슬롯은 전역 API 분기(llm_routing[CALL2]) 설정을
        # 그대로 재사용한다. slot 하나만 틀려도 전체를 다시 돌리지 않도록 한다.
        plan_id_by_slot = dict(zip(assigned_slots, assigned_plan_ids))
        routing_task_key = _CALL_TASK_KEYS["CALL2"]
        primary_slot, fallback_slot = llm_service._routing_for(routing_task_key)
        retry_policy = llm_service._routing_retry_policy(routing_task_key)
        max_cycles = max(1, int(retry_policy.get("max_retries") or 0) + 1)
        partial_slot = fallback_slot or primary_slot
        if fallback_slot is None:
            print(
                f"[ILLUST_CONTEXT:CALL2_DETAIL] 폴백 미설정: ②실패분만도 primary 슬롯 사용. "
                f"job={index}/{total}, primary={primary_slot}"
            )

        def parse_scope(raw: str, scope_slots: list[int], scope_label: str):
            scope_plan_ids = [plan_id_by_slot[s] for s in scope_slots]
            return _parse_call2_detail_partial(
                raw,
                toggles,
                scope_slots,
                scope_plan_ids,
                scope_label,
                assigned_wardrobes_by_slot,
                assigned_characters_by_slot,
                assigned_scene_context_by_slot,
            )

        def make_validator(scope_slots: list[int]):
            # 부분 허용: scope 중 1개라도 보존되면 accepted. force_slot 1회 호출이라
            # 검증 거절은 callLLMTask 내부 재시도를 유발하지 않고 [LLM 실패]로 돌아간다.
            def validate(result):
                kept, _missing, discarded, hard = parse_scope(
                    result,
                    scope_slots,
                    f"CALL2-DETAIL-{index}-RETRY-CHECK",
                )
                if kept or discarded:
                    return True, ""
                return False, (hard or "CALL2-DETAIL 보존 가능한 장면 없음")
            return validate

        kept_by_slot: dict[int, dict] = {}
        char_discarded: set[int] = set()
        raw_outputs: list[str] = []
        last_reason = ""
        cycle = 0

        def remaining_to_fill() -> list[int]:
            # 캐릭터 불일치로 폐기된 슬롯은 재시도해도 같은 결과이므로 채움 대상에서
            # 빼고, 보존된 슬롯과 함께 "확정"으로 간주해 빈 샤드를 정상으로 받아들인다.
            return [
                slot
                for slot in assigned_slots
                if slot not in kept_by_slot and slot not in char_discarded
            ]

        while cycle < max_cycles:
            missing_slots = remaining_to_fill()
            if not missing_slots:
                break
            cycle += 1

            # ① 전부 예측 (primary 슬롯 1회 강제)
            full_name = f"CALL2-DETAIL {index}/{total}"
            if attempt_kind == "duplicate":
                full_name += " [느리다고? 다시해!]"
            elif attempt_kind == "failed_shard_retry":
                full_name += " [FAILED-SHARD-RETRY]"
            full_name += f" [FULL c{cycle}/{max_cycles}]"
            try:
                raw_full = await _call_pipeline_llm(
                    full_name,
                    _normalize_messages(build_messages(plans, partial=False)),
                    job_stream_notify,
                    result_validator=make_validator(assigned_slots),
                    stream_observer=stream_observer,
                    history_id=(history_id if cycle == 1 else ""),
                    force_slot=primary_slot,
                )
                kept, _missing, discarded, hard = parse_scope(
                    raw_full,
                    assigned_slots,
                    f"CALL2-DETAIL-{index}-FULL-c{cycle}",
                )
                kept_by_slot.update(kept)
                char_discarded.update(discarded)
                char_discarded -= set(kept)
                raw_outputs.append(str(raw_full or ""))
                last_reason = hard or ""
            except Exception as e:
                last_reason = str(e) or type(e).__name__
                print(
                    f"[ILLUST_CONTEXT:CALL2_DETAIL] ①전부예측(primary={primary_slot}) "
                    f"실패/진행없음: job={index}/{total}, cycle={cycle}/{max_cycles}, "
                    f"kept={sorted(kept_by_slot)}, error={last_reason}"
                )

            missing_slots = remaining_to_fill()
            if not missing_slots:
                break

            # ② 실패분만 예측 (fallback 슬롯 1회 강제)
            partial_plans = [p for p in plans if int(p["slot"]) in set(missing_slots)]
            partial_name = f"CALL2-DETAIL {index}/{total} [PARTIAL c{cycle}/{max_cycles}]"
            try:
                raw_part = await _call_pipeline_llm(
                    partial_name,
                    _normalize_messages(build_messages(partial_plans, partial=True)),
                    job_stream_notify,
                    result_validator=make_validator(missing_slots),
                    stream_observer=stream_observer,
                    force_slot=partial_slot,
                )
                kept, _missing, discarded, hard = parse_scope(
                    raw_part,
                    missing_slots,
                    f"CALL2-DETAIL-{index}-PARTIAL-c{cycle}",
                )
                kept_by_slot.update(kept)
                char_discarded.update(discarded)
                char_discarded -= set(kept)
                raw_outputs.append(str(raw_part or ""))
                last_reason = hard or ""
            except Exception as e:
                last_reason = str(e) or type(e).__name__
                print(
                    f"[ILLUST_CONTEXT:CALL2_DETAIL] ②실패분만(partial={partial_slot}) "
                    f"실패/진행없음: job={index}/{total}, cycle={cycle}/{max_cycles}, "
                    f"missing={missing_slots}, error={last_reason}"
                )

        missing_slots = remaining_to_fill()
        if missing_slots:
            raise ValueError(
                f"CALL2-DETAIL 부분 재시도 상한({max_cycles}사이클) 초과로 미확보 슬롯 남음: "
                f"missing={missing_slots}, assigned={assigned_slots}, last_reason={last_reason}"
            )
        descriptors = [kept_by_slot[slot] for slot in assigned_slots if slot in kept_by_slot]
        combined_raw = "\n\n".join(raw for raw in raw_outputs if raw)
        return {"raw": combined_raw, "descriptors": descriptors}

    try:
        results = await _run_parallel_pipeline_jobs(
            jobs,
            group_id="CALL2_DETAIL",
            group_label="CALL2 상세 장면",
            max_concurrency=max_concurrency,
            slow_retry_enabled=bool(toggles["call2_parallel_slow_retry_enabled"]),
            slow_retry_remaining=int(toggles["call2_parallel_slow_retry_remaining"]),
            slow_retry_progress_enabled=bool(toggles["call2_parallel_slow_retry_progress_enabled"]),
            slow_retry_progress_threshold=int(toggles["call2_parallel_slow_retry_progress_threshold"]),
            slow_retry_tps_enabled=bool(toggles["call2_parallel_slow_retry_tps_enabled"]),
            slow_retry_tps_threshold=float(toggles["call2_parallel_slow_retry_tps_threshold"]),
            slow_retry_condition_operator=str(toggles["call2_parallel_slow_retry_condition_operator"]),
            stream_notify=stream_notify,
            invoke=invoke,
        )
    except ParallelPipelineJobsError as group_error:
        result_by_index = dict(group_error.resolved_values)
        print(
            f"[ILLUST_CONTEXT:CALL2_DETAIL] 성공 shard 보존 후 실패 shard만 재시도: "
            f"preserved={sorted(result_by_index)}, failed={sorted(group_error.failures)}"
        )
        for failed_index, first_reason in sorted(group_error.failures.items()):
            retry_notify = None
            if stream_notify:
                async def retry_notify(event: dict, shard_index=failed_index):
                    payload = dict(event)
                    payload["queue_subtask"] = {
                        "group_id": "call2_detail",
                        "group_label": "CALL2 상세 장면",
                        "index": shard_index,
                        "total": len(jobs),
                    }
                    await stream_notify(payload)
            try:
                result_by_index[failed_index] = await invoke(
                    jobs[failed_index - 1],
                    failed_index,
                    len(jobs),
                    "failed_shard_retry",
                    None,
                    "",
                    retry_notify,
                )
            except asyncio.CancelledError:
                print(
                    f"[ILLUST_CONTEXT:CALL2_DETAIL] 실패 shard 재시도 중 상위 작업 취소: "
                    f"job={failed_index}/{len(jobs)}"
                )
                raise
            except Exception as retry_error:
                print(
                    f"[ILLUST_CONTEXT:CALL2_DETAIL] 실패 shard 재시도 최종 실패: "
                    f"job={failed_index}/{len(jobs)}, first_reason={first_reason}, "
                    f"retry_error={retry_error}"
                )
                traceback.print_exc()
                raise RuntimeError(
                    f"CALL2 상세 장면 실패 shard {failed_index}/{len(jobs)} 재시도 실패: "
                    f"{retry_error}"
                ) from retry_error
        results = [result_by_index[index] for index in range(1, len(jobs) + 1)]
    descriptors = []
    raw_outputs = []
    for result in results:
        descriptors.extend(result.get("descriptors") or [])
        raw_outputs.append(str(result.get("raw") or ""))
    return descriptors, raw_outputs


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
    compact = re.sub(r"\s+", " ", _strip_nodes(str(value or ""))).strip()
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


def parse_speak_output(text: str, max_entries_per_scene: int | None = None) -> dict[int, str]:
    """Parse CALL3 Scene blocks and optionally enforce a structural entry limit."""
    result: dict[int, list[str]] = {}
    current = None
    dropped: dict[int, int] = {}
    for line in str(text or "").splitlines():
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
            "slot": slot,
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
    source = str(text or "")
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
    """Select the Speak/Manga prompt and append only mode-compatible instructions."""
    prompt_mode = str(toggles.get("call3_prompt_mode") or "speak").strip().lower()
    prompt_key = "call3_manga" if prompt_mode == "manga" else "call3_speak"
    selected_prompt = str(prompts.get(prompt_key) or "").strip()
    if not selected_prompt:
        print(
            f"[ILLUST_CONTEXT:CALL3] 선택한 대사 프롬프트가 비어 있음: "
            f"mode={prompt_mode}, key={prompt_key}"
        )
        raise RuntimeError(f"CALL3 {prompt_mode} 프롬프트가 비어 있습니다")

    emotion_instruction = ""
    if prompt_mode == "speak" and toggles.get("speak_emotion_enabled"):
        emotion_instruction = "\nAdd one #emotion tag to every emitted line."
        emotions = str(toggles.get("speak_emotions") or "").strip()
        if emotions:
            emotion_instruction += " Allowed labels: " + emotions
    elif prompt_mode == "manga" and toggles.get("speak_emotion_enabled"):
        print("[ILLUST_CONTEXT:CALL3] Manga 모드에서는 감정 태그 설정을 사용하지 않음")

    # NSFW 버블 타입(#nsfw_soft/#nsfw_hard) 보강 블록. manga 모드이고 nsfw 토글이
    # 켜져 있을 때만 manga 프롬프트 끝에 붙인다. 일반 장면엔 노출되지 않는다.
    nsfw_instruction = ""
    if prompt_mode == "manga" and toggles.get("nsfw"):
        nsfw_block = str(prompts.get("call3_manga_nsfw") or "").strip()
        if nsfw_block:
            nsfw_instruction = "\n" + nsfw_block
        else:
            print("[ILLUST_CONTEXT:CALL3] nsfw 토글 ON이나 manga_nsfw 프롬프트가 비어 있어 SOFT/HARD 타입 미주입")

    # CALL3에는 lb.extra 중 캐릭터 영문 이름 리스트만 넘긴다(시스템 프롬프트/복장 제외).
    # speak/manga 프롬프트의 {character_names} 자리표시자를 치환한다.
    names = str(extra_names or "").strip()
    if "{character_names}" in selected_prompt:
        system_prompt = selected_prompt.replace("{character_names}", names)
    elif names:
        system_prompt = selected_prompt + "\n\nCharacter names: " + names
    else:
        system_prompt = selected_prompt
    output_language = str(toggles.get("speak_language") or "한국어").strip() or "한국어"
    roster_boundary_instruction = ""
    if _call3_dialogue_requires_localized_names(output_language):
        roster_boundary_instruction = (
            "\nAn exact roster identifier may appear as machine-readable metadata only on the "
            "left side of the colon. Do not repeat it inside quoted dialogue or parenthesized "
            f"thought. Inside the text body, use a natural {output_language} form of address "
            "supported by the original narrative and bounded conversation, or omit direct "
            "address when uncertain."
        )
    language_instruction = (
        "# OUTPUT LANGUAGE — HARD REQUIREMENT\n"
        f"Write every dialogue, thought, inner monologue, and newly created reaction in {output_language}.\n"
        "Character names, [Scene slot=N] headers, and required output tags may remain in their "
        "prescribed form. Do not switch the spoken text to another language even when the source "
        "narrative or examples use another language. Before answering, silently verify that every "
        f"spoken or thought line follows the required output language: {output_language}."
        + roster_boundary_instruction
    )
    system_prompt = language_instruction + "\n\n" + system_prompt
    system_prompt += emotion_instruction
    system_prompt += nsfw_instruction
    return prompt_mode, system_prompt


def _build_character_history(extra_reference: str) -> str:
    # 서버가 보유한 lb.extra 자체가 가장 안정적인 외형 이력/영문 이름 사전이다.
    return str(extra_reference or "").strip()


# 삽화 CALL 이름 → 외부 LLM 분기 task_key. PLAN/DETAIL/KEYVIS는 사용자가 선택한
# 하나의 illustration_call2 경로를 공유한다. 기본 primary=llm1(server.py 참고).
_CALL_TASK_KEYS = {
    "CALL1-BACKTRANSLATE": "illustration_call1_backtranslate",
    "CALL1": "illustration_call1",
    "CALL2": "illustration_call2",
    "CALL2-PLAN": "illustration_call2",
    "CALL2-KEYVIS": "illustration_call2",
    "CALL2-AUTHORITY-AUDIT": "illustration_call2",
    "CALL2-FALLBACK": "illustration_call2",
    "CALL2-FIX": "illustration_call2_fix",
    "CALL3": "illustration_call3",
    "CALL3-CORRECTION": "illustration_call3",
    "MULTI-CHAR-MASK": "illustration_multi_char_mask",
}

# CALL1/2/2-FIX/3 각 LLM 호출을 큐 서브태스크로 표시하기 위한 그룹 정의.
# 역번역(CALL1-BACKTRANSLATE)/다중캐릭터마스크(MULTI-CHAR-MASK)는 병렬 청크용 wrapper가
# index/total을 직접 주입하므로 여기서 제외한다.
_CALL_QUEUE_SUBTASK_GROUPS = {
    "CALL1": ("call1", "CALL1 컨텍스트 보강"),
    "CALL2": ("call2", "CALL2 장면/태그 빌드"),
    "CALL2-PLAN": ("call2_plan", "CALL2 장면 PLAN"),
    "CALL2-KEYVIS": ("call2_keyvis", "CALL2 Key Visual"),
    "CALL2-AUTHORITY-AUDIT": (
        "call2_authority_audit",
        "CALL2 외형·복장 권위 감사",
    ),
    "CALL2-FALLBACK": ("call2", "CALL2 폴백"),
    "CALL2-FIX": ("call2_fix", "CALL2-FIX TOON 교정"),
    "CALL3": ("call3", "CALL3 대사 빌드"),
    "CALL3-CORRECTION": ("call3_correction", "CALL3 슬롯/언어 교정"),
}


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
    """삽화 CALL1/2/3 의 LLM 호출. 외부 LLM 분기(illustration_callN task_key)를 경유한다.

    외부 LLM 분기 탭에서 CALL별로 LLM1/LLM2/LLM3 을 선택하거나 폴백을 켤 수 있다.
    실패 시 callLLMTask 가 지정된 폴백 LLM 으로 재시도한다.
    """
    started = time.time()
    task_key = _CALL_TASK_KEYS.get(call_name)
    if task_key is None and call_name.startswith("CALL1-BACKTRANSLATE"):
        task_key = _CALL_TASK_KEYS["CALL1-BACKTRANSLATE"]
    if task_key is None and call_name.startswith("CALL1 "):
        task_key = _CALL_TASK_KEYS["CALL1"]
    if task_key is None and (
        call_name.startswith("CALL2-PLAN")
        or call_name.startswith("CALL2-DETAIL")
        or call_name.startswith("CALL2-KEYVIS")
    ):
        task_key = _CALL_TASK_KEYS["CALL2"]
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
    model = (
        llm_service.routing_primary_model(task_key)
        or llm_service._current_config.get("llm_model3")
        or llm_service._current_config.get("llm_model")
        or ""
    )
    service = llm_service.routing_primary_service(task_key) or ""
    history_record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "prompt_id": f"illustration_context:{call_name}",
        "call_name": call_name,
        "task_key": task_key,
        "model": model,
        "service": service,
        "input": messages,
        "output": "",
        "completion_tokens": 0,
        "elapsed": 0.0,
        "tps": 0.0,
        "history_id": execution_id,
        "execution_id": execution_id,
        "parent_execution_id": parent_execution_id,
        # 자세히 'LLM 실행 연결 정보' 기본값. attempt_success에서 실제 라우팅
        # 결과(폴백 슬롯 포함)로 덮어쓴다.
        "phase": "",
        "llm_slot": "",
        "attempt": 0,
        "total_attempts": 0,
        "attempt_id": "",
    }
    history_logged = False
    terminal_notified = False
    success_meta: dict = {}

    async def _notify(event: dict):
        # stream_notify 이벤트에 큐 서브태스크 그룹을 주입한다.
        # 역번역/다중캐릭터마스크 wrapper가 이미 queue_subtask를 넣은 경우 유지한다.
        if not stream_notify:
            return
        event.setdefault("execution_id", execution_id)
        event.setdefault("parent_execution_id", parent_execution_id)
        if "queue_subtask" not in event:
            base = call_name.split()[0] if call_name else ""
            grp = _CALL_QUEUE_SUBTASK_GROUPS.get(base)
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
        if etype == "attempt_success":
            success_meta.clear()
            success_meta.update({
                "phase": str(event.get("phase") or ""),
                "llm_slot": str(event.get("llm_slot") or event.get("slot") or ""),
                "attempt": int(event.get("attempt") or 0),
                "total_attempts": int(event.get("total_attempts") or 0),
                "attempt_id": str(event.get("attempt_id") or ""),
            })
            return
        if etype != "attempt_failure":
            return
        raw_result = event.get("raw_response", event.get("result"))
        attempt_exception = event.get("error") or event.get("exception")
        raw_output = "" if raw_result is None else str(raw_result)
        if not raw_output and attempt_exception is not None:
            raw_output = str(attempt_exception)
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
        try:
            result = await llm_service.callLLMTask(task_key, messages, **call_kwargs)
        finally:
            llm_service._stream_metadata_ctx.reset(stream_metadata_token)
        if not result or str(result).startswith("[LLM 실패]"):
            print(f"[ILLUST_CONTEXT:{call_name}] LLM 호출 실패: {result}")
            if stream_notify:
                await _notify({"type": "error", "call_name": call_name, "error": str(result)})
                terminal_notified = True
            raise RuntimeError(str(result or f"빈 {call_name} 응답"))
        elapsed = time.time() - started
        tokens = max(1, len(str(result)) // 3)
        prompt_tokens = llm_service._approx_input_tokens(messages)
        if stream_notify:
            await _notify({
                "type": "done",
                "call_name": call_name,
                "model": model,
                "text": str(result),
                "completion_tokens": tokens,
                "prompt_tokens": prompt_tokens,
                "elapsed": elapsed,
                "tps": tokens / elapsed if elapsed > 0 else 0.0,
                "ttft": elapsed,
            })
            terminal_notified = True
        history_record.update({
            "output": str(result),
            "completion_tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": round(elapsed, 3),
            "tps": round(tokens / elapsed, 1) if elapsed > 0 else 0.0,
            "ttft": round(elapsed, 3),
            "status": "ok",
            "phase": success_meta.get("phase", ""),
            "llm_slot": success_meta.get("llm_slot", ""),
            "attempt": success_meta.get("attempt", 0),
            "total_attempts": success_meta.get("total_attempts", 0),
            "attempt_id": success_meta.get("attempt_id", ""),
        })
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
            "CALL3",
            normalized_messages,
            stream_notify,
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
    call3_valid, call3_failure_reason = validate_call3_output_contract(
        initial_output,
        selected_slots,
        character_names,
        speak_language,
    )
    if call3_valid:
        state["output"] = initial_output
        state["silent_slots"] = [
            slot for slot in selected_slots
            if slot not in parse_speak_output(initial_output)
        ]
        return state

    state["correction_used"] = True
    state["failure_reason"] = call3_failure_reason
    roster_correction_instruction = ""
    if _call3_dialogue_requires_localized_names(speak_language):
        roster_correction_instruction = (
            "Keep each exact roster identifier as machine-readable speaker metadata only "
            "on the left side of the colon. Never repeat a roster identifier inside quoted "
            "dialogue or parenthesized thought. Infer a natural in-story form of address "
            "from the original narrative and bounded scene windows; if uncertain, omit "
            "the direct address. Preserve the speaker prefix and required output tag. "
        )
    print(
        f"[ILLUST_CONTEXT:CALL3-CORRECTION] 최초 CALL3 결과가 출력 계약을 위반해 "
        f"교정 호출 1회 실행: slots={selected_slots}, reason={call3_failure_reason}"
    )
    retry_messages = deepcopy(speak_messages)
    retry_messages.extend([{
        "role": "assistant",
        "content": initial_output,
    }, {
        "role": "user",
        "content": (
            "Your previous output violated the selected-scene contract. "
            f"Required slots, in order: {selected_slots}. "
            "Rewrite the entire output. Emit exactly one [Scene slot=N] block "
            "for every required slot and no block for any other slot. "
            "A scene that should intentionally remain silent, or has no suitable "
            "visible or narratively present speaker, must keep its Scene header with "
            "an empty body. Do not invent narration, action description, dialogue, or "
            "thought merely to fill such a block. Every other block must contain at "
            "least one dialogue, thought, or inner monologue entry. "
            f"Write every dialogue and thought in {speak_language}; this language rule is mandatory. "
            + roster_correction_instruction
            + "Character names in speaker prefixes, Scene headers, and required tags are "
            "the only language exceptions. "
            f"Validation failure to fix: {call3_failure_reason}. "
            "Output only the corrected Scene blocks."
        ),
    }])
    try:
        corrected_output = await _call_pipeline_llm(
            "CALL3-CORRECTION",
            _normalize_messages(retry_messages),
            stream_notify,
            result_validator=lambda result: validate_call3_output_contract(
                result,
                selected_slots,
                character_names,
                speak_language,
            ),
        )
        state["output"] = corrected_output
        state["silent_slots"] = [
            slot for slot in selected_slots
            if slot not in parse_speak_output(corrected_output)
        ]
        return state
    except Exception as correction_error:
        state["failure_reason"] = str(correction_error)
        print(
            "[ILLUST_CONTEXT:CALL3-RECOVERY] CALL3 교정/라우팅 폴백 소진, "
            "최초 결과에서 안전한 슬롯별 대사만 복구하고 이미지 파이프라인 계속: "
            f"slots={selected_slots}, error={type(correction_error).__name__}: "
            f"{correction_error}"
        )
        traceback.print_exc()
        recovered, recovery = recover_call3_partial_output(
            initial_output,
            selected_slots,
            character_names,
            speak_language,
        )
        removed_entries = list(recovery["removed_entries"])
        state.update({
            "output": recovered,
            "partial_recovery_used": True,
            "dialogue_drop_recovery_used": bool(removed_entries),
            "dropped_dialogue_entries": removed_entries,
            "silent_slots": recovery["silent_slots"],
        })
        return state


class ParallelPipelineJobsError(RuntimeError):
    """Parallel group failure that retains every independently valid result."""

    def __init__(self, message: str, resolved_values: dict[int, dict], failures: dict[int, str]):
        super().__init__(message)
        self.resolved_values = dict(resolved_values)
        self.failures = dict(failures)


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
) -> list[dict]:
    """Run ordered LLM jobs with a shared cap and translation-style tail hedging."""
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

    semaphore = asyncio.Semaphore(concurrency)
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
            async with semaphore:
                attempt_progress["started_at"] = time.monotonic()
                value = await invoke(
                    job,
                    index,
                    len(jobs),
                    attempt_kind,
                    observe_stream if hedge_active else None,
                    state["history_ids"][attempt_kind],
                    job_stream_notify,
                )
            raw = str(value.get("raw") or "") if isinstance(value, dict) else str(value or "")
            return {
                "ok": True,
                "value": value,
                "raw": raw,
                "output_length": len(raw),
                "attempt_kind": attempt_kind,
                "completed_at": time.monotonic(),
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
                "attempt_kind": attempt_kind,
                "completed_at": time.monotonic(),
            }

    pending: set[asyncio.Task] = set()
    task_metadata: dict[asyncio.Task, tuple[int, str]] = {}
    resolved: dict[int, dict] = {}
    failed: dict[int, str] = {}

    def start_attempt(index: int, attempt_kind: str) -> None:
        task = asyncio.create_task(run_attempt(index, attempt_kind))
        pending.add(task)
        task_metadata[task] = (index, attempt_kind)
        states[index]["tasks"].add(task)

    for job_index in range(1, len(jobs) + 1):
        start_attempt(job_index, "primary")

    try:
        while pending:
            done, waiting = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            pending = set(waiting)
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
                        "attempt_kind": attempt_kind,
                        "completed_at": time.monotonic(),
                    }
                completed.append((float(outcome["completed_at"]), index, outcome))

            for _completed_at, index, outcome in sorted(completed):
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
                    continue
                state["failure_reasons"].append(str(outcome.get("reason") or "unknown_failure"))

            for index, state in states.items():
                if index in resolved or index in failed or state["tasks"]:
                    continue
                failed[index] = state["failure_reasons"][-1] if state["failure_reasons"] else "unknown_failure"

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
        print(f"[ILLUST_CONTEXT:{group_id}] 병렬 조정 상위 작업 취소: pending={len(pending)}")
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    except Exception as e:
        print(f"[ILLUST_CONTEXT:{group_id}] 병렬 조정 예외: {e}")
        traceback.print_exc()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
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
    """Merge disjoint CALL1 shard JSON without semantic keyword inference."""
    merged = {
        "reference_assignments": [],
        "history_characters": [],
        "current_characters": [],
        "wardrobe_events": [],
        "visual_base_events": [],
        "hairstyle_events": [],
        "unresolved_references": [],
    }
    warnings = []
    fallback_errors = []
    history_seen = set()
    current_by_name: dict[str, dict] = {}
    assignment_by_key: dict[tuple, dict] = {}
    wardrobe_seen = set()
    visual_base_seen = set()
    hairstyle_seen = set()
    unresolved_seen = set()
    segment_rank = {segment_id: index for index, segment_id in enumerate(segment_order)}

    for shard_index, value in enumerate(shard_values, start=1):
        assigned_ids = set(value.get("assigned_segment_ids") or [])
        raw = value.get("value") if isinstance(value.get("value"), dict) else {}

        for item in raw.get("history_characters") or []:
            name = str(item.get("name") if isinstance(item, dict) else item or "").strip()
            folded = name.casefold()
            if name and folded not in history_seen:
                history_seen.add(folded)
                merged["history_characters"].append(name)

        for item in raw.get("current_characters") or []:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                try:
                    confidence = max(0.0, min(1.0, float(item.get("confidence", 1.0))))
                except (TypeError, ValueError):
                    confidence = 0.0
            else:
                name = str(item or "").strip()
                confidence = 1.0
            if not name:
                continue
            folded = name.casefold()
            previous = current_by_name.get(folded)
            if previous is None or confidence > float(previous.get("confidence") or 0.0):
                current_by_name[folded] = {"name": name, "confidence": confidence}

        for item in raw.get("reference_assignments") or []:
            if not isinstance(item, dict):
                warnings.append(f"CALL1 shard {shard_index} 지칭 할당 형식 오류로 폐기")
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id not in assigned_ids:
                warnings.append(
                    f"CALL1 shard {shard_index} 담당 밖 지칭 할당 폐기: segment={segment_id!r}"
                )
                continue
            key = (
                segment_id,
                str(item.get("surface") or ""),
                int(item.get("occurrence") or 1),
            )
            previous = assignment_by_key.get(key)
            if previous is not None and str(previous.get("canonical_name") or "").casefold() != str(
                item.get("canonical_name") or item.get("name") or ""
            ).casefold():
                fallback_errors.append(f"CALL1 shard 지칭 충돌: key={key!r}")
                continue
            assignment_by_key[key] = deepcopy(item)

        for item in raw.get("wardrobe_events") or []:
            if not isinstance(item, dict):
                warnings.append(f"CALL1 shard {shard_index} 복장 이벤트 형식 오류로 폐기")
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id and segment_id not in assigned_ids:
                warnings.append(
                    f"CALL1 shard {shard_index} 담당 밖 복장 이벤트 폐기: segment={segment_id!r}"
                )
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in wardrobe_seen:
                wardrobe_seen.add(key)
                merged["wardrobe_events"].append(deepcopy(item))

        for item in raw.get("visual_base_events") or []:
            if not isinstance(item, dict):
                warnings.append(f"CALL1 shard {shard_index} 외형 기반 이벤트 형식 오류로 폐기")
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id and segment_id not in assigned_ids:
                warnings.append(
                    f"CALL1 shard {shard_index} 담당 밖 외형 기반 이벤트 폐기: "
                    f"segment={segment_id!r}"
                )
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in visual_base_seen:
                visual_base_seen.add(key)
                merged["visual_base_events"].append(deepcopy(item))

        for item in raw.get("hairstyle_events") or []:
            if not isinstance(item, dict):
                warnings.append(f"CALL1 shard {shard_index} 헤어스타일 이벤트 형식 오류로 폐기")
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id and segment_id not in assigned_ids:
                warnings.append(
                    f"CALL1 shard {shard_index} 담당 밖 헤어스타일 이벤트 폐기: segment={segment_id!r}"
                )
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in hairstyle_seen:
                hairstyle_seen.add(key)
                merged["hairstyle_events"].append(deepcopy(item))

        for item in raw.get("unresolved_references") or []:
            if not isinstance(item, dict):
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id and segment_id not in assigned_ids:
                warnings.append(
                    f"CALL1 shard {shard_index} 담당 밖 미해결 지칭 폐기: segment={segment_id!r}"
                )
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in unresolved_seen:
                unresolved_seen.add(key)
                merged["unresolved_references"].append(deepcopy(item))

    merged["current_characters"] = list(current_by_name.values())
    merged["reference_assignments"] = sorted(
        assignment_by_key.values(),
        key=lambda item: (
            segment_rank.get(str(item.get("segment_id") or ""), len(segment_rank)),
            str(item.get("surface") or ""),
            int(item.get("occurrence") or 1),
        ),
    )
    merged["wardrobe_events"].sort(
        key=lambda item: segment_rank.get(
            str(item.get("segment_id") or ""),
            len(segment_rank),
        )
    )
    merged["visual_base_events"].sort(
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
        f"[ILLUST_CONTEXT:CALL1_PARALLEL] 분석 작업 준비: "
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
            "but emit reference_assignments, visual_base_events, wardrobe_events, "
            "hairstyle_events, and unresolved_references only "
            f"for these assigned segment IDs: {json.dumps(assigned, ensure_ascii=False)}.\n"
            "Emit history_characters and current_characters only for characters relevant to those "
            "assigned segments; the server unions all shard lists. Do not repeat the global roster in "
            "every shard. Use canonical-name strings and omit optional default-valued fields as allowed "
            "by the existing schema. Return one JSON object only."
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
                return False, "CALL1 shard JSON object 없음"
            for key in (
                "reference_assignments",
                "history_characters",
                "current_characters",
                "wardrobe_events",
                "visual_base_events",
                "hairstyle_events",
                "unresolved_references",
            ):
                if not isinstance(raw.get(key, []), list):
                    return False, f"CALL1 shard {key}가 list가 아님"
            return True, ""

        call_name = f"CALL1 {index}/{total}"
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
            raise ValueError(f"CALL1 shard {index}/{total} JSON 파싱 실패")
        return {
            "raw": raw_output,
            "value": raw_value,
            "assigned_segment_ids": assigned,
        }

    shard_values = await _run_parallel_pipeline_jobs(
        jobs,
        group_id="CALL1_PARALLEL",
        group_label="CALL1 병렬 분석",
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
    merged, merge_warnings, merge_fallback_errors = _merge_call1_shard_values(
        shard_values,
        segment_ids,
    )
    return json.dumps(merged, ensure_ascii=False), merge_warnings, merge_fallback_errors


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
            raw_region["outfit_state"] = deepcopy(
                character.get("outfit_state") or {}
            )
        if ignored_rewrites:
            print(
                "[ILLUST_CONTEXT:MULTI_CHAR] 모델의 캐릭터 재작성 결과 무시: "
                f"characters={ignored_rewrites}"
            )
    return multi_char_mask.validate_multi_char_layout(
        value,
        expected_names,
        require_prompt_separation=True,
        require_character_prompt=True,
        max_pairwise_overlap_ratio=CALL5_MAX_PAIRWISE_OVERLAP_RATIO,
    )


async def calculate_multi_char_layouts(
    descriptors: list[dict],
    prompt_template: str,
    stream_notify=None,
    positive_note: str = "",
) -> None:
    """CALL3 뒤 2~3인 장면의 영역과 배경/캐릭터 프롬프트를 병렬 계산한다."""
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
        elif len(characters) >= 2:
            targets.append((descriptor, characters))
    if not targets:
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
            "slot": slot,
            "camera": str(descriptor.get("camera") or ""),
            "scene": str(descriptor.get("scene") or ""),
            "supplement": str(descriptor.get("supplement") or ""),
            "speak": str(descriptor.get("speak") or ""),
            "characters": [{
                "name": str(character.get("name") or ""),
                "position_hint": str(character.get("position") or ""),
                "validated_positive": str(character.get("positive") or ""),
                "validated_negative": str(character.get("negative") or ""),
                "outfit_state": deepcopy(character.get("outfit_state") or {}),
            } for character in characters],
        }
        clean_positive_note = str(positive_note or "").strip()
        if clean_positive_note:
            scene_payload["positive_note"] = clean_positive_note
        descriptor["multi_char_layout_request"] = deepcopy(scene_payload)
        descriptor.pop("multi_char_layout_raw_response", None)
        messages = [{
            "role": "system",
            "content": system_prompt,
        }, {
            "role": "user",
            "content": json.dumps(scene_payload, ensure_ascii=False),
        }]

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
                    "group_label": "다중 캐릭터 마스크",
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

    await asyncio.gather(*(
        calculate_one(index, descriptor, characters)
        for index, (descriptor, characters) in enumerate(targets, start=1)
    ))


async def backtranslate_current_context(
    source: str,
    prompt: str,
    character_names: str,
    max_concurrency: int,
    failure_strategy: str = "fallback",
    stream_notify=None,
    slow_retry_enabled: bool = False,
    slow_retry_remaining: int = 1,
    slow_retry_progress_enabled: bool = True,
    slow_retry_progress_threshold: int = 50,
    slow_retry_tps_enabled: bool = False,
    slow_retry_tps_threshold: float = 5.0,
    slow_retry_condition_operator: str = "and",
) -> tuple[str, list[dict]]:
    """current context를 병렬 역번역하고 느린 꼬리 요청은 선택적으로 복제한다."""
    strategy = str(failure_strategy or "").strip().lower()
    if strategy not in ("fallback", "retry_abort"):
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE] 호출 시 실패 전략이 무효함: "
            f"value={failure_strategy!r}; fallback 사용"
        )
        strategy = "fallback"
    strict = strategy == "retry_abort"
    chunks = split_backtranslation_chunks(source, max_concurrency)
    if not chunks:
        print("[ILLUST_CONTEXT:BACKTRANSLATE] 번역할 청크가 없어 원문 사용")
        return str(source or ""), []

    try:
        parsed_concurrency = max(1, min(16, int(max_concurrency)))
        remaining_limit = max(1, min(16, int(slow_retry_remaining)))
        progress_threshold = max(
            1,
            min(99, int(slow_retry_progress_threshold)),
        )
        tps_threshold = max(
            0.1,
            min(1000.0, float(slow_retry_tps_threshold)),
        )
    except (TypeError, ValueError) as e:
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 느린 요청 재시도 설정 파싱 실패: "
            f"max_concurrency={max_concurrency!r}, remaining={slow_retry_remaining!r}, "
            f"progress_threshold={slow_retry_progress_threshold!r}, "
            f"tps_threshold={slow_retry_tps_threshold!r}, error={e}; "
            "기본값 concurrency=1, remaining=1, progress_threshold=50, "
            "tps_threshold=5.0 사용"
        )
        traceback.print_exc()
        parsed_concurrency = 1
        remaining_limit = 1
        progress_threshold = 50
        tps_threshold = 5.0
    condition_operator = str(slow_retry_condition_operator or "").strip().lower()
    if condition_operator not in ("and", "or"):
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 조건 결합 방식이 무효함: "
            f"value={slow_retry_condition_operator!r}; and 사용"
        )
        condition_operator = "and"
    slow_retry_active = bool(slow_retry_enabled) and parsed_concurrency >= 2 and len(chunks) >= 2
    if slow_retry_enabled and parsed_concurrency < 2:
        print(
            "[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 최대 병렬 개수가 2 미만이라 "
            f"느린 요청 재시도를 비활성화함: max_concurrency={parsed_concurrency}"
        )
    elif slow_retry_enabled and len(chunks) < 2:
        print(
            "[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 실제 역번역 청크가 2개 미만이라 "
            f"느린 요청 재시도를 건너뜀: chunks={len(chunks)}"
        )

    template = str(prompt or "").strip()
    if not template:
        print(
            "[ILLUST_CONTEXT:BACKTRANSLATE] backtranslate.txt가 비어 있어 "
            f"번역 불가: strategy={strategy}, chunks={len(chunks)}"
        )
        if strict:
            raise RuntimeError(
                "CALL1 역번역 엄격 전략 실패: backtranslate.txt가 비어 있습니다"
            )
        return str(source or ""), [
            {"index": index, "status": "fallback", "reason": "prompt_empty"}
            for index in range(1, len(chunks) + 1)
        ]

    names = str(character_names or "").strip()
    if not names:
        print(
            "[ILLUST_CONTEXT:BACKTRANSLATE] 활성 봇 캐릭터 이름 목록이 비어 있음; "
            "이름 사전 없이 번역 계속"
        )

    states = {
        index: {
            "tasks": set(),
            "duplicate_started": False,
            "hedge_evaluated": False,
            "failure_reasons": [],
            "failure_attempts": 0,
            "attempt_outcomes": {},
            "race_result": None,
            "execution_group_id": uuid.uuid4().hex if slow_retry_active else "",
            "history_ids": {
                "primary": uuid.uuid4().hex if slow_retry_active else "",
                "duplicate": "",
            },
            "progress": {
                "primary": {
                    "streaming": False,
                    "stream_id": "",
                    "partial_length": 0,
                    "started_at": 0.0,
                },
                "duplicate": {
                    "streaming": False,
                    "stream_id": "",
                    "partial_length": 0,
                    "started_at": 0.0,
                },
            },
        }
        for index in range(1, len(chunks) + 1)
    }

    async def run_translation_attempt(
        index: int,
        chunk: str,
        attempt_kind: str,
    ) -> dict:
        system_prompt = template
        replacements = {
            "{character_names}": names or "(none)",
            "{chunk_index}": str(index),
            "{chunk_total}": str(len(chunks)),
        }
        for marker, value in replacements.items():
            system_prompt = system_prompt.replace(marker, value)
        if names and "{character_names}" not in template:
            system_prompt += "\n\n# Protected character names\n" + names

        protected_chunk, protected_markers = _protect_slot_markers(chunk)

        messages = _normalize_messages([
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"[Current Response Chunk {index}/{len(chunks)}]\n"
                    "Return only the English translation of the chunk body below.\n\n"
                    "Tokens shaped like __SLOT_0__ are server-protected "
                    "slot markers. Copy every token exactly once and in the same order.\n\n"
                    + protected_chunk
                ),
            },
        ])
        last_reason = "unknown_failure"
        validation_attempts = 0

        def _restore_and_validate(translated):
            value = str(translated or "")
            if len(value.strip()) == 0:
                return value, False, "응답 길이가 0임"
            restored, protection_valid, protection_reason = _restore_slot_markers(
                value,
                protected_markers,
            )
            if not protection_valid:
                return restored, False, protection_reason
            valid, reason = _valid_backtranslation(chunk, restored)
            return restored, valid, reason

        def _validate_translation(translated):
            nonlocal validation_attempts, last_reason
            validation_attempts += 1
            _restored, valid, reason = _restore_and_validate(translated)
            last_reason = reason
            if not valid:
                print(
                    f"[ILLUST_CONTEXT:BACKTRANSLATE] 청크 응답 검증 실패: "
                    f"strategy={strategy}, chunk={index}/{len(chunks)}, "
                    f"validation_attempt={validation_attempts}, input_len={len(chunk)}, "
                    f"output_len={len(str(translated or ''))}, reason={reason}"
                )
            return valid, reason

        call_name = f"CALL1-BACKTRANSLATE {index}/{len(chunks)}"
        if attempt_kind == "duplicate":
            call_name += " [느리다고? 다시해!]"
        chunk_stream_notify = None
        if stream_notify:
            async def chunk_stream_notify(event: dict):
                payload = dict(event)
                payload["queue_subtask"] = {
                    "group_id": "backtranslation",
                    "group_label": "역번역",
                    "index": index,
                    "total": len(chunks),
                }
                await stream_notify(payload)

        attempt_progress = states[index]["progress"][attempt_kind]
        attempt_progress["started_at"] = time.monotonic()

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
                    f"[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 스트림 출력 길이 파싱 실패: "
                    f"chunk={index}/{len(chunks)}, attempt={attempt_kind}, "
                    f"value={event.get('partial_length')!r}, error={e}"
                )
                traceback.print_exc()

        try:
            call_kwargs = {"result_validator": _validate_translation}
            if slow_retry_active:
                call_kwargs["stream_observer"] = observe_stream
                call_kwargs["history_id"] = states[index]["history_ids"][attempt_kind]
            translated = await _call_pipeline_llm(
                call_name,
                messages,
                chunk_stream_notify,
                **call_kwargs,
            )
        except asyncio.CancelledError:
            print(
                f"[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 선착순에서 밀린 요청 취소: "
                f"chunk={index}/{len(chunks)}, attempt={attempt_kind}"
            )
            raise
        except Exception as e:
            last_reason = f"call_failed: {e}"
            print(
                f"[ILLUST_CONTEXT:BACKTRANSLATE] 청크 호출 실패: "
                f"strategy={strategy}, chunk={index}/{len(chunks)}, "
                f"attempt={attempt_kind}, input_len={len(chunk)}, error={e}"
            )
            traceback.print_exc()
        else:
            restored, valid, reason = _restore_and_validate(translated)
            if valid:
                return {
                    "ok": True,
                    "text": restored.strip(),
                    "reason": "",
                    "validation_attempts": max(1, validation_attempts),
                    "attempt_kind": attempt_kind,
                    "completed_at": time.monotonic(),
                }
            last_reason = reason
            print(
                f"[ILLUST_CONTEXT:BACKTRANSLATE] 청크 최종 응답 실패: "
                f"strategy={strategy}, chunk={index}/{len(chunks)}, "
                f"attempt={attempt_kind}, input_len={len(chunk)}, "
                f"output_len={len(str(translated or ''))}, reason={reason}"
            )
        return {
            "ok": False,
            "text": "",
            "reason": last_reason,
            "validation_attempts": max(1, validation_attempts),
            "attempt_kind": attempt_kind,
            "completed_at": time.monotonic(),
        }

    pending: set[asyncio.Task] = set()
    task_metadata: dict[asyncio.Task, tuple[int, str]] = {}
    resolved: dict[int, dict] = {}

    def start_attempt(index: int, attempt_kind: str) -> None:
        task = asyncio.create_task(
            run_translation_attempt(index, chunks[index - 1], attempt_kind)
        )
        pending.add(task)
        task_metadata[task] = (index, attempt_kind)
        states[index]["tasks"].add(task)

    for chunk_index in range(1, len(chunks) + 1):
        start_attempt(chunk_index, "primary")

    try:
        while pending:
            done, waiting = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            pending = set(waiting)
            completed_results = []
            for task in done:
                index, attempt_kind = task_metadata.pop(task)
                states[index]["tasks"].discard(task)
                try:
                    outcome = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    print(
                        f"[ILLUST_CONTEXT:BACKTRANSLATE] 청크 작업 예외: "
                        f"chunk={index}/{len(chunks)}, attempt={attempt_kind}, error={e}"
                    )
                    traceback.print_exception(type(e), e, e.__traceback__)
                    outcome = {
                        "ok": False,
                        "text": "",
                        "reason": f"unexpected_error: {e}",
                        "validation_attempts": 1,
                        "attempt_kind": attempt_kind,
                        "completed_at": time.monotonic(),
                    }
                completed_results.append((float(outcome["completed_at"]), index, outcome))

            # 같은 이벤트 루프 틱에서 둘 다 끝났다면 실제 완료 시각이 빠른 결과를 먼저 쓴다.
            for _completed_at, index, outcome in sorted(completed_results):
                state = states[index]
                state["attempt_outcomes"][outcome["attempt_kind"]] = outcome
                if index in resolved:
                    print(
                        f"[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 중복 완료 결과 폐기: "
                        f"chunk={index}/{len(chunks)}, attempt={outcome['attempt_kind']}"
                    )
                    continue
                if outcome["ok"]:
                    status = {
                        "index": index,
                        "status": "translated",
                        "reason": "",
                        "attempts": outcome["validation_attempts"],
                    }
                    if state["duplicate_started"]:
                        winner_kind = outcome["attempt_kind"]
                        loser_kind = (
                            "duplicate" if winner_kind == "primary" else "primary"
                        )
                        loser_outcome = state["attempt_outcomes"].get(loser_kind)
                        loser_progress_state = state["progress"][loser_kind]
                        loser_streaming = bool(loser_progress_state["streaming"])
                        if loser_outcome is not None:
                            loser_progress = 100.0
                        else:
                            reference_ratios = [
                                len(str(result.get("text") or ""))
                                / max(1, len(chunks[resolved_index - 1].strip()))
                                for resolved_index, result in resolved.items()
                                if result.get("status", {}).get("status") == "translated"
                            ]
                            reference_ratios.append(
                                len(str(outcome.get("text") or ""))
                                / max(1, len(chunks[index - 1].strip()))
                            )
                            average_ratio = sum(reference_ratios) / len(reference_ratios)
                            if loser_streaming:
                                expected_length = max(
                                    1.0,
                                    len(chunks[index - 1].strip()) * average_ratio,
                                )
                                loser_progress = min(
                                    99.0,
                                    int(loser_progress_state["partial_length"])
                                    / expected_length * 100.0,
                                )
                            else:
                                loser_progress = 0.0
                        state["race_result"] = {
                            "winner": winner_kind,
                            "loser": loser_kind,
                            "loser_progress": round(loser_progress, 1),
                            "loser_streaming": loser_streaming,
                        }
                        status.update({
                            "hedged": True,
                            "winner": winner_kind,
                            "requests": 2,
                            "loser_progress": round(loser_progress, 1),
                            "loser_streaming": loser_streaming,
                        })
                    resolved[index] = {"text": outcome["text"], "status": status}
                    for sibling in list(state["tasks"]):
                        if not sibling.done():
                            sibling.cancel()
                    continue
                state["failure_reasons"].append(outcome["reason"])
                state["failure_attempts"] += outcome["validation_attempts"]

            # 더 기다릴 같은 청크 요청이 없으면 기존 엄격/원문 폴백 정책으로 마감한다.
            for index, state in states.items():
                if index in resolved or any(not task.done() for task in state["tasks"]):
                    continue
                last_reason = state["failure_reasons"][-1] if state["failure_reasons"] else "unknown_failure"
                if strict:
                    error = RuntimeError(
                        f"청크 {index}/{len(chunks)}가 설정된 라우팅 재시도 후에도 "
                        f"실패했습니다: {last_reason}"
                    )
                    resolved[index] = {"error": error}
                else:
                    print(
                        f"[ILLUST_CONTEXT:BACKTRANSLATE] 실패 청크 원문 폴백: "
                        f"chunk={index}/{len(chunks)}, reason={last_reason}"
                    )
                    resolved[index] = {
                        "text": chunks[index - 1].strip(),
                        "status": {
                            "index": index,
                            "status": "fallback",
                            "reason": last_reason,
                            "attempts": max(1, state["failure_attempts"]),
                        },
                    }

            unresolved = [
                index for index in range(1, len(chunks) + 1)
                if index not in resolved
            ]
            if (
                slow_retry_active
                and resolved
                and unresolved
                and len(unresolved) <= remaining_limit
                and not any(result.get("error") for result in resolved.values())
            ):
                completed_ratios = [
                    len(str(result.get("text") or ""))
                    / max(1, len(chunks[index - 1].strip()))
                    for index, result in resolved.items()
                    if result.get("status", {}).get("status") == "translated"
                ]
                average_ratio = (
                    sum(completed_ratios) / len(completed_ratios)
                    if completed_ratios else None
                )
                for index in unresolved:
                    state = states[index]
                    if state["hedge_evaluated"]:
                        continue
                    state["hedge_evaluated"] = True
                    primary_progress = state["progress"]["primary"]
                    streaming = bool(primary_progress["streaming"])
                    partial_length = int(primary_progress["partial_length"])
                    if streaming and average_ratio is not None:
                        expected_length = max(
                            1.0,
                            len(chunks[index - 1].strip()) * average_ratio,
                        )
                        estimated_progress = min(
                            99.0,
                            partial_length / expected_length * 100.0,
                        )
                    else:
                        # 비스트리밍은 중간 출력 자체를 알 수 없다. 사용자 설정대로
                        # 보수적으로 0%로 보며, 완료 청크가 없어 보정 불가한 경우도 같다.
                        estimated_progress = 0.0
                    started_at = float(primary_progress.get("started_at") or 0.0)
                    elapsed = max(
                        0.001,
                        time.monotonic() - started_at,
                    ) if started_at > 0 else 0.0
                    estimated_tps = (
                        (partial_length / 3.0) / elapsed
                        if streaming and elapsed > 0
                        else 0.0
                    )
                    condition_results = []
                    if bool(slow_retry_progress_enabled):
                        condition_results.append((
                            "progress",
                            estimated_progress < progress_threshold,
                        ))
                    if bool(slow_retry_tps_enabled):
                        condition_results.append((
                            "tps",
                            estimated_tps < tps_threshold,
                        ))
                    if not condition_results:
                        should_duplicate = False
                    elif condition_operator == "or":
                        should_duplicate = any(result for _name, result in condition_results)
                    else:
                        should_duplicate = all(result for _name, result in condition_results)
                    conditions_text = ", ".join(
                        f"{name}={'met' if result else 'not_met'}"
                        for name, result in condition_results
                    ) or "none_enabled"
                    if should_duplicate:
                        state["duplicate_started"] = True
                        state["history_ids"]["duplicate"] = uuid.uuid4().hex
                        mode = "streaming" if streaming else "non_streaming"
                        print(
                            f"[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 중복 요청 시작: "
                            f"chunk={index}/{len(chunks)}, remaining={len(unresolved)}, "
                            f"mode={mode}, estimated_progress={estimated_progress:.1f}%, "
                            f"progress_threshold={progress_threshold}%, "
                            f"estimated_tps={estimated_tps:.1f}, "
                            f"tps_threshold={tps_threshold:g}, "
                            f"operator={condition_operator.upper()}, "
                            f"conditions={conditions_text}"
                        )
                        start_attempt(index, "duplicate")
                    else:
                        print(
                            f"[ILLUST_CONTEXT:BACKTRANSLATE_HEDGE] 활성 조건을 만족하지 않아 "
                            f"중복 요청하지 않음: chunk={index}/{len(chunks)}, "
                            f"estimated_progress={estimated_progress:.1f}%, "
                            f"progress_threshold={progress_threshold}%, "
                            f"estimated_tps={estimated_tps:.1f}, "
                            f"tps_threshold={tps_threshold:g}, "
                            f"operator={condition_operator.upper()}, "
                            f"conditions={conditions_text}"
                        )
    except asyncio.CancelledError:
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE] 병렬 역번역 상위 작업 취소: "
            f"pending={len(pending)}/{len(chunks)}"
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    except Exception as e:
        print(f"[ILLUST_CONTEXT:BACKTRANSLATE] 병렬 역번역 조정 예외: {e}")
        traceback.print_exc()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise

    history_updates = {}
    for index, state in states.items():
        if not state["duplicate_started"]:
            continue
        base_call_name = f"CALL1-BACKTRANSLATE {index}/{len(chunks)}"
        race_result = state.get("race_result")
        if race_result:
            winner_kind = race_result["winner"]
            loser_kind = race_result["loser"]
            loser_progress = float(race_result["loser_progress"])
            loser_streaming = bool(race_result["loser_streaming"])
            for attempt_kind, role_label in (
                ("primary", "원본"),
                ("duplicate", "느리다고? 다시해!"),
            ):
                attempt_history_id = state["history_ids"].get(attempt_kind, "")
                if not attempt_history_id:
                    continue
                if attempt_kind == winner_kind:
                    history_updates[attempt_history_id] = {
                        "call_name": f"{base_call_name} [{role_label} · 승리]",
                        "status": "race_won",
                        "race_outcome": "winner",
                        "parent_execution_id": state["execution_group_id"],
                    }
                elif attempt_kind == loser_kind:
                    if loser_streaming:
                        progress_label = f"{loser_progress:g}%"
                    else:
                        progress_label = "0% (비스트리밍)"
                    history_updates[attempt_history_id] = {
                        "call_name": (
                            f"{base_call_name} [{role_label} · 패배 · "
                            f"진행률 {progress_label}]"
                        ),
                        "status": "race_lost",
                        "race_outcome": "loser",
                        "race_progress": loser_progress,
                        "race_streaming": loser_streaming,
                        "parent_execution_id": state["execution_group_id"],
                    }
        else:
            for attempt_kind, role_label in (
                ("primary", "원본"),
                ("duplicate", "느리다고? 다시해!"),
            ):
                attempt_history_id = state["history_ids"].get(attempt_kind, "")
                if attempt_history_id:
                    history_updates[attempt_history_id] = {
                        "call_name": f"{base_call_name} [{role_label} · 경주 실패]",
                        "race_outcome": "failed",
                        "parent_execution_id": state["execution_group_id"],
                    }
    if history_updates:
        lighbd_service._update_lighbd_history_records(history_updates)

    strict_failures = [
        (index, result["error"])
        for index, result in sorted(resolved.items())
        if result.get("error") is not None
    ]
    if strict_failures:
        for index, error in strict_failures:
            print(
                f"[ILLUST_CONTEXT:BACKTRANSLATE] 청크 처리 예외: "
                f"strategy={strategy}, chunk={index}/{len(chunks)}, error={error}"
            )
            traceback.print_exception(type(error), error, error.__traceback__)
        raise RuntimeError(
            f"CALL1 역번역 엄격 전략 실패: "
            f"{len(strict_failures)}/{len(chunks)}개 청크 실패"
        ) from strict_failures[0][1]

    results = [resolved[index] for index in range(1, len(chunks) + 1)]
    translated_chunks = [result["text"] for result in results]
    statuses = [result["status"] for result in results]
    translated_count = sum(1 for status in statuses if status["status"] == "translated")
    fallback_count = len(statuses) - translated_count
    print(
        f"[ILLUST_CONTEXT:BACKTRANSLATE] 병렬 역번역 완료: "
        f"chunks={len(chunks)}, translated={translated_count}, fallback={fallback_count}"
    )
    return "\n\n".join(translated_chunks), statuses


def build_raw_prompt(descriptor: dict, narrative: str, prompts: dict, toggles: dict) -> tuple[str, str]:
    template = prompts.get("call2_preset") or "[Positive]\n[SETUP]\n{setup}\n[CHAR]\n{char}\n[SUPPLEMENT]\n{supplement}\n\n[Negative]\n"
    positive_part, marker, negative_part = template.partition("[Negative]")
    chars = descriptor.get("characters") or []
    divider = "\n\n" if toggles.get("compat_character_divider") == "newline" else " | "
    char_positive = divider.join(str(ch.get("positive") or "") for ch in chars if str(ch.get("positive") or "").strip())
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
        "{speak}": descriptor.get("speak") or "None",
        "{name}": names,
        "{setup}": setup,
        "{prompt}": setup,
        "{char}": char_positive,
        "{supplement}": supplement,
    }
    positive = positive_part
    for key, value in replacements.items():
        positive = positive.replace(key, str(value))
    if positive_note:
        positive = positive.rstrip() + "\n" + positive_note
    negative = negative_part.strip() if marker else ""
    if char_negative:
        negative = ", ".join(x for x in (negative, char_negative) if x)
    if str(toggles.get("negative_note") or "").strip():
        negative = ", ".join(x for x in (negative, str(toggles["negative_note"]).strip()) if x)
    return positive.strip(), negative.strip()


async def build_from_context(
    payload: dict,
    toggles: dict | None,
    extra_reference: str,
    progress=None,
    stream_notify=None,
    on_call2_ready=None,
    extra_instruction: str = "",
    extra_costume: str = "",
    extra_names: str = "",
    backtranslate_names: str = "",
    enable_multi_char_layout: bool = False,
    history_plan: dict | None = None,
    visual_profile_catalog: str = "",
    visual_profiles: dict[str, dict] | None = None,
) -> dict:
    toggles = merged_toggles(toggles)
    prompts = load_prompt_files()
    # 이번 삽화 생성(MULTI-CHAR-MASK~CALL3)의 모든 LLM 호출 history_id를 모은다.
    # _call_pipeline_llm 가 각 레코드 id를 여기에 append 한다.
    trace: list[str] = []
    _llm_trace_token = _llm_trace_ctx.set(trace)
    chats = payload.get("chats") or []
    target_index, narrative = _latest_narrative(chats)
    if target_index < 0 or not narrative:
        raise RuntimeError("CHAT에서 최신 CHAR 서사를 찾지 못했습니다")

    # 역번역 대상은 최신 CHAR 응답(current context) 하나뿐이다. 과거 CHAT은 원문을
    # 유지하고, 슬롯 위치 앵커도 원문 응답을 계속 사용한다.
    original_slotted = str(payload.get("target_slotted") or "").strip()
    if not original_slotted:
        original_slotted = insert_slots(_strip_nodes(narrative))
    slotted = original_slotted
    backtranslation_chunks = []
    backtranslated_narrative = ""
    backtranslation_failure_strategy = str(
        toggles["call1_backtranslate_failure_strategy"]
    )
    if toggles.get("call1_backtranslate_enabled"):
        if progress:
            await progress(3, "call1_backtranslate", "CALL1 역번역 준비")
        slotted, backtranslation_chunks = await backtranslate_current_context(
            original_slotted,
            prompts.get("call1_backtranslate", ""),
            backtranslate_names,
            int(toggles["call1_backtranslate_max_concurrency"]),
            failure_strategy=backtranslation_failure_strategy,
            stream_notify=stream_notify,
            slow_retry_enabled=bool(
                toggles["call1_backtranslate_slow_retry_enabled"]
            ),
            slow_retry_remaining=int(
                toggles["call1_backtranslate_slow_retry_remaining"]
            ),
            slow_retry_progress_enabled=bool(
                toggles["call1_backtranslate_slow_retry_progress_enabled"]
            ),
            slow_retry_progress_threshold=int(
                toggles["call1_backtranslate_slow_retry_progress_threshold"]
            ),
            slow_retry_tps_enabled=bool(
                toggles["call1_backtranslate_slow_retry_tps_enabled"]
            ),
            slow_retry_tps_threshold=float(
                toggles["call1_backtranslate_slow_retry_tps_threshold"]
            ),
            slow_retry_condition_operator=str(
                toggles["call1_backtranslate_slow_retry_condition_operator"]
            ),
        )
        backtranslated_narrative = remove_slot_markers(slotted)
        if not backtranslated_narrative:
            print(
                "[ILLUST_CONTEXT:BACKTRANSLATE] 병합 결과의 본문 길이가 0이어서 "
                f"후속 처리 불가: strategy={backtranslation_failure_strategy}"
            )
            if backtranslation_failure_strategy == "retry_abort":
                raise RuntimeError(
                    "CALL1 역번역 엄격 전략 실패: 병합 결과의 본문 길이가 0입니다"
                )
            slotted = original_slotted
            backtranslated_narrative = _strip_nodes(narrative)
            backtranslation_chunks = [{
                "index": 0,
                "status": "fallback",
                "reason": "merged_body_empty",
            }]
    else:
        print("[ILLUST_CONTEXT:BACKTRANSLATE] 토글로 비활성화됨")
    backtranslated_slotted = slotted if toggles.get("call1_backtranslate_enabled") else ""

    if progress:
        await progress(5, "call1", "CALL1 컨텍스트 준비")
    call1_output = ""
    call1_result: dict = {}
    wardrobe_events: list[dict] = []
    visual_base_events: list[dict] = []
    hairstyle_events: list[dict] = []
    reference_variables: dict[str, str] = {}
    balanced_fallback = False
    enhanced = backtranslated_narrative or _strip_nodes(narrative)
    resolved_current = enhanced
    segmented_current, current_segments = _segment_current_context(enhanced)
    persistent_history = history_plan if isinstance(history_plan, dict) else None
    reference_provenance = build_reference_provenance(persistent_history)
    if toggles.get("call1_enabled"):
        if persistent_history:
            context_slice = persistent_history.get("call1_history") or []
        else:
            n = int(toggles["call1_context_turns"])
            context_slice = chats[max(0, target_index - n):target_index]
        # CALL1에는 lb.extra 중 시스템 프롬프트를 빼고 캐릭터 복장 정보만 넘긴다.
        # enhance 프롬프트의 {lb_extra_costume} 자리표시자를 치환한다.
        # (자리표시자가 없으면 복장 정보를 뒤에 덧붙여 정보 유실을 막는다.)
        call1_system = prompts.get("call1_enhance", "")
        costume = str(extra_costume or "").strip()
        if "{lb_extra_costume}" in call1_system:
            call1_system = call1_system.replace("{lb_extra_costume}", costume)
        elif costume:
            call1_system = call1_system + "\n\n" + costume
        if "{visual_profile_catalog}" in call1_system:
            call1_system = call1_system.replace(
                "{visual_profile_catalog}",
                str(visual_profile_catalog or "").strip() or "(none)",
            )
        elif str(visual_profile_catalog or "").strip():
            call1_system += (
                "\n\n# REGISTERED CHARACTER CARDS\n"
                + str(visual_profile_catalog).strip()
            )
        call1_system = call1_system.replace("{character_names}", str(backtranslate_names or extra_names or ""))
        call1_system = call1_system.replace(
            "{character_state}",
            json.dumps(
                _call1_state_for_prompt(
                    (persistent_history or {}).get("state_before") or {},
                    costume,
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )
        history_text = _history_messages_text(context_slice)
        parallel_call1_used = False
        parallel_merge_warnings: list[str] = []
        parallel_merge_fallback_errors: list[str] = []
        should_parallel_call1 = (
            bool(toggles.get("call1_parallel_enabled"))
            and len(current_segments) > 1
        )
        if should_parallel_call1:
            try:
                (
                    call1_output,
                    parallel_merge_warnings,
                    parallel_merge_fallback_errors,
                ) = await _run_parallel_call1_analysis(
                    call1_system=call1_system,
                    segmented_current=segmented_current,
                    current_segments=current_segments,
                    history_text=history_text,
                    toggles=toggles,
                    stream_notify=stream_notify,
                )
                parallel_call1_used = True
            except Exception as e:
                print(
                    f"[ILLUST_CONTEXT:CALL1_PARALLEL] 병렬 분석 실패로 단일 CALL1 폴백: "
                    f"segments={len(current_segments)}, error={e}"
                )
                traceback.print_exc()

        if not parallel_call1_used:
            call1_messages = [{"role": "system", "content": call1_system}]
            if persistent_history:
                call1_messages.append({
                    "role": "user",
                    "content": (
                        "# PAST HISTORY\n"
                        + (history_text or "(empty)")
                        + "\n\n# CURRENT CONTEXT SEGMENTS\n"
                        + segmented_current
                    ),
                })
            else:
                call1_messages.extend({
                    "role": "assistant" if item["role"] == "char" else "user",
                    "content": _strip_nodes(item["data"]),
                } for item in context_slice)
                call1_messages.append({"role": "user", "content": "---\n[Current Response]\n" + enhanced})
            call1_output = await _call_pipeline_llm(
                "CALL1",
                _normalize_messages(call1_messages),
                stream_notify,
            )

        if persistent_history or parallel_call1_used:
            parsed_call1 = parse_call1_analysis(
                call1_output,
                enhanced,
                current_segments,
                str(backtranslate_names or extra_names or ""),
                history_context=history_text,
                visual_profiles=visual_profiles,
            )
            if parsed_call1 is None:
                balanced_fallback = True
                print(
                    "[ILLUST_CONTEXT:CALL1] 구조화 분석 실패로 균형형 CALL2 폴백 사용"
                )
            else:
                call1_result = parsed_call1
                if parallel_merge_warnings:
                    parsed_call1["validation_warnings"].extend(parallel_merge_warnings)
                    parsed_call1["validation_errors"].extend(parallel_merge_warnings)
                    print(
                        f"[ILLUST_CONTEXT:CALL1_PARALLEL] shard 병합 경고(개별 항목 폐기): "
                        f"warnings={parallel_merge_warnings}"
                    )
                if parallel_merge_fallback_errors:
                    parsed_call1["fallback_errors"].extend(
                        parallel_merge_fallback_errors
                    )
                    parsed_call1["validation_errors"].extend(
                        parallel_merge_fallback_errors
                    )
                    parsed_call1["fallback_required"] = True
                    print(
                        f"[ILLUST_CONTEXT:CALL1_PARALLEL] shard 병합 치명 오류: "
                        f"errors={parallel_merge_fallback_errors}"
                    )
                wardrobe_events = list(parsed_call1.get("wardrobe_events") or [])
                visual_base_events = list(
                    parsed_call1.get("visual_base_events") or []
                )
                hairstyle_events = list(parsed_call1.get("hairstyle_events") or [])
                resolved_current, assignment_errors, reference_variables = apply_reference_assignments(
                    enhanced,
                    current_segments,
                    parsed_call1.get("reference_assignments") or [],
                )
                if assignment_errors:
                    parsed_call1["validation_warnings"].extend(assignment_errors)
                    parsed_call1["validation_errors"].extend(assignment_errors)
                    print(
                        f"[ILLUST_CONTEXT:CALL1] 지칭 치환 경고(개별 항목 폐기): "
                        f"warnings={assignment_errors}"
                    )
                balanced_fallback = bool(parsed_call1.get("fallback_required"))
                validation_warnings = parsed_call1.get("validation_warnings") or []
                if validation_warnings:
                    print(
                        f"[ILLUST_CONTEXT:CALL1] 복구 가능한 검증 경고: "
                        f"warnings={validation_warnings}"
                    )
                if balanced_fallback:
                    print(
                        f"[ILLUST_CONTEXT:CALL1] 균형형 폴백 조건 감지: "
                        f"errors={parsed_call1.get('fallback_errors') or []}"
                    )
                enhanced = resolved_current
                slotted, slotted_assignment_errors = apply_reference_assignments_to_slotted(
                    slotted,
                    current_segments,
                    parsed_call1.get("reference_assignments") or [],
                )
                if slotted_assignment_errors:
                    parsed_call1["validation_warnings"].extend(
                        slotted_assignment_errors
                    )
                    parsed_call1["validation_errors"].extend(slotted_assignment_errors)
                    print(
                        f"[ILLUST_CONTEXT:CALL1] 슬롯 보존 지칭 치환 경고: "
                        f"warnings={slotted_assignment_errors}"
                    )
        else:
            enhanced = _splice_enhancements(enhanced, call1_output)
    else:
        print("[ILLUST_CONTEXT:CALL1] 토글로 비활성화됨")
        balanced_fallback = bool(persistent_history)

    # Risu는 결과 메타데이터를 텍스트로 받을 수 없고 generateImage만 반복 호출한다.
    # 결과를 slot 번호로 회수할 수 있도록 슬롯 번호는 원문 문단을 기준으로 고정한다.
    # 모듈이 v13 규칙으로 원문 XML을 보존한 채 먼저 삽입한 슬롯 맵을 우선한다.
    # 없으면 구버전/테스트 호환을 위해 필터된 최신 narrative로 생성한다.
    if call1_output and not call1_result:
        # 일반 CALL1에는 슬롯을 노출하지 않는다. CALL1이 반환한 Position 범위를
        # 서버가 보관한 슬롯 본문에 투영해 [Slot N]과 [Position]을 함께 보존한다.
        slotted = _merge_call1_output_into_slotted(slotted, call1_output)
    if progress:
        await progress(30, "call2", "CALL2 장면/태그 빌드")
    current_character_names = [
        str(item.get("name") or "").strip()
        for item in call1_result.get("current_characters") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    call2_instruction = str(extra_instruction or "").strip()
    call2_reference = str(extra_reference or "")
    selected_states = {}
    previous_visual = {}
    if persistent_history and toggles.get("call1_enabled") and call1_result and not balanced_fallback:
        call2_reference = _filter_character_reference(extra_reference, current_character_names)
        state_before = persistent_history.get("state_before") or {}
        selected_states = _selected_character_states(
            state_before,
            current_character_names,
        )
        previous_visual = {
            name: value
            for name, value in (
                ((persistent_history.get("record_before") or {}).get("last_pipeline") or {})
                .get("last_visual_by_character", {})
            ).items()
            if str(name).casefold() in {value.casefold() for value in current_character_names}
        }
        history_character_names = {
            str(name or "").strip().casefold()
            for name in call1_result.get("history_characters") or []
            if str(name or "").strip()
        }
        selected_state_names = {
            str((value or {}).get("canonical_name") or key).strip().casefold()
            for key, value in selected_states.items()
            if isinstance(value, dict)
        }
        missing_historical_state = sorted(
            name
            for name in current_character_names
            if name.casefold() in history_character_names
            and (
                name.casefold() not in selected_state_names
                or str(next((
                    (value.get("current_wardrobe") or {}).get("body_state")
                    for value in selected_states.values()
                    if isinstance(value, dict)
                    and str(value.get("canonical_name") or "").casefold() == name.casefold()
                ), "unknown") or "unknown").strip().lower() in ("", "unknown")
            )
        )
        if missing_historical_state:
            balanced_fallback = True
            call2_reference = str(extra_reference or "")
            selected_states = deepcopy(state_before)
            print(
                "[ILLUST_CONTEXT:CALL2] 과거 등장 캐릭터의 추적 복장이 없어 "
                f"균형형 히스토리 폴백 사용: characters={missing_historical_state}"
            )
        if current_character_names and not call2_reference.strip():
            balanced_fallback = True
            call2_reference = str(extra_reference or "")
            selected_states = deepcopy(state_before)
            print(
                "[ILLUST_CONTEXT:CALL2] 선택 캐릭터 사전이 비어 균형형 전체 lb.extra 폴백 사용: "
                f"characters={current_character_names}"
            )
    elif persistent_history:
        selected_states = deepcopy(persistent_history.get("state_before") or {})

    if persistent_history and balanced_fallback:
        previous_visual = deepcopy(
            ((persistent_history.get("record_before") or {}).get("last_pipeline") or {})
            .get("last_visual_by_character", {})
        )

    fixed_appearance = extract_authoritative_fixed_appearance(call2_reference)
    default_outfits = extract_authoritative_default_outfits(call2_reference)
    if visual_profiles and current_character_names:
        selected_states = apply_visual_base_events(
            selected_states,
            call1_result.get("current_characters") or [],
            [],
            str((persistent_history or {}).get("current_message_id") or ""),
            visual_profiles,
        )
    if persistent_history and selected_states:
        selected_states = apply_wardrobe_events(
            selected_states,
            call1_result.get("current_characters") or [],
            [],
            str(persistent_history.get("current_message_id") or ""),
            selected_reference=call2_reference,
            default_outfits=default_outfits,
        )
    last_visual_reference_classification = classify_last_visual_reference(
        reference_provenance,
        previous_visual,
    )
    classified_visual_reference = _classified_visual_reference_content(
        last_visual_reference_classification,
        previous_visual,
    )
    # The selected dictionary is already supplied as a dedicated user block and
    # fixed appearance is supplied separately. Repeating it inside the rendered
    # system prompt wastes context and makes source authority less clear.
    history = ""
    call2_system = render_call2_prompt(prompts.get("call2_system", ""), toggles, history)
    call2_thoughts = render_call2_prompt(prompts.get("call2_thoughts", ""), toggles, history)
    call2_detail_toggles = deepcopy(toggles)
    call2_detail_toggles["key_visual"] = False
    call2_detail_system = render_call2_prompt(
        prompts.get("call2_system", ""),
        call2_detail_toggles,
        history,
        include_scene_count_limit=False,
    )
    call2_detail_thoughts = render_call2_prompt(
        prompts.get("call2_thoughts", ""),
        call2_detail_toggles,
        history,
    )
    call2_keyvis_system = (
        _keyvis_only_call2_system(call2_system)
        if toggles.get("key_visual")
        else call2_system
    )
    call2_base_message = {
        "role": "system",
        "content": "\n\n".join(x for x in (
            prompts.get("call2_jailbreak", ""), prompts.get("call2_job", ""), call2_system,
        ) if x.strip()),
    }
    call2_detail_base_message = {
        "role": "system",
        "content": "\n\n".join(x for x in (
            prompts.get("call2_jailbreak", ""),
            prompts.get("call2_job", ""),
            call2_detail_system,
        ) if x.strip()),
    }
    call2_keyvis_base_message = {
        "role": "system",
        "content": "\n\n".join(x for x in (
            prompts.get("call2_jailbreak", ""),
            prompts.get("call2_job", ""),
            call2_keyvis_system,
        ) if x.strip()),
    }
    call2_messages = [deepcopy(call2_base_message)]
    call2_detail_context_messages = [deepcopy(call2_detail_base_message)]
    call2_plan_context_messages = [deepcopy(call2_base_message)]
    call2_keyvis_context_messages = [deepcopy(call2_keyvis_base_message)]

    def append_call2_context(
        message: dict,
        *,
        include_plan: bool = True,
        include_keyvis: bool = True,
        include_detail: bool = True,
    ) -> None:
        if include_detail:
            call2_messages.append(deepcopy(message))
            call2_detail_context_messages.append(deepcopy(message))
        if include_plan:
            call2_plan_context_messages.append(deepcopy(message))
        if include_keyvis:
            call2_keyvis_context_messages.append(deepcopy(message))

    if call2_instruction:
        append_call2_context({
            "role": "user",
            "content": "# ACTIVE BOT IMAGE INSTRUCTIONS\n\n" + call2_instruction,
        })
    if call2_reference.strip():
        append_call2_context({
            "role": "user",
            "content": "# CHARACTER DICTIONARY\n\n" + call2_reference,
        })
    fixed_appearance_content = _fixed_appearance_authority_content(fixed_appearance)
    if fixed_appearance_content:
        append_call2_context({
            "role": "user",
            "content": fixed_appearance_content,
        }, include_plan=False)
    if persistent_history:
        if selected_states:
            projected_states = _state_without_generated_visual_references(
                selected_states,
                source="CALL2_DETAIL_KEYVIS",
            )
            append_call2_context({
                "role": "user",
                "content": (
                    "# TRACKED WARDROBE CONTINUITY AND DEFAULT REFERENCE\n"
                    "This contains the current tracked wardrobe initialized from the default reference "
                    "plus prior sparse deltas. Preserve real continuity, but do not treat default_outfit "
                    "as fixed identity. When the assigned scene clearly calls for a different coherent "
                    "outfit, replace the fallback as a set by meaning; camera absence alone never means removal.\n\n"
                    + json.dumps(projected_states, ensure_ascii=False, indent=2)
                ),
            }, include_plan=False)
        if wardrobe_events:
            append_call2_context({
                "role": "user",
                "content": (
                    "# SPARSE CURRENT WARDROBE CHANGE HISTORY\n"
                    "Each event is a semantic instruction with `operation`, `wardrobe_change` "
                    "(a short natural-language description of what changed), and `state_after`; "
                    "`items` may be empty and that is expected. It is not a ready-made tag list and "
                    "does not contain a complete outfit. Map each `wardrobe_change` to the matching "
                    "garment cluster of the tracked outfit or default reference. Preserve compatible "
                    "continuity, but allow the full scene context to establish a coherent replacement "
                    "outfit even when it does not enumerate every garment.\n\n"
                    + json.dumps(wardrobe_events, ensure_ascii=False, indent=2)
                ),
            }, include_plan=False)
        # hairstyle history: selected_states의 누적 timeline + 이번 턴 CALL1 이벤트를 합쳐
        # CALL2에 전달한다(서버는 의미 해석 없이 전달만). persistence가 이 기능의 핵심이다.
        hairstyle_history: dict[str, list] = {}
        for value in (selected_states or {}).values():
            if not isinstance(value, dict):
                continue
            hair_name = str(value.get("canonical_name") or "").strip()
            if not hair_name:
                continue
            hair_timeline = [
                deepcopy(event) for event in (value.get("hairstyle_timeline") or [])
                if isinstance(event, dict)
            ]
            hairstyle_history[hair_name] = hair_timeline
        # 이번 턴 CALL1 hairstyle_events는 아직 timeline(이전 누적)에 반영 전이므로 덧붙인다.
        for event in hairstyle_events or []:
            hair_name = str(event.get("character") or "").strip()
            if not hair_name:
                continue
            hairstyle_history.setdefault(hair_name, [])
            hairstyle_history[hair_name].append(deepcopy(event))
        hairstyle_history = {
            name: events for name, events in hairstyle_history.items() if events
        }
        if hairstyle_history:
            append_call2_context({
                "role": "user",
                "content": (
                    "# SPARSE HAIRSTYLE CHANGE HISTORY\n"
                    "Each event is a semantic hairstyle-arrangement transition with `operation` and "
                    "`hairstyle_change`, not an image-generation tag. Resolve them chronologically "
                    "against AUTHORITATIVE FIXED APPEARANCE and change only the hairstyle-arrangement "
                    "dimension. The active hairstyle from this history is the continuity authority: "
                    "keep it across later scenes that do not repeat the hairstyle, use current-scene "
                    "prose only as a secondary cue, and never treat a generated visual reference as "
                    "hairstyle authority. `replace`/`add`/`remove` change only the conflicting "
                    "arrangement tags; `reset_default` restores the fixed hairstyle. Preserve hair "
                    "color, length, bangs, eyes, body, and every unrelated fixed trait.\n\n"
                    + json.dumps(hairstyle_history, ensure_ascii=False, indent=2)
                ),
            }, include_plan=False)
        if classified_visual_reference:
            append_call2_context({
                "role": "user",
                "content": classified_visual_reference,
            }, include_plan=False, include_keyvis=False)
        if balanced_fallback:
            fallback_text = _history_messages_text(
                persistent_history.get("call2_fallback_history") or []
            )
            if fallback_text:
                append_call2_context({
                    "role": "user",
                    "content": "# BALANCED FALLBACK PAST HISTORY\n\n" + fallback_text,
                })
            print(
                f"[ILLUST_CONTEXT:CALL2] 균형형 폴백 입력 사용: "
                f"history_chars={len(fallback_text)}, full_reference={bool(call2_reference.strip())}"
            )
    else:
        for item in chats[max(0, target_index - int(toggles["call2_context_turns"])):target_index]:
            append_call2_context({
                "role": "assistant" if item["role"] == "char" else "user",
                "content": _strip_nodes(item["data"]),
            })
    keyvis_visual_states = apply_visual_base_events(
        selected_states,
        call1_result.get("current_characters") or [],
        visual_base_events,
        str((persistent_history or {}).get("current_message_id") or ""),
        visual_profiles,
    )
    keyvis_visual_snapshot = visual_base_snapshot(
        keyvis_visual_states,
        current_character_names,
        visual_profiles,
    )
    keyvis_visual_authority = _visual_base_authority_note(
        keyvis_visual_snapshot
    )
    if keyvis_visual_authority:
        append_call2_context({
            "role": "user",
            "content": (
                "# CURRENT KEY VISUAL BASE AUTHORITY\n\n"
                + keyvis_visual_authority
            ),
        }, include_plan=False, include_detail=False, include_keyvis=True)
    append_call2_context({
        "role": "user",
        "content": "[Last log entry]\n" + slotted,
    })
    call2_context_messages = deepcopy(call2_detail_context_messages)
    try:
        detail_context_chars = sum(
            len(str(message.get("content") or ""))
            for message in call2_context_messages
        )
        plan_context_chars = sum(
            len(str(message.get("content") or ""))
            for message in call2_plan_context_messages
        )
        keyvis_context_chars = sum(
            len(str(message.get("content") or ""))
            for message in call2_keyvis_context_messages
        )
        print(
            "[ILLUST_CONTEXT:CALL2] 역할별 입력 분리: "
            f"detail_chars={detail_context_chars}, plan_chars={plan_context_chars} "
            f"(-{detail_context_chars - plan_context_chars}), "
            f"keyvis_chars={keyvis_context_chars} "
            f"(-{detail_context_chars - keyvis_context_chars})"
        )
    except Exception as e:
        print(f"[ILLUST_CONTEXT:CALL2] 역할별 입력 크기 계산 실패: error={e}")
        traceback.print_exc()
    call2_messages.append({
        "role": "user",
        "content": "# Output instructions\n\n" + call2_thoughts + "\n\n" + prompts.get("call2_format", ""),
    })
    if prompts.get("call2_prefill", "").strip():
        call2_messages.append({"role": "assistant", "content": prompts["call2_prefill"]})
    call2_messages.append({"role": "user", "content": "Return the final <lb-xnai> TOON block only after your analysis."})
    call2_output = ""
    call2_plan_output = ""
    call2_keyvis_output = ""
    call2_detail_outputs: list[str] = []
    call2_authority_audit: list[dict] = []
    call2_authority_audit_output = ""
    call2_authority_audit_status = "not_run"
    call2_parallel_fallback_stage = ""
    call2_parallel_fallback_reason = ""
    call2_fallback_expected_slots: list[int] | None = None
    call2_fallback_scene_plan: list[dict] = []
    call2_preserved_keyvis_descriptor: dict | None = None
    call2_preserved_scene_descriptors: list[dict] = []
    keyvis_allowed_names: list[str] = []
    call2_detail_completed = False
    descriptors = []
    if toggles.get("call2_parallel_enabled"):
        parallel_stage = "CALL2-PLAN"
        plan_task: asyncio.Task | None = None
        keyvis_task: asyncio.Task | None = None

        async def cancel_call2_task(task: asyncio.Task | None, label: str) -> None:
            if task is None:
                return
            was_pending = not task.done()
            if was_pending:
                print(f"[ILLUST_CONTEXT:CALL2_PARALLEL] 진행 중 {label} 취소")
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if was_pending:
                    print(f"[ILLUST_CONTEXT:CALL2_PARALLEL] {label} 취소 완료")
            except Exception as cleanup_error:
                print(
                    f"[ILLUST_CONTEXT:CALL2_PARALLEL] {label} 정리 중 실패: "
                    f"error={cleanup_error}"
                )
                traceback.print_exc()

        try:
            if progress:
                await progress(31, "call2_plan", "CALL2 장면 PLAN · Key Visual 병렬 생성")
            plan_messages = deepcopy(call2_plan_context_messages)
            _call2_segment_text, _call2_segments = _segment_current_context(enhanced)
            (
                call2_segment_slots,
                call2_segment_map,
                segment_slot_reason,
            ) = build_segment_slot_map(slotted, _call2_segments)
            if not call2_segment_slots:
                raise ValueError(segment_slot_reason or "CALL2 segment-slot 권위 매핑 실패")
            if segment_slot_reason:
                print(
                    "[ILLUST_CONTEXT:CALL2_PLAN] 부분 segment-slot 매핑으로 계속 진행: "
                    f"reason={segment_slot_reason}"
                )
            mapped_slot_set = {
                int(slot) for slot in call2_segment_slots.values()
            }
            candidates = [
                slot for slot in candidate_slots(original_slotted)
                if slot in mapped_slot_set
            ]
            if not candidates:
                reason = (
                    "segment-slot 매핑 결과에 원본 후보 Slot이 없음: "
                    f"mapped_slots={sorted(mapped_slot_set)}, "
                    f"original_slots={candidate_slots(original_slotted)}"
                )
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
                raise ValueError(reason)
            plan_toggles = deepcopy(toggles)
            plan_toggles["key_visual"] = False
            configured_minimum = int(plan_toggles["output_count_min"])
            configured_maximum = int(plan_toggles["output_count_max"])
            plan_toggles["output_count_min"] = min(
                configured_minimum,
                len(candidates),
            )
            plan_toggles["output_count_max"] = min(
                configured_maximum,
                len(candidates),
            )
            if (
                plan_toggles["output_count_min"] != configured_minimum
                or plan_toggles["output_count_max"] != configured_maximum
            ):
                print(
                    "[ILLUST_CONTEXT:CALL2_PLAN] 유효 매핑 Slot 수에 맞춰 장면 수 조정: "
                    f"configured={configured_minimum}..{configured_maximum}, "
                    f"effective={plan_toggles['output_count_min']}.."
                    f"{plan_toggles['output_count_max']}, "
                    f"available_slots={candidates}"
                )
            if plan_messages and plan_messages[0].get("role") == "system":
                planner_rules = [
                    "You are CALL2-PLAN, the global semantic visual-beat planner for an illustration pipeline.",
                    "Read the full supplied context and select binding moments before DETAIL workers expand them.",
                    "Reason silently and return only the compact JSON requested by the user message.",
                    "Do not output Danbooru tags, camera fields, outfit lists, plan_id, source_segments, slots, analysis, or prose outside JSON.",
                    "Plan narrative scene beats only. You are not an appearance, wardrobe, or Key Visual authority.",
                    "Do not copy or invent persistent visual traits in scene_brief.",
                    "Treat consecutive paragraphs sharing one time, location, and ongoing action as one visual beat; select at most one scene from that beat.",
                    "An existing <img ...> block already occupies its visual beat, so select a different beat.",
                    "Choose each anchor by semantic context and common sense, never by keyword matching.",
                    "Write scene_brief as natural language, not a field menu or tag list. Preserve the central visible action and its ongoing physical state without euphemism.",
                    "When exposure, displaced clothing, intimate contact, or another state is essential to the selected beat, state the participants, relative positions, contact/action, and visible consequence naturally enough for one physically possible image.",
                    "Across selected scenes, prefer meaningful visual progression; do not select near-identical stages of one action merely to fill the requested count.",
                ]
                focus = str(toggles.get("focus") or "").strip()
                direction = str(toggles.get("direction") or "").strip()
                if focus:
                    planner_rules.append(
                        f"Client focus is {focus!r}; do not select scenes focused on other characters."
                    )
                if direction:
                    planner_rules.append(f"Client direction: {direction}")
                plan_messages[0]["content"] = "\n".join(planner_rules)
            if str(toggles.get("scene_mode")) == "auto":
                scene_count_rule = (
                    f"Choose the appropriate count from the {len(candidates)} available slots."
                )
            else:
                minimum = int(plan_toggles["output_count_min"])
                maximum = int(plan_toggles["output_count_max"])
                scene_count_rule = f"Choose between {minimum} and {maximum} scenes."
            # PLAN에게 총장면 수(총량) 카운트 규칙을 준다. 병렬 detail worker는 이 총량을
            # worker 수로 나눈 per-worker 카운트를 별도로 주입받으므로, 여기서는 전체 값.
            plan_messages.append({
                "role": "user",
                "content": render_output_count_rule(
                    plan_toggles["output_count_min"],
                    plan_toggles["output_count_max"],
                ),
            })
            plan_messages.append({
                "role": "user",
                "content": (
                    "# GLOBAL CALL2 PLAN\n"
                    + scene_count_rule
                    + " Select at most one scene per semantic visual beat. Do not invent or output a "
                    "slot number. Select exactly one anchor_segment for each scene; the server derives "
                    "the slot from the authoritative mapping below. Every selected anchor must map to a "
                    "different server slot. Key Visual is handled by a different worker; do not output "
                    "keyvis or keyvis_plan.\n\nReturn this JSON schema only:\n"
                    "{\n"
                    '  "scene_plan": [\n'
                    "    {\n"
                    '      "anchor_segment": "C001",\n'
                    '      "characters": ["canonical name"],\n'
                    '      "scene_brief": "objective visual moment to expand"\n'
                    "    }\n"
                    "  ]\n"
                    "}\n\n"
                    "anchor_segment must be one exact Cxxx ID from the server map. Do not copy a full "
                    "outfit inventory into scene_brief, but never omit a transient wardrobe, coverage, "
                    "contact, or exposure state that defines the selected visual moment; describe that "
                    "state in ordinary natural language. The server separately carries CALL1's literal "
                    "wardrobe-change wording into DETAIL. characters must contain every named tracked character intended to "
                    "appear in that image, in canonical-name form. Use characters: [] when the visual beat "
                    "contains no named tracked character; anonymous students, crowds, staff, or other "
                    "background people belong in scene_brief and must not be given invented canonical "
                    "names.\n\n# SERVER SEGMENT MAP\n"
                    + call2_segment_map
                ),
            })

            def validate_plan(result):
                plan, reason = parse_call2_plan(
                    result,
                    plan_toggles,
                    original_slotted,
                    segment_slot_map=call2_segment_slots,
                    log_errors=False,
                )
                return bool(plan), reason or "CALL2-PLAN 파싱 실패"

            normalized_plan_messages = _normalize_messages(plan_messages)
            print(
                "[ILLUST_CONTEXT:CALL2_PLAN] 전용 입력 준비: "
                f"messages={len(normalized_plan_messages)}, "
                f"chars={sum(len(str(item.get('content') or '')) for item in normalized_plan_messages)}"
            )
            plan_task = asyncio.create_task(
                _call_pipeline_llm(
                    "CALL2-PLAN",
                    normalized_plan_messages,
                    stream_notify,
                    result_validator=validate_plan,
                    json_mode=True,
                ),
                name="call2-plan",
            )
            if toggles.get("key_visual"):
                keyvis_allowed_names = list(current_character_names)
                if not keyvis_allowed_names:
                    keyvis_allowed_names = [
                        match.group(1).strip()
                        for match in re.finditer(
                            r"(?m)^###\s+([^\r\n]+)\s*$",
                            call2_reference,
                        )
                        if match.group(1).strip()
                    ]
                keyvis_task = asyncio.create_task(
                    _run_call2_keyvis(
                        call2_context_messages=call2_keyvis_context_messages,
                        allowed_character_names=keyvis_allowed_names,
                        toggles=toggles,
                        stream_notify=stream_notify,
                    ),
                    name="call2-keyvis",
                )
            print(
                "[ILLUST_CONTEXT:CALL2_PARALLEL] 독립 LLM 동시 시작: "
                f"plan=1, keyvis={1 if keyvis_task else 0}"
            )
            call2_plan_output = await plan_task
            parsed_plan, plan_reason = parse_call2_plan(
                call2_plan_output,
                plan_toggles,
                original_slotted,
                segment_slot_map=call2_segment_slots,
            )
            if parsed_plan is None:
                raise ValueError(plan_reason or "CALL2-PLAN 파싱 실패")
            if parsed_plan["mode"] == "legacy":
                descriptors = list(parsed_plan.get("descriptors") or [])
                if keyvis_task is not None:
                    (
                        call2_preserved_keyvis_descriptor,
                        call2_keyvis_output,
                    ) = await keyvis_task
                    descriptors = [
                        deepcopy(call2_preserved_keyvis_descriptor),
                        *[
                            item for item in descriptors
                            if str(item.get("kind") or "") != "keyvis"
                        ],
                    ]
                call2_output = descriptors_to_toon(descriptors)
                print(
                    "[ILLUST_CONTEXT:CALL2_PLAN] 모델이 완성 TOON을 반환해 "
                    "장면 결과로 수용하고 독립 Key Visual과 병합"
                )
            else:
                parsed_plan["scene_plan"] = bind_scene_plan_wardrobes(
                    list(parsed_plan["scene_plan"]),
                    list(_call2_segments),
                    selected_states,
                    call1_result.get("current_characters") or [],
                    wardrobe_events,
                    str((persistent_history or {}).get("current_message_id") or ""),
                    selected_reference=call2_reference,
                    default_outfits=default_outfits,
                    visual_profiles=visual_profiles,
                    visual_base_events=visual_base_events,
                )
                call2_fallback_expected_slots = [
                    int(item["slot"]) for item in parsed_plan["scene_plan"]
                ]
                call2_fallback_scene_plan = deepcopy(parsed_plan["scene_plan"])
                parallel_stage = "CALL2-DETAIL"
                if progress:
                    await progress(
                        36,
                        "call2_detail",
                        f"CALL2 상세 장면 {len(parsed_plan['scene_plan'])}개 병렬 생성",
                    )
                detail_task = asyncio.create_task(
                    _run_parallel_call2_details(
                        scene_plan=list(parsed_plan["scene_plan"]),
                        call2_context_messages=call2_context_messages,
                        call2_format=prompts.get("call2_format", ""),
                        toggles=toggles,
                        stream_notify=stream_notify,
                        call2_thoughts=call2_detail_thoughts,
                    ),
                    name="call2-details",
                )
                if keyvis_task is not None:
                    detail_result, keyvis_result = await asyncio.gather(
                        detail_task,
                        keyvis_task,
                        return_exceptions=True,
                    )
                    if not isinstance(keyvis_result, BaseException):
                        (
                            call2_preserved_keyvis_descriptor,
                            call2_keyvis_output,
                        ) = keyvis_result
                    if isinstance(detail_result, BaseException):
                        if isinstance(detail_result, asyncio.CancelledError):
                            raise detail_result
                        raise detail_result
                    descriptors, call2_detail_outputs = detail_result
                    if isinstance(keyvis_result, BaseException):
                        if isinstance(keyvis_result, asyncio.CancelledError):
                            raise keyvis_result
                        call2_preserved_scene_descriptors = deepcopy(descriptors)
                        parallel_stage = "CALL2-KEYVIS"
                        raise keyvis_result
                else:
                    descriptors, call2_detail_outputs = await detail_task
                if call2_preserved_keyvis_descriptor is not None:
                    descriptors = [
                        deepcopy(call2_preserved_keyvis_descriptor),
                        *descriptors,
                    ]
                parallel_stage = "CALL2-DETAIL-MERGE"
                call2_output = descriptors_to_toon(descriptors)
                call2_detail_completed = True
        except asyncio.CancelledError:
            await cancel_call2_task(plan_task, "CALL2-PLAN")
            await cancel_call2_task(keyvis_task, "CALL2-KEYVIS")
            print("[ILLUST_CONTEXT:CALL2_PARALLEL] 상위 작업 취소로 병렬 CALL2 중단")
            raise
        except Exception as e:
            if parallel_stage == "CALL2-PLAN":
                await cancel_call2_task(keyvis_task, "CALL2-KEYVIS")
                call2_preserved_keyvis_descriptor = None
                call2_keyvis_output = ""
            elif keyvis_task is not None and not keyvis_task.done():
                await cancel_call2_task(keyvis_task, "CALL2-KEYVIS")
            call2_parallel_fallback_stage = parallel_stage
            call2_parallel_fallback_reason = str(e).strip() or type(e).__name__
            print(
                f"[ILLUST_CONTEXT:CALL2-FALLBACK] 폴백 시작: "
                f"failed_stage={call2_parallel_fallback_stage}, "
                f"error_type={type(e).__name__}, "
                f"reason={call2_parallel_fallback_reason}"
            )
            traceback.print_exc()
            call2_output = ""
            if parallel_stage == "CALL2-PLAN":
                call2_plan_output = ""
            if parallel_stage != "CALL2-KEYVIS":
                call2_detail_outputs = []
            descriptors = []

    if (
        not descriptors
        and not call2_detail_completed
        and call2_parallel_fallback_stage == "CALL2-KEYVIS"
        and call2_preserved_scene_descriptors
    ):
        try:
            print(
                "[ILLUST_CONTEXT:CALL2_KEYVIS] 성공한 scene DETAIL을 보존하고 "
                "Key Visual만 1회 재시도"
            )
            if progress:
                await progress(39, "call2_keyvis", "Key Visual 단독 재시도")
            (
                call2_preserved_keyvis_descriptor,
                call2_keyvis_output,
            ) = await _run_call2_keyvis(
                call2_context_messages=call2_keyvis_context_messages,
                allowed_character_names=keyvis_allowed_names,
                toggles=toggles,
                stream_notify=stream_notify,
            )
            descriptors = [
                deepcopy(call2_preserved_keyvis_descriptor),
                *deepcopy(call2_preserved_scene_descriptors),
            ]
            call2_output = descriptors_to_toon(descriptors)
            call2_detail_completed = True
            print(
                "[ILLUST_CONTEXT:CALL2_KEYVIS] 단독 재시도 성공: "
                f"preserved_scenes={len(call2_preserved_scene_descriptors)}"
            )
        except asyncio.CancelledError:
            print("[ILLUST_CONTEXT:CALL2_KEYVIS] 단독 재시도 중 상위 작업 취소")
            raise
        except Exception as keyvis_retry_error:
            call2_parallel_fallback_reason = (
                f"{call2_parallel_fallback_reason}; "
                f"keyvis_retry={keyvis_retry_error}"
            ).strip("; ")
            call2_detail_outputs = []
            print(
                "[ILLUST_CONTEXT:CALL2_KEYVIS] 단독 재시도도 실패해 전체 폴백 사용: "
                f"error={keyvis_retry_error}"
            )
            traceback.print_exc()

    if not descriptors and not call2_detail_completed:
        is_parallel_fallback = bool(call2_parallel_fallback_reason)
        call2_call_name = "CALL2-FALLBACK" if is_parallel_fallback else "CALL2"
        call2_parse_source = call2_call_name
        preserve_independent_keyvis = bool(
            call2_preserved_keyvis_descriptor is not None
            and call2_parallel_fallback_stage != "CALL2-PLAN"
        )
        fallback_toggles = deepcopy(toggles)
        if preserve_independent_keyvis:
            fallback_toggles["key_visual"] = False
            print(
                "[ILLUST_CONTEXT:CALL2-FALLBACK] 검증된 독립 Key Visual 보존, "
                "scene만 폴백 생성"
            )
        if is_parallel_fallback and progress:
            await progress(
                40,
                "call2_fallback",
                f"CALL2 폴백 · {call2_parallel_fallback_stage}: "
                f"{call2_parallel_fallback_reason}",
            )
        def validate_fallback(result):
            parsed, reason = validate_complete_call2_output(
                result,
                fallback_toggles,
                original_slotted,
                f"{call2_parse_source}-RETRY-CHECK",
                call2_fallback_expected_slots,
            )
            return bool(parsed), reason or f"{call2_call_name} TOON 검증 실패"

        fallback_messages = deepcopy(call2_messages)
        if call2_fallback_scene_plan:
            preserved_plan_payload = {
                "scene_plan": [
                    _public_call2_scene_plan(plan)
                    for plan in call2_fallback_scene_plan
                ],
            }
            keyvis_fallback_rule = (
                "A separately validated Key Visual is already preserved. Omit keyvis completely and "
                "return only the assigned scenes."
                if preserve_independent_keyvis
                else "Generate the required Key Visual independently in the same response when enabled."
            )
            fallback_messages.append({
                "role": "user",
                "content": (
                    "# PRESERVED GLOBAL PLAN AFTER DETAIL FAILURE\n"
                    "The planner already completed successfully. Expand every scene below into the final "
                    "<lb-xnai> block. Use every supplied slot exactly once, preserve character coverage and "
                    "the natural meaning of scene_brief and continuity_note, and do not reselect, omit, or "
                    "add scenes. Resolve wardrobe against the authoritative bases and event history already "
                    "present in this context; the server retains its structured snapshot internally. "
                    + keyvis_fallback_rule
                    + "\n\n"
                    + json.dumps(preserved_plan_payload, ensure_ascii=False, indent=2)
                ),
            })
        fallback_output = await _call_pipeline_llm(
            call2_call_name,
            _normalize_messages(fallback_messages),
            stream_notify,
            result_validator=validate_fallback,
        )
        descriptors, fallback_validation_reason = validate_complete_call2_output(
            fallback_output,
            fallback_toggles,
            original_slotted,
            call2_parse_source,
            call2_fallback_expected_slots,
        )
        if not descriptors:
            print(
                f"[ILLUST_CONTEXT:{call2_call_name}] 최종 폴백 검증 실패: "
                f"reason={fallback_validation_reason}"
            )
            raise RuntimeError(
                fallback_validation_reason or f"{call2_call_name} 최종 검증 실패"
            )
        if preserve_independent_keyvis:
            descriptors = [
                deepcopy(call2_preserved_keyvis_descriptor),
                *[
                    item for item in descriptors
                    if str(item.get("kind") or "") != "keyvis"
                ],
            ]
            call2_output = descriptors_to_toon(descriptors)
        else:
            call2_output = fallback_output

    # Optimized CALL1 path deliberately sends only selected character details.  If
    # CALL2 nevertheless emits another named character, retry once with the
    # bounded history and full dictionary instead of silently accepting a likely
    # CALL1 coverage miss.
    if (
        descriptors
        and persistent_history
        and toggles.get("call1_enabled")
        and call1_result
        and not balanced_fallback
    ):
        allowed = {name.casefold() for name in current_character_names}
        observed = {
            str(character.get("name") or "").strip()
            for descriptor in descriptors
            if str(descriptor.get("kind") or "") == "scene"
            for character in descriptor.get("characters") or []
            if str(character.get("name") or "").strip()
        }
        unexpected = sorted(name for name in observed if name.casefold() not in allowed)
        if unexpected:
            balanced_fallback = True
            print(
                f"[ILLUST_CONTEXT:CALL2] CALL1 선택 밖 캐릭터 감지, 균형형 1회 재시도: "
                f"unexpected={unexpected}, allowed={current_character_names}"
            )
            preserve_keyvis_during_coverage_retry = bool(
                call2_preserved_keyvis_descriptor is not None
                and toggles.get("key_visual")
            )
            coverage_retry_toggles = deepcopy(toggles)
            if preserve_keyvis_during_coverage_retry:
                coverage_retry_toggles["key_visual"] = False
            scene_only_previous_output = descriptors_to_toon([
                descriptor
                for descriptor in descriptors
                if str(descriptor.get("kind") or "") == "scene"
            ])
            retry_messages = deepcopy(call2_messages)
            retry_messages.extend([{
                "role": "assistant",
                "content": (
                    scene_only_previous_output
                    if preserve_keyvis_during_coverage_retry
                    else call2_output
                ),
            }, {
                "role": "user",
                "content": (
                    "Character coverage did not match CALL1. Re-evaluate the current context "
                    "with the bounded past history and full character dictionary below. "
                    "Preserve established wardrobe state unless a supplied wardrobe event changes it.\n\n"
                    + (
                        "# ACTIVE BOT IMAGE INSTRUCTIONS\n\n"
                        + call2_instruction
                        + "\n\n"
                        if call2_instruction
                        else ""
                    )
                    + "# FULL CHARACTER DICTIONARY\n"
                    + str(extra_reference or "")
                    + "\n\n# BOUNDED PAST HISTORY\n"
                    + (_history_messages_text(
                        persistent_history.get("call2_fallback_history") or []
                    ) or "(empty)")
                    + (
                        "\n\nAn independently validated Key Visual is already preserved and is not "
                        "included above. Return corrected scene objects only and omit keyvis."
                        if preserve_keyvis_during_coverage_retry
                        else "\n\nReturn the complete corrected <lb-xnai> block only."
                    )
                ),
            }])

            def validate_coverage_retry(result):
                parsed, reason = validate_complete_call2_output(
                    result,
                    coverage_retry_toggles,
                    original_slotted,
                    "CALL2-COVERAGE-RETRY-CHECK",
                    call2_fallback_expected_slots,
                )
                return bool(parsed), reason or "CALL2 캐릭터 커버리지 재시도 검증 실패"

            retried_output = await _call_pipeline_llm(
                "CALL2",
                _normalize_messages(retry_messages),
                stream_notify,
                result_validator=validate_coverage_retry,
            )
            retried_descriptors, coverage_retry_reason = validate_complete_call2_output(
                retried_output,
                coverage_retry_toggles,
                original_slotted,
                "CALL2-COVERAGE-RETRY",
                call2_fallback_expected_slots,
            )
            if retried_descriptors:
                if preserve_keyvis_during_coverage_retry:
                    descriptors = [
                        deepcopy(call2_preserved_keyvis_descriptor),
                        *[
                            item for item in retried_descriptors
                            if str(item.get("kind") or "") == "scene"
                        ],
                    ]
                    call2_output = descriptors_to_toon(descriptors)
                else:
                    call2_output = retried_output
                    descriptors = retried_descriptors
            else:
                print(
                    "[ILLUST_CONTEXT:CALL2] 캐릭터 커버리지 재시도 최종 검증 실패: "
                    f"reason={coverage_retry_reason}"
                )

    # CALL2 파싱 실패 시 CALL2-FIX(repair.txt)가 TOON 블록을 교정한다.
    # CALL3는 대사 생성 전용이므로 교정은 여기서 먼저 마무리한다.
    call2_fix_output = ""
    if not descriptors and not call2_detail_completed:
        if progress:
            await progress(48, "call2_fix", "CALL2-FIX TOON 교정")
        fix_system_parts = [prompts.get("call2_fix", "")]
        if call2_instruction:
            fix_system_parts.append(
                "# ACTIVE BOT IMAGE INSTRUCTIONS\n\n" + call2_instruction
            )
        if call2_reference.strip():
            fix_system_parts.append(
                "# CHARACTER DICTIONARY\n\n" + call2_reference
            )
        fix_messages = [{
            "role": "system",
            "content": "\n\n".join(
                part for part in fix_system_parts if str(part or "").strip()
            ),
        }, {
            "role": "user",
            "content": "Repair this malformed output. Return [TOON]...[/TOON].\n\n" + call2_output,
        }]
        call2_fix_output = await _call_pipeline_llm(
            "CALL2-FIX",
            _normalize_messages(fix_messages),
            stream_notify,
            result_validator=lambda result: (
                bool(parse_toon_plan(result, toggles, "CALL2-FIX-RETRY-CHECK")),
                "CALL2-FIX TOON 파싱 실패",
            ),
        )
        descriptors = parse_toon_plan(call2_fix_output, toggles, "CALL2-FIX")
        if not descriptors:
            raise RuntimeError("CALL2-FIX 교정 후에도 장면 TOON 파싱에 실패했습니다")

    # 이미지 생성에는 CALL3의 대사가 필요하지 않다. CALL2(+필요 시 FIX)가 확정되면
    # 슬롯/RAW를 먼저 고정하고 콜백으로 공개해, CALL3와 이미지 생성을 병렬로 진행한다.
    # 콜백에는 SPEAK이 없는 RAW가 전달되며 최종 반환 RAW에는 아래 CALL3 결과가 합쳐진다.
    had_descriptors_before_slot_sanitize = bool(descriptors)
    descriptors = attach_descriptor_anchors(
        descriptors,
        original_slotted,
    )
    descriptors = sanitize_descriptor_slots(descriptors, original_slotted)
    if not descriptors:
        if call2_detail_completed and not had_descriptors_before_slot_sanitize:
            print(
                "[ILLUST_CONTEXT:CALL2_DETAIL] PLAN 캐릭터 불일치로 "
                "모든 scene 슬롯이 폐기되어 빈 결과로 완료"
            )
        else:
            print("[ILLUST_CONTEXT:CALL2] 슬롯 보정 후 생성할 descriptor가 없음")
            raise RuntimeError("CALL2 결과에 유효한 장면 슬롯이 없습니다")

    plan_visual_by_slot = {
        int(plan.get("slot") or 0): deepcopy(plan.get("visual_base_snapshot") or {})
        for plan in call2_fallback_scene_plan
        if int(plan.get("slot") or 0) > 0
    }
    fallback_visual_states = apply_visual_base_events(
        selected_states,
        call1_result.get("current_characters") or [],
        visual_base_events,
        str((persistent_history or {}).get("current_message_id") or ""),
        visual_profiles,
    )
    for descriptor in descriptors:
        descriptor_names = [
            str(character.get("name") or "").strip()
            for character in descriptor.get("characters") or []
            if str(character.get("name") or "").strip()
        ]
        if str(descriptor.get("kind") or "") == "keyvis":
            candidate_snapshot = keyvis_visual_snapshot
        else:
            candidate_snapshot = plan_visual_by_slot.get(
                int(descriptor.get("slot") or 0),
                {},
            )
        if not candidate_snapshot and descriptor_names:
            candidate_snapshot = visual_base_snapshot(
                fallback_visual_states,
                descriptor_names,
                visual_profiles,
            )
        filtered_snapshot = {
            name: deepcopy(base)
            for name, base in (candidate_snapshot or {}).items()
            if str(name).casefold() in {
                value.casefold() for value in descriptor_names
            }
        }
        descriptor["visual_base_snapshot"] = filtered_snapshot
        missing_visual_names = [
            name for name in descriptor_names
            if not isinstance(
                _authority_values_for_name(filtered_snapshot, name),
                dict,
            )
        ]
        if missing_visual_names:
            print(
                f"[ILLUST_CONTEXT:VISUAL_BASE] descriptor 결속 누락: "
                f"kind={descriptor.get('kind')}, slot={descriptor.get('slot')}, "
                f"characters={missing_visual_names}"
            )

    if progress:
        await progress(49, "call2_authority_audit", "CALL2 외형·복장 권위 감사")
    # AUDIT에 hairstyle history(누적 timeline + 이번 턴 events)를 전달한다. 서버는
    # 의미 해석 없이 전달만 하고, AUDIT이 fixed appearance에 대해 temporary override
    # 여부를 판단한다. 없으면 빈 dict로 전달된다.
    audit_hairstyle_history: dict[str, list] = {}
    for value in (selected_states or {}).values():
        if not isinstance(value, dict):
            continue
        hair_name = str(value.get("canonical_name") or "").strip()
        if not hair_name:
            continue
        audit_hairstyle_history[hair_name.casefold()] = [
            deepcopy(event) for event in (value.get("hairstyle_timeline") or [])
            if isinstance(event, dict)
        ]
    for event in hairstyle_events or []:
        hair_name = str(event.get("character") or "").strip()
        if not hair_name:
            continue
        audit_hairstyle_history.setdefault(hair_name.casefold(), [])
        audit_hairstyle_history[hair_name.casefold()].append(deepcopy(event))
    audit_hairstyle_history = {
        name: events for name, events in audit_hairstyle_history.items() if events
    }
    (
        semantic_authority_decisions,
        call2_authority_audit_output,
        call2_authority_audit_status,
    ) = await _run_call2_authority_audit(
        descriptors,
        fixed_appearance,
        default_outfits,
        slotted,
        stream_notify,
        hairstyle_history=audit_hairstyle_history,
        toggles=toggles,
    )
    call2_authority_audit = apply_call2_authority_base(
        descriptors,
        fixed_appearance,
        default_outfits,
        semantic_authority_decisions,
        call2_authority_audit_status,
    )
    # Every downstream consumer, including early image enqueue, Call5 inputs,
    # generated-reference history, and diagnostics, must see the same repaired
    # descriptor set.
    call2_output = descriptors_to_toon(descriptors)

    last_visual_by_character = _last_visual_by_character(descriptors)
    character_states_after = deepcopy((persistent_history or {}).get("state_before") or {})
    if persistent_history:
        state_character_names: list[str] = []
        for item in call1_result.get("current_characters") or []:
            name = str(item.get("name") if isinstance(item, dict) else item).strip()
            if name and name.casefold() not in {
                value.casefold() for value in state_character_names
            }:
                state_character_names.append(name)
        for name in last_visual_by_character:
            if str(name).strip() and str(name).casefold() not in {
                value.casefold() for value in state_character_names
            }:
                state_character_names.append(str(name).strip())
        character_states_after = apply_visual_base_events(
            character_states_after,
            [{"name": name, "confidence": 1.0} for name in state_character_names],
            visual_base_events,
            str(persistent_history.get("current_message_id") or ""),
            visual_profiles,
        )
        final_visual_bases = visual_base_snapshot(
            character_states_after,
            state_character_names,
            visual_profiles,
        )
        final_default_outfits = deepcopy(default_outfits)
        for name, base in final_visual_bases.items():
            final_default_outfits[name] = visual_tag_values(base.get("outfit") or [])
        character_states_after = apply_wardrobe_events(
            character_states_after,
            [{"name": name, "confidence": 1.0} for name in state_character_names],
            wardrobe_events,
            str(persistent_history.get("current_message_id") or ""),
            selected_reference=call2_reference,
            default_outfits=final_default_outfits,
            hairstyle_events=hairstyle_events,
        )
        character_states_after = merge_last_visual_into_states(
            character_states_after,
            last_visual_by_character,
            str(persistent_history.get("current_message_id") or ""),
            allow_visual_initialization=(not bool(call1_result) or balanced_fallback),
        )

    downstream_chats = deepcopy(chats)
    if toggles.get("call1_backtranslate_enabled") or (persistent_history and call1_result):
        downstream_chats[target_index]["data"] = enhanced
    downstream_context = context_text(downstream_chats)
    prompt_format = str(toggles.get("prompt_format") or "v3").strip().lower()

    preliminary_items = []
    for descriptor in descriptors:
        positive, negative = build_raw_prompt(descriptor, enhanced, prompts, toggles)
        raw_item = deepcopy(descriptor)
        raw_item["raw_positive"] = positive
        raw_item["raw_negative"] = negative
        preliminary_items.append(raw_item)

    if on_call2_ready:
        if progress:
            await progress(52, "enqueue", f"CALL2 확정 · 이미지 {len(preliminary_items)}장 조기 등록")
        try:
            await on_call2_ready({
                "session_id": payload["session_id"],
                "context": downstream_context,
                "prompt_format": prompt_format,
                "items": deepcopy(preliminary_items),
            })
        except Exception as e:
            print(f"[ILLUST_CONTEXT:CALL2] 이미지 조기 등록 콜백 실패: {e}")
            traceback.print_exc()
            raise

    # CALL3는 대사 빌드(speak/manga)만 담당한다. 위에서 시작한 이미지 생성과
    # 동시에 실행되며, CALL2가 최종 선택한 일반 scene만 넘긴다.
    # 대사 말투는 역변환/보강으로 흔들리지 않도록 최신 CHAR 원문을 사용한다.
    call3_output = ""
    call3_initial_output = ""
    call3_correction_used = False
    call3_partial_recovery_used = False
    call3_dialogue_drop_recovery_used = False
    call3_dropped_dialogue_entries = []
    call3_silent_slots = []
    call3_failure_reason = ""
    call3_descriptors = [
        descriptor
        for descriptor in descriptors
        if str(descriptor.get("kind") or "") == "scene"
    ]
    if (
        toggles.get("call3_enabled")
        and toggles.get("speak_enabled")
        and call3_descriptors
    ):
        if progress:
            await progress(58, "call3", "CALL3 대사 빌드")
        call3_prompt_mode, call3_system_prompt = build_call3_dialogue_system_prompt(
            prompts,
            toggles,
            extra_names,
        )
        selected_slots, selected_scene_payload = build_call3_scene_selection(
            call3_descriptors,
            slotted if persistent_history else "",
        )
        original_narrative = (
            resolved_current
            if persistent_history and call1_result
            else _strip_nodes(narrative)
        )
        speak_language = str(toggles.get("speak_language") or "한국어").strip() or "한국어"
        speak_messages = [{
            "role": "system",
            "content": call3_system_prompt,
        }]
        if persistent_history:
            if not call1_result or balanced_fallback:
                fallback_text = _history_messages_text(
                    persistent_history.get("call3_fallback_history") or []
                )
                if fallback_text:
                    speak_messages.append({
                        "role": "user",
                        "content": "# FALLBACK PAST HISTORY FOR ATTRIBUTION\n\n" + fallback_text,
                    })
        else:
            for item in chats[max(0, target_index - int(toggles["call3_context_turns"])):target_index]:
                speak_messages.append({
                    "role": "assistant" if item["role"] == "char" else "user",
                    "content": _strip_nodes(item["data"]),
                })
        speak_messages.append({
            "role": "user",
            "content": (
                f"[Original narrative]\n{original_narrative}"
                f"\n\n[Selected scenes from CALL2]\n{selected_scene_payload}"
                f"\n\nLanguage: {speak_language}"
            ),
        })
        call3_state = await _build_call3_dialogue_with_recovery(
            speak_messages,
            selected_slots,
            extra_names,
            speak_language,
            stream_notify,
        )
        call3_output = str(call3_state["output"] or "")
        call3_initial_output = str(call3_state["initial_output"] or "")
        call3_correction_used = bool(call3_state["correction_used"])
        call3_partial_recovery_used = bool(call3_state["partial_recovery_used"])
        call3_dialogue_drop_recovery_used = bool(
            call3_state["dialogue_drop_recovery_used"]
        )
        call3_dropped_dialogue_entries = list(
            call3_state["dropped_dialogue_entries"]
        )
        call3_silent_slots = list(call3_state["silent_slots"])
        call3_failure_reason = str(call3_state["failure_reason"] or "")
        # CALL3가 닫는 따옴표/괄호 안 끝에 #감정을 붙여 내보낸 줄을 교정한다.
        # parse_speak_output이 #감정을 닫는 구분자 바깥에서만 인식하므로, 출력 직후
        # 무조건 한 번 훑어 안쪽 끝 #감정을 바깥으로 옮긴다(감정 토글과 무관).
        call3_output = postprocess.postprocess_call3_emotion_placement(call3_output)
        speak_map = parse_speak_output(
            call3_output,
            max_entries_per_scene=2 if call3_prompt_mode == "speak" else None,
        )
        for descriptor in call3_descriptors:
            descriptor["speak"] = speak_map.get(int(descriptor.get("slot", 0)), "")
    elif (
        toggles.get("call3_enabled")
        and toggles.get("speak_enabled")
        and not call3_descriptors
    ):
        key_visual_count = sum(
            1
            for descriptor in descriptors
            if str(descriptor.get("kind") or "") == "keyvis"
        )
        print(
            "[ILLUST_CONTEXT:CALL3] CALL2가 선택한 일반 장면 slot이 없어 "
            f"대사 생성 건너뜀: key_visuals={key_visual_count}"
        )
    else:
        print("[ILLUST_CONTEXT:CALL3] 토글로 비활성화되었거나 SPEAK이 꺼져 있음")

    # 다중 캐릭터 영역 계산은 이미지에 대사가 필요한 CALL3까지 끝난 뒤 수행한다.
    # 서버는 이 동안 CALL2에서 확정된 단일 캐릭터 이미지를 이미 GPU 큐에서 처리한다.
    if enable_multi_char_layout:
        if progress:
            multi_count = sum(
                1 for descriptor in descriptors
                if len(descriptor.get("characters") or []) >= 2
            )
            if multi_count:
                await progress(66, "multi_char_mask", f"다중 캐릭터 마스크 {multi_count}개 계산")
        await calculate_multi_char_layouts(
            descriptors,
            prompts.get("multi_char_mask", ""),
            stream_notify=stream_notify,
            positive_note=str(toggles.get("positive_note") or ""),
        )

    if progress:
        await progress(68, "raw_build", f"RAW 프롬프트 {len(descriptors)}개 생성")
    raw_items = []
    for descriptor in descriptors:
        positive, negative = build_raw_prompt(descriptor, enhanced, prompts, toggles)
        item = deepcopy(descriptor)
        item["raw_positive"] = positive
        item["raw_negative"] = negative
        raw_items.append(item)
    _result = {
        "session_id": payload["session_id"],
        "context": downstream_context,
        "narrative": narrative,
        "backtranslated_narrative": backtranslated_narrative,
        "backtranslated_slotted": backtranslated_slotted,
        "backtranslation_chunks": backtranslation_chunks,
        "enhanced_narrative": enhanced,
        "call1_output": call1_output,
        "call1_result": call1_result,
        "reference_provenance": reference_provenance,
        "last_visual_reference_classification": last_visual_reference_classification,
        "reference_variables": reference_variables,
        "visual_base_events": visual_base_events,
        "wardrobe_events": wardrobe_events,
        "hairstyle_events": hairstyle_events,
        "balanced_fallback_used": balanced_fallback,
        "call2_output": call2_output,
        "call2_plan_output": call2_plan_output,
        "call2_keyvis_output": call2_keyvis_output,
        "call2_detail_outputs": call2_detail_outputs,
        "call2_authority_audit": call2_authority_audit,
        "call2_authority_audit_output": call2_authority_audit_output,
        "call2_authority_audit_status": call2_authority_audit_status,
        "call2_fallback_stage": call2_parallel_fallback_stage,
        "call2_fallback_reason": call2_parallel_fallback_reason,
        "call2_fix_output": call2_fix_output,
        "call3_output": call3_output,
        "call3_initial_output": call3_initial_output,
        "call3_correction_used": call3_correction_used,
        "call3_partial_recovery_used": call3_partial_recovery_used,
        "call3_dialogue_drop_recovery_used": call3_dialogue_drop_recovery_used,
        "call3_dropped_dialogue_entries": call3_dropped_dialogue_entries,
        "call3_silent_slots": call3_silent_slots,
        "call3_failure_reason": call3_failure_reason,
        "character_states_after": character_states_after,
        "last_visual_by_character": last_visual_by_character,
        "history_input_hash": (
            hashlib.sha256(
                (
                    str((persistent_history or {}).get("base_context_hash") or "")
                    + ":"
                    + str((persistent_history or {}).get("current_context_hash") or "")
                ).encode("utf-8")
            ).hexdigest()
            if persistent_history
            else ""
        ),
        "prompt_format": prompt_format,
        "items": raw_items,
        # 이 삽화 생성에서 거친 모든 LLM 호출의 history_id (MULTI-CHAR-MASK~CALL3).
        # 백업 _info.json 의 llm_trace 로 저장되어 흐름 추적 버튼이 정확 매칭에 사용.
        "llm_trace": list(trace),
    }
    _llm_trace_ctx.reset(_llm_trace_token)
    return _result
