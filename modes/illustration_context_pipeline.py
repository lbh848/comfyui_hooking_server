"""CHAT -> CALL1/CALL2/CALL3 -> 기존 RAW 삽화 프롬프트 전단계.

RisuAI는 Comfy history에 이미지가 여러 장 있어도 첫 장만 소비한다. 이 모듈은
최초 CHAT 요청과 후속 결과 회수 요청을 구분하고, 한 세션의 모든 장면 프롬프트와
이미지를 서버 메모리에 보관할 수 있는 공통 형식을 제공한다.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts", "lighbd")
REQUIREMENTS_DIR = os.path.join(BASE_DIR, "요구사항")
SESSION_DIR = os.path.join(BASE_DIR, "logs", "illustration_context_sessions")

CONTEXT_PREFIX = "__LB_ILLUST_CONTEXT_V1__"
RESULT_PREFIX = "__LB_ILLUST_RESULT_V1__"
REGENERATE_PREFIX = "__LB_ILLUST_REGENERATE_V1__"
PROMPT_BATCH_PREFIX = "__LB_ILLUST_PROMPT_BATCH_V1__"

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
    "speak_enabled": True,
    "call3_prompt_mode": "speak",
    "speak_language": "한국어",
    "speak_emotion_enabled": False,
    "speak_emotions": "",
    "nsfw": False,
    "supplement": True,
    "key_visual": True,
    "character_limit": 3,
    # scene_mode: "manual" = 서버가 최소/최대 강제, "auto" = lb-xnai(call2)에 완전 방임
    "scene_mode": "manual",
    "scene_min": 5,
    "scene_max": 11,
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
        out["scene_min"] = max(1, min(15, int(out["scene_min"])))
        out["scene_max"] = max(1, min(15, int(out["scene_max"])))
        if out["scene_min"] > out["scene_max"]:
            print(
                f"[ILLUST_CONTEXT] scene_min({out['scene_min']}) > scene_max({out['scene_max']}), "
                f"min을 max로 보정"
            )
            out["scene_min"] = out["scene_max"]
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
            "scene_min": DEFAULT_TOGGLES["scene_min"],
            "scene_max": DEFAULT_TOGGLES["scene_max"],
        })
    # 예전 UI에서 저장한 고정 배치 크기는 더 이상 사용하지 않는다. CALL2-PLAN이
    # 선택한 전체 장면 수를 최대 동시 요청 수에 맞춰 자동 분배한다.
    out.pop("call2_parallel_batch_size", None)
    # CALL1 병렬도 segment를 최대 동시 요청 수만큼 균등 분할하므로 청크당 segment 수
    # 설정은 더 이상 사용하지 않는다. 과거 저장값이 남아 있으면 무시.
    out.pop("call1_parallel_chunk_size", None)
    return out


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
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 16:
        count = len(raw_items) if isinstance(raw_items, list) else -1
        print(
            f"[ILLUST_PROMPT_BATCH] 잘못된 items 개수: "
            f"session={session_id}, count={count}"
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


def update_session_image_by_slot(session_id: str, slot: int, image: bytes) -> bool:
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


def session_slots_by_lookup_key(lookup_key: str) -> list[int]:
    """Return only the ready slot numbers for the short HTTPS manifest route."""
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

    if not 1 <= len(slots) <= 16:
        raise ValueError(f"invalid illustration slot count: key={key}, count={len(slots)}")
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
) -> dict | None:
    """Validate CALL1's compact entity/coreference/wardrobe analysis."""
    raw = _json_object_from_text(text)
    if raw is None:
        return None
    canonical = _canonical_name_map(character_names)
    errors = []

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
        current_names.add(name.casefold())
        current_characters.append({"name": name, "confidence": confidence})
        if confidence < 0.70:
            errors.append(f"현재 캐릭터 신뢰도 낮음: {name}={confidence:.2f}")

    assignments = []
    for index, item in enumerate(raw.get("reference_assignments") or [], start=1):
        if not isinstance(item, dict):
            errors.append(f"지칭 할당 형식 오류: index={index}")
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
            errors.append(
                f"지칭 할당 필수값 오류: index={index}, segment={segment_id!r}, "
                f"surface={surface!r}, name={name!r}"
            )
            continue
        if surface not in str(segments[segment_id].get("text") or ""):
            errors.append(f"지칭 원문 불일치: segment={segment_id}, surface={surface!r}")
            continue
        if canonical and name.casefold() not in canonical:
            errors.append(f"정식 이름 목록 밖 지칭 대상: {name}")
        if name.casefold() not in replacement.casefold():
            replacement = name
        if confidence < 0.70:
            errors.append(f"지칭 할당 신뢰도 낮음: {segment_id}/{surface}={confidence:.2f}")
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
            errors.append(f"CALL1이 원문 정식 이름을 누락해 서버가 보완: {name}")

    history_names = {name.casefold() for name in history_characters}
    for folded, name in canonical.items():
        if _contains_canonical_name(history_context, name) and folded not in history_names:
            history_names.add(folded)
            history_characters.append(name)
            errors.append(f"CALL1이 과거 히스토리 정식 이름을 누락해 서버가 보완: {name}")

    wardrobe_events = []
    changing_operations = {
        "wear", "add", "remove", "replace", "set", "open", "close",
        "adjust", "nude", "topless", "bottomless", "reset_default",
        "contextual_reset",
    }
    for index, item in enumerate(raw.get("wardrobe_events") or [], start=1):
        if not isinstance(item, dict):
            errors.append(f"복장 사건 형식 오류: index={index}")
            continue
        segment_id = str(item.get("segment_id") or "").strip()
        name = normalize_name(item.get("character") or item.get("name"))
        operation = str(item.get("operation") or "keep").strip().lower()
        evidence = str(item.get("evidence") or "").strip()
        items = item.get("items") or []
        if not isinstance(items, list):
            items = [items]
        items = [str(value).strip() for value in items if str(value).strip()]
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 1.0))))
        except Exception:
            confidence = 0.0
        if not name:
            errors.append(f"복장 사건 캐릭터 없음: index={index}")
            continue
        if operation in changing_operations:
            segment_text = str((segments.get(segment_id) or {}).get("text") or "")
            if (
                not segment_id
                or not evidence
                or _normalize_analysis_text(evidence) not in _normalize_analysis_text(segment_text)
            ):
                errors.append(
                    f"복장 변경 근거 불일치: character={name}, operation={operation}, "
                    f"segment={segment_id!r}"
                )
                continue
        if confidence < 0.70:
            errors.append(f"복장 사건 신뢰도 낮음: {name}/{operation}={confidence:.2f}")
        wardrobe_events.append({
            "segment_id": segment_id,
            "character": name,
            "operation": operation,
            "items": items,
            "state_after": deepcopy(item.get("state_after")),
            "evidence": evidence,
            "confidence": confidence,
        })

    unresolved = raw.get("unresolved_references") or []
    if not isinstance(unresolved, list):
        unresolved = [unresolved]
    unresolved = [deepcopy(item) for item in unresolved if item not in (None, "", {})]
    if unresolved:
        errors.append(f"미해결 지칭 {len(unresolved)}건")
    if character_names.strip() and not current_characters:
        errors.append("현재 캐릭터 목록이 비어 있음")

    return {
        "reference_assignments": assignments,
        "history_characters": history_characters,
        "current_characters": current_characters,
        "wardrobe_events": wardrobe_events,
        "unresolved_references": unresolved,
        "validation_errors": errors,
        "fallback_required": bool(errors),
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
) -> dict:
    """Apply only CALL1-declared state operations; missing tags never remove clothes."""
    states = deepcopy(state_before or {})

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

    for item in current_characters or []:
        name = str(item.get("name") if isinstance(item, dict) else item).strip()
        if not name:
            continue
        key = state_key(name)
        if key not in states:
            states[key] = {
                "canonical_name": name,
                "default_outfit_reference": (
                    _filter_character_reference(selected_reference, [name])
                    or str(selected_reference or "")
                ),
                "current_wardrobe": {"body_state": "unknown", "worn": [], "removed": []},
                "wardrobe_timeline": [],
            }
        states[key]["last_seen_message_id"] = str(current_message_id or "")

    for event in wardrobe_events or []:
        name = str(event.get("character") or "").strip()
        if not name:
            continue
        key = state_key(name)
        if key not in states:
            states[key] = {
                "canonical_name": name,
                "default_outfit_reference": (
                    _filter_character_reference(selected_reference, [name])
                    or str(selected_reference or "")
                ),
                "current_wardrobe": {"body_state": "unknown", "worn": [], "removed": []},
                "wardrobe_timeline": [],
            }
        wardrobe = deepcopy(states[key].get("current_wardrobe") or {})
        worn = [str(value) for value in wardrobe.get("worn") or [] if str(value).strip()]
        removed = [str(value) for value in wardrobe.get("removed") or [] if str(value).strip()]
        operation = str(event.get("operation") or "keep").lower()
        items = [str(value) for value in event.get("items") or [] if str(value).strip()]
        state_after = event.get("state_after")
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
            if worn:
                removed = list(dict.fromkeys(removed + worn))
            worn = list(dict.fromkeys(items))
            wardrobe["body_state"] = state_label or ("clothed" if worn else "unknown")
        elif operation in ("reset_default", "contextual_reset"):
            worn = list(dict.fromkeys(items))
            removed = []
            wardrobe["body_state"] = state_label or ("clothed" if worn else "unknown")
        elif operation in ("open", "close", "adjust"):
            wardrobe["body_state"] = state_label or str(wardrobe.get("body_state") or "partial")
        wardrobe["worn"] = worn
        wardrobe["removed"] = removed
        wardrobe["last_event"] = deepcopy(event)
        states[key]["current_wardrobe"] = wardrobe
        timeline = list(states[key].get("wardrobe_timeline") or [])
        timeline.append(deepcopy(event))
        states[key]["wardrobe_timeline"] = timeline[-50:]
        states[key]["last_seen_message_id"] = str(current_message_id or "")
    return states


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
        existing_wardrobe = result[key].get("current_wardrobe") or {}
        existing_body_state = str(existing_wardrobe.get("body_state") or "unknown").lower()
        if (
            isinstance(outfit_state, dict)
            and outfit_state
            and (allow_visual_initialization or existing_body_state in ("", "unknown"))
        ):
            # CALL2-only mode has no semantic wardrobe writer. Keep the generated
            # state explicitly marked as a visual candidate so the next call can
            # use it without pretending it came from narrative evidence.
            candidate = deepcopy(outfit_state)
            candidate["source"] = "call2_visual_candidate"
            result[key]["current_wardrobe"] = candidate
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


def render_call2_prompt(text: str, toggles: dict, history: str = "") -> str:
    """Risu 토글 매크로를 서버 설정으로 렌더링한다."""
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
    if str(toggles.get("scene_mode")) != "auto":
        limits.append(
            f"Generate between {int(toggles['scene_min'])} and {int(toggles['scene_max'])} scenes."
        )
    limits.append(f"Maximum fully visible characters per image: {int(toggles['character_limit'])}.")
    limits.append(
        f"Key visual: {'required' if toggles.get('key_visual') else 'disabled; omit keyvis'} ."
    )
    text += "\n\n# Server limits\n- " + "\n- ".join(limits)
    return text.strip()


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
        chars.append({
            "positive": str(ch.get("positive") or "").strip(),
            "negative": str(ch.get("negative") or "").strip(),
            "name": str(ch.get("name") or "").strip(),
            "position": str(ch.get("position") or "").strip(),
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
    # 그대로 수용한다. manual 모드일 때만 scene_max 상한으로 잘라낸다.
    scene_cap = None if str(toggles.get("scene_mode")) == "auto" else int(toggles["scene_max"])
    capped = scenes if scene_cap is None else scenes[:scene_cap]
    for index, raw in enumerate(capped, start=1):
        if isinstance(raw, dict):
            out.append(_descriptor(raw, "scene", index))
    if not out:
        print(f"[ILLUST_CONTEXT:{source}] 유효한 keyvis/scene 결과가 없음")
    return out


def parse_call2_plan(
    text: str,
    toggles: dict,
    target_slotted: str,
    *,
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
    seen_slots = set()
    for index, item in enumerate(raw.get("scene_plan") or [], start=1):
        if not isinstance(item, dict):
            reason = f"scene_plan[{index}]가 object가 아님"
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}: item={item!r}")
            return None, reason
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
        if slot in seen_slots:
            reason = f"scene_plan 중복 slot: {slot}"
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason
        seen_slots.add(slot)
        source_segments = item.get("source_segments") or []
        if not isinstance(source_segments, list):
            source_segments = [source_segments]
        characters = item.get("characters") or []
        if not isinstance(characters, list):
            characters = [characters]
        normalized_characters = [
            str(value).strip() for value in characters if str(value).strip()
        ]
        scene_brief = str(item.get("scene_brief") or "").strip()
        if not normalized_characters or not scene_brief:
            reason = (
                f"scene_plan[{index}] characters 또는 scene_brief가 비어 있음: "
                f"characters={normalized_characters}, brief={scene_brief!r}"
            )
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason
        scene_plan.append({
            "plan_id": str(item.get("plan_id") or f"S{index:03d}").strip() or f"S{index:03d}",
            "slot": slot,
            "source_segments": [str(value).strip() for value in source_segments if str(value).strip()],
            "characters": normalized_characters,
            "scene_brief": scene_brief,
        })

    if not scene_plan:
        reason = "CALL2-PLAN이 장면을 선택하지 않음"
        if log_errors:
            print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
        return None, reason
    scene_plan.sort(key=lambda item: candidates.index(item["slot"]))
    for index, item in enumerate(scene_plan, start=1):
        item["plan_id"] = f"S{index:03d}"

    if str(toggles.get("scene_mode")) != "auto":
        minimum = min(int(toggles["scene_min"]), len(candidates))
        maximum = min(int(toggles["scene_max"]), len(candidates))
        if not minimum <= len(scene_plan) <= maximum:
            reason = (
                f"CALL2-PLAN 장면 수 범위 위반: count={len(scene_plan)}, "
                f"required={minimum}..{maximum}"
            )
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason

    keyvis_descriptor = None
    raw_keyvis = raw.get("keyvis")
    if toggles.get("key_visual"):
        if not isinstance(raw_keyvis, dict):
            reason = "CALL2-PLAN keyvis가 없거나 object가 아님"
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason
        keyvis_descriptor = _descriptor(raw_keyvis, "keyvis", -1)
        if (
            not keyvis_descriptor.get("camera")
            or not keyvis_descriptor.get("scene")
            or not keyvis_descriptor.get("characters")
        ):
            reason = "CALL2-PLAN keyvis 필수 camera/scene/characters가 비어 있음"
            if log_errors:
                print(f"[ILLUST_CONTEXT:CALL2_PLAN] {reason}")
            return None, reason
    elif isinstance(raw_keyvis, dict):
        print("[ILLUST_CONTEXT:CALL2_PLAN] Key Visual 비활성인데 keyvis가 반환되어 폐기")

    return {
        "mode": "plan",
        "scene_plan": scene_plan,
        "keyvis_descriptor": keyvis_descriptor,
        "descriptors": [],
    }, ""


def _parse_call2_detail_output(
    text: str,
    toggles: dict,
    assigned_slots: list[int],
    assigned_plan_ids: list[str],
    source: str,
) -> tuple[list[dict], str]:
    local_toggles = deepcopy(toggles)
    local_toggles.update({
        "key_visual": False,
        "scene_mode": "manual",
        "scene_min": len(assigned_slots),
        "scene_max": len(assigned_slots),
    })
    descriptors = [
        item
        for item in parse_toon_plan(text, local_toggles, source)
        if str(item.get("kind") or "") == "scene"
    ]
    actual_slots = []
    for item in descriptors:
        if (
            not str(item.get("camera") or "").strip()
            or not str(item.get("scene") or "").strip()
            or not (item.get("characters") or [])
        ):
            return [], f"CALL2-DETAIL 필수 camera/scene/characters가 비어 있음: item={item!r}"
        try:
            actual_slots.append(int(item.get("slot")))
        except Exception:
            return [], f"CALL2-DETAIL slot 파싱 실패: item={item!r}"
    if len(actual_slots) != len(assigned_slots):
        return [], (
            f"CALL2-DETAIL 장면 수 불일치: assigned={assigned_slots}, actual={actual_slots}"
        )
    if len(set(actual_slots)) != len(actual_slots) or set(actual_slots) != set(assigned_slots):
        return [], (
            f"CALL2-DETAIL slot 불일치: assigned={assigned_slots}, actual={actual_slots}"
        )
    actual_plan_ids = [str(item.get("plan_id") or "").strip() for item in descriptors]
    if actual_plan_ids != assigned_plan_ids:
        return [], (
            f"CALL2-DETAIL plan_id 불일치: assigned={assigned_plan_ids}, "
            f"actual={actual_plan_ids}"
        )
    by_slot = {int(item["slot"]): item for item in descriptors}
    return [by_slot[slot] for slot in assigned_slots], ""


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


async def _run_parallel_call2_details(
    *,
    scene_plan: list[dict],
    keyvis_descriptor: dict | None,
    call2_context_messages: list[dict],
    call2_format: str,
    toggles: dict,
    stream_notify,
) -> tuple[list[dict], list[str]]:
    max_concurrency = int(toggles["call2_parallel_max_concurrency"])
    batches = _balanced_call2_scene_plan_batches(scene_plan, max_concurrency)
    jobs = [{"plans": batch, "weight": len(batch)} for batch in batches]
    distribution = [len(batch) for batch in batches]
    print(
        f"[ILLUST_CONTEXT:CALL2_DETAIL] 상세 장면 배치 준비: "
        f"selected_scenes={len(scene_plan)}, workers={len(jobs)}, "
        f"distribution={distribution}"
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
        messages = deepcopy(call2_context_messages)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = str(messages[0].get("content") or "") + (
                "\n\n# Parallel CALL2-DETAIL override\n"
                f"The global planner already selected the visual beats. Output exactly {len(plans)} "
                f"scenes for assigned slots {assigned_slots}. Do not select, add, remove, or move a scene. "
                "Omit keyvis completely. This shard-specific rule overrides global scene-count and "
                "key-visual requirements above."
            )
        messages.extend([{
            "role": "user",
            "content": (
                "# ASSIGNED GLOBAL SCENE PLAN\n"
                + json.dumps(plans, ensure_ascii=False, indent=2)
                + "\n\nExpand each plan into complete Danbooru-style character tags, camera, scene, "
                "outfit_state, and supplement. Copy plan_id and slot exactly into every scene object, "
                "and preserve plan order.\n\n"
                "# OUTPUT FORMAT\n"
                + call2_format
                + "\n\nReturn one <lb-xnai> block containing scenes only. Omit keyvis."
            ),
        }])

        def validate(result):
            parsed, reason = _parse_call2_detail_output(
                result,
                toggles,
                assigned_slots,
                assigned_plan_ids,
                f"CALL2-DETAIL-{index}-RETRY-CHECK",
            )
            return bool(parsed), reason or "CALL2-DETAIL 파싱 실패"

        call_name = f"CALL2-DETAIL {index}/{total}"
        if attempt_kind == "duplicate":
            call_name += " [느리다고? 다시해!]"
        raw_output = await _call_pipeline_llm(
            call_name,
            _normalize_messages(messages),
            job_stream_notify,
            result_validator=validate,
            stream_observer=stream_observer,
            history_id=history_id,
        )
        descriptors, reason = _parse_call2_detail_output(
            raw_output,
            toggles,
            assigned_slots,
            assigned_plan_ids,
            f"CALL2-DETAIL-{index}",
        )
        if not descriptors:
            raise ValueError(reason or f"CALL2-DETAIL {index}/{total} 파싱 실패")
        return {"raw": raw_output, "descriptors": descriptors}

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
    descriptors = [deepcopy(keyvis_descriptor)] if keyvis_descriptor else []
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
    """CALL3가 선택된 모든 slot만 빠짐없이 작성했는지 검증한다."""
    expected = list(dict.fromkeys(int(slot) for slot in expected_slots))
    parsed = parse_speak_output(text)
    actual = list(parsed)
    emitted_headers = [
        int(match.group(1))
        for match in re.finditer(
            r"(?im)^\s*\[Scene\s+slot\s*=\s*(-?\d+)\]",
            str(text or ""),
        )
    ]
    missing = [slot for slot in expected if slot not in parsed]
    unexpected = [slot for slot in actual if slot not in expected]
    if missing or unexpected or emitted_headers != expected:
        reason = (
            f"CALL3 선택 slot 불일치: expected={expected}, actual={actual}, "
            f"headers={emitted_headers}, missing={missing}, unexpected={unexpected}"
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
    language_instruction = (
        "# OUTPUT LANGUAGE — HARD REQUIREMENT\n"
        f"Write every dialogue, thought, inner monologue, and newly created reaction in {output_language}.\n"
        "Character names, [Scene slot=N] headers, and required output tags may remain in their "
        "prescribed form. Do not switch the spoken text to another language even when the source "
        "narrative or examples use another language. Before answering, silently verify that every "
        f"spoken or thought line follows the required output language: {output_language}."
    )
    system_prompt = language_instruction + "\n\n" + system_prompt
    system_prompt += emotion_instruction
    system_prompt += nsfw_instruction
    return prompt_mode, system_prompt


def _build_character_history(extra_reference: str) -> str:
    # 서버가 보유한 lb.extra 자체가 가장 안정적인 외형 이력/영문 이름 사전이다.
    return str(extra_reference or "").strip()


# 삽화 CALL 이름 → 외부 API 분기 task_key. 각 CALL 을 llm_routing 에서 독립적으로
# 분기(LLM1/LLM2/LLM3)할 수 있다. 기본 primary=llm1(server.py DEFAULT_CONFIG 참고).
_CALL_TASK_KEYS = {
    "CALL1-BACKTRANSLATE": "illustration_call1_backtranslate",
    "CALL1": "illustration_call1",
    "CALL2": "illustration_call2",
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
) -> str:
    """삽화 CALL1/2/3 의 LLM 호출. 외부 API 분기(illustration_callN task_key)를 경유한다.

    외부 API 분기 탭에서 CALL별로 LLM1/LLM2/LLM3 을 선택하거나 폴백을 켤 수 있다.
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
    model = (
        llm_service.routing_primary_model(task_key)
        or llm_service._current_config.get("llm_model3")
        or llm_service._current_config.get("llm_model")
        or ""
    )
    history_record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "prompt_id": f"illustration_context:{call_name}",
        "call_name": call_name,
        "task_key": task_key,
        "model": model,
        "input": messages,
        "output": "",
        "completion_tokens": 0,
        "elapsed": 0.0,
        "tps": 0.0,
    }
    if history_id:
        history_record["history_id"] = str(history_id)
    history_logged = False
    terminal_notified = False

    async def _notify(event: dict):
        # stream_notify 이벤트에 큐 서브태스크 그룹을 주입한다.
        # 역번역/다중캐릭터마스크 wrapper가 이미 queue_subtask를 넣은 경우 유지한다.
        if not stream_notify:
            return
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

    try:
        if stream_notify:
            await _notify({
                "type": "start", "call_name": call_name, "model": model, "text": "",
            })
        call_kwargs = {}
        if result_validator is not None:
            call_kwargs["result_validator"] = result_validator
        if json_mode:
            call_kwargs["json_mode"] = True
        if stream_observer is not None:
            call_kwargs["stream_observer"] = stream_observer
        result = await llm_service.callLLMTask(task_key, messages, **call_kwargs)
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
        })
        lighbd_service._log_lighbd_history(history_record)
        history_logged = True
        return str(result)
    except asyncio.CancelledError:
        if history_id and not history_logged:
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
            f"history_id={history_id or '(none)'}"
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
                }
            elif race and attempt_kind == race["loser"]:
                history_updates[history_id] = {
                    "call_name": f"{group_label} {index}/{len(jobs)} [{role_label} · 패배]",
                    "status": "race_lost",
                    "race_outcome": "loser",
                    "race_progress": float(race["loser_progress"]),
                    "race_streaming": bool(race["loser_streaming"]),
                    "race_elapsed": round(float(race["loser_elapsed"]), 3),
                }
            else:
                history_updates[history_id] = {
                    "call_name": f"{group_label} {index}/{len(jobs)} [{role_label} · 경주 실패]",
                    "race_outcome": "failed",
                }
    if history_updates:
        lighbd_service._update_lighbd_history_records(history_updates)

    if failed:
        for index, reason in sorted(failed.items()):
            print(
                f"[ILLUST_CONTEXT:{group_id}] 병렬 작업 최종 실패: "
                f"job={index}/{len(jobs)}, reason={reason}"
            )
        raise RuntimeError(
            f"{group_label} 병렬 작업 {len(failed)}/{len(jobs)}개 실패: "
            + "; ".join(f"{index}={reason}" for index, reason in sorted(failed.items()))
        )

    print(
        f"[ILLUST_CONTEXT:{group_id}] 병렬 작업 완료: "
        f"jobs={len(jobs)}, max_concurrency={concurrency}"
    )
    return [resolved[index]["value"] for index in range(1, len(jobs) + 1)]


def _merge_call1_shard_values(
    shard_values: list[dict],
    segment_order: list[str],
) -> tuple[dict, list[str]]:
    """Merge disjoint CALL1 shard JSON without semantic keyword inference."""
    merged = {
        "reference_assignments": [],
        "history_characters": [],
        "current_characters": [],
        "wardrobe_events": [],
        "unresolved_references": [],
    }
    errors = []
    history_seen = set()
    current_by_name: dict[str, dict] = {}
    assignment_by_key: dict[tuple, dict] = {}
    wardrobe_seen = set()
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
                errors.append(f"CALL1 shard {shard_index} 지칭 할당 형식 오류")
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id not in assigned_ids:
                errors.append(
                    f"CALL1 shard {shard_index} 담당 밖 지칭 할당: segment={segment_id!r}"
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
                errors.append(f"CALL1 shard 지칭 충돌: key={key!r}")
                continue
            assignment_by_key[key] = deepcopy(item)

        for item in raw.get("wardrobe_events") or []:
            if not isinstance(item, dict):
                errors.append(f"CALL1 shard {shard_index} 복장 이벤트 형식 오류")
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id and segment_id not in assigned_ids:
                errors.append(
                    f"CALL1 shard {shard_index} 담당 밖 복장 이벤트: segment={segment_id!r}"
                )
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in wardrobe_seen:
                wardrobe_seen.add(key)
                merged["wardrobe_events"].append(deepcopy(item))

        for item in raw.get("unresolved_references") or []:
            if not isinstance(item, dict):
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id and segment_id not in assigned_ids:
                errors.append(
                    f"CALL1 shard {shard_index} 담당 밖 미해결 지칭: segment={segment_id!r}"
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
    return merged, errors


async def _run_parallel_call1_analysis(
    *,
    call1_system: str,
    segmented_current: str,
    current_segments: dict[str, dict],
    history_text: str,
    toggles: dict,
    stream_notify,
) -> tuple[str, list[str]]:
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
            "but emit reference_assignments, wardrobe_events, and unresolved_references only "
            f"for these assigned segment IDs: {json.dumps(assigned, ensure_ascii=False)}.\n"
            "history_characters and current_characters may contain the complete names needed "
            "to understand those assigned segments. Return one JSON object using the existing schema."
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
    merged, merge_errors = _merge_call1_shard_values(shard_values, segment_ids)
    return json.dumps(merged, ensure_ascii=False), merge_errors


def _parse_multi_char_layout_response(text: str, expected_names: list[str]) -> dict:
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
    return multi_char_mask.validate_multi_char_layout(
        value,
        expected_names,
        require_prompt_separation=True,
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
                "visual_tags": str(character.get("positive") or ""),
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
                _parse_multi_char_layout_response(result, expected_names)
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

        try:
            result = await _call_pipeline_llm(
                f"MULTI-CHAR-MASK slot={slot}",
                messages,
                layout_stream_notify,
                result_validator=validate_result,
                json_mode=True,
            )
            descriptor["multi_char_layout_raw_response"] = str(result or "")
            layout = _parse_multi_char_layout_response(result, expected_names)
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
    extra_costume: str = "",
    extra_names: str = "",
    backtranslate_names: str = "",
    enable_multi_char_layout: bool = False,
    history_plan: dict | None = None,
) -> dict:
    toggles = merged_toggles(toggles)
    prompts = load_prompt_files()
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
    reference_variables: dict[str, str] = {}
    balanced_fallback = False
    enhanced = backtranslated_narrative or _strip_nodes(narrative)
    resolved_current = enhanced
    segmented_current, current_segments = _segment_current_context(enhanced)
    persistent_history = history_plan if isinstance(history_plan, dict) else None
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
        call1_system = call1_system.replace("{character_names}", str(backtranslate_names or extra_names or ""))
        call1_system = call1_system.replace(
            "{character_state}",
            json.dumps(
                (persistent_history or {}).get("state_before") or {},
                ensure_ascii=False,
                indent=2,
            ),
        )
        history_text = _history_messages_text(context_slice)
        parallel_call1_used = False
        parallel_merge_errors: list[str] = []
        should_parallel_call1 = (
            bool(toggles.get("call1_parallel_enabled"))
            and len(current_segments) > 1
        )
        if should_parallel_call1:
            try:
                call1_output, parallel_merge_errors = await _run_parallel_call1_analysis(
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
            )
            if parsed_call1 is None:
                balanced_fallback = True
                print(
                    "[ILLUST_CONTEXT:CALL1] 구조화 분석 실패로 균형형 CALL2 폴백 사용"
                )
            else:
                call1_result = parsed_call1
                if parallel_merge_errors:
                    parsed_call1["validation_errors"].extend(parallel_merge_errors)
                    parsed_call1["fallback_required"] = True
                    print(
                        f"[ILLUST_CONTEXT:CALL1_PARALLEL] shard 병합 검증 오류: "
                        f"errors={parallel_merge_errors}"
                    )
                wardrobe_events = list(parsed_call1.get("wardrobe_events") or [])
                resolved_current, assignment_errors, reference_variables = apply_reference_assignments(
                    enhanced,
                    current_segments,
                    parsed_call1.get("reference_assignments") or [],
                )
                if assignment_errors:
                    parsed_call1["validation_errors"].extend(assignment_errors)
                    parsed_call1["fallback_required"] = True
                    print(
                        f"[ILLUST_CONTEXT:CALL1] 지칭 치환 검증 실패: errors={assignment_errors}"
                    )
                balanced_fallback = bool(parsed_call1.get("fallback_required"))
                if balanced_fallback:
                    print(
                        f"[ILLUST_CONTEXT:CALL1] 균형형 폴백 조건 감지: "
                        f"errors={parsed_call1.get('validation_errors') or []}"
                    )
                enhanced = resolved_current
                slotted, slotted_assignment_errors = apply_reference_assignments_to_slotted(
                    slotted,
                    current_segments,
                    parsed_call1.get("reference_assignments") or [],
                )
                if slotted_assignment_errors:
                    parsed_call1["validation_errors"].extend(slotted_assignment_errors)
                    parsed_call1["fallback_required"] = True
                    balanced_fallback = True
                    print(
                        f"[ILLUST_CONTEXT:CALL1] 슬롯 보존 지칭 치환 실패: "
                        f"errors={slotted_assignment_errors}"
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

    history = _build_character_history(call2_reference)
    call2_system = render_call2_prompt(prompts.get("call2_system", ""), toggles, history)
    call2_thoughts = render_call2_prompt(prompts.get("call2_thoughts", ""), toggles, history)
    call2_messages = [{
        "role": "system",
        "content": "\n\n".join(x for x in (
            prompts.get("call2_jailbreak", ""), prompts.get("call2_job", ""), call2_system,
        ) if x.strip()),
    }]
    if call2_reference.strip():
        call2_messages.append({"role": "user", "content": "# CHARACTER DICTIONARY\n\n" + call2_reference})
    if persistent_history:
        if selected_states:
            call2_messages.append({
                "role": "user",
                "content": (
                    "# AUTHORITATIVE WARDROBE CONTINUITY STATE\n"
                    "Carry this state forward. Absence from a camera frame never means removal.\n\n"
                    + json.dumps(selected_states, ensure_ascii=False, indent=2)
                ),
            })
        if wardrobe_events:
            call2_messages.append({
                "role": "user",
                "content": (
                    "# CURRENT WARDROBE EVENT TIMELINE\n"
                    "Apply each event only from its segment onward.\n\n"
                    + json.dumps(wardrobe_events, ensure_ascii=False, indent=2)
                ),
            })
        if previous_visual:
            call2_messages.append({
                "role": "user",
                "content": (
                    "# LAST VISUAL REFERENCE\n"
                    "Use only appearance/attire continuity; ignore old pose, action, expression and framing.\n\n"
                    + json.dumps(previous_visual, ensure_ascii=False, indent=2)
                ),
            })
        if balanced_fallback:
            fallback_text = _history_messages_text(
                persistent_history.get("call2_fallback_history") or []
            )
            if fallback_text:
                call2_messages.append({
                    "role": "user",
                    "content": "# BALANCED FALLBACK PAST HISTORY\n\n" + fallback_text,
                })
            print(
                f"[ILLUST_CONTEXT:CALL2] 균형형 폴백 입력 사용: "
                f"history_chars={len(fallback_text)}, full_reference={bool(call2_reference.strip())}"
            )
    else:
        for item in chats[max(0, target_index - int(toggles["call2_context_turns"])):target_index]:
            call2_messages.append({
                "role": "assistant" if item["role"] == "char" else "user",
                "content": _strip_nodes(item["data"]),
            })
    call2_messages.append({"role": "user", "content": "[Last log entry]\n" + slotted})
    call2_context_messages = deepcopy(call2_messages)
    call2_messages.append({
        "role": "user",
        "content": "# Output instructions\n\n" + call2_thoughts + "\n\n" + prompts.get("call2_format", ""),
    })
    if prompts.get("call2_prefill", "").strip():
        call2_messages.append({"role": "assistant", "content": prompts["call2_prefill"]})
    call2_messages.append({"role": "user", "content": "Return the final <lb-xnai> TOON block only after your analysis."})
    call2_output = ""
    call2_plan_output = ""
    call2_detail_outputs: list[str] = []
    descriptors = []
    if toggles.get("call2_parallel_enabled"):
        try:
            if progress:
                await progress(31, "call2_plan", "CALL2 전역 장면·키비주얼 계획")
            candidates = candidate_slots(original_slotted)
            plan_messages = deepcopy(call2_context_messages)
            call2_segment_map, _call2_segments = _segment_current_context(enhanced)
            if plan_messages and plan_messages[0].get("role") == "system":
                plan_messages[0]["content"] = str(plan_messages[0].get("content") or "") + (
                    "\n\n# CALL2-PLAN override\n"
                    "Select visual beats globally from the full last log entry before any parallel detail "
                    "work begins. Return compact JSON only. Do not return <lb-xnai>. The scene_plan is a "
                    "binding assignment: detail workers may describe it but may not select different scenes. "
                    "When Key Visual is enabled, generate its complete final camera/characters/tags/scene/"
                    "supplement object here so no fourth parallel key-visual call is needed."
                )
            if str(toggles.get("scene_mode")) == "auto":
                scene_count_rule = (
                    f"Choose the appropriate count from the {len(candidates)} available slots."
                )
            else:
                minimum = min(int(toggles["scene_min"]), len(candidates))
                maximum = min(int(toggles["scene_max"]), len(candidates))
                scene_count_rule = f"Choose between {minimum} and {maximum} scenes."
            keyvis_rule = (
                "Return one complete keyvis object."
                if toggles.get("key_visual")
                else "Return keyvis as null."
            )
            plan_messages.append({
                "role": "user",
                "content": (
                    "# GLOBAL CALL2 PLAN\n"
                    f"Candidate slots in narrative order: {json.dumps(candidates)}\n"
                    + scene_count_rule
                    + " Each selected slot must be unique and must belong to the candidate list. "
                    "Select at most one scene per semantic visual beat.\n"
                    + keyvis_rule
                    + "\n\n# GLOBAL SELECTION POLICY\n"
                    + call2_thoughts
                    + "\n\nApply the policy above while planning, but the final-response JSON-only "
                    "contract below overrides any draft-output wording in that policy."
                    + "\n\nReturn this JSON schema only:\n"
                    "{\n"
                    '  "scene_plan": [\n'
                    "    {\n"
                    '      "plan_id": "S001",\n'
                    '      "slot": 0,\n'
                    '      "source_segments": ["C001"],\n'
                    '      "characters": ["canonical name"],\n'
                    '      "scene_brief": "objective visual moment to expand"\n'
                    "    }\n"
                    "  ],\n"
                    '  "keyvis": {\n'
                    '    "camera": "...",\n'
                    '    "characters": [{"positive":"...","negative":"...","name":"...",'
                    '"position":"...","outfit_state":{"body_state":"clothed",'
                    '"worn":[],"removed":[]}}],\n'
                    '    "scene": "...",\n'
                    '    "supplement": "..."\n'
                    "  }\n"
                    "}\n\n"
                    "Use semantic context and common sense for visual-beat selection; do not use keyword rules.\n\n"
                    "# SERVER SEGMENT MAP\n"
                    + call2_segment_map
                ),
            })

            def validate_plan(result):
                plan, reason = parse_call2_plan(
                    result,
                    toggles,
                    original_slotted,
                    log_errors=False,
                )
                return bool(plan), reason or "CALL2-PLAN 파싱 실패"

            call2_plan_output = await _call_pipeline_llm(
                "CALL2-PLAN",
                _normalize_messages(plan_messages),
                stream_notify,
                result_validator=validate_plan,
                json_mode=True,
            )
            parsed_plan, plan_reason = parse_call2_plan(
                call2_plan_output,
                toggles,
                original_slotted,
            )
            if parsed_plan is None:
                raise ValueError(plan_reason or "CALL2-PLAN 파싱 실패")
            if parsed_plan["mode"] == "legacy":
                descriptors = list(parsed_plan.get("descriptors") or [])
                call2_output = call2_plan_output
                print(
                    "[ILLUST_CONTEXT:CALL2_PLAN] 모델이 완성 TOON을 반환해 "
                    "기존 단일 CALL2 결과로 수용"
                )
            else:
                if progress:
                    await progress(
                        36,
                        "call2_detail",
                        f"CALL2 상세 장면 {len(parsed_plan['scene_plan'])}개 병렬 생성",
                    )
                descriptors, call2_detail_outputs = await _run_parallel_call2_details(
                    scene_plan=list(parsed_plan["scene_plan"]),
                    keyvis_descriptor=parsed_plan.get("keyvis_descriptor"),
                    call2_context_messages=call2_context_messages,
                    call2_format=prompts.get("call2_format", ""),
                    toggles=toggles,
                    stream_notify=stream_notify,
                )
                call2_output = descriptors_to_toon(descriptors)
        except asyncio.CancelledError:
            print("[ILLUST_CONTEXT:CALL2_PARALLEL] 상위 작업 취소로 병렬 CALL2 중단")
            raise
        except Exception as e:
            print(
                f"[ILLUST_CONTEXT:CALL2_PARALLEL] 병렬 CALL2 실패로 단일 CALL2 폴백: {e}"
            )
            traceback.print_exc()
            call2_output = ""
            call2_plan_output = ""
            call2_detail_outputs = []
            descriptors = []

    if not descriptors:
        call2_output = await _call_pipeline_llm(
            "CALL2",
            _normalize_messages(call2_messages),
            stream_notify,
            result_validator=lambda result: (
                bool(parse_toon_plan(result, toggles, "CALL2-RETRY-CHECK")),
                "CALL2 TOON 파싱 실패",
            ),
        )
        descriptors = parse_toon_plan(call2_output, toggles, "CALL2")

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
            retry_messages = deepcopy(call2_messages)
            retry_messages.extend([{
                "role": "assistant",
                "content": call2_output,
            }, {
                "role": "user",
                "content": (
                    "Character coverage did not match CALL1. Re-evaluate the current context "
                    "with the bounded past history and full character dictionary below. "
                    "Preserve established wardrobe state unless a supplied wardrobe event changes it.\n\n"
                    "# FULL CHARACTER DICTIONARY\n"
                    + str(extra_reference or "")
                    + "\n\n# BOUNDED PAST HISTORY\n"
                    + (_history_messages_text(
                        persistent_history.get("call2_fallback_history") or []
                    ) or "(empty)")
                    + "\n\nReturn the complete corrected <lb-xnai> block only."
                ),
            }])
            retried_output = await _call_pipeline_llm(
                "CALL2",
                _normalize_messages(retry_messages),
                stream_notify,
                result_validator=lambda result: (
                    bool(parse_toon_plan(result, toggles, "CALL2-COVERAGE-RETRY-CHECK")),
                    "CALL2 캐릭터 커버리지 재시도 파싱 실패",
                ),
            )
            retried_descriptors = parse_toon_plan(
                retried_output,
                toggles,
                "CALL2-COVERAGE-RETRY",
            )
            if retried_descriptors:
                call2_output = retried_output
                descriptors = retried_descriptors

    # CALL2 파싱 실패 시 CALL2-FIX(repair.txt)가 TOON 블록을 교정한다.
    # CALL3는 대사 생성 전용이므로 교정은 여기서 먼저 마무리한다.
    call2_fix_output = ""
    if not descriptors:
        if progress:
            await progress(48, "call2_fix", "CALL2-FIX TOON 교정")
        fix_messages = [{
            "role": "system",
            "content": prompts.get("call2_fix", "") + "\n\n" + call2_reference,
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
    descriptors = attach_descriptor_anchors(
        descriptors,
        original_slotted,
    )
    descriptors = sanitize_descriptor_slots(descriptors, original_slotted)
    if not descriptors:
        print("[ILLUST_CONTEXT:CALL2] 슬롯 보정 후 생성할 descriptor가 없음")
        raise RuntimeError("CALL2 결과에 유효한 장면 슬롯이 없습니다")

    last_visual_by_character = _last_visual_by_character(descriptors)
    character_states_after = deepcopy((persistent_history or {}).get("state_before") or {})
    if persistent_history:
        character_states_after = apply_wardrobe_events(
            character_states_after,
            call1_result.get("current_characters") or [],
            wardrobe_events,
            str(persistent_history.get("current_message_id") or ""),
            selected_reference=call2_reference,
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
        call3_output = await _call_pipeline_llm("CALL3", _normalize_messages(speak_messages), stream_notify)
        call3_initial_output = call3_output
        call3_valid, call3_failure_reason = validate_call3_slot_coverage(
            call3_output,
            selected_slots,
        )
        if not call3_valid:
            call3_correction_used = True
            print(
                f"[ILLUST_CONTEXT:CALL3-CORRECTION] 최초 CALL3 결과의 선택 slot이 불완전해 "
                f"교정 호출 1회 실행: "
                f"slots={selected_slots}, reason={call3_failure_reason}"
            )
            retry_messages = deepcopy(speak_messages)
            retry_messages.extend([{
                "role": "assistant",
                "content": call3_output,
            }, {
                "role": "user",
                "content": (
                    "Your previous output violated the selected-scene contract. "
                    f"Required slots, in order: {selected_slots}. "
                    "Rewrite the entire output. Emit exactly one [Scene slot=N] block "
                    "for every required slot and no block for any other slot. "
                    "Every block must contain at least one dialogue, thought, or inner "
                    "monologue entry. "
                    f"Write every dialogue and thought in {speak_language}; this language rule is mandatory. "
                    "Character names, Scene headers, and required tags are the only exceptions. "
                    "Output only the corrected Scene blocks."
                ),
            }])
            call3_output = await _call_pipeline_llm(
                "CALL3-CORRECTION",
                _normalize_messages(retry_messages),
                stream_notify,
                result_validator=lambda result: validate_call3_slot_coverage(
                    result,
                    selected_slots,
                ),
            )
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
    return {
        "session_id": payload["session_id"],
        "context": downstream_context,
        "narrative": narrative,
        "backtranslated_narrative": backtranslated_narrative,
        "backtranslated_slotted": backtranslated_slotted,
        "backtranslation_chunks": backtranslation_chunks,
        "enhanced_narrative": enhanced,
        "call1_output": call1_output,
        "call1_result": call1_result,
        "reference_variables": reference_variables,
        "wardrobe_events": wardrobe_events,
        "balanced_fallback_used": balanced_fallback,
        "call2_output": call2_output,
        "call2_plan_output": call2_plan_output,
        "call2_detail_outputs": call2_detail_outputs,
        "call2_fix_output": call2_fix_output,
        "call3_output": call3_output,
        "call3_initial_output": call3_initial_output,
        "call3_correction_used": call3_correction_used,
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
    }
