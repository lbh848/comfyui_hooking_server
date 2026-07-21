"""CHAT -> CALL1/CALL2/CALL3 -> 기존 RAW 삽화 프롬프트 전단계.

RisuAI는 Comfy history에 이미지가 여러 장 있어도 첫 장만 소비한다. 이 모듈은
최초 CHAT 요청과 후속 결과 회수 요청을 구분하고, 한 세션의 모든 장면 프롬프트와
이미지를 서버 메모리에 보관할 수 있는 공통 형식을 제공한다.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import traceback
import datetime
from copy import deepcopy
from urllib.parse import quote

import yaml

from modes import lighbd_service, llm_service


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
}

DEFAULT_TOGGLES = {
    "call1_backtranslate_enabled": False,
    "call1_backtranslate_max_concurrency": 4,
    "call1_backtranslate_failure_strategy": "fallback",
    "call1_enabled": True,
    "call1_context_turns": 5,
    "call2_context_turns": 5,
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
    try:
        out["call1_backtranslate_max_concurrency"] = max(
            1,
            min(16, int(out["call1_backtranslate_max_concurrency"])),
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
            "call1_context_turns": DEFAULT_TOGGLES["call1_context_turns"],
            "call2_context_turns": DEFAULT_TOGGLES["call2_context_turns"],
            "call3_context_turns": DEFAULT_TOGGLES["call3_context_turns"],
            "character_limit": DEFAULT_TOGGLES["character_limit"],
            "scene_mode": DEFAULT_TOGGLES["scene_mode"],
            "scene_min": DEFAULT_TOGGLES["scene_min"],
            "scene_max": DEFAULT_TOGGLES["scene_max"],
        })
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
        print(f"[ILLUST_CONTEXT] {prefix} JSON 파싱 실패: {e}; raw={raw[:240]!r}")
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
            print(f"[ILLUST_CONTEXT:CALL1] 삽입 위치를 찾지 못함: {anchor[:120]!r}")
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
            print(f"[ILLUST_CONTEXT] dictelement 렌더 실패: {e}; expr={match.group(0)[:160]!r}")
            traceback.print_exc()
            return ""

    text = re.sub(r"\{\{dictelement::(\{[^{}]*\})::([^{}]*)\}\}", dict_element, text)
    text = _render_conditionals(text, risu_values)
    # Risu에만 존재하는 잔여 매크로는 LLM으로 보내지 않고 로그에 남긴다.
    leftovers = re.findall(r"\{\{[^\n]{0,240}?\}\}", text)
    if leftovers:
        print(f"[ILLUST_CONTEXT] 렌더 후 잔여 Risu 매크로 {len(leftovers)}개 제거: {leftovers[:8]}")
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
        })
    slot_value = -1 if kind == "keyvis" else raw.get("slot", fallback_slot)
    try:
        slot_value = int(slot_value)
    except Exception:
        slot_value = fallback_slot
    return {
        "kind": kind,
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
        print(f"[ILLUST_CONTEXT:{source}] TOON/YAML 파싱 실패: {e}\n{inner[:1000]}")
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
}


async def _call_pipeline_llm(
    call_name: str,
    messages: list[dict],
    stream_notify=None,
    result_validator=None,
) -> str:
    """삽화 CALL1/2/3 의 LLM 호출. 외부 API 분기(illustration_callN task_key)를 경유한다.

    외부 API 분기 탭에서 CALL별로 LLM1/LLM2/LLM3 을 선택하거나 폴백을 켤 수 있다.
    실패 시 callLLMTask 가 지정된 폴백 LLM 으로 재시도한다.
    """
    started = time.time()
    task_key = _CALL_TASK_KEYS.get(call_name)
    if task_key is None and call_name.startswith("CALL1-BACKTRANSLATE"):
        task_key = _CALL_TASK_KEYS["CALL1-BACKTRANSLATE"]
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
    history_logged = False
    terminal_notified = False
    try:
        if stream_notify:
            await stream_notify({
                "type": "start", "call_name": call_name, "model": model, "text": "",
            })
        if result_validator is None:
            result = await llm_service.callLLMTask(task_key, messages)
        else:
            result = await llm_service.callLLMTask(
                task_key,
                messages,
                result_validator=result_validator,
            )
        if not result or str(result).startswith("[LLM 실패]"):
            print(f"[ILLUST_CONTEXT:{call_name}] LLM 호출 실패: {result}")
            if stream_notify:
                await stream_notify({"type": "error", "call_name": call_name, "error": str(result)})
                terminal_notified = True
            raise RuntimeError(str(result or f"빈 {call_name} 응답"))
        elapsed = time.time() - started
        tokens = max(1, len(str(result)) // 3)
        prompt_tokens = llm_service._approx_input_tokens(messages)
        if stream_notify:
            await stream_notify({
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
    except Exception as e:
        if stream_notify and not terminal_notified:
            try:
                await stream_notify({
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


async def backtranslate_current_context(
    source: str,
    prompt: str,
    character_names: str,
    max_concurrency: int,
    failure_strategy: str = "fallback",
    stream_notify=None,
) -> tuple[str, list[dict]]:
    """current context만 영어로 병렬 역번역하고 선택된 실패 전략을 적용한다."""
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

    async def translate_one(index: int, chunk: str) -> tuple[str, dict]:
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

        messages = _normalize_messages([
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"[Current Response Chunk {index}/{len(chunks)}]\n"
                    "Return only the English translation of the chunk body below.\n\n"
                    + chunk
                ),
            },
        ])
        last_reason = "unknown_failure"
        validation_attempts = 0

        def _validate_translation(translated):
            nonlocal validation_attempts, last_reason
            validation_attempts += 1
            valid, reason = _valid_backtranslation(chunk, translated)
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
        try:
            translated = await _call_pipeline_llm(
                call_name,
                messages,
                chunk_stream_notify,
                result_validator=_validate_translation,
            )
        except Exception as e:
            last_reason = f"call_failed: {e}"
            print(
                f"[ILLUST_CONTEXT:BACKTRANSLATE] 청크 호출 실패: "
                f"strategy={strategy}, chunk={index}/{len(chunks)}, "
                f"input_len={len(chunk)}, error={e}"
            )
            traceback.print_exc()
        else:
            valid, reason = _valid_backtranslation(chunk, translated)
            if valid:
                return str(translated).strip(), {
                    "index": index,
                    "status": "translated",
                    "reason": "",
                    "attempts": max(1, validation_attempts),
                }
            last_reason = reason
            print(
                f"[ILLUST_CONTEXT:BACKTRANSLATE] 청크 최종 응답 실패: "
                f"strategy={strategy}, chunk={index}/{len(chunks)}, "
                f"input_len={len(chunk)}, output_len={len(str(translated or ''))}, "
                f"reason={reason}"
            )

        if strict:
            raise RuntimeError(
                f"청크 {index}/{len(chunks)}가 설정된 라우팅 재시도 후에도 실패했습니다: "
                f"{last_reason}"
            )
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE] 실패 청크 원문 폴백: "
            f"chunk={index}/{len(chunks)}, reason={last_reason}"
        )
        return chunk.strip(), {
            "index": index,
            "status": "fallback",
            "reason": last_reason,
            "attempts": max(1, validation_attempts),
        }

    gathered = await asyncio.gather(*(
        translate_one(index, chunk)
        for index, chunk in enumerate(chunks, start=1)
    ), return_exceptions=True)
    cancellations = [
        result for result in gathered if isinstance(result, asyncio.CancelledError)
    ]
    if cancellations:
        print(
            f"[ILLUST_CONTEXT:BACKTRANSLATE] 병렬 역번역 취소됨: "
            f"cancelled={len(cancellations)}/{len(chunks)}"
        )
        raise cancellations[0]
    failures = [
        (index, result)
        for index, result in enumerate(gathered, start=1)
        if isinstance(result, Exception)
    ]
    if failures:
        for index, error in failures:
            print(
                f"[ILLUST_CONTEXT:BACKTRANSLATE] 청크 처리 예외: "
                f"strategy={strategy}, chunk={index}/{len(chunks)}, error={error}"
            )
            traceback.print_exception(type(error), error, error.__traceback__)
        if strict:
            raise RuntimeError(
                f"CALL1 역번역 엄격 전략 실패: "
                f"{len(failures)}/{len(chunks)}개 청크 실패"
            ) from failures[0][1]
        for index, error in failures:
            gathered[index - 1] = (
                chunks[index - 1].strip(),
                {
                    "index": index,
                    "status": "fallback",
                    "reason": f"unexpected_error: {error}",
                    "attempts": 1,
                },
            )

    results = gathered
    translated_chunks = [value for value, _status in results]
    statuses = [status for _value, status in results]
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
    enhanced = backtranslated_narrative or _strip_nodes(narrative)
    if toggles.get("call1_enabled"):
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
        call1_messages = [{"role": "system", "content": call1_system}]
        call1_messages.extend({
            "role": "assistant" if item["role"] == "char" else "user",
            "content": _strip_nodes(item["data"]),
        } for item in context_slice)
        call1_messages.append({"role": "user", "content": "---\n[Current Response]\n" + enhanced})
        call1_output = await _call_pipeline_llm("CALL1", _normalize_messages(call1_messages), stream_notify)
        enhanced = _splice_enhancements(enhanced, call1_output)
    else:
        print("[ILLUST_CONTEXT:CALL1] 토글로 비활성화됨")

    # Risu는 결과 메타데이터를 텍스트로 받을 수 없고 generateImage만 반복 호출한다.
    # 결과를 slot 번호로 회수할 수 있도록 슬롯 번호는 원문 문단을 기준으로 고정한다.
    # 모듈이 v13 규칙으로 원문 XML을 보존한 채 먼저 삽입한 슬롯 맵을 우선한다.
    # 없으면 구버전/테스트 호환을 위해 필터된 최신 narrative로 생성한다.
    if call1_output:
        # CALL2에는 CALL1이 만든 Visual Content/DynamicPrompt를 넘기되,
        # 결과 회수에 쓰는 [Slot N] 번호는 원문 문단 기준으로 그대로 유지한다.
        slotted = _splice_enhancements(slotted, call1_output)
    if progress:
        await progress(30, "call2", "CALL2 장면/태그 빌드")
    history = _build_character_history(extra_reference)
    call2_system = render_call2_prompt(prompts.get("call2_system", ""), toggles, history)
    call2_thoughts = render_call2_prompt(prompts.get("call2_thoughts", ""), toggles, history)
    call2_messages = [{
        "role": "system",
        "content": "\n\n".join(x for x in (
            prompts.get("call2_jailbreak", ""), prompts.get("call2_job", ""), call2_system,
        ) if x.strip()),
    }]
    if extra_reference.strip():
        call2_messages.append({"role": "user", "content": "# CHARACTER DICTIONARY\n\n" + extra_reference})
    for item in chats[max(0, target_index - int(toggles["call2_context_turns"])):target_index]:
        call2_messages.append({
            "role": "assistant" if item["role"] == "char" else "user",
            "content": _strip_nodes(item["data"]),
        })
    call2_messages.append({"role": "user", "content": "[Last log entry]\n" + slotted})
    call2_messages.append({
        "role": "user",
        "content": "# Output instructions\n\n" + call2_thoughts + "\n\n" + prompts.get("call2_format", ""),
    })
    if prompts.get("call2_prefill", "").strip():
        call2_messages.append({"role": "assistant", "content": prompts["call2_prefill"]})
    call2_messages.append({"role": "user", "content": "Return the final <lb-xnai> TOON block only after your analysis."})
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

    # CALL2 파싱 실패 시 CALL2-FIX(repair.txt)가 TOON 블록을 교정한다.
    # CALL3는 대사 생성 전용이므로 교정은 여기서 먼저 마무리한다.
    call2_fix_output = ""
    if not descriptors:
        if progress:
            await progress(48, "call2_fix", "CALL2-FIX TOON 교정")
        fix_messages = [{
            "role": "system",
            "content": prompts.get("call2_fix", "") + "\n\n" + extra_reference,
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

    downstream_chats = deepcopy(chats)
    if toggles.get("call1_backtranslate_enabled"):
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
    # 동시에 실행되며, 교정이 일어났으면 교정 결과의 TOON 블록을 장면 목록으로 넘긴다.
    call3_output = ""
    scene_source = call2_fix_output or call2_output
    if toggles.get("call3_enabled") and toggles.get("speak_enabled"):
        if progress:
            await progress(58, "call3", "CALL3 대사 빌드")
        call3_prompt_mode, call3_system_prompt = build_call3_dialogue_system_prompt(
            prompts,
            toggles,
            extra_names,
        )
        speak_messages = [{
            "role": "system",
            "content": call3_system_prompt,
        }]
        for item in chats[max(0, target_index - int(toggles["call3_context_turns"])):target_index]:
            speak_messages.append({
                "role": "assistant" if item["role"] == "char" else "user",
                "content": _strip_nodes(item["data"]),
            })
        speak_messages.append({
            "role": "user",
            "content": (
                f"[Narrative to illustrate]\n{enhanced}\n\n[Scene list]\n{_extract_lb_block(scene_source)}"
                f"\n\nLanguage: {toggles.get('speak_language', '한국어')}"
            ),
        })
        call3_output = await _call_pipeline_llm("CALL3", _normalize_messages(speak_messages), stream_notify)
        speak_map = parse_speak_output(
            call3_output,
            max_entries_per_scene=2 if call3_prompt_mode == "speak" else None,
        )
        for descriptor in descriptors:
            descriptor["speak"] = speak_map.get(int(descriptor.get("slot", 0)), "")
    else:
        print("[ILLUST_CONTEXT:CALL3] 토글로 비활성화되었거나 SPEAK이 꺼져 있음")

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
        "call2_output": call2_output,
        "call2_fix_output": call2_fix_output,
        "call3_output": call3_output,
        "prompt_format": prompt_format,
        "items": raw_items,
    }
