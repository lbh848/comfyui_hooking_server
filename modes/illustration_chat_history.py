"""Persistent illustration chat history with continuation and reroll detection.

Runtime records live below ``workflow_backup`` so they are never committed.  A
record keeps the original CHAT messages separately from derived CALL results.
Incoming requests contain a past-context snapshot plus the current CHAR
response; only the non-overlapping delta is appended.  When the past snapshot
matches the base of the active turn and only the current response changed, the
active turn is replaced as a Risu reroll.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from copy import deepcopy


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_ROOT = os.path.join(BASE_DIR, "workflow_backup", "illustration_chat_history")
RECORDS_DIR = os.path.join(HISTORY_ROOT, "records")
TRASH_DIR = os.path.join(HISTORY_ROOT, "trash")
SETTINGS_PATH = os.path.join(HISTORY_ROOT, "settings.json")
REQUIREMENTS_BACKUP_DIR = os.path.join(
    BASE_DIR,
    "요구사항",
    "illustration_chat_history_backups",
)

DEFAULT_SETTINGS = {
    "storage_max_chars": 50_000,
    "call1_history_chars": 30_000,
    "call2_fallback_history_chars": 12_000,
    "call3_fallback_history_chars": 6_000,
    "reroll_archive_limit": 3,
}

_IO_LOCK = threading.RLock()
_SAFE_HISTORY_ID = re.compile(r"hist_[0-9a-f]{32}")


def _ensure_dirs() -> None:
    try:
        os.makedirs(RECORDS_DIR, exist_ok=True)
        os.makedirs(TRASH_DIR, exist_ok=True)
        os.makedirs(REQUIREMENTS_BACKUP_DIR, exist_ok=True)
    except Exception as e:
        print(f"[ILLUST_HISTORY] 저장 폴더 생성 실패: root={HISTORY_ROOT}, error={e}")
        traceback.print_exc()
        raise


def _record_path(history_id: str) -> str:
    value = str(history_id or "")
    if not _SAFE_HISTORY_ID.fullmatch(value):
        raise ValueError(f"잘못된 illustration history id: {value!r}")
    return os.path.join(RECORDS_DIR, f"{value}.json")


def _normalize_content(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _content_hash(value: str) -> str:
    normalized = _normalize_content(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _snapshot_hash(messages: list[dict]) -> str:
    joined = "\n".join(
        str(message.get("role") or "user")
        + ":"
        + str(message.get("full_content_hash") or _content_hash(message.get("content", "")))
        for message in messages
    )
    return hashlib.sha256(joined.encode("ascii", errors="ignore")).hexdigest()


def _message_from_input(item: dict) -> dict:
    content = str(item.get("data") or item.get("content") or "")
    role = "char" if str(item.get("role") or "user").lower() in ("char", "assistant") else "user"
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "role": role,
        "content": content,
        "full_content_hash": _content_hash(content),
        "trimmed_prefix_chars": 0,
        "created_at": time.time(),
    }


def _atomic_write_json(path: str, value: dict) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_illust_history_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except Exception as cleanup_error:
            print(
                f"[ILLUST_HISTORY] 임시 파일 정리 실패: path={temporary}, "
                f"error={cleanup_error}"
            )
            traceback.print_exc()
        raise


def _backup_existing(path: str, label: str) -> str:
    if not os.path.isfile(path):
        return ""
    _ensure_dirs()
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(label or "history"))[:96]
    backup = os.path.join(REQUIREMENTS_BACKUP_DIR, f"{safe_label}.{stamp}.bak")
    try:
        shutil.copy2(path, backup)
        return backup
    except Exception as e:
        print(
            f"[ILLUST_HISTORY] 기존 데이터 백업 실패: source={path}, "
            f"backup={backup}, error={e}"
        )
        traceback.print_exc()
        raise


def _load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            print(f"[ILLUST_HISTORY] JSON 루트가 object가 아님: path={path}")
            return None
        return value
    except Exception as e:
        print(f"[ILLUST_HISTORY] JSON 읽기 실패: path={path}, error={e}")
        traceback.print_exc()
        return None


def normalize_settings(value: dict | None) -> dict:
    source = value if isinstance(value, dict) else {}
    out = deepcopy(DEFAULT_SETTINGS)
    try:
        out["storage_max_chars"] = max(
            1_000,
            min(500_000, int(source.get("storage_max_chars", out["storage_max_chars"]))),
        )
        for key in (
            "call1_history_chars",
            "call2_fallback_history_chars",
            "call3_fallback_history_chars",
        ):
            out[key] = max(0, min(
                out["storage_max_chars"],
                int(source.get(key, out[key])),
            ))
        out["reroll_archive_limit"] = max(
            0,
            min(10, int(source.get("reroll_archive_limit", out["reroll_archive_limit"]))),
        )
    except Exception as e:
        print(f"[ILLUST_HISTORY] 히스토리 설정 숫자 보정 실패: value={value!r}, error={e}")
        traceback.print_exc()
        return deepcopy(DEFAULT_SETTINGS)
    return out


def load_settings() -> dict:
    with _IO_LOCK:
        _ensure_dirs()
        value = _load_json(SETTINGS_PATH)
        if value is None:
            return deepcopy(DEFAULT_SETTINGS)
        return normalize_settings(value)


def save_settings(value: dict) -> dict:
    with _IO_LOCK:
        _ensure_dirs()
        settings = normalize_settings(value)
        _backup_existing(SETTINGS_PATH, "settings.json")
        try:
            _atomic_write_json(SETTINGS_PATH, settings)
            _trim_all_records_locked(settings["storage_max_chars"])
            print(
                f"[ILLUST_HISTORY] 설정 저장 완료: max_chars={settings['storage_max_chars']}, "
                f"call1={settings['call1_history_chars']}, "
                f"call2_fallback={settings['call2_fallback_history_chars']}, "
                f"call3_fallback={settings['call3_fallback_history_chars']}"
            )
            return settings
        except Exception as e:
            print(f"[ILLUST_HISTORY] 설정 저장 실패: error={e}")
            traceback.print_exc()
            raise


def _iter_records_locked() -> list[dict]:
    _ensure_dirs()
    records = []
    try:
        names = sorted(os.listdir(RECORDS_DIR))
    except Exception as e:
        print(f"[ILLUST_HISTORY] 레코드 목록 읽기 실패: error={e}")
        traceback.print_exc()
        return []
    for name in names:
        if not name.endswith(".json"):
            continue
        record = _load_json(os.path.join(RECORDS_DIR, name))
        if not isinstance(record, dict):
            print(f"[ILLUST_HISTORY] 손상된 레코드 제외: name={name}")
            continue
        history_id = str(record.get("history_id") or "")
        if not _SAFE_HISTORY_ID.fullmatch(history_id):
            print(f"[ILLUST_HISTORY] history_id 형식 불일치 레코드 제외: name={name}")
            continue
        records.append(record)
    return records


def _message_similarity(left: dict, right: dict) -> float:
    if str(left.get("role") or "user") != str(right.get("role") or "user"):
        return 0.0
    left_hash = str(left.get("full_content_hash") or _content_hash(left.get("content", "")))
    right_hash = str(right.get("full_content_hash") or _content_hash(right.get("content", "")))
    if left_hash == right_hash:
        return 1.0
    left_text = _normalize_content(left.get("content", ""))
    right_text = _normalize_content(right.get("content", ""))
    if not left_text or not right_text:
        return 0.0
    character_ratio = difflib.SequenceMatcher(
        None,
        left_text,
        right_text,
        autojunk=True,
    ).ratio()
    if character_ratio >= 0.88:
        return character_ratio
    # SequenceMatcher's character-level autojunk heuristic can collapse on
    # repetitive prose. A capped word-level pass stays linear-sized for long
    # CHAT entries while recovering small wording edits without keyword rules.
    left_words = left_text.split()[-2_000:]
    right_words = right_text.split()[-2_000:]
    if not left_words or not right_words:
        return character_ratio
    word_ratio = difflib.SequenceMatcher(
        None,
        left_words,
        right_words,
        autojunk=False,
    ).ratio()
    return max(character_ratio, word_ratio)


def _tail_alignment(saved: list[dict], incoming: list[dict]) -> dict | None:
    """Find a contiguous incoming range that matches the end of ``saved``."""
    if not saved or not incoming:
        return None
    best = None
    saved_end = len(saved) - 1
    for incoming_end in range(len(incoming)):
        saved_index = saved_end
        incoming_index = incoming_end
        similarities = []
        overlap_chars = 0
        exact_count = 0
        while saved_index >= 0 and incoming_index >= 0:
            similarity = _message_similarity(saved[saved_index], incoming[incoming_index])
            if similarity < 0.88:
                break
            similarities.append(similarity)
            overlap_chars += min(
                len(str(saved[saved_index].get("content") or "")),
                len(str(incoming[incoming_index].get("content") or "")),
            )
            if similarity == 1.0:
                exact_count += 1
            saved_index -= 1
            incoming_index -= 1
        if not similarities:
            continue
        matched_messages = len(similarities)
        average = sum(similarities) / matched_messages
        accepted = (
            (exact_count >= 1 and overlap_chars >= 80)
            or (matched_messages >= 2 and overlap_chars >= 40 and average >= 0.90)
            or (matched_messages == 1 and overlap_chars >= 300 and average >= 0.95)
        )
        if not accepted:
            continue
        candidate = {
            "incoming_start": incoming_index + 1,
            "incoming_end": incoming_end + 1,
            "matched_messages": matched_messages,
            "exact_messages": exact_count,
            "overlap_chars": overlap_chars,
            "similarity": average,
            "score": overlap_chars * average + exact_count * 250 + matched_messages * 25,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def _active_base_messages(record: dict) -> list[dict]:
    messages = list(record.get("messages") or [])
    current_id = str((record.get("active_turn") or {}).get("current_message_id") or "")
    if not current_id:
        return messages[:-1] if messages else []
    return [message for message in messages if str(message.get("id") or "") != current_id]


def _candidate_for_record(record: dict, incoming_past: list[dict], current_hash: str) -> dict | None:
    active_messages = list(record.get("messages") or [])
    incoming_snapshot_hash = _snapshot_hash(incoming_past)
    if active_messages and _snapshot_hash(active_messages) == incoming_snapshot_hash:
        exact_chars = sum(len(str(item.get("content") or "")) for item in incoming_past)
        return {
            "record": record,
            "operation": "append",
            "alignment": {
                "incoming_start": 0,
                "incoming_end": len(incoming_past),
                "matched_messages": len(incoming_past),
                "exact_messages": len(incoming_past),
                "overlap_chars": exact_chars,
                "similarity": 1.0,
                "score": exact_chars + len(incoming_past) * 275,
            },
            "score": exact_chars + len(incoming_past) * 275 + 500,
        }
    active_alignment = _tail_alignment(active_messages, incoming_past)
    if active_alignment:
        return {
            "record": record,
            "operation": "append",
            "alignment": active_alignment,
            "score": active_alignment["score"] + 500,
        }

    base_messages = _active_base_messages(record)
    active_turn = record.get("active_turn") or {}
    if (
        incoming_past
        and str(active_turn.get("base_context_hash") or "") == incoming_snapshot_hash
    ):
        exact_chars = sum(len(str(item.get("content") or "")) for item in incoming_past)
        active_current_hash = str(active_turn.get("current_context_hash") or "")
        return {
            "record": record,
            "operation": "duplicate" if active_current_hash == current_hash else "reroll",
            "alignment": {
                "incoming_start": 0,
                "incoming_end": len(incoming_past),
                "matched_messages": len(incoming_past),
                "exact_messages": len(incoming_past),
                "overlap_chars": exact_chars,
                "similarity": 1.0,
                "score": exact_chars + len(incoming_past) * 275,
            },
            "score": exact_chars + len(incoming_past) * 275 + 300,
        }
    base_alignment = _tail_alignment(base_messages, incoming_past)
    if base_alignment and base_alignment["incoming_end"] == len(incoming_past):
        active_current_hash = str((record.get("active_turn") or {}).get("current_context_hash") or "")
        return {
            "record": record,
            "operation": "duplicate" if active_current_hash == current_hash else "reroll",
            "alignment": base_alignment,
            "score": base_alignment["score"] + 300,
        }
    return None


def _select_candidate(records: list[dict], bot_name: str, incoming_past: list[dict], current_hash: str) -> dict | None:
    candidates = []
    for record in records:
        record_bot = str((record.get("source") or {}).get("bot_name") or "")
        if bot_name and record_bot and record_bot != bot_name:
            continue
        candidate = _candidate_for_record(record, incoming_past, current_hash)
        if candidate:
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if len(candidates) > 1:
        first = candidates[0]
        second = candidates[1]
        margin = first["score"] - second["score"]
        if margin <= max(75.0, first["score"] * 0.05):
            print(
                "[ILLUST_HISTORY] 연속성 후보가 모호해 새 히스토리로 분리: "
                f"first={first['record'].get('history_id')}:{first['score']:.1f}, "
                f"second={second['record'].get('history_id')}:{second['score']:.1f}"
            )
            return None
    return candidates[0]


def _copy_messages(messages: list[dict]) -> list[dict]:
    return [deepcopy(message) for message in messages if isinstance(message, dict)]


def _slice_messages_by_chars(messages: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    remaining = int(limit)
    selected = []
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if not content:
            continue
        piece = content[-remaining:] if len(content) > remaining else content
        selected.append({"role": message.get("role", "user"), "data": piece})
        remaining -= len(piece)
        if remaining <= 0:
            break
    selected.reverse()
    return selected


def _trim_messages(messages: list[dict], max_chars: int) -> tuple[list[dict], int]:
    trimmed = _copy_messages(messages)
    total = sum(len(str(message.get("content") or "")) for message in trimmed)
    excess = max(0, total - int(max_chars))
    while excess > 0 and trimmed:
        message = trimmed[0]
        content = str(message.get("content") or "")
        if len(content) <= excess:
            excess -= len(content)
            trimmed.pop(0)
            continue
        message["content"] = content[excess:]
        message["trimmed_prefix_chars"] = int(message.get("trimmed_prefix_chars") or 0) + excess
        excess = 0
    stored_chars = sum(len(str(message.get("content") or "")) for message in trimmed)
    return trimmed, stored_chars


def _prune_character_states(
    states: dict,
    messages: list[dict],
) -> tuple[dict, list[str]]:
    retained_message_ids = {
        str(message.get("id") or "")
        for message in messages
        if str(message.get("id") or "")
    }
    result = deepcopy(states or {})
    pruned = []
    for key, value in list(result.items()):
        if not isinstance(value, dict):
            continue
        last_seen_id = str(value.get("last_seen_message_id") or "")
        if last_seen_id and last_seen_id not in retained_message_ids:
            pruned.append(str(value.get("canonical_name") or key))
            result.pop(key, None)
    return result, pruned


def prepare_history(chats: list[dict], target_index: int, bot_name: str = "") -> dict:
    """Build an uncommitted history view for CALL1/2/3."""
    if target_index < 0 or target_index >= len(chats):
        print(
            f"[ILLUST_HISTORY] 현재 context 인덱스 범위 오류: "
            f"target_index={target_index}, chats={len(chats)}"
        )
        raise ValueError("현재 context 인덱스가 CHAT 범위를 벗어났습니다")
    incoming_past = [_message_from_input(item) for item in chats[:target_index]]
    incoming_current = _message_from_input(chats[target_index])
    current_hash = incoming_current["full_content_hash"]
    settings = load_settings()

    with _IO_LOCK:
        records = _iter_records_locked()
        candidate = _select_candidate(records, str(bot_name or ""), incoming_past, current_hash)
        now = time.time()
        if candidate is None:
            history_id = f"hist_{uuid.uuid4().hex}"
            record = {
                "schema_version": 1,
                "history_id": history_id,
                "source": {"bot_name": str(bot_name or ""), "branch_id": "main"},
                "revision": 0,
                "created_at": now,
                "updated_at": now,
                "messages": [],
                "stored_chars": 0,
                "characters": {},
                "active_turn": {},
                "last_pipeline": {},
                "reroll_archive": [],
            }
            operation = "new"
            proposed_messages = incoming_past + [incoming_current]
            state_before = {}
            match = {"similarity": 0.0, "overlap_chars": 0, "matched_messages": 0}
        else:
            record = deepcopy(candidate["record"])
            history_id = str(record["history_id"])
            operation = str(candidate["operation"])
            match = deepcopy(candidate["alignment"])
            if operation == "append":
                new_past = incoming_past[int(match["incoming_end"]):]
                proposed_messages = _copy_messages(record.get("messages") or []) + new_past + [incoming_current]
                state_before = deepcopy(record.get("characters") or {})
            elif operation in ("reroll", "duplicate"):
                base_messages = _active_base_messages(record)
                if operation == "duplicate":
                    proposed_messages = _copy_messages(record.get("messages") or [])
                else:
                    # Risu reroll sends the same past snapshot and a replacement
                    # current response. Rebuild the active turn from that snapshot
                    # so a previously character-trimmed base can be recovered.
                    proposed_messages = incoming_past + [incoming_current]
                stored_state_before = (record.get("active_turn") or {}).get("state_before")
                state_before = deepcopy(
                    stored_state_before
                    if isinstance(stored_state_before, dict)
                    else (record.get("characters") or {})
                )
            else:
                print(f"[ILLUST_HISTORY] 지원하지 않는 준비 연산: operation={operation}")
                raise RuntimeError(f"지원하지 않는 히스토리 연산: {operation}")

        current_id = (
            str((record.get("active_turn") or {}).get("current_message_id") or "")
            if operation == "duplicate"
            else incoming_current["id"]
        )
        base_messages = [
            message for message in proposed_messages
            if str(message.get("id") or "") != current_id
        ]
        call1_history = _slice_messages_by_chars(base_messages, settings["call1_history_chars"])
        call2_history = _slice_messages_by_chars(base_messages, settings["call2_fallback_history_chars"])
        call3_history = _slice_messages_by_chars(base_messages, settings["call3_fallback_history_chars"])
        print(
            f"[ILLUST_HISTORY] 연속성 판정: history={history_id}, operation={operation}, "
            f"matched={match.get('matched_messages', 0)}, "
            f"overlap_chars={match.get('overlap_chars', 0)}, "
            f"similarity={float(match.get('similarity', 0.0)):.3f}"
        )
        return {
            "history_id": history_id,
            "operation": operation,
            "expected_revision": int(record.get("revision") or 0),
            "record_before": record,
            "proposed_messages": proposed_messages,
            "current_message_id": current_id,
            "current_context": str(chats[target_index].get("data") or chats[target_index].get("content") or ""),
            "current_context_hash": current_hash,
            "base_context_hash": _snapshot_hash(base_messages),
            "state_before": state_before,
            "call1_history": call1_history,
            "call2_fallback_history": call2_history,
            "call3_fallback_history": call3_history,
            "settings": settings,
            "match": match,
            "bot_name": str(bot_name or ""),
            "prepared_at": now,
        }


def _safe_multi_char_results(value: dict) -> list[dict]:
    """분석에 필요한 다중 분리 입출력만 원본 프롬프트와 분리해 보존한다."""
    saved = []
    for item in value.get("items") or []:
        if not isinstance(item, dict):
            continue
        request = item.get("multi_char_layout_request")
        raw_response = str(item.get("multi_char_layout_raw_response") or "")
        layout = item.get("multi_char_layout")
        layout_error = str(item.get("multi_char_layout_error") or "")
        if not request and not raw_response and not isinstance(layout, dict) and not layout_error:
            continue
        saved.append({
            "slot": item.get("slot"),
            "request": deepcopy(request) if isinstance(request, dict) else {},
            "raw_response": raw_response[:50_000],
            "normalized_layout": deepcopy(layout) if isinstance(layout, dict) else {},
            "error": layout_error[:2_000],
        })
    return saved


def _safe_pipeline_snapshot(result: dict | None, error: str = "") -> dict:
    value = result if isinstance(result, dict) else {}
    return {
        "input_hash": str(value.get("history_input_hash") or ""),
        "status": "error" if error else "success",
        "error": str(error or "")[:2_000],
        "call1_result": deepcopy(value.get("call1_result") or {}),
        "profile_result": deepcopy(value.get("profile_result") or {}),
        "wardrobe_events": deepcopy(value.get("wardrobe_events") or []),
        "call2_out": str(value.get("call2_output") or ""),
        "call2_authority_audit": deepcopy(value.get("call2_authority_audit") or []),
        "call2_authority_audit_output": str(
            value.get("call2_authority_audit_output") or ""
        )[:50_000],
        "call2_authority_audit_status": str(
            value.get("call2_authority_audit_status") or ""
        ),
        "call2_fix_out": str(value.get("call2_fix_output") or ""),
        "call3_initial_out": str(value.get("call3_initial_output") or ""),
        "call3_out": str(value.get("call3_output") or ""),
        "call3_correction_used": bool(value.get("call3_correction_used")),
        "last_visual_by_character": deepcopy(value.get("last_visual_by_character") or {}),
        "multi_char_results": _safe_multi_char_results(value),
        "updated_at": time.time(),
    }


def finalize_history(plan: dict, result: dict | None = None, error: str = "") -> dict:
    """Commit original CHAT after the pipeline reaches a terminal state."""
    history_id = str(plan.get("history_id") or "")
    path = _record_path(history_id)
    with _IO_LOCK:
        _ensure_dirs()
        current = _load_json(path)
        record = deepcopy(plan.get("record_before") or {})
        operation = str(plan.get("operation") or "new")
        if current is not None:
            current_revision = int(current.get("revision") or 0)
            expected_revision = int(plan.get("expected_revision") or 0)
            if current_revision != expected_revision:
                conflicted_history_id = history_id
                history_id = f"hist_{uuid.uuid4().hex}"
                path = _record_path(history_id)
                print(
                    f"[ILLUST_HISTORY] 저장 revision 충돌로 독립 분기 보존: "
                    f"source_history={conflicted_history_id}, fork_history={history_id}, "
                    f"expected={expected_revision}, current={current_revision}"
                )
                record = deepcopy(plan.get("record_before") or {})
                record["history_id"] = history_id
                record["revision"] = 0
                record["created_at"] = time.time()
                source = deepcopy(record.get("source") or {})
                source["branch_id"] = f"revision_conflict_{uuid.uuid4().hex[:12]}"
                record["source"] = source
            else:
                record = current

        previous_turn = deepcopy(record.get("active_turn") or {})
        archive = list(record.get("reroll_archive") or [])
        if operation == "reroll" and previous_turn:
            archive.append({
                "reroll_index": int(previous_turn.get("reroll_index") or 0),
                "current_context_hash": previous_turn.get("current_context_hash", ""),
                "current_content": next((
                    str(message.get("content") or "")
                    for message in record.get("messages") or []
                    if str(message.get("id") or "") == str(previous_turn.get("current_message_id") or "")
                ), ""),
                "last_pipeline": deepcopy(record.get("last_pipeline") or {}),
                "archived_at": time.time(),
            })

        settings = normalize_settings(plan.get("settings"))
        archive = archive[-int(settings["reroll_archive_limit"]):] if settings["reroll_archive_limit"] else []
        proposed_messages, stored_chars = _trim_messages(
            plan.get("proposed_messages") or [],
            settings["storage_max_chars"],
        )
        result_value = result if isinstance(result, dict) else {}
        if error and operation == "duplicate":
            state_after = deepcopy(record.get("characters") or {})
        else:
            state_after = deepcopy(
                result_value.get("character_states_after")
                or plan.get("state_before")
                or {}
            )
        state_after, pruned_characters = _prune_character_states(
            state_after,
            proposed_messages,
        )
        if pruned_characters:
            print(
                f"[ILLUST_HISTORY] 보존 범위 밖 캐릭터 상태 정리: "
                f"history={history_id}, characters={pruned_characters}"
            )
        current_message_id = str(plan.get("current_message_id") or "")
        if not any(str(message.get("id") or "") == current_message_id for message in proposed_messages):
            # 현재 본문 하나가 저장 상한보다 길어 앞이 잘렸더라도 message 자체는 남는다.
            if proposed_messages:
                current_message_id = str(proposed_messages[-1].get("id") or current_message_id)
            else:
                print(f"[ILLUST_HISTORY] 저장 상한 적용 후 현재 메시지가 사라짐: history={history_id}")

        if operation == "reroll":
            reroll_index = int(previous_turn.get("reroll_index") or 0) + 1
        elif operation == "duplicate":
            reroll_index = int(previous_turn.get("reroll_index") or 0)
        else:
            reroll_index = 0
        record.update({
            "schema_version": 1,
            "history_id": history_id,
            "source": {
                "bot_name": str(plan.get("bot_name") or (record.get("source") or {}).get("bot_name") or ""),
                "branch_id": str((record.get("source") or {}).get("branch_id") or "main"),
            },
            "revision": int(record.get("revision") or 0) + 1,
            "created_at": float(record.get("created_at") or time.time()),
            "updated_at": time.time(),
            "messages": proposed_messages,
            "stored_chars": stored_chars,
            "characters": state_after,
            "active_turn": {
                "base_context_hash": str(plan.get("base_context_hash") or ""),
                "current_context_hash": str(plan.get("current_context_hash") or ""),
                "current_message_id": current_message_id,
                "reroll_index": reroll_index,
                "state_before": deepcopy(plan.get("state_before") or {}),
                "state_after": state_after,
                "operation": operation,
                "committed_at": time.time(),
            },
            "last_pipeline": _safe_pipeline_snapshot(result, error),
            "reroll_archive": archive,
        })
        title_source = str(plan.get("current_context") or "").strip().replace("\n", " ")
        if title_source:
            record["title"] = title_source[:100]
        _backup_existing(path, f"{history_id}.json")
        try:
            _atomic_write_json(path, record)
            print(
                f"[ILLUST_HISTORY] 원문 히스토리 갱신 완료: history={history_id}, "
                f"operation={operation}, revision={record['revision']}, "
                f"chars={stored_chars}, pipeline_status={'error' if error else 'success'}"
            )
            return record
        except Exception as e:
            print(f"[ILLUST_HISTORY] 원문 히스토리 저장 실패: history={history_id}, error={e}")
            traceback.print_exc()
            raise


def _trim_all_records_locked(max_chars: int) -> None:
    for record in _iter_records_locked():
        messages, stored_chars = _trim_messages(record.get("messages") or [], max_chars)
        characters, pruned_characters = _prune_character_states(
            record.get("characters") or {},
            messages,
        )
        active_turn = deepcopy(record.get("active_turn") or {})
        if isinstance(active_turn.get("state_before"), dict):
            active_turn["state_before"], _ = _prune_character_states(
                active_turn["state_before"],
                messages,
            )
        if isinstance(active_turn.get("state_after"), dict):
            active_turn["state_after"], _ = _prune_character_states(
                active_turn["state_after"],
                messages,
            )
        if (
            messages == (record.get("messages") or [])
            and stored_chars == int(record.get("stored_chars") or 0)
            and characters == (record.get("characters") or {})
        ):
            continue
        path = _record_path(str(record.get("history_id") or ""))
        _backup_existing(path, f"{record.get('history_id')}.json")
        record["messages"] = messages
        record["stored_chars"] = stored_chars
        record["characters"] = characters
        record["active_turn"] = active_turn
        record["revision"] = int(record.get("revision") or 0) + 1
        record["updated_at"] = time.time()
        _atomic_write_json(path, record)
        if pruned_characters:
            print(
                f"[ILLUST_HISTORY] 설정 축소로 캐릭터 상태 정리: "
                f"history={record.get('history_id')}, characters={pruned_characters}"
            )
        print(
            f"[ILLUST_HISTORY] 저장 상한 즉시 적용: history={record.get('history_id')}, "
            f"chars={stored_chars}, max={max_chars}"
        )


def list_histories(query: str = "", limit: int = 100) -> list[dict]:
    normalized_query = str(query or "").strip().casefold()
    safe_limit = max(1, min(500, int(limit)))
    with _IO_LOCK:
        records = _iter_records_locked()
    summaries = []
    for record in records:
        message_text = "\n".join(str(item.get("content") or "") for item in record.get("messages") or [])
        character_names = " ".join(
            str((value or {}).get("canonical_name") or key)
            for key, value in (record.get("characters") or {}).items()
            if isinstance(value, dict)
        )
        haystack = "\n".join((
            str(record.get("history_id") or ""),
            str(record.get("title") or ""),
            str((record.get("source") or {}).get("bot_name") or ""),
            character_names,
            message_text,
        )).casefold()
        if normalized_query and normalized_query not in haystack:
            continue
        last_pipeline = record.get("last_pipeline") or {}
        active_turn = record.get("active_turn") or {}
        summaries.append({
            "history_id": record.get("history_id", ""),
            "title": record.get("title", ""),
            "bot_name": (record.get("source") or {}).get("bot_name", ""),
            "updated_at": record.get("updated_at", 0),
            "stored_chars": record.get("stored_chars", 0),
            "message_count": len(record.get("messages") or []),
            "character_count": len(record.get("characters") or {}),
            "reroll_index": active_turn.get("reroll_index", 0),
            "pipeline_status": last_pipeline.get("status", ""),
            "last_operation": active_turn.get("operation", ""),
        })
    summaries.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return summaries[:safe_limit]


def get_history(history_id: str) -> dict | None:
    with _IO_LOCK:
        record = _load_json(_record_path(history_id))
        if record is None:
            print(f"[ILLUST_HISTORY] 상세 조회 대상 없음: history={history_id}")
        return record


def delete_history(history_id: str) -> str:
    path = _record_path(history_id)
    with _IO_LOCK:
        if not os.path.isfile(path):
            print(f"[ILLUST_HISTORY] 삭제 대상 없음: history={history_id}")
            raise FileNotFoundError(f"채팅 히스토리를 찾지 못했습니다: {history_id}")
        _ensure_dirs()
        _backup_existing(path, f"{history_id}.json")
        destination = os.path.join(
            TRASH_DIR,
            f"{history_id}.{time.strftime('%Y%m%d_%H%M%S')}.{time.time_ns()}.json",
        )
        try:
            shutil.move(path, destination)
            print(
                f"[ILLUST_HISTORY] 히스토리 휴지통 이동 완료: "
                f"history={history_id}, destination={destination}"
            )
            return destination
        except Exception as e:
            print(f"[ILLUST_HISTORY] 히스토리 삭제 실패: history={history_id}, error={e}")
            traceback.print_exc()
            raise
