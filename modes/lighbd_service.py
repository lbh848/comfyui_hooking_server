"""lighbd (Lightboard-Direct) ENQUEUE + Phase B generation.

Phase A: callLLM 장면 분할 + 로그.
Phase B: 분할 결과로 N개 씬 병렬 이미지 생성 디스패치, 세션 상태 관리, 리롤.

외부 모듈 의존 (late import 로 순환참조 회피):
- server.py: prompts dict, queue_manager 인스턴스, build_prompt, find_save_image_node
- modes/llm_service.py: callLLM
"""
import os
import re
import json
import uuid
import time
import asyncio
import datetime
import hashlib
import traceback

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts", "lighbd")
LOG_DIR = os.path.join(BASE_DIR, "logs")

_PROMPTS_CACHE: dict | None = None
_PROMPTS_MTIME = 0.0


# ─── 프롬프트 로드 ─────────────────────────────────────────
def load_prompts() -> dict:
    """prompts/lighbd/*.txt를 읽어 dict로 반환. 파일 mtime 기반 캐싱."""
    global _PROMPTS_CACHE, _PROMPTS_MTIME

    files = ["system", "job", "format", "thoughts", "jailbreak", "preset"]
    latest_mtime = 0.0
    for name in files:
        p = os.path.join(PROMPTS_DIR, f"{name}.txt")
        if os.path.exists(p):
            latest_mtime = max(latest_mtime, os.path.getmtime(p))

    if _PROMPTS_CACHE is not None and latest_mtime == _PROMPTS_MTIME:
        return _PROMPTS_CACHE

    result = {}
    missing = []
    for name in files:
        p = os.path.join(PROMPTS_DIR, f"{name}.txt")
        if not os.path.exists(p):
            missing.append(name)
            result[name] = ""
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                result[name] = f.read()
        except Exception as e:
            print(f"[LIGHBD] WARN: prompt file load failed {name}.txt: {e}")
            result[name] = ""
            missing.append(name)

    if missing:
        print(f"[LIGHBD] WARN: missing prompt files: {missing}")

    _PROMPTS_CACHE = result
    _PROMPTS_MTIME = latest_mtime
    return result


# ─── ENQUEUE legacy helpers (Phase A compatibility) ────────
def is_enqueue_payload(positive: str) -> bool:
    if not positive:
        return False
    return positive.lstrip().startswith("__LB_ENQUEUE_V1__")


def extract_context(positive: str) -> str:
    if not positive:
        return ""
    m = re.search(r'\n\[CHAT\]', positive, re.IGNORECASE)
    if m:
        return positive[m.end():].strip()
    m = re.match(r'^\[CHAT\]', positive, re.IGNORECASE)
    if m:
        return positive[m.end():].strip()
    return ""


# ─── TOON plan 파싱 ────────────────────────────────────────
def _normalize_toon_for_yaml(s: str) -> str:
    """TOON 을 YAML-compatible 로 변환.
    scenes[N]: → scenes:
    characters[N]: → characters:
    """
    s = re.sub(r'\b(scenes)\[\d+\]:', r'\1:', s)
    s = re.sub(r'\b(characters)\[\d+\]:', r'\1:', s)
    return s


def parse_scenes(plan_xml: str) -> list:
    """<lb-xnai>...</lb-xnai> 블록에서 scenes 리스트 추출.

    Returns:
        [{"idx": int, "sentence_slot": int, "positive": str, "negative": str,
          "camera": str, "scene": str, "name": str}, ...]
        파싱 실패 시 빈 리스트.
    """
    if not plan_xml:
        return []

    m = re.search(r'<lb[-_]xnai[^>]*>([\s\S]*?)</lb[-_]xnai>', plan_xml, re.IGNORECASE)
    if not m:
        print("[LIGHBD] WARN: <lb-xnai> block not found in plan")
        return []

    inner = m.group(1)
    inner = _normalize_toon_for_yaml(inner)

    try:
        data = yaml.safe_load(inner)
    except yaml.YAMLError as e:
        print(f"[LIGHBD] WARN: YAML parse failed: {e}")
        return []

    if not isinstance(data, dict):
        return []

    raw_scenes = data.get("scenes", []) or []
    out = []
    for i, sc in enumerate(raw_scenes):
        if not isinstance(sc, dict):
            continue
        chars = sc.get("characters", []) or []
        # 캐릭터들의 positive/negative 병합 (multi-char 장면 대비)
        pos_parts = []
        neg_parts = []
        name_parts = []
        for ch in chars:
            if not isinstance(ch, dict):
                continue
            p = (ch.get("positive") or "").strip()
            n = (ch.get("negative") or "").strip()
            nm = (ch.get("name") or "").strip()
            if p:
                pos_parts.append(p)
            if n:
                neg_parts.append(n)
            if nm:
                name_parts.append(nm)

        slot = sc.get("slot", i)
        try:
            slot_int = int(slot)
        except (ValueError, TypeError):
            slot_int = i

        out.append({
            "idx": i,
            "sentence_slot": slot_int,
            "positive": ", ".join(pos_parts),
            "negative": ", ".join(neg_parts),
            "camera": str(sc.get("camera") or ""),
            "scene": str(sc.get("scene") or ""),
            "name": ", ".join(name_parts),
            "supplement": str(sc.get("supplement") or ""),
            "characters": chars if isinstance(chars, list) else [],
        })
    return out


# ─── 세션 파일 관리 ───────────────────────────────────────
def _session_path(session_id: str) -> str:
    return os.path.join(LOG_DIR, f"lighbd_session_{session_id[:8]}.json")


def _save_session(session_id: str, data: dict) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = _session_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LIGHBD] ERROR: session save failed: {e}")


def _load_session(session_id: str) -> dict | None:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[LIGHBD] ERROR: session load failed: {e}")
        return None


# ─── 서버 글로벌 접근 (late import) ────────────────────────
def _get_server_globals():
    """server.py 의 prompts dict, queue_manager, build_prompt 에 접근.
    server.py 가 완전히 로드된 시점엔 정상 동작.
    """
    import server
    return {
        "prompts": server.prompts,
        "queue_manager": server.queue_manager,
        "build_prompt": server.build_prompt,
    }


LIGHBD_HISTORY_PATH = os.path.join(LOG_DIR, "lighbd_history.jsonl")
# 일반(CALL1/2/3 등 삽화 LLM 호출) 보관량. "LLM 흐름 추적"으로 과거 백업 흐름을
# 분석하려면 본문이 보존되어 있어야 하므로 20 → 100 으로 상향. 이미 삭제된 과거 레코드는
# 복원되지 않으며, 이후 append 시점부터 100건까지 보존된다.
LIGHBD_GENERAL_HISTORY_MAX = 100
LIGHBD_MULTI_CHAR_HISTORY_MAX = 100
LIGHBD_HISTORY_MAX = LIGHBD_GENERAL_HISTORY_MAX + LIGHBD_MULTI_CHAR_HISTORY_MAX
_MANUAL_RACE_SUCCESS_SUPPRESSIONS: list[dict] = []


def _current_async_task_id() -> int:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return 0
    return id(task) if task is not None else 0


def _history_output_digest(output) -> str:
    return hashlib.sha256(
        str(output or "").encode("utf-8", errors="replace")
    ).hexdigest()


def _consume_manual_race_success_suppression(record: dict) -> bool:
    """core가 이미 기록한 병렬 승자와 호출자 측 일반 OK 기록의 중복을 막는다."""
    if str(record.get("status") or "").lower() != "ok":
        return False
    now = time.monotonic()
    task_id = _current_async_task_id()
    output_digest = _history_output_digest(record.get("output"))
    kept = []
    matched = False
    for pending in _MANUAL_RACE_SUCCESS_SUPPRESSIONS:
        if float(pending.get("expires_at") or 0.0) <= now:
            continue
        if (
            not matched
            and int(pending.get("task_id") or 0) == task_id
            and str(pending.get("output_digest") or "") == output_digest
        ):
            matched = True
            continue
        kept.append(pending)
    _MANUAL_RACE_SUCCESS_SUPPRESSIONS[:] = kept
    return matched


def _is_multi_char_history_record(record: object) -> bool:
    return (
        isinstance(record, dict)
        and str(record.get("task_key") or "") == "illustration_multi_char_mask"
    )


def _trim_lighbd_history_lines(lines: list[str]) -> list[str]:
    """일반 최근 100건과 분석용 다중 분리 최근 100건을 각각 보존한다."""
    general_indices = []
    multi_char_indices = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except Exception as e:
            print(
                f"[LIGHBD] history 보존 분류 중 JSON 파싱 실패, 일반 레코드로 유지: "
                f"index={index}, error={e}, line={str(line)[:200]!r}"
            )
            traceback.print_exc()
            general_indices.append(index)
            continue
        if _is_multi_char_history_record(record):
            multi_char_indices.append(index)
        else:
            general_indices.append(index)
    keep = set(general_indices[-LIGHBD_GENERAL_HISTORY_MAX:])
    keep.update(multi_char_indices[-LIGHBD_MULTI_CHAR_HISTORY_MAX:])
    return [line for index, line in enumerate(lines) if index in keep]


def _log_lighbd_history(record: dict) -> None:
    """lighbd 전용 히스토리 파일(logs/lighbd_history.jsonl)에 append.
    일반 호출 최근 100개와 다중 분리 호출 최근 100개를 별도로 유지한다.

    배포 환경에도 존재하는 logs/backups/ 폴더에 write 전 백업을 보관한다.
    """
    if _consume_manual_race_success_suppression(record):
        print(
            f"[LIGHBD] 병렬 경쟁 승자 일반 OK 중복 기록 생략: "
            f"call_name={record.get('call_name')!r}, "
            f"output_len={len(str(record.get('output') or ''))}"
        )
        return
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        history_backup_dir = os.path.join(
            os.path.dirname(LIGHBD_HISTORY_PATH),
            "backups",
        )
        os.makedirs(history_backup_dir, exist_ok=True)
        backup_path = os.path.join(history_backup_dir, "lighbd_history.jsonl.bak")

        existing_lines = []
        if os.path.exists(LIGHBD_HISTORY_PATH):
            try:
                with open(LIGHBD_HISTORY_PATH, "r", encoding="utf-8") as f:
                    existing_lines = f.readlines()
                # 백업
                with open(backup_path, "w", encoding="utf-8") as bf:
                    bf.writelines(existing_lines)
            except Exception as e:
                print(f"[LIGHBD] history 읽기/백업 실패: {e}")
                existing_lines = []

        # append 시점의 ts로 덮어쓴다: 호출자가 함수 진입(시작) 시각으로 고정한 ts가
        # 그대로 영속화되면, 재시도 끝에 성공한 최종 OK 레코드가 자기 자신의 attempt
        # 실패 기록보다 더 과거 ts를 가져 정렬 시 ERROR 위로 올라가는 버그가 생긴다.
        # 영속화 순간(= 호출 완료 시각)으로 갱신해 append 순서 = ts 순서를 보장한다.
        # 원본 객체는 부작용을 피해 변경하지 않는다.
        stamped = dict(record)
        stamped["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
        line = json.dumps(stamped, ensure_ascii=False) + "\n"
        existing_lines.append(line)
        existing_lines = _trim_lighbd_history_lines(existing_lines)

        with open(LIGHBD_HISTORY_PATH, "w", encoding="utf-8") as f:
            f.writelines(existing_lines)
    except Exception as e:
        print(f"[LIGHBD] history 쓰기 실패: {e}")
        traceback.print_exc()


def _log_manual_parallel_race(event: dict) -> None:
    """수동 병렬 재시도의 모든 시도를 승리/폐기 상태로 자세히에 기록한다."""
    race_id = str(event.get("race_id") or "")
    attempts = event.get("attempts")
    if not race_id:
        print(f"[LIGHBD] 병렬 경쟁 이력 기록 실패: race_id 없음 event={event}")
        return
    if not isinstance(attempts, list) or len(attempts) < 2:
        print(
            f"[LIGHBD] 병렬 경쟁 이력 기록 실패: 시도 정보 부족 "
            f"race_id={race_id}, attempts={attempts!r}"
        )
        return

    winner_stream_id = str(event.get("winner_stream_id") or "")
    base_call_name = str(
        event.get("call_name")
        or event.get("task_key")
        or "LLM 요청"
    )
    input_messages = event.get("input")
    if not isinstance(input_messages, list):
        print(
            f"[LIGHBD] 병렬 경쟁 입력 형식 오류: "
            f"race_id={race_id}, input_type={type(input_messages).__name__}; 빈 입력 사용"
        )
        input_messages = []

    records = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            print(
                f"[LIGHBD] 병렬 경쟁 시도 형식 오류: "
                f"race_id={race_id}, attempt={attempt!r}"
            )
            continue
        stream_id = str(attempt.get("stream_id") or "")
        if not stream_id:
            print(
                f"[LIGHBD] 병렬 경쟁 시도 기록 실패: stream_id 없음 "
                f"race_id={race_id}, attempt={attempt!r}"
            )
            continue
        role = str(attempt.get("race_role") or "")
        role_label = "병렬 재시도" if role == "parallel" else "원본"
        outcome_kind = str(attempt.get("outcome_kind") or "")
        is_winner = bool(winner_stream_id and stream_id == winner_stream_id)
        if is_winner:
            status = "race_won"
            race_result = "winner"
            error_text = ""
        elif winner_stream_id and str(attempt.get("race_status") or "") == "lost":
            status = "race_lost"
            race_result = "discarded"
            error_text = "더 빠른 응답이 채택되어 이 응답은 폐기되었습니다."
        elif outcome_kind == "cancelled":
            status = "cancelled"
            race_result = "cancelled"
            error_text = str(attempt.get("error") or "요청이 중지되었습니다.")
        elif outcome_kind == "failure":
            status = "error"
            race_result = "failed"
            error_text = str(attempt.get("error") or "병렬 요청이 실패했습니다.")
        elif winner_stream_id:
            status = "race_lost"
            race_result = "discarded"
            error_text = "더 빠른 응답이 채택되어 이 응답은 폐기되었습니다."
        else:
            status = "error"
            race_result = "failed"
            error_text = str(attempt.get("error") or "병렬 요청이 실패했습니다.")

        record = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "history_id": f"{race_id}:{stream_id}",
            "prompt_id": str(event.get("task_key") or race_id),
            "call_name": f"{base_call_name} · {role_label}",
            "task_key": str(event.get("task_key") or ""),
            "service": str(event.get("service") or ""),
            "model": str(event.get("model") or attempt.get("model") or ""),
            "llm_slot": str(event.get("llm_slot") or attempt.get("llm_slot") or ""),
            "input": input_messages,
            "output": str(attempt.get("text") or ""),
            "completion_tokens": int(attempt.get("completion_tokens") or 0),
            "prompt_tokens": int(attempt.get("prompt_tokens") or 0),
            "elapsed": round(float(attempt.get("elapsed") or 0.0), 3),
            "tps": round(float(attempt.get("tps") or 0.0), 1),
            "ttft": (
                round(float(attempt.get("ttft")), 3)
                if attempt.get("ttft") is not None
                else None
            ),
            "status": status,
            "race_id": race_id,
            "race_role": role,
            "race_result": race_result,
            "stream_id": stream_id,
            "winner_stream_id": winner_stream_id,
        }
        if error_text:
            record["error"] = error_text
        records.append(record)

    if len(records) < 2:
        print(
            f"[LIGHBD] 병렬 경쟁 이력 기록 실패: 유효한 레코드 부족 "
            f"race_id={race_id}, valid={len(records)}"
        )
        return

    # 최신 목록에서 승자가 먼저 보이도록 패배를 먼저 append한다.
    ordered = sorted(
        records,
        key=lambda record: 1 if record.get("status") == "race_won" else 0,
    )
    try:
        for record in ordered:
            _log_lighbd_history(record)
        winner_record = next(
            (record for record in records if record.get("status") == "race_won"),
            None,
        )
        if winner_record is not None:
            _MANUAL_RACE_SUCCESS_SUPPRESSIONS.append({
                "task_id": int(event.get("owner_task_id") or 0),
                "output_digest": _history_output_digest(
                    winner_record.get("output")
                ),
                "expires_at": time.monotonic() + 30.0,
            })
        print(
            f"[LIGHBD] 병렬 경쟁 자세히 기록 완료: "
            f"race_id={race_id}, winner={winner_stream_id or '(없음)'}, "
            f"records={len(records)}"
        )
    except Exception as e:
        print(
            f"[LIGHBD] 병렬 경쟁 자세히 기록 실패: "
            f"race_id={race_id}, error={type(e).__name__}: {e}"
        )
        traceback.print_exc()


def _update_lighbd_history_records(updates_by_id: dict[str, dict]) -> int:
    """history_id로 최근 LLM 히스토리 레코드를 찾아 안전하게 갱신한다."""
    if not isinstance(updates_by_id, dict) or not updates_by_id:
        print(
            f"[LIGHBD] history 갱신 건너뜀: updates가 비어 있거나 dict가 아님 "
            f"value={updates_by_id!r}"
        )
        return 0
    if not os.path.exists(LIGHBD_HISTORY_PATH):
        print(
            f"[LIGHBD] history 갱신 실패: 파일이 없음 path={LIGHBD_HISTORY_PATH}, "
            f"ids={list(updates_by_id)}"
        )
        return 0
    try:
        with open(LIGHBD_HISTORY_PATH, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        matched_ids = set()
        rewritten_lines = []
        for line in original_lines:
            stripped = line.strip()
            if not stripped:
                rewritten_lines.append(line)
                continue
            try:
                record = json.loads(stripped)
            except Exception as e:
                print(
                    f"[LIGHBD] history 갱신 중 JSON 파싱 실패, 원본 줄 유지: "
                    f"error={type(e).__name__}: {e}, line={stripped[:200]!r}"
                )
                traceback.print_exc()
                rewritten_lines.append(line)
                continue
            history_id = str(record.get("history_id") or "")
            updates = updates_by_id.get(history_id)
            if isinstance(updates, dict):
                record.update(updates)
                matched_ids.add(history_id)
            rewritten_lines.append(json.dumps(record, ensure_ascii=False) + "\n")

        missing_ids = sorted(set(updates_by_id) - matched_ids)
        if missing_ids:
            print(
                f"[LIGHBD] history 갱신 대상 일부를 찾지 못함: "
                f"missing={missing_ids}, matched={len(matched_ids)}"
            )
        if not matched_ids:
            return 0

        history_backup_dir = os.path.join(
            os.path.dirname(LIGHBD_HISTORY_PATH),
            "backups",
        )
        os.makedirs(history_backup_dir, exist_ok=True)
        backup_path = os.path.join(history_backup_dir, "lighbd_history.jsonl.bak")
        with open(backup_path, "w", encoding="utf-8") as backup_file:
            backup_file.writelines(original_lines)
        with open(LIGHBD_HISTORY_PATH, "w", encoding="utf-8") as history_file:
            history_file.writelines(rewritten_lines)
        print(
            f"[LIGHBD] history 레코드 갱신 완료: "
            f"matched={len(matched_ids)}, requested={len(updates_by_id)}"
        )
        return len(matched_ids)
    except Exception as e:
        print(
            f"[LIGHBD] history 레코드 갱신 실패: "
            f"path={LIGHBD_HISTORY_PATH}, ids={list(updates_by_id)}, error={e}"
        )
        traceback.print_exc()
        return 0


def _load_lighbd_history(limit: int = LIGHBD_HISTORY_MAX) -> list:
    """보존된 일반/다중 분리 히스토리를 오래된 → 최신 순(ts 기준)으로 반환한다.

    파일 append 순서가 항상 시간순은 아니므로(동시 호출·같은 초 기록 등),
    ts 기준 stable 정렬로 확정한다. 같은 초 내에서는 append 순서(=발생 순서)가
    유지된다. 프론트는 이 결과를 reverse 해 최신 → 오래된 순으로 표시한다.
    """
    if not os.path.exists(LIGHBD_HISTORY_PATH):
        return []
    try:
        with open(LIGHBD_HISTORY_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        records = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except Exception:
                continue
        records.sort(key=lambda r: str(r.get("ts") or ""))
        return records[-limit:]
    except Exception as e:
        print(f"[LIGHBD] history 읽기 실패: {e}")
        return []


async def _stream_with_frontend_notify(prompt_id: str, messages: list):
    """callLLMStream 을 돌면서 각 이벤트를 프론트엔드에 WS 전달.

    프론트엔드 우하단 위젯 (lighbd_llm_stream 이벤트) 이 상태/통계/출력을
    실시간 표시. lighbd 호출만 lighbd_history.jsonl 에 기록 (callLLMStream의
    llm_history.jsonl 기록은 중복 회피를 위해 끔).
    """
    from modes.llm_service import callLLMStream

    try:
        import server as _server
        notify = _server.notify_frontend
    except Exception as e:
        print(f"[LIGHBD] WARN: cannot access notify_frontend: {e}")
        notify = None

    final_record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "prompt_id": prompt_id,
        "input": messages,
        "output": "",
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "elapsed": 0.0,
        "tps": 0.0,
    }

    async for ev in callLLMStream(messages, log_history=False):
        # context: lighbd 세션 식별용 prompt_id 추가해서 프론트가 어떤 호출인지 구분
        out_ev = dict(ev)
        out_ev["prompt_id"] = prompt_id
        if notify is not None:
            try:
                await notify("lighbd_llm_stream", out_ev)
            except Exception as e:
                print(f"[LIGHBD] WARN: notify_frontend failed: {e}")
        # 히스토리용 정보 수집
        if ev["type"] == "done":
            final_record["output"] = ev.get("text", "")
            final_record["completion_tokens"] = ev.get("completion_tokens", 0)
            final_record["prompt_tokens"] = ev.get("prompt_tokens", 0)
            final_record["elapsed"] = round(ev.get("elapsed", 0.0), 3)
            final_record["tps"] = round(ev.get("tps", 0.0), 1)
            if ev.get("ttft") is not None:
                final_record["ttft"] = round(ev.get("ttft"), 3)
            final_record["status"] = "ok"
        elif ev["type"] == "error":
            final_record["error"] = ev.get("error", "")
            final_record["status"] = "error"
        yield ev

    # done/error 이후 히스토리 기록
    _log_lighbd_history(final_record)


# ─── Preset 프롬프트 빌드 ────────────────────────────────
def build_preset_prompt(preset_content: str, desc: dict, body_text: str) -> tuple[str, str]:
    """Lua gen.lua:buildPresetPrompt (comfy 모드) 포팅.

    preset 템플릿에서 [Positive]/[Negative] 영역 추출 → 플레이스홀더 치환.
    IllustPromptBuilder.parse_sections가 [Name]/[SETUP]/[CHAR]/[SUPPLEMENT]
    마커를 인식하므로, 치환 후 결과를 process_prompt로 전달하면
    캐릭터 감지/LoRA/품질 태그까지 자동 적용됨.

    Args:
        preset_content: preset.txt 원본 (빈 문자열 허용)
        desc: parse_scenes()의 씬 딕셔너리 (camera/scene/characters/supplement)
        body_text: ENQUEUE context에서 발췌한 본문 ([BODY] 이후)

    Returns:
        (positive, negative) 치환 완료된 문자열.
    """
    desc = desc or {}
    chars = desc.get("characters") or []

    # [Positive]/[Negative] 영역 추출
    positive_tmpl = ""
    negative_tmpl = ""
    if preset_content:
        pos_m = re.search(r'\[Positive\]\s*(.*?)\s*\[Negative\]',
                          preset_content, re.DOTALL | re.IGNORECASE)
        if pos_m:
            positive_tmpl = pos_m.group(1).strip()
        neg_m = re.search(r'\[Negative\]\s*(.*)\Z',
                          preset_content, re.DOTALL | re.IGNORECASE)
        if neg_m:
            negative_tmpl = neg_m.group(1).strip()

    # setup = camera + ', ' + scene (루아 line 47-52: camera first)
    setup_parts = []
    if desc.get("camera"):
        setup_parts.append(desc["camera"].strip())
    if desc.get("scene"):
        setup_parts.append(desc["scene"].strip())
    setup_prompt = ", ".join(s for s in setup_parts if s)

    # charPromptP/N (comfy 기본 divider = " | ", charPrompt transform OFF)
    char_divider = " | "
    char_prompt_p = char_divider.join(
        (c.get("positive") or "").strip()
        for c in chars if isinstance(c, dict) and (c.get("positive") or "").strip()
    )
    char_prompt_n = char_divider.join(
        (c.get("negative") or "").strip()
        for c in chars if isinstance(c, dict) and (c.get("negative") or "").strip()
    )

    supplement = (desc.get("supplement") or "").strip()

    # {name} 치환 (루아 line 174-184)
    names = [c.get("name", "").strip() for c in chars
             if isinstance(c, dict) and (c.get("name") or "").strip()]
    name_text = ", ".join(names)

    # {chat} = body_text, {slot} = '' (프로젝트 결정: Python 포트에선 fullChat 발췌 불가)
    chat_text = body_text or ""
    slot_text = ""
    # {speak} = '' (lighbd 내부 경로에선 SPEAK 발췌 불가; 외부 RAW 경로로 [SPEAK]가 채워져 들어옴)
    speak_text = ""

    positive = positive_tmpl
    positive = positive.replace("{chat}", chat_text)
    positive = positive.replace("{slot}", slot_text)
    positive = positive.replace("{speak}", speak_text)
    positive = positive.replace("{name}", name_text)

    # comfy non-{prompt} 분기 (사용자 preset엔 {prompt} 없음)
    if "{prompt}" in positive:
        prompt_body = setup_prompt
        if char_prompt_p:
            prompt_body = prompt_body + ",\n\n" + char_prompt_p
        if supplement:
            prompt_body = prompt_body + ",\n\n" + supplement
        positive = positive.replace("{prompt}", prompt_body)
    else:
        if "{setup}" in positive:
            positive = positive.replace("{setup}", setup_prompt)
        elif setup_prompt:
            positive = (positive + ", " + setup_prompt) if positive else setup_prompt

        if "{char}" in positive:
            positive = positive.replace("{char}", char_prompt_p)
        elif char_prompt_p:
            positive = (positive + "\n\n" + char_prompt_p) if positive else char_prompt_p

        if "{supplement}" in positive:
            positive = positive.replace("{supplement}", supplement)
        elif supplement:
            positive = (positive + "\n\n" + supplement) if positive else supplement

    # negative (루아 line 252-270 comfy 분기; negativeNote 없음 → 빈 문자열)
    negative = negative_tmpl
    if not negative:
        negative = "{prompt}"
    if "{prompt}" not in negative:
        negative = negative + "{prompt}"
    negative = negative.replace("{prompt}", "")
    if char_prompt_n:
        negative = (negative + "\n\n" + char_prompt_n) if negative else char_prompt_n
    negative = negative.strip()

    positive = re.sub(r'\n\n\n+', '\n\n', positive)
    negative = re.sub(r'\n\n\n+', '\n\n', negative)

    return positive, negative


# ─── 병렬 생성 디스패치 ───────────────────────────────────
def dispatch_generation(session_id: str, scenes: list, session_data: dict) -> None:
    """각 씬마다 prompt_id 발급, prompts 엔트리 사전 등록, 큐에 병렬 디스패치.

    Args:
        session_id: 세션 식별자
        scenes: parse_scenes() 결과
        session_data: 세션 JSON (scenes 필드에 prompt_id/status 갱신해서 _save_session 호출됨)
    """
    if not scenes:
        print(f"[LIGHBD] dispatch skipped: no scenes for session {session_id[:8]}")
        return

    try:
        g = _get_server_globals()
    except Exception as e:
        print(f"[LIGHBD] ERROR: cannot access server globals: {e}")
        traceback.print_exc()
        return

    prompts_dict = g["prompts"]
    try:
        from server import register_and_enqueue_illustration
    except ImportError as e:
        print(f"[LIGHBD] ERROR: cannot import register_and_enqueue_illustration: {e}")
        traceback.print_exc()
        return

    # preset 템플릿 로드 (build_preset_prompt용)
    prompts = load_prompts()
    preset_content = prompts.get("preset", "") or ""
    if not preset_content:
        print("[LIGHBD] WARN: preset.txt 비었음 — 플레이스홀더 치환 없이 flat positive만 전송")

    # 본문 (context의 [BODY] 이후 발췌)
    body_text = session_data.get("body_text", "") or ""

    for sc in scenes:
        prompt_id = sc.get("prompt_id") or str(uuid.uuid4())
        sc["prompt_id"] = prompt_id
        sc["status"] = "queued"
        sc["dispatched_at"] = time.time()

        # 루아 buildPresetPrompt 포팅 — positive/negative에
        # camera/scene/supplement/chars 가 모두 포함된 마커 포맷으로 조립.
        # process_prompt → IllustPromptBuilder.parse_sections 가
        # [Name]/[SETUP]/[CHAR]/[SUPPLEMENT] 를 인식해 최종 빌드.
        if preset_content:
            try:
                positive, negative = build_preset_prompt(preset_content, sc, body_text)
            except Exception as e:
                print(f"[LIGHBD] build_preset_prompt 실패 scene {sc['idx']}: {e}")
                traceback.print_exc()
                positive = sc.get("positive", "") or ""
                negative = sc.get("negative", "") or ""
        else:
            positive = sc.get("positive", "") or ""
            negative = sc.get("negative", "") or ""

        prompt_data = {
            f"lighbd_pos_{sc['idx']}": {
                "_meta": {"title": "긍정프롬프트"},
                "inputs": {"value": positive},
                "class_type": "STRING",
            },
            f"lighbd_neg_{sc['idx']}": {
                "_meta": {"title": "부정프롬프트"},
                "inputs": {"value": negative},
                "class_type": "STRING",
            },
        }

        # 사전 등록 + 큐 적재: server.register_and_enqueue_illustration 공유
        # — /prompt 경로와 동일 코드 경로 (한쪽 고치면 양쪽에 반영)
        label = f"lighbd scene {sc['idx']} ses={session_id[:8]}"
        register_and_enqueue_illustration(
            prompt_id=prompt_id,
            prompt_data=prompt_data,
            raw_body={},
            label=label,
        )
        print(f"[LIGHBD] dispatched scene {sc['idx']} prompt_id={prompt_id[:8]} pos_len={len(positive)} neg_len={len(negative)}")

    # idx 기반 머지 — reroll 시 일부 씬만 재디스패치해도 다른 씬이 안 날아감
    existing = {s.get("idx"): s for s in session_data.get("scenes", []) if isinstance(s, dict)}
    for sc in scenes:
        existing[sc["idx"]] = sc
    session_data["scenes"] = sorted(existing.values(), key=lambda x: x.get("idx", 0))
    session_data["status"] = "generating"
    _save_session(session_id, session_data)


# ─── 메인 ENQUEUE 핸들러 ──────────────────────────────────
def _build_character_dictionary_yaml() -> str:
    """활성 봇(app_config.bot_selected)의 _lb_extra.json 을 읽어
    LLM system 프롬프트용 YAML 캐릭터 도감 문자열을 반환.

    Returns:
        YAML 문자열. 봇 미선택/데이터 없음이면 빈 문자열.
    """
    try:
        import server as _server
        bot_name = _server.app_config.get("bot_selected", "")
        if not bot_name:
            print("[LIGHBD] char dict skip: bot_selected 없음")
            return ""
        from modes.bot_mode import _load_lb_extra
        data = _load_lb_extra(bot_name)
        if not data:
            print(f"[LIGHBD] char dict skip: 봇 '{bot_name}'에 _lb_extra.json 없음")
            return ""
        # 각 엔트리를 YAML 직렬화
        entries = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            app_tags = [t.get("tag", "").strip() for t in (entry.get("appearance") or [])
                        if isinstance(t, dict) and t.get("tag", "").strip()]
            out_tags = [t.get("tag", "").strip() for t in (entry.get("outfit") or [])
                        if isinstance(t, dict) and t.get("tag", "").strip()]
            entries.append({
                "name": name,
                "appearance": app_tags,
                "outfit": out_tags,
            })
        if not entries:
            print(f"[LIGHBD] char dict skip: '{bot_name}'에 유효 캐릭터 없음")
            return ""
        out = yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f"[LIGHBD] char dict loaded: 봇='{bot_name}' 캐릭터={len(entries)}명")
        return out
    except Exception as e:
        print(f"[LIGHBD] char dict build 실패: {e}")
        traceback.print_exc()
        return ""


async def handle_enqueue(context: str, prompt_id: str) -> dict:
    """ENQUEUE 요청 처리: callLLM → 파싱 → 병렬 디스패치 → 세션 저장."""
    if not context or not context.strip():
        msg = "ENQUEUE payload empty - context missing"
        print(f"[LIGHBD] ERROR: {msg}")
        _log_enqueue(prompt_id, context, "", status="error", error=msg)
        return {"plan": "", "status": "error", "error": msg}

    prompts = load_prompts()
    if not prompts.get("system") or not prompts.get("format"):
        msg = "required prompt files missing (system/format)"
        print(f"[LIGHBD] ERROR: {msg}")
        _log_enqueue(prompt_id, context, "", status="error", error=msg)
        return {"plan": "", "status": "error", "error": msg}

    # 루아 LightBoard XNAI 모듈의 메시지 구조 반영:
    #   1. user:    "# System rules" + jailbreak(요정 프레임) + "# Job Instruction"
    #   2. user:    [CHARACTER DICTIONARY] (lb_extra, 있을 때만 별도 메시지)
    #   3. user:    "# Chat log / --- Start of the log ---"
    #   4. assistant: context (로그 본문)
    #   5. user:    "--- End of the log ---"
    #   6. user:    "# Output" + thoughts + system(Tagging Details) + format
    #   7. user:    최종 포맷 리마인더
    # system role 사용 안 함. 로그는 assistant 메시지로 샌드위치.
    # 캐릭터 도감 lb_extra 주입 — LLM이 장면 분할 시 캐릭터 외모/복장 참조
    char_dict_yaml = _build_character_dictionary_yaml()

    jailbreak_txt = (prompts.get("jailbreak") or "").strip()
    job_txt = (prompts.get("job") or "").strip()
    thoughts_txt = (prompts.get("thoughts") or "").strip()
    system_txt = (prompts.get("system") or "").strip()
    format_txt = (prompts.get("format") or "").strip()

    # 1. system rules + job
    msg1_parts = []
    if jailbreak_txt:
        msg1_parts.append("# System rules\n" + jailbreak_txt)
    if job_txt:
        msg1_parts.append("# Job Instruction\n" + job_txt)
    msg1 = "\n\n---\n\n".join(msg1_parts)

    # 6. output instructions
    msg6_parts = []
    if thoughts_txt:
        msg6_parts.append("# Output\n" + thoughts_txt)
    if system_txt:
        msg6_parts.append(system_txt)
    if format_txt:
        msg6_parts.append("[Output format]\n" + format_txt)
    msg6 = "\n\n---\n\n".join(msg6_parts)

    messages = []
    if msg1:
        messages.append({"role": "user", "content": msg1})
    if char_dict_yaml:
        messages.append({"role": "user", "content": "[CHARACTER DICTIONARY]\n" + char_dict_yaml})
    messages.append({"role": "user", "content": "# Chat log\n\n--- Start of the log ---"})
    messages.append({"role": "assistant", "content": context})
    messages.append({"role": "user", "content": "--- End of the log ---"})
    if msg6:
        messages.append({"role": "user", "content": msg6})
    messages.append({"role": "user", "content": "---\n\nAdhere to the format. You MUST OUTPUT IN THE STRUCTURED FORMAT/SYNTAX ABOVE, AS EXPLICITLY INSTRUCTED, WITHOUT ASSUMPTIONS OR GUESSES."})

    try:
        from modes.llm_service import callLLMStream
        print(f"[LIGHBD] callLLMStream start prompt_id={prompt_id[:8]} context_len={len(context)}")
        plan_parts = []
        plan = ""
        async for ev in _stream_with_frontend_notify(prompt_id, messages):
            if ev["type"] == "done":
                plan = ev.get("text", "")
                plan_parts = [plan]
            elif ev["type"] == "error":
                err = ev.get("error", "")
                print(f"[LIGHBD] callLLMStream failed: {err}")
                _log_enqueue(prompt_id, context, "", status="error", error=err)
                return {"plan": "", "status": "error", "error": err}
        if not plan:
            print(f"[LIGHBD] callLLMStream returned empty plan")
            _log_enqueue(prompt_id, context, "", status="error", error="empty plan")
            return {"plan": "", "status": "error", "error": "empty plan"}
        print(f"[LIGHBD] callLLMStream done prompt_id={prompt_id[:8]} plan_len={len(plan)}")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[LIGHBD] EXCEPTION in handle_enqueue: {e}\n{tb}")
        _log_enqueue(prompt_id, context, "", status="error", error=f"{e}\n{tb}")
        return {"plan": "", "status": "error", "error": str(e)}

    # Phase A: 로그 저장 (하위호환)
    _log_enqueue(prompt_id, context, plan, status="ok")

    # Phase B: 파싱 + 디스패치
    scenes = parse_scenes(plan)
    if not scenes:
        print(f"[LIGHBD] WARN: no scenes parsed from plan; skipping dispatch")
        return {"plan": plan, "status": "ok", "session_id": prompt_id, "scenes_count": 0}

    # 본문 발췌 ([BODY] 태그 이후)
    body_text = ""
    bm = re.search(r'\[BODY\]\n?(.*)', context, re.DOTALL)
    if bm:
        body_text = bm.group(1).strip()

    session_data = {
        "session_id": prompt_id,
        "created_at": datetime.datetime.now().isoformat(),
        "context": context,
        "body_text": body_text,
        "plan": plan,
        "scenes": [],
        "status": "planned",
    }
    _save_session(prompt_id, session_data)

    dispatch_generation(prompt_id, scenes, session_data)

    return {
        "plan": plan,
        "status": "ok",
        "session_id": prompt_id,
        "scenes_count": len(scenes),
    }


# ─── 세션 상태 조회 ───────────────────────────────────────
def get_session_state(session_id: str) -> dict | None:
    """세션 JSON 로드 + 각 씬의 prompts[prompt_id].status 동적 반영."""
    data = _load_session(session_id)
    if data is None:
        return None

    try:
        g = _get_server_globals()
        prompts_dict = g["prompts"]
    except Exception as e:
        print(f"[LIGHBD] WARN: cannot access prompts dict: {e}")
        return data

    all_ready = True
    for sc in data.get("scenes", []):
        pid = sc.get("prompt_id")
        if not pid or pid not in prompts_dict:
            sc["status"] = sc.get("status", "unknown")
            continue
        entry = prompts_dict[pid]
        if entry.get("status") == "completed" and entry.get("image_bytes"):
            sc["status"] = "ready"
        elif entry.get("status") == "running":
            sc["status"] = "generating"
        else:
            sc["status"] = sc.get("status", "unknown")
        if sc["status"] != "ready":
            all_ready = False

    data["status"] = "ready" if all_ready and data.get("scenes") else data.get("status", "planned")
    return data


# ─── 리롤 ─────────────────────────────────────────────────
def reroll_scene(session_id: str, scene_idx: int) -> dict:
    """해당 씬을 새 prompt_id 로 재생성 디스패치.

    Returns:
        {"session_id":..., "scene_idx":..., "prompt_id": new_pid, "status": "queued"}
        실패 시 {"error": ...}
    """
    data = _load_session(session_id)
    if data is None:
        return {"error": f"session not found: {session_id}"}

    target = None
    for sc in data.get("scenes", []):
        if sc.get("idx") == scene_idx:
            target = sc
            break
    if target is None:
        return {"error": f"scene idx {scene_idx} not in session"}

    # 기존 prompt_id 엔트리 정리 (옵션: 메모리 절약)
    try:
        g = _get_server_globals()
        old_pid = target.get("prompt_id")
        if old_pid and old_pid in g["prompts"]:
            # 완료된 항목도 지움. 클라이언트가 더 이상 참조 안 함.
            try:
                del g["prompts"][old_pid]
            except KeyError:
                pass
    except Exception as e:
        print(f"[LIGHBD] WARN: cleanup old prompt entry failed: {e}")

    # 새 디스패치
    new_pid = str(uuid.uuid4())
    target["prompt_id"] = new_pid
    target["status"] = "queued"
    target["rerolled_at"] = time.time()

    _save_session(session_id, data)

    # 단일 씬 디스패치 (dispatch_generation 재사용)
    dispatch_generation(session_id, [target], data)
    return {
        "session_id": session_id,
        "scene_idx": scene_idx,
        "prompt_id": new_pid,
        "status": "queued",
    }


def get_image_bytes(prompt_id: str) -> bytes | None:
    """완료된 prompt_id 의 이미지 bytes 반환. 없으면 None."""
    try:
        g = _get_server_globals()
        entry = g["prompts"].get(prompt_id)
    except Exception as e:
        print(f"[LIGHBD] WARN: prompts access failed: {e}")
        return None
    if not entry:
        return None
    if entry.get("status") == "completed" and entry.get("image_bytes"):
        return entry["image_bytes"]
    return None


# ─── 하위호환 로그 (Phase A) ──────────────────────────────
def _log_enqueue(prompt_id: str, context: str, plan: str, status: str, error: str = ""):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt_id": prompt_id,
            "status": status,
            "context": context,
            "plan": plan,
        }
        if error:
            entry["error"] = error

        fname = f"lighbd_enqueue_{prompt_id[:8]}.json"
        path = os.path.join(LOG_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LIGHBD] ERROR: log file save failed: {e}")
