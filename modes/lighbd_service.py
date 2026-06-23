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

    files = ["system", "job", "format", "thoughts", "jailbreak"]
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
LIGHBD_HISTORY_MAX = 20


def _log_lighbd_history(record: dict) -> None:
    """lighbd 전용 히스토리 파일(logs/lighbd_history.jsonl)에 append.
    최근 LIGHBD_HISTORY_MAX(20) 개만 유지.

    CLAUDE.md 규칙: write 전 백업. 요구사항/ 폴더에 .bak 보관.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        history_backup_dir = os.path.join(BASE_DIR, "요구사항")
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

        line = json.dumps(record, ensure_ascii=False) + "\n"
        existing_lines.append(line)
        # 최근 N개만 유지
        if len(existing_lines) > LIGHBD_HISTORY_MAX:
            existing_lines = existing_lines[-LIGHBD_HISTORY_MAX:]

        with open(LIGHBD_HISTORY_PATH, "w", encoding="utf-8") as f:
            f.writelines(existing_lines)
    except Exception as e:
        print(f"[LIGHBD] history 쓰기 실패: {e}")
        traceback.print_exc()


def _load_lighbd_history(limit: int = LIGHBD_HISTORY_MAX) -> list:
    """최근 limit 개(기본 20) 히스토리 반환 (오래된 → 최신 순)."""
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

    for sc in scenes:
        prompt_id = sc.get("prompt_id") or str(uuid.uuid4())
        sc["prompt_id"] = prompt_id
        sc["status"] = "queued"
        sc["dispatched_at"] = time.time()

        # 최소 prompt_data 구조만 담아 넘김.
        # process_prompt → generate_image_with_prompt 가 update_workflow_if_needed +
        # build_prompt 를 알아서 수행. 따라서 dispatch 시점에 workflow 가 로드되어
        # 있지 않아도 됨.
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
        print(f"[LIGHBD] dispatched scene {sc['idx']} prompt_id={prompt_id[:8]}")

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

    system_content = prompts["system"]
    if prompts.get("jailbreak"):
        system_content += "\n\n---\n\n" + prompts["jailbreak"]
    if prompts.get("thoughts"):
        system_content += "\n\n---\n\n[Thought process]\n" + prompts["thoughts"]

    # 활성 봇의 캐릭터 도감(lb_extra) 주입 — LLM이 장면 분할 시 캐릭터 외모/복장 참조
    char_dict_yaml = _build_character_dictionary_yaml()
    if char_dict_yaml:
        system_content += "\n\n---\n\n[CHARACTER DICTIONARY]\n" + char_dict_yaml

    user_content = prompts.get("job", "") + "\n\n[Output format]\n" + prompts["format"] + "\n\n[Context]\n" + context

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

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
