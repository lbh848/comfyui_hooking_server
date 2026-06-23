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
    queue_manager = g["queue_manager"]
    build_prompt = g["build_prompt"]

    for sc in scenes:
        prompt_id = sc.get("prompt_id") or str(uuid.uuid4())
        sc["prompt_id"] = prompt_id
        sc["status"] = "queued"
        sc["dispatched_at"] = time.time()

        try:
            prompt_data = build_prompt(sc.get("positive", ""), sc.get("negative", ""))
        except Exception as e:
            print(f"[LIGHBD] ERROR: build_prompt failed scene={sc['idx']}: {e}")
            sc["status"] = "error"
            sc["error"] = f"build_prompt: {e}"
            continue

        # 사전 등록 — _handle_illustration → process_prompt 가 채워 넣음
        prompts_dict[prompt_id] = {
            "status": "running",
            "prompt": prompt_data,
            "client_id": "",
            "extra_data": {},
            "outputs": {},
            "filename": None,
            "save_node_id": None,
            "image_bytes": None,
            "timestamp": time.time(),
        }

        label = f"lighbd scene {sc['idx']} ses={session_id[:8]}"
        asyncio.create_task(queue_manager.add_item(
            "illustration",
            label,
            {"prompt_id": prompt_id, "prompt_data": prompt_data, "raw_body": {}},
            priority=0,
        ))
        print(f"[LIGHBD] dispatched scene {sc['idx']} prompt_id={prompt_id[:8]}")

    # idx 기반 머지 — reroll 시 일부 씬만 재디스패치해도 다른 씬이 안 날아감
    existing = {s.get("idx"): s for s in session_data.get("scenes", []) if isinstance(s, dict)}
    for sc in scenes:
        existing[sc["idx"]] = sc
    session_data["scenes"] = sorted(existing.values(), key=lambda x: x.get("idx", 0))
    session_data["status"] = "generating"
    _save_session(session_id, session_data)


# ─── 메인 ENQUEUE 핸들러 ──────────────────────────────────
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

    user_content = prompts.get("job", "") + "\n\n[Output format]\n" + prompts["format"] + "\n\n[Context]\n" + context

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    try:
        from modes.llm_service import callLLM
        print(f"[LIGHBD] callLLM start prompt_id={prompt_id[:8]} context_len={len(context)}")
        plan = await callLLM(messages)
        if plan.startswith("[LLM 실패]"):
            print(f"[LIGHBD] callLLM failed: {plan}")
            _log_enqueue(prompt_id, context, plan, status="error", error=plan)
            return {"plan": "", "status": "error", "error": plan}
        print(f"[LIGHBD] callLLM done prompt_id={prompt_id[:8]} plan_len={len(plan)}")
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
