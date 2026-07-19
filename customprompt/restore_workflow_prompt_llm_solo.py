"""
워크플로우 복원 프롬프트 - LLM 단일 캐릭터(solo) 랜덤 상황

활성 봇(bot_selected)에서 캐릭터 1명을 무작위 선택하고,
해당 캐릭터의 외모/복장(lb-extra) 정보를 LLM에 주어
[SETUP]/[CHAR]/[SUPPLEMENT] 양식의 무작위 상황을 1인(단일) 캐릭터로 생성한다.

동작 조건:
  - bot_selected 가 유효한 봇 이름이어야 한다 (삽화 모드는 항상 ON 고정).
  - 수동 그리기(/api/restore_manual_draw)에서 restore_prompt_file 로 지정해 사용한다.
    bot이 선택되어 있으면 illustration 큐로 진입해 기존 삽화 파이프라인
    (단어치환 → 캐릭터 감지 → IllustPromptBuilder 빌드)을 그대로 탄다.

필수 함수:
    async def run() -> dict

반환값:
    dict: {"positive": "...", "negative": "..."}
          (positive 는 [CHAT]/[SLOT]/[Name]/[SETUP]/[CHAR]/[SUPPLEMENT] 섹션 형식)
"""

import os
import json
import random
import traceback


# ─── 경로 ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
BOT_DATA_PATH = os.path.join(BASE_DIR, "asset_data", "bot.json")


# ─── 부정 프롬프트 (nikke v3 재사용) ────────────────────────

_NEGATIVE = (
    "lowres, worst quality, bad quality, low quality, normal quality, worst detail, "
    "displeasing, fewer details, unfinished, incomplete, sketch, watermark, username, "
    "patreon username, logo, patreon logo, sign, artist collaboration, 3d, realistic, "
    "blender, pixel art, character doll, JPEG artifacts, aliasing, dithering, scan artifacts, "
    "blurry, chromatic aberration, screentone, film grain, heavy film grain, digital dissolve, "
    "censor, censored, mosaic censoring, bar censor, cropped, split theme, split screen, "
    "head out of frame, distorted composition, bad perspective, one-hour drawing challenge, "
    "4koma, 2koma, bad anatomy, anatomically incorrect, bad proportions, mutation, deformed, "
    "disfigured, duplicate, amputee, bad hands, bad hand structure, bad arm, bad leg, bad limbs, "
    "bad feet, missing finger, extra digits, fewer digits, unclear fingertips, extra arms, "
    "extra legs, twist, bad face, mob face, bad eyes, unnatural hair, big head, big nose, "
    "nostrils, philtrum, beard, bald, long neck, futanari, breast ptosis, squiggly, "
    "bad gun anatomy, bullpup, multiple girls, multiple boys"
)


# ─── 데이터 로드 ─────────────────────────────────────────────

def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[RESTORE_LLM_SOLO] 파일 로드 실패 {path}: {e}")
        traceback.print_exc()
        return None


def _pick_random_character(bot: dict) -> dict | None:
    chars = bot.get("characters", [])
    if not chars:
        return None
    return random.choice(chars)


def _get_lb_extra_entry(bot_name: str, char_name: str) -> dict | None:
    try:
        from modes.bot_mode import _load_lb_extra
        data = _load_lb_extra(bot_name) or []
    except Exception as e:
        print(f"[RESTORE_LLM_SOLO] _load_lb_extra import/실행 실패: {e}")
        traceback.print_exc()
        return None
    return next((e for e in data if e.get("name") == char_name), None)


def _collect_tags(entry: dict, key: str) -> list[str]:
    """lb_extra 엔트리에서 key(appearance/outfit) 태그 리스트 추출."""
    if not entry:
        return []
    out = []
    for t in entry.get(key, []):
        tag = (t.get("tag", "") or "").strip()
        if tag and tag not in out:
            out.append(tag)
    return out


# ─── LLM 프롬프트 구성 ──────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "You are a helper that creates single-character (solo) illustration prompts.\n"
        "Based on the given character's appearance/outfit tags and the gender tag, "
        "imagine a random yet natural single-person scene and output it exactly in the "
        "format below.\n\n"
        "Rules:\n"
        "1. Always create a single-character (solo) scene. Do not introduce any other person.\n"
        "2. The [CHAR] section must include all of the given 'appearance tags' and 'outfit tags'.\n"
        "   Use the appearance/outfit tags verbatim, and add pose/expression/action tags to them.\n"
        "3. [SETUP] holds composition, framing, and background (e.g. cowboy shot, from above, "
        "cafe, night). Vary it randomly so each result has a different mood.\n"
        "4. [SUPPLEMENT] holds accessories, effects, lighting, and other extras. It may be left "
        "empty if unnecessary.\n"
        "5. All content must be danbooru-style English tags separated by commas. No Korean "
        "sentences.\n"
        "6. Output exactly the three sections below. No other explanation, commentary, or extra "
        "text.\n\n"
        "Output format:\n"
        "[SETUP]\n"
        "<setup tags>\n"
        "[CHAR]\n"
        "<character appearance + outfit + pose/expression/action tags>\n"
        "[SUPPLEMENT]\n"
        "<supplement tags or empty>"
    )


def _build_user_prompt(char_name: str, appearance: list[str], outfit: list[str],
                       gender: str, situation: str = "") -> str:
    appearance_str = ", ".join(appearance) if appearance else "(none)"
    outfit_str = ", ".join(outfit) if outfit else "(none)"
    base = (
        f"Character name: {char_name}\n"
        f"Gender tag: {gender}\n"
        f"Appearance tags: {appearance_str}\n"
        f"Outfit tags: {outfit_str}\n"
    )
    sit = (situation or "").strip()
    if sit:
        # User gives a rough situation directive -> the LLM must obey it, but freely
        # invent details the directive does not specify.
        base += (
            f"Situation directive: {sit}\n\n"
            "Create a scene where this character appears alone, strictly following the "
            "'Situation directive' above. Freely imagine any details not covered by the directive "
            "(composition, background, pose, expression, props, etc.) and output them in the "
            "format above."
        )
    else:
        base += "\nCreate a random scene where this character appears alone, in the format above."
    return base


async def _notify_llm_widget(event_type: str, data: dict | None = None) -> None:
    """LIGHBD 우하단 위젯용 WS 이벤트 발생.

    callLLMTask 는 비스트리밍이라 delta 가 없지만, start/done/error 만으로
    위젯은 충분히 표시된다(bot_mode/instance_lora 와 동일 패턴).
    """
    try:
        import server as _server
        await _server.notify_frontend(
            "lighbd_llm_stream", {"type": event_type, **(data or {})}
        )
    except Exception as e:
        print(f"[RESTORE_LLM_SOLO] WARN: notify_frontend 실패: {e}")


# ─── LLM 출력 파싱 ───────────────────────────────────────────

def _parse_llm_sections(text: str) -> tuple[str, str, str]:
    """LLM 출력에서 [SETUP]/[CHAR]/[SUPPLEMENT] 본문 추출."""
    import re

    def _grab(tag: str) -> str:
        # [TAG] 헤더 이후부터 다음 [xxx] 헤더(또는 끝)까지
        m = re.search(
            r"\[%s\]\s*\n(.*?)(?=\n\[[A-Z_]+\]\s*\n|$)" % tag,
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return ""
        body = m.group(1).strip()
        # 줄바꿈/여러 구분자를 쉼표로 통일 후, danbooru 태그(ASCII)만 남김.
        # LLM이 한국어 설명 등 노이즈를 섞어 넣는 경우를 방지.
        for sep in ["\n", "、"]:
            body = body.replace(sep, ",")
        parts = [p.strip() for p in body.split(",")]
        ascii_parts = [
            p for p in parts
            if p and all(ord(ch) < 0x3000 for ch in p)
        ]
        return ", ".join(ascii_parts)

    setup = _grab("SETUP")
    char = _grab("CHAR")
    supplement = _grab("SUPPLEMENT")
    return setup, char, supplement


def _ensure_tags_in_char(char_section: str, appearance: list[str],
                         outfit: list[str]) -> str:
    """[CHAR] 에 외모/복장 태그가 누락되어 있으면 중복 없이 보강.

    LLM이 빼먹을 수 있으므로 요구사항(외모/복장 포함)을 보장한다.
    """
    parts = [p.strip() for p in char_section.split(",") if p.strip()]
    existing_lower = {p.lower() for p in parts}
    for tag in list(appearance) + list(outfit):
        if tag and tag.lower() not in existing_lower:
            parts.append(tag)
            existing_lower.add(tag.lower())
    return ", ".join(parts)


# ─── 진입점 ──────────────────────────────────────────────────

async def run(char_name: str | None = None, situation: str | None = None) -> dict:
    # 1. 활성 봇 확인 (삽화 모드는 항상 ON)
    config = _read_json(CONFIG_PATH) or {}
    bot_name = config.get("bot_selected", "")

    if not bot_name:
        print(
            "[RESTORE_LLM_SOLO] bot_selected 가 없습니다. "
            "이 프롬프트는 bot이 선택된 삽화 모드에서만 동작합니다. "
            f"(bot_selected={bot_name!r})"
        )
        return {"positive": "", "negative": ""}

    # 2. 봇 데이터 → 활성 봇
    bot_data = _read_json(BOT_DATA_PATH) or {}
    bot = next((b for b in bot_data.get("bots", []) if b.get("name") == bot_name), None)
    if not bot:
        print(f"[RESTORE_LLM_SOLO] 활성 봇을 찾을 수 없음: {bot_name!r}")
        return {"positive": "", "negative": ""}

    # 3. 캐릭터 선택: char_name 이 지정되면 해당 캐릭터, 없으면 무작위
    if char_name:
        char = next(
            (c for c in bot.get("characters", []) if c.get("name") == char_name),
            None,
        )
        if not char:
            print(
                f"[RESTORE_LLM_SOLO] 지정한 캐릭터를 찾을 수 없음: {char_name!r} "
                f"(봇={bot_name!r})"
            )
            return {"positive": "", "negative": ""}
        print(f"[RESTORE_LLM_SOLO] 지정 캐릭터 사용: {char_name!r}")
    else:
        char = _pick_random_character(bot)
        if not char:
            print(f"[RESTORE_LLM_SOLO] 봇에 캐릭터가 없음: {bot_name!r}")
            return {"positive": "", "negative": ""}

    char_name = char.get("name", "")
    gender = char.get("gender_tag", "1girl")

    # 4. 외모/복장 태그 수집 (lb-extra)
    entry = _get_lb_extra_entry(bot_name, char_name)
    appearance = _collect_tags(entry, "appearance")
    outfit = _collect_tags(entry, "outfit")
    print(
        f"[RESTORE_LLM_SOLO] 선택 캐릭터: {char_name!r} | "
        f"gender={gender} | 외모 {len(appearance)}개 | 복장 {len(outfit)}개"
    )
    if not (appearance or outfit):
        print(
            f"[RESTORE_LLM_SOLO] 주의: {char_name!r} 의 lb-extra 외모/복장 태그가 비어 있습니다. "
            "LLM은 캐릭터 이름과 성별만으로 상황을 생성합니다."
        )

    # 5. LLM 호출
    sit = (situation or "").strip()
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": _build_user_prompt(char_name, appearance, outfit, gender, sit)},
    ]
    if sit:
        print(f"[RESTORE_LLM_SOLO] 상황 지시 있음({len(sit)}자) — LLM이 준수하며 자유 작성")
    else:
        print("[RESTORE_LLM_SOLO] 상황 지시 없음 — 무작위 상황 생성")

    # LLM 호출 (외부 API 분기: llm_service 가 task_key 별 primary/fallback 판단)
    # LIGHBD 우하단 위젯 표시(start/done/error) + 자세히 히스토리(lighbd_history.jsonl) 기록.
    import time as _time
    import datetime
    from modes.lighbd_service import _log_lighbd_history
    prompt_id = f"restore_llm_solo:{char_name}"

    await _notify_llm_widget("start", {"model": "restore_workflow"})
    _t0 = _time.time()
    result = None
    err_msg = None
    try:
        from modes.llm_service import callLLMTask
        result = await callLLMTask("restore_workflow", messages)
    except Exception as e:
        print(f"[RESTORE_LLM_SOLO] callLLMTask 예외: {e}")
        traceback.print_exc()
        err_msg = f"{type(e).__name__}: {e}"
        await _notify_llm_widget("error", {"error": err_msg})

    elapsed = round(_time.time() - _t0, 3)

    # 실패(예외 또는 빈 응답) → error 히스토리 기록 후 빈 프롬프트 반환
    if err_msg or not result:
        msg = err_msg or "LLM 응답을 받지 못함(빈 응답)"
        print(f"[RESTORE_LLM_SOLO] LLM 실패: {msg}")
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": prompt_id,
            "input": messages,
            "output": result or "",
            "elapsed": elapsed,
            "status": "error",
            "error": msg,
        })
        return {"positive": "", "negative": ""}

    # 성공 → done 위젯 표시 + 히스토리 기록 (이후 파싱 실패해도 LLM 출력은 남김)
    est_tokens = max(1, len(result) // 3)
    est_tps = round(est_tokens / elapsed, 1) if elapsed > 0 else 0.0
    await _notify_llm_widget("done", {
        "text": result,
        "completion_tokens": est_tokens,
        "elapsed": elapsed,
        "tps": est_tps,
    })
    _log_lighbd_history({
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "prompt_id": prompt_id,
        "input": messages,
        "output": result,
        "completion_tokens": est_tokens,
        "elapsed": elapsed,
        "tps": est_tps,
        "ttft": None,
        "status": "ok",
    })

    # 6. 섹션 파싱
    setup, char_section, supplement = _parse_llm_sections(result)
    if not setup or not char_section:
        print(
            "[RESTORE_LLM_SOLO] LLM 출력에서 [SETUP]/[CHAR] 섹션을 파싱하지 못했습니다.\n"
            f"--- 원본(앞 400자) ---\n{result[:400]}"
        )
        return {"positive": "", "negative": ""}

    # 7. 외모/복장 태그 보장 (누락 시 [CHAR] 에 추가)
    char_section = _ensure_tags_in_char(char_section, appearance, outfit)

    # 8. 삽화 섹션 조립 (nikke v3 와 동일 형식)
    positive = (
        f"[CHAT]\n(restore_llm_solo) no chat context\n"
        f"[SLOT]\n(restore slot before) || (restore slot after)\n"
        f"[Name]\n{char_name}\n"
        f"[SETUP]\n{setup}\n"
        f"[CHAR]\n{char_section}\n"
        f"[SUPPLEMENT]\n{supplement}"
    )

    print(f"[RESTORE_LLM_SOLO] 생성 완료: 캐릭터={char_name!r}, setup={setup[:40]!r}")
    return {"positive": positive, "negative": _NEGATIVE}
