"""
워크플로우 복원 프롬프트 - LLM 단일 캐릭터(solo) 랜덤 상황

활성 봇(bot_selected)에서 캐릭터 1명을 무작위 선택하고,
해당 캐릭터의 외모/복장(lb-extra) 정보를 LLM에 주어
[SETUP]/[CHAR]/[SUPPLEMENT] 양식의 무작위 상황을 1인(단일) 캐릭터로 생성한다.

동작 조건:
  - bot_mode_enabled == True 이고 bot_selected 가 유효한 봇 이름이어야 한다.
  - 수동 그리기(/api/restore_manual_draw)에서 restore_prompt_file 로 지정해 사용한다.
    bot_mode 가 켜져 있으면 illustration 큐로 진입해 기존 삽화 파이프라인
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
        "당신은 단일 캐릭터(solo) 삽화용 프롬프트를 만드는 도우미다.\n"
        "주어진 캐릭터의 외모/복장 태그와 성별 태그를 바탕으로, 무작위이면서도 "
        "자연스러운 단일 인물 상황을 상상해 아래 양식 그대로 출력한다.\n\n"
        "규칙:\n"
        "1. 반드시 단일 캐릭터(solo) 장면만 만든다. 다른 인물을 등장시키지 않는다.\n"
        "2. [CHAR] 섹션에는 주어진 '외모 태그'와 '복장 태그'를 모두 포함해야 한다.\n"
        "   외모/복장 태그는 원문 그대로 사용하고, 거기에 포즈/표정/동작 태그를 추가한다.\n"
        "3. [SETUP] 은 구도·프레이밍·배경(예: cowboy shot, from above, cafe, night)을 담는다.\n"
        "   매번 다른 분위기가 되도록 무작위로 다양하게 선택한다.\n"
        "4. [SUPPLEMENT] 은 소품/효과/광원 등 부가 요소다. 필요 없으면 비워도 된다.\n"
        "5. 모든 내용은 danbooru 스타일의 영문 태그를 쉼표로 구분한다. 한국어 문장 금지.\n"
        "6. 출력은 정확히 아래 세 섹션만. 다른 설명·주석·여분 텍스트 금지.\n\n"
        "출력 양식:\n"
        "[SETUP]\n"
        "<setup tags>\n"
        "[CHAR]\n"
        "<character appearance + outfit + pose/expression/action tags>\n"
        "[SUPPLEMENT]\n"
        "<supplement tags or empty>"
    )


def _build_user_prompt(char_name: str, appearance: list[str], outfit: list[str],
                       gender: str) -> str:
    appearance_str = ", ".join(appearance) if appearance else "(없음)"
    outfit_str = ", ".join(outfit) if outfit else "(없음)"
    return (
        f"캐릭터 이름: {char_name}\n"
        f"성별 태그: {gender}\n"
        f"외모 태그: {appearance_str}\n"
        f"복장 태그: {outfit_str}\n\n"
        "이 캐릭터 단독으로 등장하는 무작위 상황을 위 양식대로 만들어 줘."
    )


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

async def run() -> dict:
    # 1. 활성 봇 확인
    config = _read_json(CONFIG_PATH) or {}
    bot_name = config.get("bot_selected", "")
    bot_mode_enabled = config.get("bot_mode_enabled", False)

    if not (bot_name and bot_mode_enabled):
        print(
            "[RESTORE_LLM_SOLO] bot_mode 가 꺼져 있거나 bot_selected 가 없습니다. "
            "이 프롬프트는 삽화(bot) 모드에서만 동작합니다. "
            f"(bot_selected={bot_name!r}, bot_mode_enabled={bot_mode_enabled})"
        )
        return {"positive": "", "negative": ""}

    # 2. 봇 데이터 → 활성 봇
    bot_data = _read_json(BOT_DATA_PATH) or {}
    bot = next((b for b in bot_data.get("bots", []) if b.get("name") == bot_name), None)
    if not bot:
        print(f"[RESTORE_LLM_SOLO] 활성 봇을 찾을 수 없음: {bot_name!r}")
        return {"positive": "", "negative": ""}

    # 3. 무작위 캐릭터 선택
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
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": _build_user_prompt(char_name, appearance, outfit, gender)},
    ]

    result = None
    try:
        from modes.llm_service import callLLM
        result = await callLLM(messages)
    except Exception as e:
        print(f"[RESTORE_LLM_SOLO] callLLM 예외: {e}")
        traceback.print_exc()

    if not result:
        print("[RESTORE_LLM_SOLO] LLM1 응답 없음/실패 → LLM2 폴백 시도")
        try:
            from modes.llm_service import callLLM2
            result = await callLLM2(messages)
        except Exception as e:
            print(f"[RESTORE_LLM_SOLO] callLLM2 예외: {e}")
            traceback.print_exc()

    if not result:
        print("[RESTORE_LLM_SOLO] LLM 응답을 받지 못해 빈 프롬프트를 반환합니다.")
        return {"positive": "", "negative": ""}

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
