"""
워크플로우 복원 프롬프트 v3 - 삽화 모드 연동

v2의 캐릭터/손동작/표정 풀을 그대로 사용하되,
삽화 모드가 활성화되어 있으면 [SETUP]/[CHAR]/[SUPPLEMENT] 섹션 형식으로 반환합니다.
큐 매니저의 _apply_illust_pipeline()이 이 프롬프트를 파싱하여
단어 치환 → 캐릭터 감지 → IllustPromptBuilder 빌드를 수행합니다.

삽화 모드가 꺼져 있으면 v2와 완전 동일하게 동작합니다.

필수 함수:
    async def run() -> dict

반환값:
    dict: {"positive": "...", "negative": "..."}
"""

from datetime import datetime
import os
import json


# ─── 공통 ────────────────────────────────────────────────────

def _make_seed() -> int:
    now = datetime.now()
    return now.hour * 10000000 + now.minute * 100000 + now.second * 1000 + now.microsecond // 1000


def _pick_random(pool: list, seed: int) -> str:
    return pool[seed % len(pool)]


# ─── v2 풀 (그대로 사용) ────────────────────────────────────

_QUALITY = "newest, year2024, (masterpiece, best quality, score_7), highres, absurdres, solo,"

_CHARACTER_POOL = [
    "alice \\(nikke\\), pink bodysuit",
    "anis \\(nikke\\)",
    "rapi \\(nikke\\)",
    "dorothy \\(nikke\\), white dress, frilled dress, halterneck, center frills, detached sleeves, see-through sleeves, bare shoulders, sleeves past wrists, frilled choker, neck ribbon, white thighhighs",
    "cinderella \\(nikke\\), white bodysuit, red eyes",
    "red hood \\(nikke\\)",
    "privaty \\(nikke\\), maid dress",
    "anis \\(sparkling summer\\) \\(nikke\\), swimsuit",
    "modernia \\(nikke\\), bodysuit",
    "liliweiss \\(nikke\\), sparkling eyes,blue eyes",
    "shifty \\(nikke\\), blue eyes, blue hair, two side up, long hair, hair ornament, animal ear headphones, headset, sailor collar, button badge, grey serafuku, blue neckerchief, short sleeves, wrist cuffs, breast pockets, impossible clothes, breasts, pouch, white dress, covered navel, bare legs, white footwear, sneakers, socks",
]

_HAND_SIGNS = ["v", "ok sign", "thumbs up", "finger gun", "salute", "arms up", "hand on hip"]

_EXPRESSIONS = [
    "smile", "winking \\(animated\\)", "pout", "surprised", "grin", "blush",
    "closed eyes, smile", "tongue out", "cat mouth", "smirk",
]

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
    "bad gun anatomy, bullpup"
)


def _build_random_char(seed: int) -> tuple[str, str, str]:
    """시드 기반으로 캐릭터/손동작/표정 선택."""
    char = _pick_random(_CHARACTER_POOL, seed)
    hand = _pick_random(_HAND_SIGNS, seed // 7 + 3)
    expr = _pick_random(_EXPRESSIONS, seed // 13 + 5)
    return char, hand, expr


# ─── 삽화 모드: [SETUP]/[CHAR]/[SUPPLEMENT] 형식 ────────────

def _build_illust_prompt() -> dict:
    """v2 풀에서 랜덤 선택 후 섹션 형식으로 반환."""
    seed = _make_seed()
    char, hand, expr = _build_random_char(seed)
    print(f"[RESTORE_V3] 삽화 모드: seed={seed}, char={char}, hand={hand}, expr={expr}")

    setup = "cowboy shot, straight-on, yellow background"
    char_section = f"{char}, standing, {hand}, {expr}"
    supplement = f"Holding a wooden sign that reads \"Manager V4\""

    positive = f"[SETUP]\n{setup}\n[CHAR]\n{char_section}\n[SUPPLEMENT]\n{supplement}"
    return {"positive": positive, "negative": _NEGATIVE}


# ─── 비삽화 모드: v2와 완전 동일 ────────────────────────────

def _build_v2_prompt() -> dict:
    """v2 원본 동작 (ILXL/UPSCALE 섹션 포함)."""
    seed = _make_seed()
    char, hand, expr = _build_random_char(seed)
    print(f"[RESTORE_V3] v2 모드: seed={seed}, char={char}, hand={hand}, expr={expr}")

    sign = (
        f"{char}, "
        f"(holding sign:1.3), sign, wooden sign, "
        f"(she hold text on sign: \"Manager V4\":1.3), "
        f"standing, {hand}, {expr}, yellow background"
    )
    positive = f"{_QUALITY}\n{sign}\n\n[ILXL]\n{sign}\n\n[UPSCALE]\n{sign}"
    return {"positive": positive, "negative": _NEGATIVE}


# ─── 진입점 ──────────────────────────────────────────────────

async def run() -> dict:
    config = {}
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        config_path = os.path.normpath(config_path)
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
    except Exception as e:
        print(f"[RESTORE_V3] config.json 로드 실패: {e}")

    bot_mode_enabled = config.get("bot_mode_enabled", False)

    if bot_mode_enabled:
        return _build_illust_prompt()

    return _build_v2_prompt()
