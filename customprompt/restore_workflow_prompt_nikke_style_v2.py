"""
워크플로우 복원 프롬프트 - 모드 종료 후 원래 워크플로우로 전환 시 사용

모드(복장 추출 등)가 종료되면 자동으로 원래 워크플로우를 실행하여
ComfyUI에 가중치를 미리 VRAM에 로드합니다.

이 파일은 customprompt/ 폴더에 넣고 설정에서 선택하면
모드 처리 완료 후 자동으로 실행됩니다.

필수 함수:
    async def run() -> dict

반환값:
    dict: {"positive": "...", "negative": "..."}
    - positive: 긍정 프롬프트 ([ILXL], [UPSCALE] 섹션 포함)
    - negative: 부정 프롬프트
"""

from datetime import datetime

# ─── 프롬프트 구성 요소 ─────────────────────────────────

_QUALITY = "newest, year2024, (masterpiece, best quality, score_7), highres, absurdres, solo,"

# 캐릭터 목록
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

# 손모양
_HAND_SIGNS = [
    "v",
    "ok sign",
    "thumbs up",
    "finger gun",
    "salute",
    "arms up",
    "hand on hip",
]

# 표정
_EXPRESSIONS = [
    "smile",
    "winking \\(animated\\)",
    "pout",
    "surprised",
    "grin",
    "blush",
    "closed eyes, smile",
    "tongue out",
    "cat mouth",
    "smirk",
]


def _make_seed() -> int:
    """마이크로초까지 사용하여 매 호출마다 다른 시드 생성"""
    now = datetime.now()
    return now.hour * 10000000 + now.minute * 100000 + now.second * 1000 + now.microsecond // 1000


def _pick_random(pool: list, seed: int) -> str:
    """시드 기반으로 풀에서 하나 선택"""
    return pool[seed % len(pool)]


def _build_sign_prompt() -> str:
    """랜덤 캐릭터 + 랜덤 손모양 + 랜덤 표정으로 팻말 프롬프트 생성"""
    seed = _make_seed()
    char_tag = _pick_random(_CHARACTER_POOL, seed)
    hand_sign = _pick_random(_HAND_SIGNS, seed // 7 + 3)
    expression = _pick_random(_EXPRESSIONS, seed // 13 + 5)

    print(f"[RESTORE_PROMPT] seed={seed}, char={char_tag}, hand={hand_sign}, expr={expression}")

    return (
        f"{char_tag}, "
        f"(holding sign:1.3), sign, wooden sign, "
        f"(she hold text on sign: \"V3\":1.3), "
        f"standing, {hand_sign}, {expression}, yellow background"
    )


async def run() -> dict:
    """
    워크플로우 복원에 사용할 프롬프트를 반환합니다.

    Returns:
        dict: {"positive": "긍정프롬프트", "negative": "부정프롬프트"}
    """
    sign = _build_sign_prompt()
    positive = f"{_QUALITY}\n{sign}\n\n[ILXL]\n{sign}\n\n[UPSCALE]\n{sign}"
    return {
        "positive": positive,
        "negative": "lowres, worst quality, bad quality, low quality, normal quality, worst detail, displeasing, fewer details, unfinished, incomplete, sketch, watermark, username, patreon username, logo, patreon logo, sign, artist collaboration, 3d, realistic, blender, pixel art, character doll, JPEG artifacts, aliasing, dithering, scan artifacts, blurry, chromatic aberration, screentone, film grain, heavy film grain, digital dissolve, censor, censored, mosaic censoring, bar censor, cropped, split theme, split screen, head out of frame, distorted composition, bad perspective, one-hour drawing challenge, 4koma, 2koma, bad anatomy, anatomically incorrect, bad proportions, mutation, deformed, disfigured, duplicate, amputee, bad hands, bad hand structure, bad arm, bad leg, bad limbs, bad feet, missing finger, extra digits, fewer digits, unclear fingertips, extra arms, extra legs, twist, bad face, mob face, bad eyes, unnatural hair, big head, big nose, nostrils, philtrum, beard, bald, long neck, futanari, breast ptosis, squiggly, bad gun anatomy, bullpup"
    }
