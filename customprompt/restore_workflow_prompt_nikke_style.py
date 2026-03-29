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

_QUALITY = "newest, year2024, (masterpiece, best quality, score_7), highres, absurdres,"

# 캐릭터 목록: (캐릭터 태그, 복장 태그)
_CHARACTER_POOL = [
    "alice \\(nikke\\), pink bodysuit",
    "anis \\(nikke\\)",
    "rapi \\(nikke\\)",
    "dorothy \\(nikke\\)",
    "cinderella \\(nikke\\), white bodysuit",
    "red hood \\(nikke\\)",
    "privaty \\(nikke\\), maid dress",
]


def _pick_character() -> str:
    """초 단위 시간을 시드로 사용하여 캐릭터 태그 문자열을 선택한다."""
    now = datetime.now()
    seed = now.second  # 0~59
    idx = seed % len(_CHARACTER_POOL)
    char_tag = _CHARACTER_POOL[idx]
    print(f"[RESTORE_PROMPT] 캐릭터 선택: seed={seed}, idx={idx}, char={char_tag}")
    return char_tag


def _build_sign_prompt() -> str:
    """선택된 캐릭터가 현재 시간이 적힌 팻말을 든 프롬프트를 생성한다."""
    char_tag = _pick_character()
    H = datetime.now().hour
    M = datetime.now().minute
    return (
        f"{char_tag}, "
        f"(holding sign:1.3), sign, wooden sign, "
        f"(she hold text on sign: \"Job Fin\":1.3), "
        f"standing, v, smile, winking \\(animated\\), yellow background"
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
        "negative": ""
    }
