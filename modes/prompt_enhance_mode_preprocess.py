"""
프롬프트 강화 모드 전처리 - 배치 시작 시 중복 chat 정리

배치의 첫 이미지에서만 호출됨.
동일한 chat으로 그림을 다시 그리는 경우, 이전 배치의 저장된 엔트리를 삭제.

매커니즘:
1. 배치의 첫 이미지가 들어오면 현재 chat을 추출
2. 모든 캐릭터 스토리지 파일에서 최신 배치(--- 이후)의 chat과 비교
3. 80% 이상 유사도(Levenshtein ratio)면 동일 chat으로 판단
4. 해당 캐릭터의 최신 배치 엔트리를 삭제
"""

import json
import os
from difflib import SequenceMatcher

# ─── 상수 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMPROMPT_DIR = os.path.join(BASE_DIR, "customprompt")
STORAGE_DIR = os.path.join(CUSTOMPROMPT_DIR, "enhance_outfit_prompt_v4_storage")
SIMILARITY_THRESHOLD = 0.8
CHAT_COMPARE_LENGTH = 500


def _similarity(s1: str, s2: str) -> float:
    """Levenshtein 기반 유사도 (0.0 ~ 1.0).
    difflib.SequenceMatcher를 사용하여 최소 수정 거리 기반 비율 계산.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()


def _get_storage_files() -> list[str]:
    """모든 캐릭터 스토리지 JSON 파일 경로 반환"""
    if not os.path.exists(STORAGE_DIR):
        return []
    return [
        os.path.join(STORAGE_DIR, f)
        for f in os.listdir(STORAGE_DIR)
        if f.endswith('.json')
    ]


def _find_latest_batch_start(history: list) -> int:
    """마지막 --- 구분선의 위치 반환.
    --- 가 없으면 0 (전체가 하나의 배치).
    """
    for i in range(len(history) - 1, -1, -1):
        if history[i] == "---":
            return i + 1
    return 0


def _get_latest_chat(history: list, batch_start: int) -> str | None:
    """최신 배치에서 첫 번째 dict 엔트리의 chat 필드 반환."""
    for entry in reversed(history[batch_start:]):
        if isinstance(entry, dict) and entry.get("chat"):
            return entry["chat"]
    return None


async def preprocess_clean_duplicate_chats(current_chat: str) -> int:
    """배치 첫 이미지에서 호출. 동일 chat 감지 시 최신 배치 엔트리 삭제.

    Returns: 삭제된 엔트리 수
    """
    if not current_chat:
        return 0

    current_trimmed = current_chat[:CHAT_COMPARE_LENGTH]
    deleted_total = 0

    for filepath in _get_storage_files():
        char_name = os.path.splitext(os.path.basename(filepath))[0]

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if not isinstance(history, list) or not history:
            continue

        # 최신 배치 범위 찾기
        batch_start = _find_latest_batch_start(history)

        # 최신 배치의 chat 추출
        latest_chat = _get_latest_chat(history, batch_start)
        if latest_chat is None:
            continue

        # 유사도 비교
        if _similarity(current_trimmed, latest_chat[:CHAT_COMPARE_LENGTH]) < SIMILARITY_THRESHOLD:
            continue

        # 동일 chat 감지 - 최신 배치 삭제
        entries_removed = len(history) - batch_start
        cut_point = batch_start

        # 배치 앞의 --- 도 함께 삭제
        if cut_point > 0 and history[cut_point - 1] == "---":
            cut_point -= 1

        history = history[:cut_point]
        deleted_total += entries_removed

        # 파일이 비어있지 않으면 trailing --- 제거
        while history and history[-1] == "---":
            history.pop()

        # 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"[PREPROCESS] {char_name}: 최신 배치 {entries_removed}개 엔트리 삭제 (동일 chat 감지, 유사도 {_similarity(current_trimmed, latest_chat[:CHAT_COMPARE_LENGTH]):.1%})")

    if deleted_total > 0:
        print(f"[PREPROCESS] 총 {deleted_total}개 중복 엔트리 정리 완료")

    return deleted_total
