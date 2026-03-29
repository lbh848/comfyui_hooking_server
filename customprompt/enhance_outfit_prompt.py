"""
프롬프트 강화 - 캐릭터 복장 일관성 검사 및 보정

이 파일은 customprompt/ 폴더에 넣고 설정에서 선택하면
배치 프롬프트 강화 모드에서 자동으로 실행됩니다.

필수 함수:
    async def run(character_name: str, char_block: str, outfit_result: str = None,
                  chat_content: str = "") -> str

인자:
    character_name: 캐릭터 이름 (소문자)
    char_block: 현재 캐릭터 블럭 텍스트 (| 로 구분된 단위 중 하나)
    outfit_result: outfit_mode의 LLM 복장 통합 결과 JSON (없으면 None)
    chat_content: 현재 채팅 내용

반환값:
    str: 강화된 캐릭터 블럭 텍스트

주의:
- callLLM이 service/model 설정을 자동으로 처리합니다. (재시도 없음, 단일 시도)
- 프롬프트는 영어, 간단한 명령형 문장으로 작성
"""

import re
import json
import os
import datetime
import tiktoken
from modes.llm_service import callLLM, callLLM2, get_config


# ─── Token Counting ────────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc.encode(text))


# ─── Prompt Logger ────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
PROMPT_LOG_FILE = os.path.join(LOG_DIR, "enhance_prompt_io.log")


def _log_prompt_io(character_name: str, input_data: dict, output_data: str):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "character": character_name,
            "input": input_data,
            "output": output_data[:2000],
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[ENHANCE_PROMPT_LOGGER] 로그 기록 실패: {e}")


# ─── Message Conversion (GPT <-> Gemini) ────────────────────

def gpt2gemini(messages: list) -> list:
    """GPT 형식 messages -> Gemini 형식으로 변환."""
    result = []
    system_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            result.append({"role": role, "content": content})
    if system_parts and result:
        merged_system = "\n\n".join(system_parts)
        first_user = result[0]
        result[0] = {
            "role": "user",
            "content": f"[System Instructions]\n{merged_system}\n\n[User Request]\n{first_user['content']}",
        }
    elif system_parts:
        result.insert(0, {"role": "user", "content": "\n\n".join(system_parts)})
    return result


def gemini2gpt(messages: list) -> list:
    """Gemini 형식 messages -> GPT 형식으로 변환."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        sys_match = re.match(
            r'\[System Instructions\]\s*(.*?)\s*\[User Request\]\s*(.*)',
            content, re.DOTALL
        )
        if sys_match and role == "user":
            result.append({"role": "system", "content": sys_match.group(1).strip()})
            result.append({"role": "user", "content": sys_match.group(2).strip()})
        else:
            result.append({"role": role, "content": content})
    return result


def _is_gemini_model(model_name: str) -> bool:
    if not model_name:
        return False
    return any(kw in model_name.lower() for kw in ("gemini", "gemma"))


# ─── Prompt Building ───────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "### TASK\n"
        "You are a character outfit consistency checker and enhancer for AI image generation prompts.\n"
        "Given a character description block, previous outfit analysis, and current chat context:\n"
        "1. Check if the outfit description in the char_block is consistent with the outfit analysis.\n"
        "2. Check if the outfit makes sense for the current situation in the chat.\n"
        "3. If the outfit changed without a valid reason in the chat, restore it to the previous outfit.\n"
        "4. If the outfit is inconsistent or incorrect, enhance the description with precise details.\n"
        "\n"
        "### RULES\n"
        "1. Always output the FULL character block, not just the changed parts.\n"
        "2. The character's full name (e.g., 'mikazuki yozora (boku wa tomodachi ga sukunai)') MUST appear exactly once.\n"
        "3. Keep the original structure and tags that are correct (age, body type, hair, eyes, etc.).\n"
        "4. When describing headgear, clothing, and accessories, write them as SHORT DESCRIPTIVE PHRASES, not just comma-separated tags.\n"
        "   - GOOD: 'wearing a white short-sleeved cotton collared shirt with a thin black necktie'\n"
        "   - BAD: 'white shirt, black necktie'\n"
        "5. Do NOT use the | character in your output. It is reserved as a character separator.\n"
        "6. Process ONE character at a time. Never mix two characters in one block.\n"
        "7. If outfit_result is provided, use it as the ground truth for what the character is wearing.\n"
        "8. If outfit_result is None, keep the original outfit description unless it clearly contradicts the chat.\n"
        "9. If previous_chat is provided, compare it with current chat to detect outfit changes.\n"
        "   If the situation has NOT changed but the outfit description differs, prefer the outfit_result.\n"
        "10. Output ONLY the enhanced character block text. No explanations, no JSON, no markdown.\n"
        "\n"
        "### EXAMPLES\n"
        "\n"
        "Example 1 - Outfit inconsistent with previous result (should fix):\n"
        "Input char_block: '1girl, school uniform, blue shirt, red skirt, smiling'\n"
        "Input outfit_result: {\"clothing\": \"white short-sleeved cotton collared shirt, black wool pleated mini skirt\"}\n"
        "Output: '1girl, school uniform, wearing a white short-sleeved cotton collared shirt, a black wool pleated mini skirt, smiling'\n"
        "\n"
        "Example 2 - Outfit consistent (keep as is):\n"
        "Input char_block: '1girl, school uniform, white collared shirt, black pleated skirt, standing'\n"
        "Input outfit_result: {\"clothing\": \"white short-sleeved cotton collared shirt, black wool pleated mini skirt\"}\n"
        "Output: '1girl, school uniform, wearing a white short-sleeved cotton collared shirt with a black wool pleated mini skirt, standing'\n"
        "\n"
        "Example 3 - No outfit result, outfit contradicts chat:\n"
        "Input char_block: '1girl, casual clothes, t-shirt, jeans, walking outside'\n"
        "Input chat: 'She changed into her school uniform before heading out.'\n"
        "Output: '1girl, school uniform, wearing a white short-sleeved cotton collared shirt, a black pleated skirt, walking outside'\n"
        "\n"
        "Example 4 - Outfit changed with valid reason (accept change):\n"
        "Input char_block: '1girl, swimsuit, blue bikini, at the pool'\n"
        "Input chat: 'She went to the pool and changed into her swimsuit.'\n"
        "Output: '1girl, wearing a blue two-piece bikini swimsuit, at the pool'\n"
        "\n"
        "### OUTPUT FORMAT\n"
        "Output ONLY the enhanced character block. No wrapping, no explanation.\n"
        "The output must be a single continuous text suitable for Danbooru-style image generation."
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimation. Korean ~2 chars/token, English ~4 chars/token."""
    if not text:
        return 0
    korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff')
    other_chars = len(text) - korean_chars
    return korean_chars // 2 + other_chars // 4 + 1


MAX_TOTAL_TOKENS = 14000


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    """앞에서부터 잘라 max_tokens 이하로 만듦. 뒤(최신)를 유지."""
    if not text or _count_tokens(text) <= max_tokens:
        return text
    # 바이너리 서치로 자를 char 위치 찾기
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if _count_tokens(text[mid:]) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def _build_user_message(
    character_name: str,
    char_block: str,
    outfit_result: str = None,
    chat_content: str = "",
    previous_chat: str = "",
) -> str:
    parts = []
    parts.append(f"### Character Name\n{character_name}")
    parts.append(f"### Current Character Block\n{char_block}")

    if outfit_result:
        parts.append(f"### Outfit Analysis Result (ground truth)\n{outfit_result}")

    # current chat은 그대로 사용 (자르지 않음)
    if chat_content:
        parts.append(f"### Current Chat Context\n{chat_content}")

    # current chat 포함한 현재까지의 토큰 합산
    base_text = "\n\n".join(parts)
    base_tokens = _count_tokens(base_text)
    system_tokens = _count_tokens(_build_system_prompt())

    used_tokens = base_tokens + system_tokens
    remaining = MAX_TOTAL_TOKENS - used_tokens

    if previous_chat:
        prev_tokens = _count_tokens(previous_chat)
        if prev_tokens > remaining and remaining > 0:
            previous_chat = _trim_to_tokens(previous_chat, remaining)
        elif remaining <= 0:
            previous_chat = ""
        parts.append(f"### Previous Chat Context (when outfit was last tracked)\n{previous_chat}")

    return "\n\n".join(parts)


# ─── Main Entry Point ─────────────────────────────────────

async def run(
    character_name: str,
    char_block: str,
    outfit_result: str = None,
    chat_content: str = "",
    previous_chat: str = "",
) -> str:
    """
    캐릭터 블럭의 복장 묘사를 검증하고 강화.

    Args:
        character_name: 캐릭터 이름
        char_block: 현재 캐릭터 블럭
        outfit_result: outfit_mode의 LLM 복장 통합 결과 (JSON 문자열, None 가능)
        chat_content: 현재 채팅 내용
        previous_chat: 이전 복장 통합 시점의 채팅 내용

    Returns:
        강화된 캐릭터 블럭 텍스트
    """
    system_prompt = _build_system_prompt()
    user_message = _build_user_message(
        character_name, char_block, outfit_result, chat_content, previous_chat
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # 입력 로그
    log_input = {
        "character_name": character_name,
        "char_block_length": len(char_block),
        "has_outfit_result": outfit_result is not None,
        "chat_length": len(chat_content) if chat_content else 0,
    }

    # LLM1 호출
    result = await callLLM(messages)

    # LLM1 실패 시 LLM2 폴백
    if result.startswith("[LLM 실패]"):
        config = get_config()
        llm_model2 = config.get("llm_model2", "")
        if llm_model2:
            if _is_gemini_model(llm_model2):
                messages2 = gpt2gemini(messages)
            else:
                messages2 = messages
            result = await callLLM2(messages2)

    # 출력 로그
    _log_prompt_io(character_name, log_input, result)

    if result.startswith("[LLM 실패]"):
        return result

    # 결과 정리: 불필요한 마크다운/래핑 제거
    result = result.strip()
    result = re.sub(r'^```(?:\w+)?\s*', '', result)
    result = re.sub(r'\s*```$', '', result)
    result = result.strip()

    # 캐릭터 이름이 결과에 없으면 원본 반환
    if character_name.lower() not in result.lower():
        # 원본에서 풀네임 패턴 찾기
        name_pattern = re.search(
            re.escape(character_name) + r'\s*\([^)]+\)', char_block, re.IGNORECASE
        )
        if name_pattern:
            full_name = name_pattern.group(0)
            if full_name.lower() not in result.lower():
                return char_block
        else:
            if character_name.lower() not in result.lower():
                return char_block

    # | 문자가 결과에 포함되면 제거 (캐릭터 구분자로 오인)
    if '|' in result:
        result = result.replace('|', ',')

    return result
