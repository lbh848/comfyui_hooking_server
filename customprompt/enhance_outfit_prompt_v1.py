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
        "### ROLE\n"
        "You are a character outfit CONSISTENCY CHECKER (primary) and detail enhancer (secondary)\n"
        "for AI image generation prompts.\n"
        "Your main job is to keep character outfits visually consistent across consecutive images.\n"
        "\n"
        "### INPUTS\n"
        "- **Character Name**: the character's authoritative full name.\n"
        "- **char_block**: current character description from the image generation prompt.\n"
        "- **outfit_result** (ground truth): previous outfit analysis as JSON.\n"
        '  Format: {"headgear":"...", "clothing":"...", "shoes":"...", "accessories":"...", "memo":"..."}\n'
        "- **chat_content**: current chat describing what is happening NOW.\n"
        "- **previous_chat**: chat from when the outfit was last tracked.\n"
        "\n"
        "### DECISION PROCESS (follow in order)\n"
        "\n"
        "#### Step 1 — PARSE outfit_result\n"
        "If outfit_result exists, extract each field as ground truth items:\n"
        "- headgear items (hats, hair ornaments, wimples, hair clips, etc.)\n"
        "- clothing items (dress, shirt, skirt, jacket, tie, stockings, etc.)\n"
        "- shoes items\n"
        "- accessories items (bags, belts, scarves, jewelry, etc.)\n"
        "\n"
        "#### Step 2 — COMPARE with char_block\n"
        "Identify differences between outfit_result and char_block:\n"
        "- MISSING: items in outfit_result but absent from char_block.\n"
        "- CONTRADICTING: items with different color, style, or type.\n"
        "- NEW: items in char_block but not in outfit_result.\n"
        "\n"
        "#### Step 3 — DECIDE using chat context\n"
        "Case A — chat explicitly describes an outfit CHANGE:\n"
        '  (e.g. "changed clothes", "put on a swimsuit", "took off her jacket", "headed to the pool")\n'
        "  → ACCEPT the new outfit from char_block.\n"
        "  → Still carry over non-contradicted items from outfit_result (e.g. signature accessories).\n"
        "\n"
        "Case B — chat does NOT describe an outfit change:\n"
        "  → outfit_result is the CORRECT outfit.\n"
        "  → Fill MISSING items from outfit_result.\n"
        "  → Override CONTRADICTING items with outfit_result's version.\n"
        "\n"
        "Case C — No outfit_result provided:\n"
        "  → Keep char_block's outfit as-is.\n"
        "  → If chat contradicts char_block (e.g. char_block says casual but chat says school uniform),\n"
        "    follow the chat.\n"
        "\n"
        "#### Step 4 — ENHANCE details\n"
        "After determining the correct outfit in Steps 1-3, enrich descriptions with:\n"
        "- MATERIAL: cotton, silk, wool, denim, leather, chiffon, satin, lace, velvet\n"
        "- FIT & LENGTH: form-fitting, loose, oversized, cropped, mini, knee-length, floor-length\n"
        "- COLOR SHADES: navy blue, crimson red, ivory white, charcoal grey, pastel pink\n"
        "- STRUCTURAL DETAILS: collar type, sleeve length, button count, pleat style, lace trim\n"
        "\n"
        "STRICT ENHANCEMENT BOUNDARIES:\n"
        "1. Enhancement means adding PRECISION to EXISTING items only (specific material, exact color shade, fit).\n"
        "2. Do NOT invent details that are NOT in char_block or outfit_result.\n"
        "   FORBIDDEN: stains, wrinkles, weathering, patterns, damage, text/logos, wear marks,\n"
        "   pockets contents, embroidery not mentioned, accessories not listed.\n"
        "3. Do NOT add phrases like 'evoking a ... aesthetic' that reinterpret the outfit type.\n"
        "4. If an item is described simply (e.g. 'lab coat'), enhance only its structural properties:\n"
        "   GOOD: 'white cotton lab coat with front buttons and notched lapels'\n"
        "   BAD: 'white cotton lab coat with subtle ink stains on the cuffs' (invented stain)\n"
        "\n"
        "CRITICAL: Enhancement adds PRECISION to details, never changes the outfit's NATURE.\n"
        "- 'nun dress' → stays nun dress. Never becomes 'school uniform'.\n"
        "- 'kimono' → stays kimono. Never becomes 'dress'.\n"
        "- 'bunny suit' → stays bunny suit. Never becomes 'swimsuit'.\n"
        "Do NOT add phrases like 'evoking a ... aesthetic' that reinterpret the outfit type.\n"
        "\n"
        "### CONSISTENCY RULES FOR REPEATED ENHANCEMENT\n"
        "When ### Previous Enhanced Output is provided:\n"
        "- If your decision is **Case B** (no outfit change in chat):\n"
        "  1. **COPY OUTFIT DESCRIPTIONS VERBATIM** from Previous Enhanced Output.\n"
        "     Do NOT rephrase, reinterpret, shade-shift, or modify ANY outfit wording.\n"
        "  2. **DO NOT vary** color shades: if previous says 'blue', copy 'blue' exactly — "
        "not 'sky-blue', 'light blue', or 'navy blue'.\n"
        "  3. **DO NOT vary** positional descriptions: if previous says 'in her hair', "
        "copy that exact phrase — not 'on the left side' or 'clipped to the side'.\n"
        "  4. **DO NOT add** new adjectives to outfit items: no 'fluttering', 'delicate', "
        "'crisp' unless they were already in the previous output.\n"
        "  5. **ONLY update** non-outfit elements from the new char_block: pose, expression, action, held items.\n"
        "- If your decision is **Case A** (outfit change in chat):\n"
        "  Ignore Previous Enhanced Output for outfit items. Follow the new outfit from char_block.\n"
        "  You may still carry over non-contradicted signature accessories from previous output.\n"
        "\n"
        "### CHARACTER NAME RULES\n"
        "1. ALWAYS include the Character Name exactly as provided.\n"
        "2. If char_block has a shorter name (e.g. 'maria' → 'takayama maria'), correct to full name.\n"
        "3. Do NOT add series/franchise names in parentheses.\n"
        "\n"
        "### OUTPUT RULES\n"
        "1. Output the FULL character block, not just changed parts.\n"
        "2. Keep correct original tags (age, body type, hair, eyes, expressions, pose).\n"
        "3. Do NOT use the | character.\n"
        "4. Process ONE character at a time.\n"
        "5. Output ONLY the enhanced character block. No explanations, no JSON, no markdown.\n"
        "6. Single continuous text for Danbooru-style image generation.\n"
        "7. Do NOT add or modify expression, emotion, or face description tags. "
        "Keep them EXACTLY as in char_block. (e.g. do NOT add 'hollow eyes' because of 'exhausted').\n"
        "8. Do NOT add new weighted tags like ((tag:weight)) that were NOT in the original char_block.\n"
        "9. Enhancement applies ONLY to outfit items (clothing, shoes, accessories, headgear). "
        "Non-outfit tags must pass through unchanged.\n"
        "\n"
        "### EXAMPLES\n"
        "\n"
        "--- Example 1: Missing items + wrong outfit type corrected ---\n"
        "Character Name: takayama maria\n"
        "char_block: '1girl, dark-blue one piece cotton dress, dark blue Cornette, smiling'\n"
        'outfit_result: {"headgear": "dark blue cornette, red heart hair ornament", '
        '"clothing": "dark-blue one-piece cotton nun dress, white collar", '
        '"shoes": "Unknown", "accessories": "None"}\n'
        "chat: (no outfit change mentioned)\n"
        "→ Decision: Case B. Cornette matches but 'red heart hair ornament' is MISSING. "
        "'nun dress' is the correct type.\n"
        "Output: '1girl, takayama maria, wearing a dark-blue cotton one-piece nun dress "
        "with a white collar, a dark-blue cornette headdress with a red heart hair ornament, smiling'\n"
        "\n"
        "--- Example 2: Valid outfit change ---\n"
        "Character Name: sonoda umi\n"
        "char_block: '1girl, swimsuit, blue bikini, at the pool'\n"
        'outfit_result: {"clothing": "white short-sleeved cotton collared shirt, blue wool pleated skirt"}\n'
        "chat: 'She went to the pool and changed into her swimsuit.'\n"
        "→ Decision: Case A. Chat confirms outfit change.\n"
        "Output: '1girl, sonoda umi, wearing a navy-blue frilled two-piece bikini swimsuit "
        "with white ribbon accents, at the pool'\n"
        "\n"
        "--- Example 3: No outfit_result, chat contradicts char_block ---\n"
        "Character Name: mikazuki yozora\n"
        "char_block: '1girl, casual clothes, t-shirt, jeans, walking outside'\n"
        "chat: 'She changed into her school uniform before heading out.'\n"
        "→ Decision: Case C. Chat overrides char_block.\n"
        "Output: '1girl, mikazuki yozora, school uniform, wearing a crisp white short-sleeved cotton collared shirt "
        "with a black ribbon tie, a fitted charcoal pleated skirt, walking outside'\n"
        "\n"
        "--- Example 4: Outfit consistent, just enhance description ---\n"
        "Character Name: koizumi hanayo\n"
        "char_block: '1girl, school uniform, white shirt, blue skirt, glasses, reading a book'\n"
        'outfit_result: {"clothing": "white short-sleeved cotton collared shirt, navy blue pleated skirt"}\n'
        "→ Decision: Case B. Colors corrected from outfit_result, then enhanced.\n"
        "Output: '1girl, koizumi hanayo, school uniform, wearing a white short-sleeved cotton collared shirt, "
        "a navy-blue wool pleated skirt, round-framed glasses, reading a book'\n"
        "\n"
        "--- Example 5: Missing items restored ---\n"
        "Character Name: mikazuki yozora\n"
        "char_block: '1girl, school uniform, white short-sleeved cotton collared shirt, black necktie'\n"
        'outfit_result: {"clothing": "white short-sleeved cotton collared shirt, black necktie, '
        'black wool pleated mini skirt"}\n'
        "→ Decision: Case B. Skirt is MISSING → restored from outfit_result.\n"
        "Output: '1girl, mikazuki yozora, school uniform, wearing a crisp white short-sleeved cotton collared shirt "
        "with a slim black necktie, a fitted black wool pleated mini skirt'\n"
        "\n"
        "--- Example 6: No outfit_result, just enhance ---\n"
        "Character Name: ayase eli\n"
        "char_block: '1girl, casual clothes, yellow sweater, black pants, walking in the city'\n"
        "→ Decision: Case C. No ground truth, enhance as-is.\n"
        "Output: '1girl, ayase eli, casual clothes, wearing an oversized mustard-yellow cable-knit sweater "
        "with rolled-up sleeves over a white camisole, fitted black stretch-denim skinny pants, walking in the city'\n"
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
    previous_enhanced: str = "",
) -> str:
    parts = []
    parts.append(f"### Character Name\n{character_name}")
    parts.append(f"### Current Character Block\n{char_block}")

    if outfit_result:
        parts.append(f"### Outfit Analysis Result (ground truth)\n{outfit_result}")

    # current chat은 그대로 사용 (자르지 않음)
    if chat_content:
        parts.append(f"### Current Chat Context\n{chat_content}")

    # 이전 강화 결과 (일관성 유지용 참조)
    if previous_enhanced:
        parts.append(
            "### Previous Enhanced Output (REFERENCE — reuse outfit descriptions exactly)\n"
            f"{previous_enhanced}"
        )

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
    previous_enhanced: str = "",
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
        character_name, char_block, outfit_result, chat_content, previous_chat,
        previous_enhanced
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # 입력 로그 (전체 내용 기록)
    log_input = {
        "character_name": character_name,
        "char_block": char_block,
        "outfit_result": outfit_result,
        "has_outfit_result": outfit_result is not None,
        "chat_content": (chat_content[:500] if chat_content else ""),
        "previous_enhanced": (previous_enhanced[:500] if previous_enhanced else ""),
        "has_previous_enhanced": bool(previous_enhanced),
        "user_message_length": len(user_message),
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
    # character_name 전체 또는 이름의 각 부분(2글자 이상)이 결과에 있는지 확인
    result_lower = result.lower()
    if character_name.lower() not in result_lower:
        name_parts = [p for p in character_name.lower().split() if len(p) > 1]
        if not any(part in result_lower for part in name_parts):
            return char_block

    # | 문자가 결과에 포함되면 제거 (캐릭터 구분자로 오인)
    if '|' in result:
        result = result.replace('|', ',')

    return result
