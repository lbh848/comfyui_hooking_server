"""
복장정리프롬프트 - 하가나이 캐릭터 특화

이 파일은 customprompt/ 폴더에 넣고 설정에서 선택하면
"LLM 복장정리 실행" 버튼으로 실행할 수 있습니다.

필수 함수:
    async def run(character_name: str, outfit_list: list, chat_list: list = [], previous_result: str = None) -> str

인자:
    character_name: 캐릭터 이름
    outfit_list: [{"outfit_prompt": "...", "positive_prompt": "..."}, ...] (최대 10개)
    chat_list: ["채팅내용1", "채팅내용2", ...] (해당 캐릭터의 채팅 내역, 빈 값은 제외됨)
    previous_result: 이전 LLM 복장 통합 결과 (초기 None, 이후 마지막 결과 전달)

반환값:
    str: LLM이 정리한 복장 통합 결과 텍스트 (JSON)

---

GPT-4.1 요청 형식:
{
    "model": "gpt-4.1",
    "stream": false,
    "temperature": 1,
    "response_format": {"type": "json_object"},
    "messages": [
        {"role": "system", "content": "### TASK: ..."},
        {"role": "user", "content": "..."}
    ]
}

Gemini-3 계열 모델 요청 형식:
{
    "model": "gemini-3-flash-preview",
    "messages": [
        {"content": "### AI Role & System Rules...", "role": "user"},
        {"content": "Analyze...", "role": "user"}
    ],
    "max_tokens": 25000,
    "stream": false,
    "temperature": 1
}

주의:
- callLLM이 service/model 설정을 자동으로 처리합니다. (재시도 없음, 단일 시도)
"""

import re
import json
import os
import datetime
from modes.llm_service import callLLM, callLLM2, get_config


# ─── Prompt Logger ────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
PROMPT_LOG_FILE = os.path.join(LOG_DIR, "prompt_io.log")


def _log_prompt_io(character_name: str, input_data: dict, output_data: str):
    """LLM 입력/출력을 로그 파일에 기록"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "character": character_name,
            "input": input_data,
            "output": output_data[:2000],  # 출력 최대 2000자
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[PROMPT_LOGGER] 로그 기록 실패: {e}")


# ─── Output Format ─────────────────────────────────────────

OUTPUT_KEYS = ["headgear", "clothing", "shoes", "worn_accessories", "held_items", "memo"]


def _validate_output(parsed: dict) -> tuple:
    """Validate parsed output has required keys. Returns (is_valid, missing_keys)."""
    missing = [k for k in OUTPUT_KEYS if k not in parsed]
    return len(missing) == 0, missing


def _try_recover_json(text: str) -> dict:
    """Attempt to recover JSON from malformed LLM output."""
    # 1. Markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. First { ... } block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 3. Key: value extraction (loose format)
    result = {}
    for key in OUTPUT_KEYS:
        pattern = rf'["\']?{key}["\']?\s*[:=]\s*["\']?(.*?)["\']?(?:,|\n|$)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip().rstrip(',"\'')

    return result if len(result) >= 3 else None


def _fill_defaults(parsed: dict) -> dict:
    """Fill missing output keys with defaults."""
    defaults = {
        "headgear": "None",
        "clothing": "Unknown",
        "shoes": "Unknown",
        "worn_accessories": "None",
        "held_items": "None",
        "memo": ""
    }
    for key, default in defaults.items():
        if key not in parsed or not parsed[key]:
            parsed[key] = default
    return parsed


# ─── SLOT Parsing ─────────────────────────────────────────

def _parse_slot(chat_content: str) -> tuple:
    """
    [SLOT] 섹션을 chat_content에서 파싱.
    [SLOT] 아래에 || 로 구분된 텍스트가 있음:
      || 앞 = 그림 삽입 위치의 앞문장
      || 뒤 = 그림 삽입 위치의 뒷문장

    Returns:
        (chat_without_slot, text_before_insertion, text_after_insertion)
    """
    if not chat_content:
        return chat_content, "", ""

    slot_match = re.search(
        r'\[SLOT\]\s*(.*?)(?=\n\s*\[|\n\s*\n|\Z)',
        chat_content, re.DOTALL | re.IGNORECASE
    )
    if not slot_match:
        return chat_content, "", ""

    slot_content = slot_match.group(1).strip()

    if '||' in slot_content:
        parts = slot_content.split('||', 1)
        before = parts[0].strip()
        after = parts[1].strip()
    else:
        before = slot_content.strip()
        after = ""

    chat_clean = chat_content[:slot_match.start()] + chat_content[slot_match.end():]
    chat_clean = re.sub(r'\n{3,}', '\n\n', chat_clean).strip()

    return chat_clean, before, after


def _extract_insertion_context(chat_text: str, slot_before: str, slot_after: str,
                               max_chars: int = 3000) -> str:
    """
    SLOT 앵커를 기준으로 chat_text에서 그림 삽입 위치 주변 컨텍스트 추출.
    """
    if not chat_text:
        return ""
    if not slot_before and not slot_after:
        # SLOT이 비어있으면 chat의 마지막 부분 반환
        return chat_text[-max_chars:] if len(chat_text) > max_chars else chat_text

    anchor = slot_before or slot_after

    # 정확 매칭 시도
    pos = chat_text.find(anchor)

    # 앵커의 첫 문장으로 재시도
    if pos == -1:
        first_phrase = anchor.split('.')[0].strip()
        if len(first_phrase) > 10:
            pos = chat_text.find(first_phrase)

    # after_text로 재시도
    if pos == -1 and slot_after and anchor != slot_after:
        pos = chat_text.find(slot_after)
        if pos == -1:
            first_phrase = slot_after.split('.')[0].strip()
            if len(first_phrase) > 10:
                pos = chat_text.find(first_phrase)

    # 매칭 실패 → 마지막 부분
    if pos == -1:
        return chat_text[-max_chars:] if len(chat_text) > max_chars else chat_text

    # 삽입 위치 중심으로 추출
    match_center = pos + len(anchor) // 2
    half = max_chars // 2
    start = max(0, match_center - half)
    end = min(len(chat_text), start + max_chars)
    if end == len(chat_text):
        start = max(0, end - max_chars)

    return chat_text[start:end]


# ─── Chat Cleaning ─────────────────────────────────────────

def clean_chat(chat: str) -> str:
    """
    채팅 내역에서 불필요한 태그/블록을 제거하여 정리

    제거 항목:
    - <!--[asmd]-->...<!--[/asmd]--> 블록 (이미지 태그)
    - <lb-xnai ...>...</lb-xnai> 블록 (인레이 태그)
    - @Hidden Spoiler@...@END@ 블록 (스포일러)
    """
    result = re.sub(r'<!--\[asmd\]-->.*?<!--\[/asmd\]-->', '', chat, flags=re.DOTALL)
    result = re.sub(r'<lb-xnai[^>]*>.*?</lb-xnai>', '', result, flags=re.DOTALL)
    result = re.sub(r'@Hidden Spoiler@.*?@END@', '', result, flags=re.DOTALL)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


# ─── Deduplication ─────────────────────────────────────────

def _dedupe(items: list) -> list:
    """Remove duplicates while preserving order (case-insensitive)."""
    seen = set()
    result = []
    for item in items:
        stripped = item.strip()
        key = stripped.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(stripped)
    return result


# ─── Character Identity ─────────────────────────────────────────

CHAR_IDENTITIES = {
    "hasegawa kobato": (
        "Gothic lolita fashion, lolita hairband, black frilled dress, gothic lolita, "
        "long sleeves, juliet sleeves, puffy sleeves, wide sleeves, frilled sleeves, "
        "black ribbon, frills, frilled dress, bow, frilled hairband, two side up hairstyle, "
        "heterochromia (blue right, red left), blonde hair, long hair, flat chest, loli"
    ),
    "kashiwazaki pegasus": (
        "Blue kimono, wide sleeves, long sleeves, japanese clothes, "
        "hair pulled back, ponytail, black hair"
    ),
    "kashiwazaki sena": (
        "St. chronica academy school uniform, green jacket, white shirt, plaid pleated green skirt, "
        "ascot, butterfly hair ornament, long sleeves, school uniform, "
        "black thighhighs, zettai ryouiki, blonde long hair, aqua eyes, large breasts"
    ),
    "shiguma rika": (
        "Lab coat over St. chronica academy school uniform, green plaid pleated skirt, "
        "white shirt, vest, black necktie, collared shirt, hair scrunchie, miniskirt, "
        "brown ponytail, round glasses, brown hair"
    ),
    "kusunoki yukimura": (
        "Maid dress outfit, black frilled dress, maid headdress, white frilled apron, "
        "red bowtie, juliet sleeves, puffy sleeves, hair flower, red bow, "
        "white thighhighs, garter straps, bowtie, short brown hair"
    ),
    "mikazuki yozora": (
        "St. chronica academy school uniform, green jacket, white shirt, black ascot, "
        "green plaid pleated skirt, black thighhighs, zettai ryouiki, hair ribbon, braid, "
        "white gloves, long sleeves, collared shirt, very long black hair, purple eyes"
    ),
    "redfield stella": (
        "Tuxedo formal wear, black jacket (open), white vest, white dress shirt, "
        "black bowtie (traditional), black pants, white gloves, hair ornament (black bow), "
        "long sleeves, collared shirt, blonde short hair"
    ),
    "takayama kate": (
        "Nun outfit, black dress, habit (wimple), cross necklace, frilled sleeves, wide sleeves, "
        "heart hair ornament on wimple, long sleeves, holding cross, frills, thighhighs, "
        "grey-white long hair, large breasts"
    ),
    "takayama maria": (
        "Nun outfit, black dress, habit (wimple), cross necklace, frills, frilled sleeves, "
        "wide sleeves, heart hair ornament on wimple, long sleeves, "
        "grey-white long hair, flat chest, loli, small girl"
    ),
    "oreki houtarou": (
        "School uniform (gakuran), or casual: black jacket/hoodie (open), white shirt, "
        "black pants, black belt, necklace, collared shirt, hooded jacket, "
        "short brown hair, energy-saving personality"
    ),
    "yusa aoi": (
        "School uniform, brown cardigan/sweater, white collared shirt, "
        "green plaid pleated skirt, black necktie, pantyhose, ascot, miniskirt, "
        "red hair, short hair, red eyes"
    ),
    "hidaka hinata": (
        "St. chronica academy school uniform, green jacket, white shirt, "
        "black necktie, green plaid pleated skirt, hair bow, collared shirt, long sleeves, "
        "brown hair, hair bun, braid"
    ),
    "jinguuji karin": (
        "St. chronica academy school uniform, green jacket (blazer), ascot, "
        "green plaid skirt, hair ornament, collared shirt, long sleeves, "
        "black long hair"
    ),
    "ohtomo akane": (
        "School uniform, green jacket, ascot, green plaid pleated skirt, "
        "white shirt, collared shirt, long sleeves, black hair, short hair, ahoge"
    ),
    "hasegawa kodaka": (
        "White collared shirt, green plaid pants, black belt, necklace, sleeves rolled up, "
        "blonde short hair"
    ),
}


def _get_char_identity(name: str) -> str:
    return CHAR_IDENTITIES.get(name.lower(), "No specific style preference")


# ─── Prompt Building ───────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "### TASK\n"
        "Determine the character's CURRENT outfit by comparing the previous outfit state with the current situation.\n"
        "If the outfit has changed, predict the new outfit using outfit_prompt_history and positive_prompt_history.\n"
        "If not, carry forward the previous_result outfit.\n"
        "\n"
        "### INPUT DEFINITIONS\n"
        "- **previous_result**: the last known outfit state. WARNING: this may be from a much earlier tracking point — do NOT assume it corresponds to the most recent chat.\n"
        "- **outfit_prompt_history**: VLM model predictions of the character's appearance. [1]=newest.\n"
        "- **positive_prompt_history**: image generation prompts (1:1 match with outfit_prompt_history). Contains precise outfit descriptions.\n"
        "- **current_chat**: the current situation the character is experiencing.\n"
        "- **image_insertion_point** (if provided): shows exactly WHERE in the story the image is being generated.\n"
        "  This is CRITICAL — the outfit you determine must match what the character is wearing AT THIS EXACT POINT.\n"
        "  Read the text around the insertion point to determine whether the character has already changed\n"
        "  or is about to change clothes.\n"
        "\n"
        "### OUTFIT CHANGE DETECTION\n"
        "Read the current_chat carefully and use common sense to determine whether the character's outfit\n"
        "has changed since the previous_result. Do NOT look for specific keywords or phrases.\n"
        "Instead, understand the SITUATION described in the chat and judge logically.\n"
        "\n"
        "If **image_insertion_point** is provided, check it FIRST:\n"
        "- Is the insertion point AFTER a clothing change has occurred in the story?\n"
        "  → The character is already wearing the new outfit at this point.\n"
        "- Is the insertion point BEFORE the clothing change happens?\n"
        "  → The character has NOT changed yet — use the previous_result outfit.\n"
        "- No clothing change described anywhere near the insertion point?\n"
        "  → Use common sense based on the overall situation.\n"
        "\n"
        "General reasoning:\n"
        "- Does the chat describe the character actively changing clothes, or arriving at a place where\n"
        "  they would have already changed?\n"
        "- Given the location, activity, time, weather, and event, would the\n"
        "  previous_result outfit be inappropriate or physically implausible?\n"
        "- Was there enough TIME and OPPORTUNITY for the character to change? If rushed or forced to leave suddenly,\n"
        "  they likely did NOT change even if the new location would normally warrant different clothes.\n"
        "- Is the same location/activity continuing with no reason to change?\n"
        "\n"
        "Key principle: CONTEXT determines the answer, not specific words or phrases.\n"
        "Two chats using completely different words can describe the same situation.\n"
        "Always understand the FULL situation before deciding.\n"
        "\n"
        "### RULES\n"
        "1. **Cross-reference ALWAYS**: positive_prompt_history is the GROUND TRUTH of what the character was intended to wear.\n"
        "   - Items in BOTH outfit_prompt_history AND positive_prompt_history → HIGH CONFIDENCE, include.\n"
        "   - Items ONLY in positive_prompt_history (not seen by VLM) → include (may be obscured or implied).\n"
        "   - Items ONLY in outfit_prompt_history (VLM) but NOT in positive_prompt_history → LOW CONFIDENCE.\n"
        "     Include ONLY if they appear consistently across multiple outfit entries. If only in one VLM entry,\n"
        "     treat as potential hallucination and use the value from positive_prompt_history or 'Unknown'/'None'.\n"
        "2. **Completeness**: Include ALL visible items — hats, hair accessories, socks, stockings, ties, ribbons, gloves, etc.\n"
        "3. **Categorization**:\n"
        "   - headgear: hats, hair clips, ribbons, headbands, hair ornaments, wimples, glasses\n"
        "   - clothing: main outfit — shirt, skirt/pants, dress, jacket, tie, stockings/socks (if part of outfit)\n"
        "   - shoes: footwear\n"
        "   - worn_accessories: items ATTACHED to the body or clothing — earrings, necklaces, chokers, belts, scarves, wristbands, rings. \"None\" if absent\n"
        "   - held_items: items HELD in hands or carried — parasols, umbrellas, bags, purses, books, phones, weapons. \"None\" if absent\n"
        "4. **No speculation**: Do not invent items not in the data. Use \"Unknown\" or \"None\" for missing items.\n"
        "5. **Tags**: English, comma-separated Danbooru-style tags.\n"
        "6. **Evidence**: memo MUST state your reasoning — changed or not, which evidence, quote relevant chat lines.\n"
        "\n"
        "### OUTPUT FORMAT\n"
        "JSON only, no other text:\n"
        '{"headgear":"","clothing":"","shoes":"","worn_accessories":"","held_items":"","memo":""}\n'
        "\n"
        "→JSON ONLY"
    )


def _build_char_intro(character_name: str, char_identity: str) -> str:
    """첫 번째 user 메시지: 캐릭터 소개"""
    return (
        f"### Introduce character\n"
        f"[1] Name: {character_name}\n"
        f"[2] Identity: {char_identity}"
    )


def _build_data_prompt(outfits_text: str, prompts_text: str,
                       chat_context: str, insertion_context: str = "",
                       previous_result: str = None) -> str:
    """두 번째 user 메시지: 분석 데이터"""
    parts = []
    if previous_result:
        parts.append(f"<previous_result>\n{previous_result}\n</previous_result>")
    if outfits_text:
        parts.append(f"<outfit_prompt_history>\n{outfits_text}\n</outfit_prompt_history>")
    if prompts_text:
        parts.append(f"<positive_prompt_history>\n{prompts_text}\n</positive_prompt_history>")
    if chat_context:
        parts.append(f"<current_chat>\n{chat_context}\n</current_chat>")
    if insertion_context:
        parts.append(
            "<image_insertion_point>\n"
            "THIS IS WHERE THE IMAGE IS BEING GENERATED in the story.\n"
            "The image depicts the character at this exact point. "
            "Use this to determine whether the character has already changed clothes "
            "or is still in the same outfit at this specific moment.\n"
            f"{insertion_context}\n"
            "</image_insertion_point>"
        )
    return "\n\n".join(parts)


def _extract_upscale_content(prompt: str) -> str:
    """positive_prompt에서 [UPSCALE] 뒤의 내용만 추출"""
    match = re.search(r'\[UPSCALE\]\s*(.*)', prompt, re.DOTALL)
    if match:
        return match.group(1).strip().rstrip(',').strip()
    return prompt.strip()


# ─── Message Conversion (GPT ↔ Gemini) ────────────────────

def gpt2gemini(messages: list) -> list:
    """GPT 형식 messages → Gemini 형식으로 변환.
    Gemini는 system role을 지원하지 않으므로 system 메시지를 첫 user 메시지에 병합.
    """
    result = []
    system_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            result.append({"role": role, "content": content})

    # system 내용이 있으면 첫 user 메시지 앞에 병합
    if system_parts and result:
        merged_system = "\n\n".join(system_parts)
        first_user = result[0]
        result[0] = {
            "role": "user",
            "content": f"[System Instructions]\n{merged_system}\n\n[User Request]\n{first_user['content']}",
        }
    elif system_parts:
        # user 메시지가 없으면 system만 user로
        result.insert(0, {"role": "user", "content": "\n\n".join(system_parts)})

    return result


def gemini2gpt(messages: list) -> list:
    """Gemini 형식 messages → GPT 형식으로 변환.
    user 메시지 내 [System Instructions] 블록을 system role로 분리.
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")

        # [System Instructions] ... [User Request] ... 패턴 분리
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
    """모델명이 Gemini 계열인지 판별"""
    if not model_name:
        return False
    return any(kw in model_name.lower() for kw in ("gemini", "gemma"))


# ─── Main Entry Point ─────────────────────────────────────

async def run(character_name: str, outfit_list: list, chat_list: list = [],
              previous_result: str = None) -> str:
    """
    캐릭터별 복장 정보를 LLM으로 정리

    Args:
        character_name: 캐릭터 이름
        outfit_list: [{"outfit_prompt": "...", "positive_prompt": "..."}, ...]
        chat_list: ["채팅내용1", "채팅내용2", ...]
        previous_result: 이전 LLM 복장 통합 결과 (None이면 첫 실행)

    Returns:
        JSON 형식의 정리된 복장 통합 결과
    """
    # 네임드가 아닌 캐릭터는 건너뛰기
    if character_name == "unknown":
        return "LLM 작업 건너뜀"

    # 캐릭터 스타일 정보
    char_identity = _get_char_identity(character_name)

    # ─── outfit_prompt 분리: 최근 3개, 중복 제거 ───
    # outfit_list는 시간순: [0]=가장 오래됨, [-1]=가장 최신
    # reversed()로 최신순 정렬 → [0]=가장 최신, 뒤로 갈수록 오래됨
    raw_outfits = [entry.get("outfit_prompt", "") for entry in outfit_list]
    outfit_prompts = _dedupe(list(reversed(raw_outfits)))[:3]
    outfits_text = "\n".join(
        f"Outfit[{i}]: {op}" for i, op in enumerate(outfit_prompts, 1)
    )

    # ─── positive_prompt 분리: 최근 3개, [UPSCALE] 이후만 추출, 중복 제거 ───
    # outfit_list와 동일 순서이므로 reversed로 최신순 정렬
    raw_positive = [entry.get("positive_prompt", "") for entry in outfit_list]
    positive_prompts = _dedupe([_extract_upscale_content(pp) for pp in reversed(raw_positive)])[:3]
    prompts_text = "\n".join(
        f"Prompt[{i}]: {pp}" for i, pp in enumerate(positive_prompts, 1)
    )

    # ─── chat_list 분리: 최근 1개, 정리 후 중복 제거 ───
    # chat_list도 시간순: [0]=가장 오래됨, [-1]=가장 최신
    cleaned_chats = [clean_chat(c) for c in chat_list if c and c.strip()]
    chat_deduped = _dedupe(list(reversed(cleaned_chats)))[:1]

    # ─── SLOT 파싱: 그림 삽입 위치 추출 ───
    insertion_context = ""
    if chat_deduped:
        raw_chat = chat_deduped[0]
        chat_without_slot, slot_before, slot_after = _parse_slot(raw_chat)
        if slot_before or slot_after:
            insertion_context = _extract_insertion_context(
                chat_without_slot, slot_before, slot_after
            )
        chat_context = f"Chat[1]: {chat_without_slot if (slot_before or slot_after) else raw_chat}"
    else:
        chat_context = ""

    # 프롬프트 조합
    system_prompt = _build_system_prompt()
    char_intro = _build_char_intro(character_name, char_identity)
    data_prompt = _build_data_prompt(
        outfits_text, prompts_text, chat_context,
        insertion_context=insertion_context,
        previous_result=previous_result
    )

    # ─── 입력 로그 ───
    log_input = {
        "outfit_prompts": outfit_prompts,
        "positive_prompts": positive_prompts,
        "chat": chat_deduped,
        "previous_result": (previous_result[:300] if previous_result else None),
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": char_intro},
        {"role": "user", "content": data_prompt},
    ]

    # LLM1 호출
    config = get_config()
    result = await callLLM(messages)

    # LLM1 실패 시 LLM2 폴백
    if result.startswith("[LLM 실패]"):
        llm_model2 = config.get("llm_model2", "")
        if llm_model2:
            # LLM2가 Gemini 계열이면 messages 변환
            if _is_gemini_model(llm_model2):
                messages2 = gpt2gemini(messages)
            else:
                messages2 = messages
            result = await callLLM2(messages2)

    # ─── 출력 로그 ───
    _log_prompt_io(character_name, log_input, result)

    # 여전히 실패면 에러 메시지 그대로 반환
    if result.startswith("[LLM 실패]"):
        return result

    # 출력 형식 검증
    # 1차: 직접 JSON 파싱
    try:
        parsed = json.loads(result)
        is_valid, missing = _validate_output(parsed)
        if is_valid:
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        # JSON은 맞지만 키가 누락된 경우 - 기본값 채우기
        parsed = _fill_defaults(parsed)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # 2차: 하드 코딩 복구 시도
    recovered = _try_recover_json(result)
    if recovered:
        recovered = _fill_defaults(recovered)
        is_valid, _ = _validate_output(recovered)
        if is_valid:
            return json.dumps(recovered, indent=2, ensure_ascii=False)

    # 복구 불가 - 원본 텍스트 반환
    return result


