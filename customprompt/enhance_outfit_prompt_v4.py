"""
복장 프롬프트 강화 - v4
char, setup, supplement 각각 강화 후 구조화된 출력 반환

prompt_enhance_mode.py에서 파싱된 데이터를 전달받아
LLM으로 char, setup, supplement를 각각 강화합니다.

필수 함수:
    async def run(
        character_name: str,
        char: str,
        previous_enhanced: str = "",
        setup: str = "",
        supplement: str = "",
        chat: str = "",
        slot: str = "",
    ) -> dict

반환값:
    dict: {
        "char": "강화된 캐릭터 + 표정 태그 (최종 프롬프트용)",
        "outfit_only": "복장만 (일관성 추적용)",
        "setup": "강화된 setup 태그",
        "supplement": "강화된 supplement 태그",
    }

주의:
- callLLM이 service/model 설정을 자동으로 처리합니다. (재시도 없음, 단일 시도)
- prompt_enhance_mode.py에서 파싱된 데이터를 직접 전달받습니다 (내부 파싱 없음)
- 복장은 일관성 보장을 위해 다음 라운드로 전달됨
- 표정은 매번 현재 맥락 기반으로 새로 생성됨 (전달 안됨)
"""

import re
import json
import os
import datetime
import tiktoken
from modes.llm_service import callLLM, callLLM2, get_config


# ─── Storage ────────────────────────────────────────────
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "enhance_outfit_prompt_v4_storage")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
PROMPT_LOG_FILE = os.path.join(LOG_DIR, "enhance_prompt_io.log")
MAX_HISTORY = 15


def _get_storage_path(character_name: str) -> str:
    safe_name = re.sub(r'[^\w\s-]', '', character_name.lower()).strip().replace(' ', '_')
    return os.path.join(STORAGE_DIR, f"{safe_name}.json")


def _load_history(character_name: str) -> list:
    path = _get_storage_path(character_name)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_history(character_name: str, history: list):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    history = history[-MAX_HISTORY:]
    path = _get_storage_path(character_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _add_history_entry(character_name: str, entry: dict):
    history = _load_history(character_name)
    history.append(entry)
    _save_history(character_name, history)


def _get_recent_chars(character_name: str, count: int = 3) -> list:
    history = _load_history(character_name)
    recent = history[-count:] if len(history) >= count else history
    return [entry.get("char", "") for entry in reversed(recent) if entry.get("char")]


# ─── Prompt Logger ──────────────────────────────────────

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


# ─── Token Counting ────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc.encode(text))


MAX_TOTAL_TOKENS = 14000


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    """뒤(최신)를 유지하며 앞에서부터 잘라 max_tokens 이하로 만듦."""
    if not text or _count_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if _count_tokens(text[mid:]) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


# ─── Message Conversion (GPT <-> Gemini) ────────────────

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


# ─── Context Extraction ─────────────────────────────────

def _extract_context_around(text: str, anchor_before: str, anchor_after: str,
                            max_tokens: int) -> str:
    """
    앵커 텍스트가 text에서 매칭되는 위치를 찾아
    그 중심으로 max_tokens 범위의 컨텍스트를 추출.
    매칭 실패 시 뒤(최신)부터 잘라서 반환.
    """
    if not text:
        return ""
    if not anchor_before and not anchor_after:
        return _trim_to_tokens(text, max_tokens)
    if _count_tokens(text) <= max_tokens:
        return text

    pos = -1
    anchor = anchor_before or anchor_after

    pos = text.find(anchor)

    if pos == -1:
        first_phrase = anchor.split('.')[0].strip()
        if len(first_phrase) > 10:
            pos = text.find(first_phrase)

    if pos == -1 and anchor_after and anchor != anchor_after:
        pos = text.find(anchor_after)
        if pos == -1:
            first_phrase = anchor_after.split('.')[0].strip()
            if len(first_phrase) > 10:
                pos = text.find(first_phrase)

    if pos == -1:
        return _trim_to_tokens(text, max_tokens)

    match_center = pos + len(anchor) // 2
    max_chars = int(max_tokens * 3.5)
    half = max_chars // 2

    start = max(0, match_center - half)
    end = min(len(text), start + max_chars)
    if end == len(text):
        start = max(0, end - max_chars)

    extracted = text[start:end]

    if _count_tokens(extracted) > max_tokens:
        extracted = _trim_to_tokens(extracted, max_tokens)

    return extracted


# ─── System Prompt ──────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "### ROLE\n"
        "A prompt REFINER for AI image generation.\n"
        "Enhance character, setup, and supplement tags for CONSISTENCY and CONTEXT.\n"
        "\n"
        "### CORE PRINCIPLE\n"
        "LESS IS MORE. Remove unnecessary details. Keep only essential visual elements.\n"
        "Shorter prompts = better image generation. Every tag must earn its place.\n"
        "Output should be SHORTER or similar length to the input — never longer.\n"
        "\n"
        "### INPUTS\n"
        "- **Character Name** (### Character Name): authoritative full name.\n"
        "- **Character** (### Character): current character tags.\n"
        "- **Setup** (### Setup): camera/framing/location/pose tags.\n"
        "  Use camera framing to determine visible range.\n"
        "- **Supplement** (### Supplement): extra tags after character in the prompt.\n"
        "- **Chat** (### Chat): current story describing what's happening.\n"
        "- **Image Insertion Point** (### Image Insertion Point): "
        "WHERE the image is generated.\n"
        "- **Previous Enhanced** (### Previous Enhanced): "
        "previous enhanced output for consistency.\n"
        "- **Previous Characters** (### Previous Characters): "
        "recent character descriptions.\n"
        "\n"
        "### CLEANUP RULES\n"
        "1. **Context filtering**: Remove items that don't fit the story context.\n"
        "   Example: 'robot' appearing randomly when no robots are in the scene.\n"
        "2. **Atmosphere removal**: Remove tags that could be misinterpreted "
        "as physical traits.\n"
        "   'fox-like', 'robot-like', 'cat-like' → REMOVE.\n"
        "   'elegant atmosphere', 'mysterious vibe' → REMOVE.\n"
        "3. **Filler removal**: Remove meaningless decorative words.\n"
        "   'beautiful', 'stunning', 'gorgeous', 'detailed', 'intricate' → REMOVE.\n"
        "4. **Framing-based visibility**:\n"
        "   Use Setup camera/framing (cowboy shot, close-up, from side, "
        "full body, etc.)\n"
        "   to determine which parts of the character are VISIBLE in the image.\n"
        "   Items OUTSIDE the visible frame → REMOVE from output.\n"
        "   - close-up / portrait / face focus → "
        "NO shoes, lower body clothing, held items below frame\n"
        "   - upper body / bust shot → NO shoes, skirt below frame\n"
        "   - cowboy shot (mid-thigh) → most visible, shoes may be at frame edge\n"
        "   - full body → all items visible\n"
        "   - from above / high angle → feet likely out of frame\n"
        "   Only exclude when framing CLEARLY makes the item invisible.\n"
        "\n"
        "### OUTFIT CONSISTENCY\n"
        "Compare current character with Previous Characters and Previous Enhanced.\n"
        "- If outfit UNCHANGED (no outfit change described in chat):\n"
        "  COPY outfit description from previous output VERBATIM.\n"
        "  Do NOT rephrase, shade-shift, or modify ANY outfit wording.\n"
        "  'blue' stays 'blue' — not 'sky-blue', 'navy', or 'light blue'.\n"
        "- If outfit CHANGED (chat describes changing clothes):\n"
        "  Describe new outfit SIMPLY. Same detail level as previous — "
        "not more detailed.\n"
        "  Don't add material/texture/fit details unless they were in the input.\n"
        "\n"
        "### EXPRESSION\n"
        "Determine expression from Chat and Image Insertion Point.\n"
        "- Expression is ALWAYS FRESH — based on current scene emotion only.\n"
        "- No consistency needed with previous expressions.\n"
        "- Use Danbooru-compatible expression tags.\n"
        "- Tags: smiling, frowning, crying, blushing, angry, surprised, nervous,\n"
        "  embarrassed, happy, sad, scared, determined, gentle smile, smirk,\n"
        "  open mouth, closed eyes, tears, sweat drop, pout, grin, wry smile,\n"
        "  half-closed eyes, wide eyes, narrowed eyes, gritting teeth, nose blush,\n"
        "  flushed face, teary-eyed, sparkling eyes, vacant stare, confused, dazed,\n"
        "  sleepy, yawning, biting lip, shy smile, bitter smile, forced smile,\n"
        "  flustered, exasperated, worried, relieved, nostalgic, melancholy,\n"
        "  expressionless, cat smile, looking away, looking down, looking up,\n"
        "  side glance, glaring, huffing, puffed cheeks, anguish, despair, awe,\n"
        "  shock, boredom, smug, contempt, pride, light smile, warm smile,\n"
        "  sad smile, talking, shouting, screaming, drooping eyes,\n"
        "  one eye closed, both eyes closed, eyebrows raised, eyebrows furrowed.\n"
        "\n"
        "### CHARACTER NAME RULES\n"
        "1. ALWAYS include the Character Name exactly as provided.\n"
        "2. If Character has a shorter name, correct to full name.\n"
        "3. Do NOT add series/franchise names.\n"
        "\n"
        "### POSITION AND MULTI-CHARACTER (CRITICAL RULE)\n"
        "When the Character input starts with a position/placement phrase,\n"
        "you MUST copy that ENTIRE opening phrase to the very start of Part 1,\n"
        "verbatim, without ANY modification.\n"
        "Position phrases include but are NOT limited to:\n"
        "  'the girl on the left is', 'the boy on the right is',\n"
        "  'the man on the left is', 'the woman on the right is',\n"
        "  'the older woman is', 'the person behind is',\n"
        "  'the girl in the center is', 'the boy in front is', etc.\n"
        "This applies regardless of gender (boy/girl/man/woman/person).\n"
        "Do NOT replace position phrases with count tags like '1girl' or '1boy'.\n"
        "Only include count tags if they were already in the input.\n"
        "For multi-character outfit sentences, use: "
        "'the girl/boy who is wearing...'\n"
        "matching the gender from the position phrase.\n"
        "\n"
        "### OTHER CHARACTER EXCLUSION\n"
        "Output must describe ONLY the target character (Character Name).\n"
        "References to other specific characters must NOT appear in the output.\n"
        "However, POSITION/PLACEMENT descriptors (e.g., 'the girl on the left is',\n"
        "'the boy on the right is', 'the person behind is') are NOT references to\n"
        "other characters — they describe WHERE the target character is placed.\n"
        "ALWAYS preserve these position descriptors exactly as they appear.\n"
        "\n"
        "### SETUP ENHANCEMENT\n"
        "Refine the Setup tags:\n"
        "- Keep: camera, framing, angle, location, lighting, weather, "
        "time-of-day, and non-clothing props (bags, umbrellas, phones, "
        "weapons, books, etc.).\n"
        "- REMOVE from Setup and move to Supplement: clothing items the "
        "character is holding but NOT wearing\n"
        "  (e.g. 'holding t-shirt', 'holding shorts', 'carrying dress').\n"
        "  The image model may interpret these as the character's OUTFIT "
        "and override the actual outfit tags.\n"
        "- KEEP: character count tags (1girl, 1boy, 2girls, 1boy 1girl, etc.).\n"
        "  Place them at the VERY START of Setup output, before camera/framing tags.\n"
        "- Keep setup SIMPLE — only scene-level tags.\n"
        "- Interaction tags: when Chat describes physical contact between "
        "characters,\n"
        "  add the appropriate Danbooru interaction tag to Setup output.\n"
        "  Place AFTER count tags, BEFORE camera/framing tags.\n"
        "  Tags: backhug, hugging, holding hands, arm around shoulder, "
        "carrying,\n"
        "  piggyback, leaning on, headpat, handshake, linking arms, "
        "kissing, couple.\n"
        "  Do NOT add interaction tags if no physical interaction is "
        "described in Chat.\n"
        "\n"
        "### SUPPLEMENT ENHANCEMENT\n"
        "Refine the Supplement into 1-2 SHORT natural language sentences.\n"
        "- Describe the scene, atmosphere, or key visual elements the character interacts with.\n"
        "- NEVER output 'none' or leave blank. Always provide at least one sentence.\n"
        "- Keep it concise: 1-2 sentences, under 30 words total.\n"
        "- Remove tags that conflict with the character or scene.\n"
        "- If the original Supplement is too long or narrative, condense to the essential visual.\n"
        "\n"
        "### OUTPUT FORMAT\n"
        "Output FIVE parts separated by [EXPR], [SETUP], [SUPPLEMENT], "
        "and [REASON]:\n"
        "\n"
        "Part 1 (before [EXPR]): character description in this EXACT order:\n"
        "  1. Position descriptor or count tag (FIRST)\n"
        "     - Multi-character: copy position phrase verbatim "
        "(e.g., 'the girl on the left is adolescent')\n"
        "     - Single character: count tag (e.g., '1girl', '1boy')\n"
        "  2. Outfit as a natural sentence (SECOND)\n"
        "     - Single character: 'wearing a [outfit description]'\n"
        "     - Multi-character: 'the girl/boy who is wearing a "
        "[outfit description]'\n"
        "     - Include accessories with 'with' "
        "(e.g., 'wearing a dress with a hairband')\n"
        "     - Multiple clothing items: use 'and' or commas within "
        "the sentence\n"
        "  3. Body attribute tags: hair, eyes, body type "
        "(Danbooru comma-separated tags)\n"
        "  4. Character Name (LAST, at the very end)\n"
        "  Do NOT include: pose, action, location, environment, "
        "background, lighting.\n"
        "Part 2 (after [EXPR], before [SETUP]): expression/emotion tags ONLY\n"
        "Part 3 (after [SETUP], before [SUPPLEMENT]): refined Setup tags\n"
        "Part 4 (after [SUPPLEMENT], before [REASON]): refined Supplement tags\n"
        "Part 5 (after [REASON]): brief one-line decision memo\n"
        "  Format: 'outfit: <unchanged/changed>. expression: <reason>. "
        "setup: <changes>. supplement: <changes>. framing: <what removed>.'\n"
        "\n"
        "Rules:\n"
        "1. Part 1 follows the 4-step structure above. "
        "Parts 2-4 use comma-separated tags.\n"
        "2. Do NOT use the | character.\n"
        "3. Process ONE character at a time.\n"
        "4. Do NOT add new weighted tags.\n"
        "5. Part 5 (memo) must be exactly ONE line.\n"
        "\n"
        "### EXAMPLES\n"
        "\n"
        "--- Example 1: Outfit unchanged, expression from context ---\n"
        "Character Name: hasegawa kobato\n"
        "Character: 1girl, kobato, gothic lolita dress, black hairband, "
        "black ribbon, mysterious atmosphere\n"
        "Setup: cowboy shot, classroom\n"
        "Supplement: night, moonlight\n"
        "Previous: 1girl, kobato, gothic lolita dress, black hairband, "
        "blonde hair, heterochromia\n"
        "Chat: Kobato was reading quietly in her room.\n"
        "Output:\n"
        "1girl, wearing a gothic lolita dress with a black hairband, "
        "blonde hair, heterochromia, hasegawa kobato\n"
        "[EXPR]\n"
        "calm, focused, light smile\n"
        "[SETUP]\n"
        "cowboy shot, classroom\n"
        "[SUPPLEMENT]\n"
        "A quiet room lit by soft moonlight through the window.\n"
        "[REASON]\n"
        "outfit: unchanged (from previous). expression: reading calmly. "
        "setup: unchanged. supplement: unchanged. framing: cowboy shot, all visible.\n"
        "\n"
        "--- Example 2: Outfit changed, location changed ---\n"
        "Character Name: kashiwazaki sena\n"
        "Character: 1girl, sena, school uniform, green jacket, plaid skirt\n"
        "Setup: full body, school hallway\n"
        "Supplement: cherry blossoms, spring\n"
        "Previous: 1girl, sena, school uniform, green jacket, plaid skirt, "
        "blonde long hair\n"
        "Chat: Sena went to the pool and changed into her swimsuit.\n"
        "Output:\n"
        "1girl, wearing a blue bikini swimsuit, blonde long hair, "
        "aqua eyes, kashiwazaki sena\n"
        "[EXPR]\n"
        "embarrassed, blushing, nervous\n"
        "[SETUP]\n"
        "full body, pool\n"
        "[SUPPLEMENT]\n"
        "Bright sunlight reflecting off the pool water.\n"
        "[REASON]\n"
        "outfit: changed to swimsuit. expression: embarrassed at pool. "
        "setup: hallway→pool. supplement: spring→summer context. "
        "framing: full body, all visible.\n"
        "\n"
        "--- Example 3: Multi-character with position descriptors ---\n"
        "Character Name: kashiwazaki sena\n"
        "Character: the girl on the left is adolescent, kashiwazaki sena "
        "(boku wa tomodachi ga sukunai), blonde hair, long hair, aqua eyes, "
        "light blue sleeveless chiffon sun dress, grin, holding fork\n"
        "Setup: upper body, 2girls, interior, food court, wooden table\n"
        "Supplement: The blonde girl on the left leans toward a smaller girl.\n"
        "Previous: the girl on the left is adolescent, kashiwazaki sena, "
        "blonde hair, light blue sun dress\n"
        "Chat: Sena leaned forward grinning at Kobato across the food court "
        "table.\n"
        "Output:\n"
        "the girl on the left is adolescent, "
        "the girl who is wearing a light blue sleeveless chiffon sun dress, "
        "blonde hair, long hair, aqua eyes, kashiwazaki sena\n"
        "[EXPR]\n"
        "grin, holding fork, leaning forward\n"
        "[SETUP]\n"
        "2girls, upper body, interior, food court, wooden table\n"
        "[SUPPLEMENT]\n"
        "A blonde girl leans forward at a food court table.\n"
        "[REASON]\n"
        "outfit: unchanged. expression: grinning playfully. "
        "setup: kept 2girls at start, unchanged. supplement: food court scene. "
        "framing: upper body, all visible.\n"
        "\n"
        "--- Example 4: Male character with position descriptor ---\n"
        "Character Name: oreki houtarou\n"
        "Character: the boy on the left is adolescent, oreki houtarou "
        "(hyouka), (mature male:0.9), brown hair, short hair, messy hair, "
        "green eyes, navy blue thin cotton long-sleeved t-shirt, beige cotton "
        "wide-leg trousers, holding tray, walking\n"
        "Setup: full body, 1girl, 1boy, from side, interior, food court, "
        "noon, bright, tables in background\n"
        "Supplement: A tall boy and girl walk together toward a table.\n"
        "Previous: the boy on the left is adolescent, oreki houtarou, "
        "brown hair, navy blue t-shirt, beige trousers\n"
        "Chat: Oreki walked lazily carrying his food tray.\n"
        "Output:\n"
        "the boy on the left is adolescent, "
        "the boy who is wearing a navy blue thin cotton long-sleeved t-shirt "
        "and beige cotton wide-leg trousers, "
        "brown hair, short hair, messy hair, green eyes, oreki houtarou\n"
        "[EXPR]\n"
        "indifferent, half-closed eyes, holding tray, walking\n"
        "[SETUP]\n"
        "1girl, 1boy, full body, from side, interior, food court, noon, "
        "bright, tables in background\n"
        "[SUPPLEMENT]\n"
        "A boy carrying a tray walks through a bright food court.\n"
        "[REASON]\n"
        "outfit: unchanged. expression: lethargic and indifferent. "
        "setup: kept 1girl 1boy at start. supplement: food court scene. "
        "framing: full body, all visible.\n"
        "\n"
        "--- Example 5: Character interaction (backhug) ---\n"
        "Character Name: kashiwazaki sena\n"
        "Character: the girl in front is adolescent, kashiwazaki sena "
        "(boku wa tomodachi ga sukunai), blonde hair, long hair, aqua eyes, "
        "light blue sleeveless chiffon sun dress\n"
        "Setup: upper body, 2girls, interior, living room, evening\n"
        "Supplement: A girl is being hugged from behind.\n"
        "Previous: the girl in front is adolescent, kashiwazaki sena, "
        "blonde hair, light blue sun dress\n"
        "Chat: Sena was standing in the living room when Kobato suddenly "
        "hugged her from behind.\n"
        "Output:\n"
        "the girl in front is adolescent, "
        "the girl who is wearing a light blue sleeveless chiffon sun dress, "
        "blonde hair, long hair, aqua eyes, kashiwazaki sena\n"
        "[EXPR]\n"
        "surprised, blushing, wide eyes\n"
        "[SETUP]\n"
        "2girls, backhug, upper body, interior, living room, evening\n"
        "[SUPPLEMENT]\n"
        "A girl being hugged from behind in a softly lit living room.\n"
        "[REASON]\n"
        "outfit: unchanged. expression: surprised by sudden hug. "
        "setup: added backhug interaction tag. supplement: living room scene. "
        "framing: upper body, all visible.\n"
    )


# ─── User Message Building ─────────────────────────────

def _build_user_message(
    character_name: str,
    char: str,
    setup: str,
    supplement: str,
    chat: str = "",
    slot: str = "",
    previous_enhanced: str = "",
    previous_chars: list = None,
) -> str:
    slot_before = ""
    slot_after = ""
    if slot:
        if '||' in slot:
            parts = slot.split('||', 1)
            slot_before = parts[0].strip()
            slot_after = parts[1].strip()
        else:
            slot_before = slot.strip()
            slot_after = ""

    parts = []
    parts.append(f"### Character Name\n{character_name}")
    parts.append(f"### Character\n{char}")

    if setup:
        parts.append(f"### Setup\n{setup}")

    if supplement:
        parts.append(f"### Supplement\n{supplement}")

    if chat:
        parts.append(f"### Chat\n{chat}")

    # Image Insertion Point
    if slot_before or slot_after:
        insertion_context = _extract_context_around(
            chat, slot_before, slot_after, 2000
        )
        if insertion_context:
            parts.append(
                "### Image Insertion Point\n"
                "THIS IS WHERE THE IMAGE IS BEING GENERATED.\n"
                "The image depicts the scene at this exact point.\n"
                f"{insertion_context}"
            )

    if previous_enhanced:
        parts.append(
            "### Previous Enhanced (OUTFIT ONLY — reuse outfit descriptions exactly, "
            "generate NEW expression from context)\n"
            f"{previous_enhanced}"
        )

    if previous_chars:
        prev_text = "\n".join(
            f"[{i}] {pc}" for i, pc in enumerate(previous_chars, 1)
        )
        parts.append(f"### Previous Characters (newest first)\n{prev_text}")

    return "\n\n".join(parts)


# ─── Main Entry Point ──────────────────────────────────

async def run(
    character_name: str,
    char: str,
    previous_enhanced: str = "",
    setup: str = "",
    supplement: str = "",
    chat: str = "",
    slot: str = "",
) -> dict:
    """
    캐릭터, setup, supplement를 각각 강화.

    Args:
        character_name: 캐릭터 이름
        char: 현재 캐릭터 태그 ([UPSCALE]에서 추출)
        setup: setup 태그 ([ILXL]에서 char 제외)
        supplement: supplement 태그 ([Positive]에서 char 이후)
        chat: 채팅 내용
        slot: 이미지 삽입 위치 (|| 로 before/after 분리)
        previous_enhanced: 이전 강화 결과 (복장만)

    Returns:
        dict: {
            "char": "강화된 캐릭터 + 표정",
            "outfit_only": "강화된 복장만 (일관성 추적용)",
            "setup": "강화된 setup",
            "supplement": "강화된 supplement",
        }
    """
    # 이전 히스토리 로드
    previous_chars = _get_recent_chars(character_name, 3)

    # 메시지 구성
    system_prompt = _build_system_prompt()
    user_message = _build_user_message(
        character_name, char, setup, supplement,
        chat=chat,
        slot=slot,
        previous_enhanced=previous_enhanced,
        previous_chars=previous_chars,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # 입력 로그
    log_input = {
        "character_name": character_name,
        "char": char[:300] if char else "",
        "setup": setup[:300] if setup else "",
        "supplement": supplement[:300] if supplement else "",
        "chat": chat[:300] if chat else "",
        "previous_enhanced": previous_enhanced[:200] if previous_enhanced else "",
        "previous_chars": previous_chars,
    }

    # LLM1 호출
    result = await callLLM(messages)

    # LLM2 폴백
    if result.startswith("[LLM 실패]"):
        config = get_config()
        llm_model2 = config.get("llm_model2", "")
        if llm_model2:
            if _is_gemini_model(llm_model2):
                messages2 = gpt2gemini(messages)
            else:
                messages2 = messages
            result = await callLLM2(messages2)

    if result.startswith("[LLM 실패]"):
        _log_prompt_io(character_name, log_input, result)
        return {
            "char": char,
            "outfit_only": char,
            "setup": setup,
            "supplement": supplement,
        }

    # 결과 정리
    result = result.strip()
    result = re.sub(r'^```(?:\w+)?\s*', '', result)
    result = re.sub(r'\s*```$', '', result)
    result = result.strip()

    # ─── 파싱: [EXPR], [SETUP], [SUPPLEMENT], [REASON] ───
    raw_output = result  # 로그용 원본 보존

    reason_part = ""
    if '[REASON]' in result:
        parts = result.split('[REASON]', 1)
        result = parts[0].strip()
        reason_part = parts[1].strip().split('\n')[0]

    supplement_result = supplement  # 기본값: 원본
    if '[SUPPLEMENT]' in result:
        parts = result.split('[SUPPLEMENT]', 1)
        result = parts[0].strip()
        raw_supplement = parts[1].strip().rstrip(',').strip()
        if raw_supplement and raw_supplement.lower() not in ('none', 'n/a'):
            supplement_result = raw_supplement
        else:
            supplement_result = ""  # LLM이 불필요하다고 판단하면 제거

    setup_result = setup  # 기본값: 원본
    if '[SETUP]' in result:
        parts = result.split('[SETUP]', 1)
        result = parts[0].strip()
        setup_result = parts[1].strip().rstrip(',').strip()

    # 나머지 = outfit + expression
    expression_part = ""
    outfit_part = result
    if '[EXPR]' in result:
        parts = result.split('[EXPR]', 1)
        outfit_part = parts[0].strip().rstrip(',').strip()
        expression_part = parts[1].strip().rstrip(',').strip()

    if not outfit_part:
        outfit_part = char

    # 로그
    if reason_part:
        print(f"[ENHANCE_REASON] {character_name}: {reason_part}")
    log_input["reason"] = reason_part
    _log_prompt_io(character_name, log_input, raw_output)

    # 캐릭터 이름 검사
    result_lower = outfit_part.lower()
    if character_name.lower() not in result_lower:
        name_parts = [p for p in character_name.lower().split() if len(p) > 1]
        if not any(part in result_lower for part in name_parts):
            return {
                "char": char,
                "outfit_only": char,
                "setup": setup,
                "supplement": supplement,
            }

    # | 문자 제거
    outfit_part = outfit_part.replace('|', ',')
    expression_part = expression_part.replace('|', ',')
    setup_result = setup_result.replace('|', ',')
    supplement_result = supplement_result.replace('|', ',')

    # combined 조합 (outfit + expression)
    if expression_part:
        combined = f"{outfit_part}, {expression_part}"
    else:
        combined = outfit_part

    # 히스토리 저장
    slot_before = ""
    slot_after = ""
    if slot:
        if '||' in slot:
            sp = slot.split('||', 1)
            slot_before, slot_after = sp[0].strip(), sp[1].strip()
        else:
            slot_before, slot_after = slot.strip(), ""

    history_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "char": outfit_part,
        "setup": setup,
        "supplement": supplement,
        "chat": chat[:500] if chat else "",
        "slot_before": slot_before,
        "slot_after": slot_after,
        "reason": reason_part,
    }
    _add_history_entry(character_name, history_entry)

    return {
        "char": combined,
        "outfit_only": outfit_part,
        "setup": setup_result,
        "supplement": supplement_result,
    }
