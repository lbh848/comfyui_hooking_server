"""
복장 프롬프트 강화 - v5
모든 캐릭터를 한 번의 LLM 호출로 강화
JSON 구조화된 출력으로 캐릭터별 결과 + 공통 setup/supplement 반환

v4와의 차이:
- 캐릭터별 개별 LLM 호출 → 단일 호출
- [EXPR]/[SETUP] 파싱 → JSON 파싱
- setup/supplement는 캐릭터 공유 (1번만 출력)

필수 함수:
    async def run_all(
        characters: list[dict],
        setup: str = "",
        supplement: str = "",
        chat: str = "",
        slot: str = "",
    ) -> dict

    async def run(...)  # v4 호환용 (내부적으로 run_all 호출)

characters 인자:
    [{"name": str, "char": str, "previous_enhanced": str}, ...]

반환값:
    dict: {
        "characters": [
            {"name": str, "char": str, "outfit_only": str},
            ...
        ],
        "setup": str,
        "supplement": str,
    }
"""

import re
import json
import os
import datetime
import tiktoken
from modes.llm_service import callLLM, callLLM2, get_config


# ─── Storage (v4와 공유) ────────────────────────────────
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "enhance_outfit_prompt_v4_storage")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
PROMPT_LOG_FILE = os.path.join(LOG_DIR, "enhance_prompt_io.log")
MAX_HISTORY = 15

# ─── NSFW 키워드 ──────────────────────────────────────
NSFW_KEYWORDS = [
    "breast caress", "cowgirl back cumshot", "cowgirl back", "cowgirl cumshot",
    "cowgirl", "doggystyle cumshot", "doggystyle", "fellatio", "fingering",
    "footjob cumshot", "footjob", "fullnelson cumshot", "fullnelson", "handjob",
    "masturbation", "mating press cumshot", "mating press", "missionary position",
    "missionary position cumshot", "paizuri", "reverse pright straddle cumshot",
    "reverse pright straddle", "reverse standing position", "showing armpit",
    "showing nude", "spooning cumshot", "spooning", "standing position",
    "suspended congress cumshot", "suspended congress", "upright straddle cumshot",
    "upright straddle",
]


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


def _detect_nsfw(characters: list[dict]) -> tuple:
    """캐릭터 태그에서 NSFW 키워드 감지 → (detected: bool, matched: list[str])"""
    matched = []
    for char_data in characters:
        tags = (char_data.get("char") or "").lower()
        for kw in NSFW_KEYWORDS:
            if kw in tags and kw not in matched:
                matched.append(kw)
    return (len(matched) > 0, matched)


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


def _is_gemini_model(model_name: str) -> bool:
    if not model_name:
        return False
    return any(kw in model_name.lower() for kw in ("gemini", "gemma"))


# ─── Context Extraction ─────────────────────────────────

def _extract_context_around(text: str, anchor_before: str, anchor_after: str,
                            max_tokens: int) -> str:
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
        "Enhance ALL characters, setup, and supplement tags for CONSISTENCY and CONTEXT.\n"
        "You process ALL characters in a SINGLE output.\n"
        "\n"
        "### CORE PRINCIPLE\n"
        "LESS IS MORE. Remove unnecessary details. Keep only essential visual elements.\n"
        "Shorter prompts = better image generation. Every tag must earn its place.\n"
        "Output should be SHORTER or similar length to the input — never longer.\n"
        "\n"
        "CRITICAL: NEVER add details that were NOT in the input tags.\n"
        "Specifically:\n"
        "- Do NOT add fabric, material, or texture adjectives.\n"
        "  'grey hoodie' stays 'grey hoodie' — NOT 'grey cotton oversized hoodie'.\n"
        "  'black tuxedo' stays 'black tuxedo' — NOT 'black tailored cotton tuxedo'.\n"
        "- Do NOT add fit or style modifiers.\n"
        "  'skirt' stays 'skirt' — NOT 'pleated skirt' or 'A-line skirt'.\n"
        "- Do NOT add color variations.\n"
        "  'blue' stays 'blue' — NOT 'sky-blue' or 'navy blue'.\n"
        "- If Chat describes an outfit change, use ONLY the details Chat provides.\n"
        "  If Chat says 'grey hoodie and black shorts', output 'grey hoodie, black shorts'.\n"
        "  Do NOT add 'cotton', 'polyester', 'oversized', etc.\n"
        "\n"
        "### INPUTS\n"
        "You receive MULTIPLE characters to enhance simultaneously.\n"
        "\n"
        "For each character (### Character N):\n"
        "- **Name**: authoritative full name.\n"
        "- **Tags**: current character tags.\n"
        "- **Previous Enhanced**: previous enhanced output for consistency.\n"
        "- **Previous Characters**: recent character descriptions.\n"
        "\n"
        "Shared inputs:\n"
        "- **Setup** (### Setup): camera/framing/location tags.\n"
        "  Use camera framing to determine visible range.\n"
        "- **Supplement** (### Supplement): scene description tags.\n"
        "- **Chat** (### Chat): current story context.\n"
        "- **Image Insertion Point** (### Image Insertion Point):\n"
        "  WHERE the image is generated.\n"
        "\n"
        "### CLEANUP RULES\n"
        "1. **Context filtering**: Remove items that don't fit the story context.\n"
        "   Example: 'robot' appearing randomly when no robots are in the scene.\n"
        "2. **Atmosphere removal**: Remove tags that could be misinterpreted\n"
        "   as physical traits.\n"
        "   'fox-like', 'robot-like', 'cat-like' → REMOVE.\n"
        "   'elegant atmosphere', 'mysterious vibe' → REMOVE.\n"
        "3. **Filler removal**: Remove meaningless decorative words.\n"
        "   'beautiful', 'stunning', 'gorgeous', 'detailed', 'intricate' → REMOVE.\n"
        "4. **Framing-based visibility**:\n"
        "   Use Setup camera/framing to determine which parts are VISIBLE.\n"
        "   Items OUTSIDE the visible frame → REMOVE from output.\n"
        "   - close-up / portrait / face focus →\n"
        "     NO shoes, lower body clothing, held items below frame\n"
        "   - upper body / bust shot → NO shoes, skirt below frame\n"
        "   - cowboy shot (mid-thigh) → most visible, shoes may be at frame edge\n"
        "   - full body → all items visible\n"
        "   - from above / high angle → feet likely out of frame\n"
        "   Only exclude when framing CLEARLY makes the item invisible.\n"
        "\n"
        "### OUTFIT CONSISTENCY\n"
        "For each character, compare current tags with Previous Enhanced.\n"
        "- If outfit UNCHANGED (no outfit change described in chat):\n"
        "  COPY outfit description from previous output VERBATIM.\n"
        "  Do NOT rephrase, shade-shift, or modify ANY outfit wording.\n"
        "  'blue' stays 'blue' — not 'sky-blue', 'navy', or 'light blue'.\n"
        "- If outfit CHANGED (chat describes changing clothes):\n"
        "  Describe new outfit SIMPLY. Same detail level as previous.\n"
        "  Use ONLY the details mentioned in Chat or input tags.\n"
        "  If Chat says 'put on a red jacket', add 'red jacket' only.\n"
        "  Do NOT infer fabric, material, texture, or fit.\n"
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
        "### NSFW SCENE HANDLING\n"
        "When the input contains an ### NSFW Scene Detected section OR the Chat\n"
        "context clearly describes a sexual situation, apply these rules:\n"
        "\n"
        "1. **PRESERVE NSFW TAGS** (CRITICAL — highest priority):\n"
        "   EVERY NSFW keyword found in input tags MUST appear in the output,\n"
        "   in the same character's outfit field.\n"
        "   'masturbation' in input → 'masturbation' in output. No exceptions.\n"
        "   'doggystyle' in input → 'doggystyle' in output. No exceptions.\n"
        "   Do NOT soften, substitute, or paraphrase NSFW keywords.\n"
        "\n"
        "2. **ENHANCE NSFW EXPRESSION** with Danbooru-compatible tags:\n"
        "   Body reactions: arched back, head back, trembling, sweating,\n"
        "     heavy breathing, flushed face, biting lip, rolling eyes,\n"
        "     tongue out, drooling, body shake, grabbing sheets, clutching,\n"
        "     muscle tension, curled toes, vaginal fluid, bodily fluids,\n"
        "     wet, soaked, disheveled clothes, clothes pulled up,\n"
        "     clothes pulled aside, bottomless, topless\n"
        "   Emotional: crying, tears, intense blush, overwhelmed, ecstasy,\n"
        "     ahegao, mind break, half-closed eyes, drooping eyes,\n"
        "     rolled eyes, pleasure, scream, moaning, panting\n"
        "\n"
        "3. **SUPPLEMENT for NSFW scenes**:\n"
        "   Describe the atmosphere vividly in 1-2 sentences.\n"
        "   Include physical reactions, body tension, and mood.\n"
        "   Keep under 40 words but capture intensity.\n"
        "\n"
        "4. **NSFW position/action reference keywords**:\n"
        "   breast caress, cowgirl, cowgirl back, cowgirl cumshot,\n"
        "   cowgirl back cumshot, doggystyle, doggystyle cumshot,\n"
        "   fellatio, fingering, footjob, footjob cumshot, fullnelson,\n"
        "   fullnelson cumshot, handjob, masturbation, mating press,\n"
        "   mating press cumshot, missionary position,\n"
        "   missionary position cumshot, paizuri,\n"
        "   reverse pright straddle, reverse pright straddle cumshot,\n"
        "   reverse standing position, showing armpit, showing nude,\n"
        "   spooning, spooning cumshot, standing position,\n"
        "   suspended congress, suspended congress cumshot,\n"
        "   upright straddle, upright straddle cumshot\n"
        "   When input tags contain any of these → output MUST include it.\n"
        "\n"
        "### CHARACTER NAME RULES\n"
        "1. ALWAYS include each character's Name exactly as provided.\n"
        "2. Do NOT add series/franchise names.\n"
        "3. If the Name already appears in the position descriptor\n"
        "   (e.g., 'the character behind is oreki houtarou'), do NOT repeat it.\n"
        "\n"
        "### POSITION AND MULTI-CHARACTER (CRITICAL RULE)\n"
        "When a character's input starts with a position/placement phrase,\n"
        "copy that ENTIRE opening phrase to the very start of that character's outfit,\n"
        "verbatim, without ANY modification.\n"
        "Position phrases include:\n"
        "  'the girl on the left is', 'the boy on the right is',\n"
        "  'the man on the left is', 'the woman on the right is',\n"
        "  'the character in the foreground is', 'the character on the monitor is',\n"
        "  'the person behind is', 'the girl in the center is', etc.\n"
        "Do NOT replace position phrases with count tags like '1girl' or '1boy'.\n"
        "Only include count tags if they were already in the input.\n"
        "For multi-character outfit sentences, use:\n"
        "  'the girl/boy who is wearing...'\n"
        "matching the gender from the position phrase.\n"
        "\n"
        "### OTHER CHARACTER EXCLUSION\n"
        "Each character's output must describe ONLY that character.\n"
        "References to other specific characters must NOT appear.\n"
        "However, POSITION descriptors (e.g., 'the girl on the left is')\n"
        "describe WHERE the target character is placed — ALWAYS preserve these.\n"
        "\n"
        "### SETUP ENHANCEMENT\n"
        "Refine the Setup tags (shared across ALL characters):\n"
        "- Do NOT change the location, setting, or scene composition.\n"
        "  Only remove filler/quality tags and atmosphere words.\n"
        "- Keep: camera, framing, angle, location, lighting, weather,\n"
        "  time-of-day, non-clothing props.\n"
        "- REMOVE from Setup: clothing items the character is holding but NOT wearing\n"
        "  (e.g. 'holding t-shirt'). Move these to Supplement.\n"
        "- KEEP: character count tags (1girl, 1boy, 2girls, etc.).\n"
        "  Place at VERY START of setup.\n"
        "- Interaction tags: when Chat describes physical contact,\n"
        "  add Danbooru interaction tag AFTER count tags, BEFORE camera tags.\n"
        "  Tags: backhug, hugging, holding hands, arm around shoulder, carrying,\n"
        "  piggyback, leaning on, headpat, handshake, linking arms, kissing, couple.\n"
        "\n"
        "### SUPPLEMENT ENHANCEMENT\n"
        "Refine the Supplement into 1-2 SHORT natural language sentences (shared):\n"
        "- Describe the overall scene, atmosphere, or key visual elements.\n"
        "- NEVER output 'none' or leave blank.\n"
        "- Keep it concise: 1-2 sentences, under 30 words total.\n"
        "- Do NOT change the location, setting, characters, or spatial relationships.\n"
        "- MULTI-CHARACTER SCENES: Supplement MUST describe the OVERALL scene\n"
        "  including ALL characters and their spatial relationships.\n"
        "\n"
        "### OUTPUT FORMAT\n"
        "Output ONLY a JSON object. No text before or after.\n"
        "If you must use a code block, use ```json.\n"
        "\n"
        "Exact structure:\n"
        "```json\n"
        "{\n"
        '  "characters": [\n'
        "    {\n"
        '      "name": "character_full_name",\n'
        '      "outfit": "position/count, wearing a [outfit], body tags, character_name",\n'
        '      "expression": "expression_tag1, expression_tag2"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "refined setup tags",\n'
        '  "supplement": "refined supplement",\n'
        '  "reason": "one-line memo for all characters"\n'
        "}\n"
        "```\n"
        "\n"
        "Per-character \"outfit\" format (4-step order):\n"
        "  1. Position descriptor or count tag (FIRST)\n"
        "     - Multi-character: copy position phrase verbatim\n"
        "     - Single character: count tag (e.g., '1girl')\n"
        "  2. Outfit as natural sentence (SECOND)\n"
        "     - 'wearing a [outfit description]'\n"
        "     - Include accessories with 'with'\n"
        "  3. Body attribute tags: hair, eyes, body type\n"
        "  4. Character Name (LAST)\n"
        "  Do NOT include: pose, action, location, environment, lighting.\n"
        "\n"
        "\"expression\": comma-separated expression/emotion tags ONLY.\n"
        "\"setup\": refined scene tags (comma-separated).\n"
        "\"supplement\": 1-2 SHORT natural language sentences.\n"
        "\"reason\": brief one-line decision memo covering all characters.\n"
        "\n"
        "Rules:\n"
        "1. Do NOT use the | character anywhere.\n"
        "2. Do NOT add new weighted tags.\n"
        "3. Output valid JSON only.\n"
        "4. Include ALL input characters in the output.\n"
        "\n"
        "### EXAMPLES\n"
        "\n"
        "--- Example 1: Single character, outfit unchanged ---\n"
        "Input:\n"
        "Character 1: hasegawa kobato\n"
        "Tags: 1girl, kobato, gothic lolita dress, black hairband, mysterious atmosphere\n"
        "Setup: cowboy shot, classroom\n"
        "Supplement: night, moonlight\n"
        "Previous: 1girl, kobato, gothic lolita dress, black hairband, blonde hair, heterochromia\n"
        "Chat: Kobato was reading quietly in her room.\n"
        "\n"
        "Output:\n"
        "```json\n"
        "{\n"
        '  "characters": [\n'
        "    {\n"
        '      "name": "hasegawa kobato",\n'
        '      "outfit": "1girl, wearing a gothic lolita dress with a black hairband, '
        "blonde hair, heterochromia, hasegawa kobato\",\n"
        '      "expression": "calm, focused, light smile"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "cowboy shot, classroom",\n'
        '  "supplement": "A quiet room lit by soft moonlight through the window.",\n'
        '  "reason": "outfit: unchanged. expression: reading calmly. '
        "setup: unchanged. supplement: unchanged.\"\n"
        "}\n"
        "```\n"
        "\n"
        "--- Example 2: Multi-character, outfits unchanged ---\n"
        "Input:\n"
        "Character 1: redfield stella\n"
        "Tags: the character in the foreground is redfield stella, girl, blue eyes, "
        "blonde hair, short hair, black tailored tuxedo, white shirt, black necktie\n"
        "Previous: the character in the foreground is redfield stella, black tuxedo, "
        "short blonde hair, blue eyes\n"
        "Character 2: kashiwazaki sena\n"
        "Tags: the character on the monitor is kashiwazaki sena, girl, blonde hair, grey hoodie\n"
        "Previous: the character on the monitor is kashiwazaki sena, grey hoodie, "
        "blonde hair, aqua eyes\n"
        "Setup: upper body, from behind, 2girls, interior, security room, monitor, night\n"
        "Supplement: A blonde woman watches monitors in a dark room.\n"
        "Chat: Stella watched the monitors calmly. Sena sneaked through the hallway.\n"
        "\n"
        "Output:\n"
        "```json\n"
        "{\n"
        '  "characters": [\n'
        "    {\n"
        '      "name": "redfield stella",\n'
        '      "outfit": "the character in the foreground is the girl who is wearing '
        "a black tailored tuxedo with a white shirt and black necktie, "
        "short blonde hair, blue eyes, redfield stella\",\n"
        '      "expression": "smirk, calm, looking at monitor"\n'
        "    },\n"
        "    {\n"
        '      "name": "kashiwazaki sena",\n'
        '      "outfit": "the character on the monitor is the girl who is wearing '
        "a grey hoodie, blonde hair, aqua eyes, kashiwazaki sena\",\n"
        '      "expression": "scared, wide eyes, nervous"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "upper body, from behind, 2girls, interior, security room, '
        "monitor, night\",\n"
        '  "supplement": "A woman in a tuxedo watches security monitors showing '
        "a girl in a hoodie.\",\n"
        '  "reason": "outfit: unchanged for both. expression: stella calm, '
        "sena scared. setup: unchanged. supplement: condensed.\"\n"
        "}\n"
        "```\n"
        "\n"
        "--- Example 3: Outfit changed (Chat describes change) ---\n"
        "Input:\n"
        "Character 1: kashiwazaki sena\n"
        "Tags: 1girl, sena, school uniform, green jacket, plaid skirt\n"
        "Previous: 1girl, sena, school uniform, green jacket, plaid skirt, blonde long hair\n"
        "Setup: full body, school hallway\n"
        "Supplement: cherry blossoms, spring\n"
        "Chat: Sena went to the pool and changed into a blue swimsuit.\n"
        "\n"
        "Output:\n"
        "```json\n"
        "{\n"
        '  "characters": [\n'
        "    {\n"
        '      "name": "kashiwazaki sena",\n'
        '      "outfit": "1girl, wearing a blue swimsuit, blonde long hair, '
        "aqua eyes, kashiwazaki sena\",\n"
        '      "expression": "embarrassed, blushing, nervous"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "full body, pool",\n'
        '  "supplement": "Bright sunlight reflecting off the pool water.",\n'
        '  "reason": "outfit: changed to swimsuit. expression: embarrassed at pool. '
        "setup: hallway to pool. supplement: updated.\"\n"
        "}\n"
        "```\n"
    )


# ─── User Message Building ─────────────────────────────

def _build_user_message(
    characters: list[dict],
    setup: str,
    supplement: str,
    chat: str = "",
    slot: str = "",
) -> str:
    parts = []

    # 각 캐릭터 정보
    for i, char_data in enumerate(characters, 1):
        block = []
        block.append(f"### Character {i}")
        block.append(f"Name: {char_data['name']}")
        block.append(f"Tags: {char_data['char']}")

        if char_data.get("previous_enhanced"):
            block.append(
                f"Previous Enhanced (OUTFIT ONLY — reuse outfit exactly, "
                f"generate NEW expression): {char_data['previous_enhanced']}"
            )

        if char_data.get("previous_chars"):
            prev_text = "\n".join(
                f"[{j}] {pc}" for j, pc in enumerate(char_data['previous_chars'], 1)
            )
            block.append(f"Previous Characters (newest first):\n{prev_text}")

        parts.append("\n".join(block))

    # 공통 입력
    if setup:
        parts.append(f"### Setup\n{setup}")

    if supplement:
        parts.append(f"### Supplement\n{supplement}")

    if chat:
        parts.append(f"### Chat\n{chat}")

    # Image Insertion Point
    slot_before = ""
    slot_after = ""
    if slot:
        if '||' in slot:
            sp = slot.split('||', 1)
            slot_before, slot_after = sp[0].strip(), sp[1].strip()
        else:
            slot_before, slot_after = slot.strip(), ""

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

    # NSFW 감지
    nsfw_detected, nsfw_matched = _detect_nsfw(characters)
    if nsfw_detected:
        matched_str = ", ".join(nsfw_matched)
        all_kw_str = ", ".join(NSFW_KEYWORDS)
        parts.append(
            "### NSFW Scene Detected\n"
            f"Matched NSFW keywords in input: {matched_str}\n"
            "IMPORTANT: Preserve ALL matched keywords in output exactly as-is.\n"
            "Enhance the scene in NSFW direction using the NSFW SCENE HANDLING rules.\n"
            "Do NOT remove or soften any sexual content tags."
        )

    return "\n\n".join(parts)


# ─── JSON 파싱 ─────────────────────────────────────────

def _parse_json_output(raw: str) -> dict:
    """LLM 출력에서 JSON 추출. 코드블록 또는 raw JSON 지원."""
    raw = raw.strip()

    # ```json ... ``` 코드블록 추출
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()

    # ``` 가 없으면 첫 { 부터 마지막 } 까지
    if not raw.startswith('{'):
        start = raw.find('{')
        end = raw.rfind('}')
        if start >= 0 and end > start:
            raw = raw[start:end + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ─── Main Entry Point ──────────────────────────────────

async def run_all(
    characters: list[dict],
    setup: str = "",
    supplement: str = "",
    chat: str = "",
    slot: str = "",
) -> dict:
    """
    모든 캐릭터를 한 번의 LLM 호출로 강화.

    Args:
        characters: [{"name": str, "char": str, "previous_enhanced": str}, ...]
        setup: setup 태그
        supplement: supplement 태그
        chat: 채팅 내용
        slot: 이미지 삽입 위치

    Returns:
        dict: {
            "characters": [{"name": str, "char": str, "outfit_only": str}, ...],
            "setup": str,
            "supplement": str,
        }
    """
    if not characters:
        return {
            "characters": [],
            "setup": setup,
            "supplement": supplement,
        }

    # previous_chars 로드
    for char_data in characters:
        char_data["previous_chars"] = _get_recent_chars(char_data["name"], 3)

    # 메시지 구성
    system_prompt = _build_system_prompt()
    user_message = _build_user_message(
        characters, setup, supplement,
        chat=chat,
        slot=slot,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # 입력 로그
    log_names = [c["name"] for c in characters]
    log_input = {
        "characters": log_names,
        "setup": setup[:300] if setup else "",
        "supplement": supplement[:300] if supplement else "",
        "chat": chat[:300] if chat else "",
    }

    # LLM1 호출
    result = await callLLM(messages)

    # LLM2 폴백
    if result.startswith("[LLM 실패]"):
        config = get_config()
        llm_model2 = config.get("llm_model2", "")
        if llm_model2:
            messages2 = gpt2gemini(messages) if _is_gemini_model(llm_model2) else messages
            result = await callLLM2(messages2)

    if result.startswith("[LLM 실패]"):
        _log_prompt_io(",".join(log_names), log_input, result)
        # 실패 시 원본 반환
        return {
            "characters": [
                {"name": c["name"], "char": c["char"], "outfit_only": c["char"]}
                for c in characters
            ],
            "setup": setup,
            "supplement": supplement,
        }

    # 결과 정리
    result = result.strip()

    # JSON 파싱
    parsed = _parse_json_output(result)
    if not parsed or "characters" not in parsed:
        _log_prompt_io(",".join(log_names), log_input, result)
        # 파싱 실패 시 원본 반환
        return {
            "characters": [
                {"name": c["name"], "char": c["char"], "outfit_only": c["char"]}
                for c in characters
            ],
            "setup": setup,
            "supplement": supplement,
        }

    # 로그
    log_input["reason"] = parsed.get("reason", "")
    _log_prompt_io(",".join(log_names), log_input, result)

    # 결과 후처리
    setup_result = parsed.get("setup", setup)
    supplement_result = parsed.get("supplement", supplement)
    if supplement_result and supplement_result.lower() in ('none', 'n/a'):
        supplement_result = ""

    # | 문자 제거
    setup_result = setup_result.replace('|', ',')
    supplement_result = supplement_result.replace('|', ',')

    char_results = []
    for char_data in characters:
        name = char_data["name"]
        # JSON에서 해당 캐릭터 찾기
        json_char = None
        for jc in parsed.get("characters", []):
            if jc.get("name", "").lower() == name.lower():
                json_char = jc
                break

        if not json_char:
            # 이름 매칭 실패 시 원본 사용
            char_results.append({
                "name": name,
                "char": char_data["char"],
                "outfit_only": char_data["char"],
            })
            continue

        outfit = json_char.get("outfit", char_data["char"])
        expression = json_char.get("expression", "")

        # 캐릭터 이름 검사
        if name.lower() not in outfit.lower():
            name_parts = [p for p in name.lower().split() if len(p) > 1]
            if not any(part in outfit.lower() for part in name_parts):
                char_results.append({
                    "name": name,
                    "char": char_data["char"],
                    "outfit_only": char_data["char"],
                })
                continue

        # | 문자 제거
        outfit = outfit.replace('|', ',')
        expression = expression.replace('|', ',')

        # combined 조합
        combined = f"{outfit}, {expression}" if expression else outfit

        char_results.append({
            "name": name,
            "char": combined,
            "outfit_only": outfit,
        })

    # 히스토리 저장
    slot_before = ""
    slot_after = ""
    if slot:
        if '||' in slot:
            sp = slot.split('||', 1)
            slot_before, slot_after = sp[0].strip(), sp[1].strip()
        else:
            slot_before, slot_after = slot.strip(), ""

    for i, char_data in enumerate(characters):
        if i < len(char_results) and char_results[i].get("outfit_only"):
            history_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "char": char_results[i]["outfit_only"],
                "setup": setup,
                "supplement": supplement,
                "chat": chat[:500] if chat else "",
                "slot_before": slot_before,
                "slot_after": slot_after,
                "reason": parsed.get("reason", ""),
            }
            _add_history_entry(char_data["name"], history_entry)

    return {
        "characters": char_results,
        "setup": setup_result,
        "supplement": supplement_result,
    }


# ─── v4 호환 ────────────────────────────────────────────

async def run(
    character_name: str,
    char: str,
    previous_enhanced: str = "",
    setup: str = "",
    supplement: str = "",
    chat: str = "",
    slot: str = "",
) -> dict:
    """v4 호환 인터페이스. 단일 캐릭터를 run_all로 처리."""
    result = await run_all(
        characters=[{
            "name": character_name,
            "char": char,
            "previous_enhanced": previous_enhanced,
        }],
        setup=setup,
        supplement=supplement,
        chat=chat,
        slot=slot,
    )

    if result["characters"]:
        c = result["characters"][0]
        return {
            "char": c["char"],
            "outfit_only": c["outfit_only"],
            "setup": result["setup"],
            "supplement": result["supplement"],
        }

    return {
        "char": char,
        "outfit_only": char,
        "setup": setup,
        "supplement": supplement,
    }
