"""
복장 프롬프트 강화 - v6
자연어 기반 이미지 생성 모델에 최적화된 프롬프트 빌딩

v5와의 차이:
- Pipeline 구조 도입: LLM이 분석→판단→검증→출력의 순서로 사고
- 카테고리별 명확한 규칙 (카메라/의상/헤어/표정 분리)
- 자연어 서술에 최적화된 출력 포맷
- Reference Sheet 방식으로 일관성 강화

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

# ─── NSFW 키워드 (후처리용, 감지에 사용하지 않음) ──────
# _detect_nsfw() 제거됨. 문맥 감지는 LLM이 수행.


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
        "# Visual Description Refiner\n"
        "You enhance character appearance descriptions for AI image generation.\n"
        "Process ALL characters in a SINGLE output using structured analysis.\n"
        "\n"
        "# CRITICAL RULES (highest priority, override everything else)\n"
        "\n"
        "## RULE 1: LESS IS MORE\n"
        "Remove unnecessary details. Keep only essential visual elements.\n"
        "Output should be SHORTER or similar length to input — never longer.\n"
        "Every word must earn its place. Shorter prompts = better images.\n"
        "EXCEPTION: NEVER remove pose/body position tags (lying, sitting, standing, "
        "kneeling, bending, etc.) or NSFW/intimate tags (no bra, nipple outline, "
        "breast press, off-shoulder, etc.). These are ALWAYS essential.\n"
        "\n"
        "## RULE 2: NEVER INVENT DETAILS\n"
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
        "## RULE 3: POSITION DESCRIPTOR PRESERVATION\n"
        "When a character's input Tags start with a position phrase,\n"
        "copy it to the VERY START of that character's outfit field VERBATIM.\n"
        "Do NOT change, replace, paraphrase, or reinterpret it.\n"
        "\n"
        "CORRECT examples (input → output):\n"
        "  'the character on the left is X' → 'the character on the left is the boy who...'\n"
        "  'the character bottom is X' → 'the character bottom is the boy who...'\n"
        "  'the character center is X' → 'the character center is the girl who...'\n"
        "\n"
        "WRONG (NEVER do this):\n"
        "  'the character bottom is X' → 'the character on the right is...'  ✗ CHANGED\n"
        "  'the character center is X' → 'the character on the right is...'  ✗ CHANGED\n"
        "  Replacing with count tags like '1girl' instead of position phrase  ✗ REPLACED\n"
        "\n"
        "## RULE 4: POSE/ACTION SOURCE HIERARCHY\n"
        "PRIMARY: Pose, action, and body position come from INPUT TAGS.\n"
        "If Tags clearly state a pose → use it as-is.\n"
        "\n"
        "EXCEPTION — Chat pose correction (ONLY when BOTH conditions met):\n"
        "  1. Tags have NO pose info, OR Tags pose directly contradicts Chat.\n"
        "  2. Chat EXPLICITLY and DIRECTLY describes the pose (not implied/guessed).\n"
        "When both met → use pose from Chat.\n"
        "\n"
        "Chat context is ALSO used for:\n"
        "  - Determining EXPRESSION (emotion, facial expression)\n"
        "  - Detecting outfit CHANGES (character explicitly changes clothes)\n"
        "  - Understanding scene mood for Supplement\n"
        "\n"
        "Examples:\n"
        "  Tags: 'sitting on bed' + Chat clearly describes lying in bed → use 'lying on bed'\n"
        "  Tags: 'standing' + Chat says character walked across room → keep 'standing' (no direct pose)\n"
        "  Tags: 'sitting on chair' + Chat is about dialogue only → keep 'sitting on chair'\n"
        "If unsure whether Chat clearly describes pose → default to Tags only.\n"
        "\n"
        "## RULE 5: CHARACTER COUNT LOCK\n"
        "Output EXACTLY the same characters as input. No more, no fewer.\n"
        "If input has 2 characters → output has exactly 2 characters.\n"
        "NEVER add characters mentioned in Chat but not in the input Tags.\n"
        "Characters mentioned in Chat (even prominently) are NOT to be added\n"
        "unless they appear in the Character Tags input.\n"
        "Chat is context ONLY — not a character source.\n"
        "\n"
        "## RULE 6: NUDITY STATE PRESERVATION\n"
        "When input Tags indicate a character is in an unclothed state\n"
        "(e.g., naked, nude, bare, topless, bottomless, undressed, in underwear only,\n"
        "or any state clearly implying no full clothing), do NOT add clothing.\n"
        "The character MUST remain in that unclothed state in the output.\n"
        "This applies regardless of what the Chat narrative describes later.\n"
        "Clothing may only appear if the Tags themselves include clothing items,\n"
        "or if the Chat describes the character putting on clothes BEFORE the\n"
        "Image Insertion Point. (See RULE 7)\n"
        "\n"
        "Examples:\n"
        "  Tags: 'naked, soap suds, wet skin' → output: NO clothing. Keep naked state.\n"
        "  Tags: 'topless, jeans' → output: topless + jeans only. Do NOT add a shirt.\n"
        "  Tags: 'naked' but Chat AFTER insertion point says 'put on shirt'\n"
        "    → output: still naked. That event has not happened yet. (RULE 7)\n"
        "\n"
        "## RULE 7: TEMPORAL AWARENESS — INSERTION POINT BOUNDARY\n"
        "The image depicts the scene at the EXACT MOMENT of the Image Insertion Point.\n"
        "Only consider events, states, and details that exist AT or BEFORE this point.\n"
        "Events described in Chat that occur AFTER the insertion point are FUTURE events\n"
        "and must NOT be reflected in the current image.\n"
        "\n"
        "This applies to ALL visual elements: outfits, expressions, poses, actions,\n"
        "held items, location changes, and character states.\n"
        "\n"
        "How to determine what is 'before' vs 'after':\n"
        "- The Image Insertion Point marks the narrative moment being illustrated.\n"
        "- Chat text AFTER the insertion point = future events = IGNORE for this image.\n"
        "- Chat text BEFORE the insertion point = past context = can be used.\n"
        "- If unsure whether an event is before or after → default to Tags only.\n"
        "\n"
        "Example:\n"
        "  Insertion point: character is showering naked.\n"
        "  Chat after insertion: character finishes, puts on a t-shirt and shorts.\n"
        "  → Output: character is STILL naked in shower. The clothing is a future event.\n"
        "\n"
        "---\n"
        "\n"
        "# Pipeline: Analyze → Judge → Verify → Output\n"
        "Follow these phases IN ORDER for each character:\n"
        "\n"
        "## Phase 1: Context Understanding\n"
        "Read ALL inputs before writing anything:\n"
        "- **Image Insertion Point** → THIS is the exact narrative moment being depicted.\n"
        "  Establish a clear BEFORE/AFTER boundary in the Chat around this point.\n"
        "  Only use Chat content at or BEFORE this boundary. (RULE 7)\n"
        "- **Chat BEFORE insertion point** → What is happening RIGHT NOW?\n"
        "  What emotions are the characters feeling at this exact moment?\n"
        "- **Chat AFTER insertion point** → FUTURE events. Do NOT use for this image.\n"
        "- **Setup** camera/framing → What body parts are VISIBLE in the frame?\n"
        "- **Supplement** → Scene atmosphere and environment.\n"
        "\n"
        "## Phase 2: Per-Character Judgment\n"
        "For each character, make these decisions:\n"
        "\n"
        "### 2a. Position Descriptor & Framing Check\n"
        "First, check if Tags start with a position phrase (e.g., 'the character on the left is').\n"
        "If present → copy to the VERY START of outfit VERBATIM. See RULE 3.\n"
        "\n"
        "Then determine visible range from Setup camera/framing:\n"
        "| Framing | Visible | NOT visible |\n"
        "|---------|---------|-------------|\n"
        "| close-up / portrait / face focus | face, hair, accessories | shoes, lower body, held items below frame |\n"
        "| upper body / bust shot | waist up, top clothing | shoes, skirt hem below frame |\n"
        "| cowboy shot (mid-thigh) | most of body | shoes at frame edge only |\n"
        "| full body | everything | nothing excluded |\n"
        "| from above / high angle | upper body emphasis | feet likely cut off |\n"
        "Only exclude items when framing CLEARLY makes them invisible.\n"
        "\n"
        "### 2a2. Pose Judgment\n"
        "Determine character pose using RULE 4 hierarchy:\n"
        "1. Check Tags for pose keywords: sitting, lying, standing, kneeling, bending, etc.\n"
        "2. If Tags have a clear pose → use it. STOP HERE.\n"
        "3. If Tags have NO pose, or Tags pose contradicts Chat:\n"
        "   - Does Chat EXPLICITLY describe the character's body position?\n"
        "   - Is the description direct, not implied? (e.g., 'lying on the bed' = direct)\n"
        "   - If YES to both → use Chat pose.\n"
        "   - If NO → omit pose (do not guess).\n"
        "Include the final pose in the outfit output after the outfit phrase.\n"
        "\n"
        "### 2b. Outfit Judgment\n"
        "FIRST CHECK: Is the character in an unclothed state in Tags? (RULE 6)\n"
        "  If Tags indicate naked/nude/bare/topless/bottomless → PRESERVE that state.\n"
        "  Do NOT add clothing even if Chat later describes the character dressing.\n"
        "  The clothing is a FUTURE event relative to the insertion point. (RULE 7)\n"
        "\n"
        "Then compare current Tags with Previous Enhanced output:\n"
        "- **NO outfit change in Chat (before insertion point)** → COPY outfit from Previous Enhanced VERBATIM.\n"
        "  Zero modification. 'blue' stays 'blue', not 'sky-blue' or 'navy'.\n"
        "- **Outfit change described in Chat BEFORE insertion point** → Describe new outfit.\n"
        "  Use ONLY details mentioned in Chat or Tags. Same detail level as before.\n"
        "  If Chat says 'put on a red jacket', add only 'red jacket'.\n"
        "  Do NOT infer fabric, material, texture, or fit from Chat descriptions.\n"
        "- **Outfit change described in Chat AFTER insertion point** → IGNORE.\n"
        "  That change has not happened yet in the depicted moment. (RULE 7)\n"
        "\n"
        "CRITICAL RESTRICTION — Never invent details (see RULE 2):\n"
        "- Do NOT add fabric/material/texture adjectives.\n"
        "  'grey hoodie' stays 'grey hoodie' — NOT 'grey cotton oversized hoodie'.\n"
        "- Do NOT add fit/style modifiers.\n"
        "  'skirt' stays 'skirt' — NOT 'pleated skirt' or 'A-line skirt'.\n"
        "- Do NOT add color variations.\n"
        "  'blue' stays 'blue' — NOT 'sky-blue' or 'navy blue'.\n"
        "- Do NOT derive pose/action from chat narrative (see RULE 4).\n"
        "\n"
        "### 2c. Expression Judgment\n"
        "Derive expression from the narrative at Image Insertion Point:\n"
        "- ALWAYS FRESH — based on current scene emotion only.\n"
        "- NO carryover from previous expressions.\n"
        "- Use natural descriptive phrases, not isolated tags.\n"
        "- Expression covers EMOTION and FACE only.\n"
        "  Do NOT include body pose, action, or position in expression.\n"
        "  Pose/action comes from input Tags, NOT from chat. (RULE 4)\n"
        "  Good: 'gentle smile with downcast eyes', 'nervous, biting lip'\n"
        "  Bad: 'smile, closed_eyes, looking_down' (tag dump)\n"
        "\n"
        "### 2d. NSFW Tag Judgment\n"
        "Apply NSFW Tag Selection System (see dedicated section below):\n"
        "1. Classify scene intensity per character: Normal / Suggestive / Explicit.\n"
        "2. If Suggestive or Explicit → select appropriate tags from the pools.\n"
        "3. If Normal → no NSFW tags needed.\n"
        "4. Any NSFW tags from input Tags → ALWAYS preserve.\n"
        "\n"
        "## Phase 3: Verification Checklist\n"
        "Before finalizing output, check each character:\n"
        "- [ ] If Tags indicate unclothed state, is it preserved? (RULE 6)\n"
        "- [ ] No clothing added from Chat events AFTER insertion point? (RULE 7)\n"
        "- [ ] No invented fabric/material/texture/fit details? (RULE 2)\n"
        "- [ ] Outfit consistent with Previous Enhanced (unless Chat describes change)?\n"
        "- [ ] Position descriptor copied VERBATIM from input? (RULE 3)\n"
        "- [ ] Pose from Tags preserved, or corrected from Chat only when both RULE 4 conditions met?\n"
        "- [ ] Output has EXACTLY same number of characters as input? (RULE 5)\n"
        "- [ ] No characters from Chat added that weren't in input? (RULE 5)\n"
        "- [ ] Expression matches current scene mood (not previous)?\n"
        "- [ ] Character name present at end of outfit?\n"
        "- [ ] No character count tags added/removed from Setup?\n"
        "- [ ] NSFW tags from input preserved? (NSFW Tag Selection Step 3 rule 4)\n"
        "- [ ] Suggestive/Explicit scene has appropriate tags from pool?\n"
        "\n"
        "---\n"
        "\n"
        "# Description Rules\n"
        "\n"
        "## Attire\n"
        "Describe as natural descriptive phrases:\n"
        "- 'wearing a black tailored tuxedo with a white shirt and black necktie'\n"
        "- 'wearing a grey hoodie and black shorts'\n"
        "- Include accessories with 'with': 'with a silver necklace and leather belt'\n"
        "- Be specific on what's given. Never invent.\n"
        "\n"
        "## Hair\n"
        "Combine into single descriptive phrase:\n"
        "- 'long blonde wavy hair' / 'short messy black hair'\n"
        "- Include ornaments: 'with a blue ribbon hairband'\n"
        "\n"
        "## Body\n"
        "Key visual traits only: 'blue eyes', 'heterochromia', 'pale skin'.\n"
        "\n"
        "## Cleanup\n"
        "Remove from ALL output:\n"
        "- Atmosphere-as-trait: 'fox-like', 'robot-like', 'elegant atmosphere' → REMOVE\n"
        "- Filler: 'beautiful', 'stunning', 'gorgeous', 'detailed', 'intricate' → REMOVE\n"
        "- Context mismatches: items that don't fit the story situation\n"
        "\n"
        "---\n"
        "\n"
        "# Multi-Character Rules\n"
        "\n"
        "## Position & Placement (RULE 3 compliance)\n"
        "When a character's Tags start with a position phrase, copy it VERBATIM\n"
        "to the VERY START of that character's outfit field.\n"
        "Do NOT change, replace, paraphrase, or reinterpret it.\n"
        "Position phrases include:\n"
        "  'the girl on the left is', 'the boy on the right is',\n"
        "  'the character in the foreground is', 'the character on the monitor is',\n"
        "  'the character bottom is', 'the character center is',\n"
        "  'the person behind is', 'the girl in the center is', etc.\n"
        "Do NOT replace with count tags like '1girl'.\n"
        "Do NOT change 'bottom' to 'on the right'. Do NOT change 'center' to 'on the left'.\n"
        "For multi-character outfit sentences, use:\n"
        "  'the girl/boy who is wearing...'\n"
        "\n"
        "## Character Isolation\n"
        "Each character's output describes ONLY that character.\n"
        "Other character names must NOT appear.\n"
        "Position descriptors (where target character is placed) → ALWAYS preserve.\n"
        "NEVER add characters mentioned in Chat but not in input Tags. (RULE 5)\n"
        "\n"
        "## Character Name\n"
        "1. ALWAYS include Name exactly as provided.\n"
        "2. Do NOT add series/franchise names.\n"
        "3. If Name already in position descriptor, do NOT repeat.\n"
        "\n"
        "---\n"
        "\n"
        "# Scene Enhancement\n"
        "\n"
        "## Setup\n"
        "Refine shared Setup tags:\n"
        "- KEEP: camera, framing, angle, location, lighting, weather, time-of-day,\n"
        "  non-clothing props, character count tags (1girl, 1boy, 2girls, etc.)\n"
        "- Count tags → place at VERY START of setup.\n"
        "- REMOVE: filler/quality tags, atmosphere words.\n"
        "- Move held-but-not-worn items to Supplement.\n"
        "- When Chat describes physical contact, add interaction tag AFTER count tags:\n"
        "  backhug, hugging, holding hands, arm around shoulder, carrying,\n"
        "  piggyback, leaning on, headpat, handshake, linking arms, kissing, couple.\n"
        "\n"
        "## Supplement\n"
        "Refine into 1-2 SHORT natural language sentences:\n"
        "- Describe overall scene, atmosphere, key visual elements.\n"
        "- NEVER output 'none' or leave blank.\n"
        "- Keep under 30 words.\n"
        "- Do NOT change location, setting, characters, or spatial relationships.\n"
        "- Multi-character: MUST describe ALL characters and their positions.\n"
        "\n"
        "---\n"
        "\n"
        "# NSFW Tag Selection System\n"
        "You determine the scene's intimacy level from Chat narrative, Tags, and Setting.\n"
        "NO keyword matching — use context understanding.\n"
        "\n"
        "## Step 1: Scene Intensity Judgment\n"
        "Read the FULL context and classify EACH character's scene:\n"
        "- **Normal**: No sexual content. No intimate contact. Standard scene.\n"
        "- **Suggestive**: Erotic atmosphere, skin contact, revealing clothing,\n"
        "  sexual tension, partial exposure, intimate embrace, aroused state.\n"
        "  Example: girl pressing chest against boy's back while whispering in bed.\n"
        "- **Explicit**: Direct sexual activity described in Chat or Tags.\n"
        "\n"
        "Consider ALL of these signals (none alone is sufficient, weigh together):\n"
        "- Physical contact described in Chat (skin-to-skin, intimate touching)\n"
        "- Character emotional state (arousal, desire, vulnerability, excitement)\n"
        "- Clothing state in Tags (revealing, partial, missing underwear)\n"
        "- Setting intimacy (bed at night, alone together, confined space)\n"
        "- Narrative tension (romantic buildup, seduction, power dynamic)\n"
        "\n"
        "## Step 2: Tag Selection from Pools\n"
        "Based on intensity level, select appropriate tags from the pools below.\n"
        "Tags are inserted as-is (comma-separated), NOT converted to natural language.\n"
        "\n"
        "### Suggestive Pool (select 2-6 tags that match the scene context):\n"
        "**Body State**: no bra, nipple outline, covered nipples, pokies, sideboob,\n"
        "  underboob, cleavage, off-shoulder, bare shoulders, bare back, midriff,\n"
        "  navel, visible bra, visible panty line\n"
        "**Contact**: hugging from behind, breast press, clinging, lap pillow,\n"
        "  head on chest, arm around waist, holding hands, leaning on,\n"
        "  pressed against back, face pressed to back, holding arm,\n"
        "  arm between breasts, leg lock, intertwined fingers\n"
        "**Expression**: blushing, heavy breathing, flustered, embarrassed,\n"
        "  shy, aroused, sweating, trembling, heart pounding, shy smile,\n"
        "  nervous, vulnerable look, half-lidded eyes\n"
        "**Clothing State**: disheveled, clothes slightly pulled up, shirt lifted,\n"
        "  loose clothing, revealing, unbuttoned, slipping off shoulder,\n"
        "  oversized shirt, nothing underneath, transparent clothing\n"
        "\n"
        "### Explicit Pool (preserve ALL explicit tags from input + select from pool):\n"
        "**Body**: naked, nude, topless, bottomless, completely nude, bare breasts,\n"
        "  bare ass, spread legs, fully nude\n"
        "**Sexual Acts**: masturbation, cowgirl, doggystyle, missionary position,\n"
        "  fellatio, paizuri, handjob, fingering, mating press, fullnelson,\n"
        "  spooning, standing position, suspended congress, upright straddle,\n"
        "  breast caress, cunnilingus, rimming\n"
        "**Expression**: ahegao, tongue out, drooling, rolled eyes, half-closed eyes,\n"
        "  crying, tears, overwhelmed, moaning, panting, screaming, biting lip\n"
        "**Body State**: arched back, head back, grabbing sheets, vaginal fluid,\n"
        "  disheveled clothes, clothes pulled up, flushed, trembling violently\n"
        "\n"
        "## Step 3: Tag Insertion Rules\n"
        "1. **Normal scene** → no NSFW tags. Skip to outfit output.\n"
        "2. **Suggestive scene** → select tags from Suggestive Pool matching context.\n"
        "3. **Explicit scene** → preserve ALL explicit tags from input Tags,\n"
        "   then add from Explicit Pool as appropriate.\n"
        "4. **Input Tags containing pool tags** → ALWAYS preserve. Never remove.\n"
        "   'no bra' in input → 'no bra' in output. No exceptions.\n"
        "5. Do NOT over-select. Only tags clearly justified by the scene.\n"
        "6. NSFW tags are inserted as comma-separated tags, NOT natural language.\n"
        "7. These tags OVERRIDE RULE 1 (LESS IS MORE). NSFW tags are always essential.\n"
        "\n"
        "## NSFW Output Format\n"
        "When NSFW tags are present, outfit field becomes:\n"
        "  \"[position/count], wearing a [outfit phrase], [pose], [nsfw tags], "
        "[hair + body], [name]\"\n"
        "NSFW Supplement: When scene is Suggestive or Explicit, supplement may be\n"
        "up to 40 words with vivid atmosphere.\n"
        "\n"
        "---\n"
        "\n"
        "# Output Format\n"
        "Output ONLY a JSON object. No text before or after.\n"
        "If you must use a code block, use ```json.\n"
        "\n"
        "```json\n"
        "{\n"
        '  "characters": [\n'
        "    {\n"
        '      "name": "character_full_name",\n'
        '      "outfit": "[position/count], wearing a [outfit phrase], [pose], [hair + body], [name]",\n'
        '      "expression": "[natural language expression phrase]"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "[refined setup]",\n'
        '  "supplement": "[refined supplement]",\n'
        '  "reason": "[one-line memo for all characters]"\n'
        "}\n"
        "```\n"
        "\n"
        "## Outfit field order (5 parts):\n"
        "1. Position descriptor or count tag (FIRST)\n"
        "   - Multi-character: copy position phrase verbatim\n"
        "   - Single character: count tag (e.g., '1girl')\n"
        "2. Outfit as natural phrase (SECOND)\n"
        "   - 'wearing a [outfit description]'\n"
        "   - Include accessories with 'with'\n"
        "3. Pose (THIRD) — MANDATORY when present in Tags or determined by Phase 2a2\n"
        "   - e.g., 'lying on side', 'sitting on bed', 'standing'\n"
        "   - If no pose info exists anywhere → skip this part (do not guess)\n"
        "4. Body attribute phrase (FOURTH)\n"
        "   - Hair + eyes + body type combined: 'long blonde wavy hair, blue eyes'\n"
        "5. Character Name (LAST)\n"
        "\n"
        "Expression: natural descriptive phrase, NOT comma-separated tag dump.\n"
        "Setup: comma-separated scene tags.\n"
        "Supplement: 1-2 SHORT natural language sentences.\n"
        "Reason: brief one-line decision memo.\n"
        "\n"
        "Rules:\n"
        "1. No | character anywhere.\n"
        "2. No new weighted tags.\n"
        "3. Valid JSON only.\n"
        "4. Include ALL input characters.\n"
        "\n"
        "---\n"
        "\n"
        "# Examples\n"
        "\n"
        "## Example 1: Single character, outfit unchanged\n"
        "Input:\n"
        "Character 1: hasegawa kobato\n"
        "Tags: 1girl, kobato, gothic lolita dress, black hairband, mysterious atmosphere\n"
        "Setup: cowboy shot, classroom\n"
        "Supplement: night, moonlight\n"
        "Previous: 1girl, kobato, gothic lolita dress, black hairband, blonde hair, "
        "heterochromia\n"
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
        '      "expression": "calm, focused with a light smile"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "cowboy shot, classroom",\n'
        '  "supplement": "A quiet room lit by soft moonlight through the window.",\n'
        '  "reason": "outfit: unchanged (copied from previous). expression: reading '
        "calmly. setup/supplement: unchanged.\"\n"
        "}\n"
        "```\n"
        "\n"
        "## Example 2: Multi-character, outfits unchanged\n"
        "Input:\n"
        "Character 1: redfield stella\n"
        "Tags: the character in the foreground is redfield stella, girl, blue eyes, "
        "blonde hair, short hair, black tailored tuxedo, white shirt, black necktie\n"
        "Previous: the character in the foreground is redfield stella, wearing a black "
        "tailored tuxedo, short blonde hair, blue eyes\n"
        "Character 2: kashiwazaki sena\n"
        "Tags: the character on the monitor is kashiwazaki sena, girl, blonde hair, "
        "grey hoodie\n"
        "Previous: the character on the monitor is kashiwazaki sena, wearing a grey "
        "hoodie, blonde hair, aqua eyes\n"
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
        '      "expression": "calm smirk, watching the monitors"\n'
        "    },\n"
        "    {\n"
        '      "name": "kashiwazaki sena",\n'
        '      "outfit": "the character on the monitor is the girl who is wearing '
        "a grey hoodie, long blonde hair, aqua eyes, kashiwazaki sena\",\n"
        '      "expression": "nervous, wide-eyed, looking around cautiously"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "upper body, from behind, 2girls, interior, security room, '
        "monitor, night\",\n"
        '  "supplement": "A woman in a tuxedo watches security monitors showing '
        "a girl in a hoodie.\",\n"
        '  "reason": "outfit: both unchanged. expression: stella calm monitoring, '
        "sena nervous sneaking. setup/supplement: condensed.\"\n"
        "}\n"
        "```\n"
        "\n"
        "## Example 3: Outfit changed\n"
        "Input:\n"
        "Character 1: kashiwazaki sena\n"
        "Tags: 1girl, sena, school uniform, green jacket, plaid skirt\n"
        "Previous: 1girl, wearing a school uniform with a green jacket and plaid skirt, "
        "long blonde hair, aqua eyes\n"
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
        '      "outfit": "1girl, wearing a blue swimsuit, long blonde hair, '
        "aqua eyes, kashiwazaki sena\",\n"
        '      "expression": "embarrassed, blushing with nervous look"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "full body, pool",\n'
        '  "supplement": "Bright sunlight reflecting off the pool water.",\n'
        '  "reason": "outfit: changed to blue swimsuit per chat. expression: '
        "embarrassed at pool. setup: hallway→pool. supplement: updated.\"\n"
        "}\n"
        "```\n"
        "\n"
        "## Example 4: Pose preserved from Tags (lying)\n"
        "Input:\n"
        "Character 1: kashiwazaki sena\n"
        "Tags: the character on the right is kashiwazaki sena, girl, blonde hair, "
        "long hair, aqua eyes, butterfly hair ornament, navy blue t-shirt, "
        "lying on side, closed eyes, hugging arm\n"
        "Previous: the character on the right is kashiwazaki sena, wearing a "
        "navy blue t-shirt, lying on side, long blonde hair, aqua eyes\n"
        "Character 2: oreki houtarou\n"
        "Tags: the character on the left is oreki houtarou, boy, brown hair, "
        "short hair, green eyes, black cardigan, lying on side, looking at ceiling\n"
        "Previous: the character on the left is oreki houtarou, wearing a "
        "black cardigan, lying on side, short brown hair, green eyes\n"
        "Setup: from side, feet out of frame, 1girl, 1boy, bedroom, night\n"
        "Supplement: A narrow bed in a dark room.\n"
        "Chat: Sena fell asleep clinging to Hotaru's back on the narrow bed.\n"
        "\n"
        "Output:\n"
        "```json\n"
        "{\n"
        '  "characters": [\n'
        "    {\n"
        '      "name": "kashiwazaki sena",\n'
        '      "outfit": "the character on the right is the girl who is wearing '
        "a navy blue t-shirt with a butterfly hair ornament, lying on side, "
        "long blonde hair, aqua eyes, kashiwazaki sena\",\n"
        '      "expression": "peaceful, sleeping with a faint smile"\n'
        "    },\n"
        "    {\n"
        '      "name": "oreki houtarou",\n'
        '      "outfit": "the character on the left is the boy who is wearing '
        "a black cardigan, lying on side, short brown hair, green eyes, "
        "oreki houtarou\",\n"
        '      "expression": "tired, staring at the ceiling with a resigned look"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "from side, feet out of frame, 1girl, 1boy, bedroom, night",\n'
        '  "supplement": "Two figures lie on a narrow bed in a dark room, '
        "one asleep and one awake.\",\n"
        '  "reason": "outfit: both unchanged. pose: lying on side preserved '
        "from Tags for both. expression: sena peaceful sleep, houtarou tired "
        "awake.\"\n"
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

    # ─── Current Character Data ────────────────────
    # v5 방식: Previous Enhanced를 캐릭터별 인라인으로 배치
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
                f"  [{j}] {pc}" for j, pc in enumerate(char_data['previous_chars'], 1)
            )
            block.append(f"Recent outputs (newest first):\n{prev_text}")

        parts.append("\n".join(block))

    # ─── Scene Context ─────────────────────────────
    if setup:
        parts.append(f"### Setup\n{setup}")

    if supplement:
        parts.append(f"### Supplement\n{supplement}")

    if chat:
        parts.append(f"### Chat\n{chat}")

    # ─── Image Insertion Point ─────────────────────
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
    else:
        # SLOT 없음: 복장 혼선 방지 가드
        parts.append(
            "### NO IMAGE INSERTION POINT\n"
            "No specific image insertion point is provided.\n"
            "CRITICAL RESTRICTIONS when no insertion point exists:\n"
            "- Outfit MUST come from Tags ONLY. Zero Chat influence on outfit.\n"
            "- Chat may ONLY be used for: expression (emotion, facial expression) and scene mood.\n"
            "- Do NOT add, change, infer, or mix any clothing items based on Chat narrative.\n"
            "- If Tags say 'black cardigan' → output 'black cardigan' only.\n"
            "  Do NOT add 'white t-shirt' or 'shorts' just because Chat implies them.\n"
            "- Each character's outfit is strictly isolated from other characters' outfits.\n"
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

