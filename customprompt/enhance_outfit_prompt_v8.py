"""
복장 프롬프트 강화 - v8
색상/하의 일관성 수정 + Python 후처리 검증

v7와의 차이:
- RULE 0: OUTFIT CONSISTENCY 추가 (Previous Enhanced 최우선)
- RULE 3 확장: EXACT NOUN PRESERVATION (복합 명사 분리 금지)
- RULE 4 수정: position descriptor는 CURRENT Tags에서만 가져옴
- PRE-OUTPUT CHECK 확장: framing 확인, eye color 확인
- Ex 5 추가: VERBATIM copy + framing filter 결합 예시
- Python 후처리: _postprocess_outfit (색상 드리프트, 하의 누락 보완)
- NO IMAGE INSERTION POINT 블록: Previous Enhanced를 PRIMARY로 인정

필수 함수:
    async def run_all(
        characters: list[dict],
        setup: str = "",
        supplement: str = "",
        chat: str = "",
        slot: str = "",
    ) -> dict

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
    ※ outfit_only = 후처리 전 LLM 원본 (history/_last_enhanced_block용)
    ※ char = 후처리 후 출력값 (실제 프롬프트용)
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


# ─── System Prompt (v8 — outfit consistency + exact noun preservation) ──

def _build_system_prompt() -> str:
    return (
        "# Visual Description Refiner\n"
        "You enhance character appearance descriptions for AI image generation.\n"
        "Process ALL characters in a SINGLE output using structured analysis.\n"
        "\n"

        "# RULES\n"
        "\n"

        "## RULE 0: OUTFIT CONSISTENCY (HIGHEST PRIORITY)\n"
        "Previous Enhanced is the PRIMARY source for outfit. Tags is SECONDARY.\n"
        "\n"
        "### Priority Order (highest → lowest):\n"
        "1. Previous Enhanced outfit — VERBATIM copy unless Chat explicitly "
        "describes an outfit change BEFORE the Image Insertion Point.\n"
        "2. Current Tags — used ONLY for: outfit change detection, position "
        "descriptors, pose, body attributes.\n"
        "3. Framing filter — remove items NOT VISIBLE per camera framing, but "
        "PRESERVE colors of remaining items.\n"
        "\n"
        "### Tag-Anchored Color Lock:\n"
        "- If Tags contain a color for an item → Tags color takes priority.\n"
        "- If Tags do NOT contain a color for an item → keep Previous Enhanced "
        "color.\n"
        "- Example: Previous 'blue blouse', Tags 'blouse' (no color) → keep "
        "'blue blouse'.\n"
        "- Example: Previous 'blue blouse', Tags 'pink blouse' → use 'pink "
        "blouse' (Tag wins).\n"
        "\n"
        "### Consistency Rules:\n"
        "- Items in Previous Enhanced but NOT in Tags are NOT invented — they "
        "are known outfit → keep them.\n"
        "- When Chat does NOT describe outfit change → copy Previous Enhanced "
        "VERBATIM (including colors).\n"
        "- Framing filter ONLY removes invisible items. Never changes colors "
        "or adjectives.\n"
        "\n"

        "## RULE 1: NAME RESTORATION\n"
        "Place the Name field value at the VERY END of each character's outfit field.\n"
        "If Tags contain only a partial name or abbreviation, restore the FULL name "
        "from the Name field.\n"
        "  Example: Tags 'sena' + Name 'sena kashiwazaki' → use 'sena kashiwazaki'\n"
        "If name order (surname↔given) differs between Tags/Chat and Name field, "
        "use the Name field order.\n"
        "  Example: Tags 'kashiwazaki sena' + Name 'sena kashiwazaki' → "
        "'sena kashiwazaki'\n"
        "Use context understanding to identify and restore character names.\n"
        "\n"

        "## RULE 2: LESS IS MORE\n"
        "Remove unnecessary details. Keep only essential visual elements.\n"
        "Output should be SHORTER or similar length to input — never longer.\n"
        "Every word must earn its place. Shorter prompts = better images.\n"
        "EXCEPTION: NSFW/intimate tags and pose/body position tags are ALWAYS essential.\n"
        "Never remove: pose tags (lying, sitting, standing, kneeling, bending), "
        "NSFW tags (no bra, nipple outline, breast press, off-shoulder, etc.),\n"
        "or any tags selected from the NSFW Tag System pools below.\n"
        "\n"

        "## RULE 3: NEVER INVENT DETAILS + EXACT NOUN PRESERVATION\n"
        "NEVER add details NOT in the input tags.\n"
        "- Do NOT add fabric, material, texture, or fit adjectives.\n"
        "  'grey hoodie' stays 'grey hoodie' — NOT 'grey cotton oversized hoodie'.\n"
        "  'black tuxedo' stays 'black tuxedo' — NOT 'black tailored cotton tuxedo'.\n"
        "- Do NOT add style modifiers. 'skirt' stays 'skirt' — NOT 'pleated skirt'.\n"
        "- Do NOT add color variations. 'blue' stays 'blue' — NOT 'sky-blue'.\n"
        "- If Chat describes an outfit change, use ONLY details Chat provides.\n"
        "  Chat says 'grey hoodie and black shorts' → output 'grey hoodie, black shorts'.\n"
        "  Do NOT add 'cotton', 'polyester', 'oversized', etc.\n"
        "- Remove filler: 'beautiful', 'stunning', 'gorgeous', 'detailed', 'intricate'.\n"
        "- Remove atmosphere-as-trait: 'fox-like', 'robot-like', 'elegant atmosphere'.\n"
        "\n"
        "### EXACT NOUN PRESERVATION:\n"
        "When copying outfit items from Previous Enhanced or Tags, copy the FULL "
        "compound noun EXACTLY.\n"
        "- Do NOT shorten: 'grey zip-up hoodie' stays 'grey zip-up hoodie' "
        "— NEVER 'grey hoodie' or 'grey zip hoodie'.\n"
        "- Do NOT rephrase: 'black training pants' stays 'black training pants' "
        "— NEVER 'track pants', 'dark pants', or 'grey training shorts'.\n"
        "- Do NOT split compounds: 'pastel pink blouse' stays 'pastel pink blouse' "
        "— NEVER 'pink blouse' or 'pastel blouse'.\n"
        "- Word order MUST be preserved exactly.\n"
        "- Hyphenated compounds (zip-up, off-shoulder, thigh-high) MUST keep "
        "their hyphens.\n"
        "\n"

        "## RULE 4: POSITION DESCRIPTOR PRESERVATION\n"
        "When CURRENT Tags start with a position phrase, copy it to the VERY START "
        "of outfit VERBATIM.\n"
        "Position descriptors come from CURRENT Tags ONLY — never from Previous "
        "Enhanced.\n"
        "If current Tags have NO position phrase → omit (do not reuse from Previous).\n"
        "Do NOT change, replace, paraphrase, or reinterpret it.\n"
        "Do NOT replace with count tags like '1girl'.\n"
        "Position phrases: 'the girl on the left is', 'the character bottom is',\n"
        "  'the character in the foreground is', 'the character center is', etc.\n"
        "For multi-character outfit sentences, use:\n"
        "  'the girl/boy who is wearing...'\n"
        "CORRECT: 'the character bottom is X' → "
        "'the character bottom is the boy who...'\n"
        "WRONG: 'the character bottom is X' → "
        "'the character on the right is...'\n"
        "\n"

        "## RULE 5: CHARACTER COUNT LOCK\n"
        "Output EXACTLY the same characters as input. No more, no fewer.\n"
        "NEVER add characters mentioned in Chat but not in input Tags.\n"
        "Chat is context ONLY — not a character source.\n"
        "Each character's output describes ONLY that character. "
        "Other character names must NOT appear.\n"
        "\n"

        "## RULE 6: NUDITY STATE PRESERVATION\n"
        "When Tags indicate an unclothed state (naked, nude, bare, topless, bottomless, "
        "undressed, in underwear only), do NOT add clothing.\n"
        "The character MUST remain in that unclothed state.\n"
        "Clothing may only appear if Tags themselves include clothing items,\n"
        "or Chat describes putting on clothes BEFORE the Image Insertion Point (RULE 7).\n"
        "\n"

        "## RULE 7: TEMPORAL AWARENESS — INSERTION POINT BOUNDARY\n"
        "The image depicts the scene at the EXACT MOMENT of the Image Insertion Point.\n"
        "Only use events, states, and details at or BEFORE this point.\n"
        "Chat AFTER the insertion point = FUTURE events = IGNORE.\n"
        "This applies to ALL visual elements: outfits, expressions, poses, actions, "
        "held items, location changes, and character states.\n"
        "If unsure whether an event is before or after → default to Tags only.\n"
        "\n"

        "## PRE-OUTPUT CHECK\n"
        "Before writing JSON, verify ALL of the following:\n"
        "1. Correct character count (same as input).\n"
        "2. No invented details (fabric/fit/color not in source).\n"
        "3. Names match Name field order.\n"
        "4. Unclothed states preserved.\n"
        "5. Position descriptors verbatim from CURRENT Tags only.\n"
        "6. Framing filter applied: invisible items removed, colors preserved.\n"
        "7. Previous Enhanced outfit copied VERBATIM unless Chat describes "
        "outfit change.\n"
        "8. Eye color matches Previous Enhanced (unless Tags specify different).\n"
        "9. Compound nouns preserved exactly (no shortening or rephrasing).\n"
        "\n"

        "---\n"
        "\n"

        "# FORMAT GUIDE\n"
        "\n"

        "## SFW Output (Normal Scene)\n"
        "Outfit: natural language phrase with tags mixed\n"
        "  'wearing a grey hoodie and black shorts'\n"
        "  'wearing a black tailored tuxedo with a white shirt and black necktie'\n"
        "  Include accessories with 'with': "
        "'with a silver necklace and leather belt'\n"
        "Expression: natural descriptive phrase (NOT comma-separated tag dump)\n"
        "  'gentle smile with downcast eyes', 'calm, focused with a light smile'\n"
        "  Covers EMOTION and FACE only. Do NOT include pose/action in expression.\n"
        "Hair: single descriptive phrase — 'long blonde wavy hair', "
        "'short messy black hair'. Include ornaments: 'with a blue ribbon hairband'\n"
        "Body: key visual traits only — 'blue eyes', 'heterochromia', 'pale skin'\n"
        "Supplement: 1-2 short natural language sentences (under 30 words)\n"
        "\n"

        "## NSFW Output (Suggestive/Explicit Scene)\n"
        "Outfit: comma-separated tags (NOT natural language)\n"
        "  'topless, naked upper body, nipples, grey cotton dolphin pants'\n"
        "Expression: comma-separated tags\n"
        "  'intense blush, parted lips, half-lidded eyes'\n"
        "Supplement: vivid atmosphere description (up to 40 words)\n"
        "NSFW tags OVERRIDE RULE 2 (LESS IS MORE). NSFW tags are always essential.\n"
        "\n"

        "---\n"
        "\n"

        "# Pipeline: Step 1 → Step 2\n"
        "\n"

        "## Step 1: Analyze\n"
        "Read ALL inputs before writing anything:\n"
        "- **Image Insertion Point** → exact narrative moment being depicted.\n"
        "  Establish BEFORE/AFTER boundary. Only use content at or BEFORE. (RULE 7)\n"
        "- **Chat BEFORE insertion** → current emotions, outfit changes, scene mood.\n"
        "  What emotions are characters feeling at this exact moment?\n"
        "- **Chat AFTER insertion** → IGNORE for this image.\n"
        "- **Setup camera/framing** → what body parts are VISIBLE:\n"
        "  | Framing | Visible | NOT visible |\n"
        "  |---------|---------|-------------|\n"
        "  | close-up / portrait / face focus | face, hair, accessories | shoes, "
        "lower body, held items below frame |\n"
        "  | upper body / bust shot | waist up, top clothing | shoes, skirt hem "
        "below frame |\n"
        "  | cowboy shot (mid-thigh) | most of body | shoes at frame edge only |\n"
        "  | full body | everything | nothing excluded |\n"
        "  | from above / high angle | upper body emphasis | feet likely cut off |\n"
        "  Only exclude items when framing CLEARLY makes them invisible.\n"
        "- **Supplement** → scene atmosphere and environment.\n"
        "\n"

        "Per-character decisions:\n"
        "- **Pose**: Tags primary. If Tags have clear pose → use as-is.\n"
        "  Exception: Tags have NO pose, AND Chat EXPLICITLY describes body position → "
        "use Chat pose. Otherwise → omit.\n"
        "  If unsure whether Chat clearly describes pose → default to Tags only.\n"
        "- **Outfit**:\n"
        "  No change in Chat before insertion → COPY Previous Enhanced VERBATIM.\n"
        "    This includes ALL color adjectives — 'blue blouse' stays 'blue blouse',\n"
        "    never drop or change colors. If Previous says 'grey hoodie', "
        "output 'grey hoodie'.\n"
        "  Outfit change before insertion → describe using ONLY Chat/Tag details.\n"
        "    For UNCHANGED items, carry their colors from Previous Enhanced.\n"
        "    Only NEW items get colors from Chat/Tags.\n"
        "  Outfit change after insertion → IGNORE. (RULE 7)\n"
        "  Unclothed in Tags → preserve that state. (RULE 6)\n"
        "- **Expression**: ALWAYS FRESH from current scene mood. No carryover.\n"
        "- **NSFW level**: Classify as Normal / Suggestive / Explicit. "
        "See NSFW Tag System below.\n"
        "\n"

        "## Step 2: Output\n"
        "Produce the JSON output directly.\n"
        "For each character, outfit field order (5 parts):\n"
        "1. Position descriptor or count tag (FIRST)\n"
        "   - Multi-character: copy position phrase verbatim from Tags\n"
        "   - Single character: count tag (e.g., '1girl')\n"
        "2. Outfit description (SECOND)\n"
        "   - SFW: natural language phrase — 'wearing a [outfit description]'\n"
        "   - NSFW: comma-separated tags\n"
        "3. Pose (THIRD) — MANDATORY when present in Tags or determined in Step 1.\n"
        "   If no pose info exists → skip (do not guess).\n"
        "4. Body attribute phrase (FOURTH)\n"
        "   - Hair + eyes + body type combined: 'long blonde wavy hair, blue eyes'\n"
        "5. Character full name (LAST) — from Name field, per RULE 1.\n"
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
        "- Do NOT change location, setting, characters, or spatial relationships.\n"
        "- Multi-character: MUST describe ALL characters and their positions.\n"
        "\n"

        "---\n"
        "\n"

        "# NSFW Tag Selection System\n"
        "Determine scene intimacy from Chat narrative, Tags, and Setting.\n"
        "NO keyword matching — use context understanding.\n"
        "\n"

        "## Scene Intensity Classification\n"
        "Classify EACH character's scene:\n"
        "- **Normal**: No sexual content. No intimate contact. Standard scene.\n"
        "- **Suggestive**: Erotic atmosphere, skin contact, revealing clothing,\n"
        "  sexual tension, partial exposure, intimate embrace, aroused state.\n"
        "- **Explicit**: Direct sexual activity described in Chat or Tags.\n"
        "\n"

        "Consider ALL signals together (none alone is sufficient):\n"
        "- Physical contact (skin-to-skin, intimate touching)\n"
        "- Character emotional state (arousal, desire, vulnerability)\n"
        "- Clothing state in Tags (revealing, partial, missing underwear)\n"
        "- Setting intimacy (bed at night, alone together, confined space)\n"
        "- Narrative tension (romantic buildup, seduction, power dynamic)\n"
        "\n"

        "## Suggestive Pool (select 2-6 tags matching context):\n"
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

        "## Explicit Pool (preserve ALL explicit input tags + select from pool):\n"
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

        "## Tag Insertion Rules\n"
        "1. Normal scene → no NSFW tags.\n"
        "2. Suggestive → select matching tags from Suggestive Pool.\n"
        "3. Explicit → preserve ALL explicit input tags + add from Explicit Pool.\n"
        "4. Input Tags containing pool tags → ALWAYS preserve. No exceptions.\n"
        "5. Do NOT over-select. Only tags clearly justified by the scene.\n"
        "6. NSFW tags as comma-separated tags, NOT natural language.\n"
        "7. NSFW tags OVERRIDE RULE 2 (LESS IS MORE). Always essential.\n"
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
        '      "outfit": "[position/count], wearing a [outfit phrase], [pose], '
        '[hair + body], [name]",\n'
        '      "expression": "[natural language expression phrase]"\n'
        "    }\n"
        "  ],\n"
        '  "setup": "[refined setup]",\n'
        '  "supplement": "[refined supplement]",\n'
        '  "reason": "[one-line memo for all characters]"\n'
        "}\n"
        "```\n"
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

        "## Ex 1: Single character, outfit unchanged\n"
        "Tags: 1girl, kobato, gothic lolita dress, black hairband, mysterious "
        "atmosphere | Previous: 1girl, kobato, gothic lolita dress, black "
        "hairband, blonde hair, heterochromia | Chat: Kobato was reading "
        "quietly.\n"
        "→ {\"name\":\"hasegawa kobato\", "
        "\"outfit\":\"1girl, wearing a gothic lolita dress with a black "
        "hairband, blonde hair, heterochromia, hasegawa kobato\", "
        "\"expression\":\"calm, focused with a light smile\"}\n"
        "  Setup: cowboy shot, classroom | Supplement: A quiet room lit by "
        "soft moonlight.\n"
        "\n"

        "## Ex 2: Multi-character, outfits unchanged\n"
        "Char1 Tags: the character in the foreground is redfield stella, girl, "
        "blue eyes, blonde hair, short hair, black tailored tuxedo, white "
        "shirt, black necktie\n"
        "Char2 Tags: the character on the monitor is kashiwazaki sena, girl, "
        "blonde hair, grey hoodie | Chat: Stella watched monitors calmly. "
        "Sena sneaked through hallway.\n"
        "→ Char1: {\"name\":\"redfield stella\", \"outfit\":\"the character "
        "in the foreground is the girl who is wearing a black tailored tuxedo "
        "with a white shirt and black necktie, short blonde hair, blue eyes, "
        "redfield stella\", \"expression\":\"calm smirk, watching monitors\"}\n"
        "→ Char2: {\"name\":\"kashiwazaki sena\", \"outfit\":\"the character "
        "on the monitor is the girl who is wearing a grey hoodie, long "
        "blonde hair, aqua eyes, kashiwazaki sena\", "
        "\"expression\":\"nervous, wide-eyed, looking around cautiously\"}\n"
        "  Setup: upper body, from behind, 2girls, interior, security room, "
        "night | Supplement: A woman in a tuxedo watches monitors.\n"
        "\n"

        "## Ex 3: Outfit changed\n"
        "Tags: 1girl, sena, school uniform, green jacket, plaid skirt | "
        "Previous: 1girl, wearing a school uniform with a green jacket and "
        "plaid skirt, long blonde hair, aqua eyes\n"
        "Chat: Sena went to the pool and changed into a blue swimsuit.\n"
        "→ {\"name\":\"kashiwazaki sena\", \"outfit\":\"1girl, wearing a blue "
        "swimsuit, long blonde hair, aqua eyes, kashiwazaki sena\", "
        "\"expression\":\"embarrassed, blushing with nervous look\"}\n"
        "  Setup: full body, pool | Supplement: Bright sunlight reflecting "
        "off pool water.\n"
        "\n"

        "## Ex 4: Pose preserved, multi-character\n"
        "Char1 Tags: the character on the right is kashiwazaki sena, girl, "
        "blonde hair, long hair, aqua eyes, butterfly hair ornament, navy "
        "blue t-shirt, lying on side, closed eyes, hugging arm\n"
        "Char2 Tags: the character on the left is oreki houtarou, boy, brown "
        "hair, short hair, green eyes, black cardigan, lying on side, "
        "looking at ceiling\n"
        "Chat: Sena fell asleep clinging to Hotaru's back on the narrow bed.\n"
        "→ Char1: {\"name\":\"kashiwazaki sena\", \"outfit\":\"the character "
        "on the right is the girl who is wearing a navy blue t-shirt with a "
        "butterfly hair ornament, lying on side, long blonde hair, aqua eyes, "
        "kashiwazaki sena\", \"expression\":\"peaceful, sleeping with a faint "
        "smile\"}\n"
        "→ Char2: {\"name\":\"oreki houtarou\", \"outfit\":\"the character "
        "on the left is the boy who is wearing a black cardigan, lying on "
        "side, short brown hair, green eyes, oreki houtarou\", "
        "\"expression\":\"tired, staring at the ceiling with a resigned "
        "look\"}\n"
        "  Setup: from side, feet out of frame, 1girl, 1boy, bedroom, night "
        "| Supplement: Two figures lie on a narrow bed, one asleep, one "
        "awake.\n"
        "\n"

        "## Ex 5: VERBATIM copy with framing filter (outfit unchanged, upper body)\n"
        "Tags: 1girl, sena, girl, blonde hair, long hair, aqua eyes, pastel "
        "pink blouse, navy blue skirt, white thigh-high socks, black shoes\n"
        "Previous: 1girl, wearing a pastel pink blouse and navy blue skirt "
        "with white thigh-high socks and black shoes, long blonde hair, aqua "
        "eyes, kashiwazaki sena\n"
        "Setup: upper body, portrait | Chat: Sena was talking "
        "enthusiastically about her new game.\n"
        "→ {\"name\":\"kashiwazaki sena\", \"outfit\":\"1girl, wearing a "
        "pastel pink blouse, long blonde hair, aqua eyes, kashiwazaki "
        "sena\", \"expression\":\"enthusiastic, bright smile with sparkling "
        "eyes\"}\n"
        "  NOTE: upper body framing → shoes, socks, skirt removed. Blouse "
        "color 'pastel pink' preserved VERBATIM from Previous. Eye color "
        "'aqua' preserved. No color drift.\n"
        "\n"

        "## Ex 6: VERBATIM copy with framing filter (compound noun "
        "preservation)\n"
        "Tags: 1boy, oreki, boy, brown hair, short hair, green eyes, grey "
        "zip-up hoodie, black training pants, slumped posture\n"
        "Previous: 1boy, wearing a grey zip-up hoodie and black training "
        "pants, short brown hair, green eyes, oreki houtarou\n"
        "Setup: cowboy shot, cafe, day | Chat: Oreki slumped deeper into "
        "his chair, staring at the menu with zero motivation.\n"
        "→ {\"name\":\"oreki houtarou\", \"outfit\":\"1boy, wearing a grey "
        "zip-up hoodie and black training pants, slumped posture, short "
        "brown hair, green eyes, oreki houtarou\", \"expression\":\"tired, "
        "half-lidded eyes with resigned look\"}\n"
        "  NOTE: 'grey zip-up hoodie' and 'black training pants' copied "
        "VERBATIM — compound nouns NOT shortened or rephrased.\n"
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
                f"Previous Enhanced (PRIMARY OUTFIT SOURCE — copy outfit "
                f"VERBATIM, generate NEW expression): "
                f"{char_data['previous_enhanced']}"
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
        # SLOT 없음: Previous Enhanced를 PRIMARY로, Tags를 SECONDARY로
        parts.append(
            "### NO IMAGE INSERTION POINT\n"
            "No specific image insertion point is provided.\n"
            "OUTFIT CONSISTENCY (RULE 0 still applies):\n"
            "- Outfit PRIMARY source: Previous Enhanced (VERBATIM copy for "
            "unchanged items).\n"
            "- Outfit SECONDARY source: Tags (for new items not in Previous, "
            "position descriptors, pose).\n"
            "- Chat may ONLY be used for: expression (emotion, facial "
            "expression) and scene mood.\n"
            "- Do NOT add, change, infer, or mix clothing items based on "
            "Chat narrative.\n"
            "- If Tags say 'black cardigan' and Previous says 'black "
            "cardigan and grey pants' → keep BOTH from Previous.\n"
            "- Each character's outfit is strictly isolated from other "
            "characters' outfits.\n"
        )

    return "\n\n".join(parts)


# ─── Post-Processing (v8 색상/하의 일관성 보완) ──────────

_ALL_COLORS = frozenset({
    'aqua', 'beige', 'black', 'blue', 'brown', 'burgundy', 'charcoal',
    'coral', 'cream', 'crimson', 'cyan', 'gold', 'golden', 'grey', 'gray',
    'green', 'indigo', 'ivory', 'khaki', 'lavender', 'lime', 'magenta',
    'maroon', 'navy', 'olive', 'orange', 'pink', 'plum', 'purple', 'red',
    'rose', 'salmon', 'silver', 'teal', 'turquoise', 'violet', 'white',
    'yellow',
})

_COLOR_MODS = frozenset({
    'pastel', 'dark', 'light', 'bright', 'deep', 'pale', 'warm', 'cool',
})

_CLOTHING_TOP = frozenset({
    'hoodie', 'jacket', 'shirt', 'blouse', 'cardigan', 'sweater', 'dress',
    'tuxedo', 'coat', 'uniform', 'vest', 'ribbon', 'necktie', 'scarf',
    't-shirt',
})

_CLOTHING_BOTTOM = frozenset({
    'pants', 'shorts', 'skirt', 'jeans', 'trousers', 'leggings',
    'joggers', 'sweatpants',
})

_CLOTHING_ALL = _CLOTHING_TOP | _CLOTHING_BOTTOM

_FACE_FRAMING = frozenset({
    'close-up', 'portrait', 'face focus', 'bust shot', 'head shot',
})

_UPPER_FRAMING = frozenset({
    'upper body',
})


def _extract_color_from_phrase(phrase: str) -> str | None:
    """Extract color phrase from text like 'pastel pink blouse' → 'pastel pink'."""
    words = phrase.split()
    color_parts = []
    i = 0
    while i < len(words):
        w = words[i].lower().rstrip(',.')
        if w in _COLOR_MODS:
            color_parts.append(words[i])
            i += 1
        elif w in _ALL_COLORS:
            color_parts.append(words[i])
            break
        else:
            break
    return ' '.join(color_parts) if color_parts else None


def _extract_outfit_items(text: str) -> dict[str, str]:
    """Extract {item_text: color_phrase} from outfit text.

    Examples:
        "wearing a pastel pink blouse and black training pants"
        → {"blouse": "pastel pink", "training pants": "black"}
    """
    items = {}
    words = text.replace(',', ' ').replace('.', ' ').split()
    i = 0
    while i < len(words):
        w = words[i].lower()

        # Check for color modifier + color
        color_parts = []
        if w in _COLOR_MODS:
            if i + 1 < len(words) and words[i + 1].lower() in _ALL_COLORS:
                color_parts = [words[i], words[i + 1]]
                i += 2
            else:
                i += 1
                continue
        elif w in _ALL_COLORS:
            color_parts = [words[i]]
            i += 1
            # Check for compound color (e.g., "navy blue")
            if i < len(words) and words[i].lower() in _ALL_COLORS:
                color_parts.append(words[i])
                i += 1
        else:
            i += 1
            continue

        # Collect item words after color
        item_parts = []
        while i < len(words):
            iw = words[i].lower()
            if iw in ('and', 'with', 'a', 'an', 'the', 'in', 'of'):
                break
            item_parts.append(words[i])
            i += 1
            if iw in _CLOTHING_ALL:
                break

        if item_parts:
            last_lower = item_parts[-1].lower()
            if last_lower in _CLOTHING_ALL:
                color_phrase = ' '.join(p for p in color_parts)
                item_text = ' '.join(item_parts).lower()
                items[item_text] = color_phrase

    return items


def _postprocess_outfit(
    previous_enhanced: str,
    current_output: str,
    setup: str,
) -> tuple[str, list[str]]:
    """LLM 출력 후처리: 색상 드리프트, 하의 누락 보완.

    Returns:
        (processed_output, warnings)
    """
    if not previous_enhanced or not current_output:
        return current_output, []

    output = current_output
    warnings = []

    # ── 1. Eye color lock ──────────────────────────────
    _SKIP_WORDS = frozenset({
        'the', 'her', 'his', 'with', 'their', 'and',
        'bright', 'sparkling', 'beautiful', 'deep',
    })
    prev_eye = re.search(
        r'\b([a-zA-Z]\w*)\s+eyes\b', previous_enhanced, re.IGNORECASE
    )
    if prev_eye and prev_eye.group(1).lower() not in _SKIP_WORDS:
        prev_color = prev_eye.group(1)
        curr_eye = re.search(
            r'\b([a-zA-Z]\w*)\s+eyes\b', output, re.IGNORECASE
        )
        if curr_eye and curr_eye.group(1).lower() not in _SKIP_WORDS:
            if curr_eye.group(1).lower() != prev_color.lower():
                warnings.append(
                    f"eye_color_drift: {prev_color}→{curr_eye.group(1)}"
                )
                output = (
                    output[:curr_eye.start(1)]
                    + prev_color
                    + output[curr_eye.end(1):]
                )

    # ── 2. Clothing color lock ─────────────────────────
    prev_items = _extract_outfit_items(previous_enhanced)
    curr_items = _extract_outfit_items(output)

    for item_key, prev_color in prev_items.items():
        if item_key in curr_items:
            curr_color = curr_items[item_key]
            # Check any word in color phrase is a known color
            prev_has_color = any(
                w.lower() in _ALL_COLORS for w in prev_color.split()
            )
            if (
                curr_color.lower() != prev_color.lower()
                and prev_has_color
            ):
                warnings.append(
                    f"color_drift: {prev_color} {item_key}→"
                    f"{curr_color} {item_key}"
                )
                # Replace in output: find curr_color + item and replace
                pattern = (
                    r'\b' + re.escape(curr_color) + r'\s+'
                    + re.escape(item_key) + r'\b'
                )
                replacement = f"{prev_color} {item_key}"
                output, count = re.subn(
                    pattern, replacement, output,
                    flags=re.IGNORECASE, count=1,
                )

    # ── 3. Missing bottom clothing ─────────────────────
    prev_has_bottom = any(
        b in previous_enhanced.lower() for b in _CLOTHING_BOTTOM
    )
    curr_has_bottom = any(
        b in output.lower() for b in _CLOTHING_BOTTOM
    )

    if prev_has_bottom and not curr_has_bottom:
        setup_lower = (setup or '').lower()
        is_face = any(f in setup_lower for f in _FACE_FRAMING)
        is_upper = any(f in setup_lower for f in _UPPER_FRAMING)

        if not is_face and not is_upper:
            # Extract only bottom items from previous (not entire segment)
            prev_items = _extract_outfit_items(previous_enhanced)
            bottom_items = {
                k: v for k, v in prev_items.items()
                if any(b in k for b in _CLOTHING_BOTTOM)
            }
            if bottom_items:
                parts = [f"{v} {k}" for k, v in bottom_items.items()]
                bottom_text = ' and '.join(parts)
                # Insert before character name (last comma segment)
                last_comma = output.rfind(',')
                if last_comma > 0:
                    output = (
                        output[:last_comma]
                        + ', ' + bottom_text
                        + output[last_comma:]
                    )
                    warnings.append(f"bottom_added: {bottom_text}")

    return output, warnings


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

        # v8: 후처리 전 outfit 보존 (history/_last_enhanced_block용)
        outfit_raw = outfit

        # v8: 색상 드리프트, 하의 누락 보완
        previous = char_data.get("previous_enhanced", "")
        outfit, pp_warnings = _postprocess_outfit(previous, outfit, setup)
        if pp_warnings:
            _log_prompt_io(name, {"postprocess_warnings": pp_warnings}, outfit)

        # combined 조합 (후처리 후 outfit 사용)
        combined = f"{outfit}, {expression}" if expression else outfit

        char_results.append({
            "name": name,
            "char": combined,            # 후처리 후 (실제 출력)
            "outfit_only": outfit_raw,    # 후처리 전 (LLM 원본)
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
