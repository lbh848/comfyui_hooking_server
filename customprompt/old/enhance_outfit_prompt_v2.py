"""
프롬프트 강화 - 캐릭터 복장 일관성 검사/보정 + 상황별 표정 결정

이 파일은 customprompt/ 폴더에 넣고 설정에서 선택하면
배치 프롬프트 강화 모드에서 자동으로 실행됩니다.

필수 함수:
    async def run(character_name: str, char_block: str, outfit_result: str = None,
                  chat_content: str = "", previous_chat: str = "",
                  previous_enhanced: str = "") -> tuple[str, str]

반환값:
    tuple[str, str]: (combined_block, outfit_only_block)
        combined_block: 복장 + 표정 조합 (최종 프롬프트용)
        outfit_only_block: 복장만 (일관성 추적용)

주의:
- callLLM이 service/model 설정을 자동으로 처리합니다. (재시도 없음, 단일 시도)
- 프롬프트는 영어, 간단한 명령형 문장으로 작성
- 복장은 일관성 보장을 위해 다음 라운드로 전달됨
- 표정은 매번 현재 맥락 기반으로 새로 생성됨 (전달 안됨)
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
        "You are a character outfit CONSISTENCY CHECKER and detail enhancer\n"
        "for AI image generation prompts.\n"
        "You have TWO jobs:\n"
        "1. Keep character outfits visually consistent across consecutive images.\n"
        "2. Determine the most appropriate facial expression based on the current scene context.\n"
        "\n"
        "### INPUTS (each maps to a ### section in the user message)\n"
        "- **Character Name** (### Character Name): the character's authoritative full name.\n"
        "- **Current Character Block** (### Current Character Block): "
        "current character description from the image generation prompt.\n"
        "- **Scene Context** (### Scene Context): camera, location, and setting tags "
        "from the [Positive] section.\n"
        "- **Outfit Analysis Result** (### Outfit Analysis Result (ground truth)): "
        "previous outfit analysis as JSON.\n"
        '  Format: {"headgear":"...", "clothing":"...", "shoes":"...", '
        '"worn_accessories":"...", "held_items":"...", "memo":"..."}\n'
        "- **Current Chat Context** (### Current Chat Context): "
        "current chat describing what is happening NOW.\n"
        "- **Image Insertion Point Context** (### Image Insertion Point Context): "
        "CRITICAL — shows exactly WHERE in the story the image is being generated.\n"
        "  The image depicts the scene at this exact position. Read the text BEFORE and AFTER\n"
        "  the insertion point to determine:\n"
        "  * Whether the character has ALREADY changed clothes or is ABOUT TO change.\n"
        "  * The emotional state at this exact moment in the story.\n"
        "  * What outfit and expression make sense at this specific point.\n"
        "- **Previous Enhanced Output** (### Previous Enhanced Output): "
        "previous enhanced outfit description (outfit only, no expression).\n"
        "- **Previous Chat Context** (### Previous Chat Context): "
        "chat from around when the outfit was last tracked, centered on the relevant position.\n"
        "\n"
        "### ITEM CLASSIFICATION\n"
        "Every outfit item belongs to one of these categories:\n"
        "- **headgear**: hats, hair clips, ribbons, headbands, wimples, glasses\n"
        "- **clothing**: dress, shirt, skirt, jacket, tie, stockings/socks (if part of outfit)\n"
        "- **shoes**: footwear\n"
        "- **worn_accessories**: items ATTACHED to body — earrings, necklaces, chokers, belts, scarves.\n"
        "  These stay on the character regardless of scene. ALWAYS carried over.\n"
        "- **held_items**: items HELD in hands or carried — parasols, umbrellas, bags, purses,\n"
        "  books, phones, weapons, shopping bags, briefcases, etc.\n"
        "  These are SITUATIONAL — the character may pick them up or set them down at any time.\n"
        "\n"
        "### HELD ITEMS CONTEXT RULE\n"
        "held_items are NOT permanent parts of the outfit. They come and go based on context.\n"
        "Check whether the item makes sense at the image insertion point specifically —\n"
        "the character may have picked it up or set it down before/after this exact moment.\n"
        "\n"
        "CRITICAL: The image generation prompt must show what looks NATURAL in the scene.\n"
        "Prose may describe a character clutching an item that would look unnatural in the image.\n"
        "When the story says the character is holding something that doesn't belong in the scene,\n"
        "the image should NOT include it — the image must depict a visually plausible moment.\n"
        "\n"
        "Decision process for each held_item (evaluate IN THIS ORDER):\n"
        "1. FIRST — Does this item look natural in the image at this specific scene?\n"
        "   Think about the location, activity, weather, and situation at the insertion point.\n"
        "   If the item would look strange or out of place → REMOVE, regardless of what\n"
        "   the prose or chat says. Do NOT keep it just because the story mentions it.\n"
        "   If the item looks natural → proceed to step 2.\n"
        "2. If the Image Insertion Point Context or chat describes the character actively "
        "holding/using the item AND step 1 passed → KEEP.\n"
        "3. If Current Character Block includes the item, step 1 passed, and no contradiction → KEEP.\n"
        "4. If Current Character Block does NOT include the item and no reason to add → do NOT add.\n"
        "\n"
        "### OUTFIT DECISION PROCESS (Steps 1-4)\n"
        "\n"
        "#### Step 1 — PARSE Outfit Analysis Result\n"
        "If Outfit Analysis Result exists, extract each field as ground truth items.\n"
        "\n"
        "#### Step 2 — COMPARE with Current Character Block\n"
        "Identify differences between Outfit Analysis Result and Current Character Block:\n"
        "- MISSING: items in Outfit Analysis Result but absent from Current Character Block.\n"
        "- CONTRADICTING: items with different color, style, or type.\n"
        "- NEW: items in Current Character Block but not in Outfit Analysis Result.\n"
        "\n"
        "#### Step 3 — DECIDE using Image Insertion Point Context and chat\n"
        "CRITICAL: Check Image Insertion Point Context FIRST. It tells you exactly WHERE in\n"
        "the story this image is generated. Read the text BEFORE and AFTER the insertion point\n"
        "to determine whether the character has already changed clothes or is about to.\n"
        "\n"
        "Case A — Insertion point is AFTER an outfit change has occurred:\n"
        '  (e.g. insertion point comes after "changed clothes", "put on a swimsuit", '
        '"headed to the pool and changed")\n'
        "  → ACCEPT the new outfit from Current Character Block.\n"
        "  → Still carry over non-contradicted items from Outfit Analysis Result.\n"
        "\n"
        "Case A2 — Insertion point is BEFORE the outfit change happens:\n"
        '  (e.g. insertion point is before "she headed to the pool", before "went to change")\n'
        "  → The character has NOT changed yet at this point.\n"
        "  → Use the PRE-CHANGE outfit (Outfit Analysis Result or Previous Enhanced Output).\n"
        "\n"
        "Case B — No outfit change is described at the insertion point:\n"
        "  → Outfit Analysis Result is the CORRECT outfit.\n"
        "  → Fill MISSING items from Outfit Analysis Result.\n"
        "  → Override CONTRADICTING items with Outfit Analysis Result's version.\n"
        "  → worn_accessories are ALWAYS carried over.\n"
        "  → held_items follow the HELD ITEMS CONTEXT RULE above.\n"
        "\n"
        "Case C — No Outfit Analysis Result provided:\n"
        "  → Keep Current Character Block's outfit as-is.\n"
        "  → If Image Insertion Point Context or chat contradicts Current Character Block, follow the context.\n"
        "\n"
        "#### Step 4 — ENHANCE outfit details\n"
        "After determining the correct outfit, enrich descriptions with:\n"
        "- MATERIAL: cotton, silk, wool, denim, leather, chiffon, satin, lace, velvet\n"
        "- FIT & LENGTH: form-fitting, loose, oversized, cropped, mini, knee-length, floor-length\n"
        "- COLOR SHADES: navy blue, crimson red, ivory white, charcoal grey, pastel pink\n"
        "- STRUCTURAL DETAILS: collar type, sleeve length, button count, pleat style, lace trim\n"
        "\n"
        "STRICT ENHANCEMENT BOUNDARIES:\n"
        "1. Enhancement means adding PRECISION to EXISTING items only.\n"
        "2. Do NOT invent details that are NOT in Current Character Block or Outfit Analysis Result.\n"
        "   FORBIDDEN: stains, wrinkles, weathering, patterns, damage, text/logos, wear marks,\n"
        "   pockets contents, embroidery not mentioned, accessories not listed.\n"
        "3. Do NOT add phrases like 'evoking a ... aesthetic' that reinterpret the outfit type.\n"
        "\n"
        "CRITICAL: Enhancement adds PRECISION, never changes the outfit's NATURE.\n"
        "- 'nun dress' → stays nun dress. Never becomes 'school uniform'.\n"
        "- 'kimono' → stays kimono. Never becomes 'dress'.\n"
        "\n"
        "### EXPRESSION DECISION (Step 5)\n"
        "Analyze Image Insertion Point Context and Current Chat Context to determine the "
        "expression at THIS EXACT MOMENT in the story where the image is generated.\n"
        "- The expression must match the emotional state at the insertion point, "
        "not just the overall chat mood.\n"
        "- Consider what happens just BEFORE and AFTER the insertion point in the story.\n"
        "- Use Danbooru-compatible expression tags.\n"
        "- Expression tags: smiling, frowning, crying, blushing, angry, surprised, nervous,\n"
        "  embarrassed, happy, sad, scared, determined, gentle smile, smirk,\n"
        "  open mouth, closed eyes, tears, sweat drop, hollow eyes,\n"
        "  pout, grin, wry smile, half-closed eyes, wide eyes, narrowed eyes,\n"
        "  gritting teeth, nose blush, flushed face, teary-eyed, sparkling eyes,\n"
        "  vacant stare, confused, dazed, sleepy, yawning, biting lip,\n"
        "  shy smile, bitter smile, forced smile, mischievous, sly, playful,\n"
        "  flustered, exasperated, worried, relieved, nostalgic, melancholy,\n"
        "  longing, resigned, expressionless, cat smile, looking away,\n"
        "  looking down, looking up, side glance, glaring, huffing,\n"
        "  puffed cheeks, anguish, despair, awe, shock, boredom,\n"
        "  smug, contempt, pride, catlike mouth, wavy mouth,\n"
        "  seductive smile, shy look, gentle gaze, intense gaze,\n"
        "  distant look, peaceful, serene, triumphant, defeated,\n"
        "  fearful, trembling, grimacing, sniffling, sighing,\n"
        "  light smile, warm smile, sad smile, knowing smile,\n"
        "  talking, shouting, screaming, whispering, humming,\n"
        "  drooping eyes, star-shaped eyes, heart-shaped eyes,\n"
        "  heterochromia expression, one eye closed, both eyes closed,\n"
        "  eyebrows raised, eyebrows furrowed, tongue out, teeth showing,\n"
        "  lip bite, lip pout, nose wrinkle, cheek puff,\n"
        "  flattered, admiration, devotion, jealousy, envy,\n"
        "  bashful, demure, coquettish, haughty, arrogant,\n"
        "  stoic, apathetic, indifferent, curious, inquisitive,\n"
        "  startled, panicked, frantic, hysterical, euphoric,\n"
        "  wistful, bittersweet, forlorn, dejected, crestfallen.\n"
        "- If both contexts are empty or emotionally neutral, "
        "preserve expression from Current Character Block.\n"
        "- Expression is ALWAYS fresh — determined by the insertion point context only.\n"
        "\n"
        "### CONSISTENCY RULES FOR REPEATED ENHANCEMENT\n"
        "When ### Previous Enhanced Output is provided:\n"
        "It contains OUTFIT INFORMATION ONLY (no expression tags).\n"
        "- If your decision is **Case B** (no outfit change at insertion point):\n"
        "  1. **COPY OUTFIT DESCRIPTIONS VERBATIM** from Previous Enhanced Output.\n"
        "     Do NOT rephrase, reinterpret, shade-shift, or modify ANY outfit wording.\n"
        "  2. **DO NOT vary** color shades: if previous says 'blue', copy 'blue' exactly — "
        "not 'sky-blue', 'light blue', or 'navy blue'.\n"
        "  3. **DO NOT vary** positional descriptions: if previous says 'in her hair', "
        "copy that exact phrase — not 'on the left side' or 'clipped to the side'.\n"
        "  4. **DO NOT add** new adjectives to outfit items: no 'fluttering', 'delicate', "
        "'crisp' unless they were already in the previous output.\n"
        "  5. **HELD ITEMS**: Even if Previous Enhanced Output has held items, re-evaluate them\n"
        "     using the HELD ITEMS CONTEXT RULE based on the insertion point context.\n"
        "  6. **EXPRESSION**: Always generate FRESH from the insertion point context.\n"
        "     Never reuse expression from any previous output.\n"
        "- If your decision is **Case A** (outfit change confirmed at insertion point):\n"
        "  Ignore Previous Enhanced Output completely.\n"
        "  Generate new outfit AND new expression from the insertion point context.\n"
        "\n"
        "### CHARACTER NAME RULES\n"
        "1. ALWAYS include the Character Name exactly as provided.\n"
        "2. If Current Character Block has a shorter name (e.g. 'maria' → 'takayama maria'), correct to full name.\n"
        "3. Do NOT add series/franchise names in parentheses.\n"
        "\n"
        "### OUTPUT FORMAT\n"
        "Output THREE parts separated by exactly [EXPR] and [REASON]:\n"
        "\n"
        "Part 1 (before [EXPR]): character identity, body, hair, eyes, outfit, pose, action\n"
        "Part 2 (after [EXPR], before [REASON]): expression, emotion, face description tags ONLY\n"
        "Part 3 (after [REASON]): brief one-line decision memo\n"
        "  Format: 'Case X. <outfit decision>. held_items: <kept/removed/not added because...>'\n"
        "  Example: 'Case B. Outfit unchanged. held_items: parasol removed — indoors/bus at insertion point.'\n"
        "  Example: 'Case A. Outfit changed at insertion point. held_items: book kept — reading.'\n"
        "\n"
        "Rules:\n"
        "1. Parts 1 and 2 are comma-separated Danbooru-style tags.\n"
        "2. Do NOT use the | character.\n"
        "3. Process ONE character at a time.\n"
        "4. Do NOT add new weighted tags like ((tag:weight)) that were NOT in the original Current Character Block.\n"
        "5. Part 3 (memo) must be exactly ONE line. Keep it concise.\n"
        "\n"
        "### EXAMPLES\n"
        "All examples include Image Insertion Point Context showing where the image is generated.\n"
        "Note how the decision depends on what happens AT the insertion point, not just the overall chat.\n"
        "\n"
        "--- Example 1: Case B, no outfit change, expression from insertion point ---\n"
        "Character Name: takayama maria\n"
        "Current Character Block: '1girl, dark-blue one piece cotton dress, dark blue Cornette, smiling'\n"
        'Outfit Analysis Result: {"headgear": "dark blue cornette, red heart hair ornament", '
        '"clothing": "dark-blue one-piece cotton nun dress, white collar", '
        '"shoes": "Unknown", "worn_accessories": "None", "held_items": "None"}\n'
        "Current Chat Context: Maria was sitting quietly in the chapel. She flipped through "
        "her picture book with a calm face.\n"
        "Image Insertion Point Context:\n"
        "  ▼ Image insertion point ▼\n"
        "  Maria was sitting quietly in the chapel. She flipped through her picture book with a "
        "calm face. She turned the page and noticed a colorful illustration. Her eyes widened "
        "slightly with curiosity, and a small smile appeared on her lips.\n"
        "→ Insertion point: She is reading calmly, no outfit change. "
        "Expression at insertion point: calm curiosity, gentle smile.\n"
        "→ Case B. Missing hair ornament restored from Outfit Analysis Result.\n"
        "Output:\n"
        "1girl, takayama maria, wearing a dark-blue cotton one-piece nun dress "
        "with a white collar, a dark-blue cornette headdress with a red heart hair ornament, "
        "sitting, holding a book\n"
        "[EXPR]\n"
        "gentle smile, calm, curious, sparkling eyes\n"
        "[REASON]\n"
        "Case B. Outfit unchanged. Missing hair ornament restored. held_items: none.\n"
        "\n"
        "--- Example 2: Case A, insertion point AFTER outfit change ---\n"
        "Character Name: sonoda umi\n"
        "Current Character Block: '1girl, swimsuit, blue bikini, at the pool, embarrassed'\n"
        'Outfit Analysis Result: {"clothing": "white short-sleeved cotton collared shirt, blue wool pleated skirt"}\n'
        "Current Chat Context: Umi reluctantly agreed to go to the pool. She changed into her "
        "swimsuit in the changing room. Walking out to the poolside, she wrapped her arms around "
        "herself shyly.\n"
        "Image Insertion Point Context:\n"
        "  ▼ Image insertion point ▼\n"
        "  ...She changed into her swimsuit in the changing room. Walking out to the poolside, "
        "she wrapped her arms around herself shyly. Her face was completely red as her friends "
        "cheered for her.\n"
        "→ Insertion point is AFTER she changed. She is already wearing the swimsuit at this point. "
        "Expression: deeply embarrassed.\n"
        "→ Case A. Outfit change confirmed at insertion point.\n"
        "Output:\n"
        "1girl, sonoda umi, wearing a navy-blue frilled two-piece bikini swimsuit "
        "with white ribbon accents, at the pool\n"
        "[EXPR]\n"
        "blushing, embarrassed, nervous, looking away, flustered, arms wrapped around self\n"
        "[REASON]\n"
        "Case A. Outfit changed to swimsuit at insertion point. held_items: none.\n"
        "\n"
        "--- Example 3: Case A2, insertion point BEFORE outfit change ---\n"
        "Character Name: kashiwazaki sena\n"
        "Current Character Block: '1girl, school uniform, heading to pool, swimsuit in hand'\n"
        'Outfit Analysis Result: {"clothing": "St. chronica academy school uniform, green jacket, '
        'white shirt, plaid pleated green skirt", "shoes": "brown loafers", '
        '"held_items": "swimsuit bag"}\n'
        "Current Chat Context: Sena grabbed her swimsuit and headed toward the pool changing room. "
        "She was looking forward to showing off her new swimsuit.\n"
        "Image Insertion Point Context:\n"
        "  ▼ Image insertion point ▼\n"
        "  Sena grabbed her swimsuit and headed toward the pool changing room. "
        "She walked down the hallway with a confident stride. She hadn't changed yet.\n"
        "→ Insertion point is BEFORE she changes. She is still in her school uniform at this point.\n"
        "→ Case A2. Pre-change outfit. Use Outfit Analysis Result.\n"
        "Output:\n"
        "1girl, kashiwazaki sena, wearing a St. chronica academy school uniform, green jacket, "
        "white shirt, plaid pleated green skirt, brown loafers, walking, holding a bag\n"
        "[EXPR]\n"
        "confident, grin, determined\n"
        "[REASON]\n"
        "Case A2. Insertion point is before outfit change. School uniform kept. held_items: swimsuit bag kept — carrying it.\n"
        "\n"
        "--- Example 4: Case B, held item illogical at insertion point → removed ---\n"
        "Character Name: hasegawa kobato\n"
        "Current Character Block: '1girl, interior, bus, gothic lolita dress, lace frills, "
        "black lace parasol, sitting'\n"
        'Outfit Analysis Result: {"headgear": "black frilled hairband", '
        '"clothing": "black silk gothic lolita dress with lace frills", '
        '"shoes": "Unknown", "worn_accessories": "None", '
        '"held_items": "black lace parasol"}\n'
        "Current Chat Context: Kobato boarded the bus and sat by the window. The bus was "
        "heading downtown.\n"
        "Image Insertion Point Context:\n"
        "  ▼ Image insertion point ▼\n"
        "  Kobato boarded the bus and sat by the window. She stared out at the passing "
        "buildings with a bored expression. The bus was crowded and stuffy.\n"
        "→ Insertion point: On a bus, indoors. Nobody uses a parasol inside a bus. "
        "Even though Outfit Analysis Result has it, it makes no sense HERE → REMOVE.\n"
        "→ Case B. Missing headgear restored.\n"
        "Output:\n"
        "1girl, hasegawa kobato, wearing a black silk gothic lolita dress with layered lace frills, "
        "a black frilled hairband, sitting\n"
        "[EXPR]\n"
        "bored, half-closed eyes, pout, looking out window\n"
        "[REASON]\n"
        "Case B. Outfit unchanged. Missing headgear restored. held_items: parasol removed — indoors/bus at insertion point.\n"
        "\n"
        "--- Example 5: Case B, held item makes sense at insertion point → kept ---\n"
        "Character Name: oreki houtarou\n"
        "Current Character Block: '1boy, school uniform, holding a book, walking'\n"
        'Outfit Analysis Result: {"headgear": "None", '
        '"clothing": "gakuran school uniform, black jacket, white shirt", '
        '"shoes": "brown leather shoes", "worn_accessories": "None", '
        '"held_items": "book"}\n'
        "Current Chat Context: Oreki was walking home from school, flipping through a paperback "
        "he'd borrowed from the library.\n"
        "Image Insertion Point Context:\n"
        "  ▼ Image insertion point ▼\n"
        "  Oreki was walking home from school, flipping through a paperback he'd borrowed from "
        "the library. He turned a page with one hand while adjusting his bag strap with the other.\n"
        "→ Insertion point: Walking and actively reading. Book makes perfect sense here → KEEP.\n"
        "→ Case B. Outfit carried over.\n"
        "Output:\n"
        "1boy, oreki houtarou, wearing a gakuran school uniform, black jacket, white shirt, "
        "brown leather shoes, holding a book, walking\n"
        "[EXPR]\n"
        "calm, focused, half-closed eyes, reading\n"
        "[REASON]\n"
        "Case B. Outfit unchanged. held_items: book kept — actively reading at insertion point.\n"
        "\n"
        "--- Example 6: Case B, held item NOT in block, scene doesn't support → not added ---\n"
        "Character Name: kashiwazaki sena\n"
        "Current Character Block: '1girl, casual clothes, t-shirt, shorts, sitting on couch, watching TV'\n"
        'Outfit Analysis Result: {"headgear": "None", '
        '"clothing": "white t-shirt, denim shorts", '
        '"shoes": "None", "worn_accessories": "None", '
        '"held_items": "school bag"}\n'
        "Current Chat Context: Sena was lounging at home on a lazy Sunday, binge-watching her "
        "favorite anime.\n"
        "Image Insertion Point Context:\n"
        "  ▼ Image insertion point ▼\n"
        "  Sena was lounging at home on a lazy Sunday, binge-watching her favorite anime. "
        "She sprawled on the couch with a bag of chips, laughing at the screen.\n"
        "→ Insertion point: At home, weekend, relaxing. School bag in Outfit Analysis Result but "
        "she's sprawled on the couch with chips. No reason for a school bag → do NOT add.\n"
        "→ Case B. Outfit carried over.\n"
        "Output:\n"
        "1girl, kashiwazaki sena, wearing a white cotton t-shirt, denim shorts, "
        "sitting on couch, watching TV\n"
        "[EXPR]\n"
        "happy, laughing, relaxed, open mouth\n"
        "[REASON]\n"
        "Case B. Outfit unchanged. held_items: school bag not added — at home/weekend at insertion point.\n"
        "\n"
        "--- Example 7: Case B with previous enhanced, expression from insertion point ---\n"
        "Character Name: koizumi hanayo\n"
        "Current Character Block: '1girl, school uniform, white shirt, blue skirt, reading a book'\n"
        'Outfit Analysis Result: {"clothing": "white short-sleeved cotton collared shirt, navy blue pleated skirt", '
        '"worn_accessories": "None", "held_items": "None"}\n'
        "Previous Enhanced Output: '1girl, koizumi hanayo, school uniform, wearing a white "
        "short-sleeved cotton collared shirt, a navy-blue wool pleated skirt'\n"
        "Current Chat Context: Hanayo was in the library studying. Suddenly she found a passage "
        "that surprised her.\n"
        "Image Insertion Point Context:\n"
        "  ▼ Image insertion point ▼\n"
        "  Hanayo was in the library studying. She turned a page and her eyes went wide. "
        "She found a surprising passage and covered her mouth with both hands.\n"
        "→ Insertion point: In library, just discovered something surprising. "
        "Expression at this exact moment: shocked/surprised.\n"
        "→ Case B. Copy outfit VERBATIM from previous. Expression from insertion point.\n"
        "Output:\n"
        "1girl, koizumi hanayo, school uniform, wearing a white short-sleeved cotton collared shirt, "
        "a navy-blue wool pleated skirt, reading a book\n"
        "[EXPR]\n"
        "surprised, wide eyes, open mouth, hands over mouth, shock\n"
        "[REASON]\n"
        "Case B. Outfit copied verbatim from previous. held_items: none. Expression from insertion point.\n"
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


# ─── SLOT Parsing ─────────────────────────────────────────

def _parse_slot(chat_content: str) -> tuple[str, str, str]:
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

    # [SLOT] 마커 찾기 (이후 줄바꿈 또는 다른 [섹션] 전까지)
    slot_match = re.search(
        r'\[SLOT\]\s*(.*?)(?=\n\s*\[|\n\s*\n|\Z)',
        chat_content, re.DOTALL | re.IGNORECASE
    )
    if not slot_match:
        return chat_content, "", ""

    slot_content = slot_match.group(1).strip()

    # || 로 분리
    if '||' in slot_content:
        parts = slot_content.split('||', 1)
        before = parts[0].strip()
        after = parts[1].strip()
    else:
        before = slot_content.strip()
        after = ""

    # chat_content에서 SLOT 섹션 제거
    chat_clean = chat_content[:slot_match.start()] + chat_content[slot_match.end():]
    chat_clean = re.sub(r'\n{3,}', '\n\n', chat_clean).strip()

    return chat_clean, before, after


def _extract_context_around(text: str, anchor_before: str, anchor_after: str,
                            max_tokens: int) -> str:
    """
    앵커 텍스트(anchor_before || anchor_after)가 text에서 매칭되는 위치를 찾아
    그 중심으로 max_tokens 범위의 컨텍스트를 추출.
    매칭 실패 시 뒤(최신)부터 잘라서 반환.
    """
    if not text:
        return ""
    if not anchor_before and not anchor_after:
        return _trim_to_tokens(text, max_tokens)
    if _count_tokens(text) <= max_tokens:
        return text

    # 매칭 위치 찾기
    pos = -1
    anchor = anchor_before or anchor_after

    # 1) 정확 매칭
    pos = text.find(anchor)

    # 2) 앵커의 첫 문장으로 재시도
    if pos == -1:
        first_phrase = anchor.split('.')[0].strip()
        if len(first_phrase) > 10:
            pos = text.find(first_phrase)

    # 3) after_text로 재시도
    if pos == -1 and anchor_after and anchor != anchor_after:
        pos = text.find(anchor_after)
        if pos == -1:
            first_phrase = anchor_after.split('.')[0].strip()
            if len(first_phrase) > 10:
                pos = text.find(first_phrase)

    # 매칭 실패 → 뒤에서부터 자르기
    if pos == -1:
        return _trim_to_tokens(text, max_tokens)

    # 매칭 중심으로 추출
    match_center = pos + len(anchor) // 2
    max_chars = int(max_tokens * 3.5)  # 토큰당 약 3.5문자 추정
    half = max_chars // 2

    start = max(0, match_center - half)
    end = min(len(text), start + max_chars)
    if end == len(text):
        start = max(0, end - max_chars)

    extracted = text[start:end]

    # 토큰 한계 초과 시 추가 트리밍
    if _count_tokens(extracted) > max_tokens:
        extracted = _trim_to_tokens(extracted, max_tokens)

    return extracted


# ─── User Message Building ────────────────────────────────

def _build_user_message(
    character_name: str,
    char_block: str,
    outfit_result: str = None,
    chat_content: str = "",
    previous_chat: str = "",
    previous_enhanced: str = "",
    scene_tags: str = "",
) -> str:
    # SLOT 파싱: 그림 삽입 위치 정보 추출
    chat_without_slot, slot_before, slot_after = _parse_slot(chat_content)

    parts = []
    parts.append(f"### Character Name\n{character_name}")
    parts.append(f"### Current Character Block\n{char_block}")

    if scene_tags:
        parts.append(f"### Scene Context (from [Positive] section — camera, location, setting)\n{scene_tags}")

    if outfit_result:
        parts.append(f"### Outfit Analysis Result (ground truth)\n{outfit_result}")

    # Current Chat Context (SLOT 마커 제거된 버전)
    if chat_without_slot:
        parts.append(f"### Current Chat Context\n{chat_without_slot}")

    # ─── Image Insertion Point Context (새 변수) ───
    # 그림이 생성되는 정확한 위치의 앞뒤 문맥을 제공
    if slot_before or slot_after:
        insertion_context = _extract_context_around(
            chat_without_slot, slot_before, slot_after, 2000
        )
        if insertion_context:
            parts.append(
                "### Image Insertion Point Context\n"
                "THIS IS WHERE THE IMAGE IS BEING GENERATED in the story.\n"
                "The image depicts the scene at this exact point. "
                "Read the text BEFORE and AFTER the ▼ insertion point ▼ to determine:\n"
                "- Whether the character has ALREADY changed or is ABOUT TO change clothes\n"
                "- The emotional state at this exact moment\n"
                "- The correct outfit and expression for this specific point\n"
                f"{insertion_context}"
            )

    # Previous Enhanced Output
    if previous_enhanced:
        parts.append(
            "### Previous Enhanced Output (OUTFIT ONLY — reuse outfit descriptions exactly, "
            "generate NEW expression from insertion point context)\n"
            f"{previous_enhanced}"
        )

    # 토큰 예산 계산
    base_text = "\n\n".join(parts)
    base_tokens = _count_tokens(base_text)
    system_tokens = _count_tokens(_build_system_prompt())

    used_tokens = base_tokens + system_tokens
    remaining = MAX_TOTAL_TOKENS - used_tokens

    # Previous Chat Context — SLOT 기반 매칭으로 중심 추출
    if previous_chat:
        if remaining > 0:
            prev_budget = min(remaining, 2000)
            prev_context = _extract_context_around(
                previous_chat, slot_before, slot_after, prev_budget
            )
            if prev_context:
                parts.append(
                    "### Previous Chat Context (context around outfit tracking point)\n"
                    f"{prev_context}"
                )

    return "\n\n".join(parts)


# ─── Main Entry Point ─────────────────────────────────────

async def run(
    character_name: str,
    char_block: str,
    outfit_result: str = None,
    chat_content: str = "",
    previous_chat: str = "",
    previous_enhanced: str = "",
    scene_tags: str = "",
) -> tuple[str, str]:
    """
    캐릭터 블럭의 복장 묘사를 검증/강화하고, 표정을 상황에 맞게 결정.

    Args:
        character_name: 캐릭터 이름
        char_block: 현재 캐릭터 블럭
        outfit_result: outfit_mode의 LLM 복장 통합 결과 (JSON 문자열, None 가능)
        chat_content: 현재 채팅 내용 ([SLOT] 포함 가능)
        previous_chat: 이전 복장 통합 시점의 채팅 내용
        previous_enhanced: 이전 강화 결과 중 복장 부분만 (일관성 유지용)
        scene_tags: [Positive] 섹션의 장면/카메라 태그 (상황 컨텍스트)

    Returns:
        tuple[str, str]: (combined_block, outfit_only_block)
            combined_block: 복장 + 표정 조합 (최종 프롬프트용)
            outfit_only_block: 복장만 (일관성 추적용, _last_enhanced_block에 저장)
    """
    system_prompt = _build_system_prompt()
    user_message = _build_user_message(
        character_name, char_block, outfit_result, chat_content, previous_chat,
        previous_enhanced, scene_tags
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
        "chat_content": (chat_content if chat_content else ""),
        "previous_enhanced": (previous_enhanced if previous_enhanced else ""),
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

    if result.startswith("[LLM 실패]"):
        _log_prompt_io(character_name, log_input, result)
        return result, result

    # 결과 정리: 불필요한 마크다운/래핑 제거
    result = result.strip()
    result = re.sub(r'^```(?:\w+)?\s*', '', result)
    result = re.sub(r'\s*```$', '', result)
    result = result.strip()

    # ─── [EXPR] + [REASON] 파싱 ───
    outfit_part = result
    expression_part = ""
    reason_part = ""

    if '[REASON]' in result:
        parts = result.split('[REASON]', 1)
        main_content = parts[0].strip()
        reason_part = parts[1].strip().split('\n')[0]  # 첫 번째 줄만
    else:
        main_content = result

    if '[EXPR]' in main_content:
        parts = main_content.split('[EXPR]', 1)
        outfit_part = parts[0].strip().rstrip(',').strip()
        expression_part = parts[1].strip().rstrip(',').strip()

        # outfit_part가 비어있으면 원본 사용
        if not outfit_part:
            outfit_part = char_block

    # 로그 기록 (reason 포함)
    if reason_part:
        print(f"[ENHANCE_REASON] {character_name}: {reason_part}")
    log_input["reason"] = reason_part if reason_part else ""
    _log_prompt_io(character_name, log_input, result)

    # 캐릭터 이름 검사 (outfit_part 기준)
    result_lower = outfit_part.lower()
    if character_name.lower() not in result_lower:
        name_parts = [p for p in character_name.lower().split() if len(p) > 1]
        if not any(part in result_lower for part in name_parts):
            return char_block, char_block

    # | 문자 제거 (캐릭터 구분자로 오인 방지)
    if '|' in outfit_part:
        outfit_part = outfit_part.replace('|', ',')
    if '|' in expression_part:
        expression_part = expression_part.replace('|', ',')

    # combined: 복장 + 표정 조합
    if expression_part:
        combined = f"{outfit_part}, {expression_part}"
    else:
        combined = outfit_part

    return combined, outfit_part
