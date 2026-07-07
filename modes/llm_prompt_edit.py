"""삽화백업 "LLM과 함께 수정" 지원 모듈.

빌드된 긍정 프롬프트(illust_prompt_builder.build_positive_prompt 출력)에서
주제 3개 블럭(ANIMA_CONTENT/ANIMA_ALL/SDXL)의 장면(setup/char/supplement)만
LLM이 편집하고, 트리거/아티스트/품질/제어 블럭은 백엔드가 보존·재조립한다.

핵심 불변량:
- 제어 블럭([LORA_DATA]/[SEED]/[CACHE_PATH] 등)은 1바이트도 수정하지 않는다.
- 트리거/아티스트/품질 토큰은 LLM에 전달하지 않고 백엔드에서 접두부로 보존한다.
- 3개 주제 블럭은 동일 장면을 공유(supplement는 SDXL 제외) — LLM이 한 번 서술한
  scene_setup/char/supplement 를 백엔드가 3블럭에 일관 주입한다.
"""

import json
import os
import re
import shutil
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
BOT_JSON_PATH = os.path.join(ASSET_DATA_DIR, "bot.json")

# ─── LLM 보조(시스템) 프롬프트 — 배포용 builtin + 사용자 custom ───
# lb_extra_refine 패턴과 동일: builtin(배포용·읽기전용) / custom(이PC 전용·편집가능) /
# meta(use_custom 토글). effective = (use_custom && custom) ? custom : builtin.
LLM_EDIT_PROMPT_DIR = os.path.join(BASE_DIR, "prompts", "llm_prompt_edit")
LLM_EDIT_BUILTIN_FILE = os.path.join(LLM_EDIT_PROMPT_DIR, "system.txt")
LLM_EDIT_CUSTOM_FILE = os.path.join(ASSET_DATA_DIR, "llm_prompt_edit_custom.txt")
LLM_EDIT_META_FILE = os.path.join(ASSET_DATA_DIR, "llm_prompt_edit_meta.json")

# builtin 파일이 없을 때의 폴백 기본값(= 배포 전 기본 시스템 프롬프트).
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert editor for the positive prompt of a Stable Diffusion Illust workflow. "
    "Analyze the user's 'edit direction', the generated image (if provided), and the current scene tags, "
    "then modify ONLY the scene description (setup/char/supplement). "
    "Return ONLY a valid JSON object - no other text, no markdown code fences, no explanations."
)

_llm_edit_builtin_cache = None
_llm_edit_builtin_mtime = 0.0


def _load_llm_edit_builtin() -> str:
    """배포용(글로벌) 시스템 프롬프트 로드. mtime 기반 캐싱. 파일 없으면 DEFAULT_SYSTEM_PROMPT."""
    global _llm_edit_builtin_cache, _llm_edit_builtin_mtime
    if not os.path.isfile(LLM_EDIT_BUILTIN_FILE):
        return DEFAULT_SYSTEM_PROMPT
    try:
        mtime = os.path.getmtime(LLM_EDIT_BUILTIN_FILE)
        if _llm_edit_builtin_cache is not None and mtime == _llm_edit_builtin_mtime:
            return _llm_edit_builtin_cache
        with open(LLM_EDIT_BUILTIN_FILE, "r", encoding="utf-8") as f:
            txt = f.read()
        _llm_edit_builtin_cache = txt
        _llm_edit_builtin_mtime = mtime
        return txt
    except Exception as e:
        print(f"[LLM_EDIT] builtin 로드 실패: {e}")
        traceback.print_exc()
        return DEFAULT_SYSTEM_PROMPT


def _load_llm_edit_custom() -> tuple:
    """커스텀 프롬프트와 use_custom 플래그 로드. (없으면 '', False)."""
    custom = ""
    if os.path.isfile(LLM_EDIT_CUSTOM_FILE):
        try:
            with open(LLM_EDIT_CUSTOM_FILE, "r", encoding="utf-8") as f:
                custom = f.read()
        except Exception as e:
            print(f"[LLM_EDIT] custom 로드 실패: {e}")
            traceback.print_exc()

    use_custom = False
    if os.path.isfile(LLM_EDIT_META_FILE):
        try:
            with open(LLM_EDIT_META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                use_custom = bool(meta.get("use_custom", False))
        except Exception as e:
            print(f"[LLM_EDIT] meta 로드 실패: {e}")
            traceback.print_exc()

    return custom, use_custom


def _save_llm_edit_custom(text: str, use_custom: bool) -> None:
    """커스텀 프롬프트 저장. 기존 파일은 .bak 로 백업."""
    os.makedirs(ASSET_DATA_DIR, exist_ok=True)

    if os.path.isfile(LLM_EDIT_CUSTOM_FILE):
        try:
            shutil.copy2(LLM_EDIT_CUSTOM_FILE, LLM_EDIT_CUSTOM_FILE + ".bak")
        except Exception as e:
            print(f"[LLM_EDIT] custom 백업 실패: {e}")

    try:
        with open(LLM_EDIT_CUSTOM_FILE, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[LLM_EDIT] custom 저장 실패: {e}")
        traceback.print_exc()
        raise

    if os.path.isfile(LLM_EDIT_META_FILE):
        try:
            shutil.copy2(LLM_EDIT_META_FILE, LLM_EDIT_META_FILE + ".bak")
        except Exception as e:
            print(f"[LLM_EDIT] meta 백업 실패: {e}")

    try:
        with open(LLM_EDIT_META_FILE, "w", encoding="utf-8") as f:
            json.dump({"use_custom": bool(use_custom)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LLM_EDIT] meta 저장 실패: {e}")
        traceback.print_exc()
        raise


def get_effective_system_prompt() -> str:
    """실제 LLM 호출에 사용할 시스템 프롬프트. use_custom && custom 이면 custom, 아니면 builtin."""
    custom, use_custom = _load_llm_edit_custom()
    if use_custom and custom.strip():
        return custom
    return _load_llm_edit_builtin()

# 빌드본 포맷 감지에 필요한 블럭 헤더들
REQUIRED_HEADERS = [
    "ANIMA_CONTENT", "ANIMA_ALL", "SDXL",
    "CHAR_LIST", "ANIMA_ARTIST", "SDXL_ARTIST",
    "ANIMA_QUALITY", "SDXL_QUALITY",
]

# 편집 대상 주제 3개 블럭
SUBJECT_BLOCKS = ("ANIMA_CONTENT", "ANIMA_ALL", "SDXL")

# 블럭 헤더 한 줄 매칭: [NAME]\n<내용 한 줄>
_BLOCK_RE = re.compile(r"^\[([A-Z_]+)\]\n(.*)$", re.MULTILINE)
# 재조립용: 주제 3개 블럭의 헤더+내용
_SUBJECT_RE = re.compile(r"^(\[(?:ANIMA_CONTENT|ANIMA_ALL|SDXL)\]\n).*$", re.MULTILINE)


def parse_blocks(positive: str) -> dict:
    """positive 를 [블럭명] -> 내용(단일 라인) dict 로 분할.

    빌더(illust_prompt_builder.py 484~606)는 각 블럭을 "[NAME]\\n<내용>" 형태로
    "\\n" 으로 연결하므로, 블럭 내용은 항상 단일 라인이다.
    """
    blocks = {}
    for m in _BLOCK_RE.finditer(positive):
        name = m.group(1)
        content = m.group(2)
        blocks[name] = content
    return blocks


def detect_build_format(positive: str) -> bool:
    """빌드본 포맷 여부 — 편집에 필요한 8개 헤더가 모두 있는지 검사.

    배치/비삽화 백업은 [UPSCALE]/[ILXL] 등 다른 포맷이거나 헤더 자체가 없다.
    """
    for h in REQUIRED_HEADERS:
        if not re.search(rf"^\[{h}\]\n", positive, re.MULTILINE):
            return False
    return True


def _split_tokens(content: str) -> list:
    """", " 구분 문자열을 토큰 리스트로. 빈 토큰/공백 제거."""
    if not content:
        return []
    return [t.strip() for t in content.split(",") if t.strip()]


def _load_bot_data(bot_name: str):
    """bot.json 에서 bot_name 으로 bot dict 찾기 (대소문자 무관). 실패 시 None."""
    if not bot_name:
        return None
    try:
        with open(BOT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[LLM_EDIT] bot.json 로드 실패(bot_name={bot_name}): {e}")
        return None
    bots = data.get("bots", []) if isinstance(data, dict) else []
    if isinstance(bots, dict):
        # bots 가 dict 인 경우 name 키로 직접 조회
        b = bots.get(bot_name)
        if b:
            return b
        for k, v in bots.items():
            if k.lower() == bot_name.lower():
                return v
        return None
    for b in bots:
        if isinstance(b, dict) and str(b.get("name", "")).lower() == bot_name.lower():
            return b
    return None


def recover_triggers(blocks: dict, bot_name: str) -> dict:
    """감지된 캐릭터(CHAR_LIST)의 LoRA 트리거워드 집합을 복원.

    반환: {"anima": set[str], "sdxl": set[str]}

    bot.json 에서 각 캐릭터의 loras_solo/loras_group(다인이면 loras_group 우선) 의
    trigger 필드를 수집한다(illust_prompt_builder.py 369 참조).
    bot.json 로드/매칭 실패 시 폴백: CHAR_LIST 의 캐릭터명 자체를 트리거로 사용
    (트리거 기본값 = 캐릭터명). 절대 예외를 던지지 않는다(best-effort).
    """
    anima = set()
    sdxl = set()

    char_list_str = blocks.get("CHAR_LIST", "")
    char_names = [c.strip() for c in char_list_str.split(",") if c.strip()]
    if not char_names:
        print("[LLM_EDIT] recover_triggers: CHAR_LIST 비어 있음 — 트리거 복원 불가")
        return {"anima": anima, "sdxl": sdxl}

    bot = _load_bot_data(bot_name)
    if bot is None:
        # 폴백: 캐릭터명을 트리거로 사용
        print(f"[LLM_EDIT] recover_triggers: bot 데이터 없음(bot_name={bot_name!r}) — "
              f"CHAR_LIST 이름을 트리거로 폴백 사용: {char_names}")
        for n in char_names:
            anima.add(n)
            sdxl.add(n)
        return {"anima": anima, "sdxl": sdxl}

    characters = bot.get("characters", []) if isinstance(bot, dict) else []
    is_multi = len(char_names) >= 2
    lora_key = "loras_group" if is_multi else "loras_solo"

    for char_name in char_names:
        # 대소문자 무관 캐릭터 매칭
        char_data = next(
            (c for c in characters
             if isinstance(c, dict) and str(c.get("name", "")).lower() == char_name.lower()),
            None,
        )
        if not char_data:
            # 매칭 실패 시 캐릭터명을 트리거로 폴백
            anima.add(char_name)
            sdxl.add(char_name)
            continue
        loras = char_data.get(lora_key, char_data.get("loras", []))
        for lora in loras:
            if not isinstance(lora, dict):
                continue
            trigger = lora.get("trigger", char_name)
            base = lora.get("BASE", "anima")
            if base == "sdxl":
                sdxl.add(trigger)
            else:
                anima.add(trigger)

    return {"anima": anima, "sdxl": sdxl}


def extract_scene_tokens(block_content: str, prefix_tokens: set) -> str:
    """블럭 내용에서 접두부(트리거/아티스트/품질) 토큰을 제거해 장면 토큰만 추출.

    prefix_tokens: 이 블럭의 접두부로 간주할 토큰 집합(정확 매칭, strip 후 비교).
    반환: ", " 로 join 된 장면 토큰 문자열.
    """
    if not block_content:
        return ""
    prefix_norm = {t.strip() for t in prefix_tokens if t and t.strip()}
    scene = []
    for tok in _split_tokens(block_content):
        if tok in prefix_norm:
            continue
        scene.append(tok)
    return ", ".join(scene)


def build_prefix_sets(blocks: dict, triggers: dict) -> dict:
    """각 주제 블럭별 접두부 토큰 집합을 구성.

    ANIMA_CONTENT 접두부 = anima 트리거 + anima 아티스트
    ANIMA_ALL    접두부 = anima 트리거 + anima 아티스트 + anima 품질
    SDXL         접두부 = sdxl 트리거 + sdxl 아티스트 + sdxl 품질
    """
    anima_artists = set(_split_tokens(blocks.get("ANIMA_ARTIST", "")))
    sdxl_artists = set(_split_tokens(blocks.get("SDXL_ARTIST", "")))
    anima_quality = set(_split_tokens(blocks.get("ANIMA_QUALITY", "")))
    sdxl_quality = set(_split_tokens(blocks.get("SDXL_QUALITY", "")))

    return {
        "ANIMA_CONTENT": set(triggers["anima"]) | anima_artists,
        "ANIMA_ALL": set(triggers["anima"]) | anima_artists | anima_quality,
        "SDXL": set(triggers["sdxl"]) | sdxl_artists | sdxl_quality,
    }


def build_llm_messages(direction: str, scene_current: str, scene_sdxl: str) -> list:
    """LLM(비전) 호출용 messages 빌드.

    이미지는 callLLMVision 의 _build_vision_messages 가 마지막 user 메시지에
    image_url 파트로 추가하므로, 여기서는 텍스트만 작성한다.
    시스템 프롬프트는 get_effective_system_prompt() (배포용 builtin 또는 사용자 custom).
    """
    system = get_effective_system_prompt()

    schema_desc = (
        "{\n"
        '  "plan": "edit plan - briefly explain why and which tags to add/remove (write in Korean)",\n'
        '  "scene_setup": "background/location/lighting/weather/mood tags, comma+space separated",\n'
        '  "scene_char": "character appearance/pose/expression/outfit tags, comma+space separated",\n'
        '  "scene_supplement": "extra auxiliary tags, or empty string \\"\\""\n'
        "}"
    )

    user = (
        f"## User edit direction\n{direction}\n\n"
        f"## Current scene tags (ANIMA, includes supplement)\n{scene_current or '(none)'}\n\n"
        f"## Current scene tags (SDXL, no supplement)\n{scene_sdxl or '(none)'}\n\n"
        "## Instructions\n"
        "1. Look at the provided image (if any) and the tags above to understand the current scene.\n"
        "2. Adjust scene_setup/scene_char/scene_supplement to match the user's edit direction.\n"
        "3. Preserve the character's core identity (key appearance traits); focus changes on scene/mood/pose/outfit.\n"
        "4. Keep original tags for any part that does not need changing (avoid unnecessary edits).\n"
        "5. scene_supplement is NOT applied to the SDXL block, so only put SDXL-absent auxiliary descriptions there.\n"
        "6. Keep using English danbooru-style tags separated by comma+space (\", \").\n"
        "7. Write the \"plan\" field in Korean; write scene_* tags in English.\n"
        f"## Output (JSON schema - return ONLY a JSON object in this form)\n{schema_desc}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_llm_json(raw: str):
    """LLM 응답에서 JSON 객체를 파싱. 실패 시 None.

    ```json ... ``` 코드펜스 제거 → json.loads → 실패 시 첫 { ~ 마지막 } 슬라이스 재시도.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()

    # 코드펜스 제거
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # { ... } 슬라이스 추출 재시도
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except Exception as e:
            print(f"[LLM_EDIT] parse_llm_json: JSON 슬라이스 파싱 실패: {e}")
    else:
        print(f"[LLM_EDIT] parse_llm_json: JSON 객체를 찾을 수 없음 (raw 앞 200자: {text[:200]!r})")
    return None


def _coerce_scene_fields(parsed: dict) -> dict:
    """LLM JSON 에서 scene_* 필드를 안전하게 문자열로 추출(누락/타입오류 방어)."""
    if not isinstance(parsed, dict):
        return {}

    def _str(key):
        v = parsed.get(key, "")
        if v is None:
            return ""
        if isinstance(v, list):
            return ", ".join(str(x).strip() for x in v if str(x).strip())
        return str(v).strip()

    return {
        "plan": _str("plan"),
        "scene_setup": _str("scene_setup"),
        "scene_char": _str("scene_char"),
        "scene_supplement": _str("scene_supplement"),
    }


def _join_tags(*parts) -> str:
    """빈 값을 제거하고 ", " 로 join."""
    return ", ".join(p.strip() for p in parts if p and p.strip())


def reassemble(positive: str, blocks: dict, triggers: dict, parsed: dict) -> tuple:
    """LLM 결과를 3개 주제 블럭에 주입해 positive 를 재조립.

    반환: (reassembled_positive, scene_dict)
    - 제어 블럭/나머지 내용 블럭/블럭 순서·포맷은 원본 그대로 보존.
    - LLM 응답이 비정상이면 해당 블럭은 원본 유지.
    """
    scene = _coerce_scene_fields(parsed)

    # 접두부 토큰(원본 보존)
    anima_artists = _split_tokens(blocks.get("ANIMA_ARTIST", ""))
    sdxl_artists = _split_tokens(blocks.get("SDXL_ARTIST", ""))
    anima_quality = _split_tokens(blocks.get("ANIMA_QUALITY", ""))
    sdxl_quality = _split_tokens(blocks.get("SDXL_QUALITY", ""))

    setup = scene["scene_setup"]
    char = scene["scene_char"]
    supplement = scene["scene_supplement"]

    # scene 필드가 전부 비면 수정 없음 — 원본 유지
    if not (setup or char or supplement):
        print("[LLM_EDIT] reassemble: scene 필드 전부 비어 원본 유지")
        return positive, scene

    # 트리거 순서를 원본 블럭에서 보존(셋 비순회) — ANIMA/SDXL 각각
    def _ordered_triggers(block_content, trigger_set):
        seen = set()
        out = []
        for tok in _split_tokens(block_content):
            if tok in trigger_set and tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out

    anima_triggers = _ordered_triggers(blocks.get("ANIMA_CONTENT", ""), triggers["anima"])
    sdxl_triggers = _ordered_triggers(blocks.get("SDXL", ""), triggers["sdxl"])
    # 복원 실패 폴백(CHAR_LIST 이름)도 원본에 없을 수 있으니 셋 기반 보강
    for t in triggers["anima"]:
        if t not in anima_triggers:
            anima_triggers.append(t)
    for t in triggers["sdxl"]:
        if t not in sdxl_triggers:
            sdxl_triggers.append(t)

    # ANIMA_CONTENT = 트리거 + 아티스트 + setup + char + supplement
    anima_content = _join_tags(*anima_triggers, *anima_artists, setup, char, supplement)
    # ANIMA_ALL = 트리거 + 아티스트 + 품질 + setup + char + supplement
    anima_all = _join_tags(*anima_triggers, *anima_artists, *anima_quality, setup, char, supplement)
    # SDXL = 트리거 + 아티스트 + 품질 + setup + char (supplement 제외)
    sdxl = _join_tags(*sdxl_triggers, *sdxl_artists, *sdxl_quality, setup, char)

    new_contents = {
        "ANIMA_CONTENT": anima_content,
        "ANIMA_ALL": anima_all,
        "SDXL": sdxl,
    }

    def _repl(m):
        header_line = m.group(1)  # "[ANIMA_CONTENT]\n" 등
        header_name = re.match(r"\[([A-Z_]+)\]", header_line).group(1)
        new_content = new_contents.get(header_name, "")
        if not new_content:
            # 빈 결과면 원본 라인 유지(치환하지 않음)
            return m.group(0)
        return header_line + new_content

    reassembled = _SUBJECT_RE.sub(_repl, positive)
    return reassembled, scene