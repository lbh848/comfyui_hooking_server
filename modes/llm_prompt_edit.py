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
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
BOT_JSON_PATH = os.path.join(ASSET_DATA_DIR, "bot.json")

# ─── LLM 보조(시스템) 프롬프트 — 배포용 builtin + 사용자 custom ───
# lb_extra_refine 패턴과 동일: builtin(배포용·읽기전용) / custom(이PC 전용·편집가능) /
# meta(use_custom 토글). effective = (use_custom && custom) ? custom : builtin.
LLM_EDIT_PROMPT_DIR = os.path.join(BASE_DIR, "prompts", "llm_prompt_edit")

# builtin(배포용·읽기전용) 템플릿 — system 역할 / user 지시 V3 / user 지시 V1
LLM_EDIT_BUILTIN_FILE = os.path.join(LLM_EDIT_PROMPT_DIR, "system.txt")
LLM_EDIT_BUILTIN_SYSTEM_CHANSUB_FILE = os.path.join(LLM_EDIT_PROMPT_DIR, "system_chansub.txt")
LLM_EDIT_BUILTIN_USER_V3_FILE = os.path.join(LLM_EDIT_PROMPT_DIR, "user_template.txt")
LLM_EDIT_BUILTIN_USER_V1_FILE = os.path.join(LLM_EDIT_PROMPT_DIR, "user_template_v1.txt")
LLM_EDIT_BUILTIN_USER_CHANSUB_FILE = os.path.join(LLM_EDIT_PROMPT_DIR, "user_template_chansub.txt")

# custom(이 PC 전용·편집가능) 템플릿 — 슬롯별 파일. 빈 칸이면 해당 슬롯은 builtin 폴백.
LLM_EDIT_CUSTOM_FILE = os.path.join(ASSET_DATA_DIR, "llm_prompt_edit_custom.txt")  # system
LLM_EDIT_CUSTOM_SYSTEM_CHANSUB_FILE = os.path.join(ASSET_DATA_DIR, "llm_prompt_edit_custom_system_chansub.txt")
LLM_EDIT_CUSTOM_USER_V3_FILE = os.path.join(ASSET_DATA_DIR, "llm_prompt_edit_custom_user_v3.txt")
LLM_EDIT_CUSTOM_USER_V1_FILE = os.path.join(ASSET_DATA_DIR, "llm_prompt_edit_custom_user_v1.txt")
LLM_EDIT_CUSTOM_USER_CHANSUB_FILE = os.path.join(ASSET_DATA_DIR, "llm_prompt_edit_custom_user_chansub.txt")

LLM_EDIT_META_FILE = os.path.join(ASSET_DATA_DIR, "llm_prompt_edit_meta.json")

# builtin 파일이 없을 때의 폴백 기본값(배포 전/파일 누락 대비). builtin 파일이 항상 존재하므로
# 정상 동작에서는 거의 쓰이지 않는다.
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert editor for the positive prompt of a Stable Diffusion Illust workflow. "
    "Analyze the user's 'edit direction', the generated image (if provided), and the current scene tags, "
    "then modify ONLY the scene description (setup/char/supplement). "
    "Return ONLY a valid JSON object - no other text, no markdown code fences, no explanations."
)

DEFAULT_SYSTEM_CHANSUB_PROMPT = (
    "You are an expert editor for ComfyUI image prompts sent through an NAI-shaped HTTP API. "
    "Analyze the user's direction, image, and complete POSITIVE/NEGATIVE prompts. Preserve "
    "ComfyUI parenthesized weight syntax. Return only valid JSON with plan, positive, and "
    "negative fields. Never add LoRA syntax or workflow control blocks."
)

DEFAULT_USER_V3_TEMPLATE = (
    "## User edit direction\n{direction}\n\n"
    "## Current scene tags (ANIMA, includes supplement)\n{scene_current}\n\n"
    "## Current scene tags (SDXL, no supplement)\n{scene_sdxl}\n\n"
    "## Instructions\n"
    "1. Look at the provided image (if any) and the tags above to understand the current scene.\n"
    "2. Adjust scene_setup/scene_char/scene_supplement to match the user's edit direction.\n"
    "3. Preserve the character's core identity (key appearance traits); focus changes on scene/mood/pose/outfit.\n"
    "4. Keep original tags for any part that does not need changing (avoid unnecessary edits).\n"
    "5. scene_supplement is NOT applied to the SDXL block, so only put SDXL-absent auxiliary descriptions there.\n"
    "6. Keep using English danbooru-style tags separated by comma+space (\", \").\n"
    "7. Write the \"plan\" field in Korean; write scene_* tags in English.\n\n"
    "## Output (JSON schema - return ONLY a JSON object in this form)\n"
    "{\n"
    '  "plan": "edit plan - briefly explain why and which tags to add/remove (write in Korean)",\n'
    '  "scene_setup": "background/location/lighting/weather/mood tags, comma+space separated",\n'
    '  "scene_char": "character appearance/pose/expression/outfit tags, comma+space separated",\n'
    '  "scene_supplement": "extra auxiliary tags, or empty string \\"\\"\n"'
    "}"
)

DEFAULT_USER_V1_TEMPLATE = (
    "## User edit direction\n{direction}\n\n"
    "## Current scene tags (setup / background)\n{setup}\n\n"
    "## Current scene tags (char / character + pose + scene)\n{char}\n\n"
    "## Current scene tags (supplement / extra)\n{supplement}\n\n"
    "## Instructions\n"
    "1. Look at the provided image (if any) and the tags above to understand the current scene.\n"
    "2. This is a V1 prompt where `char` is the WHOLE scene (background + character + pose + "
    "expression + outfit + auxiliary tags combined). Put the ENTIRE edited scene into "
    "`scene_char`. Leave `scene_setup` and `scene_supplement` as empty strings (\"\"). "
    "Do NOT move tags between fields.\n"
    "3. Preserve the character's core identity (the character/series name tag at the start of "
    "scene_char, e.g. \"shifty \\(nikke\\)\"); focus changes on scene/mood/pose/outfit.\n"
    "4. Keep original tags for any part that does not need changing (avoid unnecessary edits).\n"
    "5. Keep using English danbooru-style tags separated by comma+space (\", \").\n"
    "6. Write the \"plan\" field in Korean; write scene_* tags in English.\n\n"
    "## Output (JSON schema - return ONLY a JSON object in this form)\n"
    "{\n"
    '  "plan": "edit plan - briefly explain why and which tags to add/remove (write in Korean)",\n'
    '  "scene_setup": "background/location/lighting/weather/mood tags, comma+space separated",\n'
    '  "scene_char": "character appearance/pose/expression/outfit tags, comma+space separated",\n'
    '  "scene_supplement": "extra auxiliary tags, or empty string \\"\\"\n"'
    "}"
)

DEFAULT_USER_CHANSUB_TEMPLATE = (
    "## User edit direction\n{direction}\n\n"
    "## Current ComfyUI POSITIVE\n{positive}\n\n"
    "## Current ComfyUI NEGATIVE\n{negative}\n\n"
    "Edit the complete ComfyUI-compatible POSITIVE and NEGATIVE prompts. Preserve unaffected "
    "character identity, artist/style, and quality tags. Never add LoRA syntax or workflow "
    "control blocks. Return only JSON with Korean plan and English positive/negative fields."
)

# builtin 텍스트 로드 캐시 — 경로별 mtime 기반
_builtin_cache = {}  # path -> (mtime, text)


def _load_builtin_text(path: str, fallback: str) -> str:
    """배포용 builtin 텍스트 로드. mtime 기반 캐싱. 파일 없으면 fallback."""
    if not os.path.isfile(path):
        return fallback
    try:
        mtime = os.path.getmtime(path)
        cached = _builtin_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        _builtin_cache[path] = (mtime, txt)
        return txt
    except Exception as e:
        print(f"[LLM_EDIT] builtin 로드 실패({path}): {e}")
        traceback.print_exc()
        return fallback


def _load_llm_edit_builtin() -> str:
    """system 역할 builtin (서버/기존 호출 호환 래퍼)."""
    return _load_builtin_text(LLM_EDIT_BUILTIN_FILE, DEFAULT_SYSTEM_PROMPT)


def _load_system_chansub_builtin() -> str:
    """챈섭 Comfy system 역할 builtin 템플릿 로드."""
    return _load_builtin_text(
        LLM_EDIT_BUILTIN_SYSTEM_CHANSUB_FILE, DEFAULT_SYSTEM_CHANSUB_PROMPT
    )


def _load_user_v3_builtin() -> str:
    """user 지시 V3 builtin 템플릿 로드."""
    return _load_builtin_text(LLM_EDIT_BUILTIN_USER_V3_FILE, DEFAULT_USER_V3_TEMPLATE)


def _load_user_v1_builtin() -> str:
    """user 지시 V1 builtin 템플릿 로드."""
    return _load_builtin_text(LLM_EDIT_BUILTIN_USER_V1_FILE, DEFAULT_USER_V1_TEMPLATE)


def _load_user_chansub_builtin() -> str:
    """챈섭 Comfy user 지시 builtin 템플릿 로드."""
    return _load_builtin_text(
        LLM_EDIT_BUILTIN_USER_CHANSUB_FILE, DEFAULT_USER_CHANSUB_TEMPLATE
    )


# 슬롯 정의 — builtin 로더 / custom 경로 매핑. 드롭다운 편집 대상 5종.
SLOTS = {
    "system": {"builtin": _load_llm_edit_builtin, "custom_file": LLM_EDIT_CUSTOM_FILE},
    "system_chansub": {
        "builtin": _load_system_chansub_builtin,
        "custom_file": LLM_EDIT_CUSTOM_SYSTEM_CHANSUB_FILE,
    },
    "user_v3": {"builtin": _load_user_v3_builtin, "custom_file": LLM_EDIT_CUSTOM_USER_V3_FILE},
    "user_v1": {"builtin": _load_user_v1_builtin, "custom_file": LLM_EDIT_CUSTOM_USER_V1_FILE},
    "user_chansub": {
        "builtin": _load_user_chansub_builtin,
        "custom_file": LLM_EDIT_CUSTOM_USER_CHANSUB_FILE,
    },
}


def _read_text_file(path: str) -> str:
    """텍스트 파일 읽기. 없거나 실패 시 ''(빈문자열). 예외 미전파."""
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[LLM_EDIT] 텍스트 로드 실패({path}): {e}")
        traceback.print_exc()
        return ""


def _load_llm_edit_custom() -> tuple:
    """3종 custom 텍스트 + use_custom 플래그 로드.

    반환: (customs, use_custom)
      customs = {"system": str, "user_v3": str, "user_v1": str} (파일 없으면 '')
      use_custom = bool (메타 없으면 False)
    """
    customs = {slot: _read_text_file(meta["custom_file"]) for slot, meta in SLOTS.items()}

    use_custom = False
    if os.path.isfile(LLM_EDIT_META_FILE):
        try:
            with open(LLM_EDIT_META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                use_custom = bool(meta.get("use_custom", False))
        except Exception as e:
            print(f"[LLM_EDIT] meta 로드 실패: {e}")
            traceback.print_exc()

    return customs, use_custom


def _save_llm_edit_custom(customs: dict, use_custom: bool) -> None:
    """3종 custom 텍스트 + use_custom 저장. 기존 파일은 .bak 로 백업.

    customs 는 SLOTS의 5개 슬롯 문자열을 담는다. 누락 슬롯은 '' 취급.
    """
    os.makedirs(ASSET_DATA_DIR, exist_ok=True)
    requirements_dir = os.path.join(BASE_DIR, "요구사항")
    os.makedirs(requirements_dir, exist_ok=True)

    def _backup_required(path: str, slot: str) -> None:
        if not os.path.isfile(path):
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = os.path.join(
            requirements_dir,
            f"llm_prompt_edit_{slot}_before_save_{stamp}{os.path.splitext(path)[1]}",
        )
        shutil.copy2(path, backup_path)
        print(f"[LLM_EDIT] {slot} 기존 파일 요구사항 백업 완료: {backup_path}")

    for slot, text in customs.items():
        path = SLOTS.get(slot, {}).get("custom_file")
        if not path:
            print(f"[LLM_EDIT] 저장: 알 수 없는 슬롯 '{slot}' 스킵")
            continue
        text = text or ""
        if os.path.isfile(path):
            try:
                _backup_required(path, slot)
                shutil.copy2(path, path + ".bak")
            except Exception as e:
                print(f"[LLM_EDIT] {slot} 백업 실패: {e}")
                traceback.print_exc()
                raise
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[LLM_EDIT] {slot} 저장 실패: {e}")
            traceback.print_exc()
            raise

    if os.path.isfile(LLM_EDIT_META_FILE):
        try:
            _backup_required(LLM_EDIT_META_FILE, "meta")
            shutil.copy2(LLM_EDIT_META_FILE, LLM_EDIT_META_FILE + ".bak")
        except Exception as e:
            print(f"[LLM_EDIT] meta 백업 실패: {e}")
            traceback.print_exc()
            raise

    try:
        with open(LLM_EDIT_META_FILE, "w", encoding="utf-8") as f:
            json.dump({"use_custom": bool(use_custom)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LLM_EDIT] meta 저장 실패: {e}")
        traceback.print_exc()
        raise


def _effective_text(slot: str, custom_text: str = None, use_custom: bool = None) -> str:
    """슬롯의 실제 적용 텍스트.

    규칙: use_custom 이 켜져 있고 해당 슬롯 custom 텍스트가 비어있지 않으면 custom,
    아니면 builtin. custom_text/use_custom 이 None 이면 디스크에서 로드한다.
    """
    if use_custom is None or custom_text is None:
        customs, uc = _load_llm_edit_custom()
        if use_custom is None:
            use_custom = uc
        if custom_text is None:
            custom_text = customs.get(slot, "")
    if use_custom and custom_text.strip():
        return custom_text
    return SLOTS[slot]["builtin"]()


def get_effective_system_prompt() -> str:
    """실제 LLM 호출에 사용할 system 프롬프트."""
    return _effective_text("system")


def get_effective_system_prompt_chansub() -> str:
    """챈섭 Comfy 프롬프트 편집에 사용할 system 프롬프트."""
    return _effective_text("system_chansub")


def get_effective_user_template_v3() -> str:
    """실제 LLM 호출에 사용할 user 지시(V3) 템플릿."""
    return _effective_text("user_v3")


def get_effective_user_template_v1() -> str:
    """실제 LLM 호출에 사용할 user 지시(V1) 템플릿."""
    return _effective_text("user_v1")


def get_effective_user_template_chansub() -> str:
    """실제 LLM 호출에 사용할 챈섭 Comfy user 지시 템플릿."""
    return _effective_text("user_chansub")

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


def _substitute_placeholders(template: str, values: dict) -> str:
    """template 안의 {key} 플레이스홀더를 values[key] 로 치환.

    str.format 이 아니다(안전) — danbooru 태그/JSON 스키마에 중괄호가 섞여 있어 포맷
    이스케이프를 강제하기보다 정확한 리터럴 치환이 안전하다.
    값이 None 이거나 빈 문자열이면 '(none)' 으로 치환(빈 섹션 방지).
    치환 후에도 남은 {word} 패턴이 있으면 경고 로그(오타/알 수 없는 키).
    """
    out = template
    for key, val in values.items():
        ph = "{" + key + "}"
        if val is None or str(val).strip() == "":
            val = "(none)"
        else:
            val = str(val)
        out = out.replace(ph, val)

    leftover = set(re.findall(r"\{[a-z_]+\}", out))
    if leftover:
        print(f"[LLM_EDIT] 치환되지 않은 플레이스홀더가 템플릿에 남음: {leftover}")
    return out


def build_llm_messages(direction: str, scene_current: str, scene_sdxl: str) -> list:
    """LLM(비전) 호출용 messages 빌드 (V3 빌드본).

    이미지는 callLLMVision 의 _build_vision_messages 가 마지막 user 메시지에
    image_url 파트로 추가하므로, 여기서는 텍스트만 작성한다.
    system/user 모두 get_effective_* (builtin 또는 사용자 custom 빈칸폴백) 템플릿을
    사용하고, 동적 데이터(direction/scene_*)는 플레이스홀더로 치환한다.
    """
    system = get_effective_system_prompt()
    template = get_effective_user_template_v3()
    user = _substitute_placeholders(template, {
        "direction": direction,
        "scene_current": scene_current,
        "scene_sdxl": scene_sdxl,
    })

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _repair_json_escapes(text: str) -> str:
    """danbooru 태그의 잘못된 JSON 이스케이프를 복구해 json.loads 가 가능하게 한다.

    JSON 이 허용하는 백슬래시 이스케이프는 \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX 뿐인데,
    LLM 이 프롬프트 안의 \\\\(, \\\\), \\\\:, \\\\! 등을 그대로 JSON 문자열에 넣어버리면
    json.loads 가 "Invalid \\escape" 로 실�다. 이 함수는 유효 이스케이프가 아닌 \\\\X 를
    \\\\\\\\X(= 리터럴 백슬래시 + X)로 변환한다. 이미 정당하게 이스케이프된 시퀀스는 그대로 둔다.
    """
    out = []
    i = 0
    n = len(text)
    # 단문자 유효 이스케이프 다음 문자 (u 제외 — \uXXXX 별도 처리)
    valid_single = set('"\\/bfnrt')
    while i < n:
        c = text[i]
        if c == '\\' and i + 1 < n:
            nxt = text[i + 1]
            if nxt in valid_single:
                out.append(c)
                out.append(nxt)
                i += 2
                continue
            if nxt == 'u' and i + 5 < n:
                # \u 뒤 4자리 hex 면 유효 이스케이프 — 그대로
                hex4 = text[i + 2:i + 6]
                if all(ch in '0123456789abcdefABCDEF' for ch in hex4):
                    out.append(c)
                    out.append(text[i + 1:i + 6])
                    i += 6
                    continue
            # 잘못된 이스케이프: 백슬래시를 두 배로(리터럴 백슬래시 + 원래 문자)
            out.append('\\\\')
            out.append(nxt)
            i += 2
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def parse_llm_json(raw: str):
    """LLM 응답에서 JSON 객체를 파싱. 실패 시 None.

    ```json ... ``` 코드펜스 제거 → json.loads → 실패 시 { ~ } 슬라이스 재시도 →
    여전히 실패하면 danbooru 태그의 잘못된 이스케이프(\\(, \\) 등)를 복구해 재시도.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()

    # 코드펜스 제거
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 1차: 그대로 파싱
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2차: { ... } 슬라이스 추출 재시도
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        print(f"[LLM_EDIT] parse_llm_json: JSON 객체를 찾을 수 없음 (raw 앞 200자: {text[:200]!r})")
        return None

    slice_ = text[first:last + 1]
    try:
        return json.loads(slice_)
    except Exception:
        pass

    # 3차: 잘못된 이스케이프 복구 후 재시도
    repaired = _repair_json_escapes(slice_)
    try:
        return json.loads(repaired)
    except Exception as e:
        print(f"[LLM_EDIT] parse_llm_json: JSON 파싱 실패(이스케이프 복구 후에도): {e}\n"
              f"  repaired(앞 300자): {repaired[:300]!r}")
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


# ─── V1 (ILXL/UPSCALE) 프롬프트 지원 ──────────────────────────
# V1 구조 (prompt_enhance_mode._parse_prompt_sections 와 동일):
#   {prefix}, {char}, {supplement}        ← 상단(헤더 없음). prefix = 품질 태그
#   [ILXL] {setup}, {char}                 ← setup = ILXL 에서 char 제거한 나머지
#   [UPSCALE] {char}                       ← char = 캐릭터+장면(= UPSCALE 내용)
#
# V3 와 달리 트리거/아티스트/품질이 블럭별로 분리되지 않고 char 안에 섞여 있어
# 백엔드 완전 보존은 불가. 대신 prefix(품질)만 보존하고 char/setup/supplement 를
# LLM이 편집한다(시스템 프롬프트로 캐릭터 정체성 유지만 지시).

V1_REQUIRED_HEADERS = ["ILXL", "UPSCALE"]


def detect_v1_format(positive: str) -> bool:
    """V1(ILXL/UPSCALE) 포맷 여부. V3 가 아니면서 ILXL+UPSCALE 헤더가 둘 다 있으면 True."""
    if detect_build_format(positive):
        return False
    for h in V1_REQUIRED_HEADERS:
        if not re.search(rf"^\[{h}\]\n", positive, re.MULTILINE):
            return False
    return True


def detect_format(positive: str, provider: str = "") -> str:
    """빌드본 프롬프트 포맷 자동 감지.

    반환: "v3" | "v1" | "chansub" | ""(지원 불가)
    - v3: 삽화 빌드본(ANIMA_CONTENT/ANIMA_ALL/SDXL 등 8개 헤더)
    - v1: ILXL/UPSCALE 스타일(배치/비삽화 백업 V1)
    """
    if (provider or "").strip().lower() == "chansub":
        return "chansub"
    if detect_build_format(positive):
        return "v3"
    if detect_v1_format(positive):
        return "v1"
    return ""


def build_chansub_llm_messages(direction: str, positive: str, negative: str) -> list:
    """챈섭 Comfy POSITIVE/NEGATIVE 편집용 LLM messages를 만든다."""
    system = get_effective_system_prompt_chansub()
    template = get_effective_user_template_chansub()
    user = _substitute_placeholders(template, {
        "direction": direction,
        "positive": positive,
        "negative": negative,
    })
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def reassemble_chansub(original_positive: str, original_negative: str, parsed: dict) -> tuple:
    """LLM 결과에서 완전한 챈섭 POSITIVE/NEGATIVE를 안전하게 꺼낸다."""
    if not isinstance(parsed, dict):
        print("[LLM_EDIT:CHANSUB] 응답이 객체가 아니어서 원본 유지")
        return original_positive, original_negative, {"plan": ""}

    plan = parsed.get("plan", "")
    positive = parsed.get("positive", "")
    negative = parsed.get("negative", "")
    plan = str(plan).strip() if plan is not None else ""
    if not isinstance(positive, str) or not positive.strip():
        print("[LLM_EDIT:CHANSUB] positive가 비었거나 문자열이 아니어서 원본 유지")
        positive = original_positive
    else:
        positive = positive.strip()
    if not isinstance(negative, str):
        print("[LLM_EDIT:CHANSUB] negative가 문자열이 아니어서 원본 유지")
        negative = original_negative
    else:
        negative = negative.strip()
    return positive, negative, {"plan": plan}


def _split_char_blocks(text: str) -> list:
    """| 로 구분된 캐릭터 블럭 분리. 괄호 안의 | 는 무시(prompt_enhance_mode 와 동일)."""
    if not text:
        return []
    blocks = []
    current = []
    paren_depth = 0
    for ch in text:
        if ch == '(':
            paren_depth += 1
            current.append(ch)
        elif ch == ')':
            paren_depth = max(0, paren_depth - 1)
            current.append(ch)
        elif ch == '|' and paren_depth == 0:
            block = ''.join(current).strip()
            if block:
                blocks.append(block)
            current = []
        else:
            current.append(ch)
    block = ''.join(current).strip()
    if block:
        blocks.append(block)
    return blocks


def _v1_extract_section(positive: str, section_name: str) -> str:
    """positive 에서 [SECTION_NAME] 내용 추출. 다음 [SECTION] 또는 끝까지."""
    pattern = rf'\[{section_name}\]\s*(.*?)(?=\n\s*\[|\Z)'
    m = re.search(pattern, positive, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _v1_replace_section_content(text: str, section_name: str, new_content: str) -> str:
    """텍스트에서 [SECTION] 의 내용을 교체. 헤더가 없으면 원본 반환."""
    pattern = rf'(\[{section_name}\]\s*)(.*?)(?=\n\s*\[|\Z)'
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        return text[:m.start(2)] + new_content + text[m.end(2):]
    return text


def parse_v1_sections(positive: str) -> dict:
    """V1 positive 를 {prefix, char, setup, supplement} 로 파싱.

    - char     = [UPSCALE] 내용
    - setup    = [ILXL] 내용에서 char 블럭 제거한 나머지
    - prefix   = 상단(첫 헤더 이전)에서 char 이전 부분(품질 태그)
    - supplement = 상단에서 char 이후 부분

    char 가 상단에 없으면 prefix/supplement 분리를 포기하고 top 전체를 prefix 로 둬
    데이터가 유실되지 않게 한다.
    """
    result = {"prefix": "", "char": "", "setup": "", "supplement": ""}

    # 1. [UPSCALE] → char
    char = _v1_extract_section(positive, "UPSCALE")
    result["char"] = char
    if not char:
        print("[LLM_EDIT][V1] UPSCALE 비어 있음 — 파싱 불가")
        return result

    # 2. [ILXL]에서 char 제거 → setup
    ilxl = _v1_extract_section(positive, "ILXL")
    if ilxl:
        setup = ilxl
        for block in _split_char_blocks(char):
            b = block.strip().rstrip(',').strip()
            if b:
                setup = setup.replace(b, '')
        setup = re.sub(r',\s*,', ',', setup).strip().strip(',').strip()
        result["setup"] = setup

    # 3. 상단에서 prefix / supplement 분리
    top = re.split(r'\n\s*\[', positive)[0].strip()
    if top:
        char_text = char.strip().rstrip(',').strip()
        idx = top.rfind(char_text) if char_text else -1
        if idx >= 0:
            result["prefix"] = top[:idx].strip().rstrip(',').strip()
            result["supplement"] = top[idx + len(char_text):].strip().lstrip(',').strip()
        else:
            # char 를 상단에서 못 찾으면 top 전체를 prefix 로 보존(데이터 유실 방지)
            print(f"[LLM_EDIT][V1] 상단에서 char 미검출 — top 전체를 prefix 로 보존 "
                  f"(char 앞 {len(char_text)}자)")
            result["prefix"] = top

    return result


def build_v1_llm_messages(direction: str, char: str, setup: str, supplement: str) -> list:
    """V1 용 LLM(비전) 호출 messages 빌드 (ILXL/UPSCALE).

    시스템 프롬프트는 V3 와 동일(get_effective_system_prompt). user 지시는 V1 전용
    템플릿(get_effective_user_template_v1)을 사용하고, 단일 scene(setup/char/supplement)
    을 플레이스홀더로 치환한다.
    """
    system = get_effective_system_prompt()
    template = get_effective_user_template_v1()
    user = _substitute_placeholders(template, {
        "direction": direction,
        "setup": setup,
        "char": char,
        "supplement": supplement,
    })

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def reassemble_v1(positive: str, parsed_v1: dict, parsed: dict) -> tuple:
    """V1 결과를 3개 블럭(상단/ILXL/UPSCALE)에 주입해 positive 재조립.

    반환: (reassembled_positive, scene_dict)
    - parsed_v1: parse_v1_sections() 결과({prefix, char, setup, supplement})
    - parsed: LLM 응답 JSON(원본). 내부에서 _coerce_scene_fields 로 정규화.
    - prefix(품질)는 보존. char/setup/supplement 만 교체.
    - 구조([ILXL]/[UPSCALE] 헤더)가 깨지면 원본 유지.

    두 경로:
    (a) nikke-style — 원본 setup/supplement 가 비어 있음(char=전체 장면). LLM 응답의
        setup/char/supplement 를 하나의 effective 장면으로 병합해 원본 char 를 전역 치환.
        LLM 이 setup/supplement 에 태그를 넣어도 유실되지 않는다.
    (b) 일반 V1 — 원본 setup/supplement 가 비어있지 않음. char는 전역 치환, setup/
        supplement 는 기존 위치(ILXL/top)에서 교체.
    """
    scene = _coerce_scene_fields(parsed)
    setup = scene.get("scene_setup", "")
    char = scene.get("scene_char", "")
    supplement = scene.get("scene_supplement", "")

    # scene 필드 전부 비면 수정 없음 — 원본 유지
    if not (setup or char or supplement):
        print("[LLM_EDIT][V1] reassemble: scene 필드 전부 비어 원본 유지")
        return positive, scene

    orig_char = parsed_v1.get("char", "")
    orig_setup = parsed_v1.get("setup", "")
    orig_supplement = parsed_v1.get("supplement", "")

    result = positive
    new_char = char if char else orig_char

    if not orig_setup and not orig_supplement:
        # (a) nikke-style: setup/char/supplement 병합 → 전체 장면으로 원본 char 치환
        parts = []
        if setup:
            parts.append(setup)
        if new_char:
            parts.append(new_char)
        if supplement:
            parts.append(supplement)
        effective = ", ".join(parts)
        if effective and orig_char and effective != orig_char:
            result = result.replace(orig_char, effective)
    else:
        # (b) 일반 V1: 위치 기반 교체
        # 1) char 전역 치환 (상단/ILXL/UPSCALE 에 동일)
        if new_char and orig_char and new_char != orig_char:
            result = result.replace(orig_char, new_char)

        # 2) setup 교체 — ILXL 내용 + 상단 prefix 안의 기존 setup
        if setup and orig_setup and setup != orig_setup:
            ilxl_content = _v1_extract_section(result, "ILXL")
            if ilxl_content:
                new_ilxl = ilxl_content.replace(orig_setup, setup, 1)
                result = _v1_replace_section_content(result, "ILXL", new_ilxl)
            first_section = re.search(r'\n\s*\[', result)
            if first_section:
                top = result[:first_section.start()]
                rest = result[first_section.start():]
                result = top.replace(orig_setup, setup, 1) + rest

        # 3) supplement 교체 — 상단 char 이후의 기존 supplement
        if supplement and orig_supplement and supplement != orig_supplement:
            first_section = re.search(r'\n\s*\[', result)
            if first_section:
                top = result[:first_section.start()]
                rest = result[first_section.start():]
                result = top.replace(orig_supplement, supplement, 1) + rest

    # 4. 구조 검증 — [ILXL]/[UPSCALE] 헤더가 유지됐는지 확인, 깨지면 원본 유지
    for section in ("ILXL", "UPSCALE"):
        if f"[{section}]" in positive and f"[{section}]" not in result:
            print(f"[LLM_EDIT][V1] reassemble: [{section}] 구조 깨짐 — 원본 유지")
            return positive, scene

    return result, scene
