"""
Instance LoRA 매니징 모듈
- LV1 평면 구조: 봇/프로젝트 계층 없이 로라 단위로 관리
- 학습 이미지 = 프리뷰 이미지, 여러 장 등록 가능
- 태그 분석 후 1-pass 자동 학습
"""

import asyncio
import base64
import json
import os
import re
import shutil
import time
import traceback
from aiohttp import web

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_LORA_DIR = os.path.join(BASE_DIR, "instance_lora")
INSTANCE_LORA_MANAGE_FILE = os.path.join(BASE_DIR, "asset_data", "instance_lora_manage.json")
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# ─── LLM 자동 LoRA 프롬프트 정제 (auto_lora_prompt) ─────────────

AUTO_LORA_PROMPT_DIR = os.path.join(BASE_DIR, "prompts", "auto_lora_prompt")
AUTO_LORA_PROMPT_BUILTIN_FILE = os.path.join(AUTO_LORA_PROMPT_DIR, "system.txt")
AUTO_LORA_PROMPT_CUSTOM_FILE = os.path.join(ASSET_DATA_DIR, "auto_lora_prompt_custom.txt")
AUTO_LORA_PROMPT_META_FILE = os.path.join(ASSET_DATA_DIR, "auto_lora_prompt_meta.json")

# 에셋 전용(성별 정보 없음) — 캐릭터 LoRA 정제 프롬프트와 완전히 분리된 별도 템플릿/설정.
AUTO_LORA_PROMPT_BUILTIN_ASSET_FILE = os.path.join(AUTO_LORA_PROMPT_DIR, "asset_system.txt")
AUTO_LORA_PROMPT_CUSTOM_ASSET_FILE = os.path.join(ASSET_DATA_DIR, "auto_lora_prompt_asset_custom.txt")
AUTO_LORA_PROMPT_META_ASSET_FILE = os.path.join(ASSET_DATA_DIR, "auto_lora_prompt_asset_meta.json")

# 봇 LoRA "테스트 이미지 일괄 세팅" 전용 템플릿. 텍스트 only LLM.
# {test_prompt} = 공통 테스트 이미지 프롬프트(품질/자세/표정), {character_prompt} = 캐릭터 카드 수정 프롬프트(복장/외모).
BOT_TEST_SETUP_PROMPT_BUILTIN_FILE = os.path.join(AUTO_LORA_PROMPT_DIR, "bot_test_setup_system.txt")
BOT_TEST_SETUP_PROMPT_CUSTOM_FILE = os.path.join(ASSET_DATA_DIR, "bot_test_setup_prompt_custom.txt")
BOT_TEST_SETUP_PROMPT_META_FILE = os.path.join(ASSET_DATA_DIR, "bot_test_setup_prompt_meta.json")

_bot_test_setup_prompt_builtin_cache: str | None = None
_bot_test_setup_prompt_builtin_mtime: float = 0.0

_auto_lora_prompt_builtin_cache: str | None = None
_auto_lora_prompt_builtin_mtime: float = 0.0
_auto_lora_prompt_builtin_asset_cache: str | None = None
_auto_lora_prompt_builtin_asset_mtime: float = 0.0


def _load_auto_lora_prompt_builtin(is_asset: bool = False) -> str:
    """글로벌(배포용) 프롬프트 로드. mtime 기반 캐싱.

    is_asset=True 이면 에셋 전용 builtin(gender 없음)을 로드한다.
    """
    global _auto_lora_prompt_builtin_cache, _auto_lora_prompt_builtin_mtime
    global _auto_lora_prompt_builtin_asset_cache, _auto_lora_prompt_builtin_asset_mtime
    if is_asset:
        path = AUTO_LORA_PROMPT_BUILTIN_ASSET_FILE
        cache_val = _auto_lora_prompt_builtin_asset_cache
        mtime_val = _auto_lora_prompt_builtin_asset_mtime
    else:
        path = AUTO_LORA_PROMPT_BUILTIN_FILE
        cache_val = _auto_lora_prompt_builtin_cache
        mtime_val = _auto_lora_prompt_builtin_mtime
    if not os.path.isfile(path):
        print(f"[INSTANCE_LORA] auto_lora_prompt builtin 파일 없음: {path}")
        return ""
    try:
        mtime = os.path.getmtime(path)
        if cache_val is not None and mtime == mtime_val:
            return cache_val
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        if is_asset:
            _auto_lora_prompt_builtin_asset_cache = txt
            _auto_lora_prompt_builtin_asset_mtime = mtime
        else:
            _auto_lora_prompt_builtin_cache = txt
            _auto_lora_prompt_builtin_mtime = mtime
        return txt
    except Exception as e:
        print(f"[INSTANCE_LORA] auto_lora_prompt builtin 로드 실패: {e}")
        traceback.print_exc()
        return ""


def _load_auto_lora_prompt_custom(is_asset: bool = False) -> tuple[str, bool]:
    """커스텀 프롬프트와 use_custom 플래그 로드. (없으면 빈 문자열, False).

    is_asset=True 이면 에셋 전용 custom/meta 파일을 로드한다.
    """
    custom_file = AUTO_LORA_PROMPT_CUSTOM_ASSET_FILE if is_asset else AUTO_LORA_PROMPT_CUSTOM_FILE
    meta_file = AUTO_LORA_PROMPT_META_ASSET_FILE if is_asset else AUTO_LORA_PROMPT_META_FILE
    custom = ""
    if os.path.isfile(custom_file):
        try:
            with open(custom_file, "r", encoding="utf-8") as f:
                custom = f.read()
        except Exception as e:
            print(f"[INSTANCE_LORA] auto_lora_prompt custom 로드 실패 (asset={is_asset}): {e}")
            traceback.print_exc()

    use_custom = False
    if os.path.isfile(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                use_custom = bool(meta.get("use_custom", False))
        except Exception as e:
            print(f"[INSTANCE_LORA] auto_lora_prompt meta 로드 실패 (asset={is_asset}): {e}")
            traceback.print_exc()

    return custom, use_custom


def _save_auto_lora_prompt_custom(text: str, use_custom: bool, is_asset: bool = False) -> None:
    """커스텀 프롬프트 저장. 기존 파일은 .bak 로 백업.

    is_asset=True 이면 에셋 전용 custom/meta 파일에 저장한다.
    """
    os.makedirs(ASSET_DATA_DIR, exist_ok=True)
    custom_file = AUTO_LORA_PROMPT_CUSTOM_ASSET_FILE if is_asset else AUTO_LORA_PROMPT_CUSTOM_FILE
    meta_file = AUTO_LORA_PROMPT_META_ASSET_FILE if is_asset else AUTO_LORA_PROMPT_META_FILE

    if os.path.isfile(custom_file):
        try:
            shutil.copy2(custom_file, custom_file + ".bak")
        except Exception as e:
            print(f"[INSTANCE_LORA] auto_lora_prompt custom 백업 실패 (asset={is_asset}): {e}")

    try:
        with open(custom_file, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[INSTANCE_LORA] auto_lora_prompt custom 저장 실패 (asset={is_asset}): {e}")
        traceback.print_exc()
        raise

    if os.path.isfile(meta_file):
        try:
            shutil.copy2(meta_file, meta_file + ".bak")
        except Exception as e:
            print(f"[INSTANCE_LORA] auto_lora_prompt meta 백업 실패 (asset={is_asset}): {e}")

    try:
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"use_custom": bool(use_custom)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[INSTANCE_LORA] auto_lora_prompt meta 저장 실패 (asset={is_asset}): {e}")
        traceback.print_exc()
        raise


def _render_auto_lora_prompt_prompt(template: str, etc_tags: str, gender: str) -> str:
    """template 의 {etc} / {gender} 변수를 치환. format() 충돌 회피용 str.replace."""
    rendered = template.replace("{etc}", etc_tags or "")
    rendered = rendered.replace("{gender}", gender or "")
    return rendered


def _load_bot_test_setup_prompt_builtin() -> str:
    """봇 LoRA '테스트 이미지 일괄 세팅' 전용 builtin 프롬프트 로드. mtime 기반 캐싱."""
    global _bot_test_setup_prompt_builtin_cache, _bot_test_setup_prompt_builtin_mtime
    path = BOT_TEST_SETUP_PROMPT_BUILTIN_FILE
    if not os.path.isfile(path):
        print(f"[INSTANCE_LORA] bot_test_setup builtin 파일 없음: {path}")
        return ""
    try:
        mtime = os.path.getmtime(path)
        if _bot_test_setup_prompt_builtin_cache is not None and mtime == _bot_test_setup_prompt_builtin_mtime:
            return _bot_test_setup_prompt_builtin_cache
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        _bot_test_setup_prompt_builtin_cache = txt
        _bot_test_setup_prompt_builtin_mtime = mtime
        return txt
    except Exception as e:
        print(f"[INSTANCE_LORA] bot_test_setup builtin 로드 실패: {e}")
        traceback.print_exc()
        return ""


def _render_bot_test_setup_prompt(template: str, test_prompt: str, character_prompt: str) -> str:
    """bot_test_setup 템플릿의 {test_prompt} / {character_prompt} 변수 치환.
    미리보기와 실제 전송이 반드시 이 단일 함수만 경유하도록 한다 (CLAUDE.md)."""
    rendered = template.replace("{test_prompt}", test_prompt or "")
    rendered = rendered.replace("{character_prompt}", character_prompt or "")
    return rendered


def _load_bot_test_setup_prompt_custom() -> tuple[str, bool]:
    """bot_test_setup 커스텀 프롬프트와 use_custom 플래그 로드. (없으면 빈 문자열, False)."""
    custom = ""
    if os.path.isfile(BOT_TEST_SETUP_PROMPT_CUSTOM_FILE):
        try:
            with open(BOT_TEST_SETUP_PROMPT_CUSTOM_FILE, "r", encoding="utf-8") as f:
                custom = f.read()
        except Exception as e:
            print(f"[INSTANCE_LORA] bot_test_setup custom 로드 실패: {e}")
            traceback.print_exc()

    use_custom = False
    if os.path.isfile(BOT_TEST_SETUP_PROMPT_META_FILE):
        try:
            with open(BOT_TEST_SETUP_PROMPT_META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                use_custom = bool(meta.get("use_custom", False))
        except Exception as e:
            print(f"[INSTANCE_LORA] bot_test_setup meta 로드 실패: {e}")
            traceback.print_exc()

    return custom, use_custom


def _save_bot_test_setup_prompt_custom(text: str, use_custom: bool) -> None:
    """bot_test_setup 커스텀 프롬프트 저장. 기존 파일은 .bak 로 백업."""
    os.makedirs(ASSET_DATA_DIR, exist_ok=True)
    custom_file = BOT_TEST_SETUP_PROMPT_CUSTOM_FILE
    meta_file = BOT_TEST_SETUP_PROMPT_META_FILE

    if os.path.isfile(custom_file):
        try:
            shutil.copy2(custom_file, custom_file + ".bak")
        except Exception as e:
            print(f"[INSTANCE_LORA] bot_test_setup custom 백업 실패: {e}")

    try:
        with open(custom_file, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[INSTANCE_LORA] bot_test_setup custom 저장 실패: {e}")
        traceback.print_exc()
        raise

    if os.path.isfile(meta_file):
        try:
            shutil.copy2(meta_file, meta_file + ".bak")
        except Exception as e:
            print(f"[INSTANCE_LORA] bot_test_setup meta 백업 실패: {e}")

    try:
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"use_custom": bool(use_custom)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[INSTANCE_LORA] bot_test_setup meta 저장 실패: {e}")
        traceback.print_exc()
        raise


def _parse_auto_lora_prompt_response(raw: str, gender_fallback: str = "") -> dict | None:
    """LLM 응답에서 {"positive": "..."} 추출. 실패 시 None."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            positive = data.get("positive")
            if isinstance(positive, str):
                pos = positive.strip()
                if pos:
                    return {"positive": pos}
                if gender_fallback:
                    return {"positive": gender_fallback}
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                positive = data.get("positive", "")
                if isinstance(positive, str) and positive.strip():
                    return {"positive": positive.strip()}
        except json.JSONDecodeError as e:
            print(f"[INSTANCE_LORA] auto_lora_prompt JSON 파싱 실패: {e}")
    return None


async def run_auto_refine_lora_prompt(
    char_name: str,
    filename: str,
    current_positive: str,
    source_type: str = "bot",
    bot_name: str = "",
    project_name: str = "",
    entry: str = "",
    gender_override: str = "",
    is_asset: bool = False,
) -> dict:
    """LLM 비전 기반 LoRA 프롬프트 정제 (core).

    source_type:
      - "bot":               bot_name + char_name + filename → 봇 캐릭터 원본 이미지
      - "bot_lora_training": bot_name + project_name + char_name + filename → 봇 LoRA 학습 이미지
      - "training":          char_name + entry + filename → 에셋 LoRA 학습 이미지

    반환: {"success": True, "data": {"positive": "..."}} 또는 {"success": False, "error": "..."}
    """
    from modes.llm_service import callLLMVision, supports_vision, get_config
    from modes.lighbd_service import _log_lighbd_history
    import datetime

    async def _notify_llm_widget(event_type: str, data: dict = None):
        try:
            import server as _server
            await _server.notify_frontend("lighbd_llm_stream", {"type": event_type, **(data or {})})
        except Exception as e:
            print(f"[INSTANCE_LORA] WARN: notify_frontend 실패: {e}")

    try:
        source_type = (source_type or "bot").strip().lower()
        if source_type not in ("bot", "bot_lora_training", "training"):
            return {"success": False, "error": f"지원하지 않는 source_type: {source_type}"}
        if not char_name or not filename:
            return {"success": False, "error": "character, filename 필드가 필요합니다."}
        if source_type == "bot" and not bot_name:
            return {"success": False, "error": "bot 소스는 bot 필드가 필요합니다."}
        if source_type == "bot_lora_training" and (not bot_name or not project_name):
            return {"success": False, "error": "bot_lora_training 소스는 bot, project 필드가 필요합니다."}
        if source_type == "training" and not entry:
            return {"success": False, "error": "training 소스는 entry 필드가 필요합니다."}
        if not current_positive or not current_positive.strip():
            return {"success": False, "error": "정제할 긍정 프롬프트가 비어 있습니다."}

        # 성별 태그: override > bot.json 캐릭터 gender_tag > 기본 1girl
        # 에셋은 성별 정보가 없으므로 gender를 아예 사용하지 않는다 (빈 문자열).
        gender_tag = (gender_override or "").strip()
        if is_asset:
            gender_tag = ""
        elif not gender_tag and source_type in ("bot", "bot_lora_training"):
            try:
                from modes.bot_mode import _load_bot_data
                data = _load_bot_data()
                bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
                char = next((c for c in (bot.get("characters", []) if bot else []) if c["name"] == char_name), None)
                if char:
                    gender_tag = (char.get("gender_tag") or "").strip()
            except Exception as e:
                print(f"[INSTANCE_LORA] gender_tag 로드 실패: {e}")
                traceback.print_exc()
        if not is_asset and not gender_tag:
            gender_tag = "1girl"
            print(f"[INSTANCE_LORA] gender_tag 비어 있어 기본값 사용: {gender_tag}")

        # 원본 이미지 경로
        img_path = None
        if source_type == "bot":
            from modes.bot_lora_mode import get_bot_char_image_path
            img_path = get_bot_char_image_path(bot_name, char_name, filename)
            if not img_path or not os.path.isfile(img_path):
                print(f"[INSTANCE_LORA] 원본 이미지 없음: bot={bot_name} char={char_name} filename={filename}")
                return {"success": False, "error": f"원본 이미지를 찾을 수 없습니다: {filename} (bot={bot_name} character={char_name})"}
        elif source_type == "bot_lora_training":
            from modes.bot_lora_mode import get_bot_training_image_path
            img_path = get_bot_training_image_path(bot_name, project_name, char_name, filename)
            if not img_path or not os.path.isfile(img_path):
                print(f"[INSTANCE_LORA] 봇 LoRA 학습 이미지 없음: bot={bot_name} project={project_name} char={char_name} filename={filename}")
                return {"success": False, "error": f"봇 LoRA 학습 이미지를 찾을 수 없습니다: {filename} (bot={bot_name} project={project_name} character={char_name})"}
        else:
            from modes.lora_mode import get_training_image_path
            img_path = get_training_image_path(char_name, entry, filename)
            if not img_path or not os.path.isfile(img_path):
                print(f"[INSTANCE_LORA] 학습 이미지 없음: char={char_name} entry={entry} filename={filename}")
                return {"success": False, "error": f"학습 이미지를 찾을 수 없습니다: {filename} (character={char_name} entry={entry})"}

        # 비전 서비스 확인
        cfg = get_config()
        service = cfg.get("llm_service", "")
        if not supports_vision(service):
            print(f"[INSTANCE_LORA] 비전 미지원 서비스: {service}")
            return {
                "success": False,
                "error": (
                    f"현재 LLM 서비스({service})는 비전(이미지 입력)을 지원하지 않습니다. "
                    "텍스트 전용 SDK를 사용하는 vertex 대신 OpenAI 호환/Gemini/Claude 등을 config.json에서 선택하세요."
                ),
            }

        # 템플릿 선택 + 변수 치환 (에셋은 gender 없는 에셋 전용 템플릿 사용)
        custom_text, use_custom = _load_auto_lora_prompt_custom(is_asset)
        if use_custom and custom_text.strip():
            template = custom_text
        else:
            template = _load_auto_lora_prompt_builtin(is_asset)
        if not template.strip():
            return {"success": False, "error": "프롬프트 템플릿이 비어 있습니다."}

        rendered = _render_auto_lora_prompt_prompt(template, current_positive, gender_tag)

        # 이미지 base64
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".png":
                image_mime = "image/png"
            elif ext in (".jpg", ".jpeg"):
                image_mime = "image/jpeg"
            elif ext == ".webp":
                image_mime = "image/webp"
            else:
                image_mime = "image/webp"
            with open(img_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            print(f"[INSTANCE_LORA] 원본 이미지 읽기 실패: {img_path} - {e}")
            traceback.print_exc()
            return {"success": False, "error": f"원본 이미지 읽기 실패: {e}"}
        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        messages = [
            {"role": "system", "content": "You are a precise prompt refiner. Follow the user's instructions exactly and respond in strict JSON."},
            {"role": "user", "content": rendered},
        ]

        if source_type == "bot":
            source_desc = f"bot={bot_name}"
        elif source_type == "bot_lora_training":
            source_desc = f"bot={bot_name} project={project_name}"
        else:
            source_desc = f"training entry={entry}"
        print(f"[INSTANCE_LORA] auto_refine_lora_prompt 호출: source={source_type} {source_desc} char={char_name} filename={filename} service={service} is_asset={is_asset} gender={gender_tag} etc_len={len(current_positive)} use_custom={use_custom}")

        use_model = cfg.get("llm_model", "")
        max_retries = max(0, int(cfg.get("auto_lora_prompt_max_retries", 2)))
        await _notify_llm_widget("start", {"model": use_model, "prompt_id": f"auto_lora_prompt:{source_type}:{char_name}/{filename}"})

        raw = None
        last_err = None
        total_elapsed = 0.0
        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                raw = await callLLMVision(messages, image_b64=img_b64, image_mime=image_mime)
            except Exception as call_err:
                print(f"[INSTANCE_LORA] callLLMVision 예외 (시도 {attempt + 1}/{max_retries + 1}): {call_err}")
                traceback.print_exc()
                last_err = f"{type(call_err).__name__}: {call_err}"
                raw = None
            total_elapsed += time.time() - t0

            if raw and not raw.startswith("[LLM 실패]"):
                parsed = _parse_auto_lora_prompt_response(raw, gender_fallback=gender_tag)
                if parsed is not None:
                    done_data = {
                        "text": raw,
                        "completion_tokens": max(1, len(raw) // 3),
                        "elapsed": round(total_elapsed, 3),
                        "tps": round((max(1, len(raw) // 3) / total_elapsed), 1) if total_elapsed > 0 else 0.0,
                        "ttft": None,
                    }
                    await _notify_llm_widget("done", done_data)
                    _log_lighbd_history({
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                        "prompt_id": f"auto_lora_prompt:{source_type}:{char_name}/{filename}",
                        "input": messages,
                        "output": raw,
                        "completion_tokens": done_data["completion_tokens"],
                        "elapsed": done_data["elapsed"],
                        "tps": done_data["tps"],
                        "status": "ok",
                    })
                    print(f"[INSTANCE_LORA] auto_refine_lora_prompt 완료: positive 길이={len(parsed['positive'])} (시도 {attempt + 1})")
                    return {"success": True, "data": parsed}
                last_err = f"LLM 응답을 JSON으로 파싱하지 못했습니다. raw: {raw[:300]}"
                print(f"[INSTANCE_LORA] LLM 응답 JSON 파싱 실패 (시도 {attempt + 1}/{max_retries + 1}). raw={raw[:500]}")
            else:
                last_err = f"LLM 호출 실패: {raw or '빈 응답'}"
                print(f"[INSTANCE_LORA] LLM 호출 실패 (시도 {attempt + 1}/{max_retries + 1}): {raw}")

            if attempt < max_retries:
                print(f"[INSTANCE_LORA] 재시도 대기 중... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1.0 * (attempt + 1))

        await _notify_llm_widget("error", {"error": last_err or "알 수 없는 오류", "elapsed": round(total_elapsed, 3)})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": f"auto_lora_prompt:{source_type}:{char_name}/{filename}",
            "input": messages,
            "output": "",
            "elapsed": round(total_elapsed, 3),
            "status": "error",
            "error": last_err or "알 수 없는 오류",
        })
        return {"success": False, "error": f"{max_retries + 1}회 시도 후 실패: {last_err}"}
    except Exception as e:
        print(f"[INSTANCE_LORA] auto_refine_lora_prompt 예외: {e}")
        traceback.print_exc()
        await _notify_llm_widget("error", {"error": f"{type(e).__name__}: {e}"})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": f"auto_lora_prompt:{source_type}:{char_name}/{filename}",
            "input": messages if "messages" in locals() else [],
            "output": "",
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        })
        return {"success": False, "error": str(e)}


async def run_auto_refine_test_setup(
    character: str,
    test_filename: str,
    card_positive: str,
    test_positive: str,
    bot_name: str = "",
    project_name: str = "",
) -> dict:
    """봇 LoRA '테스트 이미지 일괄 세팅' 텍스트 정제 (core, 비전 미사용).

    공통 테스트 이미지 프롬프트(test_positive)에서 품질/자세/표정을, 캐릭터 카드 수정
    프롬프트(card_positive)에서 복장/외모를 추출해 조합한 테스트용 positive 프롬프트 생성.

    반환: {"success": True, "data": {"positive": "..."}} 또는 {"success": False, "error": "..."}
    """
    from modes.llm_service import callLLM, get_config
    from modes.lighbd_service import _log_lighbd_history
    import datetime

    async def _notify_llm_widget(event_type: str, data: dict = None):
        try:
            import server as _server
            await _server.notify_frontend("lighbd_llm_stream", {"type": event_type, **(data or {})})
        except Exception as e:
            print(f"[INSTANCE_LORA] WARN: notify_frontend 실패: {e}")

    messages = []
    try:
        if not character:
            return {"success": False, "error": "character 필드가 필요합니다."}
        if not test_filename:
            return {"success": False, "error": "test_filename 필드가 필요합니다."}
        if not card_positive or not card_positive.strip():
            return {"success": False, "error": "캐릭터 카드 수정 프롬프트(card_positive)가 비어 있습니다."}
        if not test_positive or not test_positive.strip():
            return {"success": False, "error": "공통 테스트 이미지 프롬프트(test_positive)가 비어 있습니다."}

        custom_text, use_custom = _load_bot_test_setup_prompt_custom()
        if use_custom and custom_text.strip():
            template = custom_text
        else:
            template = _load_bot_test_setup_prompt_builtin()
        if not template.strip():
            return {"success": False, "error": "bot_test_setup 프롬프트 템플릿이 비어 있습니다."}

        rendered = _render_bot_test_setup_prompt(template, test_positive, card_positive)

        messages = [
            {"role": "system", "content": "You are a precise prompt refiner. Follow the user's instructions exactly and respond in strict JSON."},
            {"role": "user", "content": rendered},
        ]

        cfg = get_config()
        service = cfg.get("llm_service", "")
        use_model = cfg.get("llm_model", "")
        max_retries = max(0, int(cfg.get("auto_lora_prompt_max_retries", 2)))
        prompt_id = f"bot_test_setup:{bot_name}/{project_name}/{character}/{test_filename}"
        print(f"[INSTANCE_LORA] run_auto_refine_test_setup 호출: bot={bot_name} project={project_name} char={character} test={test_filename} service={service} card_len={len(card_positive)} test_len={len(test_positive)}")

        await _notify_llm_widget("start", {"model": use_model, "prompt_id": prompt_id})

        raw = None
        last_err = None
        total_elapsed = 0.0
        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                raw = await callLLM(messages)
            except Exception as call_err:
                print(f"[INSTANCE_LORA] callLLM 예외 (시도 {attempt + 1}/{max_retries + 1}): {call_err}")
                traceback.print_exc()
                last_err = f"{type(call_err).__name__}: {call_err}"
                raw = None
            total_elapsed += time.time() - t0

            if raw and not raw.startswith("[LLM 실패]"):
                parsed = _parse_auto_lora_prompt_response(raw)
                if parsed is not None:
                    done_data = {
                        "text": raw,
                        "completion_tokens": max(1, len(raw) // 3),
                        "elapsed": round(total_elapsed, 3),
                        "tps": round((max(1, len(raw) // 3) / total_elapsed), 1) if total_elapsed > 0 else 0.0,
                        "ttft": None,
                    }
                    await _notify_llm_widget("done", done_data)
                    _log_lighbd_history({
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                        "prompt_id": prompt_id,
                        "input": messages,
                        "output": raw,
                        "completion_tokens": done_data["completion_tokens"],
                        "elapsed": done_data["elapsed"],
                        "tps": done_data["tps"],
                        "status": "ok",
                    })
                    print(f"[INSTANCE_LORA] run_auto_refine_test_setup 완료: positive 길이={len(parsed['positive'])} (시도 {attempt + 1})")
                    return {"success": True, "data": parsed}
                last_err = f"LLM 응답을 JSON으로 파싱하지 못했습니다. raw: {raw[:300]}"
                print(f"[INSTANCE_LORA] LLM 응답 JSON 파싱 실패 (시도 {attempt + 1}/{max_retries + 1}). raw={raw[:500]}")
            else:
                last_err = f"LLM 호출 실패: {raw or '빈 응답'}"
                print(f"[INSTANCE_LORA] LLM 호출 실패 (시도 {attempt + 1}/{max_retries + 1}): {raw}")

            if attempt < max_retries:
                print(f"[INSTANCE_LORA] 재시도 대기 중... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1.0 * (attempt + 1))

        await _notify_llm_widget("error", {"error": last_err or "알 수 없는 오류", "elapsed": round(total_elapsed, 3)})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": prompt_id,
            "input": messages,
            "output": "",
            "elapsed": round(total_elapsed, 3),
            "status": "error",
            "error": last_err or "알 수 없는 오류",
        })
        return {"success": False, "error": f"{max_retries + 1}회 시도 후 실패: {last_err}"}
    except Exception as e:
        print(f"[INSTANCE_LORA] run_auto_refine_test_setup 예외: {e}")
        traceback.print_exc()
        await _notify_llm_widget("error", {"error": f"{type(e).__name__}: {e}"})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": f"bot_test_setup:{bot_name}/{project_name}/{character}/{test_filename}",
            "input": messages,
            "output": "",
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        })
        return {"success": False, "error": str(e)}


async def handle_get_auto_lora_prompt(request):
    """GET /api/instance_lora/auto_lora_prompt - 글로벌/커스텀 프롬프트 조회.

    ?asset=1 이면 에셋 전용(gender 없음) 프롬프트를 조회한다.
    """
    try:
        is_asset = request.query.get("asset") in ("1", "true", "True")
        builtin = _load_auto_lora_prompt_builtin(is_asset)
        custom, use_custom = _load_auto_lora_prompt_custom(is_asset)
        return web.json_response({
            "success": True,
            "data": {
                "builtin": builtin,
                "custom": custom,
                "use_custom": use_custom,
                "is_asset": is_asset,
            },
        })
    except Exception as e:
        print(f"[INSTANCE_LORA] auto_lora_prompt 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_set_auto_lora_prompt(request):
    """POST /api/instance_lora/auto_lora_prompt - 커스텀 프롬프트 저장.

    body.asset 가 true 이면 에셋 전용(gender 없음) 프롬프트로 저장한다.
    """
    is_asset = False
    try:
        body = await request.json()
        custom = body.get("custom", "") or ""
        use_custom = bool(body.get("use_custom", False))
        is_asset = bool(body.get("asset", False))
        _save_auto_lora_prompt_custom(custom, use_custom, is_asset)
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[INSTANCE_LORA] auto_lora_prompt 저장 실패 (asset={is_asset}): {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_get_bot_test_setup_prompt(request):
    """GET /api/instance_lora/bot_test_setup_prompt - 테스트 이미지 세팅 전용 LLM 프롬프트 조회."""
    try:
        builtin = _load_bot_test_setup_prompt_builtin()
        custom, use_custom = _load_bot_test_setup_prompt_custom()
        return web.json_response({
            "success": True,
            "data": {
                "builtin": builtin,
                "custom": custom,
                "use_custom": use_custom,
            },
        })
    except Exception as e:
        print(f"[INSTANCE_LORA] bot_test_setup_prompt 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_set_bot_test_setup_prompt(request):
    """POST /api/instance_lora/bot_test_setup_prompt - 테스트 이미지 세팅 전용 LLM 프롬프트 저장."""
    try:
        body = await request.json()
        custom = body.get("custom", "") or ""
        use_custom = bool(body.get("use_custom", False))
        _save_bot_test_setup_prompt_custom(custom, use_custom)
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[INSTANCE_LORA] bot_test_setup_prompt 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


def resolve_char_gender_tag(bot_name: str, char_name: str) -> str:
    """캐릭터의 gender_tag를 메인 bot.json에서 조회. 서버 정제 로직(auto_refine)과 동일 출처.
    없으면 빈 문자열을 반환한다 (조용한 기본값 치환 금지)."""
    if not bot_name or not char_name:
        return ""
    try:
        from modes.bot_mode import _load_bot_data
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
        char = next((c for c in (bot.get("characters", []) if bot else []) if c.get("name") == char_name), None)
        return (char.get("gender_tag") or "").strip() if char else ""
    except Exception as e:
        print(f"[INSTANCE_LORA] resolve_char_gender_tag 실패 (bot={bot_name} char={char_name}): {e}")
        traceback.print_exc()
        return ""


async def handle_resolve_gender_tag(request):
    """GET /api/instance_lora/resolve_gender?bot=&character= - 캐릭터 gender_tag 조회 (빈 값 허용)."""
    try:
        bot_name = request.query.get("bot", "").strip()
        char_name = request.query.get("character", "").strip()
        if not bot_name or not char_name:
            return web.json_response({"success": False, "error": "bot, character 쿼리가 필요합니다"}, status=400)
        gt = resolve_char_gender_tag(bot_name, char_name)
        return web.json_response({"success": True, "gender_tag": gt})
    except Exception as e:
        print(f"[INSTANCE_LORA] resolve_gender_tag 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_auto_refine_enqueue(request):
    """POST /api/instance_lora/auto_refine_enqueue - 단일 프롬프트 정제를 큐에 추가."""
    try:
        body = await request.json()
        source_type = (body.get("source_type") or "bot").strip().lower()
        bot_name = (body.get("bot") or "").strip()
        project_name = (body.get("project") or "").strip()
        char_name = (body.get("character") or "").strip()
        filename = (body.get("filename") or "").strip()
        entry = (body.get("entry") or "").strip()
        positive = body.get("positive", "") or ""
        gender = (body.get("gender") or "").strip()
        is_asset = bool(body.get("is_asset", False))

        if source_type not in ("bot", "bot_lora_training", "training", "bot_lora_test_setup"):
            return web.json_response({"success": False, "error": f"지원하지 않는 source_type: {source_type}"})

        # ── bot_lora_test_setup: 테스트 이미지 일괄 세팅 전용 분기 ──
        if source_type == "bot_lora_test_setup":
            card_filename = (body.get("card_filename") or "").strip()
            card_positive = body.get("card_positive", "") or ""
            test_filename = (body.get("test_filename") or "").strip()
            test_positive = body.get("test_positive", "") or ""
            if not bot_name or not project_name:
                return web.json_response({"success": False, "error": "bot_lora_test_setup 소스는 bot, project 필드가 필요합니다."})
            if not char_name:
                return web.json_response({"success": False, "error": "character 필드가 필요합니다."})
            if not card_positive.strip():
                return web.json_response({"success": False, "error": "card_positive(캐릭터 카드 수정 프롬프트) 필드가 비어 있습니다."})
            if not test_filename:
                return web.json_response({"success": False, "error": "test_filename(공통 테스트 이미지) 필드가 필요합니다."})
            if not test_positive.strip():
                return web.json_response({"success": False, "error": "test_positive(공통 테스트 이미지 프롬프트) 필드가 비어 있습니다."})

            try:
                import server as _server
                qm = _server.queue_manager
            except Exception as e:
                print(f"[INSTANCE_LORA] queue_manager 접근 실패: {e}")
                traceback.print_exc()
                return web.json_response({"success": False, "error": f"큐 매니저 접근 실패: {e}"})

            label = f"테스트 이미지 세팅: [bot_lora] {bot_name}/{project_name}/{char_name} test={test_filename}"
            item = await qm.add_item(
                item_type="instance_lora_prompt_refine",
                label=label,
                params={
                    "source_type": source_type,
                    "bot_name": bot_name,
                    "project_name": project_name,
                    "char_name": char_name,
                    "card_filename": card_filename,
                    "card_positive": card_positive,
                    "test_filename": test_filename,
                    "test_positive": test_positive,
                },
                priority=10,
            )
            print(f"[INSTANCE_LORA] auto_refine 큐 추가(bot_lora_test_setup): bot={bot_name} project={project_name} char={char_name} test={test_filename} id={item.id}")
            return web.json_response({"success": True, "data": {"id": item.id}})
        # ── 일반 정제 분기 (기존) ──

        if not char_name:
            return web.json_response({"success": False, "error": "character 필드가 필요합니다."})
        if not filename:
            return web.json_response({"success": False, "error": "filename 필드가 필요합니다."})
        if source_type == "bot" and not bot_name:
            return web.json_response({"success": False, "error": "bot 소스는 bot 필드가 필요합니다."})
        if source_type == "bot_lora_training" and (not bot_name or not project_name):
            return web.json_response({"success": False, "error": "bot_lora_training 소스는 bot, project 필드가 필요합니다."})
        if source_type == "training" and not entry:
            return web.json_response({"success": False, "error": "training 소스는 entry 필드가 필요합니다."})
        if not positive.strip():
            return web.json_response({"success": False, "error": "positive 필드가 비어 있습니다."})

        try:
            import server as _server
            qm = _server.queue_manager
        except Exception as e:
            print(f"[INSTANCE_LORA] queue_manager 접근 실패: {e}")
            traceback.print_exc()
            return web.json_response({"success": False, "error": f"큐 매니저 접근 실패: {e}"})

        if source_type == "bot":
            label = f"LoRA 프롬프트 정제: {bot_name}/{char_name}/{filename}"
        elif source_type == "bot_lora_training":
            label = f"LoRA 프롬프트 정제: [bot_lora] {bot_name}/{project_name}/{char_name}/{filename}"
        else:
            label = f"LoRA 프롬프트 정제: [training] {char_name}/{entry}/{filename}"

        item = await qm.add_item(
            item_type="instance_lora_prompt_refine",
            label=label,
            params={
                "source_type": source_type,
                "bot_name": bot_name,
                "project_name": project_name,
                "char_name": char_name,
                "filename": filename,
                "entry": entry,
                "positive": positive,
                "gender": gender,
                "is_asset": is_asset,
            },
            priority=10,
        )

        print(f"[INSTANCE_LORA] auto_refine 큐 추가: source={source_type} bot={bot_name} project={project_name} char={char_name} entry={entry} filename={filename} is_asset={is_asset} id={item.id}")
        return web.json_response({"success": True, "data": {"id": item.id}})
    except Exception as e:
        print(f"[INSTANCE_LORA] auto_refine_enqueue 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


def _safe_dirname(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-', '.')).strip() or "unnamed"


def _lora_dir(lora_id: str) -> str:
    return os.path.join(INSTANCE_LORA_DIR, _safe_dirname(lora_id))


# ─── JSON 로드/세이브 ─────────────────────────────────────────

def _load_data() -> dict:
    if not os.path.isfile(INSTANCE_LORA_MANAGE_FILE):
        return {"instance_loras": {}, "settings": {"anima": {}, "sdxl": {}, "instance": {}}}
    try:
        with open(INSTANCE_LORA_MANAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[INSTANCE_LORA] JSON 로드 실패: {e}")
        traceback.print_exc()
        return {"instance_loras": {}, "settings": {"anima": {}, "sdxl": {}, "instance": {}}}


def _save_data(data: dict):
    try:
        with open(INSTANCE_LORA_MANAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[INSTANCE_LORA] JSON 세이브 실패: {e}")
        traceback.print_exc()


# ─── CRUD ──────────────────────────────────────────────────────

def list_loras() -> list:
    data = _load_data()
    result = []
    for lora_id, lora_data in data.get("instance_loras", {}).items():
        images = lora_data.get("images", [])
        sessions = lora_data.get("sessions", {})
        entry = {
            "id": lora_id,
            "trigger": lora_data.get("trigger", ""),
            "image_count": len(images),
            "first_image": images[0] if images else None,
            "usage_count": lora_data.get("usage_count", 0),
            "has_anima": any(s.get("profile") == "anima" for s in sessions.values()),
            "has_sdxl": any(s.get("profile") == "sdxl" for s in sessions.values()),
            "created_at": lora_data.get("created_at", ""),
        }
        # 첫 이미지의 프롬프트 포함
        if images:
            prompt_result = get_image_prompt(lora_id, images[0])
            if prompt_result.get("success") and prompt_result.get("data"):
                entry["prompt"] = prompt_result["data"]
            else:
                print(f"[INSTANCE_LORA] 프롬프트 없음: {lora_id}/{images[0]} - {prompt_result.get('error', '알 수 없음')}")
        result.append(entry)
    return result


def create_lora(trigger: str) -> dict:
    data = _load_data()
    base = _safe_dirname(trigger)
    import hashlib, time
    short_hash = hashlib.md5(f"{trigger}{time.time()}".encode()).hexdigest()[:6]
    lora_id = f"{base}-{short_hash}"
    if lora_id in data.get("instance_loras", {}):
        print(f"[INSTANCE_LORA] 이미 존재: {lora_id}")
        return {"success": False, "error": "이미 존재하는 로라입니다"}

    import datetime
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data.setdefault("instance_loras", {})[lora_id] = {
        "trigger": trigger,
        "lora_name": lora_id,
        "images": [],
        "sessions": {},
        "usage_count": 0,
        "created_at": now,
    }
    _save_data(data)

    os.makedirs(_lora_dir(lora_id), exist_ok=True)
    print(f"[INSTANCE_LORA] 로라 생성: {lora_id} (trigger={trigger})")
    return {"success": True, "id": lora_id}


def import_uploaded_lora(
    trigger: str,
    profile: str,
    safetensors_path: str,
    safetensors_filename: str,
    instance_lora_load_path: str = "",
    preview_path: str = "",
    preview_filename: str = "",
) -> dict:
    """이미 학습된 safetensors를 직접 업로드하여 인스턴스 로라로 등록.
    - trigger-hash 기반 lora_id 생성 (create_lora와 동일 규칙)
    - instance_lora/{lora_id}/ : preview 이미지(옵션) 저장
    - {instance_lora_load_path}/{profile}/{lora_id}/uploaded-{ts}/ : safetensors + json 메타데이터 저장
      → 기존 피커(list_instance_lora_for_picker)가 자동으로 인식
    """
    trigger = (trigger or "").strip()
    if not trigger:
        print("[INSTANCE_LORA] import_uploaded: trigger 누락")
        return {"success": False, "error": "trigger 필수"}
    if profile not in ("anima", "sdxl"):
        print(f"[INSTANCE_LORA] import_uploaded: 잘못된 profile={profile}")
        return {"success": False, "error": "profile은 anima 또는 sdxl이어야 함"}
    if not safetensors_path or not os.path.isfile(safetensors_path):
        print(f"[INSTANCE_LORA] import_uploaded: safetensors 파일 없음: {safetensors_path}")
        return {"success": False, "error": "safetensors 파일이 없음"}
    if not instance_lora_load_path:
        print("[INSTANCE_LORA] import_uploaded: instance_lora_load_path 미설정")
        return {"success": False, "error": "instance_lora_load_path 미설정"}
    if not safetensors_filename.lower().endswith(".safetensors"):
        print(f"[INSTANCE_LORA] import_uploaded: 확장자 오류: {safetensors_filename}")
        return {"success": False, "error": ".safetensors 파일만 허용됨"}

    data = _load_data()
    base = _safe_dirname(trigger)
    hashlib = __import__("hashlib")
    time = __import__("time")
    datetime = __import__("datetime")
    short_hash = hashlib.md5(f"{trigger}{time.time()}".encode()).hexdigest()[:6]
    lora_id = f"{base}-{short_hash}"
    if lora_id in data.get("instance_loras", {}):
        print(f"[INSTANCE_LORA] import_uploaded: 이미 존재: {lora_id}")
        return {"success": False, "error": "이미 존재하는 로라입니다 (다시 시도하세요)"}

    now_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"uploaded-{now_ts}"
    step_name = "imported"

    # 1) instance_lora_manage.json 등록
    entry = {
        "trigger": trigger,
        "lora_name": lora_id,
        "images": [],
        "sessions": {
            session_name: {
                "profile": profile,
                "representative": None,
                "imported": True,
            }
        },
        "usage_count": 0,
        "created_at": now_ts,
        "imported": True,
    }

    # 2) 로컬 폴더 + preview 이미지(옵션)
    local_dir = _lora_dir(lora_id)
    os.makedirs(local_dir, exist_ok=True)
    if preview_path and os.path.isfile(preview_path) and preview_filename:
        # 확장자는 원본 유지, 카드 썸네일용
        safe_prev = _safe_dirname(preview_filename) or "preview.webp"
        dst_preview = os.path.join(local_dir, safe_prev)
        try:
            shutil.copy2(preview_path, dst_preview)
            entry["images"].append(safe_prev)
        except Exception as e:
            print(f"[INSTANCE_LORA] import_uploaded preview 복사 실패: {preview_path} -> {dst_preview} - {e}")
            traceback.print_exc()

    data.setdefault("instance_loras", {})[lora_id] = entry
    _save_data(data)

    # 3) 학습 결과 위치에 safetensors + json 메타데이터 저장
    session_dir = os.path.join(instance_lora_load_path, profile, lora_id, session_name)
    os.makedirs(session_dir, exist_ok=True)

    st_name = f"{step_name}.safetensors"
    dst_st = os.path.join(session_dir, st_name)
    try:
        shutil.copy2(safetensors_path, dst_st)
    except Exception as e:
        print(f"[INSTANCE_LORA] import_uploaded safetensors 복사 실패: {safetensors_path} -> {dst_st} - {e}")
        traceback.print_exc()
        # 롤백
        try:
            shutil.rmtree(local_dir)
        except Exception:
            pass
        data["instance_loras"].pop(lora_id, None)
        _save_data(data)
        return {"success": False, "error": f"safetensors 복사 실패: {e}"}

    # preview를 session 폴더에도 복사 (피커 preview_url용)
    previews_rel = []
    if preview_path and os.path.isfile(preview_path) and preview_filename:
        prev_in_session = os.path.join(session_dir, "preview" + os.path.splitext(preview_filename)[1])
        try:
            shutil.copy2(preview_path, prev_in_session)
            previews_rel.append(os.path.basename(prev_in_session))
        except Exception as e:
            print(f"[INSTANCE_LORA] import_uploaded session preview 복사 실패: {e}")

    # json 메타데이터 (list_instance_lora_for_picker 호환)
    meta = {
        "avr_loss": None,
        "config_file": "",
        "lora_file": st_name,
        "previews": previews_rel,
        "imported": True,
        "source_filename": safetensors_filename,
    }
    json_path = os.path.join(session_dir, f"{step_name}.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[INSTANCE_LORA] import_uploaded 메타데이터 저장 실패: {json_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": f"메타데이터 저장 실패: {e}"}

    print(f"[INSTANCE_LORA] import_uploaded 완료: {lora_id} profile={profile} session={session_name}")
    return {
        "success": True,
        "id": lora_id,
        "profile": profile,
        "session": session_name,
    }


def delete_lora(lora_id: str, instance_lora_load_path: str = "") -> dict:
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    if lora_id not in data.get("instance_loras", {}):
        print(f"[INSTANCE_LORA] 삭제 대상 없음: {lora_id}")
        return {"success": False, "error": "존재하지 않는 로라입니다"}

    del data["instance_loras"][lora_id]
    _save_data(data)

    # 학습 이미지 폴더 삭제
    lora_path = _lora_dir(lora_id)
    if os.path.isdir(lora_path):
        try:
            shutil.rmtree(lora_path)
        except Exception as e:
            print(f"[INSTANCE_LORA] 폴더 삭제 실패: {lora_path} - {e}")

    # 학습 결과물 삭제 (anima/sdxl 각각)
    if instance_lora_load_path:
        for profile in ("anima", "sdxl"):
            trained_dir = os.path.join(instance_lora_load_path, profile, lora_id)
            if os.path.isdir(trained_dir):
                try:
                    shutil.rmtree(trained_dir)
                    print(f"[INSTANCE_LORA] 학습 결과 삭제: {trained_dir}")
                except Exception as e:
                    print(f"[INSTANCE_LORA] 학습 결과 삭제 실패: {trained_dir} - {e}")

    print(f"[INSTANCE_LORA] 로라 삭제: {lora_id}")
    return {"success": True}


def reset_training(lora_id: str, instance_lora_load_path: str = "") -> dict:
    """학습 결과물(anima/sdxl)과 세션 기록만 삭제. 이미지/프롬프트는 유지."""
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    if lora_id not in data.get("instance_loras", {}):
        return {"success": False, "error": "존재하지 않는 로라입니다"}

    data["instance_loras"][lora_id]["sessions"] = {}
    _save_data(data)

    if instance_lora_load_path:
        for profile in ("anima", "sdxl"):
            trained_dir = os.path.join(instance_lora_load_path, profile, lora_id)
            if os.path.isdir(trained_dir):
                try:
                    shutil.rmtree(trained_dir)
                    print(f"[INSTANCE_LORA] 학습 결과 삭제: {trained_dir}")
                except Exception as e:
                    print(f"[INSTANCE_LORA] 학습 결과 삭제 실패: {trained_dir} - {e}")

    print(f"[INSTANCE_LORA] 학습 리셋: {lora_id}")
    return {"success": True}


def get_lora_detail(lora_id: str) -> dict:
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    lora = data.get("instance_loras", {}).get(lora_id)
    if not lora:
        print(f"[INSTANCE_LORA] 상세 조회 실패 - 없음: {lora_id}")
        return {"success": False, "error": "존재하지 않는 로라입니다"}

    return {
        "success": True,
        "data": {
            "id": lora_id,
            "trigger": lora.get("trigger", ""),
            "images": lora.get("images", []),
            "sessions": lora.get("sessions", {}),
            "usage_count": lora.get("usage_count", 0),
            "created_at": lora.get("created_at", ""),
        }
    }


def increment_usage(lora_id: str) -> dict:
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    if lora_id not in data.get("instance_loras", {}):
        return {"success": False, "error": "존재하지 않는 로라입니다"}

    data["instance_loras"][lora_id]["usage_count"] = data["instance_loras"][lora_id].get("usage_count", 0) + 1
    _save_data(data)
    return {"success": True, "usage_count": data["instance_loras"][lora_id]["usage_count"]}


# ─── 이미지 관리 ──────────────────────────────────────────────

def add_image(lora_id: str, src_path: str, filename: str) -> dict:
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    if lora_id not in data.get("instance_loras", {}):
        return {"success": False, "error": "존재하지 않는 로라입니다"}

    dst_dir = _lora_dir(lora_id)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, filename)
    try:
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        print(f"[INSTANCE_LORA] 이미지 복사 실패: {src_path} -> {dst_path} - {e}")
        return {"success": False, "error": str(e)}

    images = data["instance_loras"][lora_id].setdefault("images", [])
    if filename not in images:
        images.append(filename)
    _save_data(data)

    print(f"[INSTANCE_LORA] 이미지 추가: {lora_id}/{filename}")
    return {"success": True, "filename": filename}


def delete_image(lora_id: str, filename: str) -> dict:
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    if lora_id not in data.get("instance_loras", {}):
        return {"success": False, "error": "존재하지 않는 로라입니다"}

    images = data["instance_loras"][lora_id].get("images", [])
    if filename not in images:
        return {"success": False, "error": "이미지가 목록에 없습니다"}

    images.remove(filename)
    _save_data(data)

    img_path = os.path.join(_lora_dir(lora_id), filename)
    if os.path.isfile(img_path):
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"[INSTANCE_LORA] 이미지 파일 삭제 실패: {img_path} - {e}")

    prompt_path = os.path.join(_lora_dir(lora_id), os.path.splitext(filename)[0] + "_prompt.json")
    if os.path.isfile(prompt_path):
        try:
            os.remove(prompt_path)
        except Exception:
            pass

    print(f"[INSTANCE_LORA] 이미지 삭제: {lora_id}/{filename}")
    return {"success": True}


def get_image_path(lora_id: str, filename: str) -> str:
    return os.path.join(_lora_dir(_safe_dirname(lora_id)), filename)


def list_images(lora_id: str) -> list:
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    lora = data.get("instance_loras", {}).get(lora_id, {})
    return lora.get("images", [])


def save_image_prompt(lora_id: str, filename: str, prompt_data: dict) -> dict:
    lora_id = _safe_dirname(lora_id)
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(_lora_dir(lora_id), f"{base}_prompt.json")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)
        return {"success": True}
    except Exception as e:
        print(f"[INSTANCE_LORA] 프롬프트 저장 실패: {prompt_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_image_prompt(lora_id: str, filename: str) -> dict:
    lora_id = _safe_dirname(lora_id)
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(_lora_dir(lora_id), f"{base}_prompt.json")
    if not os.path.isfile(prompt_path):
        return {"success": False, "error": "프롬프트 없음"}
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return {"success": True, "data": json.load(f)}
    except Exception as e:
        print(f"[INSTANCE_LORA] 프롬프트 로드 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


# ─── 설정 관리 ─────────────────────────────────────────────────

def get_settings() -> dict:
    data = _load_data()
    return {"success": True, "data": data.get("settings", {"anima": {}, "sdxl": {}, "instance": {}})}


def save_settings(settings: dict) -> dict:
    data = _load_data()
    data["settings"] = settings
    _save_data(data)
    print("[INSTANCE_LORA] 설정 저장 완료")
    return {"success": True}


# ─── 세션 관리 ─────────────────────────────────────────────────

def add_session(lora_id: str, session_id: str, profile: str) -> dict:
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    if lora_id not in data.get("instance_loras", {}):
        return {"success": False, "error": "존재하지 않는 로라입니다"}

    data["instance_loras"][lora_id].setdefault("sessions", {})[session_id] = {
        "profile": profile,
        "representative": None,
    }
    _save_data(data)
    print(f"[INSTANCE_LORA] 세션 추가: {lora_id}/{session_id} (profile={profile})")
    return {"success": True}


def update_session_representative(lora_id: str, session_id: str, rep_data: dict) -> dict:
    data = _load_data()
    lora_id = _safe_dirname(lora_id)
    sessions = data.get("instance_loras", {}).get(lora_id, {}).get("sessions", {})
    if session_id not in sessions:
        return {"success": False, "error": "세션 없음"}

    sessions[session_id]["representative"] = rep_data
    _save_data(data)
    return {"success": True}


def list_instance_lora_for_picker(instance_lora_load_path: str = "") -> list:
    """LoRA 피커용 목록 반환. lora_id 목록 + 각 프로필별 safetensors 경로 포함.
    메타데이터 representative 대신 파일시스템에서 직접 최신 세션 스캔."""
    data = _load_data()
    result = []
    for lora_id, lora_data in data.get("instance_loras", {}).items():
        profiles = {}
        safe_id = _safe_dirname(lora_id)
        # 각 프로필별로 파일시스템에서 최신 세션 찾기
        for profile in ("anima", "sdxl"):
            if not instance_lora_load_path:
                continue
            profile_dir = os.path.join(instance_lora_load_path, profile, safe_id)
            if not os.path.isdir(profile_dir):
                continue
            # 가장 최신 세션 폴더 선택
            session_dirs = sorted(
                [d for d in os.listdir(profile_dir) if os.path.isdir(os.path.join(profile_dir, d))],
                reverse=True
            )
            for session_name in session_dirs:
                session_dir = os.path.join(profile_dir, session_name)
                # JSON 파일에서 safetensors 정보 읽기
                json_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
                if not json_files:
                    continue
                # 첫 번째 json 파일 사용
                json_path = os.path.join(session_dir, json_files[0])
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        jdata = json.load(f)
                    safetensors = jdata.get('lora_file', '')
                    previews = jdata.get('previews', [])
                    if safetensors and os.path.isfile(os.path.join(session_dir, safetensors)):
                        rel_path = os.path.join(profile, safe_id, session_name, safetensors)
                        preview = previews[0] if previews else ""
                        profiles[profile] = {
                            "lora_path": rel_path,
                            "preview_url": preview,
                            "session": session_name,
                        }
                        break  # 해당 프로필의 최신 세션 찾음
                except Exception as e:
                    print(f"[INSTANCE_LORA_PICKER] JSON 읽기 실패: {json_path} - {e}")
                    continue
        # 학습된 파일이 하나라도 있으면 포함
        if profiles:
            images = lora_data.get("images", [])
            result.append({
                "lora_id": lora_id,
                "trigger": lora_data.get("trigger", ""),
                "first_image": images[0] if images else "",
                "profiles": profiles,
            })
    return result
