"""
BotMode - 삽화 설정 모드

봇(bot) 단위로 캐릭터 이미지를 관리.
폴더 구조: bot/{봇이름}/{캐릭터이름}/{이미지들}
"""

import asyncio
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
import shutil
import traceback
from typing import Optional
from urllib.parse import quote
from aiohttp import web


# ─── 상수 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
BOT_DIR = os.path.join(BASE_DIR, "bot")
BOT_DATA_FILE = os.path.join(ASSET_DATA_DIR, "bot.json")
ASSET_DIR = os.path.join(BASE_DIR, "asset")

# ─── 시스템 프롬프트 프리셋 — 배포자료 builtin (git 추적) ───
# builtin(prompts/bot_system_prompt/presets.json): 배포자료·읽기전용. git 커밋으로 배포.
# local(asset_data/bot.json 의 system_prompt_presets): 이 PC 전용·편집가능. gitignore.
# 잠금은 이름이 아니라 저장 위치로 판정한다.
BUILTIN_PRESETS_DIR = os.path.join(BASE_DIR, "prompts", "bot_system_prompt")
BUILTIN_PRESETS_FILE = os.path.join(BUILTIN_PRESETS_DIR, "presets.json")
_builtin_presets_cache = None
_builtin_presets_mtime = -1.0

DEFAULT_BOT_DATA = {
    "bots": [],
    "positive_whitelist": [],
    "positive_blacklist": [],
    "system_prompt_presets": {},
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

TAG_FILTER_PROFILES_FILE = os.path.join(ASSET_DATA_DIR, "tag_filter_profiles.json")


def _migrate_solo_group(data: dict):
    """기존 loras/illust_settings를 loras_solo/illust_settings_solo로 마이그레이션."""
    changed = False
    for bot in data.get("bots", []):
        # illust_settings → illust_settings_solo
        if "illust_settings" in bot and "illust_settings_solo" not in bot:
            bot["illust_settings_solo"] = bot["illust_settings"]
            changed = True
            print(f"[BOT_MODE] 마이그레이션: illust_settings → illust_settings_solo ({bot['name']})")
        if "illust_settings_group" not in bot:
            bot["illust_settings_group"] = dict(DEFAULT_ILLUST_SETTINGS)
            changed = True
        for char in bot.get("characters", []):
            # loras → loras_solo
            if "loras" in char and "loras_solo" not in char:
                char["loras_solo"] = char["loras"]
                changed = True
                print(f"[BOT_MODE] 마이그레이션: loras → loras_solo ({bot['name']}/{char['name']})")
            if "loras_group" not in char:
                char["loras_group"] = []
                changed = True
            # gender_tag 기본값 보정 — 드롭박스 표시 기본값(1girl)과 일치. 비어 있으면 1girl 로 채운다.
            gt = (char.get("gender_tag") or "").strip()
            if gt not in ("1girl", "1boy", "1male"):
                char["gender_tag"] = "1girl"
                changed = True
                print(f"[BOT_MODE] 마이그레이션: gender_tag 기본값(1girl) 적용 ({bot['name']}/{char['name']})")
    if changed:
        _save_bot_data(data)


# 전역 시스템 프롬프트 프리셋 — 배포자료 builtin (git 추적) / local(bot.json, 편집가용) 2-레이어.
# 잠금은 이름이 아니라 저장 위치로 판정: builtin = 읽기전용, local = 편집가능.


def _load_builtin_presets() -> dict:
    """배포자료 builtin 프리셋 로드. mtime 기반 캐싱. 파일 없으면 빈 dict."""
    global _builtin_presets_cache, _builtin_presets_mtime
    if not os.path.isfile(BUILTIN_PRESETS_FILE):
        if _builtin_presets_cache is not None:
            return _builtin_presets_cache
        return {}
    try:
        mtime = os.path.getmtime(BUILTIN_PRESETS_FILE)
        if _builtin_presets_cache is not None and mtime == _builtin_presets_mtime:
            return _builtin_presets_cache
        with open(BUILTIN_PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
        _builtin_presets_cache = data
        _builtin_presets_mtime = mtime
        return data
    except Exception as e:
        print(f"[BOT_MODE] builtin 프리셋 로드 실패: {e}")
        traceback.print_exc()
        return _builtin_presets_cache if _builtin_presets_cache is not None else {}


def _ensure_bot_preset_scope(bot: dict, builtin_names, local_names):
    """봇의 preset_scope 를 보정한다. 이름이 builtin 에 있으면 'builtin', 아니면 'local'.

    과거 버전(global/private) 필드도 여기서 새 스키마로 정리한다.
    """
    preset = (bot.get("system_prompt_preset") or "").strip()
    scope = (bot.get("preset_scope") or "").strip()
    if scope == "builtin" and preset in builtin_names:
        return
    if scope == "local" and preset in local_names:
        return
    # 보정: builtin 우선, 그 다음 local
    if preset and preset in builtin_names:
        bot["preset_scope"] = "builtin"
    elif preset and preset in local_names:
        bot["preset_scope"] = "local"
    else:
        bot["preset_scope"] = "local"  # 폴백(기본은 local)


def _migrate_system_prompt_preset(data: dict):
    """각 봇에 system_prompt_preset + preset_scope(builtin|local) 를 보장한다.

    - builtin(prompts/bot_system_prompt/presets.json): 배포자료·읽기전용(git 배포).
    - local(bot.json system_prompt_presets): 편집가능. '기본' 보장.
    - local 에 builtin 과 이름이 같은 프리셋이 있으면 builtin 이 권위를 갖도록 local 에서 제거(dedup).
    - 깨진 참조는 system_prompt 본문으로 local 프리셋을 생성해 복구.
    """
    changed = False
    presets = data.get("system_prompt_presets")
    if not isinstance(presets, dict):
        presets = {}
        data["system_prompt_presets"] = presets
        changed = True
    if "기본" not in presets:
        presets["기본"] = ""
        changed = True
        print("[BOT_MODE] 마이그레이션: local 기본 프리셋 생성 (system_prompt_presets['기본'])")

    builtin = _load_builtin_presets() or {}
    builtin_names = set(builtin.keys())

    # dedup: local 에 builtin 이름이 있으면 local 쪽 제거 (builtin 권위)
    for name in list(presets.keys()):
        if name in builtin_names:
            del presets[name]
            changed = True
            print(f"[BOT_MODE] 마이그레이션: local 프리셋 '{name}' → builtin 으로 이관(중복 제거)")

    local_names = set(presets.keys())
    for bot in data.get("bots", []):
        bot_name = bot.get("name", "?")
        # 과거 스키마 잔재(custom_system_presets) 제거
        if "custom_system_presets" in bot:
            del bot["custom_system_presets"]
            changed = True
        preset = (bot.get("system_prompt_preset") or "").strip()
        prev_scope = bot.get("preset_scope")
        _ensure_bot_preset_scope(bot, builtin_names, local_names)
        # 유효한 참조면 다음 봇으로
        if preset and ((bot["preset_scope"] == "builtin" and preset in builtin_names) or
                       (bot["preset_scope"] == "local" and preset in local_names)):
            if prev_scope != bot["preset_scope"]:
                changed = True
                print(f"[BOT_MODE] 마이그레이션: preset_scope='{bot['preset_scope']}' 보정 ({bot_name})")
            continue
        # auto-adopt: 참조가 어디에도 없지만 '배포_'+이름 이 builtin 에 있으면
        # 배포자료 prefix 도입 전의 구이름 참조를 새 builtin 으로 자동 이관.
        # (단, 같은 이름의 local 프리셋이 있으면 사용자 편집을 존중해 건드리지 않는다)
        if preset and preset not in local_names:
            adopted = "배포_" + preset
            if adopted in builtin_names:
                bot["system_prompt_preset"] = adopted
                bot["preset_scope"] = "builtin"
                changed = True
                print(f"[BOT_MODE] 마이그레이션: 구참조 '{preset}' → builtin '{adopted}' 자동 이관 ({bot_name})")
                continue
        # 깨진 참조 복구
        sp_text = bot.get("system_prompt", "") or ""
        if sp_text.strip():
            base = f"{bot_name} 기본"
            new_name = base
            i = 2
            while new_name in local_names:
                new_name = f"{base} ({i})"
                i += 1
            presets[new_name] = sp_text
            local_names.add(new_name)
            bot["system_prompt_preset"] = new_name
            bot["preset_scope"] = "local"
            changed = True
            print(f"[BOT_MODE] 마이그레이션: 시스템 프롬프트 본문 → local 프리셋 '{new_name}' 생성/연결 ({bot_name})")
        else:
            bot["system_prompt_preset"] = "기본"
            bot["preset_scope"] = "local"
            changed = True
            print(f"[BOT_MODE] 마이그레이션: system_prompt_preset='기본' 할당 ({bot_name})")
    if changed:
        _save_bot_data(data)


def _load_bot_data() -> dict:
    """bot.json 로드. 없으면 기본값 생성."""
    if os.path.isfile(BOT_DATA_FILE):
        try:
            with open(BOT_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[BOT_MODE] bot.json 로드 완료")
                if "bots" not in data:
                    data["bots"] = []
                # 새 필드 자동 마이그레이션
                if "positive_whitelist" not in data:
                    data["positive_whitelist"] = []
                if "positive_blacklist" not in data:
                    data["positive_blacklist"] = []
                if "system_prompt_presets" not in data:
                    data["system_prompt_presets"] = {}
                for bot in data.get("bots", []):
                    if "system_prompt" not in bot:
                        bot["system_prompt"] = ""
                # system_prompt_preset 필드 보장 (프리셋 기반 모델)
                _migrate_system_prompt_preset(data)
                # solo/group 프로필 마이그레이션
                _migrate_solo_group(data)
                # 후처리 봇별 설정 마이그레이션 (config.json → bot.json)
                _migrate_postprocess_vn(data)
                return data
        except Exception as e:
            print(f"[BOT_MODE] bot.json 로드 실패: {e}")
            traceback.print_exc()
    data = copy_default()
    _save_bot_data(data)
    return data


def _save_bot_data(data: dict):
    """bot.json 저장."""
    try:
        os.makedirs(ASSET_DATA_DIR, exist_ok=True)
        with open(BOT_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[BOT_MODE] bot.json 저장 실패: {e}")
        traceback.print_exc()


def copy_default() -> dict:
    import copy
    return copy.deepcopy(DEFAULT_BOT_DATA)


class BotMode:
    """삽화 설정 모드 매니저"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._asset_tool = None

    def set_asset_tool(self, tool):
        self._asset_tool = tool

    # ─── 봇 데이터 조회 ──────────────────────────────────
    async def handle_get_bots(self, request):
        """GET /api/bot_mode/bots - 전체 봇 데이터 반환"""
        try:
            data = _load_bot_data()
            # 각 캐릭터의 이미지 수도 함께 반환
            for bot in data["bots"]:
                for char in bot.get("characters", []):
                    char_dir = os.path.join(BOT_DIR, bot["name"], char["name"])
                    if os.path.isdir(char_dir):
                        images = [f for f in os.listdir(char_dir)
                                  if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
                        char["image_count"] = len(images)
                    else:
                        char["image_count"] = 0
            return _json_ok(data)
        except Exception as e:
            print(f"[BOT_MODE] 봇 목록 조회 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    # ─── 봇/캐릭터 액션 ──────────────────────────────────
    async def handle_bot_action(self, request):
        """POST /api/bot_mode/action - 봇/캐릭터 CRUD"""
        try:
            body = await request.json()
            action = body.get("action", "")
            async with self._lock:
                data = _load_bot_data()

                if action == "add_bot":
                    return await self._add_bot(data, body)
                elif action == "remove_bot":
                    return await self._remove_bot(data, body)
                elif action == "add_character":
                    return await self._add_character(data, body)
                elif action == "remove_character":
                    return await self._remove_character(data, body)
                elif action == "rename_character":
                    return await self._rename_character(data, body)
                elif action == "toggle_rep_image":
                    return await self._toggle_rep_image(data, body)
                elif action == "reorder_rep_images":
                    return await self._reorder_rep_images(data, body)
                elif action == "update_eye_prompt":
                    return await self._update_eye_prompt(data, body)
                elif action == "update_char_loras":
                    return await self._update_char_loras(data, body)
                elif action == "update_char_negative":
                    return await self._update_char_negative(data, body)
                elif action == "update_char_face_tags":
                    return await self._update_char_face_tags(data, body)
                elif action == "update_char_face_loras":
                    return await self._update_char_face_loras(data, body)
                elif action == "update_char_style_loras":
                    return await self._update_char_style_loras(data, body)
                elif action == "update_char_gender_tag":
                    return await self._update_char_gender_tag(data, body)
                else:
                    return _json_error(f"알 수 없는 액션: {action}")
        except Exception as e:
            print(f"[BOT_MODE] 액션 처리 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def _add_bot(self, data, body):
        name = body.get("name", "").strip()
        if not name:
            return _json_error("봇 이름이 비어있습니다.")
        if any(b["name"] == name for b in data["bots"]):
            return _json_error(f"이미 존재하는 봇: {name}")
        data["bots"].append({"name": name, "characters": []})
        os.makedirs(os.path.join(BOT_DIR, name), exist_ok=True)
        _save_bot_data(data)
        print(f"[BOT_MODE] 봇 추가: {name}")
        return _json_ok({"bots": data["bots"]})

    async def _remove_bot(self, data, body):
        name = body.get("name", "").strip()
        if not name:
            return _json_error("봇 이름이 비어있습니다.")
        data["bots"] = [b for b in data["bots"] if b["name"] != name]
        bot_path = os.path.join(BOT_DIR, name)
        if os.path.isdir(bot_path):
            shutil.rmtree(bot_path)
            print(f"[BOT_MODE] 봇 폴더 삭제: {bot_path}")

        # soya_bot 폴더 정리
        try:
            config_path = os.path.join(BASE_DIR, "config.json")
            if os.path.isfile(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                comfy_input_dir = config.get("comfy_input_dir", "").strip()
                if comfy_input_dir:
                    soya_bot_path = os.path.join(comfy_input_dir, "soya_bot", name)
                    if os.path.isdir(soya_bot_path):
                        shutil.rmtree(soya_bot_path)
                        print(f"[BOT_MODE] soya_bot 폴더 삭제: {soya_bot_path}")
        except Exception as e:
            print(f"[BOT_MODE] soya_bot 정리 실패: {e}")
            traceback.print_exc()

        _save_bot_data(data)
        print(f"[BOT_MODE] 봇 삭제: {name}")
        return _json_ok({"bots": data["bots"]})


    async def _add_character(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        if any(c["name"] == char_name for c in bot.get("characters", [])):
            return _json_error(f"이미 존재하는 캐릭터: {char_name}")
        if "characters" not in bot:
            bot["characters"] = []
        bot["characters"].append({"name": char_name})
        os.makedirs(os.path.join(BOT_DIR, bot_name, char_name), exist_ok=True)
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 추가: {bot_name}/{char_name}")
        return _json_ok({"bots": data["bots"]})

    async def _remove_character(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        bot["characters"] = [c for c in bot.get("characters", []) if c["name"] != char_name]
        char_path = os.path.join(BOT_DIR, bot_name, char_name)
        if os.path.isdir(char_path):
            shutil.rmtree(char_path)
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 삭제: {bot_name}/{char_name}")
        return _json_ok({"bots": data["bots"]})

    async def _rename_character(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        old_name = body.get("old_name", "").strip()
        new_name = body.get("new_name", "").strip()
        if not bot_name or not old_name or not new_name:
            return _json_error("필수 값이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        if any(c["name"] == new_name for c in bot.get("characters", [])):
            return _json_error(f"이미 존재하는 캐릭터 이름: {new_name}")
        for c in bot.get("characters", []):
            if c["name"] == old_name:
                c["name"] = new_name
                break
        old_path = os.path.join(BOT_DIR, bot_name, old_name)
        new_path = os.path.join(BOT_DIR, bot_name, new_name)
        if os.path.isdir(old_path):
            os.rename(old_path, new_path)
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 이름 변경: {bot_name}/{old_name} → {new_name}")
        return _json_ok({"bots": data["bots"]})

    async def _update_eye_prompt(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        eye_prompt = body.get("eye_prompt", "")
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")
        char["eye_prompt"] = eye_prompt
        _save_bot_data(data)
        print(f"[BOT_MODE] 눈 프롬프트 업데이트: {bot_name}/{char_name}")
        return _json_ok({"bots": data["bots"]})

    async def _update_char_negative(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        character_negative = body.get("character_negative", "")
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")
        char["character_negative"] = character_negative
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 부정 프롬프트 업데이트: {bot_name}/{char_name}")
        return _json_ok({"bots": data["bots"]})

    async def _update_char_face_tags(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        face_tags = body.get("face_tags", "")
        eye_tags = body.get("eye_tags", "")
        absolute_tags = body.get("absolute_tags", "")
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")
        char["face_tags"] = face_tags
        char["eye_tags"] = eye_tags
        char["absolute_tags"] = absolute_tags
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 얼굴 태그 업데이트: {bot_name}/{char_name}")
        return _json_ok({"bots": data["bots"]})

    async def _update_char_loras(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        loras = body.get("loras", [])
        profile = body.get("profile", "solo")
        if profile not in ("solo", "group"):
            profile = "solo"
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")
        key = f"loras_{profile}"
        char[key] = loras
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 LoRA 업데이트: {bot_name}/{char_name} [{profile}] ({len(loras)}개)")
        return _json_ok({"bots": data["bots"]})

    async def _update_char_face_loras(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        face_loras = body.get("face_loras", [])
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")
        char["face_loras"] = face_loras
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 얼굴 LoRA 업데이트: {bot_name}/{char_name} ({len(face_loras)}개)")
        return _json_ok({"bots": data["bots"]})

    async def _update_char_style_loras(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        style_loras = body.get("style_loras", [])
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")
        char["style_loras"] = style_loras
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 스타일(그림체) LoRA 업데이트: {bot_name}/{char_name} ({len(style_loras)}개)")
        return _json_ok({"bots": data["bots"]})

    async def _update_char_gender_tag(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        gender_tag = body.get("gender_tag", "1girl")
        if gender_tag not in ("1girl", "1boy", "1male"):
            gender_tag = "1girl"
        if not bot_name or not char_name:
            return _json_error("봇 또는 캐릭터 이름이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")
        char["gender_tag"] = gender_tag
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 성별 태그 업데이트: {bot_name}/{char_name} → {gender_tag}")
        return _json_ok({"bots": data["bots"]})

    async def _toggle_rep_image(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        filename = body.get("filename", "").strip()
        if not bot_name or not char_name or not filename:
            return _json_error("필수 값이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")

        rep_images = char.get("rep_images", [])
        if filename in rep_images:
            rep_images = [f for f in rep_images if f != filename]
            print(f"[BOT_MODE] 대표 이미지 해제: {bot_name}/{char_name}/{filename}")
        else:
            if len(rep_images) >= 3:
                return _json_error("대표 이미지는 최대 3개까지 지정할 수 있습니다.")
            rep_images.append(filename)
            print(f"[BOT_MODE] 대표 이미지 지정: {bot_name}/{char_name}/{filename}")

        if rep_images:
            char["rep_images"] = rep_images
        else:
            char.pop("rep_images", None)
        _save_bot_data(data)
        return _json_ok({"bots": data["bots"]})

    async def _reorder_rep_images(self, data, body):
        bot_name = body.get("bot_name", "").strip()
        char_name = body.get("char_name", "").strip()
        filename = body.get("filename", "").strip()
        direction = body.get("direction", "").strip()  # "up" or "down"
        if not bot_name or not char_name or not filename or direction not in ("up", "down"):
            return _json_error("필수 값이 비어있거나 잘못되었습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return _json_error(f"캐릭터를 찾을 수 없음: {char_name}")
        rep_images = char.get("rep_images", [])
        if filename not in rep_images:
            return _json_error(f"대표 이미지가 아님: {filename}")
        idx = rep_images.index(filename)
        new_idx = idx - 1 if direction == "up" else idx + 1
        if new_idx < 0 or new_idx >= len(rep_images):
            return _json_error("이동할 수 없는 위치입니다.")
        rep_images[idx], rep_images[new_idx] = rep_images[new_idx], rep_images[idx]
        char["rep_images"] = rep_images
        _save_bot_data(data)
        print(f"[BOT_MODE] 대표 이미지 순서 변경: {bot_name}/{char_name}/{filename} {direction}")
        return _json_ok({"bots": data["bots"]})

    # ─── 이미지 목록 ─────────────────────────────────────
    async def handle_get_images(self, request):
        """GET /api/bot_mode/images?bot=xxx&character=yyy"""
        bot_name = request.query.get("bot", "").strip()
        char_name = request.query.get("character", "").strip()
        if not bot_name or not char_name:
            return _json_error("봇과 캐릭터 이름이 필요합니다.")

        char_dir = os.path.join(BOT_DIR, bot_name, char_name)
        if not os.path.isdir(char_dir):
            print(f"[BOT_MODE] 캐릭터 폴더 없음: {char_dir}")
            return _json_ok({"images": []})

        images = []
        for fname in sorted(os.listdir(char_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            base = os.path.splitext(fname)[0]
            prompt = ""
            negative = ""
            prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
            if os.path.isfile(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        pd = json.load(f)
                        prompt = pd.get("prompt", "")
                        negative = pd.get("negative", "")
                except Exception:
                    pass
            images.append({
                "filename": fname,
                "prompt": prompt,
                "negative": negative,
                "url": f"/api/bot_mode/image/{bot_name}/{char_name}/{fname}",
            })

        return _json_ok({"images": images})

    # ─── 이미지 파일 서빙 ────────────────────────────────
    def iter_character_image_filenames(self, bot_name: str, char_name: str):
        """봇 캐릭터 디렉토리(BOT_DIR/<bot>/<character>)의 이미지 파일명을 yield.

        봇으로 가져온 이미지는 '<캐릭터>-<의상>-<표정>-<해시>.<ext>' 평면 구조로 저장되므로
        표정(감정)이 파일명에 인코딩되어 있다. 후처리 감정 뽑아내기 원본으로 사용.
        """
        if not bot_name or not char_name:
            return
        char_dir = os.path.join(BOT_DIR, bot_name, char_name)
        if not os.path.isdir(char_dir):
            return
        for fname in sorted(os.listdir(char_dir)):
            if fname.startswith("_"):
                continue  # _face_image 등 특수 파일 제외
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            yield fname

    def character_image_counts(self, bot_name: str) -> dict:
        """봇의 각 캐릭터별 보유 이미지 장 수 반환. {char_name: int}.

        감정 뽑기 선택 모달에서 이미지가 없는 캐릭터를 미리 식별하기 위해 사용.
        iter_character_image_filenames 과 동일 조건(_ 프리픽스 제외)으로 집계.
        """
        counts = {}
        if not bot_name:
            return counts
        bot_dir = os.path.join(BOT_DIR, bot_name)
        if not os.path.isdir(bot_dir):
            return counts
        for cname in sorted(os.listdir(bot_dir)):
            cpath = os.path.join(bot_dir, cname)
            if not os.path.isdir(cpath):
                continue
            n = 0
            for fname in os.listdir(cpath):
                if fname.startswith("_"):
                    continue
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    n += 1
            counts[cname] = n
        return counts

    async def handle_get_image(self, request):
        """GET /api/bot_mode/image/{bot}/{character}/{filename}"""
        bot_name = request.match_info.get("bot", "")
        char_name = request.match_info.get("character", "")
        filename = request.match_info.get("filename", "")
        if not bot_name or not char_name or not filename:
            return _json_error("경로가 올바르지 않습니다.")

        filepath = os.path.join(BOT_DIR, bot_name, char_name, filename)
        filepath = os.path.normpath(filepath)
        # 경로 조작 방지
        if not filepath.startswith(os.path.normpath(BOT_DIR)):
            print(f"[BOT_MODE] 잘못된 경로 접근: {filepath}")
            return _json_error("잘못된 경로입니다.")

        if not os.path.isfile(filepath):
            print(f"[BOT_MODE] 이미지 파일 없음: {filepath}")
            return _json_error("파일을 찾을 수 없습니다.", status=404)

        import mimetypes as mt
        content_type = mt.guess_type(filepath)[0] or "image/webp"
        return web.FileResponse(filepath, headers={"Content-Type": content_type})

    # ─── 이미지 업로드 ─────────────────────────────────────
    async def handle_upload_image(self, request):
        """POST /api/bot_mode/upload - 이미지 업로드"""
        try:
            data_multipart = await request.post()
            bot_name = data_multipart.get("bot", "").strip()
            char_name = data_multipart.get("character", "").strip()
            prompt = data_multipart.get("prompt", "")
            file_field = data_multipart.get("file")

            if not bot_name or not char_name:
                return _json_error("봇과 캐릭터 이름이 필요합니다.")
            if not file_field or not hasattr(file_field, "filename"):
                return _json_error("파일이 없습니다.")

            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            os.makedirs(char_dir, exist_ok=True)

            ext = os.path.splitext(file_field.filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                ext = ".webp"
            # 원래 파일명 유지, 충돌 시 해시 추가
            base_name = os.path.splitext(file_field.filename)[0]
            filename = f"{base_name}{ext}"
            filepath = os.path.join(char_dir, filename)
            if os.path.exists(filepath):
                filename = f"{base_name}_{uuid.uuid4().hex[:6]}{ext}"
                filepath = os.path.join(char_dir, filename)

            with open(filepath, "wb") as f:
                f.write(file_field.file.read())

            # 프롬프트 저장
            if prompt:
                prompt_path = os.path.join(char_dir, f"{os.path.splitext(filename)[0]}_prompt.json")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    json.dump({"prompt": prompt, "source": "upload"}, f, ensure_ascii=False)

            print(f"[BOT_MODE] 이미지 업로드: {bot_name}/{char_name}/{filename}")
            return _json_ok({"filename": filename})
        except Exception as e:
            print(f"[BOT_MODE] 이미지 업로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    # ─── 에셋에서 이미지 가져오기 ─────────────────────────
    async def handle_import_asset(self, request):
        """POST /api/bot_mode/import_asset - 에셋 이미지를 봇으로 복사"""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            asset_paths = body.get("asset_paths", [])  # ["캐릭터/의상/표정/파일명", ...]

            if not bot_name or not char_name:
                return _json_error("봇과 캐릭터 이름이 필요합니다.")
            if not asset_paths:
                return _json_error("가져올 에셋 경로가 없습니다.")

            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            os.makedirs(char_dir, exist_ok=True)

            imported = []
            for rel_path in asset_paths:
                src = os.path.join(ASSET_DIR, rel_path)
                src = os.path.normpath(src)
                if not os.path.normpath(src).startswith(os.path.normpath(ASSET_DIR)):
                    print(f"[BOT_MODE] 잘못된 에셋 경로: {src}")
                    continue
                if not os.path.isfile(src):
                    print(f"[BOT_MODE] 에셋 파일 없음: {src}")
                    continue

                ext = os.path.splitext(src)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    ext = ".webp"
                # rel_path: "캐릭터/의상/표정/파일명" → "이름-복장-표정-해시"
                parts = rel_path.replace("\\", "/").split("/")
                asset_char = parts[0] if len(parts) > 0 else ""
                asset_outfit = parts[1] if len(parts) > 1 else ""
                asset_expr = parts[2] if len(parts) > 2 else ""
                file_hash = uuid.uuid4().hex[:8]
                name_parts = [p for p in [asset_char, asset_outfit, asset_expr] if p]
                new_name = "-".join(name_parts) + f"-{file_hash}{ext}"
                dst = os.path.join(char_dir, new_name)
                # 충돌 시 해시 변경
                if os.path.exists(dst):
                    file_hash = uuid.uuid4().hex[:8]
                    new_name = "-".join(name_parts) + f"-{file_hash}{ext}"
                    dst = os.path.join(char_dir, new_name)
                shutil.copy2(src, dst)

                # 에셋 프롬프트도 복사
                base = os.path.splitext(os.path.basename(src))[0]
                asset_prompt_path = os.path.join(os.path.dirname(src), f"{base}_prompt.json")
                prompt = ""
                negative = ""
                if os.path.isfile(asset_prompt_path):
                    try:
                        with open(asset_prompt_path, "r", encoding="utf-8") as f:
                            pd = json.load(f)
                            prompt = pd.get("positive", "")
                            negative = pd.get("negative", "")
                    except Exception:
                        pass

                new_base = os.path.splitext(new_name)[0]
                bot_prompt_path = os.path.join(char_dir, f"{new_base}_prompt.json")
                with open(bot_prompt_path, "w", encoding="utf-8") as f:
                    json.dump({"prompt": prompt, "negative": negative, "source": "asset", "original_path": rel_path}, f, ensure_ascii=False)

                imported.append({"filename": new_name, "prompt": prompt})
                print(f"[BOT_MODE] 에셋 가져오기: {rel_path} → {bot_name}/{char_name}/{new_name}")

            return _json_ok({"imported": imported, "count": len(imported)})
        except Exception as e:
            print(f"[BOT_MODE] 에셋 가져오기 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    # ─── 프롬프트 업데이트 ────────────────────────────────
    async def handle_update_prompt(self, request):
        """POST /api/bot_mode/prompt - 이미지 프롬프트 수정"""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            filename = body.get("filename", "").strip()
            prompt = body.get("prompt", "")

            if not bot_name or not char_name or not filename:
                return _json_error("필수 값이 누락되었습니다.")

            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            base = os.path.splitext(filename)[0]
            prompt_path = os.path.join(char_dir, f"{base}_prompt.json")

            # 기존 데이터 유지하면서 prompt만 업데이트
            existing = {}
            if os.path.isfile(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    pass

            existing["prompt"] = prompt
            with open(prompt_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False)

            return _json_ok({"updated": True})
        except Exception as e:
            print(f"[BOT_MODE] 프롬프트 업데이트 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    # ─── 이미지 삭제 ─────────────────────────────────────
    async def handle_delete_image(self, request):
        """POST /api/bot_mode/delete_image - 이미지 삭제"""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            filename = body.get("filename", "").strip()

            if not bot_name or not char_name or not filename:
                return _json_error("필수 값이 누락되었습니다.")

            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            filepath = os.path.join(char_dir, filename)
            filepath = os.path.normpath(filepath)
            if not filepath.startswith(os.path.normpath(BOT_DIR)):
                return _json_error("잘못된 경로입니다.")
            if not os.path.isfile(filepath):
                return _json_error("파일을 찾을 수 없습니다.")

            os.remove(filepath)
            # 프롬프트 파일도 삭제
            base = os.path.splitext(filename)[0]
            prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
            if os.path.isfile(prompt_path):
                os.remove(prompt_path)

            print(f"[BOT_MODE] 이미지 삭제: {bot_name}/{char_name}/{filename}")
            return _json_ok({"deleted": True})
        except Exception as e:
            print(f"[BOT_MODE] 이미지 삭제 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    # ─── 에셋 이미지 목록 (가져오기용) ────────────────────
    async def handle_get_asset_images(self, request):
        """GET /api/bot_mode/asset_images - 에셋에서 가져올 이미지 목록"""
        try:
            images = []
            if not os.path.isdir(ASSET_DIR):
                return _json_ok({"images": images, "characters": []})

            # 캐릭터/의상/표정 구조 탐색
            chars = []
            for char_name in sorted(os.listdir(ASSET_DIR)):
                char_dir = os.path.join(ASSET_DIR, char_name)
                if not os.path.isdir(char_dir):
                    continue
                chars.append(char_name)

            return _json_ok({"characters": chars})
        except Exception as e:
            print(f"[BOT_MODE] 에셋 목록 조회 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_asset_character_images(self, request):
        """GET /api/bot_mode/asset_character_images?character=xxx - 특정 캐릭터의 에셋 이미지"""
        char_name = request.query.get("character", "").strip()
        if not char_name:
            return _json_error("캐릭터 이름이 필요합니다.")

        char_dir = os.path.join(ASSET_DIR, char_name)
        if not os.path.isdir(char_dir):
            print(f"[BOT_MODE] 에셋 캐릭터 폴더 없음: {char_dir}")
            return _json_ok({"images": [], "outfits": []})

        # 의상/표정 구조 탐색
        outfits = []
        all_images = []
        for item in sorted(os.listdir(char_dir)):
            item_path = os.path.join(char_dir, item)
            if not os.path.isdir(item_path):
                continue
            # 의상 폴더
            expressions = []
            for expr_name in sorted(os.listdir(item_path)):
                expr_path = os.path.join(item_path, expr_name)
                if not os.path.isdir(expr_path):
                    continue
                # 표정 폴더
                expr_images = []
                for fname in sorted(os.listdir(expr_path)):
                    if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                        rel = f"{char_name}/{item}/{expr_name}/{fname}"
                        expr_images.append({
                            "filename": fname,
                            "path": rel,
                            "url": f"/api/asset_mode/characters/{char_name}/outfits/{item}/expressions/{expr_name}/images/{fname}",
                        })
                if expr_images:
                    expressions.append({"name": expr_name, "images": expr_images})
                    all_images.extend(expr_images)
            if expressions:
                outfits.append({"name": item, "expressions": expressions})

        # 의상/표정 없이 바로 이미지가 있는 경우
        direct_images = []
        for fname in sorted(os.listdir(char_dir)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                rel = f"{char_name}/{fname}"
                print(f"[BOT_MODE] 경고: 캐릭터 폴더 바로 아래 이미지({rel})는 가져오기에서 미리보기 불가")
                direct_images.append({
                    "filename": fname,
                    "path": rel,
                    "url": "",
                })
        if direct_images:
            all_images.extend(direct_images)

        return _json_ok({"outfits": outfits, "direct_images": direct_images, "all_count": len(all_images)})

    def _get_rep_image_paths(self, bot_name: str, char_name: str) -> list[dict]:
        """대표이미지 파일 경로 목록 반환."""
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
        if not bot:
            return []
        ch = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not ch:
            return []
        rep_images = ch.get("rep_images", [])
        char_dir = os.path.join(BOT_DIR, bot_name, char_name)
        results = []
        for fn in rep_images:
            fp = os.path.join(char_dir, fn)
            if os.path.isfile(fp):
                results.append({"character": char_name, "filename": fn, "filepath": fp})
            else:
                print(f"[BOT_MODE] 대표이미지 파일 없음: {fp}")
        return results

    def _get_utility_image_paths(self, bot_name: str, char_name: str = "") -> list[dict]:
        """유틸리티 결과 이미지(_face_image.webp) 경로 목록 반환."""
        results = []
        if char_name:
            chars = [char_name]
        else:
            data = _load_bot_data()
            bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
            chars = [c["name"] for c in (bot.get("characters", []) if bot else [])] if bot else []

        for cn in chars:
            fp = os.path.join(BOT_DIR, bot_name, cn, "_face_image.webp")
            if os.path.isfile(fp):
                results.append({"character": cn, "filename": "_face_image.webp", "filepath": fp})
        return results

    async def handle_get_utility_preview(self, request):
        """GET /api/bot_mode/utility_preview?bot=X&character=Y"""
        try:
            bot_name = request.query.get("bot", "").strip()
            char_name = request.query.get("character", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")

            reps = self._get_utility_image_paths(bot_name, char_name)
            results = []
            for rep in reps:
                base = os.path.splitext(rep["filename"])[0]
                char_dir = os.path.join(BOT_DIR, bot_name, rep["character"])
                prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
                prompt = ""
                negative = ""
                if os.path.isfile(prompt_path):
                    try:
                        with open(prompt_path, "r", encoding="utf-8") as pf:
                            pdata = json.load(pf)
                            prompt = pdata.get("prompt", "")
                            negative = pdata.get("negative", "")
                    except Exception:
                        pass
                results.append({
                    "character": rep["character"],
                    "filename": rep["filename"],
                    "prompt": prompt,
                    "negative": negative,
                    "url": f"/api/bot_mode/image/{bot_name}/{rep['character']}/{rep['filename']}",
                })
            return _json_ok({"images": results})
        except Exception as e:
            print(f"[BOT_MODE] utility_preview 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_asset_chars_with_rep(self, request):
        """GET /api/bot_mode/asset_chars_with_rep - 에셋 캐릭터 목록(대표 1장만 경량 로드).

        진입 시점에는 모든 캐릭터를 전체 스캔하지 않고, 각 캐릭터의 첫 대표이미지
        정보만 빠르게 반환한다. 캐릭터별 전체 대표 이미지는 선택(체크) 시점에
        /api/bot_mode/asset_character_rep_images 로 지연 조회한다.
        """
        try:
            from modes import asset_mode as _am
            reps = _am.get_characters_representative()
            chars = []
            for char_name, info in reps.items():
                fn = info.get("filename", "")
                if not fn:
                    continue
                outfit = info.get("outfit", "")
                expression = info.get("expression", "")
                rel = f"{char_name}/{outfit}/{expression}/{fn}"
                url = (f"/api/asset_mode/characters/{char_name}/outfits/"
                       f"{outfit}/expressions/{expression}/images/{fn}")
                chars.append({
                    "name": char_name,
                    "rep_image": {
                        "filename": fn, "outfit": outfit, "expression": expression,
                        "path": rel, "url": url,
                    },
                })
            return _json_ok({"characters": chars})
        except Exception as e:
            print(f"[BOT_MODE] asset_chars_with_rep 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_asset_character_rep_images(self, request):
        """GET /api/bot_mode/asset_character_rep_images?character=xxx
        단일 캐릭터의 모든 대표 이미지 목록 (선택 시점 지연 로드용)."""
        char_name = request.query.get("character", "").strip()
        if not char_name:
            return _json_error("캐릭터 이름이 필요합니다.")
        try:
            from modes import asset_mode as _am
            gallery = await asyncio.get_event_loop().run_in_executor(
                None, _am.list_character_gallery, char_name
            )
            reps = [g for g in gallery if g.get("representative")]
            rep_images = []
            for g in reps:
                fn = g["representative"]
                outfit = g["outfit"]
                expression = g["expression"]
                rel = f"{char_name}/{outfit}/{expression}/{fn}"
                url = (f"/api/asset_mode/characters/{char_name}/outfits/"
                       f"{outfit}/expressions/{expression}/images/{fn}")
                rep_images.append({
                    "filename": fn, "outfit": outfit, "expression": expression,
                    "path": rel, "url": url,
                })
            return _json_ok({
                "name": char_name,
                "rep_count": len(rep_images),
                "rep_images": rep_images,
            })
        except Exception as e:
            print(f"[BOT_MODE] asset_character_rep_images 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_import_asset_chars(self, request):
        """POST /api/bot_mode/import_asset_chars - 에셋 캐릭터를 봇으로 가져오기."""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            characters = body.get("characters", [])  # [{name, rep_images: [{path, ...}]}]
            if not bot_name or not characters:
                return _json_error("봇 이름과 캐릭터 목록이 필요합니다.")

            imported = []
            for char_info in characters:
                original_name = char_info.get("name", "").strip()
                char_name = char_info.get("import_name", "").strip() or original_name
                rep_images = char_info.get("rep_images", [])
                if not char_name or not rep_images:
                    continue

                # 캐릭터 생성 (없으면)
                data = _load_bot_data()
                bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
                if not bot:
                    return _json_error(f"봇을 찾을 수 없음: {bot_name}")
                if not any(c["name"] == char_name for c in bot.get("characters", [])):
                    if "characters" not in bot:
                        bot["characters"] = []
                    bot["characters"].append({"name": char_name})
                    _save_bot_data(data)

                char_dir = os.path.join(BOT_DIR, bot_name, char_name)
                os.makedirs(char_dir, exist_ok=True)

                imported_files = []
                for ri in rep_images:
                    src = os.path.join(ASSET_DIR, ri["path"])
                    src = os.path.normpath(src)
                    if not os.path.isfile(src):
                        print(f"[BOT_MODE] 에셋 파일 없음: {src}")
                        continue

                    ext = os.path.splitext(src)[1].lower()
                    if ext not in IMAGE_EXTENSIONS:
                        ext = ".webp"
                    outfit = ri.get("outfit", "")
                    expr = ri.get("expression", "")
                    file_hash = uuid.uuid4().hex[:8]
                    name_parts = [p for p in [char_name, outfit, expr] if p]
                    new_name = "-".join(name_parts) + f"-{file_hash}{ext}"
                    dst = os.path.join(char_dir, new_name)
                    if os.path.exists(dst):
                        file_hash = uuid.uuid4().hex[:8]
                        new_name = "-".join(name_parts) + f"-{file_hash}{ext}"
                        dst = os.path.join(char_dir, new_name)
                    shutil.copy2(src, dst)

                    # 프롬프트 복사
                    base = os.path.splitext(os.path.basename(src))[0]
                    asset_prompt_path = os.path.join(os.path.dirname(src), f"{base}_prompt.json")
                    prompt = ""
                    negative = ""
                    if os.path.isfile(asset_prompt_path):
                        try:
                            with open(asset_prompt_path, "r", encoding="utf-8") as f:
                                apd = json.load(f)
                                prompt = apd.get("positive", "")
                                negative = apd.get("negative", "")
                        except Exception:
                            pass
                    new_base = os.path.splitext(new_name)[0]
                    bot_prompt_path = os.path.join(char_dir, f"{new_base}_prompt.json")
                    with open(bot_prompt_path, "w", encoding="utf-8") as f:
                        json.dump({"prompt": prompt, "negative": negative, "source": "asset", "original_path": ri["path"]}, f, ensure_ascii=False)

                    imported_files.append(new_name)

                imported.append({"character": char_name, "files": imported_files})
                rename_info = f" ({original_name} -> {char_name})" if original_name != char_name else ""
                print(f"[BOT_MODE] 에셋 캐릭터 가져오기: {char_name}{rename_info} ({len(imported_files)}장)")

            return _json_ok({"imported": imported})
        except Exception as e:
            print(f"[BOT_MODE] 에셋 캐릭터 가져오기 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_rep_preview(self, request):
        """GET /api/bot_mode/rep_preview?bot=xxx - 대표이미지 프리뷰 (파일명+프롬프트)."""
        try:
            bot_name = request.query.get("bot", "").strip()
            char_name = request.query.get("character", "").strip()
            print(f"[BOT_MODE] rep_preview 요청: bot={bot_name!r}, character={char_name!r}")
            if not bot_name:
                print("[BOT_MODE] rep_preview 오류: 봇 이름 없음")
                return _json_error("봇 이름이 필요합니다.")

            if char_name:
                reps = self._get_rep_image_paths(bot_name, char_name)
            else:
                reps = []
                data = _load_bot_data()
                bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
                if bot:
                    chars_with_rep = [ch["name"] for ch in bot.get("characters", []) if ch.get("rep_images")]
                    print(f"[BOT_MODE] rep_preview: 봇 '{bot_name}' 캐릭터 수={len(bot.get('characters', []))}, 대표이미지 있는 캐릭터={chars_with_rep}")
                    for ch in bot.get("characters", []):
                        if ch.get("rep_images"):
                            reps.extend(self._get_rep_image_paths(bot_name, ch["name"]))
                else:
                    print(f"[BOT_MODE] rep_preview: 봇 '{bot_name}'을(를) 찾을 수 없음")

            print(f"[BOT_MODE] rep_preview: 총 대표이미지 {len(reps)}장")
            results = []
            for rep in reps:
                base = os.path.splitext(rep["filename"])[0]
                char_dir = os.path.join(BOT_DIR, bot_name, rep["character"])
                prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
                prompt = ""
                negative = ""
                if os.path.isfile(prompt_path):
                    try:
                        with open(prompt_path, "r", encoding="utf-8") as pf:
                            pdata = json.load(pf)
                            prompt = pdata.get("prompt", "")
                            negative = pdata.get("negative", "")
                    except Exception:
                        pass
                results.append({
                    "character": rep["character"],
                    "filename": rep["filename"],
                    "prompt": prompt,
                    "negative": negative,
                    "url": f"/api/bot_mode/image/{bot_name}/{rep['character']}/{rep['filename']}",
                })
            return _json_ok({"images": results})
        except Exception as e:
            print(f"[BOT_MODE] rep_preview 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_batch_analyze_rep(self, request):
        """POST /api/bot_mode/batch_analyze_rep - 대표이미지 일괄 태그 분석 → 큐에 추가."""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")

            asset_tool = self._asset_tool
            if not asset_tool or (not asset_tool.use_builtin_tagger and not asset_tool.workflow_source_path):
                return _json_error("태그 분석 워크플로우 경로가 설정되지 않았습니다")

            from queue_manager import queue_manager
            reps = queue_manager._get_bot_rep_paths(bot_name, char_name)
            filenames = body.get("filenames", [])
            if filenames:
                _fnset = set(filenames)
                reps = [r for r in reps if r["filename"] in _fnset]
            batch_label = f"태그 분석 (봇 대표: {bot_name}/{char_name or '전체'}, {len(reps)}장)"
            items_spec = []
            for r in reps:
                img = {"filepath": r["filepath"], "filename": r["filename"], "character": r["character"], "bot": bot_name}
                items_spec.append({
                    "type": "tag_analysis",
                    "label": f"태그 분석(봇 대표) {bot_name}/{r['character']}/{r['filename']}",
                    "batch_label": batch_label,
                    "params": {"source": "bot_rep", "image": img},
                })
            created = await queue_manager.add_items_batch(items_spec)
            batch_id = created[0].batch_id if created else None
            return _json_ok({"success": True, "batch_id": batch_id, "count": len(created), "total": len(reps)})
        except Exception as e:
            print(f"[BOT_MODE] 일괄 분석 큐 추가 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    # ─── 태그 필터링 ─────────────────────────────────────────
    async def handle_get_tag_filter_profiles(self, request):
        """GET /api/bot_mode/tag_filter_profiles"""
        data = _load_tag_filter_profiles()
        return _json_ok({
            "profiles": data.get("profiles", {}),
            "active_profile": data.get("active_profile", ""),
        })

    async def handle_save_tag_filter_profile(self, request):
        """POST /api/bot_mode/tag_filter_profile_save"""
        body = await request.json()
        name = body.get("name", "").strip()
        steps = body.get("steps", [])
        if not name:
            return _json_error("프로필 이름이 필요합니다.")
        data = _load_tag_filter_profiles()
        data.setdefault("profiles", {})[name] = steps
        data["active_profile"] = name
        _save_tag_filter_profiles(data)
        return _json_ok({"success": True, "name": name})

    async def handle_delete_tag_filter_profile(self, request):
        """POST /api/bot_mode/tag_filter_profile_delete"""
        body = await request.json()
        name = body.get("name", "")
        data = _load_tag_filter_profiles()
        profiles = data.get("profiles", {})
        if name not in profiles:
            return _json_error(f"프로필 '{name}'을 찾을 수 없습니다")
        if len(profiles) <= 1:
            return _json_error("마지막 프로필은 삭제할 수 없습니다")
        del profiles[name]
        if data.get("active_profile") == name:
            data["active_profile"] = next(iter(profiles))
        _save_tag_filter_profiles(data)
        return _json_ok({"success": True, "deleted": name})

    async def handle_tag_filter_preview(self, request):
        """POST /api/bot_mode/tag_filter_preview"""
        body = await request.json()
        bot_name = body.get("bot", "").strip()
        items = body.get("items", [])
        steps = body.get("steps", [])
        if not bot_name:
            return _json_error("봇 이름이 필요합니다.")
        preview = []
        for item in items:
            char_name = item.get("character", "")
            filename = item.get("filename", "")
            base = os.path.splitext(filename)[0]
            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
            original = ""
            if os.path.isfile(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        original = json.load(f).get("prompt", "")
                except Exception:
                    pass
            filtered = _apply_tag_filter_steps(original, steps) if original else ""
            preview.append({
                "character": char_name,
                "filename": filename,
                "original": original,
                "filtered": filtered,
            })
        return _json_ok({"preview": preview})

    async def handle_tag_filter_apply(self, request):
        """POST /api/bot_mode/tag_filter_apply"""
        body = await request.json()
        bot_name = body.get("bot", "").strip()
        items = body.get("items", [])
        steps = body.get("steps", [])
        if not bot_name:
            return _json_error("봇 이름이 필요합니다.")
        success_count = 0
        fail_count = 0
        for item in items:
            try:
                char_name = item.get("character", "")
                filename = item.get("filename", "")
                base = os.path.splitext(filename)[0]
                char_dir = os.path.join(BOT_DIR, bot_name, char_name)
                prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
                existing = {}
                if os.path.isfile(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                original = existing.get("prompt", "")
                filtered = _apply_tag_filter_steps(original, steps) if original else ""
                existing["prompt"] = filtered
                with open(prompt_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
                success_count += 1
            except Exception as e:
                fail_count += 1
                print(f"[BOT_MODE] 태그 필터 적용 실패: {item} - {e}")
        return _json_ok({
            "total": len(items),
            "success_count": success_count,
            "fail_count": fail_count,
        })

    async def handle_batch_set_negative(self, request):
        """POST /api/bot_mode/batch_set_negative - 대표이미지에 부정프롬프트 일괄 적용."""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            negative_tags = body.get("negative_tags", "")
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")

            if char_name:
                reps = self._get_rep_image_paths(bot_name, char_name)
            else:
                reps = []
                data = _load_bot_data()
                bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
                if bot:
                    for ch in bot.get("characters", []):
                        if (ch.get("rep_images") or []):
                            reps.extend(self._get_rep_image_paths(bot_name, ch["name"]))
            if not reps:
                return _json_ok({"total": 0, "success_count": 0, "fail_count": 0})

            # filenames 필터
            only_filenames = body.get("filenames", [])
            if only_filenames:
                reps = [r for r in reps if r["filename"] in only_filenames]
            if not reps:
                return _json_ok({"total": 0, "success_count": 0, "fail_count": 0})

            success_count = 0
            fail_count = 0
            for rep in reps:
                try:
                    base = os.path.splitext(rep["filename"])[0]
                    char_dir = os.path.join(BOT_DIR, bot_name, rep["character"])
                    prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
                    existing = {}
                    if os.path.isfile(prompt_path):
                        try:
                            with open(prompt_path, "r", encoding="utf-8") as pf:
                                existing = json.load(pf)
                        except Exception:
                            pass
                    existing["negative"] = negative_tags
                    with open(prompt_path, "w", encoding="utf-8") as pf:
                        json.dump(existing, pf, ensure_ascii=False, indent=2)
                    success_count += 1
                    print(f"[BOT_MODE] 부정 프롬프트 적용 완료: {rep['filename']}")
                except Exception as e:
                    fail_count += 1
                    print(f"[BOT_MODE] 부정 프롬프트 적용 실패: {rep['filename']} - {e}")
                    traceback.print_exc()

            return _json_ok({"total": len(reps), "success_count": success_count, "fail_count": fail_count})
        except Exception as e:
            print(f"[BOT_MODE] batch_set_negative 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_batch_analyze_utility(self, request):
        """POST /api/bot_mode/batch_analyze_utility - 유틸리티 이미지 일괄 태그 분석 → 큐에 추가."""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")

            asset_tool = self._asset_tool
            if not asset_tool or (not asset_tool.use_builtin_tagger and not asset_tool.workflow_source_path):
                return _json_error("태그 분석 워크플로우 경로가 설정되지 않았습니다")

            from queue_manager import queue_manager
            reps = queue_manager._get_bot_utility_paths(bot_name, char_name)
            # 모든 캐릭터의 얼굴 이미지 파일명이 "_face_image.webp"로 동일하므로
            # filename이 아닌 character 이름으로 선택 여부를 필터링해야 함.
            characters = body.get("characters", [])
            if characters:
                _cset = set(str(c).strip() for c in characters if str(c).strip())
                reps = [r for r in reps if r["character"] in _cset]
            # (레거시) filenames 만 온 경우에도 filename은 캐릭터를 구분하지 못하므로 무시.
            batch_label = f"태그 분석 (봇 유틸: {bot_name}, {len(reps)}장)"
            items_spec = []
            for r in reps:
                img = {"filepath": r["filepath"], "filename": r["filename"], "character": r["character"], "bot": bot_name}
                items_spec.append({
                    "type": "tag_analysis",
                    "label": f"태그 분석(봇 유틸) {bot_name}/{r['character']}/{r['filename']}",
                    "batch_label": batch_label,
                    "params": {"source": "bot_utility", "image": img},
                })
            created = await queue_manager.add_items_batch(items_spec)
            batch_id = created[0].batch_id if created else None
            return _json_ok({"success": True, "batch_id": batch_id, "count": len(created), "total": len(reps)})
        except Exception as e:
            print(f"[BOT_MODE] 유틸리티 분석 큐 추가 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_batch_set_negative_utility(self, request):
        """POST /api/bot_mode/batch_set_negative_utility - 유틸리티 이미지에 부정프롬프트 일괄 적용."""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            negative_tags = body.get("negative_tags", "")
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")

            reps = self._get_utility_image_paths(bot_name, char_name)
            # 모든 캐릭터의 얼굴 이미지 파일명이 "_face_image.webp"로 동일하므로
            # filename이 아닌 character 이름으로 선택 여부를 필터링해야 함.
            characters = body.get("characters", [])
            if characters:
                _cset = set(str(c).strip() for c in characters if str(c).strip())
                reps = [r for r in reps if r["character"] in _cset]
            # (레거시) filenames 만 온 경우에도 filename은 캐릭터를 구분하지 못하므로 무시.
            if not reps:
                return _json_ok({"total": 0, "success_count": 0, "fail_count": 0})

            success_count = 0
            fail_count = 0
            for rep in reps:
                try:
                    base = os.path.splitext(rep["filename"])[0]
                    char_dir = os.path.join(BOT_DIR, bot_name, rep["character"])
                    prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
                    existing = {}
                    if os.path.isfile(prompt_path):
                        try:
                            with open(prompt_path, "r", encoding="utf-8") as pf:
                                existing = json.load(pf)
                        except Exception:
                            pass
                    existing["negative"] = negative_tags
                    with open(prompt_path, "w", encoding="utf-8") as pf:
                        json.dump(existing, pf, ensure_ascii=False, indent=2)
                    success_count += 1
                    print(f"[BOT_MODE] 유틸리티 부정프롬프트 적용: {rep['character']}/{rep['filename']}")
                except Exception as e:
                    fail_count += 1
                    print(f"[BOT_MODE] 유틸리티 부정프롬프트 실패: {rep['character']}/{rep['filename']} - {e}")
                    traceback.print_exc()

            return _json_ok({"total": len(reps), "success_count": success_count, "fail_count": fail_count})
        except Exception as e:
            print(f"[BOT_MODE] batch_set_negative_utility 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_analyze_single(self, request):
        """POST /api/bot_mode/analyze_single - 단일 이미지 태그 분석 → 큐에 추가."""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            filename = body.get("filename", "").strip()
            if not bot_name or not char_name or not filename:
                return _json_error("봇, 캐릭터, 파일명이 필요합니다.")

            asset_tool = self._asset_tool
            if not asset_tool or (not asset_tool.use_builtin_tagger and not asset_tool.workflow_source_path):
                return _json_error("태그 분석 워크플로우 경로가 설정되지 않았습니다")

            from queue_manager import queue_manager
            label = f"태그 분석 (봇: {bot_name}/{char_name}/{filename})"
            item = await queue_manager.add_item("tag_analysis", label, {
                "source": "bot_single", "bot": bot_name, "character": char_name, "filename": filename,
            })
            return _json_ok({"success": True, "item_id": item.id})
        except Exception as e:
            print(f"[BOT_MODE] 단일 분석 큐 추가 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_set_negative_single(self, request):
        """POST /api/bot_mode/set_negative_single - 단일 이미지 부정프롬프트 적용."""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            filename = body.get("filename", "").strip()
            negative_tags = body.get("negative_tags", "")
            if not bot_name or not char_name or not filename:
                return _json_error("봇, 캐릭터, 파일명이 필요합니다.")

            base = os.path.splitext(filename)[0]
            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
            existing = {}
            if os.path.isfile(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as pf:
                        existing = json.load(pf)
                except Exception:
                    pass
            existing["negative"] = negative_tags
            with open(prompt_path, "w", encoding="utf-8") as pf:
                json.dump(existing, pf, ensure_ascii=False, indent=2)

            return _json_ok({"updated": True})
        except Exception as e:
            print(f"[BOT_MODE] 단일 부정프롬프트 적용 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    # ─── 유틸리티 설정 ──────────────────────────────────────
    async def handle_get_utility_settings(self, request):
        """GET /api/bot_mode/utility_settings?bot=X&character=Y"""
        try:
            bot_name = request.query.get("bot", "").strip()
            char_name = request.query.get("character", "").strip()
            if not bot_name or not char_name:
                return _json_error("봇, 캐릭터 이름이 필요합니다.")
            settings = _load_utility_settings(bot_name, char_name)
            prompt = build_utility_prompt(bot_name, char_name, settings)
            return _json_ok({"settings": settings, "prompt_preview": prompt})
        except Exception as e:
            print(f"[BOT_MODE] utility_settings 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_patch_settings(self, request):
        """GET /api/bot_mode/patch_settings?bot=X"""
        try:
            bot_name = request.query.get("bot", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")
            settings = _load_patch_settings(bot_name)
            return _json_ok(settings)
        except Exception as e:
            print(f"[BOT_MODE] patch_settings 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_patch_settings(self, request):
        """POST /api/bot_mode/patch_settings"""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            settings = body.get("settings", {})
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")
            _save_patch_settings(bot_name, settings)
            print(f"[BOT_MODE] 패치 설정 저장: bot={bot_name}, settings={settings}")
            return _json_ok({"saved": True})
        except Exception as e:
            print(f"[BOT_MODE] patch_settings 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_word_replacements(self, request):
        """GET /api/bot_mode/word_replacements?bot=X"""
        try:
            bot_name = request.query.get("bot", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")
            data = _load_word_replacements(bot_name)
            return _json_ok(data)
        except Exception as e:
            print(f"[BOT_MODE] word_replacements 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_word_replacements(self, request):
        """POST /api/bot_mode/word_replacements"""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            rules = body.get("rules", [])
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")
            _save_word_replacements(bot_name, {"rules": rules})
            print(f"[BOT_MODE] 단어 기반 규칙 저장: bot={bot_name}, {len(rules)}개 규칙")
            return _json_ok({"saved": True})
        except Exception as e:
            print(f"[BOT_MODE] word_replacements 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_utility_settings(self, request):
        """POST /api/bot_mode/utility_settings"""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            char_name = body.get("character", "").strip()
            settings = body.get("settings", {})
            if not bot_name or not char_name:
                return _json_error("봇, 캐릭터 이름이 필요합니다.")
            _save_utility_settings(bot_name, char_name, settings)
            prompt = build_utility_prompt(bot_name, char_name, settings)
            return _json_ok({"saved": True, "prompt_preview": prompt})
        except Exception as e:
            print(f"[BOT_MODE] utility_settings 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_postprocess_vn(self, request):
        """GET /api/bot_mode/postprocess_vn?bot=X"""
        try:
            bot_name = request.query.get("bot", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")
            vn = _load_postprocess_vn(bot_name)
            return _json_ok(vn)
        except Exception as e:
            print(f"[BOT_MODE] postprocess_vn 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_postprocess_vn(self, request):
        """POST /api/bot_mode/postprocess_vn  body: {bot, vn}"""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            vn = body.get("vn", {})
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")
            async with self._lock:
                _save_postprocess_vn(bot_name, vn)
            return _json_ok({"saved": True})
        except Exception as e:
            print(f"[BOT_MODE] postprocess_vn 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_postprocess_bubble(self, request):
        """GET /api/bot_mode/postprocess_bubble?bot=X"""
        try:
            bot_name = request.query.get("bot", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")
            bubble = _load_postprocess_bubble(bot_name)
            mode = _get_postprocess_mode(bot_name)
            return _json_ok({"bubble": bubble, "mode": mode})
        except Exception as e:
            print(f"[BOT_MODE] postprocess_bubble 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_postprocess_bubble(self, request):
        """POST /api/bot_mode/postprocess_bubble  body: {bot, bubble, mode?}"""
        try:
            body = await request.json()
            bot_name = body.get("bot", "").strip()
            bubble = body.get("bubble", {})
            mode = body.get("mode")  # 'vn' | 'bubble' (선택)
            if not bot_name:
                return _json_error("봇 이름이 필요합니다.")
            async with self._lock:
                _save_postprocess_bubble(bot_name, bubble)
                if mode in ("vn", "bubble"):
                    _set_postprocess_mode(bot_name, mode)
            return _json_ok({"saved": True})
        except Exception as e:
            print(f"[BOT_MODE] postprocess_bubble 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))


    async def handle_get_lb_extra(self, request):
        """GET /api/bot_mode/lb_extra - 저장된 분류 데이터(편집본) 로드"""
        try:
            bot_name = request.query.get("bot_name", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")

            saved = _load_lb_extra(bot_name)
            if saved is None:
                return _json_ok({"data": None})

            # 구버전 호환: {original, edited} 구조면 edited만 사용
            if isinstance(saved, dict) and "edited" in saved:
                saved = saved["edited"]

            data = _load_bot_data()
            bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
            if not bot:
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")

            current_names = {c["name"] for c in bot.get("characters", [])}
            filtered = [e for e in saved if e.get("name") in current_names]

            if len(filtered) != len(saved):
                _save_lb_extra(bot_name, filtered)
                print(f"[LB_EXTRA] 캐릭터 불일치: {len(saved)} -> {len(filtered)}개로 정리")

            return _json_ok({"data": filtered})
        except Exception as e:
            print(f"[LB_EXTRA] 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_lb_extra(self, request):
        """POST /api/bot_mode/lb_extra - 분류 데이터 저장"""
        try:
            body = await request.json()
            bot_name = body.get("bot_name", "").strip()
            extra_data = body.get("data")
            original_data = body.get("original")
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")
            if extra_data is None:
                return _json_error("데이터가 없습니다.")

            _save_lb_extra(bot_name, extra_data)
            return _json_ok({"saved": True})
        except Exception as e:
            print(f"[LB_EXTRA] 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    # ─── 시스템 프롬프트 ────────────────────────────────────
    async def handle_get_system_prompt(self, request):
        """GET /api/bot_mode/system_prompt - 봇의 시스템 프롬프트 반환

        선택 프리셋은 builtin(배포자료·읽기전용, git 배포) 또는 local(bot.json, 편집가능) 이다.
        scope 로 어느 공간인지 구분해 본문을 해석한다.
        """
        try:
            bot_name = request.query.get("bot_name", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")
            data = _load_bot_data()
            bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
            if not bot:
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")
            builtin = _load_builtin_presets() or {}
            local = data.get("system_prompt_presets", {}) or {}
            _ensure_bot_preset_scope(bot, set(builtin.keys()), set(local.keys()))

            preset = (bot.get("system_prompt_preset") or "").strip()
            scope = (bot.get("preset_scope") or "local").strip()
            # 참조 보정
            if scope == "builtin" and (not preset or preset not in builtin):
                scope = "local"
            if scope == "local" and (not preset or preset not in local):
                preset = "기본" if "기본" in local else (next(iter(local), "") if local else "")
            # 본문 해석
            if scope == "builtin":
                text = builtin.get(preset, "")
            else:
                text = local.get(preset, bot.get("system_prompt", "")) if preset else bot.get("system_prompt", "")
            return _json_ok({
                "text": text,
                "preset": preset,
                "scope": scope,
                "builtin_presets": builtin,     # 배포자료(잠금)
                "local_presets": local,          # 사용자(편집가능)
            })
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_system_prompt(self, request):
        """POST /api/bot_mode/system_prompt - 시스템 프롬프트 저장 + 봇 바인딩

        - scope=local : 프리셋 본문(덮어쓰기) + 봇 바인딩
        - scope=builtin: 본문(배포자료, 읽기전용)은 건드리지 않고 봇 '선택(바인딩)'만 영속화.
          → 사용자가 배포자료 프리셋을 선택한 사실을 저장하기 위함.
        """
        try:
            body = await request.json()
            bot_name = body.get("bot_name", "").strip()
            text = body.get("text", "")
            preset_name = (body.get("preset_name") or "").strip()
            scope = (body.get("scope") or "local").strip()
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")
            if not preset_name:
                return _json_error("저장할 프리셋이 선택되지 않았습니다. 프리셋을 먼저 선택하세요.")
            if scope not in ("local", "builtin"):
                return _json_error(f"알 수 없는 scope 입니다: {scope}")
            data = _load_bot_data()
            bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
            if not bot:
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")

            if scope == "builtin":
                # 배포자료(읽기전용): 본문 파일은 건드리지 않고, 봇이 이 프리셋을 '선택'했다는 사실만 바인딩.
                builtin = _load_builtin_presets() or {}
                if preset_name not in builtin:
                    return _json_error(f"'{preset_name}' 배포자료 프리셋을 찾을 수 없습니다.")
                bot["system_prompt_preset"] = preset_name
                bot["preset_scope"] = "builtin"
                _save_bot_data(data)
                print(f"[BOT_MODE] 시스템 프롬프트 바인딩(읽기전용): {bot_name} → [builtin] '{preset_name}'")
                return _json_ok({"saved": True, "bind_only": True})

            # local: 프리셋 본문 덮어쓰기 + 봇 바인딩
            if "system_prompt_presets" not in data:
                data["system_prompt_presets"] = {}
            data["system_prompt_presets"][preset_name] = text  # local 프리셋 덮어쓰기
            bot["system_prompt_preset"] = preset_name
            bot["preset_scope"] = "local"
            bot["system_prompt"] = text  # dead field이나 일관성 유지
            _save_bot_data(data)
            print(f"[BOT_MODE] 시스템 프롬프트 저장: {bot_name} → [local] '{preset_name}' ({len(text)}자)")
            return _json_ok({"saved": True})
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_system_prompt_presets(self, request):
        """GET /api/bot_mode/system_prompt_presets - local(편집가능) 프리셋 목록 반환"""
        try:
            data = _load_bot_data()
            return _json_ok({"presets": data.get("system_prompt_presets", {})})
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 프리셋 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_system_prompt_preset(self, request):
        """POST /api/bot_mode/system_prompt_presets - local 프리셋 저장/추가/복제

        scope=local 만 허용. builtin(배포자료) 쓰기는 거부.
        새 프리셋 추가/복제는 항상 local(bot.json) 로 들어간다.
        """
        try:
            body = await request.json()
            name = body.get("name", "").strip()
            text = body.get("text", "")
            scope = (body.get("scope") or "local").strip()
            if not name:
                return _json_error("프리셋 이름이 비어있습니다.")
            if scope == "builtin":
                return _json_error(
                    f"'{name}' 은(는) 배포자료(읽기전용)라 저장할 수 없습니다. 사용자 프리셋으로 추가해주세요."
                )
            builtin = _load_builtin_presets() or {}
            if name in builtin:
                return _json_error(
                    f"'{name}' 은(는) 배포자료(읽기전용) 이름과 같아 사용자 프리셋으로 사용할 수 없습니다. "
                    "다른 이름을 사용하세요."
                )
            data = _load_bot_data()
            if "system_prompt_presets" not in data:
                data["system_prompt_presets"] = {}
            data["system_prompt_presets"][name] = text
            _save_bot_data(data)
            print(f"[BOT_MODE] local 시스템 프롬프트 프리셋 저장: {name} ({len(text)}자)")
            return _json_ok({"saved": True, "presets": data["system_prompt_presets"]})
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 프리셋 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_delete_system_prompt_preset(self, request):
        """DELETE /api/bot_mode/system_prompt_presets - local 프리셋 삭제

        scope=local 만 삭제 허용. builtin(배포자료)은 삭제 불가.
        """
        try:
            body = await request.json()
            name = body.get("name", "").strip()
            scope = (body.get("scope") or "local").strip()
            if not name:
                return _json_error("프리셋 이름이 비어있습니다.")
            if scope == "builtin":
                return _json_error("배포자료(읽기전용) 프리셋은 삭제할 수 없습니다.")
            builtin = _load_builtin_presets() or {}
            if name in builtin:
                return _json_error(f"'{name}' 은(는) 배포자료(읽기전용)라 삭제할 수 없습니다.")
            data = _load_bot_data()
            presets = data.get("system_prompt_presets", {})
            if name not in presets:
                return _json_error(f"사용자 프리셋을 찾을 수 없습니다: {name}")
            # 사용 중인 다른 봇이 있으면 삭제 불가
            using = [b.get("name", "?") for b in data.get("bots", [])
                     if (b.get("system_prompt_preset") or "").strip() == name
                     and (b.get("preset_scope") or "local") == "local"]
            if using:
                return _json_error(f"이 프리셋을 사용 중인 봇이 있어 삭제할 수 없습니다: {', '.join(using)}")
            del presets[name]
            data["system_prompt_presets"] = presets
            _save_bot_data(data)
            print(f"[BOT_MODE] local 시스템 프롬프트 프리셋 삭제: {name}")
            return _json_ok({"deleted": True, "presets": presets})
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 프리셋 삭제 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))


# ─── 유틸리티 ──────────────────────────────────────────
def _json_ok(data, status=200):
    return web.json_response(data, status=status)


def _json_error(msg, status=400):
    print(f"[BOT_MODE] 에러: {msg}")
    return web.json_response({"error": msg}, status=status)


# ─── 태그 필터 프로필 관리 ─────────────────────────────────
def _load_tag_filter_profiles() -> dict:
    default = {"profiles": {"기본": []}, "active_profile": "기본"}
    if os.path.isfile(TAG_FILTER_PROFILES_FILE):
        try:
            with open(TAG_FILTER_PROFILES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "profiles" not in data:
                    data["profiles"] = default["profiles"]
                return data
        except Exception as e:
            print(f"[BOT_MODE] tag_filter_profiles 로드 실패: {e}")
    return default


def _save_tag_filter_profiles(data: dict):
    try:
        os.makedirs(ASSET_DATA_DIR, exist_ok=True)
        with open(TAG_FILTER_PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[BOT_MODE] tag_filter_profiles 저장 실패: {e}")
        traceback.print_exc()


import re as _re

def _apply_tag_filter_steps(tags_string: str, steps: list[dict]) -> str:
    """쉼표로 구분된 태그 문자열에 필터 단계를 적용.
    split_by는 쉼표 분리 전 전체 문자열에 먼저 적용.
    나머지 액션은 쉼표 분리 후 개별 태그에 적용.
    """
    from modes.embedding_service import _parse_take, _apply_take

    # 1단계: split_by는 전체 문자열에 순차 적용
    current_str = tags_string
    per_tag_steps = []
    for step in steps:
        if step.get("action") == "split_by":
            sep = step.get("separator", "_")
            take = step.get("take", 0)
            parts = current_str.split(sep)
            take_parsed = _parse_take(take)
            current_str = _apply_take(parts, take_parsed, join_sep=sep)
        else:
            per_tag_steps.append(step)

    # 2단계: 쉼표 분리 후 개별 태그에 나머지 액션 적용
    tags = [t.strip() for t in current_str.split(",") if t.strip()]
    result_tags = []
    for tag in tags:
        keep = True
        current = tag
        for step in per_tag_steps:
            action = step.get("action", "")
            if action == "remove_match":
                pattern = step.get("pattern", "")
                if pattern and _re.search(pattern, current):
                    keep = False
                    break
            elif action == "replace":
                current = current.replace(step.get("from", ""), step.get("to", ""))
            elif action == "strip_prefix":
                pattern = step.get("pattern", "")
                if pattern:
                    current = _re.sub(f"^{pattern}", "", current)
            elif action == "strip_suffix":
                pattern = step.get("pattern", "")
                if pattern:
                    current = _re.sub(f"{pattern}$", "", current)
            elif action == "strip":
                current = current.strip()
            elif action == "lower":
                current = current.lower()
        if keep and current.strip():
            result_tags.append(current.strip())
    return ", ".join(result_tags)


# ─── 유틸리티 설정 (캐릭터별) ─────────────────────────────
UTILITY_SETTINGS_FILE = "_utility_settings.json"
PATCH_SETTINGS_FILE = "_patch_settings.json"


def _patch_settings_path(bot_name: str) -> str:
    return os.path.join(BOT_DIR, bot_name, PATCH_SETTINGS_FILE)


def _load_patch_settings(bot_name: str) -> dict:
    path = _patch_settings_path(bot_name)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[BOT_MODE] 패치 설정 로드: {bot_name}")
                return data
        except Exception as e:
            print(f"[BOT_MODE] patch_settings 로드 실패: {e}")
    return {"face_crop_top": 1.0, "face_crop_bottom": 1.0, "emb_target": "대표만"}


def _save_patch_settings(bot_name: str, settings: dict):
    path = _patch_settings_path(bot_name)
    bot_dir = os.path.dirname(path)
    os.makedirs(bot_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


WORD_REPLACEMENTS_FILE = "_word_replacements.json"


def _word_replacements_path(bot_name: str) -> str:
    return os.path.join(BOT_DIR, bot_name, WORD_REPLACEMENTS_FILE)


def _load_word_replacements(bot_name: str) -> dict:
    path = _word_replacements_path(bot_name)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[BOT_MODE] 단어 기반 규칙 로드: {bot_name} ({len(data.get('rules', []))}개 규칙)")
                return data
        except Exception as e:
            print(f"[BOT_MODE] word_replacements 로드 실패: {e}")
    return {"rules": []}


def _save_word_replacements(bot_name: str, data: dict):
    path = _word_replacements_path(bot_name)
    bot_dir = os.path.dirname(path)
    os.makedirs(bot_dir, exist_ok=True)
    if os.path.isfile(path):
        backup_dir = os.path.join(BASE_DIR, "요구사항")
        try:
            os.makedirs(backup_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"{bot_name}_{WORD_REPLACEMENTS_FILE}.bak_{stamp}_{uuid.uuid4().hex[:8]}"
            backup_path = os.path.join(backup_dir, backup_name)
            shutil.copy2(path, backup_path)
            print(f"[BOT_MODE] 단어 기반 규칙 기존 파일 백업: {backup_path}")
        except Exception as exc:
            print(f"[BOT_MODE] 단어 기반 규칙 백업 실패(path={path!r}): {exc}")
            traceback.print_exc()
            raise RuntimeError("기존 단어 기반 규칙 백업에 실패하여 저장을 중단했습니다.") from exc
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


LB_EXTRA_FILE = "_lb_extra.json"


def _lb_extra_path(bot_name: str) -> str:
    return os.path.join(BOT_DIR, bot_name, LB_EXTRA_FILE)


def _load_lb_extra(bot_name: str):
    path = _lb_extra_path(bot_name)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[BOT_MODE] lb_extra 로드 실패: {e}")
    return None


def _save_lb_extra(bot_name: str, data):
    path = _lb_extra_path(bot_name)
    bot_dir = os.path.dirname(path)
    os.makedirs(bot_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _utility_settings_path(bot_name: str, char_name: str) -> str:
    return os.path.join(BOT_DIR, bot_name, char_name, UTILITY_SETTINGS_FILE)


def _load_utility_settings(bot_name: str, char_name: str) -> dict:
    path = _utility_settings_path(bot_name, char_name)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[BOT_MODE] utility_settings 로드 실패: {e}")
    return {"face_crop_top": 1.0, "face_crop_bottom": 1.0, "emb_target": "대표만"}


def _save_utility_settings(bot_name: str, char_name: str, settings: dict):
    path = _utility_settings_path(bot_name, char_name)
    char_dir = os.path.dirname(path)
    os.makedirs(char_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ─── 후처리 봇별 설정 (postprocess_vn) ─────────────────────
def _backup_bot_json():
    """bot.json 덮어쓰기 전 요구사항/ 폴더에 백업 (CLAUDE.md 데이터 안전 규칙)."""
    backup_dir = os.path.join(BASE_DIR, "요구사항")
    try:
        os.makedirs(backup_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        if os.path.isfile(BOT_DATA_FILE):
            shutil.copy2(BOT_DATA_FILE, os.path.join(backup_dir, f"bot.json.bak_{ts}"))
    except Exception as e:
        print(f"[BOT_MODE] WARN: bot.json 백업 실패: {e}")


def _load_postprocess_vn(bot_name: str) -> dict:
    """봇의 postprocess_vn 반환. 없으면 기본값."""
    if not bot_name:
        from modes.postprocess import _default_vn
        return _default_vn()
    try:
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
        if bot and isinstance(bot.get("postprocess_vn"), dict):
            from modes.postprocess import _merge_vn_defaults
            return _merge_vn_defaults(bot["postprocess_vn"])
    except Exception as e:
        print(f"[BOT_MODE] postprocess_vn 로드 실패({bot_name}): {e}")
        traceback.print_exc()
    from modes.postprocess import _default_vn
    return _default_vn()


def _save_postprocess_vn(bot_name: str, vn: dict):
    """봇의 postprocess_vn 저장. 백업 후 bot.json 갱신."""
    if not bot_name:
        raise ValueError("봇 이름이 필요합니다.")
    data = _load_bot_data()
    bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
    if not bot:
        raise ValueError(f"봇을 찾을 수 없음: {bot_name}")
    _backup_bot_json()
    # 임시/파생 필드는 저장 제외
    clean = dict(vn or {})
    clean.pop("emotion_rows", None)
    clean.pop("emotion_total", None)
    from modes.onnx_execution import normalize_cpu_threads, normalize_device_key
    clean["face_device"] = normalize_device_key(clean.get("face_device", "auto"))
    clean["face_cpu_threads"] = normalize_cpu_threads(
        clean.get("face_cpu_threads", 0)
    )
    bot["postprocess_vn"] = clean
    _save_bot_data(data)
    print(f"[BOT_MODE] postprocess_vn 저장: bot={bot_name}")


# ─── 후처리 봇별 설정 (postprocess_bubble: 말풍선 모드) ─────────────
def _load_postprocess_bubble(bot_name: str) -> dict:
    """봇의 postprocess_bubble 반환. 없으면 기본값."""
    if not bot_name:
        from modes.postprocess import _default_bubble
        return _default_bubble()
    try:
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
        if bot and isinstance(bot.get("postprocess_bubble"), dict):
            from modes.postprocess import _default_bubble
            base = _default_bubble()
            base.update(bot["postprocess_bubble"])
            return base
    except Exception as e:
        print(f"[BOT_MODE] postprocess_bubble 로드 실패({bot_name}): {e}")
        traceback.print_exc()
    from modes.postprocess import _default_bubble
    return _default_bubble()


def _save_postprocess_bubble(bot_name: str, bubble: dict):
    """봇의 postprocess_bubble 저장. 백업 후 bot.json 갱신."""
    if not bot_name:
        raise ValueError("봇 이름이 필요합니다.")
    data = _load_bot_data()
    bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
    if not bot:
        raise ValueError(f"봇을 찾을 수 없음: {bot_name}")
    _backup_bot_json()
    from modes.postprocess import normalize_layout_font_scale
    from modes.onnx_execution import normalize_cpu_threads, normalize_device_key
    clean = dict(bubble or {})
    # 폐기된 말풍선 옵션은 기존 클라이언트가 보내더라도 다시 저장하지 않는다.
    clean.pop("tail_len", None)
    clean.pop("conf", None)
    clean["layout_font_scale"] = normalize_layout_font_scale(
        clean.get("layout_font_scale", 2.0)
    )
    # 자간/행간/가로축소/폰트id 정규화(범위 클램프)
    try:
        clean["letter_spacing"] = max(-0.10, min(0.05, float(clean.get("letter_spacing", -0.03))))
    except (TypeError, ValueError):
        clean["letter_spacing"] = -0.03
    try:
        clean["line_height_ratio"] = max(1.0, min(1.40, float(clean.get("line_height_ratio", 1.15))))
    except (TypeError, ValueError):
        clean["line_height_ratio"] = 1.15
    try:
        clean["text_width_scale"] = max(0.70, min(1.00, float(clean.get("text_width_scale", 1.0))))
    except (TypeError, ValueError):
        clean["text_width_scale"] = 1.0
    clean["font_id"] = str(clean.get("font_id", "system") or "system")
    clean["onnx_device"] = normalize_device_key(clean.get("onnx_device", "auto"))
    clean["cpu_threads"] = normalize_cpu_threads(clean.get("cpu_threads", 0))
    bot["postprocess_bubble"] = clean
    _save_bot_data(data)
    print(f"[BOT_MODE] postprocess_bubble 저장: bot={bot_name}")


def _get_postprocess_mode(bot_name: str) -> str:
    """봇의 활성 후처리 모드: 'vn' | 'bubble'. 기본 'vn' (기존 동작 유지)."""
    if not bot_name:
        return "vn"
    try:
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
        if bot:
            m = bot.get("postprocess_mode")
            if m in ("vn", "bubble"):
                return m
    except Exception as e:
        print(f"[BOT_MODE] postprocess_mode 로드 실패({bot_name}): {e}")
    return "vn"


def _set_postprocess_mode(bot_name: str, mode: str):
    """봇의 활성 후처리 모드 저장('vn' | 'bubble'). 백업 후 갱신."""
    if not bot_name:
        raise ValueError("봇 이름이 필요합니다.")
    if mode not in ("vn", "bubble"):
        raise ValueError(f"잘못된 postprocess_mode: {mode}")
    data = _load_bot_data()
    bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
    if not bot:
        raise ValueError(f"봇을 찾을 수 없음: {bot_name}")
    _backup_bot_json()
    bot["postprocess_mode"] = mode
    _save_bot_data(data)
    print(f"[BOT_MODE] postprocess_mode 저장: bot={bot_name} → {mode}")


def _migrate_postprocess_vn(data: dict):
    """config.json postprocess.vn → 각 봇의 postprocess_vn (1회)."""
    if data.get("_postprocess_migrated"):
        return
    cfg_path = os.path.join(BASE_DIR, "config.json")
    old_vn = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            old_vn = (cfg.get("postprocess") or {}).get("vn") or {}
        except Exception as e:
            print(f"[BOT_MODE] config.json 로드 실패(마이그레이션): {e}")
    if old_vn:
        _backup_bot_json()
        from modes.postprocess import _default_vn
        changed = False
        for bot in data.get("bots", []):
            if not isinstance(bot.get("postprocess_vn"), dict):
                merged = _default_vn()
                merged.update(old_vn)
                bot["postprocess_vn"] = merged
                changed = True
                print(f"[BOT_MODE] 마이그레이션: postprocess_vn 할당 ({bot.get('name')})")
        if changed:
            _save_bot_data(data)
    data["_postprocess_migrated"] = True



def build_utility_prompt(bot_name: str, char_name: str, settings: dict) -> str:
    """캐릭터의 유틸리티 프롬프트 문자열을 생성한다."""
    emb_value = "representation" if settings.get("emb_target") == "대표만" else "representation,sub"
    return (
        f"[PATH]\nsoya_bot/{bot_name}/{char_name}\n"
        f"[FACE_CROP_TOP]\n{settings.get('face_crop_top', 1.0)}\n"
        f"[FACE_CROP_BOTTOM]\n{settings.get('face_crop_bottom', 1.0)}\n"
        f"[EMB_TARGET]\n{emb_value}\n"
        f"[END]"
    )


class BotDataPatcher:
    """Comfy Input /soya_bot/ 폴더에 봇 데이터 패치 + 유틸리티 워크플로우 실행"""

    def __init__(self):
        self._workflow_api = None
        self._workflow_hash = None
        self._program_embedding_previews = {}
        self._program_embedding_preview_lock = threading.RLock()
        self._program_embedding_preview_root = os.path.join(
            BASE_DIR, "current_work", "program_embedding_previews"
        )

    async def _load_utility_workflow(self) -> tuple[dict | None, str | None]:
        """utility_workflow_source_path에서 워크플로우를 로드한다.
        이미 API 형식이면 그대로 사용, 아니면 ComfyUI /workflow/convert로 변환.
        반환: (workflow_api_dict, error_msg)"""
        config_path = os.path.join(BASE_DIR, "config.json")
        if not os.path.isfile(config_path):
            return None, "config.json이 없습니다."
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        wf_path = config.get("utility_workflow_source_path", "").strip()
        if not wf_path:
            return None, "삽화 유틸리티 워크플로우 경로가 설정되지 않았습니다."
        if not os.path.isfile(wf_path):
            return None, f"유틸리티 워크플로우 파일이 없습니다: {wf_path}"

        import hashlib
        with open(wf_path, "r", encoding="utf-8") as f:
            raw = f.read()
        current_hash = hashlib.md5(raw.encode()).hexdigest()

        if self._workflow_api and self._workflow_hash == current_hash:
            return self._workflow_api, None

        wf_json = json.loads(raw)
        # API 형식인지 확인 (최상위가 dict이고 값에 class_type이 있으면 API 형식)
        is_api = isinstance(wf_json, dict) and any(
            isinstance(v, dict) and "class_type" in v for v in wf_json.values()
        )
        if is_api:
            self._workflow_api = wf_json
        else:
            # ComfyUI /workflow/convert로 변환
            from server import convert_workflow_via_endpoint
            api_wf, conv_err = await convert_workflow_via_endpoint(wf_json)
            if conv_err:
                return None, f"워크플로우 변환 실패: {conv_err}"
            self._workflow_api = api_wf
            print(f"[UTILITY] 워크플로우 변환 완료: {len(api_wf)} 노드")

        self._workflow_hash = current_hash
        return self._workflow_api, None

    # ─── 프로그램용 FACE embedding ──────────────────────────
    @staticmethod
    def _program_embedding_float(body, key, default, minimum, maximum):
        raw = body.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as e:
            print(f"[PROGRAM_EMBEDDING] 숫자 변환 실패: {key}={raw!r}")
            raise ValueError(f"{key} 값이 숫자가 아닙니다: {raw!r}") from e
        if not math.isfinite(value) or value < minimum or value > maximum:
            print(
                f"[PROGRAM_EMBEDDING] 설정 범위 오류: {key}={value}, "
                f"허용={minimum}~{maximum}"
            )
            raise ValueError(f"{key} 값은 {minimum}~{maximum} 범위여야 합니다.")
        return value

    @staticmethod
    def _program_embedding_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _program_embedding_safe_component(value):
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(". ")
        return safe or "_"

    def _program_embedding_remove_dir(self, path):
        if not path or not os.path.isdir(path):
            return
        try:
            root = os.path.abspath(self._program_embedding_preview_root)
            target = os.path.abspath(path)
            if os.path.commonpath([root, target]) != root or target == root:
                print(f"[PROGRAM_EMBEDDING] 임시 폴더 삭제 거부(범위 밖): {target}")
                return
            shutil.rmtree(target)
            print(f"[PROGRAM_EMBEDDING] 임시 미리보기 삭제: {target}")
        except Exception as e:
            print(f"[PROGRAM_EMBEDDING] 임시 미리보기 삭제 실패({path}): {e}")
            traceback.print_exc()

    def _program_embedding_cleanup_expired(self):
        cutoff = time.time() - 30 * 60
        expired = []
        with self._program_embedding_preview_lock:
            for preview_id, session in self._program_embedding_previews.items():
                if float(session.get("created_at", 0)) < cutoff:
                    expired.append(preview_id)
            sessions = [
                self._program_embedding_previews.pop(preview_id)
                for preview_id in expired
            ]
        for session in sessions:
            print(f"[PROGRAM_EMBEDDING] 만료된 미리보기 정리: {session.get('preview_id')}")
            self._program_embedding_remove_dir(session.get("session_dir"))
        try:
            if not os.path.isdir(self._program_embedding_preview_root):
                return
            with self._program_embedding_preview_lock:
                active_dirs = {
                    os.path.abspath(session.get("session_dir", ""))
                    for session in self._program_embedding_previews.values()
                }
            for name in os.listdir(self._program_embedding_preview_root):
                path = os.path.join(self._program_embedding_preview_root, name)
                if not os.path.isdir(path) or os.path.abspath(path) in active_dirs:
                    continue
                if os.path.getmtime(path) < cutoff:
                    print(f"[PROGRAM_EMBEDDING] 재시작 전 만료 미리보기 정리: {path}")
                    self._program_embedding_remove_dir(path)
        except Exception as e:
            print(f"[PROGRAM_EMBEDDING] 만료 미리보기 스캔 실패: {e}")
            traceback.print_exc()

    def _program_embedding_take_session(self, preview_id):
        with self._program_embedding_preview_lock:
            return self._program_embedding_previews.pop(preview_id, None)

    def _program_embedding_get_session(self, preview_id):
        with self._program_embedding_preview_lock:
            return self._program_embedding_previews.get(preview_id)

    def _create_program_embedding_preview(self, body):
        """FACE 추출 결과를 임시 폴더에 만들고 저장 전 검토 세션을 반환한다."""
        self._program_embedding_cleanup_expired()
        bot_name = str(body.get("bot_name", "")).strip()
        raw_names = body.get("char_names", [])
        if not bot_name:
            raise ValueError("봇 이름이 비어있습니다.")
        if not isinstance(raw_names, list):
            print(f"[PROGRAM_EMBEDDING] char_names가 리스트가 아님: {type(raw_names)}")
            raise ValueError("char_names는 리스트여야 합니다.")
        char_names = []
        for raw_name in raw_names:
            name = str(raw_name).strip()
            if name and name not in char_names:
                char_names.append(name)
        if not char_names:
            raise ValueError("선택된 캐릭터가 없습니다.")

        crop_top = self._program_embedding_float(body, "crop_top", 1.0, 0.1, 10.0)
        crop_bottom = self._program_embedding_float(body, "crop_bottom", 1.0, 0.1, 10.0)
        confidence = self._program_embedding_float(body, "confidence", 0.3, 0.0, 1.0)
        overwrite = bool(body.get("overwrite", False))

        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
        if not bot:
            raise ValueError(f"봇을 찾을 수 없습니다: {bot_name}")
        char_by_name = {c.get("name"): c for c in bot.get("characters", [])}
        missing = [name for name in char_names if name not in char_by_name]
        if missing:
            print(f"[PROGRAM_EMBEDDING] 존재하지 않는 캐릭터: {missing}")
            raise ValueError(f"캐릭터를 찾을 수 없습니다: {', '.join(missing)}")

        preview_id = uuid.uuid4().hex
        session_dir = os.path.join(self._program_embedding_preview_root, preview_id)
        os.makedirs(session_dir, exist_ok=False)
        session_items = []
        response_items = []

        try:
            from PIL import Image
            from modes import face_detector

            for index, char_name in enumerate(char_names):
                char = char_by_name[char_name]
                char_dir = os.path.join(BOT_DIR, bot_name, char_name)
                face_path = os.path.join(char_dir, "_face_image.webp")
                existing_face = os.path.isfile(face_path)
                face_url = (
                    f"/api/bot_mode/image/{quote(bot_name, safe='')}/"
                    f"{quote(char_name, safe='')}/_face_image.webp"
                    if existing_face else ""
                )
                rep_images = char.get("rep_images") or []
                rep_name = str(rep_images[0]) if rep_images else ""
                rep_path = os.path.join(char_dir, rep_name) if rep_name else ""
                rep_exists = bool(rep_path and os.path.isfile(rep_path))
                rep_url = (
                    f"/api/bot_mode/image/{quote(bot_name, safe='')}/"
                    f"{quote(char_name, safe='')}/{quote(rep_name, safe='')}"
                    if rep_exists else ""
                )

                item = {
                    "char_name": char_name,
                    "face_path": None,
                    "confirmed_sha256": "",
                    "save_new_face": False,
                    "preview_path": "",
                }
                response = {
                    "char_name": char_name,
                    "status": "failed",
                    "source_label": "추출 실패",
                    "display_url": rep_url,
                    "existing_url": face_url,
                    "representative_url": rep_url,
                    "detected_confidence": None,
                    "can_continue": False,
                    "message": "",
                }

                if existing_face and not overwrite:
                    source_hash = self._program_embedding_hash(face_path)
                    item.update({
                        "face_path": face_path,
                        "confirmed_sha256": source_hash,
                    })
                    response.update({
                        "status": "existing",
                        "source_label": "기존 데이터 패치 FACE",
                        "display_url": face_url,
                        "can_continue": True,
                        "message": "기존 FACE를 유지하고 임베딩합니다.",
                    })
                    print(f"[PROGRAM_EMBEDDING] 기존 FACE 우선 사용: {bot_name}/{char_name}")
                elif not rep_exists:
                    response["message"] = "대표 이미지 파일이 없어 ONNX 얼굴 추출을 할 수 없습니다."
                    print(
                        f"[PROGRAM_EMBEDDING] 대표 이미지 없음: {bot_name}/{char_name}, "
                        f"rep={rep_name!r}"
                    )
                else:
                    try:
                        with Image.open(rep_path) as opened:
                            source_image = opened.convert("RGB")
                        crop, detected_confidence = face_detector.crop_face(
                            source_image,
                            top_mult=crop_top,
                            bottom_mult=crop_bottom,
                            target_size=512,
                            conf_thres=confidence,
                            device="auto",
                            return_conf=True,
                        )
                        response["detected_confidence"] = detected_confidence
                        if crop is None:
                            if existing_face:
                                source_hash = self._program_embedding_hash(face_path)
                                item.update({
                                    "face_path": face_path,
                                    "confirmed_sha256": source_hash,
                                })
                                response.update({
                                    "status": "failed_existing",
                                    "source_label": "추출 실패 · 기존 FACE 유지",
                                    "display_url": face_url,
                                    "can_continue": True,
                                    "message": (
                                        "설정 임계치에서 얼굴을 찾지 못해 기존 FACE를 유지합니다."
                                    ),
                                })
                            else:
                                response["message"] = "설정 임계치에서 얼굴을 찾지 못했습니다."
                            print(
                                f"[PROGRAM_EMBEDDING] 얼굴 추출 실패: {bot_name}/{char_name}, "
                                f"confidence={confidence}, 최고={detected_confidence}"
                            )
                        else:
                            preview_path = os.path.join(session_dir, f"{index}.webp")
                            crop.save(preview_path, format="WEBP", quality=95, method=6)
                            source_hash = self._program_embedding_hash(preview_path)
                            item.update({
                                "face_path": preview_path,
                                "confirmed_sha256": source_hash,
                                "save_new_face": True,
                                "preview_path": preview_path,
                            })
                            response.update({
                                "status": "extracted",
                                "source_label": "ONNX 추출 결과",
                                "display_url": (
                                    f"/api/bot_mode/program_embedding/preview_image/"
                                    f"{preview_id}/{index}"
                                ),
                                "can_continue": True,
                                "message": "계속을 누르면 이 이미지를 FACE로 저장합니다.",
                            })
                            print(
                                f"[PROGRAM_EMBEDDING] ONNX 미리보기 생성: "
                                f"{bot_name}/{char_name}, conf={detected_confidence}"
                            )
                    except Exception as e:
                        print(f"[PROGRAM_EMBEDDING] ONNX 추출 예외({bot_name}/{char_name}): {e}")
                        traceback.print_exc()
                        if existing_face:
                            source_hash = self._program_embedding_hash(face_path)
                            item.update({
                                "face_path": face_path,
                                "confirmed_sha256": source_hash,
                            })
                            response.update({
                                "status": "failed_existing",
                                "source_label": "추출 실패 · 기존 FACE 유지",
                                "display_url": face_url,
                                "can_continue": True,
                                "message": f"ONNX 추출 오류로 기존 FACE를 유지합니다: {e}",
                            })
                        else:
                            response["message"] = f"ONNX 얼굴 추출 오류: {e}"

                session_items.append(item)
                response_items.append(response)

            session = {
                "preview_id": preview_id,
                "created_at": time.time(),
                "session_dir": session_dir,
                "bot_name": bot_name,
                "settings": {
                    "crop_top": crop_top,
                    "crop_bottom": crop_bottom,
                    "confidence": confidence,
                    "overwrite": overwrite,
                },
                "items": session_items,
            }
            with self._program_embedding_preview_lock:
                self._program_embedding_previews[preview_id] = session

            ready_count = sum(1 for item in response_items if item["can_continue"])
            extracted_count = sum(1 for item in response_items if item["status"] == "extracted")
            failed_count = sum(1 for item in response_items if item["status"] == "failed")
            return {
                "success": True,
                "preview_id": preview_id,
                "items": response_items,
                "ready_count": ready_count,
                "extracted_count": extracted_count,
                "failed_count": failed_count,
                "settings": session["settings"],
            }
        except Exception:
            self._program_embedding_remove_dir(session_dir)
            raise

    def _program_embedding_backup_file(self, source_path, backup_char_dir):
        if not os.path.isfile(source_path):
            return ""
        os.makedirs(backup_char_dir, exist_ok=True)
        backup_path = os.path.join(backup_char_dir, os.path.basename(source_path))
        shutil.copy2(source_path, backup_path)
        print(f"[PROGRAM_EMBEDDING] 기존 파일 백업: {source_path} → {backup_path}")
        return backup_path

    def _commit_program_embedding_preview(self, preview_id):
        """검토가 끝난 FACE를 저장하고 선택 캐릭터 임베딩을 생성한다."""
        session = self._program_embedding_take_session(preview_id)
        if not session:
            print(f"[PROGRAM_EMBEDDING] 확정할 미리보기 세션 없음: {preview_id}")
            raise ValueError("미리보기 세션이 없거나 만료되었습니다. 다시 미리보기를 생성하세요.")

        bot_name = session["bot_name"]
        backup_root = os.path.join(
            BASE_DIR,
            "요구사항",
            f"program_embedding_backup_{time.strftime('%Y%m%d_%H%M%S')}_{preview_id[:8]}",
        )
        results = []
        success_count = 0
        face_saved_count = 0
        failed_count = 0

        try:
            from modes import face_embedder

            for item in session["items"]:
                char_name = item["char_name"]
                source_path = item.get("face_path") or ""
                if not source_path:
                    print(f"[PROGRAM_EMBEDDING] 확정 스킵(FACE 없음): {bot_name}/{char_name}")
                    results.append({
                        "char_name": char_name,
                        "success": False,
                        "message": "확정 가능한 FACE가 없습니다.",
                    })
                    failed_count += 1
                    continue

                char_dir = os.path.join(BOT_DIR, bot_name, char_name)
                face_path = os.path.join(char_dir, "_face_image.webp")
                prompt_path = os.path.join(char_dir, "_face_image_prompt.json")
                cache_path = os.path.join(char_dir, "_face_image.l14.npz")
                backup_char_dir = os.path.join(
                    backup_root,
                    self._program_embedding_safe_component(bot_name),
                    self._program_embedding_safe_component(char_name),
                )
                save_new_face = bool(item.get("save_new_face"))
                face_existed = os.path.isfile(face_path)
                prompt_existed = os.path.isfile(prompt_path)
                cache_existed = os.path.isfile(cache_path)
                face_backup = ""
                prompt_backup = ""
                tmp_face_path = f"{face_path}.tmp-{uuid.uuid4().hex}"

                try:
                    current_hash = self._program_embedding_hash(source_path)
                    if current_hash != item.get("confirmed_sha256"):
                        raise RuntimeError("미리보기 이후 FACE 파일이 변경되었습니다.")

                    built = face_embedder.build_embedding_from_path(source_path)
                    if built is None:
                        raise RuntimeError("ONNX 임베딩 생성에 실패했습니다.")
                    emb, source_hash = built

                    if save_new_face:
                        face_backup = self._program_embedding_backup_file(face_path, backup_char_dir)
                        prompt_backup = self._program_embedding_backup_file(prompt_path, backup_char_dir)
                        os.makedirs(char_dir, exist_ok=True)
                        shutil.copy2(source_path, tmp_face_path)
                        os.replace(tmp_face_path, face_path)
                        if prompt_existed:
                            os.remove(prompt_path)
                            print(f"[PROGRAM_EMBEDDING] 오래된 FACE 프롬프트 제거: {prompt_path}")
                        source_hash = self._program_embedding_hash(face_path)

                    saved = face_embedder.write_embedding_cache(
                        cache_path,
                        emb,
                        source_hash,
                        backup_dir=backup_char_dir,
                    )
                    if saved is None:
                        raise RuntimeError("임베딩 캐시 저장에 실패했습니다.")

                    success_count += 1
                    if save_new_face:
                        face_saved_count += 1
                    results.append({
                        "char_name": char_name,
                        "success": True,
                        "face_saved": save_new_face,
                        "message": (
                            "ONNX FACE 저장 + 임베딩 완료"
                            if save_new_face else "기존 FACE 임베딩 완료"
                        ),
                    })
                    print(
                        f"[PROGRAM_EMBEDDING] 확정 완료: {bot_name}/{char_name}, "
                        f"face_saved={save_new_face}"
                    )
                except Exception as e:
                    print(f"[PROGRAM_EMBEDDING] 확정 실패({bot_name}/{char_name}): {e}")
                    traceback.print_exc()
                    if os.path.isfile(tmp_face_path):
                        try:
                            os.remove(tmp_face_path)
                        except Exception as cleanup_error:
                            print(
                                f"[PROGRAM_EMBEDDING] 임시 FACE 삭제 실패({tmp_face_path}): "
                                f"{cleanup_error}"
                            )
                            traceback.print_exc()
                    if save_new_face:
                        try:
                            if face_existed and face_backup:
                                shutil.copy2(face_backup, face_path)
                            elif not face_existed and os.path.isfile(face_path):
                                os.remove(face_path)
                            if prompt_existed and prompt_backup:
                                shutil.copy2(prompt_backup, prompt_path)
                        except Exception as rollback_error:
                            print(
                                f"[PROGRAM_EMBEDDING] FACE 롤백 실패({bot_name}/{char_name}): "
                                f"{rollback_error}"
                            )
                            traceback.print_exc()
                    cache_backup = os.path.join(backup_char_dir, os.path.basename(cache_path))
                    try:
                        if cache_existed and os.path.isfile(cache_backup):
                            shutil.copy2(cache_backup, cache_path)
                        elif not cache_existed and os.path.isfile(cache_path):
                            os.remove(cache_path)
                    except Exception as rollback_error:
                        print(
                            f"[PROGRAM_EMBEDDING] 캐시 롤백 실패({bot_name}/{char_name}): "
                            f"{rollback_error}"
                        )
                        traceback.print_exc()
                    results.append({
                        "char_name": char_name,
                        "success": False,
                        "message": str(e),
                    })
                    failed_count += 1

            backup_created = os.path.isdir(backup_root)
            return {
                "success": success_count > 0,
                "message": (
                    f"임베딩 {success_count}건 완료, FACE 저장 {face_saved_count}건, "
                    f"실패 {failed_count}건"
                ),
                "success_count": success_count,
                "face_saved_count": face_saved_count,
                "failed_count": failed_count,
                "backup_dir": backup_root if backup_created else "",
                "results": results,
            }
        finally:
            self._program_embedding_remove_dir(session.get("session_dir"))

    async def handle_program_embedding_preview(self, request):
        try:
            body = await request.json()
            result = await asyncio.to_thread(self._create_program_embedding_preview, body)
            return _json_ok(result)
        except ValueError as e:
            print(f"[PROGRAM_EMBEDDING] 미리보기 요청 오류: {e}")
            return _json_error(str(e))
        except Exception as e:
            print(f"[PROGRAM_EMBEDDING] 미리보기 생성 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e), status=500)

    async def handle_program_embedding_preview_image(self, request):
        try:
            self._program_embedding_cleanup_expired()
            preview_id = request.match_info.get("preview_id", "")
            index_raw = request.match_info.get("index", "")
            session = self._program_embedding_get_session(preview_id)
            if not session:
                return _json_error("미리보기 세션이 없거나 만료되었습니다.", status=404)
            try:
                index = int(index_raw)
            except ValueError:
                print(f"[PROGRAM_EMBEDDING] 잘못된 미리보기 인덱스: {index_raw!r}")
                return _json_error("잘못된 미리보기 인덱스입니다.")
            if index < 0 or index >= len(session["items"]):
                print(f"[PROGRAM_EMBEDDING] 미리보기 인덱스 범위 오류: {index}")
                return _json_error("미리보기 인덱스 범위를 벗어났습니다.", status=404)
            path = session["items"][index].get("preview_path") or ""
            if not path or not os.path.isfile(path):
                print(f"[PROGRAM_EMBEDDING] 미리보기 이미지 없음: {preview_id}/{index}")
                return _json_error("미리보기 이미지를 찾을 수 없습니다.", status=404)
            return web.FileResponse(path, headers={"Content-Type": "image/webp"})
        except Exception as e:
            print(f"[PROGRAM_EMBEDDING] 미리보기 이미지 제공 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e), status=500)

    async def handle_program_embedding_commit(self, request):
        try:
            body = await request.json()
            preview_id = str(body.get("preview_id", "")).strip()
            if not preview_id:
                return _json_error("preview_id가 비어있습니다.")
            result = await asyncio.to_thread(
                self._commit_program_embedding_preview, preview_id
            )
            return _json_ok(result)
        except ValueError as e:
            print(f"[PROGRAM_EMBEDDING] 확정 요청 오류: {e}")
            return _json_error(str(e))
        except Exception as e:
            print(f"[PROGRAM_EMBEDDING] 확정 처리 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e), status=500)

    async def handle_program_embedding_cancel(self, request):
        try:
            body = await request.json()
            preview_id = str(body.get("preview_id", "")).strip()
            if not preview_id:
                return _json_error("preview_id가 비어있습니다.")
            session = self._program_embedding_take_session(preview_id)
            if not session:
                print(f"[PROGRAM_EMBEDDING] 취소할 세션 없음: {preview_id}")
                return _json_ok({"success": True, "message": "이미 정리된 미리보기입니다."})
            self._program_embedding_remove_dir(session.get("session_dir"))
            return _json_ok({"success": True, "message": "미리보기를 취소했습니다."})
        except Exception as e:
            print(f"[PROGRAM_EMBEDDING] 미리보기 취소 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e), status=500)

    async def handle_data_patch(self, request):
        """POST /api/bot_mode/data_patch - 선택된 봇의 캐릭터 폴더 + 대표 이미지를 soya_bot/에 복사"""
        try:
            body = await request.json()
            bot_name = body.get("bot_name", "").strip()
            char_name = body.get("char_name", "").strip()
            # 다중 선택 지원: char_names(리스트)가 있으면 우선, 없으면 단일 char_name
            char_names_raw = body.get("char_names", [])
            if not isinstance(char_names_raw, list):
                print(f"[DATA_PATCH] char_names가 리스트가 아님: {type(char_names_raw)}")
                char_names_raw = []
            char_names = [str(n).strip() for n in char_names_raw if str(n).strip()]
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")

            # config.json에서 comfy_input_dir 읽기
            config_path = os.path.join(BASE_DIR, "config.json")
            if not os.path.isfile(config_path):
                return _json_error("config.json이 없습니다.")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            comfy_input_dir = config.get("comfy_input_dir", "").strip()
            if not comfy_input_dir:
                return _json_error("Comfy Input 폴더 경로가 설정되지 않았습니다.")
            if not os.path.isdir(comfy_input_dir):
                return _json_error(f"Comfy Input 폴더가 존재하지 않습니다: {comfy_input_dir}")

            # 봇 데이터에서 해당 봇 찾기
            data = _load_bot_data()
            bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
            if not bot:
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")

            bot_dst_root = os.path.join(comfy_input_dir, "soya_bot", bot_name)
            # 요청된 캐릭터명 목록: char_names(다중) 우선, 없으면 단일 char_name
            requested_names = char_names if char_names else ([char_name] if char_name else [])
            selected_only = len(requested_names) > 0

            if selected_only:
                # 선택 캐릭터 모드: 봇 폴더 전체는 건드리지 않고,
                # 선택한 캐릭터 폴더만 삭제 후 재생성 (새 캐릭터 추가 시 기존 캐릭터 유지)
                target_chars = []
                missing = []
                for name in requested_names:
                    char = next((c for c in bot.get("characters", []) if c["name"] == name), None)
                    if not char:
                        missing.append(name)
                        print(f"[DATA_PATCH] 캐릭터를 찾을 수 없음(스킵): {name}")
                        continue
                    target_chars.append(char)
                if missing:
                    return _json_error(f"캐릭터를 찾을 수 없습니다: {', '.join(missing)}")
                if not target_chars:
                    return _json_error("선택된 캐릭터가 없습니다.")
                os.makedirs(bot_dst_root, exist_ok=True)
                for char in target_chars:
                    char_dst_dir = os.path.join(bot_dst_root, char["name"])
                    if os.path.isdir(char_dst_dir):
                        shutil.rmtree(char_dst_dir)
                        print(f"[DATA_PATCH] 기존 캐릭터 폴더 삭제: {char_dst_dir}")
            else:
                # 전체 모드: 기존 봇 폴더 삭제 후 재생성
                if os.path.isdir(bot_dst_root):
                    shutil.rmtree(bot_dst_root)
                    print(f"[DATA_PATCH] 기존 폴더 삭제: {bot_dst_root}")
                target_chars = bot.get("characters", [])

            created_dirs = []
            copied_files = []
            skipped_files = []

            for char in target_chars:
                char_name = char["name"]
                dst_dir = os.path.join(bot_dst_root, char_name)
                os.makedirs(dst_dir, exist_ok=True)
                created_dirs.append(f"soya_bot/{bot_name}/{char_name}")
                print(f"[DATA_PATCH] 폴더 생성: {dst_dir}")

                rep_images = char.get("rep_images", [])
                for i, img_name in enumerate(rep_images):
                    src_file = os.path.join(BOT_DIR, bot_name, char_name, img_name)
                    if not os.path.isfile(src_file):
                        skipped_files.append(img_name)
                        print(f"[DATA_PATCH] 소스 파일 없음: {src_file}")
                        continue

                    # 첫 번째 이미지: representation, 이후: sub_1, sub_2
                    ext = os.path.splitext(img_name)[1]
                    if i == 0:
                        dst_name = f"representation{ext}"
                    else:
                        dst_name = f"sub_{i}{ext}"

                    dst_file = os.path.join(dst_dir, dst_name)
                    shutil.copy2(src_file, dst_file)
                    copied_files.append(f"{char_name}/{dst_name}")
                    print(f"[DATA_PATCH] 복사: {img_name} -> {dst_name}")

            msg = f"폴더 {len(created_dirs)}개 생성, 이미지 {len(copied_files)}개 복사"
            if skipped_files:
                msg += f", 스킵 {len(skipped_files)}개"
            print(f"[DATA_PATCH] 완료: {msg}")
            return _json_ok({
                "message": msg,
                "created_dirs": created_dirs,
                "copied_files": copied_files,
                "skipped_files": skipped_files
            })
        except Exception as e:
            print(f"[DATA_PATCH] 데이터 패치 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_run_utility(self, request):
        """POST /api/bot_mode/run_utility - 단일 캐릭터 유틸리티 워크플로우 실행"""
        try:
            body = await request.json()
            bot_name = body.get("bot_name", "").strip()
            char_name = body.get("char_name", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")
            if not char_name:
                return _json_error("캐릭터 이름이 비어있습니다.")

            # 기존 유틸리티 결과 + 프롬프트 삭제
            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            result_path = os.path.join(char_dir, "_face_image.webp")
            for old in ["_face_image.webp", "_face_image_prompt.json"]:
                old_path = os.path.join(char_dir, old)
                if os.path.isfile(old_path):
                    os.remove(old_path)
                    print(f"[UTILITY] 기존 파일 삭제: {old_path}")

            # 워크플로우 로드
            wf_api, wf_err = await self._load_utility_workflow()
            if wf_err:
                return _json_error(wf_err)

            # 봇 데이터 로드
            data = _load_bot_data()
            bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
            if not bot:
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")
            char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
            if not char:
                return _json_error(f"캐릭터를 찾을 수 없습니다: {char_name}")
            if not char.get("rep_images"):
                return _json_error(f"대표 이미지가 없는 캐릭터입니다: {char_name}")

            # 설정 로드 (봇별 패치 설정 사용)
            settings = _load_patch_settings(bot_name)
            prompt_text = build_utility_prompt(bot_name, char_name, settings)
            print(f"[UTILITY] 실행: {char_name} | 프롬프트:\n{prompt_text}")

            # 워크플로우에 프롬프트 주입
            import copy
            wf = copy.deepcopy(wf_api)
            for nid, ninfo in wf.items():
                if not isinstance(ninfo, dict):
                    continue
                title = ninfo.get("_meta", {}).get("title", "")
                if title == "긍정프롬프트":
                    ninfo["inputs"]["value"] = prompt_text

            # ComfyUI에 제출
            from server import submit_workflow_to_comfy
            img_bytes, submit_err = await submit_workflow_to_comfy(wf)
            if submit_err or not img_bytes:
                return _json_error(f"{char_name}: {submit_err or '이미지 없음'}")

            # 결과 이미지 저장
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "wb") as f:
                f.write(img_bytes)
            print(f"[UTILITY] {char_name} 결과 저장: {result_path} ({len(img_bytes):,} bytes)")
            return _json_ok({"character": char_name, "message": f"{char_name} 완료"})
        except Exception as e:
            print(f"[UTILITY] 실행 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_check_patch_files(self, request):
        """GET /api/bot_mode/check_patch_files - 각 캐릭터의 ipadpt/pt 파일 존재 여부 확인"""
        try:
            bot_name = request.query.get("bot_name", "").strip()
            print(f"[CHECK_PATCH] 요청: bot_name={bot_name}")
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")

            config_path = os.path.join(BASE_DIR, "config.json")
            if not os.path.isfile(config_path):
                return _json_error("config.json이 없습니다.")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            comfy_input_dir = config.get("comfy_input_dir", "").strip()
            print(f"[CHECK_PATCH] comfy_input_dir={comfy_input_dir}")
            if not comfy_input_dir or not os.path.isdir(comfy_input_dir):
                return _json_error("Comfy Input 폴더 경로가 설정되지 않았습니다.")

            data = _load_bot_data()
            bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
            if not bot:
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")

            results = {}
            for char in bot.get("characters", []):
                char_name = char["name"]
                char_dir = os.path.join(comfy_input_dir, "soya_bot", bot_name, char_name)
                has_ipadpt = False
                has_pt = False
                print(f"[CHECK_PATCH] 검사: {char_dir} (exists={os.path.isdir(char_dir)})")
                if os.path.isdir(char_dir):
                    files = os.listdir(char_dir)
                    print(f"[CHECK_PATCH] 파일 목록: {files}")
                    for f in files:
                        if f.endswith(".ipadpt"):
                            has_ipadpt = True
                        if f.endswith(".pt"):
                            has_pt = True
                results[char_name] = {"ipadpt": has_ipadpt, "pt": has_pt}
                print(f"[CHECK_PATCH] {char_name}: ipadpt={has_ipadpt}, pt={has_pt}")
            return _json_ok(results)
        except Exception as e:
            print(f"[CHECK_PATCH] 확인 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

# ─── 삽화 모드 설정 (봇별) ──────────────────────────────

DEFAULT_ILLUST_SETTINGS = {
    "anima_artist_preset": "",
    "sdxl_artist_preset": "",
    "anima_quality_preset": "",
    "sdxl_quality_preset": "",
    "anima_negative_preset": "",
    "sdxl_negative_preset": "",
    "hrf_activate": False,
    "hrf_size": 1.5,
    "hrf_restore_size": False,
    "anima_hrf_activate": False,
    "img_w": 756,
    "img_h": 756,
    "anima_fd_activate": False,
    "anima_hd_activate": False,
    "anima_ed_activate": False,
    "fd_activate": False,
    "hd_activate": False,
    "ed_activate": False,
    "face_id_activate": False,
    "face_id_str": 0.55,
    "seed": -1,
    "face_lora_upscale_size": "",
}


async def handle_get_illust_settings(request):
    """GET /api/bot_mode/illust_settings - 봇의 삽화 설정 반환"""
    try:
        bot_name = request.query.get("bot_name", "").strip()
        profile = request.query.get("profile", "solo").strip()
        if profile not in ("solo", "group"):
            profile = "solo"
        if not bot_name:
            return _json_error("봇 이름이 비어있습니다.")
        data = _load_bot_data()
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")
        key = f"illust_settings_{profile}"
        settings = bot.get(key, bot.get("illust_settings", DEFAULT_ILLUST_SETTINGS))
        return _json_ok(settings)
    except Exception as e:
        print(f"[BOT_MODE] 삽화 설정 조회 실패: {e}")
        traceback.print_exc()
        return _json_error(str(e))


async def handle_update_illust_settings(request):
    """POST /api/bot_mode/update_illust_settings - 봇의 삽화 설정 업데이트"""
    try:
        body = await request.json()
        bot_name = body.get("bot_name", "").strip()
        profile = body.get("profile", "solo")
        if profile not in ("solo", "group"):
            profile = "solo"
        if not bot_name:
            return _json_error("봇 이름이 비어있습니다.")
        new_settings = body.get("illust_settings", {})
        data = _load_bot_data()
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")
        # 기존 설정과 병합
        key = f"illust_settings_{profile}"
        current = bot.get(key, bot.get("illust_settings", DEFAULT_ILLUST_SETTINGS))
        merged = {**DEFAULT_ILLUST_SETTINGS, **current, **new_settings}
        bot[key] = merged
        _save_bot_data(data)
        print(f"[BOT_MODE] 삽화 설정 업데이트: {bot_name} [{profile}]")
        return _json_ok(merged)
    except Exception as e:
        print(f"[BOT_MODE] 삽화 설정 업데이트 실패: {e}")
        traceback.print_exc()
        return _json_error(str(e))


async def handle_get_positive_rules(request):
    """GET /api/bot_mode/positive_rules - POSITIVE 화이트리스트/블랙리스트 규칙 반환"""
    try:
        data = _load_bot_data()
        return _json_ok({
            "positive_whitelist": data.get("positive_whitelist", []),
            "positive_blacklist": data.get("positive_blacklist", []),
        })
    except Exception as e:
        print(f"[BOT_MODE] POSITIVE 규칙 조회 실패: {e}")
        traceback.print_exc()
        return _json_error(str(e))


async def handle_save_positive_rules(request):
    """POST /api/bot_mode/positive_rules - POSITIVE 화이트리스트/블랙리스트 규칙 저장"""
    try:
        body = await request.json()
        whitelist = body.get("positive_whitelist", None)
        blacklist = body.get("positive_blacklist", None)

        data = _load_bot_data()

        if whitelist is not None:
            if not isinstance(whitelist, list):
                return _json_error("positive_whitelist must be a list")
            for item in whitelist:
                if not isinstance(item, str):
                    return _json_error("each whitelist item must be a string")
            data["positive_whitelist"] = whitelist

        if blacklist is not None:
            if not isinstance(blacklist, list):
                return _json_error("positive_blacklist must be a list")
            for item in blacklist:
                if not isinstance(item, str):
                    return _json_error("each blacklist item must be a string")
            data["positive_blacklist"] = blacklist

        _save_bot_data(data)
        print(f"[BOT_MODE] POSITIVE 규칙 저장: whitelist={len(data.get('positive_whitelist', []))}, blacklist={len(data.get('positive_blacklist', []))}")
        return _json_ok({
            "success": True,
            "positive_whitelist": data.get("positive_whitelist", []),
            "positive_blacklist": data.get("positive_blacklist", []),
        })
    except Exception as e:
        print(f"[BOT_MODE] POSITIVE 규칙 저장 실패: {e}")
        traceback.print_exc()
        return _json_error(str(e))


# ─── LLM 자동 얼굴/눈 태그 분류 (auto_face_tag) ─────────────

AUTO_FACE_TAG_PROMPTS_DIR = os.path.join(BASE_DIR, "prompts", "auto_face_tag")
AUTO_FACE_TAG_BUILTIN_FILE = os.path.join(AUTO_FACE_TAG_PROMPTS_DIR, "system.txt")
AUTO_FACE_TAG_CUSTOM_FILE = os.path.join(ASSET_DATA_DIR, "auto_face_tag_custom.txt")
AUTO_FACE_TAG_META_FILE = os.path.join(ASSET_DATA_DIR, "auto_face_tag_meta.json")

_auto_face_tag_builtin_cache: str | None = None
_auto_face_tag_builtin_mtime: float = 0.0


def _load_auto_face_tag_builtin() -> str:
    """글로벌(배포용) 프롬프트 로드. mtime 기반 캐싱."""
    global _auto_face_tag_builtin_cache, _auto_face_tag_builtin_mtime
    if not os.path.isfile(AUTO_FACE_TAG_BUILTIN_FILE):
        print(f"[BOT_MODE] auto_face_tag builtin 파일 없음: {AUTO_FACE_TAG_BUILTIN_FILE}")
        return ""
    try:
        mtime = os.path.getmtime(AUTO_FACE_TAG_BUILTIN_FILE)
        if _auto_face_tag_builtin_cache is not None and mtime == _auto_face_tag_builtin_mtime:
            return _auto_face_tag_builtin_cache
        with open(AUTO_FACE_TAG_BUILTIN_FILE, "r", encoding="utf-8") as f:
            txt = f.read()
        _auto_face_tag_builtin_cache = txt
        _auto_face_tag_builtin_mtime = mtime
        return txt
    except Exception as e:
        print(f"[BOT_MODE] auto_face_tag builtin 로드 실패: {e}")
        traceback.print_exc()
        return ""


def _load_auto_face_tag_custom() -> tuple[str, bool]:
    """커스텀 프롬프트와 use_custom 플래그 로드. (없으면 빈 문자열, False)."""
    custom = ""
    if os.path.isfile(AUTO_FACE_TAG_CUSTOM_FILE):
        try:
            with open(AUTO_FACE_TAG_CUSTOM_FILE, "r", encoding="utf-8") as f:
                custom = f.read()
        except Exception as e:
            print(f"[BOT_MODE] auto_face_tag custom 로드 실패: {e}")
            traceback.print_exc()

    use_custom = False
    if os.path.isfile(AUTO_FACE_TAG_META_FILE):
        try:
            with open(AUTO_FACE_TAG_META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                use_custom = bool(meta.get("use_custom", False))
        except Exception as e:
            print(f"[BOT_MODE] auto_face_tag meta 로드 실패: {e}")
            traceback.print_exc()

    return custom, use_custom


def _save_auto_face_tag_custom(text: str, use_custom: bool) -> None:
    """커스텀 프롬프트 저장. 기존 파일은 .bak 로 백업."""
    os.makedirs(ASSET_DATA_DIR, exist_ok=True)

    if os.path.isfile(AUTO_FACE_TAG_CUSTOM_FILE):
        try:
            shutil.copy2(AUTO_FACE_TAG_CUSTOM_FILE, AUTO_FACE_TAG_CUSTOM_FILE + ".bak")
        except Exception as e:
            print(f"[BOT_MODE] auto_face_tag custom 백업 실패: {e}")

    try:
        with open(AUTO_FACE_TAG_CUSTOM_FILE, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[BOT_MODE] auto_face_tag custom 저장 실패: {e}")
        traceback.print_exc()
        raise

    if os.path.isfile(AUTO_FACE_TAG_META_FILE):
        try:
            shutil.copy2(AUTO_FACE_TAG_META_FILE, AUTO_FACE_TAG_META_FILE + ".bak")
        except Exception as e:
            print(f"[BOT_MODE] auto_face_tag meta 백업 실패: {e}")

    try:
        with open(AUTO_FACE_TAG_META_FILE, "w", encoding="utf-8") as f:
            json.dump({"use_custom": bool(use_custom)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[BOT_MODE] auto_face_tag meta 저장 실패: {e}")
        traceback.print_exc()
        raise


# ─── lb-xnai.lb.extra.txt Appearance/Outfit 정제 프롬프트 ────────────
LB_EXTRA_REFINE_BUILTIN_FILE = os.path.join(BASE_DIR, "prompts", "lb_extra_refine", "system.txt")
LB_EXTRA_REFINE_CUSTOM_FILE = os.path.join(ASSET_DATA_DIR, "lb_extra_refine_custom.txt")
LB_EXTRA_REFINE_META_FILE = os.path.join(ASSET_DATA_DIR, "lb_extra_refine_meta.json")

_lb_extra_refine_builtin_cache: str | None = None
_lb_extra_refine_builtin_mtime: float = 0.0


def _load_lb_extra_refine_builtin() -> str:
    """글로벌(배포용) 정제 프롬프트 로드. mtime 기반 캐싱."""
    global _lb_extra_refine_builtin_cache, _lb_extra_refine_builtin_mtime
    if not os.path.isfile(LB_EXTRA_REFINE_BUILTIN_FILE):
        print(f"[BOT_MODE] lb_extra_refine builtin 파일 없음: {LB_EXTRA_REFINE_BUILTIN_FILE}")
        return ""
    try:
        mtime = os.path.getmtime(LB_EXTRA_REFINE_BUILTIN_FILE)
        if _lb_extra_refine_builtin_cache is not None and mtime == _lb_extra_refine_builtin_mtime:
            return _lb_extra_refine_builtin_cache
        with open(LB_EXTRA_REFINE_BUILTIN_FILE, "r", encoding="utf-8") as f:
            txt = f.read()
        _lb_extra_refine_builtin_cache = txt
        _lb_extra_refine_builtin_mtime = mtime
        return txt
    except Exception as e:
        print(f"[BOT_MODE] lb_extra_refine builtin 로드 실패: {e}")
        traceback.print_exc()
        return ""


def _load_lb_extra_refine_custom() -> tuple[str, bool]:
    """커스텀 정제 프롬프트와 use_custom 플래그 로드. (없으면 빈 문자열, False)."""
    custom = ""
    if os.path.isfile(LB_EXTRA_REFINE_CUSTOM_FILE):
        try:
            with open(LB_EXTRA_REFINE_CUSTOM_FILE, "r", encoding="utf-8") as f:
                custom = f.read()
        except Exception as e:
            print(f"[BOT_MODE] lb_extra_refine custom 로드 실패: {e}")
            traceback.print_exc()

    use_custom = False
    if os.path.isfile(LB_EXTRA_REFINE_META_FILE):
        try:
            with open(LB_EXTRA_REFINE_META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                use_custom = bool(meta.get("use_custom", False))
        except Exception as e:
            print(f"[BOT_MODE] lb_extra_refine meta 로드 실패: {e}")
            traceback.print_exc()

    return custom, use_custom


def _save_lb_extra_refine_custom(text: str, use_custom: bool) -> None:
    """커스텀 정제 프롬프트 저장. 기존 파일은 .bak 로 백업."""
    os.makedirs(ASSET_DATA_DIR, exist_ok=True)

    if os.path.isfile(LB_EXTRA_REFINE_CUSTOM_FILE):
        try:
            shutil.copy2(LB_EXTRA_REFINE_CUSTOM_FILE, LB_EXTRA_REFINE_CUSTOM_FILE + ".bak")
        except Exception as e:
            print(f"[BOT_MODE] lb_extra_refine custom 백업 실패: {e}")

    try:
        with open(LB_EXTRA_REFINE_CUSTOM_FILE, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[BOT_MODE] lb_extra_refine custom 저장 실패: {e}")
        traceback.print_exc()
        raise

    if os.path.isfile(LB_EXTRA_REFINE_META_FILE):
        try:
            shutil.copy2(LB_EXTRA_REFINE_META_FILE, LB_EXTRA_REFINE_META_FILE + ".bak")
        except Exception as e:
            print(f"[BOT_MODE] lb_extra_refine meta 백업 실패: {e}")

    try:
        with open(LB_EXTRA_REFINE_META_FILE, "w", encoding="utf-8") as f:
            json.dump({"use_custom": bool(use_custom)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[BOT_MODE] lb_extra_refine meta 저장 실패: {e}")
        traceback.print_exc()
        raise


def _render_lb_extra_refine_prompt(template: str, appearance_tags: list, outfit_tags: list, etc_tags: list) -> str:
    """template 의 {Appearance}/{outfit}/{etc} 변수를 태그 문자열로 치환.

    format() 충돌을 피하기 위해 단순 str.replace 사용.
    각 태그는 가중치 구문 제거 후 ", " 로 결합.
    """
    appearance_str = ", ".join(_strip_tag_wrapper(str(t)) for t in (appearance_tags or []) if str(t).strip())
    outfit_str = ", ".join(_strip_tag_wrapper(str(t)) for t in (outfit_tags or []) if str(t).strip())
    etc_str = ", ".join(_strip_tag_wrapper(str(t)) for t in (etc_tags or []) if str(t).strip())

    rendered = template.replace("{Appearance}", appearance_str)
    rendered = rendered.replace("{outfit}", outfit_str)
    rendered = rendered.replace("{etc}", etc_str)
    return rendered


def _parse_lb_extra_refine_response(raw: str) -> dict | None:
    """LLM 응답에서 {"appearance": [...], "outfit": [...]} 추출. 실패 시 None."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            appearance = data.get("appearance", [])
            outfit = data.get("outfit", [])
            if not (isinstance(appearance, list) and isinstance(outfit, list)):
                return None
            return {
                "appearance": [str(t).strip() for t in appearance if str(t).strip()],
                "outfit": [str(t).strip() for t in outfit if str(t).strip()],
            }
    except json.JSONDecodeError:
        pass
    # fallback: 첫 {...} 블록 추출
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                appearance = data.get("appearance", [])
                outfit = data.get("outfit", [])
                return {
                    "appearance": [str(t).strip() for t in (appearance if isinstance(appearance, list) else []) if str(t).strip()],
                    "outfit": [str(t).strip() for t in (outfit if isinstance(outfit, list) else []) if str(t).strip()],
                }
        except json.JSONDecodeError as e:
            print(f"[BOT_MODE] lb_extra_refine JSON 파싱 실패: {e}")
    return None


def _strip_tag_wrapper(raw: str) -> str:
    """'(tag:1.2)' / '[tag]' / '{tag}' 같은 가중치 구문에서 순수 태그명만 추출."""
    s = raw.strip()
    m = re.search(r'[\(\[{][\d.]*:?\s*([^()\[\]{}]+)[\)\]}]', s)
    if m:
        s = m.group(1).strip()
    # 가중치 문법 "tag:1.2"에서 우측이 숫자면 제거
    if ':' in s:
        left, right = s.split(':', 1)
        if right.strip().replace('.', '').isdigit():
            s = left.strip()
    return s


def _render_auto_face_tag_prompt(template: str, groups: dict) -> str:
    """template 의 {appearance}/{attire}/{etc} 변수를 그룹 태그로 치환.

    format() 충돌을 피하기 위해 단순 str.replace 사용.
    """
    appearance_tags = ", ".join(_strip_tag_wrapper(t["tag"]) for t in groups.get("외모/신체", []))
    attire_tags = ", ".join(_strip_tag_wrapper(t["tag"]) for t in groups.get("복장", []))
    etc_tags = ", ".join(_strip_tag_wrapper(t["tag"]) for t in groups.get("미분류", []))

    rendered = template.replace("{appearance}", appearance_tags)
    rendered = rendered.replace("{attire}", attire_tags)
    rendered = rendered.replace("{etc}", etc_tags)
    return rendered


def _parse_auto_face_tag_response(raw: str) -> dict | None:
    """LLM 응답에서 JSON 객체 추출. 실패 시 None."""
    if not raw:
        return None
    cleaned = raw.strip()
    # markdown ```json ... ``` 제거
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            face = data.get("face")
            eye = data.get("eye")
            if isinstance(face, list) and isinstance(eye, list):
                return {"face": [str(t).strip() for t in face if str(t).strip()],
                        "eye": [str(t).strip() for t in eye if str(t).strip()]}
    except json.JSONDecodeError:
        pass
    # fallback: 첫 {...} 블록 추출
    m = re.search(r'\{[^{}]*"face"[^{}]*\}', cleaned, re.DOTALL)
    if not m:
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                face = data.get("face", [])
                eye = data.get("eye", [])
                return {"face": [str(t).strip() for t in (face if isinstance(face, list) else []) if str(t).strip()],
                        "eye": [str(t).strip() for t in (eye if isinstance(eye, list) else []) if str(t).strip()]}
        except json.JSONDecodeError as e:
            print(f"[BOT_MODE] auto_face_tag JSON 파싱 실패: {e}")
    return None


async def handle_get_auto_face_tag_prompt(request):
    """GET /api/bot_mode/auto_face_tag_prompt - 글로벌/커스텀 프롬프트 조회."""
    try:
        builtin = _load_auto_face_tag_builtin()
        custom, use_custom = _load_auto_face_tag_custom()
        return web.json_response({
            "success": True,
            "data": {
                "builtin": builtin,
                "custom": custom,
                "use_custom": use_custom,
            },
        })
    except Exception as e:
        print(f"[BOT_MODE] auto_face_tag_prompt 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_get_auto_face_tag_test_image(request):
    """GET /api/bot_mode/auto_face_tag_test_image - 배포 번들 테스트 이미지(base64 data URL)."""
    import base64
    path = os.path.join(AUTO_FACE_TAG_PROMPTS_DIR, "test_img.webp")
    if not os.path.isfile(path):
        print(f"[BOT_MODE] 테스트 이미지 없음: {path}")
        return web.json_response({"success": False, "error": f"테스트 이미지가 없습니다: {path}"})
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return web.json_response({
            "success": True,
            "data": {
                "image_b64": b64,
                "mime": "image/webp",
                "data_url": f"data:image/webp;base64,{b64}",
            },
        })
    except Exception as e:
        print(f"[BOT_MODE] 테스트 이미지 로드 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_set_auto_face_tag_prompt(request):
    """POST /api/bot_mode/auto_face_tag_prompt - 커스텀 프롬프트 저장."""
    try:
        body = await request.json()
        custom = body.get("custom", "") or ""
        use_custom = bool(body.get("use_custom", False))
        _save_auto_face_tag_custom(custom, use_custom)
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[BOT_MODE] auto_face_tag_prompt 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def run_auto_classify_face_tags(bot_name: str, char_name: str) -> dict:
    """LLM 비전 기반 얼굴/눈 태그 자동 분류 (HTTP 래퍼 없는 core).
    반환: {"success": True, "data": {"face": [...], "eye": [...]}} 또는 {"success": False, "error": "..."}
    """
    import base64
    import time as _time
    import datetime
    from modes.tag_classifier import classify_prompt
    from modes.llm_service import callLLMVisionTask, supports_vision, get_config, routing_primary_service
    from modes.lighbd_service import _log_lighbd_history

    async def _notify_llm_widget(event_type: str, data: dict = None):
        try:
            import server as _server
            await _server.notify_frontend("lighbd_llm_stream", {"type": event_type, **(data or {})})
        except Exception as e:
            print(f"[BOT_MODE] WARN: notify_frontend 실패: {e}")

    try:
        if not bot_name or not char_name:
            return {"success": False, "error": "bot, character 필드가 필요합니다."}

        char_dir = os.path.join(BOT_DIR, bot_name, char_name)
        face_path = os.path.join(char_dir, "_face_image.webp")
        if not os.path.isfile(face_path):
            print(f"[BOT_MODE] _face_image.webp 없음: {face_path}")
            return {"success": False, "error": f"_face_image.webp이 없습니다. 먼저 얼굴 이미지를 생성하세요. (경로: {char_dir})"}

        # 대표 프롬프트 로드 (rep_images[0])
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
        if not bot:
            return {"success": False, "error": f"봇을 찾을 수 없습니다: {bot_name}"}
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return {"success": False, "error": f"캐릭터를 찾을 수 없습니다: {char_name}"}

        rep_images = char.get("rep_images", [])
        if not rep_images:
            return {"success": False, "error": "대표 이미지(rep_images)가 없습니다."}

        rep0 = rep_images[0]
        rep_base = os.path.splitext(rep0)[0]
        prompt_path = os.path.join(char_dir, f"{rep_base}_prompt.json")
        if not os.path.isfile(prompt_path):
            return {"success": False, "error": f"대표 이미지 프롬프트 파일이 없습니다: {rep_base}_prompt.json"}
        try:
            with open(prompt_path, "r", encoding="utf-8") as pf:
                prompt_text = json.load(pf).get("prompt", "")
        except Exception as e:
            print(f"[BOT_MODE] 대표 프롬프트 로드 실패: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"대표 이미지 프롬프트 로드 실패: {e}"}

        if not prompt_text:
            return {"success": False, "error": "대표 이미지 프롬프트가 비어 있습니다."}

        # 태그 3그룹 분류
        groups = classify_prompt(prompt_text)
        if not any(groups.values()):
            return {"success": False, "error": "분류된 태그가 없습니다. 대표 프롬프트를 확인하세요."}

        # 비전 서비스 확인 (외부 API 분기: primary LLM 기준)
        cfg = get_config()
        service = routing_primary_service("classify_face_tags")
        if not supports_vision(service):
            print(f"[BOT_MODE] 비전 미지원 서비스: {service}")
            return {
                "success": False,
                "error": (
                    f"현재 LLM 서비스({service})는 비전(이미지 입력)을 지원하지 않습니다. "
                    "텍스트 전용 SDK를 사용하는 vertex 대신 OpenAI 호환/Gemini/Claude 등을 config.json에서 선택하세요."
                ),
            }

        # 프롬프트 선택 + 변수 치환
        custom_text, use_custom = _load_auto_face_tag_custom()
        if use_custom and custom_text.strip():
            template = custom_text
        else:
            template = _load_auto_face_tag_builtin()
        if not template.strip():
            return {"success": False, "error": "프롬프트 템플릿이 비어 있습니다."}

        rendered = _render_auto_face_tag_prompt(template, groups)

        # 이미지 base64 인코딩
        try:
            with open(face_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            print(f"[BOT_MODE] _face_image.webp 읽기 실패: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"얼굴 이미지 읽기 실패: {e}"}
        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        messages = [
            {"role": "system", "content": "You are a precise tag classifier. Follow the user's instructions exactly and respond in strict JSON."},
            {"role": "user", "content": rendered},
        ]

        print(f"[BOT_MODE] auto_classify_face_tags 호출: bot={bot_name} char={char_name} service={service} appearance={len(groups.get('외모/신체', []))} attire={len(groups.get('복장', []))} etc={len(groups.get('미분류', []))} use_custom={use_custom}")

        use_model = cfg.get("llm_model", "")
        max_retries = max(0, int(cfg.get("auto_face_tag_max_retries", 2)))
        await _notify_llm_widget("start", {"model": use_model, "prompt_id": f"auto_face_tag:{char_name}"})

        raw = None
        last_err = None
        total_elapsed = 0.0
        for attempt in range(max_retries + 1):
            t0 = _time.time()
            try:
                raw = await callLLMVisionTask("classify_face_tags", messages, image_b64=img_b64, image_mime="image/webp")
            except Exception as call_err:
                print(f"[BOT_MODE] callLLMVision 예외 (시도 {attempt + 1}/{max_retries + 1}): {call_err}")
                traceback.print_exc()
                last_err = f"{type(call_err).__name__}: {call_err}"
                raw = None
            total_elapsed += _time.time() - t0

            if raw and not raw.startswith("[LLM 실패]"):
                parsed = _parse_auto_face_tag_response(raw)
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
                        "prompt_id": f"auto_face_tag:{char_name}",
                        "input": messages,
                        "output": raw,
                        "completion_tokens": done_data["completion_tokens"],
                        "elapsed": done_data["elapsed"],
                        "tps": done_data["tps"],
                        "status": "ok",
                    })
                    print(f"[BOT_MODE] auto_classify_face_tags 완료: face={len(parsed['face'])}개 eye={len(parsed['eye'])}개 (시도 {attempt + 1})")
                    return {"success": True, "data": parsed}
                last_err = f"LLM 응답을 JSON으로 파싱하지 못했습니다. raw: {raw[:300]}"
                print(f"[BOT_MODE] LLM 응답 JSON 파싱 실패 (시도 {attempt + 1}/{max_retries + 1}). raw={raw[:500]}")
            else:
                last_err = f"LLM 호출 실패: {raw or '빈 응답'}"
                print(f"[BOT_MODE] LLM 호출 실패 (시도 {attempt + 1}/{max_retries + 1}): {raw}")

            if attempt < max_retries:
                retry_delay = max(0.0, float(cfg.get("auto_llm_retry_delay_sec", 1.0)))
                print(f"[BOT_MODE] 재시도 대기 중... ({attempt + 1}/{max_retries}) {retry_delay}초")
                await asyncio.sleep(retry_delay)

        await _notify_llm_widget("error", {"error": last_err or "알 수 없는 오류", "elapsed": round(total_elapsed, 3)})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": f"auto_face_tag:{char_name}",
            "input": messages,
            "output": "",
            "elapsed": round(total_elapsed, 3),
            "status": "error",
            "error": last_err or "알 수 없는 오류",
        })
        return {"success": False, "error": f"{max_retries + 1}회 시도 후 실패: {last_err}"}
    except Exception as e:
        print(f"[BOT_MODE] auto_classify_face_tags 예외: {e}")
        traceback.print_exc()
        await _notify_llm_widget("error", {"error": f"{type(e).__name__}: {e}"})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": f"auto_face_tag:{char_name}",
            "input": messages if "messages" in locals() else [],
            "output": "",
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        })
        return {"success": False, "error": str(e)}


def save_char_face_tags(bot_name: str, char_name: str, face_tags: str, eye_tags: str, absolute_tags: str) -> dict:
    """캐릭터 face/eye/absolute 태그 저장 (bot.json 갱신). 반환: {"success": bool, "error"?: str}."""
    try:
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
        if not bot:
            return {"success": False, "error": f"봇을 찾을 수 없음: {bot_name}"}
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return {"success": False, "error": f"캐릭터를 찾을 수 없음: {char_name}"}
        char["face_tags"] = face_tags
        char["eye_tags"] = eye_tags
        char["absolute_tags"] = absolute_tags
        _save_bot_data(data)
        print(f"[BOT_MODE] 캐릭터 얼굴 태그 업데이트: {bot_name}/{char_name}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_MODE] save_char_face_tags 예외: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def handle_auto_classify_face_tags(request):
    """POST /api/bot_mode/auto_classify_face_tags - LLM 비전 기반 얼굴/눈 태그 자동 분류."""
    try:
        body = await request.json()
        bot_name = (body.get("bot") or "").strip()
        char_name = (body.get("character") or "").strip()
        result = await run_auto_classify_face_tags(bot_name, char_name)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_MODE] handle_auto_classify_face_tags 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def run_lb_extra_refine(bot_name: str, char_name: str, appearance_tags: list, outfit_tags: list, etc_tags: list) -> dict:
    """LLM 비전 기반 Appearance/default_outfit 정제 (HTTP 래퍼 없는 core).

    대표 이미지(rep_images[0]) 1장 + 원본 3풀(appearance/outfit/etc)을 LLM에 전달하여
    자세/표정을 제외한 외모·복장만으로 두 그룹을 재구성.

    반환: {"success": True, "data": {"appearance": [...], "outfit": [...]}} 또는 {"success": False, "error": "..."}
    """
    import base64
    import time as _time
    import datetime
    from modes.llm_service import callLLMVisionTask, supports_vision, get_config, routing_primary_service
    from modes.lighbd_service import _log_lighbd_history

    async def _notify_llm_widget(event_type: str, data: dict = None):
        try:
            import server as _server
            await _server.notify_frontend("lighbd_llm_stream", {"type": event_type, **(data or {})})
        except Exception as e:
            print(f"[BOT_MODE] WARN: notify_frontend 실패: {e}")

    try:
        if not bot_name or not char_name:
            return {"success": False, "error": "bot, character 필드가 필요합니다."}

        char_dir = os.path.join(BOT_DIR, bot_name, char_name)
        if not os.path.isdir(char_dir):
            print(f"[BOT_MODE] 캐릭터 디렉토리 없음: {char_dir}")
            return {"success": False, "error": f"캐릭터 디렉토리가 없습니다: {char_dir}"}

        # 대표 이미지(rep_images[0]) 로드
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
        if not bot:
            return {"success": False, "error": f"봇을 찾을 수 없습니다: {bot_name}"}
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return {"success": False, "error": f"캐릭터를 찾을 수 없습니다: {char_name}"}

        rep_images = char.get("rep_images", [])
        if not rep_images:
            return {"success": False, "error": "대표 이미지(rep_images)가 없습니다."}
        rep0 = rep_images[0]
        img_path = os.path.join(char_dir, rep0)
        if not os.path.isfile(img_path):
            print(f"[BOT_MODE] 대표 이미지 파일 없음: {img_path}")
            return {"success": False, "error": f"대표 이미지 파일이 없습니다: {rep0}"}

        # 비전 서비스 확인 (외부 API 분기: primary LLM 기준)
        cfg = get_config()
        service = routing_primary_service("refine_lb_extra")
        if not supports_vision(service):
            print(f"[BOT_MODE] 비전 미지원 서비스: {service}")
            return {
                "success": False,
                "error": (
                    f"현재 LLM 서비스({service})는 비전(이미지 입력)을 지원하지 않습니다. "
                    "텍스트 전용 SDK를 사용하는 vertex 대신 OpenAI 호환/Gemini/Claude 등을 config.json에서 선택하세요."
                ),
            }

        # 프롬프트 선택 + 변수 치환
        custom_text, use_custom = _load_lb_extra_refine_custom()
        if use_custom and custom_text.strip():
            template = custom_text
        else:
            template = _load_lb_extra_refine_builtin()
        if not template.strip():
            return {"success": False, "error": "정제 프롬프트 템플릿이 비어 있습니다."}

        rendered = _render_lb_extra_refine_prompt(template, appearance_tags, outfit_tags, etc_tags)

        # 이미지 base64 인코딩 (mime 추정)
        ext = os.path.splitext(rep0)[1].lower().lstrip(".")
        mime_map = {"webp": "image/webp", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}
        mime = mime_map.get(ext, "image/webp")
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            print(f"[BOT_MODE] 대표 이미지 읽기 실패: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"대표 이미지 읽기 실패: {e}"}
        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        messages = [
            {"role": "system", "content": "You are a precise tag refiner. Follow the user's instructions exactly and respond in strict JSON."},
            {"role": "user", "content": rendered},
        ]

        print(f"[BOT_MODE] lb_extra_refine 호출: bot={bot_name} char={char_name} service={service} "
              f"appearance={len(appearance_tags or [])} outfit={len(outfit_tags or [])} etc={len(etc_tags or [])} use_custom={use_custom}")

        use_model = cfg.get("llm_model", "")
        max_retries = max(0, int(cfg.get("auto_face_tag_max_retries", 2)))
        await _notify_llm_widget("start", {"model": use_model, "prompt_id": f"lb_extra_refine:{char_name}"})

        raw = None
        last_err = None
        total_elapsed = 0.0
        for attempt in range(max_retries + 1):
            t0 = _time.time()
            try:
                raw = await callLLMVisionTask("refine_lb_extra", messages, image_b64=img_b64, image_mime=mime)
            except Exception as call_err:
                print(f"[BOT_MODE] callLLMVision 예외 (시도 {attempt + 1}/{max_retries + 1}): {call_err}")
                traceback.print_exc()
                last_err = f"{type(call_err).__name__}: {call_err}"
                raw = None
            total_elapsed += _time.time() - t0

            if raw and not raw.startswith("[LLM 실패]"):
                parsed = _parse_lb_extra_refine_response(raw)
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
                        "prompt_id": f"lb_extra_refine:{char_name}",
                        "input": messages,
                        "output": raw,
                        "completion_tokens": done_data["completion_tokens"],
                        "elapsed": done_data["elapsed"],
                        "tps": done_data["tps"],
                        "status": "ok",
                    })
                    print(f"[BOT_MODE] lb_extra_refine 완료: appearance={len(parsed['appearance'])}개 outfit={len(parsed['outfit'])}개 (시도 {attempt + 1})")
                    return {"success": True, "data": parsed}
                last_err = f"LLM 응답을 JSON으로 파싱하지 못했습니다. raw: {raw[:300]}"
                print(f"[BOT_MODE] LLM 응답 JSON 파싱 실패 (시도 {attempt + 1}/{max_retries + 1}). raw={raw[:500]}")
            else:
                last_err = f"LLM 호출 실패: {raw or '빈 응답'}"
                print(f"[BOT_MODE] LLM 호출 실패 (시도 {attempt + 1}/{max_retries + 1}): {raw}")

            if attempt < max_retries:
                retry_delay = max(0.0, float(cfg.get("auto_llm_retry_delay_sec", 1.0)))
                print(f"[BOT_MODE] 재시도 대기 중... ({attempt + 1}/{max_retries}) {retry_delay}초")
                await asyncio.sleep(retry_delay)

        await _notify_llm_widget("error", {"error": last_err or "알 수 없는 오류", "elapsed": round(total_elapsed, 3)})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": f"lb_extra_refine:{char_name}",
            "input": messages,
            "output": "",
            "elapsed": round(total_elapsed, 3),
            "status": "error",
            "error": last_err or "알 수 없는 오류",
        })
        return {"success": False, "error": f"{max_retries + 1}회 시도 후 실패: {last_err}"}
    except Exception as e:
        print(f"[BOT_MODE] lb_extra_refine 예외: {e}")
        traceback.print_exc()
        await _notify_llm_widget("error", {"error": f"{type(e).__name__}: {e}"})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": f"lb_extra_refine:{char_name}",
            "input": messages if "messages" in locals() else [],
            "output": "",
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        })
        return {"success": False, "error": str(e)}


async def handle_get_lb_extra_refine_prompt(request):
    """GET /api/bot_mode/lb_extra_refine_prompt - 글로벌/커스텀 정제 프롬프트 조회."""
    try:
        builtin = _load_lb_extra_refine_builtin()
        custom, use_custom = _load_lb_extra_refine_custom()
        return web.json_response({
            "success": True,
            "data": {
                "builtin": builtin,
                "custom": custom,
                "use_custom": use_custom,
            },
        })
    except Exception as e:
        print(f"[BOT_MODE] lb_extra_refine_prompt 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_set_lb_extra_refine_prompt(request):
    """POST /api/bot_mode/lb_extra_refine_prompt - 커스텀 정제 프롬프트 저장."""
    try:
        body = await request.json()
        custom = body.get("custom", "") or ""
        use_custom = bool(body.get("use_custom", False))
        _save_lb_extra_refine_custom(custom, use_custom)
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[BOT_MODE] lb_extra_refine_prompt 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_lb_extra_refine(request):
    """POST /api/bot_mode/lb_extra_refine - LLM 비전 기반 Appearance/default_outfit 정제."""
    try:
        body = await request.json()
        bot_name = (body.get("bot") or "").strip()
        char_name = (body.get("character") or "").strip()
        appearance_tags = body.get("appearance") or []
        outfit_tags = body.get("outfit") or []
        etc_tags = body.get("etc") or []
        if not isinstance(appearance_tags, list) or not isinstance(outfit_tags, list) or not isinstance(etc_tags, list):
            return web.json_response({"success": False, "error": "appearance/outfit/etc 는 list 여야 합니다."})
        result = await run_lb_extra_refine(bot_name, char_name, appearance_tags, outfit_tags, etc_tags)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_MODE] handle_lb_extra_refine 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_llm_batch_enqueue(request):
    """POST /api/bot_mode/llm_batch_enqueue - 선택 캐릭터들을 큐에 추가 (bot_llm_face_tag_analysis 타입)."""
    try:
        body = await request.json()
        bot_name = (body.get("bot") or "").strip()
        characters = body.get("characters") or []
        if not bot_name:
            return web.json_response({"success": False, "error": "bot 필드가 필요합니다."})
        if not isinstance(characters, list) or not characters:
            return web.json_response({"success": False, "error": "characters (비어있지 않은 list) 가 필요합니다."})

        # 큐 매니저 접근
        try:
            import server as _server
            qm = _server.queue_manager
        except Exception as e:
            print(f"[BOT_MODE] queue_manager 접근 실패: {e}")
            traceback.print_exc()
            return web.json_response({"success": False, "error": f"큐 매니저 접근 실패: {e}"})

        added = []
        for char_name in characters:
            cn = (str(char_name) or "").strip()
            if not cn:
                continue
            item = await qm.add_item(
                item_type="bot_llm_face_tag_analysis",
                label=f"LLM 얼굴/눈 태그: {bot_name}/{cn}",
                params={"bot_name": bot_name, "char_name": cn},
                priority=10,
            )
            added.append({"char_name": cn, "id": item.id})

        print(f"[BOT_MODE] LLM 일괄 분석 큐 추가: bot={bot_name} {len(added)}건")
        return web.json_response({"success": True, "data": {"added": added, "count": len(added)}})
    except Exception as e:
        print(f"[BOT_MODE] llm_batch_enqueue 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_auto_group_prompt(request):
    """POST /api/bot_mode/auto_group_prompt - 프롬프트 태그 자동 분류"""
    from modes.tag_classifier import classify_prompt
    try:
        body = await request.json()
        prompt = body.get("prompt", "").strip()
        if not prompt:
            return web.json_response({"success": False, "error": "프롬프트가 비어있습니다."})
        result = classify_prompt(prompt)
        return web.json_response({"success": True, "data": {"groups": result}})
    except Exception as e:
        print(f"[BOT_MODE] 오토 그룹핑 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


bot_mode = BotMode()
data_patcher = BotDataPatcher()
