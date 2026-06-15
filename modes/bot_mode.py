"""
BotMode - 삽화 설정 모드

봇(bot) 단위로 캐릭터 이미지를 관리.
폴더 구조: bot/{봇이름}/{캐릭터이름}/{이미지들}
"""

import asyncio
import json
import os
import time
import uuid
import shutil
import traceback
from typing import Optional
from aiohttp import web


# ─── 상수 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
BOT_DIR = os.path.join(BASE_DIR, "bot")
BOT_DATA_FILE = os.path.join(ASSET_DATA_DIR, "bot.json")
ASSET_DIR = os.path.join(BASE_DIR, "asset")

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
                # solo/group 프로필 마이그레이션
                _migrate_solo_group(data)
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
        """GET /api/bot_mode/asset_chars_with_rep - SSE 스트리밍으로 에셋 캐릭터별 대표 이미지 목록."""
        try:
            from modes import asset_mode as _am
            chars = _am.list_characters()

            async def _stream():
                yield f"data: {json.dumps({'type': 'total', 'count': len(chars)}, ensure_ascii=False)}\n\n"
                idx = 0
                for char_name in chars:
                    gallery = await asyncio.get_event_loop().run_in_executor(
                        None, _am.list_character_gallery, char_name
                    )
                    reps = [g for g in gallery if g.get("representative")]
                    if not reps:
                        continue
                    rep_images = []
                    for g in reps:
                        fn = g["representative"]
                        rel = f"{char_name}/{g['outfit']}/{g['expression']}/{fn}"
                        url = f"/api/asset_mode/characters/{char_name}/outfits/{g['outfit']}/expressions/{g['expression']}/images/{fn}"
                        rep_images.append({"filename": fn, "outfit": g["outfit"], "expression": g["expression"], "path": rel, "url": url})
                    idx += 1
                    char_data = {"name": char_name, "rep_count": len(rep_images), "rep_images": rep_images}
                    yield f"data: {json.dumps({'type': 'character', 'index': idx, 'data': char_data}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'total': idx}, ensure_ascii=False)}\n\n"

            resp = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            )
            await resp.prepare(request)
            async for chunk in _stream():
                await resp.write(chunk.encode("utf-8"))
            await resp.write_eof()
            return resp
        except Exception as e:
            print(f"[BOT_MODE] asset_chars_with_rep 오류: {e}")
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
            if not asset_tool or not asset_tool.workflow_source_path:
                return _json_error("태그 분석 워크플로우 경로가 설정되지 않았습니다")

            from queue_manager import queue_manager
            label = f"태그 분석 (봇 대표: {bot_name}/{char_name or '전체'})"
            item = await queue_manager.add_item("tag_analysis", label, {
                "source": "bot_rep", "bot": bot_name, "character": char_name,
                "filenames": body.get("filenames", []),
            })
            return _json_ok({"success": True, "item_id": item.id})
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
            if not asset_tool or not asset_tool.workflow_source_path:
                return _json_error("태그 분석 워크플로우 경로가 설정되지 않았습니다")

            from queue_manager import queue_manager
            label = f"태그 분석 (봇 유틸: {bot_name})"
            item = await queue_manager.add_item("tag_analysis", label, {
                "source": "bot_utility", "bot": bot_name, "character": char_name,
                "filenames": body.get("filenames", []),
            })
            return _json_ok({"success": True, "item_id": item.id})
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
            if not asset_tool or not asset_tool.workflow_source_path:
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
            print(f"[BOT_MODE] 단어 치환 규칙 저장: bot={bot_name}, {len(rules)}개 규칙")
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
        """GET /api/bot_mode/system_prompt - 봇의 시스템 프롬프트 반환"""
        try:
            bot_name = request.query.get("bot_name", "").strip()
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")
            data = _load_bot_data()
            bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
            if not bot:
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")
            return _json_ok({"text": bot.get("system_prompt", "")})
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_system_prompt(self, request):
        """POST /api/bot_mode/system_prompt - 봇의 시스템 프롬프트 저장"""
        try:
            body = await request.json()
            bot_name = body.get("bot_name", "").strip()
            text = body.get("text", "")
            if not bot_name:
                return _json_error("봇 이름이 비어있습니다.")
            data = _load_bot_data()
            bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
            if not bot:
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")
            bot["system_prompt"] = text
            _save_bot_data(data)
            print(f"[BOT_MODE] 시스템 프롬프트 저장: {bot_name} ({len(text)}자)")
            return _json_ok({"saved": True})
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_system_prompt_presets(self, request):
        """GET /api/bot_mode/system_prompt_presets - 전역 프리셋 목록 반환"""
        try:
            data = _load_bot_data()
            return _json_ok({"presets": data.get("system_prompt_presets", {})})
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 프리셋 로드 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_save_system_prompt_preset(self, request):
        """POST /api/bot_mode/system_prompt_presets - 프리셋 저장/업데이트"""
        try:
            body = await request.json()
            name = body.get("name", "").strip()
            text = body.get("text", "")
            if not name:
                return _json_error("프리셋 이름이 비어있습니다.")
            data = _load_bot_data()
            if "system_prompt_presets" not in data:
                data["system_prompt_presets"] = {}
            data["system_prompt_presets"][name] = text
            _save_bot_data(data)
            print(f"[BOT_MODE] 시스템 프롬프트 프리셋 저장: {name} ({len(text)}자)")
            return _json_ok({"saved": True, "presets": data["system_prompt_presets"]})
        except Exception as e:
            print(f"[BOT_MODE] 시스템 프롬프트 프리셋 저장 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_delete_system_prompt_preset(self, request):
        """DELETE /api/bot_mode/system_prompt_presets - 프리셋 삭제"""
        try:
            body = await request.json()
            name = body.get("name", "").strip()
            if not name:
                return _json_error("프리셋 이름이 비어있습니다.")
            data = _load_bot_data()
            presets = data.get("system_prompt_presets", {})
            if name not in presets:
                return _json_error(f"프리셋을 찾을 수 없습니다: {name}")
            del presets[name]
            data["system_prompt_presets"] = presets
            _save_bot_data(data)
            print(f"[BOT_MODE] 시스템 프롬프트 프리셋 삭제: {name}")
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
                print(f"[BOT_MODE] 단어 치환 로드: {bot_name} ({len(data.get('rules', []))}개 규칙)")
                return data
        except Exception as e:
            print(f"[BOT_MODE] word_replacements 로드 실패: {e}")
    return {"rules": []}


def _save_word_replacements(bot_name: str, data: dict):
    path = _word_replacements_path(bot_name)
    bot_dir = os.path.dirname(path)
    os.makedirs(bot_dir, exist_ok=True)
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

    async def handle_data_patch(self, request):
        """POST /api/bot_mode/data_patch - 선택된 봇의 캐릭터 폴더 + 대표 이미지를 soya_bot/에 복사"""
        try:
            body = await request.json()
            bot_name = body.get("bot_name", "").strip()
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

            # 기존 봇 폴더 삭제 후 재생성
            bot_dst_root = os.path.join(comfy_input_dir, "soya_bot", bot_name)
            if os.path.isdir(bot_dst_root):
                shutil.rmtree(bot_dst_root)
                print(f"[DATA_PATCH] 기존 폴더 삭제: {bot_dst_root}")

            created_dirs = []
            copied_files = []
            skipped_files = []

            for char in bot.get("characters", []):
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
