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
    "bots": []
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _load_bot_data() -> dict:
    """bot.json 로드. 없으면 기본값 생성."""
    if os.path.isfile(BOT_DATA_FILE):
        try:
            with open(BOT_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "bots" not in data:
                    data["bots"] = []
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
                elif action == "rename_bot":
                    return await self._rename_bot(data, body)
                elif action == "add_character":
                    return await self._add_character(data, body)
                elif action == "remove_character":
                    return await self._remove_character(data, body)
                elif action == "rename_character":
                    return await self._rename_character(data, body)
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
        _save_bot_data(data)
        print(f"[BOT_MODE] 봇 삭제: {name}")
        return _json_ok({"bots": data["bots"]})

    async def _rename_bot(self, data, body):
        old_name = body.get("old_name", "").strip()
        new_name = body.get("new_name", "").strip()
        if not old_name or not new_name:
            return _json_error("봇 이름이 비어있습니다.")
        if any(b["name"] == new_name for b in data["bots"]):
            return _json_error(f"이미 존재하는 봇 이름: {new_name}")
        for b in data["bots"]:
            if b["name"] == old_name:
                b["name"] = new_name
                break
        old_path = os.path.join(BOT_DIR, old_name)
        new_path = os.path.join(BOT_DIR, new_name)
        if os.path.isdir(old_path):
            os.rename(old_path, new_path)
        _save_bot_data(data)
        print(f"[BOT_MODE] 봇 이름 변경: {old_name} → {new_name}")
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
        for fname in sorted(os.listdir(char_dir), reverse=True):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            base = os.path.splitext(fname)[0]
            prompt = ""
            prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
            if os.path.isfile(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        pd = json.load(f)
                        prompt = pd.get("prompt", "")
                except Exception:
                    pass
            images.append({
                "filename": fname,
                "prompt": prompt,
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
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
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
                new_name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
                dst = os.path.join(char_dir, new_name)
                shutil.copy2(src, dst)

                # 에셋 프롬프트도 복사
                base = os.path.splitext(os.path.basename(src))[0]
                asset_prompt_path = os.path.join(os.path.dirname(src), f"{base}_prompt.json")
                prompt = ""
                if os.path.isfile(asset_prompt_path):
                    try:
                        with open(asset_prompt_path, "r", encoding="utf-8") as f:
                            pd = json.load(f)
                            prompt = pd.get("positive", "")
                    except Exception:
                        pass

                new_base = os.path.splitext(new_name)[0]
                bot_prompt_path = os.path.join(char_dir, f"{new_base}_prompt.json")
                with open(bot_prompt_path, "w", encoding="utf-8") as f:
                    json.dump({"prompt": prompt, "source": "asset", "original_path": rel_path}, f, ensure_ascii=False)

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


# ─── 유틸리티 ──────────────────────────────────────────
def _json_ok(data, status=200):
    return web.json_response(data, status=status)


def _json_error(msg, status=400):
    print(f"[BOT_MODE] 에러: {msg}")
    return web.json_response({"error": msg}, status=status)


# 싱글톤
bot_mode = BotMode()
