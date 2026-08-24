"""
BotMode - 삽화 설정 모드

봇(bot) 단위로 캐릭터 이미지를 관리.
폴더 구조: bot/{봇이름}/{캐릭터이름}/{이미지들}
"""

import asyncio
from copy import deepcopy
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

from modes.visual_profiles import (
    MAX_VISUAL_CARDS,
    PROFILE_ASSET_FOLDER,
    VisualProfileValidationError,
    cards_to_character_profiles,
    character_profiles_to_cards,
    effective_bot_profiles,
    effective_character_cards,
    effective_character_profiles,
    store_visual_cards,
    sync_root_fields_to_primary_card,
)


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

# 삽화 모드 POSITIVE 태그 규칙 모달의 "추천 태그로 덮어쓰기" 프리셋.
# 현재 프로젝트에서 검증해 사용하는 규칙을 서버 한 곳에서 관리한다.
RECOMMENDED_POSITIVE_RULES = {
    "positive_whitelist": (
        "* expressions",
        "* eyes",
        "* pupils",
        "* mouth",
        "tears",
        "happy",
        "sad",
        "smile",
        "* expression",
    ),
    "positive_blacklist": (),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
FACE_CROP_FOLDER_NAME = "FACE CROP"


def get_bot_visual_targets(
    bot_name: str,
    char_name: str = "",
    *,
    require_rep_images: bool = False,
) -> list[dict]:
    """Return the effective [character, visual-card] work targets for a bot."""
    data = _load_bot_data()
    bot = next((item for item in data.get("bots", []) if item.get("name") == bot_name), None)
    if not bot:
        print(f"[BOT_VISUAL_TARGET] 봇을 찾을 수 없음: bot={bot_name!r}")
        return []

    characters = bot.get("characters", [])
    if char_name:
        characters = [item for item in characters if item.get("name") == char_name]
        if not characters:
            print(
                f"[BOT_VISUAL_TARGET] 캐릭터를 찾을 수 없음: "
                f"bot={bot_name!r}, character={char_name!r}"
            )
            return []

    targets = []
    for character in characters:
        character_name = str(character.get("name") or "").strip()
        if not character_name:
            print(f"[BOT_VISUAL_TARGET] 이름 없는 캐릭터 스킵: bot={bot_name!r}")
            continue
        try:
            cards, _source = effective_character_cards(character, None)
        except Exception as exc:
            print(
                f"[BOT_VISUAL_TARGET] 카드 해석 실패: bot={bot_name!r}, "
                f"character={character_name!r}, error={exc}"
            )
            traceback.print_exc()
            continue
        for index, card in enumerate(cards):
            rep_images = [
                str(value).strip()
                for value in (card.get("rep_images") or [])
                if str(value).strip()
            ]
            if require_rep_images and not rep_images:
                print(
                    f"[BOT_VISUAL_TARGET] 대표 이미지 없는 카드 스킵: "
                    f"bot={bot_name!r}, character={character_name!r}, "
                    f"card={card.get('id')!r}"
                )
                continue
            targets.append({
                "character": character_name,
                "visual_card_id": str(card.get("id") or "").strip(),
                "visual_card_label": str(card.get("label") or f"카드 {index + 1}").strip(),
                "visual_card_index": index + 1,
                "is_primary": index == 0,
                "rep_images": rep_images,
            })
    return targets


def resolve_bot_visual_target(
    bot_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> dict | None:
    requested_id = str(visual_card_id or "").strip()
    targets = get_bot_visual_targets(bot_name, char_name)
    if not targets:
        return None
    if not requested_id:
        return targets[0]
    target = next(
        (item for item in targets if item.get("visual_card_id") == requested_id),
        None,
    )
    if target is None:
        print(
            f"[BOT_VISUAL_TARGET] 카드를 찾을 수 없음: bot={bot_name!r}, "
            f"character={char_name!r}, card={requested_id!r}"
        )
    return target


def bot_visual_artifact_dir(
    bot_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> str:
    target = resolve_bot_visual_target(bot_name, char_name, visual_card_id)
    if target is None:
        raise ValueError(
            f"캐릭터 카드를 찾을 수 없습니다: {bot_name}/{char_name}/{visual_card_id}"
        )
    char_dir = os.path.join(BOT_DIR, bot_name, char_name)
    if target["is_primary"]:
        return char_dir
    return os.path.join(char_dir, PROFILE_ASSET_FOLDER, target["visual_card_id"])


def bot_visual_comfy_relative_path(
    bot_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> str:
    target = resolve_bot_visual_target(bot_name, char_name, visual_card_id)
    if target is None:
        raise ValueError(
            f"캐릭터 카드를 찾을 수 없습니다: {bot_name}/{char_name}/{visual_card_id}"
        )
    base = f"soya_bot/{bot_name}/{char_name}"
    if target["is_primary"]:
        return base
    return f"{base}/{PROFILE_ASSET_FOLDER}/{target['visual_card_id']}"


def get_bot_visual_rep_paths(bot_name: str, char_name: str = "") -> list[dict]:
    """Return representative files for every effective visual card."""
    results = []
    for target in get_bot_visual_targets(bot_name, char_name, require_rep_images=True):
        char_dir = os.path.join(BOT_DIR, bot_name, target["character"])
        for filename in target["rep_images"]:
            if filename != os.path.basename(filename) or filename in {".", ".."}:
                print(
                    f"[BOT_VISUAL_TARGET] 잘못된 대표 이미지 파일명 스킵: "
                    f"bot={bot_name!r}, character={target['character']!r}, "
                    f"card={target['visual_card_id']!r}, filename={filename!r}"
                )
                continue
            filepath = os.path.join(char_dir, filename)
            if not os.path.isfile(filepath):
                print(f"[BOT_MODE] 대표이미지 파일 없음: {filepath}")
                continue
            results.append({
                **target,
                "bot": bot_name,
                "filename": filename,
                "filepath": filepath,
            })
    return results


def get_bot_visual_utility_paths(bot_name: str, char_name: str = "") -> list[dict]:
    """Return card-specific utility FACE files that currently exist."""
    results = []
    for target in get_bot_visual_targets(bot_name, char_name):
        artifact_dir = bot_visual_artifact_dir(
            bot_name, target["character"], target["visual_card_id"]
        )
        filepath = os.path.join(artifact_dir, "_face_image.webp")
        if not os.path.isfile(filepath):
            print(
                f"[BOT_VISUAL_TARGET] FACE 이미지 없는 카드 스킵: "
                f"bot={bot_name!r}, character={target['character']!r}, "
                f"card={target['visual_card_id']!r}, path={filepath!r}"
            )
            continue
        results.append({
            **target,
            "bot": bot_name,
            "filename": "_face_image.webp",
            "filepath": filepath,
        })
    return results


def dialogue_face_crop_filename(source_filename: str) -> str:
    """원본 이미지 파일명에 대응하는 저장 FACE CROP PNG 파일명을 반환한다."""
    base = os.path.splitext(os.path.basename(str(source_filename or "")))[0]
    if not base:
        print(f"[DIALOGUE_FACE_CROP] 출력 파일명 생성 실패: source={source_filename!r}")
        raise ValueError("원본 이미지 파일명이 비어 있습니다.")
    return f"{base}_face.png"


def dialogue_face_crop_dir(bot_name: str, char_name: str) -> str:
    """대사모드용 FACE CROP 고정 폴더 경로를 반환한다."""
    root = os.path.abspath(BOT_DIR)
    folder = os.path.abspath(os.path.join(
        root, str(bot_name or ""), str(char_name or ""), FACE_CROP_FOLDER_NAME
    ))
    if os.path.commonpath([root, folder]) != root:
        print(
            f"[DIALOGUE_FACE_CROP] 폴더 경로 이탈 차단: "
            f"bot={bot_name!r}, char={char_name!r}, path={folder}"
        )
        raise ValueError("잘못된 FACE CROP 폴더 경로입니다.")
    return folder


def dialogue_face_crop_path(bot_name: str, char_name: str, source_filename: str) -> str:
    """원본 이미지에 대응하는 대사모드용 FACE CROP 경로를 반환한다."""
    return os.path.join(
        dialogue_face_crop_dir(bot_name, char_name),
        dialogue_face_crop_filename(source_filename),
    )


def dialogue_face_crop_named_path(bot_name: str, char_name: str, filename: str) -> str:
    """FACE CROP 폴더 안의 단일 파일 경로를 경로 조작 없이 해석한다."""
    safe_name = os.path.basename(str(filename or ""))
    if not safe_name or safe_name != str(filename or ""):
        print(
            f"[DIALOGUE_FACE_CROP] 잘못된 파일명: bot={bot_name!r}, "
            f"char={char_name!r}, filename={filename!r}"
        )
        raise ValueError("잘못된 FACE CROP 파일명입니다.")
    folder = os.path.abspath(dialogue_face_crop_dir(bot_name, char_name))
    path = os.path.abspath(os.path.join(folder, safe_name))
    if os.path.commonpath([folder, path]) != folder:
        print(f"[DIALOGUE_FACE_CROP] FACE CROP 경로 이탈 차단: {path}")
        raise ValueError("잘못된 FACE CROP 경로입니다.")
    return path

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
            changed_card_fields = set()
            # loras → loras_solo
            if "loras" in char and "loras_solo" not in char:
                char["loras_solo"] = char["loras"]
                changed_card_fields.add("loras_solo")
                changed = True
                print(f"[BOT_MODE] 마이그레이션: loras → loras_solo ({bot['name']}/{char['name']})")
            if "loras_group" not in char:
                char["loras_group"] = []
                changed_card_fields.add("loras_group")
                changed = True
            # gender_tag 기본값 보정 — 드롭박스 표시 기본값(1girl)과 일치. 비어 있으면 1girl 로 채운다.
            gt = (char.get("gender_tag") or "").strip()
            if gt not in ("1girl", "1boy", "1male"):
                char["gender_tag"] = "1girl"
                changed_card_fields.add("gender_tag")
                changed = True
                print(f"[BOT_MODE] 마이그레이션: gender_tag 기본값(1girl) 적용 ({bot['name']}/{char['name']})")
            sync_root_fields_to_primary_card(char, changed_card_fields)
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
        if os.path.isfile(BOT_DATA_FILE):
            backup_dir = os.path.join(ASSET_DATA_DIR, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"bot.json.bak_{stamp}_{uuid.uuid4().hex[:8]}"
            backup_path = os.path.join(backup_dir, backup_name)
            shutil.copy2(BOT_DATA_FILE, backup_path)
            print(f"[BOT_MODE] bot.json 저장 전 백업 완료: {backup_path}")
        with open(BOT_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("[BOT_MODE] bot.json 저장 완료")
    except Exception as e:
        print(f"[BOT_MODE] bot.json 저장 실패: {e}")
        traceback.print_exc()
        raise


def copy_default() -> dict:
    import copy
    return copy.deepcopy(DEFAULT_BOT_DATA)


VISUAL_GUIDE_MAX_TARGETS = 120
VISUAL_GUIDE_TASK_KEY = "visual_profile_guide"
VISUAL_GUIDE_QUEUE_TYPE = "visual_profile_guide"


def _selected_bot_system_prompt(data: dict, bot: dict) -> tuple[str, str, str]:
    """Return the selected prompt text and its preset identity without saving data."""
    builtin = _load_builtin_presets() or {}
    local = data.get("system_prompt_presets", {}) or {}
    _ensure_bot_preset_scope(bot, set(builtin), set(local))

    preset = str(bot.get("system_prompt_preset") or "").strip()
    scope = str(bot.get("preset_scope") or "local").strip()
    if scope == "builtin" and (not preset or preset not in builtin):
        print(
            f"[VISUAL_GUIDE] builtin 프롬프트 참조 보정: "
            f"bot={bot.get('name')!r}, preset={preset!r}"
        )
        scope = "local"
    if scope == "local" and (not preset or preset not in local):
        fallback_preset = "기본" if "기본" in local else (next(iter(local), "") if local else "")
        print(
            f"[VISUAL_GUIDE] local 프롬프트 참조 보정: "
            f"bot={bot.get('name')!r}, preset={preset!r}, fallback={fallback_preset!r}"
        )
        preset = fallback_preset

    if scope == "builtin":
        text = builtin.get(preset, "")
    else:
        text = local.get(preset, bot.get("system_prompt", "")) if preset else bot.get("system_prompt", "")
        if not str(text or "").strip():
            print(
                f"[VISUAL_GUIDE] 선택 프롬프트 참조가 유효하지 않음: "
                f"bot={bot.get('name')!r}, preset={preset!r}, scope={scope!r}"
            )
    return str(text or "").strip(), preset, scope


def _visual_guide_tag_text(values) -> str:
    tags = []
    for value in values or []:
        tag = str(value.get("tag") if isinstance(value, dict) else value or "").strip()
        if tag:
            tags.append(tag)
    return ", ".join(tags) or "(none)"


def _build_visual_guide_messages(system_prompt: str, targets: list[dict]) -> list[dict]:
    """Build a prose-first prompt; JSON is used only for the machine-consumed reply."""
    target_sections = []
    for target in targets:
        profile = target["profile"]
        render_overrides = profile.get("render_overrides") or {}
        rep_images = [
            str(value).strip()
            for value in render_overrides.get("rep_images", [])
            if str(value).strip()
        ]
        outfits = []
        for outfit in profile.get("outfits") or []:
            outfits.append(
                f"  - {outfit.get('label') or outfit.get('id')} "
                f"(internal id: {outfit.get('id')}): "
                f"guide={str(outfit.get('selection_guide') or '').strip() or '(none)'}; "
                f"visual tags={_visual_guide_tag_text(outfit.get('tags'))}"
            )
        target_sections.append("\n".join([
            f"### Target {target['target_key']}",
            f"Character: {target['character']}",
            f"Profile internal id: {profile.get('id')}",
            f"Current profile label: {profile.get('label') or profile.get('id')}",
            f"Representative image filenames: {', '.join(rep_images) or '(none)'}",
            f"Current aliases: {', '.join(profile.get('aliases') or []) or '(none)'}",
            f"Current selection guide: {str(profile.get('selection_guide') or '').strip() or '(none)'}",
            f"Appearance evidence: {_visual_guide_tag_text(profile.get('appearance'))}",
            "Registered outfits:",
            *(outfits or ["  - (none)"]),
        ]))

    system_message = """You organize illustration character-card routing metadata.
Read the supplied image-command document as a whole and reason about its own grammar, examples, exceptions, and narrative constraints. Do not classify by hardcoded keyword spotting. Different bots may use completely different command formats.

For every target, infer which canonical character command, form, outfit, corruption/overcome state, or other profile identity it represents. Representative filenames and visual tags are supporting evidence, while the source document is authoritative.

Write Korean natural-language selection guides that explain when the profile becomes true, when it remains true, and the important situations in which it must not be selected. Preserve distinctions such as normal/corrupted/overcome forms, outfit variants, and first-event-only special assets. Do not turn ordinary emotion or action suffixes into a persistent profile unless the target itself is demonstrably that profile.

Aliases are short source-grounded names that identify this profile or form, including exact canonical command labels and confirmed in-story form/outfit titles. Do not add the character's ordinary base name as a profile alias, and do not invent unsupported nicknames or translations. If the match is unclear, use low confidence and explain why instead of fabricating certainty.

Return strict JSON only:
{"suggestions":[{"target_key":"0","aliases":["exact alias"],"selection_guide":"Korean prose","evidence":"brief source-grounded reason","confidence":"high|medium|low"}]}
Return exactly one item for every target_key and no extra targets."""

    user_message = (
        "아래는 이 봇이 실제 대화에 사용하는 이미지 출력 지침을 포함한 시스템 "
        "프롬프트 원문이다. 제목이나 필드명이 고정되어 있다고 가정하지 말고 전체 "
        "문맥을 보존해서 해석하라.\n\n"
        "===== SOURCE DOCUMENT START =====\n"
        f"{system_prompt}\n"
        "===== SOURCE DOCUMENT END =====\n\n"
        "아래 대상은 프로그램에 실제 등록된 캐릭터 카드이다. 내부 ID는 결과를 다시 "
        "연결하기 위한 기계 식별자일 뿐 의미를 추측하는 근거로 사용하지 마라.\n\n"
        + "\n\n".join(target_sections)
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _normalize_visual_guide_llm_result(
    parsed,
    targets: list[dict],
) -> tuple[list[dict] | None, str]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("suggestions"), list):
        reason = "최상위 suggestions 배열이 없습니다."
        print(f"[VISUAL_GUIDE] LLM 응답 구조 오류: {reason}, value={parsed!r}")
        return None, reason

    expected = {str(target["target_key"]): target for target in targets}
    normalized_by_key: dict[str, dict] = {}
    for index, raw in enumerate(parsed["suggestions"]):
        if not isinstance(raw, dict):
            reason = f"suggestions[{index}]가 object가 아닙니다."
            print(f"[VISUAL_GUIDE] LLM 응답 구조 오류: {reason}, value={raw!r}")
            return None, reason
        target_key = str(raw.get("target_key") or "").strip()
        if target_key not in expected:
            reason = f"알 수 없는 target_key입니다: {target_key!r}"
            print(f"[VISUAL_GUIDE] LLM 대상 오류: {reason}")
            return None, reason
        if target_key in normalized_by_key:
            reason = f"target_key가 중복되었습니다: {target_key!r}"
            print(f"[VISUAL_GUIDE] LLM 대상 오류: {reason}")
            return None, reason

        aliases = raw.get("aliases")
        if not isinstance(aliases, list):
            reason = f"target_key={target_key} aliases가 배열이 아닙니다."
            print(f"[VISUAL_GUIDE] LLM 별칭 형식 오류: {reason}")
            return None, reason
        clean_aliases = []
        seen_aliases = set()
        for alias_index, value in enumerate(aliases):
            if not isinstance(value, str):
                reason = (
                    f"target_key={target_key} aliases[{alias_index}]가 문자열이 아닙니다."
                )
                print(f"[VISUAL_GUIDE] LLM 별칭 형식 오류: {reason}")
                return None, reason
            alias = value.strip()
            if len(alias) > 160:
                reason = f"target_key={target_key} 별칭이 160자를 초과했습니다."
                print(f"[VISUAL_GUIDE] LLM 별칭 길이 오류: {reason}")
                return None, reason
            if alias and alias.casefold() not in seen_aliases:
                seen_aliases.add(alias.casefold())
                clean_aliases.append(alias)
        if len(clean_aliases) > 32:
            reason = f"target_key={target_key} 별칭이 32개를 초과했습니다."
            print(f"[VISUAL_GUIDE] LLM 별칭 개수 오류: {reason}")
            return None, reason

        selection_guide = str(raw.get("selection_guide") or "").strip()
        if not selection_guide or len(selection_guide) > 4000:
            reason = (
                f"target_key={target_key} selection_guide가 비었거나 4000자를 초과했습니다."
            )
            print(f"[VISUAL_GUIDE] LLM 선택 기준 오류: {reason}")
            return None, reason
        evidence = str(raw.get("evidence") or "").strip()
        if len(evidence) > 2000:
            reason = f"target_key={target_key} evidence가 2000자를 초과했습니다."
            print(f"[VISUAL_GUIDE] LLM 근거 길이 오류: {reason}")
            return None, reason
        confidence = str(raw.get("confidence") or "low").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            reason = f"target_key={target_key} confidence 값이 잘못되었습니다: {confidence!r}"
            print(f"[VISUAL_GUIDE] LLM 신뢰도 오류: {reason}")
            return None, reason

        target = expected[target_key]
        normalized_by_key[target_key] = {
            "character": target["character"],
            "profile_id": str(target["profile"].get("id") or ""),
            "aliases": clean_aliases,
            "selection_guide": selection_guide,
            "evidence": evidence,
            "confidence": confidence,
        }

    missing = [key for key in expected if key not in normalized_by_key]
    if missing:
        reason = f"응답에서 target_key가 누락되었습니다: {missing}"
        print(f"[VISUAL_GUIDE] LLM 대상 누락: {reason}")
        return None, reason
    return [normalized_by_key[str(target["target_key"])] for target in targets], ""


def _visual_guide_slot_identity(llm_service_module, slot: str) -> tuple[str, str]:
    """Resolve the actual provider/model used by an LLM slot for LB detail records."""
    config = llm_service_module.get_config()
    normalized = str(slot or "llm1").strip().lower()
    suffix = "" if normalized == "llm1" else normalized.removeprefix("llm")
    service = str(
        config.get(f"llm_service{suffix}")
        or config.get("llm_service")
        or ""
    )
    model = str(
        config.get(f"llm_model{suffix}")
        or config.get("llm_model")
        or ""
    )
    return service, model


def _visual_guide_character_call_name(
    *,
    character: str,
    character_index: int,
    character_count: int,
) -> str:
    position = f"{character_index}/{character_count}" if character_count else ""
    suffix = " · ".join(
        value for value in (str(character).strip(), position) if value
    )
    return f"프로필 선택 기준 자동 작성{f' · {suffix}' if suffix else ''}"


def _log_visual_guide_llm_history(
    *,
    llm_service_module,
    bot_name: str,
    messages: list[dict],
    output,
    status: str,
    error: str = "",
    usage: dict | None = None,
    elapsed: float = 0.0,
    phase: str = "",
    llm_slot: str = "",
    history_id: str = "",
    execution_id: str = "",
    parent_execution_id: str = "",
    queue_item_id: str = "",
    character: str = "",
    profile_ids: list[str] | None = None,
    profile_labels: list[str] | None = None,
    character_index: int = 0,
    character_count: int = 0,
    attempt: int | None = None,
    total_attempts: int | None = None,
) -> None:
    """Persist one visual-guide LLM result or failed attempt in LB details."""
    try:
        from modes.lighbd_service import _log_lighbd_history

        slot = str(llm_slot or "llm1")
        service, model = _visual_guide_slot_identity(llm_service_module, slot)
        token_usage = dict(usage or {})
        target_profile_ids = list(profile_ids or [])
        target_profile_labels = list(profile_labels or [])
        call_name = _visual_guide_character_call_name(
            character=character,
            character_index=character_index,
            character_count=character_count,
        )
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prompt_id": f"visual_profile_guide:{bot_name}:{character}",
            "task_key": VISUAL_GUIDE_TASK_KEY,
            "call_name": call_name,
            "history_id": history_id or execution_id or uuid.uuid4().hex,
            "execution_id": execution_id or history_id,
            "parent_execution_id": parent_execution_id,
            "llm_slot": slot,
            "phase": str(phase or "primary"),
            "service": service,
            "model": model,
            "input": messages,
            "output": output if isinstance(output, str) else str(output or ""),
            "completion_tokens": int(token_usage.get("completion_tokens") or 0),
            "prompt_tokens": int(token_usage.get("prompt_tokens") or 0),
            "elapsed": round(max(0.0, float(elapsed or 0.0)), 3),
            "tps": round(float(token_usage.get("tps") or 0.0), 1),
            "status": status,
            "bot_name": bot_name,
            "queue_item_id": queue_item_id,
            "character": character,
            "profile_id": target_profile_ids[0] if len(target_profile_ids) == 1 else "",
            "profile_ids": target_profile_ids,
            "profile_label": (
                target_profile_labels[0] if len(target_profile_labels) == 1 else ""
            ),
            "profile_labels": target_profile_labels,
            "profile_count": len(target_profile_ids),
            "character_index": character_index,
            "character_count": character_count,
            "target_count": len(target_profile_ids),
        }
        if attempt is not None:
            record["attempt"] = attempt
        if total_attempts is not None:
            record["total_attempts"] = total_attempts
        if error:
            record["error"] = str(error)
        _log_lighbd_history(record)
    except Exception as exc:
        print(
            f"[VISUAL_GUIDE:DETAIL] LB 자세히 기록 실패: bot={bot_name!r}, "
            f"character={character!r}, profiles={profile_ids!r}, "
            f"call={character_index}/{character_count}, status={status!r}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()


class BotMode:
    """삽화 설정 모드 매니저"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._asset_tool = None
        self._queue_manager = None

    def set_asset_tool(self, tool):
        self._asset_tool = tool

    def set_queue_manager(self, manager):
        self._queue_manager = manager

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
                elif action == "bulk_set_main_rep":
                    return await self._bulk_set_main_rep(data, body)
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
        sync_root_fields_to_primary_card(char, {"eye_prompt"})
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
        sync_root_fields_to_primary_card(char, {"character_negative"})
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
        if "use_image_name_tag" in body and not isinstance(body["use_image_name_tag"], bool):
            print(
                f"[BOT_MODE] 이미지 이름 태그 사용 값 형식 오류: "
                f"bot={bot_name!r}, char={char_name!r}, "
                f"value={body['use_image_name_tag']!r}"
            )
            return _json_error("이미지 이름 태그 사용 값은 true/false여야 합니다.")
        if "image_name_tag" in body and not isinstance(body["image_name_tag"], str):
            print(
                f"[BOT_MODE] 이미지 이름 태그 형식 오류: "
                f"bot={bot_name!r}, char={char_name!r}, "
                f"value={body['image_name_tag']!r}"
            )
            return _json_error("이미지 이름 태그는 문자열이어야 합니다.")
        char["face_tags"] = face_tags
        char["eye_tags"] = eye_tags
        char["absolute_tags"] = absolute_tags
        if "use_image_name_tag" in body:
            char["use_image_name_tag"] = body["use_image_name_tag"]
        if "image_name_tag" in body:
            char["image_name_tag"] = body["image_name_tag"].strip()
        changed_fields = {"face_tags", "eye_tags", "absolute_tags"}
        if "use_image_name_tag" in body:
            changed_fields.add("use_image_name_tag")
        if "image_name_tag" in body:
            changed_fields.add("image_name_tag")
        sync_root_fields_to_primary_card(char, changed_fields)
        _save_bot_data(data)
        print(
            f"[BOT_MODE] 캐릭터 태그 설정 업데이트: {bot_name}/{char_name}, "
            f"use_image_name_tag={char.get('use_image_name_tag', False)}"
        )
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
        sync_root_fields_to_primary_card(char, {key})
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
        sync_root_fields_to_primary_card(char, {"face_loras"})
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
        sync_root_fields_to_primary_card(char, {"style_loras"})
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
        sync_root_fields_to_primary_card(char, {"gender_tag"})
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
        sync_root_fields_to_primary_card(char, {"rep_images"})
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
        sync_root_fields_to_primary_card(char, {"rep_images"})
        _save_bot_data(data)
        print(f"[BOT_MODE] 대표 이미지 순서 변경: {bot_name}/{char_name}/{filename} {direction}")
        return _json_ok({"bots": data["bots"]})

    async def _bulk_set_main_rep(self, data, body):
        """캐릭터 카드별 메인 대표 지정과 새 카드 생성을 한 번에 저장한다.

        기존 호출의 ``items:[{char_name, filename}]`` 형식은 첫 번째 카드 대상으로
        계속 지원한다. 다중 카드 호출은 ``visual_card_id``를 보내며, 새 카드는
        ``create_profile=true``와 ``source_visual_card_id``를 함께 보낸다. 저장된
        보조 카드 삭제는 ``remove_profile=true``로 요청하며 기본 카드는 보호한다.
        보호 모드에서도 사용자가 직접 고른 항목은 ``manual_override=true``로 교체한다.
        대표를 지정한 카드에는 선택한 파일 하나만 남기고 기존 대표/후보는 제거한다.
        """
        bot_name = body.get("bot_name", "").strip()
        items = body.get("items", []) or []
        mode = (body.get("mode", "") or "").strip()
        if mode not in ("protect", "push"):
            print(f"[BOT_MODE] 일괄 대표 모드가 잘못되어 protect 사용: mode={mode!r}")
            mode = "protect"
        if not bot_name:
            print("[BOT_MODE] 일괄 대표 지정 실패: 봇 이름이 비어있음")
            return _json_error("봇 이름이 비어있습니다.")
        if not isinstance(items, list) or len(items) == 0:
            print(f"[BOT_MODE] 일괄 대표 지정 실패: items 형식 오류, value={items!r}")
            return _json_error("적용할 항목(items)이 비어있습니다.")
        bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
        if not bot:
            print(f"[BOT_MODE] 일괄 대표 지정 실패: 봇을 찾을 수 없음, bot={bot_name!r}")
            return _json_error(f"봇을 찾을 수 없음: {bot_name}")

        updated = []
        removed = []
        skipped = []
        character_states = {}

        def skip_item(char_name, visual_card_id, reason):
            skipped.append({
                "char_name": char_name,
                "visual_card_id": visual_card_id,
                "reason": reason,
            })
            print(
                f"[BOT_MODE] 일괄 대표 지정 스킵: bot={bot_name!r}, "
                f"character={char_name!r}, card={visual_card_id!r}, reason={reason}"
            )

        def character_state(char):
            char_name = str(char.get("name") or "")
            state = character_states.get(char_name)
            if state is not None:
                return state
            cards, source = effective_character_cards(char, None)
            state = {
                "character": char,
                "cards": cards,
                "source": source,
                "dirty": False,
            }
            character_states[char_name] = state
            return state

        def new_card_id(cards):
            used = {str(card.get("id") or "") for card in cards}
            while True:
                candidate = f"card_{uuid.uuid4().hex[:12]}"
                if candidate not in used:
                    return candidate

        for it in items:
            if not isinstance(it, dict):
                skip_item("", "", f"항목이 object가 아님: {it!r}")
                continue
            char_name = (it.get("char_name", "") or "").strip()
            filename = (it.get("filename", "") or "").strip()
            requested_card_id = (it.get("visual_card_id", "") or "").strip()
            if "remove_profile" in it and not isinstance(it.get("remove_profile"), bool):
                skip_item(char_name, requested_card_id, "remove_profile은 bool이어야 함")
                continue
            remove_profile = it.get("remove_profile") is True
            if "create_profile" in it and not isinstance(it.get("create_profile"), bool):
                skip_item(char_name, requested_card_id, "create_profile은 bool이어야 함")
                continue
            create_profile = it.get("create_profile") is True
            if remove_profile and create_profile:
                skip_item(char_name, requested_card_id, "삭제와 새 프로필 생성을 동시에 요청할 수 없음")
                continue
            if "manual_override" in it and not isinstance(it.get("manual_override"), bool):
                skip_item(char_name, requested_card_id, "manual_override는 bool이어야 함")
                continue
            manual_override = it.get("manual_override") is True
            if not char_name or (not remove_profile and not filename):
                skip_item(char_name, requested_card_id, "값이 비어있음")
                continue
            char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
            if not char:
                skip_item(char_name, requested_card_id, "캐릭터를 찾을 수 없음")
                continue

            state = character_state(char)
            cards = state["cards"]
            if remove_profile:
                if not requested_card_id:
                    skip_item(char_name, requested_card_id, "삭제할 프로필 ID가 비어있음")
                    continue
                target_index = next(
                    (
                        index for index, card in enumerate(cards)
                        if str(card.get("id") or "") == requested_card_id
                    ),
                    -1,
                )
                if target_index < 0:
                    skip_item(char_name, requested_card_id, "삭제할 프로필을 찾을 수 없음")
                    continue
                if target_index == 0:
                    skip_item(char_name, requested_card_id, "기본 프로필은 삭제할 수 없음")
                    continue
                removed_card = cards.pop(target_index)
                state["dirty"] = True
                removed.append({
                    "char_name": char_name,
                    "visual_card_id": requested_card_id,
                    "profile_label": str(removed_card.get("label") or ""),
                })
                print(
                    f"[BOT_MODE] 일괄 캐릭터 카드 삭제 반영: "
                    f"{bot_name}/{char_name}/{requested_card_id}"
                )
                continue

            char_dir = os.path.abspath(os.path.join(BOT_DIR, bot_name, char_name))
            image_path = os.path.abspath(os.path.join(char_dir, filename))
            try:
                inside_character = os.path.commonpath([char_dir, image_path]) == char_dir
            except ValueError as exc:
                print(
                    f"[BOT_MODE] 일괄 대표 이미지 경로 비교 실패: "
                    f"character={char_name!r}, filename={filename!r}, error={exc}"
                )
                traceback.print_exc()
                inside_character = False
            if (
                not inside_character
                or filename != os.path.basename(filename)
                or os.path.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS
                or not os.path.isfile(image_path)
            ):
                skip_item(char_name, requested_card_id, "캐릭터 폴더의 이미지 파일이 아님")
                continue

            target_card = None

            if create_profile:
                if len(cards) >= MAX_VISUAL_CARDS:
                    skip_item(char_name, "", f"프로필 최대 {MAX_VISUAL_CARDS}개 초과")
                    continue
                source_card_id = (it.get("source_visual_card_id", "") or "").strip()
                source_card = next(
                    (card for card in cards if str(card.get("id") or "") == source_card_id),
                    cards[0] if not source_card_id and cards else None,
                )
                if source_card is None:
                    skip_item(char_name, source_card_id, "복제 원본 프로필을 찾을 수 없음")
                    continue
                target_card = deepcopy(source_card)
                target_card["id"] = new_card_id(cards)
                target_card["label"] = (
                    str(it.get("profile_label") or "").strip()
                    or f"카드 {len(cards) + 1}"
                )
                target_card["selection_guide"] = ""
                target_card["aliases"] = []
                target_card["rep_images"] = []
                target_card["use_profile_embedding"] = True
                cards.append(target_card)
                requested_card_id = target_card["id"]
            else:
                if requested_card_id:
                    target_card = next(
                        (
                            card for card in cards
                            if str(card.get("id") or "") == requested_card_id
                        ),
                        None,
                    )
                elif cards:
                    target_card = cards[0]
                    requested_card_id = str(target_card.get("id") or "")
                if target_card is None:
                    skip_item(char_name, requested_card_id, "프로필을 찾을 수 없음")
                    continue

            rep_images = target_card.get("rep_images", []) or []
            # 보호 모드의 자동 후보는 건너뛰되, 사용자가 직접 고른 항목은 교체한다.
            if mode == "protect" and rep_images and rep_images[0] and not manual_override:
                skip_item(char_name, requested_card_id, "이미 대표 있음")
                continue
            # 일괄 설정은 빠른 대표 지정용이므로 기존 대표/후보를 보존하지 않는다.
            target_card["rep_images"] = [filename]
            state["dirty"] = True
            updated.append({
                "char_name": char_name,
                "visual_card_id": requested_card_id,
                "filename": filename,
                "created_profile": create_profile,
            })
            print(
                f"[BOT_MODE] 일괄 메인 대표 지정({mode}): "
                f"{bot_name}/{char_name}/{requested_card_id}/{filename}, "
                f"created_profile={create_profile}, manual_override={manual_override}"
            )

        if updated or removed:
            for state in character_states.values():
                if not state["dirty"]:
                    continue
                char = state["character"]
                cards = state["cards"]
                # 기존 단일 카드 데이터는 불필요하게 visual_cards로 마이그레이션하지 않는다.
                # 새 프로필이 생겼거나 이미 카드 저장 형식이면 전체 카드를 검증해 저장한다.
                if state["source"] == "legacy" and len(cards) == 1:
                    primary_reps = deepcopy(cards[0].get("rep_images") or [])
                    if primary_reps:
                        char["rep_images"] = primary_reps
                    else:
                        char.pop("rep_images", None)
                else:
                    store_visual_cards(char, cards)
            _save_bot_data(data)
        return _json_ok({
            "bots": data["bots"],
            "updated": updated,
            "removed": removed,
            "skipped": skipped,
        })

    # ─── 이미지 목록 ─────────────────────────────────────
    async def handle_get_images(self, request):
        """GET /api/bot_mode/images?bot=xxx&character=yyy&visual_card_id=zzz"""
        try:
            bot_name = request.query.get("bot", "").strip()
            char_name = request.query.get("character", "").strip()
            visual_card_id = request.query.get("visual_card_id", "").strip()
            if not bot_name or not char_name:
                print(
                    f"[BOT_MODE] 이미지 목록 필수 값 누락: "
                    f"bot={bot_name!r}, character={char_name!r}, "
                    f"card={visual_card_id!r}"
                )
                return _json_error("봇과 캐릭터 이름이 필요합니다.")

            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            if not os.path.isdir(char_dir):
                print(f"[BOT_MODE] 캐릭터 폴더 없음: {char_dir}")
                return _json_ok({"images": [], "folders": []})

            selected_target = None
            if visual_card_id:
                selected_target = resolve_bot_visual_target(
                    bot_name,
                    char_name,
                    visual_card_id,
                )
                if selected_target is None:
                    print(
                        f"[BOT_MODE] 이미지 목록 카드 해석 실패: "
                        f"bot={bot_name!r}, character={char_name!r}, "
                        f"card={visual_card_id!r}"
                    )
                    return _json_error(
                        f"캐릭터 카드를 찾을 수 없습니다: {visual_card_id}"
                    )

            profile_face_selected = bool(
                selected_target and not selected_target["is_primary"]
            )
            images = []

            def append_image(image_dir, fname, *, image_visual_card_id=""):
                base = os.path.splitext(fname)[0]
                prompt = ""
                negative = ""
                prompt_path = os.path.join(image_dir, f"{base}_prompt.json")
                if os.path.isfile(prompt_path):
                    try:
                        with open(prompt_path, "r", encoding="utf-8") as f:
                            prompt_data = json.load(f)
                        prompt = prompt_data.get("prompt", "")
                        negative = prompt_data.get("negative", "")
                    except Exception as e:
                        print(
                            f"[BOT_MODE] 이미지 프롬프트 로드 실패: "
                            f"path={prompt_path!r}, error={e}"
                        )
                        traceback.print_exc()
                card_query = (
                    f"?visual_card_id={quote(image_visual_card_id, safe='')}"
                    if image_visual_card_id else ""
                )
                images.append({
                    "filename": fname,
                    "prompt": prompt,
                    "negative": negative,
                    "visual_card_id": image_visual_card_id,
                    "url": (
                        f"/api/bot_mode/image/{quote(bot_name, safe='')}/"
                        f"{quote(char_name, safe='')}/{quote(fname, safe='')}"
                        f"{card_query}"
                    ),
                })

            for fname in sorted(os.listdir(char_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                # 보조 프로필을 보고 있을 때는 기본 카드의 FACE를 섞지 않는다.
                if profile_face_selected and fname == "_face_image.webp":
                    continue
                append_image(char_dir, fname)

            if profile_face_selected:
                profile_dir = bot_visual_artifact_dir(
                    bot_name,
                    char_name,
                    selected_target["visual_card_id"],
                )
                profile_face_name = "_face_image.webp"
                profile_face_path = os.path.join(profile_dir, profile_face_name)
                if os.path.isfile(profile_face_path):
                    append_image(
                        profile_dir,
                        profile_face_name,
                        image_visual_card_id=selected_target["visual_card_id"],
                    )
                else:
                    print(
                        f"[BOT_MODE] 프로필 FACE 이미지 없음: "
                        f"bot={bot_name!r}, character={char_name!r}, "
                        f"card={selected_target['visual_card_id']!r}, "
                        f"path={profile_face_path!r}"
                    )

            folders = []
            face_crop_dir = dialogue_face_crop_dir(bot_name, char_name)
            if os.path.isdir(face_crop_dir):
                try:
                    face_crop_count = sum(
                        1
                        for fname in os.listdir(face_crop_dir)
                        if os.path.isfile(os.path.join(face_crop_dir, fname))
                        and os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
                    )
                    folders.append({
                        "name": FACE_CROP_FOLDER_NAME,
                        "count": face_crop_count,
                        "kind": "face_crop",
                    })
                except Exception as e:
                    print(
                        f"[DIALOGUE_FACE_CROP] 폴더 정보 조회 실패: "
                        f"bot={bot_name!r}, char={char_name!r}, "
                        f"path={face_crop_dir}, error={e}"
                    )
                    traceback.print_exc()

            return _json_ok({
                "images": images,
                "folders": folders,
                "visual_card_id": (
                    selected_target["visual_card_id"] if selected_target else ""
                ),
            })
        except Exception as e:
            print(
                f"[BOT_MODE] 이미지 목록 조회 실패: "
                f"bot={locals().get('bot_name', '')!r}, "
                f"character={locals().get('char_name', '')!r}, "
                f"card={locals().get('visual_card_id', '')!r}, error={e}"
            )
            traceback.print_exc()
            return _json_error(str(e))

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
        visual_card_id = request.query.get("visual_card_id", "").strip()
        if not bot_name or not char_name or not filename:
            return _json_error("경로가 올바르지 않습니다.")

        try:
            image_dir = (
                bot_visual_artifact_dir(bot_name, char_name, visual_card_id)
                if visual_card_id else os.path.join(BOT_DIR, bot_name, char_name)
            )
            filepath = os.path.abspath(os.path.join(image_dir, filename))
            allowed_dir = os.path.abspath(image_dir)
            if (
                filename != os.path.basename(filename)
                or os.path.commonpath([allowed_dir, filepath]) != allowed_dir
            ):
                print(
                    f"[BOT_MODE] 잘못된 이미지 경로 접근: bot={bot_name!r}, "
                    f"character={char_name!r}, card={visual_card_id!r}, "
                    f"filename={filename!r}, path={filepath!r}"
                )
                return _json_error("잘못된 경로입니다.")
        except Exception as e:
            print(
                f"[BOT_MODE] 이미지 경로 해석 실패: bot={bot_name!r}, "
                f"character={char_name!r}, card={visual_card_id!r}, error={e}"
            )
            traceback.print_exc()
            return _json_error(str(e))

        if not os.path.isfile(filepath):
            print(f"[BOT_MODE] 이미지 파일 없음: {filepath}")
            return _json_error("파일을 찾을 수 없습니다.", status=404)

        import mimetypes as mt
        content_type = mt.guess_type(filepath)[0] or "image/webp"
        return web.FileResponse(filepath, headers={"Content-Type": content_type})

    async def handle_get_face_crop_images(self, request):
        """GET /api/bot_mode/face_crop_images?bot=...&character=..."""
        try:
            bot_name = request.query.get("bot", "").strip()
            char_name = request.query.get("character", "").strip()
            if not bot_name or not char_name:
                print(
                    f"[DIALOGUE_FACE_CROP] 목록 조회 실패: "
                    f"bot={bot_name!r}, char={char_name!r}"
                )
                return _json_error("봇과 캐릭터 이름이 필요합니다.")

            folder = dialogue_face_crop_dir(bot_name, char_name)
            if not os.path.isdir(folder):
                print(f"[DIALOGUE_FACE_CROP] 목록 폴더 없음: {folder}")
                return _json_ok({"images": [], "folder": FACE_CROP_FOLDER_NAME})

            images = []
            for filename in sorted(os.listdir(folder)):
                path = os.path.join(folder, filename)
                if not os.path.isfile(path):
                    continue
                if os.path.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                images.append({
                    "filename": filename,
                    "url": (
                        f"/api/bot_mode/face_crop_image/{quote(bot_name, safe='')}/"
                        f"{quote(char_name, safe='')}/{quote(filename, safe='')}"
                    ),
                })
            return _json_ok({"images": images, "folder": FACE_CROP_FOLDER_NAME})
        except Exception as e:
            print(f"[DIALOGUE_FACE_CROP] 목록 조회 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_get_face_crop_image(self, request):
        """GET /api/bot_mode/face_crop_image/{bot}/{character}/{filename}"""
        try:
            bot_name = request.match_info.get("bot", "")
            char_name = request.match_info.get("character", "")
            filename = request.match_info.get("filename", "")
            if not bot_name or not char_name or not filename:
                print(
                    f"[DIALOGUE_FACE_CROP] 이미지 조회 필수값 누락: "
                    f"bot={bot_name!r}, char={char_name!r}, filename={filename!r}"
                )
                return _json_error("경로가 올바르지 않습니다.")
            path = dialogue_face_crop_named_path(bot_name, char_name, filename)
            if not os.path.isfile(path):
                print(f"[DIALOGUE_FACE_CROP] 이미지 파일 없음: {path}")
                return _json_error("파일을 찾을 수 없습니다.", status=404)
            import mimetypes as mt
            content_type = mt.guess_type(path)[0] or "image/png"
            return web.FileResponse(path, headers={"Content-Type": content_type})
        except ValueError as e:
            print(f"[DIALOGUE_FACE_CROP] 이미지 경로 검증 실패: {e}")
            return _json_error(str(e), status=400)
        except Exception as e:
            print(f"[DIALOGUE_FACE_CROP] 이미지 조회 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

    async def handle_delete_face_crop_image(self, request):
        """POST /api/bot_mode/face_crop_image/delete - 저장 FACE CROP 단일 삭제."""
        try:
            body = await request.json()
            bot_name = str(body.get("bot", "")).strip()
            char_name = str(body.get("character", "")).strip()
            filename = str(body.get("filename", "")).strip()
            if not bot_name or not char_name or not filename:
                print(
                    f"[DIALOGUE_FACE_CROP] 삭제 필수값 누락: "
                    f"bot={bot_name!r}, char={char_name!r}, filename={filename!r}"
                )
                return _json_error("필수 값이 누락되었습니다.")
            path = dialogue_face_crop_named_path(bot_name, char_name, filename)
            if not os.path.isfile(path):
                print(f"[DIALOGUE_FACE_CROP] 삭제 대상 파일 없음: {path}")
                return _json_error("파일을 찾을 수 없습니다.", status=404)
            os.remove(path)
            print(f"[DIALOGUE_FACE_CROP] 이미지 삭제: {path}")
            return _json_ok({"deleted": True})
        except ValueError as e:
            print(f"[DIALOGUE_FACE_CROP] 삭제 경로 검증 실패: {e}")
            return _json_error(str(e), status=400)
        except Exception as e:
            print(f"[DIALOGUE_FACE_CROP] 이미지 삭제 실패: {e}")
            traceback.print_exc()
            return _json_error(str(e))

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
            visual_card_id = str(body.get("visual_card_id") or "").strip()
            prompt = body.get("prompt", "")

            if not bot_name or not char_name or not filename:
                print(
                    f"[BOT_MODE] 프롬프트 업데이트 필수 값 누락: "
                    f"bot={bot_name!r}, character={char_name!r}, "
                    f"card={visual_card_id!r}, filename={filename!r}"
                )
                return _json_error("필수 값이 누락되었습니다.")
            if filename != os.path.basename(filename) or filename in {".", ".."}:
                print(
                    f"[BOT_MODE] 프롬프트 업데이트 잘못된 파일명: "
                    f"bot={bot_name!r}, character={char_name!r}, "
                    f"card={visual_card_id!r}, filename={filename!r}"
                )
                return _json_error("잘못된 파일명입니다.")

            image_dir = (
                bot_visual_artifact_dir(bot_name, char_name, visual_card_id)
                if visual_card_id
                else os.path.join(BOT_DIR, bot_name, char_name)
            )
            if not os.path.isdir(image_dir):
                print(
                    f"[BOT_MODE] 프롬프트 업데이트 대상 폴더 없음: "
                    f"bot={bot_name!r}, character={char_name!r}, "
                    f"card={visual_card_id!r}, path={image_dir!r}"
                )
                return _json_error("이미지 폴더를 찾을 수 없습니다.")
            base = os.path.splitext(filename)[0]
            prompt_path = os.path.join(image_dir, f"{base}_prompt.json")

            # 기존 데이터 유지하면서 prompt만 업데이트
            existing = {}
            if os.path.isfile(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception as e:
                    print(
                        f"[BOT_MODE] 기존 프롬프트 로드 실패: "
                        f"path={prompt_path!r}, error={e}"
                    )
                    traceback.print_exc()

            existing["prompt"] = prompt
            with open(prompt_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False)

            print(
                f"[BOT_MODE] 프롬프트 업데이트 완료: "
                f"bot={bot_name!r}, character={char_name!r}, "
                f"card={visual_card_id!r}, filename={filename!r}"
            )
            return _json_ok({
                "updated": True,
                "visual_card_id": visual_card_id,
            })
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
        return get_bot_visual_rep_paths(bot_name, char_name)

    def _get_utility_image_paths(self, bot_name: str, char_name: str = "") -> list[dict]:
        """유틸리티 결과 이미지(_face_image.webp) 경로 목록 반환."""
        return get_bot_visual_utility_paths(bot_name, char_name)

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
                prompt_path = os.path.join(
                    os.path.dirname(rep["filepath"]), f"{base}_prompt.json"
                )
                prompt = ""
                negative = ""
                if os.path.isfile(prompt_path):
                    try:
                        with open(prompt_path, "r", encoding="utf-8") as pf:
                            pdata = json.load(pf)
                            prompt = pdata.get("prompt", "")
                            negative = pdata.get("negative", "")
                    except Exception as e:
                        print(f"[BOT_MODE] FACE 프롬프트 로드 실패: {prompt_path} - {e}")
                        traceback.print_exc()
                card_query = (
                    f"?visual_card_id={quote(rep['visual_card_id'], safe='')}"
                    if not rep["is_primary"] else ""
                )
                results.append({
                    "character": rep["character"],
                    "visual_card_id": rep["visual_card_id"],
                    "visual_card_label": rep["visual_card_label"],
                    "visual_card_index": rep["visual_card_index"],
                    "is_primary": rep["is_primary"],
                    "filename": rep["filename"],
                    "prompt": prompt,
                    "negative": negative,
                    "url": (
                        f"/api/bot_mode/image/{quote(bot_name, safe='')}/"
                        f"{quote(rep['character'], safe='')}/{rep['filename']}{card_query}"
                    ),
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
                reps = get_bot_visual_rep_paths(bot_name)

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
                    "visual_card_id": rep["visual_card_id"],
                    "visual_card_label": rep["visual_card_label"],
                    "visual_card_index": rep["visual_card_index"],
                    "is_primary": rep["is_primary"],
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
            one_click_run_id = str(body.get("one_click_run_id") or "").strip()
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
            visual_targets = body.get("visual_targets") or []
            if visual_targets:
                _target_set = {
                    (
                        str(item.get("character") or "").strip(),
                        str(item.get("visual_card_id") or "").strip(),
                        str(item.get("filename") or "").strip(),
                    )
                    for item in visual_targets if isinstance(item, dict)
                }
                reps = [
                    rep for rep in reps
                    if (
                        rep["character"], rep["visual_card_id"], rep["filename"]
                    ) in _target_set
                ]
            batch_label = f"태그 분석 (봇 대표: {bot_name}/{char_name or '전체'}, {len(reps)}장)"
            items_spec = []
            for r in reps:
                img = {
                    "filepath": r["filepath"],
                    "filename": r["filename"],
                    "character": r["character"],
                    "bot": bot_name,
                    "visual_card_id": r["visual_card_id"],
                    "visual_card_label": r["visual_card_label"],
                    "visual_card_index": r["visual_card_index"],
                }
                params = {"source": "bot_rep", "image": img}
                if one_click_run_id:
                    params["one_click_run_id"] = one_click_run_id
                items_spec.append({
                    "type": "tag_analysis",
                    "label": f"태그 분석(봇 대표) {bot_name}/{r['character']}/{r['filename']}",
                    "batch_label": batch_label,
                    "params": params,
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
                reps = get_bot_visual_rep_paths(bot_name)
            if not reps:
                return _json_ok({"total": 0, "success_count": 0, "fail_count": 0})

            # filenames 필터
            only_filenames = body.get("filenames", [])
            if only_filenames:
                reps = [r for r in reps if r["filename"] in only_filenames]
            visual_targets = body.get("visual_targets") or []
            if visual_targets:
                _target_set = {
                    (
                        str(item.get("character") or "").strip(),
                        str(item.get("visual_card_id") or "").strip(),
                        str(item.get("filename") or "").strip(),
                    )
                    for item in visual_targets if isinstance(item, dict)
                }
                reps = [
                    rep for rep in reps
                    if (
                        rep["character"], rep["visual_card_id"], rep["filename"]
                    ) in _target_set
                ]
            if not reps:
                return _json_ok({"total": 0, "success_count": 0, "fail_count": 0})

            success_count = 0
            fail_count = 0
            for rep in reps:
                try:
                    base = os.path.splitext(rep["filename"])[0]
                    rep_dir = os.path.dirname(rep.get("filepath") or "")
                    if not rep_dir:
                        rep_dir = os.path.join(BOT_DIR, bot_name, rep["character"])
                    prompt_path = os.path.join(
                        rep_dir, f"{base}_prompt.json"
                    )
                    existing = {}
                    if os.path.isfile(prompt_path):
                        try:
                            with open(prompt_path, "r", encoding="utf-8") as pf:
                                existing = json.load(pf)
                        except Exception as load_exc:
                            print(
                                "[BOT_MODE] 대표 이미지 프롬프트 로드 실패: "
                                f"path={prompt_path!r}, error={load_exc}"
                            )
                            traceback.print_exc()
                    existing["negative"] = negative_tags
                    _backup_data_file_before_overwrite(
                        prompt_path,
                        f"대표 이미지 부정 프롬프트({bot_name}/{rep['character']}/{rep['filename']})",
                    )
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
            one_click_run_id = str(body.get("one_click_run_id") or "").strip()
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
            visual_targets = body.get("visual_targets") or []
            if visual_targets:
                _target_set = {
                    (
                        str(item.get("character") or "").strip(),
                        str(item.get("visual_card_id") or "").strip(),
                    )
                    for item in visual_targets if isinstance(item, dict)
                }
                reps = [
                    rep for rep in reps
                    if (rep["character"], rep["visual_card_id"]) in _target_set
                ]
            # (레거시) filenames 만 온 경우에도 filename은 캐릭터를 구분하지 못하므로 무시.
            batch_label = f"태그 분석 (봇 유틸: {bot_name}, {len(reps)}장)"
            items_spec = []
            for r in reps:
                img = {
                    "filepath": r["filepath"],
                    "filename": r["filename"],
                    "character": r["character"],
                    "bot": bot_name,
                    "visual_card_id": r["visual_card_id"],
                    "visual_card_label": r["visual_card_label"],
                    "visual_card_index": r["visual_card_index"],
                }
                params = {"source": "bot_utility", "image": img}
                if one_click_run_id:
                    params["one_click_run_id"] = one_click_run_id
                items_spec.append({
                    "type": "tag_analysis",
                    "label": f"태그 분석(봇 유틸) {bot_name}/{r['character']}/{r['filename']}",
                    "batch_label": batch_label,
                    "params": params,
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
            visual_targets = body.get("visual_targets") or []
            if visual_targets:
                _target_set = {
                    (
                        str(item.get("character") or "").strip(),
                        str(item.get("visual_card_id") or "").strip(),
                    )
                    for item in visual_targets if isinstance(item, dict)
                }
                reps = [
                    rep for rep in reps
                    if (rep["character"], rep["visual_card_id"]) in _target_set
                ]
            # (레거시) filenames 만 온 경우에도 filename은 캐릭터를 구분하지 못하므로 무시.
            if not reps:
                print(
                    f"[BOT_MODE] 유틸리티 부정프롬프트 적용 대상 없음: "
                    f"bot={bot_name!r}, character={char_name!r}, characters={characters!r}"
                )
                return _json_ok({
                    "total": 0,
                    "success_count": 0,
                    "fail_count": 0,
                    "failed": [],
                })

            success_count = 0
            fail_count = 0
            failed = []
            for rep in reps:
                try:
                    base = os.path.splitext(rep["filename"])[0]
                    rep_dir = os.path.dirname(rep.get("filepath") or "")
                    if not rep_dir:
                        rep_dir = os.path.join(BOT_DIR, bot_name, rep["character"])
                    prompt_path = os.path.join(
                        rep_dir, f"{base}_prompt.json"
                    )
                    existing = {}
                    if os.path.isfile(prompt_path):
                        try:
                            with open(prompt_path, "r", encoding="utf-8") as pf:
                                existing = json.load(pf)
                        except Exception as load_exc:
                            print(
                                "[BOT_MODE] FACE 프롬프트 로드 실패: "
                                f"path={prompt_path!r}, error={load_exc}"
                            )
                            traceback.print_exc()
                    existing["negative"] = negative_tags
                    _backup_data_file_before_overwrite(
                        prompt_path,
                        f"FACE 부정 프롬프트({bot_name}/{rep['character']}/{rep['filename']})",
                    )
                    with open(prompt_path, "w", encoding="utf-8") as pf:
                        json.dump(existing, pf, ensure_ascii=False, indent=2)
                    success_count += 1
                    print(f"[BOT_MODE] 유틸리티 부정프롬프트 적용: {rep['character']}/{rep['filename']}")
                except Exception as e:
                    fail_count += 1
                    failed.append({
                        "char_name": rep["character"],
                        "visual_card_id": rep.get("visual_card_id", ""),
                        "filename": rep["filename"],
                        "error": str(e),
                    })
                    print(f"[BOT_MODE] 유틸리티 부정프롬프트 실패: {rep['character']}/{rep['filename']} - {e}")
                    traceback.print_exc()

            return _json_ok({
                "total": len(reps),
                "success_count": success_count,
                "fail_count": fail_count,
                "failed": failed,
            })
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

    async def handle_get_character_cards(self, request):
        """GET card data converted to the illustration pipeline's internal shape."""
        try:
            bot_name = request.query.get("bot_name", "").strip()
            char_name = request.query.get("character", "").strip()
            if not bot_name or not char_name:
                print(
                    f"[CHARACTER_CARD:API] 조회 입력 누락: "
                    f"bot={bot_name!r}, character={char_name!r}"
                )
                return _json_error("봇 이름과 캐릭터 이름이 필요합니다.")

            data = _load_bot_data()
            bot = next(
                (item for item in data.get("bots", []) if item.get("name") == bot_name),
                None,
            )
            if bot is None:
                print(f"[CHARACTER_CARD:API] 조회할 봇 없음: bot={bot_name!r}")
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}", 404)
            root_character = next(
                (
                    item for item in bot.get("characters", [])
                    if str(item.get("name") or "").casefold() == char_name.casefold()
                ),
                None,
            )
            if root_character is None:
                print(
                    f"[CHARACTER_CARD:API] 조회할 캐릭터 없음: "
                    f"bot={bot_name!r}, character={char_name!r}"
                )
                return _json_error(f"캐릭터를 찾을 수 없습니다: {char_name}", 404)

            lb_extra = _load_lb_extra(bot_name) or []
            if isinstance(lb_extra, dict) and "edited" in lb_extra:
                lb_extra = lb_extra.get("edited") or []
            extra_character = next(
                (
                    item for item in lb_extra
                    if isinstance(item, dict)
                    and str(item.get("name") or "").casefold() == char_name.casefold()
                ),
                None,
            )
            character, source = effective_character_profiles(
                str(root_character.get("name") or char_name),
                root_character,
                extra_character,
            )
            return _json_ok({
                "character": character,
                "source": source,
                "max_cards": MAX_VISUAL_CARDS,
            })
        except Exception as e:
            print(f"[CHARACTER_CARD:API] 조회 실패: error={e}")
            traceback.print_exc()
            return _json_error(str(e), 500)

    async def handle_save_character_cards(self, request):
        """POST complete character cards into bot.json on the logical character."""
        try:
            body = await request.json()
            bot_name = str(body.get("bot_name") or "").strip()
            char_name = str(body.get("character") or "").strip()
            if not bot_name or not char_name:
                print(
                    f"[CHARACTER_CARD:API] 저장 입력 누락: "
                    f"bot={bot_name!r}, character={char_name!r}"
                )
                return _json_error("봇 이름과 캐릭터 이름이 필요합니다.")

            async with self._lock:
                data = _load_bot_data()
                bot = next(
                    (item for item in data.get("bots", []) if item.get("name") == bot_name),
                    None,
                )
                if bot is None:
                    print(f"[CHARACTER_CARD:API] 저장할 봇 없음: bot={bot_name!r}")
                    return _json_error(f"봇을 찾을 수 없습니다: {bot_name}", 404)
                root_character = next(
                    (
                        item for item in bot.get("characters", [])
                        if str(item.get("name") or "").casefold() == char_name.casefold()
                    ),
                    None,
                )
                if root_character is None:
                    print(
                        f"[CHARACTER_CARD:API] 저장할 캐릭터 없음: "
                        f"bot={bot_name!r}, character={char_name!r}"
                    )
                    return _json_error(f"캐릭터를 찾을 수 없습니다: {char_name}", 404)
                canonical_name = str(root_character.get("name") or char_name)
                raw_character = body.get("data")
                if not isinstance(raw_character, dict):
                    print(
                        f"[CHARACTER_CARD:API] 저장 데이터 없음/형식 오류: "
                        f"bot={bot_name!r}, character={canonical_name!r}, "
                        f"value={raw_character!r}"
                    )
                    return _json_error("저장할 캐릭터 카드 데이터가 필요합니다.")
                supplied_name = str(raw_character.get("name") or "").strip()
                if supplied_name and supplied_name.casefold() != canonical_name.casefold():
                    print(
                        f"[CHARACTER_CARD:API] 캐릭터 이름 불일치: "
                        f"route={canonical_name!r}, data={supplied_name!r}"
                    )
                    return _json_error("경로와 데이터의 캐릭터 이름이 일치하지 않습니다.")
                cards = character_profiles_to_cards(raw_character)
                stored_cards = store_visual_cards(root_character, cards)
                _save_bot_data(data)
                saved_character = cards_to_character_profiles(
                    canonical_name,
                    stored_cards,
                )
                print(
                    f"[CHARACTER_CARD:API] 카드 저장 완료: bot={bot_name!r}, "
                    f"character={canonical_name!r}, cards={len(stored_cards)}"
                )
                return _json_ok({
                    "saved": True,
                    "source": "cards",
                    "character": saved_character,
                    "max_cards": MAX_VISUAL_CARDS,
                    "bots": data["bots"],
                })
        except VisualProfileValidationError as e:
            print(f"[CHARACTER_CARD:API] 저장 검증 실패: error={e}")
            return _json_error(str(e))
        except Exception as e:
            print(f"[CHARACTER_CARD:API] 저장 실패: error={e}")
            traceback.print_exc()
            return _json_error(str(e), 500)

    async def handle_suggest_character_card_metadata(self, request):
        """Suggest aliases and natural selection guides without modifying bot.json."""
        try:
            body = await request.json()
            bot_name = str(body.get("bot_name") or "").strip()
            requested_targets = body.get("targets")
            if not bot_name:
                print("[VISUAL_GUIDE:API] 생성 요청 거부: bot_name이 비어 있음")
                return _json_error("봇 이름이 필요합니다.")
            if not isinstance(requested_targets, list) or not requested_targets:
                print(
                    f"[VISUAL_GUIDE:API] 생성 요청 대상 없음/형식 오류: "
                    f"bot={bot_name!r}, value={requested_targets!r}"
                )
                return _json_error("자동 작성할 캐릭터 카드를 하나 이상 선택하세요.")
            if len(requested_targets) > VISUAL_GUIDE_MAX_TARGETS:
                print(
                    f"[VISUAL_GUIDE:API] 생성 요청 대상 초과: bot={bot_name!r}, "
                    f"count={len(requested_targets)}, max={VISUAL_GUIDE_MAX_TARGETS}"
                )
                return _json_error(
                    f"한 번에 최대 {VISUAL_GUIDE_MAX_TARGETS}개 카드까지 생성할 수 있습니다."
                )

            data = _load_bot_data()
            bot = next(
                (item for item in data.get("bots", []) if item.get("name") == bot_name),
                None,
            )
            if bot is None:
                print(f"[VISUAL_GUIDE:API] 생성할 봇 없음: bot={bot_name!r}")
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}", 404)
            system_prompt, preset, scope = _selected_bot_system_prompt(data, bot)
            source_override = body.get("source_text")
            if source_override is not None and not isinstance(source_override, str):
                print(
                    f"[VISUAL_GUIDE:API] source_text 형식 오류: "
                    f"bot={bot_name!r}, type={type(source_override).__name__}"
                )
                return _json_error("source_text는 문자열이어야 합니다.")
            if isinstance(source_override, str):
                if len(source_override) > 1_000_000:
                    print(
                        f"[VISUAL_GUIDE:API] source_text 길이 초과: "
                        f"bot={bot_name!r}, length={len(source_override)}"
                    )
                    return _json_error("이미지 출력 지침은 1,000,000자를 넘을 수 없습니다.")
                if source_override.strip():
                    system_prompt = source_override.strip()
                    preset = "팝업 임시 원문"
                    scope = "modal"
                    print(
                        f"[VISUAL_GUIDE:API] 팝업 임시 원문 사용: "
                        f"bot={bot_name!r}, length={len(system_prompt)}"
                    )
            if not system_prompt:
                print(
                    f"[VISUAL_GUIDE:API] 시스템 프롬프트 비어 있음: "
                    f"bot={bot_name!r}, preset={preset!r}, scope={scope!r}"
                )
                return _json_error(
                    "현재 봇의 시스템 프롬프트가 비어 있어 선택 기준을 만들 수 없습니다."
                )

            lb_extra = _load_lb_extra(bot_name) or []
            if isinstance(lb_extra, dict) and "edited" in lb_extra:
                lb_extra = lb_extra.get("edited") or []
            root_by_name = {
                str(item.get("name") or "").casefold(): item
                for item in bot.get("characters", [])
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            }
            profiles_by_name = {}
            resolved_targets = []
            seen_targets = set()
            for index, requested in enumerate(requested_targets):
                if not isinstance(requested, dict):
                    print(
                        f"[VISUAL_GUIDE:API] target 형식 오류: index={index}, "
                        f"value={requested!r}"
                    )
                    return _json_error(f"targets[{index}]는 object여야 합니다.")
                character_name = str(requested.get("character") or "").strip()
                profile_id = str(requested.get("profile_id") or "").strip()
                if not character_name or not profile_id:
                    print(
                        f"[VISUAL_GUIDE:API] target 필수값 누락: index={index}, "
                        f"character={character_name!r}, profile={profile_id!r}"
                    )
                    return _json_error(
                        f"targets[{index}]의 character와 profile_id가 필요합니다."
                    )
                identity = (character_name.casefold(), profile_id)
                if identity in seen_targets:
                    print(
                        f"[VISUAL_GUIDE:API] target 중복: character={character_name!r}, "
                        f"profile={profile_id!r}"
                    )
                    return _json_error(
                        f"같은 캐릭터 카드가 중복 선택되었습니다: {character_name}/{profile_id}"
                    )
                seen_targets.add(identity)
                root_character = root_by_name.get(character_name.casefold())
                if root_character is None:
                    print(
                        f"[VISUAL_GUIDE:API] target 캐릭터 없음: "
                        f"character={character_name!r}, profile={profile_id!r}"
                    )
                    return _json_error(
                        f"캐릭터를 찾을 수 없습니다: {character_name}", 404
                    )
                canonical_name = str(root_character.get("name") or character_name)
                cache_key = canonical_name.casefold()
                if cache_key not in profiles_by_name:
                    extra_character = next(
                        (
                            item for item in lb_extra
                            if isinstance(item, dict)
                            and str(item.get("name") or "").casefold() == cache_key
                        ),
                        None,
                    )
                    profiles_by_name[cache_key] = effective_character_profiles(
                        canonical_name,
                        root_character,
                        extra_character,
                    )[0]
                character_profiles = profiles_by_name[cache_key]
                profile = next(
                    (
                        item for item in character_profiles.get("profiles", [])
                        if str(item.get("id") or "") == profile_id
                    ),
                    None,
                )
                if profile is None:
                    print(
                        f"[VISUAL_GUIDE:API] target 카드 없음: "
                        f"character={canonical_name!r}, profile={profile_id!r}"
                    )
                    return _json_error(
                        f"캐릭터 카드를 찾을 수 없습니다: {canonical_name}/{profile_id}",
                        404,
                    )
                resolved_targets.append({
                    "target_key": str(index),
                    "character": canonical_name,
                    "profile": profile,
                })

            from modes import llm_prompt_edit
            from modes import llm_service

            character_groups = []
            group_by_character = {}
            for target in resolved_targets:
                character_key = str(target["character"]).casefold()
                group = group_by_character.get(character_key)
                if group is None:
                    group = {
                        "character": str(target["character"]),
                        "targets": [],
                    }
                    group_by_character[character_key] = group
                    character_groups.append(group)
                group["targets"].append(target)
            character_count = len(character_groups)
            if self._queue_manager is None:
                print(
                    f"[VISUAL_GUIDE:QUEUE] 큐 등록 실패: queue_manager 미주입, "
                    f"bot={bot_name!r}, targets={len(resolved_targets)}"
                )
                return _json_error("LLM 통합 큐가 준비되지 않았습니다.", 503)

            async def notify_visual_guide_progress(
                queue_item,
                *,
                stage: str,
                character: str,
                character_index: int,
                profile_ids: list[str],
                completed: int,
                suggestions: list[dict] | None = None,
                error: str = "",
            ) -> None:
                detail = {
                    "phase": "visual_profile_guide",
                    "stage": stage,
                    "bot_name": bot_name,
                    "character": character,
                    "current": character_index,
                    "completed": completed,
                    "total": character_count,
                    "profile_count": len(profile_ids),
                    "profile_ids": list(profile_ids),
                    "percentage": (
                        (completed / character_count) * 100
                        if character_count > 0
                        else 0
                    ),
                }
                if suggestions is not None:
                    detail["suggestions"] = deepcopy(suggestions)
                if error:
                    detail["error"] = error
                try:
                    notify_progress = getattr(
                        self._queue_manager,
                        "_notify_progress",
                        None,
                    )
                    if not callable(notify_progress):
                        raise RuntimeError("큐 진행률 알림 함수가 없습니다")
                    await notify_progress(queue_item, detail)
                except Exception as exc:
                    print(
                        f"[VISUAL_GUIDE:PROGRESS] 진행률 알림 실패: "
                        f"bot={bot_name!r}, character={character!r}, "
                        f"call={character_index}/{character_count}, stage={stage!r}, "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

            async def run_visual_guide_queue(queue_item):
                if not hasattr(queue_item, "_visual_guide_cancel_requested"):
                    queue_item._visual_guide_cancel_requested = False
                queue_item._visual_guide_active_stream_ids = set()
                queue_item._visual_guide_streaming = False
                suggestions = []
                completed_character_count = 0
                cancelled = False
                for character_index, group in enumerate(character_groups, start=1):
                    if bool(queue_item._visual_guide_cancel_requested):
                        cancelled = completed_character_count < character_count
                        break
                    character = str(group["character"])
                    call_targets = [
                        {**target, "target_key": str(target_index)}
                        for target_index, target in enumerate(group["targets"])
                    ]
                    profile_ids = [
                        str(target["profile"].get("id") or "")
                        for target in call_targets
                    ]
                    profile_labels = [
                        str(
                            target["profile"].get("label")
                            or target["profile"].get("id")
                            or ""
                        )
                        for target in call_targets
                    ]
                    await notify_visual_guide_progress(
                        queue_item,
                        stage="processing",
                        character=character,
                        character_index=character_index,
                        profile_ids=profile_ids,
                        completed=character_index - 1,
                    )
                    messages = _build_visual_guide_messages(system_prompt, call_targets)
                    accepted = {}
                    usage = {}
                    last_raw_result = ""
                    last_failure_reason = ""
                    execution_complete = {}
                    call_started = time.perf_counter()
                    call_name = _visual_guide_character_call_name(
                        character=character,
                        character_index=character_index,
                        character_count=character_count,
                    )
                    execution_context = llm_service.create_llm_execution_context(
                        VISUAL_GUIDE_TASK_KEY,
                        call_name=call_name,
                        json_mode=True,
                        metadata={
                            "prompt_id": f"visual_profile_guide:{bot_name}:{character}",
                            "queue_item_id": queue_item.id,
                            "bot_name": bot_name,
                            "character": character,
                            "profile_ids": profile_ids,
                            "profile_labels": profile_labels,
                            "profile_count": len(call_targets),
                            "character_index": character_index,
                            "character_count": character_count,
                        },
                    )

                    def result_validator(raw_result):
                        parsed = llm_prompt_edit.parse_llm_json(raw_result)
                        normalized, reason = _normalize_visual_guide_llm_result(
                            parsed,
                            call_targets,
                        )
                        if normalized is None:
                            return False, reason
                        accepted["suggestions"] = normalized
                        return True, ""

                    def on_attempt_failure(info):
                        nonlocal last_raw_result, last_failure_reason
                        attempt_raw = info.get("result")
                        if attempt_raw is not None:
                            last_raw_result = str(attempt_raw)
                        attempt_slot = str(info.get("slot") or "llm1")
                        attempt_phase = str(info.get("phase") or "primary")
                        attempt_number = int(info.get("attempt") or 0)
                        total_attempts = int(info.get("total_attempts") or 0)
                        attempt_reason = str(
                            info.get("reason") or "LLM 응답 검증 실패"
                        )
                        last_failure_reason = attempt_reason
                        _log_visual_guide_llm_history(
                            llm_service_module=llm_service,
                            bot_name=bot_name,
                            messages=messages,
                            output=str(attempt_raw or ""),
                            status="error",
                            error=(
                                f"[재시도 {attempt_phase} {attempt_slot} "
                                f"{attempt_number}/{total_attempts}] {attempt_reason}"
                            ),
                            usage=dict(usage),
                            elapsed=float(info.get("elapsed") or 0.0),
                            phase=attempt_phase,
                            llm_slot=attempt_slot,
                            history_id=str(info.get("attempt_id") or ""),
                            execution_id=str(info.get("attempt_id") or ""),
                            parent_execution_id=execution_context.execution_id,
                            queue_item_id=queue_item.id,
                            character=character,
                            profile_ids=profile_ids,
                            profile_labels=profile_labels,
                            character_index=character_index,
                            character_count=character_count,
                            attempt=attempt_number,
                            total_attempts=total_attempts,
                        )

                    def observe_execution(event):
                        if str(event.get("type") or "") == "execution_complete":
                            execution_complete.update(event)

                    stream_state = {"cancelled": False}

                    async def observe_stream(event):
                        event_type = str(event.get("type") or "").strip().lower()
                        stream_id = str(event.get("stream_id") or "").strip()
                        if event_type == "request_mode":
                            queue_item._visual_guide_streaming = bool(
                                event.get("streaming", False)
                            )
                        if event_type == "stream_open" and stream_id:
                            queue_item._visual_guide_active_stream_ids.add(stream_id)
                            if bool(queue_item._visual_guide_cancel_requested):
                                success, reason = llm_service.request_stream_control(
                                    stream_id,
                                    "cancel",
                                )
                                if not success:
                                    print(
                                        f"[VISUAL_GUIDE:CANCEL] 스트림 즉시 중단 실패: "
                                        f"item={queue_item.id}, stream={stream_id}, "
                                        f"reason={reason}"
                                    )
                        if event_type in {"done", "error", "cancelled"} and stream_id:
                            queue_item._visual_guide_active_stream_ids.discard(stream_id)
                        if event_type == "cancelled":
                            stream_state["cancelled"] = True

                    print(
                        f"[VISUAL_GUIDE:API] LLM 생성 시작: bot={bot_name!r}, "
                        f"preset={preset!r}, scope={scope!r}, "
                        f"character={character!r}, profiles={profile_ids!r}, "
                        f"call={character_index}/{character_count}, "
                        f"queue_item={queue_item.id}"
                    )
                    try:
                        raw = await llm_service.callLLMTask(
                            VISUAL_GUIDE_TASK_KEY,
                            messages,
                            json_mode=True,
                            result_validator=result_validator,
                            stream_observer=observe_stream,
                            metadata_sink=usage,
                            on_attempt_failure=on_attempt_failure,
                            execution_context=execution_context,
                            execution_observer=observe_execution,
                        )
                    except Exception as exc:
                        elapsed = time.perf_counter() - call_started
                        slot = str(
                            execution_complete.get("llm_slot")
                            or llm_service.routing_primary_slot(VISUAL_GUIDE_TASK_KEY)
                        )
                        phase = str(execution_complete.get("phase") or "primary")
                        print(
                            f"[VISUAL_GUIDE:API] LLM 호출 예외: bot={bot_name!r}, "
                            f"character={character!r}, profiles={profile_ids!r}, "
                            f"call={character_index}/{character_count}, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()
                        await notify_visual_guide_progress(
                            queue_item,
                            stage="failed",
                            character=character,
                            character_index=character_index,
                            profile_ids=profile_ids,
                            completed=character_index - 1,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        _log_visual_guide_llm_history(
                            llm_service_module=llm_service,
                            bot_name=bot_name,
                            messages=messages,
                            output=last_raw_result,
                            status="error",
                            error=f"{type(exc).__name__}: {exc}",
                            usage=usage,
                            elapsed=elapsed,
                            phase=phase,
                            llm_slot=slot,
                            history_id=execution_context.execution_id,
                            execution_id=execution_context.execution_id,
                            parent_execution_id=execution_context.parent_execution_id,
                            queue_item_id=queue_item.id,
                            character=character,
                            profile_ids=profile_ids,
                            profile_labels=profile_labels,
                            character_index=character_index,
                            character_count=character_count,
                        )
                        raise

                    elapsed = time.perf_counter() - call_started
                    slot = str(
                        execution_complete.get("llm_slot")
                        or llm_service.routing_primary_slot(VISUAL_GUIDE_TASK_KEY)
                    )
                    phase = str(execution_complete.get("phase") or "primary")
                    manual_cancel_type = getattr(
                        llm_service,
                        "ManualCancelledText",
                        None,
                    )
                    if (
                        bool(queue_item._visual_guide_cancel_requested)
                        and (
                            stream_state["cancelled"]
                            or (
                                manual_cancel_type is not None
                                and isinstance(raw, manual_cancel_type)
                            )
                        )
                    ):
                        cancelled = True
                        cancel_reason = "사용자가 현재 LLM 호출을 중단했습니다"
                        await notify_visual_guide_progress(
                            queue_item,
                            stage="cancelled",
                            character=character,
                            character_index=character_index,
                            profile_ids=profile_ids,
                            completed=completed_character_count,
                            error=cancel_reason,
                        )
                        _log_visual_guide_llm_history(
                            llm_service_module=llm_service,
                            bot_name=bot_name,
                            messages=messages,
                            output=raw,
                            status="cancelled",
                            error=cancel_reason,
                            usage=usage,
                            elapsed=elapsed,
                            phase=phase,
                            llm_slot=slot,
                            history_id=execution_context.execution_id,
                            execution_id=execution_context.execution_id,
                            parent_execution_id=execution_context.parent_execution_id,
                            queue_item_id=queue_item.id,
                            character=character,
                            profile_ids=profile_ids,
                            profile_labels=profile_labels,
                            character_index=character_index,
                            character_count=character_count,
                        )
                        print(
                            f"[VISUAL_GUIDE:CANCEL] 현재 스트림 중단 완료: "
                            f"bot={bot_name!r}, character={character!r}, "
                            f"call={character_index}/{character_count}, "
                            f"completed={completed_character_count}"
                        )
                        break
                    profile_suggestions = accepted.get("suggestions")
                    if profile_suggestions is None:
                        parsed = llm_prompt_edit.parse_llm_json(raw)
                        _normalized, reason = _normalize_visual_guide_llm_result(
                            parsed,
                            call_targets,
                        )
                        error = (
                            last_failure_reason
                            or reason
                            or "LLM 응답이 검증을 통과하지 못했습니다."
                        )
                        raw_for_detail = last_raw_result or str(raw or "")
                        print(
                            f"[VISUAL_GUIDE:API] LLM 생성 실패: bot={bot_name!r}, "
                            f"character={character!r}, profiles={profile_ids!r}, "
                            f"call={character_index}/{character_count}, error={error}, "
                            f"raw_preview={raw_for_detail[:500]!r}"
                        )
                        await notify_visual_guide_progress(
                            queue_item,
                            stage="failed",
                            character=character,
                            character_index=character_index,
                            profile_ids=profile_ids,
                            completed=character_index - 1,
                            error=error,
                        )
                        _log_visual_guide_llm_history(
                            llm_service_module=llm_service,
                            bot_name=bot_name,
                            messages=messages,
                            output=raw_for_detail,
                            status="error",
                            error=error,
                            usage=usage,
                            elapsed=elapsed,
                            phase=phase,
                            llm_slot=slot,
                            history_id=execution_context.execution_id,
                            execution_id=execution_context.execution_id,
                            parent_execution_id=execution_context.parent_execution_id,
                            queue_item_id=queue_item.id,
                            character=character,
                            profile_ids=profile_ids,
                            profile_labels=profile_labels,
                            character_index=character_index,
                            character_count=character_count,
                        )
                        raise RuntimeError(
                            f"선택 기준 생성 {character_index}/{character_count} "
                            f"캐릭터 호출 실패 ({character}): "
                            f"{error}"
                        )

                    _log_visual_guide_llm_history(
                        llm_service_module=llm_service,
                        bot_name=bot_name,
                        messages=messages,
                        output=raw,
                        status="ok",
                        usage=usage,
                        elapsed=elapsed,
                        phase=phase,
                        llm_slot=slot,
                        history_id=execution_context.execution_id,
                        execution_id=execution_context.execution_id,
                        parent_execution_id=execution_context.parent_execution_id,
                        queue_item_id=queue_item.id,
                        character=character,
                        profile_ids=profile_ids,
                        profile_labels=profile_labels,
                        character_index=character_index,
                        character_count=character_count,
                    )
                    suggestions.extend(profile_suggestions)
                    completed_character_count = character_index
                    await notify_visual_guide_progress(
                        queue_item,
                        stage="completed",
                        character=character,
                        character_index=character_index,
                        profile_ids=profile_ids,
                        completed=character_index,
                        suggestions=profile_suggestions,
                    )
                    print(
                        f"[VISUAL_GUIDE:API] LLM 생성 완료: bot={bot_name!r}, "
                        f"character={character!r}, profiles={profile_ids!r}, "
                        f"call={character_index}/{character_count}, "
                        f"suggestions={len(profile_suggestions)}, queue_item={queue_item.id}"
                    )
                    if bool(queue_item._visual_guide_cancel_requested):
                        cancelled = completed_character_count < character_count
                        if cancelled:
                            await notify_visual_guide_progress(
                                queue_item,
                                stage="cancelled",
                                character=character,
                                character_index=character_index,
                                profile_ids=profile_ids,
                                completed=completed_character_count,
                                error="현재 캐릭터 완료 후 사용자 요청으로 중단",
                            )
                        break
                if cancelled:
                    queue_item._runtime_cancelled = True
                    queue_item._runtime_cancel_reason = "프로필 선택 기준 자동 작성을 중단했습니다"
                    queue_item._return_result_on_cancel = True
                return {
                    "suggestions": suggestions,
                    "target_count": len(resolved_targets),
                    "character_call_count": character_count,
                    "completed_character_count": completed_character_count,
                    "cancelled": cancelled,
                }

            try:
                queue_item = await self._queue_manager.add_item(
                    VISUAL_GUIDE_QUEUE_TYPE,
                    (
                        f"프로필 선택 기준 자동 작성 · {bot_name} · "
                        f"{len(resolved_targets)}개 카드"
                    ),
                    {
                        "bot_name": bot_name,
                        "target_count": len(resolved_targets),
                        "character_call_count": character_count,
                        "source_preset": preset,
                        "source_scope": scope,
                    },
                    priority=10,
                    runtime_handler=run_visual_guide_queue,
                )
            except Exception as exc:
                print(
                    f"[VISUAL_GUIDE:QUEUE] 큐 등록 실패: bot={bot_name!r}, "
                    f"targets={len(resolved_targets)}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                return _json_error(f"LLM 통합 큐 등록 실패: {exc}", 500)

            print(
                f"[VISUAL_GUIDE:QUEUE] 큐 등록 완료: bot={bot_name!r}, "
                f"item={queue_item.id}, targets={len(resolved_targets)}, "
                f"character_calls={character_count}"
            )
            try:
                queue_result = await queue_item.completion_future
            except Exception as exc:
                print(
                    f"[VISUAL_GUIDE:QUEUE] 큐 처리 실패: bot={bot_name!r}, "
                    f"item={queue_item.id}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                return _json_error(str(exc), 422)

            if not isinstance(queue_result, dict):
                print(
                    f"[VISUAL_GUIDE:QUEUE] 큐 결과 형식 오류: bot={bot_name!r}, "
                    f"item={queue_item.id}, value={queue_result!r}"
                )
                return _json_error("LLM 통합 큐 결과 형식이 올바르지 않습니다.", 500)
            suggestions = queue_result.get("suggestions")
            if not isinstance(suggestions, list):
                print(
                    f"[VISUAL_GUIDE:QUEUE] 큐 제안 결과 누락: bot={bot_name!r}, "
                    f"item={queue_item.id}, value={queue_result!r}"
                )
                return _json_error("LLM 통합 큐 결과에 제안 목록이 없습니다.", 500)

            return _json_ok({
                "success": True,
                "suggestions": suggestions,
                "source": {"preset": preset, "scope": scope},
                "target_count": len(resolved_targets),
                "character_call_count": int(
                    queue_result.get("character_call_count") or 0
                ),
                "completed_character_count": int(
                    queue_result.get("completed_character_count") or 0
                ),
                "cancelled": bool(queue_result.get("cancelled", False)),
            })
        except VisualProfileValidationError as e:
            print(f"[VISUAL_GUIDE:API] 생성 대상 검증 실패: error={e}")
            return _json_error(str(e))
        except Exception as e:
            print(f"[VISUAL_GUIDE:API] 생성 예외: {e}")
            traceback.print_exc()
            return _json_error(str(e), 500)

    async def handle_cancel_character_card_metadata_suggestion(self, request):
        """Stop a running visual-guide job and cancel its active stream when possible."""
        try:
            body = await request.json()
            item_id = str(body.get("item_id") or "").strip()
            if not item_id:
                print(
                    f"[VISUAL_GUIDE:CANCEL] 중단 요청 거부: "
                    f"item_id가 비어 있음, body={body!r}"
                )
                return _json_error("중단할 큐 항목 ID가 필요합니다.")
            if self._queue_manager is None:
                print(
                    f"[VISUAL_GUIDE:CANCEL] 중단 요청 실패: "
                    f"queue_manager 미주입, item={item_id}"
                )
                return _json_error("LLM 통합 큐가 준비되지 않았습니다.", 503)

            queue_item = next(
                (
                    item
                    for item in self._queue_manager.items
                    if str(getattr(item, "id", "")) == item_id
                ),
                None,
            )
            if queue_item is None:
                print(
                    f"[VISUAL_GUIDE:CANCEL] 중단할 큐 항목 없음: "
                    f"item={item_id}"
                )
                return _json_error("중단할 프로필 선택 기준 작업을 찾지 못했습니다.", 404)
            if str(getattr(queue_item, "type", "")) != VISUAL_GUIDE_QUEUE_TYPE:
                print(
                    f"[VISUAL_GUIDE:CANCEL] 다른 큐 타입 중단 거부: "
                    f"item={item_id}, type={getattr(queue_item, 'type', '')!r}"
                )
                return _json_error("프로필 선택 기준 작업만 중단할 수 있습니다.")

            status = str(getattr(queue_item, "status", ""))
            if status == "pending":
                cancelled = await self._queue_manager.cancel_item(item_id)
                if not cancelled:
                    print(
                        f"[VISUAL_GUIDE:CANCEL] pending 취소 실패: "
                        f"item={item_id}, status={getattr(queue_item, 'status', '')!r}"
                    )
                    return _json_error("대기 작업을 취소하지 못했습니다.", 409)
                return _json_ok({
                    "success": True,
                    "item_id": item_id,
                    "mode": "pending_cancelled",
                    "stream_cancelled": 0,
                })
            if status not in {"waiting", "processing"}:
                print(
                    f"[VISUAL_GUIDE:CANCEL] 종료된 작업 중단 거부: "
                    f"item={item_id}, status={status!r}"
                )
                return _json_error("이미 종료된 작업입니다.", 409)

            queue_item._visual_guide_cancel_requested = True
            from modes import llm_service

            active_stream_ids = list(
                getattr(queue_item, "_visual_guide_active_stream_ids", set()) or []
            )
            stream_cancelled = 0
            for stream_id in active_stream_ids:
                success, reason = llm_service.request_stream_control(
                    str(stream_id),
                    "cancel",
                )
                if success:
                    stream_cancelled += 1
                else:
                    print(
                        f"[VISUAL_GUIDE:CANCEL] 활성 스트림 중단 실패: "
                        f"item={item_id}, stream={stream_id}, reason={reason}"
                    )

            progress_detail = dict(
                getattr(queue_item, "progress_detail", {}) or {}
            )
            progress_detail.update({
                "phase": "visual_profile_guide",
                "stage": "cancelling",
                "cancel_requested": True,
            })
            try:
                await self._queue_manager._notify_progress(
                    queue_item,
                    progress_detail,
                )
            except Exception as exc:
                print(
                    f"[VISUAL_GUIDE:CANCEL] 중단 상태 알림 실패: "
                    f"item={item_id}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

            mode = "stream_cancel_requested" if active_stream_ids else "after_current"
            print(
                f"[VISUAL_GUIDE:CANCEL] 중단 요청 접수: item={item_id}, "
                f"status={status}, mode={mode}, active_streams={len(active_stream_ids)}, "
                f"stream_cancelled={stream_cancelled}"
            )
            return _json_ok({
                "success": True,
                "item_id": item_id,
                "mode": mode,
                "stream_cancelled": stream_cancelled,
            })
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(
                f"[VISUAL_GUIDE:CANCEL] 중단 요청 파싱 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return _json_error(str(exc))
        except Exception as exc:
            print(
                f"[VISUAL_GUIDE:CANCEL] 중단 요청 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return _json_error(str(exc), 500)

    async def handle_apply_character_card_metadata(self, request):
        """Atomically apply reviewed suggestions to bot.json."""
        try:
            body = await request.json()
            bot_name = str(body.get("bot_name") or "").strip()
            items = body.get("items")
            overwrite = body.get("overwrite", False)
            if not bot_name:
                print("[VISUAL_GUIDE:APPLY] 적용 요청 거부: bot_name이 비어 있음")
                return _json_error("봇 이름이 필요합니다.")
            if not isinstance(items, list) or not items:
                print(
                    f"[VISUAL_GUIDE:APPLY] 적용 항목 없음/형식 오류: "
                    f"bot={bot_name!r}, value={items!r}"
                )
                return _json_error("적용할 제안을 하나 이상 선택하세요.")
            if len(items) > VISUAL_GUIDE_MAX_TARGETS:
                print(
                    f"[VISUAL_GUIDE:APPLY] 적용 항목 초과: bot={bot_name!r}, "
                    f"count={len(items)}, max={VISUAL_GUIDE_MAX_TARGETS}"
                )
                return _json_error(
                    f"한 번에 최대 {VISUAL_GUIDE_MAX_TARGETS}개 카드까지 적용할 수 있습니다."
                )
            if not isinstance(overwrite, bool):
                print(
                    f"[VISUAL_GUIDE:APPLY] overwrite 형식 오류: "
                    f"bot={bot_name!r}, value={overwrite!r}"
                )
                return _json_error("overwrite는 boolean이어야 합니다.")

            normalized_items = []
            seen_targets = set()
            for index, raw in enumerate(items):
                if not isinstance(raw, dict):
                    print(
                        f"[VISUAL_GUIDE:APPLY] item 형식 오류: index={index}, value={raw!r}"
                    )
                    return _json_error(f"items[{index}]는 object여야 합니다.")
                character = str(raw.get("character") or "").strip()
                profile_id = str(raw.get("profile_id") or "").strip()
                selection_guide = str(raw.get("selection_guide") or "").strip()
                aliases = raw.get("aliases")
                if not character or not profile_id or not selection_guide:
                    print(
                        f"[VISUAL_GUIDE:APPLY] item 필수값 누락: index={index}, "
                        f"character={character!r}, profile={profile_id!r}, "
                        f"guide_length={len(selection_guide)}"
                    )
                    return _json_error(
                        f"items[{index}]의 character, profile_id, selection_guide가 필요합니다."
                    )
                if len(selection_guide) > 4000:
                    print(
                        f"[VISUAL_GUIDE:APPLY] 선택 기준 길이 초과: "
                        f"character={character!r}, profile={profile_id!r}, "
                        f"length={len(selection_guide)}"
                    )
                    return _json_error("자연어 선택 기준은 4000자를 넘을 수 없습니다.")
                if not isinstance(aliases, list):
                    print(
                        f"[VISUAL_GUIDE:APPLY] aliases 형식 오류: index={index}, "
                        f"value={aliases!r}"
                    )
                    return _json_error(f"items[{index}].aliases는 문자열 배열이어야 합니다.")
                clean_aliases = []
                seen_aliases = set()
                for alias_index, value in enumerate(aliases):
                    if not isinstance(value, str):
                        print(
                            f"[VISUAL_GUIDE:APPLY] alias 형식 오류: index={index}, "
                            f"alias_index={alias_index}, value={value!r}"
                        )
                        return _json_error(
                            f"items[{index}].aliases[{alias_index}]는 문자열이어야 합니다."
                        )
                    alias = value.strip()
                    if len(alias) > 160:
                        print(
                            f"[VISUAL_GUIDE:APPLY] alias 길이 초과: "
                            f"character={character!r}, profile={profile_id!r}, "
                            f"alias_index={alias_index}, length={len(alias)}"
                        )
                        return _json_error("작중 별칭 하나는 160자를 넘을 수 없습니다.")
                    if alias and alias.casefold() not in seen_aliases:
                        seen_aliases.add(alias.casefold())
                        clean_aliases.append(alias)
                if len(clean_aliases) > 32:
                    print(
                        f"[VISUAL_GUIDE:APPLY] alias 개수 초과: "
                        f"character={character!r}, profile={profile_id!r}, "
                        f"count={len(clean_aliases)}"
                    )
                    return _json_error("카드 하나에 작중 별칭을 최대 32개까지 저장할 수 있습니다.")
                identity = (character.casefold(), profile_id)
                if identity in seen_targets:
                    print(
                        f"[VISUAL_GUIDE:APPLY] item 중복: "
                        f"character={character!r}, profile={profile_id!r}"
                    )
                    return _json_error(
                        f"같은 캐릭터 카드가 중복되었습니다: {character}/{profile_id}"
                    )
                seen_targets.add(identity)
                normalized_items.append({
                    "character": character,
                    "profile_id": profile_id,
                    "aliases": clean_aliases,
                    "selection_guide": selection_guide,
                })

            async with self._lock:
                data = _load_bot_data()
                bot = next(
                    (item for item in data.get("bots", []) if item.get("name") == bot_name),
                    None,
                )
                if bot is None:
                    print(f"[VISUAL_GUIDE:APPLY] 적용할 봇 없음: bot={bot_name!r}")
                    return _json_error(f"봇을 찾을 수 없습니다: {bot_name}", 404)
                lb_extra = _load_lb_extra(bot_name) or []
                if isinstance(lb_extra, dict) and "edited" in lb_extra:
                    lb_extra = lb_extra.get("edited") or []
                root_by_name = {
                    str(item.get("name") or "").casefold(): item
                    for item in bot.get("characters", [])
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                }
                prepared = {}
                changed_targets = set()
                for item in normalized_items:
                    root_character = root_by_name.get(item["character"].casefold())
                    if root_character is None:
                        print(
                            f"[VISUAL_GUIDE:APPLY] 캐릭터 없음: "
                            f"character={item['character']!r}, profile={item['profile_id']!r}"
                        )
                        return _json_error(
                            f"캐릭터를 찾을 수 없습니다: {item['character']}", 404
                        )
                    canonical_name = str(root_character.get("name") or item["character"])
                    cache_key = canonical_name.casefold()
                    if cache_key not in prepared:
                        extra_character = next(
                            (
                                value for value in lb_extra
                                if isinstance(value, dict)
                                and str(value.get("name") or "").casefold() == cache_key
                            ),
                            None,
                        )
                        prepared[cache_key] = {
                            "root": root_character,
                            "profiles": effective_character_profiles(
                                canonical_name,
                                root_character,
                                extra_character,
                            )[0],
                            "changed": False,
                        }
                    entry = prepared[cache_key]
                    profile = next(
                        (
                            value for value in entry["profiles"].get("profiles", [])
                            if str(value.get("id") or "") == item["profile_id"]
                        ),
                        None,
                    )
                    if profile is None:
                        print(
                            f"[VISUAL_GUIDE:APPLY] 카드 없음: "
                            f"character={canonical_name!r}, profile={item['profile_id']!r}"
                        )
                        return _json_error(
                            f"캐릭터 카드를 찾을 수 없습니다: "
                            f"{canonical_name}/{item['profile_id']}",
                            404,
                        )

                    target_changed = False
                    if overwrite or not profile.get("aliases"):
                        if profile.get("aliases") != item["aliases"]:
                            profile["aliases"] = deepcopy(item["aliases"])
                            target_changed = True
                    if overwrite or not str(profile.get("selection_guide") or "").strip():
                        if str(profile.get("selection_guide") or "").strip() != item["selection_guide"]:
                            profile["selection_guide"] = item["selection_guide"]
                            target_changed = True
                    if target_changed:
                        entry["changed"] = True
                        changed_targets.add((cache_key, item["profile_id"]))

                for entry in prepared.values():
                    if entry["changed"]:
                        cards = character_profiles_to_cards(entry["profiles"])
                        store_visual_cards(entry["root"], cards)

                if changed_targets:
                    _save_bot_data(data)
                    print(
                        f"[VISUAL_GUIDE:APPLY] 일괄 적용 완료: bot={bot_name!r}, "
                        f"overwrite={overwrite}, applied={len(changed_targets)}, "
                        f"skipped={len(normalized_items) - len(changed_targets)}"
                    )
                else:
                    print(
                        f"[VISUAL_GUIDE:APPLY] 변경할 값 없음: bot={bot_name!r}, "
                        f"overwrite={overwrite}, requested={len(normalized_items)}"
                    )
                return _json_ok({
                    "success": True,
                    "saved": bool(changed_targets),
                    "applied": len(changed_targets),
                    "skipped": len(normalized_items) - len(changed_targets),
                    "bots": data["bots"],
                })
        except VisualProfileValidationError as e:
            print(f"[VISUAL_GUIDE:APPLY] 카드 검증 실패: error={e}")
            return _json_error(str(e))
        except Exception as e:
            print(f"[VISUAL_GUIDE:APPLY] 적용 예외: {e}")
            traceback.print_exc()
            return _json_error(str(e), 500)

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
        충돌(이 프리셋을 참조 중인 봇 존재) 처리:
          - reassign 없음 → {"conflict": true, "bots": [...]} 구조화 응답(프론트 재할당 모달용).
          - reassign={bot_name: {"scope","preset"}} 있음 → 각 봇을 대체 프리셋으로 바인딩(타겟 본문
            덮어쓰기 ❌, handle_save_system_prompt 의 local 경로는 본문까지 덮어쓰므로 재사용 금지) 후 삭제.
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

            # 이 프리셋을 참조 중인 봇들 (local scope 한정)
            using_bots = [b for b in data.get("bots", [])
                          if (b.get("system_prompt_preset") or "").strip() == name
                          and (b.get("preset_scope") or "local") == "local"]

            reassigned = []
            if using_bots:
                reassign = body.get("reassign") or {}
                if not reassign:
                    # 충돌 구조화 응답: 프론트에서 모달을 띄워 재할당값을 모아 다시 호출하게 함.
                    bots_info = [
                        {
                            "name": b.get("name", "?"),
                            "preset": (b.get("system_prompt_preset") or "").strip(),
                            "scope": (b.get("preset_scope") or "local").strip(),
                        }
                        for b in using_bots
                    ]
                    msg = ("이 프리셋을 사용 중인 봇이 있어 삭제할 수 없습니다: "
                           + ", ".join(b.get("name", "?") for b in using_bots))
                    print(f"[BOT_MODE] 프리셋 삭제 충돌(name={name}): {len(using_bots)}개 봇 사용 중")
                    return web.json_response(
                        {"error": msg, "conflict": True, "bots": bots_info},
                        status=400,
                    )

                # 1) 각 충돌 봇에 재할당값이 있는지 + 타겟 유효성 검증
                for b in using_bots:
                    bname = b.get("name")
                    if bname not in reassign:
                        return _json_error(f"재할당 대상이 지정되지 않은 봇이 있습니다: {bname}")
                    tgt = reassign.get(bname) or {}
                    tscope = (tgt.get("scope") or "").strip()
                    tpreset = (tgt.get("preset") or "").strip()
                    if tscope not in ("builtin", "local"):
                        return _json_error(f"{bname} 의 대체 프리셋 scope 가 잘못되었습니다: {tscope}")
                    if not tpreset:
                        return _json_error(f"{bname} 의 대체 프리셋이 비어있습니다.")
                    if tscope == "builtin":
                        if tpreset not in builtin:
                            return _json_error(f"{bname} → 배포자료 프리셋을 찾을 수 없습니다: {tpreset}")
                    else:  # local
                        if tpreset == name:
                            return _json_error(f"{bname} 의 대체 프리셋이 삭제 대상과 같습니다: {tpreset}")
                        if tpreset not in presets:
                            return _json_error(f"{bname} → 사용자 프리셋을 찾을 수 없습니다: {tpreset}")

                # 2) 재할당 적용: 타겟 프리셋 본문은 건드리지 않고 바인딩만 변경
                for b in using_bots:
                    bname = b.get("name")
                    tgt = reassign.get(bname) or {}
                    tscope = (tgt.get("scope") or "local").strip()
                    tpreset = (tgt.get("preset") or "").strip()
                    b["system_prompt_preset"] = tpreset
                    b["preset_scope"] = tscope
                    reassigned.append({"name": bname, "scope": tscope, "preset": tpreset})
                    print(f"[BOT_MODE] 재할당(삭제 충돌 회피): {bname} → [{tscope}] '{tpreset}' (삭제 대상 '{name}')")

            del presets[name]
            data["system_prompt_presets"] = presets
            _save_bot_data(data)
            print(f"[BOT_MODE] local 시스템 프롬프트 프리셋 삭제: {name}")
            resp = {"deleted": True, "presets": presets}
            if reassigned:
                resp["reassigned"] = reassigned
            return _json_ok(resp)
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


def _backup_data_file_before_overwrite(path: str, label: str) -> str:
    """기존 데이터 파일을 배포 환경의 backups/에 백업한다."""
    if not os.path.isfile(path):
        return ""
    backup_dir = os.path.join(
        BASE_DIR,
        "backups",
        "illustration_data",
        f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
    )
    try:
        os.makedirs(backup_dir, exist_ok=False)
        backup_path = os.path.join(backup_dir, os.path.basename(path))
        shutil.copy2(path, backup_path)
        print(f"[BOT_MODE] {label} 저장 전 백업 완료: {path} → {backup_path}")
        return backup_path
    except Exception as exc:
        print(f"[BOT_MODE] {label} 저장 전 백업 실패: path={path!r}, error={exc}")
        traceback.print_exc()
        raise RuntimeError(f"{label} 백업에 실패하여 저장을 중단했습니다.") from exc


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
            traceback.print_exc()
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
    from modes.postprocess import (
        normalize_layout_font_scale,
        normalize_min_font_size,
    )
    from modes.onnx_execution import normalize_cpu_threads, normalize_device_key
    clean = dict(bubble or {})
    # 폐기된 말풍선 옵션은 기존 클라이언트가 보내더라도 다시 저장하지 않는다.
    clean.pop("tail_len", None)
    clean.pop("conf", None)
    clean["layout_font_scale"] = normalize_layout_font_scale(
        clean.get("layout_font_scale", 2.0)
    )
    clean["min_font_size"] = normalize_min_font_size(clean.get("min_font_size", 0))
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



def build_utility_prompt(
    bot_name: str,
    char_name: str,
    settings: dict,
    visual_card_id: str = "",
) -> str:
    """캐릭터의 유틸리티 프롬프트 문자열을 생성한다."""
    emb_value = "representation" if settings.get("emb_target") == "대표만" else "representation,sub"
    comfy_path = bot_visual_comfy_relative_path(bot_name, char_name, visual_card_id)
    return (
        f"[PATH]\n{comfy_path}\n"
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
        selected_names = set(char_names)
        visual_targets = [
            target
            for target in get_bot_visual_targets(bot_name, require_rep_images=True)
            if target["character"] in selected_names
        ]
        if not visual_targets:
            print(
                f"[PROGRAM_EMBEDDING] 대표 이미지 있는 카드가 없음: "
                f"bot={bot_name!r}, characters={char_names!r}"
            )
            raise ValueError("선택 캐릭터에 대표 이미지가 있는 카드가 없습니다.")

        preview_id = uuid.uuid4().hex
        session_dir = os.path.join(self._program_embedding_preview_root, preview_id)
        os.makedirs(session_dir, exist_ok=False)
        session_items = []
        response_items = []

        try:
            from PIL import Image
            from modes import face_detector

            for index, target in enumerate(visual_targets):
                char_name = target["character"]
                char_dir = os.path.join(BOT_DIR, bot_name, char_name)
                artifact_dir = bot_visual_artifact_dir(
                    bot_name, char_name, target["visual_card_id"]
                )
                face_path = os.path.join(artifact_dir, "_face_image.webp")
                existing_face = os.path.isfile(face_path)
                card_query = (
                    f"?visual_card_id={quote(target['visual_card_id'], safe='')}"
                    if not target["is_primary"] else ""
                )
                face_url = (
                    f"/api/bot_mode/image/{quote(bot_name, safe='')}/"
                    f"{quote(char_name, safe='')}/_face_image.webp{card_query}"
                    if existing_face else ""
                )
                rep_images = target["rep_images"]
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
                    "visual_card_id": target["visual_card_id"],
                    "visual_card_label": target["visual_card_label"],
                    "visual_card_index": target["visual_card_index"],
                    "is_primary": target["is_primary"],
                    "face_path": None,
                    "confirmed_sha256": "",
                    "save_new_face": False,
                    "preview_path": "",
                }
                response = {
                    "char_name": char_name,
                    "visual_card_id": target["visual_card_id"],
                    "visual_card_label": target["visual_card_label"],
                    "visual_card_index": target["visual_card_index"],
                    "display_name": f"{char_name} [{target['visual_card_index']}]",
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
                    print(
                        f"[PROGRAM_EMBEDDING] 기존 FACE 우선 사용: "
                        f"{bot_name}/{char_name}[{target['visual_card_index']}]"
                    )
                elif not rep_exists:
                    response["message"] = "대표 이미지 파일이 없어 ONNX 얼굴 추출을 할 수 없습니다."
                    print(
                        f"[PROGRAM_EMBEDDING] 대표 이미지 없음: {bot_name}/{char_name}, "
                        f"card={target['visual_card_id']!r}, rep={rep_name!r}"
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
                                f"card={target['visual_card_id']!r}, "
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
                                f"{bot_name}/{char_name}[{target['visual_card_index']}], "
                                f"conf={detected_confidence}"
                            )
                    except Exception as e:
                        print(
                            f"[PROGRAM_EMBEDDING] ONNX 추출 예외("
                            f"{bot_name}/{char_name}[{target['visual_card_index']}]): {e}"
                        )
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
            "backups",
            "program_embedding",
            f"{time.strftime('%Y%m%d_%H%M%S')}_{preview_id[:8]}",
        )
        results = []
        success_count = 0
        face_saved_count = 0
        failed_count = 0

        try:
            from modes import face_embedder

            for item in session["items"]:
                char_name = item["char_name"]
                visual_card_id = item.get("visual_card_id", "")
                visual_card_index = item.get("visual_card_index", 1)
                source_path = item.get("face_path") or ""
                if not source_path:
                    print(
                        f"[PROGRAM_EMBEDDING] 확정 스킵(FACE 없음): "
                        f"{bot_name}/{char_name}[{visual_card_index}]"
                    )
                    results.append({
                        "char_name": char_name,
                        "visual_card_id": visual_card_id,
                        "visual_card_index": visual_card_index,
                        "success": False,
                        "message": "확정 가능한 FACE가 없습니다.",
                    })
                    failed_count += 1
                    continue

                artifact_dir = bot_visual_artifact_dir(
                    bot_name, char_name, visual_card_id
                )
                face_path = os.path.join(artifact_dir, "_face_image.webp")
                prompt_path = os.path.join(artifact_dir, "_face_image_prompt.json")
                cache_path = os.path.join(artifact_dir, "_face_image.l14.npz")
                backup_char_dir = os.path.join(
                    backup_root,
                    self._program_embedding_safe_component(bot_name),
                    self._program_embedding_safe_component(char_name),
                )
                if not item.get("is_primary", True):
                    backup_char_dir = os.path.join(
                        backup_char_dir,
                        self._program_embedding_safe_component(visual_card_id),
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
                        os.makedirs(artifact_dir, exist_ok=True)
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
                        "visual_card_id": visual_card_id,
                        "visual_card_index": visual_card_index,
                        "success": True,
                        "face_saved": save_new_face,
                        "message": (
                            "ONNX FACE 저장 + 임베딩 완료"
                            if save_new_face else "기존 FACE 임베딩 완료"
                        ),
                    })
                    print(
                        f"[PROGRAM_EMBEDDING] 확정 완료: "
                        f"{bot_name}/{char_name}[{visual_card_index}], "
                        f"face_saved={save_new_face}"
                    )
                except Exception as e:
                    print(
                        f"[PROGRAM_EMBEDDING] 확정 실패("
                        f"{bot_name}/{char_name}[{visual_card_index}]): {e}"
                    )
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
                        "visual_card_id": visual_card_id,
                        "visual_card_index": visual_card_index,
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

    async def handle_dialogue_face_crop(self, request):
        """POST /api/bot_mode/dialogue_face_crop - 선택 캐릭터 이미지 크롭 큐 실행."""
        try:
            body = await request.json()
            bot_name = str(body.get("bot_name", "")).strip()
            raw_char_names = body.get("char_names") or []
            if not bot_name:
                print(f"[DIALOGUE_FACE_CROP_API] bot_name 비어있음: body={body!r}")
                return _json_error("봇 이름이 비어있습니다.")
            if not isinstance(raw_char_names, list):
                print(
                    f"[DIALOGUE_FACE_CROP_API] char_names 타입 오류: "
                    f"type={type(raw_char_names).__name__}, value={raw_char_names!r}"
                )
                return _json_error("char_names는 리스트여야 합니다.")
            char_names = []
            for value in raw_char_names:
                name = str(value or "").strip()
                if name and name not in char_names:
                    char_names.append(name)
            if not char_names:
                print(f"[DIALOGUE_FACE_CROP_API] 선택 캐릭터 없음: bot={bot_name!r}")
                return _json_error("선택된 캐릭터가 없습니다.")

            data = _load_bot_data()
            bot = next(
                (entry for entry in data.get("bots", []) if entry.get("name") == bot_name),
                None,
            )
            if not bot:
                print(f"[DIALOGUE_FACE_CROP_API] 봇 없음: {bot_name!r}")
                return _json_error(f"봇을 찾을 수 없습니다: {bot_name}")
            known_characters = {
                str(character.get("name") or "")
                for character in bot.get("characters", [])
            }
            missing = [name for name in char_names if name not in known_characters]
            if missing:
                print(
                    f"[DIALOGUE_FACE_CROP_API] 캐릭터 없음: "
                    f"bot={bot_name!r}, missing={missing!r}"
                )
                return _json_error(f"캐릭터를 찾을 수 없습니다: {', '.join(missing)}")

            config_path = os.path.join(BASE_DIR, "config.json")
            if not os.path.isfile(config_path):
                print(f"[DIALOGUE_FACE_CROP_API] config.json 없음: {config_path}")
                return _json_error("config.json이 없습니다.")
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            workflow_path = str(
                config.get("face_extract_workflow_source_path") or ""
            ).strip()
            if not workflow_path or not os.path.isfile(workflow_path):
                print(
                    f"[DIALOGUE_FACE_CROP_API] 워크플로우 경로 오류: "
                    f"{workflow_path!r}"
                )
                return _json_error("얼굴 추출 워크플로우가 설정되지 않았습니다.")

            face_crop_top = self._program_embedding_float(
                body, "face_crop_top", 1.0, 0.1, 10.0
            )
            face_crop_bottom = self._program_embedding_float(
                body, "face_crop_bottom", 1.0, 0.1, 10.0
            )
            one_click_run_id = str(body.get("one_click_run_id") or "").strip()
            from queue_manager import queue_manager

            queue_params = {
                "operation": "bot_dialogue_face_crop",
                "bot_name": bot_name,
                "char_names": char_names,
                "face_crop_top": face_crop_top,
                "face_crop_bottom": face_crop_bottom,
            }
            if one_click_run_id:
                queue_params["one_click_run_id"] = one_click_run_id
            item = await queue_manager.add_item(
                "instance_lora_face_extract",
                f"[대사 FACE CROP] {bot_name} ({len(char_names)}명)",
                queue_params,
            )
            print(
                f"[DIALOGUE_FACE_CROP_API] 큐 추가 및 완료 대기: "
                f"item={item.id}, bot={bot_name}, chars={char_names}, "
                f"top={face_crop_top}, bottom={face_crop_bottom}"
            )
            result = await item.completion_future
            if not isinstance(result, dict):
                print(
                    f"[DIALOGUE_FACE_CROP_API] 큐 결과 타입 오류: "
                    f"item={item.id}, result={result!r}"
                )
                raise TypeError("FACE CROP 큐 결과가 올바르지 않습니다.")
            return _json_ok(result)
        except (TypeError, ValueError) as e:
            print(f"[DIALOGUE_FACE_CROP_API] 요청/결과 오류: {e}")
            traceback.print_exc()
            return _json_error(str(e))
        except Exception as e:
            print(f"[DIALOGUE_FACE_CROP_API] 실행 실패: {e}")
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

                visual_cards, _source = effective_character_cards(char, None)
                for profile in visual_cards[1:]:
                    profile_id = str(profile.get("id") or "").strip()
                    profile_rep_images = profile.get("rep_images") or []
                    if not profile_rep_images:
                        skipped_files.append(
                            f"{char_name}/_visual_profiles/{profile_id}:rep_images 없음"
                        )
                        print(
                            f"[DATA_PATCH] 프로필 임베딩용 대표 이미지 없음: "
                            f"bot={bot_name!r}, character={char_name!r}, "
                            f"profile={profile_id!r}"
                        )
                        continue
                    profile_dst_dir = os.path.join(
                        dst_dir,
                        "_visual_profiles",
                        profile_id,
                    )
                    os.makedirs(profile_dst_dir, exist_ok=True)
                    created_dirs.append(
                        f"soya_bot/{bot_name}/{char_name}/"
                        f"_visual_profiles/{profile_id}"
                    )
                    for index, image_name in enumerate(profile_rep_images):
                        source_file = os.path.join(
                            BOT_DIR,
                            bot_name,
                            char_name,
                            str(image_name),
                        )
                        if not os.path.isfile(source_file):
                            skipped_files.append(
                                f"{char_name}/_visual_profiles/{profile_id}/"
                                f"{image_name}"
                            )
                            print(
                                f"[DATA_PATCH] 프로필 대표 이미지 소스 없음: "
                                f"path={source_file!r}"
                            )
                            continue
                        extension = os.path.splitext(str(image_name))[1]
                        target_name = (
                            f"representation{extension}"
                            if index == 0
                            else f"sub_{index}{extension}"
                        )
                        target_file = os.path.join(profile_dst_dir, target_name)
                        shutil.copy2(source_file, target_file)
                        relative = (
                            f"{char_name}/_visual_profiles/{profile_id}/"
                            f"{target_name}"
                        )
                        copied_files.append(relative)
                        print(
                            f"[DATA_PATCH] 프로필 대표 이미지 복사: "
                            f"{image_name} -> {relative}"
                        )

            msg = f"폴더 {len(created_dirs)}개 생성, 이미지 {len(copied_files)}개 복사"
            if skipped_files:
                msg += f", 스킵 {len(skipped_files)}개"
            print(f"[DATA_PATCH] 완료: {msg}")
            return _json_ok({
                "message": msg,
                "created_dirs": created_dirs,
                "copied_files": copied_files,
                "skipped_files": skipped_files,
                "visual_targets": [
                    target
                    for target in get_bot_visual_targets(bot_name, require_rep_images=True)
                    if not selected_only or target["character"] in requested_names
                ],
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
            img_bytes, submit_err = await submit_workflow_to_comfy(
                wf,
                task_key="utility_debug",
            )
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
                card_results = {}
                for card in char.get("visual_cards") or []:
                    card_id = str(card.get("id") or "").strip()
                    if not card_id or not card.get("use_profile_embedding"):
                        continue
                    card_dir = os.path.join(char_dir, "_visual_profiles", card_id)
                    card_files = os.listdir(card_dir) if os.path.isdir(card_dir) else []
                    card_results[card_id] = {
                        "ipadpt": any(name.endswith(".ipadpt") for name in card_files),
                        "pt": any(name.endswith(".pt") for name in card_files),
                    }
                    print(
                        f"[CHECK_PATCH] {char_name}/{card_id}: "
                        f"ipadpt={card_results[card_id]['ipadpt']}, "
                        f"pt={card_results[card_id]['pt']}"
                    )
                results[char_name] = {
                    "ipadpt": has_ipadpt,
                    "pt": has_pt,
                    "cards": card_results,
                }
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
            "recommended_positive_whitelist": list(
                RECOMMENDED_POSITIVE_RULES["positive_whitelist"]
            ),
            "recommended_positive_blacklist": list(
                RECOMMENDED_POSITIVE_RULES["positive_blacklist"]
            ),
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
    """GET /api/bot_mode/auto_face_tag_test_image - 배포 번들 테스트 이미지(base64 data URL).
    ?ref=1 이면 REF/참고용 test_img2.webp 를 반환한다(없으면 test_img.webp 로 폴백)."""
    import base64
    use_ref = request.query.get("ref") == "1"
    filename = "test_img2.webp" if use_ref else "test_img.webp"
    path = os.path.join(AUTO_FACE_TAG_PROMPTS_DIR, filename)
    if use_ref and not os.path.isfile(path):
        # test_img2 가 없으면 test_img 로 폴백
        print(f"[BOT_MODE] REF 테스트 이미지 없음, test_img 로 폴백: {path}")
        filename = "test_img.webp"
        path = os.path.join(AUTO_FACE_TAG_PROMPTS_DIR, filename)
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


async def run_auto_classify_face_tags(
    bot_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> dict:
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

        target = resolve_bot_visual_target(bot_name, char_name, visual_card_id)
        if target is None:
            return {"success": False, "error": "캐릭터 카드를 찾을 수 없습니다."}
        char_dir = os.path.join(BOT_DIR, bot_name, char_name)
        artifact_dir = bot_visual_artifact_dir(
            bot_name, char_name, target["visual_card_id"]
        )
        face_path = os.path.join(artifact_dir, "_face_image.webp")
        if not os.path.isfile(face_path):
            print(f"[BOT_MODE] _face_image.webp 없음: {face_path}")
            return {"success": False, "error": f"_face_image.webp이 없습니다. 먼저 얼굴 이미지를 생성하세요. (경로: {artifact_dir})"}

        # 대표 프롬프트 로드 (rep_images[0])
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
        if not bot:
            return {"success": False, "error": f"봇을 찾을 수 없습니다: {bot_name}"}
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return {"success": False, "error": f"캐릭터를 찾을 수 없습니다: {char_name}"}

        rep_images = target["rep_images"]
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

        # 비전 서비스 확인 (외부 LLM 분기: primary LLM 기준)
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

        prompt_id = f"auto_face_tag:{char_name}:{target['visual_card_id']}"
        print(f"[BOT_MODE] auto_classify_face_tags 호출: bot={bot_name} char={char_name} card={target['visual_card_id']} service={service} appearance={len(groups.get('외모/신체', []))} attire={len(groups.get('복장', []))} etc={len(groups.get('미분류', []))} use_custom={use_custom}")

        use_model = cfg.get("llm_model", "")
        await _notify_llm_widget("start", {"model": use_model, "prompt_id": prompt_id})

        raw = None
        last_err = None
        t0 = _time.time()
        try:
            raw = await callLLMVisionTask(
                "classify_face_tags",
                messages,
                image_b64=img_b64,
                image_mime="image/webp",
                result_validator=lambda result: (
                    _parse_auto_face_tag_response(result) is not None,
                    "얼굴/눈 태그 JSON 파싱 실패",
                ),
            )
        except Exception as call_err:
            print(f"[BOT_MODE] callLLMVision 예외: {call_err}")
            traceback.print_exc()
            last_err = f"{type(call_err).__name__}: {call_err}"
            raw = None
        total_elapsed = _time.time() - t0

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
                    "prompt_id": prompt_id,
                    "input": messages,
                    "output": raw,
                    "completion_tokens": done_data["completion_tokens"],
                    "elapsed": done_data["elapsed"],
                    "tps": done_data["tps"],
                    "status": "ok",
                })
                print(f"[BOT_MODE] auto_classify_face_tags 완료: face={len(parsed['face'])}개 eye={len(parsed['eye'])}개")
                return {"success": True, "data": parsed}
            last_err = f"LLM 응답을 JSON으로 파싱하지 못했습니다. raw: {raw[:300]}"
            print(f"[BOT_MODE] LLM 응답 JSON 파싱 실패(라우팅 재시도 소진). raw={raw}")
        else:
            last_err = last_err or f"LLM 호출 실패: {raw or '빈 응답'}"
            print(f"[BOT_MODE] LLM 호출 실패(라우팅 재시도 소진): {raw}")

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
        return {"success": False, "error": f"라우팅 재시도 후 실패: {last_err}"}
    except Exception as e:
        print(f"[BOT_MODE] auto_classify_face_tags 예외: {e}")
        traceback.print_exc()
        await _notify_llm_widget("error", {"error": f"{type(e).__name__}: {e}"})
        _log_lighbd_history({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": (
                prompt_id if "prompt_id" in locals()
                else f"auto_face_tag:{char_name}:{visual_card_id}"
            ),
            "input": messages if "messages" in locals() else [],
            "output": "",
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        })
        return {"success": False, "error": str(e)}


def save_char_face_tags(
    bot_name: str,
    char_name: str,
    face_tags: str,
    eye_tags: str,
    absolute_tags: str,
    use_image_name_tag: bool | None = None,
    image_name_tag: str | None = None,
    visual_card_id: str = "",
) -> dict:
    """캐릭터 태그 설정 저장 (bot.json 갱신).

    이미지 이름 태그 인자가 생략되면 기존 값을 보존해, 얼굴 자동 분류 같은
    기존 호출이 사용자가 설정한 옵션을 비활성화하지 않게 한다.
    """
    try:
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
        if not bot:
            return {"success": False, "error": f"봇을 찾을 수 없음: {bot_name}"}
        char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
        if not char:
            return {"success": False, "error": f"캐릭터를 찾을 수 없음: {char_name}"}
        had_visual_cards = isinstance(char.get("visual_cards"), list) and bool(char["visual_cards"])
        cards, _source = effective_character_cards(char, None)
        requested_id = str(visual_card_id or "").strip()
        card_index = 0 if not requested_id else next(
            (
                index for index, card in enumerate(cards)
                if card.get("id") == requested_id
            ),
            -1,
        )
        if card_index < 0:
            error = f"캐릭터 카드를 찾을 수 없음: {char_name}/{requested_id}"
            print(f"[BOT_MODE] save_char_face_tags 실패: {error}")
            return {"success": False, "error": error}
        target_card = cards[card_index]
        target_card["face_tags"] = face_tags
        target_card["eye_tags"] = eye_tags
        target_card["absolute_tags"] = absolute_tags
        if use_image_name_tag is not None:
            if not isinstance(use_image_name_tag, bool):
                error = f"이미지 이름 태그 사용 값은 bool이어야 합니다: {use_image_name_tag!r}"
                print(f"[BOT_MODE] save_char_face_tags 검증 실패: {error}")
                return {"success": False, "error": error}
            target_card["use_image_name_tag"] = use_image_name_tag
        if image_name_tag is not None:
            if not isinstance(image_name_tag, str):
                error = f"이미지 이름 태그는 문자열이어야 합니다: {image_name_tag!r}"
                print(f"[BOT_MODE] save_char_face_tags 검증 실패: {error}")
                return {"success": False, "error": error}
            target_card["image_name_tag"] = image_name_tag.strip()
        if had_visual_cards:
            store_visual_cards(char, cards)
        else:
            char["face_tags"] = target_card["face_tags"]
            char["eye_tags"] = target_card["eye_tags"]
            char["absolute_tags"] = target_card["absolute_tags"]
            if use_image_name_tag is not None:
                char["use_image_name_tag"] = target_card["use_image_name_tag"]
            if image_name_tag is not None:
                char["image_name_tag"] = target_card["image_name_tag"]
        _save_bot_data(data)
        print(
            f"[BOT_MODE] 캐릭터 태그 설정 업데이트: {bot_name}/{char_name}"
            f"[{card_index + 1}]"
        )
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
        visual_card_id = (body.get("visual_card_id") or "").strip()
        result = await run_auto_classify_face_tags(
            bot_name, char_name, visual_card_id
        )
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_MODE] handle_auto_classify_face_tags 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def run_lb_extra_refine(
    bot_name: str,
    char_name: str,
    appearance_tags: list,
    outfit_tags: list,
    etc_tags: list,
    visual_card_id: str = "",
) -> dict:
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

        selected_card_id = str(visual_card_id or "").strip()
        if selected_card_id:
            cards, _source = effective_character_cards(char, None)
            selected_card = next(
                (card for card in cards if str(card.get("id") or "") == selected_card_id),
                None,
            )
            if selected_card is None:
                print(
                    f"[BOT_MODE] lb_extra_refine 카드 없음: bot={bot_name!r}, "
                    f"character={char_name!r}, card={selected_card_id!r}"
                )
                return {
                    "success": False,
                    "error": f"캐릭터 카드를 찾을 수 없습니다: {selected_card_id}",
                }
            rep_images = selected_card.get("rep_images", [])
        else:
            rep_images = char.get("rep_images", [])
        if not rep_images:
            print(
                f"[BOT_MODE] lb_extra_refine 대표 이미지 없음: bot={bot_name!r}, "
                f"character={char_name!r}, card={selected_card_id!r}"
            )
            return {"success": False, "error": "대표 이미지(rep_images)가 없습니다."}
        rep0 = rep_images[0]
        if (
            not isinstance(rep0, str)
            or rep0 != os.path.basename(rep0)
            or "/" in rep0
            or "\\" in rep0
            or rep0 in {".", ".."}
        ):
            print(
                f"[BOT_MODE] lb_extra_refine 대표 이미지 경로 거부: "
                f"bot={bot_name!r}, character={char_name!r}, card={selected_card_id!r}, "
                f"image={rep0!r}"
            )
            return {"success": False, "error": "대표 이미지 파일명이 올바르지 않습니다."}
        img_path = os.path.join(char_dir, rep0)
        if not os.path.isfile(img_path):
            print(f"[BOT_MODE] 대표 이미지 파일 없음: {img_path}")
            return {"success": False, "error": f"대표 이미지 파일이 없습니다: {rep0}"}

        # 비전 서비스 확인 (외부 LLM 분기: primary LLM 기준)
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

        print(f"[BOT_MODE] lb_extra_refine 호출: bot={bot_name} char={char_name} "
              f"card={selected_card_id or '(root)'} image={rep0} service={service} "
              f"appearance={len(appearance_tags or [])} outfit={len(outfit_tags or [])} "
              f"etc={len(etc_tags or [])} use_custom={use_custom}")

        use_model = cfg.get("llm_model", "")
        await _notify_llm_widget("start", {"model": use_model, "prompt_id": f"lb_extra_refine:{char_name}"})

        raw = None
        last_err = None
        t0 = _time.time()
        try:
            raw = await callLLMVisionTask(
                "refine_lb_extra",
                messages,
                image_b64=img_b64,
                image_mime=mime,
                result_validator=lambda result: (
                    _parse_lb_extra_refine_response(result) is not None,
                    "외모/복장 태그 JSON 파싱 실패",
                ),
            )
        except Exception as call_err:
            print(f"[BOT_MODE] callLLMVision 예외: {call_err}")
            traceback.print_exc()
            last_err = f"{type(call_err).__name__}: {call_err}"
            raw = None
        total_elapsed = _time.time() - t0

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
                print(f"[BOT_MODE] lb_extra_refine 완료: appearance={len(parsed['appearance'])}개 outfit={len(parsed['outfit'])}개")
                return {"success": True, "data": parsed}
            last_err = f"LLM 응답을 JSON으로 파싱하지 못했습니다. raw: {raw[:300]}"
            print(f"[BOT_MODE] LLM 응답 JSON 파싱 실패(라우팅 재시도 소진). raw={raw}")
        else:
            last_err = last_err or f"LLM 호출 실패: {raw or '빈 응답'}"
            print(f"[BOT_MODE] LLM 호출 실패(라우팅 재시도 소진): {raw}")

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
        return {"success": False, "error": f"라우팅 재시도 후 실패: {last_err}"}
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
        visual_card_id = body.get("visual_card_id") or ""
        if not isinstance(appearance_tags, list) or not isinstance(outfit_tags, list) or not isinstance(etc_tags, list):
            print(
                f"[BOT_MODE] lb_extra_refine 태그 타입 오류: "
                f"appearance={type(appearance_tags).__name__}, "
                f"outfit={type(outfit_tags).__name__}, etc={type(etc_tags).__name__}"
            )
            return web.json_response({"success": False, "error": "appearance/outfit/etc 는 list 여야 합니다."})
        if not isinstance(visual_card_id, str):
            print(f"[BOT_MODE] lb_extra_refine visual_card_id 타입 오류: {visual_card_id!r}")
            return web.json_response({"success": False, "error": "visual_card_id는 문자열이어야 합니다."})
        result = await run_lb_extra_refine(
            bot_name,
            char_name,
            appearance_tags,
            outfit_tags,
            etc_tags,
            visual_card_id.strip(),
        )
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
        one_click_run_id = str(body.get("one_click_run_id") or "").strip()
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

        selected_names = {
            str(char_name).strip() for char_name in characters if str(char_name).strip()
        }
        requested_targets = body.get("visual_targets") or []
        requested_keys = None
        if requested_targets:
            if not isinstance(requested_targets, list):
                print(
                    f"[BOT_MODE] llm_batch_enqueue visual_targets 형식 오류: "
                    f"{type(requested_targets).__name__}"
                )
                return web.json_response({
                    "success": False,
                    "error": "visual_targets는 배열이어야 합니다.",
                })
            requested_keys = {
                (
                    str(target.get("character") or target.get("char_name") or "").strip(),
                    str(target.get("visual_card_id") or "").strip(),
                )
                for target in requested_targets
                if isinstance(target, dict)
            }
        visual_targets = [
            target
            for target in get_bot_visual_targets(bot_name, require_rep_images=True)
            if target["character"] in selected_names
            and (
                requested_keys is None
                or (target["character"], target["visual_card_id"]) in requested_keys
            )
        ]
        added = []
        for target in visual_targets:
            cn = target["character"]
            params = {
                "bot_name": bot_name,
                "char_name": cn,
                "visual_card_id": target["visual_card_id"],
                "visual_card_label": target["visual_card_label"],
                "visual_card_index": target["visual_card_index"],
            }
            if one_click_run_id:
                params["one_click_run_id"] = one_click_run_id
            item = await qm.add_item(
                item_type="bot_llm_face_tag_analysis",
                label=(
                    f"LLM 얼굴/눈 태그: {bot_name}/{cn}"
                    f"[{target['visual_card_index']}]"
                ),
                params=params,
                priority=10,
            )
            added.append({
                "char_name": cn,
                "visual_card_id": target["visual_card_id"],
                "visual_card_index": target["visual_card_index"],
                "id": item.id,
            })

        if not added:
            print(
                f"[BOT_MODE] LLM 일괄 분석 대상 없음: "
                f"bot={bot_name!r}, characters={sorted(selected_names)!r}"
            )

        print(f"[BOT_MODE] LLM 카드별 일괄 분석 큐 추가: bot={bot_name} {len(added)}건")
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
