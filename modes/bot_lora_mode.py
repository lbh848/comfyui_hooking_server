"""
Bot LoRA 매니징 모듈
- 봇 단위로 LoRA 학습 프로젝트 관리 (에셋의 엔트리와 동일 구조)
- 학습 이미지: 봇 캐릭터의 대표 이미지 + 얼굴 이미지를 프로젝트에 복사하여 관리
- 테스트 이미지: bot/<봇>/Lora/<프로젝트>/_test/ 에 저장
- 학습된 LoRA: <lora_load_path>/<봇>/Lora/<프로젝트>/<캐릭터>/ 에 저장
"""

import os
import json
import shutil
import traceback
from PIL import Image
from modes.lora_export_utils import format_lora_export_filename
from modes.lora_name_validation import validate_lora_project_name
from modes.visual_profiles import effective_character_cards

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT_DIR = os.path.join(BASE_DIR, "bot")
BOT_DATA_FILE = os.path.join(BASE_DIR, "asset_data", "bot.json")
BOT_LORA_MANAGE_FILE = os.path.join(BASE_DIR, "asset_data", "bot_lora_manage.json")

LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".bin"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TEST_DIR_NAME = "_test"


def _safe_dirname(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-', '.')).strip() or "unnamed"


def _bot_project_dir(bot_name: str, project_name: str) -> str:
    """봇 내 프로젝트 폴더 경로: bot/<봇>/Lora/<프로젝트>/"""
    return os.path.join(BOT_DIR, _safe_dirname(bot_name), "Lora", _safe_dirname(project_name))


def _bot_test_dir(bot_name: str, project_name: str) -> str:
    """프로젝트의 테스트 이미지 폴더: bot/<봇>/Lora/<프로젝트>/_test/"""
    return os.path.join(_bot_project_dir(bot_name, project_name), TEST_DIR_NAME)


def _bot_char_dir(bot_name: str, char_name: str) -> str:
    """봇 캐릭터 폴더 (학습 이미지 원본 위치)"""
    return os.path.join(BOT_DIR, _safe_dirname(bot_name), _safe_dirname(char_name))


def _bot_project_char_dir(bot_name: str, project_name: str, char_name: str) -> str:
    """프로젝트 내 캐릭터 폴더: bot/<봇>/Lora/<프로젝트>/<캐릭터>/"""
    return os.path.join(_bot_project_dir(bot_name, project_name), _safe_dirname(char_name))


def _profile_child_dir(base_dir: str, visual_card_id: str = "") -> str:
    """카드별 산출물을 캐릭터 루트와 충돌하지 않는 하위 폴더에 둔다."""
    card_id = str(visual_card_id or "").strip()
    if not card_id:
        return base_dir
    return os.path.join(base_dir, "_visual_profiles", _safe_dirname(card_id))


def _bot_project_training_dir(
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> str:
    return _profile_child_dir(
        _bot_project_char_dir(bot_name, project_name, char_name),
        visual_card_id,
    )


def _trained_lora_dir(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> str:
    """학습된 LoRA 경로. 새 카드 단위는 ``_visual_profiles/<ID>``에 격리한다."""
    char_dir = os.path.join(
        lora_load_path,
        _safe_dirname(bot_name),
        "Lora",
        _safe_dirname(project_name),
        _safe_dirname(char_name),
    )
    return _profile_child_dir(char_dir, visual_card_id)


def _profile_trigger(char_name: str, visual_card_id: str, *, is_primary: bool) -> str:
    """카드 종류와 무관한 캐릭터 공용 기본 트리거를 반환한다."""
    return char_name


def _bot_character(bot_data: dict, bot_name: str, char_name: str) -> dict | None:
    for bot in bot_data.get("bots", []):
        if bot.get("name") != bot_name:
            continue
        return next(
            (char for char in bot.get("characters", []) if char.get("name") == char_name),
            None,
        )
    return None


def _bot_visual_units(bot_data: dict, bot_name: str) -> list[dict]:
    """봇의 캐릭터 카드를 독립 LoRA 학습 단위로 펼친다."""
    bot = next((item for item in bot_data.get("bots", []) if item.get("name") == bot_name), None)
    if not bot:
        print(f"[BOT_LORA_PROFILE] 봇을 찾을 수 없음: {bot_name!r}")
        return []

    units = []
    for character in bot.get("characters", []):
        char_name = str(character.get("name") or "").strip()
        if not char_name:
            print(f"[BOT_LORA_PROFILE] 이름 없는 캐릭터 스킵: bot={bot_name!r}")
            continue
        try:
            cards, source = effective_character_cards(character, None)
        except Exception as e:
            print(
                f"[BOT_LORA_PROFILE] 캐릭터 카드 해석 실패: "
                f"bot={bot_name!r}, character={char_name!r}, error={e}"
            )
            traceback.print_exc()
            continue
        for index, card in enumerate(cards):
            card_id = str(card.get("id") or "").strip()
            if not card_id:
                print(
                    f"[BOT_LORA_PROFILE] ID 없는 카드 스킵: "
                    f"bot={bot_name!r}, character={char_name!r}, index={index}"
                )
                continue
            is_primary = index == 0
            artifact_dir = (
                _bot_char_dir(bot_name, char_name)
                if is_primary
                else os.path.join(
                    _bot_char_dir(bot_name, char_name),
                    "_visual_profiles",
                    _safe_dirname(card_id),
                )
            )
            units.append({
                "name": char_name,
                "visual_card_id": card_id,
                "visual_card_label": str(card.get("label") or f"카드 {index + 1}"),
                "visual_card_index": index + 1,
                "is_primary": is_primary,
                "source": source,
                "rep_images": list(card.get("rep_images") or []),
                "gender_tag": str(
                    card.get("gender_tag") or character.get("gender_tag") or ""
                ).strip(),
                "has_face_image": os.path.isfile(
                    os.path.join(artifact_dir, "_face_image.webp")
                ),
            })
    return units


def _project_profile_config(char_cfg: dict, visual_card_id: str = "") -> dict | None:
    """새 프로필 설정을 반환하고, 기존 캐릭터 설정은 카드 미지정 호출로 유지한다."""
    if not isinstance(char_cfg, dict):
        return None
    card_id = str(visual_card_id or "").strip()
    profiles = char_cfg.get("profiles")
    if isinstance(profiles, dict):
        if not card_id and "" in profiles:
            profile_cfg = profiles.get("")
            return profile_cfg if isinstance(profile_cfg, dict) else None
        if card_id:
            profile_cfg = profiles.get(card_id)
            return profile_cfg if isinstance(profile_cfg, dict) else None
        if len(profiles) == 1:
            only = next(iter(profiles.values()))
            return only if isinstance(only, dict) else None
        return None
    return char_cfg if not card_id else None


def _iter_project_units(project: dict):
    """새 카드별 설정과 레거시 캐릭터 설정을 동일한 형태로 순회한다."""
    for char_name, char_cfg in (project.get("characters") or {}).items():
        if not isinstance(char_cfg, dict):
            print(f"[BOT_LORA_PROFILE] 잘못된 캐릭터 설정 스킵: {char_name!r}")
            continue
        profiles = char_cfg.get("profiles")
        if isinstance(profiles, dict):
            for card_id, profile_cfg in profiles.items():
                if isinstance(profile_cfg, dict):
                    yield char_name, str(card_id), profile_cfg
                else:
                    print(
                        f"[BOT_LORA_PROFILE] 잘못된 카드 설정 스킵: "
                        f"character={char_name!r}, card={card_id!r}"
                    )
            continue
        yield char_name, "", char_cfg


def _effective_character_trigger(
    project: dict,
    char_name: str,
    visual_card_id: str = "",
    profile_cfg: dict | None = None,
) -> str:
    """캐릭터의 공용 트리거를 반환한다.

    새 카드별 스키마는 캐릭터 래퍼의 ``trigger``를 사용한다. 기존 데이터처럼
    카드마다 trigger가 들어 있으면 기본 카드(visual_card_index=1)의 값을
    캐릭터 공용값으로 간주하고, 기본 카드 표시가 없을 때만 첫 카드 값으로
    폴백한다. 이 호환 조회는 파일 자체를 자동 마이그레이션하지 않는다.
    """
    char_cfg = (project.get("characters") or {}).get(char_name)
    if not isinstance(char_cfg, dict):
        print(f"[BOT_LORA_TRIGGER] 캐릭터 설정 없음: character={char_name!r}")
        return char_name

    profiles = char_cfg.get("profiles")
    if not isinstance(profiles, dict):
        return str(char_cfg.get("trigger") or char_name).strip() or char_name

    shared_trigger = str(char_cfg.get("trigger") or "").strip()
    if shared_trigger:
        return shared_trigger

    candidates = [
        cfg for cfg in profiles.values()
        if isinstance(cfg, dict)
    ]
    primary_cfg = next(
        (cfg for cfg in candidates if cfg.get("visual_card_index", 0) == 1),
        None,
    )
    if primary_cfg is None and isinstance(profiles.get(""), dict):
        primary_cfg = profiles[""]
    if primary_cfg is None and candidates:
        primary_cfg = candidates[0]

    legacy_trigger = str((primary_cfg or {}).get("trigger") or "").strip()
    if legacy_trigger:
        return legacy_trigger

    # 잘못된/부분 데이터에서 대상 카드에만 값이 있다면 마지막 호환 폴백으로 쓴다.
    target_cfg = profile_cfg or _project_profile_config(char_cfg, visual_card_id)
    target_trigger = str((target_cfg or {}).get("trigger") or "").strip()
    return target_trigger or char_name


def _set_character_trigger(project: dict, char_name: str, trigger: str) -> bool:
    """레거시/카드별 스키마 모두에서 캐릭터 공용 트리거를 설정한다."""
    char_cfg = (project.get("characters") or {}).get(char_name)
    if not isinstance(char_cfg, dict):
        print(f"[BOT_LORA_TRIGGER] 업데이트 대상 캐릭터 없음: {char_name!r}")
        return False
    char_cfg["trigger"] = str(trigger or "").strip() or char_name
    return True


def _selection_keys(values, available_units: list[dict]) -> set[tuple[str, str]]:
    """API의 명시적 카드 선택과 레거시 캐릭터명 선택을 정규화한다."""
    result: set[tuple[str, str]] = set()
    by_character: dict[str, list[dict]] = {}
    for unit in available_units:
        by_character.setdefault(unit["name"], []).append(unit)
    for value in values or []:
        if isinstance(value, dict):
            char_name = str(value.get("character") or value.get("name") or "").strip()
            card_id = str(value.get("visual_card_id") or "").strip()
            if char_name and card_id:
                result.add((char_name, card_id))
            else:
                print(f"[BOT_LORA_PROFILE] 잘못된 카드 선택 스킵: {value!r}")
            continue
        char_name = str(value or "").strip()
        if not char_name:
            print(f"[BOT_LORA_PROFILE] 빈 캐릭터 선택 스킵: {value!r}")
            continue
        candidates = by_character.get(char_name) or []
        if not candidates:
            print(f"[BOT_LORA_PROFILE] 선택 캐릭터를 찾을 수 없음: {char_name!r}")
            continue
        # 레거시 호출은 기존 의미대로 캐릭터의 기본 카드 한 장만 선택한다.
        primary = next((unit for unit in candidates if unit["is_primary"]), candidates[0])
        result.add((char_name, primary["visual_card_id"]))
    return result


def _unit_key(unit: dict) -> tuple[str, str]:
    return unit["name"], unit["visual_card_id"]


def _project_has_unit(
    project: dict,
    char_name: str,
    visual_card_id: str,
    is_primary: bool = False,
) -> bool:
    char_cfg = (project.get("characters") or {}).get(char_name)
    if not isinstance(char_cfg, dict):
        return False
    profiles = char_cfg.get("profiles")
    if isinstance(profiles, dict):
        if visual_card_id in profiles:
            return True
        if is_primary or not visual_card_id:
            if "" in profiles:
                return True
            return any(
                isinstance(cfg, dict) and cfg.get("visual_card_index", 1) == 1
                for cfg in profiles.values()
            )
        return False
    # 레거시 캐릭터 설정은 기본 카드 한 장으로 간주한다.
    return not visual_card_id or is_primary or visual_card_id == "card_1"


def _add_project_unit(project: dict, unit: dict) -> dict:
    char_name = unit["name"]
    card_id = unit["visual_card_id"]
    characters = project.setdefault("characters", {})
    char_cfg = characters.get(char_name)
    if isinstance(char_cfg, dict) and not isinstance(char_cfg.get("profiles"), dict):
        # 기존 캐릭터 단위 설정과 새 카드 단위 설정이 공존할 때 레거시 설정을
        # 빈 카드 ID 프로필로 감싸 데이터 손실 없이 전환한다.
        legacy_cfg = char_cfg
        char_cfg = {
            "trigger": str(legacy_cfg.get("trigger") or char_name).strip() or char_name,
            "profiles": {"": legacy_cfg},
        }
        characters[char_name] = char_cfg
    elif not isinstance(char_cfg, dict):
        char_cfg = {
            "trigger": _profile_trigger(
                char_name,
                card_id,
                is_primary=bool(unit["is_primary"]),
            ),
            "profiles": {},
        }
        characters[char_name] = char_cfg
    elif not str(char_cfg.get("trigger") or "").strip():
        # 기존 카드별 trigger 데이터는 자동 삭제하지 않고 공용값만 승격한다.
        char_cfg["trigger"] = _effective_character_trigger(
            project,
            char_name,
            card_id,
        )
    profiles = char_cfg.setdefault("profiles", {})
    profile_cfg = profiles.setdefault(card_id, {})
    profile_cfg["label"] = unit["visual_card_label"]
    profile_cfg["visual_card_index"] = unit["visual_card_index"]
    return profile_cfg


# ─── 학습 이미지 프로젝트 동기화 ───────────────────────────────

def _sync_training_images_to_project(
    bot_name: str,
    project_name: str,
    char_name: str,
    rep_images: list,
    include_face: bool = True,
    visual_card_id: str = "",
    is_primary: bool = True,
) -> dict:
    """한 캐릭터 카드의 대표/FACE 이미지를 독립 학습 폴더로 복사한다."""
    char_src_dir = _bot_char_dir(bot_name, char_name)
    dst_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    if not os.path.isdir(char_src_dir):
        print(f"[BOT_LORA_SYNC] 원본 캐릭터 폴더 없음: {char_src_dir}")
        return {"synced": 0, "skipped": 0}

    os.makedirs(dst_dir, exist_ok=True)

    # 대표 이미지는 캐릭터 루트, FACE 이미지는 카드별 유틸리티 폴더에서 가져온다.
    files_to_copy = []
    for fname in rep_images:
        files_to_copy.append((char_src_dir, fname))
    face_src_dir = (
        char_src_dir
        if is_primary or not visual_card_id
        else os.path.join(
            char_src_dir, "_visual_profiles", _safe_dirname(visual_card_id)
        )
    )
    if include_face and os.path.isfile(os.path.join(face_src_dir, "_face_image.webp")):
        files_to_copy.append((face_src_dir, "_face_image.webp"))

    synced = 0
    skipped = 0
    prompts_synced = 0
    for src_dir, fname in files_to_copy:
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path):
            print(f"[BOT_LORA_SYNC] 원본 파일 없음: {src_path}")
            continue

        dst_path = os.path.join(dst_dir, fname)
        # 이미지: 이미 존재하면 스킵 (사용자가 편집했을 수 있음)
        if not os.path.isfile(dst_path):
            try:
                shutil.copy2(src_path, dst_path)
                synced += 1
            except Exception as e:
                print(f"[BOT_LORA_SYNC] 이미지 복사 실패: {src_path} -> {dst_path} - {e}")
        else:
            skipped += 1

        # 프롬프트 JSON도 복사 (이미지 스킵과 별개로 항상 체크)
        base = os.path.splitext(fname)[0]
        prompt_src = os.path.join(src_dir, f"{base}_prompt.json")
        prompt_dst = os.path.join(dst_dir, f"{base}_prompt.json")
        if os.path.isfile(prompt_dst):
            continue
        if os.path.isfile(prompt_src):
            try:
                with open(prompt_src, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                # "prompt" 키를 "positive"로 통일
                if "positive" not in pdata and "prompt" in pdata:
                    pdata["positive"] = pdata.pop("prompt")
                # 원본 보존
                if "original_positive" not in pdata:
                    pdata["original_positive"] = pdata.get("positive", "")
                if "original_negative" not in pdata:
                    pdata["original_negative"] = pdata.get("negative", "")
                with open(prompt_dst, "w", encoding="utf-8") as f:
                    json.dump(pdata, f, ensure_ascii=False, indent=2)
                prompts_synced += 1
            except Exception as e:
                print(f"[BOT_LORA_SYNC] 프롬프트 복사 실패: {prompt_src} -> {prompt_dst} - {e}")
        else:
            try:
                with open(prompt_dst, "w", encoding="utf-8") as f:
                    json.dump({"positive": "", "negative": "", "original_positive": "", "original_negative": ""}, f, ensure_ascii=False, indent=2)
                prompts_synced += 1
            except Exception as e:
                print(f"[BOT_LORA_SYNC] 빈 프롬프트 생성 실패: {prompt_dst} - {e}")

    print(
        f"[BOT_LORA_SYNC] 완료: {bot_name}/{project_name}/{char_name}/"
        f"{visual_card_id or 'legacy'} - 이미지 복사:{synced}, "
        f"스킵:{skipped}, 프롬프트:{prompts_synced}"
    )
    return {"synced": synced, "skipped": skipped, "prompts_synced": prompts_synced}


# ─── 데이터 관리 ─────────────────────────────────────────────

def _load_bot_data() -> dict:
    if not os.path.isfile(BOT_DATA_FILE):
        print("[BOT_LORA] bot.json 없음")
        return {"bots": []}
    try:
        with open(BOT_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[BOT_LORA] bot.json 로드 실패: {e}")
        traceback.print_exc()
        return {"bots": []}


def _load_bot_lora_manage() -> dict:
    if os.path.isfile(BOT_LORA_MANAGE_FILE):
        try:
            with open(BOT_LORA_MANAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[BOT_LORA_MANAGE] 로드 실패: {e}")
            traceback.print_exc()
    return {"bot_loras": {}}


def _save_bot_lora_manage(data: dict):
    try:
        with open(BOT_LORA_MANAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("[BOT_LORA_MANAGE] 저장 완료")
    except Exception as e:
        print(f"[BOT_LORA_MANAGE] 저장 실패: {e}")
        traceback.print_exc()


def _get_project_config(data: dict, bot_name: str, project_name: str) -> dict | None:
    """프로젝트 설정 조회"""
    return data.get("bot_loras", {}).get(bot_name, {}).get(project_name)


def _get_char_config(
    data: dict,
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> dict | None:
    """프로젝트 내 카드별 설정 조회. 카드 미지정은 레거시와 단일 카드만 허용한다."""
    proj = _get_project_config(data, bot_name, project_name)
    if not proj:
        return None
    return _project_profile_config(
        proj.get("characters", {}).get(char_name),
        visual_card_id,
    )


# ─── 캐릭터 임포트 ──────────────────────────────────────────

def list_importable_characters(bot_name: str, project_name: str) -> dict:
    """봇에는 있지만 프로젝트에는 없는 캐릭터 카드 학습 단위 목록 반환."""
    if not bot_name or not project_name:
        print("[BOT_LORA_IMPORT] 봇/프로젝트 이름 누락")
        return {"success": False, "error": "봇/프로젝트 이름 필수"}

    bot_data = _load_bot_data()
    bot_units = _bot_visual_units(bot_data, bot_name)

    manage_data = _load_bot_lora_manage()
    proj = _get_project_config(manage_data, bot_name, project_name)
    importable = [
        unit for unit in bot_units
        if not proj or not _project_has_unit(
            proj, unit["name"], unit["visual_card_id"], unit["is_primary"]
        )
    ]
    existing_count = len(bot_units) - len(importable)
    print(
        f"[BOT_LORA_IMPORT] 임포트 가능 캐릭터 카드: "
        f"{len(importable)}장 (기존 {existing_count}장)"
    )
    return {"success": True, "characters": importable}


def import_characters(
    bot_name: str,
    project_name: str,
    char_names: list,
    face_chars: list | None = None,
) -> dict:
    """선택한 캐릭터 카드를 프로젝트에 독립 학습 단위로 추가한다.

    새 호출은 ``{character, visual_card_id}`` 객체 배열을 사용한다. 문자열 배열은
    레거시 호환을 위해 해당 캐릭터의 기본 카드로 해석한다.
    """
    if not bot_name or not project_name:
        print("[BOT_LORA_IMPORT] 봇/프로젝트 이름 누락")
        return {"success": False, "error": "봇/프로젝트 이름 필수"}

    if char_names is None:
        char_names = []
    if face_chars is None:
        face_chars = []

    bot_data = _load_bot_data()
    bot_units = _bot_visual_units(bot_data, bot_name)
    rep_keys = _selection_keys(char_names, bot_units)
    face_keys = _selection_keys(face_chars, bot_units)
    add_keys = rep_keys | face_keys
    if not add_keys:
        print("[BOT_LORA_IMPORT] 선택된 캐릭터/얼굴 없음")
        return {"success": False, "error": "임포트할 캐릭터 카드를 선택하세요"}

    manage_data = _load_bot_lora_manage()
    proj = _get_project_config(manage_data, bot_name, project_name)
    if not proj:
        print(f"[BOT_LORA_IMPORT] 프로젝트 없음: {bot_name}/{project_name}")
        return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

    added = []
    for unit in bot_units:
        key = _unit_key(unit)
        if key not in add_keys or _project_has_unit(proj, *key, unit["is_primary"]):
            continue
        _add_project_unit(proj, unit)
        include_face = key in face_keys
        rep_imgs = unit["rep_images"] if key in rep_keys else []
        _sync_training_images_to_project(
            bot_name,
            project_name,
            unit["name"],
            rep_imgs,
            include_face,
            unit["visual_card_id"],
            unit["is_primary"],
        )
        if key not in rep_keys:
            print(
                f"[BOT_LORA_IMPORT] face-only 카드 임포트: "
                f"{bot_name}/{project_name}/{unit['name']}/{unit['visual_card_id']}"
            )
        added.append({
            "character": unit["name"],
            "visual_card_id": unit["visual_card_id"],
        })

    if added:
        _save_bot_lora_manage(manage_data)
        print(f"[BOT_LORA_IMPORT] 캐릭터 카드 임포트 완료: {added}")
    else:
        print("[BOT_LORA_IMPORT] 임포트할 새 캐릭터가 없음")

    return {"success": True, "added": added, "count": len(added)}


# ─── 프로젝트 간 캐릭터 임포트 ────────────────────────────────

def list_project_importable_characters(src_bot: str, src_project: str, dst_bot: str, dst_project: str) -> dict:
    """소스 프로젝트에는 있지만 대상(현재) 프로젝트에는 없는 캐릭터 목록 반환.
    각 캐릭터의 프로젝트 학습 이미지 수를 함께 반환한다."""
    if not src_bot or not src_project or not dst_bot or not dst_project:
        print("[BOT_LORA_PROJ_IMPORT] 필수 파라미터 누락")
        return {"success": False, "error": "소스/대상 봇·프로젝트 이름 필수"}

    manage_data = _load_bot_lora_manage()
    src_proj = _get_project_config(manage_data, src_bot, src_project)
    if not src_proj:
        print(f"[BOT_LORA_PROJ_IMPORT] 소스 프로젝트 없음: {src_bot}/{src_project}")
        return {"success": False, "error": "소스 프로젝트를 찾을 수 없습니다"}

    importable = []
    dst_proj = _get_project_config(manage_data, dst_bot, dst_project)
    for cn, card_id, ccfg in _iter_project_units(src_proj):
        if not cn or (
            dst_proj and _project_has_unit(
                dst_proj,
                cn,
                card_id,
                ccfg.get("visual_card_index", 1) == 1,
            )
        ):
            continue
        image_count = len(
            _get_project_training_images(src_bot, src_project, cn, card_id)
        )
        importable.append({
            "name": cn,
            "visual_card_id": card_id,
            "visual_card_label": ccfg.get("label") or card_id or "기본 카드",
            "trigger": _effective_character_trigger(src_proj, cn, card_id, ccfg),
            "image_count": image_count,
        })

    print(f"[BOT_LORA_PROJ_IMPORT] 임포트 가능 캐릭터 카드: {len(importable)}장")
    return {"success": True, "characters": importable}


def import_characters_from_project(src_bot: str, src_project: str, dst_bot: str, dst_project: str, char_names: list) -> dict:
    """소스 프로젝트의 캐릭터를 대상(현재) 프로젝트로 복사.
    - 학습 이미지 폴더를 통째로 복사(편집/정제 프롬프트 포함)
    - 캐릭터 설정은 trigger만 복사(skip_training, session_representatives 등은 버림)
    """
    if not src_bot or not src_project or not dst_bot or not dst_project:
        print("[BOT_LORA_PROJ_IMPORT] 필수 파라미터 누락")
        return {"success": False, "error": "소스/대상 봇·프로젝트 이름 필수"}
    if not char_names:
        print("[BOT_LORA_PROJ_IMPORT] 선택된 캐릭터 없음")
        return {"success": False, "error": "임포트할 캐릭터를 선택하세요"}

    manage_data = _load_bot_lora_manage()
    src_proj = _get_project_config(manage_data, src_bot, src_project)
    if not src_proj:
        print(f"[BOT_LORA_PROJ_IMPORT] 소스 프로젝트 없음: {src_bot}/{src_project}")
        return {"success": False, "error": "소스 프로젝트를 찾을 수 없습니다"}

    dst_proj = _get_project_config(manage_data, dst_bot, dst_project)
    if not dst_proj:
        print(f"[BOT_LORA_PROJ_IMPORT] 대상 프로젝트 없음: {dst_bot}/{dst_project}")
        return {"success": False, "error": "대상 프로젝트를 찾을 수 없습니다"}

    source_units = {
        (char_name, card_id): cfg
        for char_name, card_id, cfg in _iter_project_units(src_proj)
    }
    selected_keys = set()
    for value in char_names:
        if isinstance(value, dict):
            char_name = str(value.get("character") or value.get("name") or "").strip()
            card_id = str(value.get("visual_card_id") or "").strip()
            if char_name:
                selected_keys.add((char_name, card_id))
            else:
                print(f"[BOT_LORA_PROJ_IMPORT] 잘못된 선택 스킵: {value!r}")
            continue
        char_name = str(value or "").strip()
        candidate = next(
            (key for key in source_units if key[0] == char_name),
            None,
        )
        if candidate:
            selected_keys.add(candidate)

    added = []

    for cn, card_id in selected_keys:
        src_cfg = source_units.get((cn, card_id))
        if src_cfg is None:
            print(f"[BOT_LORA_PROJ_IMPORT] 스킵(소스에 없음): {cn}/{card_id}")
            continue
        if _project_has_unit(
            dst_proj,
            cn,
            card_id,
            src_cfg.get("visual_card_index", 1) == 1,
        ):
            print(f"[BOT_LORA_PROJ_IMPORT] 스킵(이미 존재): {cn}/{card_id}")
            continue

        dst_character_existed = isinstance(
            (dst_proj.get("characters") or {}).get(cn),
            dict,
        )
        _add_project_unit(dst_proj, {
            "name": cn,
            "visual_card_id": card_id,
            "visual_card_label": src_cfg.get("label") or card_id or "기본 카드",
            "visual_card_index": src_cfg.get("visual_card_index", 1),
            "is_primary": not card_id or src_cfg.get("visual_card_index", 1) == 1,
        })
        if not dst_character_existed:
            _set_character_trigger(
                dst_proj,
                cn,
                _effective_character_trigger(src_proj, cn, card_id, src_cfg),
            )

        # 학습 이미지 폴더 통째 복사 (이미지 + *_prompt.json)
        src_dir = _bot_project_training_dir(src_bot, src_project, cn, card_id)
        dst_dir = _bot_project_training_dir(dst_bot, dst_project, cn, card_id)
        if os.path.isdir(src_dir):
            try:
                os.makedirs(_bot_project_dir(dst_bot, dst_project), exist_ok=True)
                if not os.path.exists(dst_dir):
                    shutil.copytree(src_dir, dst_dir)
                    print(f"[BOT_LORA_PROJ_IMPORT] 폴더 복사: {src_dir} -> {dst_dir}")
                else:
                    # 폴더가 이미 존재하면 개별 파일 단위 복사(덮어쓰지 않음)
                    copied = 0
                    for root, dirnames, filenames in os.walk(src_dir):
                        # 레거시 카드 루트 아래의 다른 카드 산출물까지 함께 가져오지 않는다.
                        dirnames[:] = [
                            name for name in dirnames if name != "_visual_profiles"
                        ]
                        rel_dir = os.path.relpath(root, src_dir)
                        out_dir = (
                            dst_dir if rel_dir == "."
                            else os.path.join(dst_dir, rel_dir)
                        )
                        os.makedirs(out_dir, exist_ok=True)
                        for fname in filenames:
                            s = os.path.join(root, fname)
                            d = os.path.join(out_dir, fname)
                            if not os.path.exists(d):
                                shutil.copy2(s, d)
                                copied += 1
                    print(f"[BOT_LORA_PROJ_IMPORT] 폴더 이미 존재, 파일 복사 {copied}건: {dst_dir}")
            except Exception as e:
                print(f"[BOT_LORA_PROJ_IMPORT] 폴더 복사 실패: {src_dir} -> {dst_dir} - {e}")
                traceback.print_exc()
        else:
            print(f"[BOT_LORA_PROJ_IMPORT] 소스 캐릭터 폴더 없음(설정만 추가): {src_dir}")

        added.append({"character": cn, "visual_card_id": card_id})

    if added:
        _save_bot_lora_manage(manage_data)
        print(f"[BOT_LORA_PROJ_IMPORT] 캐릭터 임포트 완료: {added}")
    else:
        print("[BOT_LORA_PROJ_IMPORT] 임포트할 새 캐릭터가 없음")

    return {"success": True, "added": added, "count": len(added)}


def remove_character_from_project(
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> dict:
    """프로젝트에서 캐릭터 카드 학습 단위를 제거한다."""
    if not bot_name or not project_name or not char_name:
        print("[BOT_LORA_REMOVE] 필수 파라미터 누락")
        return {"success": False, "error": "봇/프로젝트/캐릭터 이름 필수"}

    manage_data = _load_bot_lora_manage()
    proj = _get_project_config(manage_data, bot_name, project_name)
    if not proj:
        print(f"[BOT_LORA_REMOVE] 프로젝트 없음: {bot_name}/{project_name}")
        return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

    characters = proj.get("characters", {})
    if char_name not in characters:
        print(f"[BOT_LORA_REMOVE] 캐릭터 없음: {char_name}")
        return {"success": False, "error": f"캐릭터 '{char_name}'가 프로젝트에 없습니다"}

    char_cfg = characters[char_name]
    profiles = char_cfg.get("profiles") if isinstance(char_cfg, dict) else None
    preserve_nested_profiles = False
    if isinstance(profiles, dict):
        profile_key = visual_card_id
        if profile_key not in profiles:
            print(f"[BOT_LORA_REMOVE] 카드 없음: {char_name}/{profile_key or 'legacy'}")
            return {"success": False, "error": "캐릭터 카드가 프로젝트에 없습니다"}
        del profiles[profile_key]
        if not profiles:
            del characters[char_name]
        elif not visual_card_id:
            preserve_nested_profiles = True
    elif visual_card_id:
        print(f"[BOT_LORA_REMOVE] 레거시 캐릭터에 카드 ID 지정됨: {char_name}/{visual_card_id}")
        return {"success": False, "error": "레거시 캐릭터에는 카드 ID를 지정할 수 없습니다"}
    else:
        del characters[char_name]
    _save_bot_lora_manage(manage_data)

    # 프로젝트 내 캐릭터 폴더 삭제 (학습 이미지, 캐릭터별 테스트 이미지)
    char_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    if os.path.isdir(char_dir):
        try:
            if preserve_nested_profiles:
                for name in os.listdir(char_dir):
                    path = os.path.join(char_dir, name)
                    if name == "_visual_profiles":
                        continue
                    if os.path.isfile(path) or os.path.islink(path):
                        os.remove(path)
                    elif name == TEST_DIR_NAME and os.path.isdir(path):
                        shutil.rmtree(path)
                print(f"[BOT_LORA_REMOVE] 레거시 카드 파일만 삭제: {char_dir}")
            else:
                shutil.rmtree(char_dir)
                print(f"[BOT_LORA_REMOVE] 캐릭터 폴더 삭제: {char_dir}")
        except Exception as e:
            print(f"[BOT_LORA_REMOVE] 캐릭터 폴더 삭제 실패: {char_dir} - {e}")

    print(
        f"[BOT_LORA_REMOVE] 캐릭터 카드 제거 완료: "
        f"{bot_name}/{project_name}/{char_name}/{visual_card_id or 'legacy'}"
    )
    return {"success": True}


# ─── 봇 목록 ─────────────────────────────────────────────────

def list_bots() -> list:
    """LoRA 학습 가능한 봇 목록 반환"""
    bot_data = _load_bot_data()
    result = []
    for bot in bot_data.get("bots", []):
        chars_by_name = {}
        for unit in _bot_visual_units(bot_data, bot.get("name", "")):
            entry = chars_by_name.setdefault(unit["name"], {
                "name": unit["name"],
                "profiles": [],
            })
            entry["profiles"].append(unit)
        result.append({
            "name": bot.get("name", ""),
            "characters": list(chars_by_name.values()),
            "visual_profile_count": sum(
                len(item["profiles"]) for item in chars_by_name.values()
            ),
        })
    return result



# ─── 프로젝트 관리 ────────────────────────────────────────────

def list_projects(bot_name: str) -> list:
    """봇의 학습 프로젝트 목록 반환"""
    data = _load_bot_lora_manage()
    projects = []
    for pname, pinfo in data.get("bot_loras", {}).get(bot_name, {}).items():
        if not isinstance(pinfo, dict):
            continue
        char_count = len(pinfo.get("characters", {}))
        profile_count = sum(1 for _ in _iter_project_units(pinfo))
        training_config = pinfo.get("training_config", {})
        projects.append({
            "name": pname,
            "character_count": char_count,
            "profile_count": profile_count,
            "profile": training_config.get("profile", "anima"),
        })
    return projects


def add_project(
    bot_name: str,
    project_name: str,
    selected_chars: list | None = None,
    face_chars: list | None = None,
) -> dict:
    """새 카드별 독립 학습 프로젝트 추가.

    선택 배열은 ``{character, visual_card_id}`` 객체를 권장하며 문자열은 기본
    카드 선택으로 호환한다.
    """
    if not bot_name:
        return {"success": False, "error": "봇 이름 누락"}
    if not project_name or not project_name.strip():
        return {"success": False, "error": "프로젝트 이름을 입력하세요"}

    project_name = project_name.strip()
    name_error = validate_lora_project_name(project_name)
    if name_error:
        print(f"[BOT_LORA] 잘못된 프로젝트명: name={project_name!r}, error={name_error}")
        return {"success": False, "error": name_error}
    data = _load_bot_lora_manage()

    bot_projects = data.setdefault("bot_loras", {}).setdefault(bot_name, {})
    if project_name in bot_projects:
        print(f"[BOT_LORA] 프로젝트 이미 존재: {bot_name}/{project_name}")
        return {"success": False, "error": "이미 존재하는 프로젝트명입니다"}

    # 프로젝트 폴더 생성
    project_dir = _bot_project_dir(bot_name, project_name)
    os.makedirs(project_dir, exist_ok=True)

    if selected_chars is None:
        selected_chars = []
    if face_chars is None:
        face_chars = []

    bot_data = _load_bot_data()
    bot_units = _bot_visual_units(bot_data, bot_name)
    rep_keys = _selection_keys(selected_chars, bot_units)
    face_keys = _selection_keys(face_chars, bot_units)
    add_keys = rep_keys | face_keys

    bot_projects[project_name] = {
        "training_config": {},
        "characters": {},
    }
    project = bot_projects[project_name]
    selected_units = []
    for unit in bot_units:
        if _unit_key(unit) not in add_keys:
            continue
        _add_project_unit(project, unit)
        selected_units.append(unit)
    _save_bot_lora_manage(data)

    # 프로젝트 생성 시에만 학습 이미지 동기화
    for unit in selected_units:
        key = _unit_key(unit)
        _sync_training_images_to_project(
            bot_name,
            project_name,
            unit["name"],
            unit["rep_images"] if key in rep_keys else [],
            key in face_keys,
            unit["visual_card_id"],
            unit["is_primary"],
        )
        if key not in rep_keys:
            print(
                f"[BOT_LORA] face-only 카드 추가: "
                f"{bot_name}/{project_name}/{unit['name']}/{unit['visual_card_id']}"
            )

    print(
        f"[BOT_LORA] 프로젝트 추가: {bot_name}/{project_name} "
        f"({len(selected_units)}개 캐릭터 카드)"
    )
    return {"success": True, "name": project_name}


def duplicate_project(bot_name: str, src_project_name: str, dst_project_name: str, lora_load_path: str = "") -> dict:
    """학습 프로젝트 복제 (학습 데이터, 설정만 복제. 학습된 LoRA는 복제하지 않음)"""
    if not bot_name:
        return {"success": False, "error": "봇 이름 누락"}
    if not src_project_name or not dst_project_name:
        return {"success": False, "error": "원본/대상 프로젝트 이름 누락"}

    dst_project_name = dst_project_name.strip()
    if not dst_project_name:
        return {"success": False, "error": "프로젝트 이름을 입력하세요"}
    name_error = validate_lora_project_name(dst_project_name)
    if name_error:
        print(f"[BOT_LORA] 복제: 잘못된 프로젝트명 - name={dst_project_name!r}, error={name_error}")
        return {"success": False, "error": name_error}

    data = _load_bot_lora_manage()
    bot_projects = data.setdefault("bot_loras", {}).setdefault(bot_name, {})

    # 원본 프로젝트 확인
    src_cfg = bot_projects.get(src_project_name)
    if not src_cfg:
        print(f"[BOT_LORA] 원본 프로젝트 없음: {bot_name}/{src_project_name}")
        return {"success": False, "error": "원본 프로젝트를 찾을 수 없습니다"}

    # 대상 프로젝트 이름 중복 확인
    if dst_project_name in bot_projects:
        print(f"[BOT_LORA] 대상 프로젝트 이미 존재: {bot_name}/{dst_project_name}")
        return {"success": False, "error": "이미 존재하는 프로젝트명입니다"}

    # 프로젝트 폴더 복제
    src_dir = _bot_project_dir(bot_name, src_project_name)
    dst_dir = _bot_project_dir(bot_name, dst_project_name)
    if os.path.isdir(src_dir):
        try:
            shutil.copytree(src_dir, dst_dir)
            print(f"[BOT_LORA] 프로젝트 폴더 복제: {src_dir} -> {dst_dir}")
        except Exception as e:
            print(f"[BOT_LORA] 프로젝트 폴더 복제 실패: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"프로젝트 폴더 복제 실패: {e}"}
    else:
        os.makedirs(dst_dir, exist_ok=True)

    # JSON 설정 복제 (session_representatives는 학습된 LoRA 참조이므로 제외)
    import copy
    dst_cfg = copy.deepcopy(src_cfg)

    # 학습된 LoRA 참조 제거
    for _char_name, card_id, unit_cfg in _iter_project_units(dst_cfg):
        unit_cfg.pop("session_representatives", None)
        unit_cfg.pop("session_priority", None)

    # lora_save_path를 새 프로젝트 기준으로 재생성
    training_config = dst_cfg.get("training_config", {})
    training_config["lora_save_path"] = f"SOYA_BOT_LORA/{_safe_dirname(bot_name)}/Lora/{_safe_dirname(dst_project_name)}"

    bot_projects[dst_project_name] = dst_cfg
    _save_bot_lora_manage(data)

    print(f"[BOT_LORA] 프로젝트 복제 완료: {bot_name}/{src_project_name} -> {dst_project_name}")
    return {"success": True, "name": dst_project_name}


def remove_project(bot_name: str, project_name: str, lora_load_path: str = "") -> dict:
    """학습 프로젝트 삭제"""
    if not bot_name or not project_name:
        return {"success": False, "error": "봇/프로젝트 이름 누락"}

    data = _load_bot_lora_manage()
    bot_projects = data.get("bot_loras", {}).get(bot_name, {})
    if project_name not in bot_projects:
        print(f"[BOT_LORA] 프로젝트 없음: {bot_name}/{project_name}")
        return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

    # 프로젝트 폴더 삭제
    project_dir = _bot_project_dir(bot_name, project_name)
    if os.path.isdir(project_dir):
        try:
            shutil.rmtree(project_dir)
            print(f"[BOT_LORA] 프로젝트 폴더 삭제: {project_dir}")
        except Exception as e:
            print(f"[BOT_LORA] 프로젝트 폴더 삭제 실패: {project_dir} - {e}")

    # 학습된 LoRA 폴더도 삭제
    if lora_load_path:
        bot_data_raw = _load_bot_data()
        for b in bot_data_raw.get("bots", []):
            if b.get("name") == bot_name:
                for ch in b.get("characters", []):
                    cn = ch.get("name", "")
                    if cn:
                        trained_dir = _trained_lora_dir(lora_load_path, bot_name, project_name, cn)
                        if os.path.isdir(trained_dir):
                            try:
                                shutil.rmtree(trained_dir)
                                print(f"[BOT_LORA] 학습 폴더 삭제: {trained_dir}")
                            except Exception as e:
                                print(f"[BOT_LORA] 학습 폴더 삭제 실패: {trained_dir} - {e}")
                break

    del bot_projects[project_name]
    if not bot_projects:
        del data["bot_loras"][bot_name]
    _save_bot_lora_manage(data)
    print(f"[BOT_LORA] 프로젝트 삭제: {bot_name}/{project_name}")
    return {"success": True}


# ─── 프로젝트 데이터 ─────────────────────────────────────────

def get_project_data(bot_name: str, project_name: str, lora_load_path: str = "") -> dict:
    """프로젝트의 상세 데이터 반환"""
    bot_data = _load_bot_data()
    bot_info = None
    for b in bot_data.get("bots", []):
        if b.get("name") == bot_name:
            bot_info = b
            break
    if not bot_info:
        print(f"[BOT_LORA] 봇 없음: {bot_name}")
        return {"success": False, "error": "봇을 찾을 수 없습니다"}

    manage_data = _load_bot_lora_manage()
    proj_cfg = _get_project_config(manage_data, bot_name, project_name)
    if not proj_cfg:
        print(f"[BOT_LORA] 프로젝트 없음: {bot_name}/{project_name}")
        return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

    training_config = proj_cfg.get("training_config", {})
    characters = []
    bot_units = {
        (unit["name"], unit["visual_card_id"]): unit
        for unit in _bot_visual_units(bot_data, bot_name)
    }
    for char_name, visual_card_id, char_cfg in _iter_project_units(proj_cfg):
        if not char_name:
            continue

        unit = bot_units.get((char_name, visual_card_id)) or {}
        training_images = _get_project_training_images(
            bot_name, project_name, char_name, visual_card_id
        )
        trigger = _effective_character_trigger(
            proj_cfg,
            char_name,
            visual_card_id,
            char_cfg,
        )
        trained_sessions = (
            _list_bot_trained_sessions(
                lora_load_path,
                bot_name,
                project_name,
                char_name,
                visual_card_id,
            )
            if lora_load_path else []
        )

        # 카드 render_overrides의 gender_tag를 우선하고 루트 값으로 폴백한다.
        bot_char = next((c for c in bot_info.get("characters", []) if c.get("name") == char_name), None)
        gender_tag = unit.get("gender_tag") or (
            (bot_char.get("gender_tag") or "") if bot_char else ""
        )

        characters.append({
            "name": char_name,
            "unit_key": f"{char_name}::{visual_card_id or 'legacy'}",
            "visual_card_id": visual_card_id,
            "visual_card_label": (
                unit.get("visual_card_label")
                or char_cfg.get("label")
                or ("기본 카드" if not visual_card_id else visual_card_id)
            ),
            "visual_card_index": unit.get("visual_card_index", 1),
            "is_primary": unit.get("is_primary", not visual_card_id),
            "trigger": trigger,
            "gender_tag": gender_tag,
            "skip_training": char_cfg.get("skip_training", False),
            "training_images": training_images,
            "trained_sessions": trained_sessions,
            "session_representatives": char_cfg.get("session_representatives", {}),
            "char_test_images": list_bot_char_test_images(
                bot_name, project_name, char_name, visual_card_id
            ),
        })

    test_images = list_bot_test_images(bot_name, project_name)

    return {
        "success": True,
        "bot_name": bot_name,
        "project_name": project_name,
        "characters": characters,
        "test_images": test_images,
        "training_config": training_config,
    }


def _get_char_training_images(bot_name: str, char_name: str, rep_images: list) -> list:
    """캐릭터의 원본 학습 이미지 목록 반환 (학습 Export용)"""
    char_dir = _bot_char_dir(bot_name, char_name)
    if not os.path.isdir(char_dir):
        print(f"[BOT_LORA] 캐릭터 폴더 없음: {char_dir}")
        return []

    images = []
    for i, fname in enumerate(rep_images):
        fpath = os.path.join(char_dir, fname)
        if not os.path.isfile(fpath):
            print(f"[BOT_LORA] 대표 이미지 없음: {fpath}")
            continue
        img_data = _load_image_with_prompt(fpath, char_dir, fname)
        if img_data:
            img_data["source"] = "rep"
            images.append(img_data)

    face_path = os.path.join(char_dir, "_face_image.webp")
    if os.path.isfile(face_path):
        img_data = _load_image_with_prompt(face_path, char_dir, "_face_image.webp")
        if img_data:
            img_data["source"] = "face"
            images.append(img_data)

    return images


def _get_project_training_images(
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> list:
    """프로젝트 폴더에서 학습 이미지 목록 반환"""
    proj_char_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    if not os.path.isdir(proj_char_dir):
        print(f"[BOT_LORA] 프로젝트 캐릭터 폴더 없음: {proj_char_dir}")
        return []

    images = []
    for fname in sorted(os.listdir(proj_char_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        fpath = os.path.join(proj_char_dir, fname)
        img_data = _load_image_with_prompt(fpath, proj_char_dir, fname)
        if img_data:
            if fname == "_face_image.webp":
                img_data["source"] = "face"
            else:
                img_data["source"] = "rep"
            images.append(img_data)

    return images


def _load_image_with_prompt(fpath: str, char_dir: str, fname: str) -> dict | None:
    try:
        fstat = os.stat(fpath)
        width, height = 0, 0
        try:
            with Image.open(fpath) as im:
                width, height = im.size
        except Exception as e:
            print(f"[BOT_LORA] 이미지 해상도 읽기 실패: {fpath} - {e}")

        base = os.path.splitext(fname)[0]
        prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
        positive = ""
        negative = ""
        original_positive = ""
        original_negative = ""
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                positive = pdata.get("positive", pdata.get("prompt", ""))
                negative = pdata.get("negative", "")
                original_positive = pdata.get("original_positive", positive)
                original_negative = pdata.get("original_negative", negative)
            except Exception as e:
                print(f"[BOT_LORA] 프롬프트 로드 실패: {prompt_path} - {e}")

        return {
            "filename": fname,
            "filepath": fpath,
            "positive": positive,
            "negative": negative,
            "original_positive": original_positive,
            "original_negative": original_negative,
            "size": fstat.st_size,
            "modified": fstat.st_mtime,
            "width": width,
            "height": height,
        }
    except Exception as e:
        print(f"[BOT_LORA] 이미지 정보 읽기 실패: {fpath} - {e}")
        return None


# ─── 캐릭터별 테스트 이미지 ─────────────────────────────────

def _bot_char_test_dir(
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> str:
    """캐릭터별 테스트 이미지 폴더: bot/<봇>/Lora/<프로젝트>/<캐릭터>/_test/"""
    return os.path.join(
        _bot_project_training_dir(
            bot_name, project_name, char_name, visual_card_id
        ),
        TEST_DIR_NAME,
    )


def list_bot_char_test_images(
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> list:
    """캐릭터별 테스트 이미지 목록 반환"""
    t_dir = _bot_char_test_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    if not os.path.isdir(t_dir):
        return []

    rep_path = os.path.join(t_dir, "_representative.json")
    representative = ""
    if os.path.isfile(rep_path):
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                representative = json.load(f).get("filename", "")
        except Exception:
            pass

    images = []
    for fname in sorted(os.listdir(t_dir)):
        if fname.startswith("_"):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        fpath = os.path.join(t_dir, fname)
        base = os.path.splitext(fname)[0]
        prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
        positive = ""
        negative = ""
        original_positive = ""
        original_negative = ""
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                positive = pdata.get("positive", "")
                negative = pdata.get("negative", "")
                original_positive = pdata.get("original_positive", positive)
                original_negative = pdata.get("original_negative", negative)
            except Exception as e:
                print(f"[BOT_LORA] 캐릭터 테스트 프롬프트 로드 실패: {prompt_path} - {e}")
        try:
            fstat = os.stat(fpath)
            images.append({
                "filename": fname,
                "positive": positive,
                "negative": negative,
                "original_positive": original_positive,
                "original_negative": original_negative,
                "is_representative": fname == representative,
                "size": fstat.st_size,
                "modified": fstat.st_mtime,
            })
        except Exception as e:
            print(f"[BOT_LORA] 캐릭터 테스트 이미지 정보 읽기 실패: {fpath} - {e}")
    return images


def add_bot_char_test_images(
    bot_name: str,
    project_name: str,
    char_name: str,
    sources: list,
    visual_card_id: str = "",
) -> dict:
    """에셋에서 캐릭터별 테스트 이미지 추가"""
    from modes.asset_mode import ASSET_DIR, AssetMode

    t_dir = _bot_char_test_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    os.makedirs(t_dir, exist_ok=True)

    added = []
    skipped = []
    for src in sources:
        outfit = src.get("outfit", "")
        expression = src.get("expression", "")
        filename = src.get("filename", "")
        src_char = src.get("character", "")
        src_char_dir = os.path.join(ASSET_DIR, AssetMode._safe_dirname(src_char)) if src_char else ""

        if not outfit or not expression or not filename or not src_char:
            print(f"[BOT_LORA] 캐릭터 테스트 이미지 추가: 필수 값 누락 - {src}")
            skipped.append({"filename": filename, "reason": "필수 값 누락"})
            continue

        src_path = os.path.join(src_char_dir, AssetMode._safe_dirname(outfit), AssetMode._safe_dirname(expression), filename)
        if not os.path.isfile(src_path):
            print(f"[BOT_LORA] 캐릭터 테스트 이미지 원본 없음: {src_path}")
            skipped.append({"filename": filename, "reason": "원본 파일 없음"})
            continue

        dest_name = filename
        dest_path = os.path.join(t_dir, dest_name)
        if os.path.exists(dest_path):
            import time
            base, ext = os.path.splitext(filename)
            dest_name = f"{int(time.time())}_{base}{ext}"
            dest_path = os.path.join(t_dir, dest_name)

        try:
            shutil.copy2(src_path, dest_path)
            base, ext = os.path.splitext(filename)
            prompt_src = os.path.join(src_char_dir, AssetMode._safe_dirname(outfit), AssetMode._safe_dirname(expression), f"{base}_prompt.json")
            prompt_dest = os.path.join(t_dir, f"{os.path.splitext(dest_name)[0]}_prompt.json")
            if os.path.isfile(prompt_src):
                with open(prompt_src, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                positive = pdata.get("positive", "")
                marker = "[FACE_ID_ACTIVATE]"
                if marker in positive:
                    pdata["positive"] = positive.split(marker)[0].strip()
                pdata["original_positive"] = pdata["positive"]
                pdata["original_negative"] = pdata.get("negative", "")
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump(pdata, f, ensure_ascii=False, indent=2)
            else:
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump({"positive": "", "negative": "", "original_positive": "", "original_negative": ""}, f, ensure_ascii=False, indent=2)
            added.append(dest_name)
            print(f"[BOT_LORA] 캐릭터 테스트 이미지 추가: {dest_path}")
        except Exception as e:
            print(f"[BOT_LORA] 캐릭터 테스트 이미지 복사 실패: {src_path} -> {dest_path} - {e}")
            traceback.print_exc()
            skipped.append({"filename": filename, "reason": str(e)})

    return {"success": True, "added": added, "skipped": skipped}


def copy_project_test_to_char(
    bot_name: str,
    project_name: str,
    char_name: str,
    filenames: list = None,
    visual_card_id: str = "",
) -> dict:
    """프로젝트 공통 테스트 이미지를 캐릭터 _test/ 로 복제"""
    src_dir = _bot_test_dir(bot_name, project_name)
    dst_dir = _bot_char_test_dir(
        bot_name, project_name, char_name, visual_card_id
    )

    if not os.path.isdir(src_dir):
        print(f"[BOT_LORA] 공통 테스트 폴더 없음: {src_dir}")
        return {"success": False, "error": "공통 테스트 이미지 폴더가 없습니다"}

    os.makedirs(dst_dir, exist_ok=True)

    # filenames가 None이면 전체 복사
    src_files = []
    for fname in sorted(os.listdir(src_dir)):
        if fname.startswith("_"):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        if filenames and fname not in filenames:
            continue
        src_files.append(fname)

    if not src_files:
        print(f"[BOT_LORA] 복사할 공통 테스트 이미지 없음")
        return {"success": True, "copied": [], "skipped": 0}

    copied = []
    skipped = 0
    for fname in src_files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if os.path.exists(dst_path):
            skipped += 1
            continue
        try:
            shutil.copy2(src_path, dst_path)
            # 프롬프트 JSON도 복사
            base = os.path.splitext(fname)[0]
            prompt_src = os.path.join(src_dir, f"{base}_prompt.json")
            prompt_dst = os.path.join(dst_dir, f"{base}_prompt.json")
            if os.path.isfile(prompt_src) and not os.path.isfile(prompt_dst):
                shutil.copy2(prompt_src, prompt_dst)
            copied.append(fname)
        except Exception as e:
            print(f"[BOT_LORA] 공통 테스트 복사 실패: {src_path} -> {dst_path} - {e}")
            skipped += 1

    print(f"[BOT_LORA] 공통→캐릭터 복제 완료: {bot_name}/{project_name}/{char_name} - 복사:{len(copied)}, 스킵:{skipped}")
    return {"success": True, "copied": copied, "skipped": skipped}


def delete_bot_char_test_image(
    bot_name: str,
    project_name: str,
    char_name: str,
    filename: str,
    visual_card_id: str = "",
) -> dict:
    """캐릭터별 테스트 이미지 삭제"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_char_test_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    fpath = os.path.join(t_dir, filename)
    if not os.path.isfile(fpath):
        print(f"[BOT_LORA] 캐릭터 테스트 이미지 없음: {fpath}")
        return {"success": False, "error": "파일을 찾을 수 없습니다"}
    try:
        os.remove(fpath)
        base = os.path.splitext(filename)[0]
        prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)
        rep_path = os.path.join(t_dir, "_representative.json")
        if os.path.isfile(rep_path):
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    if json.load(f).get("filename") == filename:
                        os.remove(rep_path)
            except Exception:
                pass
        print(f"[BOT_LORA] 캐릭터 테스트 이미지 삭제 완료: {fpath}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_LORA] 캐릭터 테스트 이미지 삭제 실패: {fpath} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def save_bot_char_test_prompt(
    bot_name: str,
    project_name: str,
    char_name: str,
    filename: str,
    positive: str,
    negative: str,
    visual_card_id: str = "",
) -> dict:
    """캐릭터별 테스트 이미지 프롬프트 저장"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_char_test_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
    try:
        existing = {}
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        if "original_positive" not in existing:
            existing["original_positive"] = existing.get("positive", "")
        if "original_negative" not in existing:
            existing["original_negative"] = existing.get("negative", "")
        existing["positive"] = positive
        existing["negative"] = negative
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[BOT_LORA] 캐릭터 테스트 프롬프트 저장 완료: {prompt_path}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_LORA] 캐릭터 테스트 프롬프트 저장 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


def save_bot_char_test_prompt_positive_only(
    bot_name: str,
    project_name: str,
    char_name: str,
    filename: str,
    positive: str,
    visual_card_id: str = "",
) -> dict:
    """LLM '테스트 이미지 세팅' 결과로 positive만 교체. negative/original_*는 기존값을 유지한다."""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_char_test_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
    try:
        existing = {}
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        if "original_positive" not in existing:
            existing["original_positive"] = existing.get("positive", "")
        if "original_negative" not in existing:
            existing["original_negative"] = existing.get("negative", "")
        # positive만 교체 — negative는 절대 건드리지 않는다.
        existing["positive"] = positive
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[BOT_LORA] 캐릭터 테스트 프롬프트 정제(positive-only) 저장 완료: {prompt_path}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_LORA] 캐릭터 테스트 프롬프트 정제(positive-only) 저장 실패: {prompt_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_bot_char_test_image_path(
    bot_name: str,
    project_name: str,
    char_name: str,
    filename: str,
    visual_card_id: str = "",
) -> str | None:
    """캐릭터별 테스트 이미지 파일 경로 반환"""
    if ".." in filename or os.path.sep in filename:
        print(f"[BOT_LORA] 잘못된 파일명: {filename}")
        return None
    fpath = os.path.join(
        _bot_char_test_dir(bot_name, project_name, char_name, visual_card_id),
        filename,
    )
    if os.path.isfile(fpath):
        return fpath
    print(f"[BOT_LORA] 캐릭터 테스트 이미지 없음: {fpath}")
    return None


# ─── 학습 설정 ─────────────────────────────────────────────────

def update_training_config(bot_name: str, project_name: str, config: dict) -> dict:
    if not bot_name or not project_name:
        return {"success": False, "error": "봇/프로젝트 이름 누락"}
    data = _load_bot_lora_manage()
    proj = data.setdefault("bot_loras", {}).setdefault(bot_name, {}).setdefault(project_name, {})
    proj["training_config"] = config
    _save_bot_lora_manage(data)
    print(f"[BOT_LORA] 학습 설정 업데이트: {bot_name}/{project_name}")
    return {"success": True}


def update_char_trigger(
    bot_name: str,
    project_name: str,
    char_name: str,
    trigger: str,
    visual_card_id: str = "",
) -> dict:
    if not bot_name or not project_name or not char_name:
        print(
            f"[BOT_LORA] trigger 업데이트 필수값 누락: "
            f"bot={bot_name!r}, project={project_name!r}, character={char_name!r}"
        )
        return {"success": False, "error": "봇/프로젝트/캐릭터 이름 누락"}
    data = _load_bot_lora_manage()
    proj = _get_project_config(data, bot_name, project_name)
    char_cfg = (
        (proj.get("characters") or {}).get(char_name)
        if isinstance(proj, dict)
        else None
    )
    if not isinstance(char_cfg, dict):
        print(
            f"[BOT_LORA] trigger 업데이트 대상 없음: "
            f"{bot_name}/{project_name}/{char_name}/{visual_card_id or 'legacy'}"
        )
        return {"success": False, "error": "캐릭터 설정을 찾을 수 없습니다"}
    _set_character_trigger(proj, char_name, trigger)
    _save_bot_lora_manage(data)
    saved_trigger = str(char_cfg.get("trigger") or char_name).strip() or char_name
    print(
        f"[BOT_LORA] 캐릭터 공용 trigger 업데이트: "
        f"{bot_name}/{project_name}/{char_name} -> {saved_trigger}"
    )
    return {"success": True}


def update_char_skip_training(
    bot_name: str,
    project_name: str,
    char_name: str,
    skip: bool,
    visual_card_id: str = "",
) -> dict:
    if not bot_name or not project_name or not char_name:
        return {"success": False, "error": "봇/프로젝트/캐릭터 이름 누락"}
    data = _load_bot_lora_manage()
    char_cfg = _get_char_config(
        data, bot_name, project_name, char_name, visual_card_id
    )
    if char_cfg is None:
        print(
            f"[BOT_LORA] skip_training 업데이트 대상 없음: "
            f"{bot_name}/{project_name}/{char_name}/{visual_card_id or 'legacy'}"
        )
        return {"success": False, "error": "캐릭터 카드 설정을 찾을 수 없습니다"}
    char_cfg["skip_training"] = bool(skip)
    _save_bot_lora_manage(data)
    print(f"[BOT_LORA] skip_training 업데이트: {bot_name}/{project_name}/{char_name} -> {bool(skip)}")
    return {"success": True}


def update_char_session_representative(
    bot_name: str,
    project_name: str,
    char_name: str,
    session_name: str,
    representative: str,
    visual_card_id: str = "",
) -> dict:
    if not bot_name or not project_name or not char_name:
        return {"success": False, "error": "봇/프로젝트/캐릭터 이름 누락"}
    data = _load_bot_lora_manage()
    char_cfg = _get_char_config(
        data, bot_name, project_name, char_name, visual_card_id
    )
    if char_cfg is None:
        print(
            f"[BOT_LORA] 세션 대표 업데이트 대상 없음: "
            f"{bot_name}/{project_name}/{char_name}/{visual_card_id or 'legacy'}"
        )
        return {"success": False, "error": "캐릭터 카드 설정을 찾을 수 없습니다"}
    if "session_representatives" not in char_cfg:
        char_cfg["session_representatives"] = {}
    char_cfg["session_representatives"][session_name] = representative
    # 대표 설정 시 session_priority에 없으면 맨 앞에 추가 (1순위 후보가 되도록).
    # style_lora_mode.update_style_session_representative 와 동일 정책.
    if session_name not in (char_cfg.get("session_priority") or []):
        char_cfg.setdefault("session_priority", []).insert(0, session_name)
    _save_bot_lora_manage(data)
    return {"success": True}


# ─── 테스트 이미지 관리 ─────────────────────────────────────

def list_bot_test_images(bot_name: str, project_name: str) -> list:
    t_dir = _bot_test_dir(bot_name, project_name)
    if not os.path.isdir(t_dir):
        return []

    rep_path = os.path.join(t_dir, "_representative.json")
    representative = ""
    if os.path.isfile(rep_path):
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                representative = json.load(f).get("filename", "")
        except Exception:
            pass

    images = []
    for fname in sorted(os.listdir(t_dir)):
        if fname.startswith("_"):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        fpath = os.path.join(t_dir, fname)
        base = os.path.splitext(fname)[0]
        prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
        positive = ""
        negative = ""
        original_positive = ""
        original_negative = ""
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                positive = pdata.get("positive", "")
                negative = pdata.get("negative", "")
                original_positive = pdata.get("original_positive", positive)
                original_negative = pdata.get("original_negative", negative)
            except Exception as e:
                print(f"[BOT_LORA] 테스트 프롬프트 로드 실패: {prompt_path} - {e}")
        try:
            fstat = os.stat(fpath)
            images.append({
                "filename": fname,
                "positive": positive,
                "negative": negative,
                "original_positive": original_positive,
                "original_negative": original_negative,
                "is_representative": fname == representative,
                "size": fstat.st_size,
                "modified": fstat.st_mtime,
            })
        except Exception as e:
            print(f"[BOT_LORA] 테스트 이미지 정보 읽기 실패: {fpath} - {e}")
    return images


def add_bot_test_images(bot_name: str, project_name: str, sources: list) -> dict:
    from modes.asset_mode import ASSET_DIR, AssetMode

    t_dir = _bot_test_dir(bot_name, project_name)
    os.makedirs(t_dir, exist_ok=True)

    added = []
    skipped = []
    for src in sources:
        outfit = src.get("outfit", "")
        expression = src.get("expression", "")
        filename = src.get("filename", "")
        src_char = src.get("character", "")
        src_char_dir = os.path.join(ASSET_DIR, AssetMode._safe_dirname(src_char)) if src_char else ""

        if not outfit or not expression or not filename or not src_char:
            print(f"[BOT_LORA] 테스트 이미지 추가: 필수 값 누락 - {src}")
            skipped.append({"filename": filename, "reason": "필수 값 누락"})
            continue

        src_path = os.path.join(src_char_dir, AssetMode._safe_dirname(outfit), AssetMode._safe_dirname(expression), filename)
        if not os.path.isfile(src_path):
            print(f"[BOT_LORA] 테스트 이미지 원본 없음: {src_path}")
            skipped.append({"filename": filename, "reason": "원본 파일 없음"})
            continue

        dest_name = filename
        dest_path = os.path.join(t_dir, dest_name)
        if os.path.exists(dest_path):
            import time
            base, ext = os.path.splitext(filename)
            dest_name = f"{int(time.time())}_{base}{ext}"
            dest_path = os.path.join(t_dir, dest_name)

        try:
            shutil.copy2(src_path, dest_path)
            base, ext = os.path.splitext(filename)
            prompt_src = os.path.join(src_char_dir, AssetMode._safe_dirname(outfit), AssetMode._safe_dirname(expression), f"{base}_prompt.json")
            prompt_dest = os.path.join(t_dir, f"{os.path.splitext(dest_name)[0]}_prompt.json")
            if os.path.isfile(prompt_src):
                with open(prompt_src, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                positive = pdata.get("positive", "")
                marker = "[FACE_ID_ACTIVATE]"
                if marker in positive:
                    pdata["positive"] = positive.split(marker)[0].strip()
                pdata["original_positive"] = pdata["positive"]
                pdata["original_negative"] = pdata.get("negative", "")
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump(pdata, f, ensure_ascii=False, indent=2)
            else:
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump({"positive": "", "negative": "", "original_positive": "", "original_negative": ""}, f, ensure_ascii=False, indent=2)
            added.append(dest_name)
            print(f"[BOT_LORA] 테스트 이미지 추가: {dest_path}")
        except Exception as e:
            print(f"[BOT_LORA] 테스트 이미지 복사 실패: {src_path} -> {dest_path} - {e}")
            traceback.print_exc()
            skipped.append({"filename": filename, "reason": str(e)})

    return {"success": True, "added": added, "skipped": skipped}


def get_bot_test_image_path(bot_name: str, project_name: str, filename: str) -> str | None:
    if ".." in filename or os.path.sep in filename:
        print(f"[BOT_LORA] 잘못된 파일명: {filename}")
        return None
    fpath = os.path.join(_bot_test_dir(bot_name, project_name), filename)
    if os.path.isfile(fpath):
        return fpath
    print(f"[BOT_LORA] 테스트 이미지 없음: {fpath}")
    return None


def delete_bot_test_image(bot_name: str, project_name: str, filename: str) -> dict:
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_test_dir(bot_name, project_name)
    fpath = os.path.join(t_dir, filename)
    if not os.path.isfile(fpath):
        print(f"[BOT_LORA] 테스트 이미지 없음: {fpath}")
        return {"success": False, "error": "파일을 찾을 수 없습니다"}
    try:
        os.remove(fpath)
        base = os.path.splitext(filename)[0]
        prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)
        rep_path = os.path.join(t_dir, "_representative.json")
        if os.path.isfile(rep_path):
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    if json.load(f).get("filename") == filename:
                        os.remove(rep_path)
            except Exception:
                pass
        print(f"[BOT_LORA] 테스트 이미지 삭제 완료: {fpath}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_LORA] 테스트 이미지 삭제 실패: {fpath} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def delete_bot_training_image(
    bot_name: str,
    project_name: str,
    char_name: str,
    filename: str,
    visual_card_id: str = "",
) -> dict:
    """봇 LoRA 학습 이미지 + 프롬프트 JSON 삭제"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    fpath = os.path.join(t_dir, filename)
    if not os.path.isfile(fpath):
        print(f"[BOT_LORA] 학습 이미지 없음: {fpath}")
        return {"success": False, "error": "파일을 찾을 수 없습니다"}
    try:
        os.remove(fpath)
        base = os.path.splitext(filename)[0]
        prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)
        rep_path = os.path.join(t_dir, "_representative.json")
        if os.path.isfile(rep_path):
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    if json.load(f).get("filename") == filename:
                        os.remove(rep_path)
            except Exception:
                pass
        print(f"[BOT_LORA] 학습 이미지 삭제 완료: {fpath}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_LORA] 학습 이미지 삭제 실패: {fpath} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def list_bot_char_available_images(
    bot_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> list:
    """봇 캐릭터 원본 폴더에서 사용 가능한 이미지 목록 반환"""
    char_dir = _bot_char_dir(bot_name, char_name)
    if not os.path.isdir(char_dir):
        print(f"[BOT_LORA] 봇 캐릭터 폴더 없음: {char_dir}")
        return []

    profile_reps = set()
    target_unit = None
    if visual_card_id:
        bot_data = _load_bot_data()
        target_unit = next((
            unit for unit in _bot_visual_units(bot_data, bot_name)
            if unit["name"] == char_name
            and unit["visual_card_id"] == visual_card_id
        ), None)
        profile_reps = set((target_unit or {}).get("rep_images") or [])

    images = []
    for fname in sorted(os.listdir(char_dir)):
        if fname == "_face_image.webp":
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        fpath = os.path.join(char_dir, fname)
        img_data = _load_image_with_prompt(fpath, char_dir, fname)
        if img_data:
            img_data["source"] = "rep"
            img_data["is_profile_representative"] = fname in profile_reps
            images.append(img_data)

    face_dir = char_dir
    if visual_card_id and target_unit is None:
        print(
            f"[BOT_LORA] 카드별 얼굴 이미지 조회 대상 없음: "
            f"{bot_name}/{char_name}/{visual_card_id}"
        )
        return images
    if target_unit and not target_unit.get("is_primary"):
        face_dir = os.path.join(
            char_dir,
            "_visual_profiles",
            _safe_dirname(target_unit["visual_card_id"]),
        )
    face_path = os.path.join(face_dir, "_face_image.webp")
    if os.path.isfile(face_path):
        face_data = _load_image_with_prompt(
            face_path, face_dir, "_face_image.webp"
        )
        if face_data:
            face_data["source"] = "face"
            face_data["is_profile_representative"] = False
            images.append(face_data)

    return images


def add_bot_training_images(
    bot_name: str,
    project_name: str,
    char_name: str,
    sources: list,
    visual_card_id: str = "",
) -> dict:
    """에셋에서 봇 LoRA 학습 이미지 추가"""
    from modes.asset_mode import ASSET_DIR, AssetMode

    t_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    os.makedirs(t_dir, exist_ok=True)

    added = []
    skipped = []
    for src in sources:
        outfit = src.get("outfit", "")
        expression = src.get("expression", "")
        filename = src.get("filename", "")
        src_char = src.get("character", "")
        src_char_dir = os.path.join(ASSET_DIR, AssetMode._safe_dirname(src_char)) if src_char else ""

        if not outfit or not expression or not filename or not src_char:
            print(f"[BOT_LORA] 학습 이미지 추가: 필수 값 누락 - {src}")
            skipped.append({"filename": filename, "reason": "필수 값 누락"})
            continue

        src_path = os.path.join(src_char_dir, AssetMode._safe_dirname(outfit), AssetMode._safe_dirname(expression), filename)
        if not os.path.isfile(src_path):
            print(f"[BOT_LORA] 학습 이미지 원본 없음: {src_path}")
            skipped.append({"filename": filename, "reason": "원본 파일 없음"})
            continue

        dest_name = filename
        dest_path = os.path.join(t_dir, dest_name)
        if os.path.exists(dest_path):
            import time
            base, ext = os.path.splitext(filename)
            dest_name = f"{int(time.time())}_{base}{ext}"
            dest_path = os.path.join(t_dir, dest_name)

        try:
            shutil.copy2(src_path, dest_path)
            base, ext = os.path.splitext(filename)
            prompt_src = os.path.join(src_char_dir, AssetMode._safe_dirname(outfit), AssetMode._safe_dirname(expression), f"{base}_prompt.json")
            prompt_dest = os.path.join(t_dir, f"{os.path.splitext(dest_name)[0]}_prompt.json")
            if os.path.isfile(prompt_src):
                with open(prompt_src, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                positive = pdata.get("positive", "")
                marker = "[FACE_ID_ACTIVATE]"
                if marker in positive:
                    pdata["positive"] = positive.split(marker)[0].strip()
                pdata["original_positive"] = pdata["positive"]
                pdata["original_negative"] = pdata.get("negative", "")
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump(pdata, f, ensure_ascii=False, indent=2)
            else:
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump({"positive": "", "negative": "", "original_positive": "", "original_negative": ""}, f, ensure_ascii=False, indent=2)
            added.append(dest_name)
            print(f"[BOT_LORA] 학습 이미지 추가: {dest_path}")
        except Exception as e:
            print(f"[BOT_LORA] 학습 이미지 복사 실패: {src_path} -> {dest_path} - {e}")
            traceback.print_exc()
            skipped.append({"filename": filename, "reason": str(e)})

    return {"success": True, "added": added, "skipped": skipped}


def add_bot_training_from_bot(
    bot_name: str,
    project_name: str,
    char_name: str,
    filenames: list,
    visual_card_id: str = "",
) -> dict:
    """봇 캐릭터 원본 폴더에서 학습 이미지를 프로젝트로 복사"""
    src_dir = _bot_char_dir(bot_name, char_name)
    dst_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    if not os.path.isdir(src_dir):
        print(f"[BOT_LORA] 봇 캐릭터 폴더 없음: {src_dir}")
        return {"success": False, "error": "봇 캐릭터 폴더 없음"}

    os.makedirs(dst_dir, exist_ok=True)

    added = []
    skipped = []
    for filename in filenames:
        if ".." in filename or os.path.sep in filename:
            skipped.append({"filename": filename, "reason": "잘못된 파일명"})
            continue
        file_src_dir = src_dir
        if filename == "_face_image.webp" and visual_card_id:
            bot_data = _load_bot_data()
            target_unit = next((
                unit for unit in _bot_visual_units(bot_data, bot_name)
                if unit["name"] == char_name
                and unit["visual_card_id"] == visual_card_id
            ), None)
            if target_unit is None:
                print(
                    f"[BOT_LORA] 카드별 얼굴 이미지 추가 대상 없음: "
                    f"{bot_name}/{char_name}/{visual_card_id}"
                )
                skipped.append({"filename": filename, "reason": "캐릭터 카드 없음"})
                continue
            if target_unit and not target_unit.get("is_primary"):
                file_src_dir = os.path.join(
                    src_dir,
                    "_visual_profiles",
                    _safe_dirname(visual_card_id),
                )
        src_path = os.path.join(file_src_dir, filename)
        if not os.path.isfile(src_path):
            print(f"[BOT_LORA] 원본 이미지 없음: {src_path}")
            skipped.append({"filename": filename, "reason": "원본 파일 없음"})
            continue

        dest_name = filename
        dest_path = os.path.join(dst_dir, dest_name)
        if os.path.exists(dest_path):
            import time
            base, ext = os.path.splitext(filename)
            dest_name = f"{int(time.time())}_{base}{ext}"
            dest_path = os.path.join(dst_dir, dest_name)

        try:
            shutil.copy2(src_path, dest_path)
            base, ext = os.path.splitext(filename)
            prompt_src = os.path.join(file_src_dir, f"{base}_prompt.json")
            prompt_dest = os.path.join(dst_dir, f"{os.path.splitext(dest_name)[0]}_prompt.json")
            pdata = None
            if os.path.isfile(prompt_src):
                with open(prompt_src, "r", encoding="utf-8") as f:
                    pdata = json.load(f)

            if pdata:
                positive = pdata.get("positive", pdata.get("prompt", ""))
                pdata["positive"] = positive
                marker = "[FACE_ID_ACTIVATE]"
                if marker in positive:
                    pdata["positive"] = positive.split(marker)[0].strip()
                if "original_positive" not in pdata:
                    pdata["original_positive"] = pdata["positive"]
                if "original_negative" not in pdata:
                    pdata["original_negative"] = pdata.get("negative", "")
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump(pdata, f, ensure_ascii=False, indent=2)
            else:
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump({"positive": "", "negative": "", "original_positive": "", "original_negative": ""}, f, ensure_ascii=False, indent=2)
            added.append(dest_name)
            print(f"[BOT_LORA] 학습 이미지 추가(봇에서): {dest_path}")
        except Exception as e:
            print(f"[BOT_LORA] 학습 이미지 복사 실패: {src_path} -> {dest_path} - {e}")
            traceback.print_exc()
            skipped.append({"filename": filename, "reason": str(e)})

    return {"success": True, "added": added, "skipped": skipped}


def save_bot_test_prompt(bot_name: str, project_name: str, filename: str, positive: str, negative: str) -> dict:
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_test_dir(bot_name, project_name)
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
    try:
        existing = {}
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        if "original_positive" not in existing:
            existing["original_positive"] = existing.get("positive", "")
        if "original_negative" not in existing:
            existing["original_negative"] = existing.get("negative", "")
        existing["positive"] = positive
        existing["negative"] = negative
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[BOT_LORA] 테스트 프롬프트 저장 완료: {prompt_path}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_LORA] 테스트 프롬프트 저장 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


# ─── 학습 이미지 프롬프트 수정 ───────────────────────────────

def save_bot_training_prompt(
    bot_name: str,
    project_name: str,
    char_name: str,
    filename: str,
    positive: str,
    negative: str,
    visual_card_id: str = "",
) -> dict:
    """프로젝트 폴더의 _prompt.json 수정"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    proj_char_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(proj_char_dir, f"{base}_prompt.json")
    try:
        existing = {}
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        if "original_positive" not in existing:
            existing["original_positive"] = existing.get("positive", "")
        if "original_negative" not in existing:
            existing["original_negative"] = existing.get("negative", "")
        existing["positive"] = positive
        existing["negative"] = negative
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[BOT_LORA] 학습 프롬프트 저장 완료: {prompt_path}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_LORA] 학습 프롬프트 저장 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


def save_bot_training_prompt_positive_only(
    bot_name: str,
    project_name: str,
    char_name: str,
    filename: str,
    positive: str,
    visual_card_id: str = "",
) -> dict:
    """LLM 정제 결과로 positive만 교체. negative/original_*는 기존값을 그대로 유지한다."""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    proj_char_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(proj_char_dir, f"{base}_prompt.json")
    try:
        existing = {}
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        if "original_positive" not in existing:
            existing["original_positive"] = existing.get("positive", "")
        if "original_negative" not in existing:
            existing["original_negative"] = existing.get("negative", "")
        # positive만 교체 — negative는 절대 건드리지 않는다 (LLM 정제는 positive에만 관여).
        existing["positive"] = positive
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[BOT_LORA] 학습 프롬프트 정제(positive-only) 저장 완료: {prompt_path}")
        return {"success": True}
    except Exception as e:
        print(f"[BOT_LORA] 학습 프롬프트 정제(positive-only) 저장 실패: {prompt_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ─── 학습된 LoRA 관리 ────────────────────────────────────────

def _list_bot_trained_sessions(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> list:
    if not lora_load_path:
        print("[BOT_LORA_TRAINED] lora_load_path 미설정")
        return []
    entry_dir = _trained_lora_dir(
        lora_load_path, bot_name, project_name, char_name, visual_card_id
    )
    if not os.path.isdir(entry_dir):
        print(f"[BOT_LORA_TRAINED] 캐릭터 카드 LoRA 폴더 없음: {entry_dir}")
        return []

    manage_data = _load_bot_lora_manage()
    char_cfg = _get_char_config(
        manage_data, bot_name, project_name, char_name, visual_card_id
    ) or {}
    session_reps = char_cfg.get("session_representatives", {})
    session_priority = _resolve_session_priority(char_cfg, entry_dir, session_reps)

    sessions = []
    for name in sorted(os.listdir(entry_dir), reverse=True):
        path = os.path.join(entry_dir, name)
        if not os.path.isdir(path):
            continue
        step_count = sum(1 for f in os.listdir(path) if f.endswith('.safetensors'))
        has_final = any('-step' not in f for f in os.listdir(path) if f.endswith('.safetensors'))
        session_rep = session_reps.get(name, "")
        preview_url = ""
        if session_rep:
            try:
                rep_data = json.loads(session_rep)
                preview_url = rep_data.get("preview", "")
                if (
                    visual_card_id
                    and "/api/bot_lora/trained/preview/" in preview_url
                    and "visual_card_id=" not in preview_url
                ):
                    separator = "&" if "?" in preview_url else "?"
                    from urllib.parse import quote
                    preview_url += (
                        f"{separator}visual_card_id={quote(visual_card_id, safe='')}"
                    )
            except Exception:
                pass
        try:
            priority_rank = session_priority.index(name) + 1 if name in session_priority else 0
        except ValueError:
            priority_rank = 0
        sessions.append({
            "name": name,
            "step_count": step_count,
            "has_final": has_final,
            "representative": session_rep,
            "preview_url": preview_url,
            "priority_rank": priority_rank,
        })
    return sessions


def _resolve_session_priority(char_cfg: dict, entry_dir: str, session_reps: dict) -> list:
    """저장된 session_priority 반환. 비어있으면 representative 있는 세션들로 자동 채움(마이그레이션, 저장 안 함).

    저장값이 있더라도 디스크에 더 이상 존재하지 않는(삭제된) 세션 이름은 무시한다.
    걸러낸 뒤 남은 것이 없으면 representative 기반 자동 채움으로 넘어간다.
    (탐색기 수동 삭제 등 어떤 경로로든 생긴 찌꺼기를 런타임에 무효화)
    """
    priority = list(char_cfg.get("session_priority", []) or [])
    if priority:
        if os.path.isdir(entry_dir):
            existing = {n for n in os.listdir(entry_dir) if os.path.isdir(os.path.join(entry_dir, n))}
            filtered = [n for n in priority if n in existing]
            if filtered:
                return filtered
            print(f"[BOT_LORA] session_priority 항목이 모두 삭제된 세션임, representative 기반으로 재계산: {priority}")
        else:
            # 디스크를 검증할 수 없으면 저장값을 신뢰
            return priority
    if not os.path.isdir(entry_dir):
        return []
    auto = []
    for name in sorted(os.listdir(entry_dir), reverse=True):
        path = os.path.join(entry_dir, name)
        if not os.path.isdir(path):
            continue
        if session_reps.get(name, ""):
            auto.append(name)
    return auto


def get_char_session_priority(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> list:
    """project API 등에서 우선순위만 조회. lora_load_path가 없으면 저장값만 반환."""
    manage_data = _load_bot_lora_manage()
    char_cfg = _get_char_config(
        manage_data, bot_name, project_name, char_name, visual_card_id
    ) or {}
    session_reps = char_cfg.get("session_representatives", {})
    if lora_load_path:
        entry_dir = _trained_lora_dir(
            lora_load_path, bot_name, project_name, char_name, visual_card_id
        )
        return _resolve_session_priority(char_cfg, entry_dir, session_reps)
    return list(char_cfg.get("session_priority", []) or [])


def update_char_session_priority(
    bot_name: str,
    project_name: str,
    char_name: str,
    sessions_list: list,
    visual_card_id: str = "",
) -> dict:
    if not bot_name or not project_name or not char_name:
        return {"success": False, "error": "봇/프로젝트/캐릭터 이름 누락"}
    if not isinstance(sessions_list, list):
        return {"success": False, "error": "sessions는 배열이어야 합니다"}
    data = _load_bot_lora_manage()
    char_cfg = _get_char_config(
        data, bot_name, project_name, char_name, visual_card_id
    )
    if char_cfg is None:
        print(
            f"[BOT_LORA] session_priority 대상 없음: "
            f"{bot_name}/{project_name}/{char_name}/{visual_card_id or 'legacy'}"
        )
        return {"success": False, "error": "캐릭터 카드 설정을 찾을 수 없습니다"}
    char_cfg["session_priority"] = sessions_list
    _save_bot_lora_manage(data)
    print(f"[BOT_LORA] session_priority 업데이트: {bot_name}/{project_name}/{char_name} -> {sessions_list}")
    return {"success": True}


def list_bot_trained_sessions(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> list:
    return _list_bot_trained_sessions(
        lora_load_path, bot_name, project_name, char_name, visual_card_id
    )


def list_bot_trained_steps(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    session: str,
    visual_card_id: str = "",
) -> list:
    if not lora_load_path:
        print("[BOT_LORA_TRAINED] lora_load_path 미설정")
        return []
    session_dir = os.path.join(
        _trained_lora_dir(
            lora_load_path, bot_name, project_name, char_name, visual_card_id
        ),
        session,
    )
    if not os.path.isdir(session_dir):
        print(f"[BOT_LORA_TRAINED] 세션 폴더 없음: {session_dir}")
        return []
    steps = []
    for fname in sorted(os.listdir(session_dir)):
        if not fname.endswith('.json'):
            continue
        if fname.endswith('.metadata.json'):
            print(f"[BOT_LORA_TRAINED] 보조 메타데이터 제외: {os.path.join(session_dir, fname)}")
            continue
        json_path = os.path.join(session_dir, fname)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[BOT_LORA_TRAINED] JSON 읽기 실패: {json_path} - {e}")
            continue
        step_name = os.path.splitext(fname)[0]
        steps.append({
            "name": step_name,
            "safetensors": data.get('lora_file', step_name + '.safetensors'),
            "previews": data.get('previews', []),
            "json_file": fname,
            "avr_loss": data.get('avr_loss', None),
        })
    return steps


def cleanup_non_representative_loras(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    visual_card_id: str = "",
) -> dict:
    """대표로 설정된 LoRA 외에 해당 캐릭터의 모든 LoRA를 정리.
    - 대표가 설정된 세션: 대표 step만 남기고 나머지 step 삭제
    - 대표가 없는 세션: 세션 전체 삭제
    """
    if not lora_load_path:
        print("[BOT_LORA_CLEANUP] lora_load_path 미설정")
        return {"success": False, "error": "lora_load_path 미설정"}

    entry_dir = _trained_lora_dir(
        lora_load_path, bot_name, project_name, char_name, visual_card_id
    )
    if not os.path.isdir(entry_dir):
        print(f"[BOT_LORA_CLEANUP] 캐릭터 LoRA 폴더 없음: {entry_dir}")
        return {"success": False, "error": "캐릭터 LoRA 폴더가 없습니다"}

    manage_data = _load_bot_lora_manage()
    char_cfg = _get_char_config(
        manage_data, bot_name, project_name, char_name, visual_card_id
    ) or {}
    session_reps = char_cfg.get("session_representatives", {})

    deleted_sessions = []
    deleted_steps = []
    errors = []

    for session_name in sorted(os.listdir(entry_dir)):
        session_dir = os.path.join(entry_dir, session_name)
        if not os.path.isdir(session_dir):
            continue

        rep_json = session_reps.get(session_name, "")
        rep_safetensors = ""
        if rep_json:
            try:
                rep_data = json.loads(rep_json)
                rep_safetensors = rep_data.get("safetensors", "")
            except Exception:
                pass

        # 대표가 없는 세션: 전체 삭제
        if not rep_safetensors:
            try:
                file_count = sum(1 for _ in os.listdir(session_dir))
                shutil.rmtree(session_dir)
                deleted_sessions.append(session_name)
                if session_name in session_reps:
                    del session_reps[session_name]
                print(f"[BOT_LORA_CLEANUP] 대표 없는 세션 삭제: {session_name} ({file_count}개 파일)")
            except Exception as e:
                errors.append(f"세션 {session_name} 삭제 실패: {e}")
                print(f"[BOT_LORA_CLEANUP] 세션 삭제 실패: {session_dir} - {e}")
                traceback.print_exc()
            continue

        # 대표가 있는 세션: 대표 step만 남기고 나머지 삭제
        for fname in sorted(os.listdir(session_dir)):
            if not fname.endswith('.json'):
                continue
            step_name = os.path.splitext(fname)[0]
            json_path = os.path.join(session_dir, fname)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                errors.append(f"JSON 읽기 실패 {fname}: {e}")
                continue

            st_name = data.get('lora_file', step_name + '.safetensors')

            # 대표 safetensors면 유지
            if st_name == rep_safetensors:
                continue

            # 대표가 아닌 step 삭제
            # safetensors
            fp = os.path.join(session_dir, st_name)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                    deleted_steps.append(f"{session_name}/{st_name}")
                except Exception as e:
                    errors.append(f"{st_name}: {e}")
            # previews
            for p in data.get('previews', []):
                fp = os.path.join(session_dir, p)
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                    except Exception as e:
                        errors.append(f"{p}: {e}")
            # toml
            toml_path = os.path.join(session_dir, step_name + ".toml")
            if os.path.isfile(toml_path):
                try:
                    os.remove(toml_path)
                except Exception as e:
                    errors.append(f"{step_name}.toml: {e}")
            # json
            try:
                os.remove(json_path)
                deleted_steps.append(f"{session_name}/{step_name}")
            except Exception as e:
                errors.append(f"{fname}: {e}")

            print(f"[BOT_LORA_CLEANUP] 비대표 step 삭제: {session_name}/{step_name}")

    # 삭제된 세션들을 session_representatives 및 session_priority에서 정리 후 저장
    if deleted_sessions:
        session_prio = char_cfg.get("session_priority") or []
        if session_prio:
            deleted_set = set(deleted_sessions)
            char_cfg["session_priority"] = [s for s in session_prio if s not in deleted_set]
        _save_bot_lora_manage(manage_data)

    result = {
        "success": True,
        "deleted_sessions": deleted_sessions,
        "deleted_steps": deleted_steps,
        "errors": errors,
    }
    print(f"[BOT_LORA_CLEANUP] 정리 완료: 세션 {len(deleted_sessions)}개 삭제, step {len(deleted_steps)}개 삭제")
    return result


def read_bot_toml_file(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    session: str,
    step_name: str,
    visual_card_id: str = "",
) -> dict:
    if not lora_load_path:
        print("[BOT_LORA_TRAINED] TOML 조회 실패: lora_load_path 미설정")
        return {"success": False, "error": "lora_load_path 미설정"}
    session_dir = os.path.join(
        _trained_lora_dir(
            lora_load_path, bot_name, project_name, char_name, visual_card_id
        ),
        session,
    )
    toml_path = os.path.join(session_dir, step_name + ".toml")
    if not os.path.isfile(toml_path):
        print(f"[BOT_LORA_TRAINED] TOML 파일 없음: {toml_path}")
        return {"success": False, "error": "TOML 파일이 없습니다"}
    try:
        with open(toml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"success": True, "content": content, "filename": step_name + ".toml"}
    except Exception as e:
        print(f"[BOT_LORA_TRAINED] TOML 읽기 실패: {toml_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def delete_bot_trained_step(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    session: str,
    step_name: str,
    visual_card_id: str = "",
) -> dict:
    if not lora_load_path:
        print("[BOT_LORA_TRAINED] step 삭제 실패: lora_load_path 미설정")
        return {"success": False, "error": "lora_load_path 미설정"}
    session_dir = os.path.join(
        _trained_lora_dir(
            lora_load_path, bot_name, project_name, char_name, visual_card_id
        ),
        session,
    )
    if not os.path.isdir(session_dir):
        print(f"[BOT_LORA_TRAINED] step 삭제 실패: 세션 폴더 없음: {session_dir}")
        return {"success": False, "error": "세션 폴더 없음"}
    json_path = os.path.join(session_dir, step_name + ".json")
    if not os.path.isfile(json_path):
        print(f"[BOT_LORA_TRAINED] step 삭제 실패: JSON 파일 없음: {json_path}")
        return {"success": False, "error": "JSON 파일 없음"}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[BOT_LORA_TRAINED] step JSON 읽기 실패: {json_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": f"JSON 읽기 실패: {e}"}
    deleted = []
    errors = []
    st_name = data.get('lora_file', step_name + '.safetensors')
    fp = os.path.join(session_dir, st_name)
    if os.path.isfile(fp):
        try: os.remove(fp); deleted.append(st_name)
        except Exception as e: errors.append(f"{st_name}: {e}")
    for p in data.get('previews', []):
        fp = os.path.join(session_dir, p)
        if os.path.isfile(fp):
            try: os.remove(fp); deleted.append(p)
            except Exception as e: errors.append(f"{p}: {e}")
    toml_path = os.path.join(session_dir, step_name + ".toml")
    if os.path.isfile(toml_path):
        try: os.remove(toml_path); deleted.append(step_name + ".toml")
        except Exception as e: errors.append(f"{step_name}.toml: {e}")
    try: os.remove(json_path); deleted.append(step_name + ".json")
    except Exception as e: errors.append(f"{step_name}.json: {e}")
    if errors:
        print(f"[BOT_LORA_TRAINED] 삭제 중 일부 실패: {errors}")
    return {"success": True, "deleted": deleted, "errors": errors}


def delete_bot_trained_session(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    session: str,
    visual_card_id: str = "",
) -> dict:
    if not lora_load_path:
        print("[BOT_LORA_TRAINED] 세션 삭제 실패: lora_load_path 미설정")
        return {"success": False, "error": "lora_load_path 미설정"}
    session_dir = os.path.join(
        _trained_lora_dir(
            lora_load_path, bot_name, project_name, char_name, visual_card_id
        ),
        session,
    )
    if not os.path.isdir(session_dir):
        print(f"[BOT_LORA_TRAINED] 세션 삭제 실패: 세션 폴더 없음: {session_dir}")
        return {"success": False, "error": "세션 폴더 없음"}
    try:
        file_count = sum(1 for _ in os.listdir(session_dir))
        shutil.rmtree(session_dir)
        # 세션 대표 설정 및 우선순위에서도 해당 세션 제거
        manage_data = _load_bot_lora_manage()
        char_cfg = _get_char_config(
            manage_data, bot_name, project_name, char_name, visual_card_id
        ) or {}
        session_reps = char_cfg.get("session_representatives", {})
        session_prio = char_cfg.get("session_priority") or []
        changed = False
        if session in session_reps:
            del session_reps[session]
            changed = True
        if session in session_prio:
            char_cfg["session_priority"] = [s for s in session_prio if s != session]
            changed = True
        if changed:
            _save_bot_lora_manage(manage_data)
            print(f"[BOT_LORA_TRAINED] 세션 관리정보에서 제거(대표/우선순위): {session}")
        print(f"[BOT_LORA_TRAINED] 세션 폴더 삭제 완료: {session_dir} ({file_count}개 파일)")
        return {"success": True, "deleted_session": session, "file_count": file_count}
    except Exception as e:
        print(f"[BOT_LORA_TRAINED] 세션 폴더 삭제 실패: {session_dir} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_bot_trained_preview_path(
    lora_load_path: str,
    bot_name: str,
    project_name: str,
    char_name: str,
    session: str,
    filename: str,
    visual_card_id: str = "",
) -> str:
    if not lora_load_path:
        print("[BOT_LORA_TRAINED] 프리뷰 조회 실패: lora_load_path 미설정")
        return ""
    path = os.path.join(
        _trained_lora_dir(
            lora_load_path, bot_name, project_name, char_name, visual_card_id
        ),
        session,
        filename,
    )
    if os.path.isfile(path):
        return path
    print(f"[BOT_LORA_TRAINED] 프리뷰 파일 없음: {path}")
    return ""


# ─── 학습 이미지 Export ──────────────────────────────────────

def export_bot_training_images(
    bot_name: str,
    project_name: str,
    char_name: str,
    comfy_input_dir: str,
    folder_name: str = "soya_lora",
    visual_card_id: str = "",
) -> dict:
    """프로젝트 폴더의 학습 이미지를 Comfy Input 폴더로 복사"""
    proj_char_dir = _bot_project_training_dir(
        bot_name, project_name, char_name, visual_card_id
    )
    if not os.path.isdir(proj_char_dir):
        print(f"[BOT_LORA_EXPORT] 프로젝트 캐릭터 폴더 없음: {proj_char_dir}")
        return {"success": False, "error": f"프로젝트 학습 이미지 폴더 없음: {bot_name}/{project_name}/{char_name}"}

    image_files = []
    for fname in sorted(os.listdir(proj_char_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        image_files.append(os.path.join(proj_char_dir, fname))

    if not image_files:
        print(f"[BOT_LORA_EXPORT] 학습 이미지 없음: {bot_name}/{char_name}")
        return {"success": False, "error": "학습용 이미지가 없습니다"}

    target_dir = os.path.join(comfy_input_dir, folder_name)
    if os.path.isdir(target_dir):
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            try:
                if os.path.isfile(item_path): os.remove(item_path)
                elif os.path.isdir(item_path): shutil.rmtree(item_path)
            except Exception as e:
                print(f"[BOT_LORA_EXPORT] 삭제 실패: {item_path} - {e}")
    else:
        os.makedirs(target_dir, exist_ok=True)

    exported = []
    errors = []
    for idx, src_path in enumerate(image_files, start=1):
        ext = os.path.splitext(src_path)[1]
        dest_name = format_lora_export_filename(idx, len(image_files), ext)
        dest_path = os.path.join(target_dir, dest_name)
        try:
            shutil.copy2(src_path, dest_path)
            exported.append(dest_name)
        except Exception as e:
            print(f"[BOT_LORA_EXPORT] 복사 실패: {src_path} -> {dest_path} - {e}")
            traceback.print_exc()
            errors.append({"filename": os.path.basename(src_path), "reason": str(e)})

    return {"success": True, "exported": exported, "errors": errors, "target_dir": target_dir, "count": len(exported)}


# ─── 학습 이미지 파일 서빙 ──────────────────────────────────

def get_bot_training_image_path(
    bot_name: str,
    project_name: str,
    char_name: str,
    filename: str,
    visual_card_id: str = "",
) -> str | None:
    """프로젝트 폴더에서 학습 이미지 경로 반환"""
    if ".." in filename or os.path.sep in filename:
        print(f"[BOT_LORA] 잘못된 파일명: {filename}")
        return None
    fpath = os.path.join(
        _bot_project_training_dir(
            bot_name, project_name, char_name, visual_card_id
        ),
        filename,
    )
    if os.path.isfile(fpath):
        return fpath
    print(f"[BOT_LORA] 학습 이미지 없음: {fpath}")
    return None


def get_bot_char_image_path(bot_name: str, char_name: str, filename: str) -> str | None:
    """봇 캐릭터 원본 이미지 경로 반환"""
    if ".." in filename or os.path.sep in filename:
        print(f"[BOT_LORA] 잘못된 파일명: {filename}")
        return None
    fpath = os.path.join(_bot_char_dir(bot_name, char_name), filename)
    if os.path.isfile(fpath):
        return fpath
    print(f"[BOT_LORA] 캐릭터 이미지 없음: {fpath}")
    return None


def list_bot_lora_for_picker(lora_load_path: str = "") -> list:
    """LoRA 피커용 목록 반환. 봇→프로젝트→캐릭터 계층 + 대표 safetensors 경로 포함."""
    data = _load_bot_lora_manage()
    result = []
    for bot_name, projects in data.get("bot_loras", {}).items():
        bot_group = {"bot_name": bot_name, "projects": []}
        for proj_name, proj_data in projects.items():
            proj_entry = {"project_name": proj_name, "characters": []}
            training_config = proj_data.get("training_config", {})
            for char_name, visual_card_id, unit_cfg in _iter_project_units(proj_data):
                # skip_training은 순차 학습 큐에서만 제외하는 설정이다. 이미
                # 선택된 대표 LoRA의 피커 노출과 원격 동기화에는 영향을 주지 않는다.
                session_reps = unit_cfg.get("session_representatives", {})
                if not session_reps:
                    continue
                # 대표가 설정된 가장 최신 세션 찾기
                rep_path = ""
                rep_preview = ""
                rep_session = ""
                for sname in sorted(session_reps.keys(), reverse=True):
                    rep_str = session_reps[sname]
                    if not rep_str:
                        continue
                    try:
                        rep_data = json.loads(rep_str)
                    except Exception:
                        continue
                    safetensors = rep_data.get("safetensors", "")
                    preview = rep_data.get("preview", "")
                    if not safetensors:
                        continue
                    # 실제 파일 존재 확인
                    if lora_load_path:
                        full_dir = _trained_lora_dir(
                            lora_load_path,
                            bot_name,
                            proj_name,
                            char_name,
                            visual_card_id,
                        )
                        if os.path.isfile(os.path.join(full_dir, sname, safetensors)):
                            rep_path = os.path.relpath(
                                os.path.join(full_dir, sname, safetensors),
                                lora_load_path,
                            )
                            rep_preview = preview
                            rep_session = sname
                            break
                if not rep_path:
                    continue
                proj_entry["characters"].append({
                    "char_name": char_name,
                    "visual_card_id": visual_card_id,
                    "visual_card_label": (
                        unit_cfg.get("label") or visual_card_id or "기본 카드"
                    ),
                    "trigger": _effective_character_trigger(
                        proj_data,
                        char_name,
                        visual_card_id,
                        unit_cfg,
                    ),
                    "lora_path": rep_path,
                    "preview_url": rep_preview,
                    "session": rep_session,
                    "BASE": training_config.get("profile", "anima"),
                })
            if proj_entry["characters"]:
                bot_group["projects"].append(proj_entry)
        if bot_group["projects"]:
            result.append(bot_group)
    return result
