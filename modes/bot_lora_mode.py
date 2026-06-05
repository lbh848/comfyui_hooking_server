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


def _trained_lora_dir(lora_load_path: str, bot_name: str, project_name: str, char_name: str) -> str:
    """학습된 LoRA 경로: <lora_load_path>/<봇>/Lora/<프로젝트>/<캐릭터>/"""
    return os.path.join(lora_load_path, _safe_dirname(bot_name), "Lora", _safe_dirname(project_name), _safe_dirname(char_name))


# ─── 학습 이미지 프로젝트 동기화 ───────────────────────────────

def _sync_training_images_to_project(bot_name: str, project_name: str, char_name: str, rep_images: list, include_face: bool = True) -> dict:
    """원본 캐릭터 폴더의 학습 이미지를 프로젝트 폴더로 복사 (기존 파일 유지)"""
    src_dir = _bot_char_dir(bot_name, char_name)
    dst_dir = _bot_project_char_dir(bot_name, project_name, char_name)
    if not os.path.isdir(src_dir):
        print(f"[BOT_LORA_SYNC] 원본 캐릭터 폴더 없음: {src_dir}")
        return {"synced": 0, "skipped": 0}

    os.makedirs(dst_dir, exist_ok=True)

    # 복사할 파일 목록: rep_images + _face_image.webp
    files_to_copy = []
    for fname in rep_images:
        files_to_copy.append(fname)
    if include_face and os.path.isfile(os.path.join(src_dir, "_face_image.webp")):
        files_to_copy.append("_face_image.webp")

    synced = 0
    skipped = 0
    prompts_synced = 0
    for fname in files_to_copy:
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

    print(f"[BOT_LORA_SYNC] 완료: {bot_name}/{project_name}/{char_name} - 이미지 복사:{synced}, 스킵:{skipped}, 프롬프트:{prompts_synced}")
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


def _get_char_config(data: dict, bot_name: str, project_name: str, char_name: str) -> dict | None:
    """프로젝트 내 캐릭터 설정 조회"""
    proj = _get_project_config(data, bot_name, project_name)
    if not proj:
        return None
    return proj.get("characters", {}).get(char_name)


# ─── 캐릭터 임포트 ──────────────────────────────────────────

def list_importable_characters(bot_name: str, project_name: str) -> dict:
    """봇에는 있지만 프로젝트에는 없는 캐릭터 목록 반환"""
    if not bot_name or not project_name:
        print("[BOT_LORA_IMPORT] 봇/프로젝트 이름 누락")
        return {"success": False, "error": "봇/프로젝트 이름 필수"}

    bot_data = _load_bot_data()
    bot_chars = []
    for b in bot_data.get("bots", []):
        if b.get("name") == bot_name:
            for ch in b.get("characters", []):
                cn = ch.get("name", "")
                if cn:
                    bot_chars.append({
                        "name": cn,
                        "rep_images": ch.get("rep_images", []),
                        "has_face_image": os.path.isfile(
                            os.path.join(BOT_DIR, bot_name, cn, "_face_image.webp")
                        ),
                    })
            break

    manage_data = _load_bot_lora_manage()
    proj = _get_project_config(manage_data, bot_name, project_name)
    existing = set(proj.get("characters", {}).keys()) if proj else set()

    importable = [ch for ch in bot_chars if ch["name"] not in existing]
    print(f"[BOT_LORA_IMPORT] 임포트 가능 캐릭터: {len(importable)}명 (기존 {len(existing)}명)")
    return {"success": True, "characters": importable}


def import_characters(bot_name: str, project_name: str, char_names: list, face_chars: list | None = None) -> dict:
    """선택한 캐릭터를 프로젝트에 추가"""
    if not bot_name or not project_name:
        print("[BOT_LORA_IMPORT] 봇/프로젝트 이름 누락")
        return {"success": False, "error": "봇/프로젝트 이름 필수"}
    if not char_names:
        print("[BOT_LORA_IMPORT] 선택된 캐릭터 없음")
        return {"success": False, "error": "임포트할 캐릭터를 선택하세요"}

    if face_chars is None:
        face_chars = []

    bot_data = _load_bot_data()
    manage_data = _load_bot_lora_manage()
    proj = _get_project_config(manage_data, bot_name, project_name)
    if not proj:
        print(f"[BOT_LORA_IMPORT] 프로젝트 없음: {bot_name}/{project_name}")
        return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

    existing_chars = proj.setdefault("characters", {})
    added = []

    for b in bot_data.get("bots", []):
        if b.get("name") == bot_name:
            for ch in b.get("characters", []):
                cn = ch.get("name", "")
                if cn in char_names and cn not in existing_chars:
                    existing_chars[cn] = {"trigger": cn}
                    include_face = cn in face_chars
                    _sync_training_images_to_project(bot_name, project_name, cn, ch.get("rep_images", []), include_face)
                    added.append(cn)
            break

    if added:
        _save_bot_lora_manage(manage_data)
        print(f"[BOT_LORA_IMPORT] 캐릭터 임포트 완료: {added}")
    else:
        print("[BOT_LORA_IMPORT] 임포트할 새 캐릭터가 없음")

    return {"success": True, "added": added, "count": len(added)}


def remove_character_from_project(bot_name: str, project_name: str, char_name: str) -> dict:
    """프로젝트에서 캐릭터 제거"""
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

    del characters[char_name]
    _save_bot_lora_manage(manage_data)

    # 프로젝트 내 캐릭터 폴더 삭제 (학습 이미지, 캐릭터별 테스트 이미지)
    char_dir = _bot_project_char_dir(bot_name, project_name, char_name)
    if os.path.isdir(char_dir):
        try:
            shutil.rmtree(char_dir)
            print(f"[BOT_LORA_REMOVE] 캐릭터 폴더 삭제: {char_dir}")
        except Exception as e:
            print(f"[BOT_LORA_REMOVE] 캐릭터 폴더 삭제 실패: {char_dir} - {e}")

    print(f"[BOT_LORA_REMOVE] 캐릭터 제거 완료: {bot_name}/{project_name}/{char_name}")
    return {"success": True}


# ─── 봇 목록 ─────────────────────────────────────────────────

def list_bots() -> list:
    """LoRA 학습 가능한 봇 목록 반환"""
    bot_data = _load_bot_data()
    result = []
    for bot in bot_data.get("bots", []):
        chars = []
        for ch in bot.get("characters", []):
            chars.append({
                "name": ch.get("name", ""),
                "rep_images": ch.get("rep_images", []),
                "has_face_image": os.path.isfile(
                    os.path.join(BOT_DIR, bot["name"], ch.get("name", ""), "_face_image.webp")
                ),
            })
        result.append({
            "name": bot.get("name", ""),
            "characters": chars,
        })
    return result


def rename_bot_in_manage(old_name: str, new_name: str):
    """삽화 모드에서 봇 이름 변경 시 로라매니징의 키도 같이 변경"""
    data = _load_bot_lora_manage()
    bot_loras = data.get("bot_loras", {})
    if old_name in bot_loras:
        bot_loras[new_name] = bot_loras.pop(old_name)
        _save_bot_lora_manage(data)
        print(f"[BOT_LORA] 봇 이름 동기화: {old_name} -> {new_name}")
    else:
        print(f"[BOT_LORA] 봇 '{old_name}' 로라매니징에 없음, 스킵")


# ─── 프로젝트 관리 ────────────────────────────────────────────

def list_projects(bot_name: str) -> list:
    """봇의 학습 프로젝트 목록 반환"""
    data = _load_bot_lora_manage()
    projects = []
    for pname, pinfo in data.get("bot_loras", {}).get(bot_name, {}).items():
        if not isinstance(pinfo, dict):
            continue
        char_count = len(pinfo.get("characters", {}))
        training_config = pinfo.get("training_config", {})
        projects.append({
            "name": pname,
            "character_count": char_count,
            "profile": training_config.get("profile", "anima"),
        })
    return projects


def add_project(bot_name: str, project_name: str, selected_chars: list | None = None, face_chars: list | None = None) -> dict:
    """새 학습 프로젝트 추가 (selected_chars: 포함할 캐릭터, face_chars: 얼굴 이미지 포함할 캐릭터)"""
    if not bot_name:
        return {"success": False, "error": "봇 이름 누락"}
    if not project_name or not project_name.strip():
        return {"success": False, "error": "프로젝트 이름을 입력하세요"}

    project_name = project_name.strip()
    data = _load_bot_lora_manage()

    bot_projects = data.setdefault("bot_loras", {}).setdefault(bot_name, {})
    if project_name in bot_projects:
        print(f"[BOT_LORA] 프로젝트 이미 존재: {bot_name}/{project_name}")
        return {"success": False, "error": "이미 존재하는 프로젝트명입니다"}

    # 프로젝트 폴더 생성
    project_dir = _bot_project_dir(bot_name, project_name)
    os.makedirs(project_dir, exist_ok=True)

    if face_chars is None:
        face_chars = []

    # 캐릭터별 기본 설정 생성
    bot_data = _load_bot_data()
    characters = {}
    for b in bot_data.get("bots", []):
        if b.get("name") == bot_name:
            for ch in b.get("characters", []):
                cn = ch.get("name", "")
                if cn:
                    if selected_chars is None or cn in selected_chars:
                        characters[cn] = {"trigger": cn}
            break

    bot_projects[project_name] = {
        "training_config": {},
        "characters": characters,
    }
    _save_bot_lora_manage(data)

    # 프로젝트 생성 시에만 학습 이미지 동기화
    for ch_entry in characters:
        bot_data2 = _load_bot_data()
        for b in bot_data2.get("bots", []):
            if b.get("name") == bot_name:
                for ch in b.get("characters", []):
                    if ch.get("name") == ch_entry:
                        include_face = ch_entry in face_chars
                        _sync_training_images_to_project(bot_name, project_name, ch_entry, ch.get("rep_images", []), include_face)
                        break
                break

    print(f"[BOT_LORA] 프로젝트 추가: {bot_name}/{project_name} ({len(characters)}명)")
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
    for char_name, char_data in dst_cfg.get("characters", {}).items():
        if "session_representatives" in char_data:
            del char_data["session_representatives"]

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
    char_configs = proj_cfg.get("characters", {})

    characters = []
    for char_name in char_configs:
        if not char_name:
            continue

        training_images = _get_project_training_images(bot_name, project_name, char_name)
        char_cfg = char_configs.get(char_name, {})
        trigger = char_cfg.get("trigger", "") or char_name
        trained_sessions = _list_bot_trained_sessions(lora_load_path, bot_name, project_name, char_name) if lora_load_path else []

        characters.append({
            "name": char_name,
            "trigger": trigger,
            "skip_training": char_cfg.get("skip_training", False),
            "training_images": training_images,
            "trained_sessions": trained_sessions,
            "session_representatives": char_cfg.get("session_representatives", {}),
            "char_test_images": list_bot_char_test_images(bot_name, project_name, char_name),
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


def _get_project_training_images(bot_name: str, project_name: str, char_name: str) -> list:
    """프로젝트 폴더에서 학습 이미지 목록 반환"""
    proj_char_dir = _bot_project_char_dir(bot_name, project_name, char_name)
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

def _bot_char_test_dir(bot_name: str, project_name: str, char_name: str) -> str:
    """캐릭터별 테스트 이미지 폴더: bot/<봇>/Lora/<프로젝트>/<캐릭터>/_test/"""
    return os.path.join(_bot_project_char_dir(bot_name, project_name, char_name), TEST_DIR_NAME)


def list_bot_char_test_images(bot_name: str, project_name: str, char_name: str) -> list:
    """캐릭터별 테스트 이미지 목록 반환"""
    t_dir = _bot_char_test_dir(bot_name, project_name, char_name)
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


def add_bot_char_test_images(bot_name: str, project_name: str, char_name: str, sources: list) -> dict:
    """에셋에서 캐릭터별 테스트 이미지 추가"""
    from modes.asset_mode import ASSET_DIR, AssetMode

    t_dir = _bot_char_test_dir(bot_name, project_name, char_name)
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


def copy_project_test_to_char(bot_name: str, project_name: str, char_name: str, filenames: list = None) -> dict:
    """프로젝트 공통 테스트 이미지를 캐릭터 _test/ 로 복제"""
    src_dir = _bot_test_dir(bot_name, project_name)
    dst_dir = _bot_char_test_dir(bot_name, project_name, char_name)

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


def delete_bot_char_test_image(bot_name: str, project_name: str, char_name: str, filename: str) -> dict:
    """캐릭터별 테스트 이미지 삭제"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_char_test_dir(bot_name, project_name, char_name)
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


def save_bot_char_test_prompt(bot_name: str, project_name: str, char_name: str, filename: str, positive: str, negative: str) -> dict:
    """캐릭터별 테스트 이미지 프롬프트 저장"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_char_test_dir(bot_name, project_name, char_name)
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


def get_bot_char_test_image_path(bot_name: str, project_name: str, char_name: str, filename: str) -> str | None:
    """캐릭터별 테스트 이미지 파일 경로 반환"""
    if ".." in filename or os.path.sep in filename:
        print(f"[BOT_LORA] 잘못된 파일명: {filename}")
        return None
    fpath = os.path.join(_bot_char_test_dir(bot_name, project_name, char_name), filename)
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


def update_char_trigger(bot_name: str, project_name: str, char_name: str, trigger: str) -> dict:
    if not bot_name or not project_name or not char_name:
        return {"success": False, "error": "봇/프로젝트/캐릭터 이름 누락"}
    data = _load_bot_lora_manage()
    char_cfg = data.setdefault("bot_loras", {}).setdefault(bot_name, {}).setdefault(project_name, {}).setdefault("characters", {}).setdefault(char_name, {})
    char_cfg["trigger"] = trigger.strip()
    _save_bot_lora_manage(data)
    print(f"[BOT_LORA] trigger 업데이트: {bot_name}/{project_name}/{char_name} -> {trigger.strip()}")
    return {"success": True}


def update_char_skip_training(bot_name: str, project_name: str, char_name: str, skip: bool) -> dict:
    if not bot_name or not project_name or not char_name:
        return {"success": False, "error": "봇/프로젝트/캐릭터 이름 누락"}
    data = _load_bot_lora_manage()
    char_cfg = data.setdefault("bot_loras", {}).setdefault(bot_name, {}).setdefault(project_name, {}).setdefault("characters", {}).setdefault(char_name, {})
    char_cfg["skip_training"] = bool(skip)
    _save_bot_lora_manage(data)
    print(f"[BOT_LORA] skip_training 업데이트: {bot_name}/{project_name}/{char_name} -> {bool(skip)}")
    return {"success": True}


def update_char_session_representative(bot_name: str, project_name: str, char_name: str, session_name: str, representative: str) -> dict:
    if not bot_name or not project_name or not char_name:
        return {"success": False, "error": "봇/프로젝트/캐릭터 이름 누락"}
    data = _load_bot_lora_manage()
    char_cfg = data.setdefault("bot_loras", {}).setdefault(bot_name, {}).setdefault(project_name, {}).setdefault("characters", {}).setdefault(char_name, {})
    if "session_representatives" not in char_cfg:
        char_cfg["session_representatives"] = {}
    char_cfg["session_representatives"][session_name] = representative
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


def delete_bot_training_image(bot_name: str, project_name: str, char_name: str, filename: str) -> dict:
    """봇 LoRA 학습 이미지 + 프롬프트 JSON 삭제"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    t_dir = _bot_project_char_dir(bot_name, project_name, char_name)
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


def list_bot_char_available_images(bot_name: str, char_name: str) -> list:
    """봇 캐릭터 원본 폴더에서 사용 가능한 이미지 목록 반환"""
    char_dir = _bot_char_dir(bot_name, char_name)
    if not os.path.isdir(char_dir):
        print(f"[BOT_LORA] 봇 캐릭터 폴더 없음: {char_dir}")
        return []

    images = []
    for fname in sorted(os.listdir(char_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        fpath = os.path.join(char_dir, fname)
        img_data = _load_image_with_prompt(fpath, char_dir, fname)
        if img_data:
            if fname == "_face_image.webp":
                img_data["source"] = "face"
            else:
                img_data["source"] = "rep"
            images.append(img_data)

    return images


def add_bot_training_images(bot_name: str, project_name: str, char_name: str, sources: list) -> dict:
    """에셋에서 봇 LoRA 학습 이미지 추가"""
    from modes.asset_mode import ASSET_DIR, AssetMode

    t_dir = _bot_project_char_dir(bot_name, project_name, char_name)
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


def add_bot_training_from_bot(bot_name: str, project_name: str, char_name: str, filenames: list) -> dict:
    """봇 캐릭터 원본 폴더에서 학습 이미지를 프로젝트로 복사"""
    src_dir = _bot_char_dir(bot_name, char_name)
    dst_dir = _bot_project_char_dir(bot_name, project_name, char_name)
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
        src_path = os.path.join(src_dir, filename)
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
            prompt_src = os.path.join(src_dir, f"{base}_prompt.json")
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

def save_bot_training_prompt(bot_name: str, project_name: str, char_name: str, filename: str, positive: str, negative: str) -> dict:
    """프로젝트 폴더의 _prompt.json 수정"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}
    proj_char_dir = _bot_project_char_dir(bot_name, project_name, char_name)
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


# ─── 학습된 LoRA 관리 ────────────────────────────────────────

def _list_bot_trained_sessions(lora_load_path: str, bot_name: str, project_name: str, char_name: str) -> list:
    if not lora_load_path:
        print("[BOT_LORA_TRAINED] lora_load_path 미설정")
        return []
    entry_dir = _trained_lora_dir(lora_load_path, bot_name, project_name, char_name)
    if not os.path.isdir(entry_dir):
        return []

    manage_data = _load_bot_lora_manage()
    char_cfg = _get_char_config(manage_data, bot_name, project_name, char_name) or {}
    session_reps = char_cfg.get("session_representatives", {})

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
            except Exception:
                pass
        sessions.append({
            "name": name,
            "step_count": step_count,
            "has_final": has_final,
            "representative": session_rep,
            "preview_url": preview_url,
        })
    return sessions


def list_bot_trained_sessions(lora_load_path: str, bot_name: str, project_name: str, char_name: str) -> list:
    return _list_bot_trained_sessions(lora_load_path, bot_name, project_name, char_name)


def list_bot_trained_steps(lora_load_path: str, bot_name: str, project_name: str, char_name: str, session: str) -> list:
    if not lora_load_path:
        print("[BOT_LORA_TRAINED] lora_load_path 미설정")
        return []
    session_dir = os.path.join(_trained_lora_dir(lora_load_path, bot_name, project_name, char_name), session)
    if not os.path.isdir(session_dir):
        print(f"[BOT_LORA_TRAINED] 세션 폴더 없음: {session_dir}")
        return []
    steps = []
    for fname in sorted(os.listdir(session_dir)):
        if not fname.endswith('.json'):
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


def cleanup_non_representative_loras(lora_load_path: str, bot_name: str, project_name: str, char_name: str) -> dict:
    """대표로 설정된 LoRA 외에 해당 캐릭터의 모든 LoRA를 정리.
    - 대표가 설정된 세션: 대표 step만 남기고 나머지 step 삭제
    - 대표가 없는 세션: 세션 전체 삭제
    """
    if not lora_load_path:
        print("[BOT_LORA_CLEANUP] lora_load_path 미설정")
        return {"success": False, "error": "lora_load_path 미설정"}

    entry_dir = _trained_lora_dir(lora_load_path, bot_name, project_name, char_name)
    if not os.path.isdir(entry_dir):
        print(f"[BOT_LORA_CLEANUP] 캐릭터 LoRA 폴더 없음: {entry_dir}")
        return {"success": False, "error": "캐릭터 LoRA 폴더가 없습니다"}

    manage_data = _load_bot_lora_manage()
    char_cfg = _get_char_config(manage_data, bot_name, project_name, char_name) or {}
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

    # session_representatives 업데이트 저장
    if deleted_sessions:
        _save_bot_lora_manage(manage_data)

    result = {
        "success": True,
        "deleted_sessions": deleted_sessions,
        "deleted_steps": deleted_steps,
        "errors": errors,
    }
    print(f"[BOT_LORA_CLEANUP] 정리 완료: 세션 {len(deleted_sessions)}개 삭제, step {len(deleted_steps)}개 삭제")
    return result


def read_bot_toml_file(lora_load_path: str, bot_name: str, project_name: str, char_name: str, session: str, step_name: str) -> dict:
    if not lora_load_path:
        return {"success": False, "error": "lora_load_path 미설정"}
    session_dir = os.path.join(_trained_lora_dir(lora_load_path, bot_name, project_name, char_name), session)
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


def delete_bot_trained_step(lora_load_path: str, bot_name: str, project_name: str, char_name: str, session: str, step_name: str) -> dict:
    if not lora_load_path:
        return {"success": False, "error": "lora_load_path 미설정"}
    session_dir = os.path.join(_trained_lora_dir(lora_load_path, bot_name, project_name, char_name), session)
    if not os.path.isdir(session_dir):
        return {"success": False, "error": "세션 폴더 없음"}
    json_path = os.path.join(session_dir, step_name + ".json")
    if not os.path.isfile(json_path):
        return {"success": False, "error": "JSON 파일 없음"}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
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


def delete_bot_trained_session(lora_load_path: str, bot_name: str, project_name: str, char_name: str, session: str) -> dict:
    if not lora_load_path:
        return {"success": False, "error": "lora_load_path 미설정"}
    session_dir = os.path.join(_trained_lora_dir(lora_load_path, bot_name, project_name, char_name), session)
    if not os.path.isdir(session_dir):
        return {"success": False, "error": "세션 폴더 없음"}
    try:
        file_count = sum(1 for _ in os.listdir(session_dir))
        shutil.rmtree(session_dir)
        # 세션 대표 설정에서도 해당 세션 키 제거
        manage_data = _load_bot_lora_manage()
        char_cfg = _get_char_config(manage_data, bot_name, project_name, char_name) or {}
        session_reps = char_cfg.get("session_representatives", {})
        if session in session_reps:
            del session_reps[session]
            _save_bot_lora_manage(manage_data)
            print(f"[BOT_LORA_TRAINED] 세션 대표 설정에서 제거: {session}")
        print(f"[BOT_LORA_TRAINED] 세션 폴더 삭제 완료: {session_dir} ({file_count}개 파일)")
        return {"success": True, "deleted_session": session, "file_count": file_count}
    except Exception as e:
        print(f"[BOT_LORA_TRAINED] 세션 폴더 삭제 실패: {session_dir} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_bot_trained_preview_path(lora_load_path: str, bot_name: str, project_name: str, char_name: str, session: str, filename: str) -> str:
    if not lora_load_path:
        return ""
    path = os.path.join(_trained_lora_dir(lora_load_path, bot_name, project_name, char_name), session, filename)
    if os.path.isfile(path):
        return path
    return ""


# ─── 학습 이미지 Export ──────────────────────────────────────

def export_bot_training_images(bot_name: str, project_name: str, char_name: str, comfy_input_dir: str, folder_name: str = "soya_lora") -> dict:
    """프로젝트 폴더의 학습 이미지를 Comfy Input 폴더로 복사"""
    proj_char_dir = _bot_project_char_dir(bot_name, project_name, char_name)
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
        dest_name = f"[{idx}]{ext}"
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

def get_bot_training_image_path(bot_name: str, project_name: str, char_name: str, filename: str) -> str | None:
    """프로젝트 폴더에서 학습 이미지 경로 반환"""
    if ".." in filename or os.path.sep in filename:
        print(f"[BOT_LORA] 잘못된 파일명: {filename}")
        return None
    fpath = os.path.join(_bot_project_char_dir(bot_name, project_name, char_name), filename)
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
            for char_name, char_cfg in proj_data.get("characters", {}).items():
                if char_cfg.get("skip_training"):
                    continue
                session_reps = char_cfg.get("session_representatives", {})
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
                        full_dir = _trained_lora_dir(lora_load_path, bot_name, proj_name, char_name)
                        if os.path.isfile(os.path.join(full_dir, sname, safetensors)):
                            rep_path = os.path.join(
                                _safe_dirname(bot_name), "Lora",
                                _safe_dirname(proj_name), _safe_dirname(char_name),
                                sname, safetensors
                            )
                            rep_preview = preview
                            rep_session = sname
                            break
                if not rep_path:
                    continue
                proj_entry["characters"].append({
                    "char_name": char_name,
                    "trigger": char_cfg.get("trigger", char_name),
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
