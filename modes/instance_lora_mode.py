"""
Instance LoRA 매니징 모듈
- LV1 평면 구조: 봇/프로젝트 계층 없이 로라 단위로 관리
- 학습 이미지 = 프리뷰 이미지, 여러 장 등록 가능
- 태그 분석 후 1-pass 자동 학습
"""

import os
import json
import shutil
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_LORA_DIR = os.path.join(BASE_DIR, "instance_lora")
INSTANCE_LORA_MANAGE_FILE = os.path.join(BASE_DIR, "asset_data", "instance_lora_manage.json")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


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
