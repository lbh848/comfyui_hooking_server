"""
Style LoRA(그림체 로라) 매니징 모듈
- 평면 구조: 프로젝트(=그림체 로라 1개) 단일 계층 (과거 그룹>프로젝트 2단계 제거)
- 프로젝트가 학습 이미지 풀 + 학습 세션 + 프로젝트별 training_config 보유
- 인스턴스 로라(instance_lora_mode)의 함수형 API 구조를 미러.
- 태깅/정제/학습은 모두 수동 버튼 트리거 (자동 E2E 체인 없음).
- 이미지는 프로젝트 폴더에 새로 복사된다(원본 참조 X).

데이터 파일: asset_data/style_lora_manage.json
이미지 복사본: style_lora_data/{project_id}/{filename}
캡션 파일: style_lora_data/{project_id}/{base}_prompt.json
"""

import datetime
import hashlib
import json
import os
import shutil
import time
import traceback
from aiohttp import web

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE_LORA_DIR = os.path.join(BASE_DIR, "style_lora_data")
STYLE_LORA_MANAGE_FILE = os.path.join(BASE_DIR, "asset_data", "style_lora_manage.json")
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
BACKUP_DIR = os.path.join(BASE_DIR, "요구사항")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# 프로젝트별 학습 설정 디폴트 (ANIMA/SDXL 각각 독립 보관)
DEFAULT_PROFILE_SETTINGS = {
    "step_per_image": 125,
    "il_rate": 0.00025,
    "save_per_step": 25,
    "multi_img_folder_name": "soya_lora",
    "gen_w": 1024,
    "gen_h": 1024,
    "upscale": False,
    "resolution": 1024,
    "save_after": 0,
    "dim": 32,
    "alpha": 16,
}


# ─── 유틸 ──────────────────────────────────────────────────────

def _safe_dirname(name: str) -> str:
    return "".join(c for c in str(name) if c.isalnum() or c in (' ', '_', '-', '.')).strip() or "unnamed"


def _project_dir(project_id: str) -> str:
    return os.path.join(STYLE_LORA_DIR, _safe_dirname(project_id))


def _gen_id(name: str) -> str:
    base = _safe_dirname(name)
    short_hash = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:6]
    return f"{base}-{short_hash}"


# ─── JSON 로드/세이브 + 마이그레이션 ───────────────────────────

def _backup_file(path: str):
    """데이터 파일 덮어쓰기 전 요구사항/ 폴더에 백업."""
    try:
        if not os.path.isfile(path):
            return
        os.makedirs(BACKUP_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BACKUP_DIR, f"{os.path.basename(path)}.bak.{ts}")
        shutil.copy2(path, dst)
        print(f"[STYLE_LORA] 데이터 백업: {dst}")
    except Exception as e:
        print(f"[STYLE_LORA] 백업 실패({path}): {e}")
        traceback.print_exc()


def _migrate_legacy(data: dict) -> dict:
    """구 스키마(groups>projects, 최상위 settings)를 평면 projects 로 변환.
    파일시스템 이미지 디렉터리도 style_lora_data/{group}/{project}/ -> {project}/ 로 이동."""
    if "groups" not in data and "projects" not in data:
        return data
    if "projects" in data and "groups" not in data:
        return data  # 이미 신스키마

    print("[STYLE_LORA] 구 스키마 감지 → 평면 projects 로 마이그레이션")
    _backup_file(STYLE_LORA_MANAGE_FILE)

    new_projects = dict(data.get("projects", {}))
    for group_id, gdata in (data.get("groups") or {}).items():
        for project_id, pdata in (gdata.get("projects") or {}).items():
            # 프로젝트 id 충돌 회피
            pid = project_id
            if pid in new_projects:
                pid = f"{_safe_dirname(group_id)}_{project_id}"
            new_projects[pid] = pdata
            # 이미지 디렉터리 이동
            legacy_dir = os.path.join(STYLE_LORA_DIR, _safe_dirname(group_id), _safe_dirname(project_id))
            new_dir = _project_dir(pid)
            if os.path.isdir(legacy_dir) and legacy_dir != new_dir:
                try:
                    if os.path.isdir(new_dir):
                        # 병합: 파일 단위 이동
                        for fn in os.listdir(legacy_dir):
                            src = os.path.join(legacy_dir, fn)
                            dst = os.path.join(new_dir, fn)
                            if not os.path.exists(dst):
                                shutil.move(src, dst)
                    else:
                        shutil.move(legacy_dir, new_dir)
                    print(f"[STYLE_LORA] 디렉터리 이동: {legacy_dir} -> {new_dir}")
                except Exception as e:
                    print(f"[STYLE_LORA] 디렉터리 이동 실패({legacy_dir}): {e}")
                    traceback.print_exc()
            # 빈 그룹 폴더 정리
            gpath = os.path.join(STYLE_LORA_DIR, _safe_dirname(group_id))
            try:
                if os.path.isdir(gpath) and not os.listdir(gpath):
                    os.rmdir(gpath)
            except OSError:
                pass

    migrated = {"projects": new_projects}
    return migrated


def _load_data() -> dict:
    if not os.path.isfile(STYLE_LORA_MANAGE_FILE):
        return {"projects": {}}
    try:
        with open(STYLE_LORA_MANAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        migrated = _migrate_legacy(data)
        if migrated is not data:
            _save_data(migrated)
        return migrated
    except Exception as e:
        print(f"[STYLE_LORA] JSON 로드 실패: {e}")
        traceback.print_exc()
        return {"projects": {}}


def _save_data(data: dict):
    try:
        with open(STYLE_LORA_MANAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[STYLE_LORA] JSON 세이브 실패: {e}")
        traceback.print_exc()


# ─── 프로젝트 CRUD ─────────────────────────────────────────────

def list_projects() -> list:
    data = _load_data()
    result = []
    for project_id, pdata in data.get("projects", {}).items():
        images = pdata.get("images", [])
        sessions = pdata.get("sessions", {})
        entry = {
            "id": project_id,
            "name": pdata.get("name", project_id),
            "trigger": pdata.get("trigger", ""),
            "description": pdata.get("description", ""),
            "image_count": len(images),
            "first_image": images[0] if images else None,
            "usage_count": pdata.get("usage_count", 0),
            "has_anima": any(s.get("profile") == "anima" for s in sessions.values()),
            "has_sdxl": any(s.get("profile") == "sdxl" for s in sessions.values()),
            "created_at": pdata.get("created_at", ""),
        }
        if images:
            prompt_result = get_image_prompt(project_id, images[0])
            if prompt_result.get("success") and prompt_result.get("data"):
                entry["prompt"] = prompt_result["data"]
        result.append(entry)
    return result


def create_project(name: str, trigger: str = "", description: str = "") -> dict:
    name = (name or "").strip()
    if not name:
        return {"success": False, "error": "프로젝트 이름이 필요합니다"}
    data = _load_data()
    project_id = _gen_id(name)
    projects = data.setdefault("projects", {})
    if project_id in projects:
        return {"success": False, "error": "이미 존재하는 프로젝트입니다 (다시 시도하세요)"}

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    projects[project_id] = {
        "name": name,
        "trigger": (trigger or "").strip() or name,
        "description": description or "",
        "images": [],
        "test_images": [],
        "sessions": {},
        "training_config": {"anima": {}, "sdxl": {}},
        "usage_count": 0,
        "created_at": now,
    }
    _save_data(data)
    os.makedirs(_project_dir(project_id), exist_ok=True)
    print(f"[STYLE_LORA] 프로젝트 생성: {project_id} (name={name}, trigger={trigger})")
    return {"success": True, "id": project_id}


def delete_project(project_id: str, style_lora_load_path: str = "", _data: dict = None) -> dict:
    own_data = _data is None
    data = _data if _data is not None else _load_data()
    project_id = _safe_dirname(project_id)
    projects = data.get("projects", {})
    if project_id not in projects:
        if own_data:
            print(f"[STYLE_LORA] 삭제 대상 없음: {project_id}")
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    projects.pop(project_id, None)
    if own_data:
        _save_data(data)

    # 학습 이미지 폴더 삭제
    ppath = _project_dir(project_id)
    if os.path.isdir(ppath):
        try:
            shutil.rmtree(ppath)
        except Exception as e:
            print(f"[STYLE_LORA] 프로젝트 폴더 삭제 실패: {ppath} - {e}")

    # 학습 결과물 삭제 (anima/sdxl). 저장 경로 키: {project}
    storage_key = _safe_dirname(project_id)
    if style_lora_load_path:
        for profile in ("anima", "sdxl"):
            trained_dir = os.path.join(style_lora_load_path, profile, storage_key)
            if os.path.isdir(trained_dir):
                try:
                    shutil.rmtree(trained_dir)
                    print(f"[STYLE_LORA] 학습 결과 삭제: {trained_dir}")
                except Exception as e:
                    print(f"[STYLE_LORA] 학습 결과 삭제 실패: {trained_dir} - {e}")

    print(f"[STYLE_LORA] 프로젝트 삭제: {project_id}")
    return {"success": True}


def get_project_detail(project_id: str) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        print(f"[STYLE_LORA] 상세 조회 실패 - 없음: {project_id}")
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    return {
        "success": True,
        "data": {
            "id": project_id,
            "name": project.get("name", project_id),
            "trigger": project.get("trigger", ""),
            "description": project.get("description", ""),
            "images": project.get("images", []),
            "image_count": len(project.get("images", [])),
            "test_images": project.get("test_images", []),
            "sessions": project.get("sessions", {}),
            "training_config": project.get("training_config", {"anima": {}, "sdxl": {}}),
            "usage_count": project.get("usage_count", 0),
            "created_at": project.get("created_at", ""),
        },
    }


def update_project(project_id: str, trigger: str = None, description: str = None) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    if trigger is not None:
        project["trigger"] = trigger.strip()
    if description is not None:
        project["description"] = description
    _save_data(data)
    return {"success": True}


def increment_usage(project_id: str) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    project["usage_count"] = project.get("usage_count", 0) + 1
    _save_data(data)
    return {"success": True, "usage_count": project["usage_count"]}


# ─── 이미지 관리 ──────────────────────────────────────────────

def add_image(project_id: str, src_path: str, filename: str) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    dst_dir = _project_dir(project_id)
    os.makedirs(dst_dir, exist_ok=True)
    # 파일명 충돌 회피
    dst_name = filename
    if os.path.exists(os.path.join(dst_dir, dst_name)):
        stem, ext = os.path.splitext(filename)
        dst_name = f"{stem}_{int(time.time() * 1000) % 100000}{ext}"
    dst_path = os.path.join(dst_dir, dst_name)
    try:
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        print(f"[STYLE_LORA] 이미지 복사 실패: {src_path} -> {dst_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    images = project.setdefault("images", [])
    if dst_name not in images:
        images.append(dst_name)
    _save_data(data)

    print(f"[STYLE_LORA] 이미지 추가: {project_id}/{dst_name}")
    return {"success": True, "filename": dst_name}


def delete_image(project_id: str, filename: str) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    images = project.get("images", [])
    if filename not in images:
        return {"success": False, "error": "이미지가 목록에 없습니다"}

    images.remove(filename)
    _save_data(data)

    pdir = _project_dir(project_id)
    img_path = os.path.join(pdir, filename)
    if os.path.isfile(img_path):
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"[STYLE_LORA] 이미지 파일 삭제 실패: {img_path} - {e}")

    prompt_path = os.path.join(pdir, os.path.splitext(filename)[0] + "_prompt.json")
    if os.path.isfile(prompt_path):
        try:
            os.remove(prompt_path)
        except Exception:
            pass

    print(f"[STYLE_LORA] 이미지 삭제: {project_id}/{filename}")
    return {"success": True}


# ─── 테스트 이미지 관리 ─────────────────────────────────────────
# 학습 이미지(images)와 동일 폴더(_project_dir)에 저장하지만 별도의 test_images 배열로 추적.
# 학습 흐름과 분리된 테스트용 이미지 풀. 추가/삭제 시 add_image/delete_image 와 동일한
# 복사·충돌회피·파일제거 패턴을 따른다.

def add_test_image(project_id: str, src_path: str, filename: str) -> dict:
    """테스트 이미지로 복사 추가. test_images 배열에 기록."""
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        print(f"[STYLE_LORA] 테스트 이미지 추가 실패 - 프로젝트 없음: {project_id}")
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    dst_dir = _project_dir(project_id)
    os.makedirs(dst_dir, exist_ok=True)
    # 파일명 충돌 회피 (학습 이미지와 동일 폴더이므로 충돌 가능)
    dst_name = filename
    if os.path.exists(os.path.join(dst_dir, dst_name)):
        stem, ext = os.path.splitext(filename)
        dst_name = f"{stem}_{int(time.time() * 1000) % 100000}{ext}"
    dst_path = os.path.join(dst_dir, dst_name)
    try:
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        print(f"[STYLE_LORA] 테스트 이미지 복사 실패: {src_path} -> {dst_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    test_images = project.setdefault("test_images", [])
    if dst_name not in test_images:
        test_images.append(dst_name)
    _save_data(data)

    # 원본 에셋 프롬프트 시드: 소스의 {base}_prompt.json 이 있으면 읽어(원본 수정 금지 - 읽기만)
    # {dst_base}_test_prompt.json 으로 복사본 생성. original_positive/negative 보존.
    dst_base = os.path.splitext(dst_name)[0]
    try:
        src_dir = os.path.dirname(src_path)
        src_prompt_path = os.path.join(src_dir, os.path.splitext(filename)[0] + "_prompt.json")
        dst_prompt_path = os.path.join(dst_dir, f"{dst_base}_test_prompt.json")
        if os.path.isfile(src_prompt_path):
            with open(src_prompt_path, "r", encoding="utf-8") as pf:
                pdata = json.load(pf)
            seeded = {
                "positive": pdata.get("positive", pdata.get("original_positive", "")),
                "negative": pdata.get("negative", pdata.get("original_negative", "")),
                "original_positive": pdata.get("original_positive", pdata.get("positive", "")),
                "original_negative": pdata.get("original_negative", pdata.get("negative", "")),
            }
            with open(dst_prompt_path, "w", encoding="utf-8") as pf:
                json.dump(seeded, pf, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[STYLE_LORA] 테스트 프롬프트 시드 실패(무시): {filename} - {e}")

    print(f"[STYLE_LORA] 테스트 이미지 추가: {project_id}/{dst_name}")
    return {"success": True, "filename": dst_name}


def add_test_image_from_train(project_id: str, filename: str) -> dict:
    """현재 프로젝트의 학습 이미지를 테스트 이미지로 등록.
    학습 이미지와 같은 폴더에 이미 존재하므로 파일 복사 없이 test_images 배열에만 추가.
    학습 캡션({base}_prompt.json)이 있으면 {base}_test_prompt.json 으로 시드 복사."""
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        print(f"[STYLE_LORA] 테스트 이미지(학습) 추가 실패 - 프로젝트 없음: {project_id}")
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    filename = (filename or "").strip()
    if not filename:
        return {"success": False, "error": "filename 필수"}
    if filename not in project.get("images", []):
        print(f"[STYLE_LORA] 테스트 이미지(학습) 추가 실패 - 학습 이미지 아님: {project_id}/{filename}")
        return {"success": False, "error": "학습 이미지 목록에 없습니다"}

    test_images = project.setdefault("test_images", [])
    if filename in test_images:
        # 이미 테스트 이미지로 등록됨 — 멱등하게 성공 처리
        return {"success": True, "filename": filename, "skipped": True}
    test_images.append(filename)
    _save_data(data)

    # 학습 캡션 시드: {base}_prompt.json → {base}_test_prompt.json (있을 때만)
    pdir = _project_dir(project_id)
    dst_base = os.path.splitext(filename)[0]
    src_prompt_path = os.path.join(pdir, f"{dst_base}_prompt.json")
    dst_prompt_path = os.path.join(pdir, f"{dst_base}_test_prompt.json")
    if os.path.isfile(src_prompt_path) and not os.path.isfile(dst_prompt_path):
        try:
            with open(src_prompt_path, "r", encoding="utf-8") as pf:
                pdata = json.load(pf)
            seeded = {
                "positive": pdata.get("positive", pdata.get("original_positive", "")),
                "negative": pdata.get("negative", pdata.get("original_negative", "")),
                "original_positive": pdata.get("original_positive", pdata.get("positive", "")),
                "original_negative": pdata.get("original_negative", pdata.get("negative", "")),
            }
            with open(dst_prompt_path, "w", encoding="utf-8") as pf:
                json.dump(seeded, pf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[STYLE_LORA] 테스트 프롬프트 시드 실패(무시): {filename} - {e}")

    print(f"[STYLE_LORA] 테스트 이미지(학습에서) 추가: {project_id}/{filename}")
    return {"success": True, "filename": filename}


def _test_prompt_path(project_id: str, filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return os.path.join(_project_dir(_safe_dirname(project_id)), f"{base}_test_prompt.json")


def get_test_image_prompt(project_id: str, filename: str) -> dict:
    """테스트 이미지 1장의 프롬프트 조회. 없으면 빈 값."""
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    if filename not in project.get("test_images", []):
        return {"success": False, "error": "이미지가 테스트 목록에 없습니다"}

    prompt_path = _test_prompt_path(project_id, filename)
    if os.path.isfile(prompt_path):
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                pdata = json.load(f)
            return {"success": True, "data": {
                "positive": pdata.get("positive", ""),
                "negative": pdata.get("negative", ""),
                "original_positive": pdata.get("original_positive", ""),
                "original_negative": pdata.get("original_negative", ""),
            }}
        except Exception as e:
            print(f"[STYLE_LORA] 테스트 프롬프트 읽기 실패: {prompt_path} - {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    return {"success": True, "data": {"positive": "", "negative": "", "original_positive": "", "original_negative": ""}}


def save_test_image_prompt(project_id: str, filename: str, prompt_data: dict) -> dict:
    """테스트 이미지 프롬프트 저장. original_* 는 최초 1회 보존, positive/negative 만 갱신.
    학습 이미지와 파일명이 분리({base}_test_prompt.json)되며 프로젝트 폴더에 저장 → 원본 에셋 영향 없음."""
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    if filename not in project.get("test_images", []):
        return {"success": False, "error": "이미지가 테스트 목록에 없습니다"}

    prompt_path = _test_prompt_path(project_id, filename)
    existing = {}
    if os.path.isfile(prompt_path):
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as e:
            print(f"[STYLE_LORA] 기존 테스트 프롬프트 읽기 실패(무시): {prompt_path} - {e}")

    positive = (prompt_data.get("positive") or "").strip()
    negative = (prompt_data.get("negative") or "").strip()
    out = {
        "positive": positive,
        "negative": negative,
        "original_positive": existing.get("original_positive") or prompt_data.get("original_positive") or positive,
        "original_negative": existing.get("original_negative") or prompt_data.get("original_negative") or negative,
    }
    try:
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[STYLE_LORA] 테스트 프롬프트 저장 실패: {prompt_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    print(f"[STYLE_LORA] 테스트 프롬프트 저장: {project_id}/{filename}")
    return {"success": True}


def delete_test_image(project_id: str, filename: str) -> dict:
    """테스트 이미지 1건 삭제 (목록에서 제거 + 파일 삭제)."""
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        print(f"[STYLE_LORA] 테스트 이미지 삭제 실패 - 프로젝트 없음: {project_id}")
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    test_images = project.get("test_images", [])
    if filename not in test_images:
        print(f"[STYLE_LORA] 테스트 이미지 삭제 실패 - 목록에 없음: {project_id}/{filename}")
        return {"success": False, "error": "이미지가 테스트 목록에 없습니다"}

    test_images.remove(filename)
    _save_data(data)

    pdir = _project_dir(project_id)
    img_path = os.path.join(pdir, filename)
    # 학습 이미지와 같은 폴더이므로, 동일 파일명이 images 에도 있으면 파일은 남김
    if filename in project.get("images", []):
        print(f"[STYLE_LORA] 테스트 이미지 파일 유지(학습 이미지와 공유): {project_id}/{filename}")
    elif os.path.isfile(img_path):
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"[STYLE_LORA] 테스트 이미지 파일 삭제 실패: {img_path} - {e}")

    # 테스트 프롬프트 파일 정리 ({base}_test_prompt.json — 학습 캡션과 분리)
    tp = _test_prompt_path(project_id, filename)
    if os.path.isfile(tp):
        try:
            os.remove(tp)
        except Exception as e:
            print(f"[STYLE_LORA] 테스트 프롬프트 파일 삭제 실패: {tp} - {e}")

    print(f"[STYLE_LORA] 테스트 이미지 삭제: {project_id}/{filename}")
    return {"success": True}


def delete_images_bulk(project_id: str, filenames_to_remove) -> dict:
    """이미지 일괄 삭제(이미지 필터링용).
    - 메타데이터 images 리스트에서 일괄 제거 (단일 _save_data 1회)
    - 실제 이미지 파일 + 캡션(_prompt.json) 파일 삭제
    - 삭제 전 asset_data/style_lora_manage.json 을 요구사항/ 에 백업 (데이터 안전 규칙)
    반환: {success, deleted:int, failed:[...]}
    """
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    images = project.setdefault("images", [])
    remove_set = set(filenames_to_remove)

    # ── 메타데이터 백업 (덮어쓰기 전) ──
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"style_lora_manage.backup_{ts}.json"
        backup_path = os.path.join(BACKUP_DIR, backup_name)
        shutil.copy2(STYLE_LORA_MANAGE_FILE, backup_path)
        print(f"[STYLE_LORA] 메타데이터 백업: {backup_path}")
    except Exception as e:
        print(f"[STYLE_LORA] 경고: 메타데이터 백업 실패 - {e}")
        traceback.print_exc()

    # ── 메타데이터 images 리스트 갱신 (1회) ──
    new_images = [f for f in images if f not in remove_set]
    removed_in_meta = len(images) - len(new_images)
    project["images"] = new_images
    _save_data(data)

    # ── 실제 파일 삭제 ──
    pdir = _project_dir(project_id)
    deleted = 0
    failed = []
    for fn in filenames_to_remove:
        img_path = os.path.join(pdir, fn)
        ok = True
        if os.path.isfile(img_path):
            try:
                os.remove(img_path)
                deleted += 1
            except Exception as e:
                print(f"[STYLE_LORA] 이미지 파일 삭제 실패: {img_path} - {e}")
                failed.append({"filename": fn, "error": str(e)})
                ok = False
        # 캡션 파일도 함께 정리
        prompt_path = os.path.join(pdir, os.path.splitext(fn)[0] + "_prompt.json")
        if os.path.isfile(prompt_path):
            try:
                os.remove(prompt_path)
            except Exception:
                pass
        # 파일은 없었지만 메타에는 있었던 경우 카운트 보정
        if ok and not os.path.isfile(img_path):
            # 파일이 이미 없었으면 deleted 에 포함시키지 않음 (메타에서만 제거됨)
            pass

    print(f"[STYLE_LORA] 일괄 삭제: project={project_id} meta_removed={removed_in_meta} "
          f"file_deleted={deleted} failed={len(failed)}")
    return {"success": True, "deleted": deleted, "failed": failed,
            "meta_removed": removed_in_meta}


def get_image_path(project_id: str, filename: str) -> str:
    return os.path.join(_project_dir(_safe_dirname(project_id)), filename)


def list_images(project_id: str) -> list:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id, {})
    return project.get("images", [])


def save_image_prompt(project_id: str, filename: str, prompt_data: dict) -> dict:
    project_id = _safe_dirname(project_id)
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(_project_dir(project_id), f"{base}_prompt.json")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)
        return {"success": True}
    except Exception as e:
        print(f"[STYLE_LORA] 프롬프트 저장 실패: {prompt_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_image_prompt(project_id: str, filename: str) -> dict:
    project_id = _safe_dirname(project_id)
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(_project_dir(project_id), f"{base}_prompt.json")
    if not os.path.isfile(prompt_path):
        return {"success": False, "error": "프롬프트 없음"}
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return {"success": True, "data": json.load(f)}
    except Exception as e:
        print(f"[STYLE_LORA] 프롬프트 로드 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


def batch_set_negative(project_id: str, filenames, negative_tags: str) -> dict:
    """학습 이미지 캡션({base}_prompt.json)의 negative 필드를 일괄 덮어쓰기.
    positive 등 기존 필드는 보존. asset_mode.batch_set_negative 와 동일 패턴.
    반환: {success, total, success_count, fail_count, failed:[{filename,error}]}"""
    project_id = _safe_dirname(project_id)
    data = _load_data()
    project = data.get("projects", {}).get(project_id)
    if not project:
        print(f"[STYLE_LORA] 부정 프롬프트 일괄 적용 실패 - 프로젝트 없음: {project_id}")
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    images = set(project.get("images", []))
    pdir = _project_dir(project_id)
    negative_tags = (negative_tags or "").strip()

    success_count = 0
    fail_count = 0
    failed = []
    for fn in filenames:
        fn = (fn or "").strip()
        if not fn:
            continue
        if fn not in images:
            print(f"[STYLE_LORA] 부정 프롬프트 적용 스킵 - 학습 이미지 아님: {project_id}/{fn}")
            fail_count += 1
            failed.append({"filename": fn, "error": "학습 이미지 목록에 없음"})
            continue
        prompt_path = os.path.join(pdir, os.path.splitext(fn)[0] + "_prompt.json")
        existing = {}
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as pf:
                    existing = json.load(pf)
            except Exception as e:
                print(f"[STYLE_LORA] 기존 프롬프트 읽기 실패(무시): {prompt_path} - {e}")
        existing["negative"] = negative_tags
        try:
            with open(prompt_path, "w", encoding="utf-8") as pf:
                json.dump(existing, pf, ensure_ascii=False, indent=2)
            success_count += 1
        except Exception as e:
            print(f"[STYLE_LORA] 부정 프롬프트 저장 실패: {prompt_path} - {e}")
            traceback.print_exc()
            fail_count += 1
            failed.append({"filename": fn, "error": str(e)})

    print(f"[STYLE_LORA] 부정 프롬프트 일괄 적용: project={project_id} "
          f"ok={success_count} fail={fail_count}")
    return {"success": True, "total": len(filenames),
            "success_count": success_count, "fail_count": fail_count, "failed": failed}


# ─── 설정 관리 (프로젝트별 학습 설정, ANIMA/SDXL) ───────────────

def _merged_profile_settings(stored: dict) -> dict:
    merged = dict(DEFAULT_PROFILE_SETTINGS)
    if isinstance(stored, dict):
        for k, v in stored.items():
            merged[k] = v
    # 타입 정규화
    for int_key in ("step_per_image", "save_per_step", "gen_w", "gen_h", "resolution", "save_after", "dim", "alpha"):
        try:
            merged[int_key] = int(merged.get(int_key))
        except (TypeError, ValueError):
            pass
    try:
        merged["il_rate"] = float(merged.get("il_rate"))
    except (TypeError, ValueError):
        pass
    merged["upscale"] = bool(merged.get("upscale"))
    return merged


def get_project_settings(project_id: str) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    cfg = project.get("training_config", {}) or {}
    return {
        "success": True,
        "data": {
            "anima": _merged_profile_settings(cfg.get("anima", {})),
            "sdxl": _merged_profile_settings(cfg.get("sdxl", {})),
            "selected_profile": cfg.get("selected_profile", "both"),
        },
    }


def save_project_settings(project_id: str, settings: dict) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    cfg = {"anima": {}, "sdxl": {}, "selected_profile": "both"}
    if isinstance(settings, dict):
        for profile in ("anima", "sdxl"):
            cfg[profile] = settings.get(profile, {}) or {}
        sp = settings.get("selected_profile")
        if sp in ("anima", "sdxl", "both"):
            cfg["selected_profile"] = sp
    project["training_config"] = cfg
    _save_data(data)
    print(f"[STYLE_LORA] 프로젝트 설정 저장: {project_id}")
    return {"success": True}


# ─── 세션 관리 ─────────────────────────────────────────────────

def add_session(project_id: str, session_id: str, profile: str) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    project.setdefault("sessions", {})[session_id] = {
        "profile": profile,
        "representative": None,
    }
    _save_data(data)
    print(f"[STYLE_LORA] 세션 추가: {project_id}/{session_id} (profile={profile})")
    return {"success": True}


# ─── 학습된 LoRA 관리 (ANIMA/SDXL profile별) ───────────────────
# 학습 결과 파일 구조: {style_lora_load_path}/{profile}/{project}/{session}/
#   안에 .json(lora_file, previews, avr_loss) · .safetensors · .previews · .toml
# 세션 폴더명은 ComfyUI가 생성하므로 파일시스템을 직접 스캔한다.

def _style_trained_dir(style_lora_load_path: str, profile: str, project_id: str) -> str:
    return os.path.join(style_lora_load_path, profile, _safe_dirname(project_id))


def _get_trained_manage(data: dict, project_id: str, profile: str) -> dict:
    """project의 trained_manage[profile] 반환(없으면 빈 구조). profile은 'anima'/'sdxl'."""
    project = data.get("projects", {}).get(_safe_dirname(project_id))
    if not project:
        return {"representatives": {}, "priority": []}
    tm = project.setdefault("trained_manage", {}).setdefault(profile, {})
    tm.setdefault("representatives", {})
    tm.setdefault("priority", [])
    return tm


def _save_with_backup(data: dict):
    _backup_file(STYLE_LORA_MANAGE_FILE)
    _save_data(data)


def _resolve_style_session_priority(tm: dict, entry_dir: str, reps: dict) -> list:
    """저장된 priority 반환. 비어있으면 representative 있는 세션들로 자동 채움(저장 안 함)."""
    priority = list(tm.get("priority", []) or [])
    if priority:
        return priority
    if not os.path.isdir(entry_dir):
        return []
    auto = []
    for name in sorted(os.listdir(entry_dir), reverse=True):
        path = os.path.join(entry_dir, name)
        if not os.path.isdir(path):
            continue
        if reps.get(name):
            auto.append(name)
    return auto


def list_style_trained_sessions(style_lora_load_path: str, profile: str, project_id: str) -> list:
    if not style_lora_load_path:
        print("[STYLE_LORA_TRAINED] style_lora_load_path 미설정")
        return []
    if profile not in ("anima", "sdxl"):
        print(f"[STYLE_LORA_TRAINED] 잘못된 profile: {profile}")
        return []
    entry_dir = _style_trained_dir(style_lora_load_path, profile, project_id)
    if not os.path.isdir(entry_dir):
        return []

    data = _load_data()
    tm = _get_trained_manage(data, project_id, profile)
    reps = tm.get("representatives", {})
    session_priority = _resolve_style_session_priority(tm, entry_dir, reps)

    sessions = []
    for name in sorted(os.listdir(entry_dir), reverse=True):
        path = os.path.join(entry_dir, name)
        if not os.path.isdir(path):
            continue
        step_count = sum(1 for f in os.listdir(path) if f.endswith('.safetensors'))
        has_final = any('-step' not in f for f in os.listdir(path) if f.endswith('.safetensors'))
        rep = reps.get(name) or {}
        preview_url = rep.get("preview", "") if isinstance(rep, dict) else ""
        try:
            priority_rank = session_priority.index(name) + 1 if name in session_priority else 0
        except ValueError:
            priority_rank = 0
        sessions.append({
            "name": name,
            "step_count": step_count,
            "has_final": has_final,
            "representative": rep,
            "preview_url": preview_url,
            "priority_rank": priority_rank,
        })
    return sessions


def list_style_trained_steps(style_lora_load_path: str, profile: str, project_id: str, session: str) -> list:
    if not style_lora_load_path:
        print("[STYLE_LORA_TRAINED] style_lora_load_path 미설정")
        return []
    session_dir = os.path.join(_style_trained_dir(style_lora_load_path, profile, project_id), session)
    if not os.path.isdir(session_dir):
        print(f"[STYLE_LORA_TRAINED] 세션 폴더 없음: {session_dir}")
        return []
    steps = []
    for fname in sorted(os.listdir(session_dir)):
        if not fname.endswith('.json'):
            continue
        json_path = os.path.join(session_dir, fname)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                jdata = json.load(f)
        except Exception as e:
            print(f"[STYLE_LORA_TRAINED] JSON 읽기 실패: {json_path} - {e}")
            continue
        step_name = os.path.splitext(fname)[0]
        steps.append({
            "name": step_name,
            "safetensors": jdata.get('lora_file', step_name + '.safetensors'),
            "previews": jdata.get('previews', []),
            "json_file": fname,
            "avr_loss": jdata.get('avr_loss', None),
        })
    return steps


def read_style_toml_file(style_lora_load_path: str, profile: str, project_id: str, session: str, step_name: str) -> dict:
    if not style_lora_load_path:
        return {"success": False, "error": "style_lora_load_path 미설정"}
    session_dir = os.path.join(_style_trained_dir(style_lora_load_path, profile, project_id), session)
    toml_path = os.path.join(session_dir, step_name + ".toml")
    if not os.path.isfile(toml_path):
        print(f"[STYLE_LORA_TRAINED] TOML 파일 없음: {toml_path}")
        return {"success": False, "error": "TOML 파일이 없습니다"}
    try:
        with open(toml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"success": True, "content": content, "filename": step_name + ".toml"}
    except Exception as e:
        print(f"[STYLE_LORA_TRAINED] TOML 읽기 실패: {toml_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_style_trained_preview_path(style_lora_load_path: str, profile: str, project_id: str, session: str, filename: str) -> str:
    if not style_lora_load_path:
        return ""
    path = os.path.join(_style_trained_dir(style_lora_load_path, profile, project_id), session, filename)
    if os.path.isfile(path):
        return path
    return ""


def delete_style_trained_step(style_lora_load_path: str, profile: str, project_id: str, session: str, step_name: str) -> dict:
    if not style_lora_load_path:
        return {"success": False, "error": "style_lora_load_path 미설정"}
    session_dir = os.path.join(_style_trained_dir(style_lora_load_path, profile, project_id), session)
    if not os.path.isdir(session_dir):
        return {"success": False, "error": "세션 폴더 없음"}
    json_path = os.path.join(session_dir, step_name + ".json")
    if not os.path.isfile(json_path):
        return {"success": False, "error": "JSON 파일 없음"}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            jdata = json.load(f)
    except Exception as e:
        return {"success": False, "error": f"JSON 읽기 실패: {e}"}
    deleted = []
    errors = []
    st_name = jdata.get('lora_file', step_name + '.safetensors')
    fp = os.path.join(session_dir, st_name)
    if os.path.isfile(fp):
        try:
            os.remove(fp); deleted.append(st_name)
        except Exception as e:
            errors.append(f"{st_name}: {e}")
    for p in jdata.get('previews', []):
        fp = os.path.join(session_dir, p)
        if os.path.isfile(fp):
            try:
                os.remove(fp); deleted.append(p)
            except Exception as e:
                errors.append(f"{p}: {e}")
    toml_path = os.path.join(session_dir, step_name + ".toml")
    if os.path.isfile(toml_path):
        try:
            os.remove(toml_path); deleted.append(step_name + ".toml")
        except Exception as e:
            errors.append(f"{step_name}.toml: {e}")
    try:
        os.remove(json_path); deleted.append(step_name + ".json")
    except Exception as e:
        errors.append(f"{step_name}.json: {e}")

    # 삭제한 step이 이 세션의 대표였다면 대표 해제
    try:
        data = _load_data()
        tm = _get_trained_manage(data, project_id, profile)
        reps = tm.get("representatives", {})
        rep = reps.get(session)
        if isinstance(rep, dict) and rep.get("safetensors") == st_name:
            del reps[session]
            _save_with_backup(data)
            print(f"[STYLE_LORA_TRAINED] 삭제된 대표 step 해제: {session}/{st_name}")
    except Exception as e:
        print(f"[STYLE_LORA_TRAINED] 대표 해제 실패: {e}")
        traceback.print_exc()

    if errors:
        print(f"[STYLE_LORA_TRAINED] 삭제 중 일부 실패: {errors}")
    return {"success": True, "deleted": deleted, "errors": errors}


def delete_style_trained_session(style_lora_load_path: str, profile: str, project_id: str, session: str) -> dict:
    if not style_lora_load_path:
        return {"success": False, "error": "style_lora_load_path 미설정"}
    session_dir = os.path.join(_style_trained_dir(style_lora_load_path, profile, project_id), session)
    if not os.path.isdir(session_dir):
        return {"success": False, "error": "세션 폴더 없음"}
    try:
        file_count = sum(1 for _ in os.listdir(session_dir))
        shutil.rmtree(session_dir)
        # trained_manage 에서 해당 세션 제거
        data = _load_data()
        tm = _get_trained_manage(data, project_id, profile)
        changed = False
        if session in tm.get("representatives", {}):
            del tm["representatives"][session]
            changed = True
        if session in (tm.get("priority") or []):
            tm["priority"] = [s for s in tm["priority"] if s != session]
            changed = True
        if changed:
            _save_with_backup(data)
            print(f"[STYLE_LORA_TRAINED] 세션 관리정보에서 제거: {session}")
        print(f"[STYLE_LORA_TRAINED] 세션 폴더 삭제 완료: {session_dir} ({file_count}개 파일)")
        return {"success": True, "deleted_session": session, "file_count": file_count}
    except Exception as e:
        print(f"[STYLE_LORA_TRAINED] 세션 폴더 삭제 실패: {session_dir} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def update_style_session_representative(project_id: str, profile: str, session: str, representative: dict) -> dict:
    if not project_id or profile not in ("anima", "sdxl") or not session:
        return {"success": False, "error": "project/profile/session 누락"}
    data = _load_data()
    project = data.get("projects", {}).get(_safe_dirname(project_id))
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    tm = _get_trained_manage(data, project_id, profile)
    tm.setdefault("representatives", {})[session] = representative
    # 대표 설정 시 priority에 없으면 추가(1순위 후보가 되도록 맨 앞)
    if session not in (tm.get("priority") or []):
        tm.setdefault("priority", []).insert(0, session)
    _save_with_backup(data)
    print(f"[STYLE_LORA] 세션 대표 설정: {project_id}/{profile}/{session}")
    return {"success": True}


def update_style_session_priority(project_id: str, profile: str, sessions_list: list) -> dict:
    if not project_id or profile not in ("anima", "sdxl"):
        return {"success": False, "error": "project/profile 누락"}
    if not isinstance(sessions_list, list):
        return {"success": False, "error": "sessions는 배열이어야 합니다"}
    data = _load_data()
    project = data.get("projects", {}).get(_safe_dirname(project_id))
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    tm = _get_trained_manage(data, project_id, profile)
    tm["priority"] = sessions_list
    _save_with_backup(data)
    print(f"[STYLE_LORA] session_priority 업데이트: {project_id}/{profile} -> {sessions_list}")
    return {"success": True}


def cleanup_style_non_representative(style_lora_load_path: str, profile: str, project_id: str) -> dict:
    """해당 profile의 대표 LoRA 외 모든 LoRA 정리.
    - 대표가 설정된 세션: 대표 step만 남기고 나머지 step 삭제
    - 대표가 없는 세션: 세션 전체 삭제
    """
    if not style_lora_load_path:
        print("[STYLE_LORA_CLEANUP] style_lora_load_path 미설정")
        return {"success": False, "error": "style_lora_load_path 미설정"}
    if profile not in ("anima", "sdxl"):
        return {"success": False, "error": f"잘못된 profile: {profile}"}

    entry_dir = _style_trained_dir(style_lora_load_path, profile, project_id)
    if not os.path.isdir(entry_dir):
        print(f"[STYLE_LORA_CLEANUP] profile LoRA 폴더 없음: {entry_dir}")
        return {"success": False, "error": "해당 profile의 LoRA 폴더가 없습니다"}

    data = _load_data()
    tm = _get_trained_manage(data, project_id, profile)
    reps = tm.get("representatives", {})

    deleted_sessions = []
    deleted_steps = []
    errors = []
    reps_changed = False

    for session_name in sorted(os.listdir(entry_dir)):
        session_dir = os.path.join(entry_dir, session_name)
        if not os.path.isdir(session_dir):
            continue

        rep = reps.get(session_name) or {}
        rep_safetensors = rep.get("safetensors", "") if isinstance(rep, dict) else ""

        # 대표가 없는 세션: 전체 삭제
        if not rep_safetensors:
            try:
                file_count = sum(1 for _ in os.listdir(session_dir))
                shutil.rmtree(session_dir)
                deleted_sessions.append(session_name)
                if session_name in reps:
                    del reps[session_name]
                    reps_changed = True
                print(f"[STYLE_LORA_CLEANUP] 대표 없는 세션 삭제: {session_name} ({file_count}개 파일)")
            except Exception as e:
                errors.append(f"세션 {session_name} 삭제 실패: {e}")
                print(f"[STYLE_LORA_CLEANUP] 세션 삭제 실패: {session_dir} - {e}")
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
                    jdata = json.load(f)
            except Exception as e:
                errors.append(f"JSON 읽기 실패 {fname}: {e}")
                continue

            st_name = jdata.get('lora_file', step_name + '.safetensors')
            if st_name == rep_safetensors:
                continue

            # 비대표 step 삭제
            fp = os.path.join(session_dir, st_name)
            if os.path.isfile(fp):
                try:
                    os.remove(fp); deleted_steps.append(f"{session_name}/{st_name}")
                except Exception as e:
                    errors.append(f"{st_name}: {e}")
            for p in jdata.get('previews', []):
                fp = os.path.join(session_dir, p)
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                    except Exception as e:
                        errors.append(f"{p}: {e}")
            toml_path = os.path.join(session_dir, step_name + ".toml")
            if os.path.isfile(toml_path):
                try:
                    os.remove(toml_path)
                except Exception as e:
                    errors.append(f"{step_name}.toml: {e}")
            try:
                os.remove(json_path); deleted_steps.append(f"{session_name}/{step_name}")
            except Exception as e:
                errors.append(f"{fname}: {e}")
            print(f"[STYLE_LORA_CLEANUP] 비대표 step 삭제: {session_name}/{step_name}")

    if reps_changed:
        _save_with_backup(data)

    result = {
        "success": True,
        "deleted_sessions": deleted_sessions,
        "deleted_steps": deleted_steps,
        "errors": errors,
    }
    print(f"[STYLE_LORA_CLEANUP] 정리 완료: 세션 {len(deleted_sessions)}개 삭제, step {len(deleted_steps)}개 삭제")
    return result


# ─── 피커용 ────────────────────────────────────────────────────

def list_style_lora_for_picker(style_lora_load_path: str = "") -> list:
    """Style LoRA 피커용 목록. 학습 결과 파일시스템 스캔(instance 패턴).
    저장 경로 키: {project_id}."""
    data = _load_data()
    result = []
    for project_id, pdata in data.get("projects", {}).items():
        profiles = {}
        storage_key = _safe_dirname(project_id)
        for profile in ("anima", "sdxl"):
            if not style_lora_load_path:
                continue
            profile_dir = os.path.join(style_lora_load_path, profile, storage_key)
            if not os.path.isdir(profile_dir):
                continue
            all_sessions = sorted(
                [d for d in os.listdir(profile_dir) if os.path.isdir(os.path.join(profile_dir, d))],
                reverse=True,
            )
            # 대표 우선: priority 순 → 나머지 세션(최신순) 순회.
            tm = _get_trained_manage(data, project_id, profile)
            reps = tm.get("representatives", {})
            priority = [s for s in (tm.get("priority") or []) if s in all_sessions]
            ordered = priority + [s for s in all_sessions if s not in priority]

            picked = False
            for session_name in ordered:
                session_dir = os.path.join(profile_dir, session_name)
                # (1) 이 세션에 대표가 있으면 대표 safetensors 우선
                rep = reps.get(session_name)
                if isinstance(rep, dict) and rep.get("safetensors"):
                    rep_safe = rep.get("safetensors")
                    if os.path.isfile(os.path.join(session_dir, rep_safe)):
                        rel_path = os.path.join(profile, storage_key, session_name, rep_safe)
                        profiles[profile] = {
                            "lora_path": rel_path,
                            "preview_url": rep.get("preview", ""),
                            "session": session_name,
                        }
                        picked = True
                        break
                # (2) 대표 없으면 이 세션의 첫 step 사용(기존 동작)
                json_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
                if not json_files:
                    continue
                json_path = os.path.join(session_dir, json_files[0])
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        jdata = json.load(f)
                    safetensors = jdata.get('lora_file', '')
                    previews = jdata.get('previews', [])
                    if safetensors and os.path.isfile(os.path.join(session_dir, safetensors)):
                        rel_path = os.path.join(profile, storage_key, session_name, safetensors)
                        preview = previews[0] if previews else ""
                        profiles[profile] = {
                            "lora_path": rel_path,
                            "preview_url": preview,
                            "session": session_name,
                        }
                        picked = True
                        break
                except Exception as e:
                    print(f"[STYLE_LORA_PICKER] JSON 읽기 실패: {json_path} - {e}")
                    continue
            if picked:
                continue
        if profiles:
            images = pdata.get("images", [])
            result.append({
                "project_id": project_id,
                "id": project_id,
                "name": pdata.get("name", project_id),
                "trigger": pdata.get("trigger", ""),
                "first_image": images[0] if images else None,
                "profiles": profiles,
            })
    return result


# ─── LLM 정제 프롬프트 템플릿 조회/저장 (style 전용, instance_lora_mode 의 로더 재사용) ──

async def handle_get_style_lora_prompt(request):
    """GET /api/style_lora/auto_lora_prompt - 스타일 LoRA 정제 프롬프트(builtin/custom/use_custom) 조회."""
    try:
        from modes.instance_lora_mode import (
            _load_auto_lora_prompt_builtin, _load_auto_lora_prompt_custom,
        )
        builtin = _load_auto_lora_prompt_builtin(False, template_set="style")
        custom, use_custom = _load_auto_lora_prompt_custom(False, template_set="style")
        return web.json_response({
            "success": True,
            "data": {
                "builtin": builtin,
                "custom": custom,
                "use_custom": use_custom,
                "template_set": "style",
            },
        })
    except Exception as e:
        print(f"[STYLE_LORA] auto_lora_prompt 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_set_style_lora_prompt(request):
    """POST /api/style_lora/auto_lora_prompt - 스타일 LoRA 정제 커스텀 프롬프트 저장."""
    try:
        from modes.instance_lora_mode import _save_auto_lora_prompt_custom
        body = await request.json()
        custom = body.get("custom", "") or ""
        use_custom = bool(body.get("use_custom", False))
        _save_auto_lora_prompt_custom(custom, use_custom, False, template_set="style")
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[STYLE_LORA] auto_lora_prompt 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_style_lora_auto_refine_enqueue(request):
    """POST /api/style_lora/auto_refine_enqueue - 스타일 프로젝트 단일 이미지 LLM 정제 큐 적재.
    body: { project, filename } (또는 filenames 배열 → 각각 별도 큐 아이템)."""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        if not project:
            return web.json_response({"success": False, "error": "project 필드가 필요합니다."}, status=400)
        filenames = body.get("filenames")
        if filenames:
            if not isinstance(filenames, list) or not filenames:
                return web.json_response({"success": False, "error": "filenames 가 비어 있습니다."}, status=400)
        else:
            filename = (body.get("filename") or "").strip()
            if not filename:
                return web.json_response({"success": False, "error": "filename 필드가 필요합니다."}, status=400)
            filenames = [filename]

        try:
            import server as _server
            qm = _server.queue_manager
        except Exception as e:
            print(f"[STYLE_LORA] queue_manager 접근 실패: {e}")
            traceback.print_exc()
            return web.json_response({"success": False, "error": f"큐 매니저 접근 실패: {e}"})

        items_spec = []
        for fn in filenames:
            items_spec.append({
                "type": "instance_lora_prompt_refine",
                "label": f"스타일 LoRA 정제: {project}/{fn}",
                "batch_label": f"스타일 LoRA 정제: {project} ({len(filenames)}장)",
                "params": {
                    "source_type": "style",
                    "project": project,
                    "filename": fn,
                },
            })
        created = await qm.add_items_batch(items_spec, priority=10)
        batch_id = created[0].batch_id if created else None
        print(f"[STYLE_LORA] auto_refine 배치 큐 추가: project={project} count={len(created)} batch_id={batch_id}")
        return web.json_response({"success": True, "data": {"ids": [i.id for i in created], "count": len(created), "batch_id": batch_id}})
    except Exception as e:
        print(f"[STYLE_LORA] auto_refine_enqueue 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_style_lora_test_auto_refine_enqueue(request):
    """POST /api/style_lora/test_auto_refine_enqueue - 스타일 프로젝트 테스트 이미지 LLM 정제 큐 적재.
    body: { project, filename } (또는 filenames 배열 → 각각 별도 큐 아이템).
    학습 이미지 정제(handle_style_lora_auto_refine_enqueue)와 동일한 비전 LLM 프롬프트(template_set="style")를
    사용하되, 저장은 테스트 전용 프롬프트 파일({base}_test_prompt.json)에 한다(source_type="style_test")."""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        if not project:
            return web.json_response({"success": False, "error": "project 필드가 필요합니다."}, status=400)
        filenames = body.get("filenames")
        if filenames:
            if not isinstance(filenames, list) or not filenames:
                return web.json_response({"success": False, "error": "filenames 가 비어 있습니다."}, status=400)
        else:
            filename = (body.get("filename") or "").strip()
            if not filename:
                return web.json_response({"success": False, "error": "filename 필드가 필요합니다."}, status=400)
            filenames = [filename]

        try:
            import server as _server
            qm = _server.queue_manager
        except Exception as e:
            print(f"[STYLE_LORA] queue_manager 접근 실패: {e}")
            traceback.print_exc()
            return web.json_response({"success": False, "error": f"큐 매니저 접근 실패: {e}"})

        items_spec = []
        for fn in filenames:
            items_spec.append({
                "type": "instance_lora_prompt_refine",
                "label": f"스타일 LoRA 테스트 정제: {project}/{fn}",
                "batch_label": f"스타일 LoRA 테스트 정제: {project} ({len(filenames)}장)",
                "params": {
                    "source_type": "style_test",
                    "project": project,
                    "filename": fn,
                },
            })
        created = await qm.add_items_batch(items_spec, priority=10)
        batch_id = created[0].batch_id if created else None
        print(f"[STYLE_LORA] test_auto_refine 배치 큐 추가: project={project} count={len(created)} batch_id={batch_id}")
        return web.json_response({"success": True, "data": {"ids": [i.id for i in created], "count": len(created), "batch_id": batch_id}})
    except Exception as e:
        print(f"[STYLE_LORA] test_auto_refine_enqueue 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})
