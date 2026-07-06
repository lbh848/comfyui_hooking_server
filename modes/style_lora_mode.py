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
        },
    }


def save_project_settings(project_id: str, settings: dict) -> dict:
    data = _load_data()
    project_id = _safe_dirname(project_id)
    project = data.get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    cfg = {"anima": {}, "sdxl": {}}
    if isinstance(settings, dict):
        for profile in ("anima", "sdxl"):
            cfg[profile] = settings.get(profile, {}) or {}
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
            session_dirs = sorted(
                [d for d in os.listdir(profile_dir) if os.path.isdir(os.path.join(profile_dir, d))],
                reverse=True,
            )
            for session_name in session_dirs:
                session_dir = os.path.join(profile_dir, session_name)
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
                        break
                except Exception as e:
                    print(f"[STYLE_LORA_PICKER] JSON 읽기 실패: {json_path} - {e}")
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

        items = []
        for fn in filenames:
            label = f"스타일 LoRA 정제: {project}/{fn}"
            item = await qm.add_item(
                item_type="instance_lora_prompt_refine",
                label=label,
                params={
                    "source_type": "style",
                    "project": project,
                    "filename": fn,
                },
                priority=10,
            )
            items.append(item)
        print(f"[STYLE_LORA] auto_refine 큐 추가: project={project} count={len(items)}")
        return web.json_response({"success": True, "data": {"ids": [i.id for i in items], "count": len(items)}})
    except Exception as e:
        print(f"[STYLE_LORA] auto_refine_enqueue 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})
