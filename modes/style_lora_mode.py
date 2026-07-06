"""
Style LoRA(그림체 로라) 매니징 모듈
- 2단계 계층: 그룹 > 프로젝트(=그림체 로라 1개)
- 프로젝트가 학습 이미지 풀 + 학습 세션 보유
- 인스턴스 로라(instance_lora_mode)의 함수형 API 구조를 미러.
- 태깅/정제/학습은 모두 수동 버튼 트리거 (자동 E2E 체인 없음).
- 이미지는 프로젝트 폴더에 새로 복사된다(원본 참조 X).

데이터 파일: asset_data/style_lora_manage.json
이미지 복사본: style_lora_data/{group_id}/{project_id}/{filename}
캡션 파일: style_lora_data/{group_id}/{project_id}/{base}_prompt.json
"""

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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

DEFAULT_SETTINGS = {"anima": {}, "sdxl": {}}


# ─── 유틸 ──────────────────────────────────────────────────────

def _safe_dirname(name: str) -> str:
    return "".join(c for c in str(name) if c.isalnum() or c in (' ', '_', '-', '.')).strip() or "unnamed"


def _group_dir(group_id: str) -> str:
    return os.path.join(STYLE_LORA_DIR, _safe_dirname(group_id))


def _project_dir(group_id: str, project_id: str) -> str:
    return os.path.join(_group_dir(group_id), _safe_dirname(project_id))


def _gen_id(name: str) -> str:
    base = _safe_dirname(name)
    short_hash = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:6]
    return f"{base}-{short_hash}"


# ─── JSON 로드/세이브 ─────────────────────────────────────────

def _load_data() -> dict:
    if not os.path.isfile(STYLE_LORA_MANAGE_FILE):
        return {"groups": {}, "settings": dict(DEFAULT_SETTINGS)}
    try:
        with open(STYLE_LORA_MANAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[STYLE_LORA] JSON 로드 실패: {e}")
        traceback.print_exc()
        return {"groups": {}, "settings": dict(DEFAULT_SETTINGS)}


def _save_data(data: dict):
    try:
        with open(STYLE_LORA_MANAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[STYLE_LORA] JSON 세이브 실패: {e}")
        traceback.print_exc()


# ─── 그룹 관리 ─────────────────────────────────────────────────

def list_groups() -> list:
    data = _load_data()
    result = []
    for group_id, gdata in data.get("groups", {}).items():
        projects = gdata.get("projects", {})
        result.append({
            "id": group_id,
            "name": gdata.get("name", group_id),
            "project_count": len(projects),
        })
    return result


def create_group(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        return {"success": False, "error": "그룹 이름이 필요합니다"}
    data = _load_data()
    group_id = _gen_id(name)
    groups = data.setdefault("groups", {})
    if group_id in groups:
        return {"success": False, "error": "이미 존재하는 그룹입니다 (다시 시도하세요)"}
    groups[group_id] = {"name": name, "projects": {}}
    _save_data(data)
    os.makedirs(_group_dir(group_id), exist_ok=True)
    print(f"[STYLE_LORA] 그룹 생성: {group_id} (name={name})")
    return {"success": True, "id": group_id}


def delete_group(group_id: str, style_lora_load_path: str = "") -> dict:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    group = data.get("groups", {}).get(group_id)
    if not group:
        return {"success": False, "error": "존재하지 않는 그룹입니다"}

    # 하위 프로젝트 학습 결과물 정리
    for project_id in list(group.get("projects", {}).keys()):
        delete_project(group_id, project_id, style_lora_load_path=style_lora_load_path, _data=data)

    data.setdefault("groups", {}).pop(group_id, None)
    _save_data(data)

    gpath = _group_dir(group_id)
    if os.path.isdir(gpath):
        try:
            shutil.rmtree(gpath)
        except Exception as e:
            print(f"[STYLE_LORA] 그룹 폴더 삭제 실패: {gpath} - {e}")

    print(f"[STYLE_LORA] 그룹 삭제: {group_id}")
    return {"success": True}


# ─── 프로젝트 CRUD ─────────────────────────────────────────────

def list_projects(group_id: str) -> list:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    group = data.get("groups", {}).get(group_id, {})
    result = []
    for project_id, pdata in group.get("projects", {}).items():
        images = pdata.get("images", [])
        sessions = pdata.get("sessions", {})
        entry = {
            "group_id": group_id,
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
            prompt_result = get_image_prompt(group_id, project_id, images[0])
            if prompt_result.get("success") and prompt_result.get("data"):
                entry["prompt"] = prompt_result["data"]
        result.append(entry)
    return result


def create_project(group_id: str, name: str, trigger: str = "", description: str = "") -> dict:
    import datetime
    name = (name or "").strip()
    if not name:
        return {"success": False, "error": "프로젝트 이름이 필요합니다"}
    data = _load_data()
    group_id = _safe_dirname(group_id)
    group = data.setdefault("groups", {}).get(group_id)
    if not group:
        return {"success": False, "error": "존재하지 않는 그룹입니다"}

    project_id = _gen_id(name)
    projects = group.setdefault("projects", {})
    if project_id in projects:
        return {"success": False, "error": "이미 존재하는 프로젝트입니다 (다시 시도하세요)"}

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    projects[project_id] = {
        "name": name,
        "trigger": (trigger or "").strip() or name,
        "description": description or "",
        "images": [],
        "sessions": {},
        "usage_count": 0,
        "created_at": now,
    }
    _save_data(data)
    os.makedirs(_project_dir(group_id, project_id), exist_ok=True)
    print(f"[STYLE_LORA] 프로젝트 생성: {group_id}/{project_id} (name={name}, trigger={trigger})")
    return {"success": True, "id": project_id, "group_id": group_id}


def delete_project(group_id: str, project_id: str, style_lora_load_path: str = "", _data: dict = None) -> dict:
    own_data = _data is None
    data = _data if _data is not None else _load_data()
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    group = data.get("groups", {}).get(group_id)
    if not group or project_id not in group.get("projects", {}):
        if own_data:
            print(f"[STYLE_LORA] 삭제 대상 없음: {group_id}/{project_id}")
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    group["projects"].pop(project_id, None)
    if own_data:
        _save_data(data)

    # 학습 이미지 폴더 삭제
    ppath = _project_dir(group_id, project_id)
    if os.path.isdir(ppath):
        try:
            shutil.rmtree(ppath)
        except Exception as e:
            print(f"[STYLE_LORA] 프로젝트 폴더 삭제 실패: {ppath} - {e}")

    # 학습 결과물 삭제 (anima/sdxl). 저장 경로 키: {group}_{project}
    storage_key = f"{_safe_dirname(group_id)}_{project_id}"
    if style_lora_load_path:
        for profile in ("anima", "sdxl"):
            trained_dir = os.path.join(style_lora_load_path, profile, storage_key)
            if os.path.isdir(trained_dir):
                try:
                    shutil.rmtree(trained_dir)
                    print(f"[STYLE_LORA] 학습 결과 삭제: {trained_dir}")
                except Exception as e:
                    print(f"[STYLE_LORA] 학습 결과 삭제 실패: {trained_dir} - {e}")

    print(f"[STYLE_LORA] 프로젝트 삭제: {group_id}/{project_id}")
    return {"success": True}


def get_project_detail(group_id: str, project_id: str) -> dict:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    project = data.get("groups", {}).get(group_id, {}).get("projects", {}).get(project_id)
    if not project:
        print(f"[STYLE_LORA] 상세 조회 실패 - 없음: {group_id}/{project_id}")
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    return {
        "success": True,
        "data": {
            "group_id": group_id,
            "id": project_id,
            "name": project.get("name", project_id),
            "trigger": project.get("trigger", ""),
            "description": project.get("description", ""),
            "images": project.get("images", []),
            "sessions": project.get("sessions", {}),
            "usage_count": project.get("usage_count", 0),
            "created_at": project.get("created_at", ""),
        },
    }


def update_project(group_id: str, project_id: str, trigger: str = None, description: str = None) -> dict:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    project = data.get("groups", {}).get(group_id, {}).get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    if trigger is not None:
        project["trigger"] = trigger.strip()
    if description is not None:
        project["description"] = description
    _save_data(data)
    return {"success": True}


def increment_usage(group_id: str, project_id: str) -> dict:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    project = data.get("groups", {}).get(group_id, {}).get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    project["usage_count"] = project.get("usage_count", 0) + 1
    _save_data(data)
    return {"success": True, "usage_count": project["usage_count"]}


# ─── 이미지 관리 ──────────────────────────────────────────────

def add_image(group_id: str, project_id: str, src_path: str, filename: str) -> dict:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    project = data.get("groups", {}).get(group_id, {}).get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    dst_dir = _project_dir(group_id, project_id)
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

    print(f"[STYLE_LORA] 이미지 추가: {group_id}/{project_id}/{dst_name}")
    return {"success": True, "filename": dst_name}


def delete_image(group_id: str, project_id: str, filename: str) -> dict:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    project = data.get("groups", {}).get(group_id, {}).get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}

    images = project.get("images", [])
    if filename not in images:
        return {"success": False, "error": "이미지가 목록에 없습니다"}

    images.remove(filename)
    _save_data(data)

    pdir = _project_dir(group_id, project_id)
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

    print(f"[STYLE_LORA] 이미지 삭제: {group_id}/{project_id}/{filename}")
    return {"success": True}


def get_image_path(group_id: str, project_id: str, filename: str) -> str:
    return os.path.join(_project_dir(_safe_dirname(group_id), _safe_dirname(project_id)), filename)


def list_images(group_id: str, project_id: str) -> list:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    project = data.get("groups", {}).get(group_id, {}).get("projects", {}).get(project_id, {})
    return project.get("images", [])


def save_image_prompt(group_id: str, project_id: str, filename: str, prompt_data: dict) -> dict:
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(_project_dir(group_id, project_id), f"{base}_prompt.json")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)
        return {"success": True}
    except Exception as e:
        print(f"[STYLE_LORA] 프롬프트 저장 실패: {prompt_path} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_image_prompt(group_id: str, project_id: str, filename: str) -> dict:
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(_project_dir(group_id, project_id), f"{base}_prompt.json")
    if not os.path.isfile(prompt_path):
        return {"success": False, "error": "프롬프트 없음"}
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return {"success": True, "data": json.load(f)}
    except Exception as e:
        print(f"[STYLE_LORA] 프롬프트 로드 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


# ─── 설정 관리 (글로벌 학습 설정, instance 와 동일 스키마) ──────

def get_settings() -> dict:
    data = _load_data()
    return {"success": True, "data": data.get("settings", dict(DEFAULT_SETTINGS))}


def save_settings(settings: dict) -> dict:
    data = _load_data()
    data["settings"] = settings
    _save_data(data)
    print("[STYLE_LORA] 설정 저장 완료")
    return {"success": True}


# ─── 세션 관리 ─────────────────────────────────────────────────

def add_session(group_id: str, project_id: str, session_id: str, profile: str) -> dict:
    data = _load_data()
    group_id = _safe_dirname(group_id)
    project_id = _safe_dirname(project_id)
    project = data.get("groups", {}).get(group_id, {}).get("projects", {}).get(project_id)
    if not project:
        return {"success": False, "error": "존재하지 않는 프로젝트입니다"}
    project.setdefault("sessions", {})[session_id] = {
        "profile": profile,
        "representative": None,
    }
    _save_data(data)
    print(f"[STYLE_LORA] 세션 추가: {group_id}/{project_id}/{session_id} (profile={profile})")
    return {"success": True}


# ─── 피커용 (v2 통합 대비) ─────────────────────────────────────

def list_style_lora_for_picker(style_lora_load_path: str = "") -> list:
    """Style LoRA 피커용 목록. 학습 결과 파일시스템 스캔(instance 패턴).
    저장 경로 키: {group_id}_{project_id}."""
    data = _load_data()
    result = []
    for group_id, gdata in data.get("groups", {}).items():
        for project_id, pdata in gdata.get("projects", {}).items():
            profiles = {}
            storage_key = f"{_safe_dirname(group_id)}_{_safe_dirname(project_id)}"
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
                    "group_id": group_id,
                    "project_id": project_id,
                    "id": f"{group_id}/{project_id}",
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
    body: { group, project, filename } (또는 filenames 배열 → 각각 별도 큐 아이템)."""
    try:
        body = await request.json()
        group = (body.get("group") or "").strip()
        project = (body.get("project") or "").strip()
        if not group or not project:
            return web.json_response({"success": False, "error": "group, project 필드가 필요합니다."}, status=400)
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
            label = f"스타일 LoRA 정제: {group}/{project}/{fn}"
            item = await qm.add_item(
                item_type="instance_lora_prompt_refine",
                label=label,
                params={
                    "source_type": "style",
                    "group": group,
                    "project": project,
                    "filename": fn,
                },
                priority=10,
            )
            items.append(item)
        print(f"[STYLE_LORA] auto_refine 큐 추가: group={group} project={project} count={len(items)}")
        return web.json_response({"success": True, "data": {"ids": [i.id for i in items], "count": len(items)}})
    except Exception as e:
        print(f"[STYLE_LORA] auto_refine_enqueue 예외: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})
