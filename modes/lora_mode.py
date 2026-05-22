"""
LoRA 매니징 모듈
- asset/<character>/Lora/<entry>/ 폴더에서 LoRA 파일 관리
- asset/<character>/Lora/<entry>/학습용 이미지/ 폴더에서 학습용 이미지 관리
"""

import os
import json
import shutil
import traceback
from PIL import Image
from modes.asset_mode import ASSET_DIR, TAGS_FILE, AssetMode

LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".bin"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TRAINING_DIR_NAME = "학습용 이미지"
TEST_DIR_NAME = "테스트 이미지"
LORA_MANAGE_FILE = os.path.join(os.path.dirname(TAGS_FILE), "lora_manage.json")


def _safe_dirname(name: str) -> str:
    return AssetMode._safe_dirname(name)


def _lora_dir(character: str) -> str:
    """캐릭터의 Lora 폴더 경로 반환"""
    return os.path.join(ASSET_DIR, _safe_dirname(character), "Lora")


def _lora_entry_dir(character: str, entry_name: str) -> str:
    """캐릭터의 특정 LoRA 항목 폴더 경로 반환"""
    return os.path.join(_lora_dir(character), _safe_dirname(entry_name))


def _training_dir(character: str, entry: str = "") -> str:
    """LoRA 엔트리 내 학습용 이미지 폴더 경로 반환"""
    base = _lora_entry_dir(character, entry) if entry else _lora_dir(character)
    return os.path.join(base, TRAINING_DIR_NAME)


def _test_dir(character: str, entry: str = "") -> str:
    """LoRA 엔트리 내 테스트 이미지 폴더 경로 반환"""
    base = _lora_entry_dir(character, entry) if entry else _lora_dir(character)
    return os.path.join(base, TEST_DIR_NAME)


def list_characters() -> list:
    """tags.json에서 캐릭터 목록 반환"""
    if not os.path.isfile(TAGS_FILE):
        print("[LORA] tags.json 없음")
        return []
    try:
        with open(TAGS_FILE, "r", encoding="utf-8") as f:
            tags = json.load(f)
        return list(tags.get("characters", {}).keys())
    except Exception as e:
        print(f"[LORA] 캐릭터 목록 로드 실패: {e}")
        return []


def list_lora_files(character: str, entry: str = "") -> list:
    """LoRA 파일 목록 반환. entry 지정 시 해당 항목 폴더 내, 미지정 시 Lora 루트"""
    lora_path = _lora_entry_dir(character, entry) if entry else _lora_dir(character)
    if not os.path.isdir(lora_path):
        return []

    files = []
    for fname in sorted(os.listdir(lora_path)):
        fpath = os.path.join(lora_path, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in LORA_EXTENSIONS:
            continue
        try:
            fstat = os.stat(fpath)
            files.append({
                "filename": fname,
                "size": fstat.st_size,
                "modified": fstat.st_mtime,
            })
        except Exception as e:
            print(f"[LORA] 파일 정보 읽기 실패: {fpath} - {e}")
    return files


def list_trained_sessions(lora_load_path: str, character: str, entry: str) -> list:
    """학습된 LoRA 세션(타임스탬프 폴더) 목록 반환"""
    if not lora_load_path:
        print("[LORA_TRAINED] lora_load_path 미설정")
        return []
    # 구조: lora_load_path/<character>/Lora/<entry>/
    entry_dir = os.path.join(lora_load_path, _safe_dirname(character), "Lora", _safe_dirname(entry))
    if not os.path.isdir(entry_dir):
        print(f"[LORA_TRAINED] 엔트리 폴더 없음: {entry_dir}")
        return []

    # 세션별 대표 로라 조회
    manage_data = _load_lora_manage()
    manage_entry = _get_entry(manage_data, character, entry)
    session_reps = (manage_entry or {}).get("session_representatives", {})

    sessions = []
    for name in sorted(os.listdir(entry_dir), reverse=True):
        path = os.path.join(entry_dir, name)
        if not os.path.isdir(path):
            continue
        step_count = sum(1 for f in os.listdir(path) if f.endswith('.safetensors'))
        has_final = any('-step' not in f for f in os.listdir(path) if f.endswith('.safetensors'))

        # 세션 대표 이미지 정보
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


def list_trained_steps(lora_load_path: str, character: str, entry: str, session: str) -> list:
    """학습된 LoRA step 파일 목록 반환 (JSON 메타데이터 포함)"""
    if not lora_load_path:
        print("[LORA_TRAINED] lora_load_path 미설정")
        return []
    session_dir = os.path.join(lora_load_path, _safe_dirname(character), "Lora", _safe_dirname(entry), session)
    if not os.path.isdir(session_dir):
        print(f"[LORA_TRAINED] 세션 폴더 없음: {session_dir}")
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
            print(f"[LORA_TRAINED] JSON 읽기 실패: {json_path} - {e}")
            continue
        step_name = os.path.splitext(fname)[0]
        safetensors = data.get('lora_file', step_name + '.safetensors')
        previews = data.get('previews', [])
        steps.append({
            "name": step_name,
            "safetensors": safetensors,
            "previews": previews,
            "json_file": fname,
        })
    return steps


def delete_trained_step(lora_load_path: str, character: str, entry: str, session: str, step_name: str) -> dict:
    """학습된 LoRA step 삭제 (safetensors, json, toml, 프리뷰 이미지)"""
    if not lora_load_path:
        print("[LORA_TRAINED] lora_load_path 미설정")
        return {"success": False, "error": "lora_load_path 미설정"}
    session_dir = os.path.join(lora_load_path, _safe_dirname(character), "Lora", _safe_dirname(entry), session)
    if not os.path.isdir(session_dir):
        print(f"[LORA_TRAINED] 세션 폴더 없음: {session_dir}")
        return {"success": False, "error": "세션 폴더 없음"}
    json_path = os.path.join(session_dir, step_name + ".json")
    if not os.path.isfile(json_path):
        print(f"[LORA_TRAINED] JSON 파일 없음: {json_path}")
        return {"success": False, "error": "JSON 파일 없음"}
    # JSON에서 연관 파일 목록 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[LORA_TRAINED] JSON 읽기 실패: {json_path} - {e}")
        return {"success": False, "error": f"JSON 읽기 실패: {e}"}
    deleted = []
    errors = []
    # safetensors
    st_name = data.get('lora_file', step_name + '.safetensors')
    for fname in [st_name]:
        fp = os.path.join(session_dir, fname)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
                deleted.append(fname)
            except Exception as e:
                errors.append(f"{fname}: {e}")
    # previews
    for p in data.get('previews', []):
        fp = os.path.join(session_dir, p)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
                deleted.append(p)
            except Exception as e:
                errors.append(f"{p}: {e}")
    # toml
    toml_path = os.path.join(session_dir, step_name + ".toml")
    if os.path.isfile(toml_path):
        try:
            os.remove(toml_path)
            deleted.append(step_name + ".toml")
        except Exception as e:
            errors.append(f"{step_name}.toml: {e}")
    # json
    try:
        os.remove(json_path)
        deleted.append(step_name + ".json")
    except Exception as e:
        errors.append(f"{step_name}.json: {e}")
    if errors:
        print(f"[LORA_TRAINED] 삭제 중 일부 실패: {errors}")
    print(f"[LORA_TRAINED] 삭제 완료: {deleted}")
    return {"success": True, "deleted": deleted, "errors": errors}


def delete_trained_session(lora_load_path: str, character: str, entry: str, session: str) -> dict:
    """학습된 LoRA 세션 폴더 전체 삭제"""
    import shutil
    if not lora_load_path:
        print("[LORA_TRAINED] lora_load_path 미설정")
        return {"success": False, "error": "lora_load_path 미설정"}
    session_dir = os.path.join(lora_load_path, _safe_dirname(character), "Lora", _safe_dirname(entry), session)
    if not os.path.isdir(session_dir):
        print(f"[LORA_TRAINED] 세션 폴더 없음: {session_dir}")
        return {"success": False, "error": "세션 폴더 없음"}
    try:
        file_count = sum(1 for _ in os.listdir(session_dir))
        shutil.rmtree(session_dir)
        print(f"[LORA_TRAINED] 세션 폴더 삭제 완료: {session_dir} ({file_count}개 파일)")
        return {"success": True, "deleted_session": session, "file_count": file_count}
    except Exception as e:
        print(f"[LORA_TRAINED] 세션 폴더 삭제 실패: {session_dir} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_trained_preview_path(lora_load_path: str, character: str, entry: str, session: str, filename: str) -> str:
    """학습된 LoRA 프리뷰 이미지 경로 반환"""
    if not lora_load_path:
        return ""
    path = os.path.join(lora_load_path, _safe_dirname(character), "Lora", _safe_dirname(entry), session, filename)
    if os.path.isfile(path):
        return path
    return ""


def save_uploaded_file(character: str, filename: str, file_data: bytes, entry: str = "") -> dict:
    """업로드된 LoRA 파일을 저장. entry 지정 시 항목 폴더 내에 저장"""
    # 파일명 정제
    safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '_', '-', '.', '(', ')', '[', ']', '(')).strip()
    if not safe_name:
        return {"success": False, "error": "유효하지 않은 파일명"}

    lora_path = _lora_entry_dir(character, entry) if entry else _lora_dir(character)
    os.makedirs(lora_path, exist_ok=True)

    dest = os.path.join(lora_path, safe_name)
    if os.path.exists(dest):
        return {"success": False, "error": f"이미 존재하는 파일: {safe_name}"}

    try:
        with open(dest, "wb") as f:
            f.write(file_data)
        fstat = os.stat(dest)
        print(f"[LORA] 파일 저장 완료: {dest}")
        return {
            "success": True,
            "filename": safe_name,
            "size": fstat.st_size,
        }
    except Exception as e:
        print(f"[LORA] 파일 저장 실패: {dest} - {e}")
        return {"success": False, "error": str(e)}


def delete_lora_file(character: str, filename: str, entry: str = "") -> dict:
    """LoRA 파일 삭제. entry 지정 시 항목 폴더 내에서 삭제"""
    # 경로 탈출 방지
    if ".." in filename or os.path.sep in filename:
        print(f"[LORA] 잘못된 파일명: {filename}")
        return {"success": False, "error": "잘못된 파일명"}

    lora_path = _lora_entry_dir(character, entry) if entry else _lora_dir(character)
    fpath = os.path.join(lora_path, filename)

    if not os.path.isfile(fpath):
        print(f"[LORA] 파일 없음: {fpath}")
        return {"success": False, "error": "파일을 찾을 수 없습니다"}

    try:
        os.remove(fpath)
        print(f"[LORA] 파일 삭제 완료: {fpath}")
        return {"success": True}
    except Exception as e:
        print(f"[LORA] 파일 삭제 실패: {fpath} - {e}")
        return {"success": False, "error": str(e)}


# ─── 학습용 이미지 관리 ─────────────────────────────────────

def add_training_images(character: str, entry: str, sources: list) -> dict:
    """
    에셋 폴더에서 학습용 이미지로 복사
    sources: [{ "outfit": "...", "expression": "...", "filename": "..." }, ...]
    """
    t_dir = _training_dir(character, entry)
    os.makedirs(t_dir, exist_ok=True)

    added = []
    skipped = []

    for src in sources:
        outfit = src.get("outfit", "")
        expression = src.get("expression", "")
        filename = src.get("filename", "")
        src_char = src.get("character") or character
        src_char_dir = os.path.join(ASSET_DIR, _safe_dirname(src_char))

        if not outfit or not expression or not filename:
            print(f"[LORA] 학습 이미지 추가: 필수 값 누락 - {src}")
            skipped.append({"filename": filename, "reason": "필수 값 누락"})
            continue

        # 원본 경로
        src_path = os.path.join(src_char_dir, _safe_dirname(outfit), _safe_dirname(expression), filename)
        if not os.path.isfile(src_path):
            print(f"[LORA] 학습 이미지 원본 없음: {src_path}")
            skipped.append({"filename": filename, "reason": "원본 파일 없음"})
            continue

        # 대상 경로 (이름 충돌 방지)
        dest_name = filename
        dest_path = os.path.join(t_dir, dest_name)
        if os.path.exists(dest_path):
            # 이미 존재하면 타임스탬프 접두어 추가
            import time
            base, ext = os.path.splitext(filename)
            dest_name = f"{int(time.time())}_{base}{ext}"
            dest_path = os.path.join(t_dir, dest_name)

        try:
            shutil.copy2(src_path, dest_path)

            # 프롬프트 JSON도 복사
            base, ext = os.path.splitext(filename)
            prompt_src = os.path.join(src_char_dir, _safe_dirname(outfit), _safe_dirname(expression), f"{base}_prompt.json")
            prompt_dest = os.path.join(t_dir, f"{os.path.splitext(dest_name)[0]}_prompt.json")
            if os.path.isfile(prompt_src):
                # 복사하면서 positive에서 [FACE_ID_ACTIVATE] 위쪽만 추출
                with open(prompt_src, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                positive = pdata.get("positive", "")
                marker = "[FACE_ID_ACTIVATE]"
                if marker in positive:
                    pdata["positive"] = positive.split(marker)[0].strip()
                # 원본 프롬프트 저장
                pdata["original_positive"] = pdata["positive"]
                pdata["original_negative"] = pdata.get("negative", "")
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump(pdata, f, ensure_ascii=False, indent=2)
            else:
                # 프롬프트 파일이 없으면 빈 프롬프트 생성
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump({"positive": "", "negative": "", "original_positive": "", "original_negative": ""}, f, ensure_ascii=False, indent=2)

            added.append(dest_name)
            print(f"[LORA] 학습 이미지 추가: {dest_path}")
        except Exception as e:
            print(f"[LORA] 학습 이미지 복사 실패: {src_path} -> {dest_path} - {e}")
            traceback.print_exc()
            skipped.append({"filename": filename, "reason": str(e)})

    return {"success": True, "added": added, "skipped": skipped}


def list_training_images(character: str, entry: str = "") -> list:
    """학습용 이미지 목록 반환 (프롬프트 포함)"""
    t_dir = _training_dir(character, entry)
    if not os.path.isdir(t_dir):
        return []

    # 대표 이미지 로드
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
        # 프롬프트 로드
        base = os.path.splitext(fname)[0]
        prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
        positive = ""
        negative = ""
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                    positive = pdata.get("positive", "")
                    negative = pdata.get("negative", "")
                    # original이 없으면 현재값을 원본으로 사용 (기존 데이터 마이그레이션)
                    original_positive = pdata.get("original_positive", positive)
                    original_negative = pdata.get("original_negative", negative)
            except Exception as e:
                print(f"[LORA] 프롬프트 로드 실패: {prompt_path} - {e}")

        try:
            fstat = os.stat(fpath)
            # 이미지 해상도 읽기
            width, height = 0, 0
            try:
                with Image.open(fpath) as im:
                    width, height = im.size
            except Exception as e:
                print(f"[LORA] 이미지 해상도 읽기 실패: {fpath} - {e}")
            images.append({
                "filename": fname,
                "positive": positive,
                "negative": negative,
                "original_positive": original_positive,
                "original_negative": original_negative,
                "is_representative": fname == representative,
                "size": fstat.st_size,
                "modified": fstat.st_mtime,
                "width": width,
                "height": height,
            })
        except Exception as e:
            print(f"[LORA] 학습 이미지 정보 읽기 실패: {fpath} - {e}")

    return images


def get_training_image_path(character: str, entry: str, filename: str) -> str | None:
    """학습용 이미지 파일 경로 반환"""
    if ".." in filename or os.path.sep in filename:
        print(f"[LORA] 잘못된 파일명: {filename}")
        return None
    fpath = os.path.join(_training_dir(character, entry), filename)
    if os.path.isfile(fpath):
        return fpath
    print(f"[LORA] 학습 이미지 없음: {fpath}")
    return None


def delete_training_image(character: str, entry: str, filename: str) -> dict:
    """학습용 이미지 + 프롬프트 JSON 삭제"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}

    t_dir = _training_dir(character, entry)
    fpath = os.path.join(t_dir, filename)

    if not os.path.isfile(fpath):
        print(f"[LORA] 학습 이미지 없음: {fpath}")
        return {"success": False, "error": "파일을 찾을 수 없습니다"}

    try:
        os.remove(fpath)

        # 프롬프트 JSON도 삭제
        base = os.path.splitext(filename)[0]
        prompt_path = os.path.join(t_dir, f"{base}_prompt.json")
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)

        # 대표 이미지였으면 대표 설정도 해제
        rep_path = os.path.join(t_dir, "_representative.json")
        if os.path.isfile(rep_path):
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    if json.load(f).get("filename") == filename:
                        os.remove(rep_path)
            except Exception:
                pass

        print(f"[LORA] 학습 이미지 삭제 완료: {fpath}")
        return {"success": True}
    except Exception as e:
        print(f"[LORA] 학습 이미지 삭제 실패: {fpath} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def set_training_representative(character: str, entry: str, filename: str) -> dict:
    """학습용 이미지 대표 설정/해제"""
    t_dir = _training_dir(character, entry)
    rep_path = os.path.join(t_dir, "_representative.json")

    # 이미 대표면 해제
    if os.path.isfile(rep_path):
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                current = json.load(f).get("filename", "")
            if current == filename:
                os.remove(rep_path)
                print(f"[LORA] 대표 해제: {filename}")
                return {"success": True, "representative": ""}
        except Exception as e:
            print(f"[LORA] 대표 확인 실패: {e}")

    # 대표 설정
    try:
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump({"filename": filename}, f, ensure_ascii=False)
        print(f"[LORA] 대표 설정: {filename}")
        return {"success": True, "representative": filename}
    except Exception as e:
        print(f"[LORA] 대표 설정 실패: {e}")
        return {"success": False, "error": str(e)}


def save_training_prompt(character: str, entry: str, filename: str, positive: str, negative: str) -> dict:
    """학습용 이미지의 프롬프트 저장"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}

    t_dir = _training_dir(character, entry)
    base = os.path.splitext(filename)[0]
    prompt_path = os.path.join(t_dir, f"{base}_prompt.json")

    try:
        # 기존 데이터 유지하면서 프롬프트만 업데이트
        existing = {}
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        # original이 없으면 최초 저장 시 현재값을 원본으로 보존
        if "original_positive" not in existing:
            existing["original_positive"] = existing.get("positive", "")
        if "original_negative" not in existing:
            existing["original_negative"] = existing.get("negative", "")
        existing["positive"] = positive
        existing["negative"] = negative

        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[LORA] 프롬프트 저장 완료: {prompt_path}")
        return {"success": True}
    except Exception as e:
        print(f"[LORA] 프롬프트 저장 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


# ─── 테스트 이미지 관리 ─────────────────────────────────────

def add_test_images(character: str, entry: str, sources: list) -> dict:
    """
    에셋 폴더에서 테스트 이미지로 복사
    sources: [{ "outfit": "...", "expression": "...", "filename": "..." }, ...]
    """
    t_dir = _test_dir(character, entry)
    os.makedirs(t_dir, exist_ok=True)

    added = []
    skipped = []

    for src in sources:
        outfit = src.get("outfit", "")
        expression = src.get("expression", "")
        filename = src.get("filename", "")
        src_char = src.get("character") or character
        src_char_dir = os.path.join(ASSET_DIR, _safe_dirname(src_char))

        if not outfit or not expression or not filename:
            print(f"[LORA] 테스트 이미지 추가: 필수 값 누락 - {src}")
            skipped.append({"filename": filename, "reason": "필수 값 누락"})
            continue

        src_path = os.path.join(src_char_dir, _safe_dirname(outfit), _safe_dirname(expression), filename)
        if not os.path.isfile(src_path):
            print(f"[LORA] 테스트 이미지 원본 없음: {src_path}")
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
            prompt_src = os.path.join(src_char_dir, _safe_dirname(outfit), _safe_dirname(expression), f"{base}_prompt.json")
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
            print(f"[LORA] 테스트 이미지 추가: {dest_path}")
        except Exception as e:
            print(f"[LORA] 테스트 이미지 복사 실패: {src_path} -> {dest_path} - {e}")
            traceback.print_exc()
            skipped.append({"filename": filename, "reason": str(e)})

    return {"success": True, "added": added, "skipped": skipped}


def list_test_images(character: str, entry: str = "") -> list:
    """테스트 이미지 목록 반환 (프롬프트 포함)"""
    t_dir = _test_dir(character, entry)
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
                print(f"[LORA] 테스트 프롬프트 로드 실패: {prompt_path} - {e}")

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
            print(f"[LORA] 테스트 이미지 정보 읽기 실패: {fpath} - {e}")

    return images


def get_test_image_path(character: str, entry: str, filename: str) -> str | None:
    """테스트 이미지 파일 경로 반환"""
    if ".." in filename or os.path.sep in filename:
        print(f"[LORA] 잘못된 파일명: {filename}")
        return None
    fpath = os.path.join(_test_dir(character, entry), filename)
    if os.path.isfile(fpath):
        return fpath
    print(f"[LORA] 테스트 이미지 없음: {fpath}")
    return None


def delete_test_image(character: str, entry: str, filename: str) -> dict:
    """테스트 이미지 + 프롬프트 JSON 삭제"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}

    t_dir = _test_dir(character, entry)
    fpath = os.path.join(t_dir, filename)

    if not os.path.isfile(fpath):
        print(f"[LORA] 테스트 이미지 없음: {fpath}")
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

        print(f"[LORA] 테스트 이미지 삭제 완료: {fpath}")
        return {"success": True}
    except Exception as e:
        print(f"[LORA] 테스트 이미지 삭제 실패: {fpath} - {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def set_test_representative(character: str, entry: str, filename: str) -> dict:
    """테스트 이미지 대표 설정/해제"""
    t_dir = _test_dir(character, entry)
    rep_path = os.path.join(t_dir, "_representative.json")

    if os.path.isfile(rep_path):
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                current = json.load(f).get("filename", "")
            if current == filename:
                os.remove(rep_path)
                print(f"[LORA] 테스트 대표 해제: {filename}")
                return {"success": True, "representative": ""}
        except Exception as e:
            print(f"[LORA] 테스트 대표 확인 실패: {e}")

    try:
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump({"filename": filename}, f, ensure_ascii=False)
        print(f"[LORA] 테스트 대표 설정: {filename}")
        return {"success": True, "representative": filename}
    except Exception as e:
        print(f"[LORA] 테스트 대표 설정 실패: {e}")
        return {"success": False, "error": str(e)}


def save_test_prompt(character: str, entry: str, filename: str, positive: str, negative: str) -> dict:
    """테스트 이미지의 프롬프트 저장"""
    if ".." in filename or os.path.sep in filename:
        return {"success": False, "error": "잘못된 파일명"}

    t_dir = _test_dir(character, entry)
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
        print(f"[LORA] 테스트 프롬프트 저장 완료: {prompt_path}")
        return {"success": True}
    except Exception as e:
        print(f"[LORA] 테스트 프롬프트 저장 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


# ─── LoRA 항목 관리 (lora_manage.json) ─────────────────────────

def _load_lora_manage() -> dict:
    """lora_manage.json 로드 (필요시 마이그레이션)"""
    if os.path.isfile(LORA_MANAGE_FILE):
        try:
            with open(LORA_MANAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("block_tag_rules", [])
            data = _migrate_lora_manage(data)
            return data
        except Exception as e:
            print(f"[LORA_MANAGE] 로드 실패: {e}")
            traceback.print_exc()
            return {"loras": {}, "block_tag_rules": []}
    return {"loras": {}, "block_tag_rules": []}


def _migrate_lora_manage(data: dict) -> dict:
    """구형 평평한 구조를 캐릭터 중첩 구조로 마이그레이션"""
    loras = data.get("loras", {})
    if not loras:
        return data

    # 구형 감지: 값에 "character" 필드가 있으면 평평한 구조
    first_value = next(iter(loras.values()), {})
    if "character" not in first_value:
        return data  # 이미 새 구조

    print("[LORA_MANAGE] 마이그레이션: 평평한 구조 -> 캐릭터 중첩 구조")
    new_loras = {}
    for name, info in loras.items():
        char = info.get("character", "")
        if not char:
            print(f"[LORA_MANAGE] 마이그레이션 스킵: 캐릭터 없음 - {name}")
            continue
        # character 필드 제거하고 중첩
        entry_data = {k: v for k, v in info.items() if k != "character"}
        new_loras.setdefault(char, {})[name] = entry_data

    data["loras"] = new_loras
    _save_lora_manage(data)
    print(f"[LORA_MANAGE] 마이그레이션 완료: {len(loras)}개 항목")
    return data


def _save_lora_manage(data: dict):
    """lora_manage.json 저장"""
    try:
        with open(LORA_MANAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[LORA_MANAGE] 저장 완료")
    except Exception as e:
        print(f"[LORA_MANAGE] 저장 실패: {e}")
        traceback.print_exc()


def get_block_tag_rules() -> list:
    """전역 블록 태그 규칙 목록 반환"""
    data = _load_lora_manage()
    return data.get("block_tag_rules", [])


def save_block_tag_rules(rules: list) -> dict:
    """전역 블록 태그 규칙 저장"""
    if not isinstance(rules, list):
        return {"success": False, "error": "rules must be a list"}
    for r in rules:
        if not isinstance(r, str):
            return {"success": False, "error": "each rule must be a string"}
    data = _load_lora_manage()
    data["block_tag_rules"] = rules
    _save_lora_manage(data)
    return {"success": True, "rules": rules}


def _get_entry(data: dict, character: str, name: str) -> dict | None:
    """중첩 구조에서 엔트리 조회"""
    return data.get("loras", {}).get(character, {}).get(name)


def list_lora_entries(character: str = "") -> list:
    """LoRA 항목 목록 반환. character 지정 시 해당 캐릭터 것만."""
    data = _load_lora_manage()
    entries = []
    for char_name, char_entries in data.get("loras", {}).items():
        if character and char_name != character:
            continue
        for name, info in char_entries.items():
            entries.append({
                "name": name,
                "trigger": info.get("trigger", ""),
                "description": info.get("description", ""),
                "character": char_name,
                "representative": info.get("representative", ""),
                "training_config": info.get("training_config", {}),
            })
    return entries


def list_lora_for_picker(lora_load_path: str = "") -> list:
    """LoRA 피커용 목록 반환. 캐릭터별 그룹 + 대표이미지 + 사용 가능한 safetensors 경로 포함."""
    data = _load_lora_manage()
    result = []
    for char_name, char_entries in data.get("loras", {}).items():
        char_group = {"character": char_name, "entries": []}
        for name, info in char_entries.items():
            entry = {
                "name": name,
                "trigger": info.get("trigger", ""),
                "description": info.get("description", ""),
                "representative": info.get("representative", ""),
                "training_config": info.get("training_config", {}),
                "session_representatives": info.get("session_representatives", {}),
                "lora_files": [],
            }
            # 학습된 safetensors 파일 목록
            if lora_load_path:
                entry_dir = os.path.join(lora_load_path, _safe_dirname(char_name), "Lora", _safe_dirname(name))
                if os.path.isdir(entry_dir):
                    for session_name in sorted(os.listdir(entry_dir)):
                        session_dir = os.path.join(entry_dir, session_name)
                        if not os.path.isdir(session_dir):
                            continue
                        for fname in sorted(os.listdir(session_dir)):
                            if fname.endswith('.json'):
                                json_path = os.path.join(session_dir, fname)
                                try:
                                    with open(json_path, 'r', encoding='utf-8') as f:
                                        jdata = json.load(f)
                                    safetensors = jdata.get('lora_file', '')
                                    if safetensors:
                                        relative_path = os.path.join(
                                            _safe_dirname(char_name), "Lora",
                                            _safe_dirname(name), session_name, safetensors
                                        )
                                        previews = jdata.get('previews', [])
                                        entry["lora_files"].append({
                                            "path": relative_path,
                                            "session": session_name,
                                            "safetensors": safetensors,
                                            "previews": previews,
                                        })
                                except Exception as e:
                                    print(f"[LORA_PICKER] JSON 읽기 실패: {json_path} - {e}")
            # 수동 업로드된 safetensors도 포함
            manual_dir = os.path.join(ASSET_DIR, _safe_dirname(char_name), "Lora", _safe_dirname(name))
            if os.path.isdir(manual_dir):
                for fname in os.listdir(manual_dir):
                    if fname.endswith('.safetensors'):
                        full_path = os.path.join(manual_dir, fname)
                        entry["lora_files"].append({
                            "path": f"(수동) {_safe_dirname(char_name)}/Lora/{_safe_dirname(name)}/{fname}",
                            "session": "manual",
                            "safetensors": fname,
                            "previews": [],
                            "local_path": full_path,
                        })
            char_group["entries"].append(entry)
        if char_group["entries"]:
            result.append(char_group)
    return result


def add_lora_entry(name: str, character: str, trigger: str, description: str) -> dict:
    """새 LoRA 항목 추가"""
    if not name or not name.strip():
        print("[LORA_MANAGE] 이름 누락")
        return {"success": False, "error": "이름을 입력하세요"}
    if not trigger or not trigger.strip():
        print("[LORA_MANAGE] 트리거 키워드 누락")
        return {"success": False, "error": "트리거 키워드를 입력하세요"}
    if not character:
        print("[LORA_MANAGE] 캐릭터 미지정")
        return {"success": False, "error": "캐릭터를 선택하세요"}

    name = name.strip()
    data = _load_lora_manage()

    if _get_entry(data, character, name):
        print(f"[LORA_MANAGE] 이미 존재: {name} (캐릭터: {character})")
        return {"success": False, "error": "이미 존재하는 LoRA명"}

    # 캐릭터 Lora 폴더 + 항목 폴더 생성
    entry_path = _lora_entry_dir(character, name)
    os.makedirs(entry_path, exist_ok=True)

    data.setdefault("loras", {}).setdefault(character, {})[name] = {
        "trigger": trigger.strip(),
        "description": description.strip(),
    }
    _save_lora_manage(data)
    print(f"[LORA_MANAGE] LoRA 추가: {name} (캐릭터: {character}, 트리거: {trigger.strip()})")
    return {"success": True, "name": name}


def duplicate_lora_entry(
    source_character: str, source_entry: str,
    target_character: str, target_entry: str,
    trigger: str, description: str,
    training_config: dict = None
) -> dict:
    """LoRA 항목 복제 (학습/테스트 이미지 + 메타데이터)"""
    # 1) 입력 검증
    if not target_entry or not target_entry.strip():
        print("[LORA_MANAGE] 복제: 대상 항목명 누락")
        return {"success": False, "error": "대상 항목명을 입력하세요"}
    if not trigger or not trigger.strip():
        print("[LORA_MANAGE] 복제: 트리거 키워드 누락")
        return {"success": False, "error": "트리거 키워드를 입력하세요"}
    if not target_character:
        print("[LORA_MANAGE] 복제: 대상 캐릭터 누락")
        return {"success": False, "error": "대상 캐릭터를 선택하세요"}

    target_entry = target_entry.strip()
    data = _load_lora_manage()

    # 2) 소스 엔트리 존재 확인
    src = _get_entry(data, source_character, source_entry)
    if not src:
        print(f"[LORA_MANAGE] 복제: 소스 없음 {source_character}/{source_entry}")
        return {"success": False, "error": "원본 항목을 찾을 수 없습니다"}

    # 3) 타겟 중복 확인
    if _get_entry(data, target_character, target_entry):
        print(f"[LORA_MANAGE] 복제: 대상 이미 존재 {target_character}/{target_entry}")
        return {"success": False, "error": "대상 항목명이 이미 존재합니다"}

    # 4) 타겟 폴더 생성
    target_entry_path = _lora_entry_dir(target_character, target_entry)
    target_training = _training_dir(target_character, target_entry)
    target_test = _test_dir(target_character, target_entry)
    try:
        os.makedirs(target_training, exist_ok=True)
        os.makedirs(target_test, exist_ok=True)
    except Exception as e:
        print(f"[LORA_MANAGE] 복제: 폴더 생성 실패 - {e}")
        traceback.print_exc()
        return {"success": False, "error": f"폴더 생성 실패: {e}"}

    # 5) 학습 이미지 복사 (_representative.json 제외)
    src_training = _training_dir(source_character, source_entry)
    copied_training = 0
    if os.path.isdir(src_training):
        for fname in os.listdir(src_training):
            if fname == "_representative.json":
                continue
            src_file = os.path.join(src_training, fname)
            if os.path.isfile(src_file):
                try:
                    shutil.copy2(src_file, os.path.join(target_training, fname))
                    copied_training += 1
                except Exception as e:
                    print(f"[LORA_MANAGE] 복제: 학습 이미지 복사 실패 {fname} - {e}")

    # 6) 테스트 이미지 복사
    src_test = _test_dir(source_character, source_entry)
    copied_test = 0
    if os.path.isdir(src_test):
        for fname in os.listdir(src_test):
            if fname == "_representative.json":
                continue
            src_file = os.path.join(src_test, fname)
            if os.path.isfile(src_file):
                try:
                    shutil.copy2(src_file, os.path.join(target_test, fname))
                    copied_test += 1
                except Exception as e:
                    print(f"[LORA_MANAGE] 복제: 테스트 이미지 복사 실패 {fname} - {e}")

    # 7) JSON 메타데이터 생성
    entry_meta = {
        "trigger": trigger.strip(),
        "description": description.strip() if description else "",
    }
    if training_config and isinstance(training_config, dict):
        entry_meta["training_config"] = training_config

    data.setdefault("loras", {}).setdefault(target_character, {})[target_entry] = entry_meta
    _save_lora_manage(data)

    print(f"[LORA_MANAGE] 복제 완료: {source_character}/{source_entry} → {target_character}/{target_entry} "
          f"(학습 {copied_training}장, 테스트 {copied_test}장)")
    return {
        "success": True,
        "name": target_entry,
        "copied_training": copied_training,
        "copied_test": copied_test,
    }


def _cleanup_empty_dirs(path: str):
    """빈 디렉토리를 상위로 거슬러 올라가며 삭제"""
    try:
        while os.path.isdir(path):
            if not os.listdir(path):
                os.rmdir(path)
                print(f"[LORA_MANAGE] 빈 폴더 삭제: {path}")
                path = os.path.dirname(path)
            else:
                break
    except Exception as e:
        print(f"[LORA_MANAGE] 빈 폴더 정리 실패: {path} - {e}")

def remove_lora_entry(name: str, character: str, lora_load_path: str = "") -> dict:
    """LoRA 항목 삭제 (메타데이터 + 폴더)"""
    if not name:
        return {"success": False, "error": "이름 누락"}
    if not character:
        return {"success": False, "error": "캐릭터 누락"}

    data = _load_lora_manage()
    if not _get_entry(data, character, name):
        print(f"[LORA_MANAGE] 항목 없음: {character}/{name}")
        return {"success": False, "error": "항목을 찾을 수 없습니다"}

    # 에셋 항목 폴더 삭제
    entry_path = _lora_entry_dir(character, name)
    if os.path.isdir(entry_path):
        try:
            shutil.rmtree(entry_path)
            print(f"[LORA_MANAGE] 에셋 폴더 삭제: {entry_path}")
        except Exception as e:
            print(f"[LORA_MANAGE] 에셋 폴더 삭제 실패: {entry_path} - {e}")
            traceback.print_exc()
    _cleanup_empty_dirs(os.path.dirname(entry_path))

    # lora_load_path 항목 폴더도 삭제
    if lora_load_path:
        load_entry_dir = os.path.join(lora_load_path, _safe_dirname(character), "Lora", _safe_dirname(name))
        if os.path.isdir(load_entry_dir):
            try:
                shutil.rmtree(load_entry_dir)
                print(f"[LORA_MANAGE] 로드 경로 폴더 삭제: {load_entry_dir}")
            except Exception as e:
                print(f"[LORA_MANAGE] 로드 경로 폴더 삭제 실패: {load_entry_dir} - {e}")
                traceback.print_exc()
            _cleanup_empty_dirs(os.path.dirname(load_entry_dir))

    del data["loras"][character][name]
    # 캐릭터 하위가 비었으면 캐릭터 키도 삭제
    if not data["loras"][character]:
        del data["loras"][character]
    _save_lora_manage(data)
    print(f"[LORA_MANAGE] LoRA 삭제: {character}/{name}")
    return {"success": True}


def update_lora_entry(name: str, character: str, trigger: str = None, description: str = None, representative: str = None, training_config: dict = None, session_name: str = None, session_representative: str = None) -> dict:
    """LoRA 항목 메타데이터 수정"""
    if not name:
        return {"success": False, "error": "이름 누락"}
    if not character:
        return {"success": False, "error": "캐릭터 누락"}

    data = _load_lora_manage()
    entry = _get_entry(data, character, name)
    if entry is None:
        print(f"[LORA_MANAGE] 항목 없음: {character}/{name}")
        return {"success": False, "error": "항목을 찾을 수 없습니다"}

    if trigger is not None:
        entry["trigger"] = trigger.strip()
    if description is not None:
        entry["description"] = description.strip()
    if representative is not None:
        entry["representative"] = representative.strip()
    if training_config is not None:
        entry["training_config"] = training_config
    if session_name is not None and session_representative is not None:
        if "session_representatives" not in entry:
            entry["session_representatives"] = {}
        entry["session_representatives"][session_name] = session_representative

    _save_lora_manage(data)
    print(f"[LORA_MANAGE] LoRA 수정: {character}/{name}")
    return {"success": True}


def export_training_images(character: str, entry: str, comfy_input_dir: str) -> dict:
    """
    학습용 이미지를 Comfy Input 폴더 하위로 복사
    - multi_img_folder_name 서브폴더 생성 (없으면 생성, 있으면 내부 비우기)
    - 이미지를 [1].ext, [2].ext ... 형식으로 이름 변경하여 복사
    """
    # 학습 이미지 폴더 확인
    t_dir = _training_dir(character, entry)
    if not os.path.isdir(t_dir):
        print(f"[LORA_EXPORT] 학습 이미지 폴더 없음: {t_dir}")
        return {"success": False, "error": "학습용 이미지가 없습니다"}

    # 학습 이미지 파일 목록 (이미지 파일만)
    image_files = []
    for fname in sorted(os.listdir(t_dir)):
        if fname.startswith("_"):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_files.append(fname)

    if not image_files:
        print(f"[LORA_EXPORT] 학습 이미지 파일 없음: {t_dir}")
        return {"success": False, "error": "학습용 이미지 파일이 없습니다"}

    # multi_img_folder_name 읽기
    data = _load_lora_manage()
    entry_info = _get_entry(data, character, entry) or {}
    training_config = entry_info.get("training_config", {})
    folder_name = training_config.get("multi_img_folder_name", "soya_lora")

    # 대상 폴더 경로
    target_dir = os.path.join(comfy_input_dir, folder_name)

    # 대상 폴더가 있으면 내부 비우기
    if os.path.isdir(target_dir):
        print(f"[LORA_EXPORT] 대상 폴더 비우기: {target_dir}")
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"[LORA_EXPORT] 삭제 실패: {item_path} - {e}")
    else:
        os.makedirs(target_dir, exist_ok=True)
        print(f"[LORA_EXPORT] 대상 폴더 생성: {target_dir}")

    # 이미지 복사 ([1].ext, [2].ext ...)
    exported = []
    errors = []
    for idx, fname in enumerate(image_files, start=1):
        src_path = os.path.join(t_dir, fname)
        ext = os.path.splitext(fname)[1]
        dest_name = f"[{idx}]{ext}"
        dest_path = os.path.join(target_dir, dest_name)
        try:
            shutil.copy2(src_path, dest_path)
            exported.append(dest_name)
            print(f"[LORA_EXPORT] 복사: {src_path} -> {dest_path}")
        except Exception as e:
            print(f"[LORA_EXPORT] 복사 실패: {src_path} -> {dest_path} - {e}")
            traceback.print_exc()
            errors.append({"filename": fname, "reason": str(e)})

    return {
        "success": True,
        "exported": exported,
        "errors": errors,
        "target_dir": target_dir,
        "count": len(exported),
    }


def scan_untracked_loras(lora_load_path: str) -> dict:
    """lora_load_path에서 추적되지 않는 캐릭터/엔트리 스캔
    추적 조건: lora_manage.json에 등록되어 있고, 에셋(tags.json)에도 캐릭터가 존재해야 함"""
    if not lora_load_path or not os.path.isdir(lora_load_path):
        print(f"[LORA_UNTRACKED] lora_load_path 없음: {lora_load_path}")
        return {"success": False, "error": "로라 로드 경로가 존재하지 않습니다", "items": []}

    manage_data = _load_lora_manage()
    tracked = manage_data.get("loras", {})

    # 에셋 시스템의 캐릭터 목록 로드
    asset_chars = set()
    try:
        with open(TAGS_FILE, "r", encoding="utf-8") as f:
            asset_tags = json.load(f)
        asset_chars = set(asset_tags.get("characters", {}).keys())
    except Exception as e:
        print(f"[LORA_UNTRACKED] 에셋 캐릭터 로드 실패, 에셋 체크 생략: {e}")

    untracked = []

    # lora_load_path 바로 아래 항목 스캔
    try:
        top_items = os.listdir(lora_load_path)
    except Exception as e:
        print(f"[LORA_UNTRACKED] 경로 스캔 실패: {e}")
        traceback.print_exc()
        return {"success": False, "error": f"경로 스캔 실패: {e}", "items": []}

    for char_dir_name in top_items:
        char_path = os.path.join(lora_load_path, char_dir_name)
        if not os.path.isdir(char_path):
            continue

        # 이 디렉토리가 tracked의 어떤 캐릭터와 매칭되는지 확인
        # 단, 에셋 시스템에도 존재해야 추적으로 인정
        matched_char = None
        for tracked_char in tracked:
            if _safe_dirname(tracked_char) == char_dir_name:
                if asset_chars and tracked_char not in asset_chars:
                    # lora_manage에는 있지만 에셋에서 삭제된 캐릭터 → 비추적
                    print(f"[LORA_UNTRACKED] 에셋에 없는 캐릭터: {tracked_char}")
                    continue
                matched_char = tracked_char
                break

        if matched_char is None:
            # 캐릭터 자체가 추적 안됨 - 하위 전체가 비추적
            file_count = 0
            size_mb = 0
            for root, dirs, files in os.walk(char_path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        size_mb += os.path.getsize(fp)
                        file_count += 1
                    except Exception:
                        pass
            size_mb = round(size_mb / (1024 * 1024), 1)
            untracked.append({
                "type": "character",
                "path": char_path,
                "display": char_dir_name,
                "file_count": file_count,
                "size_mb": size_mb,
            })
            continue

        # 추적된 캐릭터: Lora 하위 엔트리 중 비추적 항목 스캔
        lora_dir = os.path.join(char_path, "Lora")
        if not os.path.isdir(lora_dir):
            continue
        tracked_entries = tracked.get(matched_char, {})
        try:
            entry_dirs = os.listdir(lora_dir)
        except Exception:
            continue
        for entry_dir_name in entry_dirs:
            entry_path = os.path.join(lora_dir, entry_dir_name)
            if not os.path.isdir(entry_path):
                continue
            # 이 엔트리 디렉토리가 tracked 엔트리와 매칭되는지 확인
            is_tracked = any(
                _safe_dirname(e_name) == entry_dir_name
                for e_name in tracked_entries
            )
            if not is_tracked:
                file_count = 0
                size_mb = 0
                for root, dirs, files in os.walk(entry_path):
                    for f in files:
                        try:
                            size_mb += os.path.getsize(os.path.join(root, f))
                            file_count += 1
                        except Exception:
                            pass
                size_mb = round(size_mb / (1024 * 1024), 1)
                untracked.append({
                    "type": "entry",
                    "path": entry_path,
                    "display": f"{char_dir_name}/Lora/{entry_dir_name}",
                    "file_count": file_count,
                    "size_mb": size_mb,
                })

    # 디스크에 없지만 lora_manage.json에만 있는 항목도 비추적으로 추가
    disk_dirs = set(top_items) if os.path.isdir(lora_load_path) else set()
    for tracked_char, entries in tracked.items():
        safe_char = _safe_dirname(tracked_char)
        # 에셋에 없는 캐릭터 or 디스크에 없는 캐릭터 → 비추적
        if (asset_chars and tracked_char not in asset_chars) or safe_char not in disk_dirs:
            if not any(u["type"] == "character" and u["display"] == safe_char for u in untracked):
                # 캐릭터 전체가 비추적
                char_path = os.path.join(lora_load_path, safe_char) if os.path.isdir(lora_load_path) else safe_char
                file_count = 0
                size_mb = 0.0
                if os.path.isdir(char_path):
                    for root, dirs, files in os.walk(char_path):
                        for f in files:
                            try:
                                size_mb += os.path.getsize(os.path.join(root, f))
                                file_count += 1
                            except Exception:
                                pass
                    size_mb = round(size_mb / (1024 * 1024), 1)
                reason = "에셋에 없음" if (asset_chars and tracked_char not in asset_chars) else "폴더 없음"
                print(f"[LORA_UNTRACKED] JSON에만 있는 캐릭터 ({reason}): {tracked_char}")
                untracked.append({
                    "type": "character",
                    "path": char_path,
                    "display": safe_char,
                    "file_count": file_count,
                    "size_mb": size_mb,
                })

    print(f"[LORA_UNTRACKED] 스캔 완료: {len(untracked)}개 비추적 항목 발견")
    return {"success": True, "items": untracked}


def remove_untracked_loras(items: list, cleanup_manage: bool = False) -> dict:
    """비추적 LoRA 항목 일괄 삭제
    cleanup_manage=True이면 lora_manage.json에서도 해당 항목 제거"""
    removed = []
    errors = []

    manage_data = None
    if cleanup_manage:
        manage_data = _load_lora_manage()

    for item in items:
        path = item.get("path", "")
        item_type = item.get("type", "")
        display = item.get("display", "")

        if not path or not os.path.exists(path):
            # 폴더가 없으면 정리만 수행하고 성공으로 처리
            if cleanup_manage and manage_data and display:
                _cleanup_manage_entry(manage_data, display, item_type)
            removed.append(display or path)
            print(f"[LORA_UNTRACKED] 폴더 없음 (정리만 완료): {display or path}")
            continue

        try:
            shutil.rmtree(path)
            removed.append(path)
            print(f"[LORA_UNTRACKED] 삭제: {path}")
            _cleanup_empty_dirs(os.path.dirname(path))
        except Exception as e:
            errors.append({"path": path, "reason": str(e)})
            print(f"[LORA_UNTRACKED] 삭제 실패: {path} - {e}")
            traceback.print_exc()
            continue

        # 폴더 삭제 성공 시 manage 정리
        if cleanup_manage and manage_data and display:
            _cleanup_manage_entry(manage_data, display, item_type)

    if cleanup_manage and manage_data:
        _save_lora_manage(manage_data)

    return {"success": True, "removed": removed, "errors": errors, "removed_count": len(removed)}


def _cleanup_manage_entry(manage_data: dict, display: str, item_type: str):
    """display 이름으로 lora_manage.json에서 항목 제거
    character 타입: 캐릭터 키 전체 제거
    entry 타입: display에서 캐릭터/엔트리 파싱하여 엔트리만 제거"""
    if item_type == "entry" and "/" in display:
        # display 형식: "캐릭터디렉토리/Lora/엔트리디렉토리"
        parts = display.split("/Lora/", 1)
        if len(parts) == 2:
            safe_char = parts[0]
            safe_entry = parts[1]
            # safe_dirname에서 원래 이름 찾기
            for char_name in list(manage_data.get("loras", {}).keys()):
                if _safe_dirname(char_name) == safe_char:
                    for entry_name in list(manage_data["loras"][char_name].keys()):
                        if _safe_dirname(entry_name) == safe_entry:
                            del manage_data["loras"][char_name][entry_name]
                            print(f"[LORA_UNTRACKED] lora_manage에서 엔트리 제거: {char_name}/{entry_name}")
                            # 캐릭터 하위가 비었으면 캐릭터 키도 제거
                            if not manage_data["loras"][char_name]:
                                del manage_data["loras"][char_name]
                                print(f"[LORA_UNTRACKED] lora_manage에서 빈 캐릭터 제거: {char_name}")
                            return
    # character 타입: 캐릭터 키 전체 제거
    char_name = display
    if char_name in manage_data.get("loras", {}):
        del manage_data["loras"][char_name]
        print(f"[LORA_UNTRACKED] lora_manage에서 캐릭터 제거: {char_name}")


def get_entry_image_path(character: str, entry_name: str, filename: str) -> str | None:
    """LoRA 항목 폴더 내 이미지 파일 경로 반환"""
    if ".." in entry_name or ".." in filename or os.path.sep in entry_name or os.path.sep in filename:
        print(f"[LORA_MANAGE] 잘못된 경로: {entry_name}/{filename}")
        return None
    fpath = os.path.join(_lora_entry_dir(character, entry_name), filename)
    if os.path.isfile(fpath):
        return fpath
    # 엔트리 내 학습용 이미지 폴더도 확인
    tpath = os.path.join(_training_dir(character, entry_name), filename)
    if os.path.isfile(tpath):
        return tpath
    # 엔트리 내 테스트 이미지 폴더도 확인
    testpath = os.path.join(_test_dir(character, entry_name), filename)
    if os.path.isfile(testpath):
        return testpath
    print(f"[LORA_MANAGE] 이미지 없음: {fpath}")
    return None
