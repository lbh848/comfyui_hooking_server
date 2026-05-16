"""
LoRA 매니징 모듈
- asset/<character>/Lora/<entry>/ 폴더에서 LoRA 파일 관리
- asset/<character>/Lora/<entry>/학습용 이미지/ 폴더에서 학습용 이미지 관리
"""

import os
import json
import shutil
import traceback
from modes.asset_mode import ASSET_DIR, TAGS_FILE, AssetMode

LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".bin"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TRAINING_DIR_NAME = "학습용 이미지"
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
    char_dir = os.path.join(ASSET_DIR, _safe_dirname(character))

    for src in sources:
        outfit = src.get("outfit", "")
        expression = src.get("expression", "")
        filename = src.get("filename", "")

        if not outfit or not expression or not filename:
            print(f"[LORA] 학습 이미지 추가: 필수 값 누락 - {src}")
            skipped.append({"filename": filename, "reason": "필수 값 누락"})
            continue

        # 원본 경로
        src_path = os.path.join(char_dir, _safe_dirname(outfit), _safe_dirname(expression), filename)
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
            prompt_src = os.path.join(char_dir, _safe_dirname(outfit), _safe_dirname(expression), f"{base}_prompt.json")
            prompt_dest = os.path.join(t_dir, f"{os.path.splitext(dest_name)[0]}_prompt.json")
            if os.path.isfile(prompt_src):
                # 복사하면서 positive에서 [FACE_ID_ACTIVATE] 위쪽만 추출
                with open(prompt_src, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                positive = pdata.get("positive", "")
                marker = "[FACE_ID_ACTIVATE]"
                if marker in positive:
                    pdata["positive"] = positive.split(marker)[0].strip()
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump(pdata, f, ensure_ascii=False, indent=2)
            else:
                # 프롬프트 파일이 없으면 빈 프롬프트 생성
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    json.dump({"positive": "", "negative": ""}, f, ensure_ascii=False, indent=2)

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
            except Exception as e:
                print(f"[LORA] 프롬프트 로드 실패: {prompt_path} - {e}")

        try:
            fstat = os.stat(fpath)
            images.append({
                "filename": fname,
                "positive": positive,
                "negative": negative,
                "is_representative": fname == representative,
                "size": fstat.st_size,
                "modified": fstat.st_mtime,
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
        existing["positive"] = positive
        existing["negative"] = negative

        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[LORA] 프롬프트 저장 완료: {prompt_path}")
        return {"success": True}
    except Exception as e:
        print(f"[LORA] 프롬프트 저장 실패: {prompt_path} - {e}")
        return {"success": False, "error": str(e)}


# ─── LoRA 항목 관리 (lora_manage.json) ─────────────────────────

def _load_lora_manage() -> dict:
    """lora_manage.json 로드"""
    if os.path.isfile(LORA_MANAGE_FILE):
        try:
            with open(LORA_MANAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[LORA_MANAGE] 로드 실패: {e}")
            traceback.print_exc()
            return {"loras": {}}
    return {"loras": {}}


def _save_lora_manage(data: dict):
    """lora_manage.json 저장"""
    try:
        with open(LORA_MANAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[LORA_MANAGE] 저장 완료")
    except Exception as e:
        print(f"[LORA_MANAGE] 저장 실패: {e}")
        traceback.print_exc()


def list_lora_entries(character: str = "") -> list:
    """LoRA 항목 목록 반환. character 지정 시 해당 캐릭터 것만."""
    data = _load_lora_manage()
    entries = []
    for name, info in data.get("loras", {}).items():
        if character and info.get("character") != character:
            continue
        entries.append({
            "name": name,
            "trigger": info.get("trigger", ""),
            "description": info.get("description", ""),
            "character": info.get("character", ""),
            "representative": info.get("representative", ""),
        })
    return entries


def add_lora_entry(name: str, character: str, trigger: str, description: str = "") -> dict:
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

    if name in data.get("loras", {}):
        print(f"[LORA_MANAGE] 이미 존재: {name}")
        return {"success": False, "error": "이미 존재하는 LoRA명"}

    # 캐릭터 Lora 폴더 + 항목 폴더 생성
    entry_path = _lora_entry_dir(character, name)
    os.makedirs(entry_path, exist_ok=True)

    data.setdefault("loras", {})[name] = {
        "trigger": trigger.strip(),
        "description": description.strip(),
        "character": character,
    }
    _save_lora_manage(data)
    print(f"[LORA_MANAGE] LoRA 추가: {name} (캐릭터: {character}, 트리거: {trigger.strip()})")
    return {"success": True, "name": name}


def remove_lora_entry(name: str, character: str = "") -> dict:
    """LoRA 항목 삭제 (메타데이터 + 폴더)"""
    if not name:
        return {"success": False, "error": "이름 누락"}

    data = _load_lora_manage()
    if name not in data.get("loras", {}):
        print(f"[LORA_MANAGE] 항목 없음: {name}")
        return {"success": False, "error": "항목을 찾을 수 없습니다"}

    # character가 안 주어지면 메타데이터에서 가져옴
    if not character:
        character = data["loras"][name].get("character", "")

    # 항목 폴더 삭제
    if character:
        entry_path = _lora_entry_dir(character, name)
        if os.path.isdir(entry_path):
            try:
                shutil.rmtree(entry_path)
                print(f"[LORA_MANAGE] 폴더 삭제: {entry_path}")
            except Exception as e:
                print(f"[LORA_MANAGE] 폴더 삭제 실패: {entry_path} - {e}")
                traceback.print_exc()

    del data["loras"][name]
    _save_lora_manage(data)
    print(f"[LORA_MANAGE] LoRA 삭제: {name}")
    return {"success": True}


def update_lora_entry(name: str, trigger: str = None, description: str = None, representative: str = None) -> dict:
    """LoRA 항목 메타데이터 수정"""
    if not name:
        return {"success": False, "error": "이름 누락"}

    data = _load_lora_manage()
    if name not in data.get("loras", {}):
        print(f"[LORA_MANAGE] 항목 없음: {name}")
        return {"success": False, "error": "항목을 찾을 수 없습니다"}

    if trigger is not None:
        data["loras"][name]["trigger"] = trigger.strip()
    if description is not None:
        data["loras"][name]["description"] = description.strip()
    if representative is not None:
        data["loras"][name]["representative"] = representative.strip()

    _save_lora_manage(data)
    print(f"[LORA_MANAGE] LoRA 수정: {name}")
    return {"success": True}


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
    print(f"[LORA_MANAGE] 이미지 없음: {fpath}")
    return None
