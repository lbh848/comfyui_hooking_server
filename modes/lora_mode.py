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


# ─── LoRA 항목 관리 (lora_manage.json) ─────────────────────────

def _load_lora_manage() -> dict:
    """lora_manage.json 로드 (필요시 마이그레이션)"""
    if os.path.isfile(LORA_MANAGE_FILE):
        try:
            with open(LORA_MANAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_lora_manage(data)
            return data
        except Exception as e:
            print(f"[LORA_MANAGE] 로드 실패: {e}")
            traceback.print_exc()
            return {"loras": {}}
    return {"loras": {}}


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


def remove_lora_entry(name: str, character: str) -> dict:
    """LoRA 항목 삭제 (메타데이터 + 폴더)"""
    if not name:
        return {"success": False, "error": "이름 누락"}
    if not character:
        return {"success": False, "error": "캐릭터 누락"}

    data = _load_lora_manage()
    if not _get_entry(data, character, name):
        print(f"[LORA_MANAGE] 항목 없음: {character}/{name}")
        return {"success": False, "error": "항목을 찾을 수 없습니다"}

    # 항목 폴더 삭제
    entry_path = _lora_entry_dir(character, name)
    if os.path.isdir(entry_path):
        try:
            shutil.rmtree(entry_path)
            print(f"[LORA_MANAGE] 폴더 삭제: {entry_path}")
        except Exception as e:
            print(f"[LORA_MANAGE] 폴더 삭제 실패: {entry_path} - {e}")
            traceback.print_exc()

    del data["loras"][character][name]
    # 캐릭터 하위가 비었으면 캐릭터 키도 삭제
    if not data["loras"][character]:
        del data["loras"][character]
    _save_lora_manage(data)
    print(f"[LORA_MANAGE] LoRA 삭제: {character}/{name}")
    return {"success": True}


def update_lora_entry(name: str, character: str, trigger: str = None, description: str = None, representative: str = None, training_config: dict = None) -> dict:
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
