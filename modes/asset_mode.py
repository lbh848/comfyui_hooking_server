"""
AssetMode - 에셋 생성 모드

외모/복장/표정은 전역 관리, 캐릭터는 조합만 참조.
"""

import asyncio
import json
import os
import copy
import time
import uuid
import hashlib
import shutil
import traceback
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, Awaitable
import workflow_profiles


# ─── 상수 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
ASSET_DIR = os.path.join(BASE_DIR, "asset")
AUTOMATCH_DEFAULT_OUTFIT_DIR = "_automatch_defaults"
# 캐릭터 메이커 작업공간: 서버 재시작에도 살아있는 단일 영속 세션.
# 프로젝트 루트 안의 안정 디렉터리를 사용하며 .gitignore로 배포에서 제외한다.
CHARACTER_MAKER_TEMP_DIR = os.path.join(BASE_DIR, "character_maker_data")
os.makedirs(CHARACTER_MAKER_TEMP_DIR, exist_ok=True)
# 캐릭터 메이커 단일 영속 세션 식별자. character_maker_mode.SINGLE_SESSION_ID 와
# 동일한 값이어야 한다(순환 import 회피를 위해 이쪽에 리터럴로 정의).
CHARACTER_MAKER_SINGLE_SESSION_ID = "default"
TAGS_FILE = os.path.join(ASSET_DATA_DIR, "tags.json")
HIDDEN_TAGS_FILE = os.path.join(ASSET_DATA_DIR, "hidden_tags.json")
NAME_MAPPING_FILE = os.path.join(ASSET_DATA_DIR, "name_mapping.json")
NAME_MAPPING_BACKUP_DIR = os.path.join(BASE_DIR, "요구사항")

EXPORT_NAMING_BLOCKS = ("character", "outfit", "expression")
_INVALID_EXPORT_TOKEN_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# 프리셋매니징 대상 카테고리
PRESET_MGMT_CATEGORIES = [
    "appearances", "outfits", "expressions",
    "quality_presets", "composition_presets",
    "negative_presets", "character_negative_presets",
    "artist_presets", "natural_language_presets",
]
CURRENT_MODE_WORK_DIR = os.path.join(BASE_DIR, "current_mode_workflow")
MODE_WORKFLOW_DIR = os.path.join(BASE_DIR, "mode_workflow")

DEFAULT_POSE_DATA = {
    "people": [
        {
            "pose_keypoints_2d": [
                381.10, 379.86, 1,
                413.21, 506.58, 1,
                223.64, 436.41, 1,
                55.33, 624.12, 1,
                64.77, 454.24, 1,
                579.09, 575.30, 1,
                362.22, 935.42, 1,
                331.06, 972.15, 1,
                196.97, 797.67, 1,
                264.96, 1013.47, 1,
                257.41, 1024.49, 1,
                350.89, 820.63, 1,
                419.82, 1011.63, 1,
                419.82, 1024.49, 1,
                315.95, 343.13, 1,
                462.31, 294.46, 1,
                263.07, 365.17, 1,
                546.35, 301.80, 1,
            ]
        }
    ],
    "canvas_height": 1024,
    "canvas_width": 700,
}

DEFAULT_TAGS = {
    "quality": ["masterpiece, best quality"],
    "composition": [],
    "negative": ["lowres, bad anatomy, bad hands"],
    "character_negative": [],
    "appearances": {},
    "outfits": {},
    "expressions": {},
    "characters": {},
    "quality_presets": {},
    "composition_presets": {},
    "negative_presets": {},
    "character_negative_presets": {},
    "artist_presets": {},          # { "name": ["tag1", "tag2"] }
    "natural_language_presets": {},  # { "name": "긴 텍스트" }
    "anima_quality": [],
    "anima_negative": [],
}


# ─── 프리셋 추적 병렬화 (I/O 바운드 → 스레드풀) ───────────────────────
def _trace_worker_count() -> int:
    """스레드풀 워커 수. 시스템 여유 확보를 위해 cpu-2 기반, 최소 2, 상한 16."""
    cpu = os.cpu_count() or 4
    base = max(2, cpu - 2)
    return min(16, base * 2)


_trace_executor: Optional[ThreadPoolExecutor] = None


def _get_trace_executor() -> ThreadPoolExecutor:
    """프리셋 추적 전용 공유 스레드풀(지연 생성, 재사용)."""
    global _trace_executor
    if _trace_executor is None:
        _trace_executor = ThreadPoolExecutor(
            max_workers=_trace_worker_count(), thread_name_prefix="preset-trace"
        )
    return _trace_executor


# 카테고리 → prompt_data 필드 매핑 (필드값 일치 후 태그 카운트)
_TRACE_FIELD_MAP = {
    "appearances": "appearance",
    "outfits": "outfit",
    "expressions": "expression",
}


def _collect_prompt_files() -> list:
    """asset/ 트리의 모든 *_prompt.json 을 (char, outfit, expr, expr_dir, fname, path) 튜플로 수집.
    순회 1회만 수행하여 total 카운트/스캔 중복 순회를 제거한다."""
    files = []
    if not os.path.isdir(ASSET_DIR):
        return files
    for char_name in os.listdir(ASSET_DIR):
        char_dir = os.path.join(ASSET_DIR, char_name)
        if not os.path.isdir(char_dir):
            continue
        for outfit_name in os.listdir(char_dir):
            outfit_dir = os.path.join(char_dir, outfit_name)
            if not os.path.isdir(outfit_dir):
                continue
            for expr_name in os.listdir(outfit_dir):
                expr_dir = os.path.join(outfit_dir, expr_name)
                if not os.path.isdir(expr_dir):
                    continue
                for fname in os.listdir(expr_dir):
                    if fname.endswith("_prompt.json"):
                        files.append((
                            char_name, outfit_name, expr_name,
                            expr_dir, fname, os.path.join(expr_dir, fname),
                        ))
    return files


def _match_one_prompt(args) -> Optional[dict]:
    """프롬프트 파일 1개를 읽어 프리셋 매칭. 스레드 안전(공유 가변 상태 없음, 읽기 전용).
    인자: (prompt_path, category, name, preset_tags, char_name, outfit_name, expr_name, expr_dir, fname)
    반환: 매치 시 dict(positive/negative/_prompt_data 포함 슈퍼셋), 미매치·실패 시 None."""
    (prompt_path, category, name, preset_tags,
     char_name, outfit_name, expr_name, expr_dir, fname) = args
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except Exception as e:
        print(f"[ASSET_MODE] trace: {prompt_path} 읽기 실패: {e}")
        return None

    match_count = 0
    if category in _TRACE_FIELD_MAP:
        field = _TRACE_FIELD_MAP[category]
        if prompt_data.get(field) == name:
            positive = prompt_data.get("positive", "")
            match_count = sum(1 for tag in preset_tags if tag and tag.lower() in positive.lower())
            if match_count == 0:
                match_count = 1
    elif category in ("quality_presets", "composition_presets", "artist_presets"):
        positive = prompt_data.get("positive", "")
        match_count = sum(1 for tag in preset_tags if tag and tag.lower() in positive.lower())
    elif category in ("negative_presets", "character_negative_presets"):
        negative = prompt_data.get("negative", "")
        match_count = sum(1 for tag in preset_tags if tag and tag.lower() in negative.lower())
    elif category == "natural_language_presets":
        positive = prompt_data.get("positive", "")
        nl_text = preset_tags[0] if preset_tags else ""
        if nl_text and nl_text.lower() in positive.lower():
            match_count = 1

    if match_count <= 0:
        return None

    img_file = fname.replace("_prompt.json", "")
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        if os.path.isfile(os.path.join(expr_dir, img_file + ext)):
            img_file = img_file + ext
            break
    else:
        img_file = fname

    return {
        "character": prompt_data.get("character", char_name),
        "outfit": prompt_data.get("outfit", outfit_name),
        "expression": prompt_data.get("expression", expr_name),
        "image_file": img_file,
        "match_count": match_count,
        "positive": prompt_data.get("positive", ""),
        "negative": prompt_data.get("negative", ""),
        "_prompt_data": prompt_data,  # 내부용: 동기 경로만 사용
    }


def _split_chunks(seq, n):
    """seq를 n개 청크로 분할(균등). n > len(seq) 이면 빈 청크 없이 분배."""
    n = max(1, min(n, len(seq))) if seq else 1
    k, m = divmod(len(seq), n)
    out = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        out.append(seq[start:start + size])
        start += size
    return out


def _match_chunk(file_args) -> list:
    """한 청크(파일 인자 리스트)를 순차 스캔해 매치 결과 리스트 반환.
    청크 단위로 future를 만들어 10k future 제출 오버헤드를 회피한다.
    웜 캐시에서 순차와 동급, 콜드 캐시에서 스레드풀 I/O 병렬 이득 유지."""
    out = []
    for a in file_args:
        m = _match_one_prompt(a)
        if m is not None:
            out.append(m)
    return out


class AssetMode:
    """에셋 생성 모드 매니저"""

    def __init__(self):
        self.enabled: bool = False
        self.workflow_source_path: str = ""
        self.anima_workflow_source_path: str = ""
        self.anima_only_workflow_source_path: str = ""
        self.workflow_type: str = workflow_profiles.ASSET_ILXL
        self._asset_api_workflow: Optional[dict] = None
        self._asset_hash: str = ""
        self._tags: dict = copy.deepcopy(DEFAULT_TAGS)
        self._tags_loaded: bool = False
        self._is_generating: bool = False
        self._lock = asyncio.Lock()

        # 콜백
        self.mode_log_func: Optional[Callable] = None
        self.notify_frontend_func: Optional[Callable] = None
        self.convert_workflow_func: Optional[Callable] = None
        self.compute_hash_func: Optional[Callable] = None
        self.submit_workflow_func: Optional[Callable] = None
        self.build_prompt_with_workflow_func: Optional[Callable] = None
        self.upload_reference_image_func: Optional[Callable] = None

    def _log(self, action: str, data: dict = None):
        if self.mode_log_func:
            self.mode_log_func("asset_mode", action, data)

    # ─── 태그 로드 / 마이그레이션 ──────────────────────────
    def load_tags(self):
        os.makedirs(ASSET_DATA_DIR, exist_ok=True)
        if os.path.isfile(TAGS_FILE):
            try:
                with open(TAGS_FILE, "r", encoding="utf-8") as f:
                    self._tags = json.load(f)
                self._migrate_if_needed()
                for k, v in DEFAULT_TAGS.items():
                    if k not in self._tags:
                        self._tags[k] = copy.deepcopy(v)
                self._tags_loaded = True
                self._log("tags_loaded", {"characters": len(self._tags.get("characters", {}))})
            except Exception as e:
                self._log("tags_load_error", {"error": str(e)})
                self._tags = copy.deepcopy(DEFAULT_TAGS)
                self._tags_loaded = True
        else:
            self._tags = copy.deepcopy(DEFAULT_TAGS)
            self._tags_loaded = True

    def _migrate_if_needed(self):
        """옛날 구조(캐릭터 하위 appearance/outfits/expressions 딕셔너리) → 새 구조로 변환."""
        chars = self._tags.get("characters", {})
        if not chars:
            return
        first_char = next(iter(chars.values()), None)
        if not isinstance(first_char, dict):
            return
        # 옛날 형식: {"appearance": {"name": [tags]}, "outfits": {...}, "expressions": {...}}
        if "appearance" in first_char and isinstance(first_char["appearance"], dict):
            g_apps = self._tags.setdefault("appearances", {})
            g_outs = self._tags.setdefault("outfits", {})
            g_exprs = self._tags.setdefault("expressions", {})
            for char_name, char_data in list(chars.items()):
                for n, t in char_data.get("appearance", {}).items():
                    if n not in g_apps:
                        g_apps[n] = list(t)
                for n, t in char_data.get("outfits", {}).items():
                    if n not in g_outs:
                        g_outs[n] = list(t)
                for n, t in char_data.get("expressions", {}).items():
                    if n not in g_exprs:
                        g_exprs[n] = list(t)
                chars[char_name] = {
                    "appearance": next(iter(char_data.get("appearance", {})), ""),
                    "outfit": next(iter(char_data.get("outfits", {})), ""),
                    "expression": next(iter(char_data.get("expressions", {})), ""),
                }
            for key in ["appearance_presets", "outfit_presets", "expression_presets"]:
                self._tags.pop(key, None)
            self.save_tags()
            self._log("tags_migrated", {})

    def save_tags(self):
        if not self._tags_loaded:
            print("[ASSET_MODE] WARNING: save_tags() called before load_tags(). Skipping to prevent data loss.")
            return
        os.makedirs(ASSET_DATA_DIR, exist_ok=True)
        with open(TAGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._tags, f, indent=2, ensure_ascii=False)

    def get_tags(self) -> dict:
        return copy.deepcopy(self._tags)

    # ─── 전역 외모 관리 ──────────────────────────────────
    def list_appearances(self) -> list[str]:
        return list(self._tags.get("appearances", {}).keys())

    def add_appearance(self, name: str) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        apps = self._tags.setdefault("appearances", {})
        if name in apps:
            return {"success": False, "error": "이미 존재하는 외모"}
        apps[name] = [""]
        self.save_tags()
        return {"success": True}

    def remove_appearance(self, name: str) -> dict:
        apps = self._tags.get("appearances", {})
        if name not in apps:
            return {"success": False, "error": "존재하지 않는 외모"}
        del apps[name]
        self.save_tags()
        return {"success": True}

    def duplicate_appearance(self, name: str, new_name: str) -> dict:
        apps = self._tags.get("appearances", {})
        if name not in apps:
            return {"success": False, "error": "원본 외모가 존재하지 않음"}
        if not new_name.strip():
            return {"success": False, "error": "빈 이름"}
        if new_name in apps:
            return {"success": False, "error": "이미 존재하는 외모명"}
        apps[new_name] = list(apps[name])
        self.save_tags()
        return {"success": True}

    def add_appearance_tag(self, name: str, value: str) -> dict:
        apps = self._tags.get("appearances", {})
        if name not in apps:
            return {"success": False, "error": "존재하지 않는 외모"}
        if value.strip() in [t.strip() for t in apps[name]]:
            return {"success": False, "error": "이미 존재하는 태그"}
        apps[name].append(value)
        self.save_tags()
        return {"success": True}

    def remove_appearance_tag(self, name: str, index: int) -> dict:
        apps = self._tags.get("appearances", {})
        if name not in apps:
            return {"success": False, "error": "존재하지 않는 외모"}
        tags = apps[name]
        if index < 0 or index >= len(tags):
            return {"success": False, "error": "잘못된 인덱스"}
        tags.pop(index)
        self.save_tags()
        return {"success": True}

    # ─── 전역 복장 관리 ──────────────────────────────────
    def list_outfits(self) -> list[str]:
        return list(self._tags.get("outfits", {}).keys())

    def add_outfit(self, name: str) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        outs = self._tags.setdefault("outfits", {})
        if name in outs:
            return {"success": False, "error": "이미 존재하는 복장"}
        outs[name] = [""]
        self.save_tags()
        return {"success": True}

    def remove_outfit(self, name: str) -> dict:
        outs = self._tags.get("outfits", {})
        if name not in outs:
            return {"success": False, "error": "존재하지 않는 복장"}
        del outs[name]
        self.save_tags()
        return {"success": True}

    def add_outfit_tag(self, name: str, value: str) -> dict:
        outs = self._tags.get("outfits", {})
        if name not in outs:
            return {"success": False, "error": "존재하지 않는 복장"}
        if value.strip() in [t.strip() for t in outs[name]]:
            return {"success": False, "error": "이미 존재하는 태그"}
        outs[name].append(value)
        self.save_tags()
        return {"success": True}

    def remove_outfit_tag(self, name: str, index: int) -> dict:
        outs = self._tags.get("outfits", {})
        if name not in outs:
            return {"success": False, "error": "존재하지 않는 복장"}
        tags = outs[name]
        if index < 0 or index >= len(tags):
            return {"success": False, "error": "잘못된 인덱스"}
        tags.pop(index)
        self.save_tags()
        return {"success": True}

    def duplicate_outfit(self, name: str, new_name: str) -> dict:
        outs = self._tags.get("outfits", {})
        if name not in outs:
            return {"success": False, "error": "원본 복장이 존재하지 않음"}
        if not new_name.strip():
            return {"success": False, "error": "빈 이름"}
        if new_name in outs:
            return {"success": False, "error": "이미 존재하는 복장명"}
        outs[new_name] = list(outs[name])
        self.save_tags()
        return {"success": True}

    # ─── 전역 표정 관리 ──────────────────────────────────
    def list_expressions(self) -> list[str]:
        return list(self._tags.get("expressions", {}).keys())

    def add_expression(self, name: str) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        exprs = self._tags.setdefault("expressions", {})
        if name in exprs:
            return {"success": False, "error": "이미 존재하는 표정"}
        exprs[name] = [""]
        self.save_tags()
        return {"success": True}

    def remove_expression(self, name: str) -> dict:
        exprs = self._tags.get("expressions", {})
        if name not in exprs:
            return {"success": False, "error": "존재하지 않는 표정"}
        del exprs[name]
        self.save_tags()
        return {"success": True}

    def add_expression_tag(self, name: str, value: str) -> dict:
        exprs = self._tags.get("expressions", {})
        if name not in exprs:
            return {"success": False, "error": "존재하지 않는 표정"}
        if value.strip() in [t.strip() for t in exprs[name]]:
            return {"success": False, "error": "이미 존재하는 태그"}
        exprs[name].append(value)
        self.save_tags()
        return {"success": True}

    def remove_expression_tag(self, name: str, index: int) -> dict:
        exprs = self._tags.get("expressions", {})
        if name not in exprs:
            return {"success": False, "error": "존재하지 않는 표정"}
        tags = exprs[name]
        if index < 0 or index >= len(tags):
            return {"success": False, "error": "잘못된 인덱스"}
        tags.pop(index)
        self.save_tags()
        return {"success": True}

    def duplicate_expression(self, name: str, new_name: str) -> dict:
        exprs = self._tags.get("expressions", {})
        if name not in exprs:
            return {"success": False, "error": "원본 표정이 존재하지 않음"}
        if not new_name.strip():
            return {"success": False, "error": "빈 이름"}
        if new_name in exprs:
            return {"success": False, "error": "이미 존재하는 표정명"}
        exprs[new_name] = list(exprs[name])
        self.save_tags()
        return {"success": True}

    # ─── 공통 품질/구도/부정 태그 관리 ──────────────────────
    def add_quality_tag(self, value: str) -> dict:
        if value.strip() in [t.strip() for t in self._tags["quality"]]:
            return {"success": False, "error": "이미 존재하는 태그"}
        self._tags["quality"].append(value)
        self.save_tags()
        return {"success": True}

    def remove_quality_tag(self, index: int) -> dict:
        if index < 0 or index >= len(self._tags["quality"]):
            return {"success": False, "error": "잘못된 인덱스"}
        self._tags["quality"].pop(index)
        self.save_tags()
        return {"success": True}

    def add_composition_tag(self, value: str) -> dict:
        tags = self._tags.setdefault("composition", [])
        if value.strip() in [t.strip() for t in tags]:
            return {"success": False, "error": "이미 존재하는 태그"}
        tags.append(value)
        self.save_tags()
        return {"success": True}

    def remove_composition_tag(self, index: int) -> dict:
        tags = self._tags.get("composition", [])
        if index < 0 or index >= len(tags):
            return {"success": False, "error": "잘못된 인덱스"}
        tags.pop(index)
        self.save_tags()
        return {"success": True}

    def add_negative_tag(self, value: str) -> dict:
        if value.strip() in [t.strip() for t in self._tags["negative"]]:
            return {"success": False, "error": "이미 존재하는 태그"}
        self._tags["negative"].append(value)
        self.save_tags()
        return {"success": True}

    def remove_negative_tag(self, index: int) -> dict:
        if index < 0 or index >= len(self._tags["negative"]):
            return {"success": False, "error": "잘못된 인덱스"}
        self._tags["negative"].pop(index)
        self.save_tags()
        return {"success": True}

    # ─── 태그 순서 변경 ──────────────────────────────────
    def reorder_global_tags(self, category: str, order: list[int] = None, tags: list[str] = None) -> dict:
        # tags 파라미터로 직접 순서가 지정된 태그 리스트를 받음
        if tags is not None:
            self._tags[category] = [t for t in tags if t]
            self.save_tags()
            return {"success": True}
        # 기존 인덱스 방식 (fallback)
        current = self._tags.get(category, [])
        if not order or len(order) != len(current):
            return {"success": False, "error": "순서 길이 불일치"}
        try:
            self._tags[category] = [current[i] for i in order]
        except (IndexError, TypeError):
            return {"success": False, "error": "잘못된 인덱스"}
        self.save_tags()
        return {"success": True}

    def reorder_sub_tags(self, sub: str, name: str, order: list[int]) -> dict:
        """sub: appearances / outfits / expressions"""
        group = self._tags.get(sub, {})
        if name not in group:
            return {"success": False, "error": "존재하지 않음"}
        tags = group[name]
        if not order or len(order) != len(tags):
            return {"success": False, "error": "순서 길이 불일치"}
        try:
            group[name] = [tags[i] for i in order]
        except (IndexError, TypeError):
            return {"success": False, "error": "잘못된 인덱스"}
        self.save_tags()
        return {"success": True}

    # ─── 품질 프리셋 ──────────────────────────────────────
    def get_quality_presets(self) -> dict:
        return copy.deepcopy(self._tags.get("quality_presets", {}))

    def save_quality_preset(self, name: str, tags: list[str]) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        self._tags.setdefault("quality_presets", {})[name.strip()] = list(tags)
        self.save_tags()
        return {"success": True}

    def delete_quality_preset(self, name: str) -> dict:
        presets = self._tags.get("quality_presets", {})
        if name not in presets:
            return {"success": False, "error": "존재하지 않는 프리셋"}
        del presets[name]
        self.save_tags()
        return {"success": True}

    # ─── 구도 프리셋 ──────────────────────────────────────
    def get_composition_presets(self) -> dict:
        return copy.deepcopy(self._tags.get("composition_presets", {}))

    def save_composition_preset(self, name: str, tags: list[str]) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        self._tags.setdefault("composition_presets", {})[name.strip()] = list(tags)
        self.save_tags()
        return {"success": True}

    def delete_composition_preset(self, name: str) -> dict:
        presets = self._tags.get("composition_presets", {})
        if name not in presets:
            return {"success": False, "error": "존재하지 않는 프리셋"}
        del presets[name]
        self.save_tags()
        return {"success": True}

    # ─── 부정 프리셋 ──────────────────────────────────────
    def get_negative_presets(self) -> dict:
        return copy.deepcopy(self._tags.get("negative_presets", {}))

    def save_negative_preset(self, name: str, tags: list[str]) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        self._tags.setdefault("negative_presets", {})[name.strip()] = list(tags)
        self.save_tags()
        return {"success": True}

    def delete_negative_preset(self, name: str) -> dict:
        presets = self._tags.get("negative_presets", {})
        if name not in presets:
            return {"success": False, "error": "존재하지 않는 프리셋"}
        del presets[name]
        self.save_tags()
        return {"success": True}

    # ─── 캐릭터 부정 태그 ──────────────────────────────────
    def add_character_negative_tag(self, value: str) -> dict:
        if value.strip() in [t.strip() for t in self._tags.get("character_negative", [])]:
            return {"success": False, "error": "이미 존재하는 태그"}
        self._tags.setdefault("character_negative", []).append(value)
        self.save_tags()
        return {"success": True}

    def remove_character_negative_tag(self, index: int) -> dict:
        tags = self._tags.get("character_negative", [])
        if index < 0 or index >= len(tags):
            return {"success": False, "error": "잘못된 인덱스"}
        tags.pop(index)
        self.save_tags()
        return {"success": True}

    def get_character_negative_presets(self) -> dict:
        return copy.deepcopy(self._tags.get("character_negative_presets", {}))

    def save_character_negative_preset(self, name: str, tags: list[str]) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        self._tags.setdefault("character_negative_presets", {})[name.strip()] = list(tags)
        self.save_tags()
        return {"success": True}

    def delete_character_negative_preset(self, name: str) -> dict:
        presets = self._tags.get("character_negative_presets", {})
        if name not in presets:
            return {"success": False, "error": "존재하지 않는 프리셋"}
        del presets[name]
        self.save_tags()
        return {"success": True}

    # ─── ANIMA 품질 태그 ──────────────────────────────────
    def add_anima_quality_tag(self, value: str) -> dict:
        if value.strip() in [t.strip() for t in self._tags.get("anima_quality", [])]:
            return {"success": False, "error": "이미 존재하는 태그"}
        self._tags.setdefault("anima_quality", []).append(value)
        self.save_tags()
        return {"success": True}

    def remove_anima_quality_tag(self, index: int) -> dict:
        tags = self._tags.get("anima_quality", [])
        if index < 0 or index >= len(tags):
            return {"success": False, "error": "잘못된 인덱스"}
        tags.pop(index)
        self.save_tags()
        return {"success": True}

    # ─── ANIMA 부정 태그 ──────────────────────────────────
    def add_anima_negative_tag(self, value: str) -> dict:
        if value.strip() in [t.strip() for t in self._tags.get("anima_negative", [])]:
            return {"success": False, "error": "이미 존재하는 태그"}
        self._tags.setdefault("anima_negative", []).append(value)
        self.save_tags()
        return {"success": True}

    def remove_anima_negative_tag(self, index: int) -> dict:
        tags = self._tags.get("anima_negative", [])
        if index < 0 or index >= len(tags):
            return {"success": False, "error": "잘못된 인덱스"}
        tags.pop(index)
        self.save_tags()
        return {"success": True}

    # ─── 캐릭터 관리 (조합 참조만) ─────────────────────────
    def add_character(self, name: str) -> dict:
        if name in self._tags["characters"]:
            return {"success": False, "error": "이미 존재하는 캐릭터"}
        self._tags["characters"][name] = {
            "appearance": "",
            "outfit": "",
            "expression": "",
        }
        self.save_tags()
        self._log("character_added", {"name": name})
        return {"success": True}

    def remove_character(self, name: str) -> dict:
        if name not in self._tags["characters"]:
            return {"success": False, "error": "존재하지 않는 캐릭터"}
        del self._tags["characters"][name]
        self.save_tags()
        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(name))
        if os.path.isdir(char_dir):
            shutil.rmtree(char_dir)
        # lora_manage.json에서도 해당 캐릭터 제거
        try:
            from modes.lora_mode import _load_lora_manage, _save_lora_manage
            lora_data = _load_lora_manage()
            if name in lora_data.get("loras", {}):
                del lora_data["loras"][name]
                _save_lora_manage(lora_data)
                print(f"[ASSET] lora_manage에서 캐릭터 제거: {name}")
        except Exception as e:
            print(f"[ASSET] lora_manage 캐릭터 제거 실패: {e}")
        self._log("character_removed", {"name": name})
        return {"success": True}

    def duplicate_character(self, source_name: str, new_name: str) -> dict:
        if source_name not in self._tags["characters"]:
            return {"success": False, "error": "존재하지 않는 캐릭터"}
        if new_name in self._tags["characters"]:
            return {"success": False, "error": "이미 존재하는 캐릭터"}
        src = self._tags["characters"][source_name]
        self._tags["characters"][new_name] = {
            "appearance": src.get("appearance", ""),
            "outfit": src.get("outfit", ""),
            "expression": src.get("expression", ""),
        }
        self.save_tags()
        # 에셋 폴더 전체 복사
        src_dir = os.path.join(ASSET_DIR, self._safe_dirname(source_name))
        new_dir = os.path.join(ASSET_DIR, self._safe_dirname(new_name))
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, new_dir)
        # 이름 매핑도 복사
        mapping = self._load_name_mapping()
        if source_name in mapping:
            mapping[new_name] = dict(mapping[source_name])
            self._save_name_mapping(mapping)
        self._log("character_duplicated", {"source": source_name, "new": new_name})
        return {"success": True}

    def update_character(self, name: str, appearance: str = "", outfit: str = "", expression: str = "") -> dict:
        if name not in self._tags["characters"]:
            return {"success": False, "error": "존재하지 않는 캐릭터"}
        self._tags["characters"][name]["appearance"] = appearance
        self._tags["characters"][name]["outfit"] = outfit
        self._tags["characters"][name]["expression"] = expression
        self.save_tags()
        return {"success": True}

    def list_characters(self) -> list[str]:
        return list(self._tags.get("characters", {}).keys())

    def get_characters_representative(self) -> dict[str, dict]:
        """각 캐릭터의 첫 번째 대표이미지 정보를 반환.
        {char_name: {outfit, expression, filename}} 또는 {}
        """
        result = {}
        for char_name in self.list_characters():
            char_dir = os.path.join(ASSET_DIR, self._safe_dirname(char_name))
            if not os.path.isdir(char_dir):
                continue
            for outfit_dir_name in sorted(os.listdir(char_dir)):
                outfit_path = os.path.join(char_dir, outfit_dir_name)
                if not os.path.isdir(outfit_path):
                    continue
                if outfit_dir_name == "Lora":
                    continue
                for expr_dir_name in sorted(os.listdir(outfit_path)):
                    expr_path = os.path.join(outfit_path, expr_dir_name)
                    if not os.path.isdir(expr_path):
                        continue
                    rep_path = os.path.join(expr_path, "_representative.json")
                    if os.path.isfile(rep_path):
                        try:
                            with open(rep_path, "r", encoding="utf-8") as f:
                                rep_file = json.load(f).get("filename", "")
                            if rep_file:
                                result[char_name] = {
                                    "outfit": outfit_dir_name,
                                    "expression": expr_dir_name,
                                    "filename": rep_file,
                                }
                                break
                        except Exception:
                            pass
                if char_name in result:
                    break
        return result

    # ─── 복장×표정 그룹 (Level 1) ────────────────────────────
    def _expr_group_path(self, character: str, outfit: str, expression: str) -> str:
        return os.path.join(ASSET_DIR, self._safe_dirname(character),
                            self._safe_dirname(outfit), self._safe_dirname(expression),
                            "_expr_group.json")

    def get_outfit_groups(self, character: str) -> dict:
        """캐릭터의 복장×표정 그룹을 {group_id: [{outfit, expression}, ...]} 반환."""
        groups: dict[str, list[dict]] = {}
        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        if not os.path.isdir(char_dir):
            return groups
        for outfit_dir in sorted(os.listdir(char_dir)):
            outfit_path = os.path.join(char_dir, outfit_dir)
            if not os.path.isdir(outfit_path):
                continue
            if outfit_dir == "Lora":
                continue
            for expr_dir in sorted(os.listdir(outfit_path)):
                expr_path = os.path.join(outfit_path, expr_dir)
                gfile = os.path.join(expr_path, "_expr_group.json")
                if not os.path.isfile(gfile):
                    continue
                try:
                    with open(gfile, "r", encoding="utf-8") as f:
                        gid = json.load(f).get("group_id")
                    if gid:
                        groups.setdefault(gid, []).append({"outfit": outfit_dir, "expression": expr_dir})
                except Exception:
                    pass
        return groups

    def set_outfit_group(self, character: str, src_outfit: str, src_expr: str,
                         tgt_outfit: str, tgt_expr: str) -> dict:
        """두 복장×표정 조합을 같은 그룹으로 묶기."""
        if not all([character, src_outfit, src_expr, tgt_outfit, tgt_expr]):
            return {"success": False, "error": "필드 누락"}

        src_gfile = self._expr_group_path(character, src_outfit, src_expr)
        tgt_gfile = self._expr_group_path(character, tgt_outfit, tgt_expr)

        # 기존 그룹 ID 읽기
        tgt_gid = None
        if os.path.isfile(tgt_gfile):
            try:
                with open(tgt_gfile, "r", encoding="utf-8") as f:
                    tgt_gid = json.load(f).get("group_id")
            except Exception:
                pass

        src_gid = None
        if os.path.isfile(src_gfile):
            try:
                with open(src_gfile, "r", encoding="utf-8") as f:
                    src_gid = json.load(f).get("group_id")
            except Exception:
                pass

        if tgt_gid and src_gid and tgt_gid == src_gid:
            return {"success": True, "message": "이미 같은 그룹"}

        final_gid = tgt_gid or str(uuid.uuid4())

        if not tgt_gid:
            os.makedirs(os.path.dirname(tgt_gfile), exist_ok=True)
            with open(tgt_gfile, "w", encoding="utf-8") as f:
                json.dump({"group_id": final_gid}, f, ensure_ascii=False, indent=2)

        os.makedirs(os.path.dirname(src_gfile), exist_ok=True)
        with open(src_gfile, "w", encoding="utf-8") as f:
            json.dump({"group_id": final_gid}, f, ensure_ascii=False, indent=2)

        # 기존 그룹 병합
        if src_gid and src_gid != final_gid:
            groups = self.get_outfit_groups(character)
            for m in groups.get(src_gid, []):
                mf = self._expr_group_path(character, m["outfit"], m["expression"])
                os.makedirs(os.path.dirname(mf), exist_ok=True)
                with open(mf, "w", encoding="utf-8") as f:
                    json.dump({"group_id": final_gid}, f, ensure_ascii=False, indent=2)

        return {"success": True}

    def ungroup_outfit(self, character: str, outfit: str, expression: str) -> dict:
        """복장×표정 조합을 그룹에서 제거."""
        if not all([character, outfit, expression]):
            return {"success": False, "error": "필드 누락"}

        gfile = self._expr_group_path(character, outfit, expression)
        if not os.path.isfile(gfile):
            return {"success": True, "message": "그룹 없음"}

        try:
            with open(gfile, "r", encoding="utf-8") as f:
                old_gid = json.load(f).get("group_id")
        except Exception:
            old_gid = None

        os.remove(gfile)

        return {"success": True}

    def ensure_upload_character(self):
        """업로드이미지 캐릭터가 tags.json에 없으면 자동 등록."""
        chars = self._tags.setdefault("characters", {})
        if "업로드이미지" not in chars:
            chars["업로드이미지"] = {"appearance": "", "outfit": "", "expression": ""}
            self.save_tags()

    def get_character(self, name: str) -> dict:
        return copy.deepcopy(self._tags.get("characters", {}).get(name, {}))

    # ─── 아티스트 프리셋 ─────────────────────────────────
    def save_artist_preset(self, name: str, tags: list) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        self._tags.setdefault("artist_presets", {})[name.strip()] = [t for t in tags if t.strip()]
        self.save_tags()
        return {"success": True}

    def delete_artist_preset(self, name: str) -> dict:
        presets = self._tags.get("artist_presets", {})
        if name not in presets:
            return {"success": False, "error": "존재하지 않는 프리셋"}
        del presets[name]
        self.save_tags()
        return {"success": True}

    def get_artist_presets(self) -> dict:
        return self._tags.get("artist_presets", {})

    # ─── 자연어 프리셋 ────────────────────────────────────
    def save_natural_language_preset(self, name: str, text: str) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        self._tags.setdefault("natural_language_presets", {})[name.strip()] = text.strip()
        self.save_tags()
        return {"success": True}

    def delete_natural_language_preset(self, name: str) -> dict:
        presets = self._tags.get("natural_language_presets", {})
        if name not in presets:
            return {"success": False, "error": "존재하지 않는 프리셋"}
        del presets[name]
        self.save_tags()
        return {"success": True}

    def get_natural_language_presets(self) -> dict:
        return self._tags.get("natural_language_presets", {})

    # ─── 범용 프리셋 태그 편집 ──────────────────────────────
    VALID_PRESET_TYPES = (
        "artist_presets", "quality_presets", "negative_presets",
        "composition_presets", "character_negative_presets",
    )

    def add_preset_tag(self, preset_type: str, preset_name: str, value: str) -> dict:
        if preset_type not in self.VALID_PRESET_TYPES:
            return {"success": False, "error": f"지원하지 않는 프리셋 타입: {preset_type}"}
        if not preset_name or not value.strip():
            return {"success": False, "error": "프리셋명과 태그값 필요"}
        presets = self._tags.get(preset_type, {})
        if preset_name not in presets:
            return {"success": False, "error": f"프리셋 '{preset_name}' 없음"}
        tags = presets[preset_name]
        if not isinstance(tags, list):
            return {"success": False, "error": "리스트 형태 프리셋만 지원"}
        tags.append(value.strip())
        self.save_tags()
        return {"success": True, "tags": list(tags)}

    def remove_preset_tag(self, preset_type: str, preset_name: str, index: int) -> dict:
        if preset_type not in self.VALID_PRESET_TYPES:
            return {"success": False, "error": f"지원하지 않는 프리셋 타입: {preset_type}"}
        if not preset_name:
            return {"success": False, "error": "프리셋명 필요"}
        presets = self._tags.get(preset_type, {})
        if preset_name not in presets:
            return {"success": False, "error": f"프리셋 '{preset_name}' 없음"}
        tags = presets[preset_name]
        if not isinstance(tags, list):
            return {"success": False, "error": "리스트 형태 프리셋만 지원"}
        if index < 0 or index >= len(tags):
            return {"success": False, "error": "잘못된 인덱스"}
        tags.pop(index)
        self.save_tags()
        return {"success": True, "tags": list(tags)}

    def reorder_preset_tags(self, preset_type: str, preset_name: str, order: list) -> dict:
        if preset_type not in self.VALID_PRESET_TYPES:
            return {"success": False, "error": f"지원하지 않는 프리셋 타입: {preset_type}"}
        if not preset_name:
            return {"success": False, "error": "프리셋명 필요"}
        presets = self._tags.get(preset_type, {})
        if preset_name not in presets:
            return {"success": False, "error": f"프리셋 '{preset_name}' 없음"}
        tags = presets[preset_name]
        if not isinstance(tags, list):
            return {"success": False, "error": "리스트 형태 프리셋만 지원"}
        if len(order) != len(tags):
            return {"success": False, "error": "순서 길이 불일치"}
        try:
            reordered = [tags[i] for i in order]
        except (IndexError, TypeError):
            return {"success": False, "error": "잘못된 순서"}
        presets[preset_name] = reordered
        self.save_tags()
        return {"success": True, "tags": list(reordered)}

    # ─── 프롬프트 빌드 ────────────────────────────────────
    def build_prompts(
        self,
        appearance: str = "",
        outfit: str = "",
        expression: str = "",
        face_id_enabled: bool = False,
        face_id_strength: float = 0.55,
        face_id_dir: str = "",
        style_ref_enabled: bool = False,
        style_ref_strength: float = 0.55,
        style_ref_dir: str = "",
        lora_activate: bool = False,
        lora_data: str = "",
        pose_enabled: bool = False,
        pose_data: dict = None,
        hrf_activate: bool = False,
        anima_hrf_activate: bool = False,
        hrf_size: float = 2.0,
        hrf_restore_size: bool = True,
        hrf_control_net: bool = False,
        img_w: int = 700,
        img_h: int = 1024,
        anima_fd_activate: bool = False,
        anima_hd_activate: bool = False,
        anima_ed_activate: bool = False,
        fd_activate: bool = False,
        hd_activate: bool = False,
        ed_activate: bool = False,
        face_lora_activate: bool = False,
        face_lora_data: str = "",
        style_lora_activate: bool = False,
        style_lora_data: str = "",
        face_crop_top: float = 2.5,
        face_crop_bottom: float = 1.0,
        char_face_tag_inform: str = "",
        seed: int = -1,
        artist_preset: str = "",
        natural_language: str = "",
        lora_trigger_words: str = "",
        anima_artist_preset: str = "",
        asset_workflow_type: str = workflow_profiles.ASSET_ILXL,
        anima_lora_trigger_words: str = "",
        sdxl_lora_trigger_words: str = "",
    ) -> tuple[str, str]:
        asset_workflow_type = workflow_profiles.normalize_asset_workflow_type(
            asset_workflow_type
        )
        capabilities = workflow_profiles.asset_capabilities(asset_workflow_type)
        is_anima = capabilities["anima"]
        is_dual = capabilities["dual"]
        q_tags = self._tags.get("quality", [])
        c_tags = self._tags.get("composition", [])
        app_tags = self._tags.get("appearances", {}).get(appearance, [])
        outfit_tags = self._tags.get("outfits", {}).get(outfit, [])
        expr_tags = self._tags.get("expressions", {}).get(expression, [])
        n_tags = self._tags.get("negative", [])
        cn_tags = self._tags.get("character_negative", [])
        artist_tags = self._tags.get("artist_presets", {}).get(artist_preset, [])
        anima_artist_tags = self._tags.get("artist_presets", {}).get(anima_artist_preset, [])
        anima_q_tags = self._tags.get("anima_quality", [])
        anima_n_tags = self._tags.get("anima_negative", [])

        if is_anima:
            # ANIMA 계열: ANIMA 블럭을 만들고, dual 프로필만 ILXL 블럭을 덧붙인다.
            anima_quality_parts = [t.strip() for t in anima_q_tags if t.strip()]
            anima_artist_parts = [t.strip() for t in anima_artist_tags if t.strip()]
            sdxl_quality_parts = [t.strip() for t in q_tags if t.strip()]
            sdxl_artist_parts = [t.strip() for t in artist_tags if t.strip()]

            section1_rest = []
            # ANIMA LoRA 트리거 워드
            if anima_lora_trigger_words.strip():
                section1_rest.append(anima_lora_trigger_words.strip())
            # ANIMA 아티스트 프리셋
            for t in anima_artist_tags:
                if t.strip():
                    section1_rest.append(t.strip())
            # ANIMA 품질 태그
            for t in anima_q_tags:
                if t.strip():
                    section1_rest.append(t.strip())
            # 공통 태그 (composition → appearance → expression → outfit)
            for t in c_tags:
                if t.strip():
                    section1_rest.append(t.strip())
            for t in app_tags:
                if t.strip():
                    section1_rest.append(t.strip())
            for t in expr_tags:
                if t.strip():
                    section1_rest.append(t.strip())
            for t in outfit_tags:
                if t.strip():
                    section1_rest.append(t.strip())
            # 자연어
            if natural_language.strip():
                section1_rest.append(natural_language.strip())

            section2_rest = []
            # SDXL LoRA 트리거 워드
            if sdxl_lora_trigger_words.strip():
                section2_rest.append(sdxl_lora_trigger_words.strip())
            # SDXL 아티스트 프리셋
            for t in artist_tags:
                if t.strip():
                    section2_rest.append(t.strip())
            # SDXL 품질 태그
            for t in q_tags:
                if t.strip():
                    section2_rest.append(t.strip())
            # 공통 태그
            for t in c_tags:
                if t.strip():
                    section2_rest.append(t.strip())
            for t in app_tags:
                if t.strip():
                    section2_rest.append(t.strip())
            for t in expr_tags:
                if t.strip():
                    section2_rest.append(t.strip())
            for t in outfit_tags:
                if t.strip():
                    section2_rest.append(t.strip())

            positive = "[ANIMA_QUALITY]\n" + ", ".join(anima_quality_parts)
            positive += "\n[ANIMA_ARTIST]\n" + ", ".join(anima_artist_parts)
            positive += "\n[ANIMA]\n" + ", ".join(section1_rest)
            if is_dual:
                positive += "\n[SDXL_QUALITY]\n" + ", ".join(sdxl_quality_parts)
                positive += "\n[SDXL_ARTIST]\n" + ", ".join(sdxl_artist_parts)
                positive += "\n[SDXL]"
                positive += "\n" + ", ".join(section2_rest)
            positive += "\n[CHAR_LIST]\nasset_mode"
        else:
            # ILXL 모드: 기존 평탄 프롬프트 유지
            positive_parts = []
            # 1. LoRA 트리거 워드
            if lora_trigger_words.strip():
                positive_parts.append(lora_trigger_words.strip())
            # 2. artist tags
            for t in artist_tags:
                if t.strip():
                    positive_parts.append(t.strip())
            # 3. quality tags
            for t in q_tags:
                if t.strip():
                    positive_parts.append(t.strip())
            # 4. composition tags
            for t in c_tags:
                if t.strip():
                    positive_parts.append(t.strip())
            # 5. appearance tags
            for t in app_tags:
                if t.strip():
                    positive_parts.append(t.strip())
            # 6. expression tags
            for t in expr_tags:
                if t.strip():
                    positive_parts.append(t.strip())
            # 7. outfit tags
            for t in outfit_tags:
                if t.strip():
                    positive_parts.append(t.strip())
            # 8. natural language text
            if natural_language.strip():
                positive_parts.append(natural_language.strip())

            positive = ", ".join(positive_parts)

        ipadapter_enabled = capabilities["ipadapter"]
        if not ipadapter_enabled and (face_id_enabled or style_ref_enabled):
            print(
                f"[ASSET] {asset_workflow_type}에서 지원하지 않는 IPAdapter 옵션 무시: "
                f"face_id={face_id_enabled}, style_ref={style_ref_enabled}"
            )
        positive += f"\n[FACE_ID_ACTIVATE]\n{'true' if face_id_enabled and ipadapter_enabled else 'false'}"
        positive += f"\n[FACE_ID_STR]\n{face_id_strength}"
        positive += f"\n[FACE_ID_DIR]\n{face_id_dir or 'soya_char_ref/fallback'}"
        positive += f"\n[STYLE_ACTIVATE]\n{'true' if style_ref_enabled and ipadapter_enabled else 'false'}"
        positive += f"\n[STYLE_STR]\n{style_ref_strength}"
        positive += f"\n[STYLE_DIR]\n{style_ref_dir or 'soya_style_ref/fallback'}"
        positive += f"\n[FACE_CROP_TOP]\n{face_crop_top}"
        positive += f"\n[FACE_CROP_BOTTOM]\n{face_crop_bottom}"
        positive += f"\n[LORA_ACTIVATE]\n{'true' if lora_activate else 'false'}"
        positive += f"\n[LORA_DATA]\n{lora_data or '{"list":[]}'}"
        effective_face_lora_activate = bool(
            face_lora_activate and capabilities["face_lora"]
        )
        if face_lora_activate and not capabilities["face_lora"]:
            print(f"[ASSET] {asset_workflow_type}에서 지원하지 않는 Face LoRA 옵션 무시")
        positive += (
            f"\n[FACE_LORA_ACTIVATE]\n"
            f"{'true' if effective_face_lora_activate else 'false'}"
        )
        # FACE_LORA_DATA에 CHAR 필드 추가
        if face_lora_data:
            try:
                flora_parsed = json.loads(face_lora_data)
                for item in flora_parsed.get("list", []):
                    item["CHAR"] = "asset_mode"
                positive += f"\n[FACE_LORA_DATA]\n{json.dumps(flora_parsed, ensure_ascii=False)}"
            except (json.JSONDecodeError, TypeError):
                positive += f"\n[FACE_LORA_DATA]\n{face_lora_data}"
        else:
            positive += f"\n[FACE_LORA_DATA]\n{'{\"list\":[]}'}"
        # Style(그림체) LoRA — ANIMA 모드에서만 별도 토큰(분리 로더) 출력.
        # ILXL 모드는 스타일 LoRA가 LORA_DATA에 흡수되므로 이 토큰들을 출력하지 않음.
        if is_anima:
            positive += f"\n[STYLE_LORA_ACTIVATE]\n{'true' if style_lora_activate else 'false'}"
            positive += f"\n[STYLE_LORA_DATA]\n{style_lora_data or '{"list":[]}'}"
        positive += f"\n[CHAR_FACE_TAG_INFORM]\n{char_face_tag_inform or '{"list":[]}'}"
        effective_pose_enabled = bool(pose_enabled and capabilities["pose"])
        if pose_enabled and not capabilities["pose"]:
            print(f"[ASSET] {asset_workflow_type}에서 지원하지 않는 포즈 옵션 무시")
        positive += f"\n[POSE_ACTIVATE]\n{'true' if effective_pose_enabled else 'false'}"
        if effective_pose_enabled and pose_data:
            positive += f"\n[POSE_DATA]\n{json.dumps(pose_data, ensure_ascii=False)}"
        else:
            positive += f"\n[POSE_DATA]\n{json.dumps(DEFAULT_POSE_DATA, ensure_ascii=False)}"
        positive += f"\n[HRF_ACTIVATE]\n{'true' if hrf_activate and capabilities['ilxl'] else 'false'}"
        if is_anima:
            positive += f"\n[ANIMA_HRF_ACTIVATE]\n{'true' if anima_hrf_activate else 'false'}"
        positive += f"\n[HRF_SIZE]\n{hrf_size}"
        positive += f"\n[HRF_RESTORE_SIZE]\n{'true' if hrf_restore_size else 'false'}"
        positive += f"\n[HRF_CONTROL_NET]\n{'true' if hrf_control_net else 'false'}"
        positive += f"\n[IMG_W]\n{img_w}"
        positive += f"\n[IMG_H]\n{img_h}"
        positive += f"\n[ANIMA_FD_ACTIVATE]\n{'true' if anima_fd_activate and is_anima else 'false'}"
        positive += f"\n[ANIMA_HD_ACTIVATE]\n{'true' if anima_hd_activate and is_anima else 'false'}"
        positive += f"\n[ANIMA_ED_ACTIVATE]\n{'true' if anima_ed_activate and is_anima else 'false'}"
        positive += f"\n[FD_ACTIVATE]\n{'true' if fd_activate and capabilities['ilxl'] else 'false'}"
        positive += f"\n[HD_ACTIVATE]\n{'true' if hd_activate and capabilities['ilxl'] else 'false'}"
        positive += f"\n[ED_ACTIVATE]\n{'true' if ed_activate and capabilities['ilxl'] else 'false'}"
        positive += f"\n[SEED]\n{seed if capabilities['seed'] else -1}"
        positive += "\n[END]"

        if is_anima:
            anima_neg_parts = [t.strip() for t in cn_tags if t.strip()] + [t.strip() for t in anima_n_tags if t.strip()]
            negative = ", ".join(anima_neg_parts)
            if is_dual:
                sdxl_neg_parts = [t.strip() for t in cn_tags if t.strip()] + [t.strip() for t in n_tags if t.strip()]
                negative += "\n[SDXL]\n" + ", ".join(sdxl_neg_parts)
        else:
            negative_parts = [t.strip() for t in cn_tags if t.strip()] + [t.strip() for t in n_tags if t.strip()]
            negative = ", ".join(negative_parts)
        return positive, negative

    # ─── 워크플로우 관리 ──────────────────────────────────
    def _compute_file_hash(self, filepath: str) -> str:
        if self.compute_hash_func:
            return self.compute_hash_func(filepath)
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _is_api_format(self, wf: dict) -> bool:
        return isinstance(wf, dict) and any(
            isinstance(v, dict) and "class_type" in v for v in wf.values()
        )

    def _load_stored_hash(self) -> str:
        hash_path = os.path.join(CURRENT_MODE_WORK_DIR, "asset_hash.txt")
        if os.path.isfile(hash_path):
            try:
                with open(hash_path, "r") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    def _save_stored_hash(self, h: str):
        os.makedirs(CURRENT_MODE_WORK_DIR, exist_ok=True)
        with open(os.path.join(CURRENT_MODE_WORK_DIR, "asset_hash.txt"), "w") as f:
            f.write(h)

    def _save_cached_api(self, wf: dict):
        try:
            os.makedirs(CURRENT_MODE_WORK_DIR, exist_ok=True)
            with open(os.path.join(CURRENT_MODE_WORK_DIR, "asset_api.json"), "w", encoding="utf-8") as f:
                json.dump(wf, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    async def _fallback_load_workflow(self) -> bool:
        """workflow_source_path 실패 시 mode_workflow 폴더에서 같은 이름의 워크플로우를 찾아 로드한다."""
        if not os.path.isdir(MODE_WORKFLOW_DIR):
            print("[ASSET][FALLBACK] mode_workflow 폴더가 없음")
            self._log("fallback_no_dir", {"dir": MODE_WORKFLOW_DIR})
            return False

        src = self.workflow_source_path or ""
        fname = os.path.basename(src) if src else ""
        fpath = os.path.join(MODE_WORKFLOW_DIR, fname) if fname else ""

        if not fname or not os.path.isfile(fpath):
            print(f"[ASSET][FALLBACK] mode_workflow에 같은 이름 파일 없음: '{fname or '(이름 없음)'}'")
            self._log("fallback_no_match", {"name": fname})
            return False

        print(f"[ASSET][FALLBACK] mode_workflow에서 '{fname}' 로드 시도")
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                wf_data = json.load(f)
        except Exception as e:
            print(f"[ASSET][FALLBACK] '{fname}' 로드 실패: {e}")
            self._log("fallback_load_error", {"name": fname, "error": str(e)})
            return False

        if self._is_api_format(wf_data):
            self._asset_api_workflow = wf_data
            self._save_cached_api(wf_data)
            print(f"[ASSET][FALLBACK] '{fname}' API 포맷 로드 성공 ({len(wf_data)}개 노드)")
            self._log("fallback_loaded_api", {"file": fname, "nodes": len(wf_data)})
            return True

        if self.convert_workflow_func:
            try:
                api_wf, error = await self.convert_workflow_func(wf_data)
                if api_wf is not None:
                    self._asset_api_workflow = api_wf
                    self._save_cached_api(api_wf)
                    print(f"[ASSET][FALLBACK] '{fname}' 변환 성공 ({len(api_wf)}개 노드)")
                    self._log("fallback_converted", {"file": fname, "nodes": len(api_wf)})
                    return True
                else:
                    print(f"[ASSET][FALLBACK] '{fname}' 변환 실패: {error}")
            except Exception as e:
                print(f"[ASSET][FALLBACK] '{fname}' 변환 예외: {e}")
        else:
            print(f"[ASSET][FALLBACK] '{fname}' - API 포맷이 아니고 변환 함수도 없음")

        self._log("fallback_failed", {"name": fname})
        return False

    async def update_asset_workflow(self) -> bool:
        src = self.workflow_source_path
        if not src or not os.path.isfile(src):
            reason = "경로 미설정" if not src else f"파일 없음: '{src}'"
            print(f"[ASSET] 워크플로우 소스 {reason} → mode_workflow 폴더에서 폴백 탐색")
            self._log("workflow_skip", {"reason": "no_source", "path": src or ""})
            return await self._fallback_load_workflow()

        try:
            file_hash = self._compute_file_hash(src)
        except Exception as e:
            print(f"[ASSET] 워크플로우 해시 계산 실패: {e} → mode_workflow 폴더에서 폴백 탐색")
            self._log("workflow_hash_error", {"error": str(e)})
            return False

        stored_hash = self._load_stored_hash()
        cache_api_path = os.path.join(CURRENT_MODE_WORK_DIR, "asset_api.json")

        if file_hash == stored_hash and self._asset_api_workflow is not None:
            self._log("workflow_cache_hit", {"hash": file_hash[:12]})
            return True

        if file_hash == stored_hash and os.path.exists(cache_api_path):
            try:
                with open(cache_api_path, "r", encoding="utf-8") as f:
                    self._asset_api_workflow = json.load(f)
                if self._asset_api_workflow:
                    self._asset_hash = file_hash
                    self._log("workflow_cache_loaded_from_disk", {"nodes": len(self._asset_api_workflow)})
                    return True
            except Exception:
                pass

        os.makedirs(MODE_WORKFLOW_DIR, exist_ok=True)
        dest = os.path.join(MODE_WORKFLOW_DIR, os.path.basename(src))
        shutil.copy2(src, dest)
        self._log("workflow_copied", {"src": src, "dest": dest})

        try:
            with open(dest, "r", encoding="utf-8") as f:
                wf_data = json.load(f)
        except Exception as e:
            print(f"[ASSET] 워크플로우 로드 실패: {e} → mode_workflow 폴더에서 폴백 탐색")
            self._log("workflow_load_error", {"error": str(e)})
            return await self._fallback_load_workflow()

        if self._is_api_format(wf_data):
            self._asset_api_workflow = wf_data
            self._asset_hash = file_hash
            self._save_stored_hash(file_hash)
            self._save_cached_api(wf_data)
            self._log("workflow_loaded_api", {"nodes": len(wf_data)})
            return True

        if self.convert_workflow_func:
            api_wf, error = await self.convert_workflow_func(wf_data)
            if api_wf is None:
                print(f"[ASSET] 워크플로우 변환 실패: {error} → mode_workflow 폴더에서 폴백 탐색")
                self._log("workflow_convert_error", {"error": str(error)})
                return await self._fallback_load_workflow()
            self._asset_api_workflow = api_wf
            self._asset_hash = file_hash
            self._save_stored_hash(file_hash)
            self._save_cached_api(api_wf)
            self._log("workflow_converted", {"nodes": len(api_wf)})
            return True

        self._log("workflow_no_converter", {})
        print("[ASSET] 워크플로우 변환 함수 없음 → mode_workflow 폴더에서 폴백 탐색")
        return await self._fallback_load_workflow()

    # ─── 이미지 생성 ──────────────────────────────────────
    async def generate(
        self,
        character: str,
        appearance: str = "",
        outfit: str = "",
        expression: str = "",
        face_id_enabled: bool = False,
        face_id_strength: float = 0.55,
        reference_subfolder: str = "",
        style_ref_enabled: bool = False,
        style_ref_strength: float = 0.55,
        style_ref_subfolder: str = "",
        lora_activate: bool = False,
        lora_data: str = "",
        pose_enabled: bool = False,
        pose_id: str = "",
        hrf_activate: bool = False,
        anima_hrf_activate: bool = False,
        hrf_size: float = 2.0,
        hrf_restore_size: bool = True,
        hrf_control_net: bool = False,
        img_w: int = 700,
        img_h: int = 1024,
        fd_activate: bool = False,
        hd_activate: bool = False,
        ed_activate: bool = False,
        artist_preset: str = "",
        natural_language: str = "",
        lora_trigger_words: str = "",
        anima_artist_preset: str = "",
        asset_workflow_type: str = workflow_profiles.ASSET_ILXL,
        anima_lora_trigger_words: str = "",
        sdxl_lora_trigger_words: str = "",
        positive_prompt: str = None,
        negative_prompt: str = None,
        style_lora_activate: bool = False,
        style_lora_data: str = "",
        storage_group: str = "",
        storage_session: str = "",
    ) -> dict:
        async with self._lock:
            self._is_generating = True
            try:
                return await self._generate_internal(
                    character, appearance, outfit, expression,
                    face_id_enabled, face_id_strength, reference_subfolder,
                    style_ref_enabled, style_ref_strength, style_ref_subfolder,
                    lora_activate, lora_data,
                    pose_enabled, pose_id,
                    hrf_activate, anima_hrf_activate, hrf_size, hrf_restore_size, hrf_control_net, img_w, img_h,
                    fd_activate, hd_activate, ed_activate,
                    artist_preset, natural_language, lora_trigger_words,
                    anima_artist_preset, asset_workflow_type,
                    anima_lora_trigger_words, sdxl_lora_trigger_words,
                    positive_prompt, negative_prompt,
                    style_lora_activate, style_lora_data,
                    storage_group, storage_session,
                )
            finally:
                self._is_generating = False

    async def _generate_internal(
        self,
        character: str,
        appearance: str,
        outfit: str,
        expression: str,
        face_id_enabled: bool,
        face_id_strength: float,
        reference_subfolder: str,
        style_ref_enabled: bool,
        style_ref_strength: float,
        style_ref_subfolder: str,
        lora_activate: bool,
        lora_data: str,
        pose_enabled: bool,
        pose_id: str,
        hrf_activate: bool,
        anima_hrf_activate: bool,
        hrf_size: float,
        hrf_restore_size: bool,
        hrf_control_net: bool,
        img_w: int,
        img_h: int,
        fd_activate: bool,
        hd_activate: bool,
        ed_activate: bool,
        artist_preset: str,
        natural_language: str,
        lora_trigger_words: str,
        anima_artist_preset: str = "",
        asset_workflow_type: str = workflow_profiles.ASSET_ILXL,
        anima_lora_trigger_words: str = "",
        sdxl_lora_trigger_words: str = "",
        positive_prompt: str = None,
        negative_prompt: str = None,
        style_lora_activate: bool = False,
        style_lora_data: str = "",
        storage_group: str = "",
        storage_session: str = "",
    ) -> dict:
        if storage_group not in ("", "automatch_defaults", "character_maker"):
            error_msg = f"지원하지 않는 에셋 저장 분류: {storage_group}"
            print(f"[ASSET] {error_msg}")
            return {"success": False, "error": error_msg}
        if storage_group == "character_maker" and str(
            storage_session or ""
        ) != CHARACTER_MAKER_SINGLE_SESSION_ID:
            error_msg = (
                f"캐릭터 메이커 단일 영속 세션 ID가 유효하지 않음: {storage_session!r} "
                f"(예상: {CHARACTER_MAKER_SINGLE_SESSION_ID!r})"
            )
            print(f"[ASSET] {error_msg}")
            return {"success": False, "error": error_msg}
        storage_outfit = (
            AUTOMATCH_DEFAULT_OUTFIT_DIR
            if storage_group == "automatch_defaults"
            else outfit
        )

        asset_workflow_type = workflow_profiles.normalize_asset_workflow_type(
            asset_workflow_type
        )
        # 선택 에셋 프로필에 맞는 워크플로우 경로로 교체
        saved_workflow_path = self.workflow_source_path
        selected_workflow_path = {
            workflow_profiles.ASSET_ILXL: self.workflow_source_path,
            workflow_profiles.ASSET_ANIMA_ILXL: self.anima_workflow_source_path,
            workflow_profiles.ASSET_ANIMA_ONLY: self.anima_only_workflow_source_path,
        }[asset_workflow_type]
        if not selected_workflow_path and asset_workflow_type != workflow_profiles.ASSET_ILXL:
            error_msg = f"{asset_workflow_type} 에셋 워크플로우 소스 경로가 비어 있음"
            print(f"[ASSET] {error_msg}")
            return {"success": False, "error": error_msg}
        if selected_workflow_path != self.workflow_source_path:
            self.workflow_source_path = selected_workflow_path
            self._asset_api_workflow = None
            self._asset_hash = ""
        try:
            ok = await self.update_asset_workflow()
            if not ok:
                error_msg = "워크플로우 준비 실패 (소스 경로 및 mode_workflow 폴더 모두 탐색 실패)"
                print(f"[ASSET] {error_msg}")
                return {"success": False, "error": error_msg}

            pose_data = None
            if pose_enabled and pose_id:
                pose_data = self._load_pose_data(pose_id)

            if positive_prompt is not None and negative_prompt is not None:
                positive, negative = positive_prompt, negative_prompt
                print(f"[ASSET] 프론트엔드 pre-built 프롬프트 사용 (길이: {len(positive)})")
            else:
                positive, negative = self.build_prompts(
                    appearance, outfit, expression,
                    face_id_enabled=face_id_enabled,
                    face_id_strength=face_id_strength,
                    face_id_dir=reference_subfolder,
                    style_ref_enabled=style_ref_enabled,
                    style_ref_strength=style_ref_strength,
                    style_ref_dir=style_ref_subfolder,
                    lora_activate=lora_activate,
                    lora_data=lora_data,
                    pose_enabled=pose_enabled,
                    pose_data=pose_data,
                    hrf_activate=hrf_activate,
                    anima_hrf_activate=anima_hrf_activate,
                    hrf_size=hrf_size,
                    hrf_restore_size=hrf_restore_size,
                    hrf_control_net=hrf_control_net,
                    img_w=img_w,
                    img_h=img_h,
                    fd_activate=fd_activate,
                    hd_activate=hd_activate,
                    ed_activate=ed_activate,
                    artist_preset=artist_preset,
                    natural_language=natural_language,
                    lora_trigger_words=lora_trigger_words,
                    anima_artist_preset=anima_artist_preset,
                    asset_workflow_type=asset_workflow_type,
                    anima_lora_trigger_words=anima_lora_trigger_words,
                    sdxl_lora_trigger_words=sdxl_lora_trigger_words,
                    style_lora_activate=style_lora_activate,
                    style_lora_data=style_lora_data,
                )
            if not positive:
                return {"success": False, "error": "프롬프트가 비어있음"}

            # seed=-1이면 매 생성마다 랜덤값으로 치환
            def _replace_seed(m):
                import random
                return "[SEED]\n" + str(random.randint(0, 2**32 - 1))
            positive = re.sub(r'\[SEED\]\n-1(?!\d)', _replace_seed, positive)

            self._log("generate_start", {
                "character": character, "outfit": outfit, "expression": expression,
                "positive_preview": positive[:100],
            })

            if self.notify_frontend_func:
                await self.notify_frontend_func("asset_generation_started", {
                    "character": character, "outfit": outfit, "expression": expression,
                })

            if self.build_prompt_with_workflow_func:
                workflow = self.build_prompt_with_workflow_func(
                    self._asset_api_workflow, positive, negative,
                )
            else:
                workflow = copy.deepcopy(self._asset_api_workflow)
                for nid, ninfo in workflow.items():
                    if not isinstance(ninfo, dict):
                        continue
                    title = ninfo.get("_meta", {}).get("title", "")
                    if title == "긍정프롬프트":
                        ninfo["inputs"]["value"] = positive
                    elif title == "부정프롬프트":
                        ninfo["inputs"]["value"] = negative

            final_positive = positive
            final_negative = negative
            for nid, ninfo in workflow.items():
                if not isinstance(ninfo, dict):
                    continue
                title = ninfo.get("_meta", {}).get("title", "")
                if title == "긍정프롬프트":
                    final_positive = ninfo.get("inputs", {}).get("value", positive)
                elif title == "부정프롬프트":
                    final_negative = ninfo.get("inputs", {}).get("value", negative)

            # 프롬프트 주입된 workflow를 asset_api.json에도 저장
            self._save_cached_api(workflow)

            if self.submit_workflow_func:
                async def _on_progress(value, max_value):
                    if self.notify_frontend_func:
                        await self.notify_frontend_func("asset_generation_progress", {
                            "value": value, "max": max_value,
                            "character": character, "outfit": outfit, "expression": expression,
                        })

                img_bytes, error = await self.submit_workflow_func(workflow, progress_callback=_on_progress)
            else:
                return {"success": False, "error": "submit_workflow_func 미설정"}

            if not img_bytes:
                error_msg = error if isinstance(error, str) else "이미지 생성 실패"
                print(f"[ASSET] 에셋 생성 실패 - 캐릭터: {character}, 복장: {outfit}, 표정: {expression}")
                print(f"[ASSET] 실패 사유: {error_msg}")
                if isinstance(error, dict):
                    print(f"[ASSET] 상세 에러: {json.dumps(error, ensure_ascii=False, indent=2)}")
                self._log("generate_failed", {"error": error_msg})
                if self.notify_frontend_func:
                    await self.notify_frontend_func("asset_generation_completed", {
                        "status": "error", "error": error_msg,
                        "character": character, "outfit": outfit, "expression": expression,
                    })
                return {"success": False, "error": error_msg}

            if storage_group == "character_maker":
                save_dir = os.path.join(
                    CHARACTER_MAKER_TEMP_DIR,
                    storage_session,
                    "images",
                )
            else:
                save_dir = os.path.join(
                    ASSET_DIR,
                    self._safe_dirname(character),
                    self._safe_dirname(storage_outfit),
                    self._safe_dirname(expression),
                )
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.webp"
            filepath = os.path.join(save_dir, filename)

            try:
                from PIL import Image
                from io import BytesIO
                img = Image.open(BytesIO(img_bytes))
                save_img = img if img.mode == "RGBA" else img.convert("RGB")
                save_img.save(filepath, format="WEBP", quality=90, method=4)
            except Exception as webp_error:
                print(
                    f"[ASSET] WEBP 저장 실패, PNG 원본 저장 시도: "
                    f"path={filepath}, error={type(webp_error).__name__}: {webp_error}"
                )
                traceback.print_exc()
                filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
                filepath = os.path.join(save_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

            prompt_record_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_prompt.json")
            try:
                with open(prompt_record_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "positive": final_positive,
                        "negative": final_negative,
                        "character": character,
                        "appearance": appearance,
                        "outfit": outfit,
                        "expression": expression,
                        "storage_group": storage_group,
                        "storage_outfit": storage_outfit,
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[ASSET] 프롬프트 기록 저장 실패: {prompt_record_path} ({e})")
                traceback.print_exc()
                if storage_group == "character_maker":
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                    except Exception as cleanup_error:
                        print(
                            f"[ASSET] 캐릭터 메이커 실패 이미지 정리 실패: "
                            f"path={filepath}, error={cleanup_error}"
                        )
                        traceback.print_exc()
                    return {
                        "success": False,
                        "error": "캐릭터 메이커 임시 프롬프트 기록 저장 실패",
                    }

            self._log("generate_saved", {
                "character": character, "outfit": outfit, "expression": expression,
                "filename": filename, "size": len(img_bytes),
            })

            if self.notify_frontend_func:
                await self.notify_frontend_func("asset_generation_completed", {
                    "status": "success",
                    "character": character, "outfit": outfit, "expression": expression,
                    "filename": filename,
                    "storage_group": storage_group,
                    "storage_outfit": storage_outfit,
                })

            result = {
                "success": True,
                "filename": filename,
                "character": character,
                "outfit": outfit,
                "expression": expression,
                "storage_group": storage_group,
                "storage_outfit": storage_outfit,
            }
            if storage_group == "character_maker":
                result["local_path"] = filepath
                result["prompt_record_path"] = prompt_record_path
            return result
        finally:
            # 임시 선택 워크플로우 경로 복원
            if self.workflow_source_path != saved_workflow_path:
                self.workflow_source_path = saved_workflow_path

    # ─── 폴더/이미지 관리 ─────────────────────────────────
    @staticmethod
    def _load_pose_data(pose_id: str) -> Optional[dict]:
        pose_dir = os.path.join(BASE_DIR, "pose_data")
        json_path = os.path.join(pose_dir, f"{pose_id}.json")
        if not os.path.isfile(json_path):
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _safe_dirname(name: str) -> str:
        safe = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-', '.')).strip()
        return safe or f"unknown_{hash(name) % 10000}"

    def list_images(self, character: str, outfit: str, expression: str) -> dict:
        img_dir = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
            self._safe_dirname(expression),
        )
        if not os.path.isdir(img_dir):
            return {"images": [], "representative": ""}

        rep_path = os.path.join(img_dir, "_representative.json")
        representative = ""
        if os.path.isfile(rep_path):
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    representative = json.load(f).get("filename", "")
            except Exception:
                pass

        images = []
        for fname in sorted(os.listdir(img_dir)):
            if fname.startswith("_"):
                continue
            fpath = os.path.join(img_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            prompt_data = {}
            prompt_path = os.path.join(img_dir, f"{os.path.splitext(fname)[0]}_prompt.json")
            if os.path.isfile(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as pf:
                        prompt_data = json.load(pf)
                except Exception as e:
                    print(
                        f"[ASSET_MODE] 이미지 프롬프트 로드 실패: "
                        f"path={prompt_path!r}, error={type(e).__name__}: {e}"
                    )
                    traceback.print_exc()

            images.append({
                "filename": fname,
                "is_representative": fname == representative,
                "has_prompt": bool(prompt_data),
                "positive": prompt_data.get("positive", ""),
                "negative": prompt_data.get("negative", ""),
                "prompt_character": prompt_data.get("character", ""),
                "prompt_appearance": prompt_data.get("appearance", ""),
                "prompt_outfit": prompt_data.get("outfit", ""),
                "prompt_expression": prompt_data.get("expression", ""),
                "is_edited": bool(prompt_data.get("is_edited", False)),
                "edit_prompt": prompt_data.get("edit_prompt", ""),
                "edit_prompt_original": prompt_data.get("edit_prompt_original", ""),
                "edit_negative_prompt": prompt_data.get("edit_negative_prompt", ""),
                "edit_source_filename": prompt_data.get("edit_source_filename", ""),
                "edit_model": prompt_data.get("edit_model", ""),
                "edited_at": prompt_data.get("edited_at", ""),
                "local_path": fpath,
            })
        return {"images": images, "representative": representative}

    @staticmethod
    def _preferred_image_filename(image_listing: dict) -> str:
        images = image_listing.get("images", []) if isinstance(image_listing, dict) else []
        filenames = [item.get("filename", "") for item in images if item.get("filename")]
        representative = image_listing.get("representative", "") if isinstance(image_listing, dict) else ""
        if representative and representative in filenames:
            return representative
        return filenames[0] if filenames else ""

    def list_automatch_compare_images(
        self,
        character: str,
        outfit: str,
        include_existing: bool = False,
    ) -> dict:
        """오토매치 비교 이미지의 명시적 우선순위를 반환한다.

        선택 복장의 ILXL 에셋 대표 이미지를 우선한다. ``include_existing``이
        활성화되면 외모/복장 프리셋과 무관하게 같은 캐릭터의 다른 일반 복장을
        표정 폴더명으로 탐색하고, 그래도 없을 때만 ``_automatch_defaults``
        분류 이미지를 사용한다.
        """
        if not character:
            print("[AUTOMATCH] 비교 이미지 조회 실패: character가 비어있음")
            return {
                "success": False,
                "error": "character 필수",
                "images": {},
                "default_outfit": AUTOMATCH_DEFAULT_OUTFIT_DIR,
            }

        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        other_outfits = []
        if include_existing:
            if os.path.isdir(char_dir):
                selected_outfit_dir = self._safe_dirname(outfit) if outfit else ""
                other_outfits = [
                    dirname
                    for dirname in sorted(os.listdir(char_dir))
                    if dirname not in ("Lora", AUTOMATCH_DEFAULT_OUTFIT_DIR, selected_outfit_dir)
                    and os.path.isdir(os.path.join(char_dir, dirname))
                ]
            else:
                print(
                    f"[AUTOMATCH] 기존 에셋 탐색 결과 없음: "
                    f"character={character!r}, 캐릭터 폴더가 존재하지 않음"
                )

        results = {}
        for expression in self.list_expressions():
            direct_filename = ""
            if outfit and outfit != AUTOMATCH_DEFAULT_OUTFIT_DIR:
                direct_listing = self.list_images(character, outfit, expression)
                direct_filename = self._preferred_image_filename(direct_listing)

            if direct_filename:
                results[expression] = {
                    "source": "direct",
                    "outfit": outfit,
                    "expression": expression,
                    "filename": direct_filename,
                }
                continue

            if include_existing:
                existing_match = None
                for existing_outfit in other_outfits:
                    existing_listing = self.list_images(
                        character,
                        existing_outfit,
                        expression,
                    )
                    existing_filename = self._preferred_image_filename(existing_listing)
                    if existing_filename:
                        existing_match = {
                            "source": "existing_asset",
                            "outfit": existing_outfit,
                            "expression": expression,
                            "filename": existing_filename,
                        }
                        break
                if existing_match:
                    results[expression] = existing_match
                    continue

            default_listing = self.list_images(
                character,
                AUTOMATCH_DEFAULT_OUTFIT_DIR,
                expression,
            )
            default_filename = self._preferred_image_filename(default_listing)
            if default_filename:
                results[expression] = {
                    "source": "generated_default",
                    "outfit": AUTOMATCH_DEFAULT_OUTFIT_DIR,
                    "expression": expression,
                    "filename": default_filename,
                }

        existing_count = sum(
            1 for item in results.values() if item.get("source") == "existing_asset"
        )
        if include_existing:
            print(
                f"[AUTOMATCH] 기존 에셋 탐색 완료: character={character!r}, "
                f"후보복장={len(other_outfits)}, 표정매칭={existing_count}"
            )

        return {
            "success": True,
            "character": character,
            "outfit": outfit,
            "include_existing": include_existing,
            "default_outfit": AUTOMATCH_DEFAULT_OUTFIT_DIR,
            "images": results,
        }

    def set_representative(self, character: str, outfit: str, expression: str, filename: str) -> dict:
        img_dir = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
            self._safe_dirname(expression),
        )
        os.makedirs(img_dir, exist_ok=True)

        rep_path = os.path.join(img_dir, "_representative.json")

        if not filename:
            if os.path.isfile(rep_path):
                os.remove(rep_path)
            return {"success": True, "action": "unset"}

        current_rep = ""
        if os.path.isfile(rep_path):
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    current_rep = json.load(f).get("filename", "")
            except Exception:
                pass

        if current_rep == filename:
            os.remove(rep_path)
            return {"success": True, "action": "unset"}

        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump({"filename": filename}, f, ensure_ascii=False)
        return {"success": True, "action": "set"}

    def get_image_path(self, character: str, outfit: str, expression: str, filename: str) -> Optional[str]:
        path = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
            self._safe_dirname(expression),
            filename,
        )
        if os.path.isfile(path):
            return path
        return None

    def delete_combination(self, character: str, outfit: str, expression: str) -> dict:
        """복장×표정 조합 전체(모든 이미지)를 삭제."""
        img_dir = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
            self._safe_dirname(expression),
        )
        if not os.path.isdir(img_dir):
            return {"success": False, "error": "존재하지 않는 조합"}
        shutil.rmtree(img_dir)
        self._log("combination_deleted", {
            "character": character, "outfit": outfit, "expression": expression,
        })
        return {"success": True}

    def delete_image(self, character: str, outfit: str, expression: str, filename: str) -> dict:
        img_dir = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
            self._safe_dirname(expression),
        )
        fpath = os.path.join(img_dir, filename)
        if not os.path.isfile(fpath):
            return {"success": False, "error": "파일이 존재하지 않음"}

        os.remove(fpath)

        # 프롬프트 JSON 삭제
        base, _ = os.path.splitext(filename)
        prompt_path = os.path.join(img_dir, f"{base}_prompt.json")
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)

        rep_path = os.path.join(img_dir, "_representative.json")
        if os.path.isfile(rep_path):
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    if json.load(f).get("filename") == filename:
                        os.remove(rep_path)
            except Exception:
                pass
        return {"success": True}

    def upload_image(self, character: str, outfit: str, expression: str,
                     filename: str, image_data: bytes) -> dict:
        """외부 이미지를 에셋 폴더에 저장."""
        import re
        img_dir = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
            self._safe_dirname(expression),
        )
        os.makedirs(img_dir, exist_ok=True)

        # 안전한 파일명 생성
        safe_name = os.path.splitext(filename)[0]
        safe_name = re.sub(r'[^\w\s\-\.]', '', safe_name).strip() or "upload"
        ext = os.path.splitext(filename)[1].lower() or ".png"
        safe_filename = f"{safe_name}{ext}"

        # 중복 시 숫자 추가
        counter = 1
        final_path = os.path.join(img_dir, safe_filename)
        while os.path.exists(final_path):
            final_path = os.path.join(img_dir, f"{safe_name}_{counter}{ext}")
            counter += 1

        with open(final_path, "wb") as f:
            f.write(image_data)

        # 업로드 이미지용 빈 프롬프트 JSON 생성
        prompt_path = os.path.join(img_dir, f"{os.path.splitext(os.path.basename(final_path))[0]}_prompt.json")
        try:
            with open(prompt_path, "w", encoding="utf-8") as pf:
                json.dump({
                    "positive": "",
                    "negative": "",
                    "character": character,
                    "appearance": "",
                    "outfit": outfit,
                    "expression": expression,
                }, pf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ASSET_MODE] 프롬프트 JSON 생성 실패: {e}")

        print(f"[ASSET_MODE] 이미지 업로드: {final_path}")
        return {"success": True, "filename": os.path.basename(final_path)}

    def list_character_gallery(self, character: str) -> list[dict]:
        """캐릭터 폴더의 실제 복장×표정 조합을 스캔하여 반환."""
        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        if not os.path.isdir(char_dir):
            return []

        results = []
        for outfit_dir_name in sorted(os.listdir(char_dir)):
            outfit_path = os.path.join(char_dir, outfit_dir_name)
            if not os.path.isdir(outfit_path):
                continue
            if outfit_dir_name == "Lora":
                continue
            for expr_dir_name in sorted(os.listdir(outfit_path)):
                expr_path = os.path.join(outfit_path, expr_dir_name)
                if not os.path.isdir(expr_path):
                    continue

                rep_file = ""
                rep_path = os.path.join(expr_path, "_representative.json")
                if os.path.isfile(rep_path):
                    try:
                        with open(rep_path, "r", encoding="utf-8") as f:
                            rep_file = json.load(f).get("filename", "")
                    except Exception:
                        pass

                image_count = 0
                for fname in os.listdir(expr_path):
                    if fname.startswith("_"):
                        continue
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in (".png", ".jpg", ".jpeg", ".webp"):
                        image_count += 1

                if image_count > 0:
                    local_path = self.get_image_path(character, outfit_dir_name, expr_dir_name, rep_file) if rep_file else ""
                    results.append({
                        "outfit": outfit_dir_name,
                        "expression": expr_dir_name,
                        "representative": rep_file,
                        "image_count": image_count,
                        "local_path": local_path,
                    })
        return results

    def batch_analyze_representatives(self, character: str) -> list[dict]:
        """대표이미지가 있는 조합의 실제 파일 경로 목록을 반환."""
        gallery = self.list_character_gallery(character)
        results = []
        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))

        for item in gallery:
            rep_file = item.get("representative", "")
            if not rep_file:
                continue
            filepath = os.path.join(
                char_dir,
                self._safe_dirname(item["outfit"]),
                self._safe_dirname(item["expression"]),
                rep_file,
            )
            if os.path.isfile(filepath):
                results.append({
                    "outfit": item["outfit"],
                    "expression": item["expression"],
                    "filename": rep_file,
                    "filepath": filepath,
                })
            else:
                print(f"[ASSET_MODE] 대표이미지 파일 없음: {filepath}")

        print(f"[ASSET_MODE] 대표이미지 일괄 분석 대상: {len(results)}개")
        return results

    # ─── 이름 치환 규칙 ───────────────────────────────────────
    def _load_name_mapping(self) -> dict:
        if os.path.isfile(NAME_MAPPING_FILE):
            try:
                with open(NAME_MAPPING_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    print(
                        f"[ASSET_NAME_MAPPING] 로드 실패: 최상위 값이 객체가 아님 "
                        f"path={NAME_MAPPING_FILE}, type={type(data).__name__}"
                    )
                    return {}
                return data
            except Exception as e:
                print(f"[ASSET_NAME_MAPPING] 로드 실패: path={NAME_MAPPING_FILE}, error={e}")
                traceback.print_exc()
        else:
            print(f"[ASSET_NAME_MAPPING] 매핑 파일 없음, 빈 규칙 사용: {NAME_MAPPING_FILE}")
        return {}

    def _save_name_mapping(self, data: dict):
        if not isinstance(data, dict):
            print(f"[ASSET_NAME_MAPPING] 저장 거부: data type={type(data).__name__}")
            raise ValueError("이름 치환 데이터는 JSON 객체여야 합니다.")
        try:
            os.makedirs(os.path.dirname(NAME_MAPPING_FILE), exist_ok=True)
        except Exception as e:
            print(
                f"[ASSET_NAME_MAPPING] 저장 폴더 생성 실패: "
                f"path={os.path.dirname(NAME_MAPPING_FILE)}, error={e}"
            )
            traceback.print_exc()
            raise
        if os.path.isfile(NAME_MAPPING_FILE):
            try:
                os.makedirs(NAME_MAPPING_BACKUP_DIR, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                suffix = uuid.uuid4().hex[:8]
                backup_path = os.path.join(
                    NAME_MAPPING_BACKUP_DIR,
                    f"name_mapping_{stamp}_{suffix}.json",
                )
                shutil.copy2(NAME_MAPPING_FILE, backup_path)
                print(f"[ASSET_NAME_MAPPING] 저장 전 백업 완료: {backup_path}")
            except Exception as e:
                print(
                    f"[ASSET_NAME_MAPPING] 저장 중단: 백업 실패 "
                    f"source={NAME_MAPPING_FILE}, error={e}"
                )
                traceback.print_exc()
                raise RuntimeError("기존 이름 치환 파일 백업에 실패하여 저장을 중단했습니다.") from e
        temp_path = f"{NAME_MAPPING_FILE}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, NAME_MAPPING_FILE)
        except Exception as e:
            print(f"[ASSET_NAME_MAPPING] UTF-8 저장 실패: path={NAME_MAPPING_FILE}, error={e}")
            traceback.print_exc()
            if os.path.isfile(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"[ASSET_NAME_MAPPING] 실패한 임시 저장 파일 정리: {temp_path}")
                except Exception as cleanup_error:
                    print(
                        f"[ASSET_NAME_MAPPING] 임시 저장 파일 정리 실패: "
                        f"path={temp_path}, error={cleanup_error}"
                    )
                    traceback.print_exc()
            raise
        print(f"[ASSET_NAME_MAPPING] UTF-8 저장 완료: {NAME_MAPPING_FILE}")

    def get_character_export_info(self, character: str) -> dict:
        """캐릭터 폴더에서 실제 복장/표정 목록을 스캔하여 이름 치환 규칙과 함께 반환."""
        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        outfits = set()
        expressions = set()
        if os.path.isdir(char_dir):
            for outfit_dir_name in sorted(os.listdir(char_dir)):
                outfit_path = os.path.join(char_dir, outfit_dir_name)
                if not os.path.isdir(outfit_path):
                    continue
                if outfit_dir_name == "Lora":
                    continue
                outfit_has_files = False
                for expr_dir_name in sorted(os.listdir(outfit_path)):
                    expr_path = os.path.join(outfit_path, expr_dir_name)
                    if os.path.isdir(expr_path) and os.listdir(expr_path):
                        outfit_has_files = True
                        expressions.add(expr_dir_name)
                if outfit_has_files:
                    outfits.add(outfit_dir_name)

        mapping = self._load_name_mapping().get(character, {})

        return {
            "character": character,
            "export_name": mapping.get("export_name", ""),
            "outfits": sorted(outfits),
            "outfit_mapping": mapping.get("outfits", {}),
            "expressions": sorted(expressions),
            "expression_mapping": mapping.get("expressions", {}),
            "export_format": mapping.get("export_format", "webp"),
            "export_quality": mapping.get("export_quality", 90),
            "naming_order": mapping.get("naming_order", ["character", "outfit", "expression"]),
            "naming_enabled": mapping.get("naming_enabled", {"character": True, "outfit": True, "expression": True}),
        }

    @staticmethod
    def _export_issue(code: str, message: str, *, details=None, **extra) -> dict:
        issue = {"code": code, "message": message}
        if details:
            issue["details"] = details
        issue.update(extra)
        return issue

    @staticmethod
    def _normalize_export_collision_key(filename: str) -> str:
        """Windows/ZIP 소비 환경에서도 같은 이름으로 취급될 값을 하나로 묶는다."""
        return unicodedata.normalize("NFC", filename).rstrip(" .").casefold()

    @staticmethod
    def _validate_export_token(value, label: str) -> list[dict]:
        errors = []
        if not isinstance(value, str):
            return [AssetMode._export_issue(
                "invalid_mapping_type",
                f"{label} 치환값은 문자열이어야 합니다. 현재 형식: {type(value).__name__}",
            )]
        if not value:
            return [AssetMode._export_issue("empty_mapping", f"{label} 치환값이 비어 있습니다.")]
        if value != value.strip():
            errors.append(AssetMode._export_issue(
                "unsafe_mapping_name",
                f"{label} 치환값 '{value}'의 앞뒤 공백을 제거하세요.",
            ))
        if value.endswith((".", " ")):
            errors.append(AssetMode._export_issue(
                "unsafe_mapping_name",
                f"{label} 치환값 '{value}'은(는) 점이나 공백으로 끝날 수 없습니다.",
            ))
        if value in (".", "..") or _INVALID_EXPORT_TOKEN_RE.search(value):
            errors.append(AssetMode._export_issue(
                "unsafe_mapping_name",
                f"{label} 치환값 '{value}'에 파일명으로 사용할 수 없는 문자가 있습니다. "
                "금지 문자: < > : \" / \\ | ? *",
            ))
        stem = value.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            errors.append(AssetMode._export_issue(
                "unsafe_mapping_name",
                f"{label} 치환값 '{value}'은(는) Windows 예약 이름이라 사용할 수 없습니다.",
            ))
        return errors

    @staticmethod
    def _coerce_export_mapping(raw_mapping: dict | None) -> dict:
        """저장 스키마와 프론트엔드 draft 스키마를 하나의 내부 형식으로 맞춘다."""
        if not isinstance(raw_mapping, dict):
            return {}
        return {
            "export_name": raw_mapping.get("export_name", ""),
            "outfits": raw_mapping.get("outfits", raw_mapping.get("outfit_mapping", {})),
            "expressions": raw_mapping.get("expressions", raw_mapping.get("expression_mapping", {})),
            "export_format": raw_mapping.get("export_format", "webp"),
            "export_quality": raw_mapping.get("export_quality", 90),
            "naming_order": raw_mapping.get("naming_order", list(EXPORT_NAMING_BLOCKS)),
            "naming_enabled": raw_mapping.get(
                "naming_enabled",
                {"character": True, "outfit": True, "expression": True},
            ),
        }

    def build_character_export_plan(
        self,
        character: str,
        selected_outfits=None,
        selected_expressions=None,
        mapping_override: dict | None = None,
    ) -> dict:
        """실제 대표 이미지와 최종 파일명을 기준으로 내보내기를 사전 검증한다."""
        errors = []
        warnings = []
        files = []

        if not isinstance(character, str) or not character.strip():
            print(f"[ASSET_EXPORT_VALIDATE] 캐릭터 이름 누락: value={character!r}")
            errors.append(self._export_issue("missing_character", "캐릭터 이름이 비어 있습니다."))
            return {"success": False, "errors": errors, "warnings": warnings, "files": files, "file_count": 0}

        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        if not os.path.isdir(char_dir):
            print(f"[ASSET_EXPORT_VALIDATE] 캐릭터 폴더 없음: {char_dir}")
            errors.append(self._export_issue(
                "character_directory_missing",
                f"캐릭터 에셋 폴더가 없습니다: {character}",
            ))
            return {"success": False, "errors": errors, "warnings": warnings, "files": files, "file_count": 0}

        stored = self._load_name_mapping().get(character, {})
        mapping = self._coerce_export_mapping(mapping_override if mapping_override is not None else stored)
        if mapping_override is not None and not isinstance(mapping_override, dict):
            print(f"[ASSET_EXPORT_VALIDATE] mapping_override 형식 오류: {type(mapping_override).__name__}")
            errors.append(self._export_issue("invalid_mapping", "이름 치환 규칙이 JSON 객체가 아닙니다."))

        outfit_map = mapping.get("outfits")
        expression_map = mapping.get("expressions")
        if not isinstance(outfit_map, dict):
            print(f"[ASSET_EXPORT_VALIDATE] outfits 매핑 형식 오류: {type(outfit_map).__name__}")
            errors.append(self._export_issue("invalid_mapping", "복장 치환 규칙이 JSON 객체가 아닙니다."))
            outfit_map = {}
        if not isinstance(expression_map, dict):
            print(f"[ASSET_EXPORT_VALIDATE] expressions 매핑 형식 오류: {type(expression_map).__name__}")
            errors.append(self._export_issue("invalid_mapping", "표정 치환 규칙이 JSON 객체가 아닙니다."))
            expression_map = {}

        naming_order = mapping.get("naming_order")
        if not isinstance(naming_order, list):
            naming_order = []
        invalid_blocks = [
            b for b in naming_order
            if not isinstance(b, str) or b not in EXPORT_NAMING_BLOCKS
        ]
        duplicate_blocks = sorted({
            b for b in naming_order
            if isinstance(b, str) and naming_order.count(b) > 1
        })
        missing_blocks = [b for b in EXPORT_NAMING_BLOCKS if b not in naming_order]
        if invalid_blocks or duplicate_blocks or missing_blocks or not naming_order:
            details = []
            if invalid_blocks:
                details.append(f"알 수 없는 블록: {', '.join(map(str, invalid_blocks))}")
            if duplicate_blocks:
                details.append(f"중복 블록: {', '.join(duplicate_blocks)}")
            if missing_blocks:
                details.append(f"누락 블록: {', '.join(missing_blocks)}")
            if not naming_order:
                details.append("파일명 블록 순서가 비어 있음")
            print(f"[ASSET_EXPORT_VALIDATE] 네이밍 순서 오류: {details}")
            errors.append(self._export_issue(
                "invalid_naming_order",
                "파일명 구성 순서가 올바르지 않습니다.",
                details=details,
            ))
            naming_order = list(EXPORT_NAMING_BLOCKS)

        naming_enabled = mapping.get("naming_enabled")
        if not isinstance(naming_enabled, dict):
            print(f"[ASSET_EXPORT_VALIDATE] naming_enabled 형식 오류: {type(naming_enabled).__name__}")
            errors.append(self._export_issue("invalid_naming_enabled", "파일명 블록 설정이 올바르지 않습니다."))
            naming_enabled = {block: True for block in EXPORT_NAMING_BLOCKS}
        invalid_enabled = [
            block for block in EXPORT_NAMING_BLOCKS
            if block in naming_enabled and not isinstance(naming_enabled[block], bool)
        ]
        if invalid_enabled:
            print(f"[ASSET_EXPORT_VALIDATE] 파일명 블록 토글 형식 오류: {invalid_enabled}")
            errors.append(self._export_issue(
                "invalid_naming_enabled",
                "파일명 블록의 켜기/끄기 값은 true 또는 false여야 합니다.",
                details=invalid_enabled,
            ))
            naming_enabled = {
                block: naming_enabled.get(block, True)
                if isinstance(naming_enabled.get(block, True), bool)
                else True
                for block in EXPORT_NAMING_BLOCKS
            }
        enabled_order = [block for block in naming_order if bool(naming_enabled.get(block, True))]
        if not enabled_order:
            print("[ASSET_EXPORT_VALIDATE] 모든 파일명 블록이 비활성화됨")
            errors.append(self._export_issue(
                "all_naming_blocks_disabled",
                "캐릭터·복장·표정 파일명 블록이 모두 꺼져 있습니다. 하나 이상 켜세요.",
            ))

        available_outfits = []
        available_expressions = set()
        for outfit_name in sorted(os.listdir(char_dir)):
            outfit_path = os.path.join(char_dir, outfit_name)
            if outfit_name == "Lora" or not os.path.isdir(outfit_path):
                continue
            available_outfits.append(outfit_name)
            for expression_name in sorted(os.listdir(outfit_path)):
                if os.path.isdir(os.path.join(outfit_path, expression_name)):
                    available_expressions.add(expression_name)

        def _selection(raw, available, label, empty_code):
            if raw is None:
                return set(available)
            if not isinstance(raw, (list, tuple, set)):
                print(f"[ASSET_EXPORT_VALIDATE] {label} 선택 형식 오류: {type(raw).__name__}")
                errors.append(self._export_issue(
                    "invalid_selection",
                    f"선택한 {label} 목록 형식이 올바르지 않습니다.",
                ))
                return set()
            chosen = {value for value in raw if isinstance(value, str)}
            if not chosen:
                print(f"[ASSET_EXPORT_VALIDATE] 선택된 {label} 없음")
                errors.append(self._export_issue(empty_code, f"선택된 {label}이 없습니다."))
                return set()
            unknown = sorted(chosen - set(available))
            if unknown:
                print(f"[ASSET_EXPORT_VALIDATE] 존재하지 않는 {label} 선택: {unknown}")
                errors.append(self._export_issue(
                    "stale_selection",
                    f"선택한 {label} 중 현재 에셋 폴더에 없는 항목이 있습니다.",
                    details=unknown,
                ))
            return chosen & set(available)

        selected_outfit_set = _selection(
            selected_outfits, available_outfits, "복장", "empty_outfit_selection"
        )
        selected_expression_set = _selection(
            selected_expressions, available_expressions, "표정", "empty_expression_selection"
        )

        export_format = str(mapping.get("export_format") or "webp").lower()
        format_map = {
            "webp": ("WEBP", ".webp"),
            "png": ("PNG", ".png"),
            "jpeg": ("JPEG", ".jpg"),
            "jpg": ("JPEG", ".jpg"),
            "avif": ("AVIF", ".avif"),
        }
        if export_format not in format_map:
            print(f"[ASSET_EXPORT_VALIDATE] 지원하지 않는 출력 형식: {export_format!r}")
            errors.append(self._export_issue(
                "unsupported_export_format",
                f"지원하지 않는 이미지 형식입니다: {export_format}",
            ))
            export_format = "webp"
        pil_format, extension = format_map[export_format]

        missing_representatives = []
        raw_candidates = []
        for outfit_name in available_outfits:
            if outfit_name not in selected_outfit_set:
                continue
            outfit_path = os.path.join(char_dir, outfit_name)
            for expression_name in sorted(os.listdir(outfit_path)):
                expression_path = os.path.join(outfit_path, expression_name)
                if not os.path.isdir(expression_path) or expression_name not in selected_expression_set:
                    continue
                rep_path = os.path.join(expression_path, "_representative.json")
                rep_file = ""
                if os.path.isfile(rep_path):
                    try:
                        with open(rep_path, "r", encoding="utf-8") as f:
                            rep_data = json.load(f)
                        if isinstance(rep_data, dict):
                            rep_file = rep_data.get("filename", "")
                    except Exception as e:
                        print(f"[ASSET_EXPORT_VALIDATE] 대표 이미지 정보 로드 실패: path={rep_path}, error={e}")
                        traceback.print_exc()
                if rep_file and (
                    not isinstance(rep_file, str)
                    or os.path.basename(rep_file) != rep_file
                    or rep_file in (".", "..")
                ):
                    print(
                        f"[ASSET_EXPORT_VALIDATE] 안전하지 않은 대표 이미지 파일명 무시: "
                        f"path={rep_path}, filename={rep_file!r}"
                    )
                    rep_file = ""
                if not rep_file or not os.path.isfile(os.path.join(expression_path, rep_file)):
                    missing_representatives.append({"outfit": outfit_name, "expression": expression_name})
                    continue
                raw_candidates.append({
                    "outfit": outfit_name,
                    "expression": expression_name,
                    "source_filename": rep_file,
                    "image_path": os.path.join(expression_path, rep_file),
                })

        if missing_representatives:
            details = [f"{item['outfit']} / {item['expression']}" for item in missing_representatives]
            print(
                f"[ASSET_EXPORT_VALIDATE] 대표 이미지 누락으로 제외: "
                f"character={character!r}, combinations={details}"
            )
            warnings.append(self._export_issue(
                "missing_representative",
                f"대표 이미지가 없어 제외되는 조합이 {len(details)}개 있습니다.",
                details=details,
            ))
        if not raw_candidates:
            print(
                f"[ASSET_EXPORT_VALIDATE] 내보낼 대표 이미지 없음: character={character!r}, "
                f"outfits={sorted(selected_outfit_set)}, expressions={sorted(selected_expression_set)}"
            )
            errors.append(self._export_issue(
                "no_exportable_representatives",
                "선택한 복장·표정 조합에 내보낼 대표 이미지가 없습니다.",
                details=[f"복장: {', '.join(sorted(selected_outfit_set)) or '(없음)'}",
                         f"표정: {', '.join(sorted(selected_expression_set)) or '(없음)'}"],
            ))

        export_name = mapping.get("export_name") or character
        missing_outfits = set()
        missing_expressions = set()
        invalid_issue_keys = set()

        def _record_token_errors(token, label):
            token_errors = self._validate_export_token(token, label)
            for issue in token_errors:
                key = (issue["code"], issue["message"])
                if key not in invalid_issue_keys:
                    invalid_issue_keys.add(key)
                    errors.append(issue)
            return not token_errors

        for candidate in raw_candidates:
            parts = []
            valid = True
            for block in enabled_order:
                if block == "character":
                    token = export_name
                    label = "캐릭터 이름"
                elif block == "outfit":
                    token = outfit_map.get(candidate["outfit"], "")
                    label = f"복장 '{candidate['outfit']}'"
                    if not token:
                        missing_outfits.add(candidate["outfit"])
                        valid = False
                        continue
                else:
                    token = expression_map.get(candidate["expression"], "")
                    label = f"표정 '{candidate['expression']}'"
                    if not token:
                        missing_expressions.add(candidate["expression"])
                        valid = False
                        continue
                if not _record_token_errors(token, label):
                    valid = False
                parts.append(token)
            if not valid or not parts:
                continue
            filename = "_".join(parts) + extension
            if len(filename) > 240:
                errors.append(self._export_issue(
                    "filename_too_long",
                    f"최종 파일명이 240자를 초과합니다: {filename[:120]}…",
                ))
                continue
            files.append({**candidate, "filename": filename})

        if missing_outfits:
            names = sorted(missing_outfits)
            print(f"[ASSET_EXPORT_VALIDATE] 복장 매핑 누락으로 조합 제외: {names}")
            warnings.append(self._export_issue(
                "missing_outfit_mapping",
                f"선택한 복장 {len(names)}개의 치환 이름이 비어 있어 해당 조합을 제외합니다.",
                details=names,
            ))
        if missing_expressions:
            names = sorted(missing_expressions)
            print(f"[ASSET_EXPORT_VALIDATE] 표정 매핑 누락으로 조합 제외: {names}")
            warnings.append(self._export_issue(
                "missing_expression_mapping",
                f"선택한 표정 {len(names)}개의 치환 이름이 비어 있어 해당 조합을 제외합니다.",
                details=names,
            ))

        if raw_candidates and not files and (missing_outfits or missing_expressions):
            print(
                f"[ASSET_EXPORT_VALIDATE] 치환 이름이 채워진 내보내기 대상 없음: "
                f"character={character!r}, missing_outfits={sorted(missing_outfits)}, "
                f"missing_expressions={sorted(missing_expressions)}"
            )
            errors.append(self._export_issue(
                "no_mapped_export_files",
                "선택한 조합 중 치환 이름이 모두 채워진 이미지가 없습니다.",
                details=[
                    "치환 이름을 하나 이상 채우거나 LLM 자동 수정을 실행하세요.",
                    "빈 치환값이 있는 조합은 자동으로 제외됩니다.",
                ],
            ))

        collision_groups = {}
        for item in files:
            collision_groups.setdefault(
                self._normalize_export_collision_key(item["filename"]), []
            ).append(item)
        collisions = []
        for group in collision_groups.values():
            if len(group) < 2:
                continue
            collisions.append({
                "filename": group[0]["filename"],
                "sources": [
                    {"outfit": item["outfit"], "expression": item["expression"]}
                    for item in group
                ],
            })
        if collisions:
            details = []
            for collision in collisions:
                source_text = ", ".join(
                    f"{source['outfit']} / {source['expression']}"
                    for source in collision["sources"]
                )
                details.append(f"{collision['filename']} ← {source_text}")
            print(f"[ASSET_EXPORT_VALIDATE] 최종 파일명 충돌: {details}")
            errors.append(self._export_issue(
                "filename_collision",
                f"서로 다른 이미지가 같은 최종 파일명으로 매핑됩니다 ({len(collisions)}건).",
                details=details,
                collisions=collisions,
                resolution=(
                    "충돌한 복장·표정 치환값을 서로 다르게 바꾸거나, 파일명 구성에서 "
                    "복장/표정 블록을 다시 켠 뒤 재시도하세요."
                ),
            ))

        try:
            export_quality = max(1, min(90, int(mapping.get("export_quality", 90))))
        except (TypeError, ValueError) as e:
            print(f"[ASSET_EXPORT_VALIDATE] 출력 품질 형식 오류: {mapping.get('export_quality')!r}, error={e}")
            traceback.print_exc()
            errors.append(self._export_issue("invalid_export_quality", "이미지 품질은 1~90 사이 숫자여야 합니다."))
            export_quality = 90

        return {
            "success": not errors,
            "errors": errors,
            "warnings": warnings,
            "files": files,
            "file_count": len(files),
            "mapping": {
                **mapping,
                "export_name": export_name,
                "outfits": outfit_map,
                "expressions": expression_map,
                "export_format": export_format,
                "export_quality": export_quality,
                "pil_format": pil_format,
                "extension": extension,
                "naming_order": naming_order,
                "naming_enabled": naming_enabled,
            },
            "selection": {
                "outfits": sorted(selected_outfit_set),
                "expressions": sorted(selected_expression_set),
            },
        }

    @staticmethod
    def public_export_plan(plan: dict) -> dict:
        """로컬 절대 경로를 제외하고 API로 반환 가능한 검증 결과를 만든다."""
        return {
            "success": bool(plan.get("success")),
            "errors": plan.get("errors", []),
            "warnings": plan.get("warnings", []),
            "file_count": int(plan.get("file_count", 0)),
            "files": [
                {
                    "outfit": item.get("outfit", ""),
                    "expression": item.get("expression", ""),
                    "filename": item.get("filename", ""),
                }
                for item in plan.get("files", [])
            ],
            "selection": plan.get("selection", {}),
        }

    def save_character_name_mapping(self, character: str, export_name: str,
                                    outfit_mapping: dict, expression_mapping: dict,
                                    export_format: str = "webp", export_quality: int = 90,
                                    naming_order: list = None, naming_enabled: dict = None) -> dict:
        """캐릭터 이름 치환 규칙 저장."""
        if not isinstance(character, str) or not character.strip():
            print(f"[ASSET_NAME_MAPPING] 저장 거부: character={character!r}")
            raise ValueError("캐릭터 이름이 필요합니다.")
        if not isinstance(export_name, str):
            print(f"[ASSET_NAME_MAPPING] 저장 거부: export_name type={type(export_name).__name__}")
            raise ValueError("캐릭터 치환 이름은 문자열이어야 합니다.")
        if not isinstance(outfit_mapping, dict) or not isinstance(expression_mapping, dict):
            print(
                f"[ASSET_NAME_MAPPING] 저장 거부: outfits={type(outfit_mapping).__name__}, "
                f"expressions={type(expression_mapping).__name__}"
            )
            raise ValueError("복장/표정 치환 규칙은 JSON 객체여야 합니다.")

        if export_name:
            token_errors = self._validate_export_token(export_name, "캐릭터 이름")
            if token_errors:
                print(f"[ASSET_NAME_MAPPING] 저장 거부: {token_errors[0]['message']}")
                raise ValueError(token_errors[0]["message"])
        cleaned_mappings = {}
        for category, mapping in (("outfits", outfit_mapping), ("expressions", expression_mapping)):
            label = "복장" if category == "outfits" else "표정"
            cleaned = {}
            for original, mapped in mapping.items():
                if not isinstance(original, str):
                    print(
                        f"[ASSET_NAME_MAPPING] 저장 거부: {label} 원본 키 형식="
                        f"{type(original).__name__}"
                    )
                    raise ValueError(f"{label} 원본 이름은 문자열이어야 합니다.")
                if mapped == "":
                    print(f"[ASSET_NAME_MAPPING] 빈 치환값 제거: {label}={original!r}")
                    continue
                token_errors = self._validate_export_token(mapped, f"{label} '{original}'")
                if token_errors:
                    print(f"[ASSET_NAME_MAPPING] 저장 거부: {token_errors[0]['message']}")
                    raise ValueError(token_errors[0]["message"])
                cleaned[original] = mapped
            cleaned_mappings[category] = cleaned
        outfit_mapping = cleaned_mappings["outfits"]
        expression_mapping = cleaned_mappings["expressions"]

        export_format = str(export_format or "webp").lower()
        if export_format not in ("webp", "png", "jpeg", "jpg", "avif"):
            print(f"[ASSET_NAME_MAPPING] 저장 거부: export_format={export_format!r}")
            raise ValueError(f"지원하지 않는 이미지 형식입니다: {export_format}")
        try:
            export_quality = max(1, min(90, int(export_quality)))
        except (TypeError, ValueError) as e:
            print(f"[ASSET_NAME_MAPPING] 저장 거부: export_quality={export_quality!r}, error={e}")
            traceback.print_exc()
            raise ValueError("이미지 품질은 1~90 사이 숫자여야 합니다.") from e

        if naming_order is None:
            naming_order = list(EXPORT_NAMING_BLOCKS)
        if (
            not isinstance(naming_order, list)
            or len(naming_order) != len(EXPORT_NAMING_BLOCKS)
            or not all(isinstance(block, str) for block in naming_order)
            or set(naming_order) != set(EXPORT_NAMING_BLOCKS)
        ):
            print(f"[ASSET_NAME_MAPPING] 저장 거부: naming_order={naming_order!r}")
            raise ValueError("파일명 블록 순서는 캐릭터·복장·표정을 각각 한 번씩 포함해야 합니다.")

        if naming_enabled is None:
            naming_enabled = {block: True for block in EXPORT_NAMING_BLOCKS}
        if not isinstance(naming_enabled, dict) or any(
            block in naming_enabled and not isinstance(naming_enabled[block], bool)
            for block in EXPORT_NAMING_BLOCKS
        ):
            print(f"[ASSET_NAME_MAPPING] 저장 거부: naming_enabled={naming_enabled!r}")
            raise ValueError("파일명 블록의 켜기/끄기 값은 true 또는 false여야 합니다.")
        naming_enabled = {
            block: naming_enabled.get(block, True) for block in EXPORT_NAMING_BLOCKS
        }
        if not any(naming_enabled.values()):
            print("[ASSET_NAME_MAPPING] 저장 거부: 모든 파일명 블록 비활성화")
            raise ValueError("캐릭터·복장·표정 파일명 블록을 하나 이상 켜야 합니다.")

        data = self._load_name_mapping()
        data[character] = {
            "export_name": export_name,
            "outfits": outfit_mapping,
            "expressions": expression_mapping,
            "export_format": export_format,
            "export_quality": export_quality,
            "naming_order": naming_order,
            "naming_enabled": naming_enabled,
        }
        self._save_name_mapping(data)
        return {"success": True}

    def get_ep_settings(self, character: str) -> dict:
        data = self._load_name_mapping()
        return data.get(character, {}).get("ep_settings", {})

    def get_last_ep_settings(self) -> dict:
        data = self._load_name_mapping()
        last_char = data.get("_last_ep_settings_character", "")
        if not last_char:
            return {}
        settings = data.get(last_char, {}).get("ep_settings", {})
        if settings:
            settings["character"] = last_char
        return settings

    def save_ep_settings(self, character: str, settings: dict) -> dict:
        data = self._load_name_mapping()
        if character not in data:
            data[character] = {
                "export_name": character,
                "outfits": {},
                "expressions": {},
            }
        data[character]["ep_settings"] = settings
        data["_last_ep_settings_character"] = character
        self._save_name_mapping(data)
        return {"success": True}

    def export_character_zip(
        self,
        character: str,
        selected_outfits=None,
        selected_expressions=None,
        mapping_override: dict | None = None,
        export_plan: dict | None = None,
    ):
        """캐릭터의 대표 이미지를 이름 치환 규칙에 따라 이름_복장_표정.ext로 만들어 zip 반환.
        selected_outfits / selected_expressions(디렉터리명 리스트)가 주어지면 해당 항목만 내보낸다.
        None이면 전체 내보내기(기존 동작)."""
        import zipfile, io, logging
        from PIL import Image

        log = logging.getLogger("asset_export")
        plan = export_plan or self.build_character_export_plan(
            character,
            selected_outfits,
            selected_expressions,
            mapping_override,
        )
        if not plan.get("success"):
            for issue in plan.get("errors", []):
                print(
                    f"[ZIP 내보내기] 사전 검증 실패: code={issue.get('code')}, "
                    f"message={issue.get('message')}, details={issue.get('details', [])}"
                )
            return None

        mapping = plan["mapping"]
        export_format = mapping["export_format"]
        export_quality = mapping["export_quality"]
        pil_format = mapping["pil_format"]
        ext = mapping["extension"]
        log.info(
            f"[ZIP 내보내기] 시작 — 캐릭터: {character}, "
            f"선택 복장={plan['selection']['outfits']}, 선택 표정={plan['selection']['expressions']}, "
            f"파일={plan['file_count']}개, 포맷={export_format}, 품질={export_quality}"
        )

        # 로컬 저장 품질 (90) 대비 보정값 계산
        # 사용자가 80 설정 → PIL quality = round(80/0.9) ≈ 89 → 유효 품질 ~80%
        LOCAL_QUALITY = 90
        if export_quality >= LOCAL_QUALITY:
            pil_quality = LOCAL_QUALITY
            need_recompress = False
        else:
            pil_quality = min(100, round(export_quality / (LOCAL_QUALITY / 100)))
            need_recompress = True

        buf = io.BytesIO()
        added = 0
        used_names = set()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in plan["files"]:
                rep_file = item["source_filename"]
                img_path = item["image_path"]
                zip_name = item["filename"]
                collision_key = self._normalize_export_collision_key(zip_name)
                if collision_key in used_names:
                    print(f"[ZIP 내보내기] 내부 충돌 감지: {zip_name}")
                    raise RuntimeError(f"사전 검증 후 파일명 충돌이 다시 감지되었습니다: {zip_name}")
                used_names.add(collision_key)

                orig_ext = os.path.splitext(rep_file)[1].lower()
                if orig_ext == ext and not need_recompress:
                    zf.write(img_path, zip_name)
                    log.info(f"[ZIP 내보내기] [{added + 1}] 원본 그대로 추가: {zip_name}")
                else:
                    try:
                        log.info(
                            f"[ZIP 내보내기] [{added + 1}] 변환 중: "
                            f"{rep_file} → {zip_name} ({pil_format}, q={pil_quality})"
                        )
                        with Image.open(img_path) as opened:
                            opened.load()
                            img = opened
                            if pil_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                                img = img.convert("RGB")
                            elif pil_format != "AVIF" and img.mode not in ("RGB", "RGBA"):
                                img = img.convert("RGBA") if pil_format != "JPEG" else img.convert("RGB")

                            img_buf = io.BytesIO()
                            save_kwargs = {"format": pil_format}
                            if pil_format in ("WEBP", "JPEG", "AVIF"):
                                save_kwargs["quality"] = pil_quality
                            if pil_format == "WEBP":
                                save_kwargs["method"] = 6
                            img.save(img_buf, **save_kwargs)
                        img_buf.seek(0)
                        zf.writestr(zip_name, img_buf.read())
                        log.info(
                            f"[ZIP 내보내기] [{added + 1}] 변환 완료: "
                            f"{zip_name} ({len(img_buf.getvalue())} bytes)"
                        )
                    except Exception as e:
                        print(
                            f"[ZIP 내보내기] 이미지 변환 실패: source={img_path}, "
                            f"target={zip_name}, format={pil_format}, error={e}"
                        )
                        traceback.print_exc()
                        raise RuntimeError(
                            f"이미지 변환에 실패했습니다: {rep_file} → {zip_name} ({e})"
                        ) from e

                added += 1

        if added == 0:
            print(f"[ZIP 내보내기] 추가된 파일 없음: character={character!r}")
            return None
        buf.seek(0)
        log.info(
            f"[ZIP 내보내기] 완료 — 총 {added}개 파일, "
            f"ZIP 크기={buf.getbuffer().nbytes / 1024:.1f}KB"
        )
        return buf

    # ─── 표정 프로필 ─────────────────────────────────────
    def scan_expression_profiles(self, character: str, outfit: str) -> dict:
        """캐릭터/복장 경로에서 표정 프로필 폴더 상태를 스캔.
        tags.json의 표정 목록과 실제 폴더를 비교하여 상태 반환."""
        expr_list = self.list_expressions()
        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        outfit_dir = os.path.join(char_dir, self._safe_dirname(outfit))
        results = []
        for expr_name in expr_list:
            expr_dir = os.path.join(outfit_dir, self._safe_dirname(expr_name))
            exists = os.path.isdir(expr_dir)
            has_images = False
            representative = ""
            image_count = 0
            if exists:
                for f in os.listdir(expr_dir):
                    if f.startswith("_"):
                        continue
                    fp = os.path.join(expr_dir, f)
                    if os.path.isfile(fp):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in (".webp", ".png", ".jpg", ".jpeg", ".avif"):
                            image_count += 1
                            has_images = True
                rep_path = os.path.join(expr_dir, "_representative.json")
                if os.path.isfile(rep_path):
                    try:
                        with open(rep_path, "r", encoding="utf-8") as f:
                            representative = json.load(f).get("filename", "")
                    except Exception:
                        pass
            results.append({
                "name": expr_name,
                "folder_exists": exists,
                "has_images": has_images,
                "image_count": image_count,
                "representative": representative,
            })
        return {"profiles": results, "character": character, "outfit": outfit}

    def create_expression_profile_folders(self, character: str, outfit: str, expressions: list = None) -> dict:
        """지정한 캐릭터/복장 경로에 표정 폴더를 생성.
        expressions가 None이면 tags.json의 모든 표정에 대해 생성."""
        if expressions is None:
            expressions = self.list_expressions()
        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        outfit_dir = os.path.join(char_dir, self._safe_dirname(outfit))
        created = []
        skipped = []
        for expr_name in expressions:
            expr_dir = os.path.join(outfit_dir, self._safe_dirname(expr_name))
            if os.path.isdir(expr_dir):
                skipped.append(expr_name)
            else:
                os.makedirs(expr_dir, exist_ok=True)
                created.append(expr_name)
        return {"success": True, "created": created, "skipped": skipped}

    # ─── 프리셋매니징: hidden_tags I/O ──────────────────────
    def load_hidden_tags(self):
        """hidden_tags.json 로드"""
        os.makedirs(ASSET_DATA_DIR, exist_ok=True)
        if os.path.isfile(HIDDEN_TAGS_FILE):
            try:
                with open(HIDDEN_TAGS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ASSET_MODE] hidden_tags 로드 실패: {e}")
                traceback.print_exc()
        return {}

    def save_hidden_tags(self, data: dict):
        """hidden_tags.json 저장"""
        os.makedirs(ASSET_DATA_DIR, exist_ok=True)
        with open(HIDDEN_TAGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_hidden_tags(self) -> dict:
        """프리셋매니징용: hidden_tags + 활성 tags 병합 반환"""
        return {
            "active": self._get_active_presets(),
            "hidden": self.load_hidden_tags(),
        }

    def _get_active_presets(self) -> dict:
        """현재 tags.json에서 프리셋매니징 대상 카테고리만 추출"""
        result = {}
        for cat in PRESET_MGMT_CATEGORIES:
            val = self._tags.get(cat, {})
            # appearances/outfits/expressions은 dict, quality_presets 등도 dict
            result[cat] = copy.deepcopy(val) if isinstance(val, dict) else list(val) if isinstance(val, list) else val
        return result

    # ─── 프리셋매니징: 숨기기 / 복원 ───────────────────────
    def hide_preset(self, category: str, name: str) -> dict:
        """프리셋을 tags.json에서 hidden_tags.json으로 이동"""
        if category not in PRESET_MGMT_CATEGORIES:
            print(f"[ASSET_MODE] hide_preset: 지원하지 않는 카테고리 '{category}'")
            return {"success": False, "error": f"지원하지 않는 카테고리: {category}"}

        cat_data = self._tags.get(category, {})
        if isinstance(cat_data, dict):
            if name not in cat_data:
                print(f"[ASSET_MODE] hide_preset: '{name}'을(를) {category}에서 찾을 수 없음")
                return {"success": False, "error": f"'{name}'을(를) 찾을 수 없습니다."}
            tag_value = copy.deepcopy(cat_data[name])
        else:
            print(f"[ASSET_MODE] hide_preset: 카테고리 '{category}'가 dict가 아님")
            return {"success": False, "error": f"카테고리 '{category}' 구조 오류"}

        # hidden_tags에 추가
        hidden = self.load_hidden_tags()
        hidden_cat = hidden.setdefault(category, {})
        if name in hidden_cat:
            print(f"[ASSET_MODE] hide_preset: '{name}'이(가) 이미 숨김 상태")
            return {"success": False, "error": f"'{name}'은(는) 이미 숨김 처리되어 있습니다."}
        hidden_cat[name] = tag_value
        self.save_hidden_tags(hidden)

        # tags.json에서 제거
        del cat_data[name]
        self.save_tags()

        self._log("preset_hidden", {"category": category, "name": name})
        return {"success": True}

    def hide_presets_batch(self, category: str, names: list) -> dict:
        """여러 프리셋을 일괄 숨기기"""
        results = []
        for name in names:
            r = self.hide_preset(category, name)
            results.append({"name": name, **r})
        return {"success": True, "results": results}

    def restore_preset(self, category: str, name: str) -> dict:
        """숨김 프리셋을 hidden_tags.json에서 tags.json으로 복원"""
        if category not in PRESET_MGMT_CATEGORIES:
            print(f"[ASSET_MODE] restore_preset: 지원하지 않는 카테고리 '{category}'")
            return {"success": False, "error": f"지원하지 않는 카테고리: {category}"}

        hidden = self.load_hidden_tags()
        hidden_cat = hidden.get(category, {})
        if name not in hidden_cat:
            print(f"[ASSET_MODE] restore_preset: '{name}'을(를) 숨김 목록에서 찾을 수 없음")
            return {"success": False, "error": f"'{name}'을(를) 숨김 목록에서 찾을 수 없습니다."}

        tag_value = copy.deepcopy(hidden_cat[name])

        # tags.json에 복원 (이미 존재하면 에러)
        cat_data = self._tags.setdefault(category, {})
        if isinstance(cat_data, dict) and name in cat_data:
            print(f"[ASSET_MODE] restore_preset: '{name}'이(가) 이미 tags.json에 존재함")
            return {"success": False, "error": f"'{name}'은(는) 이미 활성 상태입니다."}
        if isinstance(cat_data, dict):
            cat_data[name] = tag_value
        self.save_tags()

        # hidden_tags에서 제거
        del hidden_cat[name]
        if not hidden_cat:
            hidden.pop(category, None)
        self.save_hidden_tags(hidden)

        self._log("preset_restored", {"category": category, "name": name})
        return {"success": True}

    def restore_presets_batch(self, category: str, names: list) -> dict:
        """여러 숨김 프리셋을 일괄 복원"""
        results = []
        for name in names:
            r = self.restore_preset(category, name)
            results.append({"name": name, **r})
        return {"success": True, "results": results}

    # ─── 프리셋매니징: 이름 변경 ───────────────────────────
    # 카테고리 → characters[*] 참조 필드 매핑
    _PRESET_REF_FIELD = {
        "appearances": "appearance",
        "outfits": "outfit",
        "expressions": "expression",
    }

    def rename_preset(self, category: str, old_name: str, new_name: str) -> dict:
        """프리셋 이름 변경 (활성/숨김 모두 지원).
        appearances/outfits/expressions 인 경우 characters[*] 참조도 함께 갱신.
        """
        if category not in PRESET_MGMT_CATEGORIES:
            print(f"[ASSET_MODE] rename_preset: 지원하지 않는 카테고리 '{category}'")
            return {"success": False, "error": f"지원하지 않는 카테고리: {category}"}

        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()
        if not old_name:
            print("[ASSET_MODE] rename_preset: 기존 이름이 비어있음")
            return {"success": False, "error": "기존 이름이 비어있습니다."}
        if not new_name:
            print("[ASSET_MODE] rename_preset: 새 이름이 비어있음")
            return {"success": False, "error": "새 이름을 입력해주세요."}
        if old_name == new_name:
            return {"success": False, "error": "새 이름이 현재 이름과 같습니다."}

        # 활성(tags.json) / 숨김(hidden_tags.json) 양쪽 탐색
        cat_data = self._tags.get(category, {})
        hidden = self.load_hidden_tags()
        hidden_cat = hidden.get(category, {})

        in_active = isinstance(cat_data, dict) and old_name in cat_data
        in_hidden = isinstance(hidden_cat, dict) and old_name in hidden_cat
        if not in_active and not in_hidden:
            print(f"[ASSET_MODE] rename_preset: '{old_name}'을(를) {category}에서 찾을 수 없음")
            return {"success": False, "error": f"'{old_name}'을(를) 찾을 수 없습니다."}

        # 새 이름 충돌 검사 (같은 카테고리의 활성/숨김 양쪽)
        if isinstance(cat_data, dict) and new_name in cat_data and new_name != old_name:
            return {"success": False, "error": f"이미 존재하는 이름입니다: '{new_name}'"}
        if isinstance(hidden_cat, dict) and new_name in hidden_cat and new_name != old_name:
            return {"success": False, "error": f"숨김 목록에 이미 존재하는 이름입니다: '{new_name}'"}

        # 순서 보존하며 키 교체
        def _rebuild(d: dict) -> dict:
            return {(new_name if k == old_name else k): v for k, v in d.items()}

        touched_tags = False
        touched_hidden = False
        if in_active:
            self._tags[category] = _rebuild(cat_data)
            touched_tags = True
        if in_hidden:
            hidden[category] = _rebuild(hidden_cat)
            touched_hidden = True

        # 캐릭터 참조 갱신 (appearances/outfits/expressions)
        ref_field = self._PRESET_REF_FIELD.get(category)
        ref_count = 0
        if ref_field:
            chars = self._tags.get("characters", {})
            for c in chars.values():
                if isinstance(c, dict) and c.get(ref_field) == old_name:
                    c[ref_field] = new_name
                    ref_count += 1
            if ref_count > 0:
                touched_tags = True

        if touched_tags:
            self.save_tags()
        if touched_hidden:
            self.save_hidden_tags(hidden)

        self._log("preset_renamed", {
            "category": category, "old": old_name, "new": new_name,
            "ref_updated": ref_count,
        })
        print(f"[ASSET_MODE] rename_preset: '{old_name}' -> '{new_name}' ({category}), 참조 {ref_count}건 갱신")
        return {"success": True, "ref_updated": ref_count}

    @staticmethod
    def _split_tags_preserving_parens(text: str) -> list:
        """쉼표로 분리하되 괄호 () [] {} 내부의 쉼표는 무시."""
        tags = []
        depth = 0
        buf = []
        for ch in text:
            if ch in '({[':
                depth += 1
                buf.append(ch)
            elif ch in ')}]':
                depth = max(0, depth - 1)
                buf.append(ch)
            elif ch == ',' and depth == 0:
                tag = ''.join(buf).strip()
                if tag:
                    tags.append(tag)
                buf = []
            else:
                buf.append(ch)
        tag = ''.join(buf).strip()
        if tag:
            tags.append(tag)
        return tags

    # ─── 프리셋매니징: 추가 / 수정 ─────────────────────────
    def _parse_managed_preset_value(self, category: str, tags_text: str):
        """관리 화면 입력값을 저장 형식으로 변환한다. 실패 시 (None, error)를 반환한다."""
        if category == "natural_language_presets":
            text = (tags_text or "").strip()
            if not text:
                print("[ASSET_MODE] save_managed_preset: 텍스트가 비어있음")
                return None, "텍스트를 입력해주세요."
            return text, None

        tags = self._split_tags_preserving_parens(tags_text or "")
        if not tags:
            print("[ASSET_MODE] save_managed_preset: 태그가 비어있음")
            return None, "태그를 입력해주세요."
        return tags, None

    def save_managed_preset(
        self,
        category: str,
        name: str,
        tags_text: str,
        operation: str = "create",
        original_name: str = "",
        target_state: str = "active",
    ) -> dict:
        """프리셋을 충돌 검사 후 생성하거나 활성/숨김 위치에서 명시적으로 수정한다."""
        if category not in PRESET_MGMT_CATEGORIES:
            print(f"[ASSET_MODE] save_managed_preset: 지원하지 않는 카테고리 '{category}'")
            return {"success": False, "error": f"지원하지 않는 카테고리: {category}"}

        operation = (operation or "").strip().lower()
        if operation not in {"create", "update"}:
            print(f"[ASSET_MODE] save_managed_preset: 지원하지 않는 작업 '{operation}'")
            return {"success": False, "error": f"지원하지 않는 작업: {operation}"}

        name = (name or "").strip()
        if not name:
            print("[ASSET_MODE] save_managed_preset: 이름이 비어있음")
            return {"success": False, "error": "이름을 입력해주세요."}

        cat_data = self._tags.setdefault(category, {})
        if not isinstance(cat_data, dict):
            print(f"[ASSET_MODE] save_managed_preset: 카테고리 '{category}'가 dict가 아님")
            return {"success": False, "error": f"카테고리 '{category}' 구조 오류"}

        hidden = self.load_hidden_tags()
        hidden_cat = hidden.get(category, {})
        if not isinstance(hidden_cat, dict):
            print(f"[ASSET_MODE] save_managed_preset: 숨김 카테고리 '{category}'가 dict가 아님")
            return {"success": False, "error": f"숨김 카테고리 '{category}' 구조 오류"}

        value, value_error = self._parse_managed_preset_value(category, tags_text)
        if value_error:
            return {"success": False, "error": value_error}

        active_exists = name in cat_data
        hidden_exists = name in hidden_cat

        if operation == "create":
            if active_exists or hidden_exists:
                conflict_state = "hidden" if hidden_exists and not active_exists else "active"
                state_label = "숨김" if conflict_state == "hidden" else "활성"
                print(
                    f"[ASSET_MODE] save_managed_preset: 신규 이름 충돌 "
                    f"category={category}, name={name!r}, state={conflict_state}"
                )
                return {
                    "success": False,
                    "error": f"'{name}' 이름의 {state_label} 프리셋이 이미 존재합니다.",
                    "conflict": True,
                    "conflict_state": conflict_state,
                }

            cat_data[name] = value
            self.save_tags()
            count = len(value)
            self._log("preset_created", {"category": category, "name": name, "count": count})
            print(f"[ASSET_MODE] save_managed_preset: 신규 저장 완료 ({category}/{name}, count={count})")
            return {
                "success": True,
                "operation": "create",
                "state": "active",
                "name": name,
                "count": count,
            }

        original_name = (original_name or "").strip()
        if not original_name:
            print("[ASSET_MODE] save_managed_preset: 수정할 기존 이름이 비어있음")
            return {"success": False, "error": "수정할 프리셋 이름이 비어있습니다."}
        if target_state not in {"active", "hidden"}:
            print(f"[ASSET_MODE] save_managed_preset: 잘못된 대상 상태 '{target_state}'")
            return {"success": False, "error": f"잘못된 프리셋 상태: {target_state}"}

        source = cat_data if target_state == "active" else hidden_cat
        if original_name not in source:
            print(
                f"[ASSET_MODE] save_managed_preset: 수정 대상 없음 "
                f"category={category}, name={original_name!r}, state={target_state}"
            )
            return {"success": False, "error": f"수정할 프리셋 '{original_name}'을(를) 찾을 수 없습니다."}

        if name != original_name and (name in cat_data or name in hidden_cat):
            conflict_state = "active" if name in cat_data else "hidden"
            state_label = "활성" if conflict_state == "active" else "숨김"
            print(
                f"[ASSET_MODE] save_managed_preset: 수정 이름 충돌 "
                f"category={category}, old={original_name!r}, new={name!r}, state={conflict_state}"
            )
            return {
                "success": False,
                "error": f"'{name}' 이름의 {state_label} 프리셋이 이미 존재합니다.",
                "conflict": True,
                "conflict_state": conflict_state,
            }

        # dict 순서를 유지하면서 이름과 값을 한 번에 교체한다.
        updated_source = {}
        for key, old_value in source.items():
            if key == original_name:
                updated_source[name] = value
            else:
                updated_source[key] = old_value

        if target_state == "active":
            self._tags[category] = updated_source
        else:
            hidden[category] = updated_source

        ref_count = 0
        if name != original_name:
            ref_field = self._PRESET_REF_FIELD.get(category)
            if ref_field:
                characters = self._tags.get("characters", {})
                for character in characters.values():
                    if isinstance(character, dict) and character.get(ref_field) == original_name:
                        character[ref_field] = name
                        ref_count += 1

        # 숨김 값 수정은 hidden_tags만, 활성 값 또는 참조 이름 수정은 tags를 저장한다.
        if target_state == "active" or ref_count > 0:
            self.save_tags()
        if target_state == "hidden":
            self.save_hidden_tags(hidden)

        count = len(value)
        self._log("preset_updated", {
            "category": category,
            "old": original_name,
            "name": name,
            "state": target_state,
            "count": count,
            "ref_updated": ref_count,
        })
        print(
            f"[ASSET_MODE] save_managed_preset: 수정 완료 "
            f"({category}/{original_name} -> {name}, state={target_state}, count={count}, refs={ref_count})"
        )
        return {
            "success": True,
            "operation": "update",
            "state": target_state,
            "name": name,
            "count": count,
            "ref_updated": ref_count,
        }

    def batch_insert_preset(self, category: str, name: str, tags_text: str) -> dict:
        """이전 호출 호환용 신규 추가. 같은 이름은 덮어쓰지 않는다."""
        return self.save_managed_preset(
            category=category,
            name=name,
            tags_text=tags_text,
            operation="create",
        )

    # ─── 프리셋매니징: 에셋 추적 ────────────────────────────
    def _get_preset_tags_raw(self, category: str, name: str) -> Optional[list]:
        """활성/숨김에서 프리셋 태그 원본 리스트를 가져온다. 못 찾으면 None."""
        active_cat = self._tags.get(category, {})
        hidden = self.load_hidden_tags()
        hidden_cat = hidden.get(category, {})
        if isinstance(active_cat, dict) and name in active_cat:
            v = active_cat[name]
            return v if isinstance(v, list) else [v]
        if isinstance(hidden_cat, dict) and name in hidden_cat:
            v = hidden_cat[name]
            return v if isinstance(v, list) else [v]
        return None

    def trace_preset_assets(self, category: str, name: str) -> dict:
        """프리셋이 사용된 에셋 이미지를 추적 (스레드풀 병렬 스캔)."""
        if category not in PRESET_MGMT_CATEGORIES:
            print(f"[ASSET_MODE] trace_preset_assets: 지원하지 않는 카테고리 '{category}'")
            return {"success": False, "error": f"지원하지 않는 카테고리: {category}"}

        preset_tags = self._get_preset_tags_raw(category, name)
        if preset_tags is None:
            print(f"[ASSET_MODE] trace_preset_assets: '{name}'을(를) 찾을 수 없음")
            return {"success": False, "error": f"'{name}'을(를) 찾을 수 없습니다."}

        # 파일 목록 1회 수집
        files = _collect_prompt_files()
        if not files:
            print(f"[ASSET_MODE] trace_preset_assets: asset/ 디렉토리 없음 또는 프롬프트 파일 없음")
            return {"success": True, "results": [], "preset_tags": preset_tags}

        # 스레드풀 병렬 매칭 (청크 단위, I/O 바운드)
        executor = _get_trace_executor()
        args_list = [
            (path, category, name, preset_tags, char, outfit, expr, expr_dir, fname)
            for (char, outfit, expr, expr_dir, fname, path) in files
        ]
        chunks = _split_chunks(args_list, executor._max_workers)
        results = []
        for chunk_matches in executor.map(_match_chunk, chunks):
            for m in chunk_matches:
                results.append({
                    "character": m["character"],
                    "outfit": m["outfit"],
                    "expression": m["expression"],
                    "image_file": m["image_file"],
                    "prompt_data": m["_prompt_data"],
                    "match_count": m["match_count"],
                })

        # 일치순 정렬 (매칭 태그 수 많은 순)
        results.sort(key=lambda r: r.get("match_count", 0), reverse=True)
        self._log("preset_traced", {"category": category, "name": name, "matches": len(results)})
        return {"success": True, "results": results, "preset_tags": preset_tags}

    def trace_preset_assets_stream(self, category: str, name: str):
        """프리셋 추적 제너레이터 - 스레드풀 병렬 스캔, 매치 결과를 개별 이벤트로 yield.

        SSE 이벤트 계약 유지: start / progress / match / done / error (프론트엔드 변경 없음).
        매치는 완료 순서로 도착한다(디렉토리 순서 아님). UI는 도착 순 렌더라 무관."""
        if category not in PRESET_MGMT_CATEGORIES:
            yield ("error", {"error": f"지원하지 않는 카테고리: {category}"})
            return

        preset_tags_raw = self._get_preset_tags_raw(category, name)
        if preset_tags_raw is None:
            yield ("error", {"error": f"'{name}'을(를) 찾을 수 없습니다."})
            return

        # 중복 태그 제거 (순서 유지, 대소문자 무시, 빈 문자열 제거)
        seen = set()
        preset_tags = []
        for t in preset_tags_raw:
            if not isinstance(t, str) or not t.strip():
                continue
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                preset_tags.append(t)

        # 파일 목록 1회 수집 (기존 2회 순회 → 1회)
        files = _collect_prompt_files()
        total = len(files)

        yield ("start", {"total": total, "preset_tags": preset_tags})

        scanned = 0
        found = 0
        if total == 0:
            yield ("done", {"total_found": 0})
            self._log("preset_traced", {"category": category, "name": name, "matches": 0})
            return

        # 스레드풀 병렬 매칭(청크 단위), 완료 순서대로 스트리밍
        executor = _get_trace_executor()
        args_list = [
            (path, category, name, preset_tags, char, outfit, expr, expr_dir, fname)
            for (char, outfit, expr, expr_dir, fname, path) in files
        ]
        chunks = _split_chunks(args_list, executor._max_workers)
        fut_to_len = {executor.submit(_match_chunk, ch): len(ch) for ch in chunks}

        for fut in as_completed(fut_to_len):
            ch_len = fut_to_len[fut]
            try:
                chunk_matches = fut.result()
            except Exception as e:
                print(f"[ASSET_MODE] trace: 청크 작업 예외 {type(e).__name__}: {e}")
                traceback.print_exc()
                chunk_matches = []
            for m in chunk_matches:
                found += 1
                yield ("match", {
                    "character": m["character"],
                    "outfit": m["outfit"],
                    "expression": m["expression"],
                    "image_file": m["image_file"],
                    "match_count": m["match_count"],
                    "positive": m["positive"],
                    "negative": m["negative"],
                })
            scanned += ch_len
            yield ("progress", {"scanned": scanned, "total": total, "found": found})

        yield ("done", {"total_found": found})
        self._log("preset_traced", {"category": category, "name": name, "matches": found})
    def get_status(self) -> dict:
        return {
            "workflow_source_path": self.workflow_source_path,
            "workflow_loaded": self._asset_api_workflow is not None,
            "is_generating": self._is_generating,
            "character_count": len(self._tags.get("characters", {})),
            "quality_tags": self._tags.get("quality", []),
            "composition_tags": self._tags.get("composition", []),
            "negative_tags": self._tags.get("negative", []),
            "quality_presets": self._tags.get("quality_presets", {}),
            "composition_presets": self._tags.get("composition_presets", {}),
        }


asset_mode = AssetMode()
