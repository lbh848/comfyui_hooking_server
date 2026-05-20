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
from typing import Optional, Callable, Awaitable


# ─── 상수 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
ASSET_DIR = os.path.join(BASE_DIR, "asset")
TAGS_FILE = os.path.join(ASSET_DATA_DIR, "tags.json")
HIDDEN_TAGS_FILE = os.path.join(ASSET_DATA_DIR, "hidden_tags.json")
NAME_MAPPING_FILE = os.path.join(ASSET_DATA_DIR, "name_mapping.json")

# 프리셋매니징 대상 카테고리
PRESET_MGMT_CATEGORIES = [
    "appearances", "outfits", "expressions",
    "quality_presets", "composition_presets",
    "negative_presets", "character_negative_presets",
]
CURRENT_MODE_WORK_DIR = os.path.join(BASE_DIR, "current_mode_workflow")
MODE_WORKFLOW_DIR = os.path.join(BASE_DIR, "mode_workflow")

DEFAULT_POSE_DATA = {
    "people": [
        {
            "pose_keypoints_2d": [
                509.59, 522.31, 1,
                552.52, 696.55, 1,
                299.04, 600.06, 1,
                73.98, 858.17, 1,
                86.61, 624.58, 1,
                774.33, 791.04, 1,
                484.34, 1286.20, 1,
                442.67, 1336.70, 1,
                263.38, 1096.80, 1,
                354.29, 1393.52, 1,
                344.19, 1408.67, 1,
                469.19, 1128.37, 1,
                561.36, 1390.99, 1,
                561.36, 1408.67, 1,
                422.47, 471.80, 1,
                618.18, 404.88, 1,
                351.76, 502.11, 1,
                730.55, 414.98, 1,
            ]
        }
    ],
    "canvas_height": 1408,
    "canvas_width": 936,
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
    "character_presets": {},
}


class AssetMode:
    """에셋 생성 모드 매니저"""

    def __init__(self):
        self.enabled: bool = False
        self.workflow_source_path: str = ""
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

    # ─── 캐릭터 프리셋 ────────────────────────────────────
    def get_character_presets(self) -> dict:
        return copy.deepcopy(self._tags.get("character_presets", {}))

    def save_character_preset(self, name: str, appearance: str = "", outfit: str = "") -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        self._tags.setdefault("character_presets", {})[name.strip()] = {
            "appearance": appearance,
            "outfit": outfit,
        }
        self.save_tags()
        return {"success": True}

    @staticmethod
    def _convert_to_new_format(data: dict) -> dict:
        """구형 프리셋/캐릭터 데이터를 신형으로 변환.
        구형: {"appearance": {"이름": [태그]}, "outfits": {...}, "expressions": {...}}
        신형: {"appearance": "이름", "outfit": "이름", "expression": "이름"}
        """
        result = {}
        app = data.get("appearance", "")
        if isinstance(app, dict):
            result["appearance"] = next(iter(app), "")
        else:
            result["appearance"] = str(app)

        outfit = data.get("outfit", "")
        if isinstance(outfit, str) and outfit:
            result["outfit"] = outfit
        elif isinstance(data.get("outfits"), dict):
            result["outfit"] = next(iter(data["outfits"]), "")
        else:
            result["outfit"] = str(outfit) if outfit else ""

        expr = data.get("expression", "")
        if isinstance(expr, str) and expr:
            result["expression"] = expr
        elif isinstance(data.get("expressions"), dict):
            result["expression"] = next(iter(data["expressions"]), "")
        else:
            result["expression"] = str(expr) if expr else ""

        return result

    def load_character_preset(self, name: str, character: str) -> dict:
        preset = self._tags.get("character_presets", {}).get(name)
        if not preset:
            return {"success": False, "error": "존재하지 않는 프리셋"}
        if not character.strip():
            return {"success": False, "error": "캐릭터명 필요"}
        converted = self._convert_to_new_format(preset)
        self._tags.setdefault("characters", {})[character.strip()] = converted
        self.save_tags()
        return {"success": True}

    def delete_character_preset(self, name: str) -> dict:
        presets = self._tags.get("character_presets", {})
        if name not in presets:
            return {"success": False, "error": "존재하지 않는 프리셋"}
        del presets[name]
        self.save_tags()
        return {"success": True}

    def create_character_preset(self, name: str) -> dict:
        if not name.strip():
            return {"success": False, "error": "빈 이름"}
        presets = self._tags.setdefault("character_presets", {})
        if name.strip() in presets:
            return {"success": False, "error": "이미 존재하는 프리셋"}
        presets[name.strip()] = {"appearance": "", "outfit": "", "expression": ""}
        self.save_tags()
        return {"success": True}

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
        fd_activate: bool = False,
        hd_activate: bool = False,
        ed_activate: bool = False,
    ) -> tuple[str, str]:
        q_tags = self._tags.get("quality", [])
        c_tags = self._tags.get("composition", [])
        app_tags = self._tags.get("appearances", {}).get(appearance, [])
        outfit_tags = self._tags.get("outfits", {}).get(outfit, [])
        expr_tags = self._tags.get("expressions", {}).get(expression, [])
        n_tags = self._tags.get("negative", [])
        cn_tags = self._tags.get("character_negative", [])

        positive_parts = []
        for t in q_tags:
            if t.strip():
                positive_parts.append(t.strip())
        for t in c_tags:
            if t.strip():
                positive_parts.append(t.strip())
        for t in app_tags:
            if t.strip():
                positive_parts.append(t.strip())
        for t in expr_tags:
            if t.strip():
                positive_parts.append(t.strip())
        for t in outfit_tags:
            if t.strip():
                positive_parts.append(t.strip())

        positive = ", ".join(positive_parts)

        positive += f"\n[FACE_ID_ACTIVATE]\n{'true' if face_id_enabled else 'false'}"
        positive += f"\n[FACE_ID_STR]\n{face_id_strength}"
        positive += f"\n[FACE_ID_DIR]\n{face_id_dir or 'soya_char_ref/fallback'}"
        positive += f"\n[STYLE_ACTIVATE]\n{'true' if style_ref_enabled else 'false'}"
        positive += f"\n[STYLE_STR]\n{style_ref_strength}"
        positive += f"\n[STYLE_DIR]\n{style_ref_dir or 'soya_style_ref/fallback'}"
        positive += f"\n[LORA_ACTIVATE]\n{'true' if lora_activate else 'false'}"
        positive += f"\n[LORA_DATA]\n{lora_data or '{"list":[]}'}"
        positive += f"\n[POSE_ACTIVATE]\n{'true' if pose_enabled else 'false'}"
        if pose_enabled and pose_data:
            positive += f"\n[POSE_DATA]\n{json.dumps(pose_data, ensure_ascii=False)}"
        else:
            positive += f"\n[POSE_DATA]\n{json.dumps(DEFAULT_POSE_DATA, ensure_ascii=False)}"
        positive += f"\n[HRF_ACTIVATE]\n{'true' if hrf_activate else 'false'}"
        positive += f"\n[FD_ACTIVATE]\n{'true' if fd_activate else 'false'}"
        positive += f"\n[HD_ACTIVATE]\n{'true' if hd_activate else 'false'}"
        positive += f"\n[ED_ACTIVATE]\n{'true' if ed_activate else 'false'}"
        positive += "\n[END]"

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
        fd_activate: bool = False,
        hd_activate: bool = False,
        ed_activate: bool = False,
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
                    hrf_activate, fd_activate, hd_activate, ed_activate,
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
        fd_activate: bool,
        hd_activate: bool,
        ed_activate: bool,
    ) -> dict:
        ok = await self.update_asset_workflow()
        if not ok:
            error_msg = "워크플로우 준비 실패 (소스 경로 및 mode_workflow 폴더 모두 탐색 실패)"
            print(f"[ASSET] {error_msg}")
            return {"success": False, "error": error_msg}

        pose_data = None
        if pose_enabled and pose_id:
            pose_data = self._load_pose_data(pose_id)

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
            fd_activate=fd_activate,
            hd_activate=hd_activate,
            ed_activate=ed_activate,
        )
        if not positive:
            return {"success": False, "error": "프롬프트가 비어있음"}

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

        save_dir = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
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
        except Exception:
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
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        self._log("generate_saved", {
            "character": character, "outfit": outfit, "expression": expression,
            "filename": filename, "size": len(img_bytes),
        })

        if self.notify_frontend_func:
            await self.notify_frontend_func("asset_generation_completed", {
                "status": "success",
                "character": character, "outfit": outfit, "expression": expression,
                "filename": filename,
            })

        return {
            "success": True,
            "filename": filename,
            "character": character,
            "outfit": outfit,
            "expression": expression,
        }

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
                except Exception:
                    pass

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
            })
        return {"images": images, "representative": representative}

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
                    results.append({
                        "outfit": outfit_dir_name,
                        "expression": expr_dir_name,
                        "representative": rep_file,
                        "image_count": image_count,
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
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_name_mapping(self, data: dict):
        os.makedirs(os.path.dirname(NAME_MAPPING_FILE), exist_ok=True)
        with open(NAME_MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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
                # 빈 표정 폴더 정리
                for expr_dir_name in sorted(os.listdir(outfit_path)):
                    expr_path = os.path.join(outfit_path, expr_dir_name)
                    if os.path.isdir(expr_path) and not os.listdir(expr_path):
                        os.rmdir(expr_path)
                # 빈 복장 폴더 정리
                if not os.listdir(outfit_path):
                    os.rmdir(outfit_path)
                    continue
                outfits.add(outfit_dir_name)
                for expr_dir_name in sorted(os.listdir(outfit_path)):
                    expr_path = os.path.join(outfit_path, expr_dir_name)
                    if os.path.isdir(expr_path):
                        expressions.add(expr_dir_name)

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

    def save_character_name_mapping(self, character: str, export_name: str,
                                    outfit_mapping: dict, expression_mapping: dict,
                                    export_format: str = "webp", export_quality: int = 90,
                                    naming_order: list = None, naming_enabled: dict = None) -> dict:
        """캐릭터 이름 치환 규칙 저장."""
        data = self._load_name_mapping()
        data[character] = {
            "export_name": export_name,
            "outfits": outfit_mapping,
            "expressions": expression_mapping,
            "export_format": export_format,
            "export_quality": max(1, min(100, int(export_quality))),
            "naming_order": naming_order or ["character", "outfit", "expression"],
            "naming_enabled": naming_enabled or {"character": True, "outfit": True, "expression": True},
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

    def export_character_zip(self, character: str) -> str:
        """캐릭터의 대표 이미지를 이름 치환 규칙에 따라 이름_복장_표정.ext로 만들어 zip 반환."""
        import zipfile, io, tempfile, logging
        from PIL import Image

        log = logging.getLogger("asset_export")
        log.info(f"[ZIP 내보내기] 시작 — 캐릭터: {character}")

        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        if not os.path.isdir(char_dir):
            log.warning(f"[ZIP 내보내기] 캐릭터 디렉토리 없음: {char_dir}")
            return None

        mapping = self._load_name_mapping().get(character, {})
        export_name = mapping.get("export_name", "") or character
        outfit_map = mapping.get("outfits", {})
        expr_map = mapping.get("expressions", {})
        export_format = mapping.get("export_format", "webp").lower()
        export_quality = max(1, min(90, int(mapping.get("export_quality", 90))))
        naming_order = mapping.get("naming_order", ["character", "outfit", "expression"])
        naming_enabled = mapping.get("naming_enabled", {"character": True, "outfit": True, "expression": True})
        log.info(f"[ZIP 내보내기] 포맷={export_format}, 품질={export_quality}, 내보내기 이름={export_name}, 순서={naming_order}")

        # 포맷별 PIL 포맷 문자열 및 확장자
        FORMAT_MAP = {
            "webp": ("WEBP", ".webp"),
            "png": ("PNG", ".png"),
            "jpeg": ("JPEG", ".jpg"),
            "jpg": ("JPEG", ".jpg"),
            "avif": ("AVIF", ".avif"),
        }
        pil_format, ext = FORMAT_MAP.get(export_format, ("WEBP", ".webp"))

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
        skipped_no_mapping = 0
        skipped_no_rep = 0
        used_names = set()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
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

                    # 대표 이미지 찾기
                    rep_file = ""
                    rep_json = os.path.join(expr_path, "_representative.json")
                    if os.path.isfile(rep_json):
                        try:
                            with open(rep_json, "r", encoding="utf-8") as f:
                                rep_file = json.load(f).get("filename", "")
                        except Exception:
                            pass
                    if not rep_file:
                        skipped_no_rep += 1
                        continue

                    img_path = os.path.join(expr_path, rep_file)
                    if not os.path.isfile(img_path):
                        continue

                    # 치환 이름이 설정되지 않은 항목은 건너뛰기
                    o_name = outfit_map.get(outfit_dir_name, "")
                    e_name = expr_map.get(expr_dir_name, "")

                    # 활성화된 블록 중 매핑 누락 확인
                    skip = False
                    for block_id in naming_order:
                        if not naming_enabled.get(block_id, True):
                            continue
                        if block_id == "outfit" and not o_name:
                            skip = True
                            break
                        elif block_id == "expression" and not e_name:
                            skip = True
                            break
                    if skip:
                        skipped_no_mapping += 1
                        continue

                    # 순서와 활성화 상태에 따라 파일명 구성
                    parts = []
                    for block_id in naming_order:
                        if not naming_enabled.get(block_id, True):
                            continue
                        if block_id == "character":
                            parts.append(export_name)
                        elif block_id == "outfit":
                            parts.append(o_name)
                        elif block_id == "expression":
                            parts.append(e_name)
                    if not parts:
                        continue
                    base = "_".join(parts)
                    zip_name = base + ext
                    # 동일 이름 처리
                    if zip_name in used_names:
                        idx = 2
                        while f"{base}_{idx}{ext}" in used_names:
                            idx += 1
                        zip_name = f"{base}_{idx}{ext}"
                    used_names.add(zip_name)

                    orig_ext = os.path.splitext(rep_file)[1].lower()
                    # 같은 포맷 + 재압축 불필요 → 원본 그대로
                    if orig_ext == ext and not need_recompress:
                        zf.write(img_path, zip_name)
                        log.info(f"[ZIP 내보내기] [{added + 1}] 원본 그대로 추가: {zip_name}")
                    else:
                        try:
                            log.info(f"[ZIP 내보내기] [{added + 1}] 변환 중: {rep_file} → {zip_name} ({pil_format}, q={pil_quality})")
                            img = Image.open(img_path)
                            # JPEG는 알파 채널 미지원 → RGB 변환
                            if pil_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                                img = img.convert("RGB")
                            elif pil_format == "AVIF" and img.mode == "RGBA":
                                pass  # AVIF는 RGBA 지원
                            elif img.mode not in ("RGB", "RGBA"):
                                img = img.convert("RGBA") if pil_format != "JPEG" else img.convert("RGB")

                            img_buf = io.BytesIO()
                            save_kwargs = {"format": pil_format}
                            if pil_format in ("WEBP", "JPEG", "AVIF"):
                                save_kwargs["quality"] = pil_quality
                            if pil_format == "WEBP":
                                save_kwargs["method"] = 6  # 압축 속도 느리지만 최고 품질
                            img.save(img_buf, **save_kwargs)
                            img_buf.seek(0)
                            zf.writestr(zip_name, img_buf.read())
                            log.info(f"[ZIP 내보내기] [{added + 1}] 변환 완료: {zip_name} ({len(img_buf.getvalue())} bytes)")
                        except Exception as e:
                            # 변환 실패 시 원본 그대로 사용
                            log.warning(f"[ZIP 내보내기] [{added + 1}] 변환 실패, 원본 사용: {zip_name} — {e}")
                            zf.write(img_path, zip_name)

                    added += 1

        if added == 0:
            log.warning(f"[ZIP 내보내기] 추가된 파일 없음 (매핑 누락={skipped_no_mapping}, 대표이미지 없음={skipped_no_rep})")
            return None
        buf.seek(0)
        log.info(f"[ZIP 내보내기] 완료 — 총 {added}개 파일, 매핑 누락={skipped_no_mapping}, 대표이미지 없음={skipped_no_rep}, ZIP 크기={buf.getbuffer().nbytes / 1024:.1f}KB")
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

    # ─── 프리셋매니징: 일괄 삽입 ────────────────────────────
    def batch_insert_preset(self, category: str, name: str, tags_text: str) -> dict:
        """쉼표 구분 태그 문자열을 리스트로 파싱하여 tags.json에 저장"""
        if category not in PRESET_MGMT_CATEGORIES:
            print(f"[ASSET_MODE] batch_insert_preset: 지원하지 않는 카테고리 '{category}'")
            return {"success": False, "error": f"지원하지 않는 카테고리: {category}"}

        if not name or not name.strip():
            print("[ASSET_MODE] batch_insert_preset: 이름이 비어있음")
            return {"success": False, "error": "이름을 입력해주세요."}

        name = name.strip()

        # 쉼표 구분 파싱
        tags = [t.strip() for t in tags_text.split(",") if t.strip()]
        if not tags:
            print("[ASSET_MODE] batch_insert_preset: 태그가 비어있음")
            return {"success": False, "error": "태그를 입력해주세요."}

        cat_data = self._tags.setdefault(category, {})
        if not isinstance(cat_data, dict):
            print(f"[ASSET_MODE] batch_insert_preset: 카테고리 '{category}'가 dict가 아님")
            return {"success": False, "error": f"카테고리 '{category}' 구조 오류"}

        cat_data[name] = tags
        self.save_tags()

        self._log("preset_batch_inserted", {"category": category, "name": name, "count": len(tags)})
        return {"success": True, "name": name, "count": len(tags)}

    # ─── 프리셋매니징: 에셋 추적 ────────────────────────────
    def trace_preset_assets(self, category: str, name: str) -> dict:
        """프리셋이 사용된 에셋 이미지를 추적"""
        if category not in PRESET_MGMT_CATEGORIES:
            print(f"[ASSET_MODE] trace_preset_assets: 지원하지 않는 카테고리 '{category}'")
            return {"success": False, "error": f"지원하지 않는 카테고리: {category}"}

        # 프리셋 태그 가져오기 (활성 + 숨김 모두 확인)
        preset_tags = []
        active_cat = self._tags.get(category, {})
        hidden = self.load_hidden_tags()
        hidden_cat = hidden.get(category, {})

        if isinstance(active_cat, dict) and name in active_cat:
            preset_tags = active_cat[name] if isinstance(active_cat[name], list) else [active_cat[name]]
        elif isinstance(hidden_cat, dict) and name in hidden_cat:
            preset_tags = hidden_cat[name] if isinstance(hidden_cat[name], list) else [hidden_cat[name]]
        else:
            print(f"[ASSET_MODE] trace_preset_assets: '{name}'을(를) 찾을 수 없음")
            return {"success": False, "error": f"'{name}'을(를) 찾을 수 없습니다."}

        # asset/ 디렉토리 순회
        results = []
        if not os.path.isdir(ASSET_DIR):
            print(f"[ASSET_MODE] trace_preset_assets: asset/ 디렉토리 없음")
            return {"success": True, "results": [], "preset_tags": preset_tags}

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
                    # _prompt.json 파일 순회
                    for fname in os.listdir(expr_dir):
                        if not fname.endswith("_prompt.json"):
                            continue
                        prompt_path = os.path.join(expr_dir, fname)
                        try:
                            with open(prompt_path, "r", encoding="utf-8") as f:
                                prompt_data = json.load(f)
                        except Exception as e:
                            print(f"[ASSET_MODE] trace: {prompt_path} 읽기 실패: {e}")
                            continue

                        matched = False
                        if category in ("appearances", "outfits", "expressions"):
                            # 필드값 매칭
                            field_map = {
                                "appearances": "appearance",
                                "outfits": "outfit",
                                "expressions": "expression",
                            }
                            field = field_map[category]
                            if prompt_data.get(field) == name:
                                matched = True
                        elif category in ("quality_presets", "composition_presets"):
                            # positive 텍스트에서 태그 포함 여부
                            positive = prompt_data.get("positive", "")
                            if any(tag.lower() in positive.lower() for tag in preset_tags if tag):
                                matched = True
                        elif category in ("negative_presets", "character_negative_presets"):
                            # negative 텍스트에서 태그 포함 여부
                            negative = prompt_data.get("negative", "")
                            if any(tag.lower() in negative.lower() for tag in preset_tags if tag):
                                matched = True

                        if matched:
                            img_file = fname.replace("_prompt.json", "")
                            # 실제 이미지 확장자 확인
                            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                                if os.path.isfile(os.path.join(expr_dir, img_file + ext)):
                                    img_file = img_file + ext
                                    break
                            else:
                                # 이미지 파일이 없으면 prompt 파일명만 유지
                                img_file = fname

                            results.append({
                                "character": prompt_data.get("character", char_name),
                                "outfit": prompt_data.get("outfit", outfit_name),
                                "expression": prompt_data.get("expression", expr_name),
                                "image_file": img_file,
                                "prompt_data": prompt_data,
                            })

        self._log("preset_traced", {"category": category, "name": name, "matches": len(results)})
        return {"success": True, "results": results, "preset_tags": preset_tags}

    # ─── 상태 ─────────────────────────────────────────────
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
