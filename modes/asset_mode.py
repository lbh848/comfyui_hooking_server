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
NAME_MAPPING_FILE = os.path.join(ASSET_DATA_DIR, "name_mapping.json")
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
    "appearances": {},
    "outfits": {},
    "expressions": {},
    "characters": {},
    "quality_presets": {},
    "composition_presets": {},
    "negative_presets": {},
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
                self._log("tags_loaded", {"characters": len(self._tags.get("characters", {}))})
            except Exception as e:
                self._log("tags_load_error", {"error": str(e)})
                self._tags = copy.deepcopy(DEFAULT_TAGS)
        else:
            self._tags = copy.deepcopy(DEFAULT_TAGS)

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

        negative_parts = [t.strip() for t in n_tags if t.strip()]
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

    async def update_asset_workflow(self) -> bool:
        src = self.workflow_source_path
        if not src or not os.path.isfile(src):
            self._log("workflow_skip", {"reason": "no_source", "path": src or ""})
            return False

        try:
            file_hash = self._compute_file_hash(src)
        except Exception as e:
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
            self._log("workflow_load_error", {"error": str(e)})
            return False

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
                self._log("workflow_convert_error", {"error": str(error)})
                return False
            self._asset_api_workflow = api_wf
            self._asset_hash = file_hash
            self._save_stored_hash(file_hash)
            self._save_cached_api(api_wf)
            self._log("workflow_converted", {"nodes": len(api_wf)})
            return True

        self._log("workflow_no_converter", {})
        return False

    # ─── 이미지 생성 ──────────────────────────────────────
    async def generate(
        self,
        character: str,
        appearance: str = "",
        outfit: str = "",
        expression: str = "",
        face_id_enabled: bool = False,
        face_id_strength: float = 0.55,
        reference_image: str = "",
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
                    face_id_enabled, face_id_strength, reference_image,
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
        reference_image: str,
        pose_enabled: bool,
        pose_id: str,
        hrf_activate: bool,
        fd_activate: bool,
        hd_activate: bool,
        ed_activate: bool,
    ) -> dict:
        ok = await self.update_asset_workflow()
        if not ok:
            return {"success": False, "error": "워크플로우 준비 실패"}

        pose_data = None
        if pose_enabled and pose_id:
            pose_data = self._load_pose_data(pose_id)

        positive, negative = self.build_prompts(
            appearance, outfit, expression,
            face_id_enabled=face_id_enabled,
            face_id_strength=face_id_strength,
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
                reference_image=reference_image if face_id_enabled else "",
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
                elif title == "레퍼런스이미지로드" and face_id_enabled and reference_image:
                    ninfo["inputs"]["image"] = reference_image
                    ninfo["inputs"]["subfolder"] = ""
                    ninfo["inputs"]["type"] = "input"

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

    def list_images(self, character: str, outfit: str, expression: str) -> list[dict]:
        img_dir = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
            self._safe_dirname(expression),
        )
        if not os.path.isdir(img_dir):
            return []

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
                "positive": prompt_data.get("positive", ""),
                "negative": prompt_data.get("negative", ""),
            })
        return images

    def set_representative(self, character: str, outfit: str, expression: str, filename: str) -> dict:
        img_dir = os.path.join(
            ASSET_DIR,
            self._safe_dirname(character),
            self._safe_dirname(outfit),
            self._safe_dirname(expression),
        )
        os.makedirs(img_dir, exist_ok=True)

        fpath = os.path.join(img_dir, filename)
        if not os.path.isfile(fpath):
            return {"success": False, "error": "파일이 존재하지 않음"}

        rep_path = os.path.join(img_dir, "_representative.json")

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
        }

    def save_character_name_mapping(self, character: str, export_name: str,
                                    outfit_mapping: dict, expression_mapping: dict) -> dict:
        """캐릭터 이름 치환 규칙 저장."""
        data = self._load_name_mapping()
        data[character] = {
            "export_name": export_name,
            "outfits": outfit_mapping,
            "expressions": expression_mapping,
        }
        self._save_name_mapping(data)
        return {"success": True}

    def export_character_zip(self, character: str) -> str:
        """캐릭터의 대표 이미지를 이름 치환 규칙에 따라 이름_복장_표정.webp로 만들어 zip 반환."""
        import zipfile, io, tempfile

        char_dir = os.path.join(ASSET_DIR, self._safe_dirname(character))
        if not os.path.isdir(char_dir):
            return None

        mapping = self._load_name_mapping().get(character, {})
        export_name = mapping.get("export_name", "") or character
        outfit_map = mapping.get("outfits", {})
        expr_map = mapping.get("expressions", {})

        buf = io.BytesIO()
        added = 0
        used_names = set()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for outfit_dir_name in sorted(os.listdir(char_dir)):
                outfit_path = os.path.join(char_dir, outfit_dir_name)
                if not os.path.isdir(outfit_path):
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
                        continue

                    img_path = os.path.join(expr_path, rep_file)
                    if not os.path.isfile(img_path):
                        continue

                    # 치환 이름이 설정되지 않은 항목은 건너뛰기
                    o_name = outfit_map.get(outfit_dir_name, "")
                    e_name = expr_map.get(expr_dir_name, "")
                    if not o_name or not e_name:
                        continue

                    base = f"{export_name}_{o_name}_{e_name}"
                    zip_name = base + ".webp"
                    # 동일 이름 처리
                    if zip_name in used_names:
                        idx = 2
                        while f"{base}_{idx}.webp" in used_names:
                            idx += 1
                        zip_name = f"{base}_{idx}.webp"
                    used_names.add(zip_name)

                    zf.write(img_path, zip_name)
                    added += 1

        if added == 0:
            return None
        buf.seek(0)
        return buf

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
